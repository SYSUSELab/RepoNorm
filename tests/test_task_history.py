"""History isolation uses disposable synthetic repos, never the host checkout."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from task_history import SANITIZE_SCRIPT, sanitize_container_history


def git(repo, *args, input=None, check=True):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=repo, input=input, capture_output=True, text=True, check=check)


def repository(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "source.py").write_text("base = True\n")
    git(tmp_path, "add", "source.py")
    git(tmp_path, "commit", "-m", "base")
    return git(tmp_path, "rev-parse", "HEAD").stdout.strip()


def test_future_refs_and_dangling_secret_objects_are_removed(tmp_path):
    ancestor = repository(tmp_path)
    (tmp_path / "note.txt").write_text("base ancestor remains available\n")
    git(tmp_path, "add", "note.txt")
    git(tmp_path, "commit", "-m", "frozen snapshot")
    base = git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    (tmp_path / "solution.py").write_text("secret_future_answer = True\n")
    git(tmp_path, "add", "solution.py")
    git(tmp_path, "commit", "-m", "future solution")
    future = git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    git(tmp_path, "tag", "future-answer", future)
    git(tmp_path, "update-ref", "refs/remotes/origin/future", future)
    dangling = git(tmp_path, "hash-object", "-w", "--stdin", input="secret dangling gold patch").stdout.strip()
    git(tmp_path, "checkout", "--detach", base)
    hooks = tmp_path / ".git/hooks/post-checkout"
    hooks.write_text("#!/bin/sh\ntouch hook-ran\n")
    hooks.chmod(0o755)
    (tmp_path / ".git/ORIG_HEAD").write_text(future + "\n")
    result = subprocess.run([sys.executable, "-c", SANITIZE_SCRIPT, base], cwd=tmp_path, text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["remaining_refs"] == [f"refs/heads/base {base}"]
    assert receipt["unreachable_objects"] == 0
    assert git(tmp_path, "cat-file", "-e", future, check=False).returncode != 0
    assert git(tmp_path, "cat-file", "-e", dangling, check=False).returncode != 0
    assert git(tmp_path, "cat-file", "-e", ancestor).returncode == 0
    assert not (tmp_path / "solution.py").exists()
    assert not (tmp_path / "hook-ran").exists()
    assert not (tmp_path / ".git/ORIG_HEAD").exists()
    assert (tmp_path / "source.py").read_text() == "base = True\n"


def test_dirty_worktree_is_rejected_without_modification(tmp_path):
    base = repository(tmp_path)
    (tmp_path / "source.py").write_text("user change\n")
    result = subprocess.run([sys.executable, "-c", SANITIZE_SCRIPT, base], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Tracked worktree changes" in result.stderr
    assert (tmp_path / "source.py").read_text() == "user change\n"


def test_api_rejects_non_container_target_before_subprocess():
    with pytest.raises(ValueError, match="container identity"):
        sanitize_container_history("/workspace/repo", "a" * 40)
