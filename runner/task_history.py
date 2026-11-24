"""Remove post-snapshot Git objects in the disposable solver container only."""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any


# Kept as a standalone trusted script so the exact production operation can be
# exercised against synthetic repositories without requiring a Docker daemon.
SANITIZE_SCRIPT = r'''
import json
import os
from pathlib import Path
import re
import subprocess
import sys

base = sys.argv[1]
if not re.fullmatch(r"[0-9a-f]{40}", base):
    raise ValueError("Expected a complete SHA-1 snapshot identity")
environment = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}

def git(*args):
    result = subprocess.run(["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", *args], env=environment, text=True, capture_output=True, check=True)
    return result.stdout.strip()

directory = Path(git("rev-parse", "--absolute-git-dir"))
if not (Path.cwd() / ".git").is_dir() or directory.resolve() != (Path.cwd() / ".git").resolve():
    raise ValueError("Only a standalone container repository is supported")
if (directory / "objects/info/alternates").exists() or (directory / "info/grafts").exists():
    raise ValueError("External or rewritten Git ancestry is not admitted")
if git("rev-parse", "HEAD") != base:
    raise ValueError("HEAD does not match the validated snapshot")
if git("status", "--porcelain", "--untracked-files=no"):
    raise ValueError("Tracked worktree changes exist before solver starts")
before_refs = git("for-each-ref", "--format=%(refname)").splitlines()
git("checkout", "--detach", base)
for ref in before_refs:
    git("update-ref", "--no-deref", "-d", ref)
git("update-ref", "refs/heads/base", base)
# These pseudorefs can retain object identities outside refs/ and reflogs.
for name in ("ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD", "AUTO_MERGE", "REBASE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"):
    (directory / name).unlink(missing_ok=True)
git("reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
git("-c", "gc.cruftPacks=false", "gc", "--prune=now")
refs = git("for-each-ref", "--format=%(refname) %(objectname)").splitlines()
if refs != ["refs/heads/base " + base] or git("rev-parse", "HEAD") != base:
    raise ValueError("Unexpected ref or HEAD after history isolation")
fsck = git("fsck", "--full", "--no-reflogs", "--unreachable")
if re.search(r"\b(unreachable|dangling)\b", fsck):
    raise ValueError("Unreachable Git objects remain accessible")
reachable = set(git("rev-list", "--objects", "--all").splitlines())
base_objects = set(git("rev-list", "--objects", base).splitlines())
if reachable != base_objects:
    raise ValueError("Objects outside base ancestry remain reachable")
if git("status", "--porcelain", "--untracked-files=no"):
    raise ValueError("History isolation changed the tracked worktree")
print(json.dumps({"schema_version": "cooperbench-task-history-1", "base_commit": base, "head": git("rev-parse", "HEAD"), "remaining_refs": refs, "removed_ref_count": len(before_refs), "reachable_object_count": len(base_objects), "unreachable_objects": 0, "post_snapshot_history_removed": True}))
'''


def sanitize_container_history(container: str, base: str) -> dict[str, Any]:
    """Mutate only the named disposable container, before GitConnector/LLM."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
        raise ValueError("Invalid container identity")
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Expected a complete snapshot SHA-1")
    completed = subprocess.run(
        ["docker", "exec", "-w", "/workspace/repo", container, "python", "-c", SANITIZE_SCRIPT, base],
        capture_output=True, text=True, check=True, timeout=300,
    )
    return json.loads(completed.stdout)
