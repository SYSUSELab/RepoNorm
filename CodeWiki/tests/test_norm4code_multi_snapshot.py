"""Frozen multi-snapshot selection, preparation and offline CLI dispatch."""

from collections import Counter
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from integration import run
from integration.runtime import AdapterError


def test_registry_matches_frozen_task_releases_and_norm4code_bindings():
    manifest = json.loads((run.ROOT / "benchmark/manifest.json").read_text())
    counts = Counter(task["snapshot_id"] for task in manifest["tasks"])
    assert len(run.SNAPSHOTS) == 14
    assert sum(counts.values()) == 121
    for row in manifest["snapshots"]:
        assert run.SNAPSHOTS[row["id"]] == (row["repository_url"], row["commit"], counts[row["id"]])
        assert run.repository_for(row["id"]) == run.ROOT / "outputs/repositories" / row["id"]


@pytest.fixture
def registered_repo(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    (repository / "example.py").write_text("value = 1\n")
    git("add", "example.py")
    git("commit", "-qm", "Fixture")
    target = "cli-d8763b93021c"
    origin = run.SNAPSHOTS[target][0]
    commit = git("rev-parse", "HEAD")
    git("remote", "add", "origin", origin)
    monkeypatch.setitem(run.SNAPSHOTS, target, (origin, commit, 7))
    monkeypatch.setattr(run, "repository_for", lambda selected: repository if selected == target else pytest.fail("wrong target"))
    return repository, target, commit, git


@pytest.mark.parametrize("damage", ["none", "origin", "head", "dirty", "untracked", "symlink"])
def test_existing_snapshot_is_verified_without_repair(registered_repo, monkeypatch, damage):
    repository, target, commit, git = registered_repo
    error = None
    if damage == "origin":
        git("remote", "set-url", "origin", "https://example.invalid/other.git")
        error = "unexpected_target_origin"
    elif damage == "head":
        monkeypatch.setitem(run.SNAPSHOTS, target, (run.SNAPSHOTS[target][0], "f" * 40, 7))
        error = "snapshot_commit_mismatch"
    elif damage in {"dirty", "untracked"}:
        (repository / ("example.py" if damage == "dirty" else "extra.py")).write_text("changed\n")
        error = "target_worktree_not_clean"
    elif damage == "symlink":
        alias = repository.parent / "alias"
        alias.symlink_to(repository, target_is_directory=True)
        monkeypatch.setattr(run, "repository_for", lambda _: alias)
        error = "target_repository_symlink"
    before = {str(path.relative_to(repository)): path.read_bytes()
              for path in repository.rglob("*") if path.is_file()}
    original_run = subprocess.run
    def read_only(command, **kwargs):
        assert command[0] == "git"
        assert command[command.index("-C") + 2:] in [
            ["rev-parse", "HEAD"], ["status", "--porcelain", "--untracked-files=all", "--ignored"],
            ["ls-files", "-z"], ["remote", "get-url", "origin"],
        ]
        return original_run(command, **kwargs)
    monkeypatch.setattr(run.subprocess, "run", read_only)
    if error:
        with pytest.raises(AdapterError, match=error):
            run.prepare_snapshot(target)
    else:
        assert run.prepare_snapshot(target)["commit"] == commit
    after = {str(path.relative_to(repository)): path.read_bytes()
             for path in repository.rglob("*") if path.is_file()}
    assert after == before


def test_missing_snapshot_uses_only_selected_norm4code_clone(tmp_path, monkeypatch):
    target = "jin-20eb7f5e3312"
    monkeypatch.setattr(run, "ROOT", tmp_path)
    calls = []
    def clone(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run.subprocess, "run", clone)
    def verify(selected):
        assert selected == target
        assert len(calls) == 1
        return {"commit": run.SNAPSHOTS[target][1]}
    monkeypatch.setattr(run, "registered_snapshot", verify)
    assert run.prepare_snapshot(target) == {"commit": run.SNAPSHOTS[target][1]}
    assert calls == [([run.sys.executable, str(tmp_path / "runner/dataset.py"), "clone", "--snapshot", target], {"check": True})]


@pytest.mark.parametrize("arguments, expected", [
    ([], [run.TARGET]),
    (["--all"], list(run.SNAPSHOTS)),
    (["--all"], list(run.SNAPSHOTS)),
    (["--target", "cli-d8763b93021c", "--target", run.TARGET,
      "--target", "cli-d8763b93021c"], ["cli-d8763b93021c", run.TARGET]),
])
def test_cli_selection_is_sequential_and_deduplicated(monkeypatch, arguments, expected):
    calls = []
    monkeypatch.setattr(run.sys, "argv", ["run.py", *arguments])
    monkeypatch.setattr(run, "run_target", lambda args, target: calls.append(target) or 0)
    assert run.main() == 0
    assert calls == expected


@pytest.mark.parametrize("failure", ["exception", "exit_code"])
def test_batch_continues_after_failure_and_returns_nonzero(monkeypatch, capsys, failure):
    calls = []
    def execute(args, target):
        calls.append(target)
        if len(calls) == 2:
            if failure == "exception":
                raise AdapterError("fixture_failure")
            return 3
        return 0
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--all"])
    monkeypatch.setattr(run, "run_target", execute)
    assert run.main() == 1
    assert calls == list(run.SNAPSHOTS)
    if failure == "exception":
        assert "fixture_failure" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [
    ["--target", "unknown/task"], ["--all", "--target", run.TARGET],
    ["--all", "--target", run.TARGET],

])
def test_invalid_selection_fails_before_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(run.sys, "argv", ["run.py", *arguments])
    monkeypatch.setattr(run, "run_target", lambda *args: pytest.fail("invalid dispatch"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


@pytest.mark.parametrize("arguments, reason", [
    (["--mock"], "mock_requires_probe_mode"),
    (["--clone-only", "--mode", "generate"], "clone_only_requires_check_mode"),
    (["--all", "--worker-run-dir", "/unused"], "worker_requires_one_target"),
    (["--all", "--worker-run-dir", "/unused"], "worker_requires_one_target"),
])
def test_invalid_modes_fail_before_dispatch(monkeypatch, capsys, arguments, reason):
    monkeypatch.setattr(run.sys, "argv", ["run.py", *arguments])
    monkeypatch.setattr(run, "run_target", lambda *args: pytest.fail("invalid dispatch"))
    assert run.main() == 1
    assert reason in capsys.readouterr().err


def test_list_and_clone_only_never_load_profile_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(run, "load_execution_model_profile", lambda *args: pytest.fail("profile load"))
    monkeypatch.setattr(run, "dotenv_values", lambda *args: pytest.fail("credential read"))
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--list"])
    assert run.main() == 0
    output = capsys.readouterr().out
    for target in run.SNAPSHOTS:
        assert target in output
    calls = []
    monkeypatch.setattr(run, "prepare_snapshot", lambda target: calls.append(target) or {"commit": run.SNAPSHOTS[target][1]})
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--all", "--clone-only"])
    assert run.main() == 0
    assert calls == list(run.SNAPSHOTS)


@pytest.mark.parametrize("selection", [None, "matching", "conflicting", "unknown_receipt"])
@pytest.mark.parametrize("allow_partial", [False, True])
def test_offline_export_infers_receipt_target_and_passes_frozen_commit(tmp_path, monkeypatch, capsys, selection, allow_partial):
    from integration import consumer
    target = "cli-d8763b93021c"
    (tmp_path / "receipt.json").write_text(json.dumps({"target": target if selection != "unknown_receipt" else "unknown/task"}))
    arguments = ["run.py", "--export-consumer", str(tmp_path)]
    if allow_partial:
        arguments.append("--allow-partial")
    if selection in {"matching", "conflicting"}:
        arguments += ["--target", target if selection == "matching" else run.TARGET]
    monkeypatch.setattr(run.sys, "argv", arguments)
    verified, exported = [], []
    monkeypatch.setattr(run, "registered_snapshot", lambda selected: verified.append(selected))
    monkeypatch.setattr(run, "load_execution_model_profile", lambda *args: pytest.fail("profile load"))
    def export(folder, repository, *, expected_commit, allow_partial):
        exported.append((folder, repository, expected_commit, allow_partial))
        return folder / "consumer/knowledge"
    monkeypatch.setattr(consumer, "export_consumer", export)
    if selection in {None, "matching"}:
        assert run.main() == 0
        assert verified == [target]
        assert exported == [(tmp_path, run.repository_for(target), run.SNAPSHOTS[target][1], allow_partial)]
    else:
        assert run.main() == 1
        assert not exported
        assert not verified
        assert ("target_mismatch" if selection == "conflicting" else "unregistered_target") in capsys.readouterr().err


def test_partial_flag_requires_standalone_export(monkeypatch, capsys):
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--allow-partial", "--mode", "generate"])
    assert run.main() == 1
    assert "allow_partial_requires_export_consumer" in capsys.readouterr().err
