"""Offline consumer publication and host handoff checks."""

import json
import subprocess
import shutil
from types import SimpleNamespace

import pytest

from integration.consumer import LEGACY_START, export_consumer, original_markdown, verify_consumer
from integration.runtime import AdapterError, digest
from integration.run import snapshot as real_snapshot
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile, stable_profile_text


@pytest.fixture
def source(tmp_path, monkeypatch):
    from integration import run
    repository = tmp_path / "repo"
    repository.mkdir()
    state = {"commit": "frozen", "content_digest": "fixture"}
    def snapshot(repository_path, expected=None):
        assert repository_path == repository
        if expected is not None and expected != state["commit"]:
            raise AdapterError("snapshot_commit_mismatch")
        return dict(state)
    monkeypatch.setattr(run, "snapshot", snapshot)
    folder = tmp_path / "run"
    docs = folder / "docs"
    docs.mkdir(parents=True)
    profile = load_execution_model_profile()
    (folder / "execution-profile.json").write_text(stable_profile_text(profile))
    (folder / "receipt.json").write_text(json.dumps({
        "status": "complete", "mode": "generate", "snapshot_unchanged": True,
        "snapshot": state, "profile_digest": profile.profile_digest, "target": "fixture/task",
    }))
    (folder / "command.json").write_text(json.dumps({
        "status": "complete", "exit_code": 0, "profile_digest": profile.profile_digest,
    }))
    (docs / "overview.md").write_bytes(b"# Overview\n\n[Module](module.md)\n")
    (docs / "module.md").write_bytes(b"# Module\n\nOriginal content.\n")
    (docs / "module_tree.json").write_text(json.dumps({"module": {"children": {}}}))
    (docs / "metadata.json").write_text(json.dumps({"generation_info": {"commit_id": "frozen"}}))
    return folder, repository, state


def test_exact_markdown_projection_and_no_overwrite(source):
    folder, repository, _ = source
    knowledge = export_consumer(folder, repository)
    assert verify_consumer(folder, repository) == knowledge
    assert set(p.name for p in knowledge.iterdir()) == {"overview.md", "module.md"}
    manifest = json.loads((folder / "consumer/manifest.json").read_text())
    assert manifest["schema_version"] == "codewiki-consumer-2.0"
    assert manifest["content_policy"] == "original_markdown_bytes_only"
    assert "entrypoint" not in manifest
    assert original_markdown(folder, repository) == {
        path.name: path.read_bytes() for path in (folder / "docs").glob("*.md")
    }
    assert (knowledge / "module.md").read_bytes() == (folder / "docs/module.md").read_bytes()
    assert (knowledge / "module.md").stat().st_mode & 0o222 == 0
    with pytest.raises(AdapterError, match="destination_exists"):
        export_consumer(folder, repository)


def _incomplete_source(folder):
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt.update(status="failed", error="IncompleteDocumentationError")
    (folder / "receipt.json").write_text(json.dumps(receipt))
    command = json.loads((folder / "command.json").read_text())
    command.update(status="failed", exit_code=1)
    (folder / "command.json").write_text(json.dumps(command))
    (folder / "docs/module_tree.json").write_text(json.dumps({
        "module": {"children": {}}, "missing": {"children": {}}}))


def test_partial_export_explicit_original_bytes_and_coverage(source):
    folder, repository, _ = source
    _incomplete_source(folder)
    before = (folder / "receipt.json").read_bytes()
    with pytest.raises(AdapterError, match="requires_successful"):
        export_consumer(folder, repository)
    knowledge = export_consumer(folder, repository, allow_partial=True)
    assert verify_consumer(folder, repository) == knowledge
    assert original_markdown(folder, repository) == {
        p.name: p.read_bytes() for p in (folder / "docs").glob("*.md")}
    assert (folder / "receipt.json").read_bytes() == before
    manifest_path = folder / "consumer/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "codewiki-consumer-2.1"
    assert manifest["coverage"]["missing_modules"] == ["missing"]
    manifest["coverage"]["status"] = "complete"
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AdapterError, match="partial_coverage"):
        verify_consumer(folder, repository)


@pytest.mark.parametrize("change", ["error", "provider", "snapshot", "profile", "overview", "empty"])
def test_partial_does_not_bypass_other_failures(source, change):
    folder, repository, state = source
    _incomplete_source(folder)
    receipt = json.loads((folder / "receipt.json").read_text())
    if change == "error":
        receipt["error"] = "ProfileViolation"
    elif change == "provider":
        receipt["provider"] = {"failure": "profile_violation"}
    elif change == "snapshot":
        state["content_digest"] = "changed"
    elif change == "profile":
        receipt["profile_digest"] = "changed"
    elif change == "overview":
        (folder / "docs/overview.md").unlink()
    elif change == "empty":
        (folder / "docs/module.md").write_text("")
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(AdapterError):
        export_consumer(folder, repository, allow_partial=True)
    assert not (folder / "consumer").exists()


@pytest.mark.parametrize("change", ["failed", "probe", "snapshot", "missing", "symlink", "nested"])
def test_invalid_runs_never_publish(source, change):
    folder, repository, state = source
    receipt = json.loads((folder / "receipt.json").read_text())
    if change == "failed":
        receipt["status"] = "failed"
    elif change == "probe":
        receipt["mode"] = "probe"
    elif change == "snapshot":
        state["content_digest"] = "changed"
    elif change == "missing":
        (folder / "docs/module.md").unlink()
    elif change == "symlink":
        (folder / "docs/module.md").unlink()
        (folder / "docs/module.md").symlink_to(folder / "receipt.json")
    else:
        (folder / "docs/nested").mkdir()
        (folder / "docs/nested/document.md").write_text("must not silently omit")
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(AdapterError):
        export_consumer(folder, repository)
    assert not (folder / "consumer").exists()


@pytest.mark.parametrize("change", ["snapshot", "content", "extra", "source"])
def test_handoff_rejects_mismatch_and_tampering(source, change):
    folder, repository, state = source
    knowledge = export_consumer(folder, repository)
    if change == "snapshot":
        state["content_digest"] = "changed"
    elif change == "content":
        (knowledge / "module.md").chmod(0o644)
        (knowledge / "module.md").write_text("changed")
    elif change == "extra":
        knowledge.chmod(0o755)
        (knowledge / "extra.json").write_text("{}")
    else:
        (folder / "docs/module.md").write_text("changed")
    with pytest.raises(AdapterError):
        verify_consumer(folder, repository)


def _legacy_export(folder, repository):
    """Reconstruct the exact historical wrapper in an isolated test fixture."""
    knowledge = export_consumer(folder, repository)
    knowledge.chmod(0o755)
    if (knowledge / "START.md").exists():
        (knowledge / "START.md").chmod(0o644)
    (knowledge / "START.md").write_text(LEGACY_START)
    manifest_path = folder / "consumer/manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(schema_version="codewiki-consumer-1.0", entrypoint="START.md",
                    content_policy="original_markdown_bytes_plus_navigation_only")
    manifest["knowledge_sha256"]["START.md"] = digest(LEGACY_START.encode())
    manifest_path.write_text(json.dumps(manifest))
    return knowledge


def test_legacy_projection_is_verified_and_source_is_immutable(source):
    folder, repository, _ = source
    knowledge = _legacy_export(folder, repository)
    before = {str(path.relative_to(folder)): path.read_bytes()
              for path in folder.rglob("*") if path.is_file()}
    assert verify_consumer(folder, repository) == knowledge
    native = original_markdown(folder, repository)
    assert set(native) == {"overview.md", "module.md"}
    assert native["module.md"] == (folder / "docs/module.md").read_bytes()
    assert before == {str(path.relative_to(folder)): path.read_bytes()
                      for path in folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", ["START.md", "start.md", "Start.md"])
def test_native_start_document_is_never_removed(source, name):
    folder, repository, _ = source
    # Even exact legacy-wrapper bytes are native if recorded as generated docs.
    (folder / "docs" / name).write_text(LEGACY_START)
    knowledge = export_consumer(folder, repository)
    assert (knowledge / name).read_text() == LEGACY_START
    assert original_markdown(folder, repository)[name] == LEGACY_START.encode()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("change", ["omit", "rewrite", "add_source", "remove_source_hash", "entrypoint"])
def test_metadata_cannot_drop_or_rewrite_native_documents(source, legacy, change):
    folder, repository, _ = source
    knowledge = (_legacy_export if legacy else export_consumer)(folder, repository)
    knowledge.chmod(0o755)
    manifest_path = folder / "consumer/manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    if change == "omit":
        (knowledge / "module.md").unlink()
        del manifest["knowledge_sha256"]["module.md"]
        del manifest["source_sha256"]["docs/module.md"]
    elif change == "rewrite":
        (knowledge / "module.md").chmod(0o644)
        (knowledge / "module.md").write_text("replacement")
        manifest["knowledge_sha256"]["module.md"] = digest(b"replacement")
    elif change == "add_source":
        (folder / "docs/another.md").write_text("new original document")
    elif change == "remove_source_hash":
        del manifest["source_sha256"]["receipt.json"]
    else:
        manifest["entrypoint"] = "module.md"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AdapterError):
        original_markdown(folder, repository)


def test_legacy_host_start_must_match_frozen_bytes(source):
    folder, repository, _ = source
    knowledge = _legacy_export(folder, repository)
    (knowledge / "START.md").write_text("extra instructions")
    manifest_path = folder / "consumer/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["knowledge_sha256"]["START.md"] = digest(b"extra instructions")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AdapterError, match="projection_content"):
        original_markdown(folder, repository)


def test_legacy_manifest_cannot_label_native_start_as_host_navigation(source):
    folder, repository, _ = source
    (folder / "docs/START.md").write_text(LEGACY_START)
    _legacy_export(folder, repository)
    with pytest.raises(AdapterError, match="legacy_start_collision"):
        original_markdown(folder, repository)


@pytest.fixture
def nonregistered_consumer(source, monkeypatch):
    """An ordinary published consumer whose initial Git HEAD is not Tiktoken."""
    from integration import run
    folder, repository, state = source
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Offline fixture")
    (repository / "example.py").write_text("value = 1\n")
    git("add", "example.py")
    git("commit", "-m", "initial fixture")
    head = git("rev-parse", "HEAD")
    state.clear()
    state.update(real_snapshot(repository, expected=head))
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["snapshot"] = state
    (folder / "receipt.json").write_text(json.dumps(receipt))
    (folder / "docs/metadata.json").write_text(json.dumps({"generation_info": {"commit_id": head}}))
    # Exercise export and host verification against an actual nondefault Git
    # snapshot, without invoking generation or making any model call.
    monkeypatch.setattr(run, "snapshot", real_snapshot)
    knowledge = export_consumer(folder, repository, expected_commit=head)
    return folder, repository, knowledge, head, git


def test_host_accepts_caller_bound_nonregistered_snapshot(nonregistered_consumer):
    folder, repository, knowledge, head, _ = nonregistered_consumer
    assert verify_consumer(folder, repository, expected_commit=head) == knowledge
    # Existing callers retain the registered Tiktoken default.
    with pytest.raises(AdapterError, match="snapshot_commit_mismatch"):
        verify_consumer(folder, repository)


@pytest.mark.parametrize("change", ["wrong_expected", "changed_head", "wrong_manifest"])
def test_explicit_snapshot_binding_rejects_mismatch(nonregistered_consumer, change):
    folder, repository, _, head, git = nonregistered_consumer
    expected = head
    if change == "wrong_expected":
        expected = "f" * 40
    elif change == "changed_head":
        (repository / "example.py").write_text("value = 2\n")
        git("add", "example.py")
        git("commit", "-m", "changed fixture")
    else:
        manifest_path = folder / "consumer/manifest.json"
        manifest_path.chmod(0o644)
        manifest = json.loads(manifest_path.read_text())
        manifest["snapshot"]["commit"] = "f" * 40
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AdapterError, match="snapshot.*mismatch"):
        verify_consumer(folder, repository, expected_commit=expected)


def test_readonly_isolated_knowledge_mount(source):
    folder, repository, _ = source
    knowledge = export_consumer(folder, repository)
    # Empty filesystem with just shell dependencies and the task Markdown tree.
    command = ["bwrap", "--unshare-net", "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/bin", "/bin", "--ro-bind", "/lib", "/lib",
               "--ro-bind", "/lib64", "/lib64", "--ro-bind", str(knowledge), "/workspace/repo/docs/codewiki",
               "--", "/bin/sh", "-c",
               'test ! -e /workspace/repo/docs/codewiki/START.md && '
               'test -r /workspace/repo/docs/codewiki/module.md && '
               'test ! -e /workspace/repo/docs/codewiki/manifest.json && test ! -e /data && '
               'test ! -e /knowledge && '
               'if touch /workspace/repo/docs/codewiki/new.md 2>/dev/null; then exit 1; fi']
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.parametrize("worker_exit", [0, 1])
@pytest.mark.parametrize("target", ["tik-4560a8896f5f", "cli-d8763b93021c"])
def test_generate_automatically_exports_only_success(source, monkeypatch, worker_exit, target):
    from integration import run
    folder, repository, state = source
    state["commit"] = run.SNAPSHOTS[target][1]
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt.update(snapshot=state, target=target)
    (folder / "receipt.json").write_text(json.dumps(receipt))
    (folder / "docs/metadata.json").write_text(json.dumps({"generation_info": {"commit_id": state["commit"]}}))
    monkeypatch.setattr(run, "repository_for", lambda selected: repository if selected == target else pytest.fail("wrong target"))
    monkeypatch.setattr(run, "RUNS_ROOT", folder.parent / "outputs")
    monkeypatch.setattr(run, "git", lambda *args: run.SNAPSHOTS[target][0])
    monkeypatch.setenv("APIYI_API_KEY", "offline-fixture-key")
    monkeypatch.setattr(run, "stage_tokenizer_cache", lambda directory: directory)
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--mode", "generate", "--target", target])
    def process(command, **kwargs):
        if "--worker-run-dir" in command:
            assert kwargs["env"]["PYTHONPATH"] == str(run.ROOT / "CodeWiki")
            assert command[command.index("--target") + 1] == target
            destination = run.Path(command[command.index("--worker-run-dir") + 1])
            shutil.copytree(folder / "docs", destination / "docs")
            shutil.copyfile(folder / "receipt.json", destination / "receipt.json")
        return SimpleNamespace(returncode=worker_exit)
    monkeypatch.setattr(run.subprocess, "run", process)
    assert run.main() == worker_exit
    consumers = list((folder.parent / "outputs").rglob("consumer"))
    assert len(consumers) == (1 if worker_exit == 0 else 0)

    command_path, = (folder.parent / "outputs").rglob("command.json")
    assert target in str(command_path)
    command = json.loads(command_path.read_text())
    assert command["target"] == target
    assert command["snapshot"]["commit"] == run.SNAPSHOTS[target][1]
    if worker_exit == 0:
        manifest = json.loads((consumers[0] / "manifest.json").read_text())
        assert manifest["target"] == target
        assert manifest["snapshot"]["commit"] == run.SNAPSHOTS[target][1]


def test_export_uses_caller_bound_snapshot_and_rejects_wrong_commit(source, monkeypatch):
    from integration import run
    folder, repository, state = source
    calls = []
    def bound_snapshot(path, expected=run.COMMIT):
        calls.append((path, expected))
        if expected != state["commit"]:
            raise AdapterError("snapshot_commit_mismatch")
        return dict(state)
    monkeypatch.setattr(run, "snapshot", bound_snapshot)
    with pytest.raises(AdapterError, match="snapshot_commit_mismatch"):
        export_consumer(folder, repository, expected_commit="wrong")
    assert not (folder / "consumer").exists()
    knowledge = export_consumer(folder, repository, expected_commit=state["commit"])
    assert calls[-2:] == [(repository, state["commit"])] * 2
    assert verify_consumer(folder, repository, expected_commit=state["commit"]) == knowledge
    with pytest.raises(AdapterError, match="snapshot_commit_mismatch"):
        verify_consumer(folder, repository)
