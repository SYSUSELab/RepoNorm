"""Host binding and clean projection tests; no target code or model is executed."""
import hashlib
from pathlib import Path
import stat

import pytest

from evaluation.consumer_handoff import prepare_consumer_handoff
from repo_constraint_memory_v3.unified.publication import directory_tree_fingerprint
from test_consumer_cli import _mock_source_run


@pytest.fixture
def source(tmp_path):
    run, repository, _ = _mock_source_run(tmp_path / "source", incomplete=True)
    return run, repository, run / "consumers" / "accepted_only"


def test_partial_handoff_is_clean_readonly_and_does_not_run_replay(source, tmp_path, monkeypatch):
    run, repository, view = source
    import repo_constraint_memory_v3.unified.validation as validation
    monkeypatch.setattr(validation, "validate_unified_directory", lambda *a, **k: pytest.fail("full replay is not a gate"))
    output = tmp_path / "agent" / "knowledge"
    receipt = prepare_consumer_handoff(view, source_run=run, repository_root=repository, output_directory=output)
    assert receipt["source_manifest"]["package_status"] == "partial"
    assert receipt["task_tree_fingerprint"] == directory_tree_fingerprint(output, max_bytes=1_000_000_000)
    assert receipt["task_tree_digest"] == receipt["source_manifest"]["task_tree_digest"]
    assert (output / "START.md").is_file()
    assert all(path.suffix == ".md" for path in output.rglob("*") if path.is_file())
    assert all(not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for path in [output, *output.rglob("*")])
    assert not (output / "consumer-manifest.json").exists()
    # Agent edits are ordinary task work, not a request for a new binding check.
    (repository / "task-edit.py").write_text("result = 1\n", encoding="utf-8")
    assert (output / "START.md").is_file()


@pytest.mark.parametrize("damage", ["content", "extra", "snapshot", "symlink"])
def test_handoff_rejects_bad_inputs_without_publishing(source, tmp_path, damage):
    run, repository, view = source
    if damage == "content":
        path = view / "START.md"
        path.chmod(0o644)
        path.write_text("tampered", encoding="utf-8")
    elif damage == "extra":
        view.chmod(0o755)
        (view / "extra.md").write_text("foreign", encoding="utf-8")
    elif damage == "symlink":
        path = view / "START.md"
        view.chmod(0o755)
        path.unlink()
        path.symlink_to(repository / "README.md")
    else:
        (repository / "unexpected.py").write_text("changed = True\n", encoding="utf-8")
    output = tmp_path / "task-knowledge"
    with pytest.raises(ValueError):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository, output_directory=output)
    assert not output.exists()


def test_handoff_rejects_overlapping_and_existing_destinations(source, tmp_path):
    run, repository, view = source
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    for output in [repository / "knowledge", run / "knowledge", occupied]:
        with pytest.raises(ValueError):
            prepare_consumer_handoff(view, source_run=run, repository_root=repository, output_directory=output)


def test_handoff_rechecks_snapshot_before_publication(source, tmp_path, monkeypatch):
    run, repository, view = source
    import evaluation.consumer_handoff as handoff
    original = handoff.directory_tree_fingerprint
    changed = False
    def fingerprint(path, **kwargs):
        nonlocal changed
        result = original(path, **kwargs)
        if ".handoff-" in path.name and not changed:
            changed = True
            (repository / "concurrent.py").write_text("changed = True\n", encoding="utf-8")
        return result
    monkeypatch.setattr(handoff, "directory_tree_fingerprint", fingerprint)
    output = tmp_path / "task-knowledge"
    with pytest.raises(ValueError):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository, output_directory=output)
    assert not output.exists()


def test_handoff_rejects_copy_corruption_even_if_source_stays_valid(source, tmp_path, monkeypatch):
    run, repository, view = source
    original = Path.write_bytes
    def write(path, data):
        if any(".handoff-" in part for part in path.parts) and path.name == "START.md":
            data = b"transient untrusted bytes\n"
        return original(path, data)
    monkeypatch.setattr(Path, "write_bytes", write)
    output = tmp_path / "task-knowledge"
    with pytest.raises(ValueError, match="staging differs"):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository, output_directory=output)
    assert not output.exists()


def test_benchmark_handoff_is_independent_atomic_authenticated_projection(source, tmp_path, monkeypatch):
    run, repository, view = source
    import repo_constraint_memory_v3.unified.validation as validation
    monkeypatch.setattr(validation, "validate_unified_directory", lambda *a, **k: pytest.fail("full replay is not a gate"))
    original = directory_tree_fingerprint(view, max_bytes=1_000_000_000)
    output = tmp_path / "benchmark-knowledge"
    receipt = prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                                      output_directory=output, benchmark=True)
    assert receipt["schema_version"] == "task-consumer-handoff-2.0"
    assert receipt["projection_version"] == "benchmark-consumer-surface-1.0"
    assert receipt["source_manifest"]["package_status"] == "partial"
    assert receipt["source_task_tree_digest"] == receipt["source_manifest"]["task_tree_digest"]
    assert receipt["task_tree_digest"] != receipt["source_task_tree_digest"]
    assert receipt["task_tree_fingerprint"] == directory_tree_fingerprint(output, max_bytes=1_000_000_000)
    assert directory_tree_fingerprint(view, max_bytes=1_000_000_000) == original
    start = (output / "START.md").read_text()
    assert not any(word in start for word in ("omit", "incomplete", "SKILL", "by-path"))
    for path in output.rglob("*"):
        assert not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        if path.is_file():
            assert path.suffix == ".md"
    for detail in (view / "claims").glob("*.md"):
        assert (output / "claims" / detail.name).read_bytes() == detail.read_bytes()
    assert not (output / "consumer-manifest.json").exists()
    actual_files = {path.relative_to(output).as_posix(): path.read_bytes()
                    for path in output.rglob("*.md")}
    assert set(receipt["projection"]["files"]) == set(actual_files)
    for relative, data in actual_files.items():
        row = receipt["projection"]["files"][relative]
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["size_bytes"] == len(data)
        if relative.startswith("claims/"):
            assert row["claim_ids"] == [Path(relative).stem]
        elif relative == "START.md":
            assert row["claim_ids"] == []
        else:
            for claim_id in row["claim_ids"]:
                assert claim_id in data.decode()


@pytest.mark.parametrize("damage", ["source", "repository", "staging"])
def test_benchmark_handoff_rejects_tamper_without_publication(source, tmp_path, monkeypatch, damage):
    run, repository, view = source
    output = tmp_path / "benchmark-knowledge"
    if damage == "source":
        path = view / "START.md"
        path.chmod(0o644)
        path.write_text("foreign")
    elif damage == "repository":
        (repository / "foreign.py").write_text("foreign")
    else:
        original = Path.write_bytes
        def write(path, data):
            if ".handoff-" in str(path) and path.name == "START.md":
                data = b"corrupted staging"
            return original(path, data)
        monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(ValueError):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                                 output_directory=output, benchmark=True)
    assert not output.exists()


def test_benchmark_reviewed_handoff_keeps_frozen_membership_and_acceptance(source, tmp_path):
    from repo_constraint_memory_v3.review.service import create_review, publish_review, save_decision
    run, repository, _ = source
    review = tmp_path / "review"
    session = create_review(run, repository_root=repository, output_directory=review)
    claim_id = next(iter(session.claim_digests))
    save_decision(review, repository_root=repository, claim_id=claim_id, decision="accept")
    plan = publish_review(review, repository_root=repository)
    view = plan.publication_directory / "consumer"
    output = tmp_path / "reviewed-benchmark"
    receipt = prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                                      review_directory=review, output_directory=output, benchmark=True)
    assert receipt["source_manifest"]["consumer_profile"] == "accepted_after_review"
    assert receipt["projection"]["claims"][claim_id]["source_epistemic_status"] == "review"
    assert receipt["projection"]["claims"][claim_id]["delivery_policy"] != "not_applicable"
    detail = f"claims/{claim_id}.md"
    assert (output / detail).read_bytes() == (view / detail).read_bytes()
    assert "developer_confirmation" in (output / detail).read_text()
    assert "Unconfirmed (review)" not in (output / "catalog.md").read_text()
    assert set(receipt["projection"]["claims"]) == {path.stem for path in (view / "claims").glob("*.md")}


def test_benchmark_handoff_requires_exact_projection_regeneration(source, tmp_path, monkeypatch):
    import evaluation.consumer_handoff as handoff
    from evaluation.benchmark_projection import BenchmarkProjection
    run, repository, view = source
    original = handoff.render_benchmark_projection
    calls = 0

    def inconsistent(*args, **kwargs):
        nonlocal calls
        calls += 1
        plan = original(*args, **kwargs)
        if calls == 2:
            return BenchmarkProjection(plan.files, {**plan.metadata, "unexpected": True})
        return plan

    monkeypatch.setattr(handoff, "render_benchmark_projection", inconsistent)
    output = tmp_path / "inconsistent-projection"
    with pytest.raises(ValueError, match="projection changed during handoff regeneration"):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                                 output_directory=output, benchmark=True)
    assert calls == 2
    assert not output.exists()


def test_benchmark_reuses_each_authentication_typed_claims_without_extra_source_load(source, tmp_path, monkeypatch):
    import evaluation.consumer_handoff as handoff
    run, repository, view = source
    original_load = handoff.surface._load_standalone_inputs
    original_render = handoff.render_benchmark_projection
    loaded_claims = []
    rendered_claims = []

    def load(*args, **kwargs):
        inputs = original_load(*args, **kwargs)
        loaded_claims.append(inputs.canonical.claims)
        return inputs

    def render(claims, **kwargs):
        rendered_claims.append(claims)
        return original_render(claims, **kwargs)

    monkeypatch.setattr(handoff.surface, "_load_standalone_inputs", load)
    monkeypatch.setattr(handoff, "render_benchmark_projection", render)
    prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                             output_directory=tmp_path / "cached-projection", benchmark=True)
    assert len(loaded_claims) == len(rendered_claims) == 2
    for loaded, rendered in zip(loaded_claims, rendered_claims, strict=True):
        assert all(left is right for left, right in zip(loaded, rendered, strict=True))


def test_benchmark_checks_source_metadata_after_final_in_memory_render(source, tmp_path, monkeypatch):
    import evaluation.consumer_handoff as handoff
    run, repository, view = source
    original = handoff.render_benchmark_projection
    calls = 0

    def render(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            (run / "concurrent-foreign.md").write_text("changed after final authentication")
        return result

    monkeypatch.setattr(handoff, "render_benchmark_projection", render)
    output = tmp_path / "source-raced-projection"
    with pytest.raises(ValueError, match="source changed during rendering"):
        prepare_consumer_handoff(view, source_run=run, repository_root=repository,
                                 output_directory=output, benchmark=True)
    assert calls == 2
    assert not output.exists()
