from __future__ import annotations

from pathlib import Path

import pytest

from repo_constraint_memory_v3.review.models import ReviewEvidence, ReviewSession
from repo_constraint_memory_v3.review.service import (
    ReviewDecisionConflictError,
    create_review, load_review, preflight_review_consumer, publish_review, review_queue, save_decision,
    validate_review_consumer,
)
from repo_constraint_memory_v3.unified import consumer
from repo_constraint_memory_v3.unified.ids import seal_payload
from repo_constraint_memory_v3.unified.publication import directory_tree_fingerprint
from test_consumer_cli import _mock_source_run


@pytest.fixture(scope="module")
def review_source(tmp_path_factory):
    return _mock_source_run(tmp_path_factory.mktemp("review-source"), incomplete=True)


def test_review_independent_rounds_partial_union_and_immutable_publications(review_source, tmp_path, monkeypatch):
    source, repository, canonical = review_source
    before = directory_tree_fingerprint(source, max_bytes=1_000_000_000)
    session = create_review(source, repository_root=repository, output_directory=tmp_path / "round-one")
    second = create_review(source, repository_root=repository, output_directory=tmp_path / "round-two")
    assert session.session_id != second.session_id
    assert session.claim_digests and session.progress()["unprocessed"] == len(session.claim_digests)
    ids = list(session.claim_digests)
    updated = save_decision(tmp_path / "round-one", repository_root=repository, claim_id=ids[0], decision="accept")
    assert updated.progress()["accepted"] == 1
    assert load_review(tmp_path / "round-two").progress()["accepted"] == 0

    def forbidden(*args, **kwargs):
        pytest.fail("review publication must not invoke full source replay")

    import repo_constraint_memory_v3.unified.validation as validation
    original = validation.validate_unified_directory
    monkeypatch.setattr(validation, "validate_unified_directory", forbidden)
    plan = publish_review(tmp_path / "round-one", repository_root=repository)
    assert plan.publication_directory == tmp_path / "round-one/publications/000001"
    assert preflight_review_consumer(
        plan.publication_directory / "consumer", review_directory=tmp_path / "round-one",
        repository_root=repository, source_run=source,
    ) == plan.manifest
    monkeypatch.setattr(validation, "validate_unified_directory", original)
    view = tmp_path / "round-one/publications/000001/consumer"
    assert plan.manifest.review_accepted_count == 1
    assert plan.manifest.review_rejected_count == 0
    assert plan.manifest.package_status == "partial"
    accepted = {claim.claim_id for claim in canonical.claims if claim.epistemic_status == "accepted"}
    assert {file.stem for file in (view / "claims").iterdir()} == accepted | {ids[0]}
    text = (view / f"claims/{ids[0]}.md").read_text()
    assert "developer_confirmation" in text
    assert "not_applicable" not in text
    assert "technical_or_coverage_gap" not in text
    assert not (view / "decisions.json").exists()
    assert not (view / "canonical").exists()
    assert validate_review_consumer(view, review_directory=tmp_path / "round-one", repository_root=repository) == []
    assert validate_review_consumer(view, review_directory=tmp_path / "round-one", repository_root=repository, require_complete=True)
    frozen = directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000)
    save_decision(tmp_path / "round-one", repository_root=repository, claim_id=ids[0], decision="reject")
    publish_review(tmp_path / "round-one", repository_root=repository)
    assert (tmp_path / "round-one/publications/000002/consumer").exists()
    assert directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000) == frozen
    assert validate_review_consumer(view, review_directory=tmp_path / "round-one", repository_root=repository) == []
    assert directory_tree_fingerprint(source, max_bytes=1_000_000_000) == before


def test_review_binding_rejects_unknown_claim_tampering_and_snapshot_change(review_source, tmp_path):
    source, repository, _ = review_source
    directory = tmp_path / "review"
    session = create_review(source, repository_root=repository, output_directory=directory)
    with pytest.raises(ValueError, match="canonical review"):
        save_decision(directory, repository_root=repository, claim_id="CLAIM-" + "0" * 24, decision="accept")
    payload = session.model_dump(mode="json")
    payload["canonical_claims_digest"] = "0" * 64
    forged = ReviewSession.model_validate(seal_payload(payload))
    (directory / "session.json").write_text(consumer.canonical_json_text(forged))
    with pytest.raises(ValueError, match="identity"):
        load_review(directory)
    (directory / "session.json").write_text(consumer.canonical_json_text(session))
    changed = repository / "review-snapshot-change.py"
    try:
        changed.write_text("CHANGED = True\n")
        with pytest.raises(ValueError, match="snapshot"):
            publish_review(directory, repository_root=repository)
    finally:
        changed.unlink()


def test_review_queue_evidence_validation_and_tampered_view(review_source, tmp_path):
    source, repository, _ = review_source
    directory = tmp_path / "review"
    session = create_review(source, repository_root=repository, output_directory=directory)
    queue = review_queue(directory, repository_root=repository)
    assert len(queue["claims"]) == session.progress()["planned"]
    assert queue["claims"][0]["review_explanations"]
    assert "detail_markdown" in queue["claims"][0]
    claim_id = next(iter(session.claim_digests))
    with pytest.raises(ValueError, match="repository file"):
        save_decision(directory, repository_root=repository, claim_id=claim_id, decision="accept", evidence=[ReviewEvidence(path="absent.py", start_line=1, end_line=1)])
    with pytest.raises(ValueError, match="rationale"):
        save_decision(directory, repository_root=repository, claim_id=claim_id, decision="reject", reviewer_kind="model")
    publish_review(directory, repository_root=repository)
    view = directory / "publications/000001/consumer"
    (view / "extra.md").write_text("unexpected")
    assert validate_review_consumer(view, review_directory=directory, repository_root=repository)
    with pytest.raises(ValueError, match="fresh"):
        publish_review(directory, repository_root=repository, output_directory=view.parent)


def test_model_decisions_bind_receipt_and_published_audit_is_independent(review_source, tmp_path):
    source, repository, _ = review_source
    directory = tmp_path / "review"
    session = create_review(source, repository_root=repository, output_directory=directory)
    claim_id = next(iter(session.claim_digests))
    receipt = directory / "receipt.json"
    receipt.write_text(consumer.canonical_json_text({"effective_model": "fake-review-model"}))
    evidence_path = next(path for path in sorted(repository.rglob("*.py")) if path.read_text().splitlines())
    saved = save_decision(
        directory, repository_root=repository, claim_id=claim_id,
        decision="accept", reviewer_kind="model", reviewer_id="fake-review-model",
        rationale="The repository implements this convention.",
        evidence=[ReviewEvidence(path=evidence_path.relative_to(repository).as_posix(), start_line=1, end_line=1)],
        execution_ref="receipt.json",
    )
    assert saved.latest_decisions()[claim_id].execution_digest
    assert saved.latest_decisions()[claim_id].evidence[0].content_digest
    publish_review(directory, repository_root=repository)
    view = directory / "publications/000001/consumer"
    assert "model_review" in (view / f"claims/{claim_id}.md").read_text()
    assert not (view / "receipt.json").exists()
    receipt.write_text("tampered")
    with pytest.raises(ValueError, match="execution"):
        publish_review(directory, repository_root=repository)
    assert validate_review_consumer(view, review_directory=directory, repository_root=repository) == []
    audit = view.parent / "audit/receipt.json"
    audit.chmod(0o644)
    audit.write_text("tampered")
    assert validate_review_consumer(view, review_directory=directory, repository_root=repository)


def test_review_projection_preserves_conditions_risk_and_original_status():
    from types import SimpleNamespace
    from test_consumer_surface import _bundle, _pair
    from repo_constraint_memory_v3.review.models import ReviewDecision
    from repo_constraint_memory_v3.review.render import render_review_consumer

    pairs = [
        _pair("review-condition", status="review", delivery="not_applicable", facets=("factual_uncertainty",), condition="When changing the parser", exception="Except generated files"),
        _pair("review-risk", status="review", delivery="not_applicable", facets=("technical_or_coverage_gap",), risk="external_side_effect"),
        _pair("unprocessed", status="review", delivery="not_applicable", facets=("factual_uncertainty",)),
        _pair("accepted"),
    ]
    bundle = _bundle(pairs, partial=True)
    canonical = consumer.make_canonical_claims_file(bundle)
    completeness = consumer.make_canonical_completeness_file(bundle)
    inputs = SimpleNamespace(bundle=bundle, canonical=canonical, completeness=completeness, git_artifacts={})
    decisions = [ReviewDecision(claim_id=claim.claim_id, claim_digest=claim.artifact_digest, decision="accept", reviewer_kind="developer", reviewer_id="developer", decided_at="2026-09-08T00:00:00+00:00", version=1).model_dump(mode="json") for claim, _ in pairs[:2]]
    session = ReviewSession.model_validate(seal_payload({
        "format_version": "developer-review-1.0", "session_id": "REVIEW-" + "a" * 32,
        "source_run": "/fixture/source", "repository_root": "/fixture/repository", "repository": "fixture",
        "snapshot_digest": canonical.snapshot_digest, "snapshot_commit": canonical.snapshot_commit,
        "canonical_claims_digest": canonical.artifact_digest, "completeness_digest": completeness.artifact_digest,
        "source_tree_digest": "f" * 64, "created_at": "2026-09-08T00:00:00+00:00",
        "claim_digests": {claim.claim_id: claim.artifact_digest for claim in canonical.claims if claim.epistemic_status == "review"},
        "decisions": decisions,
    }))
    plan = render_review_consumer(inputs, session)
    assert plan.manifest.exposed_accepted_count == 3
    assert plan.manifest.review_unprocessed_count == 1
    conditional = plan.files[f"claims/{pairs[0][0].claim_id}.md"].decode()
    assert "Normative strength" in conditional
    assert "When changing the parser" in conditional
    assert "Except generated files" in conditional
    risky = plan.files[f"claims/{pairs[1][0].claim_id}.md"].decode()
    assert "reference_only" in risky
    path_bytes = b"".join(data for path, data in plan.files.items() if path.startswith("by-path/"))
    assert pairs[1][0].claim_id.encode() not in path_bytes
    assert all(claim.epistemic_status == "review" and claim.delivery_policy == "not_applicable" for claim, _ in pairs[:3])
    assert plan.manifest.consumer_surface_version == "review-consumer-surface-2.0"
    start = plan.files["START.md"].decode()
    assert "may omit repository conventions" in start
    assert all(word not in start.lower() for word in ("snapshot", "profile", "audit", "validation", "digest"))
    for path, data in plan.files.items():
        if not path.endswith(".md"):
            continue
        text = data.decode()
        assert "**Status:**" not in text
        assert "Condition:** none" not in text
        assert "Evidence items:" not in text and "Population support:" not in text
        if not path.startswith("claims/"):
            assert "Acceptance source" not in text
    catalog = plan.files["catalog.md"].decode()
    for claim, _ in (pairs[0], pairs[1], pairs[3]):
        assert claim.semantics.rule in catalog
        assert all(scope in catalog for scope in claim.semantics.applies_to)
    assert "external_side_effect" in risky
    assert "developer_confirmation" in conditional



def test_review_rejects_symlink_outputs_and_foreign_publication_artifacts(review_source, tmp_path):
    source, repository, _ = review_source
    link = tmp_path / "source-link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        create_review(link, repository_root=repository, output_directory=tmp_path / "bad")
    with pytest.raises(ValueError, match="protected"):
        create_review(source, repository_root=repository, output_directory=source / "review")
    directory = tmp_path / "review"
    create_review(source, repository_root=repository, output_directory=directory)
    publish_review(directory, repository_root=repository)
    view = directory / "publications/000001/consumer"
    (view.parent / "foreign.txt").write_text("not in frozen publication")
    errors = validate_review_consumer(view, review_directory=directory, repository_root=repository)
    assert any("foreign" in error for error in errors)


def test_review_preflight_binds_selected_session_and_consumer_without_replay(review_source, tmp_path, monkeypatch):
    source, repository, _ = review_source
    directory = tmp_path / "review"
    create_review(source, repository_root=repository, output_directory=directory)
    other = tmp_path / "other-review"
    create_review(source, repository_root=repository, output_directory=other)
    plan = publish_review(directory, repository_root=repository)
    view = plan.publication_directory / "consumer"
    before = directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000)
    import repo_constraint_memory_v3.unified.validation as validation

    def forbidden(*args, **kwargs):
        pytest.fail("consumer preflight must not replay source")

    monkeypatch.setattr(validation, "validate_unified_directory", forbidden)
    assert preflight_review_consumer(view, review_directory=directory, repository_root=repository) == plan.manifest
    assert directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000) == before
    with pytest.raises(ValueError, match="different review session"):
        preflight_review_consumer(view, review_directory=other, repository_root=repository)
    with pytest.raises(ValueError, match="authorized source"):
        preflight_review_consumer(view, review_directory=directory, repository_root=repository, source_run=tmp_path)
    with pytest.raises(ValueError, match="consumer directory"):
        preflight_review_consumer(view / "claims", review_directory=directory, repository_root=repository)
    with pytest.raises(ValueError, match="existing publication"):
        publish_review(directory, repository_root=repository, output_directory=view.parent / "nested")
    (view / "catalog.md").chmod(0o644)
    (view / "catalog.md").write_text("tampered")
    with pytest.raises(ValueError, match="invalid review consumer"):
        preflight_review_consumer(view, review_directory=directory, repository_root=repository)


def test_copied_session_inside_repository_never_writes_a_lock(review_source, tmp_path):
    import shutil

    source, repository, _ = review_source
    directory = tmp_path / "review"
    session = create_review(source, repository_root=repository, output_directory=directory)
    copied = repository / "copied-review"
    try:
        shutil.copytree(directory, copied)
        with pytest.raises(ValueError, match="protected input"):
            save_decision(copied, repository_root=repository, claim_id=next(iter(session.claim_digests)), decision="accept")
        assert not (copied / ".review.lock").exists()
    finally:
        shutil.rmtree(copied)


def test_review_evidence_hash_catches_content_change_outside_original_snapshot(tmp_path):
    from repo_constraint_memory_v3.review.service import _check_evidence

    evidence_file = tmp_path / "CONVENTIONS.md"
    evidence_file.write_text("Use the common test fixture.\n")
    refs = _check_evidence(tmp_path, [ReviewEvidence(path="CONVENTIONS.md", start_line=1, end_line=1)], bind=True)
    assert refs[0].content_digest
    evidence_file.write_text("Each test must build its own fixture.\n")
    with pytest.raises(ValueError, match="content differs"):
        _check_evidence(tmp_path, refs)


def test_conditional_decision_save_preserves_concurrent_developer_choice(review_source, tmp_path):
    source, repository, _ = review_source
    directory = tmp_path / "review"
    session = create_review(source, repository_root=repository, output_directory=directory)
    claim_id = next(iter(session.claim_digests))
    developer = save_decision(
        directory, repository_root=repository, claim_id=claim_id,
        decision="reject", rationale="Developer does not endorse this rule.",
        expected_version=0,
    )
    before = (directory / "session.json").read_bytes()
    with pytest.raises(ReviewDecisionConflictError, match="changed"):
        save_decision(
            directory, repository_root=repository, claim_id=claim_id,
            decision="accept", expected_version=0,
        )
    assert (directory / "session.json").read_bytes() == before
    assert load_review(directory).latest_decisions()[claim_id] == developer.latest_decisions()[claim_id]
    revised = save_decision(
        directory, repository_root=repository, claim_id=claim_id,
        decision="accept", expected_version=1,
    )
    assert revised.latest_decisions()[claim_id].version == 2


def test_review_catalog_shards_are_bounded_readable_and_project_accepted_rules():
    from test_consumer_surface import _pair
    from repo_constraint_memory_v3.review.render import ReviewProjection, _catalog

    items = []
    for index in range(60):
        claim, _ = _pair(f"catalog-{index}")
        semantics = claim.semantics.model_copy(update={
            "rule": f"Rule {index}: " + "preserve_repository_convention " * 30,
        })
        items.append(ReviewProjection(claim.model_copy(update={"semantics": semantics}), None))
    files = _catalog(items)
    shards = {path: data for path, data in files.items() if path.startswith("catalog/")}
    assert len(shards) > 1
    assert all(len(data) <= consumer.MAX_CATALOG_FILE_BYTES for data in files.values())
    index_text = files["catalog.md"].decode()
    assert "Rules 1–" in index_text and "scope preview:" in index_text
    assert all(item.claim.claim_id not in index_text for item in items)
    content = b"".join(shards.values()).decode()
    for item in items:
        assert content.count(item.claim.semantics.rule) == 1
        assert f"{item.claim.claim_id}.md" in content
    assert "**Status:**" not in content and "Acceptance source" not in content
