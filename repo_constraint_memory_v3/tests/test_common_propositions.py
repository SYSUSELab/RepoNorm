from __future__ import annotations

from pathlib import Path

import pytest

from helpers import build_fixture, copied_fixture
from repo_constraint_memory_v3.chunking import chunk_sources
from repo_constraint_memory_v3.discovery import load_source_file
from repo_constraint_memory_v3.models import CandidateClaim
from repo_constraint_memory_v3.propositions import (
    PropositionEnvelope,
    normalize_explicit_proposition,
    validate_explicit_proposition_route,
    validate_proposition_references,
)


def test_explicit_normalization_is_category_neutral(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")
    claim = result.memory.claims[0]
    discovery = __import__(
        "repo_constraint_memory_v3.models",
        fromlist=["DiscoveryArtifact"],
    ).DiscoveryArtifact.model_validate_json(
        (tmp_path / "run" / "discovery.json").read_bytes()
    )
    metadata = next(
        item
        for item in discovery.baseline_sources
        if item.path == claim.evidence[0].path
    )
    source = load_source_file(repository, metadata, origin="deterministic")
    unit = chunk_sources([source], max_unit_chars=100_000)[0]
    candidate = CandidateClaim(
        rule=claim.rule,
        claim_kind=claim.claim_kind,
        normative_strength=claim.normative_strength,
        applies_to=claim.applies_to,
        condition=claim.condition,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[0].span_id,
                "span_end_id": unit.spans[0].span_id,
            }
        ],
    )
    first = normalize_explicit_proposition(
        snapshot_digest=result.memory.repository.snapshot_digest,
        unit=unit,
        candidate=candidate,
        derived_kind=claim.claim_kind,
        derived_strength=claim.normative_strength,
        canonical_scope=claim.applies_to,
        canonical_condition=claim.condition or "",
        grounded_evidence=claim.evidence,
        category="new repository-specific label",
    )
    second = normalize_explicit_proposition(
        snapshot_digest=result.memory.repository.snapshot_digest,
        unit=unit,
        candidate=candidate,
        derived_kind=claim.claim_kind,
        derived_strength=claim.normative_strength,
        canonical_scope=claim.applies_to,
        canonical_condition=claim.condition or "",
        grounded_evidence=claim.evidence,
        category="another unseen label",
    )
    assert first.proposition_id == second.proposition_id
    validate_explicit_proposition_route(first)
    manifests = {
        reference.kind: {
            item.stable_id
            for item in first.evidence_refs + first.inference_refs
            if item.kind == reference.kind
        }
        for reference in first.evidence_refs + first.inference_refs
    }
    validate_proposition_references(first, manifests)
    with pytest.raises(ValueError, match="unresolved"):
        validate_proposition_references(first, {})
    assert all(item.origin_channels == ["explicit"] for item in result.memory.claims)


def test_future_route_cannot_authorize_a_35_accepted_claim(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")
    claim = result.memory.claims[0]
    # A structurally valid future proposition is a common envelope, but the
    # Schema-3.6 accepted-claim gate must still reject it.
    payload = {
        "contract_version": "common-proposition-1.1",
        "snapshot_digest": result.memory.repository.snapshot_digest,
        "proposition_id": "PROP-" + "1" * 24,
        "rule": claim.rule,
        "category": "unseen",
        "proposed_claim_kind": "observed_convention",
        "proposed_normative_strength": "observed",
        "applies_to": ["**/*.py"],
        "condition": None,
        "origin_channels": ["implicit_current"],
        "evidence_routes": ["current_semantic_population"],
        "evidence_refs": [],
        "inference_refs": [],
        "population_refs": [],
        "limitation_refs": [],
        "completeness_refs": ["future"],
        "provenance": ["test"],
    }
    from repo_constraint_memory_v3.observations.canonical import payload_digest

    payload["artifact_digest"] = payload_digest(payload)
    proposition = PropositionEnvelope.model_validate(payload)
    with pytest.raises(ValueError):
        validate_explicit_proposition_route(proposition)
