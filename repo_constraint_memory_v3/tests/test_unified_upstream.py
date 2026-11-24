from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from helpers import config_for_tests
from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.explicit_verification import (
    ExplicitVerificationCoverage,
    ExplicitVerificationJudgment,
    ExplicitVerificationResult,
    apply_explicit_verification_policy,
    explicit_candidate_root_id,
)
from repo_constraint_memory_v3.inference import InferenceBudget
from repo_constraint_memory_v3.models import (
    CandidateClaim,
    Evidence,
    EvidenceRelation,
    EvidenceSpan,
    EvidenceUnit,
)
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.propositions import normalize_explicit_proposition
from repo_constraint_memory_v3.semantic.validation import CurrentSourceIndex
from repo_constraint_memory_v3.source_adapters import GroundedEvidence
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    canonical_bytes,
)
from repo_constraint_memory_v3.unified.pipeline import run_unified_layer
from repo_constraint_memory_v3.unified.inputs import (
    ASSESSMENT_CONSTRUCTION_ATTRITION,
    TERMINAL_INVALID_SCOPE_ATTRITION,
)
from repo_constraint_memory_v3.unified.upstream import (
    UnifiedUpstreamInputs,
    UnifiedUpstreamLoadError,
    canonical_unified_upstream_bytes,
    load_unified_upstream_inputs,
    make_unified_upstream_inputs,
)


SNAPSHOT = "sha256:" + "a" * 64
PROFILE = "b" * 64
PROTOCOL_MANIFEST = "c" * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decision() -> CandidateDecision:
    quote = "Contributors must run pytest."
    unit_id = "UNIT-" + _digest("unit")[:24]
    span_id = "SPAN-" + _digest("span")[:24]
    evidence = Evidence(
        type="direct_declaration",
        adapter="prose",
        semantic_key="instruction:pytest",
        path="AGENTS.md",
        span_ids=[span_id],
        line_start=7,
        line_end=7,
        quote=quote,
        quote_sha256=_digest(quote),
        source_content_sha256=_digest("whole source"),
        source_kind="instruction",
        discovery_origin="deterministic",
    )
    grounded = GroundedEvidence(
        evidence=evidence,
        semantic_flags=frozenset({"prose", "instruction_section:testing"}),
        contributor_context=True,
        hard_modal=True,
        soft_modal=False,
        hard_directive=True,
        soft_directive=False,
        example_context=False,
    )
    candidate = CandidateClaim(
        rule="Contributors must run pytest.",
        claim_kind="normative_declaration",
        normative_strength="required",
        applies_to=["src/**"],
        condition=None,
        evidence=[
            {
                "unit_id": unit_id,
                "span_start_id": span_id,
                "span_end_id": span_id,
            }
        ],
    )
    unit = EvidenceUnit(
        unit_id=unit_id,
        path="AGENTS.md",
        source_kind="instruction",
        discovery_origin="deterministic",
        line_start=7,
        line_end=7,
        default_scope=["src/**"],
        content=quote,
        content_sha256=_digest(quote),
        source_content_sha256=evidence.source_content_sha256,
        spans=[EvidenceSpan(span_id=span_id, line_number=7, text=quote)],
    )
    proposition = normalize_explicit_proposition(
        snapshot_digest=SNAPSHOT,
        unit=unit,
        candidate=candidate,
        derived_kind="normative_declaration",
        derived_strength="required",
        canonical_scope=["src/**"],
        canonical_condition="",
        grounded_evidence=[evidence],
        inference_request_id="REQ-explicit-1",
        category="testing",
    )
    fingerprint = _digest("candidate")
    return CandidateDecision(
        unit_id=unit_id,
        candidate_fingerprint=fingerprint,
        outcome="accepted",
        outcome_stage="explicit_semantic_verify",
        reason_code="direct_norm",
        raw_candidate=candidate,
        errors=["retained host diagnostic"],
        grounded=[grounded],
        relations=[
            EvidenceRelation(
                relation="corroborates",
                route="explicit_typed_evidence",
                target_fingerprint=_digest("other candidate"),
                reason="same repository instruction",
                effect="advisory",
            )
        ],
        derived_kind="normative_declaration",
        derived_strength="required",
        acceptance_basis=["direct_modal_text", "direct_instruction_text"],
        semantic_core="contributors run pytest",
        canonical_scope=("src/**",),
        canonical_condition="",
        condition_source="host_normalized",
        host_eligible=True,
        verification_disposition="eligible",
        verification_request_id="REQ-explicit-1",
        verification_reason_codes=["verified_eligible"],
        verification_technical_reason_code=None,
        claim_id="RCM3-" + _digest("claim")[:24],
        proposition=proposition,
    )


def _verification(decision: CandidateDecision) -> ExplicitVerificationResult:
    root_id = explicit_candidate_root_id(decision.candidate_fingerprint)
    judgment = ExplicitVerificationJudgment(
        git_need={"status": "no_established_need", "questions": []},
        candidate_root_id=root_id,
        coding_relevance="pass",
        evidence_entailment="pass",
        scope_condition_support="pass",
        ambiguity="none",
    )
    policy = apply_explicit_verification_policy(root_id, judgment)
    coverage = ExplicitVerificationCoverage(
        status="complete",
        grounded_roots=1,
        planned_roots=1,
        verified_roots=1,
        technical_failure_roots=0,
        omitted_roots=0,
        planned_batches=1,
        successful_batches=1,
        failed_batches=0,
        policy_eligible=1,
        policy_review=0,
        policy_rejected=0,
        incomplete_reasons=[],
    )
    payload: dict[str, object] = {
        "result_version": "explicit-semantic-verification-result-1.1",
        "policy_version": "explicit-semantic-verification-policy-1.0",
        "plan_digest": _digest("verification plan"),
        "decisions": [policy.model_dump(mode="json")],
        "coverage": coverage.model_dump(mode="json"),
    }
    payload["result_digest"] = payload_digest(payload)
    return ExplicitVerificationResult.model_validate(payload)


def _artifact(
    *,
    current_source=None,
    observations=None,
) -> tuple[CandidateDecision, UnifiedUpstreamInputs]:
    decision = _decision()
    return decision, make_unified_upstream_inputs(
        snapshot_digest=SNAPSHOT,
        profile_digest=PROFILE,
        snapshot_commit=None,
        explicit_decisions=[decision],
        explicit_verification=_verification(decision),
        current_source=current_source,
        observations=observations,
        explicit_discovery_complete=True,
        raw_candidate_events=9,
    )


def _write_artifact(tmp_path, artifact: UnifiedUpstreamInputs):
    path = tmp_path / "unified_upstream_inputs.json"
    path.write_bytes(canonical_unified_upstream_bytes(artifact))
    return path


def _empty_current_source():
    index_payload: dict[str, object] = {
        "index_version": "current-source-index-1.10",
        "snapshot_digest": SNAPSHOT,
        "required": True,
        "status": "complete",
        "semantic_map_digest": _digest("semantic map"),
        "hypothesis_reduction_digest": _digest("reduction"),
        "semantic_observations": 0,
        "candidate_events": 0,
        "unique_candidate_roots": 0,
        "hypothesis_versions": 0,
        "complete_populations": 0,
        "partial_populations": 0,
        "failed_populations": 0,
        "classified_members": 0,
        "population_matched_members": 0,
        "population_inspected_members": 0,
        "population_uninspected_members": 0,
        "population_provider_eligible_members": 0,
        "population_provider_inspected_members": 0,
        "population_provider_uninspected_members": 0,
        "census_populations": 0,
        "sampled_populations": 0,
        "technical_unknown_members": 0,
        "completed_falsifications": 0,
        "partial_falsifications": 0,
        "refinements": 0,
        "propositions": 0,
        "root_hypothesis": 0,
        "root_review": 0,
        "root_rejected": 0,
        "root_accepted": 0,
        "experimental_accept_candidates": 0,
        "experimental_review": 0,
        "experimental_rejected": 0,
        "inference_ids": [],
        "artifact_set_digest": _digest("artifact set"),
        "population_schedule": [],
        "classification_admission_schedule": [],
        "classification_transport_plans": [],
        "resource_use": {},
        "incomplete_reasons": [],
        "limitations": [],
    }
    index_payload["index_digest"] = payload_digest(index_payload)
    index = CurrentSourceIndex.model_validate(index_payload)
    return SimpleNamespace(
        index=index,
        semantic_map=SimpleNamespace(observations=()),
        hypotheses=(),
        populations=(),
        member_packets=(),
        falsifications=(),
        refinements=(),
        policy_decisions=(),
    )


def test_full_candidate_and_minimal_channel_views_round_trip(tmp_path) -> None:
    current = _empty_current_source()
    observations = SimpleNamespace(
        index=SimpleNamespace(
            snapshot_digest=SNAPSHOT,
            observation_set_digest=_digest("observation set"),
            overall="complete",
            incomplete_reasons=[],
        ),
        shards=(),
    )
    decision, artifact = _artifact(
        current_source=current,
        observations=observations,
    )
    assert artifact.input_version == "unified-upstream-inputs-1.1"
    upstream_schema = UnifiedUpstreamInputs.model_json_schema()
    field_schema = upstream_schema["properties"]["explicit_discovery_complete"]
    assert "Candidate-enumeration completeness only" in field_schema[
        "description"
    ]
    path = _write_artifact(tmp_path, artifact)

    runtime = load_unified_upstream_inputs(
        path,
        expected_snapshot_digest=SNAPSHOT,
        expected_profile_digest=PROFILE,
    )

    assert runtime.explicit_decisions == (decision,)
    assert runtime.explicit_verification == _verification(decision)
    assert runtime.raw_candidate_events == 9
    assert runtime.current_source is not None
    assert runtime.current_source.index == current.index
    assert runtime.current_source.semantic_map.observations == ()
    assert runtime.observations is not None
    assert runtime.observations.snapshot_digest == SNAPSHOT
    assert runtime.observations.overall == "complete"
    assert runtime.observations.incomplete_reasons == ()
    assert runtime.observations.shards == ()
    assert runtime.pipeline_kwargs() == dict(runtime)
    assert canonical_unified_upstream_bytes(artifact) == canonical_bytes(artifact)


def test_tamper_and_unexpected_secret_shaped_field_are_rejected(tmp_path) -> None:
    _, artifact = _artifact()
    payload = artifact.model_dump(mode="json")
    payload["explicit_decisions"][0]["canonical_condition"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnifiedUpstreamLoadError, match="artifact digest"):
        load_unified_upstream_inputs(
            path,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )

    payload = artifact.model_dump(mode="json")
    payload["provider_response"] = {"api_key": "must-not-be-stored"}
    payload["artifact_digest"] = artifact_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnifiedUpstreamLoadError, match="Extra inputs"):
        load_unified_upstream_inputs(
            path,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )


@pytest.mark.parametrize(
    ("expected_snapshot", "expected_profile", "message"),
    [
        ("sha256:" + "d" * 64, PROFILE, "snapshot mismatch"),
        (SNAPSHOT, "e" * 64, "profile mismatch"),
    ],
)
def test_expected_binding_mismatch_is_rejected(
    tmp_path,
    expected_snapshot,
    expected_profile,
    message,
) -> None:
    _, artifact = _artifact()
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(UnifiedUpstreamLoadError, match=message):
        load_unified_upstream_inputs(
            path,
            expected_snapshot_digest=expected_snapshot,
            expected_profile_digest=expected_profile,
        )


def test_loader_rejects_symlink_non_regular_oversize_and_duplicate_keys(
    tmp_path,
) -> None:
    _, artifact = _artifact()
    path = _write_artifact(tmp_path, artifact)
    link = tmp_path / "linked.json"
    link.symlink_to(path)

    with pytest.raises(UnifiedUpstreamLoadError, match="symlink"):
        load_unified_upstream_inputs(
            link,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )
    with pytest.raises(UnifiedUpstreamLoadError, match="regular file"):
        load_unified_upstream_inputs(
            tmp_path,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )
    with pytest.raises(UnifiedUpstreamLoadError, match="byte limit"):
        load_unified_upstream_inputs(
            path,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
            max_bytes=path.stat().st_size - 1,
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"same":1,"same":2}', encoding="utf-8")
    with pytest.raises(UnifiedUpstreamLoadError, match="duplicate JSON key"):
        load_unified_upstream_inputs(
            duplicate,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(UnifiedUpstreamLoadError, match="non-finite JSON"):
        load_unified_upstream_inputs(
            non_finite,
            expected_snapshot_digest=SNAPSHOT,
            expected_profile_digest=PROFILE,
        )


class _NoCallClient:
    def complete_with_settings(self, *args, **kwargs):
        raise AssertionError("explicit fast path must not invoke a provider")

    def complete(self, *args, **kwargs):
        raise AssertionError("explicit fast path must not invoke a provider")


def _run_decisions(tmp_path, decisions, verification):
    config = config_for_tests()
    llm_settings = config.llm.model_copy(
        update={"max_retries": 0, "backoff_seconds": 0.0, "jitter_seconds": 0.0}
    )
    return run_unified_layer(
        repository_root=tmp_path,
        settings=config.unified,
        llm_settings=llm_settings,
        client=_NoCallClient(),
        protocol_manifest_digest=PROTOCOL_MANIFEST,
        budget=InferenceBudget(llm_settings),
        snapshot_digest=SNAPSHOT,
        snapshot_commit=None,
        profile_digest=PROFILE,
        explicit_decisions=decisions,
        explicit_verification=verification,
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=len(decisions),
    )


def test_host_rejected_unsafe_scope_is_terminal_invalid_without_rewrite(
    tmp_path,
) -> None:
    safe = _decision()
    raw_scope = [
        "*/lib/python*/site-packages/pluggy/*",
        "*\\Lib\\site-packages\\pluggy\\*",
        "testing/*",
    ]
    assert safe.raw_candidate is not None
    unsafe_candidate = safe.raw_candidate.model_copy(
        update={"applies_to": raw_scope}
    )
    unsafe = replace(
        safe,
        candidate_fingerprint=_digest("unsafe candidate"),
        outcome="rejected",
        outcome_stage="semantics",
        reason_code="scope_invalid",
        raw_candidate=unsafe_candidate,
        errors=[
            "invalid repository-relative applies_to glob: "
            "*\\Lib\\site-packages\\pluggy\\*"
        ],
        canonical_scope=tuple(raw_scope),
        host_eligible=False,
        verification_disposition=None,
        verification_request_id=None,
        verification_reason_codes=[],
        claim_id=None,
        proposition=None,
    )

    # Strict model output may repeat an identical claim.  Preserve all three
    # raw events, but materialize the content-addressed unsafe root once.
    result = _run_decisions(
        tmp_path,
        [safe, unsafe, unsafe],
        _verification(safe),
    )
    bundle = result.bundle
    capability = {
        item.capability: item for item in bundle.completeness.capabilities
    }["channel_assessment_construction"]

    assert len(bundle.assessments) == 1
    assert bundle.assessments[0].candidate_root_id == explicit_candidate_root_id(
        safe.candidate_fingerprint
    )
    assert all(
        item.semantics.applies_to != ["**"] for item in bundle.assessments
    )
    assert bundle.funnel.raw_candidate_events == 3
    assert bundle.funnel.unique_candidate_roots == 2
    assert bundle.funnel.invalid_members == 1
    assert bundle.funnel.attrition_reasons[TERMINAL_INVALID_SCOPE_ATTRITION] == 1
    assert bundle.funnel.attrition_reasons["candidate_event_not_materialized"] == 1
    assert bundle.funnel.by_channel["explicit"]["invalid"] == 1
    assert bundle.funnel.by_channel["explicit"][
        f"attrition:{TERMINAL_INVALID_SCOPE_ATTRITION}"
    ] == 1
    assert bundle.funnel.by_evidence_route["direct_normative"][
        "materialized_candidate_versions"
    ] == 2
    assert bundle.funnel.by_evidence_route["direct_normative"]["invalid"] == 1
    assert bundle.completeness.package_status == "complete"
    assert capability.status == "complete"
    assert capability.reason_codes == []

    upstream = make_unified_upstream_inputs(
        snapshot_digest=SNAPSHOT,
        profile_digest=PROFILE,
        snapshot_commit=None,
        explicit_decisions=[safe, unsafe],
        explicit_verification=_verification(safe),
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=2,
    )
    assert upstream.explicit_decisions[1].raw_candidate is not None
    assert upstream.explicit_decisions[1].raw_candidate.applies_to == raw_scope


def test_unexpected_valid_candidate_semantics_failure_is_partial(
    tmp_path,
) -> None:
    decision = _decision()
    assert decision.raw_candidate is not None
    overlong_condition = "x" * 501
    candidate = decision.raw_candidate.model_copy(
        update={"condition": overlong_condition}
    )
    decision = replace(
        decision,
        raw_candidate=candidate,
        canonical_condition=overlong_condition,
    )

    result = _run_decisions(tmp_path, [decision], _verification(decision))
    bundle = result.bundle
    capability = {
        item.capability: item for item in bundle.completeness.capabilities
    }["channel_assessment_construction"]

    assert bundle.assessments == []
    assert bundle.funnel.unique_candidate_roots == 1
    assert bundle.funnel.invalid_members == 1
    assert bundle.funnel.attrition_reasons[ASSESSMENT_CONSTRUCTION_ATTRITION] == 1
    assert bundle.completeness.package_status == "partial"
    assert bundle.completeness.omitted_possible is True
    assert capability.status == "partial"
    assert capability.reason_codes == [
        "channel_assessment_construction_incomplete"
    ]


def test_loaded_inputs_replay_the_same_canonical_unified_bundle(tmp_path) -> None:
    decision, artifact = _artifact()
    path = _write_artifact(tmp_path, artifact)
    runtime = load_unified_upstream_inputs(
        path,
        expected_snapshot_digest=SNAPSHOT,
        expected_profile_digest=PROFILE,
    )
    config = config_for_tests()
    llm_settings = config.llm.model_copy(
        update={"max_retries": 0, "backoff_seconds": 0.0, "jitter_seconds": 0.0}
    )
    execution = {
        "repository_root": tmp_path,
        "settings": config.unified,
        "llm_settings": llm_settings,
        "client": _NoCallClient(),
        "protocol_manifest_digest": PROTOCOL_MANIFEST,
    }
    live = run_unified_layer(
        **execution,
        budget=InferenceBudget(llm_settings),
        snapshot_digest=SNAPSHOT,
        snapshot_commit=None,
        profile_digest=PROFILE,
        explicit_decisions=[decision],
        explicit_verification=_verification(decision),
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=9,
    )
    replayed = run_unified_layer(
        **execution,
        budget=InferenceBudget(llm_settings),
        **runtime,
    )

    assert live.bundle.artifact_digest == replayed.bundle.artifact_digest
    assert canonical_bytes(live.bundle) == canonical_bytes(replayed.bundle)
    assert (
        live.bundle.index.artifact_set_digest
        == replayed.bundle.index.artifact_set_digest
    )
