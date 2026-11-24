from __future__ import annotations

import json
from pathlib import Path

from helpers import build_fixture, copied_current_source_fixture
from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.exporters import stable_json_text
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.traceability import build_explicit_channel


def test_channel_funnels_keep_event_and_root_denominators_separate(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")
    traceability = result.traceability
    assert result.current_source is not None
    current = traceability.current_source
    assert current.event_funnel is not None
    assert current.root_dispositions is not None
    assert current.event_funnel.generated_candidate_events == len(
        result.current_source.reduction.events
    )
    assert current.root_dispositions.total == (
        result.current_source.index.unique_candidate_roots
    )
    assert current.root_dispositions.accepted == 0
    assert traceability.explicit.root_dispositions is not None
    assert not hasattr(traceability, "history")
    assert not hasattr(traceability, "review")
    assert all(
        trace.accepted_constraint_id is None
        for trace in traceability.traces
        if trace.channel == "implicit_current"
    )
    assert all(
        trace.evidence_route
        in {
            "current_structural_population",
            "current_semantic_population",
            "unoperationalized",
        }
        for trace in traceability.traces
        if trace.channel == "implicit_current"
    )
    assert result.build_report["channel_funnels"]["current_source"] == (
        current.model_dump(mode="json")
    )


def test_trace_count_tamper_is_rejected_by_schema_and_replay(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    path = run / "traceability.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["current_source"]["root_dispositions"]["review"] += 1
    path.write_text(stable_json_text(value), encoding="utf-8")
    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "traceability.json schema error" in item
        or "traceability.json differs" in item
        for item in errors
    ), errors


def test_explicit_trace_marks_technical_verification_as_partial() -> None:
    decision = CandidateDecision(
        unit_id="UNIT-" + "1" * 24,
        candidate_fingerprint="a" * 64,
        outcome="review",
        outcome_stage="explicit_semantic_verify",
        reason_code="direct_norm",
        raw_candidate=None,
        verification_disposition="review",
        verification_reason_codes=["technical_missing"],
        verification_technical_reason_code="provider_failure",
    )

    traces, _, channel = build_explicit_channel(
        snapshot_digest="sha256:" + "2" * 64,
        decisions=[decision],
        extraction_complete=False,
    )

    transition = traces[0].stage_transitions[0]
    assert transition.stage == "explicit_semantic_verify"
    assert transition.status == "partial"
    assert transition.reason_codes == [
        "explicit_verification:technical_missing",
        "explicit_verification_technical:provider_failure",
    ]
    assert channel.status == "partial"
