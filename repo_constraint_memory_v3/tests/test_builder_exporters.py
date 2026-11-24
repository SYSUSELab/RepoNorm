from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3 import builder as builder_module
from helpers import (
    DelayedMockClient,
    build_fixture,
    config_for_tests,
    copied_fixture,
    discover_fixture,
    supplement_for,
)
from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.builder import (
    SnapshotMismatchError,
    _explicit_outcome_counts,
)
from repo_constraint_memory_v3.llm import LLMProviderError, MockLLMClient
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.run_validation import validate_run_directory


def test_build_exports_canonical_catalog_delivery_and_safe_views(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")
    run = tmp_path / "run"

    assert result.build_report["complete"]
    assert result.memory.schema_version == "4.11-partition-only-required-evidence"
    source_processing = result.build_report["processing"]["current_source"]
    assert source_processing["reducer_cancelled_stages"] == [
        "component_merge",
        "repository_merge",
    ]
    assert not any(key.startswith("reducer_merge_child") for key in source_processing)
    coverage = result.memory.coverage.model_dump(mode="json")
    assert set(coverage) == {
        "explicit",
        "observation",
        "current_source",
        "unified",
        "delivery",
    }
    assert coverage["current_source"]["semantic_units"]["status"] == "complete"
    assert coverage["current_source"]["semantic_map"]["status"] == "complete"
    assert coverage["current_source"]["hypothesis_reduction"]["status"] == (
        "complete"
    )
    assert coverage["current_source"]["population_verification"]["status"] == (
        "complete"
    )
    assert coverage["current_source"]["falsification"]["status"] == "complete"
    assert coverage["unified"]["candidate_grouping"]["status"] == "complete"
    grouping_report = result.build_report["unified"]["candidate_grouping"]
    assert grouping_report["mode"] == "semantic"
    assert grouping_report["reason"] is None
    assert grouping_report["planned_pairs"] >= grouping_report["planned_calls"]
    assert (
        grouping_report["terminal_calls"]
        <= grouping_report["planned_calls"]
    )
    assert coverage["unified"]["git_temporal_evidence"]["status"] == "not_required"
    assert {item.channel for item in result.unified.bundle.assessments} == {
        "explicit",
        "implicit_current",
    }
    legacy_payload = result.memory.model_dump(mode="json")
    legacy_payload["schema_version"] = "3.3.1-current-implicit"
    legacy_payload["coverage"]["implicit_current"] = {"mode": "not_run"}
    with pytest.raises(ValidationError):
        type(result.memory).model_validate(legacy_payload)
    for relative in (
        "audit.jsonl",
        "evidence_relations.json",
        "build_report.json",
        "observations/index.json",
        "observations/queries/python_snapshot.json",
        "observations/queries/python_clone_clusters.json",
        "observations/semantic-units/index.json",
        "inference/index.json",
        "traceability.json",
        "current-source/index.json",
        "current-source/semantic-observations/index.json",
        "current-source/hypotheses/index.json",
        "current-source/propositions.json",
        "canonical/claims.json",
        "canonical/completeness.json",
        "canonical/delivery_report.json",
        "canonical/audit/index.json",
        "canonical/audit/upstream-inputs.json",
        "canonical/audit/member-dispositions.json",
        "canonical/audit/funnel.json",
        "consumers/accepted_only/consumer-manifest.json",
        "consumers/accepted_only/START.md",
        "consumers/accepted_only/catalog.md",
        "consumers/accepted_only/by-path/index.md",
    ):
        assert (run / relative).is_file()
    for forbidden in (
        "by-trigger",
        "implicit",
        "history",
        "adjudication",
    ):
        assert not (run / forbidden).exists()
    start_text = (
        run / "consumers" / "accepted_only" / "START.md"
    ).read_text(encoding="utf-8")
    assert "Repository knowledge" in start_text
    assert "catalog.md" in start_text and "by-path/index.md" in start_text
    assert "Snapshot commit" not in start_text
    assert "history_not_implemented" not in start_text

    formatter = next(
        item
        for item in result.unified.bundle.canonical_claims
        if "ruff format" in item.semantics.rule
    )
    # Fact acceptance is independent from delivery risk.  This command recipe
    # is locally verified without a Git route, but remains reference-only
    # because it mutates the repository broadly.
    assert formatter.epistemic_status == "accepted"
    assert formatter.delivery_policy == "reference_only"
    assert formatter.execution_risk == "repository_wide_mutation"
    canonical = json.loads(
        (run / "canonical" / "claims.json").read_text(encoding="utf-8")
    )
    accepted_ids = {
        item["claim_id"]
        for item in canonical["claims"]
        if item["epistemic_status"] == "accepted"
    }
    catalog_text = (
        run / "consumers" / "accepted_only" / "catalog.md"
    ).read_text(encoding="utf-8")
    assert formatter.claim_id in accepted_ids
    assert formatter.claim_id in catalog_text
    assert not any(
        formatter.claim_id in path.read_text(encoding="utf-8")
        for path in (
            run / "consumers" / "accepted_only" / "by-path"
        ).rglob("constraints.md")
    )
    current_accepted = [
        claim
        for claim in result.unified.bundle.canonical_claims
        if claim.epistemic_status == "accepted"
        and any(
            assessment.channel == "implicit_current"
            and assessment.assessment_id
            in {
                member.assessment_id
                for case in result.unified.bundle.cases
                if case.case_id == claim.case_id
                for member in case.members
            }
            for assessment in result.unified.bundle.assessments
        )
    ]
    assert current_accepted
    assert {item.claim_id for item in current_accepted} <= accepted_ids
    delivery = json.loads(
        (run / "canonical" / "delivery_report.json").read_text(encoding="utf-8")
    )
    assert delivery["accepted_claim_count"] == len(accepted_ids)
    assert delivery["delivered_claim_count"] <= delivery["accepted_claim_count"]


def test_singleton_build_reports_not_required_zero_grouping_denominator(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "singleton-run"
    config = config_for_tests(
        unified={"candidate_grouping_mode": "singleton"}
    )

    result = build_fixture(repository, run, config=config)

    assert result.build_report["unified"]["candidate_grouping"] == {
        "mode": "singleton",
        "status": "not_required",
        "reason": "candidate_grouping_mode_singleton",
        "planned_pairs": 0,
        "planned_calls": 0,
        "terminal_calls": 0,
        "case_count": len(result.unified.bundle.cases),
    }
    assert result.unified.bundle.funnel.candidate_grouping_mode == "singleton"
    assert result.unified.bundle.funnel.grouping_planned_pairs == 0
    assert result.unified.bundle.funnel.grouping_planned_calls == 0
    assert result.unified.bundle.funnel.grouping_case_count == len(
        result.unified.bundle.cases
    )
    assert not any(
        item.stage == "candidate_semantic_group"
        for item in result.inference_records
    )
    assert all(
        len(case.members) == 1 and case.grouping_status == "complete"
        for case in result.unified.bundle.cases
    )
    assert validate_run_directory(run, repository_root=repository) == []


def test_build_never_removes_destination_created_after_initial_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    output = tmp_path / "racing-run"
    real_write = builder_module.write_build_outputs

    def write_then_race(*args, **kwargs):
        result = real_write(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(builder_module, "write_build_outputs", write_then_race)
    with pytest.raises(ValueError, match="appeared during export"):
        build_fixture(repository, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".racing-run.staging-*"))


def test_full_build_writes_channels_once_in_final_bounded_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3.unified import export as export_module

    repository = copied_fixture(tmp_path)
    output = tmp_path / "direct-run"
    original_write = builder_module.write_build_outputs
    original_copy = export_module._copy_upstream_tree
    staged_channels: dict[str, bytes] = {}
    writes = 0

    def capture_channels(directory, **kwargs):
        nonlocal writes
        writes += 1
        assert kwargs["writer"].root == directory
        assert directory.name.startswith(".direct-run.staging-")
        paths = original_write(directory, **kwargs)
        staged_channels.update({
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in paths.values()
        })
        return paths

    def require_direct_write(writer, upstream_directory, **kwargs):
        assert upstream_directory is None
        return original_copy(writer, upstream_directory, **kwargs)

    monkeypatch.setattr(builder_module, "write_build_outputs", capture_channels)
    monkeypatch.setattr(export_module, "_copy_upstream_tree", require_direct_write)
    result = build_fixture(repository, output)
    assert writes == 1
    assert staged_channels
    assert all((output / path).read_bytes() == value for path, value in staged_channels.items())
    assert not list(tmp_path.glob(".direct-run.upstream-*"))
    assert not {"memory.json", "review.json", "catalog.json", "index.md"} & set(staged_channels)
    assert (output / "inference/index.json").is_file()
    assert len(list((output / "inference/requests").glob("*.json"))) == len(result.inference_records)
    assert validate_run_directory(output, repository_root=repository) == []


def test_source_supplement_funnel_reaches_merged_accepted_claim(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config,
        supplement=supplement_for(discovery, "docs/team_practices.rst"),
    )

    funnel = result.build_report["source_funnel"]
    source = next(
        item
        for item in funnel["sources"]
        if item["path"] == "docs/team_practices.rst"
    )
    assert source["proposed"]
    assert source["source_plan_admitted"]
    assert source["loaded"]
    assert source["unit_count"] > 0
    assert source["raw_candidate_count"] > 0
    assert source["grounded_count"] > 0
    assert len(source["accepted_claim_ids"]) == 1
    assert funnel["groups"]["agent_only"]["accepted_claim_ids"] == source[
        "accepted_claim_ids"
    ]


def test_observation_partial_writes_run_but_required_capability_is_incomplete(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path, incomplete=True)
    result = build_fixture(repository, tmp_path / "run")

    assert not result.build_report["complete"]
    assert result.memory.coverage.observation.overall == "partial"
    snapshot = next(
        item
        for item in result.memory.coverage.observation.queries
        if item.name == "python_snapshot"
    )
    assert snapshot.status == "partial"
    assert snapshot.incomplete_reasons
    assert result.build_report["incomplete_reasons"] == (
        result.unified.bundle.completeness.reasons
    )


def test_mock_build_is_deterministic_and_parallelism_is_bounded(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests(
        chunking={
            "max_unit_chars": 120,
            "max_batch_chars": 240,
            "max_units_per_batch": 1,
        }
    )
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"max_concurrency": 3})
        }
    )
    first_client = DelayedMockClient()
    second_client = DelayedMockClient()
    first = build_fixture(
        repository,
        tmp_path / "first",
        config=config,
        client=first_client,
    )
    second = build_fixture(
        repository,
        tmp_path / "second",
        config=config,
        client=second_client,
    )

    assert 2 <= first_client.peak_in_flight <= 3
    assert 2 <= second_client.peak_in_flight <= 3
    for relative in (
        "audit.jsonl",
        "canonical/claims.json",
        "canonical/completeness.json",
        "canonical/delivery_report.json",
        "canonical/audit/index.json",
        "consumers/accepted_only/consumer-manifest.json",
        "consumers/accepted_only/START.md",
        "consumers/accepted_only/catalog.md",
        "consumers/accepted_only/by-path/index.md",
    ):
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()
    assert first.memory == second.memory


@pytest.mark.parametrize("failure_kind", ["schema", "provider"])
def test_provider_or_schema_failure_writes_visible_incomplete_run(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    repository = copied_fixture(tmp_path)

    class ProviderFaultClient:
        def __init__(self) -> None:
            self.delegate = MockLLMClient()

        def complete(self, system_prompt: str, user_prompt: str):
            payload = json.loads(user_prompt)
            request = payload.get("original_request", payload)
            if any(
                unit.get("path") == "AGENTS.md"
                for unit in request.get("units", [])
            ):
                raise LLMProviderError(
                    "fixture provider failure",
                    retryable=False,
                )
            return self.delegate.complete(system_prompt, user_prompt)

    client = (
        DelayedMockClient(delay_seconds=0, invalid_path="AGENTS.md")
        if failure_kind == "schema"
        else ProviderFaultClient()
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)

    assert not result.build_report["complete"]
    assert result.memory.coverage.explicit.extraction == "partial"
    assert result.memory.coverage.explicit.grounding == "complete"
    assert result.build_report["incomplete_reasons"] == (
        result.unified.bundle.completeness.reasons
    )
    assert "explicit_candidate_discovery_incomplete" in (
        result.build_report["incomplete_reasons"]
    )
    assert any(
        item.reason_code == "extraction_failed" for item in result.audit_records
    )
    assert any(
        item.terminal_status == "failure" for item in result.inference_records
    )
    conflict_dependencies = [
        dependency
        for bundle in result.unified.bundle.current_evidence
        for dependency in bundle.dependencies
        if dependency.capability
        == "explicit_discovery_for_current_conflicts"
    ]
    assert conflict_dependencies
    assert {item.requirement_mode for item in conflict_dependencies} == {
        "augmentative"
    }
    assert {item.status for item in conflict_dependencies} == {"partial"}
    assert all(
        item.reason_codes == ["explicit_discovery_incomplete"]
        for item in conflict_dependencies
    )
    assert (run / "canonical" / "claims.json").is_file()
    assert (
        run / "consumers" / "accepted_only" / "START.md"
    ).is_file()
    assert validate_run_directory(run, repository_root=repository) == []
    assert "complete package required but unified package is partial" in (
        validate_run_directory(
            run,
            repository_root=repository,
            require_complete=True,
        )
    )
    request_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run / "inference" / "requests").glob("INFER-*.json")
    ]
    assert any(item["terminal_status"] == "failure" for item in request_records)

    current_assessment_ids = {
        item.assessment_id
        for item in result.unified.bundle.assessments
        if item.channel == "implicit_current"
    }
    current_case_ids = {
        case.case_id
        for case in result.unified.bundle.cases
        if any(
            member.assessment_id in current_assessment_ids
            for member in case.members
        )
    }
    assert current_case_ids
    assert all(
        result.unified.case_routes[case_id] != "not_required_current_incomplete"
        for case_id in current_case_ids
    )
    assert any(
        claim.case_id in current_case_ids
        and claim.epistemic_status == "accepted"
        for claim in result.unified.bundle.canonical_claims
    )


def test_dropped_explicit_unit_remains_gap_without_blocking_current_cases(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests(chunking={"max_units": 1})
    run = tmp_path / "run"
    result = build_fixture(repository, run, config=config)

    assert result.memory.coverage.explicit.eligible_units == "partial"
    assert result.memory.coverage.explicit.extraction == "partial"
    assert result.upstream_inputs.explicit_discovery_complete is False
    assert any(
        item.model_classification == "not_sent"
        for item in result.audit_records
    )
    conflict_dependencies = [
        dependency
        for bundle in result.unified.bundle.current_evidence
        for dependency in bundle.dependencies
        if dependency.capability
        == "explicit_discovery_for_current_conflicts"
    ]
    assert conflict_dependencies
    assert {item.requirement_mode for item in conflict_dependencies} == {
        "augmentative"
    }
    assert {item.status for item in conflict_dependencies} == {"partial"}
    assert "explicit_candidate_discovery_incomplete" in (
        result.unified.bundle.completeness.reasons
    )
    assert all(
        route != "not_required_current_incomplete"
        for route in result.unified.case_routes.values()
    )
    assert validate_run_directory(run, repository_root=repository) == []


def test_explicit_verifier_technical_failure_is_not_a_discovery_gap(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)

    class ExplicitVerifierFaultClient:
        def __init__(self) -> None:
            self.delegate = MockLLMClient()

        def complete(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> ProviderResponse:
            payload = json.loads(user_prompt)
            request = payload.get("original_request", payload)
            if request.get("stage") == "explicit_semantic_verify":
                return ProviderResponse(
                    content="{not-json",
                    model="explicit-verifier-schema-fault",
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    latency_ms=0,
                )
            return self.delegate.complete(system_prompt, user_prompt)

    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=ExplicitVerifierFaultClient(),
    )

    assert result.memory.coverage.explicit.extraction == "complete"
    assert result.memory.coverage.explicit.semantic_verification in {
        "partial",
        "failed",
    }
    assert not result.build_report["complete"]
    assert result.upstream_inputs.explicit_discovery_complete is True
    conflict_dependencies = [
        dependency
        for bundle in result.unified.bundle.current_evidence
        for dependency in bundle.dependencies
        if dependency.capability
        == "explicit_discovery_for_current_conflicts"
    ]
    assert conflict_dependencies
    assert {item.requirement_mode for item in conflict_dependencies} == {
        "augmentative"
    }
    assert {item.status for item in conflict_dependencies} == {"complete"}
    assert validate_run_directory(run, repository_root=repository) == []


def test_known_explicit_semantic_failure_does_not_broadcast_discovery_gap(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    workflow = repository / ".github" / "workflows" / "tests.yml"
    workflow.write_text(
        "name: tests\non: [push]\n",
        encoding="utf-8",
    )

    class PredicateOnlyWorkflowClient:
        def __init__(self) -> None:
            self.delegate = MockLLMClient()

        def complete(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> ProviderResponse:
            response = self.delegate.complete(system_prompt, user_prompt)
            payload = json.loads(user_prompt)
            request = payload.get("original_request", payload)
            workflow_units = [
                unit
                for unit in request.get("units", [])
                if unit.get("path") == ".github/workflows/tests.yml"
            ]
            if not workflow_units:
                return response
            value = json.loads(response.content)
            results_by_unit = {
                item["unit_id"]: item for item in value["results"]
            }
            for unit in workflow_units:
                predicate = next(
                    span
                    for span in unit["spans"]
                    if str(span["text"]).lstrip().startswith("on:")
                )
                results_by_unit[unit["unit_id"]] = {
                    "unit_id": unit["unit_id"],
                    "classification": "contains_claims",
                    "reason_code": "workflow_fact",
                    "claims": [
                        {
                            "rule": "The tests workflow runs on pushes.",
                            "claim_kind": "workflow_fact",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "on pushes",
                            "evidence": [
                                {
                                    "unit_id": unit["unit_id"],
                                    "span_start_id": predicate["span_id"],
                                    "span_end_id": predicate["span_id"],
                                }
                            ],
                        }
                    ],
                }
            content = json.dumps(
                {
                    "results": [
                        results_by_unit[unit["unit_id"]]
                        for unit in request["units"]
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return ProviderResponse(
                content=content,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                reasoning_tokens=response.reasoning_tokens,
                reasoning_content_present=response.reasoning_content_present,
                reasoning_content_bytes=response.reasoning_content_bytes,
                latency_ms=response.latency_ms,
                finish_reason=response.finish_reason,
            )

    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=PredicateOnlyWorkflowClient(),
    )

    assert result.memory.coverage.explicit.extraction == "complete"
    assert result.upstream_inputs.explicit_discovery_complete is True
    workflow_audit = next(
        item
        for item in result.audit_records
        if item.path == ".github/workflows/tests.yml"
    )
    assert workflow_audit.model_classification == "contains_claims"
    assert workflow_audit.local_outcome == "review"
    assert any(
        "workflow evidence must identify an exact typed run/uses action" in error
        for error in workflow_audit.validation_errors
    )
    conflict_dependencies = [
        dependency
        for bundle in result.unified.bundle.current_evidence
        for dependency in bundle.dependencies
        if dependency.capability
        == "explicit_discovery_for_current_conflicts"
    ]
    assert conflict_dependencies
    assert {item.requirement_mode for item in conflict_dependencies} == {
        "augmentative"
    }
    assert {item.status for item in conflict_dependencies} == {"complete"}
    assert validate_run_directory(run, repository_root=repository) == []


def test_non_empty_output_is_refused(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "owned.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        build_fixture(repository, output)
    assert (output / "owned.txt").read_text(encoding="utf-8") == "preserve\n"


def test_build_refuses_to_publish_if_snapshot_changes_during_inference(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)

    class MutatingMockClient:
        def __init__(self) -> None:
            self.delegate = MockLLMClient()
            self.lock = threading.Lock()
            self.mutated = False

        def complete(self, system_prompt: str, user_prompt: str):
            result = self.delegate.complete(system_prompt, user_prompt)
            with self.lock:
                if not self.mutated:
                    (repository / "README.md").write_text(
                        "# Changed during inference\n",
                        encoding="utf-8",
                    )
                    self.mutated = True
            return result

    run = tmp_path / "run"
    with pytest.raises(SnapshotMismatchError, match="snapshot changed"):
        build_fixture(
            repository,
            run,
            client=MutatingMockClient(),
        )

    assert not run.exists() or not any(run.iterdir())


def test_build_rechecks_snapshot_after_staging_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    real_writer = builder_module.write_unified_outputs

    def writer_with_late_repository_race(*args, **kwargs):
        real_check = kwargs["_prepublication_check"]

        def mutate_then_check() -> None:
            (repository / "README.md").write_text(
                "# Changed after staging\n",
                encoding="utf-8",
            )
            real_check()

        kwargs["_prepublication_check"] = mutate_then_check
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(
        builder_module,
        "write_unified_outputs",
        writer_with_late_repository_race,
    )
    run = tmp_path / "late-race-run"
    with pytest.raises(SnapshotMismatchError, match="snapshot changed"):
        build_fixture(repository, run)

    assert not run.exists()
    assert not list(tmp_path.glob(".late-race-run.staging-*"))


def test_explicit_funnel_counts_raw_events_and_unique_roots_separately() -> None:
    first = CandidateDecision(
        unit_id="UNIT-" + "1" * 24,
        candidate_fingerprint="a" * 64,
        outcome="accepted",
        outcome_stage="merge",
        reason_code="direct_norm",
        raw_candidate=None,
    )
    duplicate = CandidateDecision(
        unit_id=first.unit_id,
        candidate_fingerprint=first.candidate_fingerprint,
        outcome="accepted",
        outcome_stage="merge",
        reason_code="direct_norm",
        raw_candidate=None,
    )

    raw, roots = _explicit_outcome_counts([first, duplicate])
    assert raw == {"accepted": 2}
    assert roots == {"accepted": 1}

    duplicate.outcome = "review"
    with pytest.raises(ValueError, match="conflicting host outcomes"):
        _explicit_outcome_counts([first, duplicate])
