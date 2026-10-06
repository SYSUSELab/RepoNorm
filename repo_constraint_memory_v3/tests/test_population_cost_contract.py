from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.semantic.classify import (
    _classification_request,
    _classification_request_bytes,
)
from repo_constraint_memory_v3.semantic.models import PopulationArtifact


class _PopulationCallRecordingClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.population_requests: list[dict[str, object]] = []
        self.population_request_bytes: list[int] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") == "population_semantic_classify":
            self.population_requests.append(request)
            self.population_request_bytes.append(len(user_prompt.encode("utf-8")))
        return super().complete(system_prompt, user_prompt)


def _client(category: str) -> _PopulationCallRecordingClient:
    return _PopulationCallRecordingClient(
        category=category,
        rule="Contributors preserve the observed fixture convention.",
    )


def test_classification_plan_accounts_for_exact_canonical_request_bytes(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _client("classification-request-accounting")
    result = build_fixture(repository, tmp_path / "run", client=client)

    assert result.current_source is not None
    population = result.current_source.populations[0]
    hypothesis = next(
        item
        for item in result.current_source.hypotheses
        if item.hypothesis_id == population.hypothesis_id
    )
    packets_by_observation = {
        item.observation_ref.observation_id: item
        for item in result.current_source.member_packets
        if item.hypothesis_id == hypothesis.hypothesis_id
    }
    members_by_id = {item.member_id: item for item in population.members}
    packet_by_member = {
        member_id: packets_by_observation[member.observation_ref.observation_id]
        for member_id, member in members_by_id.items()
        if member.observation_ref.observation_id in packets_by_observation
    }
    exact_batch_bytes = [
        _classification_request_bytes(
            _classification_request(
                    hypothesis=hypothesis,
                    member_ids=batch.member_ids,
                    packet_by_member=packet_by_member,
                    sampling_plan=population.sampling_plan,
            )
        )
        for batch in population.classification_plan.batches
    ]

    assert exact_batch_bytes
    assert [
        item.request_bytes for item in population.classification_plan.batches
    ] == exact_batch_bytes
    assert population.classification_plan.planned_request_bytes == sum(
        exact_batch_bytes
    )
    # Provider execution may be concurrent, so compare the exact multiset.
    assert Counter(client.population_request_bytes) == Counter(exact_batch_bytes)
    schedule = result.current_source.index.population_schedule
    assert [item.ordinal for item in schedule] == list(range(len(schedule)))
    assert [item.hypothesis_id for item in schedule] == [
        hypothesis.hypothesis_id
    ]
    admission = result.current_source.index.classification_admission_schedule
    assert len(admission) == 1
    assert admission[0].outcome == "admitted"
    assert admission[0].hypothesis_id == hypothesis.hypothesis_id
    assert admission[0].semantic_members == result.current_source.index.resource_use[
        "classification_admitted_semantic_members"
    ]
    assert admission[0].planned_calls == population.classification_plan.planned_calls
    assert admission[0].planned_request_bytes == (
        population.classification_plan.planned_request_bytes
    )


@pytest.mark.parametrize(
    ("semantic_update", "expected_reason"),
    [
        (
            {
                "max_classification_members_per_request": 1,
                "max_classification_calls_per_hypothesis": 1,
            },
            "max_classification_calls_per_hypothesis",
        ),
        (
            {
                "max_classification_members_per_request": 1,
                "max_member_packet_bytes": 6_000,
                "max_classification_request_bytes": 6_000,
                "max_classification_request_bytes_per_hypothesis": 6_000,
            },
            "max_classification_request_bytes_per_hypothesis",
        ),
    ],
)
def test_per_root_call_or_byte_limit_rejects_before_population_provider_call(
    tmp_path: Path,
    semantic_update: dict[str, int],
    expected_reason: str,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    client = _client(f"per-root-cost-rejection-{expected_reason}")
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(semantic=semantic_update),
        client=client,
    )

    assert result.current_source is not None
    current = result.current_source
    assert current.reduction.hypotheses
    assert client.population_requests == []
    assert current.populations == ()
    assert current.index.status == "partial"
    assert current.index.failed_populations == 1
    assert any(
        reason.startswith(f"population_cost_not_admitted:{expected_reason}:")
        for reason in current.index.incomplete_reasons
    )
    assert current.index.resource_use[
        "classification_not_admitted_hypotheses"
    ] == 1
    assert len(current.index.classification_admission_schedule) == 1
    admission = current.index.classification_admission_schedule[0]
    assert admission.outcome == "not_admitted"
    assert admission.reason_code == expected_reason
    assert admission.limit is not None
    assert current.index.resource_use["classification_admission_attempts"] == 1
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


class _NarrowThenRecordClient(_PopulationCallRecordingClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.parent_hypothesis_id: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") != "counterexample_verify":
            return response
        hypothesis_id = str(request["hypothesis"]["hypothesis_id"])
        if self.parent_hypothesis_id is None:
            self.parent_hypothesis_id = hypothesis_id
        value = json.loads(response.content)
        if hypothesis_id == self.parent_hypothesis_id:
            value.update(
                {
                    "verdict": "narrow_scope",
                    "proposed_applies_to": ["src/aurora/**"],
                    "rationale_summary": (
                        "The supplied evidence supports a narrower scope."
                    ),
                }
            )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="narrow-then-record-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


def test_refinement_child_cannot_bypass_run_wide_classification_ledger(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    client = _NarrowThenRecordClient(
        category="refinement-shared-admission-ledger",
        rule="Contributors preserve the observed fixture convention.",
    )
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(
            semantic={
                # The parent consumes the single allowed canonical job. The
                # refinement still fits its local bound but cannot spend a
                # second call from the shared run-wide ledger.
                "max_classification_members_per_request": 10_000,
                "max_classification_calls_per_hypothesis": 1,
                "max_total_classification_calls": 1,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    current = result.current_source
    parent = next(
        item for item in current.hypotheses if item.parent_hypothesis_id is None
    )
    child = next(
        item for item in current.hypotheses if item.parent_hypothesis_id is not None
    )
    assert child.parent_hypothesis_id == parent.hypothesis_id
    assert len(client.population_requests) == 1
    assert client.population_requests[0]["roots"]["r0"]["hypothesis"]["hypothesis_id"] == (
        parent.hypothesis_id
    )
    assert {item.hypothesis_id for item in current.populations} == {
        parent.hypothesis_id
    }
    assert all(item.hypothesis_id != child.hypothesis_id for item in current.populations)
    assert current.refinements == ()
    assert current.index.status == "partial"
    assert current.index.failed_populations == 1
    assert any(
        reason.startswith(
            "refinement_population_cost_not_admitted:"
            "max_total_classification_calls:"
        )
        for reason in current.index.incomplete_reasons
    )
    assert current.index.resource_use["classification_admitted_hypotheses"] == 1
    assert current.index.resource_use[
        "classification_not_admitted_hypotheses"
    ] == 1
    assert current.index.resource_use["classification_admitted_calls"] == 1
    assert current.index.resource_use[
        "classification_not_admitted_planned_calls"
    ] == 1
    assert not result.build_report["complete"]
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


def test_wide_population_is_sampled_without_reducer_attrition(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    client = _client("bounded-population-sampling")
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(
            semantic={
                "max_classified_population_members_per_hypothesis": 1,
                "max_total_classified_population_members": 1,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    current = result.current_source
    assert current.reduction.hypotheses
    assert not any(
        event.normalization == "cost_omitted"
        for event in current.reduction.events
    )
    population = current.populations[0]
    sampling = population.sampling_plan
    assert sampling.mode == "deterministic_stratified_sample"
    assert sampling.census_enumeration_status == "complete"
    assert sampling.sampling_plan_status == "complete"
    assert sampling.provider_inspected_count == 1
    assert sampling.provider_uninspected_count > 0
    assert sampling.matched_count == (
        sampling.inspected_count + sampling.uninspected_count
    )
    assert client.population_requests
    assert validate_run_directory(run, repository_root=repository) == []


def test_packet_failure_is_not_counted_as_provider_inspected(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "packet-failure-denominator"
    client = _client("packet-failure-provider-denominator")
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(
            semantic={"max_member_packet_bytes": 1}
        ),
        client=client,
    )

    assert result.current_source is not None
    population = result.current_source.populations[0]
    sampling = population.sampling_plan
    assert sampling.mode == "census"
    assert sampling.uninspected_count == 0
    assert sampling.provider_eligible_count > 0
    assert sampling.provider_inspected_count == 0
    assert sampling.provider_uninspected_count == (
        sampling.provider_eligible_count
    )
    assert sum(item.inspected for item in sampling.strata) == 0
    assert sum(item.uninspected for item in sampling.strata) == (
        sampling.provider_eligible_count
    )
    assert population.technical_unknown_count == len(population.members)
    assert population.classification_plan.batches == []
    assert set(population.classification_plan.deterministic_member_ids) == {
        item.member_id for item in population.members
    }
    assert client.population_requests == []
    assert result.current_source.index.resource_use[
        "classification_admitted_semantic_members"
    ] == 0
    assert result.current_source.index.resource_use[
        "classification_admitted_hypotheses"
    ] == 0
    assert result.current_source.index.classification_admission_schedule == []
    assert result.current_source.index.population_provider_inspected_members == 0
    assert result.current_source.index.population_provider_uninspected_members == (
        sampling.provider_eligible_count
    )
    assert result.current_source.index.resource_use[
        "population_schedule_provider_allocated_members"
    ] == sampling.provider_eligible_count
    assert result.current_source.index.resource_use[
        "population_provider_inspected_members"
    ] == 0
    assert validate_run_directory(run, repository_root=repository) == []

    forged = population.model_dump(mode="json")
    forged_sampling = dict(forged["sampling_plan"])
    forged_strata = [dict(item) for item in forged_sampling["strata"]]
    assert forged_strata and forged_strata[0]["uninspected"] > 0
    forged_sampling["provider_inspected_count"] = 1
    forged_sampling["provider_uninspected_count"] -= 1
    forged_strata[0]["inspected"] = 1
    forged_strata[0]["uninspected"] -= 1
    forged_sampling["strata"] = forged_strata
    forged_sampling["plan_digest"] = payload_digest(
        {
            key: value
            for key, value in forged_sampling.items()
            if key != "plan_digest"
        }
    )
    forged["sampling_plan"] = forged_sampling
    forged["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in forged.items()
            if key != "artifact_digest"
        }
    )
    with pytest.raises(
        ValidationError,
        match="provider-inspected denominator differs from provider plan",
    ):
        PopulationArtifact.model_validate(forged)
