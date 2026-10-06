from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import repo_constraint_memory_v3.semantic.classify as semantic_classify

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    attach_response_json_schema,
)
from repo_constraint_memory_v3.llm import LLMProviderError, MockLLMClient
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    payload_digest,
)
from repo_constraint_memory_v3.semantic.classify import (
    HOST_REASON_SUMMARY_TRUNCATION_MARKER,
    MemberClassificationResponse,
    PopulationCostNotAdmitted,
    _fact_index,
    _allowed_classification_reason_codes,
    _classification_request,
    _classification_request_bytes,
    _classification_provider_response_model,
    _normalize_response,
    _repair_prompt,
    _terminal_validate_response,
    _validate_response,
    classify_population,
)
from repo_constraint_memory_v3.semantic.validation import (
    _ScheduledPopulationRoot,
    _population_root_allocations,
    _population_result_non_conclusion_reasons,
    _round_robin_population_roots,
)


def test_population_schedule_is_component_fair_cheap_first_and_order_independent(
) -> None:
    def entry(
        component: str,
        cost: int,
        root_id: str,
        *,
        conclusion_eligible: bool = True,
    ) -> _ScheduledPopulationRoot:
        return _ScheduledPopulationRoot(
            hypothesis=SimpleNamespace(candidate_root_id=root_id),  # type: ignore[arg-type]
            component_key=component,
            semantic_member_cost=cost,
            matched_member_count=cost,
            conclusion_eligible=conclusion_eligible,
            non_conclusion_reasons=(
                ()
                if conclusion_eligible
                else ("population_dependency_partial",)
            ),
        )

    unordered = [
        entry(
            "component-b",
            0,
            "root-b-partial-cheap",
            conclusion_eligible=False,
        ),
        entry("component-b", 20, "root-b-wide"),
        entry("component-a", 10, "root-a-wide"),
        entry(
            "component-a",
            3,
            "root-a-partial-wide",
            conclusion_eligible=False,
        ),
        entry("component-b", 1, "root-b-cheap"),
        entry("component-a", 2, "root-a-cheap"),
        entry(
            "component-a",
            1,
            "root-a-partial-cheap",
            conclusion_eligible=False,
        ),
    ]
    expected = [
        "root-a-cheap",
        "root-b-cheap",
        "root-a-wide",
        "root-b-wide",
        "root-a-partial-cheap",
        "root-b-partial-cheap",
        "root-a-partial-wide",
    ]

    assert [
        item.hypothesis.candidate_root_id
        for item in _round_robin_population_roots(unordered)
    ] == expected
    assert [
        item.hypothesis.candidate_root_id
        for item in _round_robin_population_roots(list(reversed(unordered)))
    ] == expected


def test_population_schedule_preserves_overlapping_non_conclusion_reasons() -> None:
    result = SimpleNamespace(
        status="partial",
        omitted_count=7,
        dependency_file_ids=["FILE-partial"],
        dependency_paths=[],
        dependency_paths_omitted_count=0,
    )

    assert _population_result_non_conclusion_reasons(result) == (
        "population_dependency_partial",
        "population_member_limit",
    )


def test_population_allocation_preserves_small_census_roots_before_sampling(
) -> None:
    def entry(
        component: str,
        matched: int,
        cost: int,
        root_id: str,
    ) -> _ScheduledPopulationRoot:
        return _ScheduledPopulationRoot(
            hypothesis=SimpleNamespace(  # type: ignore[arg-type]
                candidate_root_id=root_id,
                hypothesis_id=f"hypothesis-{root_id}",
                predicate_mode="semantic",
            ),
            component_key=component,
            semantic_member_cost=cost,
            matched_member_count=matched,
            conclusion_eligible=True,
            non_conclusion_reasons=(),
        )

    settings = config_for_tests(
        semantic={
            "max_classified_population_members_per_hypothesis": 20,
            "max_total_classified_population_members": 35,
        }
    ).semantic
    ordered = _round_robin_population_roots(
        [
            entry("component-a", 10, 10, "small-a"),
            entry("component-b", 10, 10, "small-b"),
            entry("component-c", 10, 10, "small-c"),
            entry("component-a", 100, 100, "wide-a"),
            entry("component-b", 100, 100, "wide-b"),
        ]
    )

    allocations = _population_root_allocations(
        ordered,
        semantic_settings=settings,
    )

    assert allocations["hypothesis-small-a"] == 10
    assert allocations["hypothesis-small-b"] == 10
    assert allocations["hypothesis-small-c"] == 10
    assert allocations["hypothesis-wide-a"] + allocations[
        "hypothesis-wide-b"
    ] == 5
    assert abs(
        allocations["hypothesis-wide-a"]
        - allocations["hypothesis-wide-b"]
    ) <= 1
    assert sum(allocations.values()) == 35


def test_population_allocation_never_partially_inspects_census_root() -> None:
    def entry(root_id: str) -> _ScheduledPopulationRoot:
        return _ScheduledPopulationRoot(
            hypothesis=SimpleNamespace(  # type: ignore[arg-type]
                candidate_root_id=root_id,
                hypothesis_id=f"hypothesis-{root_id}",
                predicate_mode="semantic",
            ),
            component_key=root_id,
            semantic_member_cost=10,
            matched_member_count=10,
            conclusion_eligible=True,
            non_conclusion_reasons=(),
        )

    settings = config_for_tests(
        semantic={
            "max_classified_population_members_per_hypothesis": 20,
            "max_total_classified_population_members": 25,
        }
    ).semantic
    allocations = _population_root_allocations(
        _round_robin_population_roots(
            [entry("root-a"), entry("root-b"), entry("root-c")]
        ),
        semantic_settings=settings,
    )

    assert sorted(allocations.values()) == [0, 10, 10]
    assert sum(allocations.values()) == 20


class _SemanticDependencyPartialClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.population_classification_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "population_semantic_classify":
            self.population_classification_calls += 1
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = {
                "op": "universe",
                "universe": "entity",
            }
            hypothesis["applies_to"] = ["**"]
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="semantic-dependency-partial-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


def test_dependency_partial_semantic_population_is_closed_without_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    (repository / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    client = _SemanticDependencyPartialClient(
        category="semantic-dependency-partial",
        rule="Contributors preserve the observed fixture convention.",
    )

    def forbidden_fact_index(*_args, **_kwargs):
        raise AssertionError("dependency-partial root expanded full source facts")

    monkeypatch.setattr(semantic_classify, "_fact_index", forbidden_fact_index)

    def forbidden_classification_count(*_args, **_kwargs):
        raise AssertionError("dependency-partial root recomputed provider cost")

    monkeypatch.setattr(
        semantic_classify,
        "count_population_members_for_admission",
        forbidden_classification_count,
    )
    run = tmp_path / "run"
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
    assert not any(event.normalization == "cost_omitted" for event in current.reduction.events)
    population = current.populations[0]
    plan = population.classification_plan
    assert population.population_result.status == "partial"
    assert population.population_result.serialized_count == len(
        population.members
    )
    assert population.population_result.matched_count == len(population.members)
    assert set(plan.member_ids) == {item.member_id for item in population.members}
    assert set(plan.deterministic_member_ids) == set(plan.member_ids)
    assert plan.unplanned_member_ids == []
    assert plan.batches == []
    assert plan.planned_calls == 0
    assert plan.planned_bytes > 0
    assert plan.planned_request_bytes == 0
    assert plan.execution.request_plan_complete
    assert plan.execution.planned_items == 0
    assert plan.execution.terminal_items == 0
    assert plan.execution.omitted_items == 0
    assert client.population_classification_calls == 0
    assert not any(
        record.stage == "population_semantic_classify"
        for record in result.inference_records
    )
    assert {item.label for item in population.members} == {"unknown"}
    assert {
        item.classification_basis for item in population.members
    } == {"technical_unknown"}
    assert {item.reason_code for item in population.members} == {
        "semantic_population_dependency_partial"
    }
    population_packets = [
        item
        for item in current.member_packets
        if item.hypothesis_id == population.hypothesis_id
    ]
    assert len(population_packets) == len(population.members)
    assert all(
        set(item.fact)
        == {"observation_id", "population_dependency_status"}
        and item.fact["population_dependency_status"] == "partial"
        and item.missing_context == ["population_query_dependency_partial"]
        for item in population_packets
    )
    assert {
        "population_query_dependency_partial",
        "candidate_local_dependencies_partial",
        "semantic_population_dependency_partial",
        "technical_unknown_members",
    } <= set(population.incomplete_reasons)
    assert current.index.resource_use[
        "population_schedule_conclusion_eligible_roots"
    ] == 0
    assert current.index.resource_use[
        "population_schedule_non_conclusion_eligible_roots"
    ] == 1
    assert current.index.resource_use[
        "population_schedule_non_conclusion_dependency_partial_roots"
    ] == 1
    assert current.index.resource_use[
        "population_schedule_non_conclusion_member_limit_roots"
    ] == 0
    assert current.index.resource_use[
        "population_schedule_non_conclusion_precost_failed_roots"
    ] == 0
    assert current.index.resource_use[
        "population_schedule_conclusion_eligible_semantic_members"
    ] == 0
    assert current.index.resource_use[
        "population_schedule_non_conclusion_eligible_semantic_members"
    ] == len(population.members)
    assert len(current.index.population_schedule) == 1
    scheduled = current.index.population_schedule[0]
    assert scheduled.ordinal == 0
    assert scheduled.hypothesis_id == population.hypothesis_id
    assert not scheduled.conclusion_eligible
    assert scheduled.non_conclusion_reasons == [
        "population_dependency_partial"
    ]
    assert current.index.classification_admission_schedule == []
    assert current.index.resource_use["classification_admitted_hypotheses"] == 0
    assert current.index.resource_use["classification_admitted_calls"] == 0
    assert current.index.resource_use[
        "classification_not_admitted_hypotheses"
    ] == 0
    assert current.index.resource_use["classification_admission_attempts"] == 0
    assert current.index.resource_use["falsification_admitted_hypotheses"] == 0
    assert current.index.resource_use["falsification_admitted_calls"] == 0
    assert current.index.resource_use[
        "falsification_admitted_request_bytes"
    ] == 0
    assert current.index.resource_use[
        "falsification_cost_not_admitted_hypotheses"
    ] == 0
    assert current.index.resource_use["falsification_not_planned_roots"] == 1
    assert current.index.resource_use[
        "falsification_terminal_inference_record_hypotheses"
    ] == 0
    assert validate_run_directory(run, repository_root=repository) == []


def test_semantic_population_freezes_and_classifies_every_member_once(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=ParameterizedSemanticClient(
            category="open-four-label-fixture",
            rule="Contributors follow the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.route == "current_semantic_population"
    assert population.status == "complete"
    assert population.member_set_digest is not None
    assert population.population_result.serialized_count == len(
        population.members
    )
    assert {
        item.label for item in population.members
    } == {"support", "counterexample", "ineligible", "unknown"}
    assert population.semantic_unknown_count == population.unknown_count
    assert population.technical_unknown_count == 0
    source_context_packets = [
        packet
        for packet in result.current_source.member_packets
        if packet.fact.get("semantic_unit")
    ]
    assert source_context_packets
    assert all(
        packet.fact["semantic_unit"]["source_ranges"]  # type: ignore[index]
        for packet in source_context_packets
    )
    assert all(
        any(
            source_range["range_id"] in packet.evidence_refs
            and source_range["text"]
            for source_range in packet.fact["semantic_unit"]["source_ranges"]  # type: ignore[index]
        )
        for packet in source_context_packets
    )
    assigned = [
        member_id
        for batch in population.classification_plan.batches
        for member_id in batch.member_ids
    ]
    assert assigned == population.classification_plan.member_ids
    assert len(assigned) == len(set(assigned)) == len(population.members)
    assert population.clone_aware_independent_support <= (
        population.occurrence_support
    )
    assert set(population.distributions) >= {
        "file",
        "module",
        "component",
        "clone_aware",
    }


def test_exhausted_population_never_builds_classification_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    baseline = build_fixture(
        repository,
        tmp_path / "baseline",
        config=config,
        client=ParameterizedSemanticClient(
            category="budget-preclose-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert baseline.current_source is not None
    hypothesis = baseline.current_source.hypotheses[0]
    assert hypothesis.predicate_mode == "semantic"
    assert baseline.observation is not None
    assert baseline.semantic_units is not None

    def forbidden_payload(**kwargs):
        del kwargs
        raise AssertionError("exhausted classification built a request payload")

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.classify._classification_request",
        forbidden_payload,
    )
    budget = InferenceBudget(
        config.llm,
        initial_input_tokens=config.llm.max_total_input_tokens,
    )
    built = classify_population(
        repository_root=repository,
        discovery=baseline.source_acquisition.discovery,
        hypothesis=hypothesis,
        observations=baseline.observation,
        semantic_units=baseline.semantic_units,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        client=MockLLMClient(),
        budget=budget,
        protocol_manifest_digest=(
            baseline.protocol_manifest.protocol_manifest_digest
        ),
        candidate_local_dependencies_complete=True,
    )

    execution = built.artifact.classification_plan.execution
    assert not execution.request_plan_complete
    assert execution.planned_items == execution.omitted_items
    assert execution.terminal_items == 0
    assert built.inference_records == ()
    assert all(
        member.inference_ref is None
        for member in built.artifact.members
        if member.classification_basis == "technical_unknown"
    )


def test_total_population_suffix_is_omitted_before_packet_or_provider_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    baseline = build_fixture(
        repository,
        tmp_path / "baseline-total-population",
        config=config,
        client=ParameterizedSemanticClient(
            category="whole-population-admission-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert baseline.current_source is not None
    assert baseline.observation is not None
    assert baseline.semantic_units is not None
    hypothesis = baseline.current_source.hypotheses[0]
    population = baseline.current_source.populations[0]
    assert population.population_result.matched_count > 1

    def forbidden_fact_index(*_args, **_kwargs):
        raise AssertionError("partial population constructed provider packets")

    monkeypatch.setattr(semantic_classify, "_fact_index", forbidden_fact_index)
    with pytest.raises(PopulationCostNotAdmitted) as captured:
        classify_population(
            repository_root=repository,
            discovery=baseline.source_acquisition.discovery,
            hypothesis=hypothesis,
            observations=baseline.observation,
            semantic_units=baseline.semantic_units,
            semantic_settings=config.semantic,
            llm_settings=config.llm,
            client=MockLLMClient(),
            budget=InferenceBudget(config.llm),
            protocol_manifest_digest=(
                baseline.protocol_manifest.protocol_manifest_digest
            ),
            candidate_local_dependencies_complete=True,
            max_members=1,
        )

    assert captured.value.reason_code == "max_total_population_members"
    assert captured.value.planned_calls == 0


def test_mid_population_budget_omission_round_trips_offline(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    base_config = config_for_tests(
        semantic={"max_classification_members_per_request": 1}
    )
    base_config = base_config.model_copy(
        update={
            "llm": base_config.llm.model_copy(update={"max_concurrency": 1})
        }
    )
    client_kwargs = {
        "category": "mid-population-budget-fixture",
        "rule": "Contributors preserve the observed fixture convention.",
    }
    baseline = build_fixture(
        repository,
        tmp_path / "baseline-mid-population",
        config=base_config,
        client=ParameterizedSemanticClient(**client_kwargs),
    )
    pre_population_calls = sum(
        record.usage.calls
        for record in baseline.inference_records
        if record.stage
        in {
            "explicit_extract",
            "explicit_semantic_verify",
            "current_semantic_map",
            "current_hypothesis_reduce",
        }
    )
    limited_config = base_config.model_copy(
        update={
            "llm": base_config.llm.model_copy(
                update={"max_total_calls": pre_population_calls + 1}
            )
        }
    )
    run = tmp_path / "limited-mid-population"
    limited = build_fixture(
        repository,
        run,
        config=limited_config,
        client=ParameterizedSemanticClient(**client_kwargs),
    )
    assert limited.current_source is not None
    population = limited.current_source.populations[0]
    execution = population.classification_plan.execution
    assert execution.terminal_calls == 1
    assert execution.omitted_calls > 0
    assert execution.planned_calls == (
        execution.terminal_calls + execution.omitted_calls
    )
    classification_stage = next(
        stage
        for stage in limited.inference_index.stages
        if stage.stage == "population_semantic_classify"
    )
    assert classification_stage.omitted == execution.omitted_calls
    assert classification_stage.omission_reasons == {
        "max_total_calls": execution.omitted_calls
    }
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


def test_current_omission_coordinated_index_reseal_is_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    base_config = config_for_tests(
        semantic={"max_classification_members_per_request": 1}
    )
    base_config = base_config.model_copy(
        update={
            "llm": base_config.llm.model_copy(update={"max_concurrency": 1})
        }
    )
    baseline = build_fixture(
        repository,
        tmp_path / "baseline-reseal",
        config=base_config,
        client=ParameterizedSemanticClient(
            category="omission-reseal-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    pre_population_calls = sum(
        record.usage.calls
        for record in baseline.inference_records
        if record.stage
        in {
            "explicit_extract",
            "explicit_semantic_verify",
            "current_semantic_map",
            "current_hypothesis_reduce",
        }
    )
    config = base_config.model_copy(
        update={
            "llm": base_config.llm.model_copy(
                update={"max_total_calls": pre_population_calls + 1}
            )
        }
    )
    run = tmp_path / "resealed-current-omission"
    build_fixture(
        repository,
        run,
        config=config,
        client=ParameterizedSemanticClient(
            category="omission-reseal-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    path = run / "inference" / "index.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    stage = next(
        item
        for item in value["stages"]
        if item["stage"] == "population_semantic_classify"
    )
    stage["planned"] += 1
    stage["omitted"] += 1
    stage["omission_reasons"]["max_total_calls"] += 1
    value.pop("inference_set_digest")
    value["inference_set_digest"] = payload_digest(value)
    path.write_bytes(canonical_json_bytes(value))

    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "current-source inference omissions differ from recorded replay" in item
        for item in errors
    )


class _OneUnrelatedReducerFailureClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.failed_component: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
            and self.failed_component is None
        ):
            self.failed_component = str(request.get("component_key"))
            raise LLMProviderError(
                "isolated reducer transport failure",
                retryable=False,
            )
        return super().complete(system_prompt, user_prompt)


def test_unrelated_reducer_failure_does_not_taint_materialized_populations(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _OneUnrelatedReducerFailureClient(
        category="candidate-local-reduction-dependency",
        rule="Contributors preserve the independently materialized pattern.",
    )
    result = build_fixture(repository, tmp_path / "run", client=client)

    assert client.failed_component is not None
    assert result.current_source is not None
    current = result.current_source
    assert current.reduction.index.status == "partial"
    assert current.reduction.index.failed_calls == 1
    assert current.index.status == "partial"
    assert current.populations
    assert all(
        population.status == "complete" for population in current.populations
    )
    assert all(
        "candidate_local_dependencies_partial"
        not in population.incomplete_reasons
        for population in current.populations
    )
    assert "hypothesis_reducer_failure" in current.index.incomplete_reasons


class _OneUnrelatedMapFailureClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self.failed_map_batch = False

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") == "current_semantic_map":
            with self._lock:
                if not self.failed_map_batch:
                    self.failed_map_batch = True
                    raise LLMProviderError(
                        "isolated semantic-map transport failure",
                        retryable=False,
                    )
        return super().complete(system_prompt, user_prompt)


def test_unrelated_map_failure_does_not_taint_materialized_populations(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests(
        semantic={"max_map_packets_per_request": 1}
    )
    client = _OneUnrelatedMapFailureClient(
        category="candidate-local-map-dependency",
        rule="Contributors preserve the independently materialized pattern.",
    )
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config,
        client=client,
    )

    assert client.failed_map_batch
    assert result.current_source is not None
    current = result.current_source
    assert current.semantic_map.index.status == "partial"
    assert current.semantic_map.index.packet_failure == 1
    assert current.semantic_map.observations
    assert current.reduction.hypotheses
    assert current.index.status == "partial"
    assert current.populations
    assert all(
        population.status == "complete" for population in current.populations
    )
    assert all(
        "candidate_local_dependencies_partial"
        not in population.incomplete_reasons
        for population in current.populations
    )
    assert "semantic_map_packet_failure" in current.index.incomplete_reasons


class _ClassificationRequestBoundClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.classification_request_bytes: list[int] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        original = request.get("original_request", request)
        if original.get("stage") == "population_semantic_classify":
            self.classification_request_bytes.append(
                len(user_prompt.encode("utf-8"))
            )
        return super().complete(system_prompt, user_prompt)


class _ClassificationReasonSchemaClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.reason_contracts: list[tuple[list[str], list[str]]] = []
        self.evidence_contracts: list[tuple[list[str], list[str]]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") == "population_semantic_classify":
            schema = payload["response_json_schema"]
            draft = schema["$defs"]["BoundedMemberClassificationDraft"]
            self.reason_contracts.append(
                (
                    request["allowed_reason_codes"],
                    draft["properties"]["reason_code"]["enum"],
                )
            )
            self.evidence_contracts.append(
                (
                    request["allowed_evidence_refs"],
                    draft["properties"]["evidence_refs"]["items"]["enum"],
                )
            )
        return super().complete(system_prompt, user_prompt)


def test_classification_reason_vocabulary_is_bound_in_request_and_schema(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _ClassificationReasonSchemaClient(
        category="reason-schema-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)

    assert client.reason_contracts
    assert all(
        allowed == sorted(set(allowed)) and allowed == schema_enum
        for allowed, schema_enum in client.reason_contracts
    )
    assert all(
        allowed == sorted(set(allowed)) and allowed == schema_enum
        for allowed, schema_enum in client.evidence_contracts
    )
    records = [
        item
        for item in result.inference_records
        if item.stage == "population_semantic_classify"
    ]
    assert records
    assert all(
        item.terminal_status == "success"
        and len(item.attempts) == 1
        and not item.attempts[0].repair
        for item in records
    )
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
    packets_by_member = {
        item.member_id: packets_by_observation[item.observation_ref.observation_id]
        for item in population.members
        if item.observation_ref.observation_id in packets_by_observation
    }
    first_batch = population.classification_plan.batches[0]
    request = _classification_request(
        hypothesis=hypothesis,
        member_ids=first_batch.member_ids,
        packet_by_member=packets_by_member,
        sampling_plan=population.sampling_plan,
    )
    request_value = json.loads(request)
    provider_model = _classification_provider_response_model(
        tuple(request_value["allowed_reason_codes"]),
        tuple(request_value["allowed_evidence_refs"]),
    )
    contractual = attach_response_json_schema(request, provider_model)
    first_record = next(
        item for item in records if item.request_id == first_batch.inference_id
    )
    assert _classification_request_bytes(request) == len(
        contractual.encode("utf-8")
    )
    assert first_batch.request_bytes == len(contractual.encode("utf-8"))
    assert first_record.request_payload_digest == hashlib.sha256(
        contractual.encode("utf-8")
    ).hexdigest()
    assert MemberClassificationResponse.model_validate(
        first_record.structured_response
    ).model_dump(mode="json") == first_record.structured_response
    # Offline replay reconstructs the same provider schema and therefore the
    # same request digest/cache key while parsing the stable stored response.
    assert validate_run_directory(run, repository_root=repository) == []

    repair = json.loads(
        _repair_prompt(
            json.dumps(
                {
                    "stage": "population_semantic_classify",
                    "allowed_reason_codes": ["allowed_a", "allowed_b"],
                }
            ),
            '{"results":[]}',
            ["semantic classification used a reason code outside the contract"],
        )
    )
    assert repair["allowed_reason_codes"] == ["allowed_a", "allowed_b"]
    assert "never invent" in repair["instruction"]


def test_classification_normalizes_only_bounded_auxiliary_summary_text() -> None:
    long_summary = "reason   " + "x" * 700
    normalized = _normalize_response(
        {
            "results": [
                {
                    "member_ordinal": 0,
                    "label": "unknown",
                    "evidence_context": "unknown",
                    "reason_code": "insufficient_context",
                    "reason_summary": long_summary,
                    "evidence_refs": ["EVIDENCE-a"],
                }
            ]
        },
        member_ids=["MEMBER-a"],
    )

    summary = normalized["results"][0]["reason_summary"]
    assert len(summary) == 600
    assert summary.endswith(HOST_REASON_SUMMARY_TRUNCATION_MARKER)
    assert "  " not in summary
    assert MemberClassificationResponse.model_validate(normalized)

    whitespace_only = _normalize_response(
        {
            "results": [
                {
                    "member_ordinal": 0,
                    "label": "unknown",
                    "evidence_context": "unknown",
                    "reason_code": "insufficient_context",
                    "reason_summary": "   \n  ",
                    "evidence_refs": ["EVIDENCE-a"],
                }
            ]
        },
        member_ids=["MEMBER-a"],
    )
    with pytest.raises(ValidationError):
        MemberClassificationResponse.model_validate(whitespace_only)


def test_terminal_classification_quarantines_only_foreign_member_refs() -> None:
    response = MemberClassificationResponse.model_validate(
        {
            "results": [
                {
                    "member_ordinal": 0,
                    "label": "support",
                    "evidence_context": "production",
                    "reason_code": "api_invariant",
                    "reason_summary": "supported",
                    "evidence_refs": ["FOREIGN-ref"],
                },
                {
                    "member_ordinal": 1,
                    "label": "support",
                    "evidence_context": "production",
                    "reason_code": "api_invariant",
                    "reason_summary": "supported sibling",
                    "evidence_refs": ["EVIDENCE-b"],
                },
            ]
        }
    )
    member_ids = ["MEMBER-a", "MEMBER-b"]
    allowed = {
        "MEMBER-a": {"EVIDENCE-a"},
        "MEMBER-b": {"EVIDENCE-b"},
    }
    reason_codes = {"api_invariant", "insufficient_context"}
    with pytest.raises(ValueError, match="unsupplied evidence"):
        _validate_response(
            response,
            member_ids=member_ids,
            allowed_by_member=allowed,
            allowed_reason_codes=reason_codes,
        )

    terminal = _terminal_validate_response(
        response,
        member_ids=member_ids,
        allowed_by_member=allowed,
        allowed_reason_codes=reason_codes,
    )
    first, second = terminal.results
    assert (first.label, first.evidence_context) == ("unknown", "unknown")
    assert first.reason_code == "insufficient_context"
    assert first.evidence_refs == ["EVIDENCE-a"]
    assert second == response.results[1]


def test_classification_request_bound_includes_response_schema(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    request_limit = 6_000
    config = config_for_tests(
        semantic={
            "max_classification_request_bytes": request_limit,
            "max_member_packet_bytes": request_limit,
        }
    )
    client = _ClassificationRequestBoundClient(
        category="classification-request-bound-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )

    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config,
        client=client,
    )

    assert result.current_source is not None
    population = result.current_source.populations[0]
    hypothesis = next(
        item
        for item in result.current_source.hypotheses
        if item.hypothesis_id == population.hypothesis_id
    )
    packet_by_observation = {
        item.observation_ref.observation_id: item
        for item in result.current_source.member_packets
        if item.hypothesis_id == hypothesis.hypothesis_id
    }
    packet_by_member = {
        item.member_id: packet_by_observation[item.observation_ref.observation_id]
        for item in population.members
    }
    planned_sizes = [
        (
            len(batch.member_ids),
            _classification_request_bytes(
                _classification_request(
                    hypothesis=hypothesis,
                    member_ids=batch.member_ids,
                    packet_by_member=packet_by_member,
                    sampling_plan=population.sampling_plan,
                )
            ),
        )
        for batch in population.classification_plan.batches
    ]

    assert client.classification_request_bytes
    assert max(client.classification_request_bytes) <= request_limit
    assert all(size <= request_limit for count, size in planned_sizes if count > 1)
    assert any(size > request_limit for count, size in planned_sizes if count == 1)
    classification_records = [
        item
        for item in result.inference_records
        if item.stage == "population_semantic_classify"
    ]
    assert any(
        item.terminal_status == "failure"
        and "max_classification_request_bytes" in item.incomplete_reasons
        for item in classification_records
    )
    assert population.status == "partial"
    assert validate_run_directory(
        tmp_path / "run",
        repository_root=repository,
    ) == []


class _StructuralClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.population_classification_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "population_semantic_classify":
            self.population_classification_calls += 1
        response = super().complete(system_prompt, user_prompt)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = {
                "op": "universe",
                "universe": "entity",
            }
            hypothesis["predicate_mode"] = "structural"
            hypothesis["semantic_predicate"] = None
            hypothesis["structural_predicate"] = {
                "eligible_spec": {
                    "op": "filter",
                    "source": {"op": "universe", "universe": "entity"},
                    "predicates": [
                        {
                            "field": "kind",
                            "operator": "in",
                            "value": ["function", "async_function", "class"],
                        }
                    ],
                },
                "support_spec": {
                    "op": "filter",
                    "source": {"op": "universe", "universe": "entity"},
                    "predicates": [
                        {
                            "field": "kind",
                            "operator": "in",
                            "value": ["function", "async_function"],
                        }
                    ],
                },
            }
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="structural-population-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _StructuralEligiblePartialClient(_StructuralClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        scoped = {
            "op": "filter",
            "source": {"op": "universe", "universe": "entity"},
            "predicates": [
                {
                    "field": "path",
                    "operator": "path_under",
                    "value": "src/aurora",
                }
            ],
        }
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = scoped
            hypothesis["structural_predicate"] = {
                "eligible_spec": {
                    "op": "difference",
                    "left": scoped,
                    "right": {
                        "op": "filter",
                        "source": {"op": "universe", "universe": "entity"},
                        "predicates": [
                            {
                                "field": "kind",
                                "operator": "eq",
                                "value": "class",
                            }
                        ],
                    },
                },
                "support_spec": {
                    "op": "filter",
                    "source": scoped,
                    "predicates": [
                        {
                            "field": "kind",
                            "operator": "in",
                            "value": ["function", "async_function"],
                        }
                    ],
                },
            }
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="structural-local-partial-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _StructuralSupportPartialClient(_StructuralClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        scoped = {
            "op": "filter",
            "source": {"op": "universe", "universe": "entity"},
            "predicates": [
                {
                    "field": "path",
                    "operator": "path_under",
                    "value": "src/aurora",
                }
            ],
        }
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = scoped
            hypothesis["structural_predicate"] = {
                "eligible_spec": scoped,
                "support_spec": {
                    "op": "union",
                    "operands": [
                        {
                            "op": "filter",
                            "source": scoped,
                            "predicates": [
                                {
                                    "field": "kind",
                                    "operator": "in",
                                    "value": ["function", "async_function"],
                                }
                            ],
                        },
                        {
                            "op": "filter",
                            "source": {
                                "op": "traverse",
                                "source": scoped,
                                "traversal": "relation",
                                "relation_kind": "calls",
                                "direction": "outgoing",
                            },
                            "predicates": [
                                {
                                    "field": "path",
                                    "operator": "path_under",
                                    "value": "src/aurora",
                                }
                            ],
                        },
                    ],
                },
            }
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="structural-support-partial-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def _assert_structural_dependency_partial(
    *,
    result,
    client: _StructuralClient,
    repository: Path,
    run: Path,
) -> None:
    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.route == "current_structural_population"
    assert population.classification_plan.planned_calls == 0
    assert client.population_classification_calls == 0
    assert not any(
        record.stage == "population_semantic_classify"
        for record in result.inference_records
    )
    assert population.support_count == 0
    assert population.counterexample_count == 0
    assert population.unknown_count == len(population.members)
    assert {
        member.label for member in population.members
    } == {"unknown"}
    assert {
        member.classification_basis for member in population.members
    } == {"technical_unknown"}
    assert {
        member.reason_code for member in population.members
    } == {"structural_population_dependency_partial"}
    assert "structural_population_dependency_partial" in (
        population.incomplete_reasons
    )
    assert validate_run_directory(run, repository_root=repository) == []


def test_structural_population_uses_subset_algebra_without_llm_labels(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_StructuralClient(
            category="generic-structural-fixture",
            rule="Contributors preserve the structurally observed pattern.",
        ),
    )
    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.route == "current_structural_population"
    assert population.classification_plan.planned_calls == 0
    assert {
        item.classification_basis for item in population.members
    } == {"structural"}
    assert {"support", "counterexample", "ineligible"} <= {
        item.label for item in population.members
    }


def test_structural_population_unions_super_eligible_and_support_dependencies(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    (repository / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    client = _StructuralEligiblePartialClient(
        category="generic-structural-local-partial",
        rule="Contributors preserve the structurally observed pattern.",
    )
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=client,
    )

    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.population_result.status == "complete"
    assert population.population_result.dependency_file_ids == []
    assert "population_query_dependency_partial" in (
        population.incomplete_reasons
    )
    _assert_structural_dependency_partial(
        result=result,
        client=client,
        repository=repository,
        run=run,
    )


def test_structural_partial_superpopulation_does_not_materialize_labels(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    (repository / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    client = _StructuralClient(
        category="generic-structural-super-partial",
        rule="Contributors preserve the structurally observed pattern.",
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)

    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.population_result.status == "partial"
    _assert_structural_dependency_partial(
        result=result,
        client=client,
        repository=repository,
        run=run,
    )


def test_structural_partial_support_does_not_materialize_labels(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    (repository / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    client = _StructuralSupportPartialClient(
        category="generic-structural-support-partial",
        rule="Contributors preserve the structurally observed pattern.",
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)

    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.population_result.status == "complete"
    assert population.population_result.dependency_file_ids == []
    _assert_structural_dependency_partial(
        result=result,
        client=client,
        repository=repository,
        run=run,
    )


class _MissingMemberClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        value["results"] = value["results"][:-1]
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="missing-member-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _ReorderedMemberClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        value["results"] = list(reversed(value["results"]))
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="reordered-member-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _CopiedOpaqueMemberIdClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        member_id_by_ordinal = {
            item["member_ordinal"]: item["member_id"]
            for item in request["members"]
        }
        for result in value["results"]:
            ordinal = result.pop("member_ordinal")
            result["member_id"] = member_id_by_ordinal[ordinal]
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="copied-opaque-member-id-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _ConflictingOpaqueMemberIdClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        if len(request["members"]) >= 2:
            value["results"][0]["member_id"] = request["members"][1]["member_id"]
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="conflicting-opaque-member-id-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _ContradictoryClassificationAxesClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        for item in value["results"]:
            if item["label"] == "support":
                item["evidence_context"] = "irrelevant"
                break
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="contradictory-classification-axes-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_exact_copied_member_ids_are_safely_repaired_to_ordinals(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_CopiedOpaqueMemberIdClient(
            category="opaque-id-repair-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.populations[0].status == "complete"
    assert all(
        record.terminal_status == "success"
        for record in result.inference_records
        if record.stage == "population_semantic_classify"
    )


def test_conflicting_copied_member_id_and_ordinal_fail_closed(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_ConflictingOpaqueMemberIdClient(
            category="conflicting-id-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.populations[0].status == "partial"
    assert any(
        record.terminal_status == "failure"
        for record in result.inference_records
        if record.stage == "population_semantic_classify"
    )


def test_contradictory_classification_axes_fail_closed(tmp_path: Path) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_ContradictoryClassificationAxesClient(
            category="classification-axis-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.populations[0].status == "partial"
    assert any(
        record.terminal_status == "failure"
        for record in result.inference_records
        if record.stage == "population_semantic_classify"
    )


def test_reordered_semantic_members_are_canonicalized(tmp_path: Path) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_ReorderedMemberClient(
            category="reordered-member-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.populations[0].status == "complete"
    assert all(
        record.terminal_status == "success"
        for record in result.inference_records
        if record.stage == "population_semantic_classify"
    )


class _NarrowScopeRecordingClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.classified_paths: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "current_hypothesis_reduce":
            value = json.loads(response.content)
            for hypothesis in value.get("hypotheses", []):
                hypothesis["applies_to"] = ["src/aurora/**"]
                if request.get("mode") == "component_partition":
                    bindings = request["population_context"].get(
                        "witness_bindings", []
                    )
                    scoped = next(
                        (
                            item["witness_id"]
                            for item in bindings
                            if str(item.get("path", "")).startswith(
                                "src/aurora/"
                            )
                        ),
                        None,
                    )
                    if scoped is not None:
                        hypothesis["witness_refs"] = [scoped]
            return ProviderResponse(
                content=json.dumps(value, sort_keys=True, separators=(",", ":")),
                model="narrow-scope-fake",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )
        if request.get("stage") == "population_semantic_classify":
            self.classified_paths.extend(
                member["packet"]["observation_ref"]["path"]
                for member in request["members"]
            )
        return response


def test_out_of_scope_members_are_not_sent_to_the_llm(tmp_path: Path) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _NarrowScopeRecordingClient(
        category="narrow-scope-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)
    assert result.current_source is not None
    population = result.current_source.populations[0]
    deterministic = set(population.classification_plan.deterministic_member_ids)
    outside = [
        member
        for member in population.members
        if not (member.observation_ref.path or "").startswith("src/aurora/")
    ]
    assert outside
    assert all(member.member_id in deterministic for member in outside)
    assert all(
        member.label == "ineligible"
        and member.classification_basis == "structural"
        and member.inference_ref is None
        for member in outside
    )
    assert client.classified_paths
    assert all(path.startswith("src/aurora/") for path in client.classified_paths)
    packet_observation_ids = {
        item.observation_ref.observation_id
        for item in result.current_source.member_packets
        if item.hypothesis_id == population.hypothesis_id
    }
    assert not {
        member.observation_ref.observation_id for member in outside
    } & packet_observation_ids

    evidence_bundle = next(
        bundle
        for bundle in result.unified.bundle.current_evidence
        if any(
            denominator.population_id == population.population_id
            for denominator in bundle.denominators
        )
    )
    denominator = next(
        item
        for item in evidence_bundle.denominators
        if item.population_id == population.population_id
    )
    assert (
        denominator.support,
        denominator.counterexample,
        denominator.unknown,
        denominator.ineligible,
    ) == (
        population.support_count,
        population.counterexample_count,
        population.unknown_count,
        population.ineligible_count,
    )
    assert "population_member_evidence_incomplete" not in (
        evidence_bundle.incomplete_reasons
    )
    assert validate_run_directory(run, repository_root=repository) == []


class _RecursiveScopeRecordingClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.classified_paths: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "current_hypothesis_reduce":
            value = json.loads(response.content)
            for hypothesis in value.get("hypotheses", []):
                hypothesis["applies_to"] = ["src/aurora/**/*.py"]
                if request.get("mode") == "component_partition":
                    bindings = request["population_context"].get(
                        "witness_bindings", []
                    )
                    scoped = next(
                        (
                            item["witness_id"]
                            for item in bindings
                            if str(item.get("path", "")).startswith(
                                "src/aurora/"
                            )
                        ),
                        None,
                    )
                    if scoped is not None:
                        hypothesis["witness_refs"] = [scoped]
            return ProviderResponse(
                content=json.dumps(value, sort_keys=True, separators=(",", ":")),
                model="recursive-scope-fake",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )
        if request.get("stage") == "population_semantic_classify":
            self.classified_paths.extend(
                member["packet"]["observation_ref"]["path"]
                for member in request["members"]
            )
        return response


def test_recursive_scope_survives_preflight_and_classifies_all_depths(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    nested = repository / "src" / "aurora" / "nested" / "deep.py"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        "def recursive_scope_member() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    client = _RecursiveScopeRecordingClient(
        category="recursive-scope-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )

    result = build_fixture(repository, tmp_path / "run", client=client)

    assert result.current_source is not None
    assert result.current_source.reduction.index.unique_candidate_roots == 1
    assert {"src/aurora/api.py", "src/aurora/nested/deep.py"} <= set(
        client.classified_paths
    )
    assert not any(
        path.startswith("src/extensions/") for path in client.classified_paths
    )
    scoped_members = [
        member
        for member in result.current_source.populations[0].members
        if member.observation_ref.path
        in {"src/aurora/api.py", "src/aurora/nested/deep.py"}
    ]
    assert scoped_members
    assert all(
        member.reason_code != "outside_hypothesis_scope"
        for member in scoped_members
    )


def test_mock_classifier_uses_recursive_scope_for_zero_and_nested_depths() -> None:
    response = MockLLMClient().complete(
        "unused",
        json.dumps(
            {
                "stage": "population_semantic_classify",
                "roots": {"r0": {"hypothesis": {"applies_to": ["src/pkg/**/*.py"]}}},
                "reason_catalog": {
                    "c0": "mock_support", "c1": "mock_counterexample",
                    "c2": "mock_ineligible", "c3": "mock_unknown",
                },
                "members": [
                    {
                        "member_ordinal": 0,
                        "root_ref": "r0",
                        "member_id": "top-level",
                        "packet": {
                            "observation_ref": {"path": "src/pkg/mod.py"},
                            "evidence_refs": ["EVIDENCE-top"],
                        },
                    },
                    {
                        "member_ordinal": 1,
                        "root_ref": "r0",
                        "member_id": "nested",
                        "packet": {
                            "observation_ref": {
                                "path": "src/pkg/nested/deep/mod.py"
                            },
                            "evidence_refs": ["EVIDENCE-nested"],
                        },
                    },
                ],
            }
        ),
    )

    labels = {
        item["member_ordinal"]: item["label"]
        for item in json.loads(response.content)["results"]
    }
    assert labels == {0: "support", 1: "counterexample"}


def test_missing_semantic_member_fails_batch_to_technical_unknown(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=_MissingMemberClient(
            category="missing-member-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    population = result.current_source.populations[0]
    assert population.status == "partial"
    assert population.technical_unknown_count == len(population.members)
    assert population.semantic_unknown_count == 0
    assert not result.build_report["complete"]
    decision = result.current_source.policy_decisions[0]
    assert decision.experimental_disposition == "review"
    assert decision.reason_codes == ["population_incomplete"]
    assert result.current_source.falsifications == ()
    assert not any(
        record.stage == "counterexample_verify"
        for record in result.inference_records
    )
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


class _InvalidPopulationSpecClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = {
                "op": "filter",
                "source": {"op": "universe", "universe": "entity"},
                "predicates": [
                    {
                        "field": "category_specific_magic",
                        "operator": "eq",
                        "value": True,
                    }
                ],
            }
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="invalid-population-spec-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_invalid_population_spec_is_quarantined_after_bounded_repair(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_InvalidPopulationSpecClient(
            category="invalid-query-fixture",
            rule="This candidate carries an invalid generic query.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.index.status == "complete"
    assert result.current_source.index.failed_populations == 0
    assert result.current_source.reduction.index.failed_calls == 0
    assert result.current_source.reduction.index.unique_candidate_roots == 0
    quarantined = [
        item
        for call in result.current_source.reduction.call_results
        for item in call.quarantined_hypotheses
    ]
    assert quarantined
    assert result.current_source.reduction.index.invalid_candidate_events == len(
        quarantined
    )
    assert len(result.current_source.reduction.events) == len(quarantined)
    assert all(
        event.normalization == "invalid"
        and "population_repair_exhausted" in event.reason_codes
        for event in result.current_source.reduction.events
    )
    assert "invalid_hypotheses_quarantined_after_repair" in {
        reason
        for call in result.current_source.reduction.call_results
        for reason in call.reason_codes
    }


class _EmptyWitnessPopulationClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.partition_repair_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        response = super().complete(system_prompt, user_prompt)
        if (
            request.get("stage") != "current_hypothesis_reduce"
            or request.get("mode") != "component_partition"
        ):
            return response
        if payload.get("repair"):
            self.partition_repair_calls += 1
        from test_open_hypothesis_reduction import _editable_reducer_response

        value = _editable_reducer_response(response, request)
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = (
                {
                    "op": "universe",
                    "universe": "entity",
                }
                if payload.get("repair")
                else {
                    "op": "filter",
                    "source": {"op": "universe", "universe": "entity"},
                    "predicates": [
                        {
                            "field": "qualified_name",
                            "operator": "eq",
                            "value": "definitely.missing",
                        }
                    ],
                }
            )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="population-preflight-repair-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_empty_witness_population_is_excluded_at_final_admission(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _EmptyWitnessPopulationClient(
        category="population-preflight-repair",
        rule="Contributors preserve the fixture convention.",
    )
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=client,
    )
    assert result.current_source is not None
    repository_reduce = [
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
    ]
    assert repository_reduce
    assert all(record.terminal_status == "success" and len(record.attempts) == 1
               for record in repository_reduce)
    assert client.partition_repair_calls == 0
    reduction = result.current_source.reduction
    assert reduction.index.unique_candidate_roots == 0
    assert reduction.events
    assert all(event.normalization == "invalid"
               and event.reason_codes == ["invalid_population_preflight"]
               for event in reduction.events)


class _FilePopulationClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        for hypothesis in value.get("hypotheses", []):
            hypothesis["superpopulation_spec"] = {
                "op": "universe",
                "universe": "file",
            }
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="file-population-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _RecordingFilePopulationClient(_FilePopulationClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.user_prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        self.user_prompts.append(user_prompt)
        return super().complete(system_prompt, user_prompt)


def test_file_population_packets_include_bounded_full_file_source(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_FilePopulationClient(
            category="file-source-context",
            rule="Contributors preserve the fixture file-level convention.",
        ),
    )
    assert result.current_source is not None
    file_packets = [
        packet
        for packet in result.current_source.member_packets
        if packet.observation_ref.universe == "file"
    ]
    assert file_packets
    assert all(packet.fact.get("full_file_source") for packet in file_packets)
    for packet in file_packets:
        source = packet.fact["full_file_source"]
        assert source["text"] == (
            repository / packet.observation_ref.path
        ).read_text(encoding="utf-8")


def test_noncomplete_file_member_is_quarantined_from_source_and_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    quarantined_path = "src/aurora/quarantined.py"
    sentinel = "SENTINEL_NONCOMPLETE_SOURCE_MUST_NOT_ESCAPE"
    (repository / quarantined_path).write_text(
        f"{sentinel} = (\n",
        encoding="utf-8",
    )
    source_reads: list[str] = []
    original_read = semantic_classify.safe_read_manifest_file

    def recording_read(repository_root, manifest_file):
        source_reads.append(manifest_file.path)
        return original_read(repository_root, manifest_file)

    monkeypatch.setattr(
        semantic_classify,
        "safe_read_manifest_file",
        recording_read,
    )
    client = _RecordingFilePopulationClient(
        category="quarantined-file-source",
        rule="Contributors preserve the fixture file-level convention.",
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=client)

    assert result.current_source is not None
    quarantined_packet = next(
        packet
        for packet in result.current_source.member_packets
        if packet.observation_ref.path == quarantined_path
    )
    population = next(
        item
        for item in result.current_source.populations
        if item.hypothesis_id == quarantined_packet.hypothesis_id
    )
    member = next(
        item
        for item in population.members
        if item.observation_ref.observation_id
        == quarantined_packet.observation_ref.observation_id
    )
    assigned = {
        member_id
        for batch in population.classification_plan.batches
        for member_id in batch.member_ids
    }

    assert quarantined_path not in source_reads
    assert all(sentinel not in prompt for prompt in client.user_prompts)
    assert sentinel not in json.dumps(quarantined_packet.fact, sort_keys=True)
    assert quarantined_packet.owning_unit_id is None
    assert not (
        {"full_file_source", "semantic_unit", "source_ranges", "text"}
        & set(quarantined_packet.fact)
    )
    assert "source_quarantined" in quarantined_packet.missing_context
    assert member.label == "unknown"
    assert member.classification_basis == "technical_unknown"
    assert member.inference_ref is None
    assert member.member_id not in assigned
    assert member.member_id in population.classification_plan.deterministic_member_ids
    assert population.population_result.status == "complete"
    assert population.population_result.dependency_file_ids == []
    assert population.status == "partial"
    assert "technical_unknown_members" in population.incomplete_reasons
    assert validate_run_directory(run, repository_root=repository) == []

    packet_path = (
        run
        / "current-source"
        / "member-packets"
        / f"{quarantined_packet.packet_id}.json"
    )
    packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_payload["observation_ref"]["file_id"] = next(
        packet.observation_ref.file_id
        for packet in result.current_source.member_packets
        if packet.observation_ref.file_id
        != quarantined_packet.observation_ref.file_id
    )
    packet_payload["fact"]["text"] = sentinel
    packet_payload["packet_digest"] = payload_digest(
        {
            key: value
            for key, value in packet_payload.items()
            if key != "packet_digest"
        }
    )
    packet_path.write_bytes(canonical_json_bytes(packet_payload))
    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "non-complete shard packet contains source context" in error
        for error in errors
    )
    assert any(
        "non-complete shard packet has a spoofed owning file" in error
        for error in errors
    )


def test_fact_index_does_not_access_or_mutate_quarantined_cache_entry(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    quarantined_path = "src/aurora/quarantined.py"
    (repository / quarantined_path).write_text("def broken(:\n", encoding="utf-8")
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_FilePopulationClient(
            category="quarantined-cache",
            rule="Contributors preserve the fixture file-level convention.",
        ),
    )
    assert result.current_source is not None
    assert result.observation is not None
    assert result.semantic_units is not None
    hypothesis = result.current_source.hypotheses[0]
    quarantined_shard = next(
        shard
        for shard in result.observation.shards
        if shard.path == quarantined_path
    )
    sentinel = b"SENTINEL_CALLER_OWNED_CACHE"
    cache = {quarantined_shard.file_id: sentinel}

    facts = _fact_index(
        repository,
        result.source_acquisition.discovery,
        result.observation,
        result.semantic_units,
        hypothesis,
        config_for_tests().semantic,
        cache,
        {quarantined_shard.file_id},
    )

    assert cache == {quarantined_shard.file_id: sentinel}
    fact = facts[quarantined_shard.file_id]
    assert fact.provider_eligible is False
    assert sentinel.decode("utf-8") not in json.dumps(fact.payload, sort_keys=True)
