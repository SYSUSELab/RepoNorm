from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.run_validation import validate_run_directory


class _DelayedMultiRootClient(ParameterizedSemanticClient):
    """Create three roots and force completion out of worker-start order."""

    def __init__(self) -> None:
        super().__init__(
            category="cross-root-wave-fixture",
            rule="Contributors preserve the cross-root fixture convention.",
        )
        self.lock = threading.Lock()
        self.started_by_stage: dict[str, list[str]] = {}
        self.completed_by_stage: dict[str, list[str]] = {}
        self.in_flight_by_stage: dict[str, int] = {}
        self.peak_by_stage: dict[str, int] = {}
        self.release_first_by_stage: dict[str, threading.Event] = {}

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        stage = str(request.get("stage") or "explicit_extract")
        hypothesis = request.get("hypothesis")
        hypothesis_id = (
            str(hypothesis.get("hypothesis_id"))
            if isinstance(hypothesis, dict)
            and isinstance(hypothesis.get("hypothesis_id"), str)
            else ""
        )
        if stage == "population_semantic_classify":
            hypothesis_id = "|".join(item["packet"]["packet_id"] for item in request["members"])
        delayed = stage in {
            "population_semantic_classify",
            "counterexample_verify",
        }
        start_ordinal = 0
        if delayed:
            with self.lock:
                started = self.started_by_stage.setdefault(stage, [])
                start_ordinal = len(started)
                started.append(hypothesis_id)
                active = self.in_flight_by_stage.get(stage, 0) + 1
                self.in_flight_by_stage[stage] = active
                self.peak_by_stage[stage] = max(
                    self.peak_by_stage.get(stage, 0),
                    active,
                )
        try:
            if delayed:
                # The first planned item waits for another worker to finish,
                # proving real overlap and a completion/start-order difference
                # without assuming which planned item the scheduler starts first.
                release_first = self.release_first_by_stage.setdefault(
                    stage, threading.Event()
                )
                if start_ordinal == 0:
                    release_first.wait(timeout=1.0)
                else:
                    time.sleep(0.005)
            response = super().complete(system_prompt, user_prompt)
            if (
                stage == "current_hypothesis_reduce"
                and request.get("mode") == "component_partition"
            ):
                from test_open_hypothesis_reduction import _editable_reducer_response

                value = _editable_reducer_response(response, request)
                hypotheses = value.get("hypotheses", [])
                if hypotheses:
                    original = hypotheses[0]
                    value["hypotheses"] = []
                    for ordinal in range(min(3, request["hypothesis_budget"]["max_output_hypotheses"])):
                        draft = json.loads(json.dumps(original))
                        draft["category"] = f"cross-root-wave-{ordinal}"
                        draft["rule"] = (
                            "Contributors preserve cross-root fixture variant "
                            f"{ordinal}."
                        )
                        value["hypotheses"].append(draft)
                    value["outcome"] = "hypotheses"
                response = ProviderResponse(
                    content=json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                )
            return response
        finally:
            if delayed:
                with self.lock:
                    self.completed_by_stage.setdefault(stage, []).append(
                        hypothesis_id
                    )
                    self.in_flight_by_stage[stage] -= 1
                    if start_ordinal > 0:
                        self.release_first_by_stage[stage].set()


def _config():
    config = config_for_tests(
        semantic={
            "max_classification_members_per_request": 32,
            "max_classified_population_members_per_hypothesis": 20,
            "max_falsification_items_per_request": 32,
            "max_total_hypotheses": 3,
        }
    )
    return config.model_copy(
        update={"llm": config.llm.model_copy(update={"max_concurrency": 3})}
    )


def _three_component_repository(tmp_path: Path) -> Path:
    """Give the three-root fixture three real non-expanding reducer inputs."""

    repository = copied_current_source_fixture(tmp_path)
    example = repository / "examples" / "cross_root_wave.py"
    example.parent.mkdir()
    example.write_text(
        "def preserve_cross_root_wave() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    return repository


def _classification_record_roots(result) -> list[str]:
    roots: list[str] = []
    for record in result.inference_records:
        if record.stage != "population_semantic_classify":
            continue
        packet_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "population_member_packet"
        ]
        roots.append("|".join(packet_ids))
    return roots


def _falsification_record_roots(result) -> list[str]:
    return [
        next(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "norm_hypothesis"
        )
        for record in result.inference_records
        if record.stage == "counterexample_verify"
    ]


def test_single_batch_roots_run_concurrently_but_fold_in_plan_order_and_cache(
    tmp_path: Path,
) -> None:
    repository = _three_component_repository(tmp_path)
    config = _config()
    cache = tmp_path / "inference-cache"
    client = _DelayedMultiRootClient()
    run = tmp_path / "cross-root-run"
    result = build_fixture(
        repository,
        run,
        config=config,
        client=client,
        inference_cache=cache,
    )

    assert result.current_source is not None
    current = result.current_source
    assert len(current.index.population_schedule) == 3
    population_by_hypothesis = {
        item.hypothesis_id: item for item in current.populations
    }
    transport = current.index.classification_transport_plans[0]
    packet_ids = {packet.packet_digest: packet.packet_id for packet in current.member_packets}
    planned_roots = [
        "|".join(packet_ids[pair.packet_digest] for pair in request.pairs)
        for request in transport.requests
    ]
    assert transport.admission_estimate_calls == 3
    assert 2 <= transport.execution.planned_calls <= 3
    assert any(len({pair.hypothesis_id for pair in request.pairs}) > 1 for request in transport.requests)
    assert len({record.request_id for record in result.inference_records}) == len(result.inference_records)
    assert 2 <= client.peak_by_stage["population_semantic_classify"] <= 3
    assert _classification_record_roots(result) == planned_roots
    assert set(client.completed_by_stage["population_semantic_classify"]) == set(
        planned_roots
    )
    assert client.completed_by_stage["population_semantic_classify"] != (
        client.started_by_stage["population_semantic_classify"]
    )

    falsification_roots = _falsification_record_roots(result)
    assert len(falsification_roots) == 3
    assert 2 <= client.peak_by_stage["counterexample_verify"] <= 3
    assert falsification_roots == [
        item.hypothesis_id for item in current.index.population_schedule
    ]
    assert set(client.completed_by_stage["counterexample_verify"]) == set(
        falsification_roots
    )
    assert client.completed_by_stage["counterexample_verify"] != (
        client.started_by_stage["counterexample_verify"]
    )
    assert validate_run_directory(run, repository_root=repository) == []

    cached_client = _DelayedMultiRootClient()
    cached = build_fixture(
        repository,
        tmp_path / "cross-root-cached-run",
        config=config,
        client=cached_client,
        inference_cache=cache,
    )
    assert cached.current_source == current
    assert cached.inference_records == result.inference_records
    assert "population_semantic_classify" not in cached_client.started_by_stage
    assert "counterexample_verify" not in cached_client.started_by_stage


def test_cross_root_budget_suffix_is_exact_and_replays_offline(
    tmp_path: Path,
) -> None:
    repository = _three_component_repository(tmp_path)
    config = _config()
    baseline = build_fixture(
        repository,
        tmp_path / "cross-root-budget-baseline",
        config=config,
        client=_DelayedMultiRootClient(),
    )
    pre_classification_calls = sum(
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
    limited_config = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={"max_total_calls": pre_classification_calls + 1}
            )
        }
    )
    run = tmp_path / "cross-root-budget-limited"
    limited = build_fixture(
        repository,
        run,
        config=limited_config,
        client=_DelayedMultiRootClient(),
    )

    assert limited.current_source is not None
    current = limited.current_source
    assert len(current.index.population_schedule) == 3
    population_by_hypothesis = {
        item.hypothesis_id: item for item in current.populations
    }
    executions = [
        population_by_hypothesis[
            item.hypothesis_id
        ].classification_plan.execution
        for item in current.index.population_schedule
    ]
    assert all(item.request_plan_complete for item in executions)
    transport = current.index.classification_transport_plans[0]
    assert transport.execution.terminal_calls == 1
    assert transport.execution.omitted_calls == len(transport.requests) - 1
    assert sum(item.terminal_items for item in executions) == len(transport.requests[0].pairs)
    assert sum(item.omitted_items for item in executions) == transport.execution.omitted_items
    for scheduled, execution in zip(current.index.population_schedule, executions):
        expected = [
            request for request in transport.requests
            if any(pair.hypothesis_id == scheduled.hypothesis_id for pair in request.pairs)
        ]
        assert execution.planned_calls == len(expected)
        assert execution.terminal_calls == sum(request.ordinal == 0 for request in expected)
        assert execution.omitted_calls == sum(request.ordinal > 0 for request in expected)
    stage = next(
        item
        for item in limited.inference_index.stages
        if item.stage == "population_semantic_classify"
    )
    assert stage.omitted == transport.execution.omitted_calls
    assert stage.omission_reasons == {"max_total_calls": transport.execution.omitted_calls}
    assert current.index.resource_use["classification_admitted_hypotheses"] == 3
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


def test_failed_root_artifact_fold_keeps_unique_physical_classification_usage(
    tmp_path: Path, monkeypatch,
) -> None:
    from repo_constraint_memory_v3.semantic import validation as semantic_validation

    original = semantic_validation.finish_population_classification
    failed = False

    def fail_first_root(pending, stage):
        nonlocal failed
        if not failed and stage.outcomes:
            failed = True
            raise ValueError("injected population artifact fold failure")
        return original(pending, stage)

    monkeypatch.setattr(semantic_validation, "finish_population_classification", fail_first_root)
    repository = _three_component_repository(tmp_path)
    result = build_fixture(
        repository, tmp_path / "failed-root-fold", config=_config(),
        client=_DelayedMultiRootClient(),
    )
    assert failed
    current = result.current_source
    assert current.index.failed_populations == 1
    transport = current.index.classification_transport_plans[0]
    records = [record for record in result.inference_records if record.stage == "population_semantic_classify"]
    assert len(records) == transport.execution.terminal_calls
    assert len({record.request_id for record in records}) == len(records)
    assert sum(record.usage.calls for record in records) == len(records)


def test_zero_run_wide_inspection_share_is_partial_not_zero_support_rejection(
    tmp_path: Path,
) -> None:
    repository = _three_component_repository(tmp_path)
    base = _config()
    config = base.model_copy(
        update={
            "semantic": base.semantic.model_copy(
                update={
                    "max_classified_population_members_per_hypothesis": 1,
                    "max_total_classified_population_members": 1,
                }
            )
        }
    )
    run = tmp_path / "zero-inspection-share"
    result = build_fixture(
        repository,
        run,
        config=config,
        client=_DelayedMultiRootClient(),
    )

    assert result.current_source is not None
    current = result.current_source
    assert len(current.index.population_schedule) == 3
    zero_share_ids = {
        item.hypothesis_id
        for item in current.index.population_schedule
        if item.classified_member_limit == 0
    }
    assert len(zero_share_ids) == 2
    population_by_hypothesis = {
        item.hypothesis_id: item for item in current.populations
    }
    for hypothesis_id in zero_share_ids:
        population = population_by_hypothesis[hypothesis_id]
        assert population.sampling_plan.provider_eligible_count > 0
        assert population.sampling_plan.provider_inspected_count == 0
        assert population.status == "partial"
        assert "population_provider_inspection_not_admitted" in (
            population.incomplete_reasons
        )

    decisions = {
        item.hypothesis_id: item for item in current.policy_decisions
    }
    reviews = {item.hypothesis_id: item for item in current.review_items}
    for hypothesis_id in zero_share_ids:
        assert decisions[hypothesis_id].experimental_disposition == "review"
        assert decisions[hypothesis_id].canonical_epistemic_status == "hypothesis"
        assert "no_population_support" not in decisions[hypothesis_id].reason_codes
        assert reviews[hypothesis_id].epistemic_status == "hypothesis"
        assert "no_population_support" not in reviews[hypothesis_id].reason_codes
    assert validate_run_directory(run, repository_root=repository) == []
