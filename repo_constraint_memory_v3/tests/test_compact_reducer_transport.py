from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from helpers import config_for_tests
from test_open_hypothesis_reduction import _draft_fixture_input
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceInputRef,
    MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    StageDefinition,
    error_strings,
    run_structured_inference,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.semantic.reduce import (
    HypothesisDraft,
    ReductionResponse,
    _ReductionRepairController,
    _validate_response,
    REDUCE_PROTOCOL_VERSION,
    REDUCE_RESPONSE_SCHEMA_VERSION,
)
from repo_constraint_memory_v3.semantic.reduce_transport import (
    LocalDraftRepairResponse,
    ReducerWireController,
)


def _controller(*, limit: int = 4) -> ReducerWireController:
    child = _draft_fixture_input(1, 3)
    settings = config_for_tests().semantic

    def preflight(draft: HypothesisDraft) -> None:
        if draft.applies_to == ["bad.py"]:
            raise ValueError("target scope lacks population member")

    def validate(value: ReductionResponse) -> ReductionResponse:
        _validate_response(
            value, mode="component_partition", input_ids=[child.stable_id],
            settings=settings, effective_output_limit=limit,
        )
        for draft in value.hypotheses:
            preflight(draft)
        return value

    controller = ReducerWireController(
        _ReductionRepairController(
            mode="component_partition", input_ids=[child.stable_id], settings=settings,
            effective_output_limit=limit, validate_overflow_draft=preflight,
        ),
        [child], validate, validate,
    )

    controller.drafts = {f"d{ordinal}": draft for ordinal, draft in enumerate(child.payload["hypotheses"])}
    return controller


def _parse(controller: ReducerWireController, value: dict) -> ReductionResponse:
    content = json.dumps(value, ensure_ascii=False)
    controller.inner.bind_response_digest(hashlib.sha256(content.encode()).hexdigest())
    controller.bind_content(content)
    return ReductionResponse.model_validate(controller.normalize(controller.raw(value)))


def _record(controller: ReducerWireController, errors: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(schema_repairs=len(errors) - 1, attempts=[
        SimpleNamespace(
            response_digest=step.response_digest, provider_error=False,
            retryable=False, parsed=not failures, validation_errors=failures,
            schema_attempt=ordinal, repair=ordinal > 0,
        )
        for ordinal, (step, failures) in enumerate(zip(controller.steps, errors))
    ])


def test_full_draft_expands_exact_child_and_replays_wire_provenance() -> None:
    controller = _controller()
    value = _parse(controller, {
        "outcome": "hypotheses", "hypotheses": [_controller().drafts["d1"]], "reason_code": "selected",
    })
    assert value.hypotheses[0].model_dump(mode="json") == controller.drafts["d1"]
    assert controller.validate(value) == value
    record = _record(controller, [[]])
    _controller().validate_record(value, record)
    tampered = value.model_copy(update={"hypotheses": [value.hypotheses[0].model_copy(update={"rule": "Forged rule"})]})
    with pytest.raises(ValueError, match="canonical response"):
        _controller().validate_record(tampered, record)




def test_full_drafts_share_original_output_ceiling() -> None:
    controller = _controller(limit=1)
    value = _parse(controller, {
        "outcome": "hypotheses",
        "hypotheses": [_controller().drafts["d0"], controller.drafts["d1"]],
        "reason_code": "too_many",
    })
    with pytest.raises(ValueError, match="effective request hypothesis limit"):
        controller.validate(value)


def test_local_repair_fixes_only_failed_draft_and_replays_source() -> None:
    controller = _controller()
    good = controller.drafts["d0"]
    bad = {**controller.drafts["d1"], "applies_to": ["bad.py"]}
    first = _parse(controller, {
        "outcome": "hypotheses", "hypotheses": [good, bad], "reason_code": "initial",
    })
    with pytest.raises(ValueError) as failure:
        controller.validate(first)
    errors = error_strings(failure.value)
    assert controller.response_model_for_repair(controller.steps[-1].raw_response, errors) is LocalDraftRepairResponse
    prompt = json.loads(controller.repair_prompt("{}", "", errors))
    assert [item["ordinal"] for item in prompt["replacements_required"]] == [1]
    assert list(prompt["fixed_draft_digests"]) == ["0"]
    final = _parse(controller, {"replacements": [{"ordinal": 1, "draft": controller.drafts["d2"]}]})
    assert final.hypotheses[0].model_dump(mode="json") == good
    controller.validate(final)
    _controller().validate_record(final, _record(controller, [errors, []]))


def test_local_repair_rejects_sibling_edits_and_missing_replacements() -> None:
    for replacements in ([], [{"ordinal": 0, "draft": _controller().drafts["d0"]}]):
        controller = _controller()
        first = _parse(controller, {
            "outcome": "hypotheses",
            "hypotheses": [controller.drafts["d0"], {**controller.drafts["d1"], "applies_to": ["bad.py"]}],
            "reason_code": "initial",
        })
        with pytest.raises(ValueError) as failure:
            controller.validate(first)
        controller.response_model_for_repair("", error_strings(failure.value))
        with pytest.raises(ValueError, match="cover each failed ordinal"):
            _parse(controller, {"replacements": replacements})


def test_local_repair_missing_ordinals_get_exact_feedback_and_replay() -> None:
    controller = _controller()
    cache = _MemoryCache()

    class IncompleteRepairClient(_WireClient):
        def complete(self, system_prompt, user_prompt):
            request = json.loads(user_prompt)
            if self.calls == 2:
                assert [item["ordinal"] for item in request["replacements_required"]] == [1, 2]
                assert list(request["fixed_draft_digests"]) == ["0"]
                assert "missing=[2]" in request["validation_errors"][0]
                assert "including still-failing" in request["instruction"]
                assert "return that required draft unchanged" in request["instruction"]
            return super().complete(system_prompt, user_prompt)

    client = IncompleteRepairClient([
        {"outcome": "hypotheses", "reason_code": "initial", "hypotheses": [
            controller.drafts["d0"],
            {**controller.drafts["d1"], "applies_to": ["bad.py"]},
            {**controller.drafts["d2"], "applies_to": ["bad.py"]},
        ]},
        {"replacements": [{"ordinal": 1, "draft": controller.drafts["d1"]}]},
        {"replacements": [
            {"ordinal": 1, "draft": controller.drafts["d1"]},
            {"ordinal": 2, "draft": controller.drafts["d2"]},
        ]},
    ])
    value, record = _inference(controller, client, cache, repairs=2)
    assert value is not None
    assert client.calls == len(record.attempts) == 3
    assert [draft.model_dump(mode="json") for draft in value.hypotheses] == list(controller.drafts.values())
    cached, cached_record = _inference(_controller(), _WireClient([]), cache, repairs=2)
    assert cached == value
    assert cached_record == record


def test_local_repair_reports_duplicate_and_unexpected_ordinals() -> None:
    controller = _controller()
    first = _parse(controller, {
        "outcome": "hypotheses", "reason_code": "initial", "hypotheses": [
            controller.drafts["d0"],
            {**controller.drafts["d1"], "applies_to": ["bad.py"]},
        ],
    })
    with pytest.raises(ValueError) as failure:
        controller.validate(first)
    controller.response_model_for_repair("", error_strings(failure.value))
    with pytest.raises(ValueError) as failure:
        _parse(controller, {"replacements": [
            {"ordinal": ordinal, "draft": controller.drafts["d1"]}
            for ordinal in (0, 1, 1)
        ]})
    assert "missing=[]" in str(failure.value)
    assert "unexpected=[0] (count=1)" in str(failure.value)
    assert "duplicates=[1] (count=1)" in str(failure.value)


def test_unrepairable_local_draft_remains_invalid_until_terminal_quarantine() -> None:
    controller = _controller()
    cache = _MemoryCache()
    bad = {**controller.drafts["d1"], "independence_spec": {"mode": "entity"}}
    value, record = _inference(controller, _WireClient([
        {"outcome": "hypotheses", "reason_code": "initial", "hypotheses": [
            controller.drafts["d0"], bad,
        ]},
        {"replacements": [{"ordinal": 1, "draft": bad}]},
    ]), cache, repairs=1)
    assert value is not None
    assert len(record.attempts) == 2
    assert value.hypothesis_ordinals == [0]
    assert [item.hypothesis_ordinal for item in value.quarantined_hypotheses] == [1]
    assert value.hypotheses[0].model_dump(mode="json") == controller.drafts["d0"]
    cached, cached_record = _inference(_controller(), _WireClient([]), cache, repairs=1)
    assert cached == value
    assert cached_record == record


def test_local_repair_terminally_quarantines_only_malformed_replacement() -> None:
    controller = _controller()
    first = _parse(controller, {
        "outcome": "hypotheses",
        "hypotheses": [controller.drafts["d0"], {**controller.drafts["d1"], "applies_to": ["bad.py"]}],
        "reason_code": "initial",
    })
    with pytest.raises(ValueError) as failure:
        controller.validate(first)
    errors = error_strings(failure.value)
    controller.response_model_for_repair("", errors)
    malformed = dict(controller.drafts["d1"])
    malformed.pop("superpopulation_spec")
    with pytest.raises(ValueError) as final_error:
        _parse(controller, {"replacements": [{"ordinal": 1, "draft": malformed}]})
    terminal = controller.terminal_parse(controller._attach(controller.last_normalized), final_error.value)
    assert terminal.hypothesis_ordinals == [0]
    assert [item.hypothesis_ordinal for item in terminal.quarantined_hypotheses] == [1]
    record = _record(controller, [errors, []])
    _controller().validate_record(terminal, record)
    record.schema_repairs = 2
    with pytest.raises(ValueError, match="terminal repair"):
        _controller().validate_record(terminal, record)


def test_local_repair_does_not_partition_damaged_top_level() -> None:
    controller = _controller()
    with pytest.raises(ValueError):
        _parse(controller, {
            "outcome": "hypotheses", "reason_code": "invalid_envelope", "unexpected": True,
            "hypotheses": [controller.drafts["d0"], {**controller.drafts["d1"], "applies_to": ["bad.py"]}],
        })
    assert controller.response_model_for_repair("", ["unexpected extra field"]) is controller.provider_model


def test_full_draft_catalog_and_raw_response_digest_tampering_fail_closed() -> None:
    controller = _controller()
    value = _parse(controller, {"outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"]], "reason_code": "full_draft"})
    record = _record(controller, [[]])
    payload = value.model_dump(mode="json")
    payload["host_transport"]["steps"][0]["raw_response"] = "{}"
    with pytest.raises(ValueError, match="source response digest"):
        ReductionResponse.model_validate(payload)


def _inference(controller, client, cache=None, *, repairs=1):
    settings = config_for_tests().llm.model_copy(update={"schema_repairs": repairs})
    definition = StageDefinition(
        stage="current_hypothesis_reduce", mode="component_partition",
        protocol_version=REDUCE_PROTOCOL_VERSION,
        response_schema_version=REDUCE_RESPONSE_SCHEMA_VERSION,
        system_prompt="Use the supplied reducer contract.",
        response_model=ReductionResponse,
        provider_response_model=controller.provider_model,
        repair_response_model=controller.response_model_for_repair,
        validate_response=controller.validate_live,
        terminal_validate_response=controller.terminal_validate_live,
        normalize_response=controller.normalize,
        validate_raw_response=controller.raw,
        terminal_parse_response=controller.terminal_parse,
        repair_prompt=controller.repair_prompt,
        bind_response_digest=controller.inner.bind_response_digest,
        bind_response_content=controller.bind_content,
        validate_record_response=controller.validate_record,
        max_response_bytes=MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    )
    child = controller.inputs[0]
    request = json.dumps({
        "stage": "current_hypothesis_reduce", "mode": "component_partition",
        "hypothesis_budget": {"max_output_hypotheses": controller.inner.effective_output_limit},
    })
    controller.original_request = request
    return run_structured_inference(
        definition=definition,
        inputs=[InferenceInputRef(kind="semantic_observation", stable_id=child.stable_id,
                                  digest=child.digest, snapshot_digest="sha256:" + "a" * 64)],
        request_ordinal=0, request_payload=request,
        client=client, settings=settings, budget=InferenceBudget(settings, cache=cache),
        protocol_manifest_digest="b" * 64, snapshot_digest="sha256:" + "a" * 64,
    )


class _WireClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system_prompt, user_prompt):
        assert self.responses, "unexpected provider call"
        self.calls += 1
        value = self.responses.pop(0)
        content = value if isinstance(value, str) else json.dumps(value)
        return ProviderResponse(content=content, model="mock-reducer",
                                input_tokens=1, output_tokens=1, latency_ms=0)


class _MemoryCache:
    def __init__(self):
        self.values = {}

    def read(self, key):
        return self.values.get(key)

    def write(self, key, record):
        self.values[key] = record


def test_full_draft_live_success_cache_replays_without_provider() -> None:
    cache = _MemoryCache()
    client = _WireClient([{"outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"]], "reason_code": "full_draft"}])
    value, record = _inference(_controller(), client, cache)
    assert value is not None
    assert record.terminal_status == "success"
    second = _WireClient([])
    cached_value, cached_record = _inference(_controller(), second, cache)
    assert second.calls == 0
    assert cached_value == value
    assert cached_record == record


def test_full_draft_terminal_overselection_preserves_original_ordinal_accounting() -> None:
    controller = _controller(limit=1)
    cache = _MemoryCache()
    client = _WireClient([
        {"outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"], _controller().drafts["d1"], _controller().drafts["d2"]], "reason_code": "overflow"},
        {"selected_hypothesis_ordinals": [0, 1]},
    ])
    value, record = _inference(controller, client, cache)
    assert value is not None
    assert record.terminal_status == "success"
    assert value.hypothesis_ordinals == [0]
    assert value.host_cardinality_selection.cardinality_not_selected_source_ordinals == [1, 2]
    cached, cached_record = _inference(_controller(limit=1), _WireClient([]), cache)
    assert cached == value
    assert cached_record == record


def test_local_repair_live_cache_preserves_fixed_sibling() -> None:
    controller = _controller()
    client = _WireClient([
        {"outcome": "hypotheses", "hypotheses": [controller.drafts["d0"], {**controller.drafts["d1"], "applies_to": ["bad.py"]}], "reason_code": "local"},
        {"replacements": [{"ordinal": 1, "draft": controller.drafts["d2"]}]},
    ])
    cache = _MemoryCache()
    value, record = _inference(controller, client, cache)
    assert value is not None
    assert value.hypotheses[0].model_dump(mode="json") == controller.drafts["d0"]
    cached, cached_record = _inference(_controller(), _WireClient([]), cache)
    assert cached == value
    assert cached_record == record


def test_no_hypothesis_does_not_create_a_draft() -> None:
    controller = _controller()
    value = _parse(controller, {"outcome": "no_hypothesis", "hypotheses": [], "reason_code": "explicitly_omitted"})
    assert value.hypotheses == []
    _controller().validate_record(value, _record(controller, [[]]))


def test_local_repair_keeps_fixed_sibling_normalization_provenance() -> None:
    controller = _controller()
    fixed = dict(controller.drafts["d0"])
    fixed.pop("structural_predicate")
    client = _WireClient([
        {"outcome": "hypotheses", "hypotheses": [fixed, {**controller.drafts["d1"], "applies_to": ["bad.py"]}], "reason_code": "local"},
        {"replacements": [{"ordinal": 1, "draft": controller.drafts["d2"]}]},
    ])
    cache = _MemoryCache()
    value, record = _inference(controller, client, cache)
    assert value is not None
    assert [(item.hypothesis_ordinal, item.repair_code) for item in value.host_population_repairs] == [
        (0, "inactive_structural_predicate_defaulted_null")
    ]
    cached, cached_record = _inference(_controller(), _WireClient([]), cache)
    assert cached == value
    assert cached_record == record


def test_reducer_fenced_json_replays_exact_raw_bytes_from_success_cache() -> None:
    cache = _MemoryCache()
    response = "```json\n" + json.dumps({
        "outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"]],
        "reason_code": "full_draft",
    }) + "\n```"
    value, record = _inference(_controller(), _WireClient([response]), cache)
    assert value is not None
    assert value.host_transport.steps[0].raw_response == response
    cached, cached_record = _inference(_controller(), _WireClient([]), cache)
    assert cached == value
    assert cached_record == record


def test_all_actual_reducer_responses_require_wire_provenance() -> None:
    controller = _controller()
    value, record = _inference(controller, _WireClient([{
        "outcome": "hypotheses", "hypotheses": [controller.drafts["d0"]],
        "reason_code": "full_draft",
    }]))
    assert value is not None
    assert value.host_transport is not None
    tampered = value.model_copy(update={"host_transport": None})
    with pytest.raises(ValueError, match="wire provenance"):
        _controller().validate_record(tampered, record)


def test_later_local_repair_fixes_newly_valid_siblings_without_extra_rounds() -> None:
    controller = _controller()
    cache = _MemoryCache()

    class ProgressiveClient(_WireClient):
        def complete(self, system_prompt, user_prompt):
            request = json.loads(user_prompt)
            if self.calls == 2:
                assert list(request["fixed_draft_digests"]) == ["0", "1"]
                assert [item["ordinal"] for item in request["replacements_required"]] == [2]
            return super().complete(system_prompt, user_prompt)

    client = ProgressiveClient([
        {"outcome": "hypotheses", "reason_code": "initial", "hypotheses": [
            controller.drafts["d0"],
            {**controller.drafts["d1"], "applies_to": ["bad.py"]},
            {**controller.drafts["d2"], "applies_to": ["bad.py"]},
        ]},
        {"replacements": [
            {"ordinal": 1, "draft": controller.drafts["d1"]},
            {"ordinal": 2, "draft": {**controller.drafts["d2"], "applies_to": ["bad.py"]}},
        ]},
        {"replacements": [{"ordinal": 2, "draft": controller.drafts["d2"]}]},
    ])
    value, record = _inference(controller, client, cache, repairs=2)
    assert value is not None
    assert client.calls == len(record.attempts) == 3
    assert [draft.model_dump(mode="json") for draft in value.hypotheses] == list(controller.drafts.values())
    cached, cached_record = _inference(_controller(), _WireClient([]), cache, repairs=2)
    assert cached == value
    assert cached_record == record


def test_reducer_cache_replays_failed_json_attempt_before_valid_full_draft() -> None:
    cache = _MemoryCache()
    value, record = _inference(_controller(), _WireClient([
        '{"hypotheses":',
        {"outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"]],
         "reason_code": "corrected"},
    ]), cache)
    assert value is not None
    assert len(value.host_transport.steps) == 2
    assert record.attempts[0].validation_errors
    cached, cached_record = _inference(_controller(), _WireClient([]), cache)
    assert cached == value
    assert cached_record == record


def test_live_reducer_reuses_its_exact_successful_preflight() -> None:
    controller = _controller()
    preflighted = []
    validate = controller.validate

    def counted(value):
        preflighted.append(value)
        return validate(value)

    controller.validate = counted
    value, record = _inference(controller, _WireClient([{
        "outcome": "hypotheses", "hypotheses": [_controller().drafts["d0"]],
        "reason_code": "full_draft",
    }]))
    assert value is not None
    assert record.terminal_status == "success"
    assert len(preflighted) == 1
