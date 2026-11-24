from __future__ import annotations

import json
from functools import partial
from types import SimpleNamespace

import pytest

from helpers import config_for_tests
from repo_constraint_memory_v3.inference import InferenceBudget
from repo_constraint_memory_v3.models import ProviderResponse, SemanticSettings
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.semantic.classify import (
    MemberClassificationResponse,
    _classification_request_bytes,
    _invoke_classification_pairs,
    _expand_compact_classification_wire_response,
    _normalize_response,
    _terminal_validate_response,
    _validate_response,
)
from repo_constraint_memory_v3.semantic.classification_transport import (
    ClassificationPair,
    ClassificationTransportPlan,
    compact_classification_request,
    execute_classification_transport,
    expand_classification_request,
    freeze_classification_transport,
    inspection_payload,
)
from repo_constraint_memory_v3.semantic.ids import semantic_id
from repo_constraint_memory_v3.semantic.models import NormHypothesis, PopulationMemberPacket
from repo_constraint_memory_v3.semantic.validation import (
    _RecordedInferenceCache, _RecordedInferenceClient,
    _ScheduledPopulationRoot, _population_root_allocations,
)


def _pairs(root: int, count: int, *, reason: str = "supports") -> list[ClassificationPair]:
    snapshot = "sha256:" + "0" * 64
    hypothesis_payload = {
        "hypothesis_id": semantic_id("HYP", {"root": root}),
        "candidate_root_id": semantic_id("ROOT", {"root": root}),
        "snapshot_digest": snapshot,
        "version": 0, "parent_hypothesis_id": None, "refinement_round": 0,
        "rule": f"Repository rule {root}.", "category": None,
        "proposed_claim_kind": "observed_convention", "proposed_strength": "observed",
        "applies_to": ["src/**"], "condition": f"Condition {root}.",
        "witness_refs": ["ENTITY-witness"], "witness_ref_repairs": [],
        "explicit_context_refs": [],
        "superpopulation_spec": {"op": "universe", "universe": "entity"},
        "predicate_mode": "semantic", "structural_predicate": None,
        "semantic_predicate": {
            "eligibility_question": f"Does rule {root} apply?",
            "support_question": f"Does the member satisfy rule {root}?",
            "allowed_reason_codes": [reason],
            "expected_evidence_form": "Current source.",
            "ambiguity_instruction": "Unknown when ambiguous.",
            "repository_specificity_test": "Repository convention.",
            "protocol_version": "semantic-member-classification-1.0",
        },
        "independence_spec": {"mode": "clone_aware_observation"},
        "limitations": [], "origin_inference_ids": ["INFER-origin"],
        "completeness_refs": ["complete"], "component_keys": ["src"],
    }
    hypothesis_payload["artifact_digest"] = payload_digest(hypothesis_payload)
    hypothesis = NormHypothesis.model_validate(hypothesis_payload)
    result = []
    for ordinal in range(count):
        packet = {
            "packet_id": semantic_id("MEMPACKET", {"root": root, "member": ordinal}),
            "snapshot_digest": snapshot, "hypothesis_id": hypothesis.hypothesis_id,
            "observation_ref": {
                "universe": "entity", "observation_id": f"ENTITY-{ordinal}",
                "file_id": "FILE-source", "path": "src/example.py",
            },
            "owning_unit_id": "UNIT-owner",
            "fact": {
                "observation": {"entity_id": f"ENTITY-{ordinal}", "name": f"member_{ordinal}"},
                "semantic_unit": {
                    "unit_id": "UNIT-owner", "text": "complete source context\n" * 100,
                },
            },
            "evidence_refs": [f"EVIDENCE-{ordinal}"],
            "clone_group": None, "independence_group": f"root-{root}", "missing_context": [],
        }
        packet["packet_digest"] = payload_digest(packet)
        result.append(ClassificationPair(
            member_id=semantic_id("POPMEMBER", {"root": root, "member": ordinal}),
            hypothesis=hypothesis, packet=PopulationMemberPacket.model_validate(packet),
            inspection=inspection_payload(None, count),
        ))
    return result


class _SupportClient:
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        code = next(key for key, value in request["reason_catalog"].items() if value == "supports")
        return ProviderResponse(
            content=json.dumps({"results": [{
                "member_ordinal": item["member_ordinal"], "label": "support",
                "evidence_context": "production", "reason_code": code,
                "reason_summary": "The supplied complete fact supports this pair.",
                "evidence_refs": item["packet"]["evidence_refs"],
            } for item in request["members"]]}),
            model="classification-transport-fake", input_tokens=1, output_tokens=1,
            latency_ms=0,
        )


def _pending(pairs, *, settings, budget, client=None, batch_size=None):
    batch_size = batch_size or settings.semantic.max_classification_members_per_request
    ids = [pair.member_id for pair in pairs]
    return SimpleNamespace(work=SimpleNamespace(
        hypothesis_id=pairs[0].hypothesis.hypothesis_id,
        pairs=tuple(pairs),
        items=tuple(enumerate([ids[start:start + batch_size] for start in range(0, len(ids), batch_size)])),
        batches=[], semantic_batches=[],
        invoke_pairs=partial(
            _invoke_classification_pairs, semantic_settings=settings.semantic,
            llm_settings=settings.llm, client=client or _SupportClient(), budget=budget,
            protocol_manifest_digest="0" * 64,
        ),
    ))


def test_compact_catalog_roundtrips_complete_facts_and_root_packet_context() -> None:
    pairs = [*_pairs(0, 2), *_pairs(1, 2)]
    request = json.loads(compact_classification_request(pairs))
    assert len(request["facts"]) == 2
    assert len(request["structures"]) == 1
    assert len(request["roots"]) == 2
    restored = expand_classification_request(request)
    for pair, member in zip(pairs, restored["members"]):
        assert member["packet"] == pair.packet.model_dump(mode="json")
        assert member["hypothesis"]["rule"] == pair.hypothesis.rule
        assert member["hypothesis"]["condition"] == pair.hypothesis.condition
        assert member["population_inspection"] == pair.inspection


def test_equal_observation_ids_do_not_alias_different_full_facts() -> None:
    left, right = _pairs(0, 1)[0], _pairs(1, 1)[0]
    payload = right.packet.model_dump(mode="json", exclude={"packet_digest"})
    payload["fact"]["additional_current_fact"] = "different source context"
    payload["packet_digest"] = payload_digest(payload)
    right = ClassificationPair(right.member_id, right.hypothesis, PopulationMemberPacket.model_validate(payload), right.inspection)
    request = json.loads(compact_classification_request([left, right]))
    assert len(request["facts"]) == 2


def test_short_aliases_are_disjoint_and_normalization_is_cache_idempotent() -> None:
    pair = _pairs(0, 1, reason="c0")[0]
    packet = pair.packet.model_dump(mode="json", exclude={"packet_digest"})
    packet["evidence_refs"] = ["e0"]
    packet["packet_digest"] = payload_digest(packet)
    pair = ClassificationPair(pair.member_id, pair.hypothesis, PopulationMemberPacket.model_validate(packet), pair.inspection)
    request = json.loads(compact_classification_request([pair]))
    assert set(request["reason_catalog"]).isdisjoint(request["reason_catalog"].values())
    assert set(request["evidence_catalog"]).isdisjoint(request["evidence_catalog"].values())
    wire = {"results": [{
        "member_ordinal": 0, "label": "support", "evidence_context": "production",
        "reason_code": next(key for key, value in request["reason_catalog"].items() if value == "c0"),
        "reason_summary": "Current source supports this root.",
        "evidence_refs": request["members"][0]["packet"]["evidence_refs"],
    }]}
    canonical = _expand_compact_classification_wire_response(wire, request=request)
    assert canonical["results"][0]["reason_code"] == "c0"
    assert canonical["results"][0]["evidence_refs"] == ["e0"]
    assert _normalize_response(canonical, member_ids=[pair.member_id]) == canonical
    with pytest.raises(ValueError, match="wire reason code"):
        _expand_compact_classification_wire_response(canonical, request=request)


@pytest.mark.parametrize("canonical_field", ["reason_code", "evidence_refs"])
def test_canonical_wire_values_cannot_bypass_short_reference_contract(canonical_field: str) -> None:
    class CanonicalWireClient(_SupportClient):
        def complete(self, system_prompt, user_prompt):
            response = super().complete(system_prompt, user_prompt)
            value = json.loads(response.content)
            request = json.loads(user_prompt)
            if canonical_field == "reason_code":
                value["results"][0]["reason_code"] = "supports"
            else:
                value["results"][0]["evidence_refs"] = [
                    request["evidence_catalog"][ref]
                    for ref in value["results"][0]["evidence_refs"]
                ]
            return response.model_copy(update={"content": json.dumps(value)})

    config = config_for_tests()
    settings = config.llm.model_copy(update={"schema_repairs": 0})
    _, _, response, record = _invoke_classification_pairs(
        _pairs(0, 2), 0, semantic_settings=config.semantic,
        llm_settings=settings, client=CanonicalWireClient(), budget=InferenceBudget(settings),
        protocol_manifest_digest="0" * 64,
    )
    assert len(record.attempts) == 1
    if canonical_field == "reason_code":
        assert response is None
        assert record.terminal_status == "failure"
        assert any("wire reason code" in reason for reason in record.incomplete_reasons)
    else:
        assert response.results[0].label == "unknown"
        assert response.results[0].reason_code == "insufficient_context"
        assert response.results[1].label == "support"


def test_canonical_success_cache_uses_host_normalization_without_wire_decoding() -> None:
    config = config_for_tests()
    pairs = _pairs(0, 2)
    original = _invoke_classification_pairs(
        pairs, 0, semantic_settings=config.semantic, llm_settings=config.llm,
        client=_SupportClient(), budget=InferenceBudget(config.llm),
        protocol_manifest_digest="0" * 64,
    )
    assert original[2] is not None
    replay_client = _RecordedInferenceClient([original[3]])
    replay_budget = InferenceBudget(config.llm, cache=_RecordedInferenceCache([original[3]]))
    replayed = _invoke_classification_pairs(
        pairs, 0, semantic_settings=config.semantic, llm_settings=config.llm,
        client=replay_client, budget=replay_budget, protocol_manifest_digest="0" * 64,
    )
    assert replayed == original
    assert replay_client.calls == 0


def test_schema_attached_repair_over_byte_cap_is_unissued_and_replays_exactly() -> None:
    class InvalidFirstClient(_SupportClient):
        def __init__(self):
            self.request_bytes = []

        def complete(self, system_prompt, user_prompt):
            self.request_bytes.append(len(user_prompt.encode("utf-8")))
            if len(self.request_bytes) == 1:
                return ProviderResponse(
                    content="{}", model="classification-transport-fake",
                    input_tokens=1, output_tokens=1, latency_ms=0,
                )
            return super().complete(system_prompt, user_prompt)

    config = config_for_tests()
    pairs = _pairs(0, 1)
    byte_cap = _classification_request_bytes(compact_classification_request(pairs))
    semantic = config.semantic.model_copy(update={"max_classification_request_bytes": byte_cap})
    settings = config.llm.model_copy(update={"schema_repairs": 1})
    client = InvalidFirstClient()
    original = _invoke_classification_pairs(
        pairs, 0, semantic_settings=semantic, llm_settings=settings,
        client=client, budget=InferenceBudget(settings), protocol_manifest_digest="0" * 64,
    )
    assert original[2] is None
    record = original[3]
    assert record.terminal_status == "failure"
    assert client.request_bytes == [byte_cap]
    assert record.usage.calls == len(record.attempts) == 1
    assert "max_population_semantic_classify_request_bytes" in record.incomplete_reasons
    replay_client = _RecordedInferenceClient([record])
    replayed = _invoke_classification_pairs(
        pairs, 0, semantic_settings=semantic, llm_settings=settings,
        client=replay_client,
        budget=InferenceBudget(settings, cache=_RecordedInferenceCache([record])),
        protocol_manifest_digest="0" * 64,
    )
    assert replayed == original
    assert replay_client.calls == 0


def test_cross_root_reason_union_does_not_grant_another_pairs_reason() -> None:
    pairs = [_pairs(0, 1, reason="root_zero")[0], _pairs(1, 1, reason="root_one")[0]]
    response = MemberClassificationResponse.model_validate({"results": [{
        "member_ordinal": ordinal, "label": "support", "evidence_context": "production",
        "reason_code": "root_zero", "reason_summary": "Independent judgment.",
        "evidence_refs": list(pair.packet.evidence_refs),
    } for ordinal, pair in enumerate(pairs)]})
    kwargs = {
        "member_ids": [pair.member_id for pair in pairs],
        "allowed_by_member": {pair.member_id: set(pair.packet.evidence_refs) for pair in pairs},
        "allowed_reason_codes": {"root_zero", "root_one", "insufficient_context"},
        "allowed_reasons_by_member": {
            pair.member_id: set(pair.hypothesis.semantic_predicate.allowed_reason_codes) | {"insufficient_context"}
            for pair in pairs
        },
    }
    with pytest.raises(ValueError, match="reason code outside"):
        _validate_response(response, **kwargs)
    with pytest.raises(ValueError, match="reason code outside"):
        _terminal_validate_response(response, **kwargs)


def test_terminal_foreign_pair_evidence_preserves_legal_sibling_and_strict_coverage() -> None:
    pairs = [_pairs(0, 2)[0], _pairs(1, 2)[1]]
    response = MemberClassificationResponse.model_validate({"results": [{
        "member_ordinal": ordinal, "label": "support", "evidence_context": "production",
        "reason_code": "supports", "reason_summary": "Independent judgment.",
        "evidence_refs": list(pairs[1].packet.evidence_refs),
    } for ordinal in range(2)]})
    kwargs = {
        "member_ids": [pair.member_id for pair in pairs],
        "allowed_by_member": {pair.member_id: set(pair.packet.evidence_refs) for pair in pairs},
        "allowed_reason_codes": {"supports", "insufficient_context"},
    }
    with pytest.raises(ValueError, match="unsupplied evidence"):
        _validate_response(response, **kwargs)
    terminal = _terminal_validate_response(response, **kwargs)
    assert terminal.results[0].label == "unknown"
    assert terminal.results[0].evidence_refs == pairs[0].packet.evidence_refs
    assert terminal.results[1] == response.results[1]
    with pytest.raises(ValueError, match="coverage differs"):
        _terminal_validate_response(response.model_copy(update={"results": response.results[:1]}), **kwargs)
    with pytest.raises(ValueError, match="duplicate member"):
        _terminal_validate_response(response.model_copy(update={"results": [response.results[0], response.results[0]]}), **kwargs)


def test_transport_cannot_cross_refinement_barrier() -> None:
    pairs = [_pairs(0, 1)[0], _pairs(1, 1)[0]]
    payload = pairs[1].hypothesis.model_dump(mode="json", exclude={"artifact_digest"})
    payload.update(version=1, refinement_round=1, parent_hypothesis_id=pairs[0].hypothesis.hypothesis_id)
    payload["artifact_digest"] = payload_digest(payload)
    child = NormHypothesis.model_validate(payload)
    pairs[1] = ClassificationPair(pairs[1].member_id, child, pairs[1].packet, pairs[1].inspection)
    with pytest.raises(ValueError, match="refinement barrier"):
        compact_classification_request(pairs)


@pytest.mark.parametrize(("batch_size", "request_sizes"), [(16, [16, 16, 8]), (32, [32, 8])])
def test_twenty_member_roots_share_facts_without_expanding_pair_admission(
    batch_size, request_sizes,
) -> None:
    settings = config_for_tests(semantic={"max_classification_members_per_request": batch_size})
    budget = InferenceBudget(settings.llm)
    pending = [_pending(_pairs(root, 20), settings=settings, budget=budget) for root in range(2)]
    frozen = freeze_classification_transport(pending, semantic_settings=settings.semantic)
    assert frozen.admission_estimate_calls == (4 if batch_size == 16 else 2)
    assert [len(request.pairs) for request in frozen.requests] == request_sizes
    assert all(len({pair.hypothesis_id for pair in request.pairs}) == 2 for request in frozen.requests)
    assert sum(request.request_bytes for request in frozen.requests) < frozen.admission_estimate_request_bytes
    assert len({(pair.hypothesis_id, pair.member_id) for request in frozen.requests for pair in request.pairs}) == 40
    for item in pending:
        assert {
            pair.member_id for request in frozen.requests for pair in request.pairs
            if pair.hypothesis_id == item.work.hypothesis_id
        } == {pair.member_id for pair in item.work.pairs}
    assert freeze_classification_transport(pending, semantic_settings=settings.semantic) == frozen


@pytest.mark.parametrize(("batch_size", "planned_calls"), [(16, 3), (32, 2)])
def test_physical_budget_omission_projects_to_each_roots_exact_pair_subset(
    batch_size, planned_calls,
) -> None:
    settings = config_for_tests(semantic={"max_classification_members_per_request": batch_size})
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"max_total_calls": 1})})
    budget = InferenceBudget(settings.llm)
    pending = [_pending(_pairs(root, 20), settings=settings, budget=budget) for root in range(2)]
    execution = execute_classification_transport(
        pending, semantic_settings=settings.semantic, llm_settings=settings.llm, budget=budget,
    )
    assert len(execution.records) == 1
    assert sum(record.usage.calls for record in execution.records) == 1
    assert execution.plan.execution.planned_calls == planned_calls
    assert execution.plan.execution.terminal_calls == 1
    assert execution.plan.execution.omitted_calls == planned_calls - 1
    assert execution.plan.execution.terminal_items == batch_size
    assert execution.plan.execution.omitted_items == 40 - batch_size
    assert all(stage.planned == planned_calls and len(stage.outcomes) == 1 and stage.omitted == planned_calls - 1 for stage in execution.root_stages)
    assert sum(len(stage.outcomes[0][0]) for stage in execution.root_stages) == batch_size
    assert execution.root_stages[0].outcomes[0][3] is execution.root_stages[1].outcomes[0][3]
    resealed = execution.plan.model_dump(mode="json", exclude={"plan_digest"})
    resealed["requests"][0]["pairs"][0]["allowed_evidence_refs"] = ["e99999"]
    resealed["plan_digest"] = payload_digest(resealed)
    with pytest.raises(ValueError, match="catalog binding"):
        ClassificationTransportPlan.model_validate(resealed)


def test_real_request_byte_limits_are_not_fractionally_allocated_to_roots() -> None:
    settings = config_for_tests(semantic={"max_classification_members_per_request": 32})
    budget = InferenceBudget(settings.llm)
    pending = [_pending(_pairs(root, 20), settings=settings, budget=budget) for root in range(2)]
    initial = freeze_classification_transport(pending, semantic_settings=settings.semantic)
    root_limit = initial.admission_estimate_request_bytes // 2
    assert root_limit < sum(request.request_bytes for request in initial.requests)
    constrained = settings.semantic.model_copy(update={"max_classification_request_bytes_per_hypothesis": root_limit})
    frozen = freeze_classification_transport(pending, semantic_settings=constrained)
    assert all(len({pair.hypothesis_id for pair in request.pairs}) == 1 for request in frozen.requests)


def test_sixteen_member_batches_preserve_census_and_wide_root_allocations() -> None:
    roots = [
        _ScheduledPopulationRoot(
            hypothesis=SimpleNamespace(
                candidate_root_id=f"root-{ordinal}",
                hypothesis_id=f"hypothesis-{ordinal}",
                predicate_mode="semantic",
            ),
            component_key=f"component-{ordinal % 2}",
            semantic_member_cost=count,
            matched_member_count=count,
            conclusion_eligible=True,
            non_conclusion_reasons=(),
        )
        for ordinal, count in enumerate([20, 20, 100, 100])
    ]
    current = SemanticSettings()
    previous = current.model_copy(update={"max_classification_members_per_request": 32})
    assert _population_root_allocations(roots, semantic_settings=current) == (
        _population_root_allocations(roots, semantic_settings=previous)
    ) == {f"hypothesis-{ordinal}": 20 for ordinal in range(4)}
