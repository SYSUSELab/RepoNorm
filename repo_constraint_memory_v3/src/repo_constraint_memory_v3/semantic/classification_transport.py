"""Lossless request catalogs and a frozen transport for independent root pairs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import Field, model_validator

from repo_constraint_memory_v3.inference import (
    BudgetedStageRunResult, InferenceBudget, InferenceRecord, attach_response_json_schema,
    request_identity, run_budgeted_stage_calls,
)
from repo_constraint_memory_v3.models import (
    LLMSettings, NonEmptyString, SemanticSettings, Sha256Hex, SnapshotDigest, StrictModel,
)

from repo_constraint_memory_v3.observations.canonical import canonical_json_bytes, payload_digest, sha256_hex
from repo_constraint_memory_v3.semantic.models import (
    BoundedStageExecution,
    ClassificationBatch,
    HypothesisId,
    NormHypothesis,
    PopulationMemberPacket,
    PopulationMemberId,
    PopulationArtifact,
    PopulationSamplingPlan,
)

CLASSIFICATION_TRANSPORT_PLAN_VERSION = "classification-transport-plan-1.0"


class ClassificationTransportPair(StrictModel):
    ordinal: int = Field(ge=0)
    hypothesis_id: HypothesisId
    hypothesis_version: int = Field(ge=0)
    member_id: PopulationMemberId
    packet_digest: Sha256Hex
    fact_digest: Sha256Hex
    root_ref: NonEmptyString
    fact_ref: NonEmptyString
    allowed_evidence_refs: list[NonEmptyString]
    allowed_reason_codes: list[NonEmptyString]


class ClassificationTransportRequest(StrictModel):
    ordinal: int = Field(ge=0)
    inference_id: NonEmptyString
    request_payload_digest: Sha256Hex
    request_bytes: int = Field(ge=1)
    pairs: list[ClassificationTransportPair]
    evidence_catalog: dict[str, str]
    reason_catalog: dict[str, str]
    fact_catalog_digests: dict[NonEmptyString, Sha256Hex]
    structure_catalog_digests: dict[NonEmptyString, Sha256Hex]
    root_catalog_digests: dict[NonEmptyString, Sha256Hex]

    @model_validator(mode="after")
    def exact_bindings(self) -> ClassificationTransportRequest:
        if not self.pairs or [pair.ordinal for pair in self.pairs] != list(range(len(self.pairs))):
            raise ValueError("classification transport pair ordinals differ")
        if len({(pair.hypothesis_id, pair.member_id) for pair in self.pairs}) != len(self.pairs):
            raise ValueError("classification transport duplicates a pair")
        for pair in self.pairs:
            if (
                not pair.allowed_evidence_refs or not pair.allowed_reason_codes
                or not set(pair.allowed_evidence_refs) <= self.evidence_catalog.keys()
                or not set(pair.allowed_reason_codes) <= self.reason_catalog.keys()
                or self.fact_catalog_digests.get(pair.fact_ref) != pair.fact_digest
                or pair.root_ref not in self.root_catalog_digests
            ):
                raise ValueError("classification transport catalog binding differs")
        return self


class ClassificationTransportPlan(StrictModel):
    plan_version: Literal["classification-transport-plan-1.0"] = "classification-transport-plan-1.0"
    snapshot_digest: SnapshotDigest
    refinement_round: int = Field(ge=0)
    requests: list[ClassificationTransportRequest]
    admission_estimate_calls: int = Field(ge=0)
    admission_estimate_request_bytes: int = Field(ge=0)
    execution: BoundedStageExecution
    plan_digest: Sha256Hex

    @model_validator(mode="after")
    def exact_algebra(self) -> ClassificationTransportPlan:
        if [item.ordinal for item in self.requests] != list(range(len(self.requests))):
            raise ValueError("classification transport request ordinals differ")
        pairs = [(pair.hypothesis_id, pair.member_id) for request in self.requests for pair in request.pairs]
        if len(pairs) != len(set(pairs)):
            raise ValueError("classification transport repeats a planned pair")
        if self.execution.planned_calls != len(self.requests) or self.execution.planned_items != len(pairs):
            raise ValueError("classification transport denominator differs")
        if len(self.requests) > self.admission_estimate_calls or sum(request.request_bytes for request in self.requests) > self.admission_estimate_request_bytes:
            raise ValueError("classification transport exceeds admission estimates")
        if self.plan_digest != payload_digest(self.model_dump(mode="json", exclude={"plan_digest"})):
            raise ValueError("classification transport digest differs")
        return self

if TYPE_CHECKING:
    from repo_constraint_memory_v3.semantic.classify import (
        ClassificationBatchOutcome, PendingPopulationBuild,
    )


@dataclass(frozen=True)
class ClassificationPair:
    member_id: str
    hypothesis: NormHypothesis
    packet: PopulationMemberPacket
    inspection: dict[str, object]


def inspection_payload(
    sampling_plan: PopulationSamplingPlan | None, member_count: int
) -> dict[str, object]:
    if sampling_plan is not None:
        fields = (
            "mode", "census_enumeration_status", "sampling_plan_status",
            "matched_count", "inspected_count", "uninspected_count",
            "provider_eligible_count", "provider_inspected_count",
            "provider_uninspected_count", "required_inspected_member_ids",
        )
        return {
            **{field: getattr(sampling_plan, field) for field in fields},
            "sampling_plan_digest": sampling_plan.plan_digest,
        }
    return {
        "mode": "census", "census_enumeration_status": "complete",
        "sampling_plan_status": "complete", "matched_count": member_count,
        "inspected_count": member_count, "uninspected_count": 0,
        "provider_eligible_count": member_count,
        "provider_inspected_count": member_count,
        "provider_uninspected_count": 0,
        "required_inspected_member_ids": [], "sampling_plan_digest": None,
    }


def compact_classification_request(pairs: Sequence[ClassificationPair]) -> str:
    """Catalog only complete equal facts; preserve every original packet binding."""

    from repo_constraint_memory_v3.semantic.classify import (
        CLASSIFY_PROTOCOL_VERSION, _allowed_classification_reason_codes,
    )

    if not pairs:
        raise ValueError("classification transport requires pairs")
    snapshots = {pair.hypothesis.snapshot_digest for pair in pairs}
    rounds = {pair.hypothesis.refinement_round for pair in pairs}
    if len(snapshots) != 1 or len(rounds) != 1:
        raise ValueError("classification transport crosses snapshot or refinement barrier")
    evidence = sorted({ref for pair in pairs for ref in pair.packet.evidence_refs})
    reasons = sorted({
        code for pair in pairs
        for code in _allowed_classification_reason_codes(pair.hypothesis)
    })
    evidence_keys = _short_aliases(evidence, "e")
    reason_keys = _short_aliases(reasons, "c")
    roots: dict[str, object] = {}
    root_keys: dict[str, str] = {}
    facts: dict[str, object] = {}
    fact_keys: dict[str, str] = {}
    members: list[dict[str, object]] = []
    for ordinal, pair in enumerate(pairs):
        hypothesis = pair.hypothesis
        if (
            pair.packet.hypothesis_id != hypothesis.hypothesis_id
            or pair.packet.snapshot_digest != hypothesis.snapshot_digest
        ):
            raise ValueError("classification packet belongs to another root or snapshot")
        root_payload = {
            "hypothesis": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_version": hypothesis.version,
                "rule": hypothesis.rule,
                "applies_to": hypothesis.applies_to,
                "condition": hypothesis.condition,
                "predicate": hypothesis.semantic_predicate.model_dump(mode="json"),
            },
            "population_inspection": pair.inspection,
            "allowed_reason_codes": [
                reason_keys[code]
                for code in _allowed_classification_reason_codes(hypothesis)
            ],
        }
        root_key = root_keys.get(hypothesis.hypothesis_id)
        if root_key is None:
            root_key = f"r{len(root_keys)}"
            root_keys[hypothesis.hypothesis_id] = root_key
            roots[root_key] = root_payload
        elif roots[root_key] != root_payload:
            raise ValueError("classification root context differs between pairs")
        fact_digest = payload_digest(pair.packet.fact)
        fact_key = fact_keys.get(fact_digest)
        if fact_key is None:
            fact_key = f"f{len(fact_keys)}"
            fact_keys[fact_digest] = fact_key
            facts[fact_key] = {"fact_digest": fact_digest, "fact": pair.packet.fact}
        elif facts[fact_key]["fact"] != pair.packet.fact:
            raise ValueError("classification fact digest collision")
        packet = pair.packet.model_dump(mode="json", exclude={"fact"})
        packet["fact_ref"] = fact_key
        packet["evidence_refs"] = [evidence_keys[ref] for ref in pair.packet.evidence_refs]
        members.append({
            "member_ordinal": ordinal, "member_id": pair.member_id,
            "root_ref": root_key, "packet": packet,
        })
    # Only exact, typed field values can be interned. No source selection,
    # semantic approximation, or ID-only comparison occurs in this projection.
    shared_fields = {"semantic_unit", "full_file_source", "related_sites", "related_relations"}
    structural_counts = Counter(
        (field, payload_digest({"value": value}))
        for entry in facts.values() for field, value in entry["fact"].items()
        if field in shared_fields and len(canonical_json_bytes(value)) > 128
    )
    structures: dict[str, object] = {}
    structure_keys: dict[tuple[str, str], str] = {}
    for entry in facts.values():
        projected = dict(entry["fact"])
        bindings = {}
        for field, value in entry["fact"].items():
            identity = (field, payload_digest({"value": value}))
            if structural_counts[identity] < 2:
                continue
            key = structure_keys.get(identity)
            if key is None:
                key = f"s{len(structure_keys)}"
                structure_keys[identity] = key
                structures[key] = {"field": field, "value_digest": identity[1], "value": value}
            elif structures[key]["value"] != value:
                raise ValueError("classification structure digest collision")
            bindings[field] = key
            del projected[field]
        entry["fact"] = projected
        entry["structure_refs"] = bindings
    return json.dumps({
        "stage": "population_semantic_classify",
        "mode": "inspected_member_classification",
        "protocol_version": CLASSIFY_PROTOCOL_VERSION,
        "snapshot_digest": pairs[0].hypothesis.snapshot_digest,
        "roots": roots, "facts": facts, "structures": structures, "members": members,
        "evidence_catalog": {evidence_keys[ref]: ref for ref in evidence},
        "reason_catalog": {reason_keys[code]: code for code in reasons},
        "allowed_evidence_refs": sorted(evidence_keys.values()),
        "allowed_reason_codes": sorted(reason_keys.values()),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _short_aliases(values: Sequence[str], prefix: str) -> dict[str, str]:
    """Keep wire aliases disjoint from canonical values for cache normalization."""

    occupied = set(values)
    aliases: dict[str, str] = {}
    ordinal = 0
    for value in values:
        while f"{prefix}{ordinal}" in occupied:
            ordinal += 1
        aliases[value] = f"{prefix}{ordinal}"
        ordinal += 1
    return aliases


def expand_classification_request(request: dict[str, object]) -> dict[str, object]:
    """Restore canonical pair facts for host diagnostics and deterministic fakes."""

    expanded = dict(request)
    expanded_members = []
    for member in request["members"]:
        root = request["roots"][member["root_ref"]]
        packet = dict(member["packet"])
        entry = request["facts"][packet.pop("fact_ref")]
        fact = dict(entry["fact"])
        for field, ref in entry["structure_refs"].items():
            structure = request["structures"][ref]
            if structure["field"] != field or field in fact or payload_digest({"value": structure["value"]}) != structure["value_digest"]:
                raise ValueError("classification structure binding differs")
            fact[field] = structure["value"]
        if payload_digest(fact) != entry["fact_digest"]:
            raise ValueError("classification full fact digest differs")
        packet["fact"] = fact
        packet["evidence_refs"] = [request["evidence_catalog"][ref] for ref in packet["evidence_refs"]]
        PopulationMemberPacket.model_validate(packet)
        expanded_members.append({**member, "packet": packet, **root})
    expanded["members"] = expanded_members
    return expanded


@dataclass(frozen=True)
class FrozenClassificationTransport:
    """One immutable work list, constructed before any provider response."""

    pairs_by_request: tuple[tuple[ClassificationPair, ...], ...]
    requests: tuple[ClassificationTransportRequest, ...]
    admission_estimate_calls: int
    admission_estimate_request_bytes: int


@dataclass(frozen=True)
class ClassificationTransportExecution:
    plan: ClassificationTransportPlan
    records: tuple[InferenceRecord, ...]
    root_stages: tuple[BudgetedStageRunResult[ClassificationBatchOutcome], ...]


def _request_descriptor(
    pairs: Sequence[ClassificationPair], ordinal: int,
) -> ClassificationTransportRequest:
    from repo_constraint_memory_v3.semantic.classify import (
        CLASSIFY_PROTOCOL_VERSION, _classification_input_refs,
        _classification_provider_response_model,
    )

    request = compact_classification_request(pairs)
    payload = json.loads(request)
    contractual = attach_response_json_schema(
        request,
        _classification_provider_response_model(
            tuple(payload["allowed_reason_codes"]),
            tuple(payload["allowed_evidence_refs"]),
        ),
    ).encode("utf-8")
    return ClassificationTransportRequest(
        ordinal=ordinal,
        inference_id=request_identity(
            stage="population_semantic_classify",
            mode="inspected_member_classification",
            protocol_version=CLASSIFY_PROTOCOL_VERSION,
            inputs=_classification_input_refs(pairs), request_ordinal=ordinal,
        ),
        request_payload_digest=sha256_hex(contractual),
        request_bytes=len(contractual),
        pairs=[
            ClassificationTransportPair(
                ordinal=index, hypothesis_id=pair.hypothesis.hypothesis_id,
                hypothesis_version=pair.hypothesis.version,
                member_id=pair.member_id, packet_digest=pair.packet.packet_digest,
                fact_digest=payload["facts"][member["packet"]["fact_ref"]]["fact_digest"],
                root_ref=member["root_ref"], fact_ref=member["packet"]["fact_ref"],
                allowed_evidence_refs=member["packet"]["evidence_refs"],
                allowed_reason_codes=payload["roots"][member["root_ref"]]["allowed_reason_codes"],
            )
            for index, (pair, member) in enumerate(zip(pairs, payload["members"]))
        ],
        evidence_catalog=payload["evidence_catalog"], reason_catalog=payload["reason_catalog"],
        fact_catalog_digests={key: item["fact_digest"] for key, item in payload["facts"].items()},
        structure_catalog_digests={key: payload_digest(item) for key, item in payload["structures"].items()},
        root_catalog_digests={key: payload_digest(item) for key, item in payload["roots"].items()},
    )


def freeze_classification_transport(
    pending: Sequence[PendingPopulationBuild], *, semantic_settings: SemanticSettings,
) -> FrozenClassificationTransport:
    """Compare a bounded fact-ordered plan to already-admitted single-root work.

    The alternative never changes selected pairs or admission. A whole-plan
    comparison avoids assigning arbitrary fractions of physical request bytes
    to roots to obtain a passing estimate.
    """

    from repo_constraint_memory_v3.semantic.classify import _classification_request_bytes

    baseline: list[tuple[ClassificationPair, ...]] = []
    stable_pairs: list[tuple[int, int, ClassificationPair]] = []
    for root_ordinal, item in enumerate(pending):
        pair_by_id = {pair.member_id: pair for pair in item.work.pairs}
        baseline.extend(tuple(pair_by_id[member] for member in batch) for _, batch in item.work.items)
        stable_pairs.extend((root_ordinal, member_ordinal, pair) for member_ordinal, pair in enumerate(item.work.pairs))
    if not baseline:
        raise ValueError("classification transport has no admitted requests")
    baseline_descriptors = tuple(_request_descriptor(pairs, ordinal) for ordinal, pairs in enumerate(baseline))
    baseline_bytes = sum(request.request_bytes for request in baseline_descriptors)
    expected = {(pair.hypothesis.hypothesis_id, pair.member_id) for _, _, pair in stable_pairs}
    if len(expected) != len(stable_pairs):
        raise ValueError("classification transport input repeats a pair")
    chosen = baseline
    descriptors = baseline_descriptors
    if len(pending) > 1:
        ordered = sorted(
            stable_pairs,
            key=lambda item: (payload_digest(item[2].packet.fact), item[0], item[1]),
        )
        candidate: list[tuple[ClassificationPair, ...]] = []
        current: list[ClassificationPair] = []
        for _, _, pair in ordered:
            proposed = [*current, pair]
            fits = (
                len(proposed) <= semantic_settings.max_classification_members_per_request
                and _classification_request_bytes(compact_classification_request(proposed))
                <= semantic_settings.max_classification_request_bytes
            )
            if current and not fits:
                candidate.append(tuple(current))
                current = [pair]
            else:
                current = proposed
        if current:
            candidate.append(tuple(current))
        candidate_descriptors = tuple(_request_descriptor(pairs, ordinal) for ordinal, pairs in enumerate(candidate))
        per_root_calls: dict[str, int] = {}
        per_root_bytes: dict[str, int] = {}
        for request in candidate_descriptors:
            for root_id in {pair.hypothesis_id for pair in request.pairs}:
                per_root_calls[root_id] = per_root_calls.get(root_id, 0) + 1
                per_root_bytes[root_id] = per_root_bytes.get(root_id, 0) + request.request_bytes
        candidate_bytes = sum(request.request_bytes for request in candidate_descriptors)
        if (
            len(candidate) <= len(baseline)
            and candidate_bytes <= baseline_bytes
            and len(candidate) <= semantic_settings.max_total_classification_calls
            and candidate_bytes <= semantic_settings.max_total_classification_request_bytes
            and all(request.request_bytes <= semantic_settings.max_classification_request_bytes for request in candidate_descriptors)
            and all(count <= semantic_settings.max_classification_calls_per_hypothesis for count in per_root_calls.values())
            and all(count <= semantic_settings.max_classification_request_bytes_per_hypothesis for count in per_root_bytes.values())
        ):
            chosen = candidate
            descriptors = candidate_descriptors
    actual = [(pair.hypothesis_id, pair.member_id) for request in descriptors for pair in request.pairs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("classification transport changed the admitted pair set")
    return FrozenClassificationTransport(
        pairs_by_request=tuple(chosen), requests=descriptors,
        admission_estimate_calls=len(baseline),
        admission_estimate_request_bytes=baseline_bytes,
    )


def execute_classification_transport(
    pending: Sequence[PendingPopulationBuild], *, semantic_settings: SemanticSettings,
    llm_settings: LLMSettings, budget: InferenceBudget,
) -> ClassificationTransportExecution:
    """Run frozen physical work once, then project exact pair subsets to roots."""

    from repo_constraint_memory_v3.semantic.classify import MemberClassificationResponse

    frozen = freeze_classification_transport(pending, semantic_settings=semantic_settings)
    # Planning and binding precede all model work. The owning root's closure
    # reads these lists only after the fixed bounded-wave execution completes.
    for item in pending:
        root_id = item.work.hypothesis_id
        batches = []
        for request in frozen.requests:
            pairs = [pair for pair in request.pairs if pair.hypothesis_id == root_id]
            if not pairs:
                continue
            batches.append(ClassificationBatch(
                ordinal=len(batches), member_ids=[pair.member_id for pair in pairs],
                packet_digests=[pair.packet_digest for pair in pairs],
                inference_id=request.inference_id, request_bytes=request.request_bytes,
                transport_ordinal=request.ordinal,
                pair_ordinals=[pair.ordinal for pair in pairs],
                request_payload_digest=request.request_payload_digest,
            ))
        item.work.batches[:] = batches
        item.work.semantic_batches[:] = [list(batch.member_ids) for batch in batches]

    def invoke(ordinal: int):
        outcome = pending[0].work.invoke_pairs(frozen.pairs_by_request[ordinal], ordinal)
        record = outcome[3]
        expected = frozen.requests[ordinal]
        if record.request_id != expected.inference_id or record.request_payload_digest != expected.request_payload_digest:
            raise ValueError("classification inference differs from frozen transport")
        return outcome

    stage = run_budgeted_stage_calls(
        tuple(range(len(frozen.requests))), invoke,
        stage="population_semantic_classify", budget=budget,
        max_concurrency=llm_settings.max_concurrency,
        thread_name_prefix="repo-memory-v3-population-classify",
    )
    records = tuple(outcome[3] for outcome in stage.outcomes)
    if len({record.request_id for record in records}) != len(records):
        raise ValueError("classification transport duplicated a physical inference")
    root_stages = []
    for item in pending:
        outcomes = []
        for batch in item.work.batches:
            ordinal = batch.transport_ordinal
            if ordinal >= len(stage.outcomes):
                continue
            _, packets, parsed, record = stage.outcomes[ordinal]
            by_ordinal = {result.member_ordinal: result for result in parsed.results} if parsed is not None else {}
            projected = (
                MemberClassificationResponse(results=[
                    by_ordinal[global_ordinal].model_copy(update={"member_ordinal": local_ordinal})
                    for local_ordinal, global_ordinal in enumerate(batch.pair_ordinals)
                ]) if parsed is not None else None
            )
            outcomes.append((
                list(batch.member_ids), [packets[index] for index in batch.pair_ordinals],
                projected, record,
            ))
        omitted = len(item.work.batches) - len(outcomes)
        root_stages.append(BudgetedStageRunResult(
            outcomes=tuple(outcomes), planned=len(item.work.batches), omitted=omitted,
            exhaustion_reason=stage.exhaustion_reason if omitted else None,
        ))
    terminal_pairs = sum(len(outcome[0]) for outcome in stage.outcomes)
    planned_pairs = sum(len(request.pairs) for request in frozen.requests)
    payload = {
        "plan_version": CLASSIFICATION_TRANSPORT_PLAN_VERSION,
        "snapshot_digest": frozen.pairs_by_request[0][0].hypothesis.snapshot_digest,
        "refinement_round": frozen.pairs_by_request[0][0].hypothesis.refinement_round,
        "requests": [request.model_dump(mode="json") for request in frozen.requests],
        "admission_estimate_calls": frozen.admission_estimate_calls,
        "admission_estimate_request_bytes": frozen.admission_estimate_request_bytes,
        "execution": BoundedStageExecution(
            request_plan_complete=True, planned_calls=len(frozen.requests),
            terminal_calls=len(stage.outcomes), omitted_calls=stage.omitted,
            planned_items=planned_pairs, terminal_items=terminal_pairs,
            omitted_items=planned_pairs - terminal_pairs,
            closure_reason=stage.exhaustion_reason,
        ).model_dump(mode="json"),
    }
    payload["plan_digest"] = payload_digest(payload)
    return ClassificationTransportExecution(
        plan=ClassificationTransportPlan.model_validate(payload), records=records,
        root_stages=tuple(root_stages),
    )


def classification_transport_totals(
    plans: Sequence[ClassificationTransportPlan],
) -> dict[str, int]:
    """Physical transport totals, deliberately independent of root references."""

    return {
        "classification_transport_planned_calls": sum(plan.execution.planned_calls for plan in plans),
        "classification_transport_terminal_calls": sum(plan.execution.terminal_calls for plan in plans),
        "classification_transport_omitted_calls": sum(plan.execution.omitted_calls for plan in plans),
        "classification_transport_planned_pairs": sum(plan.execution.planned_items for plan in plans),
        "classification_transport_terminal_pairs": sum(plan.execution.terminal_items for plan in plans),
        "classification_transport_omitted_pairs": sum(plan.execution.omitted_items for plan in plans),
        "classification_transport_planned_request_bytes": sum(request.request_bytes for plan in plans for request in plan.requests),
    }


def validate_classification_transport_bindings(
    *, plans: Sequence[ClassificationTransportPlan],
    populations: Sequence[PopulationArtifact], records: Sequence[InferenceRecord],
) -> None:
    """Prove exact root subsets and unique physical accounting before publication."""

    requests = {request.inference_id: request for plan in plans for request in plan.requests}
    if len(requests) != sum(len(plan.requests) for plan in plans):
        raise ValueError("classification plans repeat a physical request identity")
    terminal = {
        request.inference_id for plan in plans
        for request in plan.requests[:plan.execution.terminal_calls]
    }
    physical = [record for record in records if record.stage == "population_semantic_classify"]
    if len(physical) != len(terminal) or {record.request_id for record in physical} != terminal:
        raise ValueError("classification records do not cover unique terminal transport")
    for record in physical:
        request = requests[record.request_id]
        if record.request_payload_digest != request.request_payload_digest or record.request_ordinal != request.ordinal:
            raise ValueError("classification record is not bound to frozen transport")
        if [ref.digest for ref in record.input_manifest] != [pair.packet_digest for pair in request.pairs]:
            raise ValueError("classification record packet bindings differ")
    for population in populations:
        plan = population.classification_plan
        for batch in plan.batches:
            request = requests.get(batch.inference_id)
            if request is None:
                raise ValueError("classification root cites an unplanned transport")
            subset = [pair for pair in request.pairs if pair.hypothesis_id == population.hypothesis_id]
            if (
                batch.transport_ordinal != request.ordinal
                or batch.pair_ordinals != [pair.ordinal for pair in subset]
                or batch.member_ids != [pair.member_id for pair in subset]
                or batch.packet_digests != [pair.packet_digest for pair in subset]
                or batch.request_bytes != request.request_bytes
                or batch.request_payload_digest != request.request_payload_digest
            ):
                raise ValueError("classification root subset differs from transport")
            member_by_id = {member.member_id: member for member in population.members}
            for pair in subset:
                expected = request.inference_id if request.inference_id in terminal else None
                if member_by_id[pair.member_id].inference_ref != expected:
                    raise ValueError("classification member inference ownership differs")
        terminal_batches = [batch for batch in plan.batches if batch.inference_id in terminal]
        if plan.execution.terminal_calls != len(terminal_batches) or plan.execution.terminal_items != sum(len(batch.member_ids) for batch in terminal_batches):
            raise ValueError("classification root execution differs from transport")
