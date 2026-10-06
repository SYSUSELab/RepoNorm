"""Canonical per-candidate lineage and channel/evidence-route funnels."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from repo_constraint_memory_v3.models import (
    NonEmptyString,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import (
    model_payload_digest,
    payload_digest,
)
from repo_constraint_memory_v3.semantic.ids import semantic_id
from repo_constraint_memory_v3.semantic.models import (
    FalsificationResult,
    NormHypothesis,
    PopulationArtifact,
    RefinementRecord,
)
from repo_constraint_memory_v3.semantic.reduce import CandidateEvent

TraceStage = Literal[
    "explicit_semantic_verify",
    "map",
    "reduce",
    "operationalize",
    "classify",
    "falsify",
    "refine",
    "disposition",
]
TraceDisposition = Literal["accepted", "hypothesis", "review", "rejected"]
TraceChannel = Literal["explicit", "implicit_current"]


class StageTransition(StrictModel):
    stage: TraceStage
    status: Literal["complete", "partial", "failed", "not_run"]
    input_refs: list[NonEmptyString]
    output_refs: list[NonEmptyString]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]


class CandidateTrace(StrictModel):
    trace_id: Annotated[str, Field(pattern=r"^TRACE-[0-9a-f]{24}$")]
    snapshot_digest: SnapshotDigest
    channel: TraceChannel
    candidate_root_id: NonEmptyString
    raw_event_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    normalization_result: Literal["unique", "merged_duplicate"]
    normalization_reason_codes: Annotated[
        list[NonEmptyString], Field(min_length=1)
    ]
    stage_transitions: Annotated[list[StageTransition], Field(min_length=1)]
    hypothesis_version_refs: list[NonEmptyString]
    population_refs: list[NonEmptyString]
    falsification_refs: list[NonEmptyString]
    proposition_ref: str | None
    evidence_route: Literal[
        "explicit_typed_evidence",
        "current_structural_population",
        "current_semantic_population",
        "unoperationalized",
    ]
    terminal_disposition: TraceDisposition
    terminal_reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    accepted_constraint_id: str | None
    category: str | None
    component_keys: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def boundary_and_digest(self) -> CandidateTrace:
        if self.channel == "implicit_current" and (
            self.terminal_disposition == "accepted"
            or self.accepted_constraint_id is not None
        ):
            raise ValueError(
                "current-source channel-local trace cannot carry canonical acceptance"
            )
        if (
            self.terminal_disposition == "accepted"
        ) != (self.accepted_constraint_id is not None):
            raise ValueError("accepted constraint ID/disposition relationship is invalid")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect candidate-trace digest")
        return self


class CandidateEventTrace(StrictModel):
    event_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    channel: TraceChannel
    normalization_result: Literal[
        "invalid",
        "cost_omitted",
        "selection_omitted",
        "merged_duplicate",
    ]
    winning_root_id: str | None
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def shape(self) -> CandidateEventTrace:
        if (
            self.normalization_result
            in {"invalid", "cost_omitted", "selection_omitted"}
        ) != (self.winning_root_id is None):
            raise ValueError("candidate event winning-root relationship is invalid")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect candidate-event trace digest")
        return self


class EventFunnel(StrictModel):
    generated_candidate_events: int = Field(ge=0)
    invalid_candidate_events: int = Field(ge=0)
    cost_omitted_candidate_events: int = Field(ge=0, default=0)
    selection_omitted_candidate_events: int = Field(ge=0, default=0)
    duplicate_merged_events: int = Field(ge=0)
    normalized_unique_candidate_events: int = Field(ge=0)

    @model_validator(mode="after")
    def algebra(self) -> EventFunnel:
        if (
            self.invalid_candidate_events
            + self.cost_omitted_candidate_events
            + self.selection_omitted_candidate_events
            + self.duplicate_merged_events
            + self.normalized_unique_candidate_events
            != self.generated_candidate_events
        ):
            raise ValueError("candidate-event funnel algebra is invalid")
        return self


class RootDispositions(StrictModel):
    accepted: int = Field(ge=0)
    hypothesis: int = Field(ge=0)
    review: int = Field(ge=0)
    rejected: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.accepted + self.hypothesis + self.review + self.rejected


class FunnelSlice(StrictModel):
    trace_count: int = Field(ge=0)
    trace_ids: list[NonEmptyString]
    trace_set_digest: Sha256Hex
    dispositions: RootDispositions

    @model_validator(mode="after")
    def trace_algebra(self) -> FunnelSlice:
        if self.trace_count != len(self.trace_ids):
            raise ValueError("trace slice count differs from trace IDs")
        if self.trace_count != self.dispositions.total:
            raise ValueError("trace slice disposition algebra is invalid")
        if self.trace_set_digest != payload_digest({"trace_ids": self.trace_ids}):
            raise ValueError("trace slice digest is invalid")
        return self


class ChannelTraceability(StrictModel):
    status: Literal["complete", "partial", "failed", "not_run"]
    reason: str | None
    event_funnel: EventFunnel | None
    root_dispositions: RootDispositions | None
    by_evidence_route: dict[str, FunnelSlice]
    by_component: dict[str, FunnelSlice]
    by_category_audit_only: dict[str, FunnelSlice]
    transition_reason_histograms: dict[str, dict[str, int]]
    trace_ids: list[NonEmptyString]
    trace_set_digest: Sha256Hex | None

    @model_validator(mode="after")
    def not_run_shape(self) -> ChannelTraceability:
        if self.status == "not_run":
            if (
                self.event_funnel is not None
                or self.root_dispositions is not None
                or self.trace_ids
                or self.trace_set_digest is not None
            ):
                raise ValueError("not-run channel cannot carry a zero-work funnel")
        elif self.event_funnel is None or self.root_dispositions is None:
            raise ValueError("run channel requires event and root funnels")
        if self.trace_set_digest is not None and self.trace_set_digest != payload_digest(
            {"trace_ids": self.trace_ids}
        ):
            raise ValueError("channel trace-set digest is invalid")
        return self


class TraceabilityArtifact(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    traces: list[CandidateTrace]
    invalid_or_duplicate_events: list[CandidateEventTrace]
    explicit: ChannelTraceability
    current_source: ChannelTraceability
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def global_algebra(self) -> TraceabilityArtifact:
        trace_ids = [item.trace_id for item in self.traces]
        if trace_ids != sorted(trace_ids) or len(trace_ids) != len(set(trace_ids)):
            raise ValueError("candidate traces must be uniquely sorted")
        current = [
            item for item in self.traces if item.channel == "implicit_current"
        ]
        if any(item.terminal_disposition == "accepted" for item in current):
            raise ValueError(
                "current-source channel-local audit accepted count must remain zero"
            )
        if self.current_source.root_dispositions is not None and (
            self.current_source.root_dispositions.accepted != 0
        ):
            raise ValueError("current-source funnel accepted count must be zero")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect traceability artifact digest")
        return self


def _dispositions(traces: Sequence[CandidateTrace]) -> RootDispositions:
    counts = Counter(item.terminal_disposition for item in traces)
    return RootDispositions(
        accepted=counts["accepted"],
        hypothesis=counts["hypothesis"],
        review=counts["review"],
        rejected=counts["rejected"],
    )


def _slice(traces: Sequence[CandidateTrace]) -> FunnelSlice:
    ids = sorted(item.trace_id for item in traces)
    return FunnelSlice(
        trace_count=len(ids),
        trace_ids=ids,
        trace_set_digest=payload_digest({"trace_ids": ids}),
        dispositions=_dispositions(traces),
    )


def _channel(
    *,
    status: Literal["complete", "partial", "failed", "not_run"],
    reason: str | None,
    traces: Sequence[CandidateTrace],
    events: Sequence[CandidateEventTrace],
    generated_events: int,
) -> ChannelTraceability:
    if status == "not_run":
        return ChannelTraceability(
            status=status,
            reason=reason,
            event_funnel=None,
            root_dispositions=None,
            by_evidence_route={},
            by_component={},
            by_category_audit_only={},
            transition_reason_histograms={},
            trace_ids=[],
            trace_set_digest=None,
        )
    event_counts = Counter(item.normalization_result for item in events)
    unique_events = generated_events - (
        event_counts["invalid"]
        + event_counts["cost_omitted"]
        + event_counts["selection_omitted"]
        + event_counts["merged_duplicate"]
    )
    by_route: dict[str, list[CandidateTrace]] = defaultdict(list)
    by_component: dict[str, list[CandidateTrace]] = defaultdict(list)
    by_category: dict[str, list[CandidateTrace]] = defaultdict(list)
    histograms: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        by_route[trace.evidence_route].append(trace)
        for component in trace.component_keys or ["unknown"]:
            by_component[component].append(trace)
        by_category[trace.category or "uncategorized"].append(trace)
        for transition in trace.stage_transitions:
            histograms[transition.stage].update(transition.reason_codes)
        histograms["terminal"].update(trace.terminal_reason_codes)
    trace_ids = sorted(item.trace_id for item in traces)
    return ChannelTraceability(
        status=status,
        reason=reason,
        event_funnel=EventFunnel(
            generated_candidate_events=generated_events,
            invalid_candidate_events=event_counts["invalid"],
            cost_omitted_candidate_events=event_counts["cost_omitted"],
            selection_omitted_candidate_events=event_counts[
                "selection_omitted"
            ],
            duplicate_merged_events=event_counts["merged_duplicate"],
            normalized_unique_candidate_events=unique_events,
        ),
        root_dispositions=_dispositions(traces),
        by_evidence_route={
            key: _slice(value) for key, value in sorted(by_route.items())
        },
        by_component={
            key: _slice(value) for key, value in sorted(by_component.items())
        },
        by_category_audit_only={
            key: _slice(value) for key, value in sorted(by_category.items())
        },
        transition_reason_histograms={
            stage: dict(sorted(counts.items()))
            for stage, counts in sorted(histograms.items())
        },
        trace_ids=trace_ids,
        trace_set_digest=payload_digest({"trace_ids": trace_ids}),
    )


def build_current_traces(
    *,
    snapshot_digest: str,
    events: Sequence[CandidateEvent],
    hypotheses: Sequence[NormHypothesis],
    populations: Sequence[PopulationArtifact],
    falsifications: Sequence[FalsificationResult],
    refinements: Sequence[RefinementRecord],
    dispositions: Mapping[str, tuple[TraceDisposition, list[str], str | None]],
    pipeline_status: Literal["complete", "partial", "failed"],
) -> tuple[list[CandidateTrace], list[CandidateEventTrace], ChannelTraceability]:
    events_by_root: dict[str, list[CandidateEvent]] = defaultdict(list)
    event_traces: list[CandidateEventTrace] = []
    for event in events:
        if event.winning_root_id is not None:
            events_by_root[event.winning_root_id].append(event)
        if event.normalization != "unique":
            raw = {
                "event_id": event.event_id,
                "snapshot_digest": snapshot_digest,
                "channel": "implicit_current",
                "normalization_result": event.normalization,
                "winning_root_id": event.winning_root_id,
                "reason_codes": event.reason_codes,
            }
            raw["artifact_digest"] = payload_digest(raw)
            event_traces.append(CandidateEventTrace.model_validate(raw))
    hypotheses_by_root: dict[str, list[NormHypothesis]] = defaultdict(list)
    for hypothesis in hypotheses:
        hypotheses_by_root[hypothesis.candidate_root_id].append(hypothesis)
    pop_by_hyp = {item.hypothesis_id: item for item in populations}
    fals_by_hyp = {item.hypothesis_id: item for item in falsifications}
    refined_parents = {item.parent_hypothesis_id for item in refinements}
    traces: list[CandidateTrace] = []
    for root_id, versions in sorted(hypotheses_by_root.items()):
        ordered_versions = sorted(versions, key=lambda item: item.version)
        leaves = [
            item for item in ordered_versions if item.hypothesis_id not in refined_parents
        ]
        leaf = max(leaves or ordered_versions, key=lambda item: item.version)
        population = pop_by_hyp.get(leaf.hypothesis_id)
        falsification = fals_by_hyp.get(leaf.hypothesis_id)
        disposition, terminal_reasons, proposition_ref = dispositions[root_id]
        transitions = [
            StageTransition(
                stage="map",
                status="complete" if pipeline_status == "complete" else "partial",
                input_refs=[],
                output_refs=sorted(set(leaf.witness_refs)),
                reason_codes=["semantic_map_accounted"],
            ),
            StageTransition(
                stage="reduce",
                status="complete",
                input_refs=sorted(set(leaf.witness_refs)),
                output_refs=[item.hypothesis_id for item in ordered_versions],
                reason_codes=["normalized_unique_root"],
            ),
            StageTransition(
                stage="operationalize",
                status=population.status if population is not None else "failed",
                input_refs=[leaf.hypothesis_id],
                output_refs=[population.population_id] if population else [],
                reason_codes=(
                    ["population_frozen_before_classification"]
                    if population
                    else ["population_unavailable"]
                ),
            ),
            StageTransition(
                stage="classify",
                status=population.status if population is not None else "failed",
                input_refs=[population.population_id] if population else [],
                output_refs=(
                    [item.member_id for item in population.members]
                    if population
                    else []
                ),
                reason_codes=(
                    ["all_serialized_members_terminal"]
                    if population
                    else ["classification_not_run"]
                ),
            ),
            StageTransition(
                stage="falsify",
                status=(
                    falsification.status if falsification is not None else "not_run"
                ),
                input_refs=[population.population_id] if population else [],
                output_refs=(
                    [falsification.falsification_id] if falsification else []
                ),
                reason_codes=(
                    [
                        f"verdict:{falsification.verdict}",
                        *(
                            [
                                "falsification_cost_not_admitted:"
                                f"{falsification.admission_estimate.reason_code}"
                            ]
                            if falsification.admission_estimate is not None
                            else []
                        ),
                    ]
                    if falsification
                    else ["falsification_not_run"]
                ),
            ),
            StageTransition(
                stage="refine",
                status="complete",
                input_refs=[
                    item.parent_hypothesis_id
                    for item in refinements
                    if item.candidate_root_id == root_id
                ],
                output_refs=[
                    item.child_hypothesis_id
                    for item in refinements
                    if item.candidate_root_id == root_id
                ],
                reason_codes=[
                    "bounded_refinement"
                    if any(
                        item.candidate_root_id == root_id for item in refinements
                    )
                    else "no_refinement"
                ],
            ),
            StageTransition(
                stage="disposition",
                status="complete",
                input_refs=[leaf.hypothesis_id],
                output_refs=[proposition_ref] if proposition_ref else [],
                reason_codes=terminal_reasons,
            ),
        ]
        trace_identity = {
            "snapshot_digest": snapshot_digest,
            "channel": "implicit_current",
            "candidate_root_id": root_id,
        }
        raw = {
            "trace_id": semantic_id("TRACE", trace_identity),
            "snapshot_digest": snapshot_digest,
            "channel": "implicit_current",
            "candidate_root_id": root_id,
            "raw_event_refs": sorted(
                item.event_id for item in events_by_root[root_id]
            ),
            "normalization_result": (
                "merged_duplicate"
                if len(events_by_root[root_id]) > 1
                else "unique"
            ),
            "normalization_reason_codes": [
                "canonical_duplicate_merged"
                if len(events_by_root[root_id]) > 1
                else "normalized_unique"
            ],
            "stage_transitions": [
                item.model_dump(mode="json") for item in transitions
            ],
            "hypothesis_version_refs": [
                item.hypothesis_id for item in ordered_versions
            ],
            "population_refs": [
                pop_by_hyp[item.hypothesis_id].population_id
                for item in ordered_versions
                if item.hypothesis_id in pop_by_hyp
            ],
            "falsification_refs": [
                fals_by_hyp[item.hypothesis_id].falsification_id
                for item in ordered_versions
                if item.hypothesis_id in fals_by_hyp
            ],
            "proposition_ref": proposition_ref,
            "evidence_route": (
                population.route if population else "unoperationalized"
            ),
            "terminal_disposition": disposition,
            "terminal_reason_codes": terminal_reasons,
            "accepted_constraint_id": None,
            "category": leaf.category,
            "component_keys": leaf.component_keys,
        }
        raw["artifact_digest"] = payload_digest(raw)
        traces.append(CandidateTrace.model_validate(raw))
    traces.sort(key=lambda item: item.trace_id)
    event_traces.sort(key=lambda item: item.event_id)
    channel = _channel(
        status=pipeline_status,
        reason=None,
        traces=traces,
        events=event_traces,
        generated_events=len(events),
    )
    return traces, event_traces, channel


def build_explicit_channel(
    *,
    snapshot_digest: str,
    decisions: Sequence[object],
    extraction_complete: bool,
) -> tuple[list[CandidateTrace], list[CandidateEventTrace], ChannelTraceability]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for decision in decisions:
        key = getattr(decision, "claim_id", None) or getattr(
            decision, "candidate_fingerprint"
        )
        grouped[key].append(decision)
    traces: list[CandidateTrace] = []
    events: list[CandidateEventTrace] = []
    for root_id, values in sorted(grouped.items()):
        for duplicate in values[1:]:
            event_id = semantic_id(
                "EVENT",
                {
                    "channel": "explicit",
                    "root": root_id,
                    "fingerprint": getattr(duplicate, "candidate_fingerprint"),
                },
            )
            raw_event = {
                "event_id": event_id,
                "snapshot_digest": snapshot_digest,
                "channel": "explicit",
                "normalization_result": "merged_duplicate",
                "winning_root_id": root_id,
                "reason_codes": ["explicit_duplicate_merge"],
            }
            raw_event["artifact_digest"] = payload_digest(raw_event)
            events.append(CandidateEventTrace.model_validate(raw_event))
        accepted = next(
            (item for item in values if getattr(item, "outcome") == "accepted"),
            None,
        )
        disposition: TraceDisposition = (
            "accepted"
            if accepted is not None
            else (
                "review"
                if any(getattr(item, "outcome") == "review" for item in values)
                else "rejected"
            )
        )
        event_refs = sorted(
            semantic_id(
                "EVENT",
                {
                    "channel": "explicit",
                    "fingerprint": getattr(item, "candidate_fingerprint"),
                    "unit": getattr(item, "unit_id"),
                },
            )
            for item in values
        )
        verification_dispositions = [
            getattr(item, "verification_disposition", None) for item in values
        ]
        verification_technical_codes = [
            getattr(item, "verification_technical_reason_code", None)
            for item in values
        ]
        verification_reason_codes = {
            reason
            for item in values
            for reason in getattr(item, "verification_reason_codes", [])
        }
        if not any(verification_dispositions):
            verification_status = "not_run"
        elif all(verification_dispositions) and not any(
            verification_technical_codes
        ):
            verification_status = "complete"
        else:
            verification_status = "partial"
        transition_reasons = {
            f"explicit_verification:{reason}"
            for reason in verification_reason_codes
        }
        transition_reasons.update(
            f"explicit_verification_technical:{reason}"
            for reason in verification_technical_codes
            if reason is not None
        )
        if not transition_reasons:
            transition_reasons.add(
                "explicit_verification_not_run_ungrounded"
                if not any(verification_dispositions)
                else "explicit_verification_untyped_outcome"
            )
        verification_transition = StageTransition(
            stage="explicit_semantic_verify",
            status=verification_status,
            input_refs=sorted(
                {
                    getattr(item, "candidate_fingerprint")
                    for item in values
                }
            ),
            output_refs=sorted(
                {
                    request_id
                    for item in values
                    if (
                        request_id := getattr(
                            item, "verification_request_id", None
                        )
                    )
                }
            ),
            reason_codes=sorted(transition_reasons),
        )
        transition = StageTransition(
            stage="disposition",
            status="complete" if extraction_complete else "partial",
            input_refs=event_refs,
            output_refs=[getattr(accepted, "claim_id")] if accepted else [],
            reason_codes=sorted(
                {getattr(item, "reason_code") for item in values}
            ),
        )
        raw = {
            "trace_id": semantic_id(
                "TRACE",
                {
                    "snapshot_digest": snapshot_digest,
                    "channel": "explicit",
                    "root": root_id,
                },
            ),
            "snapshot_digest": snapshot_digest,
            "channel": "explicit",
            "candidate_root_id": root_id,
            "raw_event_refs": event_refs,
            "normalization_result": (
                "merged_duplicate" if len(values) > 1 else "unique"
            ),
            "normalization_reason_codes": [
                "explicit_duplicate_merge" if len(values) > 1 else "explicit_unique"
            ],
            "stage_transitions": [
                verification_transition.model_dump(mode="json"),
                transition.model_dump(mode="json"),
            ],
            "hypothesis_version_refs": [],
            "population_refs": [],
            "falsification_refs": [],
            "proposition_ref": None,
            "evidence_route": "explicit_typed_evidence",
            "terminal_disposition": disposition,
            "terminal_reason_codes": transition.reason_codes,
            "accepted_constraint_id": (
                getattr(accepted, "claim_id") if accepted is not None else None
            ),
            "category": None,
            "component_keys": ["explicit"],
        }
        raw["artifact_digest"] = payload_digest(raw)
        traces.append(CandidateTrace.model_validate(raw))
    traces.sort(key=lambda item: item.trace_id)
    events.sort(key=lambda item: item.event_id)
    return traces, events, _channel(
        status="complete" if extraction_complete else "partial",
        reason=None,
        traces=traces,
        events=events,
        generated_events=sum(len(value) for value in grouped.values()),
    )


def make_traceability(
    *,
    snapshot_digest: str,
    explicit_traces: Sequence[CandidateTrace],
    explicit_events: Sequence[CandidateEventTrace],
    explicit_channel: ChannelTraceability,
    current_traces: Sequence[CandidateTrace],
    current_events: Sequence[CandidateEventTrace],
    current_channel: ChannelTraceability,
) -> TraceabilityArtifact:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "traces": [
            item.model_dump(mode="json")
            for item in sorted(
                [*explicit_traces, *current_traces],
                key=lambda value: value.trace_id,
            )
        ],
        "invalid_or_duplicate_events": [
            item.model_dump(mode="json")
            for item in sorted(
                [*explicit_events, *current_events],
                key=lambda value: value.event_id,
            )
        ],
        "explicit": explicit_channel.model_dump(mode="json"),
        "current_source": current_channel.model_dump(mode="json"),
    }
    payload["artifact_digest"] = payload_digest(payload)
    return TraceabilityArtifact.model_validate(payload)


def not_run_channel(reason: str) -> ChannelTraceability:
    """Expose the canonical null-funnel shape for an unrun channel."""

    return _channel(
        status="not_run",
        reason=reason,
        traces=[],
        events=[],
        generated_events=0,
    )
