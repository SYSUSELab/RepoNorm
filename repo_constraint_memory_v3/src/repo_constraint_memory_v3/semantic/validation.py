"""Current-source pipeline orchestration, propositions, and artifact contract."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, Sequence

from pydantic import Field, model_validator

from repo_constraint_memory_v3.inference import (
    BudgetedStageRunResult,
    InferenceBudget,
    InferenceRecord,
    inference_cache_key_for_record,
    run_budgeted_stage_calls,
)
from pydantic import ValidationError

from repo_constraint_memory_v3.llm import (
    LLMClient,
    LLMProviderError,
)
from repo_constraint_memory_v3.models import (
    CurrentSourceReviewItem,
    DiscoveryArtifact,
    LLMSettings,
    NonEmptyString,
    SemanticSettings,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
    ProviderResponse,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    canonical_json_text,
    model_payload_digest,
    payload_digest,
)
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.observations.semantic_units import (
    SemanticUnitBundle,
)
from repo_constraint_memory_v3.propositions import (
    PROPOSITION_CONTRACT_VERSION,
    PropositionEnvelope,
    TypedObjectRef,
)
from repo_constraint_memory_v3.semantic.classify import (
    ClassificationAdmissionLedger,
    PendingPopulationBuild,
    PopulationCostNotAdmitted,
    finish_population_classification,
    start_population_classification,
)
from repo_constraint_memory_v3.semantic.classification_transport import (
    ClassificationTransportPlan,
    classification_transport_totals,
    execute_classification_transport,
    validate_classification_transport_bindings,
)
from repo_constraint_memory_v3.semantic.falsification import (
    FalsificationBatchOutcome,
    FalsificationAdmissionLedger,
    FalsificationBuildResult,
    PendingFalsificationBuild,
    falsification_ineligibility_reason,
    finish_falsification,
    make_zero_support_falsification,
    make_refined_hypothesis,
    make_refinement_record,
    run_falsification,
    start_falsification,
)
from repo_constraint_memory_v3.semantic.ids import semantic_id
from repo_constraint_memory_v3.semantic.map import (
    SemanticMapBundle,
    SemanticMapIndex,
    SemanticMapPacketResult,
    run_semantic_map,
)
from repo_constraint_memory_v3.semantic.models import (
    FalsificationResult,
    NormHypothesis,
    PopulationArtifact,
    PopulationMember,
    PopulationMemberPacket,
    PopulationResult,
    RefinementRecord,
    SemanticObservation,
)
from repo_constraint_memory_v3.semantic.reduce import (
    CandidateEvent,
    HypothesisReductionBundle,
    HypothesisReductionIndex,
    ReductionCallResult,
    run_hypothesis_reduction,
)
from repo_constraint_memory_v3.semantic.queries import (
    count_population_members_for_admission,
    evaluate_population,
)
from repo_constraint_memory_v3.traceability import (
    CandidateEventTrace,
    CandidateTrace,
    ChannelTraceability,
    build_current_traces,
)

CURRENT_SOURCE_INDEX_VERSION = "current-source-index-1.10"
CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION = (
    "current-source-experimental-policy-1.4"
)
_CURRENT_SOURCE_INFERENCE_STAGES = frozenset(
    {
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
    }
)

_POPULATION_OBSERVATION_KINDS = {
    "file": "observation_file",
    "entity": "observation_entity",
    "site": "observation_site",
    "relation": "observation_relation",
}


class CurrentSourcePolicyDecision(StrictModel):
    """Auditable pre-calibration disposition that never enters delivery."""

    policy_version: Literal["current-source-experimental-policy-1.4"]
    snapshot_digest: SnapshotDigest
    candidate_root_id: NonEmptyString
    hypothesis_id: Annotated[
        str, Field(pattern=r"^HYP-[0-9a-f]{24}$")
    ]
    proposition_id: Annotated[
        str, Field(pattern=r"^PROP-[0-9a-f]{24}$")
    ]
    experimental_disposition: Literal[
        "accept_candidate", "review", "rejected"
    ]
    canonical_epistemic_status: Literal["hypothesis", "review", "rejected"]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def policy_and_digest(self) -> CurrentSourcePolicyDecision:
        if (
            self.experimental_disposition == "accept_candidate"
            and self.canonical_epistemic_status != "review"
        ):
            raise ValueError(
                "experimental accept candidates must remain canonical review"
            )
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("policy reason codes are not canonical")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect current-source policy digest")
        return self


class PopulationScheduleRecord(StrictModel):
    ordinal: int = Field(ge=0)
    candidate_root_id: NonEmptyString
    hypothesis_id: Annotated[str, Field(pattern=r"^HYP-[0-9a-f]{24}$")]
    component_key: NonEmptyString
    semantic_member_cost: int = Field(ge=0)
    matched_member_count: int = Field(ge=0)
    classified_member_limit: int = Field(ge=0)
    provider_unallocated_count: int = Field(ge=0)
    conclusion_eligible: bool
    non_conclusion_reasons: list[
        Literal[
            "population_dependency_partial",
            "population_member_limit",
            "population_precost_failed",
        ]
    ]

    @model_validator(mode="after")
    def conclusion_reason_is_exact(self) -> PopulationScheduleRecord:
        if self.non_conclusion_reasons != sorted(
            set(self.non_conclusion_reasons)
        ):
            raise ValueError("population schedule reasons are not canonical")
        if self.conclusion_eligible == bool(self.non_conclusion_reasons):
            raise ValueError(
                "population schedule conclusion eligibility/reason mismatch"
            )
        if (
            self.classified_member_limit + self.provider_unallocated_count
            != self.semantic_member_cost
        ):
            raise ValueError("population schedule allocation counts do not sum")
        return self


class ClassificationAdmissionRecord(StrictModel):
    cost_basis: Literal["compact_single_root_admission_estimate"] = (
        "compact_single_root_admission_estimate"
    )
    ordinal: int = Field(ge=0)
    phase: Literal["initial", "refinement"]
    candidate_root_id: NonEmptyString
    hypothesis_id: Annotated[str, Field(pattern=r"^HYP-[0-9a-f]{24}$")]
    outcome: Literal["admitted", "not_admitted"]
    semantic_members: int = Field(ge=0)
    planned_calls: int = Field(ge=0)
    planned_request_bytes: int = Field(ge=0)
    reason_code: NonEmptyString | None
    limit: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def outcome_fields_are_exact(self) -> ClassificationAdmissionRecord:
        rejected = self.outcome == "not_admitted"
        if rejected != (self.reason_code is not None and self.limit is not None):
            raise ValueError(
                "classification admission outcome fields are inconsistent"
            )
        return self


class CurrentSourceIndex(StrictModel):
    index_version: Literal["current-source-index-1.10"]
    snapshot_digest: SnapshotDigest
    required: bool
    status: Literal["complete", "partial", "failed"]
    semantic_map_digest: Sha256Hex
    hypothesis_reduction_digest: Sha256Hex
    semantic_observations: int = Field(ge=0)
    candidate_events: int = Field(ge=0)
    unique_candidate_roots: int = Field(ge=0)
    hypothesis_versions: int = Field(ge=0)
    complete_populations: int = Field(ge=0)
    partial_populations: int = Field(ge=0)
    failed_populations: int = Field(ge=0)
    classified_members: int = Field(ge=0)
    population_matched_members: int = Field(ge=0)
    population_inspected_members: int = Field(ge=0)
    population_uninspected_members: int = Field(ge=0)
    population_provider_eligible_members: int = Field(ge=0)
    population_provider_inspected_members: int = Field(ge=0)
    population_provider_uninspected_members: int = Field(ge=0)
    census_populations: int = Field(ge=0)
    sampled_populations: int = Field(ge=0)
    technical_unknown_members: int = Field(ge=0)
    completed_falsifications: int = Field(ge=0)
    partial_falsifications: int = Field(ge=0)
    refinements: int = Field(ge=0)
    propositions: int = Field(ge=0)
    root_hypothesis: int = Field(ge=0)
    root_review: int = Field(ge=0)
    root_rejected: int = Field(ge=0)
    root_accepted: Literal[0]
    experimental_accept_candidates: int = Field(ge=0)
    experimental_review: int = Field(ge=0)
    experimental_rejected: int = Field(ge=0)
    inference_ids: list[NonEmptyString]
    artifact_set_digest: Sha256Hex
    population_schedule: list[PopulationScheduleRecord]
    classification_admission_schedule: list[ClassificationAdmissionRecord]
    classification_transport_plans: list[ClassificationTransportPlan]
    resource_use: dict[str, int]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    index_digest: Sha256Hex

    @model_validator(mode="after")
    def algebra_and_digest(self) -> CurrentSourceIndex:
        if (
            self.root_hypothesis + self.root_review + self.root_rejected
            != self.unique_candidate_roots
        ):
            raise ValueError("current-source root disposition algebra is invalid")
        if (
            self.experimental_accept_candidates
            + self.experimental_review
            + self.experimental_rejected
            != self.unique_candidate_roots
        ):
            raise ValueError(
                "current-source experimental disposition algebra is invalid"
            )
        if (
            self.complete_populations
            + self.partial_populations
            + self.failed_populations
            != self.hypothesis_versions
        ):
            raise ValueError("current-source population/version algebra is invalid")
        if (
            self.completed_falsifications + self.partial_falsifications
            != self.hypothesis_versions
        ):
            raise ValueError(
                "current-source falsification/version algebra is invalid"
            )
        if self.population_inspected_members + self.population_uninspected_members != (
            self.population_matched_members
        ):
            raise ValueError("current-source population sampling counts do not sum")
        if (
            self.population_provider_inspected_members
            + self.population_provider_uninspected_members
            != self.population_provider_eligible_members
        ):
            raise ValueError(
                "current-source provider population counts do not sum"
            )
        if self.classified_members != self.population_inspected_members:
            raise ValueError("classified members differ from inspected members")
        materialized_populations = (
            self.complete_populations
            + self.partial_populations
            + self.failed_populations
            - self.resource_use.get("failed_population_hypotheses", 0)
        )
        if self.census_populations + self.sampled_populations != (
            materialized_populations
        ):
            raise ValueError("population sampling modes do not cover artifacts")
        if self.status == "complete" and self.incomplete_reasons:
            raise ValueError("complete current-source index cannot be incomplete")
        if [item.ordinal for item in self.population_schedule] != list(
            range(len(self.population_schedule))
        ):
            raise ValueError("population schedule ordinals are not canonical")
        if [item.ordinal for item in self.classification_admission_schedule] != list(
            range(len(self.classification_admission_schedule))
        ):
            raise ValueError(
                "classification admission schedule ordinals are not canonical"
            )
        if len({item.hypothesis_id for item in self.population_schedule}) != len(
            self.population_schedule
        ):
            raise ValueError("population schedule repeats a hypothesis")
        if self.resource_use.get("population_schedule_roots", 0) != len(
            self.population_schedule
        ):
            raise ValueError("population schedule root count differs")
        sampling_totals = {
            "population_matched_members": self.population_matched_members,
            "population_inspected_members": self.population_inspected_members,
            "population_uninspected_members": self.population_uninspected_members,
            "population_provider_eligible_members": (
                self.population_provider_eligible_members
            ),
            "population_provider_inspected_members": (
                self.population_provider_inspected_members
            ),
            "population_provider_uninspected_members": (
                self.population_provider_uninspected_members
            ),
            "population_census_roots": self.census_populations,
            "population_sampled_roots": self.sampled_populations,
        }
        if any(
            self.resource_use.get(key, 0) != value
            for key, value in sampling_totals.items()
        ):
            raise ValueError("population sampling totals differ from resources")
        conclusion_eligible = sum(
            item.conclusion_eligible for item in self.population_schedule
        )
        non_conclusion = len(self.population_schedule) - conclusion_eligible
        schedule_counts = {
            "population_schedule_conclusion_eligible_roots": conclusion_eligible,
            "population_schedule_non_conclusion_eligible_roots": non_conclusion,
            "population_schedule_non_conclusion_dependency_partial_roots": sum(
                "population_dependency_partial" in item.non_conclusion_reasons
                for item in self.population_schedule
            ),
            "population_schedule_non_conclusion_member_limit_roots": sum(
                "population_member_limit" in item.non_conclusion_reasons
                for item in self.population_schedule
            ),
            "population_schedule_non_conclusion_precost_failed_roots": sum(
                "population_precost_failed" in item.non_conclusion_reasons
                for item in self.population_schedule
            ),
            "population_schedule_provider_allocated_members": sum(
                item.classified_member_limit
                for item in self.population_schedule
            ),
            "population_schedule_provider_unallocated_members": sum(
                item.provider_unallocated_count
                for item in self.population_schedule
            ),
        }
        if any(
            self.resource_use.get(key, 0) != value
            for key, value in schedule_counts.items()
        ):
            raise ValueError("population schedule reason counts differ")
        admitted = [
            item
            for item in self.classification_admission_schedule
            if item.outcome == "admitted"
        ]
        not_admitted = [
            item
            for item in self.classification_admission_schedule
            if item.outcome == "not_admitted"
        ]
        admission_totals = {
            "classification_admitted_hypotheses": len(admitted),
            "classification_admitted_semantic_members": sum(
                item.semantic_members for item in admitted
            ),
            "classification_admitted_calls": sum(
                item.planned_calls for item in admitted
            ),
            "classification_admitted_request_bytes": sum(
                item.planned_request_bytes for item in admitted
            ),
            "classification_not_admitted_hypotheses": len(not_admitted),
            "classification_not_admitted_semantic_members": sum(
                item.semantic_members for item in not_admitted
            ),
            "classification_not_admitted_planned_calls": sum(
                item.planned_calls for item in not_admitted
            ),
            "classification_not_admitted_planned_request_bytes": sum(
                item.planned_request_bytes for item in not_admitted
            ),
        }
        if any(
            self.resource_use.get(key, 0) != value
            for key, value in admission_totals.items()
        ):
            raise ValueError(
                "classification admission schedule differs from resource totals"
            )
        if self.resource_use.get(
            "classification_admission_attempts", 0
        ) != len(self.classification_admission_schedule):
            raise ValueError("classification admission attempt count differs")
        if any(
            self.resource_use.get(key, 0) != value
            for key, value in classification_transport_totals(
                self.classification_transport_plans
            ).items()
        ):
            raise ValueError("classification transport totals differ")
        if self.index_digest != model_payload_digest(
            self, digest_field="index_digest"
        ):
            raise ValueError("incorrect current-source index digest")
        return self


@dataclass(frozen=True)
class CurrentSourceBundle:
    index: CurrentSourceIndex
    semantic_map: SemanticMapBundle
    reduction: HypothesisReductionBundle
    hypotheses: tuple[NormHypothesis, ...]
    populations: tuple[PopulationArtifact, ...]
    member_packets: tuple[PopulationMemberPacket, ...]
    falsifications: tuple[FalsificationResult, ...]
    refinements: tuple[RefinementRecord, ...]
    propositions: tuple[PropositionEnvelope, ...]
    review_items: tuple[CurrentSourceReviewItem, ...]
    policy_decisions: tuple[CurrentSourcePolicyDecision, ...]
    traces: tuple[CandidateTrace, ...]
    event_traces: tuple[CandidateEventTrace, ...]
    trace_channel: ChannelTraceability
    inference_records: tuple[InferenceRecord, ...]


class _RecordedInferenceClient:
    """Replay strict structured results by deterministic request-payload digest."""

    def __init__(self, records: Sequence[InferenceRecord]) -> None:
        self.by_request_digest = {
            item.request_payload_digest: item for item in records
        }
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt
        self.calls += 1
        digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        record = self.by_request_digest.get(digest)
        if record is None:
            raise LLMProviderError(
                "recorded inference request is unavailable",
                retryable=False,
            )
        if (
            record.terminal_status != "success"
            or record.structured_response is None
        ):
            raise LLMProviderError(
                (
                    record.incomplete_reasons[0]
                    if record.incomplete_reasons
                    else "recorded_terminal_failure"
                ),
                retryable=False,
            )
        return ProviderResponse(
            content=json.dumps(
                record.structured_response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            model=record.model_parameters.model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _RecordedInferenceCache:
    """Expose recorded successes through the normal canonical cache path."""

    def __init__(self, records: Sequence[InferenceRecord]) -> None:
        self.by_cache_key = {
            inference_cache_key_for_record(item): item.model_dump(mode="json")
            for item in records
            if item.terminal_status == "success"
            and item.structured_response is not None
        }
        self.terminal_by_cache_key = {
            inference_cache_key_for_record(item): item.model_dump(mode="json")
            for item in records
            if item.terminal_status != "success"
        }

    def read(self, cache_key: str) -> dict[str, object] | None:
        value = self.by_cache_key.get(cache_key)
        return dict(value) if value is not None else None

    def write(self, cache_key: str, value: dict[str, object]) -> None:
        del cache_key, value
        raise ValueError("offline replay cannot create inference cache entries")

    def read_terminal(self, cache_key: str) -> dict[str, object] | None:
        value = self.terminal_by_cache_key.get(cache_key)
        return dict(value) if value is not None else None


def _current_proposition(
    *,
    hypothesis: NormHypothesis,
    population: PopulationArtifact | None,
    falsification: FalsificationResult | None,
) -> PropositionEnvelope:
    route = (
        population.route if population is not None else "current_semantic_population"
    )
    evidence_refs_by_key: dict[tuple[str, str, str], TypedObjectRef] = {}
    for item in hypothesis.witness_refs:
        reference = TypedObjectRef(
            kind="semantic_observation",
            stable_id=item,
            snapshot_digest=hypothesis.snapshot_digest,
        )
        evidence_refs_by_key[
            (reference.kind, reference.stable_id, reference.snapshot_digest)
        ] = reference
    if population is not None:
        for member in population.members:
            if member.label != "support":
                continue
            observation = member.observation_ref
            reference = TypedObjectRef(
                kind=_POPULATION_OBSERVATION_KINDS[observation.universe],
                stable_id=observation.observation_id,
                snapshot_digest=hypothesis.snapshot_digest,
            )
            evidence_refs_by_key[
                (reference.kind, reference.stable_id, reference.snapshot_digest)
            ] = reference
    evidence_refs = [
        evidence_refs_by_key[key] for key in sorted(evidence_refs_by_key)
    ]
    inference_ids = sorted(
        set(hypothesis.origin_inference_ids)
        | set(falsification.inference_ids if falsification else [])
    )
    population_refs = (
        [
            TypedObjectRef(
                kind="population_result",
                stable_id=population.population_id,
                snapshot_digest=hypothesis.snapshot_digest,
            )
        ]
        if population is not None
        else []
    )
    limitation_refs = (
        [
            TypedObjectRef(
                kind="falsification_result",
                stable_id=falsification.falsification_id,
                snapshot_digest=hypothesis.snapshot_digest,
            )
        ]
        if falsification is not None
        else []
    )
    identity = {
        "contract_version": PROPOSITION_CONTRACT_VERSION,
        "snapshot_digest": hypothesis.snapshot_digest,
        "candidate_root_id": hypothesis.candidate_root_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "rule": hypothesis.rule,
        "route": route,
    }
    raw = {
        "contract_version": PROPOSITION_CONTRACT_VERSION,
        "snapshot_digest": hypothesis.snapshot_digest,
        "proposition_id": semantic_id("PROP", identity),
        "rule": hypothesis.rule,
        "category": hypothesis.category,
        "proposed_claim_kind": hypothesis.proposed_claim_kind,
        "proposed_normative_strength": hypothesis.proposed_strength,
        "applies_to": hypothesis.applies_to,
        "condition": hypothesis.condition,
        "origin_channels": ["implicit_current"],
        "evidence_routes": [route],
        "evidence_refs": [item.model_dump(mode="json") for item in evidence_refs],
        "inference_refs": [
            TypedObjectRef(
                kind="inference_request",
                stable_id=item,
                snapshot_digest=hypothesis.snapshot_digest,
            ).model_dump(mode="json")
            for item in inference_ids
        ],
        "population_refs": [
            item.model_dump(mode="json") for item in population_refs
        ],
        "limitation_refs": [
            item.model_dump(mode="json") for item in limitation_refs
        ],
        "completeness_refs": hypothesis.completeness_refs,
        "provenance": [
            f"candidate_root:{hypothesis.candidate_root_id}",
            f"hypothesis:{hypothesis.hypothesis_id}",
            "pre_calibration_review_only",
        ],
    }
    raw["artifact_digest"] = payload_digest(raw)
    return PropositionEnvelope.model_validate(raw)


def _disposition(
    population: PopulationArtifact | None,
    falsification: FalsificationResult | None,
    *,
    blocked_reason: str | None = None,
) -> tuple[
    Literal["hypothesis", "review", "rejected"],
    list[str],
]:
    if blocked_reason is not None:
        if blocked_reason == "no_population_support":
            return "rejected", [blocked_reason]
        if blocked_reason.startswith(
            (
                "population_operationalization_failed:",
                "refinement_population_failed:",
                "invalid_refinement:",
                "complete_empty_population",
            )
        ):
            return "rejected", [blocked_reason]
        return "hypothesis", [blocked_reason]
    if population is None:
        return "rejected", ["population_operationalization_failed"]
    if population.population_result.serialized_count == 0:
        return "rejected", ["complete_empty_population"]
    if population.status != "complete":
        return "hypothesis", ["population_or_falsification_incomplete"]
    if population.support_count == 0:
        return "rejected", ["no_population_support"]
    if falsification is None:
        return "hypothesis", ["population_or_falsification_incomplete"]
    if falsification.status != "complete":
        reasons = ["falsification_incomplete"]
        if falsification.admission_estimate is not None:
            reasons.append(
                "falsification_cost_not_admitted:"
                f"{falsification.admission_estimate.reason_code}"
            )
        return "hypothesis", reasons
    if falsification.verdict in {
        "alternative_non_norm_explanation",
        "reject_insufficient",
    }:
        return "rejected", [f"falsification:{falsification.verdict}"]
    return "review", [
        f"falsification:{falsification.verdict}",
        "pre_calibration_review_only",
    ]


def _experimental_disposition(
    population: PopulationArtifact | None,
    falsification: FalsificationResult | None,
    *,
    canonical_status: Literal["hypothesis", "review", "rejected"],
    refinement_round: int,
    blocked_reason: str | None = None,
) -> tuple[Literal["accept_candidate", "review", "rejected"], list[str]]:
    """Apply the local quality gate used to collect calibration candidates.

    This decision is deliberately separate from canonical epistemic status and
    delivery. It is broad enough to admit a single independently supported
    repository-specific norm, but only after complete population and independent
    falsification with no unresolved member or quality-axis concern.
    """

    if canonical_status == "rejected":
        return "rejected", [blocked_reason or "canonical_rejection"]
    if blocked_reason is not None:
        return "review", [blocked_reason]
    if population is None:
        return "rejected", ["population_missing"]
    if population.status != "complete":
        return "review", ["population_incomplete"]
    if population.population_result.serialized_count == 0:
        return "rejected", ["population_empty"]
    if population.support_count == 0:
        return "rejected", ["population_has_no_support"]
    if falsification is None or falsification.status != "complete":
        reasons = ["falsification_incomplete"]
        if (
            falsification is not None
            and falsification.admission_estimate is not None
        ):
            reasons.append(
                "falsification_cost_not_admitted:"
                f"{falsification.admission_estimate.reason_code}"
            )
        return "review", reasons
    if falsification.verdict in {
        "alternative_non_norm_explanation",
        "reject_insufficient",
    }:
        return "rejected", [f"falsification:{falsification.verdict}"]

    reasons: list[str] = []
    interpretations = {
        item.interpretation for item in falsification.issue_interpretations
    }
    if "true_counterexample" in interpretations:
        reasons.append("falsification_true_counterexample")
    if "unresolved" in interpretations:
        reasons.append("falsification_unresolved_member")
    if "legitimate_exception" in interpretations:
        reasons.append("falsification_legitimate_exception")
    if not falsification.evidence_refs:
        reasons.append("falsification_missing_evidence_refs")
    if falsification.unresolved_questions:
        reasons.append("falsification_unresolved_questions")
    if falsification.verdict != "supported_as_written":
        reasons.append(f"falsification:{falsification.verdict}")
    for field_name in (
        "repository_specificity",
        "contributor_value",
        "scope_condition_fit",
        "evidence_sufficiency",
    ):
        value = getattr(falsification, field_name)
        if value != "pass":
            reasons.append(f"quality:{field_name}:{value}")
    if reasons:
        if any(reason.endswith(":fail") for reason in reasons):
            return "rejected", sorted(set(reasons))
        return "review", sorted(set(reasons))
    if refinement_round > 0:
        return "review", [
            "post_falsification_refinement",
            "refinement_requires_calibration",
        ]
    return "accept_candidate", [
        "complete_population_and_falsification",
        "local_quality_axes_passed",
        "pre_calibration_non_delivery",
    ]


@dataclass(frozen=True)
class _ScheduledPopulationRoot:
    hypothesis: NormHypothesis
    component_key: str
    semantic_member_cost: int
    matched_member_count: int
    conclusion_eligible: bool
    non_conclusion_reasons: tuple[
        Literal[
            "population_dependency_partial",
            "population_member_limit",
            "population_precost_failed",
        ],
        ...,
    ]
    classified_member_limit: int = 0


def _round_robin_population_roots(
    entries: Sequence[_ScheduledPopulationRoot],
) -> tuple[_ScheduledPopulationRoot, ...]:
    """Order complete roots first, then component-fair and cheap-first."""

    scheduled: list[_ScheduledPopulationRoot] = []
    for conclusion_eligible in (True, False):
        buckets: dict[str, list[_ScheduledPopulationRoot]] = defaultdict(list)
        for entry in entries:
            if entry.conclusion_eligible == conclusion_eligible:
                buckets[entry.component_key].append(entry)
        for values in buckets.values():
            values.sort(
                key=lambda item: (
                    item.semantic_member_cost,
                    item.matched_member_count,
                    item.hypothesis.candidate_root_id,
                )
            )
        offsets = {key: 0 for key in buckets}
        while True:
            advanced = False
            for component_key in sorted(buckets):
                offset = offsets[component_key]
                if offset >= len(buckets[component_key]):
                    continue
                scheduled.append(buckets[component_key][offset])
                offsets[component_key] += 1
                advanced = True
            if not advanced:
                break
    return tuple(scheduled)


def _population_result_non_conclusion_reasons(
    result: PopulationResult,
) -> tuple[
    Literal["population_dependency_partial", "population_member_limit"], ...
]:
    reasons: list[
        Literal["population_dependency_partial", "population_member_limit"]
    ] = []
    if result.omitted_count:
        reasons.append("population_member_limit")
    if (
        result.dependency_file_ids
        or result.dependency_paths
        or result.dependency_paths_omitted_count
        or (
            result.status != "complete"
            and not result.omitted_count
        )
    ):
        reasons.append("population_dependency_partial")
    return tuple(sorted(set(reasons)))


def _hypothesis_population_conclusion_eligible(
    hypothesis: NormHypothesis,
    *,
    superpopulation: PopulationResult,
    observations: ObservationBundle,
    semantic_settings: SemanticSettings,
) -> tuple[
    bool,
    tuple[
        Literal["population_dependency_partial", "population_member_limit"],
        ...,
    ],
]:
    results = [superpopulation]
    if hypothesis.predicate_mode == "structural":
        assert hypothesis.structural_predicate is not None
        results.extend(
            (
                evaluate_population(
                    hypothesis.structural_predicate.eligible_spec,
                    observations,
                    semantic_settings,
                ),
                evaluate_population(
                    hypothesis.structural_predicate.support_spec,
                    observations,
                    semantic_settings,
                ),
            )
        )
    reasons = tuple(
        sorted(
            {
                reason
                for item in results
                for reason in _population_result_non_conclusion_reasons(item)
            }
        )
    )
    return not reasons, reasons


def _population_root_schedule(
    hypotheses: Sequence[NormHypothesis],
    *,
    observations: ObservationBundle,
    semantic_settings: SemanticSettings,
) -> tuple[_ScheduledPopulationRoot, ...]:
    """Build a deterministic component-fair, cheap-first population order.

    The deterministic query census remains independent of provider inspection.
    Complete small-census semantic roots receive atomic whole-root allocations
    first; the remaining allowance is component/root-round-robin across wide
    roots. Structural and dependency-partial roots consume no provider-member
    allocation.
    """

    entries: list[_ScheduledPopulationRoot] = []
    for hypothesis in hypotheses:
        component_key = min(hypothesis.component_keys)
        try:
            result = evaluate_population(
                hypothesis.superpopulation_spec,
                observations,
                semantic_settings,
            )
            semantic_member_cost = (
                0
                if hypothesis.predicate_mode == "structural"
                else count_population_members_for_admission(
                    hypothesis.superpopulation_spec,
                    observations,
                    semantic_settings,
                    applies_to=hypothesis.applies_to,
                ).provider_eligible_in_scope_member_count
            )
            matched_member_count = result.matched_count
            conclusion_eligible, non_conclusion_reasons = (
                _hypothesis_population_conclusion_eligible(
                    hypothesis,
                    superpopulation=result,
                    observations=observations,
                    semantic_settings=semantic_settings,
                )
            )
        except (TypeError, ValueError):
            # Keep an operationally invalid root in the deterministic plan so
            # normal population construction records its typed failure.  Sort
            # it behind executable roots without inventing semantic evidence.
            semantic_member_cost = 0
            matched_member_count = (
                semantic_settings.max_population_members_per_hypothesis + 1
            )
            conclusion_eligible = False
            non_conclusion_reasons = ("population_precost_failed",)
        entries.append(
            _ScheduledPopulationRoot(
                hypothesis=hypothesis,
                component_key=component_key,
                semantic_member_cost=semantic_member_cost,
                matched_member_count=matched_member_count,
                conclusion_eligible=conclusion_eligible,
                non_conclusion_reasons=non_conclusion_reasons,
            )
        )
    ordered = _round_robin_population_roots(entries)
    allocations = _population_root_allocations(
        ordered,
        semantic_settings=semantic_settings,
    )
    return tuple(
        replace(
            item,
            classified_member_limit=allocations[item.hypothesis.hypothesis_id],
        )
        for item in ordered
    )


def _population_root_allocations(
    ordered: Sequence[_ScheduledPopulationRoot],
    *,
    semantic_settings: SemanticSettings,
) -> dict[str, int]:
    """Reserve true census roots atomically, then fairly sample wide roots.

    A root whose complete PopulationSpec matches no more than the per-root
    provider limit must not silently become a partial sample merely because a
    run-wide member allowance ends between round-robin turns.  Such roots are
    admitted as whole census jobs while capacity remains.  If the census jobs
    themselves exceed the run-wide allowance, a job that does not fit receives
    zero provider members and is later closed as a typed partial result.  The
    remaining allowance is distributed member-by-member across genuinely wide
    populations using the existing component/root round robin.
    """

    allocations = {item.hypothesis.hypothesis_id: 0 for item in ordered}
    desired = {
        item.hypothesis.hypothesis_id: min(
            item.semantic_member_cost,
            semantic_settings.max_classified_population_members_per_hypothesis,
        )
        for item in ordered
        if item.hypothesis.predicate_mode == "semantic"
        and item.conclusion_eligible
    }
    remaining = semantic_settings.max_total_classified_population_members
    census_threshold = (
        semantic_settings.max_classified_population_members_per_hypothesis
    )
    for item in ordered:
        hypothesis_id = item.hypothesis.hypothesis_id
        census_cost = desired.get(hypothesis_id, 0)
        if not census_cost or item.matched_member_count > census_threshold:
            continue
        if census_cost <= remaining:
            allocations[hypothesis_id] = census_cost
            remaining -= census_cost

    buckets: dict[str, list[_ScheduledPopulationRoot]] = defaultdict(list)
    for item in ordered:
        if (
            desired.get(item.hypothesis.hypothesis_id, 0)
            and item.matched_member_count > census_threshold
        ):
            buckets[item.component_key].append(item)
    offsets = {component: 0 for component in buckets}
    while remaining > 0:
        advanced = False
        for component in sorted(buckets):
            values = buckets[component]
            for _ in range(len(values)):
                offset = offsets[component] % len(values)
                offsets[component] += 1
                item = values[offset]
                hypothesis_id = item.hypothesis.hypothesis_id
                if allocations[hypothesis_id] >= desired[hypothesis_id]:
                    continue
                allocations[hypothesis_id] += 1
                remaining -= 1
                advanced = True
                break
            if remaining <= 0:
                break
        if not advanced:
            break
    return allocations


def _candidate_population_dependencies_complete(
    hypothesis: NormHypothesis,
    *,
    observations: ObservationBundle,
    semantic_settings: SemanticSettings,
) -> bool:
    """Replay query-local source dependencies instead of trusting a global flag."""

    result = evaluate_population(
        hypothesis.superpopulation_spec,
        observations,
        semantic_settings,
    )
    conclusion_eligible, _ = _hypothesis_population_conclusion_eligible(
        hypothesis,
        superpopulation=result,
        observations=observations,
        semantic_settings=semantic_settings,
    )
    return conclusion_eligible


def run_current_source_pipeline(
    *,
    repository_root,
    discovery: DiscoveryArtifact,
    observations: ObservationBundle,
    semantic_units: SemanticUnitBundle,
    semantic_settings: SemanticSettings,
    llm_settings: LLMSettings,
    client: LLMClient,
    budget: InferenceBudget,
    protocol_manifest_digest: str,
    explicit_context: Sequence[dict[str, object]] = (),
) -> CurrentSourceBundle:
    """Execute the bounded map-reduce-classify-falsify current-source channel."""

    semantic_map = run_semantic_map(
        repository_root=repository_root,
        discovery=discovery,
        semantic_units=semantic_units,
        semantic_settings=semantic_settings,
        llm_settings=llm_settings,
        client=client,
        budget=budget,
        protocol_manifest_digest=protocol_manifest_digest,
    )
    reduction = run_hypothesis_reduction(
        semantic_map=semantic_map,
        observations=observations,
        semantic_units=semantic_units,
        semantic_settings=semantic_settings,
        llm_settings=llm_settings,
        client=client,
        budget=budget,
        protocol_manifest_digest=protocol_manifest_digest,
        explicit_context=explicit_context,
    )
    all_hypotheses: list[NormHypothesis] = list(reduction.hypotheses)
    populations: list[PopulationArtifact] = []
    member_packets: list[PopulationMemberPacket] = []
    falsifications: list[FalsificationResult] = []
    refinements: list[RefinementRecord] = []
    inference_records: list[InferenceRecord] = [
        *semantic_map.inference_records,
        *reduction.inference_records,
    ]
    final_by_root: dict[str, NormHypothesis] = {}
    blocked_by_root: dict[str, str] = {}
    failed_population_hypotheses: set[str] = set()
    total_members = 0
    source_content_cache: dict[str, bytes] = {}
    classification_admission = ClassificationAdmissionLedger(
        semantic_settings
    )
    falsification_admission = FalsificationAdmissionLedger(
        semantic_settings
    )
    population_schedule = _population_root_schedule(
        reduction.hypotheses,
        observations=observations,
        semantic_settings=semantic_settings,
    )
    population_schedule_records = [
        PopulationScheduleRecord(
            ordinal=ordinal,
            candidate_root_id=item.hypothesis.candidate_root_id,
            hypothesis_id=item.hypothesis.hypothesis_id,
            component_key=item.component_key,
            semantic_member_cost=item.semantic_member_cost,
            matched_member_count=item.matched_member_count,
            classified_member_limit=item.classified_member_limit,
            provider_unallocated_count=max(
                item.semantic_member_cost - item.classified_member_limit,
                0,
            ),
            conclusion_eligible=item.conclusion_eligible,
            non_conclusion_reasons=list(item.non_conclusion_reasons),
        )
        for ordinal, item in enumerate(population_schedule)
    ]
    classification_admission_schedule: list[ClassificationAdmissionRecord] = []

    def classification_admission_counters() -> tuple[int, ...]:
        return (
            classification_admission.admitted_hypotheses,
            classification_admission.admitted_semantic_members,
            classification_admission.admitted_calls,
            classification_admission.admitted_request_bytes,
            classification_admission.not_admitted_hypotheses,
            classification_admission.not_admitted_semantic_members,
            classification_admission.not_admitted_planned_calls,
            classification_admission.not_admitted_planned_request_bytes,
        )

    def record_admission_delta(
        hypothesis: NormHypothesis,
        *,
        phase: Literal["initial", "refinement"],
        before: tuple[int, ...],
    ) -> None:
        after = classification_admission_counters()
        delta = tuple(right - left for left, right in zip(before, after))
        if not any(delta):
            return
        if delta[0] != 1 or any(delta[index] for index in range(4, 8)):
            raise ValueError("unexpected classification admission ledger delta")
        classification_admission_schedule.append(
            ClassificationAdmissionRecord(
                ordinal=len(classification_admission_schedule),
                phase=phase,
                candidate_root_id=hypothesis.candidate_root_id,
                hypothesis_id=hypothesis.hypothesis_id,
                outcome="admitted",
                semantic_members=delta[1],
                planned_calls=delta[2],
                planned_request_bytes=delta[3],
                reason_code=None,
                limit=None,
            )
        )

    def record_not_admitted(
        hypothesis: NormHypothesis,
        error: PopulationCostNotAdmitted,
        *,
        phase: Literal["initial", "refinement"],
        before: tuple[int, ...],
    ) -> None:
        after = classification_admission_counters()
        delta = tuple(right - left for left, right in zip(before, after))
        expected = (
            0,
            0,
            0,
            0,
            1,
            error.semantic_members,
            error.planned_calls,
            error.planned_request_bytes,
        )
        if delta != expected:
            raise ValueError("classification rejection ledger delta differs")
        classification_admission_schedule.append(
            ClassificationAdmissionRecord(
                ordinal=len(classification_admission_schedule),
                phase=phase,
                candidate_root_id=hypothesis.candidate_root_id,
                hypothesis_id=hypothesis.hypothesis_id,
                outcome="not_admitted",
                semantic_members=error.semantic_members,
                planned_calls=error.planned_calls,
                planned_request_bytes=error.planned_request_bytes,
                reason_code=error.reason_code,
                limit=error.limit,
            )
        )
    initial_population_by_root: dict[
        str,
        tuple[PopulationArtifact, tuple[PopulationMemberPacket, ...]],
    ] = {}

    # Phase one freezes and admits every reducer-admitted root before executing
    # any classification call.  Calls are then flattened in stable
    # population-schedule/batch order and share one bounded-wave runner.  This
    # preserves whole-root admission while allowing the configured concurrency
    # to work across the common one-batch-per-root case.
    pending_initial_populations: list[
        tuple[NormHypothesis, PendingPopulationBuild]
    ] = []
    for scheduled in population_schedule:
        root = scheduled.hypothesis
        remaining_members = (
            semantic_settings.max_total_population_members - total_members
        )
        if remaining_members <= 0:
            blocked_by_root[root.candidate_root_id] = (
                "max_total_population_members"
            )
            failed_population_hypotheses.add(root.hypothesis_id)
            final_by_root[root.candidate_root_id] = root
            continue
        admission_before = classification_admission_counters()
        try:
            started_population = start_population_classification(
                repository_root=repository_root,
                discovery=discovery,
                hypothesis=root,
                observations=observations,
                semantic_units=semantic_units,
                semantic_settings=semantic_settings,
                llm_settings=llm_settings,
                client=client,
                budget=budget,
                protocol_manifest_digest=protocol_manifest_digest,
                # Reducer/map failures outside this PopulationSpec remain
                # independent. Missing source dependencies are replayed from
                # the exact candidate query rather than assumed complete.
                candidate_local_dependencies_complete=(
                    scheduled.conclusion_eligible
                ),
                max_members=remaining_members,
                max_classified_members=scheduled.classified_member_limit,
                source_content_cache=source_content_cache,
                classification_admission=classification_admission,
            )
        except PopulationCostNotAdmitted as error:
            record_not_admitted(
                root,
                error,
                phase="initial",
                before=admission_before,
            )
            blocked_by_root[root.candidate_root_id] = (
                "population_cost_not_admitted:"
                f"{error.reason_code}:members={error.semantic_members}:"
                f"calls={error.planned_calls}:"
                f"request_bytes={error.planned_request_bytes}:"
                f"limit={error.limit}"
            )
            failed_population_hypotheses.add(root.hypothesis_id)
            final_by_root[root.candidate_root_id] = root
            continue
        except (ValueError, TypeError) as error:
            record_admission_delta(
                root,
                phase="initial",
                before=admission_before,
            )
            blocked_by_root[root.candidate_root_id] = (
                "population_operationalization_failed:"
                + " ".join(str(error).split())[:160]
            )
            failed_population_hypotheses.add(root.hypothesis_id)
            final_by_root[root.candidate_root_id] = root
            continue
        record_admission_delta(
            root,
            phase="initial",
            before=admission_before,
        )
        if isinstance(started_population, PendingPopulationBuild):
            # Reserve the exact frozen denominator before planning the next
            # root.  A later provider/finalization failure cannot grant a
            # schedule-dependent population allowance to a suffix root.
            total_members += started_population.work.member_count
            pending_initial_populations.append((root, started_population))
            continue
        built_population = started_population
        population = built_population.artifact
        packets = built_population.packets
        populations.append(population)
        member_packets.extend(packets)
        inference_records.extend(built_population.inference_records)
        total_members += population.population_result.serialized_count
        initial_population_by_root[root.candidate_root_id] = (
            population,
            packets,
        )
        if (
            population.status == "complete"
            and population.population_result.matched_count == 0
        ):
            blocked_by_root[root.candidate_root_id] = (
                "complete_empty_population"
            )
            final_by_root[root.candidate_root_id] = root

    classification_transport_plans: list[ClassificationTransportPlan] = []
    classification_transport = (
        execute_classification_transport(
            [pending for _, pending in pending_initial_populations],
            semantic_settings=semantic_settings,
            llm_settings=llm_settings, budget=budget,
        )
        if pending_initial_populations else None
    )
    if classification_transport is not None:
        classification_transport_plans.append(classification_transport.plan)
        # A physical request is charged once even when a root artifact cannot
        # be materialized. Root projections never contribute another record.
        inference_records.extend(classification_transport.records)
    for pending_ordinal, (root, pending) in enumerate(pending_initial_populations):
        local_stage = classification_transport.root_stages[pending_ordinal]
        try:
            built_population = finish_population_classification(
                pending,
                local_stage,
            )
        except (ValueError, TypeError) as error:
            blocked_by_root[root.candidate_root_id] = (
                "population_operationalization_failed:"
                + " ".join(str(error).split())[:160]
            )
            failed_population_hypotheses.add(root.hypothesis_id)
            final_by_root[root.candidate_root_id] = root
            continue
        population = built_population.artifact
        packets = built_population.packets
        populations.append(population)
        member_packets.extend(packets)
        initial_population_by_root[root.candidate_root_id] = (
            population,
            packets,
        )
        if (
            population.status == "complete"
            and population.population_result.matched_count == 0
        ):
            blocked_by_root[root.candidate_root_id] = (
                "complete_empty_population"
            )
            final_by_root[root.candidate_root_id] = root

    # Phase two first plans and admits every eligible round-zero falsification,
    # then executes their calls in one stable cross-root bounded wave.  The
    # refinement barrier remains unchanged: child work starts only after every
    # independent round-zero result is terminal.
    initial_falsification_by_root: dict[str, FalsificationBuildResult] = {}
    pending_initial_falsifications: list[
        tuple[NormHypothesis, PendingFalsificationBuild]
    ] = []
    for scheduled in population_schedule:
        root = scheduled.hypothesis
        if root.candidate_root_id in final_by_root:
            continue
        initial = initial_population_by_root.get(root.candidate_root_id)
        if initial is None:
            continue
        current_population, current_packets = initial
        ineligibility = falsification_ineligibility_reason(current_population)
        if ineligibility == "no_population_support":
            initial_falsification_by_root[root.candidate_root_id] = (
                make_zero_support_falsification(
                    hypothesis=root,
                    population=current_population,
                )
            )
            continue
        if ineligibility is not None:
            blocked_by_root[root.candidate_root_id] = ineligibility
            final_by_root[root.candidate_root_id] = root
            continue
        try:
            started_falsification = start_falsification(
                hypothesis=root,
                population=current_population,
                member_packets=current_packets,
                semantic_settings=semantic_settings,
                llm_settings=llm_settings,
                client=client,
                budget=budget,
                protocol_manifest_digest=protocol_manifest_digest,
                explicit_context=explicit_context,
                falsification_admission=falsification_admission,
            )
        except (ValueError, TypeError) as error:
            blocked_by_root[root.candidate_root_id] = (
                "falsification_failed:"
                + " ".join(str(error).split())[:160]
            )
            final_by_root[root.candidate_root_id] = root
            continue
        if isinstance(started_falsification, PendingFalsificationBuild):
            pending_initial_falsifications.append(
                (root, started_falsification)
            )
        else:
            initial_falsification_by_root[root.candidate_root_id] = (
                started_falsification
            )

    falsification_work = [
        (pending_ordinal, item)
        for pending_ordinal, (_, pending) in enumerate(
            pending_initial_falsifications
        )
        for item in pending.work.items
    ]

    def invoke_falsification_work(
        work: tuple[
            int,
            tuple[
                int,
                tuple[list[PopulationMember], list[PopulationMember]],
            ],
        ],
    ) -> FalsificationBatchOutcome:
        pending_ordinal, item = work
        return pending_initial_falsifications[
            pending_ordinal
        ][1].work.invoke(item)

    falsification_stage = (
        run_budgeted_stage_calls(
            falsification_work,
            invoke_falsification_work,
            stage="counterexample_verify",
            budget=budget,
            max_concurrency=llm_settings.max_concurrency,
            thread_name_prefix="repo-memory-v3-falsification",
        )
        if falsification_work
        else BudgetedStageRunResult(
            outcomes=(),
            planned=0,
            omitted=0,
            exhaustion_reason=None,
        )
    )
    falsification_outcome_ordinal = 0
    remaining_falsification_outcomes = len(falsification_stage.outcomes)
    for root, pending in pending_initial_falsifications:
        planned = len(pending.work.items)
        terminal = min(planned, remaining_falsification_outcomes)
        local_outcomes = falsification_stage.outcomes[
            falsification_outcome_ordinal :
            falsification_outcome_ordinal + terminal
        ]
        falsification_outcome_ordinal += terminal
        remaining_falsification_outcomes -= terminal
        omitted = planned - terminal
        local_stage = BudgetedStageRunResult[FalsificationBatchOutcome](
            outcomes=local_outcomes,
            planned=planned,
            omitted=omitted,
            exhaustion_reason=(
                falsification_stage.exhaustion_reason if omitted else None
            ),
        )
        try:
            initial_falsification_by_root[root.candidate_root_id] = (
                finish_falsification(pending, local_stage)
            )
        except (ValueError, TypeError) as error:
            blocked_by_root[root.candidate_root_id] = (
                "falsification_failed:"
                + " ".join(str(error).split())[:160]
            )
            final_by_root[root.candidate_root_id] = root

    for scheduled in population_schedule:
        root = scheduled.hypothesis
        built_falsification = initial_falsification_by_root.get(
            root.candidate_root_id
        )
        if built_falsification is None:
            continue
        initial = initial_population_by_root[root.candidate_root_id]
        current = root
        current_population, current_packets = initial
        while True:
            falsification = built_falsification.result
            falsifications.append(falsification)
            inference_records.extend(built_falsification.inference_records)
            can_refine = (
                falsification.status == "complete"
                and falsification.verdict
                in {
                    "narrow_scope",
                    "narrow_condition",
                    "legitimate_exception",
                    "missing_query",
                }
            )
            if not can_refine:
                final_by_root[root.candidate_root_id] = current
                break
            if current.refinement_round >= semantic_settings.max_refinement_rounds:
                blocked_by_root[root.candidate_root_id] = (
                    "refinement_round_limit_reached"
                )
                final_by_root[root.candidate_root_id] = current
                break
            try:
                child, change_kind = make_refined_hypothesis(
                    current, falsification
                )
            except ValueError as error:
                blocked_by_root[root.candidate_root_id] = (
                    "invalid_refinement:"
                    + " ".join(str(error).split())[:160]
                )
                final_by_root[root.candidate_root_id] = current
                break
            all_hypotheses.append(child)
            parent = current
            parent_population = current_population
            remaining_members = (
                semantic_settings.max_total_population_members - total_members
            )
            if remaining_members <= 0:
                blocked_by_root[root.candidate_root_id] = (
                    "refinement_population_budget_exhausted"
                )
                failed_population_hypotheses.add(child.hypothesis_id)
                final_by_root[root.candidate_root_id] = child
                break
            admission_before = classification_admission_counters()
            try:
                child_started = start_population_classification(
                    repository_root=repository_root,
                    discovery=discovery,
                    hypothesis=child,
                    observations=observations,
                    semantic_units=semantic_units,
                    semantic_settings=semantic_settings,
                    llm_settings=llm_settings,
                    client=client,
                    budget=budget,
                    protocol_manifest_digest=protocol_manifest_digest,
                    candidate_local_dependencies_complete=(
                        _candidate_population_dependencies_complete(
                            child,
                            observations=observations,
                            semantic_settings=semantic_settings,
                        )
                    ),
                    max_members=remaining_members,
                    max_classified_members=min(
                        semantic_settings
                        .max_classified_population_members_per_hypothesis,
                        max(
                            semantic_settings
                            .max_total_classified_population_members
                            - classification_admission
                            .admitted_semantic_members,
                            0,
                        ),
                    ),
                    required_inspected_observation_ids=tuple(
                        item.observation_ref.observation_id
                        for item in parent_population.members
                    ),
                    source_content_cache=source_content_cache,
                    classification_admission=classification_admission,
                )
                if isinstance(child_started, PendingPopulationBuild):
                    child_transport = execute_classification_transport(
                        [child_started], semantic_settings=semantic_settings,
                        llm_settings=llm_settings, budget=budget,
                    )
                    inference_records.extend(child_transport.records)
                    classification_transport_plans.append(child_transport.plan)
                    child_build = finish_population_classification(
                        child_started, child_transport.root_stages[0],
                    )
                else:
                    child_build = child_started
                refinement = make_refinement_record(
                    parent=parent,
                    child=child,
                    parent_population=parent_population,
                    child_population=child_build.artifact,
                    falsification=falsification,
                    change_kind=change_kind,
                )
            except PopulationCostNotAdmitted as error:
                record_not_admitted(
                    child,
                    error,
                    phase="refinement",
                    before=admission_before,
                )
                blocked_by_root[root.candidate_root_id] = (
                    "refinement_population_cost_not_admitted:"
                    f"{error.reason_code}:members={error.semantic_members}:"
                    f"calls={error.planned_calls}:"
                    f"request_bytes={error.planned_request_bytes}:"
                    f"limit={error.limit}"
                )
                failed_population_hypotheses.add(child.hypothesis_id)
                final_by_root[root.candidate_root_id] = child
                break
            except (ValueError, TypeError) as error:
                record_admission_delta(
                    child,
                    phase="refinement",
                    before=admission_before,
                )
                blocked_by_root[root.candidate_root_id] = (
                    "refinement_population_failed:"
                    + " ".join(str(error).split())[:160]
                )
                failed_population_hypotheses.add(child.hypothesis_id)
                final_by_root[root.candidate_root_id] = child
                break
            record_admission_delta(
                child,
                phase="refinement",
                before=admission_before,
            )
            populations.append(child_build.artifact)
            member_packets.extend(child_build.packets)
            total_members += (
                child_build.artifact.population_result.serialized_count
            )
            refinements.append(refinement)
            current = child
            # Reuse the frozen child population; the next iteration starts at
            # independent falsification and can perform another bounded round.
            current_population = child_build.artifact
            current_packets = child_build.packets
            ineligibility = falsification_ineligibility_reason(
                current_population
            )
            if ineligibility == "no_population_support":
                built_falsification = make_zero_support_falsification(
                    hypothesis=current,
                    population=current_population,
                )
                continue
            if ineligibility is not None:
                blocked_by_root[root.candidate_root_id] = ineligibility
                final_by_root[root.candidate_root_id] = current
                break
            try:
                built_falsification = run_falsification(
                    hypothesis=current,
                    population=current_population,
                    member_packets=current_packets,
                    semantic_settings=semantic_settings,
                    llm_settings=llm_settings,
                    client=client,
                    budget=budget,
                    protocol_manifest_digest=protocol_manifest_digest,
                    explicit_context=explicit_context,
                    falsification_admission=falsification_admission,
                )
            except (ValueError, TypeError) as error:
                blocked_by_root[root.candidate_root_id] = (
                    "falsification_failed:"
                    + " ".join(str(error).split())[:160]
                )
                final_by_root[root.candidate_root_id] = current
                break

    population_by_hypothesis = {item.hypothesis_id: item for item in populations}
    falsification_by_hypothesis = {
        item.hypothesis_id: item for item in falsifications
    }
    propositions: list[PropositionEnvelope] = []
    review_items: list[CurrentSourceReviewItem] = []
    policy_decisions: list[CurrentSourcePolicyDecision] = []
    trace_dispositions: dict[
        str,
        tuple[
            Literal["accepted", "hypothesis", "review", "rejected"],
            list[str],
            str | None,
        ],
    ] = {}
    for root_id, hypothesis in sorted(final_by_root.items()):
        population = population_by_hypothesis.get(hypothesis.hypothesis_id)
        falsification = falsification_by_hypothesis.get(
            hypothesis.hypothesis_id
        )
        disposition, reasons = _disposition(
            population,
            falsification,
            blocked_reason=blocked_by_root.get(root_id),
        )
        proposition = _current_proposition(
            hypothesis=hypothesis,
            population=population,
            falsification=falsification,
        )
        propositions.append(proposition)
        trace_id = semantic_id(
            "TRACE",
            {
                "snapshot_digest": hypothesis.snapshot_digest,
                "channel": "implicit_current",
                "candidate_root_id": root_id,
            },
        )
        review_items.append(
            CurrentSourceReviewItem(
                trace_id=trace_id,
                candidate_root_id=root_id,
                hypothesis_id=hypothesis.hypothesis_id,
                proposition_id=proposition.proposition_id,
                epistemic_status=disposition,
                evidence_route=(
                    population.route if population else "unoperationalized"
                ),
                category=hypothesis.category,
                rule=hypothesis.rule,
                applies_to=hypothesis.applies_to,
                condition=hypothesis.condition,
                population_id=population.population_id if population else None,
                falsification_ids=(
                    [falsification.falsification_id]
                    if falsification
                    else []
                ),
                reason_codes=reasons,
                limitations=[
                    *hypothesis.limitations,
                    *(population.limitations if population else []),
                ],
            )
        )
        experimental_disposition, experimental_reasons = (
            _experimental_disposition(
                population,
                falsification,
                canonical_status=disposition,
                refinement_round=hypothesis.refinement_round,
                blocked_reason=blocked_by_root.get(root_id),
            )
        )
        policy_payload = {
            "policy_version": CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
            "snapshot_digest": hypothesis.snapshot_digest,
            "candidate_root_id": root_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "proposition_id": proposition.proposition_id,
            "experimental_disposition": experimental_disposition,
            "canonical_epistemic_status": disposition,
            "reason_codes": sorted(set(experimental_reasons)),
        }
        policy_payload["artifact_digest"] = payload_digest(policy_payload)
        policy_decisions.append(
            CurrentSourcePolicyDecision.model_validate(policy_payload)
        )
        trace_dispositions[root_id] = (
            disposition,
            reasons,
            proposition.proposition_id,
        )
    operational_block_reasons = {
        reason
        for reason in blocked_by_root.values()
        if reason
        not in {
            "refinement_round_limit_reached",
            "no_population_support",
        }
    }
    incomplete = {
        *semantic_map.index.incomplete_reasons,
        *reduction.index.incomplete_reasons,
        *operational_block_reasons,
        *(
            reason
            for item in populations
            for reason in item.incomplete_reasons
        ),
        *(
            reason
            for item in falsifications
            for reason in item.incomplete_reasons
        ),
    }
    pipeline_status: Literal["complete", "partial", "failed"] = (
        "complete" if not incomplete else "partial"
    )
    traces, event_traces, trace_channel = build_current_traces(
        snapshot_digest=discovery.repository.snapshot_digest,
        events=reduction.events,
        hypotheses=all_hypotheses,
        populations=populations,
        falsifications=falsifications,
        refinements=refinements,
        dispositions=trace_dispositions,
        pipeline_status=pipeline_status,
    )
    disposition_counts = {
        value: sum(item.epistemic_status == value for item in review_items)
        for value in ("hypothesis", "review", "rejected")
    }
    experimental_counts = {
        value: sum(
            item.experimental_disposition == value
            for item in policy_decisions
        )
        for value in ("accept_candidate", "review", "rejected")
    }
    population_status = {
        value: sum(item.status == value for item in populations)
        for value in ("complete", "partial", "failed")
    }
    artifact_digests = [
        semantic_map.index.index_digest,
        *(item.artifact_digest for item in semantic_map.results),
        *(item.artifact_digest for item in semantic_map.observations),
        reduction.index.index_digest,
        *(item.artifact_digest for item in reduction.call_results),
        *(item.artifact_digest for item in reduction.events),
        *(item.artifact_digest for item in all_hypotheses),
        *(item.artifact_digest for item in populations),
        *(item.packet_digest for item in member_packets),
        *(item.artifact_digest for item in falsifications),
        *(item.artifact_digest for item in refinements),
        *(item.artifact_digest for item in propositions),
        *(item.artifact_digest for item in policy_decisions),
        *(item.artifact_digest for item in traces),
        *(item.artifact_digest for item in event_traces),
    ]
    validate_classification_transport_bindings(
        plans=classification_transport_plans,
        populations=populations,
        records=inference_records,
    )
    index_payload = {
        "index_version": CURRENT_SOURCE_INDEX_VERSION,
        "snapshot_digest": discovery.repository.snapshot_digest,
        "required": semantic_settings.required,
        "status": pipeline_status,
        "semantic_map_digest": semantic_map.index.index_digest,
        "hypothesis_reduction_digest": reduction.index.index_digest,
        "semantic_observations": len(semantic_map.observations),
        "candidate_events": len(reduction.events),
        "unique_candidate_roots": len(reduction.hypotheses),
        "hypothesis_versions": len(all_hypotheses),
        "complete_populations": population_status["complete"],
        "partial_populations": population_status["partial"],
        "failed_populations": (
            population_status["failed"] + len(failed_population_hypotheses)
        ),
        "classified_members": sum(len(item.members) for item in populations),
        "population_matched_members": sum(
            item.sampling_plan.matched_count for item in populations
        ),
        "population_inspected_members": sum(
            item.sampling_plan.inspected_count for item in populations
        ),
        "population_uninspected_members": sum(
            item.sampling_plan.uninspected_count for item in populations
        ),
        "population_provider_eligible_members": sum(
            item.sampling_plan.provider_eligible_count for item in populations
        ),
        "population_provider_inspected_members": sum(
            item.sampling_plan.provider_inspected_count for item in populations
        ),
        "population_provider_uninspected_members": sum(
            item.sampling_plan.provider_uninspected_count for item in populations
        ),
        "census_populations": sum(
            item.sampling_plan.mode == "census" for item in populations
        ),
        "sampled_populations": sum(
            item.sampling_plan.mode == "deterministic_stratified_sample"
            for item in populations
        ),
        "technical_unknown_members": sum(
            item.technical_unknown_count for item in populations
        ),
        "completed_falsifications": sum(
            item.status == "complete" for item in falsifications
        ),
        "partial_falsifications": (
            len(all_hypotheses)
            - sum(item.status == "complete" for item in falsifications)
        ),
        "refinements": len(refinements),
        "propositions": len(propositions),
        "root_hypothesis": disposition_counts["hypothesis"],
        "root_review": disposition_counts["review"],
        "root_rejected": disposition_counts["rejected"],
        "root_accepted": 0,
        "experimental_accept_candidates": experimental_counts[
            "accept_candidate"
        ],
        "experimental_review": experimental_counts["review"],
        "experimental_rejected": experimental_counts["rejected"],
        "inference_ids": sorted(
            item.request_id for item in inference_records
        ),
        "artifact_set_digest": payload_digest(
            {"artifact_digests": artifact_digests}
        ),
        "population_schedule": [
            item.model_dump(mode="json") for item in population_schedule_records
        ],
        "classification_admission_schedule": [
            item.model_dump(mode="json")
            for item in classification_admission_schedule
        ],
        "classification_transport_plans": [
            item.model_dump(mode="json") for item in classification_transport_plans
        ],
        "resource_use": {
            **classification_transport_totals(classification_transport_plans),
            "inference_requests": len(inference_records),
            "semantic_observations": len(semantic_map.observations),
            "hypothesis_versions": len(all_hypotheses),
            "population_members": sum(len(item.members) for item in populations),
            "population_matched_members": sum(
                item.sampling_plan.matched_count for item in populations
            ),
            "population_inspected_members": sum(
                item.sampling_plan.inspected_count for item in populations
            ),
            "population_uninspected_members": sum(
                item.sampling_plan.uninspected_count for item in populations
            ),
            "population_provider_eligible_members": sum(
                item.sampling_plan.provider_eligible_count
                for item in populations
            ),
            "population_provider_inspected_members": sum(
                item.sampling_plan.provider_inspected_count
                for item in populations
            ),
            "population_provider_uninspected_members": sum(
                item.sampling_plan.provider_uninspected_count
                for item in populations
            ),
            "population_census_roots": sum(
                item.sampling_plan.mode == "census" for item in populations
            ),
            "population_sampled_roots": sum(
                item.sampling_plan.mode == "deterministic_stratified_sample"
                for item in populations
            ),
            "failed_population_hypotheses": len(failed_population_hypotheses),
            "population_schedule_roots": len(population_schedule),
            "population_schedule_conclusion_eligible_roots": sum(
                item.conclusion_eligible for item in population_schedule
            ),
            "population_schedule_non_conclusion_eligible_roots": sum(
                not item.conclusion_eligible for item in population_schedule
            ),
            "population_schedule_non_conclusion_dependency_partial_roots": sum(
                "population_dependency_partial" in item.non_conclusion_reasons
                for item in population_schedule
            ),
            "population_schedule_non_conclusion_member_limit_roots": sum(
                "population_member_limit" in item.non_conclusion_reasons
                for item in population_schedule
            ),
            "population_schedule_non_conclusion_precost_failed_roots": sum(
                "population_precost_failed" in item.non_conclusion_reasons
                for item in population_schedule
            ),
            "population_schedule_conclusion_eligible_semantic_members": sum(
                item.semantic_member_cost
                for item in population_schedule
                if item.conclusion_eligible
                and item.semantic_member_cost
                <= semantic_settings.max_population_members_per_hypothesis
            ),
            "population_schedule_non_conclusion_eligible_semantic_members": sum(
                item.semantic_member_cost
                for item in population_schedule
                if not item.conclusion_eligible
                and item.semantic_member_cost
                <= semantic_settings.max_population_members_per_hypothesis
            ),
            "population_schedule_semantic_members": sum(
                item.semantic_member_cost
                for item in population_schedule
                if item.semantic_member_cost
                <= semantic_settings.max_population_members_per_hypothesis
            ),
            "population_schedule_provider_allocated_members": sum(
                item.classified_member_limit for item in population_schedule
            ),
            "population_schedule_provider_unallocated_members": sum(
                max(
                    item.semantic_member_cost - item.classified_member_limit,
                    0,
                )
                for item in population_schedule
            ),
            "falsification_terminal_inference_record_hypotheses": sum(
                bool(item.inference_ids) for item in falsifications
            ),
            "falsification_deterministic_rejections": sum(
                item.status == "complete"
                and item.verdict == "reject_insufficient"
                and not item.inference_ids
                for item in falsifications
            ),
            "falsification_not_planned_roots": sum(
                reason
                in {
                    "population_incomplete",
                    "classification_incomplete",
                    "technical_unknown_members",
                }
                for reason in blocked_by_root.values()
            ),
            "classification_admitted_hypotheses": (
                classification_admission.admitted_hypotheses
            ),
            "classification_admission_attempts": len(
                classification_admission_schedule
            ),
            "classification_admitted_semantic_members": (
                classification_admission.admitted_semantic_members
            ),
            "classification_admitted_calls": (
                classification_admission.admitted_calls
            ),
            "classification_admitted_request_bytes": (
                classification_admission.admitted_request_bytes
            ),
            "classification_not_admitted_hypotheses": (
                classification_admission.not_admitted_hypotheses
            ),
            "classification_not_admitted_semantic_members": (
                classification_admission.not_admitted_semantic_members
            ),
            "classification_not_admitted_planned_calls": (
                classification_admission.not_admitted_planned_calls
            ),
            "classification_not_admitted_planned_request_bytes": (
                classification_admission
                .not_admitted_planned_request_bytes
            ),
            "falsification_admitted_hypotheses": (
                falsification_admission.admitted_hypotheses
            ),
            "falsification_admitted_calls": (
                falsification_admission.admitted_calls
            ),
            "falsification_admitted_request_bytes": (
                falsification_admission.admitted_request_bytes
            ),
            "falsification_cost_not_admitted_hypotheses": (
                falsification_admission.not_admitted_hypotheses
            ),
            "falsification_cost_not_admitted_calls": (
                falsification_admission.not_admitted_calls
            ),
            "falsification_cost_not_admitted_request_bytes": (
                falsification_admission.not_admitted_request_bytes
            ),
            "artifact_bytes": 0,
        },
        "incomplete_reasons": sorted(incomplete),
        "limitations": [
            "current_source_outputs_are_review_only_pre_calibration",
            "engineering_coverage_does_not_establish_semantic_quality",
        ],
    }
    index_payload["index_digest"] = payload_digest(index_payload)
    provisional = CurrentSourceIndex.model_validate(index_payload)
    artifact_bytes = sum(
        len(canonical_json_bytes(item))
        for item in (
            provisional,
            semantic_map.index,
            *semantic_map.observations,
            *semantic_map.results,
            reduction.index,
            *reduction.call_results,
            *reduction.events,
            *all_hypotheses,
            *populations,
            *member_packets,
            *falsifications,
            *refinements,
            *propositions,
            *policy_decisions,
            *traces,
            *event_traces,
        )
    )
    if artifact_bytes > semantic_settings.max_current_source_artifact_bytes:
        incomplete.add("max_current_source_artifact_bytes")
        pipeline_status = "partial"
        trace_channel = trace_channel.model_copy(
            update={
                "status": "partial",
                "reason": "max_current_source_artifact_bytes",
            }
        )
    index_payload.update(
        {
            "status": pipeline_status,
            "resource_use": {
                **index_payload["resource_use"],
                "artifact_bytes": artifact_bytes,
            },
            "incomplete_reasons": sorted(incomplete),
        }
    )
    index_payload["index_digest"] = payload_digest(
        {key: value for key, value in index_payload.items() if key != "index_digest"}
    )
    return CurrentSourceBundle(
        index=CurrentSourceIndex.model_validate(index_payload),
        semantic_map=semantic_map,
        reduction=reduction,
        hypotheses=tuple(
            sorted(all_hypotheses, key=lambda item: item.hypothesis_id)
        ),
        populations=tuple(
            sorted(populations, key=lambda item: item.population_id)
        ),
        member_packets=tuple(
            sorted(member_packets, key=lambda item: item.packet_id)
        ),
        falsifications=tuple(
            sorted(falsifications, key=lambda item: item.falsification_id)
        ),
        refinements=tuple(
            sorted(refinements, key=lambda item: item.refinement_id)
        ),
        propositions=tuple(
            sorted(propositions, key=lambda item: item.proposition_id)
        ),
        review_items=tuple(
            sorted(review_items, key=lambda item: item.trace_id)
        ),
        policy_decisions=tuple(
            sorted(
                policy_decisions,
                key=lambda item: item.candidate_root_id,
            )
        ),
        traces=tuple(traces),
        event_traces=tuple(event_traces),
        trace_channel=trace_channel,
        inference_records=tuple(inference_records),
    )


def _read_canonical_model(
    path: Path,
    model: type[StrictModel],
    errors: list[str],
):
    relative = path.as_posix()
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("missing or symlinked artifact")
        text = path.read_text(encoding="utf-8")
        value = model.model_validate_json(text)
        if text != canonical_json_text(value):
            errors.append(f"{relative} is not canonical JSON")
        return value
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        errors.append(f"{relative} error: {error}")
        return None


def validate_current_source_tree(
    run_directory: Path,
    *,
    repository_root: Path,
    discovery: DiscoveryArtifact,
    observations: ObservationBundle,
    semantic_units: SemanticUnitBundle,
    semantic_settings: SemanticSettings,
    llm_settings: LLMSettings,
    protocol_manifest_digest: str,
    inference_records: Sequence[InferenceRecord],
    explicit_context: Sequence[dict[str, object]],
    expected_stage_omissions: (
        Mapping[str, tuple[int, Mapping[str, int]]] | None
    ) = None,
) -> tuple[list[str], CurrentSourceBundle | None]:
    """Strict-parse and provider-free replay the entire current-source subtree."""

    errors: list[str] = []
    root = run_directory / "current-source"
    if root.is_symlink() or not root.is_dir():
        return ["current-source/ is missing or a symlink"], None
    index = _read_canonical_model(root / "index.json", CurrentSourceIndex, errors)
    map_index = _read_canonical_model(
        root / "semantic-observations" / "index.json",
        SemanticMapIndex,
        errors,
    )
    reduction_index = _read_canonical_model(
        root / "hypotheses" / "index.json",
        HypothesisReductionIndex,
        errors,
    )
    if index is None or map_index is None or reduction_index is None:
        return errors, None

    map_results: list[SemanticMapPacketResult] = []
    result_ref_by_id = {item.result_id: item for item in map_index.results}
    for result_id in sorted(result_ref_by_id):
        value = _read_canonical_model(
            root / "semantic-observations" / "units" / f"{result_id}.json",
            SemanticMapPacketResult,
            errors,
        )
        if value is not None:
            reference = result_ref_by_id[result_id]
            if (
                value.packet_id != reference.packet_id
                or value.artifact_digest != reference.artifact_digest
            ):
                errors.append(f"{result_id}: semantic-map reference mismatch")
            map_results.append(value)
    semantic_observations: list[SemanticObservation] = []
    for reference in map_index.observations:
        value = _read_canonical_model(
            root
            / "semantic-observations"
            / "observations"
            / f"{reference.observation_id}.json",
            SemanticObservation,
            errors,
        )
        if value is not None:
            if (
                value.artifact_digest != reference.artifact_digest
                or value.packet_id != reference.packet_id
            ):
                errors.append(
                    f"{reference.observation_id}: semantic-observation reference mismatch"
                )
            semantic_observations.append(value)
    reduction_results: list[ReductionCallResult] = []
    for result_id in sorted(reduction_index.call_results):
        value = _read_canonical_model(
            root
            / "hypotheses"
            / "reduction-results"
            / f"{result_id}.json",
            ReductionCallResult,
            errors,
        )
        if value is not None:
            reduction_results.append(value)
    hypotheses: list[NormHypothesis] = []
    for path in sorted((root / "hypotheses").glob("HYP-*.json")):
        value = _read_canonical_model(path, NormHypothesis, errors)
        if value is not None:
            hypotheses.append(value)
    populations: list[PopulationArtifact] = []
    for path in sorted((root / "populations").glob("POP-*.json")):
        value = _read_canonical_model(path, PopulationArtifact, errors)
        if value is not None:
            populations.append(value)
    packets: list[PopulationMemberPacket] = []
    packet_root = root / "member-packets"
    if packet_root.exists():
        for path in sorted(packet_root.glob("MEMPACKET-*.json")):
            value = _read_canonical_model(path, PopulationMemberPacket, errors)
            if value is not None:
                packets.append(value)
    # Non-complete shards are source-quarantined.  This is an artifact-integrity
    # rule, not merely a replay preference: neither saved source text nor an
    # inference assignment may survive for one of these members.
    incomplete_file_ids = {
        shard.file_id
        for shard in observations.shards
        if shard.parse_status != "complete"
    }
    file_id_by_observation = {
        observation_id: shard.file_id
        for shard in observations.shards
        for observation_id in (
            shard.file_id,
            *(item.entity_id for item in shard.entities),
            *(item.site_id for item in shard.sites),
            *(item.relation_id for item in shard.relations),
        )
    }
    member_by_key = {
        (population.hypothesis_id, member.observation_ref.observation_id): member
        for population in populations
        for member in population.members
    }
    planned_member_ids = {
        member_id
        for population in populations
        for batch in population.classification_plan.batches
        for member_id in batch.member_ids
    }
    inferred_packet_ids = {
        reference.stable_id
        for record in inference_records
        for reference in record.input_manifest
        if reference.kind == "population_member_packet"
    }

    def contains_forbidden_source(value: object) -> bool:
        if isinstance(value, dict):
            if set(value) & {"full_file_source", "semantic_unit", "source_ranges", "text"}:
                return True
            return any(contains_forbidden_source(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_forbidden_source(item) for item in value)
        return False

    for packet in packets:
        file_id = file_id_by_observation.get(
            packet.observation_ref.observation_id,
            packet.observation_ref.file_id,
        )
        if file_id not in incomplete_file_ids:
            continue
        prefix = f"{packet.packet_id}: non-complete shard packet"
        if packet.observation_ref.file_id != file_id:
            errors.append(f"{prefix} has a spoofed owning file")
        if packet.owning_unit_id is not None or contains_forbidden_source(packet.fact):
            errors.append(f"{prefix} contains source context")
        member = member_by_key.get(
            (packet.hypothesis_id, packet.observation_ref.observation_id)
        )
        if member is None:
            errors.append(f"{prefix} has no population member")
            continue
        if (
            member.classification_basis != "technical_unknown"
            or member.inference_ref is not None
        ):
            errors.append(f"{prefix} is not a deterministic technical unknown")
        if member.member_id in planned_member_ids:
            errors.append(f"{prefix} is assigned to a provider batch")
        if packet.packet_id in inferred_packet_ids:
            errors.append(f"{prefix} appears in an inference input manifest")
    falsifications: list[FalsificationResult] = []
    falsification_root = root / "falsifications"
    if falsification_root.exists():
        for path in sorted(falsification_root.glob("FALSIFY-*.json")):
            value = _read_canonical_model(path, FalsificationResult, errors)
            if value is not None:
                falsifications.append(value)
    refinements: list[RefinementRecord] = []
    refinement_root = root / "refinements"
    if refinement_root.exists():
        for path in sorted(refinement_root.glob("REFINE-*.json")):
            value = _read_canonical_model(path, RefinementRecord, errors)
            if value is not None:
                refinements.append(value)
    traces: list[CandidateTrace] = []
    for path in sorted((root / "lineage").glob("TRACE-*.json")):
        value = _read_canonical_model(path, CandidateTrace, errors)
        if value is not None:
            traces.append(value)
    events_path = root / "lineage" / "events.json"
    reduction_events: list[CandidateEvent] = []
    event_traces: list[CandidateEventTrace] = []
    try:
        if events_path.is_symlink() or not events_path.is_file():
            raise OSError("missing or symlinked event artifact")
        raw_events = json.loads(events_path.read_text(encoding="utf-8"))
        if (
            raw_events.get("schema_version")
            != SCHEMA_VERSION
            or raw_events.get("snapshot_digest")
            != discovery.repository.snapshot_digest
        ):
            raise ValueError("event artifact schema or snapshot mismatch")
        reduction_events = [
            CandidateEvent.model_validate(item)
            for item in raw_events["events"]
        ]
        event_traces = [
            CandidateEventTrace.model_validate(item)
            for item in raw_events["event_traces"]
        ]
        expected_events = canonical_json_text(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_digest": discovery.repository.snapshot_digest,
                "events": [
                    item.model_dump(mode="json") for item in reduction_events
                ],
                "event_traces": [
                    item.model_dump(mode="json") for item in event_traces
                ],
            }
        )
        if events_path.read_text(encoding="utf-8") != expected_events:
            errors.append("current-source lineage/events.json is not canonical")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        errors.append(f"current-source lineage/events.json error: {error}")
    propositions_path = root / "propositions.json"
    propositions: list[PropositionEnvelope] = []
    try:
        raw_propositions = json.loads(propositions_path.read_text(encoding="utf-8"))
        propositions = [
            PropositionEnvelope.model_validate(item)
            for item in raw_propositions["propositions"]
        ]
        expected = canonical_json_text(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_digest": discovery.repository.snapshot_digest,
                "propositions": [
                    item.model_dump(mode="json") for item in propositions
                ],
            }
        )
        if propositions_path.read_text(encoding="utf-8") != expected:
            errors.append("current-source/propositions.json is not canonical")
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as error:
        errors.append(f"current-source/propositions.json error: {error}")
    policy_path = root / "policy-decisions.json"
    policy_decisions: list[CurrentSourcePolicyDecision] = []
    try:
        raw_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if (
            raw_policy.get("schema_version") != SCHEMA_VERSION
            or raw_policy.get("snapshot_digest")
            != discovery.repository.snapshot_digest
            or raw_policy.get("policy_version")
            != CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION
        ):
            raise ValueError(
                "policy artifact schema, snapshot, or policy mismatch"
            )
        policy_decisions = [
            CurrentSourcePolicyDecision.model_validate(item)
            for item in raw_policy["decisions"]
        ]
        expected = canonical_json_text(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_digest": discovery.repository.snapshot_digest,
                "policy_version": CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
                "decisions": [
                    item.model_dump(mode="json")
                    for item in policy_decisions
                ],
            }
        )
        if policy_path.read_text(encoding="utf-8") != expected:
            errors.append(
                "current-source/policy-decisions.json is not canonical"
            )
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as error:
        errors.append(f"current-source/policy-decisions.json error: {error}")
    semantic_records = [
        item
        for item in inference_records
        if item.stage in _CURRENT_SOURCE_INFERENCE_STAGES
    ]
    prior_channel_records = [
        item
        for item in inference_records
        if item.stage in {"explicit_extract", "explicit_semantic_verify"}
    ]
    replay_client = _RecordedInferenceClient(semantic_records)
    replay_budget = InferenceBudget(
        llm_settings,
        cache=_RecordedInferenceCache(semantic_records),
        initial_calls=sum(item.usage.calls for item in prior_channel_records),
        initial_input_tokens=sum(
            item.usage.input_tokens for item in prior_channel_records
        ),
        initial_output_tokens=sum(
            item.usage.output_tokens for item in prior_channel_records
        ),
    )
    try:
        replayed = run_current_source_pipeline(
            repository_root=repository_root,
            discovery=discovery,
            observations=observations,
            semantic_units=semantic_units,
            semantic_settings=semantic_settings,
            llm_settings=llm_settings,
            client=replay_client,
            budget=replay_budget,
            protocol_manifest_digest=protocol_manifest_digest,
            explicit_context=explicit_context,
        )
    except (OSError, ValueError, TypeError, ValidationError) as error:
        errors.append(f"current-source replay failed: {error}")
        return errors, None
    if expected_stage_omissions is not None:
        expected_omissions = {
            stage: (count, dict(sorted(reasons.items())))
            for stage, (count, reasons) in sorted(
                expected_stage_omissions.items()
            )
        }
        if replay_budget.stage_omissions != expected_omissions:
            errors.append(
                "current-source inference omissions differ from recorded replay"
            )
    comparisons = (
        (
            "inference records",
            tuple(sorted(semantic_records, key=lambda item: item.request_id)),
            tuple(
                sorted(
                    replayed.inference_records,
                    key=lambda item: item.request_id,
                )
            ),
        ),
        ("index", index, replayed.index),
        ("semantic map index", map_index, replayed.semantic_map.index),
        (
            "semantic map results",
            tuple(sorted(map_results, key=lambda item: item.packet_id)),
            replayed.semantic_map.results,
        ),
        (
            "semantic observations",
            tuple(
                sorted(
                    semantic_observations,
                    key=lambda item: item.observation_id,
                )
            ),
            replayed.semantic_map.observations,
        ),
        ("reduction index", reduction_index, replayed.reduction.index),
        (
            "reduction results",
            tuple(sorted(reduction_results, key=lambda item: item.result_id)),
            tuple(
                sorted(
                    replayed.reduction.call_results,
                    key=lambda item: item.result_id,
                )
            ),
        ),
        (
            "hypotheses",
            tuple(sorted(hypotheses, key=lambda item: item.hypothesis_id)),
            replayed.hypotheses,
        ),
        (
            "populations",
            tuple(sorted(populations, key=lambda item: item.population_id)),
            replayed.populations,
        ),
        (
            "member packets",
            tuple(sorted(packets, key=lambda item: item.packet_id)),
            replayed.member_packets,
        ),
        (
            "falsifications",
            tuple(
                sorted(
                    falsifications,
                    key=lambda item: item.falsification_id,
                )
            ),
            replayed.falsifications,
        ),
        (
            "refinements",
            tuple(sorted(refinements, key=lambda item: item.refinement_id)),
            replayed.refinements,
        ),
        (
            "propositions",
            tuple(
                sorted(propositions, key=lambda item: item.proposition_id)
            ),
            replayed.propositions,
        ),
        (
            "policy decisions",
            tuple(
                sorted(
                    policy_decisions,
                    key=lambda item: item.candidate_root_id,
                )
            ),
            replayed.policy_decisions,
        ),
        (
            "lineage traces",
            tuple(sorted(traces, key=lambda item: item.trace_id)),
            replayed.traces,
        ),
        (
            "candidate events",
            tuple(sorted(reduction_events, key=lambda item: item.event_id)),
            tuple(
                sorted(
                    replayed.reduction.events,
                    key=lambda item: item.event_id,
                )
            ),
        ),
        (
            "candidate event traces",
            tuple(sorted(event_traces, key=lambda item: item.event_id)),
            replayed.event_traces,
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            errors.append(f"current-source {label} differs from recorded replay")
    expected_files = {
        "index.json",
        "propositions.json",
        "policy-decisions.json",
        "semantic-observations/index.json",
        "lineage/events.json",
        "hypotheses/index.json",
        *(
            f"semantic-observations/units/{item.result_id}.json"
            for item in map_index.results
        ),
        *(
            f"semantic-observations/observations/{item.observation_id}.json"
            for item in map_index.observations
        ),
        *(
            f"hypotheses/reduction-results/{item.result_id}.json"
            for item in replayed.reduction.call_results
        ),
        *(
            f"hypotheses/{item.hypothesis_id}.json"
            for item in replayed.hypotheses
        ),
        *(
            f"populations/{item.population_id}.json"
            for item in replayed.populations
        ),
        *(
            f"member-packets/{item.packet_id}.json"
            for item in replayed.member_packets
        ),
        *(
            f"falsifications/{item.falsification_id}.json"
            for item in replayed.falsifications
        ),
        *(
            f"refinements/{item.refinement_id}.json"
            for item in replayed.refinements
        ),
        *(f"lineage/{item.trace_id}.json" for item in replayed.traces),
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(
                f"current-source tree contains symlink: {path.relative_to(root)}"
            )
        elif path.is_dir():
            actual_directories.add(path.relative_to(root).as_posix())
        elif path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"missing current-source artifacts: {missing}")
        if extra:
            errors.append(f"unexpected current-source artifacts: {extra}")
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        if missing:
            errors.append(f"missing current-source directories: {missing}")
        if extra:
            errors.append(f"unexpected current-source directories: {extra}")
    return errors, replayed
