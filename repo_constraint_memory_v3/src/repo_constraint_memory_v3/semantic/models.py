"""Strict category-neutral models for current-source open semantic mining."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import (
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from repo_constraint_memory_v3.discovery import matches_glob
from repo_constraint_memory_v3.models import (
    NonEmptyString,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.temporal_need import GitNeedAssessment

POPULATION_SPEC_VERSION = "population-spec-1.2"
PopulationUniverse = Literal["file", "entity", "site", "relation"]
LiteralValue = str | bool


class UniverseSpec(StrictModel):
    op: Literal["universe"]
    universe: PopulationUniverse


class FilterPredicate(StrictModel):
    field: NonEmptyString
    operator: Literal["eq", "in", "path_under", "path_glob", "has"]
    value: LiteralValue | list[LiteralValue]

    @model_validator(mode="after")
    def operator_shape(self) -> FilterPredicate:
        if self.operator in {"in", "path_glob"}:
            if not isinstance(self.value, list) or not self.value:
                if self.operator == "in":
                    raise ValueError(
                        "membership predicate requires non-empty literal list"
                    )
            if self.operator == "path_glob":
                values = self.value if isinstance(self.value, list) else [self.value]
                if not values or any(not isinstance(item, str) for item in values):
                    raise ValueError("path_glob requires one or more string patterns")
                for item in values:
                    _validate_relative_glob(item)
        elif isinstance(self.value, list):
            raise ValueError(f"{self.operator} predicate requires one literal")
        if self.operator == "path_under":
            if not isinstance(self.value, str):
                raise ValueError("path_under requires a repository-relative path")
            _validate_relative_path(self.value)
        if self.operator == "has" and not isinstance(self.value, str):
            raise ValueError("has requires a string attribute or symbol")
        return self


class FilterSpec(StrictModel):
    op: Literal["filter"]
    source: "PopulationExpression"
    predicates: Annotated[list[FilterPredicate], Field(min_length=1)]


class TraverseSpec(StrictModel):
    op: Literal["traverse"]
    source: "PopulationExpression"
    traversal: Literal[
        "parent",
        "child",
        "ancestor",
        "descendant",
        "relation",
    ]
    relation_kind: (
        Literal[
            "contains",
            "defines",
            "imports",
            "imports_from",
            "references",
            "inherits",
            "calls",
        ]
        | None
    ) = None
    direction: Literal["outgoing", "incoming"] | None = None

    @model_validator(mode="after")
    def traversal_shape(self) -> TraverseSpec:
        if self.traversal == "relation":
            if self.relation_kind is None or self.direction is None:
                raise ValueError(
                    "relation traversal requires relation_kind and direction"
                )
        elif self.relation_kind is not None or self.direction is not None:
            raise ValueError("lexical traversal cannot carry relation settings")
        return self


class SetSpec(StrictModel):
    op: Literal["union", "intersection"]
    operands: Annotated[list["PopulationExpression"], Field(min_length=2)]


class DifferenceSpec(StrictModel):
    op: Literal["difference"]
    left: "PopulationExpression"
    right: "PopulationExpression"


PopulationExpression = Annotated[
    Union[UniverseSpec, FilterSpec, TraverseSpec, SetSpec, DifferenceSpec],
    Field(discriminator="op"),
]


class PopulationSpec(RootModel[PopulationExpression]):
    model_config = ConfigDict(strict=True)


FilterSpec.model_rebuild()
TraverseSpec.model_rebuild()
SetSpec.model_rebuild()
DifferenceSpec.model_rebuild()
PopulationSpec.model_rebuild()


class PopulationMemberRef(StrictModel):
    universe: PopulationUniverse
    observation_id: NonEmptyString
    file_id: str | None
    path: str | None


class PopulationResourceUse(StrictModel):
    query_nodes: int = Field(ge=1)
    query_depth: int = Field(ge=1)
    literal_values: int = Field(ge=0)
    scanned: int = Field(ge=0)
    serialized: int = Field(ge=0)
    artifact_bytes: int = Field(ge=0)


class PopulationResult(StrictModel):
    spec_version: Literal["population-spec-1.2"]
    spec: dict[str, object]
    spec_digest: Sha256Hex
    snapshot_digest: SnapshotDigest
    observation_set_digest: Sha256Hex
    status: Literal["complete", "partial", "failed"]
    source_universe: PopulationUniverse
    scanned_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    serialized_count: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    members: list[PopulationMemberRef]
    member_set_digest: Sha256Hex | None
    dependency_file_ids: list[NonEmptyString]
    dependency_paths: list[NonEmptyString]
    dependency_paths_omitted_count: int = Field(ge=0)
    resource_use: PopulationResourceUse
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    result_digest: Sha256Hex

    @model_validator(mode="after")
    def algebra(self) -> PopulationResult:
        if self.serialized_count != len(self.members):
            raise ValueError("population serialized count is inconsistent")
        if self.serialized_count + self.omitted_count != self.matched_count:
            raise ValueError("population member count algebra is inconsistent")
        if self.status == "complete":
            if (
                self.incomplete_reasons
                or self.omitted_count
                or self.dependency_file_ids
                or self.dependency_paths
                or self.dependency_paths_omitted_count
            ):
                raise ValueError("complete population result is incomplete")
            if self.member_set_digest is None:
                raise ValueError("complete population requires member-set digest")
        elif self.member_set_digest is not None:
            raise ValueError("partial/failed population cannot claim complete member set")
        if self.dependency_file_ids != sorted(set(self.dependency_file_ids)):
            raise ValueError("population dependency file IDs are not canonical")
        if self.dependency_paths != sorted(set(self.dependency_paths)):
            raise ValueError("population dependency paths are not canonical")
        for path in self.dependency_paths:
            _validate_relative_path(path)
        if self.result_digest != payload_digest(
            self.model_dump(mode="json", exclude={"result_digest"})
        ):
            raise ValueError("incorrect population-result digest")
        return self


def _validate_relative_path(value: str) -> None:
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or "://" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("path literal must be safe and repository-relative")


def _validate_relative_glob(value: str) -> None:
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or "://" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("path glob must be safe and repository-relative")
    # Compile through the exact deterministic matcher now. Otherwise a malformed
    # character range can escape much later as an uncaught ``re.error`` while an
    # untrusted LLM-authored population is being evaluated.
    matches_glob("__population_glob_validation__", value)


SemanticObservationId = Annotated[
    str, Field(pattern=r"^SEMOBS-[0-9a-f]{24}$")
]
HypothesisId = Annotated[str, Field(pattern=r"^HYP-[0-9a-f]{24}$")]
PopulationId = Annotated[str, Field(pattern=r"^POP-[0-9a-f]{24}$")]
PopulationMemberId = Annotated[
    str, Field(pattern=r"^POPMEMBER-[0-9a-f]{24}$")
]
FalsificationId = Annotated[
    str, Field(pattern=r"^FALSIFY-[0-9a-f]{24}$")
]
RefinementId = Annotated[str, Field(pattern=r"^REFINE-[0-9a-f]{24}$")]


def validate_member_classification_semantics(
    *,
    label: str,
    evidence_context: str,
    reason_code: str,
) -> None:
    """Reject contradictory classification axes chosen by the semantic LLM."""

    context_label = {
        "conforming_example": "support",
        "intentional_violation_test": "support",
        "irrelevant": "ineligible",
        "unknown": "unknown",
    }.get(evidence_context)
    if context_label is not None and label != context_label:
        raise ValueError(
            f"{evidence_context} evidence context requires {context_label} label"
        )
    if evidence_context == "allowed_exception" and label == "support":
        raise ValueError("allowed-exception context cannot be support")

    reserved_reason_label = {
        "supports": "support",
        "confirming_negative_test": "support",
        "conforming_example": "support",
        "contradicts": "counterexample",
        "allowed_exception": "counterexample",
        "not_applicable": "ineligible",
        "irrelevant_member": "ineligible",
        "ambiguous": "unknown",
        "insufficient_context": "unknown",
    }.get(reason_code)
    if reserved_reason_label is not None and label != reserved_reason_label:
        raise ValueError(
            f"{reason_code} reason code requires {reserved_reason_label} label"
        )


class SemanticObservation(StrictModel):
    observation_id: SemanticObservationId
    snapshot_digest: SnapshotDigest
    primary_unit_id: NonEmptyString
    segment_id: NonEmptyString
    packet_id: NonEmptyString
    category: Annotated[str, Field(min_length=1, max_length=120)]
    summary: Annotated[str, Field(min_length=1, max_length=600)]
    contributor_relevance: Literal[
        "actionable", "contextual", "none", "uncertain"
    ]
    evidence_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    related_unit_ids: list[NonEmptyString]
    uncertainty: Annotated[str, Field(max_length=400)] | None
    origin_inference_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @field_validator("category")
    @classmethod
    def category_is_single_line(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or "\n" in stripped or "\r" in stripped:
            raise ValueError("semantic observation category must be one line")
        return stripped

    @model_validator(mode="after")
    def canonical(self) -> SemanticObservation:
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("semantic observation evidence refs must be unique")
        if len(set(self.origin_inference_ids)) != len(self.origin_inference_ids):
            raise ValueError("semantic observation inference refs must be unique")
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect semantic-observation digest")
        return self


class SemanticMapPacketResult(StrictModel):
    packet_id: NonEmptyString
    primary_unit_id: NonEmptyString
    outcome: Literal[
        "observations",
        "no_observation",
        "uncertain",
        "failure",
    ]
    observation_ids: list[SemanticObservationId]
    inference_id: NonEmptyString | None
    reason_codes: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def shape(self) -> SemanticMapPacketResult:
        if self.outcome == "observations" and not self.observation_ids:
            raise ValueError("observation outcome requires observations")
        if self.outcome != "observations" and self.observation_ids:
            raise ValueError("non-observation outcome cannot carry observations")
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect semantic-map result digest")
        return self


class StructuralPredicate(StrictModel):
    eligible_spec: PopulationSpec
    support_spec: PopulationSpec


class SemanticPredicate(StrictModel):
    eligibility_question: Annotated[str, Field(min_length=1, max_length=800)]
    support_question: Annotated[str, Field(min_length=1, max_length=800)]
    allowed_reason_codes: Annotated[
        list[NonEmptyString], Field(min_length=1, max_length=64)
    ]
    expected_evidence_form: Annotated[str, Field(min_length=1, max_length=400)]
    ambiguity_instruction: Annotated[str, Field(min_length=1, max_length=400)]
    repository_specificity_test: Annotated[
        str, Field(min_length=1, max_length=400)
    ]
    protocol_version: Literal["semantic-member-classification-1.0"]


class IndependenceSpec(StrictModel):
    mode: Literal[
        "clone_aware_observation",
        "file",
        "module",
        "component",
        "primary_entity",
    ] = "clone_aware_observation"


class WitnessReferenceRepair(StrictModel):
    source_evidence_ref: NonEmptyString
    target_observation_ref: SemanticObservationId
    repair_code: Literal[
        "unique_evidence_range_to_observation",
        "shared_evidence_range_to_observation",
    ]


class NormHypothesis(StrictModel):
    hypothesis_id: HypothesisId
    candidate_root_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    version: int = Field(ge=0, le=2)
    parent_hypothesis_id: HypothesisId | None
    refinement_round: int = Field(ge=0, le=2)
    rule: Annotated[str, Field(min_length=1, max_length=500)]
    category: Annotated[str, Field(min_length=1, max_length=120)] | None
    proposed_claim_kind: Literal["observed_convention"]
    proposed_strength: Literal["observed", "recommended"]
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: Annotated[str, Field(min_length=1, max_length=500)] | None
    witness_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    witness_ref_repairs: list[WitnessReferenceRepair] = Field(
        default_factory=list
    )
    explicit_context_refs: list[NonEmptyString]
    superpopulation_spec: PopulationSpec
    predicate_mode: Literal["structural", "semantic"]
    structural_predicate: StructuralPredicate | None
    semantic_predicate: SemanticPredicate | None
    independence_spec: IndependenceSpec
    limitations: list[NonEmptyString]
    origin_inference_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    completeness_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    component_keys: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @field_validator("category")
    @classmethod
    def category_is_single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or "\n" in stripped or "\r" in stripped:
            raise ValueError("hypothesis category must be one line")
        return stripped

    @field_validator("applies_to")
    @classmethod
    def safe_scope(cls, values: list[str]) -> list[str]:
        for value in values:
            if (
                not value
                or value.startswith(("/", "\\"))
                or "\\" in value
                or "://" in value
                or ".." in value.split("/")
            ):
                raise ValueError("applies_to must be repository-relative")
            prefix = value.split("*", 1)[0].rstrip("/")
            if prefix:
                _validate_relative_path(prefix)
        return values

    @model_validator(mode="after")
    def shape_and_digest(self) -> NormHypothesis:
        if self.version != self.refinement_round:
            raise ValueError("hypothesis version and refinement round differ")
        if (self.version == 0) != (self.parent_hypothesis_id is None):
            raise ValueError("hypothesis parent/version relationship is invalid")
        if self.predicate_mode == "structural":
            if self.structural_predicate is None or self.semantic_predicate is not None:
                raise ValueError("structural hypothesis has invalid predicate shape")
        elif self.semantic_predicate is None or self.structural_predicate is not None:
            raise ValueError("semantic hypothesis has invalid predicate shape")
        repair_pairs = [
            (item.source_evidence_ref, item.target_observation_ref)
            for item in self.witness_ref_repairs
        ]
        if len(repair_pairs) != len(set(repair_pairs)):
            raise ValueError("duplicate witness-reference repair")
        if not {
            item.target_observation_ref for item in self.witness_ref_repairs
        } <= set(self.witness_refs):
            raise ValueError("witness-reference repair target is not a witness")
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect hypothesis digest")
        return self


class PopulationMemberPacket(StrictModel):
    packet_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    hypothesis_id: HypothesisId
    observation_ref: PopulationMemberRef
    owning_unit_id: str | None
    fact: dict[str, object]
    evidence_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    clone_group: str | None
    independence_group: NonEmptyString
    missing_context: list[NonEmptyString]
    packet_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> PopulationMemberPacket:
        if self.packet_digest != payload_digest(
            self.model_dump(mode="json", exclude={"packet_digest"})
        ):
            raise ValueError("incorrect population-member packet digest")
        return self


class ClassificationBatch(StrictModel):
    ordinal: int = Field(ge=0)
    member_ids: Annotated[list[PopulationMemberId], Field(min_length=1)]
    packet_digests: Annotated[list[Sha256Hex], Field(min_length=1)]
    inference_id: str | None
    request_bytes: int = Field(ge=1)
    transport_ordinal: int | None = Field(default=None, ge=0)
    pair_ordinals: list[int] = Field(default_factory=list)
    request_payload_digest: Sha256Hex | None = None

    @model_validator(mode="after")
    def aligned(self) -> ClassificationBatch:
        if len(self.member_ids) != len(self.packet_digests):
            raise ValueError("classification batch member/packet counts differ")
        if self.pair_ordinals and (
            len(self.pair_ordinals) != len(self.member_ids)
            or self.pair_ordinals != sorted(set(self.pair_ordinals))
        ):
            raise ValueError("classification transport pair assignment differs")
        return self


class BoundedStageExecution(StrictModel):
    """Compact denominator and terminal/omitted algebra for one LLM stage."""

    request_plan_complete: bool
    planned_calls: int = Field(ge=0)
    terminal_calls: int = Field(ge=0)
    omitted_calls: int = Field(ge=0)
    planned_items: int = Field(ge=0)
    terminal_items: int = Field(ge=0)
    omitted_items: int = Field(ge=0)
    closure_reason: NonEmptyString | None

    @model_validator(mode="after")
    def exact_algebra(self) -> BoundedStageExecution:
        if self.terminal_calls + self.omitted_calls != self.planned_calls:
            raise ValueError("bounded-stage call counts do not sum to planned")
        if self.terminal_items + self.omitted_items != self.planned_items:
            raise ValueError("bounded-stage item counts do not sum to planned")
        omitted = self.omitted_calls > 0 or self.omitted_items > 0
        if omitted != (self.closure_reason is not None):
            raise ValueError("bounded-stage closure reason differs from omissions")
        if not self.request_plan_complete and (
            self.planned_calls or self.terminal_calls or self.omitted_calls
        ):
            raise ValueError(
                "an incomplete request plan cannot claim a call denominator"
            )
        return self


class PopulationSamplingStratum(StrictModel):
    stratum_key: NonEmptyString
    matched: int = Field(ge=0)
    inspected: int = Field(ge=0)
    uninspected: int = Field(ge=0)

    @model_validator(mode="after")
    def exact_algebra(self) -> PopulationSamplingStratum:
        if self.inspected + self.uninspected != self.matched:
            raise ValueError("population sampling stratum counts do not sum")
        return self


class PopulationSamplingPlan(StrictModel):
    """Frozen population selection and feasible provider-planning boundary.

    Overall inspected/uninspected counts describe the deterministic population
    sample. Provider-inspected counts exclude selected members that become
    host-classified technical unknowns before request planning; later execution
    omissions remain represented by the classification execution denominator.
    """

    plan_version: Literal["population-sampling-plan-1.3"]
    hypothesis_id: HypothesisId
    population_id: PopulationId
    mode: Literal["census", "deterministic_stratified_sample"]
    census_enumeration_status: Literal["complete", "partial", "failed"]
    sampling_plan_status: Literal["complete", "failed"]
    seed_digest: Sha256Hex
    matched_count: int = Field(ge=0)
    inspected_count: int = Field(ge=0)
    uninspected_count: int = Field(ge=0)
    provider_eligible_count: int = Field(ge=0)
    provider_inspected_count: int = Field(ge=0)
    provider_uninspected_count: int = Field(ge=0)
    selected_member_ids: list[PopulationMemberId]
    required_inspected_member_ids: list[PopulationMemberId]
    uninspected_member_set_digest: Sha256Hex
    strata: list[PopulationSamplingStratum]
    plan_digest: Sha256Hex

    @model_validator(mode="after")
    def exact_algebra(self) -> PopulationSamplingPlan:
        if self.inspected_count + self.uninspected_count != self.matched_count:
            raise ValueError("population sampling counts do not sum")
        if self.provider_inspected_count + self.provider_uninspected_count != (
            self.provider_eligible_count
        ):
            raise ValueError("population provider sampling counts do not sum")
        if self.provider_inspected_count > self.inspected_count:
            raise ValueError("provider-inspected members exceed inspected members")
        if self.provider_eligible_count > self.matched_count:
            raise ValueError("provider-eligible members exceed matched members")
        if self.provider_uninspected_count < self.uninspected_count:
            raise ValueError(
                "provider-uninspected members omit population omissions"
            )
        if len(self.selected_member_ids) != self.inspected_count or (
            self.selected_member_ids != sorted(set(self.selected_member_ids))
        ):
            raise ValueError("population selected member IDs are not canonical")
        if self.required_inspected_member_ids != sorted(
            set(self.required_inspected_member_ids)
        ) or not set(self.required_inspected_member_ids) <= set(
            self.selected_member_ids
        ):
            raise ValueError(
                "required population inspection members are not canonical"
            )
        if self.mode == "census" and self.uninspected_count:
            raise ValueError("census sampling plan cannot omit members")
        if self.mode == "deterministic_stratified_sample" and not (
            self.uninspected_count
        ):
            raise ValueError("sampled population requires uninspected members")
        if [item.stratum_key for item in self.strata] != sorted(
            {item.stratum_key for item in self.strata}
        ):
            raise ValueError("population sampling strata are not canonical")
        if sum(item.matched for item in self.strata) != self.provider_eligible_count:
            raise ValueError("population sampling strata matched total differs")
        if sum(item.inspected for item in self.strata) != (
            self.provider_inspected_count
        ):
            raise ValueError("population sampling strata inspected total differs")
        if sum(item.uninspected for item in self.strata) != (
            self.provider_uninspected_count
        ):
            raise ValueError("population sampling strata omitted total differs")
        if self.plan_digest != payload_digest(
            self.model_dump(mode="json", exclude={"plan_digest"})
        ):
            raise ValueError("incorrect population sampling-plan digest")
        return self


class ClassificationPlan(StrictModel):
    plan_version: Literal["population-classification-plan-1.8"]
    hypothesis_id: HypothesisId
    population_id: PopulationId
    member_ids: list[PopulationMemberId]
    deterministic_member_ids: list[PopulationMemberId]
    unplanned_member_ids: list[PopulationMemberId]
    batches: list[ClassificationBatch]
    planned_calls: int = Field(ge=0)
    planned_bytes: int = Field(ge=0)
    planned_request_bytes: int = Field(ge=0)
    admission_estimate_calls: int = Field(ge=0)
    admission_estimate_request_bytes: int = Field(ge=0)
    execution: BoundedStageExecution
    plan_digest: Sha256Hex

    @model_validator(mode="after")
    def exact_assignment(self) -> ClassificationPlan:
        assigned = [item for batch in self.batches for item in batch.member_ids]
        classified = [
            *self.deterministic_member_ids,
            *assigned,
            *self.unplanned_member_ids,
        ]
        if (
            len(classified) != len(set(classified))
            or set(classified) != set(self.member_ids)
            or len(classified) != len(self.member_ids)
        ):
            raise ValueError("classification plan is not a one-to-one assignment")
        if not self.planned_calls and self.batches:
            raise ValueError("zero-call classification plan cannot carry batches")
        if self.planned_calls != len(self.batches):
            raise ValueError("classification planned call count differs")
        if bool(self.planned_calls) != bool(self.admission_estimate_calls) or (
            bool(self.admission_estimate_calls)
            != bool(self.admission_estimate_request_bytes)
        ):
            raise ValueError("classification admission estimate presence differs")
        if self.planned_request_bytes != sum(
            item.request_bytes for item in self.batches
        ):
            raise ValueError(
                "classification planned request bytes differ from batches"
            )
        if self.execution.planned_calls != self.planned_calls:
            raise ValueError("classification execution differs from request plan")
        if self.execution.planned_items != (
            len(assigned) + len(self.unplanned_member_ids)
        ):
            raise ValueError("classification execution item denominator differs")
        if self.execution.request_plan_complete and self.unplanned_member_ids:
            raise ValueError("complete classification request plan has unplanned members")
        if not self.execution.request_plan_complete and self.batches:
            raise ValueError("incomplete classification request plan carries batches")
        if self.plan_digest != payload_digest(
            self.model_dump(mode="json", exclude={"plan_digest"})
        ):
            raise ValueError("incorrect classification-plan digest")
        return self


class PopulationMember(StrictModel):
    member_id: PopulationMemberId
    hypothesis_id: HypothesisId
    population_id: PopulationId
    observation_ref: PopulationMemberRef
    label: Literal["support", "counterexample", "ineligible", "unknown"]
    evidence_context: Literal[
        "production",
        "conforming_example",
        "intentional_violation_test",
        "allowed_exception",
        "irrelevant",
        "unknown",
    ]
    classification_basis: Literal[
        "structural", "semantic", "technical_unknown"
    ]
    reason_code: NonEmptyString
    reason_summary: Annotated[str, Field(min_length=1, max_length=600)]
    evidence_refs: list[NonEmptyString]
    inference_ref: str | None
    independence_group: NonEmptyString
    clone_group: str | None
    limitation: str | None
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> PopulationMember:
        if self.classification_basis == "technical_unknown" and self.label != "unknown":
            raise ValueError("technical-unknown basis requires unknown label")
        validate_member_classification_semantics(
            label=self.label,
            evidence_context=self.evidence_context,
            reason_code=self.reason_code,
        )
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect population-member digest")
        return self


class PopulationArtifact(StrictModel):
    artifact_version: Literal["current-population-1.5"]
    population_id: PopulationId
    snapshot_digest: SnapshotDigest
    hypothesis_id: HypothesisId
    hypothesis_version: int = Field(ge=0, le=2)
    route: Literal[
        "current_structural_population", "current_semantic_population"
    ]
    status: Literal["complete", "partial", "failed"]
    population_result: PopulationResult
    spec_digest: Sha256Hex
    member_set_digest: Sha256Hex | None
    sampling_plan: PopulationSamplingPlan
    classification_plan: ClassificationPlan
    members: list[PopulationMember]
    support_count: int = Field(ge=0)
    counterexample_count: int = Field(ge=0)
    ineligible_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    semantic_unknown_count: int = Field(ge=0)
    technical_unknown_count: int = Field(ge=0)
    occurrence_support: int = Field(ge=0)
    independent_support: int = Field(ge=0)
    clone_aware_independent_support: int = Field(ge=0)
    distributions: dict[str, dict[str, int]]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def algebra_and_digest(self) -> PopulationArtifact:
        if self.sampling_plan.hypothesis_id != self.hypothesis_id or (
            self.sampling_plan.population_id != self.population_id
        ):
            raise ValueError("population sampling plan belongs to another population")
        if self.sampling_plan.matched_count != (
            self.population_result.serialized_count
        ):
            raise ValueError("population sampling census differs from population result")
        if len(self.members) != self.sampling_plan.inspected_count:
            raise ValueError("population members do not cover inspected population")
        if self.sampling_plan.selected_member_ids != sorted(
            item.member_id for item in self.members
        ):
            raise ValueError("population members differ from sampling selection")
        if self.classification_plan.member_ids != (
            self.sampling_plan.selected_member_ids
        ):
            raise ValueError("classification plan differs from sampling selection")
        provider_planned_ids = [
            item
            for batch in self.classification_plan.batches
            for item in batch.member_ids
        ] + list(self.classification_plan.unplanned_member_ids)
        if len(provider_planned_ids) != (
            self.sampling_plan.provider_inspected_count
        ):
            raise ValueError(
                "provider-inspected denominator differs from provider plan"
            )
        if (
            self.support_count
            + self.counterexample_count
            + self.ineligible_count
            + self.unknown_count
            != len(self.members)
        ):
            raise ValueError("population terminal label algebra is invalid")
        if self.semantic_unknown_count + self.technical_unknown_count != self.unknown_count:
            raise ValueError("population unknown algebra is invalid")
        if self.occurrence_support != self.support_count:
            raise ValueError("occurrence support differs from support count")
        if self.status == "complete" and (
            self.technical_unknown_count or self.incomplete_reasons
        ):
            raise ValueError("complete population cannot contain technical failures")
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect population artifact digest")
        return self


FALSIFICATION_RESULT_VERSION = "current-falsification-1.5"

FalsificationVerdict = Literal[
    "supported_as_written",
    "narrow_scope",
    "narrow_condition",
    "legitimate_exception",
    "alternative_non_norm_explanation",
    "missing_query",
    "reject_insufficient",
    "uncertain",
]

SemanticQualityAxis = Literal["pass", "fail", "uncertain"]


class FalsificationIssueInterpretation(StrictModel):
    member_id: PopulationMemberId
    interpretation: Literal[
        "true_counterexample",
        "confirming_negative_test",
        "legitimate_exception",
        "ineligible",
        "not_counterexample",
        "unresolved",
    ]
    rationale_summary: Annotated[str, Field(min_length=1, max_length=500)]


class FalsificationAdmissionEstimate(StrictModel):
    """Exact pre-inference cost estimate for a job denied admission."""

    status: Literal["not_admitted"]
    estimated_calls: int = Field(ge=1)
    estimated_request_bytes: int = Field(ge=1)
    reason_code: Literal[
        "max_falsification_calls_per_hypothesis",
        "max_falsification_request_bytes_per_hypothesis",
        "max_total_falsification_calls",
        "max_total_falsification_request_bytes",
    ]
    limit: int = Field(ge=0)


class FalsificationResult(StrictModel):
    result_version: Literal["current-falsification-1.5"]
    falsification_id: FalsificationId
    snapshot_digest: SnapshotDigest
    hypothesis_id: HypothesisId
    population_id: PopulationId
    population_sampling_mode: Literal[
        "census", "deterministic_stratified_sample"
    ]
    population_matched_count: int = Field(ge=0)
    population_inspected_count: int = Field(ge=0)
    population_uninspected_count: int = Field(ge=0)
    status: Literal["complete", "partial", "failed"]
    verdict: FalsificationVerdict
    git_need: GitNeedAssessment
    reviewed_member_ids: list[PopulationMemberId]
    inference_ids: list[NonEmptyString]
    evidence_refs: list[NonEmptyString]
    issue_interpretations: list[FalsificationIssueInterpretation]
    repository_specificity: SemanticQualityAxis
    contributor_value: SemanticQualityAxis
    scope_condition_fit: SemanticQualityAxis
    evidence_sufficiency: SemanticQualityAxis
    proposed_applies_to: list[NonEmptyString] | None
    proposed_condition: str | None
    exception_summary: str | None
    alternative_explanation: str | None
    proposed_population_spec: PopulationSpec | None
    unresolved_questions: list[NonEmptyString]
    rationale_summary: Annotated[str, Field(min_length=1, max_length=800)]
    incomplete_reasons: list[NonEmptyString]
    admission_estimate: FalsificationAdmissionEstimate | None
    execution: BoundedStageExecution
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> FalsificationResult:
        if self.population_inspected_count + self.population_uninspected_count != (
            self.population_matched_count
        ):
            raise ValueError("falsification population sampling counts do not sum")
        if self.status == "complete" and self.incomplete_reasons:
            raise ValueError("complete falsification cannot be incomplete")
        if self.status == "complete" and not self.evidence_refs:
            raise ValueError("complete falsification requires cited evidence")
        if self.admission_estimate is not None:
            execution = self.execution
            stable_reason = (
                "falsification_cost_not_admitted:"
                f"{self.admission_estimate.reason_code}"
            )
            detail_reason = (
                f"{stable_reason}:calls={self.admission_estimate.estimated_calls}:"
                "request_bytes="
                f"{self.admission_estimate.estimated_request_bytes}:"
                f"limit={self.admission_estimate.limit}"
            )
            if (
                self.status != "partial"
                or self.verdict != "uncertain"
                or self.inference_ids
                or self.reviewed_member_ids
                or self.evidence_refs
                or self.issue_interpretations
                or any(
                    item != "uncertain"
                    for item in (
                        self.repository_specificity,
                        self.contributor_value,
                        self.scope_condition_fit,
                        self.evidence_sufficiency,
                    )
                )
                or execution.request_plan_complete
                or execution.planned_calls
                or execution.terminal_calls
                or execution.omitted_calls
                or execution.terminal_items
                or execution.omitted_items != execution.planned_items
                or stable_reason not in self.incomplete_reasons
                or detail_reason not in self.incomplete_reasons
                or self.unresolved_questions != [detail_reason]
                or (
                    execution.planned_items > 0
                    and execution.closure_reason != stable_reason
                )
                or (
                    execution.planned_items == 0
                    and execution.closure_reason is not None
                )
            ):
                raise ValueError(
                    "a falsification admission estimate requires a partial "
                    "pre-inference result with no planned calls"
                )
        elif any(
            item.startswith("falsification_cost_not_admitted:")
            for item in [*self.incomplete_reasons, self.execution.closure_reason or ""]
        ):
            raise ValueError(
                "a falsification cost-omission reason requires an admission estimate"
            )
        interpreted = [item.member_id for item in self.issue_interpretations]
        if (
            len(interpreted) != len(set(interpreted))
            or set(interpreted) != set(self.reviewed_member_ids)
            or len(interpreted) != len(self.reviewed_member_ids)
        ):
            raise ValueError(
                "falsification issue interpretations must cover reviewed members"
            )
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect falsification-result digest")
        return self


class CarriedMemberDisposition(StrictModel):
    observation_id: NonEmptyString
    parent_label: Literal["support", "counterexample", "ineligible", "unknown"]
    child_label: Literal["support", "counterexample", "ineligible", "unknown"]
    reason_code: NonEmptyString


class RefinementRecord(StrictModel):
    refinement_id: RefinementId
    snapshot_digest: SnapshotDigest
    candidate_root_id: NonEmptyString
    parent_hypothesis_id: HypothesisId
    child_hypothesis_id: HypothesisId
    refinement_round: int = Field(ge=1, le=2)
    change_kind: Literal["scope", "condition", "exception", "superpopulation"]
    carried_member_dispositions: list[CarriedMemberDisposition]
    origin_falsification_refs: Annotated[list[FalsificationId], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> RefinementRecord:
        if self.parent_hypothesis_id == self.child_hypothesis_id:
            raise ValueError("refinement parent and child must differ")
        if self.artifact_digest != payload_digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect refinement-record digest")
        return self
