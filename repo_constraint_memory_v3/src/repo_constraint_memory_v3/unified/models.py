"""Strict, immutable contracts for the canonical unified evidence layer."""

from __future__ import annotations

from itertools import combinations
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from repo_constraint_memory_v3.temporal_need import GitNeedAssessment, GitNeedJudgment

from repo_constraint_memory_v3.models import (
    ClaimKind,
    ExecutionRisk,
    NonEmptyString,
    NormativeStrength,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.repository_paths import (
    is_canonical_repository_path,
    is_consumer_routable_repository_path,
    is_nonaccepted_directory_marker,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    make_assessment_id,
    make_bundle_id,
    make_canonical_claim_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
    make_finding_id,
    make_member_id,
    make_refinement_id,
    make_temporal_matrix_id,
)

UNIFIED_CONTRACT_VERSION = "unified-evidence-contract-1.6"
UNIFIED_POLICY_VERSION = "unified-canonical-policy-1.13"
MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS = 8_000
MAX_OTHER_EVIDENCE_DESCRIPTION_CHARS = 600

ProfileDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
AssessmentId = Annotated[str, Field(pattern=r"^ASSESS-[0-9a-f]{24}$")]
MemberId = Annotated[str, Field(pattern=r"^MEMBER-[0-9a-f]{24}$")]
CaseId = Annotated[str, Field(pattern=r"^CASE-[0-9a-f]{24}$")]
EvidenceId = Annotated[str, Field(pattern=r"^EVIDENCE-[0-9a-f]{24}$")]
EvidenceOriginId = Annotated[str, Field(pattern=r"^EVORIGIN-[0-9a-f]{24}$")]
CurrentBundleId = Annotated[str, Field(pattern=r"^CURREV-[0-9a-f]{24}$")]
TemporalMatrixId = Annotated[str, Field(pattern=r"^TEMPMATRIX-[0-9a-f]{24}$")]
FindingId = Annotated[str, Field(pattern=r"^FINDING-[0-9a-f]{24}$")]
RefinementId = Annotated[str, Field(pattern=r"^UREFINE-[0-9a-f]{24}$")]
CanonicalClaimId = Annotated[str, Field(pattern=r"^CLAIM-[0-9a-f]{24}$")]
UnifiedBundleId = Annotated[str, Field(pattern=r"^UNIFIED-[0-9a-f]{24}$")]

ChannelName = Literal["explicit", "implicit_current"]
NormalizedLocalStatus = Literal[
    "locally_supported", "locally_uncertain", "locally_rejected", "invalid"
]
CanonicalStatus = Literal["accepted", "review", "rejected"]
CanonicalDeliveryPolicy = Literal[
    "default_consider",
    "condition_matched_only",
    "reference_only",
    "not_applicable",
]
ReviewClass = Literal[
    "technical_or_coverage_gap",
    "conflict_or_currentness",
    "refinement_or_qualifier_needed",
    "negative_basis_insufficient",
    "channel_disagreement",
    "factual_uncertainty",
]
RejectionKind = Literal[
    "current_snapshot_contradiction",
    "outside_repository_coding_scope",
]
AxisFinding = Literal["pass", "fail", "uncertain", "not_applicable"]
CapabilityStatus = Literal[
    "complete",
    "bounded_no_match",
    "partial",
    "unavailable",
    "failed",
    "not_required",
]


class ImmutableArtifactRef(StrictModel):
    kind: Literal[
        "channel_assessment",
        "explicit_candidate",
        "explicit_evidence",
        "explicit_verification",
        "norm_hypothesis",
        "population",
        "population_member",
        "falsification",
        "channel_lineage",
        "observation",
        "semantic_unit",
        "inference_request",
        "git_manifest",
        "git_temporal_bundle",
        "git_change_slice",
        "temporal_matrix",
        "current_evidence_bundle",
        "case_finding",
        "refinement",
    ]
    stable_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    artifact_path: NonEmptyString
    artifact_digest: Sha256Hex

    @field_validator("artifact_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if (
            value.startswith(("/", "\\"))
            or "\\" in value
            or "://" in value
            or ".." in value.split("/")
        ):
            raise ValueError("artifact path must be safe and repository-relative")
        return value


class CandidateSemantics(StrictModel):
    rule: Annotated[str, Field(min_length=1, max_length=500)]
    normalized_action: Annotated[str, Field(min_length=1, max_length=500)]
    polarity: Literal["positive", "negative"]
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: Annotated[str, Field(min_length=1, max_length=500)] | None
    exception: Annotated[str, Field(min_length=1, max_length=500)] | None

    @field_validator("applies_to")
    @classmethod
    def canonical_safe_scope(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("candidate scope must be sorted and unique")
        for value in values:
            if (
                value.startswith(("/", "\\"))
                or "\\" in value
                or "://" in value
                or ".." in value.split("/")
            ):
                raise ValueError("candidate scope must be repository-relative")
        return values


class ChannelAssessment(StrictModel):
    contract_version: Literal["unified-evidence-contract-1.6"] = (
        UNIFIED_CONTRACT_VERSION
    )
    assessment_id: AssessmentId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    channel: ChannelName
    candidate_root_id: NonEmptyString
    candidate_version: Annotated[int, Field(ge=0, le=2)]
    parent_candidate_id: str | None
    semantics: CandidateSemantics
    proposed_claim_kind: ClaimKind
    proposed_strength: NormativeStrength
    raw_channel_status: NonEmptyString
    raw_policy_version: NonEmptyString
    normalized_status: NormalizedLocalStatus
    axis_findings: dict[NonEmptyString, AxisFinding]
    typed_reasons: list[NonEmptyString]
    support_refs: list[ImmutableArtifactRef]
    counterevidence_refs: list[ImmutableArtifactRef]
    coverage_refs: list[ImmutableArtifactRef]
    lineage_refs: list[ImmutableArtifactRef]
    extension_refs: list[ImmutableArtifactRef]
    integrity: Literal["valid", "invalid"]
    source_authority: Literal[
        "direct_normative",
        "machine_enforced",
        "descriptive_typed",
        "current_population",
        "weak_or_unknown",
    ]
    git_need: GitNeedAssessment
    provenance: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_snapshot_and_digest(self) -> ChannelAssessment:
        expected = make_assessment_id(
            self.snapshot_digest,
            self.profile_digest,
            self.channel,
            self.candidate_root_id,
            self.candidate_version,
        )
        if self.assessment_id != expected:
            raise ValueError("incorrect channel assessment ID")
        refs = [
            *self.support_refs,
            *self.counterevidence_refs,
            *self.coverage_refs,
            *self.lineage_refs,
            *self.extension_refs,
        ]
        if any(item.snapshot_digest != self.snapshot_digest for item in refs):
            raise ValueError("channel assessment reference snapshot mismatch")
        if self.integrity == "invalid" and self.normalized_status != "invalid":
            raise ValueError("invalid assessment integrity must be typed invalid")
        if self.typed_reasons != sorted(set(self.typed_reasons)):
            raise ValueError("assessment reasons must be sorted and unique")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect channel assessment digest")
        return self


class CandidateCaseMember(StrictModel):
    member_id: MemberId
    assessment_id: AssessmentId
    snapshot_digest: SnapshotDigest
    channel: ChannelName
    semantics: CandidateSemantics
    assessment_ref: ImmutableArtifactRef

    @model_validator(mode="after")
    def identity_contract(self) -> CandidateCaseMember:
        if self.member_id != make_member_id(
            self.snapshot_digest, self.assessment_id
        ):
            raise ValueError("incorrect case member ID")
        if self.assessment_ref.stable_id != self.assessment_id:
            raise ValueError("case member assessment ref mismatch")
        if self.assessment_ref.snapshot_digest != self.snapshot_digest:
            raise ValueError("case member snapshot mismatch")
        return self


class CandidateMemberRelation(StrictModel):
    left_member_id: MemberId
    right_member_id: MemberId
    relation: Literal[
        "equivalent",
        "related_variant",
        "distinct",
        "uncertain",
    ]
    source: Literal["exact_identity", "semantic_grouping", "host_distinct"]
    reason_code: NonEmptyString

    @model_validator(mode="after")
    def canonical_pair(self) -> CandidateMemberRelation:
        if self.left_member_id >= self.right_member_id:
            raise ValueError("member relation pair must be canonical")
        return self


class CandidateCase(StrictModel):
    case_id: CaseId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    members: Annotated[list[CandidateCaseMember], Field(min_length=1)]
    relations: list[CandidateMemberRelation]
    grouping_status: Literal["complete", "incomplete"]
    grouping_reason_codes: list[NonEmptyString]
    exact_identity_groups: list[list[MemberId]]
    required_capabilities: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def membership_matrix_and_digest(self) -> CandidateCase:
        member_ids = [item.member_id for item in self.members]
        if member_ids != sorted(member_ids) or len(member_ids) != len(set(member_ids)):
            raise ValueError("case members must be uniquely sorted")
        if any(item.snapshot_digest != self.snapshot_digest for item in self.members):
            raise ValueError("case member snapshot mismatch")
        if self.case_id != make_case_id(self.snapshot_digest, member_ids):
            raise ValueError("incorrect CandidateCase ID")
        expected_pairs = set(combinations(member_ids, 2))
        actual_pairs = {
            (item.left_member_id, item.right_member_id) for item in self.relations
        }
        if len(actual_pairs) != len(self.relations) or not actual_pairs <= expected_pairs:
            raise ValueError("case relation matrix is duplicated or references a foreign pair")
        if self.grouping_status == "complete" and actual_pairs != expected_pairs:
            raise ValueError("complete case relation matrix is incomplete")
        grouped = [item for group in self.exact_identity_groups for item in group]
        if len(grouped) != len(set(grouped)) or not set(grouped) <= set(member_ids):
            raise ValueError("exact identity groups overlap or reference foreign member")
        if self.grouping_reason_codes != sorted(set(self.grouping_reason_codes)):
            raise ValueError("case grouping reasons must be canonical")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect CandidateCase digest")
        return self


class EvidenceOrigin(StrictModel):
    origin_id: EvidenceOriginId
    origin_kind: Literal[
        "explicit_span",
        "configuration_atom",
        "command_atom",
        "workflow_atom",
        "observation",
        "population_member",
        "git_change_slice",
    ]
    path: NonEmptyString
    line_start: Annotated[int, Field(ge=1)] | None
    line_end: Annotated[int, Field(ge=1)] | None
    source_content_digest: Sha256Hex
    independence_group: NonEmptyString

    @model_validator(mode="after")
    def line_contract(self) -> EvidenceOrigin:
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("evidence origin line range must be complete")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("evidence origin line range is reversed")
        return self


class CurrentEvidenceItem(StrictModel):
    evidence_id: EvidenceId
    origin_id: EvidenceOriginId
    role: Literal["support", "counterevidence", "unknown", "ineligible", "context"]
    member_ids: Annotated[list[MemberId], Field(min_length=1)]
    artifact_ref: ImmutableArtifactRef
    excerpt_digest: Sha256Hex
    description: Annotated[
        str,
        Field(min_length=1, max_length=MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS),
    ]
    anchor_literals: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        default_factory=list
    )
    anchor_symbols: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list
    )

    @field_validator("member_ids", "anchor_literals", "anchor_symbols")
    @classmethod
    def canonical_members(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("current evidence list must be canonical")
        if any("\n" in item or "\r" in item or "\x00" in item for item in value):
            raise ValueError("current evidence anchors must be bounded single lines")
        return value


class PopulationDenominator(StrictModel):
    member_id: MemberId
    population_id: NonEmptyString
    status: Literal["complete", "partial", "failed", "not_applicable"]
    census_enumeration_status: Literal[
        "complete", "partial", "failed", "not_applicable"
    ]
    sampling_plan_status: Literal[
        "complete", "failed", "not_applicable"
    ]
    sampling_mode: Literal[
        "census", "deterministic_stratified_sample", "not_applicable"
    ]
    scanned: Annotated[int, Field(ge=0)]
    serialized: Annotated[int, Field(ge=0)]
    matched: Annotated[int, Field(ge=0)]
    inspected: Annotated[int, Field(ge=0)]
    uninspected: Annotated[int, Field(ge=0)]
    support: Annotated[int, Field(ge=0)]
    counterexample: Annotated[int, Field(ge=0)]
    unknown: Annotated[int, Field(ge=0)]
    ineligible: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def algebra(self) -> PopulationDenominator:
        if self.support + self.counterexample + self.unknown + self.ineligible != (
            self.inspected
        ):
            raise ValueError("population denominator labels do not sum")
        if self.inspected + self.uninspected != self.matched:
            raise ValueError("population inspection denominator does not sum")
        if self.matched != self.serialized:
            raise ValueError("population matched/serialized denominator differs")
        if self.serialized > self.scanned:
            raise ValueError("serialized population exceeds scanned denominator")
        if self.sampling_mode == "census" and self.uninspected:
            raise ValueError("census denominator cannot have uninspected members")
        if self.sampling_mode == "deterministic_stratified_sample" and not (
            self.uninspected
        ):
            raise ValueError("sampled denominator requires uninspected members")
        not_applicable = self.status == "not_applicable"
        if not_applicable != (
            self.census_enumeration_status == "not_applicable"
            and self.sampling_plan_status == "not_applicable"
            and self.sampling_mode == "not_applicable"
        ):
            raise ValueError("population denominator applicability differs")
        return self


class CapabilityDependency(StrictModel):
    capability: NonEmptyString
    requirement_mode: Literal["required", "augmentative", "not_required"]
    status: CapabilityStatus
    reason_codes: list[NonEmptyString]

    @model_validator(mode="after")
    def status_contract(self) -> CapabilityDependency:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("capability dependency reasons must be canonical")
        if self.status in {"complete", "bounded_no_match", "not_required"} and (
            self.reason_codes
        ):
            raise ValueError("successful capability dependency cannot carry gaps")
        if self.status in {"partial", "unavailable", "failed"} and not (
            self.reason_codes
        ):
            raise ValueError("failed capability dependency requires a typed reason")
        return self


class CurrentEvidenceBundle(StrictModel):
    bundle_id: CurrentBundleId
    case_id: CaseId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    origins: list[EvidenceOrigin]
    items: list[CurrentEvidenceItem]
    denominators: list[PopulationDenominator]
    population_refs: list[ImmutableArtifactRef]
    falsification_refs: list[ImmutableArtifactRef]
    scope_condition_refs: list[ImmutableArtifactRef]
    dependencies: Annotated[list[CapabilityDependency], Field(min_length=1)]
    integrity: Literal["valid", "invalid"]
    status: Literal["complete", "partial", "failed"]
    incomplete_reasons: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def references_and_digest(self) -> CurrentEvidenceBundle:
        if self.bundle_id != make_current_evidence_bundle_id(
            self.snapshot_digest, self.profile_digest, self.case_id
        ):
            raise ValueError("incorrect current-evidence bundle ID")
        origin_ids = [item.origin_id for item in self.origins]
        evidence_ids = [item.evidence_id for item in self.items]
        if origin_ids != sorted(set(origin_ids)):
            raise ValueError("current evidence origins must be canonical")
        if evidence_ids != sorted(set(evidence_ids)):
            raise ValueError("current evidence items must be canonical")
        origin_by_id = {item.origin_id: item for item in self.origins}
        if not {item.origin_id for item in self.items} <= set(origin_by_id):
            raise ValueError("current evidence references a foreign origin")
        for origin in self.origins:
            span = (
                None
                if origin.line_start is None
                else [origin.line_start, origin.line_end]
            )
            if origin.origin_id != make_evidence_origin_id(
                self.snapshot_digest,
                origin.origin_kind,
                origin.path,
                span,
                origin.source_content_digest,
            ):
                raise ValueError("incorrect current evidence origin ID")
        for item in self.items:
            if item.evidence_id != make_evidence_id(
                item.origin_id,
                item.role,
                item.member_ids,
                item.artifact_ref.stable_id,
            ):
                raise ValueError("incorrect current evidence item ID")
            if item.artifact_ref.snapshot_digest != self.snapshot_digest:
                raise ValueError("current evidence item snapshot mismatch")
            origin = origin_by_id[item.origin_id]
            if (
                origin.origin_kind
                not in {
                    "explicit_span",
                    "configuration_atom",
                    "command_atom",
                    "workflow_atom",
                }
                and len(item.description) > MAX_OTHER_EVIDENCE_DESCRIPTION_CHARS
            ):
                raise ValueError("non-explicit current evidence description is too long")
        denominator_ids = [item.member_id for item in self.denominators]
        if denominator_ids != sorted(set(denominator_ids)):
            raise ValueError("population denominators must be canonical")
        artifact_refs = [
            *self.population_refs,
            *self.falsification_refs,
            *self.scope_condition_refs,
        ]
        if any(item.snapshot_digest != self.snapshot_digest for item in artifact_refs):
            raise ValueError("current evidence artifact snapshot mismatch")
        for refs in (
            self.population_refs,
            self.falsification_refs,
            self.scope_condition_refs,
        ):
            keys = [
                (
                    item.kind,
                    item.stable_id,
                    item.snapshot_digest,
                    item.artifact_path,
                    item.artifact_digest,
                )
                for item in refs
            ]
            if keys != sorted(set(keys)):
                raise ValueError("current evidence refs must be canonical")
        if any(item.kind != "population" for item in self.population_refs):
            raise ValueError("population refs must point to population artifacts")
        if any(item.kind != "falsification" for item in self.falsification_refs):
            raise ValueError("falsification refs must point to falsification artifacts")
        if any(
            item.kind not in {"explicit_evidence", "falsification"}
            for item in self.scope_condition_refs
        ):
            raise ValueError("scope/condition refs have an unsupported artifact kind")
        if self.integrity == "invalid" and self.status != "failed":
            raise ValueError("invalid current evidence must fail")
        dependency_names = [item.capability for item in self.dependencies]
        if dependency_names != sorted(set(dependency_names)):
            raise ValueError("current evidence dependencies must be canonical")
        if self.incomplete_reasons != sorted(set(self.incomplete_reasons)):
            raise ValueError("current evidence reasons must be canonical")
        if self.status == "complete" and self.incomplete_reasons:
            raise ValueError("complete current evidence cannot carry gaps")
        if self.status in {"partial", "failed"} and not self.incomplete_reasons:
            raise ValueError("incomplete current evidence requires a typed reason")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect current-evidence bundle digest")
        return self


class TemporalEvidenceRef(StrictModel):
    item_id: NonEmptyString
    commit_id: NonEmptyString
    parent_id: str | None
    artifact_ref: ImmutableArtifactRef


class TemporalMemberRelation(StrictModel):
    item_id: NonEmptyString
    member_id: MemberId
    relation: Literal[
        "supports_as_written",
        "contradicts_as_written",
        "qualifies_scope_or_condition",
        "retires_or_replaces",
        "context_only",
        "uncertain",
    ]
    affected_facets: list[Literal["rule", "polarity", "scope", "condition", "currentness"]]
    evidence_span: NonEmptyString | None
    reason_code: NonEmptyString
    uncertainty_reason: str | None


class TemporalRelationMatrix(StrictModel):
    matrix_id: TemporalMatrixId
    case_id: CaseId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    git_manifest_ref: ImmutableArtifactRef | None = None
    item_ids: list[NonEmptyString]
    member_ids: list[MemberId]
    relations: list[TemporalMemberRelation]
    status: Literal["complete", "partial", "failed", "not_required"]
    incomplete_reasons: list[NonEmptyString]
    inference_refs: list[ImmutableArtifactRef]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def complete_matrix_and_digest(self) -> TemporalRelationMatrix:
        if self.status == "not_required" and self.git_manifest_ref is not None:
            raise ValueError("not-required temporal matrix cannot carry a Git manifest")
        if self.status != "not_required" and (
            self.git_manifest_ref is None
            or self.git_manifest_ref.kind != "git_manifest"
            or self.git_manifest_ref.snapshot_digest != self.snapshot_digest
        ):
            raise ValueError("temporal matrix requires its snapshot Git manifest")
        if self.item_ids != sorted(set(self.item_ids)) or self.member_ids != sorted(
            set(self.member_ids)
        ):
            raise ValueError("temporal matrix axes must be canonical")
        expected = {(item, member) for item in self.item_ids for member in self.member_ids}
        actual = {(item.item_id, item.member_id) for item in self.relations}
        if len(actual) != len(self.relations) or (
            self.status == "complete" and actual != expected
        ):
            raise ValueError("temporal item/member relation matrix is incomplete")
        if not actual <= expected:
            raise ValueError("temporal relation matrix contains a foreign cell")
        if self.status == "complete" and self.item_ids and not self.inference_refs:
            raise ValueError("complete temporal matrix requires inference provenance")
        if self.matrix_id != make_temporal_matrix_id(
            self.snapshot_digest,
            self.profile_digest,
            self.case_id,
            self.item_ids,
            self.member_ids,
        ):
            raise ValueError("incorrect temporal matrix ID")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect temporal matrix digest")
        return self


class UnifiedMemberFinding(StrictModel):
    member_id: MemberId
    norm_relevance: AxisFinding
    rule_support: AxisFinding
    scope_support: AxisFinding
    condition_support: AxisFinding
    temporal_status: Literal[
        "current",
        "supported",
        "adverse_resolved",
        "adverse_unresolved",
        "uncertain",
        "not_available",
        "not_required",
    ]
    evidence_sufficiency: AxisFinding
    evidence_refs: list[NonEmptyString]
    closed_local_reason_codes: list[NonEmptyString]
    reason_codes: list[NonEmptyString]


class UnifiedMemberRelation(StrictModel):
    left_member_id: MemberId
    right_member_id: MemberId
    relation: Literal[
        "equivalent",
        "compatible",
        "conflicting",
        "distinct",
        "uncertain",
    ]
    evidence_refs: list[NonEmptyString]
    reason_code: NonEmptyString

    @model_validator(mode="after")
    def canonical_pair(self) -> UnifiedMemberRelation:
        if self.left_member_id >= self.right_member_id:
            raise ValueError("unified member relation pair must be canonical")
        return self


class SupportedConclusion(StrictModel):
    conclusion_id: NonEmptyString
    base_member_ids: Annotated[list[MemberId], Field(min_length=1)]
    relation: Literal["equivalent", "compatible", "conflicting", "distinct"]
    evidence_refs: Annotated[list[NonEmptyString], Field(min_length=1)]


class RefinementProposal(StrictModel):
    git_need: GitNeedJudgment
    parent_member_id: MemberId
    semantics: CandidateSemantics
    evidence_refs: Annotated[list[NonEmptyString], Field(min_length=1)]
    change_kinds: Annotated[
        list[Literal["wording", "scope", "condition", "exception", "split"]],
        Field(min_length=1),
    ]


class UnifiedCaseFinding(StrictModel):
    finding_id: FindingId
    case_id: CaseId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    verification_round: Annotated[int, Field(ge=0, le=1)]
    member_findings: Annotated[list[UnifiedMemberFinding], Field(min_length=1)]
    member_relations: list[UnifiedMemberRelation]
    unresolved_conflicts: list[NonEmptyString]
    conclusions: list[SupportedConclusion]
    refinement_proposals: list[RefinementProposal]
    inference_ref: ImmutableArtifactRef | None
    status: Literal["complete", "failed", "not_required"]
    failure_reasons: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_and_digest(self) -> UnifiedCaseFinding:
        if self.finding_id != make_finding_id(
            self.snapshot_digest,
            self.profile_digest,
            self.case_id,
            self.verification_round,
        ):
            raise ValueError("incorrect unified finding ID")
        member_ids = [item.member_id for item in self.member_findings]
        if member_ids != sorted(set(member_ids)):
            raise ValueError("unified finding members must be canonical")
        for item in self.member_findings:
            for values in (
                item.evidence_refs,
                item.closed_local_reason_codes,
                item.reason_codes,
            ):
                if values != sorted(set(values)):
                    raise ValueError("unified member finding refs must be canonical")
        expected_pairs = set(combinations(member_ids, 2))
        relation_pairs = [
            (item.left_member_id, item.right_member_id)
            for item in self.member_relations
        ]
        if relation_pairs != sorted(set(relation_pairs)):
            raise ValueError("unified finding relations must be canonical")
        if not set(relation_pairs) <= expected_pairs:
            raise ValueError("unified finding relation references a foreign member")
        if self.status == "complete" and set(relation_pairs) != expected_pairs:
            raise ValueError("complete unified finding relation matrix is incomplete")
        for item in self.member_relations:
            if item.evidence_refs != sorted(set(item.evidence_refs)):
                raise ValueError("unified relation refs must be canonical")
        for item in self.conclusions:
            if (
                item.base_member_ids != sorted(set(item.base_member_ids))
                or not set(item.base_member_ids) <= set(member_ids)
                or item.evidence_refs != sorted(set(item.evidence_refs))
            ):
                raise ValueError("unified conclusion refs must be canonical")
        conclusion_ids = [item.conclusion_id for item in self.conclusions]
        if conclusion_ids != sorted(set(conclusion_ids)):
            raise ValueError("unified conclusions must be canonical")
        if len(self.refinement_proposals) > 2:
            raise ValueError("unified finding refinement child limit exceeded")
        if self.refinement_proposals and len(
            {item.parent_member_id for item in self.refinement_proposals}
        ) != 1:
            raise ValueError("unified finding cannot refine multiple parents")
        for item in self.refinement_proposals:
            if (
                item.parent_member_id not in set(member_ids)
                or item.evidence_refs != sorted(set(item.evidence_refs))
                or item.change_kinds != sorted(set(item.change_kinds))
            ):
                raise ValueError("unified refinement proposal is not canonical")
        if self.verification_round == 1 and self.refinement_proposals:
            raise ValueError("round-one finding cannot propose another refinement")
        if self.unresolved_conflicts != sorted(set(self.unresolved_conflicts)):
            raise ValueError("unified conflicts must be canonical")
        if self.failure_reasons != sorted(set(self.failure_reasons)):
            raise ValueError("unified finding failures must be canonical")
        if self.status == "complete" and self.failure_reasons:
            raise ValueError("complete unified finding cannot carry failure")
        if self.status == "complete" and self.inference_ref is None:
            raise ValueError("complete unified finding requires inference provenance")
        if self.status == "failed" and not self.failure_reasons:
            raise ValueError("failed unified finding requires a typed reason")
        if self.inference_ref is not None and (
            self.inference_ref.kind != "inference_request"
            or self.inference_ref.snapshot_digest != self.snapshot_digest
        ):
            raise ValueError("unified finding inference ref is invalid")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect unified finding digest")
        return self


class RefinementRecord(StrictModel):
    refinement_id: RefinementId
    case_id: CaseId
    snapshot_digest: SnapshotDigest
    parent_member_id: MemberId
    child_member_ids: Annotated[list[MemberId], Field(min_length=1, max_length=2)]
    round: Literal[1]
    change_kinds: Annotated[list[NonEmptyString], Field(min_length=1)]
    reacquired_current_bundle_ids: Annotated[list[CurrentBundleId], Field(min_length=1)]
    git_bundle_refs: list[ImmutableArtifactRef]
    finding_ids: Annotated[list[FindingId], Field(min_length=1)]
    status: Literal["complete", "review"]
    reason_codes: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def lineage_and_digest(self) -> RefinementRecord:
        if self.refinement_id != make_refinement_id(
            self.snapshot_digest,
            self.case_id,
            self.parent_member_id,
            self.child_member_ids,
        ):
            raise ValueError("incorrect unified refinement ID")
        if len(set(self.child_member_ids)) != len(self.child_member_ids):
            raise ValueError("refinement children must be unique")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect unified refinement digest")
        return self


class MemberDisposition(StrictModel):
    member_id: MemberId
    case_id: CaseId
    disposition: Literal[
        "adopted_as_support",
        "superseded_by_refinement",
        "retained_as_separate",
        "review",
        "rejected",
        "invalid",
    ]
    canonical_claim_ids: list[CanonicalClaimId]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]

    @model_validator(mode="after")
    def canonical_references(self) -> MemberDisposition:
        if self.canonical_claim_ids != sorted(set(self.canonical_claim_ids)):
            raise ValueError("member disposition claim refs must be canonical")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("member disposition reasons must be canonical")
        return self


class CanonicalEvidenceRef(StrictModel):
    evidence_id: NonEmptyString
    origin_id: NonEmptyString
    role: Literal["support", "counterevidence"]
    kind: Literal["explicit", "current_observation", "temporal"]
    path: NonEmptyString
    line_start: Annotated[int, Field(ge=1)] | None
    line_end: Annotated[int, Field(ge=1)] | None
    artifact_ref: ImmutableArtifactRef

    @field_validator("path")
    @classmethod
    def canonical_evidence_path(cls, value: str) -> str:
        if not is_canonical_repository_path(value):
            raise ValueError("canonical evidence path must be repository-relative")
        return value


class CanonicalClaim(StrictModel):
    claim_id: CanonicalClaimId
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    case_id: CaseId
    semantics: CandidateSemantics
    epistemic_status: CanonicalStatus
    claim_kind: ClaimKind
    normative_strength: NormativeStrength
    delivery_policy: CanonicalDeliveryPolicy
    execution_risk: ExecutionRisk
    support: list[CanonicalEvidenceRef]
    counterevidence: list[CanonicalEvidenceRef]
    conflicts_with: list[CanonicalClaimId]
    contributing_member_ids: Annotated[list[MemberId], Field(min_length=1)]
    refinement_parent_id: str | None
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    review_facets: list[ReviewClass] = Field(default_factory=list)
    rejection_kind: RejectionKind | None = None
    rejection_basis_refs: list[NonEmptyString] = Field(default_factory=list)
    dependency_status: Literal["independent", "complete", "incomplete"]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_and_delivery(self) -> CanonicalClaim:
        if self.claim_id != make_canonical_claim_id(
            self.snapshot_digest, self.semantics.model_dump(mode="json")
        ):
            raise ValueError("incorrect canonical claim ID")
        if self.epistemic_status == "accepted" and not self.support:
            raise ValueError("accepted canonical claim requires support evidence")
        unsafe_scope = [
            value
            for value in self.semantics.applies_to
            if not (
                is_consumer_routable_repository_path(value)
                or is_nonaccepted_directory_marker(value)
            )
        ]
        if unsafe_scope:
            raise ValueError("canonical claim contains an unsafe scope")
        directory_markers = [
            value
            for value in self.semantics.applies_to
            if is_nonaccepted_directory_marker(value)
        ]
        if directory_markers and not (
            self.epistemic_status == "review"
            and "refinement_or_qualifier_needed" in self.review_facets
        ):
            raise ValueError(
                "directory-marker scope requires a scope-failed review claim"
            )
        if self.epistemic_status == "accepted":
            if self.delivery_policy == "not_applicable":
                raise ValueError("accepted canonical claim requires a delivery policy")
            if self.dependency_status == "incomplete":
                raise ValueError("accepted canonical claim requires complete dependencies")
        elif self.delivery_policy != "not_applicable":
            raise ValueError(
                "review/rejected canonical claim delivery policy must be not_applicable"
            )
        ordered_facets = sorted(set(self.review_facets))
        if self.review_facets != ordered_facets:
            raise ValueError("canonical review facets must be lexicographically canonical")
        if self.rejection_basis_refs != sorted(set(self.rejection_basis_refs)):
            raise ValueError("canonical rejection basis refs must be canonical")
        if self.epistemic_status == "accepted":
            if (
                self.review_facets
                or self.rejection_kind is not None
                or self.rejection_basis_refs
            ):
                raise ValueError("accepted canonical claim cannot carry review/rejection audit")
        elif self.epistemic_status == "review":
            if not self.review_facets:
                raise ValueError("review canonical claim requires review facets")
            if self.rejection_kind is not None or self.rejection_basis_refs:
                raise ValueError("review canonical claim cannot carry rejection audit")
        else:
            if self.review_facets:
                raise ValueError("rejected canonical claim cannot carry review audit")
            if self.rejection_kind is None or not self.rejection_basis_refs:
                raise ValueError("rejected canonical claim requires kind and basis refs")
            if self.rejection_kind == "current_snapshot_contradiction":
                permitted_basis_ids = {
                    item.evidence_id
                    for item in self.counterevidence
                    if item.kind != "temporal"
                }
                basis_label = "current counterevidence"
            else:
                # A candidate can be faithfully entailed by current source and
                # still fall outside Norm4Code's repository-coding knowledge
                # boundary.  In that narrow case the current support span is
                # the positive basis for what was classified as out of scope.
                permitted_basis_ids = {
                    item.evidence_id
                    for item in self.support
                    if item.kind != "temporal"
                }
                basis_label = "current support evidence"
            if not set(self.rejection_basis_refs) <= permitted_basis_ids:
                raise ValueError(
                    "canonical rejection basis must reference " + basis_label
                )
        if self.contributing_member_ids != sorted(set(self.contributing_member_ids)):
            raise ValueError("canonical contributing members must be canonical")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect canonical claim digest")
        return self


class UnifiedFunnel(StrictModel):
    git_need_counts: dict[str, Annotated[int, Field(ge=0)]]
    git_case_counts: dict[str, Annotated[int, Field(ge=0)]]
    candidate_grouping_mode: Literal["semantic", "singleton"]
    grouping_status: Literal["complete", "partial", "not_required"]
    grouping_reason: str | None
    grouping_planned_pairs: Annotated[int, Field(ge=0)]
    grouping_planned_calls: Annotated[int, Field(ge=0)]
    grouping_terminal_calls: Annotated[int, Field(ge=0)]
    grouping_case_count: Annotated[int, Field(ge=0)]
    raw_candidate_events: Annotated[int, Field(ge=0)]
    unique_candidate_roots: Annotated[int, Field(ge=0)]
    refinement_versions: Annotated[int, Field(ge=0)]
    invalid_members: Annotated[int, Field(ge=0)]
    accepted: Annotated[int, Field(ge=0)]
    review: Annotated[int, Field(ge=0)]
    rejected: Annotated[int, Field(ge=0)]
    by_channel: dict[ChannelName, dict[str, Annotated[int, Field(ge=0)]]]
    by_evidence_route: dict[str, dict[str, Annotated[int, Field(ge=0)]]]
    attrition_reasons: dict[str, Annotated[int, Field(ge=0)]]

    @model_validator(mode="after")
    def denominator_algebra(self) -> UnifiedFunnel:
        need_statuses = {"affirmative", "no_established_need", "undetermined", "not_assessed", "invalid"}
        if set(self.git_need_counts) != need_statuses | {"candidate_versions"}:
            raise ValueError("Git need denominator is incomplete")
        if sum(self.git_need_counts[name] for name in need_statuses) != self.git_need_counts["candidate_versions"]:
            raise ValueError("Git need judgments do not close their candidate denominator")
        required_git_counts = {
            "cases", "eligible_cases", "affirmative_cases", "optional_cases", "required_cases", "queried_cases",
            "raw_logical_calls", "raw_physical_calls", "raw_cache_hits",
            *(f"result_{status}" for status in ("complete", "bounded_no_match", "partial", "unavailable", "failed", "not_required")),
        }
        if set(self.git_case_counts) != required_git_counts:
            raise ValueError("Git case denominator is incomplete")
        counts = self.git_case_counts
        if sum(value for key, value in counts.items() if key.startswith("result_")) != counts["cases"]:
            raise ValueError("Git terminal states do not cover every case")
        if counts["optional_cases"] + counts["required_cases"] != counts["affirmative_cases"]:
            raise ValueError("Git affirmative requirement modes do not close")
        if not counts["queried_cases"] <= counts["affirmative_cases"] <= counts["eligible_cases"] <= counts["cases"]:
            raise ValueError("Git route and query denominator is invalid")
        if counts["raw_logical_calls"] != counts["raw_physical_calls"] + counts["raw_cache_hits"]:
            raise ValueError("Git physical execution receipts do not close")
        if self.grouping_terminal_calls > self.grouping_planned_calls:
            raise ValueError("terminal grouping calls exceed planned calls")
        if self.candidate_grouping_mode == "singleton" and (
            self.grouping_planned_pairs
            or self.grouping_planned_calls
            or self.grouping_terminal_calls
            or self.grouping_status != "not_required"
            or self.grouping_reason != "candidate_grouping_mode_singleton"
        ):
            raise ValueError("singleton grouping funnel denominator is invalid")
        if self.raw_candidate_events < self.unique_candidate_roots:
            raise ValueError("raw candidate events cannot be below unique roots")
        required = {
            "raw_candidate_events",
            "unique_candidate_roots",
            "refinement_versions",
            "materialized_candidate_versions",
            "accepted",
            "review",
            "rejected",
            "invalid",
            "superseded",
        }
        if set(self.by_channel) != {"explicit", "implicit_current"}:
            raise ValueError("funnel discovery-channel denominator is incomplete")
        for dimension, breakdowns in (
            ("channel", self.by_channel),
            ("evidence route", self.by_evidence_route),
        ):
            for name, counts in breakdowns.items():
                if not required <= set(counts):
                    raise ValueError(f"funnel {dimension} counts are incomplete: {name}")
                if counts["raw_candidate_events"] < counts["unique_candidate_roots"]:
                    raise ValueError(f"funnel {dimension} raw/root algebra is invalid")
                terminal = sum(
                    counts[key]
                    for key in (
                        "accepted",
                        "review",
                        "rejected",
                        "invalid",
                        "superseded",
                    )
                )
                if terminal != counts["materialized_candidate_versions"]:
                    raise ValueError(
                        f"funnel {dimension} final disposition algebra is invalid"
                    )
                if counts["refinement_versions"] > counts[
                    "materialized_candidate_versions"
                ]:
                    raise ValueError(
                        f"funnel {dimension} refinement denominator is invalid"
                    )
                if any(
                    key not in required and not key.startswith("attrition:")
                    for key in counts
                ):
                    raise ValueError(f"funnel {dimension} count key is unknown")

        for field, expected in (
            ("raw_candidate_events", self.raw_candidate_events),
            ("unique_candidate_roots", self.unique_candidate_roots),
            ("refinement_versions", self.refinement_versions),
            ("invalid", self.invalid_members),
        ):
            if sum(item[field] for item in self.by_channel.values()) != expected:
                raise ValueError(f"funnel channel {field} total is invalid")
            if sum(item[field] for item in self.by_evidence_route.values()) != expected:
                raise ValueError(f"funnel route {field} total is invalid")
        if self.attrition_reasons != {
            key: value
            for key, value in sorted(self.attrition_reasons.items())
            if value > 0
        }:
            raise ValueError("funnel attrition reasons must be positive and canonical")
        for reason, expected in self.attrition_reasons.items():
            key = f"attrition:{reason}"
            if sum(item.get(key, 0) for item in self.by_channel.values()) != expected:
                raise ValueError("funnel channel attrition total is invalid")
            if sum(item.get(key, 0) for item in self.by_evidence_route.values()) != expected:
                raise ValueError("funnel route attrition total is invalid")
        known_attrition_keys = {
            f"attrition:{reason}" for reason in self.attrition_reasons
        }
        if any(
            key.startswith("attrition:") and key not in known_attrition_keys
            for breakdown in (*self.by_channel.values(), *self.by_evidence_route.values())
            for key in breakdown
        ):
            raise ValueError("funnel breakdown contains an unknown attrition reason")
        return self


class UnifiedCompleteness(StrictModel):
    structurally_valid: bool
    package_status: Literal["complete", "partial", "invalid"]
    capabilities: list[CapabilityDependency]
    omitted_possible: bool
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def canonical_capabilities(self) -> UnifiedCompleteness:
        names = [item.capability for item in self.capabilities]
        if names != sorted(set(names)):
            raise ValueError("completeness capabilities must be canonical")
        if self.reasons != sorted(set(self.reasons)):
            raise ValueError("completeness reasons must be canonical")
        return self


class UnifiedIndex(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = SCHEMA_VERSION
    contract_version: Literal["unified-evidence-contract-1.6"] = (
        UNIFIED_CONTRACT_VERSION
    )
    policy_version: Literal["unified-canonical-policy-1.13"] = UNIFIED_POLICY_VERSION
    bundle_id: UnifiedBundleId
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    profile_digest: ProfileDigest
    channel_assessment_ids: list[AssessmentId]
    case_ids: list[CaseId]
    finding_ids: list[FindingId]
    canonical_claim_ids: list[CanonicalClaimId]
    accepted_count: Annotated[int, Field(ge=0)]
    review_count: Annotated[int, Field(ge=0)]
    rejected_count: Annotated[int, Field(ge=0)]
    artifact_set_digest: Sha256Hex
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_and_digest(self) -> UnifiedIndex:
        if self.bundle_id != make_bundle_id(
            self.snapshot_digest, self.profile_digest
        ):
            raise ValueError("incorrect unified bundle ID")
        for values in (
            self.channel_assessment_ids,
            self.case_ids,
            self.finding_ids,
            self.canonical_claim_ids,
        ):
            if values != sorted(set(values)):
                raise ValueError("unified index IDs must be uniquely sorted")
        if self.accepted_count + self.review_count + self.rejected_count != len(
            self.canonical_claim_ids
        ):
            raise ValueError("unified index canonical count algebra is invalid")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect unified index digest")
        return self


class UnifiedBundle(StrictModel):
    index: UnifiedIndex
    assessments: list[ChannelAssessment]
    cases: list[CandidateCase]
    current_evidence: list[CurrentEvidenceBundle]
    git_bundle_refs: list[ImmutableArtifactRef]
    temporal_matrices: list[TemporalRelationMatrix]
    findings: list[UnifiedCaseFinding]
    refinements: list[RefinementRecord]
    canonical_claims: list[CanonicalClaim]
    member_dispositions: list[MemberDisposition]
    funnel: UnifiedFunnel
    completeness: UnifiedCompleteness
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def global_references_and_digest(self) -> UnifiedBundle:
        if self.index.channel_assessment_ids != sorted(
            item.assessment_id for item in self.assessments
        ):
            raise ValueError("unified assessment index mismatch")
        if self.index.case_ids != sorted(item.case_id for item in self.cases):
            raise ValueError("unified case index mismatch")
        if self.index.finding_ids != sorted(item.finding_id for item in self.findings):
            raise ValueError("unified finding index mismatch")
        if self.index.canonical_claim_ids != sorted(
            item.claim_id for item in self.canonical_claims
        ):
            raise ValueError("unified canonical index mismatch")
        member_ids = {item.member_id for case in self.cases for item in case.members}
        inactive_markers = {
            "refinement_parent_superseded",
            "refinement_review_branch",
        }
        active_cases = [
            case
            for case in self.cases
            if not inactive_markers.intersection(case.grouping_reason_codes)
        ]
        active_members = [
            item.member_id for case in active_cases for item in case.members
        ]
        if len(active_members) != len(set(active_members)):
            raise ValueError("a member belongs to multiple active cases")
        active_case_ids = {item.case_id for item in active_cases}
        case_members = {
            item.case_id: {member.member_id for member in item.members}
            for item in active_cases
        }
        for claim in self.canonical_claims:
            if (
                claim.case_id not in active_case_ids
                or not set(claim.contributing_member_ids)
                <= case_members[claim.case_id]
            ):
                raise ValueError("canonical claim references an inactive or foreign case")
        disposition_ids = [item.member_id for item in self.member_dispositions]
        if len(disposition_ids) != len(set(disposition_ids)) or set(
            disposition_ids
        ) != member_ids:
            raise ValueError("every case member requires one final disposition")
        case_members_all = {
            item.case_id: {member.member_id for member in item.members}
            for item in self.cases
        }
        claim_ids = {item.claim_id for item in self.canonical_claims}
        git_ref_keys = {
            (
                item.kind,
                item.stable_id,
                item.snapshot_digest,
                item.artifact_path,
                item.artifact_digest,
            )
            for item in self.git_bundle_refs
        }
        for matrix in self.temporal_matrices:
            if matrix.git_manifest_ref is None:
                continue
            ref = matrix.git_manifest_ref
            if (
                ref.kind,
                ref.stable_id,
                ref.snapshot_digest,
                ref.artifact_path,
                ref.artifact_digest,
            ) not in git_ref_keys:
                raise ValueError("temporal matrix references a foreign Git manifest")
        for disposition in self.member_dispositions:
            if (
                disposition.case_id not in case_members_all
                or disposition.member_id
                not in case_members_all[disposition.case_id]
                or not set(disposition.canonical_claim_ids) <= claim_ids
            ):
                raise ValueError("member disposition references a foreign case or claim")
        if self.funnel.accepted + self.funnel.review + self.funnel.rejected != len(
            self.canonical_claims
        ):
            raise ValueError("unified funnel canonical count algebra is invalid")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect unified bundle digest")
        return self


__all__ = [
    "CandidateCase",
    "CandidateCaseMember",
    "CandidateMemberRelation",
    "CandidateSemantics",
    "CanonicalClaim",
    "CanonicalDeliveryPolicy",
    "CanonicalEvidenceRef",
    "CapabilityDependency",
    "ChannelAssessment",
    "CurrentEvidenceBundle",
    "CurrentEvidenceItem",
    "EvidenceOrigin",
    "ImmutableArtifactRef",
    "MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS",
    "MAX_OTHER_EVIDENCE_DESCRIPTION_CHARS",
    "MemberDisposition",
    "PopulationDenominator",
    "RejectionKind",
    "RefinementProposal",
    "RefinementRecord",
    "ReviewClass",
    "SupportedConclusion",
    "TemporalEvidenceRef",
    "TemporalMemberRelation",
    "TemporalRelationMatrix",
    "UNIFIED_CONTRACT_VERSION",
    "UNIFIED_POLICY_VERSION",
    "UnifiedBundle",
    "UnifiedCaseFinding",
    "UnifiedCompleteness",
    "UnifiedFunnel",
    "UnifiedIndex",
    "UnifiedMemberFinding",
    "UnifiedMemberRelation",
]
