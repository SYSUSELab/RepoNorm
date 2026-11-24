"""One-round, evidence-bound unified refinement protocol."""

from __future__ import annotations
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Protocol

from repo_constraint_memory_v3.discovery import matches_glob
from repo_constraint_memory_v3.inference import InferenceRecord
from repo_constraint_memory_v3.temporal_need import (
    aggregate_git_needs, unassessed_git_need, validate_git_need,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    make_assessment_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_finding_id,
    make_member_id,
    make_refinement_id,
    stable_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    ImmutableArtifactRef,
    PopulationDenominator,
    RefinementProposal,
    RefinementRecord,
    TemporalRelationMatrix,
    UnifiedCaseFinding,
    UnifiedMemberFinding,
)


UNIFIED_REFINEMENT_PROTOCOL_VERSION = "unified-refinement-round-1.1"


@dataclass(frozen=True)
class ValidatedRefinementProposal:
    proposal: RefinementProposal
    requires_current_reacquisition: bool
    requires_git_reroute: bool
    reusable_evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class RefinementEvidence:
    current_evidence: CurrentEvidenceBundle
    git_bundle_refs: tuple[ImmutableArtifactRef, ...]
    complete: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class RefinementRoundResult:
    validated: tuple[ValidatedRefinementProposal, ...]
    findings: tuple[UnifiedCaseFinding, ...]
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CurrentEvidenceReacquisition:
    """Host proof that current evidence was rebuilt from a complete substrate."""

    current_evidence: CurrentEvidenceBundle | None
    substrate_kind: Literal[
        "complete_population",
        "complete_observation",
        "complete_population_and_observation",
        "unavailable",
    ]
    complete: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GitReacquisition:
    """Candidate-conditioned Git reroute output for a refined case."""

    git_bundle_refs: tuple[ImmutableArtifactRef, ...] = ()
    temporal_matrix: TemporalRelationMatrix | None = None
    complete: bool = True
    reason_codes: tuple[str, ...] = ()
    dependencies: tuple[CapabilityDependency, ...] = ()
    inference_records: tuple[InferenceRecord, ...] = ()


@dataclass(frozen=True)
class RoundOneVerification:
    """Validated-domain output plus replay provenance from verifier round 1."""

    finding: UnifiedCaseFinding | None
    inference_record: InferenceRecord | None = None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefinementLineageMetadata:
    """Host-owned links needed to seal a :class:`RefinementRecord`."""

    parent_case_id: str
    parent_member_id: str
    child_case_id: str
    child_assessment_ids: tuple[str, ...]
    child_member_ids: tuple[str, ...]
    current_evidence_bundle_id: str
    current_evidence_mode: Literal[
        "filtered_complete_reuse",
        "full_reacquisition",
        "failed",
    ]
    change_kinds: tuple[str, ...]
    requires_git_reroute: Literal[True]


@dataclass(frozen=True)
class RefinementExecutionResult:
    """Append-only artifacts produced by the host refinement adapter.

    The result deliberately does not mutate a ``UnifiedBundle`` or apply
    canonical policy.  A pipeline can append complete artifacts and hand the
    returned round-one finding to the existing policy stage.
    """

    validated: tuple[ValidatedRefinementProposal, ...]
    child_assessments: tuple[ChannelAssessment, ...]
    child_members: tuple[CandidateCaseMember, ...]
    child_cases: tuple[CandidateCase, ...]
    current_evidence_bundles: tuple[CurrentEvidenceBundle, ...]
    temporal_matrices: tuple[TemporalRelationMatrix, ...]
    findings: tuple[UnifiedCaseFinding, ...]
    refinement_records: tuple[RefinementRecord, ...]
    inference_records: tuple[InferenceRecord, ...]
    lineage: tuple[RefinementLineageMetadata, ...]
    status: Literal["complete", "review"]
    reason_codes: tuple[str, ...]


class Reacquirer(Protocol):
    """Host implementation that rebuilds evidence for one proposed version."""

    def reacquire(
        self,
        *,
        case: CandidateCase,
        parent: CandidateCaseMember,
        proposal: ValidatedRefinementProposal,
    ) -> RefinementEvidence: ...


class RoundOneVerifier(Protocol):
    def verify_round_one(
        self,
        *,
        case: CandidateCase,
        parent: CandidateCaseMember,
        proposal: ValidatedRefinementProposal,
        evidence: RefinementEvidence,
    ) -> UnifiedCaseFinding: ...


class CurrentEvidenceReacquirer(Protocol):
    """Rebuild a refined case from a complete population/observation base."""

    def reacquire_current(
        self,
        *,
        parent_case: CandidateCase,
        refined_case: CandidateCase,
        parent: CandidateCaseMember,
        child_assessments: Sequence[ChannelAssessment],
        proposals: Sequence[ValidatedRefinementProposal],
        previous_evidence: CurrentEvidenceBundle,
    ) -> CurrentEvidenceReacquisition: ...


class GitReacquirer(Protocol):
    """Reroute Git and classify temporal evidence for one refined case."""

    def reacquire_git(
        self,
        *,
        parent_case: CandidateCase,
        refined_case: CandidateCase,
        assessments: Sequence[ChannelAssessment],
        current_evidence: CurrentEvidenceBundle,
        proposals: Sequence[ValidatedRefinementProposal],
    ) -> GitReacquisition: ...


class HostRoundOneVerifier(Protocol):
    """Adapter over the ordinary UnifiedCaseVerifier in explicit round-1 mode."""

    def verify_round_one_case(
        self,
        *,
        case: CandidateCase,
        assessments: Sequence[ChannelAssessment],
        current_evidence: CurrentEvidenceBundle,
        temporal_matrix: TemporalRelationMatrix | None,
        verification_round: Literal[1],
    ) -> RoundOneVerification: ...


def _path_in_scope(path: str, patterns: Sequence[str]) -> bool:
    return any(matches_glob(path, pattern) for pattern in patterns)


def _scope_is_demonstrably_narrower(
    parent: Sequence[str], child: Sequence[str]
) -> bool:
    parent_set = set(parent)
    child_set = set(child)
    if child_set < parent_set:
        return True
    if parent_set == {"**"} and child_set != {"**"}:
        return True
    # Glob containment is undecidable in the general case.  Only accept exact
    # prefix containment; all ambiguous changes force reacquisition.
    for child_pattern in child:
        child_prefix = child_pattern.split("*", 1)[0]
        if not any(
            parent_pattern == "**"
            or child_prefix.startswith(parent_pattern.split("*", 1)[0])
            for parent_pattern in parent
        ):
            return False
    return set(child) != set(parent)


def _condition_is_stricter(parent: str | None, child: str | None) -> bool:
    if parent == child:
        return False
    if parent is None:
        return child is not None
    if child is None:
        return False
    return parent.casefold() in child.casefold() and len(child) > len(parent)


def validate_refinement_proposals(
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
    proposals: Sequence[RefinementProposal],
    max_children: int,
    parent_finding: UnifiedCaseFinding | None = None,
) -> tuple[tuple[ValidatedRefinementProposal, ...], tuple[str, ...]]:
    """Validate direction/core/lineage and determine reacquisition needs."""

    if max_children < 1:
        return (), ("refinement_child_limit_invalid",)
    if not proposals:
        return (), ("refinement_proposal_missing",)
    if len(proposals) > min(max_children, 2):
        return (), ("refinement_child_limit_exceeded",)
    if case.grouping_status != "complete":
        return (), ("refinement_parent_case_incomplete",)
    if (
        current_evidence.case_id != case.case_id
        or current_evidence.snapshot_digest != case.snapshot_digest
        or current_evidence.profile_digest != case.profile_digest
    ):
        return (), ("refinement_current_evidence_case_mismatch",)
    if parent_finding is not None and (
        parent_finding.case_id != case.case_id
        or parent_finding.snapshot_digest != case.snapshot_digest
        or parent_finding.profile_digest != case.profile_digest
        or parent_finding.verification_round != 0
        or not {item.model_dump_json() for item in proposals}
        <= {item.model_dump_json() for item in parent_finding.refinement_proposals}
    ):
        return (), ("refinement_parent_finding_binding_invalid",)
    member_by_id = {item.member_id: item for item in case.members}
    allowed_evidence_ids = {item.evidence_id for item in current_evidence.items}
    per_parent = Counter(item.parent_member_id for item in proposals)
    if len(per_parent) != 1:
        return (), ("refinement_multiple_parents_forbidden",)
    child_semantics: set[tuple[str, str]] = set()
    validated: list[ValidatedRefinementProposal] = []
    errors: set[str] = set()
    for proposal in proposals:
        parent_member = member_by_id.get(proposal.parent_member_id)
        if parent_member is None:
            errors.add("refinement_foreign_parent")
            continue
        parent = assessments.get(parent_member.assessment_id)
        if parent is None:
            errors.add("refinement_parent_assessment_missing")
            continue
        member_findings = [] if parent_finding is None else parent_finding.member_findings
        unresolved_axes = {
            name for finding in member_findings if finding.member_id == parent_member.member_id
            for name in ("norm_relevance", "rule_support", "scope_support", "condition_support", "temporal_status", "evidence_sufficiency")
            if getattr(finding, name) in {"uncertain", "adverse_unresolved"}
        }
        unresolved_axes.update(name for name, value in parent.axis_findings.items() if value in {"uncertain", "material"})
        if parent_finding is not None and any(
            parent_member.member_id in conflict.split(":", 2)[:2]
            for conflict in parent_finding.unresolved_conflicts
        ):
            unresolved_axes.add("unresolved_conflicts")
        reference_paths = {
            item.evidence_id: {origin.path for origin in current_evidence.origins if origin.origin_id == item.origin_id}
            for item in current_evidence.items if parent_member.member_id in item.member_ids and item.role != "ineligible"
        }
        try:
            validate_git_need(
                proposal.git_need,
                allowed_refs=set(reference_paths),
                allowed_paths={path for paths in reference_paths.values() for path in paths},
                reference_paths=reference_paths,
                unresolved_axes=unresolved_axes,
            )
        except ValueError:
            errors.add("refinement_git_need_binding_invalid")
            continue
        if (
            parent.snapshot_digest != case.snapshot_digest
            or parent.profile_digest != case.profile_digest
            or parent.assessment_id != parent_member.assessment_id
            or parent.channel != parent_member.channel
            or parent.semantics != parent_member.semantics
        ):
            errors.add("refinement_parent_assessment_mismatch")
            continue
        if (
            "unified_refinement_child" in parent.typed_reasons
            or "unified_refinement_round:1" in parent.provenance
        ):
            errors.add("second_refinement_forbidden")
            continue
        if proposal.semantics.normalized_action != parent.semantics.normalized_action:
            errors.add("refinement_changes_core_action")
            continue
        if proposal.semantics.polarity != parent.semantics.polarity:
            errors.add("refinement_reverses_polarity")
            continue
        if not set(proposal.evidence_refs) <= allowed_evidence_ids:
            errors.add("refinement_foreign_evidence")
            continue
        if proposal.evidence_refs != sorted(set(proposal.evidence_refs)):
            errors.add("refinement_evidence_refs_not_canonical")
            continue
        if proposal.change_kinds != sorted(set(proposal.change_kinds)):
            errors.add("refinement_change_kinds_not_canonical")
            continue
        child_key = (
            proposal.parent_member_id,
            stable_id("REFSEM", proposal.semantics.model_dump(mode="json")),
        )
        if child_key in child_semantics:
            errors.add("refinement_duplicate_child_semantics")
            continue
        child_semantics.add(child_key)
        actual_changes: set[str] = set()
        if proposal.semantics.rule != parent.semantics.rule:
            actual_changes.add("wording")
        if proposal.semantics.applies_to != parent.semantics.applies_to:
            actual_changes.add("scope")
        if proposal.semantics.condition != parent.semantics.condition:
            actual_changes.add("condition")
        if proposal.semantics.exception != parent.semantics.exception:
            actual_changes.add("exception")
        if per_parent[proposal.parent_member_id] > 1:
            actual_changes.add("split")
        if not actual_changes:
            errors.add("refinement_has_no_change")
            continue
        if set(proposal.change_kinds) != actual_changes:
            errors.add("refinement_change_kinds_mismatch")
            continue

        scope_changed = "scope" in actual_changes
        scope_narrower = scope_changed and _scope_is_demonstrably_narrower(
            parent.semantics.applies_to, proposal.semantics.applies_to
        )
        condition_changed = "condition" in actual_changes
        condition_stricter = condition_changed and _condition_is_stricter(
            parent.semantics.condition, proposal.semantics.condition
        )
        needs_reacquisition = (
            "split" in actual_changes
            or (scope_changed and not scope_narrower)
            or (condition_changed and not condition_stricter)
            or "exception" in actual_changes
            or current_evidence.integrity != "valid"
            or current_evidence.status != "complete"
            or any(item.status != "complete" for item in current_evidence.denominators)
        )
        reusable = tuple(
            sorted(
                item.evidence_id
                for item in current_evidence.items
                if parent_member.member_id in item.member_ids
                and item.evidence_id in proposal.evidence_refs
                and (
                    not scope_changed
                    or _path_in_scope(
                        next(
                            origin.path
                            for origin in current_evidence.origins
                            if origin.origin_id == item.origin_id
                        ),
                        proposal.semantics.applies_to,
                    )
                )
            )
        )
        if not reusable:
            needs_reacquisition = True
        validated.append(
            ValidatedRefinementProposal(
                proposal=proposal,
                requires_current_reacquisition=needs_reacquisition,
                # Any semantic version change receives a fresh candidate-
                # conditioned temporal route; no old Git conclusion is inherited.
                requires_git_reroute=True,
                reusable_evidence_ids=reusable,
                reason_codes=(
                    (
                        "complete_population_or_observation_reacquisition_required",
                    )
                    if needs_reacquisition
                    else ("filtered_complete_population_reuse_allowed",)
                ),
            )
        )
    if errors:
        return (), tuple(sorted(errors))
    return tuple(validated), ()


def run_refinement_round(
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
    proposals: Sequence[RefinementProposal],
    max_children: int,
    reacquirer: Reacquirer,
    verifier: RoundOneVerifier,
) -> RefinementRoundResult:
    """Run the only permitted refinement round and fail closed on any gap."""

    validated, validation_errors = validate_refinement_proposals(
        case=case,
        assessments=assessments,
        current_evidence=current_evidence,
        proposals=proposals,
        max_children=max_children,
    )
    if validation_errors:
        return RefinementRoundResult(
            validated=(),
            findings=(),
            status="review",
            reason_codes=validation_errors,
        )
    member_by_id = {item.member_id: item for item in case.members}
    findings: list[UnifiedCaseFinding] = []
    reasons: set[str] = set()
    for proposal in validated:
        parent = member_by_id[proposal.proposal.parent_member_id]
        evidence = reacquirer.reacquire(
            case=case,
            parent=parent,
            proposal=proposal,
        )
        if not evidence.complete:
            reasons.update(evidence.reason_codes or ("refinement_reacquisition_gap",))
            continue
        finding = verifier.verify_round_one(
            case=case,
            parent=parent,
            proposal=proposal,
            evidence=evidence,
        )
        if finding.verification_round != 1 or finding.status != "complete":
            reasons.update(finding.failure_reasons or ["round_one_verifier_failed"])
            continue
        if finding.refinement_proposals:
            reasons.add("second_refinement_forbidden")
            continue
        findings.append(finding)
    complete = len(findings) == len(validated) and not reasons
    return RefinementRoundResult(
        validated=validated,
        findings=tuple(findings),
        status="complete" if complete else "review",
        reason_codes=tuple(sorted(reasons)),
    )


def _proposal_key(value: ValidatedRefinementProposal) -> tuple[str, ...]:
    proposal = value.proposal
    return (
        stable_id("REFPROPOSAL", proposal.model_dump(mode="json")),
        proposal.parent_member_id,
    )


def _artifact_ref_key(value: ImmutableArtifactRef) -> tuple[str, ...]:
    return (
        value.kind,
        value.stable_id,
        value.snapshot_digest,
        value.artifact_path,
        value.artifact_digest,
    )


def _canonical_git_refs(
    values: Sequence[ImmutableArtifactRef], *, snapshot_digest: str
) -> tuple[ImmutableArtifactRef, ...]:
    allowed_kinds = {
        "git_manifest",
        "git_temporal_bundle",
        "git_change_slice",
        "temporal_matrix",
        "inference_request",
    }
    by_identity: dict[tuple[str, str], ImmutableArtifactRef] = {}
    for value in values:
        if value.snapshot_digest != snapshot_digest:
            raise ValueError("refinement Git reference snapshot mismatch")
        if value.kind not in allowed_kinds:
            raise ValueError("refinement Git reference kind is invalid")
        identity = (value.kind, value.stable_id)
        previous = by_identity.get(identity)
        if previous is not None and previous != value:
            raise ValueError("refinement Git reference identity conflict")
        by_identity[identity] = value
    return tuple(sorted(by_identity.values(), key=_artifact_ref_key))


def _make_child_assessment(
    *,
    parent: ChannelAssessment,
    proposal: ValidatedRefinementProposal,
    ordinal: int,
) -> ChannelAssessment:
    child_root_id = stable_id(
        "REFROOT",
        [
            parent.candidate_root_id,
            proposal.proposal.semantics.model_dump(mode="json"),
            ordinal,
        ],
    )
    child_version = parent.candidate_version + 1
    assessment_id = make_assessment_id(
        parent.snapshot_digest,
        parent.profile_digest,
        parent.channel,
        child_root_id,
        child_version,
    )
    parent_ref = ImmutableArtifactRef(
        kind="channel_assessment",
        stable_id=parent.assessment_id,
        snapshot_digest=parent.snapshot_digest,
        artifact_path=(
            "canonical/audit/channel-assessments/"
            f"{parent.assessment_id}.json"
        ),
        artifact_digest=parent.artifact_digest,
    )
    lineage_by_key = {
        _artifact_ref_key(value): value for value in [*parent.lineage_refs, parent_ref]
    }
    payload = parent.model_dump(
        mode="python",
        exclude={
            "assessment_id",
            "candidate_root_id",
            "candidate_version",
            "parent_candidate_id",
            "semantics",
            "typed_reasons",
            "lineage_refs",
            "provenance",
            "artifact_digest",
        },
    )
    payload.update(
        {
            "assessment_id": assessment_id,
            "candidate_root_id": child_root_id,
            "candidate_version": child_version,
            "parent_candidate_id": parent.assessment_id,
            "semantics": proposal.proposal.semantics,
            "git_need": aggregate_git_needs([proposal.proposal.git_need]),
            "typed_reasons": sorted(
                {
                    *parent.typed_reasons,
                    "unified_refinement_child",
                    "refinement_requires_git_reroute",
                    *(
                        {"refinement_requires_current_reacquisition"}
                        if proposal.requires_current_reacquisition
                        else {"refinement_allows_filtered_current_reuse"}
                    ),
                }
            ),
            "lineage_refs": [
                lineage_by_key[key] for key in sorted(lineage_by_key)
            ],
            "provenance": list(
                dict.fromkeys(
                    [
                        *parent.provenance,
                        "unified_refinement_round:1",
                        f"refinement_parent:{parent.assessment_id}",
                    ]
                )
            ),
        }
    )
    payload["artifact_digest"] = artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def _make_child_member(assessment: ChannelAssessment) -> CandidateCaseMember:
    member_id = make_member_id(
        assessment.snapshot_digest, assessment.assessment_id
    )
    return CandidateCaseMember(
        member_id=member_id,
        assessment_id=assessment.assessment_id,
        snapshot_digest=assessment.snapshot_digest,
        channel=assessment.channel,
        semantics=assessment.semantics,
        assessment_ref=ImmutableArtifactRef(
            kind="channel_assessment",
            stable_id=assessment.assessment_id,
            snapshot_digest=assessment.snapshot_digest,
            artifact_path=(
                "canonical/audit/channel-assessments/"
                f"{assessment.assessment_id}.json"
            ),
            artifact_digest=assessment.artifact_digest,
        ),
    )


def _make_refined_case(
    *,
    parent_case: CandidateCase,
    parent: CandidateCaseMember,
    children: Sequence[CandidateCaseMember],
) -> CandidateCase:
    members = sorted(
        [
            *(item for item in parent_case.members if item.member_id != parent.member_id),
            *children,
        ],
        key=lambda item: item.member_id,
    )
    child_ids = {item.member_id for item in children}
    old_relations = {
        (item.left_member_id, item.right_member_id): item
        for item in parent_case.relations
    }
    relations: list[CandidateMemberRelation] = []
    for left, right in combinations([item.member_id for item in members], 2):
        if left not in child_ids and right not in child_ids:
            relation = old_relations.get((left, right))
            if relation is None:
                raise ValueError("parent case relation matrix is incomplete")
            relations.append(relation)
            continue
        relations.append(
            CandidateMemberRelation(
                left_member_id=left,
                right_member_id=right,
                relation="uncertain",
                source="semantic_grouping",
                reason_code="refinement_requires_relation_reverification",
            )
        )
    exact_groups = [
        sorted(item for item in group if item != parent.member_id)
        for group in parent_case.exact_identity_groups
    ]
    exact_groups = [item for item in exact_groups if len(item) > 1]
    member_ids = [item.member_id for item in members]
    payload: dict[str, object] = {
        "case_id": make_case_id(parent_case.snapshot_digest, member_ids),
        "snapshot_digest": parent_case.snapshot_digest,
        "profile_digest": parent_case.profile_digest,
        "members": members,
        "relations": relations,
        "grouping_status": "complete",
        "grouping_reason_codes": sorted(
            {
                *parent_case.grouping_reason_codes,
                *(
                    {"refinement_relation_reverification_required"}
                    if len(members) > 1
                    else set()
                ),
            }
        ),
        "exact_identity_groups": sorted(exact_groups),
        "required_capabilities": sorted(
            set(parent_case.required_capabilities)
        ),
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CandidateCase.model_validate(payload)


def _current_dependency(
    *, mode: str, complete: bool, reason_codes: Sequence[str] = ()
) -> CapabilityDependency:
    reasons = tuple(sorted(set(reason_codes)))
    if complete:
        return CapabilityDependency(
            capability=f"refinement_current_{mode}",
            requirement_mode="required",
            status="complete",
            reason_codes=[],
        )
    return CapabilityDependency(
        capability="refinement_current_reacquisition",
        requirement_mode="required",
        status="failed",
        reason_codes=list(reasons or ("refinement_current_reacquisition_gap",)),
    )


_STALE_TEMPORAL_DEPENDENCIES = {
    "git_temporal_evidence",
    "temporal_evidence_classification",
    "refinement_git_reroute",
}


def _with_current_dependency(
    bundle: CurrentEvidenceBundle,
    *,
    mode: Literal["filtered_complete_reuse", "full_reacquisition"],
) -> CurrentEvidenceBundle:
    dependencies = {
        item.capability: item
        for item in bundle.dependencies
        if not item.capability.startswith("refinement_current_")
        and item.capability not in _STALE_TEMPORAL_DEPENDENCIES
    }
    marker = _current_dependency(mode=mode, complete=True)
    dependencies[marker.capability] = marker
    payload = bundle.model_dump(mode="python", exclude={"artifact_digest"})
    payload["dependencies"] = [dependencies[key] for key in sorted(dependencies)]
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _with_git_reroute_dependencies(
    bundle: CurrentEvidenceBundle,
    *,
    supplied: Sequence[CapabilityDependency],
    complete: bool,
    reason_codes: Sequence[str],
) -> CurrentEvidenceBundle:
    dependencies = {
        item.capability: item
        for item in bundle.dependencies
        if item.capability not in _STALE_TEMPORAL_DEPENDENCIES
    }
    seen: set[str] = set()
    for item in supplied:
        if item.capability in seen:
            raise ValueError("refinement Git dependency is duplicated")
        if item.capability not in {
            "git_temporal_evidence",
            "temporal_evidence_classification",
        }:
            raise ValueError("refinement Git dependency kind is invalid")
        seen.add(item.capability)
        dependencies[item.capability] = item
    marker = CapabilityDependency(
        capability="refinement_git_reroute",
        requirement_mode="required",
        status="complete" if complete else "failed",
        reason_codes=(
            []
            if complete
            else list(
                sorted(set(reason_codes))
                or ["refinement_git_reroute_incomplete"]
            )
        ),
    )
    dependencies[marker.capability] = marker
    payload = bundle.model_dump(mode="python", exclude={"artifact_digest"})
    payload["dependencies"] = [dependencies[key] for key in sorted(dependencies)]
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _population_counts_for_member(
    items: Sequence[CurrentEvidenceItem], member_id: str
) -> dict[str, int]:
    result = {
        "support": 0,
        "counterexample": 0,
        "unknown": 0,
        "ineligible": 0,
    }
    for item in items:
        if (
            member_id not in item.member_ids
            or item.artifact_ref.kind != "population_member"
        ):
            continue
        if item.role == "support":
            result["support"] += 1
        elif item.role == "counterevidence":
            result["counterexample"] += 1
        elif item.role in {"unknown", "ineligible"}:
            result[item.role] += 1
    return result


def _rebind_filtered_current_evidence(
    *,
    refined_case: CandidateCase,
    parent: CandidateCaseMember,
    child: CandidateCaseMember,
    previous: CurrentEvidenceBundle,
) -> CurrentEvidenceBundle:
    if previous.integrity != "valid" or previous.status != "complete":
        raise ValueError("previous current evidence is not complete")
    if any(item.status != "complete" for item in previous.denominators):
        raise ValueError("previous population denominator is not complete")

    origin_by_id = {item.origin_id: item for item in previous.origins}
    item_by_id: dict[str, CurrentEvidenceItem] = {}
    for item in previous.items:
        member_ids = [
            member_id
            for member_id in item.member_ids
            if member_id != parent.member_id
        ]
        if parent.member_id in item.member_ids:
            origin = origin_by_id[item.origin_id]
            applicability_bound = origin.origin_kind in {
                "observation",
                "population_member",
            }
            if not applicability_bound or _path_in_scope(
                origin.path, child.semantics.applies_to
            ):
                member_ids.append(child.member_id)
        member_ids = sorted(set(member_ids))
        if not member_ids:
            continue
        evidence_id = make_evidence_id(
            item.origin_id,
            item.role,
            member_ids,
            item.artifact_ref.stable_id,
        )
        rebound = item.model_copy(
            update={"evidence_id": evidence_id, "member_ids": member_ids}
        )
        previous_item = item_by_id.get(evidence_id)
        if previous_item is not None and previous_item != rebound:
            raise ValueError("filtered evidence identity conflict")
        item_by_id[evidence_id] = rebound
    items = [item_by_id[key] for key in sorted(item_by_id)]
    retained_origin_ids = {item.origin_id for item in items}
    origins = [
        origin_by_id[key]
        for key in sorted(retained_origin_ids)
    ]

    denominators: list[PopulationDenominator] = []
    found_parent_denominator = False
    for denominator in previous.denominators:
        if denominator.member_id != parent.member_id:
            denominators.append(denominator)
            continue
        found_parent_denominator = True
        if child.channel != "implicit_current":
            continue
        counts = _population_counts_for_member(items, child.member_id)
        inspected = sum(counts.values())
        serialized = inspected + denominator.uninspected
        denominators.append(
            PopulationDenominator(
                member_id=child.member_id,
                population_id=denominator.population_id,
                status=("partial" if denominator.uninspected else "complete"),
                census_enumeration_status=(
                    "partial"
                    if denominator.uninspected
                    else denominator.census_enumeration_status
                ),
                sampling_plan_status=denominator.sampling_plan_status,
                sampling_mode=denominator.sampling_mode,
                scanned=denominator.scanned,
                serialized=serialized,
                matched=serialized,
                inspected=inspected,
                uninspected=denominator.uninspected,
                **counts,
            )
        )
    if child.channel == "implicit_current" and not found_parent_denominator:
        raise ValueError("complete implicit population denominator is missing")
    denominators.sort(key=lambda item: (item.member_id, item.population_id))

    dependencies = {
        item.capability: item
        for item in previous.dependencies
        if not item.capability.startswith("refinement_current_")
        and item.capability not in _STALE_TEMPORAL_DEPENDENCIES
    }
    partial_denominator = any(
        item.status != "complete" for item in denominators
    )
    incomplete_reasons = (
        ["refinement_sampled_population_denominator_partial"]
        if partial_denominator
        else []
    )
    marker = _current_dependency(
        mode="filtered_complete_reuse",
        complete=not partial_denominator,
        reason_codes=incomplete_reasons,
    )
    dependencies[marker.capability] = marker
    payload: dict[str, object] = {
        "bundle_id": make_current_evidence_bundle_id(
            refined_case.snapshot_digest,
            refined_case.profile_digest,
            refined_case.case_id,
        ),
        "case_id": refined_case.case_id,
        "snapshot_digest": refined_case.snapshot_digest,
        "profile_digest": refined_case.profile_digest,
        "origins": origins,
        "items": items,
        "denominators": denominators,
        "population_refs": previous.population_refs,
        "falsification_refs": previous.falsification_refs,
        "scope_condition_refs": previous.scope_condition_refs,
        "dependencies": [dependencies[key] for key in sorted(dependencies)],
        "integrity": "valid",
        "status": "partial" if partial_denominator else "complete",
        "incomplete_reasons": incomplete_reasons,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _failed_current_evidence(
    *,
    refined_case: CandidateCase,
    previous: CurrentEvidenceBundle,
    reason_codes: Sequence[str],
) -> CurrentEvidenceBundle:
    reasons = sorted(set(reason_codes) or {"refinement_current_reacquisition_gap"})
    dependency = _current_dependency(
        mode="reacquisition", complete=False, reason_codes=reasons
    )
    payload: dict[str, object] = {
        "bundle_id": make_current_evidence_bundle_id(
            refined_case.snapshot_digest,
            refined_case.profile_digest,
            refined_case.case_id,
        ),
        "case_id": refined_case.case_id,
        "snapshot_digest": refined_case.snapshot_digest,
        "profile_digest": refined_case.profile_digest,
        "origins": [],
        "items": [],
        "denominators": [],
        "population_refs": previous.population_refs,
        "falsification_refs": previous.falsification_refs,
        "scope_condition_refs": previous.scope_condition_refs,
        "dependencies": [dependency],
        "integrity": "valid",
        "status": "failed",
        "incomplete_reasons": reasons,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _current_binding_errors(
    *,
    bundle: CurrentEvidenceBundle,
    refined_case: CandidateCase,
    child_ids: set[str],
) -> tuple[str, ...]:
    errors: set[str] = set()
    try:
        checked = CurrentEvidenceBundle.model_validate(
            bundle.model_dump(mode="json")
        )
    except ValueError:
        return ("refinement_current_evidence_invalid",)
    if (
        checked.case_id != refined_case.case_id
        or checked.snapshot_digest != refined_case.snapshot_digest
        or checked.profile_digest != refined_case.profile_digest
    ):
        errors.add("refinement_current_evidence_case_mismatch")
    case_member_ids = {item.member_id for item in refined_case.members}
    evidence_member_ids = {
        member_id for item in checked.items for member_id in item.member_ids
    }
    denominator_member_ids = {item.member_id for item in checked.denominators}
    raw_denominator_member_ids = [
        item.member_id for item in checked.denominators
    ]
    if not evidence_member_ids <= case_member_ids:
        errors.add("refinement_current_evidence_foreign_member")
    if not denominator_member_ids <= case_member_ids:
        errors.add("refinement_current_denominator_foreign_member")
    if len(raw_denominator_member_ids) != len(set(raw_denominator_member_ids)):
        errors.add("refinement_current_denominator_duplicate")
    if any(item.status != "complete" for item in checked.denominators):
        errors.add("refinement_current_denominator_incomplete")
    implicit_child_ids = {
        item.member_id
        for item in refined_case.members
        if item.member_id in child_ids and item.channel == "implicit_current"
    }
    if not implicit_child_ids <= denominator_member_ids:
        errors.add("refinement_current_denominator_missing")
    denominator_by_member = {
        item.member_id: item for item in checked.denominators
    }
    origin_by_id = {item.origin_id: item for item in checked.origins}
    child_by_id = {
        item.member_id: item
        for item in refined_case.members
        if item.member_id in child_ids
    }
    for child_id, child in child_by_id.items():
        for item in checked.items:
            if (
                child_id in item.member_ids
                and origin_by_id[item.origin_id].origin_kind
                in {"observation", "population_member"}
                and not _path_in_scope(
                    origin_by_id[item.origin_id].path,
                    child.semantics.applies_to,
                )
            ):
                errors.add("refinement_current_evidence_outside_child_scope")
        if child.channel != "implicit_current":
            continue
        denominator = denominator_by_member.get(child_id)
        if denominator is None:
            continue
        counts = _population_counts_for_member(checked.items, child_id)
        if any(
            getattr(denominator, field_name) != value
            for field_name, value in counts.items()
        ):
            errors.add("refinement_current_population_counts_mismatch")
    if checked.integrity != "valid" or checked.status != "complete":
        errors.add("refinement_current_reacquisition_incomplete")
    if checked.incomplete_reasons:
        errors.add("refinement_current_reacquisition_incomplete")
    if any(
        item.requirement_mode == "required"
        and item.status not in {"complete", "bounded_no_match", "not_required"}
        for item in checked.dependencies
    ):
        errors.add("refinement_current_required_dependency_gap")
    return tuple(sorted(errors))


def _failed_finding(
    *, case: CandidateCase, reason_codes: Sequence[str]
) -> UnifiedCaseFinding:
    reasons = sorted(set(reason_codes) or {"round_one_verifier_failed"})
    member_findings = [
        UnifiedMemberFinding(
            member_id=item.member_id,
            norm_relevance="uncertain",
            rule_support="uncertain",
            scope_support="uncertain",
            condition_support="uncertain",
            temporal_status="not_available",
            evidence_sufficiency="uncertain",
            evidence_refs=[],
            closed_local_reason_codes=[],
            reason_codes=reasons,
        )
        for item in case.members
    ]
    payload: dict[str, object] = {
        "finding_id": make_finding_id(
            case.snapshot_digest, case.profile_digest, case.case_id, 1
        ),
        "case_id": case.case_id,
        "snapshot_digest": case.snapshot_digest,
        "profile_digest": case.profile_digest,
        "verification_round": 1,
        "member_findings": member_findings,
        "member_relations": [],
        "unresolved_conflicts": [],
        "conclusions": [],
        "refinement_proposals": [],
        "inference_ref": None,
        "status": "failed",
        "failure_reasons": reasons,
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return UnifiedCaseFinding.model_validate(payload)


def _finding_errors(
    *,
    finding: UnifiedCaseFinding,
    case: CandidateCase,
    assessments: Sequence[ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
    temporal_matrix: TemporalRelationMatrix | None,
) -> tuple[str, ...]:
    try:
        checked = UnifiedCaseFinding.model_validate(
            finding.model_dump(mode="json")
        )
    except ValueError:
        return ("round_one_finding_invalid",)
    errors: set[str] = set()
    if (
        checked.case_id != case.case_id
        or checked.snapshot_digest != case.snapshot_digest
        or checked.profile_digest != case.profile_digest
        or checked.verification_round != 1
    ):
        errors.add("round_one_finding_case_mismatch")
    member_ids = [item.member_id for item in case.members]
    finding_member_ids = [item.member_id for item in checked.member_findings]
    if len(finding_member_ids) != len(set(finding_member_ids)) or set(
        finding_member_ids
    ) != set(member_ids):
        errors.add("round_one_finding_member_matrix_invalid")
    expected_pairs = set(combinations(member_ids, 2))
    raw_relation_pairs = [
        (item.left_member_id, item.right_member_id)
        for item in checked.member_relations
    ]
    relation_pairs = set(raw_relation_pairs)
    if checked.status == "complete" and (
        relation_pairs != expected_pairs
        or len(raw_relation_pairs) != len(relation_pairs)
    ):
        errors.add("round_one_finding_relation_matrix_invalid")
    assessment_by_member = {
        member.member_id: next(
            item
            for item in assessments
            if item.assessment_id == member.assessment_id
        )
        for member in case.members
    }
    for item in checked.member_findings:
        if (
            item.evidence_refs != sorted(set(item.evidence_refs))
            or item.reason_codes != sorted(set(item.reason_codes))
            or item.closed_local_reason_codes
            != sorted(set(item.closed_local_reason_codes))
        ):
            errors.add("round_one_finding_member_fields_not_canonical")
        if not set(item.closed_local_reason_codes) <= set(
            assessment_by_member[item.member_id].typed_reasons
        ):
            errors.add("round_one_finding_closes_foreign_local_reason")
    for item in checked.member_relations:
        if item.evidence_refs != sorted(set(item.evidence_refs)):
            errors.add("round_one_finding_relation_refs_not_canonical")
    for item in checked.conclusions:
        if not set(item.base_member_ids) <= set(member_ids):
            errors.add("round_one_finding_conclusion_foreign_member")
    for item in checked.refinement_proposals:
        if item.parent_member_id not in set(member_ids):
            errors.add("round_one_refinement_foreign_parent")
        if (
            item.evidence_refs != sorted(set(item.evidence_refs))
            or item.change_kinds != sorted(set(item.change_kinds))
        ):
            errors.add("round_one_refinement_fields_not_canonical")
    allowed_refs = {
        ref.stable_id
        for assessment in assessments
        for ref in [*assessment.support_refs, *assessment.counterevidence_refs]
    }
    allowed_refs.update(item.evidence_id for item in current_evidence.items)
    if temporal_matrix is not None:
        allowed_refs.update(item.item_id for item in temporal_matrix.relations)
        allowed_refs.update(
            item.evidence_span
            for item in temporal_matrix.relations
            if item.evidence_span is not None
        )
    cited_refs = {
        evidence_ref
        for item in checked.member_findings
        for evidence_ref in item.evidence_refs
    }
    cited_refs.update(
        evidence_ref
        for item in checked.member_relations
        for evidence_ref in item.evidence_refs
    )
    cited_refs.update(
        evidence_ref
        for item in checked.conclusions
        for evidence_ref in item.evidence_refs
    )
    cited_refs.update(
        evidence_ref
        for item in checked.refinement_proposals
        for evidence_ref in item.evidence_refs
    )
    if not cited_refs <= allowed_refs:
        errors.add("round_one_finding_foreign_evidence")
    if checked.inference_ref is not None and (
        checked.inference_ref.snapshot_digest != case.snapshot_digest
        or checked.inference_ref.kind != "inference_request"
    ):
        errors.add("round_one_finding_inference_ref_invalid")
    return tuple(sorted(errors))


def _make_refinement_record(
    *,
    parent_case: CandidateCase,
    parent: CandidateCaseMember,
    children: Sequence[CandidateCaseMember],
    current_evidence: CurrentEvidenceBundle,
    git_bundle_refs: Sequence[ImmutableArtifactRef],
    finding: UnifiedCaseFinding,
    change_kinds: Sequence[str],
    status: Literal["complete", "review"],
    reason_codes: Sequence[str],
) -> RefinementRecord:
    child_ids = sorted(item.member_id for item in children)
    payload: dict[str, object] = {
        "refinement_id": make_refinement_id(
            parent_case.snapshot_digest,
            parent_case.case_id,
            parent.member_id,
            child_ids,
        ),
        "case_id": parent_case.case_id,
        "snapshot_digest": parent_case.snapshot_digest,
        "parent_member_id": parent.member_id,
        "child_member_ids": child_ids,
        "round": 1,
        "change_kinds": sorted(set(change_kinds)),
        "reacquired_current_bundle_ids": [current_evidence.bundle_id],
        "git_bundle_refs": sorted(git_bundle_refs, key=_artifact_ref_key),
        "finding_ids": [finding.finding_id],
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return RefinementRecord.model_validate(payload)


def _validate_git_reacquisition(
    *,
    value: GitReacquisition,
    refined_case: CandidateCase,
) -> tuple[tuple[ImmutableArtifactRef, ...], TemporalRelationMatrix | None]:
    refs = _canonical_git_refs(
        value.git_bundle_refs, snapshot_digest=refined_case.snapshot_digest
    )
    matrix = value.temporal_matrix
    if matrix is None:
        return refs, None
    try:
        checked = TemporalRelationMatrix.model_validate(
            matrix.model_dump(mode="json")
        )
    except ValueError as error:
        raise ValueError("refinement temporal matrix is invalid") from error
    if (
        checked.case_id != refined_case.case_id
        or checked.snapshot_digest != refined_case.snapshot_digest
        or checked.profile_digest != refined_case.profile_digest
        or checked.member_ids
        != sorted(item.member_id for item in refined_case.members)
    ):
        raise ValueError("refinement temporal matrix does not belong to the case")
    return refs, checked


def rebind_child_git_need(
    child: ChannelAssessment,
    *, parent_current: CurrentEvidenceBundle,
    child_current: CurrentEvidenceBundle,
    child_member_id: str,
) -> ChannelAssessment:
    """Rebind proposal citations by exact retained immutable evidence identity."""
    if child.git_need.status != "affirmative":
        return child
    parent_items = {item.evidence_id: item for item in parent_current.items}
    rebound: dict[str, str] = {}
    for ref in {ref for judgment in child.git_need.judgments for question in judgment.questions for ref in question.evidence_refs}:
        previous = parent_items.get(ref)
        matches = [] if previous is None else [item.evidence_id for item in child_current.items
            if child_member_id in item.member_ids and item.role != "ineligible"
            and item.origin_id == previous.origin_id and item.artifact_ref == previous.artifact_ref
            and item.excerpt_digest == previous.excerpt_digest and item.role == previous.role]
        if len(matches) != 1:
            need = unassessed_git_need("refinement_git_need_current_binding_invalid", invalid=True)
            break
        rebound[ref] = matches[0]
    else:
        judgments = [judgment.model_copy(update={"questions": [question.model_copy(update={"evidence_refs": [rebound[ref] for ref in question.evidence_refs]}) for question in judgment.questions]}) for judgment in child.git_need.judgments]
        need = aggregate_git_needs(judgments)
    payload = child.model_dump(mode="python", exclude={"artifact_digest"})
    payload["git_need"] = need
    payload["artifact_digest"] = artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def execute_refinement_round(
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
    proposals: Sequence[RefinementProposal],
    max_children: int,
    git_reacquirer: GitReacquirer | None,
    verifier: HostRoundOneVerifier | None,
    current_reacquirer: CurrentEvidenceReacquirer | None = None,
    parent_finding: UnifiedCaseFinding | None = None,
) -> RefinementExecutionResult:
    """Materialize and reverify the sole permitted refinement round.

    Safe scope narrowing and strict-condition changes can reuse a complete
    serialized current population.  Every other applicability change requires
    an injected host reacquirer which attests that a complete population or
    observation substrate was queried.  Every child is then Git-rerouted and
    sent through the ordinary case verifier with ``verification_round=1``.
    """

    validated, errors = validate_refinement_proposals(
        case=case,
        assessments=assessments,
        current_evidence=current_evidence,
        proposals=proposals,
        max_children=max_children,
        parent_finding=parent_finding,
    )
    if errors:
        return RefinementExecutionResult(
            validated=(),
            child_assessments=(),
            child_members=(),
            child_cases=(),
            current_evidence_bundles=(),
            temporal_matrices=(),
            findings=(),
            refinement_records=(),
            inference_records=(),
            lineage=(),
            status="review",
            reason_codes=errors,
        )

    assessment_results: list[ChannelAssessment] = []
    member_results: list[CandidateCaseMember] = []
    case_results: list[CandidateCase] = []
    current_results: list[CurrentEvidenceBundle] = []
    temporal_results: list[TemporalRelationMatrix] = []
    finding_results: list[UnifiedCaseFinding] = []
    record_results: list[RefinementRecord] = []
    inference_results: list[InferenceRecord] = []
    lineage_results: list[RefinementLineageMetadata] = []
    all_reasons: set[str] = set()

    member_by_id = {item.member_id: item for item in case.members}
    proposals_by_parent: dict[str, list[ValidatedRefinementProposal]] = {}
    for item in sorted(validated, key=_proposal_key):
        proposals_by_parent.setdefault(
            item.proposal.parent_member_id, []
        ).append(item)

    for parent_member_id in sorted(proposals_by_parent):
        group = proposals_by_parent[parent_member_id]
        parent_member = member_by_id[parent_member_id]
        parent_assessment = assessments[parent_member.assessment_id]
        children_assessments = [
            _make_child_assessment(
                parent=parent_assessment,
                proposal=item,
                ordinal=ordinal,
            )
            for ordinal, item in enumerate(group)
        ]
        children_members = [
            _make_child_member(item) for item in children_assessments
        ]
        refined_case = _make_refined_case(
            parent_case=case,
            parent=parent_member,
            children=children_members,
        )

        needs_full_reacquisition = any(
            item.requires_current_reacquisition for item in group
        )
        current_mode: Literal[
            "filtered_complete_reuse", "full_reacquisition", "failed"
        ]
        current_errors: set[str] = set()
        if not needs_full_reacquisition and len(children_members) == 1:
            try:
                refined_current = _rebind_filtered_current_evidence(
                    refined_case=refined_case,
                    parent=parent_member,
                    child=children_members[0],
                    previous=current_evidence,
                )
                current_errors.update(
                    _current_binding_errors(
                        bundle=refined_current,
                        refined_case=refined_case,
                        child_ids={children_members[0].member_id},
                    )
                )
                if current_errors:
                    refined_current = _failed_current_evidence(
                        refined_case=refined_case,
                        previous=current_evidence,
                        reason_codes=tuple(current_errors),
                    )
                    current_mode = "failed"
                else:
                    current_mode = "filtered_complete_reuse"
            except (KeyError, ValueError):
                current_errors.add("refinement_filtered_reuse_failed")
                refined_current = _failed_current_evidence(
                    refined_case=refined_case,
                    previous=current_evidence,
                    reason_codes=tuple(current_errors),
                )
                current_mode = "failed"
        elif current_reacquirer is None:
            current_errors.add("refinement_complete_substrate_reacquirer_missing")
            refined_current = _failed_current_evidence(
                refined_case=refined_case,
                previous=current_evidence,
                reason_codes=tuple(current_errors),
            )
            current_mode = "failed"
        else:
            try:
                reacquired = current_reacquirer.reacquire_current(
                    parent_case=case,
                    refined_case=refined_case,
                    parent=parent_member,
                    child_assessments=children_assessments,
                    proposals=group,
                    previous_evidence=current_evidence,
                )
            except Exception:  # host adapters convert provider/substrate failure
                reacquired = CurrentEvidenceReacquisition(
                    current_evidence=None,
                    substrate_kind="unavailable",
                    complete=False,
                    reason_codes=("refinement_current_reacquirer_failed",),
                )
            if not isinstance(reacquired, CurrentEvidenceReacquisition):
                reacquired = CurrentEvidenceReacquisition(
                    current_evidence=None,
                    substrate_kind="unavailable",
                    complete=False,
                    reason_codes=("refinement_current_reacquisition_invalid",),
                )
            if not reacquired.complete:
                current_errors.update(
                    reacquired.reason_codes
                    or ("refinement_current_reacquisition_gap",)
                )
            if reacquired.substrate_kind == "unavailable":
                current_errors.add("refinement_complete_substrate_unavailable")
            if reacquired.current_evidence is None:
                current_errors.add("refinement_current_evidence_missing")
            elif not current_errors:
                current_errors.update(
                    _current_binding_errors(
                        bundle=reacquired.current_evidence,
                        refined_case=refined_case,
                        child_ids={item.member_id for item in children_members},
                    )
                )
            if current_errors or reacquired.current_evidence is None:
                refined_current = _failed_current_evidence(
                    refined_case=refined_case,
                    previous=current_evidence,
                    reason_codes=tuple(current_errors),
                )
                current_mode = "failed"
            else:
                refined_current = _with_current_dependency(
                    reacquired.current_evidence,
                    mode="full_reacquisition",
                )
                current_mode = "full_reacquisition"
        children_assessments = [
            rebind_child_git_need(child, parent_current=current_evidence, child_current=refined_current,
                                  child_member_id=member.member_id)
            for child, member in zip(children_assessments, children_members, strict=True)
        ]
        children_members = [_make_child_member(item) for item in children_assessments]
        refined_case = _make_refined_case(parent_case=case, parent=parent_member, children=children_members)
        assessment_results.extend(children_assessments)
        member_results.extend(children_members)
        case_results.append(refined_case)
        need_binding_invalid = any(child.git_need.status == "invalid" for child in children_assessments)
        if need_binding_invalid:
            reason = "refinement_git_need_current_binding_invalid"
            current_errors.add(reason)
            # Current source facts can remain complete even when the proposed
            # child's historical question cannot bind to retained evidence.
            # Preserve that distinction while closing this required host step
            # before another member's affirmative need could route the case.
            payload = refined_current.model_dump(mode="python", exclude={"artifact_digest"})
            payload["dependencies"] = sorted(
                [*refined_current.dependencies, CapabilityDependency(
                    capability="refinement_git_need_binding",
                    requirement_mode="required",
                    status="failed",
                    reason_codes=[reason],
                )],
                key=lambda item: item.capability,
            )
            payload["artifact_digest"] = artifact_digest(payload)
            refined_current = CurrentEvidenceBundle.model_validate(payload)
        refined_assessments = [
            *(
                assessments[item.assessment_id]
                for item in refined_case.members
                if item.member_id
                not in {child.member_id for child in children_members}
            ),
            *children_assessments,
        ]
        refined_assessments.sort(key=lambda item: item.assessment_id)

        git_refs: tuple[ImmutableArtifactRef, ...] = ()
        temporal_matrix: TemporalRelationMatrix | None = None
        group_reasons = set(current_errors)
        finding: UnifiedCaseFinding
        group_status: Literal["complete", "review"] = "review"
        if current_mode == "failed" or need_binding_invalid:
            finding = _failed_finding(
                case=refined_case, reason_codes=tuple(group_reasons)
            )
        else:
            supplied_git_dependencies: Sequence[CapabilityDependency] = ()
            if git_reacquirer is None:
                group_reasons.add("refinement_git_reacquirer_missing")
                git_result = None
            else:
                try:
                    git_result = git_reacquirer.reacquire_git(
                        parent_case=case,
                        refined_case=refined_case,
                        assessments=refined_assessments,
                        current_evidence=refined_current,
                        proposals=group,
                    )
                except Exception:
                    git_result = GitReacquisition(
                        complete=False,
                        reason_codes=("refinement_git_reacquirer_failed",),
                    )
            if git_result is not None:
                if not isinstance(git_result, GitReacquisition):
                    group_reasons.add("refinement_git_reacquisition_invalid")
                else:
                    inference_results.extend(git_result.inference_records)
                    supplied_git_dependencies = git_result.dependencies
                    try:
                        git_refs, temporal_matrix = _validate_git_reacquisition(
                            value=git_result, refined_case=refined_case
                        )
                    except ValueError:
                        group_reasons.add("refinement_git_reacquisition_invalid")
                    if not git_result.complete:
                        group_reasons.update(
                            git_result.reason_codes
                            or ("refinement_git_reroute_incomplete",)
                        )
                    if temporal_matrix is not None:
                        temporal_results.append(temporal_matrix)

            try:
                refined_current = _with_git_reroute_dependencies(
                    refined_current,
                    supplied=supplied_git_dependencies,
                    complete=not group_reasons,
                    reason_codes=tuple(group_reasons),
                )
            except ValueError:
                group_reasons.add("refinement_git_dependency_invalid")
                refined_current = _with_git_reroute_dependencies(
                    refined_current,
                    supplied=(),
                    complete=False,
                    reason_codes=tuple(group_reasons),
                )

            if group_reasons:
                finding = _failed_finding(
                    case=refined_case, reason_codes=tuple(group_reasons)
                )
            elif verifier is None:
                group_reasons.add("round_one_verifier_missing")
                finding = _failed_finding(
                    case=refined_case, reason_codes=tuple(group_reasons)
                )
            else:
                try:
                    verification = verifier.verify_round_one_case(
                        case=refined_case,
                        assessments=refined_assessments,
                        current_evidence=refined_current,
                        temporal_matrix=temporal_matrix,
                        verification_round=1,
                    )
                except Exception:
                    verification = RoundOneVerification(
                        finding=None,
                        reason_codes=("round_one_verifier_failed",),
                    )
                if not isinstance(verification, RoundOneVerification):
                    verification = RoundOneVerification(
                        finding=None,
                        reason_codes=("round_one_verifier_result_invalid",),
                    )
                if verification.inference_record is not None:
                    inference_results.append(verification.inference_record)
                if verification.finding is None:
                    group_reasons.update(
                        verification.reason_codes
                        or ("round_one_verifier_failed",)
                    )
                    finding = _failed_finding(
                        case=refined_case, reason_codes=tuple(group_reasons)
                    )
                else:
                    finding_errors = _finding_errors(
                        finding=verification.finding,
                        case=refined_case,
                        assessments=refined_assessments,
                        current_evidence=refined_current,
                        temporal_matrix=temporal_matrix,
                    )
                    if finding_errors:
                        group_reasons.update(finding_errors)
                        finding = _failed_finding(
                            case=refined_case,
                            reason_codes=tuple(group_reasons),
                        )
                    else:
                        finding = verification.finding
                        if finding.status != "complete":
                            group_reasons.update(
                                finding.failure_reasons
                                or ["round_one_verifier_failed"]
                            )
                        if finding.refinement_proposals:
                            group_reasons.add("second_refinement_forbidden")
                        if not group_reasons:
                            group_status = "complete"

        current_results.append(refined_current)
        if group_status == "complete":
            record_reasons = {
                f"current_evidence:{current_mode}",
                "refinement_git_rerouted",
                "round_one_reverified",
            }
        else:
            record_reasons = group_reasons or {"refinement_round_one_failed"}
        finding_results.append(finding)
        record = _make_refinement_record(
            parent_case=case,
            parent=parent_member,
            children=children_members,
            current_evidence=refined_current,
            git_bundle_refs=git_refs,
            finding=finding,
            change_kinds=[
                change
                for item in group
                for change in item.proposal.change_kinds
            ],
            status=group_status,
            reason_codes=tuple(record_reasons),
        )
        record_results.append(record)
        lineage_results.append(
            RefinementLineageMetadata(
                parent_case_id=case.case_id,
                parent_member_id=parent_member.member_id,
                child_case_id=refined_case.case_id,
                child_assessment_ids=tuple(
                    sorted(item.assessment_id for item in children_assessments)
                ),
                child_member_ids=tuple(
                    sorted(item.member_id for item in children_members)
                ),
                current_evidence_bundle_id=refined_current.bundle_id,
                current_evidence_mode=current_mode,
                change_kinds=tuple(
                    sorted(
                        {
                            change
                            for item in group
                            for change in item.proposal.change_kinds
                        }
                    )
                ),
                requires_git_reroute=True,
            )
        )
        if group_status != "complete":
            all_reasons.update(record.reason_codes)

    return RefinementExecutionResult(
        validated=tuple(sorted(validated, key=_proposal_key)),
        child_assessments=tuple(
            sorted(assessment_results, key=lambda item: item.assessment_id)
        ),
        child_members=tuple(
            sorted(member_results, key=lambda item: item.member_id)
        ),
        child_cases=tuple(sorted(case_results, key=lambda item: item.case_id)),
        current_evidence_bundles=tuple(
            sorted(current_results, key=lambda item: item.bundle_id)
        ),
        temporal_matrices=tuple(
            sorted(temporal_results, key=lambda item: item.matrix_id)
        ),
        findings=tuple(sorted(finding_results, key=lambda item: item.finding_id)),
        refinement_records=tuple(
            sorted(record_results, key=lambda item: item.refinement_id)
        ),
        inference_records=tuple(inference_results),
        lineage=tuple(
            sorted(lineage_results, key=lambda item: item.parent_member_id)
        ),
        status="review" if all_reasons else "complete",
        reason_codes=tuple(sorted(all_reasons)),
    )


__all__ = [
    "CurrentEvidenceReacquirer",
    "CurrentEvidenceReacquisition",
    "GitReacquirer",
    "GitReacquisition",
    "HostRoundOneVerifier",
    "Reacquirer",
    "RefinementEvidence",
    "RefinementExecutionResult",
    "RefinementLineageMetadata",
    "RefinementRoundResult",
    "RoundOneVerification",
    "RoundOneVerifier",
    "ValidatedRefinementProposal",
    "execute_refinement_round",
    "run_refinement_round",
    "validate_refinement_proposals",
]
