"""Frozen, deterministic canonical policy for the unified layer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from repo_constraint_memory_v3.adjudication import (
    classify_execution_risk,
    delivery_policy_for,
)
from repo_constraint_memory_v3.repository_paths import is_nonaccepted_directory_marker
from repo_constraint_memory_v3.unified.ids import (
    make_canonical_claim_id,
    seal_payload,
    stable_id,
)
from repo_constraint_memory_v3.unified.git_evidence import GitTemporalBundle
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CanonicalClaim,
    CanonicalEvidenceRef,
    ChannelAssessment,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    MemberDisposition,
    ReviewClass,
    TemporalRelationMatrix,
    UnifiedCaseFinding,
    UnifiedMemberFinding,
    UNIFIED_POLICY_VERSION,
)


_AUTHORITY_RANK = {
    "machine_enforced": 0,
    "direct_normative": 1,
    "current_population": 2,
    "descriptive_typed": 3,
    "weak_or_unknown": 4,
}

_NON_FACTUAL_CHANNEL_AUDIT_REASONS = {
    "complete_population_and_falsification",
    "local_quality_axes_passed",
    "pre_calibration_non_delivery",
    "pre_calibration_review_only",
    "post_falsification_refinement",
    "refinement_requires_calibration",
}

_CURRENT_SNAPSHOT_ORIGIN_KINDS = {
    "explicit_span",
    "configuration_atom",
    "command_atom",
    "workflow_atom",
    "observation",
    "population_member",
}
# ``ineligible`` is an audit label for a member outside the candidate's typed
# population predicate, not positive evidence that the candidate is false.
# Its complete count remains in PopulationDenominator, while only an explicit
# current-snapshot counterexample may serve as a canonical rejection basis.
_CURRENT_REJECTION_ROLES = {"counterevidence"}
CANONICAL_IDENTITY_COLLISION_REASON = (
    "canonical_identity_cross_case_collision"
)
_DIRECTORY_MARKER_SCOPE_REASON = "directory_marker_scope_requires_refinement"


@dataclass(frozen=True)
class CasePolicyResult:
    claims: tuple[CanonicalClaim, ...]
    dispositions: tuple[MemberDisposition, ...]
    fast_path: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalPolicyAggregation:
    claims: tuple[CanonicalClaim, ...]
    dispositions: tuple[MemberDisposition, ...]
    collision_claim_ids: tuple[str, ...]
    collision_member_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


class CanonicalPolicyAccumulator:
    """Combine per-case policy projections without lossy ID overwrites.

    ``CanonicalClaim.case_id`` cannot truthfully represent members from two
    active cases.  A cross-case identity collision therefore fails closed: the
    ambiguous claim is omitted, every contributing member remains visible as
    review, and the caller can expose ``reason_codes`` as a required capability
    gap.  A completed refinement may explicitly replace its superseded parent
    case because only the child remains active.
    """

    def __init__(self) -> None:
        self._claims: dict[str, CanonicalClaim] = {}
        self._dispositions: dict[str, MemberDisposition] = {}
        self._blocked_claim_ids: set[str] = set()
        self._collision_member_ids: set[str] = set()

    def _block_collision(
        self,
        *claims: CanonicalClaim,
    ) -> None:
        for claim in claims:
            self._blocked_claim_ids.add(claim.claim_id)
            self._collision_member_ids.update(claim.contributing_member_ids)
            self._claims.pop(claim.claim_id, None)

    def absorb(
        self,
        result: CasePolicyResult,
        *,
        include_claims: bool = True,
        include_claim_member_ids: set[str] | None = None,
        prefer_dispositions: bool = False,
        supersedes_case_id: str | None = None,
    ) -> None:
        included_claim_ids: set[str] = set()
        if include_claims:
            for claim in result.claims:
                if (
                    include_claim_member_ids is not None
                    and not include_claim_member_ids.intersection(
                        claim.contributing_member_ids
                    )
                ):
                    continue
                if claim.claim_id in self._blocked_claim_ids:
                    self._block_collision(claim)
                    continue
                existing = self._claims.get(claim.claim_id)
                if existing is None:
                    self._claims[claim.claim_id] = claim
                elif existing == claim:
                    included_claim_ids.add(claim.claim_id)
                    continue
                elif (
                    supersedes_case_id is not None
                    and existing.case_id == supersedes_case_id
                    and claim.case_id != existing.case_id
                ):
                    self._claims[claim.claim_id] = claim
                else:
                    self._block_collision(existing, claim)
                    continue
                if claim.claim_id in self._claims:
                    included_claim_ids.add(claim.claim_id)

        for disposition in result.dispositions:
            disposition = disposition.model_copy(
                update={
                    "canonical_claim_ids": sorted(
                        claim_id
                        for claim_id in disposition.canonical_claim_ids
                        if claim_id in included_claim_ids
                        and claim_id in self._claims
                        and claim_id not in self._blocked_claim_ids
                    )
                }
            )
            existing = self._dispositions.get(disposition.member_id)
            if existing is None:
                self._dispositions[disposition.member_id] = disposition
                continue
            chosen = disposition if prefer_dispositions else existing
            self._dispositions[disposition.member_id] = chosen.model_copy(
                update={
                    "canonical_claim_ids": sorted(
                        {
                            *existing.canonical_claim_ids,
                            *disposition.canonical_claim_ids,
                        }
                        - self._blocked_claim_ids
                    ),
                    "reason_codes": sorted(
                        {*existing.reason_codes, *disposition.reason_codes}
                    ),
                }
            )

    def supersede_member(
        self,
        *,
        parent_member_id: str,
        child_member_ids: Sequence[str],
    ) -> None:
        replacement_claim_ids = sorted(
            claim.claim_id
            for claim in self._claims.values()
            if set(claim.contributing_member_ids).intersection(child_member_ids)
        )
        self._claims = {
            claim_id: claim
            for claim_id, claim in self._claims.items()
            if parent_member_id not in claim.contributing_member_ids
        }
        previous = self._dispositions[parent_member_id]
        self._dispositions[parent_member_id] = previous.model_copy(
            update={
                "disposition": "superseded_by_refinement",
                "canonical_claim_ids": replacement_claim_ids,
                "reason_codes": ["refinement_round_one_complete"],
            }
        )

    def finalize(self) -> CanonicalPolicyAggregation:
        claim_ids = set(self._claims)
        dispositions: list[MemberDisposition] = []
        for member_id in sorted(self._dispositions):
            disposition = self._dispositions[member_id]
            if member_id in self._collision_member_ids:
                disposition = disposition.model_copy(
                    update={
                        "disposition": "review",
                        "canonical_claim_ids": [],
                        "reason_codes": [CANONICAL_IDENTITY_COLLISION_REASON],
                    }
                )
            else:
                disposition = disposition.model_copy(
                    update={
                        "canonical_claim_ids": sorted(
                            set(disposition.canonical_claim_ids) & claim_ids
                        )
                    }
                )
            dispositions.append(disposition)
        return CanonicalPolicyAggregation(
            claims=tuple(sorted(self._claims.values(), key=lambda item: item.claim_id)),
            dispositions=tuple(dispositions),
            collision_claim_ids=tuple(sorted(self._blocked_claim_ids)),
            collision_member_ids=tuple(sorted(self._collision_member_ids)),
            reason_codes=(
                (CANONICAL_IDENTITY_COLLISION_REASON,)
                if self._blocked_claim_ids
                else ()
            ),
        )


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def is_explicit_fast_path(
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
) -> bool:
    """Return whether the host-owned explicit-local-positive shortcut applies."""

    members = [assessments[item.assessment_id] for item in case.members]
    if case.grouping_status != "complete" or any(
        relation.relation == "uncertain" for relation in case.relations
    ):
        return False
    if not members or any(item.channel != "explicit" for item in members):
        return False
    if any(
        item.integrity != "valid"
        or item.git_need.status in {"affirmative", "invalid"}
        for item in members
    ):
        return False
    if any(item.normalized_status != "locally_supported" for item in members):
        return False
    if any(
        is_nonaccepted_directory_marker(scope)
        for item in members
        for scope in item.semantics.applies_to
    ):
        return False
    if current_evidence.integrity != "valid" or current_evidence.status != "complete":
        return False
    if any(item.role == "counterevidence" for item in current_evidence.items):
        return False
    if any(
        dependency.requirement_mode == "required"
        and dependency.status not in {"complete", "bounded_no_match", "not_required"}
        for dependency in current_evidence.dependencies
    ):
        return False
    required_axes = {
        "host_grounding",
        "coding_relevance",
        "evidence_entailment",
        "scope_condition_support",
        "ambiguity",
    }
    return all(
        required_axes <= set(item.axis_findings)
        and all(item.axis_findings[axis] == "pass" for axis in required_axes)
        for item in members
    )


def _equivalence_groups(
    case: CandidateCase,
    finding: UnifiedCaseFinding | None,
) -> list[list[str]]:
    member_ids = [item.member_id for item in case.members]
    union = _UnionFind(member_ids)
    for relation in case.relations:
        if relation.relation == "equivalent":
            union.union(relation.left_member_id, relation.right_member_id)
    if finding is not None:
        for relation in finding.member_relations:
            if relation.relation == "equivalent":
                union.union(relation.left_member_id, relation.right_member_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for member_id in member_ids:
        groups[union.find(member_id)].append(member_id)
    return sorted((sorted(group) for group in groups.values()), key=lambda item: item[0])


def _choose_base(
    member_ids: Sequence[str],
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
) -> tuple[str, ChannelAssessment]:
    member_by_id = {item.member_id: item for item in case.members}

    def rank(member_id: str) -> tuple[int, int, int, str]:
        assessment = assessments[member_by_id[member_id].assessment_id]
        scope_specificity = sum(len(value.replace("*", "")) for value in assessment.semantics.applies_to)
        condition_specificity = len(assessment.semantics.condition or "")
        return (
            _AUTHORITY_RANK[assessment.source_authority],
            -scope_specificity,
            -condition_specificity,
            member_id,
        )

    member_id = min(member_ids, key=rank)
    return member_id, assessments[member_by_id[member_id].assessment_id]


def _canonical_evidence(
    *,
    member_ids: set[str],
    current_evidence: CurrentEvidenceBundle,
    cited_ids: set[str] | None,
    role: str,
) -> list[CanonicalEvidenceRef]:
    origin_by_id = {item.origin_id: item for item in current_evidence.origins}
    result: list[CanonicalEvidenceRef] = []
    seen_origins: set[str] = set()
    for item in sorted(current_evidence.items, key=lambda value: value.evidence_id):
        allowed_roles = (
            {"support"}
            if role == "support"
            else _CURRENT_REJECTION_ROLES
        )
        if item.role not in allowed_roles or not member_ids.intersection(
            item.member_ids
        ):
            continue
        if (
            cited_ids is not None
            and item.evidence_id not in cited_ids
            and item.artifact_ref.stable_id not in cited_ids
        ):
            continue
        if item.origin_id in seen_origins:
            continue
        origin = origin_by_id[item.origin_id]
        seen_origins.add(item.origin_id)
        kind = (
            "explicit"
            if origin.origin_kind
            in {
                "explicit_span",
                "configuration_atom",
                "command_atom",
                "workflow_atom",
            }
            else "current_observation"
        )
        result.append(
            CanonicalEvidenceRef(
                evidence_id=item.evidence_id,
                origin_id=item.origin_id,
                role="support" if role == "support" else "counterevidence",
                kind=kind,
                path=origin.path,
                line_start=origin.line_start,
                line_end=origin.line_end,
                artifact_ref=item.artifact_ref,
            )
        )
    return result


def _cited_current_rejection_basis_refs(
    *,
    member_id: str,
    finding: UnifiedMemberFinding,
    current_evidence: CurrentEvidenceBundle,
) -> list[str]:
    """Require a cited, current-snapshot positive basis for ``rejected``.

    Temporal evidence is deliberately absent from this predicate.  A Git
    contradiction or retirement can force review/currentness resolution, but
    cannot itself establish that the candidate is false or inapplicable at the
    fixed current snapshot.
    """

    cited = set(finding.evidence_refs)
    origins = {item.origin_id: item for item in current_evidence.origins}
    result: list[str] = []
    for item in current_evidence.items:
        origin = origins.get(item.origin_id)
        if (
            member_id in item.member_ids
            and item.role in _CURRENT_REJECTION_ROLES
            and origin is not None
            and origin.origin_kind in _CURRENT_SNAPSHOT_ORIGIN_KINDS
            and (
                item.evidence_id in cited
                or item.artifact_ref.stable_id in cited
            )
        ):
            result.append(item.evidence_id)
    return sorted(set(result))


def _cited_current_support_basis_refs(
    *,
    member_id: str,
    finding: UnifiedMemberFinding,
    current_evidence: CurrentEvidenceBundle,
) -> list[str]:
    """Return cited current support that grounds an out-of-scope decision."""

    cited = set(finding.evidence_refs)
    origins = {item.origin_id: item for item in current_evidence.origins}
    result: list[str] = []
    for item in current_evidence.items:
        origin = origins.get(item.origin_id)
        if (
            member_id in item.member_ids
            and item.role == "support"
            and origin is not None
            and origin.origin_kind in _CURRENT_SNAPSHOT_ORIGIN_KINDS
            and (
                item.evidence_id in cited
                or item.artifact_ref.stable_id in cited
            )
        ):
            result.append(item.evidence_id)
    return sorted(set(result))


def _review_facets_for_member(
    *,
    reason_codes: Sequence[str],
    assessment: ChannelAssessment,
    finding: UnifiedMemberFinding | None,
) -> list[ReviewClass]:
    """Map host policy blockers to a stable, provider-independent taxonomy."""

    facets: set[ReviewClass] = set()
    for reason in reason_codes:
        if reason in {
            "grouping_incomplete",
            "required_capability_incomplete",
            "unified_verification_missing",
            "canonical_support_not_cited",
        }:
            facets.add("technical_or_coverage_gap")
        elif reason in {
            "unresolved_case_conflict",
            "temporal_status_unresolved",
        }:
            facets.add("conflict_or_currentness")
        elif reason in {
            "refinement_requires_round_one_verification",
            _DIRECTORY_MARKER_SCOPE_REASON,
        }:
            facets.add("refinement_or_qualifier_needed")
        elif reason == "current_snapshot_rejection_basis_missing":
            facets.add("negative_basis_insufficient")
        elif reason in {
            "fast_path_member_not_locally_supported",
            "local_rejection_requires_independent_member",
            "local_review_reason_unclosed",
        }:
            facets.add("channel_disagreement")
        elif reason == "outside_scope_rejection_not_admitted":
            facets.add("factual_uncertainty")
        elif reason == "unified_fact_axis_not_passed":
            has_qualifier = (
                assessment.semantics.condition is not None
                or assessment.semantics.exception is not None
            )
            condition_passed = finding is not None and (
                finding.condition_support == "pass"
                if has_qualifier
                else finding.condition_support == "not_applicable"
            )
            if finding is not None and (
                finding.scope_support != "pass" or not condition_passed
            ):
                facets.add("refinement_or_qualifier_needed")
            else:
                facets.add("factual_uncertainty")
        else:
            facets.add("factual_uncertainty")
    return sorted(facets)


def _canonical_temporal_evidence(
    *,
    member_ids: set[str],
    temporal_matrix: TemporalRelationMatrix | None,
    git_bundle: GitTemporalBundle | None,
    cited_ids: set[str] | None,
    role: str,
) -> list[CanonicalEvidenceRef]:
    if temporal_matrix is None or git_bundle is None:
        return []
    slice_by_id = {item.item_id: item for item in git_bundle.slices}
    support_relations = {"supports_as_written"}
    counter_relations = {
        "contradicts_as_written",
        "qualifies_scope_or_condition",
        "retires_or_replaces",
    }
    allowed = support_relations if role == "support" else counter_relations
    selected_ids = {
        item.item_id
        for item in temporal_matrix.relations
        if item.member_id in member_ids
        and item.relation in allowed
        and (cited_ids is None or item.item_id in cited_ids)
    }
    result: list[CanonicalEvidenceRef] = []
    for item_id in sorted(selected_ids):
        item = slice_by_id.get(item_id)
        if item is None:
            continue
        path = item.after_path or item.before_path
        if path is None:
            continue
        result.append(
            CanonicalEvidenceRef(
                evidence_id=item.item_id,
                origin_id=stable_id(
                    "TEMPORIGIN",
                    [item.snapshot_digest, item.commit_id, item.parent_id, path],
                ),
                role="support" if role == "support" else "counterevidence",
                kind="temporal",
                path=path,
                line_start=None,
                line_end=None,
                artifact_ref=ImmutableArtifactRef(
                    kind="git_change_slice",
                    stable_id=item.item_id,
                    snapshot_digest=item.snapshot_digest,
                    artifact_path=(
                        f"canonical/audit/git/slices/{item.item_id}.json"
                    ),
                    artifact_digest=item.artifact_digest,
                ),
            )
        )
    return result


def _member_finding_status(
    assessment: ChannelAssessment,
    finding: UnifiedMemberFinding | None,
    *,
    grouping_complete: bool,
    dependency_complete: bool,
    unresolved_conflict: bool,
    refinement_pending: bool,
    current_contradiction_basis: bool,
    current_scope_basis: bool,
) -> tuple[str, list[str]]:
    if assessment.integrity == "invalid" or assessment.normalized_status == "invalid":
        return "invalid", ["invalid_channel_assessment"]
    # CanonicalClaim permits a lexical directory marker only as review pending
    # scope refinement. This is a host-known shape fact, independent of whether
    # the verifier ran or of its semantic scope/rule verdict. Preserve the exact
    # scope; never guess a recursive glob or erase another causal blocker.
    scope_reasons = (
        [_DIRECTORY_MARKER_SCOPE_REASON]
        if any(is_nonaccepted_directory_marker(scope) for scope in assessment.semantics.applies_to)
        else []
    )
    if not grouping_complete:
        return "review", ["grouping_incomplete", *scope_reasons]
    # A route may deliberately omit the verifier when its candidate-local
    # current evidence is already incomplete. Preserve that causal blocker
    # instead of misreporting the expected missing finding as a verifier fault.
    if not dependency_complete:
        return "review", ["required_capability_incomplete", *scope_reasons]
    if finding is None:
        return "review", ["unified_verification_missing", *scope_reasons]
    if unresolved_conflict:
        return "review", ["unresolved_case_conflict", *scope_reasons]
    if refinement_pending:
        return "review", ["refinement_requires_round_one_verification", *scope_reasons]
    if scope_reasons:
        return "review", scope_reasons
    if finding.norm_relevance == "fail":
        has_qualifier = (
            assessment.semantics.condition is not None
            or assessment.semantics.exception is not None
        )
        condition_grounded = finding.condition_support == (
            "pass" if has_qualifier else "not_applicable"
        )
        # Current support can establish that the declaration exists, but a
        # semantic relevance failure is not an independently typed proof that
        # it lies outside repository coding work.  Real runs showed that review,
        # validation, configuration, and delivery norms were otherwise
        # over-rejected.  Until a positive domain-boundary verifier exists,
        # relevance failures therefore abstain; current contradictions below
        # remain the canonical terminal-negative route.
        if (
            current_scope_basis
            and finding.rule_support == "pass"
            and finding.scope_support == "pass"
            and condition_grounded
            and finding.evidence_sufficiency == "pass"
        ):
            return "review", ["outside_scope_rejection_not_admitted"]
        return "review", ["outside_scope_rejection_not_admitted"]
    if finding.rule_support == "fail":
        if current_contradiction_basis:
            return "rejected", ["current_snapshot_positive_rejection"]
        return "review", ["current_snapshot_rejection_basis_missing"]
    axes = [
        finding.norm_relevance,
        finding.rule_support,
        finding.scope_support,
        finding.evidence_sufficiency,
    ]
    condition_axis_passed = finding.condition_support == "pass" or (
        assessment.semantics.condition is None
        and assessment.semantics.exception is None
        and finding.condition_support == "not_applicable"
    )
    if any(value != "pass" for value in axes) or not condition_axis_passed:
        return "review", ["unified_fact_axis_not_passed"]
    adverse_temporal = finding.temporal_status in {
        "adverse_unresolved",
        "uncertain",
    }
    if adverse_temporal:
        return "review", ["temporal_status_unresolved"]
    if (
        assessment.normalized_status == "locally_rejected"
        and assessment.channel != "explicit"
    ):
        # Current-source local rejection remains an acceptance veto. Explicit
        # local rejection is deliberately recoverable here because that verifier
        # is conservatively calibrated, while this branch is reached only after
        # all unified factual, dependency, conflict, and temporal checks above pass.
        return "review", ["local_rejection_requires_independent_member"]
    if assessment.normalized_status == "locally_uncertain":
        # The old current-source policy's calibration/non-delivery labels are
        # preserved for channel audit but have no factual force in schema 4.
        # Only unresolved local evidence reasons must be explicitly closed by
        # the unified verifier.
        closable = set(assessment.typed_reasons) - _NON_FACTUAL_CHANNEL_AUDIT_REASONS
        closed = set(finding.closed_local_reason_codes)
        if not closable <= closed:
            return "review", ["local_review_reason_unclosed"]
    return "accepted", ["unified_fact_axes_passed"]


def _risk_and_delivery(
    assessment: ChannelAssessment,
    *,
    status: str,
    force_reference_only: bool,
) -> tuple[str, str]:
    # The shared risk classifier is deterministic and accepts an empty typed
    # evidence sequence; current evidence controls truth, not action parsing.
    risk = classify_execution_risk(
        assessment.proposed_claim_kind,
        assessment.semantics.rule,
        [],
    )
    if status != "accepted":
        delivery = "not_applicable"
    elif force_reference_only:
        delivery = "reference_only"
    else:
        delivery = delivery_policy_for(assessment.semantics.condition, risk)
    return risk, delivery


def _canonical_strength(assessment: ChannelAssessment) -> str:
    """Derive strength from current authority, independently of truth status."""

    if (
        assessment.source_authority == "machine_enforced"
        or assessment.proposed_claim_kind == "enforced_configuration"
    ):
        return "required"
    if assessment.source_authority == "current_population":
        return "observed"
    return assessment.proposed_strength


def apply_case_policy(
    *,
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
    temporal_matrix: TemporalRelationMatrix | None,
    git_bundle: GitTemporalBundle | None = None,
    finding: UnifiedCaseFinding | None,
) -> CasePolicyResult:
    """Recompute canonical status without provider calls or mutable state."""

    assessment_by_member = {
        item.member_id: assessments[item.assessment_id] for item in case.members
    }
    fast_path = is_explicit_fast_path(
        case=case,
        assessments=assessments,
        current_evidence=current_evidence,
    )
    finding_by_member = (
        {item.member_id: item for item in finding.member_findings}
        if finding is not None and finding.status == "complete"
        else {}
    )
    unresolved_conflict = bool(
        finding is not None and finding.unresolved_conflicts
    )
    refinement_pending = bool(
        finding is not None and finding.refinement_proposals
    )
    dependency_complete = all(
        item.requirement_mode != "required"
        or item.status in {"complete", "bounded_no_match", "not_required"}
        for item in current_evidence.dependencies
    )
    groups = _equivalence_groups(case, finding)
    claims: list[CanonicalClaim] = []
    status_by_member: dict[str, tuple[str, list[str]]] = {}
    review_facets_by_member: dict[str, list[ReviewClass]] = {}
    rejection_basis_by_member: dict[str, list[str]] = {}
    rejection_kind_by_member: dict[str, str] = {}

    for group in groups:
        if fast_path:
            for member_id in group:
                assessment = assessment_by_member[member_id]
                status_by_member[member_id] = (
                    ("accepted", ["explicit_local_positive_fast_path"])
                    if assessment.normalized_status == "locally_supported"
                    else ("review", ["fast_path_member_not_locally_supported"])
                )
                if status_by_member[member_id][0] == "review":
                    review_facets_by_member[member_id] = _review_facets_for_member(
                        reason_codes=status_by_member[member_id][1],
                        assessment=assessment,
                        finding=None,
                    )
        else:
            for member_id in group:
                member_finding = finding_by_member.get(member_id)
                rejection_basis_refs = (
                    _cited_current_rejection_basis_refs(
                        member_id=member_id,
                        finding=member_finding,
                        current_evidence=current_evidence,
                    )
                    if member_finding is not None
                    else []
                )
                scope_basis_refs = (
                    _cited_current_support_basis_refs(
                        member_id=member_id,
                        finding=member_finding,
                        current_evidence=current_evidence,
                    )
                    if member_finding is not None
                    else []
                )
                status_by_member[member_id] = _member_finding_status(
                    assessment_by_member[member_id],
                    member_finding,
                    grouping_complete=case.grouping_status == "complete",
                    dependency_complete=dependency_complete,
                    unresolved_conflict=unresolved_conflict,
                    refinement_pending=refinement_pending,
                    current_contradiction_basis=bool(rejection_basis_refs),
                    current_scope_basis=bool(scope_basis_refs),
                )
                member_status, member_reasons = status_by_member[member_id]
                if member_status == "review":
                    review_facets_by_member[member_id] = _review_facets_for_member(
                        reason_codes=member_reasons,
                        assessment=assessment_by_member[member_id],
                        finding=member_finding,
                    )
                elif member_status == "rejected":
                    rejection_kind_by_member[member_id] = (
                        "current_snapshot_contradiction"
                    )
                    rejection_basis_by_member[member_id] = rejection_basis_refs

        accepted_members = [
            item for item in group if status_by_member[item][0] == "accepted"
        ]
        rejected_members = [
            item for item in group if status_by_member[item][0] == "rejected"
        ]
        valid_members = [
            item for item in group if status_by_member[item][0] != "invalid"
        ]
        if not valid_members:
            continue
        if accepted_members:
            canonical_status = "accepted"
            contributing = accepted_members
        elif len(rejected_members) == len(valid_members):
            canonical_status = "rejected"
            contributing = valid_members
        else:
            canonical_status = "review"
            contributing = valid_members

        base_member_id, base = _choose_base(
            contributing, case=case, assessments=assessments
        )
        cited_ids: set[str] | None
        if fast_path:
            cited_ids = None
        else:
            cited_ids = {
                evidence_id
                for member_id in contributing
                for evidence_id in (
                    finding_by_member.get(member_id).evidence_refs
                    if finding_by_member.get(member_id) is not None
                    else []
                )
            }
        support = _canonical_evidence(
            member_ids=set(contributing),
            current_evidence=current_evidence,
            cited_ids=cited_ids,
            role="support",
        )
        counterevidence = _canonical_evidence(
            member_ids=set(group),
            current_evidence=current_evidence,
            cited_ids=cited_ids,
            role="counterevidence",
        )
        if canonical_status == "accepted" and not support:
            canonical_status = "review"
            for member_id in accepted_members:
                status_by_member[member_id] = (
                    "review",
                    ["canonical_support_not_cited"],
                )
                review_facets_by_member[member_id] = [
                    "technical_or_coverage_gap"
                ]
        support.extend(
            _canonical_temporal_evidence(
                member_ids=set(contributing),
                temporal_matrix=temporal_matrix,
                git_bundle=git_bundle,
                cited_ids=cited_ids,
                role="support",
            )
        )
        counterevidence.extend(
            _canonical_temporal_evidence(
                member_ids=set(group),
                temporal_matrix=temporal_matrix,
                git_bundle=git_bundle,
                cited_ids=cited_ids,
                role="counterevidence",
            )
        )

        force_reference = False
        canonical_strength = _canonical_strength(base)
        if canonical_strength == "observed":
            # An observed convention that conflicts with a direct requirement is
            # truthful but is never default-delivered.
            for other_member_id, other in assessment_by_member.items():
                if other_member_id in group:
                    continue
                if (
                    _canonical_strength(other) == "required"
                    and other.source_authority
                    in {"direct_normative", "machine_enforced"}
                ):
                    relation_pairs = {
                        frozenset((item.left_member_id, item.right_member_id))
                        for item in (finding.member_relations if finding else [])
                        if item.relation == "conflicting"
                    }
                    if any(
                        frozenset((member_id, other_member_id)) in relation_pairs
                        for member_id in group
                    ):
                        force_reference = True
                        break
        risk, delivery = _risk_and_delivery(
            base,
            status=canonical_status,
            force_reference_only=force_reference,
        )
        claim_id = make_canonical_claim_id(
            case.snapshot_digest, base.semantics.model_dump(mode="json")
        )
        status_driving_members = [
            member_id
            for member_id in contributing
            if status_by_member[member_id][0] == canonical_status
        ]
        reasons = {
            reason
            for member_id in status_driving_members
            for reason in status_by_member[member_id][1]
        }
        if force_reference:
            reasons.add("conflicting_observed_reference_only")
        review_facets = sorted(
            {
                facet
                for member_id in group
                if status_by_member[member_id][0] == "review"
                for facet in review_facets_by_member.get(member_id, [])
            }
        )
        rejection_kind: str | None = None
        if canonical_status == "rejected":
            member_rejection_kinds = {
                rejection_kind_by_member[member_id]
                for member_id in contributing
            }
            if member_rejection_kinds != {"current_snapshot_contradiction"}:
                raise ValueError("canonical rejection lacks a current contradiction")
            rejection_kind = "current_snapshot_contradiction"
        rejection_basis_refs = sorted(
            {
                evidence_id
                for member_id in contributing
                if rejection_kind_by_member.get(member_id) == rejection_kind
                for evidence_id in rejection_basis_by_member.get(member_id, [])
            }
            & {
                item.evidence_id
                for item in (
                    counterevidence
                )
                if item.kind != "temporal"
            }
        )
        payload = {
            "claim_id": claim_id,
            "snapshot_digest": case.snapshot_digest,
            "profile_digest": case.profile_digest,
            "case_id": case.case_id,
            "semantics": base.semantics,
            "epistemic_status": canonical_status,
            "claim_kind": base.proposed_claim_kind,
            "normative_strength": canonical_strength,
            "delivery_policy": delivery,
            "execution_risk": risk,
            "support": support,
            "counterevidence": counterevidence,
            "conflicts_with": [],
            "contributing_member_ids": sorted(contributing),
            "refinement_parent_id": (
                base.parent_candidate_id if base.candidate_version else None
            ),
            "reason_codes": sorted(reasons) or ["unified_policy_no_positive_basis"],
            "review_facets": review_facets if canonical_status == "review" else [],
            "rejection_kind": (
                rejection_kind if canonical_status == "rejected" else None
            ),
            "rejection_basis_refs": (
                rejection_basis_refs if canonical_status == "rejected" else []
            ),
            "dependency_status": (
                "independent"
                if fast_path
                else "complete"
                if dependency_complete
                else "incomplete"
            ),
        }
        claims.append(CanonicalClaim.model_validate(seal_payload(payload)))

    # Add symmetric conflict links without changing claim identity.
    conflict_member_pairs = {
        frozenset((item.left_member_id, item.right_member_id))
        for item in (finding.member_relations if finding is not None else [])
        if item.relation == "conflicting"
    }
    claim_group_by_id = {
        claim.claim_id: set(claim.contributing_member_ids) for claim in claims
    }
    conflict_claims: dict[str, set[str]] = defaultdict(set)
    for left in claims:
        for right in claims:
            if left.claim_id >= right.claim_id:
                continue
            if any(
                frozenset((left_member, right_member)) in conflict_member_pairs
                for left_member in claim_group_by_id[left.claim_id]
                for right_member in claim_group_by_id[right.claim_id]
            ):
                conflict_claims[left.claim_id].add(right.claim_id)
                conflict_claims[right.claim_id].add(left.claim_id)
    sealed_claims: list[CanonicalClaim] = []
    for claim in claims:
        payload = claim.model_dump(mode="python", exclude={"artifact_digest"})
        payload["conflicts_with"] = sorted(conflict_claims[claim.claim_id])
        sealed_claims.append(CanonicalClaim.model_validate(seal_payload(payload)))

    claim_ids_by_member: dict[str, list[str]] = defaultdict(list)
    for claim in sealed_claims:
        for member_id in claim.contributing_member_ids:
            claim_ids_by_member[member_id].append(claim.claim_id)
    dispositions: list[MemberDisposition] = []
    for member_id in sorted(assessment_by_member):
        status, reasons = status_by_member.get(
            member_id, ("invalid", ["invalid_channel_assessment"])
        )
        claim_ids = sorted(claim_ids_by_member.get(member_id, []))
        if status == "accepted":
            disposition = "adopted_as_support"
        elif status == "rejected":
            disposition = "rejected"
        elif status == "invalid":
            disposition = "invalid"
        else:
            disposition = "review"
        dispositions.append(
            MemberDisposition(
                member_id=member_id,
                case_id=case.case_id,
                disposition=disposition,
                canonical_claim_ids=claim_ids,
                reason_codes=sorted(set(reasons)),
            )
        )
    return CasePolicyResult(
        claims=tuple(sorted(sealed_claims, key=lambda item: item.claim_id)),
        dispositions=tuple(dispositions),
        fast_path=fast_path,
        reason_codes=(
            ("explicit_local_positive_fast_path",)
            if fast_path
            else (f"policy:{UNIFIED_POLICY_VERSION}",)
        ),
    )


__all__ = [
    "CANONICAL_IDENTITY_COLLISION_REASON",
    "CanonicalPolicyAccumulator",
    "CanonicalPolicyAggregation",
    "CasePolicyResult",
    "apply_case_policy",
    "is_explicit_fast_path",
]
