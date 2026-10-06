"""Lossless adapters from the two discovery channels into unified assessments.

The adapters deliberately preserve the channel-owned status and policy.  The
normalized status is only a routing view; canonical status is assigned later by
the frozen unified policy.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import ValidationError
from repo_constraint_memory_v3.temporal_need import aggregate_git_needs, unassessed_git_need

from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.explicit_verification import (
    EXPLICIT_VERIFICATION_POLICY_VERSION,
    ExplicitVerificationResult,
    explicit_candidate_root_id,
)
from repo_constraint_memory_v3.models import ClaimKind, NormativeStrength
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.observations.models import (
    PythonFileShard,
    SourceLocation,
)
from repo_constraint_memory_v3.semantic.falsification import (
    falsification_ineligibility_reason,
)
from repo_constraint_memory_v3.semantic.models import PopulationMember, NormHypothesis, PopulationArtifact, FalsificationResult, PopulationMemberPacket
from repo_constraint_memory_v3.semantic.validation import (
    CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
    CurrentSourceBundle,
)
from repo_constraint_memory_v3.unified.ids import (
    content_digest,
    make_assessment_id,
    seal_payload,
    stable_id,
)
from repo_constraint_memory_v3.unified.current_evidence import (
    EvidenceContribution,
    PopulationEvidenceContribution,
)
from repo_constraint_memory_v3.unified.grouping import MemberBlockingAnchors
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateSemantics,
    CapabilityDependency,
    ChannelAssessment,
    ImmutableArtifactRef,
    MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS,
    PopulationDenominator,
    UNIFIED_CONTRACT_VERSION,
)


_NEGATIVE_RE = re.compile(
    r"\b(?:do\s+not|does\s+not|must\s+not|should\s+not|shall\s+not|"
    r"may\s+not|can(?:not|'t)|is\s+not\s+allowed|are\s+not\s+allowed|"
    r"never|avoid|forbid(?:den)?|prohibit(?:ed)?|disallow(?:ed)?)\b|"
    r"\b(?:don't|doesn't|mustn't|shouldn't|shan't)\b",
    re.IGNORECASE,
)
_LEADING_DIRECTIVE_RE = re.compile(
    r"^(?:(?:all|the|repository|contributors?|code|tests?)\s+)?"
    r"(?:(?:must|shall|should|needs?\s+to|is\s+required\s+to)\s+)",
    re.IGNORECASE,
)


TERMINAL_INVALID_SCOPE_ATTRITION = "channel_assessment_omitted_invalid_scope"
ASSESSMENT_CONSTRUCTION_ATTRITION = (
    "channel_assessment_omitted_construction_failure"
)
UNIFIED_CHANNEL_ADAPTER_VERSION = "unified-channel-assessment-adapter-1.5"


@dataclass(frozen=True)
class ChannelAssessmentOmission:
    """One candidate root that cannot safely become a semantic assessment.

    The raw candidate and its host decision remain losslessly available in the
    unified upstream envelope and explicit review artifact.  This record is a
    deterministic routing/accounting fact only; it must never carry rewritten
    candidate semantics.
    """

    channel: Literal["explicit", "implicit_current"]
    candidate_root_id: str
    candidate_version: int
    source_authority: str
    reason_code: Literal[
        "channel_assessment_omitted_invalid_scope",
        "channel_assessment_omitted_construction_failure",
    ]
    affects_completeness: bool


@dataclass(frozen=True)
class ChannelAssessmentBuildResult:
    assessments: tuple[ChannelAssessment, ...]
    omissions: tuple[ChannelAssessmentOmission, ...]


@dataclass(frozen=True)
class CaseEvidenceInputs:
    explicit_evidence: tuple[EvidenceContribution, ...]
    population_evidence: tuple[PopulationEvidenceContribution, ...]
    denominators: tuple[PopulationDenominator, ...]
    population_refs: tuple[ImmutableArtifactRef, ...]
    falsification_refs: tuple[ImmutableArtifactRef, ...]
    scope_condition_refs: tuple[ImmutableArtifactRef, ...]
    dependencies: tuple[CapabilityDependency, ...]
    observation_evidence: tuple[EvidenceContribution, ...] = ()


@dataclass(frozen=True)
class _ResolvedLocation:
    path: str
    source_content_digest: str
    line_start: int | None
    line_end: int | None
    excerpt_digest: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class _ObservationAnchorMatch:
    shard_id: str
    shard_digest: str
    path: str
    source_content_digest: str
    line_start: int | None
    line_end: int | None
    excerpt_digest: str
    exact_values: tuple[str, ...]
    shard_status: str
    shard_incomplete_reasons: tuple[str, ...]

    @property
    def origin_key(self) -> tuple[object, ...]:
        return (
            self.path,
            self.line_start,
            self.line_end,
            self.source_content_digest,
        )


@dataclass(frozen=True)
class ObservationAnchorIndex:
    """Deterministic exact lookup over already-supplied Python shards."""

    snapshot_digest: str
    status: str
    incomplete_reasons: tuple[str, ...]
    by_symbol: Mapping[str, tuple[_ObservationAnchorMatch, ...]]
    by_path: Mapping[str, _ObservationAnchorMatch]


@dataclass(frozen=True)
class ObservationAnchorSupplement:
    evidence: tuple[EvidenceContribution, ...]
    dependency: CapabilityDependency
    matched_origins: int
    omitted_origins: int


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def infer_polarity(rule: str) -> str:
    """Extract the directive direction without treating it as evidence.

    The legacy channel schemas did not expose a direction field.  Schema 4.1
    therefore binds this deterministic extractor in the unified profile.  It is
    intentionally conservative and only recognizes explicit negative markers.
    """

    return "negative" if _NEGATIVE_RE.search(rule) else "positive"


def normalized_action(rule: str) -> str:
    value = _normal(rule)
    value = _LEADING_DIRECTIVE_RE.sub("", value)
    value = _NEGATIVE_RE.sub(" ", value)
    value = re.sub(r"[^\w./:*+-]+", " ", value)
    return " ".join(value.split()) or _normal(rule)


def candidate_semantics(
    *,
    rule: str,
    applies_to: Sequence[str],
    condition: str | None,
    exception: str | None = None,
) -> CandidateSemantics:
    return CandidateSemantics(
        rule=rule.strip(),
        normalized_action=normalized_action(rule),
        polarity=infer_polarity(rule),  # type: ignore[arg-type]
        applies_to=sorted(set(applies_to)) or ["**"],
        condition=condition.strip() if condition and condition.strip() else None,
        exception=exception.strip() if exception and exception.strip() else None,
    )


def _ref(
    *,
    kind: str,
    stable_id_value: str,
    snapshot_digest: str,
    artifact_path: str,
    payload: object,
    artifact_digest_value: str | None = None,
) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,  # type: ignore[arg-type]
        stable_id=stable_id_value,
        snapshot_digest=snapshot_digest,
        artifact_path=artifact_path,
        artifact_digest=artifact_digest_value or content_digest(payload),
    )


def _source_authority(decision: CandidateDecision) -> str:
    if "executable_enforcement" in decision.acceptance_basis:
        return "machine_enforced"
    if {
        "direct_modal_text",
        "direct_instruction_text",
    } & set(decision.acceptance_basis):
        return "direct_normative"
    if decision.derived_kind in {
        "enforced_configuration",
        "operational_fact",
        "command_recipe",
        "workflow_fact",
    }:
        return "descriptive_typed"
    return "weak_or_unknown"


def _build_explicit_assessment_result(
    *,
    decisions: Sequence[CandidateDecision],
    verification: ExplicitVerificationResult,
    snapshot_digest: str,
    profile_digest: str,
) -> ChannelAssessmentBuildResult:
    """Freeze safe explicit roots and close unsafe roots candidate-locally."""

    verification_by_root = {
        item.candidate_root_id: item for item in verification.decisions
    }
    assessments: list[ChannelAssessment] = []
    omissions: list[ChannelAssessmentOmission] = []
    for decision in decisions:
        candidate = decision.raw_candidate
        if candidate is None:
            # A bounded abstention is an event/funnel fact, not a candidate.
            continue
        root_id = explicit_candidate_root_id(decision.candidate_fingerprint)
        verified = verification_by_root.get(root_id)
        complete_grounding = len(decision.grounded) == len(candidate.evidence)
        integrity = "valid" if complete_grounding else "invalid"
        if integrity == "invalid" or decision.outcome == "failed":
            normalized_status = "invalid"
        elif decision.outcome == "accepted":
            normalized_status = "locally_supported"
        elif decision.outcome == "rejected":
            normalized_status = "locally_rejected"
        else:
            normalized_status = "locally_uncertain"

        axes: dict[str, str] = {
            "host_grounding": "pass" if complete_grounding else "fail",
        }
        if verified is None or verified.judgment is None:
            axes.update(
                {
                    "coding_relevance": "uncertain",
                    "evidence_entailment": "uncertain",
                    "scope_condition_support": "uncertain",
                    "ambiguity": "uncertain",
                }
            )
        else:
            judgment = verified.judgment
            axes.update(
                {
                    "coding_relevance": judgment.coding_relevance,
                    "evidence_entailment": judgment.evidence_entailment,
                    "scope_condition_support": judgment.scope_condition_support,
                    "ambiguity": (
                        "pass"
                        if judgment.ambiguity == "none"
                        else "fail"
                        if judgment.ambiguity == "material"
                        else "uncertain"
                    ),
                }
            )

        support_refs: list[ImmutableArtifactRef] = []
        for grounded in decision.grounded:
            evidence = grounded.evidence
            evidence_payload = evidence.model_dump(mode="json")
            evidence_id = stable_id(
                "EXPLEVID",
                [root_id, evidence.path, evidence.line_start, evidence.line_end,
                 evidence.semantic_key, evidence.quote_sha256],
            )
            support_refs.append(
                _ref(
                    kind="explicit_evidence",
                    stable_id_value=evidence_id,
                    snapshot_digest=snapshot_digest,
                    artifact_path="explicit_verification.json",
                    payload=evidence_payload,
                )
            )
        support_refs.sort(key=lambda item: item.stable_id)
        candidate_ref = _ref(
            kind="explicit_candidate",
            stable_id_value=root_id,
            snapshot_digest=snapshot_digest,
            artifact_path="explicit_verification.json",
            payload=candidate.model_dump(mode="json"),
        )
        coverage_ref = _ref(
            kind="explicit_verification",
            stable_id_value=root_id,
            snapshot_digest=snapshot_digest,
            artifact_path="explicit_verification.json",
            payload=(
                verified.model_dump(mode="json")
                if verified is not None
                else {"candidate_root_id": root_id, "missing": True}
            ),
        )
        reasons = {
            decision.reason_code,
            *(decision.verification_reason_codes or []),
            *(f"host_error:{_normal(item)[:160]}" for item in decision.errors),
        }
        if not complete_grounding:
            reasons.add("explicit_grounding_incomplete")
        known_invalid_scope = (
            not decision.host_eligible
            and decision.outcome != "accepted"
            and any(
                item.startswith(
                    "invalid repository-relative applies_to glob:"
                )
                for item in decision.errors
            )
        )
        if known_invalid_scope:
            omissions.append(
                ChannelAssessmentOmission(
                    channel="explicit",
                    candidate_root_id=root_id,
                    candidate_version=0,
                    source_authority=_source_authority(decision),
                    reason_code=TERMINAL_INVALID_SCOPE_ATTRITION,
                    affects_completeness=False,
                )
            )
            continue
        try:
            semantics = candidate_semantics(
                rule=candidate.rule,
                applies_to=(
                    decision.canonical_scope or tuple(candidate.applies_to)
                ),
                condition=decision.canonical_condition or candidate.condition,
            )
        except ValidationError:
            omissions.append(
                ChannelAssessmentOmission(
                    channel="explicit",
                    candidate_root_id=root_id,
                    candidate_version=0,
                    source_authority=_source_authority(decision),
                    reason_code=ASSESSMENT_CONSTRUCTION_ATTRITION,
                    affects_completeness=True,
                )
            )
            continue
        proposed_kind: ClaimKind = decision.derived_kind or candidate.claim_kind
        proposed_strength: NormativeStrength = (
            decision.derived_strength or candidate.normative_strength
        )
        assessment_id = make_assessment_id(
            snapshot_digest, profile_digest, "explicit", root_id, 0
        )
        payload = {
            "contract_version": UNIFIED_CONTRACT_VERSION,
            "assessment_id": assessment_id,
            "snapshot_digest": snapshot_digest,
            "profile_digest": profile_digest,
            "channel": "explicit",
            "candidate_root_id": root_id,
            "candidate_version": 0,
            "parent_candidate_id": None,
            "semantics": semantics,
            "proposed_claim_kind": proposed_kind,
            "proposed_strength": proposed_strength,
            "raw_channel_status": decision.outcome,
            "raw_policy_version": EXPLICIT_VERIFICATION_POLICY_VERSION,
            "normalized_status": normalized_status,
            "axis_findings": axes,
            "typed_reasons": sorted(reason for reason in reasons if reason),
            "support_refs": support_refs,
            "counterevidence_refs": [],
            "coverage_refs": [coverage_ref],
            "lineage_refs": [],
            "extension_refs": [candidate_ref],
            "integrity": integrity,
            "source_authority": _source_authority(decision),
            "git_need": (
                aggregate_git_needs([verified.judgment.git_need])
                if verified is not None and verified.judgment is not None
                else unassessed_git_need(
                    "explicit_verification_no_valid_judgment",
                    invalid=(
                        verified is not None
                        and verified.reason_codes == ["technical_missing"]
                        and verified.technical_reason_code not in {
                            "structured_inference_not_executed",
                            "missing_batch_response",
                            "missing_batch_judgment",
                        }
                    ),
                )
            ),
            "provenance": [
                f"candidate_fingerprint:{decision.candidate_fingerprint}",
                f"unit_id:{decision.unit_id}",
            ],
        }
        assessments.append(ChannelAssessment.model_validate(seal_payload(payload)))
    # A strict extraction response may repeat the same claim event.  Candidate
    # identity is content-addressed, so exact repetitions are one unique root,
    # while the raw-event denominator remains upstream and is closed by the
    # funnel's ``candidate_event_not_materialized`` attrition.  Conflicting
    # projections for one identity remain an integrity failure rather than a
    # first/last-wins fallback.
    unique_assessments: dict[
        tuple[str, str, int], ChannelAssessment
    ] = {}
    for assessment in assessments:
        key = (
            assessment.channel,
            assessment.candidate_root_id,
            assessment.candidate_version,
        )
        previous = unique_assessments.get(key)
        if previous is not None and previous != assessment:
            raise ValueError("channel assessment root projections conflict")
        unique_assessments[key] = assessment

    unique_omissions: dict[
        tuple[str, str, int], ChannelAssessmentOmission
    ] = {}
    for omission in omissions:
        key = (
            omission.channel,
            omission.candidate_root_id,
            omission.candidate_version,
        )
        previous = unique_omissions.get(key)
        if previous is not None and previous != omission:
            raise ValueError("channel assessment omission projections conflict")
        unique_omissions[key] = omission
    if set(unique_assessments) & set(unique_omissions):
        raise ValueError("channel assessment and omission roots overlap")
    return ChannelAssessmentBuildResult(
        assessments=tuple(
            sorted(
                unique_assessments.values(),
                key=lambda item: item.assessment_id,
            )
        ),
        omissions=tuple(
            sorted(
                unique_omissions.values(),
                key=lambda item: (
                    item.channel,
                    item.candidate_root_id,
                    item.candidate_version,
                ),
            )
        ),
    )


def build_explicit_assessments(
    *,
    decisions: Sequence[CandidateDecision],
    verification: ExplicitVerificationResult,
    snapshot_digest: str,
    profile_digest: str,
) -> tuple[ChannelAssessment, ...]:
    """Freeze safely representable explicit candidate decisions."""

    return _build_explicit_assessment_result(
        decisions=decisions,
        verification=verification,
        snapshot_digest=snapshot_digest,
        profile_digest=profile_digest,
    ).assessments


def build_current_source_assessments(
    *,
    current_source: CurrentSourceBundle | None,
    snapshot_digest: str,
    profile_digest: str,
) -> tuple[ChannelAssessment, ...]:
    """Freeze final current-source candidate versions and their full lineage."""

    if current_source is None:
        return ()
    hypothesis_by_id = {
        item.hypothesis_id: item for item in current_source.hypotheses
    }
    population_by_hypothesis = {
        item.hypothesis_id: item for item in current_source.populations
    }
    falsification_by_hypothesis = {
        item.hypothesis_id: item for item in current_source.falsifications
    }
    refinements_by_child = {
        item.child_hypothesis_id: item for item in current_source.refinements
    }
    assessments: list[ChannelAssessment] = []
    for policy in current_source.policy_decisions:
        hypothesis = hypothesis_by_id[policy.hypothesis_id]
        population = population_by_hypothesis.get(hypothesis.hypothesis_id)
        falsification = falsification_by_hypothesis.get(hypothesis.hypothesis_id)
        optional_falsification_reason = (
            falsification_ineligibility_reason(population)
            if population is not None and falsification is None
            else None
        )
        falsification_optional = optional_falsification_reason in {
            "population_incomplete",
            "classification_incomplete",
            "technical_unknown_members",
        }
        population_cost_omitted = population is None and any(
            reason.startswith(
                (
                    "population_cost_not_admitted:",
                    "refinement_population_cost_not_admitted:",
                )
            )
            or reason
            in {
                "max_total_population_members",
                "refinement_population_budget_exhausted",
            }
            for reason in policy.reason_codes
        )
        integrity_valid = (
            (
                population_cost_omitted
                or (
                    population is not None
                    and population.snapshot_digest == snapshot_digest
                    # Absence is a required coverage gap, not corruption.
                    and (
                        falsification is None
                        or falsification.snapshot_digest == snapshot_digest
                    )
                )
            )
        )
        if not integrity_valid:
            normalized_status = "invalid"
        elif policy.experimental_disposition == "accept_candidate":
            normalized_status = "locally_supported"
        elif policy.experimental_disposition == "rejected":
            normalized_status = "locally_rejected"
        else:
            normalized_status = "locally_uncertain"

        axes: dict[str, str] = {
            "population_integrity": (
                "pass"
                if population is not None and population.status == "complete"
                else "uncertain"
            ),
            "repository_specificity": (
                falsification.repository_specificity
                if falsification is not None
                else "uncertain"
            ),
            "contributor_value": (
                falsification.contributor_value
                if falsification is not None
                else "uncertain"
            ),
            "scope_condition_support": (
                falsification.scope_condition_fit
                if falsification is not None
                else "uncertain"
            ),
            "evidence_sufficiency": (
                falsification.evidence_sufficiency
                if falsification is not None
                else "uncertain"
            ),
        }
        hypothesis_ref = _ref(
            kind="norm_hypothesis",
            stable_id_value=hypothesis.hypothesis_id,
            snapshot_digest=snapshot_digest,
            artifact_path=(
                f"current-source/hypotheses/{hypothesis.hypothesis_id}.json"
            ),
            payload=hypothesis,
            artifact_digest_value=hypothesis.artifact_digest,
        )
        support_refs: list[ImmutableArtifactRef] = []
        counter_refs: list[ImmutableArtifactRef] = []
        extension_refs: list[ImmutableArtifactRef] = [hypothesis_ref]
        coverage_refs: list[ImmutableArtifactRef] = []
        if population is not None:
            population_ref = _ref(
                kind="population",
                stable_id_value=population.population_id,
                snapshot_digest=snapshot_digest,
                artifact_path=(
                    f"current-source/populations/{population.population_id}.json"
                ),
                payload=population,
                artifact_digest_value=population.artifact_digest,
            )
            extension_refs.append(population_ref)
            coverage_refs.append(population_ref)
            for member in population.members:
                member_ref = _ref(
                    kind="population_member",
                    stable_id_value=member.member_id,
                    snapshot_digest=snapshot_digest,
                    artifact_path=(
                        f"current-source/populations/{population.population_id}.json"
                    ),
                    payload=member,
                    artifact_digest_value=member.artifact_digest,
                )
                if member.label == "support":
                    support_refs.append(member_ref)
                elif member.label == "counterexample":
                    counter_refs.append(member_ref)
        if falsification is not None:
            falsification_ref = _ref(
                kind="falsification",
                stable_id_value=falsification.falsification_id,
                snapshot_digest=snapshot_digest,
                artifact_path=(
                    "current-source/falsifications/"
                    f"{falsification.falsification_id}.json"
                ),
                payload=falsification,
                artifact_digest_value=falsification.artifact_digest,
            )
            extension_refs.append(falsification_ref)
            coverage_refs.append(falsification_ref)

        lineage_refs: list[ImmutableArtifactRef] = []
        cursor = hypothesis
        while cursor.parent_hypothesis_id is not None:
            refinement = refinements_by_child.get(cursor.hypothesis_id)
            if refinement is None:
                break
            lineage_refs.append(
                _ref(
                    kind="channel_lineage",
                    stable_id_value=refinement.refinement_id,
                    snapshot_digest=snapshot_digest,
                    artifact_path=(
                        f"current-source/refinements/{refinement.refinement_id}.json"
                    ),
                    payload=refinement,
                    artifact_digest_value=refinement.artifact_digest,
                )
            )
            parent = hypothesis_by_id.get(refinement.parent_hypothesis_id)
            if parent is None:
                break
            cursor = parent

        reasons = set(policy.reason_codes)
        if population is None:
            reasons.add(
                "population_cost_not_materialized"
                if population_cost_omitted
                else "population_missing"
            )
        elif population.status != "complete":
            reasons.update(population.incomplete_reasons or ["population_incomplete"])
        if falsification is None:
            reasons.add("falsification_missing")
            if falsification_optional and optional_falsification_reason:
                reasons.add(
                    "falsification_not_eligible:"
                    + optional_falsification_reason
                )
        elif falsification.status != "complete":
            reasons.update(
                falsification.incomplete_reasons or ["falsification_incomplete"]
            )
        semantics = candidate_semantics(
            rule=hypothesis.rule,
            applies_to=hypothesis.applies_to,
            condition=hypothesis.condition,
            exception=(
                falsification.exception_summary
                if falsification is not None
                else None
            ),
        )
        assessment_id = make_assessment_id(
            snapshot_digest,
            profile_digest,
            "implicit_current",
            hypothesis.candidate_root_id,
            hypothesis.version,
        )
        payload = {
            "contract_version": UNIFIED_CONTRACT_VERSION,
            "assessment_id": assessment_id,
            "snapshot_digest": snapshot_digest,
            "profile_digest": profile_digest,
            "channel": "implicit_current",
            "candidate_root_id": hypothesis.candidate_root_id,
            "candidate_version": hypothesis.version,
            "parent_candidate_id": hypothesis.parent_hypothesis_id,
            "semantics": semantics,
            "proposed_claim_kind": hypothesis.proposed_claim_kind,
            "proposed_strength": hypothesis.proposed_strength,
            "raw_channel_status": policy.experimental_disposition,
            "raw_policy_version": CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
            "normalized_status": normalized_status,
            "axis_findings": axes,
            "typed_reasons": sorted(reason for reason in reasons if reason),
            "support_refs": sorted(support_refs, key=lambda item: item.stable_id),
            "counterevidence_refs": sorted(
                counter_refs, key=lambda item: item.stable_id
            ),
            "coverage_refs": sorted(
                coverage_refs, key=lambda item: item.stable_id
            ),
            "lineage_refs": sorted(
                lineage_refs, key=lambda item: item.stable_id
            ),
            "extension_refs": sorted(
                extension_refs, key=lambda item: item.stable_id
            ),
            "integrity": "valid" if integrity_valid else "invalid",
            "source_authority": "current_population",
            "git_need": (
                falsification.git_need if falsification is not None
                else unassessed_git_need("falsification_not_executed")
            ),
            "provenance": [
                f"candidate_root_id:{hypothesis.candidate_root_id}",
                f"hypothesis_id:{hypothesis.hypothesis_id}",
            ],
        }
        assessments.append(ChannelAssessment.model_validate(seal_payload(payload)))
    return tuple(sorted(assessments, key=lambda item: item.assessment_id))


def build_channel_assessments(
    *,
    explicit_decisions: Sequence[CandidateDecision],
    explicit_verification: ExplicitVerificationResult,
    current_source: CurrentSourceBundle | None,
    snapshot_digest: str,
    profile_digest: str,
) -> tuple[ChannelAssessment, ...]:
    return build_channel_assessment_result(
        explicit_decisions=explicit_decisions,
        explicit_verification=explicit_verification,
        current_source=current_source,
        snapshot_digest=snapshot_digest,
        profile_digest=profile_digest,
    ).assessments


def build_channel_assessment_result(
    *,
    explicit_decisions: Sequence[CandidateDecision],
    explicit_verification: ExplicitVerificationResult,
    current_source: CurrentSourceBundle | None,
    snapshot_digest: str,
    profile_digest: str,
) -> ChannelAssessmentBuildResult:
    """Build channel assessments plus typed candidate-local omissions."""

    explicit = _build_explicit_assessment_result(
        decisions=explicit_decisions,
        verification=explicit_verification,
        snapshot_digest=snapshot_digest,
        profile_digest=profile_digest,
    )
    current = build_current_source_assessments(
        current_source=current_source,
        snapshot_digest=snapshot_digest,
        profile_digest=profile_digest,
    )
    assessment_ids = [
        item.assessment_id for item in (*explicit.assessments, *current)
    ]
    if len(assessment_ids) != len(set(assessment_ids)):
        raise ValueError("channel assessment IDs collide")
    return ChannelAssessmentBuildResult(
        assessments=tuple(
            sorted(
                (*explicit.assessments, *current),
                key=lambda item: item.assessment_id,
            )
        ),
        omissions=explicit.omissions,
    )


def _location_from_source(
    location: SourceLocation,
    *,
    symbols: Sequence[str] = (),
) -> _ResolvedLocation:
    return _ResolvedLocation(
        path=location.path,
        source_content_digest=location.source_content_sha256,
        line_start=location.start_line,
        line_end=location.end_line,
        excerpt_digest=location.excerpt_sha256,
        symbols=tuple(sorted(set(symbols))),
    )


def _observation_locations(
    observations: ObservationBundle | None,
    current_source: CurrentSourceBundle | None,
) -> dict[str, _ResolvedLocation]:
    result: dict[str, _ResolvedLocation] = {}
    if observations is not None:
        for shard in observations.shards:
            file_location = _ResolvedLocation(
                path=shard.path,
                source_content_digest=shard.content_sha256,
                line_start=None,
                line_end=None,
                excerpt_digest=shard.content_sha256,
                symbols=(),
            )
            result[shard.file_id] = file_location
            site_by_id = {item.site_id: item for item in shard.sites}
            for site in shard.sites:
                symbols = tuple(
                    sorted(
                        {
                            *(site.decorator_chain or []),
                            *(site.decorator_symbols or []),
                        }
                    )
                )
                result[site.site_id] = _location_from_source(
                    site.location, symbols=symbols
                )
            for entity in shard.entities:
                if entity.definition_site_id in result:
                    location = result[entity.definition_site_id]
                    result[entity.entity_id] = _ResolvedLocation(
                        path=location.path,
                        source_content_digest=location.source_content_digest,
                        line_start=location.line_start,
                        line_end=location.line_end,
                        excerpt_digest=location.excerpt_digest,
                        symbols=tuple(
                            sorted(
                                {
                                    *location.symbols,
                                    entity.name,
                                    entity.qualified_name,
                                }
                            )
                        ),
                    )
                else:
                    result[entity.entity_id] = _ResolvedLocation(
                        **{
                            **file_location.__dict__,
                            "symbols": tuple(
                                sorted({entity.name, entity.qualified_name})
                            ),
                        }
                    )
            for relation in shard.relations:
                base = (
                    result.get(relation.site_id)
                    if relation.site_id is not None
                    else file_location
                ) or file_location
                result[relation.relation_id] = _ResolvedLocation(
                    path=base.path,
                    source_content_digest=base.source_content_digest,
                    line_start=base.line_start,
                    line_end=base.line_end,
                    excerpt_digest=base.excerpt_digest,
                    symbols=tuple(
                        sorted(
                            {
                                *base.symbols,
                                relation.source.rendered,
                                relation.target.rendered,
                            }
                        )
                    ),
                )
    if current_source is not None:
        unresolved = list(current_source.semantic_map.observations)
        for _ in range(2):
            pending = []
            for observation in unresolved:
                locations = [
                    result[item]
                    for item in observation.evidence_refs
                    if item in result
                ]
                if not locations:
                    pending.append(observation)
                    continue
                base = min(
                    locations,
                    key=lambda item: (
                        item.path,
                        item.line_start or 0,
                        item.line_end or 0,
                    ),
                )
                result[observation.observation_id] = base
            unresolved = pending
    return result


def _packet_fallback_location(
    *,
    packet: object,
    member: PopulationMember,
) -> _ResolvedLocation:
    observation_ref = getattr(packet, "observation_ref")
    fact = getattr(packet, "fact")
    path = observation_ref.path
    source_digest: str | None = None
    excerpt_digest: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    symbols: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        nonlocal path, source_digest, excerpt_digest, line_start, line_end
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and path is None:
                path = value["path"]
            for digest_key in (
                "source_content_sha256",
                "content_sha256",
            ):
                digest_value = value.get(digest_key)
                if (
                    source_digest is None
                    and isinstance(digest_value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest_value)
                ):
                    source_digest = digest_value
            digest_value = value.get("excerpt_sha256")
            if (
                excerpt_digest is None
                and isinstance(digest_value, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest_value)
            ):
                excerpt_digest = digest_value
            if isinstance(value.get("start_line"), int):
                line_start = value["start_line"]
            if isinstance(value.get("end_line"), int):
                line_end = value["end_line"]
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in {
            "name",
            "qualified_name",
            "rendered",
            "callee",
            "decorator",
            "semantic_key",
        }:
            if 0 < len(value) <= 256 and "\n" not in value:
                symbols.add(value)

    visit(fact)
    safe_path = path or "unknown-observation"
    digest = source_digest or member.artifact_digest
    return _ResolvedLocation(
        path=safe_path,
        source_content_digest=digest,
        line_start=line_start,
        line_end=line_end,
        excerpt_digest=excerpt_digest or digest,
        symbols=tuple(sorted(symbols)),
    )


def _bounded_exact_values(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for value in values
                if value
                and len(value) <= 256
                and not any(token in value for token in ("\n", "\r", "\x00"))
            }
        )
    )


def _explicit_quote_literal_anchors(quote: str) -> tuple[str, ...]:
    """Return canonical exact source lines from one validated explicit span.

    Evidence recovery preserves source line endings, so even a one-line atom
    normally contains a trailing newline.  Git literals must instead be
    bounded non-empty single lines.  Remove only CR/LF line separators, keep
    every other source character unchanged, and discard values that cannot
    safely cross the fixed-argv Git boundary.
    """

    return tuple(
        sorted(
            {
                line
                for line in re.split(r"\r\n|\n|\r", quote)
                if line.strip()
                and len(line) <= 512
                and not any(
                    unicodedata.category(character) == "Cc" for character in line
                )
            }
        )
    )


def _explicit_semantic_anchor_symbols(
    evidence_type: str,
    semantic_key: str,
) -> tuple[str, ...]:
    """Return only bounded, typed explicit keys as symbol anchors.

    Prose semantic keys contain a normalized declaration and can be both long
    and semantically broad.  Treating them as symbols creates accidental exact
    anchors; prose remains available through the claim semantics and evidence
    excerpt instead.
    """

    if evidence_type not in {
        "configuration_entry",
        "command_definition",
        "workflow_step",
    }:
        return ()
    return _bounded_exact_values((semantic_key,))


def _explicit_typed_decision_slots(
    evidence_type: str,
    semantic_key: str,
    *,
    semantic_flags: Sequence[str] = (),
    path: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> tuple[str, ...]:
    """Return only host-provable stable decision-slot identities.

    Typed proposition keys include the selected value, command body, or
    workflow action.  Comparing those full keys would incorrectly treat two
    conflicting values of one slot as independent decisions.  Use typed parser
    flags (and, for workflow atoms, the exact source location) to remove the
    value while retaining enough identity to prove distinct slots.  If the
    adapter did not expose that structure, abstain from the negative grouping
    gate rather than guessing by parsing repository prose.
    """

    flags = tuple(sorted(set(semantic_flags)))
    if evidence_type == "configuration_entry":
        adapter = semantic_key.partition(":")[0]
        if adapter not in {"toml", "ini_cfg", "yaml_json"}:
            return ()
        sections = tuple(
            sorted(
                flag.removeprefix("config_section:")
                for flag in flags
                if flag.startswith("config_section:")
            )
        )
        keys = tuple(
            sorted(
                flag.removeprefix("config_key:")
                for flag in flags
                if flag.startswith("config_key:")
                and flag.removeprefix("config_key:")
            )
        )
        if len(sections) != 1 or not keys:
            return ()
        section = sections[0] or "<root>"
        slots = tuple(
            f"configuration:{adapter}:{section}:{key}" for key in keys
        )
        bounded = _bounded_exact_values(slots)
        return bounded if len(bounded) == len(slots) else ()
    if evidence_type == "command_definition":
        targets = tuple(
            sorted(
                flag.removeprefix("command_target:")
                for flag in flags
                if flag.startswith("command_target:")
                and flag.removeprefix("command_target:")
            )
        )
        if len(targets) != 1:
            return ()
        slots = (f"command-target:{targets[0]}",)
        bounded = _bounded_exact_values(slots)
        return bounded if len(bounded) == len(slots) else ()
    if evidence_type == "workflow_step":
        action_keys = tuple(
            sorted(
                flag.removeprefix("workflow_key:")
                for flag in flags
                if flag in {"workflow_key:run", "workflow_key:uses"}
            )
        )
        if (
            len(action_keys) != 1
            or not path
            or line_start is None
            or line_end is None
            or line_start < 1
            or line_end < line_start
        ):
            return ()
        # `run` and `uses` are alternative values of the same exact workflow
        # action location, not separate decision slots.  The proposition key
        # and semantic flags still preserve which action was selected.
        slots = (f"workflow-action:{path}:{line_start}:{line_end}",)
        bounded = _bounded_exact_values(slots)
        return bounded if len(bounded) == len(slots) else ()
    return ()


def _observation_runtime_metadata(
    observations: object,
) -> tuple[str, str, tuple[str, ...]]:
    index = getattr(observations, "index", None)
    if index is not None:
        return (
            str(getattr(index, "snapshot_digest")),
            str(getattr(index, "overall")),
            tuple(sorted(set(getattr(index, "incomplete_reasons", ())))),
        )
    return (
        str(getattr(observations, "snapshot_digest")),
        str(getattr(observations, "overall")),
        tuple(sorted(set(getattr(observations, "incomplete_reasons", ())))),
    )


def build_observation_anchor_index(
    observations: ObservationBundle | object | None,
) -> ObservationAnchorIndex | None:
    """Index exact AST facts from supplied shards without repository access."""

    if observations is None:
        return None
    snapshot_digest, status, incomplete_reasons = _observation_runtime_metadata(
        observations
    )
    if status not in {"complete", "partial"}:
        status = "partial"

    aggregate: dict[
        tuple[object, ...],
        tuple[_ObservationAnchorMatch, set[str]],
    ] = {}

    def add(
        shard: PythonFileShard,
        *,
        location: SourceLocation,
        exact_values: Sequence[str],
    ) -> None:
        values = _bounded_exact_values(exact_values)
        if not values:
            return
        match = _ObservationAnchorMatch(
            shard_id=shard.file_id,
            shard_digest=shard.shard_digest,
            path=shard.path,
            source_content_digest=shard.content_sha256,
            line_start=location.start_line,
            line_end=location.end_line,
            excerpt_digest=location.excerpt_sha256,
            exact_values=values,
            shard_status=shard.parse_status,
            shard_incomplete_reasons=tuple(
                sorted(set(shard.incomplete_reasons))
            ),
        )
        previous = aggregate.get(match.origin_key)
        if previous is None:
            aggregate[match.origin_key] = (match, set(values))
        else:
            previous[1].update(values)

    by_path: dict[str, _ObservationAnchorMatch] = {}
    for shard in sorted(
        getattr(observations, "shards"), key=lambda item: (item.path, item.file_id)
    ):
        by_path[shard.path] = _ObservationAnchorMatch(
            shard_id=shard.file_id,
            shard_digest=shard.shard_digest,
            path=shard.path,
            source_content_digest=shard.content_sha256,
            line_start=None,
            line_end=None,
            excerpt_digest=shard.content_sha256,
            exact_values=(),
            shard_status=shard.parse_status,
            shard_incomplete_reasons=tuple(
                sorted(set(shard.incomplete_reasons))
            ),
        )
        site_by_id = {item.site_id: item for item in shard.sites}
        for entity in shard.entities:
            site = site_by_id.get(entity.definition_site_id or "")
            if site is not None:
                add(
                    shard,
                    location=site.location,
                    exact_values=(entity.name, entity.qualified_name),
                )
        for site in shard.sites:
            if site.kind == "decorator":
                add(
                    shard,
                    location=site.location,
                    exact_values=(
                        *(site.decorator_chain or ()),
                        *(site.decorator_symbols or ()),
                    ),
                )
        for relation in shard.relations:
            site = site_by_id.get(relation.site_id or "")
            if site is not None:
                add(
                    shard,
                    location=site.location,
                    exact_values=(
                        relation.source.rendered,
                        relation.target.rendered,
                        *((relation.alias,) if relation.alias else ()),
                    ),
                )

    matches: list[_ObservationAnchorMatch] = []
    for key in sorted(aggregate, key=repr):
        representative, values = aggregate[key]
        matches.append(
            replace(representative, exact_values=tuple(sorted(values)))
        )
    symbol_values: dict[str, list[_ObservationAnchorMatch]] = {}
    for match in matches:
        for value in match.exact_values:
            symbol_values.setdefault(value, []).append(match)
    return ObservationAnchorIndex(
        snapshot_digest=snapshot_digest,
        status=status,
        incomplete_reasons=incomplete_reasons,
        by_symbol={
            key: tuple(
                sorted(
                    value,
                    key=lambda item: (
                        item.path,
                        item.line_start or 0,
                        item.line_end or 0,
                        item.shard_id,
                    ),
                )
            )
            for key, value in sorted(symbol_values.items())
        },
        by_path=dict(sorted(by_path.items())),
    )


def build_observation_anchor_supplement(
    *,
    case: CandidateCase,
    anchors_by_member: Mapping[str, MemberBlockingAnchors],
    observation_index: ObservationAnchorIndex | None,
    max_items: int,
) -> ObservationAnchorSupplement:
    """Resolve exact blocking anchors to bounded, shard-backed context facts."""

    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise ValueError("observation anchor supplement cap must be positive")
    case_member_ids = {item.member_id for item in case.members}
    foreign = set(anchors_by_member) - case_member_ids
    if foreign:
        raise ValueError("observation anchor supplement contains foreign members")

    member_anchors: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    any_anchor = False
    for member in case.members:
        supplied = anchors_by_member.get(member.member_id, MemberBlockingAnchors())
        exact_values = _bounded_exact_values(
            (
                *supplied.symbols,
                *supplied.config_keys,
                *supplied.commands,
                *supplied.workflow_actions,
                *supplied.literals,
            )
        )
        paths = tuple(sorted(set(supplied.paths)))
        member_anchors[member.member_id] = (exact_values, paths)
        any_anchor = any_anchor or bool(exact_values or paths)

    if not any_anchor:
        return ObservationAnchorSupplement(
            evidence=(),
            dependency=CapabilityDependency(
                capability="observation_anchor_supplement",
                requirement_mode="augmentative",
                status="bounded_no_match",
                reason_codes=[],
            ),
            matched_origins=0,
            omitted_origins=0,
        )
    if observation_index is None:
        return ObservationAnchorSupplement(
            evidence=(),
            dependency=CapabilityDependency(
                capability="observation_anchor_supplement",
                requirement_mode="not_required",
                status="not_required",
                reason_codes=[],
            ),
            matched_origins=0,
            omitted_origins=0,
        )
    if observation_index.snapshot_digest != case.snapshot_digest:
        return ObservationAnchorSupplement(
            evidence=(),
            dependency=CapabilityDependency(
                capability="observation_anchor_supplement",
                requirement_mode="augmentative",
                status="failed",
                reason_codes=["observation_snapshot_mismatch"],
            ),
            matched_origins=0,
            omitted_origins=0,
        )

    matched: dict[
        tuple[object, ...],
        tuple[_ObservationAnchorMatch, set[str], set[str]],
    ] = {}

    def bind(
        member_id: str,
        match: _ObservationAnchorMatch,
        anchors: Sequence[str],
    ) -> None:
        previous = matched.get(match.origin_key)
        if previous is None:
            matched[match.origin_key] = (match, {member_id}, set(anchors))
        else:
            previous[1].add(member_id)
            previous[2].update(anchors)

    for member_id in sorted(member_anchors):
        exact_values, paths = member_anchors[member_id]
        for value in exact_values:
            for match in observation_index.by_symbol.get(value, ()):
                bind(member_id, match, (value,))
        for path in paths:
            match = observation_index.by_path.get(path)
            if match is not None:
                bind(member_id, match, ())

    ordered = [matched[key] for key in sorted(matched, key=repr)]
    source_eligible = [
        item for item in ordered if item[0].shard_status == "complete"
    ]
    quarantined = [
        item for item in ordered if item[0].shard_status != "complete"
    ]
    retained = source_eligible[:max_items]
    cap_omitted = len(source_eligible) - len(retained)
    omitted = len(ordered) - len(retained)
    evidence: list[EvidenceContribution] = []
    for match, member_ids, exact_values in retained:
        location = (
            match.path
            if match.line_start is None
            else f"{match.path}:{match.line_start}-{match.line_end}"
        )
        evidence.append(
            EvidenceContribution(
                origin_kind="observation",
                path=match.path,
                line_start=match.line_start,
                line_end=match.line_end,
                source_content_digest=match.source_content_digest,
                independence_group=(
                    f"observation:{match.path}:{match.source_content_digest}"
                ),
                role="context",
                member_ids=tuple(sorted(member_ids)),
                artifact_ref=ImmutableArtifactRef(
                    kind="observation",
                    stable_id=match.shard_id,
                    snapshot_digest=case.snapshot_digest,
                    artifact_path=(
                        f"observations/python-files/{match.shard_id}.json"
                    ),
                    artifact_digest=match.shard_digest,
                ),
                excerpt_digest=match.excerpt_digest,
                description=f"Saved Python observation context at {location}"[:600],
                anchor_symbols=tuple(sorted(exact_values)),
            )
        )

    reasons: list[str] = []
    if quarantined:
        reasons.append("observation_anchor_dependency_partial")
    if cap_omitted:
        reasons.append("observation_anchor_supplement_limit")
    if reasons:
        status = "partial"
    elif evidence:
        status = "complete"
    else:
        status = "bounded_no_match"
    return ObservationAnchorSupplement(
        evidence=tuple(evidence),
        dependency=CapabilityDependency(
            capability="observation_anchor_supplement",
            requirement_mode="augmentative",
            status=status,  # type: ignore[arg-type]
            reason_codes=sorted(set(reasons)),
        ),
        matched_origins=len(ordered),
        omitted_origins=omitted,
    )


@dataclass(frozen=True)
class CaseInputIndex:
    assessments: Mapping[str, ChannelAssessment]
    decisions: Mapping[str, CandidateDecision]
    locations: Mapping[str, _ResolvedLocation]
    hypotheses: Mapping[str, NormHypothesis]
    populations: Mapping[str, PopulationArtifact]
    falsifications: Mapping[str, FalsificationResult]
    packets: Mapping[tuple[str, str], PopulationMemberPacket]


def build_case_input_index(
    *, assessments: Sequence[ChannelAssessment],
    explicit_decisions: Sequence[CandidateDecision],
    observations: ObservationBundle | None,
    current_source: CurrentSourceBundle | None,
) -> CaseInputIndex:
    return CaseInputIndex(
        assessments={item.assessment_id: item for item in assessments},
        decisions={explicit_candidate_root_id(item.candidate_fingerprint): item for item in explicit_decisions if item.raw_candidate is not None},
        locations=_observation_locations(observations, current_source),
        hypotheses={item.hypothesis_id: item for item in current_source.hypotheses} if current_source is not None else {},
        populations={item.hypothesis_id: item for item in current_source.populations} if current_source is not None else {},
        falsifications={item.hypothesis_id: item for item in current_source.falsifications} if current_source is not None else {},
        packets={(item.hypothesis_id, item.observation_ref.observation_id): item for item in current_source.member_packets} if current_source is not None else {},
    )


def _falsification_capability(
    population: PopulationArtifact | None,
    falsification: FalsificationResult | None,
) -> tuple[str, tuple[str, ...]]:
    """Derive capability from immutable results, never reference existence.

    A terminal inference can fail and a saved result can describe an entirely
    omitted job. Both its semantic status and execution/admission must close.
    This function is shared by live adaptation and read-only upstream replay.
    """
    if falsification is None:
        if population is not None and falsification_ineligibility_reason(
            population
        ) in {"population_incomplete", "classification_incomplete", "technical_unknown_members"}:
            # Classification already closes this case before falsification.
            return "not_required", ()
        return "unavailable", ("falsification_artifact_ref_missing",)
    execution = falsification.execution
    reasons = set(falsification.incomplete_reasons)
    if falsification.admission_estimate is not None:
        reasons.add("falsification_cost_not_admitted")
    if (
        not execution.request_plan_complete
        or execution.omitted_calls
        or execution.omitted_items
        or execution.closure_reason is not None
        or execution.terminal_calls != execution.planned_calls
        or execution.terminal_items != execution.planned_items
    ):
        reasons.add("falsification_execution_incomplete")
    if (
        population is not None
        and falsification_ineligibility_reason(population) is None
        and (execution.planned_calls == 0 or not falsification.inference_ids)
    ):
        reasons.add("falsification_execution_not_run")
    if falsification.status != "complete":
        reasons.add("falsification_" + falsification.status)
    if falsification.status == "failed":
        return "failed", tuple(sorted(reasons))
    if reasons:
        return "partial", tuple(sorted(reasons))
    return "complete", ()


def build_case_evidence_inputs(
    *,
    case: CandidateCase,
    assessments: Sequence[ChannelAssessment],
    explicit_decisions: Sequence[CandidateDecision],
    current_source: CurrentSourceBundle | None,
    observations: ObservationBundle | None,
    explicit_discovery_complete: bool,
    anchors_by_member: Mapping[str, MemberBlockingAnchors] | None = None,
    observation_anchor_index: ObservationAnchorIndex | None = None,
    max_observation_context_items: int = 64,
    input_index: CaseInputIndex | None = None,
) -> CaseEvidenceInputs:
    """Resolve legacy channel artifacts into the mandatory case evidence view."""

    index = input_index or build_case_input_index(
        assessments=assessments, explicit_decisions=explicit_decisions,
        observations=observations, current_source=current_source,
    )
    assessment_by_id = index.assessments
    decision_by_root = index.decisions
    location_by_observation = index.locations
    explicit_contributions: list[EvidenceContribution] = []
    population_contributions: list[PopulationEvidenceContribution] = []
    denominators: list[PopulationDenominator] = []
    population_refs: list[ImmutableArtifactRef] = []
    falsification_refs: list[ImmutableArtifactRef] = []
    scope_condition_refs: list[ImmutableArtifactRef] = []
    current_members = 0
    falsification_capabilities: list[tuple[str, tuple[str, ...]]] = []

    hypothesis_by_id = index.hypotheses
    population_by_hypothesis = index.populations
    falsification_by_hypothesis = index.falsifications
    packet_by_key = index.packets

    for case_member in case.members:
        assessment = assessment_by_id[case_member.assessment_id]
        refs_by_id = {
            item.stable_id: item
            for item in [
                *assessment.support_refs,
                *assessment.counterevidence_refs,
                *assessment.coverage_refs,
                *assessment.extension_refs,
            ]
        }
        if assessment.channel == "explicit":
            decision = decision_by_root.get(assessment.candidate_root_id)
            if decision is None:
                continue
            for grounded in decision.grounded:
                evidence = grounded.evidence
                stable_evidence_id = stable_id(
                    "EXPLEVID",
                    [
                        assessment.candidate_root_id,
                        evidence.path,
                        evidence.line_start,
                        evidence.line_end,
                        evidence.semantic_key,
                        evidence.quote_sha256,
                    ],
                )
                reference = refs_by_id.get(stable_evidence_id)
                if reference is None:
                    continue
                origin_kind = {
                    "configuration_entry": "configuration_atom",
                    "command_definition": "command_atom",
                    "workflow_step": "workflow_atom",
                }.get(evidence.type, "explicit_span")
                literal_anchors = _explicit_quote_literal_anchors(evidence.quote)
                contribution = EvidenceContribution(
                    origin_kind=origin_kind,  # type: ignore[arg-type]
                    path=evidence.path,
                    line_start=evidence.line_start,
                    line_end=evidence.line_end,
                    source_content_digest=evidence.source_content_sha256,
                    independence_group=f"explicit:{evidence.path}",
                    role="support",
                    member_ids=(case_member.member_id,),
                    artifact_ref=reference,
                    excerpt_digest=evidence.quote_sha256,
                    description=evidence.quote[
                        :MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS
                    ],
                    anchor_literals=literal_anchors,
                    anchor_symbols=_explicit_semantic_anchor_symbols(
                        evidence.type,
                        evidence.semantic_key,
                    ),
                )
                explicit_contributions.append(contribution)
                scope_condition_refs.append(reference)
            continue

        current_members += 1
        hypothesis_ref = next(
            (
                item
                for item in assessment.extension_refs
                if item.kind == "norm_hypothesis"
            ),
            None,
        )
        if hypothesis_ref is None:
            falsification_capabilities.append(
                ("unavailable", ("falsification_artifact_ref_missing",))
            )
            continue
        hypothesis = hypothesis_by_id.get(hypothesis_ref.stable_id)
        population = population_by_hypothesis.get(hypothesis_ref.stable_id)
        falsification = falsification_by_hypothesis.get(hypothesis_ref.stable_id)
        falsification_capabilities.append(
            _falsification_capability(population, falsification)
        )
        if hypothesis is None or population is None:
            continue
        population_ref = next(
            (
                item
                for item in assessment.extension_refs
                if item.kind == "population" and item.stable_id == population.population_id
            ),
            None,
        )
        if population_ref is not None:
            population_refs.append(population_ref)
        if falsification is not None:
            falsification_ref = next(
                (
                    item
                    for item in assessment.extension_refs
                    if item.kind == "falsification"
                    and item.stable_id == falsification.falsification_id
                ),
                None,
            )
            if falsification_ref is not None:
                falsification_refs.append(falsification_ref)
                scope_condition_refs.append(falsification_ref)
        denominators.append(
            PopulationDenominator(
                member_id=case_member.member_id,
                population_id=population.population_id,
                status=population.status,
                census_enumeration_status=(
                    population.sampling_plan.census_enumeration_status
                ),
                sampling_plan_status=(
                    population.sampling_plan.sampling_plan_status
                ),
                sampling_mode=population.sampling_plan.mode,
                scanned=population.population_result.scanned_count,
                serialized=population.population_result.serialized_count,
                matched=population.sampling_plan.matched_count,
                inspected=population.sampling_plan.inspected_count,
                uninspected=population.sampling_plan.uninspected_count,
                support=population.support_count,
                counterexample=population.counterexample_count,
                unknown=population.unknown_count,
                ineligible=population.ineligible_count,
            )
        )
        for member in population.members:
            location = location_by_observation.get(
                member.observation_ref.observation_id
            )
            packet = packet_by_key.get(
                (hypothesis.hypothesis_id, member.observation_ref.observation_id)
            )
            if packet is None:
                # Semantic population construction intentionally does not
                # materialize source packets for deterministically out-of-scope
                # members.  Their saved observation location and sealed member
                # artifact are sufficient to preserve the inspected four-way
                # labels alongside the explicit matched/inspected/uninspected
                # denominator.  Every reviewable member remains fail-closed:
                # without its packet it contributes no evidence and the
                # denominator check reports a typed completeness gap.
                if member.label != "ineligible" or location is None:
                    continue
            elif location is None:
                location = _packet_fallback_location(packet=packet, member=member)
            member_ref = refs_by_id.get(member.member_id)
            if member_ref is None:
                member_ref = _ref(
                    kind="population_member",
                    stable_id_value=member.member_id,
                    snapshot_digest=case.snapshot_digest,
                    artifact_path=(
                        f"current-source/populations/{population.population_id}.json"
                    ),
                    payload=member,
                    artifact_digest_value=member.artifact_digest,
                )
            role = "counterevidence" if member.label == "counterexample" else member.label
            evidence = EvidenceContribution(
                origin_kind="population_member",
                path=location.path,
                line_start=location.line_start,
                line_end=location.line_end,
                source_content_digest=location.source_content_digest,
                independence_group=member.independence_group,
                role=role,  # type: ignore[arg-type]
                member_ids=(case_member.member_id,),
                artifact_ref=member_ref,
                excerpt_digest=location.excerpt_digest,
                description=member.reason_summary,
                anchor_literals=(),
                anchor_symbols=location.symbols,
            )
            population_contributions.append(
                PopulationEvidenceContribution(
                    case_member_id=case_member.member_id,
                    population_id=population.population_id,
                    population_member_id=member.member_id,
                    label=member.label,
                    evidence=evidence,
                )
            )

    dependencies: list[CapabilityDependency] = []
    if current_members:
        # Keep every root/version's requirement: one complete member must not
        # cover another member whose saved falsification was omitted.
        status_order = ("failed", "unavailable", "partial", "complete", "not_required")
        falsification_status = next(
            status for status in status_order
            if any(item[0] == status for item in falsification_capabilities)
        )
        dependencies.append(
            CapabilityDependency(
                capability="current_falsification",
                requirement_mode=(
                    "not_required" if falsification_status == "not_required" else "required"
                ),
                status=falsification_status,
                reason_codes=sorted({
                    reason for _, reasons in falsification_capabilities for reason in reasons
                }),
            )
        )
        # A repository-wide explicit enumeration gap remains visible in the
        # package, but it is not candidate-local missing substrate for an
        # independently evidenced current-source observation. Successfully
        # extracted explicit members still enter grouping and conflict review.
        dependencies.append(
            CapabilityDependency(
                capability="explicit_discovery_for_current_conflicts",
                requirement_mode="augmentative",
                status="complete" if explicit_discovery_complete else "partial",
                reason_codes=(
                    []
                    if explicit_discovery_complete
                    else ["explicit_discovery_incomplete"]
                ),
            )
        )
    observation_evidence: tuple[EvidenceContribution, ...] = ()
    if anchors_by_member is not None:
        case_anchors = {
            item.member_id: anchors_by_member.get(
                item.member_id, MemberBlockingAnchors()
            )
            for item in case.members
        }
        anchor_index = observation_anchor_index
        if anchor_index is None and observations is not None:
            anchor_index = build_observation_anchor_index(observations)
        supplement = build_observation_anchor_supplement(
            case=case,
            anchors_by_member=case_anchors,
            observation_index=anchor_index,
            max_items=max_observation_context_items,
        )
        existing_independence: dict[tuple[object, ...], str] = {}
        for contribution in [
            *explicit_contributions,
            *(item.evidence for item in population_contributions),
        ]:
            key = (
                contribution.path,
                contribution.line_start,
                contribution.line_end,
                contribution.source_content_digest,
            )
            existing_independence.setdefault(key, contribution.independence_group)
        observation_evidence = tuple(
            replace(
                contribution,
                independence_group=existing_independence.get(
                    (
                        contribution.path,
                        contribution.line_start,
                        contribution.line_end,
                        contribution.source_content_digest,
                    ),
                    contribution.independence_group,
                ),
            )
            for contribution in supplement.evidence
        )
        dependencies.append(supplement.dependency)
    return CaseEvidenceInputs(
        explicit_evidence=tuple(explicit_contributions),
        population_evidence=tuple(population_contributions),
        denominators=tuple(denominators),
        population_refs=tuple(
            sorted({item.stable_id: item for item in population_refs}.values(), key=lambda item: item.stable_id)
        ),
        falsification_refs=tuple(
            sorted({item.stable_id: item for item in falsification_refs}.values(), key=lambda item: item.stable_id)
        ),
        scope_condition_refs=tuple(
            sorted({item.stable_id: item for item in scope_condition_refs}.values(), key=lambda item: item.stable_id)
        ),
        dependencies=tuple(
            sorted(dependencies, key=lambda item: item.capability)
        ),
        observation_evidence=observation_evidence,
    )


def build_blocking_anchors(
    *,
    members: Sequence[object],
    assessments: Sequence[ChannelAssessment],
    explicit_decisions: Sequence[CandidateDecision],
    current_source: CurrentSourceBundle | None,
    observations: ObservationBundle | None,
    input_index: CaseInputIndex | None = None,
) -> dict[str, MemberBlockingAnchors]:
    """Build deterministic exact anchors without exposing channel status."""

    index = input_index or build_case_input_index(
        assessments=assessments, explicit_decisions=explicit_decisions,
        observations=observations, current_source=current_source,
    )
    assessment_by_id = index.assessments
    decision_by_root = index.decisions
    locations = index.locations
    hypothesis_by_id = index.hypotheses
    population_by_hypothesis = index.populations
    result: dict[str, MemberBlockingAnchors] = {}
    for member in members:
        member_id = str(getattr(member, "member_id"))
        assessment_id = str(getattr(member, "assessment_id"))
        assessment = assessment_by_id[assessment_id]
        symbols: set[str] = set()
        config_keys: set[str] = set()
        commands: set[str] = set()
        workflow_actions: set[str] = set()
        decision_slots: set[str] = set()
        literals: set[str] = set()
        paths: set[str] = set()
        if assessment.channel == "explicit":
            decision = decision_by_root.get(assessment.candidate_root_id)
            for grounded in decision.grounded if decision is not None else []:
                evidence = grounded.evidence
                paths.add(evidence.path)
                typed_symbols = _explicit_semantic_anchor_symbols(
                    evidence.type,
                    evidence.semantic_key,
                )
                symbols.update(typed_symbols)
                decision_slots.update(
                    _explicit_typed_decision_slots(
                        evidence.type,
                        evidence.semantic_key,
                        semantic_flags=tuple(
                            getattr(grounded, "semantic_flags", ())
                        ),
                        path=evidence.path,
                        line_start=getattr(evidence, "line_start", None),
                        line_end=getattr(evidence, "line_end", None),
                    )
                )
                if len(evidence.quote) <= 512 and "\n" not in evidence.quote:
                    literals.add(evidence.quote)
                if evidence.type == "configuration_entry":
                    config_keys.update(typed_symbols)
                elif evidence.type == "command_definition":
                    commands.update(typed_symbols)
                elif evidence.type == "workflow_step":
                    workflow_actions.update(typed_symbols)
        else:
            hypothesis_ref = next(
                (
                    item
                    for item in assessment.extension_refs
                    if item.kind == "norm_hypothesis"
                ),
                None,
            )
            hypothesis = (
                hypothesis_by_id.get(hypothesis_ref.stable_id)
                if hypothesis_ref is not None
                else None
            )
            population = (
                population_by_hypothesis.get(hypothesis.hypothesis_id)
                if hypothesis is not None
                else None
            )
            if population is not None:
                for population_member in population.members:
                    # Population enumeration is a complete deterministic
                    # superset.  Ineligible/unknown members are deliberately
                    # retained for denominators and later falsification, but
                    # they do not express the candidate's semantics and must
                    # not connect otherwise unrelated candidates during
                    # grouping.  Only classified positive or negative
                    # semantic contributors may supply blocking anchors.
                    if population_member.label not in {
                        "support",
                        "counterexample",
                    }:
                        continue
                    if population_member.observation_ref.path:
                        paths.add(population_member.observation_ref.path)
                    location = locations.get(
                        population_member.observation_ref.observation_id
                    )
                    if location is not None:
                        paths.add(location.path)
                        symbols.update(location.symbols)
        result[member_id] = MemberBlockingAnchors(
            symbols=tuple(sorted(symbols)),
            config_keys=tuple(sorted(config_keys)),
            commands=tuple(sorted(commands)),
            workflow_actions=tuple(sorted(workflow_actions)),
            decision_slots=tuple(sorted(decision_slots)),
            literals=tuple(sorted(literals)),
            paths=tuple(sorted(paths)),
        )
    return result


__all__ = [
    "ASSESSMENT_CONSTRUCTION_ATTRITION",
    "CaseEvidenceInputs",
    "ChannelAssessmentBuildResult",
    "ChannelAssessmentOmission",
    "ObservationAnchorIndex",
    "ObservationAnchorSupplement",
    "TERMINAL_INVALID_SCOPE_ATTRITION",
    "UNIFIED_CHANNEL_ADAPTER_VERSION",
    "build_case_evidence_inputs",
    "build_blocking_anchors",
    "build_channel_assessment_result",
    "build_channel_assessments",
    "build_current_source_assessments",
    "build_explicit_assessments",
    "build_observation_anchor_index",
    "build_observation_anchor_supplement",
    "candidate_semantics",
    "infer_polarity",
    "normalized_action",
]
