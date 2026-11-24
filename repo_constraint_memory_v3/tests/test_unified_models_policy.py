from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.unified.ids import (
    content_digest,
    make_assessment_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
    make_finding_id,
    make_member_id,
    seal_payload,
)
from repo_constraint_memory_v3.unified.git_evidence import (
    GitChangeSlice,
    GitTemporalBundle,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    CanonicalClaim,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    PopulationDenominator,
    TemporalMemberRelation,
    TemporalRelationMatrix,
    UnifiedCaseFinding,
    UnifiedMemberFinding,
    UnifiedMemberRelation,
    UNIFIED_CONTRACT_VERSION,
)
from repo_constraint_memory_v3.unified.policy import (
    CANONICAL_IDENTITY_COLLISION_REASON,
    CanonicalPolicyAccumulator,
    apply_case_policy,
    is_explicit_fast_path,
)


SNAPSHOT = "sha256:" + "1" * 64
PROFILE = "2" * 64
DIGEST = "3" * 64


def _ref(kind: str, stable_id: str, path: str = "explicit_verification.json"):
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id,
        snapshot_digest=SNAPSHOT,
        artifact_path=path,
        artifact_digest=DIGEST,
    )


def _assessment(
    root: str,
    *,
    channel: str = "explicit",
    polarity: str = "positive",
    status: str = "locally_supported",
    authority: str = "direct_normative",
    strength: str = "required",
    claim_kind: str | None = None,
    reasons: list[str] | None = None,
    rule: str = "Run pytest for every change.",
    condition: str | None = None,
    exception: str | None = None,
) -> ChannelAssessment:
    assessment_id = make_assessment_id(SNAPSHOT, PROFILE, channel, root, 0)
    semantics = CandidateSemantics(
        rule=rule,
        normalized_action="run pytest for every change",
        polarity=polarity,
        applies_to=["**"],
        condition=condition,
        exception=exception,
    )
    axes = (
        {
            "host_grounding": "pass",
            "coding_relevance": "pass",
            "evidence_entailment": "pass",
            "scope_condition_support": "pass",
            "ambiguity": "pass",
        }
        if channel == "explicit"
        else {
            "population_integrity": "pass",
            "repository_specificity": "pass",
            "contributor_value": "pass",
            "scope_condition_support": "pass",
            "evidence_sufficiency": "pass",
        }
    )
    payload = {
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "assessment_id": assessment_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "channel": channel,
        "candidate_root_id": root,
        "candidate_version": 0,
        "parent_candidate_id": None,
        "semantics": semantics,
        "proposed_claim_kind": claim_kind
        or (
            "normative_declaration"
            if channel == "explicit"
            else "observed_convention"
        ),
        "proposed_strength": strength,
        "raw_channel_status": "eligible",
        "raw_policy_version": "test-policy",
        "normalized_status": status,
        "axis_findings": axes,
        "typed_reasons": reasons or ["verified_eligible"],
        "support_refs": [_ref("explicit_evidence" if channel == "explicit" else "population_member", root)],
        "counterevidence_refs": [],
        "coverage_refs": [],
        "lineage_refs": [],
        "extension_refs": [],
        "integrity": "valid",
        "source_authority": authority,
        "git_need": {"status": "not_assessed", "judgments": [], "reason": "host_fixture"},
        "provenance": [f"root:{root}"],
    }
    return ChannelAssessment.model_validate(seal_payload(payload))


def _locally_relevance_rejected_normative_assessment(
    root: str,
    *,
    rule: str = "When answering chat messages, list pros and cons.",
) -> ChannelAssessment:
    assessment = _assessment(
        root,
        status="locally_rejected",
        authority="direct_normative",
        strength="required",
        claim_kind="normative_declaration",
        reasons=["coding_relevance_failed"],
        rule=rule,
    )
    payload = assessment.model_dump(mode="python", exclude={"artifact_digest"})
    payload["axis_findings"]["coding_relevance"] = "fail"
    return ChannelAssessment.model_validate(seal_payload(payload))


def _case(assessments: list[ChannelAssessment], *, complete: bool = True):
    members = []
    for assessment in assessments:
        member_id = make_member_id(SNAPSHOT, assessment.assessment_id)
        members.append(
            CandidateCaseMember(
                member_id=member_id,
                assessment_id=assessment.assessment_id,
                snapshot_digest=SNAPSHOT,
                channel=assessment.channel,
                semantics=assessment.semantics,
                assessment_ref=ImmutableArtifactRef(
                    kind="channel_assessment",
                    stable_id=assessment.assessment_id,
                    snapshot_digest=SNAPSHOT,
                    artifact_path=(
                        "canonical/audit/channel-assessments/"
                        f"{assessment.assessment_id}.json"
                    ),
                    artifact_digest=assessment.artifact_digest,
                ),
            )
        )
    members.sort(key=lambda item: item.member_id)
    relations = [
        CandidateMemberRelation(
            left_member_id=left.member_id,
            right_member_id=right.member_id,
            relation="distinct",
            source="host_distinct",
            reason_code="test_distinct",
        )
        for left, right in combinations(members, 2)
    ]
    payload = {
        "case_id": make_case_id(SNAPSHOT, [item.member_id for item in members]),
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "members": members,
        "relations": relations if complete else [],
        "grouping_status": "complete" if complete else "incomplete",
        "grouping_reason_codes": [] if complete else ["grouping_overflow"],
        "exact_identity_groups": [],
        "required_capabilities": ["current_evidence", "unified_verifier"],
    }
    return CandidateCase.model_validate(seal_payload(payload))


def _current_evidence(case: CandidateCase, *, status: str = "complete"):
    origins = []
    items = []
    denominators = []
    for ordinal, member in enumerate(case.members, start=1):
        assessment_kind = member.channel
        path = "CONTRIBUTING.md" if member.channel == "explicit" else "src/pkg.py"
        source_digest = content_digest([path, ordinal])
        origin_kind = "explicit_span" if member.channel == "explicit" else "observation"
        origin_id = make_evidence_origin_id(
            SNAPSHOT, origin_kind, path, [ordinal, ordinal], source_digest
        )
        origins.append(
            EvidenceOrigin(
                origin_id=origin_id,
                origin_kind=origin_kind,
                path=path,
                line_start=ordinal,
                line_end=ordinal,
                source_content_digest=source_digest,
                independence_group=f"file:{path}",
            )
        )
        artifact_ref = _ref(
            "explicit_evidence" if member.channel == "explicit" else "observation",
            f"source-{ordinal}",
            path=(
                "explicit_verification.json"
                if member.channel == "explicit"
                else "observations/index.json"
            ),
        )
        evidence_id = make_evidence_id(
            origin_id, "support", [member.member_id], artifact_ref.stable_id
        )
        items.append(
            CurrentEvidenceItem(
                evidence_id=evidence_id,
                origin_id=origin_id,
                role="support",
                member_ids=[member.member_id],
                artifact_ref=artifact_ref,
                excerpt_digest=source_digest,
                description="Exact current evidence.",
            )
        )
        if member.channel == "implicit_current":
            denominators.append(
                PopulationDenominator(
                    member_id=member.member_id,
                    population_id=f"POP-{ordinal}",
                    status="complete",
                    census_enumeration_status="complete",
                    sampling_plan_status="complete",
                    sampling_mode="census",
                    scanned=1,
                    serialized=1,
                    matched=1,
                    inspected=1,
                    uninspected=0,
                    support=1,
                    counterexample=0,
                    unknown=0,
                    ineligible=0,
                )
            )
    payload = {
        "bundle_id": make_current_evidence_bundle_id(SNAPSHOT, PROFILE, case.case_id),
        "case_id": case.case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "origins": sorted(origins, key=lambda item: item.origin_id),
        "items": sorted(items, key=lambda item: item.evidence_id),
        "denominators": sorted(denominators, key=lambda item: item.member_id),
        "population_refs": [],
        "falsification_refs": [],
        "scope_condition_refs": [],
        "dependencies": [
            CapabilityDependency(
                capability="current_evidence",
                requirement_mode="required",
                status="complete" if status == "complete" else "partial",
                reason_codes=[] if status == "complete" else ["test_gap"],
            )
        ],
        "integrity": "valid",
        "status": status,
        "incomplete_reasons": [] if status == "complete" else ["test_gap"],
    }
    return CurrentEvidenceBundle.model_validate(seal_payload(payload))


def _finding(
    case: CandidateCase,
    evidence: CurrentEvidenceBundle,
    assessments: list[ChannelAssessment],
    *,
    conflicts: bool = False,
    condition_support: str = "pass",
):
    item_by_member = {
        member_id: item.evidence_id
        for item in evidence.items
        for member_id in item.member_ids
    }
    findings = []
    assessment_by_id = {item.assessment_id: item for item in assessments}
    for member in case.members:
        assessment = assessment_by_id[member.assessment_id]
        findings.append(
            UnifiedMemberFinding(
                member_id=member.member_id,
                norm_relevance="pass",
                rule_support="pass",
                scope_support="pass",
                condition_support=condition_support,
                temporal_status="not_required",
                evidence_sufficiency="pass",
                evidence_refs=[item_by_member[member.member_id]],
                closed_local_reason_codes=list(assessment.typed_reasons),
                reason_codes=["verified"],
            )
        )
    relations = []
    for left, right in combinations(case.members, 2):
        relations.append(
            UnifiedMemberRelation(
                left_member_id=left.member_id,
                right_member_id=right.member_id,
                relation="conflicting" if conflicts else "distinct",
                evidence_refs=sorted(
                    [item_by_member[left.member_id], item_by_member[right.member_id]]
                ),
                reason_code="verified_relation",
            )
        )
    payload = {
        "finding_id": make_finding_id(SNAPSHOT, PROFILE, case.case_id, 0),
        "case_id": case.case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "verification_round": 0,
        "member_findings": findings,
        "member_relations": relations,
        "unresolved_conflicts": [],
        "conclusions": [],
        "refinement_proposals": [],
        "inference_ref": _ref(
            "inference_request",
            "INFER-" + "a" * 24,
            path="inference/requests/INFER-" + "a" * 24 + ".json",
        ),
        "status": "complete",
        "failure_reasons": [],
    }
    return UnifiedCaseFinding.model_validate(seal_payload(payload))


def _finding_with_factual_failure(
    case: CandidateCase,
    evidence: CurrentEvidenceBundle,
    assessment: ChannelAssessment,
    *,
    axis: str,
    evidence_refs: list[str],
    temporal_status: str = "not_required",
    condition_support: str = "pass",
) -> UnifiedCaseFinding:
    finding = _finding(
        case,
        evidence,
        [assessment],
        condition_support=condition_support,
    )
    payload = finding.model_dump(mode="python", exclude={"artifact_digest"})
    payload["member_findings"][0][axis] = "fail"
    payload["member_findings"][0]["temporal_status"] = temporal_status
    payload["member_findings"][0]["evidence_refs"] = sorted(evidence_refs)
    return UnifiedCaseFinding.model_validate(seal_payload(payload))


def _with_current_rejection_role(
    evidence: CurrentEvidenceBundle,
    *,
    role: str,
) -> CurrentEvidenceBundle:
    payload = evidence.model_dump(mode="python", exclude={"artifact_digest"})
    item = payload["items"][0]
    item["role"] = role
    item["evidence_id"] = make_evidence_id(
        item["origin_id"],
        role,
        item["member_ids"],
        item["artifact_ref"]["stable_id"],
    )
    return CurrentEvidenceBundle.model_validate(seal_payload(payload))


def _adverse_temporal(
    case: CandidateCase,
    *,
    relation: str,
) -> tuple[str, TemporalRelationMatrix, GitTemporalBundle]:
    item_id = "GITITEM-" + "9" * 24
    member_id = case.members[0].member_id
    change_slice = GitChangeSlice.model_construct(
        item_id=item_id,
        snapshot_digest=SNAPSHOT,
        commit_id="4" * 40,
        parent_id="5" * 40,
        before_path="src/old.py",
        after_path="src/new.py",
        artifact_digest="6" * 64,
    )
    matrix = TemporalRelationMatrix.model_construct(
        case_id=case.case_id,
        relations=[
            TemporalMemberRelation(
                item_id=item_id,
                member_id=member_id,
                relation=relation,
                affected_facets=["currentness"],
                evidence_span="git-span:1",
                reason_code="adverse_history",
                uncertainty_reason=None,
            )
        ],
    )
    bundle = GitTemporalBundle.model_construct(slices=[change_slice])
    return item_id, matrix, bundle


def test_incomplete_case_may_omit_over_budget_relation_matrix() -> None:
    assessments = [_assessment(f"root-{ordinal}") for ordinal in range(3)]
    case = _case(assessments, complete=False)
    assert case.grouping_status == "incomplete"
    assert case.relations == []

    payload = case.model_dump(mode="python", exclude={"artifact_digest"})
    payload["grouping_status"] = "complete"
    payload["grouping_reason_codes"] = []
    with pytest.raises(ValidationError, match="relation matrix is incomplete"):
        CandidateCase.model_validate(seal_payload(payload))


def test_channel_assessment_digest_and_snapshot_are_immutable() -> None:
    assessment = _assessment("root-a")
    payload = assessment.model_dump(mode="python")
    payload["raw_channel_status"] = "tampered"
    with pytest.raises(ValidationError, match="assessment digest"):
        ChannelAssessment.model_validate(payload)


def test_high_authority_explicit_fast_path_accepts_without_verifier_or_git() -> None:
    assessment = _assessment("root-fast")
    case = _case([assessment])
    evidence = _current_evidence(case)
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=None,
    )
    assert result.fast_path is True
    assert [item.epistemic_status for item in result.claims] == ["accepted"]
    assert result.claims[0].delivery_policy == "default_consider"
    assert result.claims[0].review_facets == []
    assert result.claims[0].rejection_kind is None
    assert result.claims[0].rejection_basis_refs == []
    assert result.dispositions[0].disposition == "adopted_as_support"
    assert result.dispositions[0].reason_codes == [
        "explicit_local_positive_fast_path"
    ]
    assert result.reason_codes == ("explicit_local_positive_fast_path",)


def test_explicit_local_positive_fast_path_is_independent_of_authority() -> None:
    direct = _assessment("root-direct")
    weak = _assessment(
        "root-weak",
        authority="weak_or_unknown",
        strength="recommended",
    )
    case = _case([direct, weak])
    evidence = _current_evidence(case)
    result = apply_case_policy(
        case=case,
        assessments={
            direct.assessment_id: direct,
            weak.assessment_id: weak,
        },
        current_evidence=evidence,
        temporal_matrix=None,
        finding=None,
    )

    assert result.fast_path is True
    assert {item.epistemic_status for item in result.claims} == {"accepted"}
    assert all(
        item.reason_codes == ["explicit_local_positive_fast_path"]
        for item in result.dispositions
    )


@pytest.mark.parametrize(
    "blocked_gate",
    [
        "grouping_incomplete",
        "affirmative_git_need",
        "counterevidence",
        "required_gap",
        "locally_uncertain",
        "locally_rejected",
        "invalid",
        "axis_not_passed",
    ],
)
def test_explicit_local_positive_fast_path_preserves_safety_gates(
    blocked_gate: str,
) -> None:
    assessment = _assessment(
        f"root-fast-path-gate-{blocked_gate}",
        authority="weak_or_unknown",
        strength="observed",
    )
    assessment_payload = assessment.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    if blocked_gate == "affirmative_git_need":
        assessment_payload["git_need"] = {
            "status": "affirmative", "reason": None,
            "judgments": [{"status": "affirmative", "questions": [{
                "question": "Was the recorded migration reverted?",
                "current_gap": "The migration note does not establish the replacement relationship.",
                "answer_use": "Resolve currentness of the recorded replacement.",
                "path": "README.md", "evidence_refs": ["fixture-evidence"], "blocking_axis": None,
            }]}],
        }
    elif blocked_gate in {"locally_uncertain", "locally_rejected"}:
        assessment_payload["normalized_status"] = blocked_gate
    elif blocked_gate == "invalid":
        assessment_payload["normalized_status"] = "invalid"
        assessment_payload["integrity"] = "invalid"
    elif blocked_gate == "axis_not_passed":
        assessment_payload["axis_findings"]["evidence_entailment"] = "uncertain"
    assessment = ChannelAssessment.model_validate(seal_payload(assessment_payload))
    case = _case(
        [assessment],
        complete=blocked_gate != "grouping_incomplete",
    )
    evidence = _current_evidence(
        case,
        status="partial" if blocked_gate == "required_gap" else "complete",
    )
    if blocked_gate == "counterevidence":
        evidence = _with_current_rejection_role(
            evidence,
            role="counterevidence",
        )

    assert not is_explicit_fast_path(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
    )


def test_incomplete_dependency_precedes_expected_missing_verifier() -> None:
    assessment = _assessment(
        "root-incomplete-current",
        channel="implicit_current",
        authority="current_population",
        strength="observed",
    )
    case = _case([assessment])
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=_current_evidence(case, status="partial"),
        temporal_matrix=None,
        finding=None,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["technical_or_coverage_gap"]
    assert result.claims[0].delivery_policy == "not_applicable"
    assert result.dispositions[0].reason_codes == [
        "required_capability_incomplete"
    ]


@pytest.mark.parametrize(
    ("blocker", "other_reason"),
    [
        ("dependency_no_finding", "required_capability_incomplete"),
        ("dependency_all_pass", "required_capability_incomplete"),
        ("grouping", "grouping_incomplete"),
        ("missing_finding", "unified_verification_missing"),
        ("unresolved_conflict", "unresolved_case_conflict"),
        ("refinement_pending", "refinement_requires_round_one_verification"),
        ("all_pass", None),
        ("current_contradiction", None),
        ("explicit_fast_path", None),
    ],
)
def test_directory_marker_retains_scope_blocker_across_policy_exits(
    blocker: str, other_reason: str | None,
) -> None:
    explicit = blocker == "explicit_fast_path"
    assessment = _assessment(
        "directory-marker-" + blocker,
        channel="explicit" if explicit else "implicit_current",
        authority="direct_normative" if explicit else "current_population",
        strength="required" if explicit else "observed",
    )
    payload = assessment.model_dump(mode="python", exclude={"artifact_digest"})
    payload["semantics"]["applies_to"] = ["src/"]
    assessment = ChannelAssessment.model_validate(seal_payload(payload))
    case = _case([assessment], complete=blocker != "grouping")
    evidence = _current_evidence(
        case, status="partial" if blocker.startswith("dependency_") else "complete",
    )
    if blocker == "current_contradiction":
        evidence = _with_current_rejection_role(evidence, role="counterevidence")
        finding = _finding_with_factual_failure(
            case, evidence, assessment, axis="rule_support",
            evidence_refs=[evidence.items[0].evidence_id],
        )
    elif blocker in {"dependency_no_finding", "grouping", "missing_finding"}:
        finding = None
    else:
        finding = _finding(case, evidence, [assessment])
    if blocker in {"unresolved_conflict", "refinement_pending"}:
        payload = finding.model_dump(mode="python", exclude={"artifact_digest"})
        if blocker == "unresolved_conflict":
            payload["unresolved_conflicts"] = ["fixture_unresolved_conflict"]
        else:
            payload["refinement_proposals"] = [{
                "git_need": {"status": "no_established_need", "questions": []},
                "parent_member_id": case.members[0].member_id,
                "semantics": assessment.semantics.model_copy(update={"applies_to": ["src/core.py"]}),
                "evidence_refs": [evidence.items[0].evidence_id],
                "change_kinds": ["scope"],
            }]
        finding = UnifiedCaseFinding.model_validate(seal_payload(payload))
    result = apply_case_policy(
        case=case, assessments={assessment.assessment_id: assessment},
        current_evidence=evidence, temporal_matrix=None, finding=finding,
    )
    assert not result.fast_path
    claim = result.claims[0]
    assert claim.epistemic_status == "review"
    assert claim.semantics.applies_to == ["src/"]
    assert claim.delivery_policy == "not_applicable"
    assert claim.rejection_kind is None
    expected_reasons = ["directory_marker_scope_requires_refinement"]
    expected_facets = ["refinement_or_qualifier_needed"]
    if other_reason is not None:
        expected_reasons.append(other_reason)
        if blocker == "unresolved_conflict":
            expected_facets.append("conflict_or_currentness")
        elif blocker != "refinement_pending":
            expected_facets.append("technical_or_coverage_gap")
    assert claim.reason_codes == sorted(expected_reasons)
    assert claim.review_facets == sorted(expected_facets)
    assert result.dispositions[0].reason_codes == sorted(expected_reasons)
    if blocker.startswith("dependency_"):
        assert claim.dependency_status == "incomplete"
        assert "unified_verification_missing" not in claim.reason_codes
    if blocker == "current_contradiction":
        assert claim.counterevidence


def test_current_source_can_be_accepted_without_positive_git_support() -> None:
    assessment = _assessment(
        "root-current",
        channel="implicit_current",
        authority="current_population",
        strength="observed",
        reasons=["pre_calibration_review_only"],
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [assessment])
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )
    assert result.fast_path is False
    assert result.claims[0].epistemic_status == "accepted"
    assert result.claims[0].normative_strength == "observed"


def test_required_falsification_gap_cannot_be_closed_by_passing_model() -> None:
    reason = "falsification_cost_not_admitted"
    assessment = _assessment(
        "required-falsification-gap", channel="implicit_current",
        authority="current_population", strength="observed", reasons=[reason],
    )
    case = _case([assessment])
    evidence = _current_evidence(case, status="partial")
    payload = evidence.model_dump(mode="python", exclude={"artifact_digest"})
    payload["dependencies"] = [CapabilityDependency(
        capability="current_falsification", requirement_mode="required",
        status="partial", reason_codes=[reason],
    )]
    evidence = CurrentEvidenceBundle.model_validate(seal_payload(payload))
    finding = _finding(case, evidence, [assessment])
    assert finding.member_findings[0].closed_local_reason_codes == [reason]
    result = apply_case_policy(
        case=case, assessments={assessment.assessment_id: assessment},
        current_evidence=evidence, temporal_matrix=None, finding=finding,
    )
    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["technical_or_coverage_gap"]
    assert result.dispositions[0].reason_codes == ["required_capability_incomplete"]


@pytest.mark.parametrize(
    ("channel", "authority", "strength", "expected_status", "expected_reason"),
    [
        (
            "explicit",
            "direct_normative",
            "required",
            "accepted",
            "unified_fact_axes_passed",
        ),
        (
            "implicit_current",
            "current_population",
            "observed",
            "review",
            "local_rejection_requires_independent_member",
        ),
    ],
)
def test_local_rejection_override_is_explicit_only(
    channel: str,
    authority: str,
    strength: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    if channel == "explicit":
        assessment = _locally_relevance_rejected_normative_assessment(
            "root-local-rejection-explicit",
            rule="Run pytest for every change.",
        )
    else:
        assessment = _assessment(
            "root-local-rejection-implicit-current",
            channel=channel,
            status="locally_rejected",
            authority=authority,
            strength=strength,
            reasons=["repository_specificity_failed"],
        )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [assessment])
    if channel == "explicit":
        finding_payload = finding.model_dump(
            mode="python", exclude={"artifact_digest"}
        )
        finding_payload["member_findings"][0]["closed_local_reason_codes"] = []
        finding = UnifiedCaseFinding.model_validate(seal_payload(finding_payload))

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.fast_path is False
    assert result.claims[0].epistemic_status == expected_status
    assert result.dispositions[0].reason_codes == [expected_reason]
    assert result.dispositions[0].disposition == (
        "adopted_as_support" if channel == "explicit" else "review"
    )


def test_explicit_local_rejection_still_reviews_when_unified_axis_fails() -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        "root-explicit-local-rejection-unified-failure",
        rule="Run pytest for every change.",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="scope_support",
        evidence_refs=[evidence.items[0].evidence_id],
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.fast_path is False
    assert result.claims[0].epistemic_status == "review"
    assert result.dispositions[0].reason_codes == [
        "unified_fact_axis_not_passed"
    ]
    assert "local_rejection_requires_independent_member" not in (
        result.dispositions[0].reason_codes
    )


def test_explicit_local_rejection_still_obeys_current_contradiction() -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        "root-explicit-local-rejection-current-contradiction",
        rule="Run pytest for every change.",
    )
    case = _case([assessment])
    evidence = _with_current_rejection_role(
        _current_evidence(case),
        role="counterevidence",
    )
    evidence_id = evidence.items[0].evidence_id
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="rule_support",
        evidence_refs=[evidence_id],
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.fast_path is False
    assert result.claims[0].epistemic_status == "rejected"
    assert result.claims[0].rejection_kind == (
        "current_snapshot_contradiction"
    )
    assert result.claims[0].rejection_basis_refs == [evidence_id]
    assert result.dispositions[0].reason_codes == [
        "current_snapshot_positive_rejection"
    ]


def test_explicit_local_rejection_still_obeys_adverse_temporal_gate() -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        "root-explicit-local-rejection-adverse-temporal",
        rule="Run pytest for every change.",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    item_id, matrix, git_bundle = _adverse_temporal(
        case,
        relation="contradicts_as_written",
    )
    finding = _finding(case, evidence, [assessment])
    finding_payload = finding.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    finding_payload["member_findings"][0]["temporal_status"] = (
        "adverse_unresolved"
    )
    finding_payload["member_findings"][0]["evidence_refs"] = sorted(
        [evidence.items[0].evidence_id, item_id]
    )
    finding = UnifiedCaseFinding.model_validate(seal_payload(finding_payload))

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=matrix,
        git_bundle=git_bundle,
        finding=finding,
    )

    assert result.fast_path is False
    assert result.claims[0].epistemic_status == "review"
    assert result.dispositions[0].reason_codes == [
        "temporal_status_unresolved"
    ]


def test_conditionless_member_accepts_not_applicable_condition_axis() -> None:
    assessment = _assessment(
        "root-conditionless",
        status="locally_uncertain",
        authority="descriptive_typed",
        strength="observed",
        claim_kind="operational_fact",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(
        case,
        evidence,
        [assessment],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.fast_path is False
    assert result.claims[0].epistemic_status == "accepted"
    assert result.claims[0].delivery_policy == "default_consider"
    assert result.dispositions[0].reason_codes == ["unified_fact_axes_passed"]


@pytest.mark.parametrize(
    ("condition", "exception", "condition_support"),
    [
        ("When editing documentation.", None, "not_applicable"),
        (None, "Except generated files.", "not_applicable"),
        (None, None, "fail"),
        (None, None, "uncertain"),
    ],
)
def test_condition_axis_still_fails_closed_when_not_applicable_is_invalid(
    condition: str | None,
    exception: str | None,
    condition_support: str,
) -> None:
    assessment = _assessment(
        f"root-condition-{condition_support}",
        status="locally_uncertain",
        authority="descriptive_typed",
        strength="observed",
        claim_kind="operational_fact",
        condition=condition,
        exception=exception,
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(
        case,
        evidence,
        [assessment],
        condition_support=condition_support,
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == [
        "refinement_or_qualifier_needed"
    ]
    assert result.dispositions[0].reason_codes == [
        "unified_fact_axis_not_passed"
    ]


@pytest.mark.parametrize(
    "relation",
    ["contradicts_as_written", "retires_or_replaces"],
)
def test_adverse_temporal_evidence_cannot_singlehandedly_reject(
    relation: str,
) -> None:
    assessment = _assessment(
        f"root-temporal-{relation}",
        channel="implicit_current",
        authority="current_population",
        strength="observed",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    item_id, matrix, git_bundle = _adverse_temporal(case, relation=relation)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="rule_support",
        evidence_refs=[item_id],
        temporal_status="adverse_unresolved",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=matrix,
        git_bundle=git_bundle,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["negative_basis_insufficient"]
    assert result.dispositions[0].disposition == "review"
    assert result.dispositions[0].reason_codes == [
        "current_snapshot_rejection_basis_missing"
    ]
    assert [item.kind for item in result.claims[0].counterevidence] == [
        "temporal"
    ]


def test_cited_current_snapshot_counterevidence_can_reject() -> None:
    role = "counterevidence"
    assessment = _assessment(
        f"root-current-rejection-{role}",
        channel="implicit_current",
        authority="current_population",
        strength="observed",
    )
    case = _case([assessment])
    evidence = _with_current_rejection_role(
        _current_evidence(case),
        role=role,
    )
    evidence_id = evidence.items[0].evidence_id
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="rule_support",
        evidence_refs=[evidence_id],
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "rejected"
    assert result.claims[0].review_facets == []
    assert result.claims[0].delivery_policy == "not_applicable"
    assert result.claims[0].rejection_kind == (
        "current_snapshot_contradiction"
    )
    assert result.claims[0].rejection_basis_refs == [evidence_id]
    assert result.dispositions[0].disposition == "rejected"
    assert result.dispositions[0].reason_codes == [
        "current_snapshot_positive_rejection"
    ]
    assert [item.evidence_id for item in result.claims[0].counterevidence] == [
        evidence_id
    ]

    payload = result.claims[0].model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    payload["rejection_kind"] = None
    with pytest.raises(
        ValidationError,
        match="rejected canonical claim requires kind and basis refs",
    ):
        CanonicalClaim.model_validate(seal_payload(payload))


def test_supported_relevance_failure_abstains_without_typed_domain_proof() -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        "root-outside-coding-scope"
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    evidence_id = evidence.items[0].evidence_id
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence_id],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    claim = result.claims[0]
    assert claim.epistemic_status == "review"
    assert claim.review_facets == ["factual_uncertainty"]
    assert claim.rejection_kind is None
    assert claim.rejection_basis_refs == []
    assert [item.evidence_id for item in claim.support] == [evidence_id]
    assert claim.counterevidence == []
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


def test_relevance_failure_with_current_counterevidence_still_abstains() -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        "root-relevance-failure-with-current-counterevidence"
    )
    case = _case([assessment])
    evidence = _with_current_rejection_role(
        _current_evidence(case),
        role="counterevidence",
    )
    evidence_id = evidence.items[0].evidence_id
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence_id],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    claim = result.claims[0]
    assert claim.epistemic_status == "review"
    assert claim.review_facets == ["factual_uncertainty"]
    assert claim.rejection_kind is None
    assert claim.rejection_basis_refs == []
    assert claim.support == []
    assert [item.evidence_id for item in claim.counterevidence] == [evidence_id]
    assert result.dispositions[0].disposition == "review"
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


@pytest.mark.parametrize(
    "claim_kind",
    [
        "command_recipe",
        "workflow_fact",
        "operational_fact",
        "enforced_configuration",
    ],
)
def test_descriptive_fact_cannot_be_rejected_as_outside_coding_scope(
    claim_kind: str,
) -> None:
    assessment = _assessment(
        f"root-outside-scope-{claim_kind}",
        status="locally_rejected",
        authority="descriptive_typed",
        strength="observed",
        claim_kind=claim_kind,
        reasons=["coding_relevance_failed"],
    )
    assessment_payload = assessment.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    assessment_payload["axis_findings"]["coding_relevance"] = "fail"
    assessment = ChannelAssessment.model_validate(
        seal_payload(assessment_payload)
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence.items[0].evidence_id],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    claim = result.claims[0]
    assert claim.epistemic_status == "review"
    assert claim.review_facets == ["factual_uncertainty"]
    assert claim.rejection_kind is None
    assert claim.rejection_basis_refs == []
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


def test_out_of_scope_rejection_requires_independent_local_rejection() -> None:
    assessment = _assessment(
        "root-outside-scope-without-local-rejection",
        status="locally_uncertain",
        authority="descriptive_typed",
        claim_kind="normative_declaration",
        rule="When answering chat messages, list pros and cons.",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence.items[0].evidence_id],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["factual_uncertainty"]
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


@pytest.mark.parametrize(
    ("channel", "strength"),
    [
        ("explicit", "observed"),
        ("implicit_current", "required"),
    ],
)
def test_out_of_scope_rejection_requires_explicit_normative_strength(
    channel: str,
    strength: str,
) -> None:
    assessment = _assessment(
        f"root-outside-scope-{channel}-{strength}",
        channel=channel,
        status="locally_rejected",
        authority=(
            "direct_normative" if channel == "explicit" else "current_population"
        ),
        strength=strength,
        claim_kind="normative_declaration",
        reasons=["coding_relevance_failed"],
        rule="When answering chat messages, list pros and cons.",
    )
    payload = assessment.model_dump(mode="python", exclude={"artifact_digest"})
    if channel == "explicit":
        payload["axis_findings"]["coding_relevance"] = "fail"
    assessment = ChannelAssessment.model_validate(seal_payload(payload))
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence.items[0].evidence_id],
        condition_support="not_applicable",
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["factual_uncertainty"]
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


@pytest.mark.parametrize(
    ("axis", "value"),
    [
        ("rule_support", "fail"),
        ("scope_support", "fail"),
        ("evidence_sufficiency", "fail"),
    ],
)
def test_relevance_failure_with_additional_axis_failure_stays_factual_review(
    axis: str,
    value: str,
) -> None:
    assessment = _locally_relevance_rejected_normative_assessment(
        f"root-outside-scope-ungrounded-{axis}"
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence.items[0].evidence_id],
        condition_support="not_applicable",
    )
    payload = finding.model_dump(mode="python", exclude={"artifact_digest"})
    payload["member_findings"][0][axis] = value
    finding = UnifiedCaseFinding.model_validate(seal_payload(payload))

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["factual_uncertainty"]
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


def test_ineligible_population_member_is_audit_only_not_rejection_basis() -> None:
    assessment = _assessment(
        "root-current-rejection-ineligible",
        channel="implicit_current",
        authority="current_population",
        strength="observed",
    )
    case = _case([assessment])
    evidence = _with_current_rejection_role(
        _current_evidence(case),
        role="ineligible",
    )
    finding = _finding_with_factual_failure(
        case,
        evidence,
        assessment,
        axis="norm_relevance",
        evidence_refs=[evidence.items[0].evidence_id],
    )

    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert result.claims[0].epistemic_status == "review"
    assert result.claims[0].review_facets == ["factual_uncertainty"]
    assert result.claims[0].counterevidence == []
    assert result.dispositions[0].disposition == "review"
    assert result.dispositions[0].reason_codes == [
        "outside_scope_rejection_not_admitted"
    ]


def test_canonical_decision_audit_fields_are_status_exclusive() -> None:
    assessment = _assessment("root-audit-exclusive")
    case = _case([assessment])
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=_current_evidence(case),
        temporal_matrix=None,
        finding=None,
    )
    accepted = result.claims[0]

    payload = accepted.model_dump(mode="python", exclude={"artifact_digest"})
    payload["review_facets"] = ["factual_uncertainty"]
    with pytest.raises(
        ValidationError,
        match="accepted canonical claim cannot carry review/rejection audit",
    ):
        CanonicalClaim.model_validate(seal_payload(payload))

    payload = accepted.model_dump(mode="python", exclude={"artifact_digest"})
    payload["epistemic_status"] = "review"
    payload["delivery_policy"] = "reference_only"
    payload["review_facets"] = ["factual_uncertainty"]
    with pytest.raises(
        ValidationError,
        match="review/rejected canonical claim delivery policy must be not_applicable",
    ):
        CanonicalClaim.model_validate(seal_payload(payload))


def test_cross_case_claim_identity_accumulator_fails_closed_losslessly() -> None:
    assessments = [
        _assessment("root-collision-a"),
        _assessment("root-collision-b"),
    ]
    results = []
    for assessment in assessments:
        case = _case([assessment])
        results.append(
            apply_case_policy(
                case=case,
                assessments={assessment.assessment_id: assessment},
                current_evidence=_current_evidence(case),
                temporal_matrix=None,
                finding=None,
            )
        )
    assert results[0].claims[0].claim_id == results[1].claims[0].claim_id

    accumulator = CanonicalPolicyAccumulator()
    for result in results:
        accumulator.absorb(result)
    aggregated = accumulator.finalize()

    assert aggregated.claims == ()
    assert aggregated.reason_codes == (CANONICAL_IDENTITY_COLLISION_REASON,)
    expected_members = {
        disposition.member_id
        for result in results
        for disposition in result.dispositions
    }
    assert set(aggregated.collision_member_ids) == expected_members
    assert {item.member_id for item in aggregated.dispositions} == expected_members
    assert all(
        item.disposition == "review"
        and item.canonical_claim_ids == []
        and item.reason_codes == [CANONICAL_IDENTITY_COLLISION_REASON]
        for item in aggregated.dispositions
    )


def test_effective_machine_configuration_maps_to_required_strength() -> None:
    assessment = _assessment(
        "root-config",
        authority="descriptive_typed",
        strength="observed",
        claim_kind="enforced_configuration",
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [assessment])
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )
    assert result.claims[0].epistemic_status == "accepted"
    assert result.claims[0].normative_strength == "required"


def test_obsolete_channel_calibration_reason_has_no_canonical_gate() -> None:
    assessment = _assessment(
        "root-refined-current",
        channel="implicit_current",
        status="locally_uncertain",
        authority="current_population",
        strength="observed",
        reasons=[
            "post_falsification_refinement",
            "refinement_requires_calibration",
        ],
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [assessment])
    payload = finding.model_dump(mode="python", exclude={"artifact_digest"})
    payload["member_findings"][0]["closed_local_reason_codes"] = []
    finding = UnifiedCaseFinding.model_validate(seal_payload(payload))
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )
    assert result.claims[0].epistemic_status == "accepted"


def test_local_review_reason_must_be_explicitly_closed() -> None:
    assessment = _assessment(
        "root-review",
        channel="implicit_current",
        status="locally_uncertain",
        authority="current_population",
        strength="observed",
        reasons=["unresolved_member"],
    )
    case = _case([assessment])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [assessment])
    finding_payload = finding.model_dump(mode="python", exclude={"artifact_digest"})
    finding_payload["member_findings"][0]["closed_local_reason_codes"] = []
    finding = UnifiedCaseFinding.model_validate(seal_payload(finding_payload))
    result = apply_case_policy(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )
    assert result.claims[0].epistemic_status == "review"
    assert "local_review_reason_unclosed" in result.claims[0].reason_codes
    assert result.claims[0].review_facets == ["channel_disagreement"]


def test_accepted_equivalence_claim_excludes_review_member_reasons() -> None:
    supported = _assessment("root-equivalent-supported")
    local_rejected = _assessment(
        "root-equivalent-rejected",
        channel="implicit_current",
        status="locally_rejected",
        authority="current_population",
        strength="observed",
    )
    assessments = [supported, local_rejected]
    case = _case(assessments)
    case_payload = case.model_dump(mode="python", exclude={"artifact_digest"})
    case_payload["relations"][0].update(
        {
            "relation": "equivalent",
            "source": "semantic_grouping",
            "reason_code": "same_coding_decision",
        }
    )
    case = CandidateCase.model_validate(seal_payload(case_payload))
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, assessments)
    finding_payload = finding.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    finding_payload["member_relations"][0]["relation"] = "equivalent"
    finding = UnifiedCaseFinding.model_validate(seal_payload(finding_payload))

    result = apply_case_policy(
        case=case,
        assessments={item.assessment_id: item for item in assessments},
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.epistemic_status == "accepted"
    assert claim.reason_codes == ["unified_fact_axes_passed"]
    assert "local_rejection_requires_independent_member" not in claim.reason_codes
    assert {item.disposition for item in result.dispositions} == {
        "adopted_as_support",
        "review",
    }


def test_explicit_requirement_and_conflicting_observed_behavior_both_survive() -> None:
    explicit = _assessment("root-required")
    current = _assessment(
        "root-observed",
        channel="implicit_current",
        polarity="negative",
        authority="current_population",
        strength="observed",
        reasons=["pre_calibration_review_only"],
        rule="Do not run pytest for every change.",
    )
    case = _case([explicit, current])
    evidence = _current_evidence(case)
    finding = _finding(case, evidence, [explicit, current], conflicts=True)
    result = apply_case_policy(
        case=case,
        assessments={
            explicit.assessment_id: explicit,
            current.assessment_id: current,
        },
        current_evidence=evidence,
        temporal_matrix=None,
        finding=finding,
    )
    assert [item.epistemic_status for item in result.claims] == ["accepted", "accepted"]
    by_strength = {item.normative_strength: item for item in result.claims}
    assert by_strength["required"].delivery_policy == "default_consider"
    assert by_strength["observed"].delivery_policy == "reference_only"
    assert by_strength["required"].conflicts_with == [by_strength["observed"].claim_id]
    assert by_strength["observed"].conflicts_with == [by_strength["required"].claim_id]
