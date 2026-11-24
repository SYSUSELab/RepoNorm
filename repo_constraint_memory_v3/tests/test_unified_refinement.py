from __future__ import annotations

from itertools import combinations

import pytest

from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    make_assessment_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
    make_finding_id,
    make_member_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateSemantics,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    PopulationDenominator,
    RefinementProposal,
    UNIFIED_CONTRACT_VERSION,
    UnifiedCaseFinding,
    UnifiedMemberFinding,
    UnifiedMemberRelation,
)
from repo_constraint_memory_v3.unified.refinement import (
    CurrentEvidenceReacquisition,
    GitReacquisition,
    RoundOneVerification,
    execute_refinement_round,
)

SNAPSHOT = "sha256:" + "a" * 64
PROFILE = "b" * 64


def _ref(kind: str, stable_id: str, seed: str) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id,
        snapshot_digest=SNAPSHOT,
        artifact_path=f"upstream/{seed}.json",
        artifact_digest=content_digest([kind, stable_id, seed]),
    )


def _semantics(
    *,
    rule: str = "Register plugins through PluginManager",
    action: str = "register-plugins-through-manager",
    polarity: str = "positive",
    scope: tuple[str, ...] = ("src/**",),
    condition: str | None = None,
) -> CandidateSemantics:
    return CandidateSemantics(
        rule=rule,
        normalized_action=action,
        polarity=polarity,
        applies_to=sorted(scope),
        condition=condition,
        exception=None,
    )


def _assessment(
    semantics: CandidateSemantics | None = None,
    *,
    seed: str = "plugin-manager",
    candidate_version: int = 0,
) -> ChannelAssessment:
    candidate_root_id = f"ROOT-{seed}"
    assessment_id = make_assessment_id(
        SNAPSHOT,
        PROFILE,
        "implicit_current",
        candidate_root_id,
        candidate_version,
    )
    payload: dict[str, object] = {
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "assessment_id": assessment_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "channel": "implicit_current",
        "candidate_root_id": candidate_root_id,
        "candidate_version": candidate_version,
        "parent_candidate_id": (
            None if candidate_version == 0 else f"CHANNEL-PARENT-{seed}"
        ),
        "semantics": semantics or _semantics(),
        "proposed_claim_kind": "observed_convention",
        "proposed_strength": "observed",
        "raw_channel_status": "accept_candidate",
        "raw_policy_version": "current-source-policy-1",
        "normalized_status": "locally_supported",
        "axis_findings": {"host_grounding": "pass"},
        "typed_reasons": ["population_supported"],
        "support_refs": [_ref("population", f"POP-{seed}", f"population-{seed}")],
        "counterevidence_refs": [],
        "coverage_refs": [],
        "lineage_refs": [],
        "extension_refs": [],
        "integrity": "valid",
        "source_authority": "current_population",
        "git_need": {"status": "not_assessed", "judgments": [], "reason": "host_fixture"},
        "provenance": [f"test:{seed}"],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def _member(assessment: ChannelAssessment) -> CandidateCaseMember:
    member_id = make_member_id(SNAPSHOT, assessment.assessment_id)
    return CandidateCaseMember(
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
                "unified/channel-assessments/"
                f"{assessment.assessment_id}.json"
            ),
            artifact_digest=assessment.artifact_digest,
        ),
    )


def _case(member: CandidateCaseMember) -> CandidateCase:
    case_id = make_case_id(SNAPSHOT, [member.member_id])
    payload: dict[str, object] = {
        "case_id": case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "members": [member],
        "relations": [],
        "grouping_status": "complete",
        "grouping_reason_codes": [],
        "exact_identity_groups": [],
        "required_capabilities": ["current_evidence"],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CandidateCase.model_validate(payload)


def _evidence_parts(
    member_id: str,
    *,
    path: str,
    seed: str,
    role: str,
    population_id: str,
) -> tuple[EvidenceOrigin, CurrentEvidenceItem]:
    source_digest = content_digest(["source", seed])
    origin_id = make_evidence_origin_id(
        SNAPSHOT,
        "population_member",
        path,
        [10, 10],
        source_digest,
    )
    origin = EvidenceOrigin(
        origin_id=origin_id,
        origin_kind="population_member",
        path=path,
        line_start=10,
        line_end=10,
        source_content_digest=source_digest,
        independence_group=f"independent-{seed}",
    )
    artifact = _ref("population_member", f"POPMEMBER-{seed}", seed)
    evidence_id = make_evidence_id(
        origin_id, role, [member_id], artifact.stable_id
    )
    item = CurrentEvidenceItem(
        evidence_id=evidence_id,
        origin_id=origin_id,
        role=role,
        member_ids=[member_id],
        artifact_ref=artifact,
        excerpt_digest=content_digest(["excerpt", seed]),
        description=f"Population evidence {seed} from {population_id}",
        anchor_literals=["PluginManager"],
        anchor_symbols=["PluginManager.register"],
    )
    return origin, item


def _current_bundle(
    case: CandidateCase,
    *,
    entries: tuple[tuple[str, str, str], ...] = (
        ("src/core/plugin.py", "core", "support"),
        ("src/legacy/plugin.py", "legacy", "counterevidence"),
    ),
    uninspected: int = 0,
) -> CurrentEvidenceBundle:
    member_id = case.members[0].member_id
    population_id = "POP-parent"
    pairs = [
        _evidence_parts(
            member_id,
            path=path,
            seed=seed,
            role=role,
            population_id=population_id,
        )
        for path, seed, role in entries
    ]
    counts = {
        "support": sum(role == "support" for _, _, role in entries),
        "counterexample": sum(
            role == "counterevidence" for _, _, role in entries
        ),
        "unknown": sum(role == "unknown" for _, _, role in entries),
        "ineligible": sum(role == "ineligible" for _, _, role in entries),
    }
    inspected = sum(counts.values())
    serialized = inspected + uninspected
    payload: dict[str, object] = {
        "bundle_id": make_current_evidence_bundle_id(
            SNAPSHOT, PROFILE, case.case_id
        ),
        "case_id": case.case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "origins": sorted(
            [origin for origin, _ in pairs], key=lambda item: item.origin_id
        ),
        "items": sorted(
            [item for _, item in pairs], key=lambda item: item.evidence_id
        ),
        "denominators": [
            PopulationDenominator(
                member_id=member_id,
                population_id=population_id,
                status="complete",
                census_enumeration_status="complete",
                sampling_plan_status="complete",
                sampling_mode=(
                    "deterministic_stratified_sample"
                    if uninspected
                    else "census"
                ),
                scanned=serialized,
                serialized=serialized,
                matched=serialized,
                inspected=inspected,
                uninspected=uninspected,
                **counts,
            )
        ],
        "population_refs": [_ref("population", population_id, "population")],
        "falsification_refs": [
            _ref("falsification", "FALSIFY-parent", "falsification")
        ],
        "scope_condition_refs": [
            _ref("falsification", "FALSIFY-scope", "scope")
        ],
        "dependencies": [
            CapabilityDependency(
                capability="current_population",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            )
        ],
        "integrity": "valid",
        "status": "complete",
        "incomplete_reasons": [],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _proposal(
    member_id: str,
    evidence_id: str,
    semantics: CandidateSemantics,
    change_kinds: tuple[str, ...],
) -> RefinementProposal:
    return RefinementProposal(
        git_need={"status": "no_established_need", "questions": []},
        parent_member_id=member_id,
        semantics=semantics,
        evidence_refs=[evidence_id],
        change_kinds=sorted(change_kinds),
    )


def _finding(
    case: CandidateCase,
    evidence: CurrentEvidenceBundle,
    *,
    second_proposal: bool = False,
) -> UnifiedCaseFinding:
    evidence_ref = evidence.items[0].evidence_id if evidence.items else None
    member_findings = [
        UnifiedMemberFinding(
            member_id=member.member_id,
            norm_relevance="pass",
            rule_support="pass",
            scope_support="pass",
            condition_support="pass",
            temporal_status="not_required",
            evidence_sufficiency="pass",
            evidence_refs=[] if evidence_ref is None else [evidence_ref],
            closed_local_reason_codes=[],
            reason_codes=[],
        )
        for member in case.members
    ]
    relations = [
        UnifiedMemberRelation(
            left_member_id=left,
            right_member_id=right,
            relation="compatible",
            evidence_refs=[] if evidence_ref is None else [evidence_ref],
            reason_code="round_one_compatible",
        )
        for left, right in combinations(
            [item.member_id for item in case.members], 2
        )
    ]
    refinements = []
    if second_proposal:
        parent = case.members[0]
        refinements = [
            RefinementProposal(
                git_need={"status": "no_established_need", "questions": []},
                parent_member_id=parent.member_id,
                semantics=parent.semantics.model_copy(
                    update={"rule": parent.semantics.rule + " precisely"}
                ),
                evidence_refs=[evidence_ref] if evidence_ref is not None else ["none"],
                change_kinds=["wording"],
            )
        ]
    payload: dict[str, object] = {
        "finding_id": make_finding_id(SNAPSHOT, PROFILE, case.case_id, 1),
        "case_id": case.case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "verification_round": 1,
        "member_findings": member_findings,
        "member_relations": relations,
        "unresolved_conflicts": [],
        "conclusions": [],
        "refinement_proposals": refinements,
        "inference_ref": _ref(
            "inference_request", "INFER-" + "a" * 24, "round-one"
        ),
        "status": "complete",
        "failure_reasons": [],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    if second_proposal:
        # Deliberately simulate an untrusted round-one artifact that bypassed
        # construction; the host adapter must revalidate and fail closed.
        return UnifiedCaseFinding.model_construct(**payload)
    return UnifiedCaseFinding.model_validate(payload)


class _Git:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reacquire_git(self, **kwargs: object) -> GitReacquisition:
        refined_case = kwargs["refined_case"]
        assert isinstance(refined_case, CandidateCase)
        self.calls.append(refined_case.case_id)
        return GitReacquisition(complete=True)


class _Verifier:
    def __init__(self, *, second_proposal: bool = False) -> None:
        self.second_proposal = second_proposal
        self.rounds: list[int] = []

    def verify_round_one_case(self, **kwargs: object) -> RoundOneVerification:
        case = kwargs["case"]
        evidence = kwargs["current_evidence"]
        verification_round = kwargs["verification_round"]
        assert isinstance(case, CandidateCase)
        assert isinstance(evidence, CurrentEvidenceBundle)
        assert isinstance(verification_round, int)
        self.rounds.append(verification_round)
        return RoundOneVerification(
            finding=_finding(
                case, evidence, second_proposal=self.second_proposal
            )
        )


class _SplitCurrent:
    def __init__(self) -> None:
        self.calls = 0

    def reacquire_current(self, **kwargs: object) -> CurrentEvidenceReacquisition:
        self.calls += 1
        refined_case = kwargs["refined_case"]
        assert isinstance(refined_case, CandidateCase)
        origins: list[EvidenceOrigin] = []
        items: list[CurrentEvidenceItem] = []
        denominators: list[PopulationDenominator] = []
        population_refs: list[ImmutableArtifactRef] = []
        for ordinal, member in enumerate(refined_case.members):
            population_id = f"POP-child-{ordinal}"
            path = member.semantics.applies_to[0].replace("**", "plugin.py")
            origin, item = _evidence_parts(
                member.member_id,
                path=path,
                seed=f"child-{ordinal}",
                role="support",
                population_id=population_id,
            )
            origins.append(origin)
            items.append(item)
            denominators.append(
                PopulationDenominator(
                    member_id=member.member_id,
                    population_id=population_id,
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
            population_refs.append(
                _ref("population", population_id, f"child-{ordinal}")
            )
        payload: dict[str, object] = {
            "bundle_id": make_current_evidence_bundle_id(
                SNAPSHOT, PROFILE, refined_case.case_id
            ),
            "case_id": refined_case.case_id,
            "snapshot_digest": SNAPSHOT,
            "profile_digest": PROFILE,
            "origins": sorted(origins, key=lambda item: item.origin_id),
            "items": sorted(items, key=lambda item: item.evidence_id),
            "denominators": sorted(
                denominators, key=lambda item: item.member_id
            ),
            "population_refs": sorted(
                population_refs, key=lambda item: item.stable_id
            ),
            "falsification_refs": [
                _ref("falsification", "FALSIFY-child", "child")
            ],
            "scope_condition_refs": [
                _ref("falsification", "FALSIFY-child-scope", "child-scope")
            ],
            "dependencies": [
                CapabilityDependency(
                    capability="current_population",
                    requirement_mode="required",
                    status="complete",
                    reason_codes=[],
                )
            ],
            "integrity": "valid",
            "status": "complete",
            "incomplete_reasons": [],
        }
        payload["artifact_digest"] = artifact_digest(payload)
        bundle = CurrentEvidenceBundle.model_validate(payload)
        return CurrentEvidenceReacquisition(
            current_evidence=bundle,
            substrate_kind="complete_population",
            complete=True,
        )


def _base() -> tuple[
    ChannelAssessment, CandidateCaseMember, CandidateCase, CurrentEvidenceBundle
]:
    assessment = _assessment()
    member = _member(assessment)
    case = _case(member)
    return assessment, member, case, _current_bundle(case)


def test_narrowing_filters_complete_population_rebinds_and_reverifies() -> None:
    assessment, member, case, evidence = _base()
    core_evidence = next(
        item
        for item in evidence.items
        if next(
            origin for origin in evidence.origins if origin.origin_id == item.origin_id
        ).path.startswith("src/core/")
    )
    proposal = _proposal(
        member.member_id,
        core_evidence.evidence_id,
        _semantics(scope=("src/core/**",)),
        ("scope",),
    )
    git = _Git()
    verifier = _Verifier()

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=git,
        verifier=verifier,
    )

    assert result.status == "complete"
    assert result.reason_codes == ()
    assert len(result.child_assessments) == 1
    child_assessment = result.child_assessments[0]
    assert child_assessment.parent_candidate_id == assessment.assessment_id
    assert child_assessment.candidate_version == 1
    assert child_assessment.semantics.normalized_action == (
        assessment.semantics.normalized_action
    )
    assert child_assessment.semantics.polarity == assessment.semantics.polarity
    child_evidence = result.current_evidence_bundles[0]
    assert child_evidence.status == "complete"
    assert [origin.path for origin in child_evidence.origins] == [
        "src/core/plugin.py"
    ]
    assert len(child_evidence.items) == 1
    assert child_evidence.items[0].member_ids == [
        result.child_members[0].member_id
    ]
    denominator = child_evidence.denominators[0]
    assert denominator.scanned == 2
    assert denominator.serialized == 1
    assert denominator.support == 1
    assert denominator.counterexample == 0
    assert any(
        item.capability == "refinement_current_filtered_complete_reuse"
        and item.status == "complete"
        for item in child_evidence.dependencies
    )
    assert result.lineage[0].current_evidence_mode == "filtered_complete_reuse"
    assert result.lineage[0].requires_git_reroute is True
    assert result.refinement_records[0].status == "complete"
    assert git.calls == [result.child_cases[0].case_id]
    assert verifier.rounds == [1]


def test_narrowing_sampled_population_cannot_claim_complete_filtered_reuse() -> None:
    assessment, member, case, _ = _base()
    evidence = _current_bundle(case, uninspected=3)
    core_evidence = next(
        item
        for item in evidence.items
        if next(
            origin
            for origin in evidence.origins
            if origin.origin_id == item.origin_id
        ).path.startswith("src/core/")
    )
    proposal = _proposal(
        member.member_id,
        core_evidence.evidence_id,
        _semantics(scope=("src/core/**",)),
        ("scope",),
    )
    git = _Git()
    verifier = _Verifier()

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=git,
        verifier=verifier,
    )

    assert result.status == "review"
    assert "refinement_current_denominator_incomplete" in result.reason_codes
    assert result.current_evidence_bundles[0].status == "failed"
    assert verifier.rounds == []
    assert git.calls == []


def test_scope_expansion_without_complete_substrate_is_typed_review() -> None:
    assessment, member, case, evidence = _base()
    proposal = _proposal(
        member.member_id,
        evidence.items[0].evidence_id,
        _semantics(scope=("src/**", "tests/**")),
        ("scope",),
    )
    git = _Git()
    verifier = _Verifier()

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        current_reacquirer=None,
        git_reacquirer=git,
        verifier=verifier,
    )

    assert result.status == "review"
    assert "refinement_complete_substrate_reacquirer_missing" in result.reason_codes
    assert result.current_evidence_bundles[0].status == "failed"
    assert result.findings[0].status == "failed"
    assert result.refinement_records[0].status == "review"
    assert result.lineage[0].current_evidence_mode == "failed"
    assert git.calls == []
    assert verifier.rounds == []


@pytest.mark.parametrize(
    ("semantics", "reason"),
    [
        (
            _semantics(action="register-plugins-directly"),
            "refinement_changes_core_action",
        ),
        (
            _semantics(polarity="negative"),
            "refinement_reverses_polarity",
        ),
    ],
)
def test_core_or_polarity_change_is_rejected_before_artifact_creation(
    semantics: CandidateSemantics, reason: str
) -> None:
    assessment, member, case, evidence = _base()
    proposal = _proposal(
        member.member_id,
        evidence.items[0].evidence_id,
        semantics,
        ("wording",),
    )

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )

    assert result.status == "review"
    assert result.reason_codes == (reason,)
    assert result.child_assessments == ()
    assert result.refinement_records == ()


def test_foreign_proposal_evidence_is_rejected() -> None:
    assessment, member, case, evidence = _base()
    proposal = _proposal(
        member.member_id,
        "EVIDENCE-foreign",
        _semantics(scope=("src/core/**",)),
        ("scope",),
    )

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )

    assert result.status == "review"
    assert result.reason_codes == ("refinement_foreign_evidence",)
    assert result.child_cases == ()


def test_split_allows_two_unique_children_with_full_reacquisition_but_not_three() -> None:
    assessment, member, case, evidence = _base()
    proposal_values = [
        _proposal(
            member.member_id,
            evidence.items[0].evidence_id,
            _semantics(scope=(scope,)),
            ("scope", "split"),
        )
        for scope in ("src/core/**", "src/plugins/**")
    ]
    current = _SplitCurrent()

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=proposal_values,
        max_children=8,
        current_reacquirer=current,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )

    assert result.status == "complete"
    assert current.calls == 1
    assert len(result.child_assessments) == 2
    assert len({item.candidate_root_id for item in result.child_assessments}) == 2
    assert len({item.member_id for item in result.child_members}) == 2
    assert len(result.child_cases) == 1
    assert len(result.child_cases[0].relations) == 1
    assert result.child_cases[0].relations[0].relation == "uncertain"
    assert result.lineage[0].current_evidence_mode == "full_reacquisition"
    assert len(result.refinement_records[0].child_member_ids) == 2

    third = _proposal(
        member.member_id,
        evidence.items[0].evidence_id,
        _semantics(scope=("src/third/**",)),
        ("scope", "split"),
    )
    rejected = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[*proposal_values, third],
        max_children=8,
        current_reacquirer=current,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )
    assert rejected.status == "review"
    assert rejected.reason_codes == ("refinement_child_limit_exceeded",)
    assert rejected.child_assessments == ()


def test_round_one_second_proposal_is_not_executed_and_forces_review() -> None:
    assessment, member, case, evidence = _base()
    proposal = _proposal(
        member.member_id,
        evidence.items[0].evidence_id,
        _semantics(scope=("src/core/**",)),
        ("scope",),
    )
    verifier = _Verifier(second_proposal=True)

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=_Git(),
        verifier=verifier,
    )

    assert verifier.rounds == [1]
    assert result.status == "review"
    assert result.reason_codes == ("round_one_finding_invalid",)
    assert result.findings[0].refinement_proposals == []
    assert result.refinement_records[0].status == "review"


def test_one_round_cannot_refine_multiple_parent_members() -> None:
    first = _assessment(seed="first")
    second = _assessment(seed="second")
    members = sorted([_member(first), _member(second)], key=lambda item: item.member_id)
    case_payload: dict[str, object] = {
        "case_id": make_case_id(SNAPSHOT, [item.member_id for item in members]),
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "members": members,
        "relations": [
            {
                "left_member_id": members[0].member_id,
                "right_member_id": members[1].member_id,
                "relation": "related_variant",
                "source": "semantic_grouping",
                "reason_code": "same_test_case",
            }
        ],
        "grouping_status": "complete",
        "grouping_reason_codes": [],
        "exact_identity_groups": [],
        "required_capabilities": ["current_evidence"],
    }
    case_payload["artifact_digest"] = artifact_digest(case_payload)
    case = CandidateCase.model_validate(case_payload)
    evidence = _current_bundle(case)
    proposals = [
        _proposal(
            member.member_id,
            evidence.items[0].evidence_id,
            member.semantics.model_copy(update={"applies_to": ["src/core/**"]}),
            ("scope",),
        )
        for member in members
    ]

    result = execute_refinement_round(
        case=case,
        assessments={item.assessment_id: item for item in (first, second)},
        current_evidence=evidence,
        proposals=proposals,
        max_children=2,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )

    assert result.status == "review"
    assert result.reason_codes == ("refinement_multiple_parents_forbidden",)
    assert result.child_cases == ()


def test_upstream_channel_refined_leaf_can_use_the_unified_round() -> None:
    assessment = _assessment(
        seed="channel-refined",
        candidate_version=1,
    )
    member = _member(assessment)
    case = _case(member)
    evidence = _current_bundle(case)
    proposal = _proposal(
        member.member_id,
        evidence.items[0].evidence_id,
        assessment.semantics.model_copy(
            update={"applies_to": ["src/core/**"]}
        ),
        ("scope",),
    )

    result = execute_refinement_round(
        case=case,
        assessments={assessment.assessment_id: assessment},
        current_evidence=evidence,
        proposals=[proposal],
        max_children=2,
        git_reacquirer=_Git(),
        verifier=_Verifier(),
    )

    assert result.status == "complete"
    assert result.child_assessments[0].candidate_version == 2
    assert "unified_refinement_child" in result.child_assessments[0].typed_reasons
