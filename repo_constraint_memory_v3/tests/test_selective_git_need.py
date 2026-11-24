"""Affirmative historical-use admission and bounded raw reuse regressions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.explicit_verification import (
    ExplicitVerificationJudgment,
    ExplicitVerificationResponse,
    plan_explicit_verification,
    validate_explicit_verification_response,
)
from repo_constraint_memory_v3.temporal_need import (
    GitNeedAssessment,
    GitNeedJudgment,
    aggregate_git_needs,
    unassessed_git_need,
    validate_git_need,
)
from repo_constraint_memory_v3.unified import pipeline
from repo_constraint_memory_v3.unified.git_evidence import (
    GitProcessOutcome,
    RunLocalGitRunner,
)
from test_explicit_verification import _candidate, _judgment


def _yes(ref: str = "current-ref", path: str = "README.md", axis=None):
    return GitNeedJudgment.model_validate({
        "status": "affirmative",
        "questions": [{
            "question": "Was the recorded A-to-B replacement later reverted?",
            "current_gap": "The supplied migration note identifies both branches but leaves the reversal relationship open.",
            "answer_use": "Resolve which documented replacement relationship applies to this candidate at the snapshot.",
            "path": path,
            "evidence_refs": [ref],
            "blocking_axis": axis,
        }],
    })


@pytest.mark.parametrize("status", ["no_established_need", "undetermined"])
def test_no_need_does_not_fabricate_questions(status):
    judgment = GitNeedJudgment(status=status, questions=[])
    assert aggregate_git_needs([judgment]).status == status
    with pytest.raises(ValidationError):
        GitNeedJudgment(status=status, questions=_yes().questions)


def test_missing_need_and_missing_affirmative_question_are_invalid():
    payload = _judgment(_candidate("tests")).model_dump(mode="json")
    payload.pop("git_need")
    with pytest.raises(ValidationError):
        ExplicitVerificationJudgment.model_validate(payload)
    with pytest.raises(ValidationError):
        GitNeedJudgment(status="affirmative", questions=[])
    with pytest.raises(ValidationError):
        GitNeedAssessment(status="not_assessed", judgments=[], reason=None)


def test_need_aggregation_is_exact_stable_and_preserves_undetermined():
    yes = _yes()
    unknown = GitNeedJudgment(status="undetermined", questions=[])
    forward = aggregate_git_needs([yes, unknown, yes])
    assert forward == aggregate_git_needs([unknown, yes])
    assert forward.status == "affirmative"
    assert len(forward.judgments) == 2


@pytest.mark.parametrize("mutation", ["reference", "path", "axis", "path_ref_binding"])
def test_host_rejects_unbound_need_without_semantic_keyword_heuristics(mutation):
    value = _yes(axis="scope_support")
    refs = {"current-ref"}
    paths = {"README.md", "OTHER.md"}
    axes = {"scope_support"}
    mapping = {"current-ref": {"README.md"}}
    if mutation == "reference":
        refs = {"foreign"}
    elif mutation == "path":
        paths = {"OTHER.md"}
    elif mutation == "axis":
        axes = set()
    else:
        mapping = {"current-ref": {"OTHER.md"}}
    with pytest.raises(ValueError):
        validate_git_need(value, allowed_refs=refs, allowed_paths=paths,
                          unresolved_axes=axes, reference_paths=mapping)


def test_explicit_need_replays_against_exact_candidate_evidence():
    candidate = _candidate("tests")
    plan = plan_explicit_verification([candidate])
    evidence = candidate.evidence[0]
    judgment = _judgment(candidate).model_copy(update={
        "git_need": _yes(evidence.evidence_digest, evidence.evidence.path),
    })
    response = ExplicitVerificationResponse(results=[judgment])
    assert validate_explicit_verification_response(response, plan.batches[0]) == response
    modified = judgment.model_copy(update={"git_need": _yes("foreign", evidence.evidence.path)})
    with pytest.raises(ValueError, match="unsupplied"):
        validate_explicit_verification_response(
            ExplicitVerificationResponse(results=[modified]), plan.batches[0]
        )


@pytest.mark.parametrize("status", ["no_established_need", "undetermined", "not_assessed", "invalid"])
def test_active_current_conflict_and_keywords_do_not_establish_need(monkeypatch, status):
    need = (
        aggregate_git_needs([GitNeedJudgment(status=status, questions=[])])
        if status in {"no_established_need", "undetermined"}
        else unassessed_git_need("host_test", invalid=status == "invalid")
    )
    case = SimpleNamespace(
        grouping_status="complete", members=[SimpleNamespace(assessment_id="a")],
        relations=[SimpleNamespace(relation="uncertain")],
    )
    assessment = SimpleNamespace(
        channel="implicit_current", integrity="valid", normalized_status="locally_supported",
        git_need=need,
    )
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **kw: False)
    assert pipeline._route_case(case, {"a": assessment}, SimpleNamespace()) == "not_required"


@pytest.mark.parametrize("axis, expected", [(None, "augmentative"), ("scope_support", "ambiguity_resolving")])
def test_only_affirmative_routes_and_requiredness_is_explicit(monkeypatch, axis, expected):
    case = SimpleNamespace(grouping_status="complete", members=[SimpleNamespace(assessment_id="a")])
    assessment = SimpleNamespace(integrity="valid", normalized_status="locally_supported", git_need=aggregate_git_needs([_yes(axis=axis)]))
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **kw: False)
    assert pipeline._route_case(case, {"a": assessment}, SimpleNamespace()) == expected


def test_no_match_does_not_resolve_a_required_historical_question():
    bundle = SimpleNamespace(status="bounded_no_match")
    required = pipeline._dependency_for_temporal("ambiguity_resolving", bundle, None)
    optional = pipeline._dependency_for_temporal("augmentative", bundle, None)
    assert required.requirement_mode == "required"
    assert required.status == "unavailable"
    assert required.reason_codes == ["required_git_question_unanswered_no_match"]
    assert optional.requirement_mode == "not_required"
    assert optional.status == "not_required"


class _RawRunner:
    executable = "/usr/bin/git"

    def __init__(self):
        self.calls = 0
        self.failed = False

    def run(self, repository_root, argv, *, timeout_seconds, max_output_bytes):
        self.calls += 1
        return GitProcessOutcome(tuple(argv), 1 if self.failed else 0, b"raw-object", b"", False, False, False, 17)


def test_run_local_git_reuses_only_success_with_identical_boundaries(tmp_path):
    raw = _RawRunner()
    runner = RunLocalGitRunner(raw, snapshot_digest="snapshot-a")
    args = ["show", "immutable-object"]
    first = runner.run(tmp_path, args, timeout_seconds=10, max_output_bytes=100)
    second = runner.run(tmp_path, args, timeout_seconds=10, max_output_bytes=100)
    assert second == replace(first, duration_ms=0, cache_hit=True)
    assert runner.counters() == {"logical_calls": 2, "physical_calls": 1, "cache_hits": 1}
    runner.run(tmp_path, args, timeout_seconds=10, max_output_bytes=101)
    runner.run(tmp_path, args, timeout_seconds=11, max_output_bytes=100)
    assert raw.calls == 3
    raw.failed = True
    for _ in range(2):
        runner.run(tmp_path, ["show", "missing-object"], timeout_seconds=10, max_output_bytes=100)
    assert raw.calls == 5
    independent = RunLocalGitRunner(raw, snapshot_digest="snapshot-b")
    independent.run(tmp_path, args, timeout_seconds=10, max_output_bytes=100)
    assert raw.calls == 6


def test_case_input_index_reuses_locations_without_changing_complete_dto(monkeypatch):
    from repo_constraint_memory_v3.unified import inputs
    from repo_constraint_memory_v3.unified.grouping import (
        assemble_candidate_cases, plan_candidate_grouping,
    )
    from helpers import config_for_tests
    from test_unified_upstream import _decision, _verification, SNAPSHOT, PROFILE

    decision = _decision()
    assessments = inputs.build_explicit_assessments(
        decisions=[decision], verification=_verification(decision),
        snapshot_digest=SNAPSHOT, profile_digest=PROFILE,
    )
    members = pipeline._case_members(assessments)
    plan = plan_candidate_grouping(members, mode="singleton", anchors_by_member={}, limits=pipeline._grouping_limits(config_for_tests().unified))
    case = assemble_candidate_cases(plan, [], profile_digest=PROFILE).cases[0]
    arguments = dict(assessments=assessments, explicit_decisions=[decision],
                     observations=None, current_source=None)
    baseline = inputs.build_case_evidence_inputs(
        case=case, explicit_discovery_complete=True, **arguments,
    )
    original = inputs._observation_locations
    calls = []

    def locations(observations, current_source):
        calls.append(1)
        return original(observations, current_source)

    monkeypatch.setattr(inputs, "_observation_locations", locations)
    index = inputs.build_case_input_index(**arguments)
    inputs.build_blocking_anchors(members=members, input_index=index, **arguments)
    for _ in range(2):
        assert inputs.build_case_evidence_inputs(
            case=case, explicit_discovery_complete=True, input_index=index, **arguments,
        ) == baseline
    assert calls == [1]


@pytest.mark.parametrize("axis_value", ["pass", "fail"])
def test_required_need_cannot_bind_a_resolved_explicit_fact(axis_value):
    candidate = _candidate("tests")
    plan = plan_explicit_verification([candidate])
    evidence = candidate.evidence[0]
    judgment = _judgment(candidate, evidence_entailment=axis_value).model_copy(update={
        "git_need": _yes(evidence.evidence_digest, evidence.evidence.path, "evidence_entailment"),
    })
    with pytest.raises(ValueError, match="unresolved fact axis"):
        validate_explicit_verification_response(ExplicitVerificationResponse(results=[judgment]), plan.batches[0])


def test_legal_required_need_binds_an_uncertain_explicit_fact():
    candidate = _candidate("tests")
    plan = plan_explicit_verification([candidate])
    evidence = candidate.evidence[0]
    judgment = _judgment(candidate, evidence_entailment="uncertain").model_copy(update={
        "git_need": _yes(evidence.evidence_digest, evidence.evidence.path, "evidence_entailment"),
    })
    response = ExplicitVerificationResponse(results=[judgment])
    assert validate_explicit_verification_response(response, plan.batches[0]) == response


def test_need_aggregation_respects_existing_configurable_batch_ceiling():
    # The ordinary default is 16 calls, but the existing profile permits more.
    values = [_yes(ref=f"current-{index}") for index in range(17)]
    result = aggregate_git_needs([*values, values[0]])
    assert result.status == "affirmative"
    assert len(result.judgments) == 17
    with pytest.raises(ValueError, match="host-only state"):
        aggregate_git_needs([])


@pytest.mark.parametrize("status", ["no_established_need", "undetermined"])
@pytest.mark.parametrize("channel", ["explicit", "implicit_current"])
def test_no_affirmative_skips_all_git_work_and_keeps_normal_fact_flow(monkeypatch, tmp_path, status, channel):
    from test_unified_pipeline import _assessment, _run, _PipelineClient
    need = aggregate_git_needs([GitNeedJudgment(status=status, questions=[])])
    assessment = _assessment(
        "no-history", channel=channel, rule="Use the current replacement for the deprecated interface.",
        git_need=need,
    )
    client = _PipelineClient()
    result = _run(monkeypatch, tmp_path, [assessment], client=client,
                  snapshot_commit="e" * 40, fail_on_git_preflight=True)
    assert result.git_bundles == ()
    assert all(route.startswith("not_required") for route in result.case_routes.values())
    assert [request["stage"] for request in client.requests] == ([] if channel == "explicit" else ["unified_case_verify"])
    assert result.bundle.canonical_claims[0].epistemic_status == "accepted"
    counts = result.bundle.funnel.git_case_counts
    assert counts["cases"] == counts["result_not_required"] == 1
    assert counts["raw_logical_calls"] == counts["queried_cases"] == 0


def test_optional_and_required_disabled_queries_have_distinct_completeness(monkeypatch, tmp_path):
    from test_unified_pipeline import _assessment, _run, _PipelineClient, _affirmative_need
    for required in (False, True):
        with monkeypatch.context() as patch:
            assessment = _assessment(f"required-{required}", channel="implicit_current", rule="Register plugins through PluginManager.", git_need=_affirmative_need(required=required))
            result = _run(patch, tmp_path, [assessment], client=_PipelineClient(), unified_updates={"git_enabled": False}, fail_on_git_preflight=True)
        counts = result.bundle.funnel.git_case_counts
        assert counts["cases"] == counts["affirmative_cases"] == counts["result_unavailable"] == 1
        assert counts["required_cases"] == int(required)
        assert counts["optional_cases"] == int(not required)
        assert result.bundle.canonical_claims[0].epistemic_status == ("review" if required else "accepted")


def _child_need_pipeline(monkeypatch, tmp_path):
    from test_unified_pipeline import _assessment, _run, _PipelineClient

    class ChildNeedClient(_PipelineClient):
        def _response(self, request):
            response = super()._response(request)
            proposal = response.get("refinement_proposal")
            if proposal is not None:
                item = next(item for item in request["current_evidence"]["items"] if item["evidence_id"] == proposal["evidence_refs"][0])
                origin = next(origin for origin in request["current_evidence"]["origins"] if origin["origin_id"] == item["origin_id"])
                proposal["git_need"] = _yes(item["evidence_id"], origin["path"]).model_dump(mode="json")
            return response

    assessment = _assessment("child-need", channel="implicit_current", rule="Register plugins through PluginManager.")
    return _run(monkeypatch, tmp_path, [assessment], client=ChildNeedClient(propose_narrowing=True), unified_updates={"git_enabled": False}, fail_on_git_preflight=True)


def test_child_need_is_independent_and_rebound_to_exact_child_evidence(monkeypatch, tmp_path):
    from repo_constraint_memory_v3.unified.validation import _refinement_closure
    result = _child_need_pipeline(monkeypatch, tmp_path)
    parent = next(item for item in result.bundle.assessments if item.candidate_version == 0)
    child = next(item for item in result.bundle.assessments if item.candidate_version == 1)
    assert parent.git_need.status == "not_assessed"
    assert child.git_need.status == "affirmative"
    proposal = next(item for finding in result.bundle.findings for item in finding.refinement_proposals)
    child_case = next(case for case in result.bundle.cases if any(member.assessment_id == child.assessment_id for member in case.members))
    child_current = next(item for item in result.bundle.current_evidence if item.case_id == child_case.case_id)
    question = child.git_need.judgments[0].questions[0]
    assert question.evidence_refs != proposal.git_need.questions[0].evidence_refs
    assert set(question.evidence_refs) <= {item.evidence_id for item in child_current.items}
    assert question.question == proposal.git_need.questions[0].question
    errors = []
    _refinement_closure(result.bundle, {}, errors)
    assert errors == []


def test_offline_refinement_replay_rejects_resealed_child_git_need(monkeypatch, tmp_path):
    from repo_constraint_memory_v3.unified.models import ChannelAssessment
    from repo_constraint_memory_v3.unified.ids import seal_payload
    from repo_constraint_memory_v3.unified.validation import _refinement_closure
    result = _child_need_pipeline(monkeypatch, tmp_path)
    child = next(item for item in result.bundle.assessments if item.candidate_version == 1)
    payload = child.model_dump(mode="json", exclude={"artifact_digest"})
    payload["git_need"] = aggregate_git_needs([GitNeedJudgment(status="no_established_need", questions=[])]).model_dump(mode="json")
    altered = ChannelAssessment.model_validate(seal_payload(payload))
    bundle = result.bundle.model_copy(update={"assessments": [altered if item.assessment_id == child.assessment_id else item for item in result.bundle.assessments]})
    errors = []
    _refinement_closure(bundle, {}, errors)
    assert any("Git need differs from rebound proposal" in error for error in errors)



def test_filtered_child_need_is_typed_invalid_without_faking_current_incompleteness():
    from test_unified_refinement import _base, _proposal, _semantics, _Git, _Verifier
    from repo_constraint_memory_v3.unified.refinement import execute_refinement_round
    assessment, member, case, current = _base()
    paths = {origin.origin_id: origin.path for origin in current.origins}
    kept = next(item for item in current.items if paths[item.origin_id].startswith("src/core/"))
    removed = next(item for item in current.items if paths[item.origin_id].startswith("src/legacy/"))
    proposal = _proposal(member.member_id, kept.evidence_id, _semantics(scope=("src/core/**",)), ("scope",)).model_copy(update={
        "git_need": _yes(removed.evidence_id, paths[removed.origin_id]),
    })
    git, verifier = _Git(), _Verifier()
    result = execute_refinement_round(case=case, assessments={assessment.assessment_id: assessment},
        current_evidence=current, proposals=[proposal], max_children=2,
        git_reacquirer=git, verifier=verifier)
    assert result.status == "review"
    child = result.child_assessments[0]
    assert child.git_need.status == "invalid"
    evidence = result.current_evidence_bundles[0]
    assert evidence.status == "complete"
    assert result.lineage[0].current_evidence_mode == "filtered_complete_reuse"
    dependency = next(item for item in evidence.dependencies if item.capability == "refinement_git_need_binding")
    assert (dependency.requirement_mode, dependency.status) == ("required", "failed")
    assert pipeline._route_case(result.child_cases[0], {child.assessment_id: child}, evidence) == "not_required_current_incomplete"
    assert git.calls == verifier.rounds == []


def test_child_no_need_does_not_inherit_parent_affirmative(monkeypatch, tmp_path):
    from test_unified_pipeline import _assessment, _run, _PipelineClient, _affirmative_need
    parent = _assessment("parent-need", channel="implicit_current", rule="Register plugins through PluginManager.", git_need=_affirmative_need())
    result = _run(monkeypatch, tmp_path, [parent], client=_PipelineClient(propose_narrowing=True), unified_updates={"git_enabled": False}, fail_on_git_preflight=True)
    child = next(item for item in result.bundle.assessments if item.candidate_version == 1)
    assert child.git_need.status == "no_established_need"
    assert list(result.case_routes.values()).count("augmentative") == 1
    assert list(result.case_routes.values()).count("not_required") == 1
