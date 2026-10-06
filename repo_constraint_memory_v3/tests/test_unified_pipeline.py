from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from helpers import config_for_tests

from repo_constraint_memory_v3.inference import InferenceBudget
from repo_constraint_memory_v3.llm import LLMProviderError
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.temporal_need import GitNeedAssessment
from repo_constraint_memory_v3.unified import pipeline
from repo_constraint_memory_v3.unified.current_evidence import (
    EvidenceContribution,
    PopulationEvidenceContribution,
)
from repo_constraint_memory_v3.unified.git_evidence import (
    GitChangeSlice,
    GitQueryManifest,
    GitPreflight,
    GitRawExecution,
    GitTemporalBundle,
)
from repo_constraint_memory_v3.unified.grouping import (
    MemberBlockingAnchors,
    plan_grouping_batches,
)
from repo_constraint_memory_v3.unified.ids import (
    content_digest,
    make_assessment_id,
    make_case_id,
    seal_payload,
    stable_id,
)
from repo_constraint_memory_v3.unified.inputs import (
    CaseEvidenceInputs,
    ChannelAssessmentBuildResult,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CapabilityDependency,
    ChannelAssessment,
    ImmutableArtifactRef,
    PopulationDenominator,
    UnifiedFunnel,
    UNIFIED_CONTRACT_VERSION,
)
from repo_constraint_memory_v3.unified.policy import (
    CANONICAL_IDENTITY_COLLISION_REASON,
)
from repo_constraint_memory_v3.unified.semantic import (
    _candidate_group_request_ordinal,
)

SNAPSHOT = "sha256:" + "a" * 64
PROFILE = "b" * 64
PROTOCOL_MANIFEST = "c" * 64


def _ref(
    kind: str,
    stable_id_value: str,
    *,
    seed: str,
) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id_value,
        snapshot_digest=SNAPSHOT,
        artifact_path=f"channel/{seed}.json",
        artifact_digest=content_digest([kind, stable_id_value, seed]),
    )


def _affirmative_need(*, required: bool = False) -> GitNeedAssessment:
    return GitNeedAssessment.model_validate({
        "status": "affirmative", "reason": None,
        "judgments": [{"status": "affirmative", "questions": [{
            "question": "Was the documented replacement reverted?",
            "current_gap": "Current supplied evidence leaves the replacement relationship unresolved.",
            "answer_use": "Resolve the replacement relationship used by the currentness finding.",
            "path": "src/evidence_1.py", "evidence_refs": ["fixture-evidence"],
            "blocking_axis": "scope_condition_support" if required else None,
        }]}],
    })


def _assessment(
    seed: str,
    *,
    channel: str,
    rule: str,
    profile_digest: str = PROFILE,
    source_authority: str | None = None,
    normalized_status: str = "locally_supported",
    git_need: GitNeedAssessment | None = None,
) -> ChannelAssessment:
    root_id = f"ROOT-{seed}"
    assessment_id = make_assessment_id(
        SNAPSHOT, profile_digest, channel, root_id, 0
    )
    support = _ref(
        "explicit_evidence" if channel == "explicit" else "population_member",
        f"SUPPORT-{seed}",
        seed=f"support-{seed}",
    )
    axes = {
        "host_grounding": "pass",
        "coding_relevance": "pass",
        "evidence_entailment": "pass",
        "scope_condition_support": "pass",
        "ambiguity": "pass",
    }
    payload: dict[str, object] = {
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "assessment_id": assessment_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": profile_digest,
        "channel": channel,
        "candidate_root_id": root_id,
        "candidate_version": 0,
        "parent_candidate_id": None,
        "semantics": {
            "rule": rule,
            "normalized_action": " ".join(rule.casefold().split()),
            "polarity": "positive",
            "applies_to": ["src/**"],
            "condition": None,
            "exception": None,
        },
        "proposed_claim_kind": (
            "normative_declaration"
            if channel == "explicit"
            else "observed_convention"
        ),
        "proposed_strength": "required" if channel == "explicit" else "observed",
        "raw_channel_status": "channel_supported",
        "raw_policy_version": "channel-policy-test-1",
        "normalized_status": normalized_status,
        "axis_findings": axes,
        "typed_reasons": [],
        "support_refs": [support],
        "counterevidence_refs": [],
        "coverage_refs": [],
        "lineage_refs": [],
        "extension_refs": [],
        "integrity": "valid",
        "source_authority": source_authority
        or ("direct_normative" if channel == "explicit" else "current_population"),
        "git_need": git_need or {"status": "not_assessed", "judgments": [], "reason": "host_fixture"},
        "provenance": [f"test:{seed}"],
    }
    payload["artifact_digest"] = pipeline.artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def _hard_invalid_assessment(seed: str) -> ChannelAssessment:
    assessment = _assessment(
        seed,
        channel="explicit",
        rule="Run the invalid command.",
    )
    payload = assessment.model_dump(mode="python", exclude={"artifact_digest"})
    payload.update(
        {
            "normalized_status": "invalid",
            "integrity": "invalid",
            "typed_reasons": ["channel_artifact_invalid"],
        }
    )
    payload["artifact_digest"] = pipeline.artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def _case_evidence_inputs(
    case: CandidateCase,
    assessments: Mapping[str, ChannelAssessment],
) -> CaseEvidenceInputs:
    explicit: list[EvidenceContribution] = []
    population: list[PopulationEvidenceContribution] = []
    denominators: list[PopulationDenominator] = []
    population_refs: list[ImmutableArtifactRef] = []
    falsification_refs: list[ImmutableArtifactRef] = []
    scope_refs: list[ImmutableArtifactRef] = []
    current_members = 0
    for index, member in enumerate(case.members, start=1):
        assessment = assessments[member.assessment_id]
        source_digest = content_digest(["source", member.member_id])
        excerpt_digest = content_digest(["excerpt", member.member_id])
        path = f"src/evidence_{index}.py"
        if assessment.channel == "explicit":
            reference = assessment.support_refs[0]
            explicit.append(
                EvidenceContribution(
                    origin_kind="explicit_span",
                    path=path,
                    line_start=index,
                    line_end=index,
                    source_content_digest=source_digest,
                    independence_group=f"explicit:{index}",
                    role="support",
                    member_ids=(member.member_id,),
                    artifact_ref=reference,
                    excerpt_digest=excerpt_digest,
                    description=assessment.semantics.rule,
                    anchor_literals=(assessment.semantics.rule,),
                    anchor_symbols=(),
                )
            )
            scope_refs.append(reference)
            continue

        current_members += 1
        population_id = f"POPULATION-{member.member_id}"
        population_member_id = assessment.support_refs[0].stable_id
        population_ref = _ref(
            "population", population_id, seed=f"population-{index}"
        )
        falsification_ref = _ref(
            "falsification",
            f"FALSIFICATION-{member.member_id}",
            seed=f"falsification-{index}",
        )
        population_refs.append(population_ref)
        falsification_refs.append(falsification_ref)
        scope_refs.append(falsification_ref)
        evidence = EvidenceContribution(
            origin_kind="population_member",
            path=path,
            line_start=index,
            line_end=index,
            source_content_digest=source_digest,
            independence_group=f"current:{index}",
            role="support",
            member_ids=(member.member_id,),
            artifact_ref=assessment.support_refs[0],
            excerpt_digest=excerpt_digest,
            description=assessment.semantics.rule,
            anchor_literals=(),
            anchor_symbols=("PluginManager.register",),
        )
        population.append(
            PopulationEvidenceContribution(
                case_member_id=member.member_id,
                population_id=population_id,
                population_member_id=population_member_id,
                label="support",
                evidence=evidence,
            )
        )
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
    dependencies = (
        (
            CapabilityDependency(
                capability="current_falsification",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            ),
            CapabilityDependency(
                capability="explicit_discovery_for_current_conflicts",
                requirement_mode="augmentative",
                status="complete",
                reason_codes=[],
            ),
        )
        if current_members
        else ()
    )
    return CaseEvidenceInputs(
        explicit_evidence=tuple(explicit),
        population_evidence=tuple(population),
        denominators=tuple(denominators),
        population_refs=tuple(sorted(population_refs, key=lambda item: item.stable_id)),
        falsification_refs=tuple(
            sorted(falsification_refs, key=lambda item: item.stable_id)
        ),
        scope_condition_refs=tuple(
            sorted(scope_refs, key=lambda item: item.stable_id)
        ),
        dependencies=dependencies,
    )


class _NoCallClient:
    def complete_with_settings(self, *args, **kwargs):
        raise AssertionError("the fast/empty pipeline must not invoke a provider")

    def complete(self, *args, **kwargs):
        raise AssertionError("the fast/empty pipeline must not invoke a provider")


def test_required_gap_directory_marker_publishes_review_without_verifier(
    monkeypatch, tmp_path: Path,
) -> None:
    from dataclasses import replace

    assessment = _assessment(
        "directory-marker-required-gap", channel="implicit_current",
        rule="Preserve the fixture convention.",
    )
    payload = assessment.model_dump(mode="python", exclude={"artifact_digest"})
    payload["semantics"]["applies_to"] = ["src/"]
    assessment = ChannelAssessment.model_validate(seal_payload(payload))
    original_inputs = _case_evidence_inputs

    def incomplete_inputs(case, assessment_by_id):
        inputs = original_inputs(case, assessment_by_id)
        return replace(inputs, dependencies=tuple(
            dependency.model_copy(update={
                "status": "partial", "reason_codes": ["saved_falsification_gap"],
            }) if dependency.capability == "current_falsification" else dependency
            for dependency in inputs.dependencies
        ))

    monkeypatch.setitem(globals(), "_case_evidence_inputs", incomplete_inputs)
    result = _run(
        monkeypatch, tmp_path, [assessment], client=_NoCallClient(),
        unified_updates={"git_enabled": False}, fail_on_git_preflight=True,
    )
    assert result.inference_records == ()
    assert result.bundle.findings[0].status == "not_required"
    assert result.bundle.findings[0].inference_ref is None
    claim = result.bundle.canonical_claims[0]
    assert claim.epistemic_status == "review"
    assert claim.dependency_status == "incomplete"
    assert claim.semantics.applies_to == ["src/"]
    assert claim.reason_codes == [
        "directory_marker_scope_requires_refinement", "required_capability_incomplete",
    ]
    assert claim.review_facets == [
        "refinement_or_qualifier_needed", "technical_or_coverage_gap",
    ]

class _PipelineClient:
    def __init__(
        self,
        *,
        fail_grouping_provider: bool = False,
        propose_narrowing: bool = False,
        repeat_refinement_round_one: bool = False,
        invalid_refinement: str | None = None,
    ) -> None:
        self.fail_grouping_provider = fail_grouping_provider
        self.propose_narrowing = propose_narrowing
        self.repeat_refinement_round_one = repeat_refinement_round_one
        self.invalid_refinement = invalid_refinement
        self.requests: list[dict[str, object]] = []

    def _response(self, request: dict[str, object]) -> dict[str, object]:
        stage = request["stage"]
        if stage == "candidate_semantic_group":
            if self.fail_grouping_provider:
                raise LLMProviderError("grouping provider failed", retryable=False)
            return {
                "relations": [
                    {
                        "left_member_id": pair[0],
                        "right_member_id": pair[1],
                        "relation": "conflicting_variant",
                    }
                    for pair in request["required_pairs"]
                ]
            }
        if stage == "temporal_evidence_classify":
            spans = {
                item["item_id"]: item["evidence_span_ref"]
                for item in request["items"]
            }
            return {
                "relations": [
                    {
                        "item_id": cell[0],
                        "member_id": cell[1],
                        "relation": "supports_as_written",
                        "affected_facets": ["rule"],
                        "evidence_span": spans[cell[0]],
                        "reason_code": "temporal_patch_supports",
                        "uncertainty_reason": None,
                    }
                    for cell in request["required_cells"]
                ]
            }
        assert stage == "unified_case_verify"
        member_ids = request["required_member_ids"]
        qualifier_by_member = {
            item["member_id"]: bool(
                item["semantics"]["condition"]
                or item["semantics"]["exception"]
            )
            for item in request["members"]
        }
        evidence_id = request["current_evidence"]["items"][0]["evidence_id"]
        temporal_item_ids = request["temporal_evidence"].get("item_ids", [])
        cited_evidence = sorted(
            {evidence_id, *temporal_item_ids[:1]}
        )
        temporal_status = (
            "supported"
            if request["temporal_evidence"].get("status") == "complete"
            else "not_required"
        )
        refinement_proposal = None
        verification_round = request["verification_round"]
        if self.propose_narrowing and (
            verification_round == 0 or self.repeat_refinement_round_one
        ):
            semantics = dict(request["members"][0]["semantics"])
            semantics["applies_to"] = ["src/evidence_1.py"]
            refinement_proposal = {
                "git_need": {"status": "no_established_need", "questions": []},
                "parent_member_id": member_ids[0],
                "semantics": semantics,
                "evidence_refs": [evidence_id],
                "change_kinds": ["scope"],
            }
        elif self.invalid_refinement is not None and verification_round == 0:
            member = request["members"][0]
            semantics = dict(member["semantics"])
            evidence_refs = [evidence_id]
            if self.invalid_refinement == "changes_action":
                semantics["normalized_action"] = "perform a different action"
            elif self.invalid_refinement == "reverses_polarity":
                semantics["polarity"] = "negative"
            elif self.invalid_refinement == "foreign_evidence":
                evidence_refs = [member["support_ref_ids"][0]]
            refinement_proposal = {
                "git_need": {"status": "no_established_need", "questions": []},
                "parent_member_id": member_ids[0],
                "semantics": semantics,
                "evidence_refs": evidence_refs,
                "change_kinds": ["wording"],
            }
        return {
            "member_findings": [
                {
                    "member_id": member_id,
                    "norm_relevance": "pass",
                    "rule_support": "pass",
                    "scope_support": "pass",
                    "condition_support": (
                        "pass"
                        if qualifier_by_member[member_id]
                        else "not_applicable"
                    ),
                    "temporal_status": temporal_status,
                    "evidence_sufficiency": "pass",
                    "evidence_refs": cited_evidence,
                    "closed_local_reason_codes": [],
                    "reason_codes": ["saved_current_evidence_supports"],
                }
                for member_id in member_ids
            ],
            "member_relations": [
                {
                    "left_member_id": pair[0],
                    "right_member_id": pair[1],
                    "relation": "compatible",
                    "evidence_refs": cited_evidence,
                    "reason_code": "members_are_compatible",
                }
                for pair in request["required_pairs"]
            ],
            "unresolved_conflicts": [],
            "conclusions": [
                {
                    "base_member_ids": sorted(member_ids),
                    "relation": "compatible",
                    "evidence_refs": cited_evidence,
                }
            ],
            "refinement_proposal": refinement_proposal,
        }

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings,
    ) -> ProviderResponse:
        del system_prompt
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        self.requests.append(request)
        response = self._response(request)
        return ProviderResponse(
            content=json.dumps(response),
            model=settings.model,
            input_tokens=10,
            output_tokens=10,
            reasoning_tokens=0,
            latency_ms=1,
            finish_reason="stop",
        )

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        raise AssertionError("pipeline semantic stages must bind effective settings")


class _DelayedPipelineClient(_PipelineClient):
    """Record real host concurrency while retaining deterministic responses."""

    def __init__(
        self,
        *,
        barrier_sizes: Mapping[tuple[str, int], int] | None = None,
        delay_by_request_ordinal: Mapping[int, float] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.lock = threading.Lock()
        self.active: dict[tuple[str, int], int] = {}
        self.peak: dict[tuple[str, int], int] = {}
        self.events: list[tuple[str, str, int]] = []
        self.completion_ordinals: list[int] = []
        self.delay_by_request_ordinal = dict(delay_by_request_ordinal or {})
        self.barriers = {
            key: threading.Barrier(size)
            for key, size in (barrier_sizes or {}).items()
        }

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings,
    ) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        stage = str(request["stage"])
        round_number = int(request.get("verification_round", -1))
        key = (stage, round_number)
        request_ordinal = -1
        if stage == "candidate_semantic_group":
            request_ordinal = _candidate_group_request_ordinal(
                str(request["phase"]),
                tuple(
                    (str(pair[0]), str(pair[1]))
                    for pair in request["required_pairs"]
                ),
            )
        with self.lock:
            self.active[key] = self.active.get(key, 0) + 1
            self.peak[key] = max(self.peak.get(key, 0), self.active[key])
            self.events.append(("start", stage, round_number))
        try:
            barrier = self.barriers.get(key)
            if barrier is not None:
                barrier.wait(timeout=5)
            # Long enough for all configured workers to overlap, without
            # putting wall-clock timing into any canonical record field.
            time.sleep(self.delay_by_request_ordinal.get(request_ordinal, 0.03))
            return super().complete_with_settings(
                system_prompt,
                user_prompt,
                settings=settings,
            )
        finally:
            with self.lock:
                self.events.append(("end", stage, round_number))
                if request_ordinal >= 0:
                    self.completion_ordinals.append(request_ordinal)
                self.active[key] -= 1


def _git_bundle(case_id: str, *, count: int, cutoff: str) -> GitTemporalBundle:
    slices: list[GitChangeSlice] = []
    for index in range(count):
        hunk = f"@@ -1 +1 @@\n-old_{index}\n+new_{index}\n"
        slices.append(
            GitChangeSlice.model_construct(
                item_id=stable_id("GITITEM", [case_id, index]),
                case_id=case_id,
                snapshot_digest=SNAPSHOT,
                commit_id=f"{index + 1:040x}",
                parent_id=f"{index:040x}",
                commit_parents=[f"{index:040x}"],
                commit_time=index,
                comparison_kind="ordinary",
                change_kind="modification",
                before_path="src/plugin.py",
                after_path="src/plugin.py",
                old_blob_id=f"{index + 10:040x}",
                new_blob_id=f"{index + 20:040x}",
                hunk_header="@@ -1 +1 @@",
                hunk=hunk,
                patch_sha256=hashlib.sha256(hunk.encode("utf-8")).hexdigest(),
                query_command_ids=[stable_id("GITCMD", [index])],
                query_families=["path_history"],
                anchor_ids=["ANCHOR-plugin"],
                commit_message_context_status="complete",
                commit_message_context=f"change plugin registration {index}",
                commit_message_context_sha256=hashlib.sha256(
                    f"change plugin registration {index}".encode("utf-8")
                ).hexdigest(),
                artifact_digest=content_digest([case_id, index, hunk]),
            )
        )
    manifest = GitQueryManifest.model_construct(
        command_results=[],
        manifest_id=stable_id("GITMAN", [case_id]),
        artifact_digest=content_digest(["manifest", case_id]),
    )
    return GitTemporalBundle.model_construct(
        preflight=GitPreflight.model_construct(checks=[]),
        raw_execution=GitRawExecution(logical_calls=0, physical_calls=0, cache_hits=0),
        raw_execution_scope="collection_only",
        bundle_id=stable_id("GITBUNDLE", [case_id]),
        case_id=case_id,
        snapshot_digest=SNAPSHOT,
        temporal_cutoff_commit=cutoff,
        status="complete",
        reason_codes=[],
        manifest=manifest,
        slices=slices,
        artifact_digest=content_digest(["bundle", case_id, count]),
    )


def _run(
    monkeypatch,
    repository_root: Path,
    assessments: Sequence[ChannelAssessment],
    *,
    client,
    unified_updates: Mapping[str, object] | None = None,
    snapshot_commit: str | None = None,
    anchors_for_members=None,
    git_bundle_size: int | None = None,
    profile_digest: str = PROFILE,
    protocol_manifest_digest: str = PROTOCOL_MANIFEST,
    raw_candidate_events: int | None = None,
    llm_updates: Mapping[str, object] | None = None,
    fail_on_git_preflight: bool = False,
):
    config = config_for_tests(unified=dict(unified_updates or {}))
    llm_settings = config.llm.model_copy(
        update={
            "schema_repairs": 0,
            "provider_retries": 0,
            "output_limit_retries": 0,
            **dict(llm_updates or {}),
        }
    )
    assessment_tuple = tuple(sorted(assessments, key=lambda item: item.assessment_id))
    assessment_by_id = {item.assessment_id: item for item in assessment_tuple}
    monkeypatch.setattr(
        pipeline,
        "build_channel_assessment_result",
        lambda **kwargs: ChannelAssessmentBuildResult(
            assessments=assessment_tuple,
            omissions=(),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_blocking_anchors",
        (
            anchors_for_members
            if anchors_for_members is not None
            else lambda **kwargs: {
                item.member_id: MemberBlockingAnchors()
                for item in kwargs["members"]
            }
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_case_evidence_inputs",
        lambda *, case, **kwargs: _case_evidence_inputs(case, assessment_by_id),
    )
    monkeypatch.setattr(
        pipeline,
        "preflight_git_repository",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(AssertionError("Git preflight was not expected"))
            if fail_on_git_preflight else object()
        ),
    )
    if git_bundle_size is None:
        monkeypatch.setattr(
            pipeline,
            "collect_git_evidence",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Git collection was not expected")
            ),
        )
    else:
        assert snapshot_commit is not None
        monkeypatch.setattr(
            pipeline,
            "collect_git_evidence",
            lambda *args, **kwargs: _git_bundle(
                kwargs["case_id"], count=git_bundle_size, cutoff=snapshot_commit
            ),
        )
    return pipeline.run_unified_layer(
        repository_root=repository_root,
        snapshot_digest=SNAPSHOT,
        snapshot_commit=snapshot_commit,
        profile_digest=profile_digest,
        settings=config.unified,
        llm_settings=llm_settings,
        client=client,
        budget=InferenceBudget(llm_settings),
        protocol_manifest_digest=protocol_manifest_digest,
        explicit_decisions=[],
        explicit_verification=None,  # type: ignore[arg-type]
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=raw_candidate_events,
    )


def test_empty_pipeline_produces_complete_zero_candidate_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run(monkeypatch, tmp_path, [], client=_NoCallClient())

    assert result.grouping_plan is None
    assert result.bundle.index.case_ids == []
    assert result.bundle.canonical_claims == []
    assert result.bundle.funnel.raw_candidate_events == 0
    assert result.bundle.completeness.package_status == "complete"
    assert result.grouping_execution.report_metadata() == {
        "mode": "semantic",
        "status": "not_required",
        "reason": "no_candidate_members",
        "planned_pairs": 0,
        "planned_calls": 0,
        "terminal_calls": 0,
        "case_count": 0,
    }


def test_singleton_pipeline_has_zero_grouping_calls_and_verifies_empty_pairs(
    monkeypatch, tmp_path: Path
) -> None:
    assessments = [
        _assessment(
            f"singleton-pipeline-{ordinal}",
            channel="implicit_current",
            rule="Register shared plugins before tests.",
        )
        for ordinal in range(2)
    ]

    def shared_anchors(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
            for item in kwargs["members"]
        }

    client = _PipelineClient()
    result = _run(
        monkeypatch,
        tmp_path,
        assessments,
        client=client,
        unified_updates={
            "candidate_grouping_mode": "singleton",
            "git_enabled": False,
        },
        anchors_for_members=shared_anchors,
    )

    assert result.grouping_plan is not None
    assert result.grouping_plan.semantic_pairs == ()
    assert len(result.bundle.cases) == len(assessments)
    assert all(len(case.members) == 1 for case in result.bundle.cases)
    assert all(
        case.grouping_status == "complete" for case in result.bundle.cases
    )
    assert all(not case.grouping_reason_codes for case in result.bundle.cases)
    assert not any(
        record.stage == "candidate_semantic_group"
        for record in result.inference_records
    )
    assert not any(
        request["stage"] == "candidate_semantic_group"
        for request in client.requests
    )
    verifier_requests = [
        request
        for request in client.requests
        if request["stage"] == "unified_case_verify"
    ]
    assert len(verifier_requests) == len(assessments)
    assert all(request["required_pairs"] == [] for request in verifier_requests)
    assert all(not finding.member_relations for finding in result.bundle.findings)
    assert all(not finding.unresolved_conflicts for finding in result.bundle.findings)
    assert result.grouping_execution.report_metadata() == {
        "mode": "singleton",
        "status": "not_required",
        "reason": "candidate_grouping_mode_singleton",
        "planned_pairs": 0,
        "planned_calls": 0,
        "terminal_calls": 0,
        "case_count": len(assessments),
    }
    assert result.bundle.funnel.candidate_grouping_mode == "singleton"
    assert result.bundle.funnel.grouping_status == "not_required"
    assert result.bundle.funnel.grouping_reason == (
        "candidate_grouping_mode_singleton"
    )
    assert result.bundle.funnel.grouping_planned_pairs == 0
    assert result.bundle.funnel.grouping_planned_calls == 0
    assert result.bundle.funnel.grouping_terminal_calls == 0
    assert result.bundle.funnel.grouping_case_count == len(assessments)


def test_grouping_batches_run_concurrently_in_stable_plan_order(
    monkeypatch, tmp_path: Path
) -> None:
    assessments = [
        _assessment(
            f"parallel-group-{ordinal}",
            channel="explicit",
            rule=f"Require registration variant {ordinal} for its adapter.",
        )
        for ordinal in range(4)
    ]

    def shared_anchors(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
            for item in kwargs["members"]
        }

    updates = {
        "git_enabled": False,
        "max_group_pairs_per_request": 1,
    }
    grouping_barrier = {("candidate_semantic_group", -1): 3}
    first_client = _DelayedPipelineClient(barrier_sizes=grouping_barrier)
    first = _run(
        monkeypatch,
        tmp_path / "first",
        assessments,
        client=first_client,
        unified_updates=updates,
        llm_updates={"max_concurrency": 3},
        anchors_for_members=shared_anchors,
    )
    assert first.grouping_plan is not None
    assert first.grouping_execution.mode == "semantic"
    assert first.grouping_execution.status == "complete"
    assert first.grouping_execution.planned_pairs > 0
    assert first.grouping_execution.planned_calls > 0
    assert (
        first.grouping_execution.terminal_calls
        == first.grouping_execution.planned_calls
    )
    direct_plan = plan_grouping_batches(
        first.grouping_plan,
        first.grouping_plan.semantic_pairs,
        phase="direct",
    )
    direct_records = [
        item
        for item in first.inference_records
        if item.stage == "candidate_semantic_group"
        and item.mode == "blind_pair_relation_direct"
    ]
    assert len(direct_records) >= 3
    assert first_client.peak[("candidate_semantic_group", -1)] == 3
    assert [item.request_ordinal for item in direct_records] == [
        _candidate_group_request_ordinal(batch.phase, batch.pairs)
        for batch in direct_plan.batches
    ]

    first_wave = [
        _candidate_group_request_ordinal(batch.phase, batch.pairs)
        for batch in direct_plan.batches[:3]
    ]
    reversed_latency = {
        ordinal: 0.01 + (len(first_wave) - index) * 0.04
        for index, ordinal in enumerate(first_wave)
    }
    second_client = _DelayedPipelineClient(
        barrier_sizes=grouping_barrier,
        delay_by_request_ordinal=reversed_latency,
    )
    second = _run(
        monkeypatch,
        tmp_path / "second",
        assessments,
        client=second_client,
        unified_updates=updates,
        llm_updates={"max_concurrency": 3},
        anchors_for_members=shared_anchors,
    )
    assert first.bundle.artifact_digest == second.bundle.artifact_digest
    assert first.inference_records == second.inference_records
    assert second_client.completion_ordinals[:3] == list(reversed(first_wave))


def test_round_zero_temporal_wave_spans_cases_and_folds_stably(
    monkeypatch, tmp_path: Path
) -> None:
    rules = [
        "Alphazoid files invoke frobnicate.",
        "Betamax records preserve quuxification.",
        "Gammadion modules prohibit zorbulation.",
        "Deltawave packages require wibblecheck.",
    ]
    assessments = [
        _assessment(
            f"parallel-temporal-{ordinal}",
            git_need=_affirmative_need(),
            channel="implicit_current",
            rule=rule,
        )
        for ordinal, rule in enumerate(rules)
    ]
    parallel_client = _DelayedPipelineClient(
        barrier_sizes={("temporal_evidence_classify", -1): 4}
    )
    parallel = _run(
        monkeypatch,
        tmp_path / "parallel",
        assessments,
        client=parallel_client,
        snapshot_commit="e" * 40,
        git_bundle_size=1,
        unified_updates={"max_temporal_batch_items": 1},
        llm_updates={"max_concurrency": 4},
    )
    sequential = _run(
        monkeypatch,
        tmp_path / "sequential",
        assessments,
        client=_PipelineClient(),
        snapshot_commit="e" * 40,
        git_bundle_size=1,
        unified_updates={"max_temporal_batch_items": 1},
        llm_updates={"max_concurrency": 1},
    )

    assert parallel_client.peak[("temporal_evidence_classify", -1)] == 4
    assert parallel.bundle.artifact_digest == sequential.bundle.artifact_digest
    assert parallel.inference_records == sequential.inference_records
    base_cases = {
        case.case_id: case
        for case in parallel.bundle.cases
        if case.case_id in parallel.case_routes
    }
    temporal_records = [
        record
        for record in parallel.inference_records
        if record.stage == "temporal_evidence_classify"
    ]
    record_member_ids = [
        next(
            ref.stable_id
            for ref in record.input_manifest
            if ref.kind == "candidate_case_member"
        )
        for record in temporal_records
    ]
    expected_member_ids = [
        base_cases[case_id].members[0].member_id
        for case_id in sorted(parallel.case_routes)
    ]
    assert record_member_ids == expected_member_ids


def test_initial_verifiers_are_parallel_before_serial_refinement_phase(
    monkeypatch, tmp_path: Path
) -> None:
    rules = [
        "Alphazoid files invoke frobnicate.",
        "Betamax records preserve quuxification.",
        "Gammadion modules prohibit zorbulation.",
        "Deltawave packages require wibblecheck.",
    ]
    assessments = [
        _assessment(
            f"parallel-verifier-{ordinal}",
            channel="implicit_current",
            rule=rule,
        )
        for ordinal, rule in enumerate(rules)
    ]
    client = _DelayedPipelineClient(
        propose_narrowing=True,
        barrier_sizes={("unified_case_verify", 0): 4},
    )
    result = _run(
        monkeypatch,
        tmp_path,
        assessments,
        client=client,
        unified_updates={"git_enabled": False},
        llm_updates={"max_concurrency": 4},
    )

    assert client.peak[("unified_case_verify", 0)] == 4
    assert client.peak[("unified_case_verify", 1)] == 1
    last_round_zero_end = max(
        ordinal
        for ordinal, event in enumerate(client.events)
        if event == ("end", "unified_case_verify", 0)
    )
    first_round_one_start = min(
        ordinal
        for ordinal, event in enumerate(client.events)
        if event == ("start", "unified_case_verify", 1)
    )
    assert last_round_zero_end < first_round_one_start
    verifier_modes = [
        item.mode
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ]
    assert verifier_modes == (
        ["candidate_case_round_0"] * len(assessments)
        + ["candidate_case_round_1"] * len(assessments)
    )
    assert len(result.bundle.refinements) == len(assessments)


def test_explicit_fast_path_is_accepted_and_rerun_digest_is_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "explicit-fast",
        channel="explicit",
        rule="Use the registered plugin manager.",
        source_authority="weak_or_unknown",
    )
    cutoff = "a" * 40
    first = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_NoCallClient(),
        snapshot_commit=cutoff,
    )
    second = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_NoCallClient(),
        snapshot_commit=cutoff,
    )

    assert set(first.case_routes.values()) == {"not_required_fast_path"}
    assert [item.epistemic_status for item in first.bundle.canonical_claims] == [
        "accepted"
    ]
    assert first.bundle.index.accepted_count == 1
    assert first.git_bundles == ()
    assert first.inference_records == ()
    assert first.bundle.findings[0].member_findings[0].reason_codes == [
        "explicit_local_positive_fast_path"
    ]
    assert first.bundle.member_dispositions[0].reason_codes == [
        "explicit_local_positive_fast_path"
    ]
    assert first.bundle.funnel.by_channel["explicit"] == {
        "accepted": 1,
        "invalid": 0,
        "materialized_candidate_versions": 1,
        "raw_candidate_events": 1,
        "refinement_versions": 0,
        "rejected": 0,
        "review": 0,
        "superseded": 0,
        "unique_candidate_roots": 1,
    }
    assert first.bundle.funnel.by_evidence_route["weak_or_unknown"][
        "accepted"
    ] == 1
    assert first.bundle.artifact_digest == second.bundle.artifact_digest
    assert first.bundle.index.artifact_set_digest == second.bundle.index.artifact_set_digest


@pytest.mark.parametrize(
    "normalized_status",
    ["locally_uncertain", "locally_rejected"],
)
def test_explicit_nonpositive_status_continues_to_unified_verifier(
    monkeypatch,
    tmp_path: Path,
    normalized_status: str,
) -> None:
    assessment = _assessment(
        f"explicit-{normalized_status}",
        channel="explicit",
        rule="Use the registered plugin manager.",
        source_authority="weak_or_unknown",
        normalized_status=normalized_status,
    )
    client = _PipelineClient()
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        unified_updates={"git_enabled": False},
    )

    assert set(result.case_routes.values()) == {"not_required"}
    assert [
        item.stage
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ] == ["unified_case_verify"]
    assert [
        item["stage"]
        for item in client.requests
        if item["stage"] == "unified_case_verify"
    ] == ["unified_case_verify"]
    assert result.bundle.canonical_claims[0].epistemic_status == "accepted"


def test_current_only_without_positive_git_support_can_be_accepted_observed(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "current-only",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    client = _PipelineClient()
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        unified_updates={"git_enabled": False},
    )

    assert set(result.case_routes.values()) == {"not_required"}
    assert result.git_bundles == ()
    claim = result.bundle.canonical_claims[0]
    assert claim.epistemic_status == "accepted"
    assert claim.normative_strength == "observed"
    assert any(
        item.stage == "unified_case_verify" for item in result.inference_records
    )
    evidence = result.bundle.current_evidence[0]
    git_dependency = next(
        item for item in evidence.dependencies if item.capability == "git_temporal_evidence"
    )
    assert git_dependency.requirement_mode == "not_required"
    assert git_dependency.status == "not_required"


def test_git_anchors_use_only_support_and_counterevidence_roles() -> None:
    roles = [
        "support",
        "counterevidence",
        "unknown",
        "ineligible",
        "context",
    ]
    evidence = SimpleNamespace(
        origins=[
            SimpleNamespace(
                origin_id=f"ORIGIN-{role}",
                path=f"src/{role}.py",
                line_start=1,
                line_end=2,
                origin_kind="population_member",
            )
            for role in roles
        ],
        items=[
            SimpleNamespace(
                origin_id=f"ORIGIN-{role}",
                role=role,
                anchor_literals=(f"{role}_literal",),
                anchor_symbols=(f"{role}_symbol",),
            )
            for role in roles
        ],
    )

    anchors = pipeline._git_anchors(
        SimpleNamespace(case_id="CASE-anchor-role-filter"),
        evidence,
    )

    assert {item.path for item in anchors} == {
        "src/support.py",
        "src/counterevidence.py",
    }
    assert {item.value for item in anchors if item.value is not None} == {
        "support_literal",
        "support_symbol",
        "counterevidence_literal",
        "counterevidence_symbol",
    }
    assert all(
        item.precise_origin_ids == item.origin_ids
        for item in anchors
        if item.kind in {"literal", "symbol"}
    )
    # Routing is a read-only view: excluded context remains available in the
    # original current-evidence bundle for the unified verifier.
    assert [item.role for item in evidence.items] == roles


def test_explicit_semantic_symbol_is_not_a_precise_source_anchor() -> None:
    evidence = SimpleNamespace(
        origins=[
            SimpleNamespace(
                origin_id="ORIGIN-explicit",
                origin_kind="configuration_atom",
                path="pyproject.toml",
                line_start=1,
                line_end=1,
            )
        ],
        items=[
            SimpleNamespace(
                origin_id="ORIGIN-explicit",
                role="support",
                anchor_literals=("requires-python = '>=3.10'",),
                anchor_symbols=("project.requires-python",),
            )
        ],
    )
    anchors = pipeline._git_anchors(
        SimpleNamespace(case_id="CASE-explicit-anchor"),
        evidence,
    )
    literal = next(item for item in anchors if item.kind == "literal")
    symbol = next(item for item in anchors if item.kind == "symbol")
    assert literal.precise_origin_ids == ["ORIGIN-explicit"]
    assert symbol.precise_origin_ids == []


@pytest.mark.parametrize(
    ("relation", "rule", "polarities", "channels", "expected"),
    [
        (
            "related_variant",
            "Register plugins through PluginManager.",
            ("positive", "positive"),
            ("explicit", "explicit"),
            "not_required",
        ),
        (
            "uncertain",
            "Register plugins through PluginManager.",
            ("positive", "positive"),
            ("explicit", "explicit"),
            "not_required",
        ),
        (
            "related_variant",
            "Migrate away from the deprecated compatibility interface.",
            ("positive", "positive"),
            ("explicit", "explicit"),
            "not_required",
        ),
        (
            "related_variant",
            "Register plugins through PluginManager.",
            ("positive", "negative"),
            ("explicit", "explicit"),
            "not_required",
        ),
        (
            "related_variant",
            "Register plugins through PluginManager.",
            ("positive", "positive"),
            ("implicit_current", "explicit"),
            "not_required",
        ),
    ],
)
def test_git_route_does_not_infer_need_from_keywords_or_status(
    monkeypatch,
    relation: str,
    rule: str,
    polarities: tuple[str, str],
    channels: tuple[str, str],
    expected: str,
) -> None:
    member_ids = ("MEMBER-left", "MEMBER-right")
    assessment_ids = ("ASSESS-left", "ASSESS-right")
    case = SimpleNamespace(
        grouping_status="complete",
        members=[
            SimpleNamespace(member_id=member_id, assessment_id=assessment_id)
            for member_id, assessment_id in zip(member_ids, assessment_ids)
        ],
        relations=[SimpleNamespace(relation=relation)],
    )
    assessments = {
        assessment_id: SimpleNamespace(
            integrity="valid",
            normalized_status="locally_supported",
            channel=channel,
            git_need=SimpleNamespace(status="not_assessed"),
            semantics=SimpleNamespace(polarity=polarity, rule=rule),
        )
        for assessment_id, channel, polarity in zip(
            assessment_ids, channels, polarities
        )
    }
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **_: False)

    assert pipeline._route_case(case, assessments, SimpleNamespace()) == expected


def test_git_and_verifier_route_closes_required_current_dependency_gap(
    monkeypatch,
) -> None:
    case = SimpleNamespace(
        grouping_status="complete",
        members=[
            SimpleNamespace(
                member_id="MEMBER-current-gap",
                assessment_id="ASSESS-current-gap",
            )
        ],
        relations=[],
    )
    assessments = {
        "ASSESS-current-gap": SimpleNamespace(
            integrity="valid",
            normalized_status="locally_uncertain",
            channel="implicit_current",
            semantics=SimpleNamespace(polarity="positive"),
        )
    }
    evidence = SimpleNamespace(
        dependencies=(
            CapabilityDependency(
                capability="population_classification",
                requirement_mode="required",
                status="partial",
                reason_codes=["population_query_dependency_partial"],
            ),
        )
    )
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **_: False)

    assert (
        pipeline._route_case(case, assessments, evidence)
        == "not_required_current_incomplete"
    )


def test_explicit_discovery_gap_is_augmentative_for_current_case(monkeypatch) -> None:
    case = SimpleNamespace(
        grouping_status="complete",
        members=[
            SimpleNamespace(
                member_id="MEMBER-current-explicit-gap",
                assessment_id="ASSESS-current-explicit-gap",
            )
        ],
        relations=[],
    )
    assessments = {
        "ASSESS-current-explicit-gap": SimpleNamespace(
            git_need=SimpleNamespace(status="not_assessed"),
            integrity="valid",
            normalized_status="locally_uncertain",
            channel="implicit_current",
            semantics=SimpleNamespace(polarity="positive"),
        )
    }
    evidence = SimpleNamespace(
        dependencies=(
            CapabilityDependency(
                capability="explicit_discovery_for_current_conflicts",
                requirement_mode="augmentative",
                status="partial",
                reason_codes=["explicit_discovery_incomplete"],
            ),
        )
    )
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **_: False)

    assert pipeline._route_case(case, assessments, evidence) == "not_required"


def test_git_route_ignores_only_post_route_dependency_gaps(monkeypatch) -> None:
    case = SimpleNamespace(
        grouping_status="complete",
        members=[
            SimpleNamespace(
                member_id="MEMBER-temporal-gap",
                assessment_id="ASSESS-temporal-gap",
            )
        ],
        relations=[],
    )
    assessments = {
        "ASSESS-temporal-gap": SimpleNamespace(
            git_need=_affirmative_need(required=True),
            integrity="valid",
            normalized_status="locally_uncertain",
            channel="explicit",
            semantics=SimpleNamespace(polarity="positive"),
        )
    }
    evidence = SimpleNamespace(
        dependencies=tuple(
            CapabilityDependency(
                capability=capability,
                requirement_mode="required",
                status=(
                    "failed" if capability == "refinement_git_reroute" else "partial"
                ),
                reason_codes=[f"{capability}_incomplete"],
            )
            for capability in (
                "git_temporal_evidence",
                "temporal_evidence_classification",
                "refinement_git_reroute",
            )
        )
    )
    monkeypatch.setattr(pipeline, "is_explicit_fast_path", lambda **_: False)

    assert pipeline._route_case(case, assessments, evidence) == "ambiguity_resolving"

    evidence.dependencies += (
        CapabilityDependency(
            capability="new_required_current_capability",
            requirement_mode="required",
            status="partial",
            reason_codes=["new_required_current_capability_incomplete"],
        ),
    )
    assert (
        pipeline._route_case(case, assessments, evidence)
        == "not_required_current_incomplete"
    )

    evidence.dependencies = evidence.dependencies[:-1] + (
        CapabilityDependency(
            capability="refinement_current_reacquisition",
            requirement_mode="required",
            status="failed",
            reason_codes=["refinement_current_reacquisition_gap"],
        ),
    )
    assert (
        pipeline._route_case(case, assessments, evidence)
        == "not_required_current_incomplete"
    )


def test_temporal_total_item_overflow_fails_closed_for_required_route(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "temporal-overflow",
        git_need=_affirmative_need(required=True),
        channel="explicit",
        rule="Keep the legacy plugin adapter during migration.",
    )
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates={"max_temporal_items_per_case": 64},
        snapshot_commit="e" * 40,
        git_bundle_size=65,
    )

    assert set(result.case_routes.values()) == {"ambiguity_resolving"}
    assert not any(
        item.stage == "temporal_evidence_classify"
        for item in result.inference_records
    )
    evidence = result.bundle.current_evidence[0]
    dependency = next(
        item
        for item in evidence.dependencies
        if item.capability == "temporal_evidence_classification"
    )
    assert dependency.requirement_mode == "required"
    assert dependency.status == "failed"
    assert dependency.reason_codes == ["temporal_classifier_failed"]
    assert result.bundle.canonical_claims[0].epistemic_status == "review"


def test_grouping_provider_failure_reviews_only_failed_component(
    monkeypatch, tmp_path: Path
) -> None:
    assessments = [
        _assessment(
            "failed-left",
            channel="explicit",
            rule="Require alpha registration.",
        ),
        _assessment(
            "failed-right",
            channel="explicit",
            rule="Require beta registration.",
        ),
        _assessment(
            "independent-survivor",
            channel="explicit",
            rule="Enforce isolated omega validation.",
        ),
    ]
    failed_assessment_ids = {
        assessments[0].assessment_id,
        assessments[1].assessment_id,
    }

    def anchors_for_members(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(
                symbols=("SharedRegistry",)
                if item.assessment_id in failed_assessment_ids
                else ()
            )
            for item in kwargs["members"]
        }

    result = _run(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(fail_grouping_provider=True),
        anchors_for_members=anchors_for_members,
    )

    incomplete = [
        case for case in result.bundle.cases if case.grouping_status == "incomplete"
    ]
    complete = [
        case for case in result.bundle.cases if case.grouping_status == "complete"
    ]
    assert len(incomplete) == 1 and len(incomplete[0].members) == 2
    assert "candidate_semantic_group_failed" in incomplete[0].grouping_reason_codes
    assert len(complete) == 1 and len(complete[0].members) == 1
    statuses = [item.epistemic_status for item in result.bundle.canonical_claims]
    assert statuses.count("accepted") == 1
    assert statuses.count("review") == 2
    assert result.bundle.completeness.package_status == "partial"
    assert sum(
        item.epistemic_status == "accepted"
        for item in result.bundle.canonical_claims
    ) == 1


def test_temporal_and_verifier_close_compactly_after_global_budget(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "temporal-budget-closure",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        snapshot_commit="e" * 40,
        git_bundle_size=3,
        unified_updates={"max_temporal_batch_items": 1},
        llm_updates={"max_total_calls": 1, "max_concurrency": 1},
    )

    temporal_records = [
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert len(temporal_records) == 1
    assert not any(
        item.stage == "unified_case_verify"
        for item in result.inference_records
    )
    matrix = result.bundle.temporal_matrices[0]
    assert matrix.status == "partial"
    assert len(matrix.item_ids) == 3
    assert len(matrix.relations) == 1
    assert {
        "max_total_calls",
        "temporal_batch_incomplete",
        "temporal_batches_unprocessed",
    } <= set(matrix.incomplete_reasons)
    finding = result.bundle.findings[0]
    assert finding.status == "failed"
    assert finding.inference_ref is None
    assert finding.failure_reasons == [
        "max_total_calls",
        "unified_case_verifier_budget_exhausted",
    ]


def test_git_disabled_ambiguity_resolving_case_is_review(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "legacy-ambiguity",
        git_need=_affirmative_need(required=True),
        channel="explicit",
        rule="Keep the legacy plugin adapter during migration.",
    )
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates={"git_enabled": False},
        snapshot_commit="d" * 40,
    )

    assert set(result.case_routes.values()) == {"ambiguity_resolving"}
    assert result.bundle.canonical_claims[0].epistemic_status == "review"
    evidence = result.bundle.current_evidence[0]
    dependency = next(
        item for item in evidence.dependencies if item.capability == "git_temporal_evidence"
    )
    assert dependency.requirement_mode == "required"
    assert dependency.status == "unavailable"
    assert result.bundle.completeness.package_status == "partial"


def test_partial_git_capability_filters_normal_query_closure_reasons() -> None:
    partial = SimpleNamespace(
        status="partial",
        reason_codes=[
            "query_complete",
            "query_complete_no_match",
            "merge_change_not_resolved_to_parent",
        ],
    )

    reasons = pipeline._git_temporal_gap_reasons([partial])
    capability = pipeline._capability(
        "git_temporal_evidence",
        ["partial"],
        reasons=reasons,
    )
    assert capability.status == "partial"
    assert capability.reason_codes == ["merge_change_not_resolved_to_parent"]


def test_temporal_evidence_is_classified_in_configured_batches(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "temporal-batches",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    client = _PipelineClient()
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        unified_updates={"max_temporal_batch_items": 2},
        snapshot_commit="e" * 40,
        git_bundle_size=5,
    )

    temporal_requests = [
        item for item in client.requests if item["stage"] == "temporal_evidence_classify"
    ]
    assert [len(item["items"]) for item in temporal_requests] == [2, 2, 1]
    assert len(result.bundle.temporal_matrices) == 1
    assert len(result.bundle.temporal_matrices[0].item_ids) == 5
    assert len(result.bundle.temporal_matrices[0].inference_refs) == 3
    assert result.bundle.canonical_claims[0].epistemic_status == "accepted"
    assert any(
        item.kind == "temporal"
        for item in result.bundle.canonical_claims[0].support
    )
    assert sum(
        item.stage == "temporal_evidence_classify"
        for item in result.inference_records
    ) == 3


def test_round_zero_narrowing_materializes_and_verifies_round_one(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "refinement-parent",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    client = _PipelineClient(propose_narrowing=True)

    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        unified_updates={"git_enabled": False},
    )

    verifier_requests = [
        item for item in client.requests if item["stage"] == "unified_case_verify"
    ]
    assert [item["verification_round"] for item in verifier_requests] == [0, 1]
    assert len(result.bundle.refinements) == 1
    refinement = result.bundle.refinements[0]
    assert refinement.status == "complete"
    assert len(result.bundle.assessments) == 2
    child_assessment = next(
        item for item in result.bundle.assessments if item.candidate_version == 1
    )
    assert child_assessment.parent_candidate_id == assessment.assessment_id
    assert child_assessment.semantics.applies_to == ["src/evidence_1.py"]
    assert len(result.bundle.cases) == 2
    parent_case = next(
        item for item in result.bundle.cases if item.case_id == refinement.case_id
    )
    assert "refinement_parent_superseded" in parent_case.grouping_reason_codes
    assert len(result.bundle.current_evidence) == 2
    child_current = next(
        item
        for item in result.bundle.current_evidence
        if item.bundle_id in refinement.reacquired_current_bundle_ids
    )
    assert child_current.status == "complete"
    assert any(
        item.capability == "refinement_current_filtered_complete_reuse"
        for item in child_current.dependencies
    )
    assert any(
        item.capability == "refinement_git_reroute"
        and item.status == "complete"
        for item in child_current.dependencies
    )
    assert len(result.bundle.canonical_claims) == 1
    claim = result.bundle.canonical_claims[0]
    assert claim.epistemic_status == "accepted"
    assert claim.refinement_parent_id == assessment.assessment_id
    dispositions = {
        item.member_id: item for item in result.bundle.member_dispositions
    }
    assert dispositions[refinement.parent_member_id].disposition == (
        "superseded_by_refinement"
    )
    assert dispositions[refinement.child_member_ids[0]].disposition == (
        "adopted_as_support"
    )
    capability = next(
        item
        for item in result.bundle.completeness.capabilities
        if item.capability == "unified_refinement"
    )
    assert capability.status == "complete"
    assert result.git_bundles == ()


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("changes_action", "refinement_changes_core_action"),
        ("reverses_polarity", "refinement_reverses_polarity"),
        ("foreign_evidence", "refinement_foreign_evidence"),
        ("no_op", "refinement_has_no_change"),
    ],
)
def test_invalid_optional_refinement_keeps_parent_policy_and_completeness(
    monkeypatch,
    tmp_path: Path,
    failure_kind: str,
    expected_reason: str,
) -> None:
    assessment = _assessment(
        f"ignored-{failure_kind}",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    client = _PipelineClient(invalid_refinement=failure_kind)
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        unified_updates={"git_enabled": False},
    )

    assert not result.bundle.refinements
    assert len(result.bundle.findings) == 1
    finding = result.bundle.findings[0]
    assert finding.status == "complete"
    assert finding.refinement_proposals == []
    assert len(result.bundle.canonical_claims) == 1
    assert result.bundle.canonical_claims[0].epistemic_status == "accepted"
    assert result.bundle.member_dispositions[0].disposition == "adopted_as_support"
    capability = next(
        item
        for item in result.bundle.completeness.capabilities
        if item.capability == "unified_refinement"
    )
    assert capability.status == "complete"
    verifier_records = [
        item
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ]
    assert len(verifier_records) == 1
    assert verifier_records[0].structured_response is not None
    assert verifier_records[0].structured_response["refinement_quarantine"][
        "reason_code"
    ] == expected_reason
    assert [
        item["verification_round"]
        for item in client.requests
        if item["stage"] == "unified_case_verify"
    ] == [0]


def test_multimember_refinement_has_one_active_case_and_rehomes_policy(
    monkeypatch, tmp_path: Path
) -> None:
    assessments = [
        _assessment(
            "refinement-multi-a",
            channel="implicit_current",
            rule="Register plugins through PluginManager.",
        ),
        _assessment(
            "refinement-multi-b",
            channel="implicit_current",
            rule="Register plugins through PluginManager.",
        ),
    ]
    result = _run(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(propose_narrowing=True),
        unified_updates={"git_enabled": False},
    )

    inactive = {
        "refinement_parent_superseded",
        "refinement_review_branch",
    }
    active_cases = [
        case
        for case in result.bundle.cases
        if not inactive.intersection(case.grouping_reason_codes)
    ]
    assert len(active_cases) == 1
    active = active_cases[0]
    assert len(active.members) == 2
    active_member_ids = [
        member.member_id for case in active_cases for member in case.members
    ]
    assert len(active_member_ids) == len(set(active_member_ids))
    assert result.bundle.canonical_claims
    assert {item.case_id for item in result.bundle.canonical_claims} == {
        active.case_id
    }
    unchanged_member = next(
        member for member in active.members if member.channel == "implicit_current"
        and next(
            assessment
            for assessment in result.bundle.assessments
            if assessment.assessment_id == member.assessment_id
        ).candidate_version == 0
    )
    unchanged_disposition = next(
        item
        for item in result.bundle.member_dispositions
        if item.member_id == unchanged_member.member_id
    )
    assert unchanged_disposition.case_id == active.case_id


def test_all_hard_invalid_case_skips_git_and_verifier_but_keeps_attrition(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _hard_invalid_assessment("hard-invalid")
    client = _PipelineClient()

    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=client,
        snapshot_commit="e" * 40,
    )

    assert set(result.case_routes.values()) == {"not_required_hard_invalid"}
    assert not [
        item
        for item in client.requests
        if item["stage"]
        in {"temporal_evidence_classify", "unified_case_verify"}
    ]
    assert result.git_bundles == ()
    assert result.bundle.canonical_claims == []
    assert len(result.bundle.member_dispositions) == 1
    assert result.bundle.member_dispositions[0].disposition == "invalid"
    assert result.bundle.funnel.invalid_members == 1
    assert result.bundle.funnel.by_channel["explicit"]["invalid"] == 1
    assert any(
        key.startswith("attrition:")
        for key in result.bundle.funnel.by_channel["explicit"]
    )
    finding = result.bundle.findings[0]
    assert finding.status == "not_required"
    assert finding.member_findings[0].norm_relevance == "fail"
    assert finding.member_findings[0].reason_codes == [
        "all_members_hard_invalid"
    ]


def test_round_one_optional_repeat_is_closed_without_invalidating_child_core(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "refinement-round-one-repeat",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(
            propose_narrowing=True,
            repeat_refinement_round_one=True,
        ),
        unified_updates={"git_enabled": False},
    )

    assert len(result.bundle.refinements) == 1
    assert result.bundle.refinements[0].status == "complete"
    parent_case = next(
        case
        for case in result.bundle.cases
        if case.case_id == result.bundle.refinements[0].case_id
    )
    child_case = next(
        case
        for case in result.bundle.cases
        if case.case_id != parent_case.case_id
    )
    assert "refinement_parent_superseded" in parent_case.grouping_reason_codes
    assert "refinement_review_branch" not in child_case.grouping_reason_codes
    assert {claim.case_id for claim in result.bundle.canonical_claims} == {
        child_case.case_id
    }
    child_member_ids = {member.member_id for member in child_case.members}
    assert any(
        child_member_ids.intersection(claim.contributing_member_ids)
        for claim in result.bundle.canonical_claims
    )
    round_one_record = next(
        item
        for item in result.inference_records
        if item.stage == "unified_case_verify"
        and item.mode == "candidate_case_round_1"
    )
    assert round_one_record.structured_response is not None
    assert round_one_record.structured_response["refinement_quarantine"][
        "reason_code"
    ] == "refinement_round_one_forbidden"
    capability = next(
        item
        for item in result.bundle.completeness.capabilities
        if item.capability == "unified_refinement"
    )
    assert capability.status == "complete"


def test_cross_case_canonical_identity_collision_fails_closed_without_member_loss(
    monkeypatch, tmp_path: Path
) -> None:
    assessments = [
        _assessment(
            "collision-a",
            channel="explicit",
            rule="Run pytest for every change.",
        ),
        _assessment(
            "collision-b",
            channel="explicit",
            rule="Run pytest for every change.",
        ),
    ]

    def force_separate_cases(plan, relations, **kwargs):
        del relations
        cases = []
        for member in plan.members:
            payload = {
                "case_id": make_case_id(SNAPSHOT, [member.member_id]),
                "snapshot_digest": SNAPSHOT,
                "profile_digest": kwargs["profile_digest"],
                "members": [member],
                "relations": [],
                "grouping_status": "complete",
                "grouping_reason_codes": [],
                "exact_identity_groups": [],
                "required_capabilities": ["current_evidence"],
            }
            cases.append(CandidateCase.model_validate(seal_payload(payload)))
        return SimpleNamespace(cases=tuple(cases))

    monkeypatch.setattr(
        pipeline,
        "assemble_candidate_cases",
        force_separate_cases,
    )
    result = _run(
        monkeypatch,
        tmp_path,
        assessments,
        client=_NoCallClient(),
        unified_updates={"git_enabled": False},
    )

    assert result.bundle.canonical_claims == []
    expected_members = {
        member.member_id
        for case in result.bundle.cases
        for member in case.members
    }
    assert {
        item.member_id for item in result.bundle.member_dispositions
    } == expected_members
    assert all(
        item.disposition == "review"
        and item.canonical_claim_ids == []
        and item.reason_codes == [CANONICAL_IDENTITY_COLLISION_REASON]
        for item in result.bundle.member_dispositions
    )
    capability = next(
        item
        for item in result.bundle.completeness.capabilities
        if item.capability == "canonical_identity_resolution"
    )
    assert capability.status == "partial"
    assert capability.reason_codes == [CANONICAL_IDENTITY_COLLISION_REASON]
    assert result.bundle.completeness.package_status == "partial"
    assert result.bundle.completeness.omitted_possible
    assert result.bundle.funnel.attrition_reasons[
        CANONICAL_IDENTITY_COLLISION_REASON
    ] == 2
    assert result.bundle.funnel.by_channel["explicit"]["review"] == 2
    assert result.bundle.funnel.by_channel["explicit"][
        f"attrition:{CANONICAL_IDENTITY_COLLISION_REASON}"
    ] == 2


def test_funnel_preserves_raw_to_root_pre_normalization_attrition(
    monkeypatch, tmp_path: Path
) -> None:
    assessment = _assessment(
        "raw-denominator",
        channel="explicit",
        rule="Run pytest before submitting.",
    )
    result = _run(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_NoCallClient(),
        raw_candidate_events=3,
    )

    channel = result.bundle.funnel.by_channel["explicit"]
    assert channel["raw_candidate_events"] == 3
    assert channel["unique_candidate_roots"] == 1
    assert channel["attrition:candidate_event_not_materialized"] == 2
    assert result.bundle.funnel.by_evidence_route[
        "pre_normalization_unattributed"
    ]["raw_candidate_events"] == 2

    invalid = result.bundle.funnel.model_dump(mode="python")
    invalid["by_channel"]["explicit"]["accepted"] += 1
    with pytest.raises(ValidationError, match="final disposition algebra"):
        UnifiedFunnel.model_validate(invalid)
