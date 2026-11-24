from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable

import pytest

from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceRecord,
    contractual_request_bytes,
    inference_cache_key_for_record,
)
from repo_constraint_memory_v3.models import LLMSettings, ProviderResponse
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.unified.git_evidence import (
    GitChangeSlice,
    GitLimits,
    GitQueryManifest,
    GitTemporalBundle,
    plan_git_queries,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    make_assessment_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
    make_member_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    PopulationDenominator,
    UNIFIED_CONTRACT_VERSION,
)
from repo_constraint_memory_v3.unified.pipeline import _git_anchors
from repo_constraint_memory_v3.unified.semantic import (
    CANDIDATE_GROUP_PROTOCOL_VERSION,
    CANDIDATE_GROUP_RESPONSE_VERSION,
    TEMPORAL_CLASSIFIER_PROTOCOL_VERSION,
    UNIFIED_VERIFIER_PROTOCOL_VERSION,
    UNIFIED_VERIFIER_RESPONSE_VERSION,
    VERIFIER_SYSTEM_PROMPT,
    CandidateSemanticGroupResponse,
    TemporalEvidenceClassificationResponse,
    UnifiedCaseVerificationResponse,
    _verifier_provider_response_model,
    allowed_verifier_evidence_refs,
    build_candidate_semantic_group_prompt,
    build_temporal_evidence_prompt,
    build_unified_case_verify_prompt,
    build_verifier_current_evidence_view,
    candidate_semantic_group,
    git_manifest_ref,
    replay_candidate_semantic_group_record,
    replay_temporal_evidence_record,
    replay_unified_case_verify_record,
    temporal_evidence_classify,
    temporal_evidence_classify_bounded,
    temporal_evidence_classify_cases_bounded,
    unified_case_verify,
)


SNAPSHOT = "sha256:" + "1" * 64
PROFILE = "2" * 64
MANIFEST = "3" * 64


def test_verifier_prompt_contrastively_checks_governed_scope() -> None:
    assert "governed-change test" in VERIFIER_SYSTEM_PROMPT
    assert "behavioral\n  witness" in VERIFIER_SYSTEM_PROMPT
    assert "test construction directly governs test files" in (
        VERIFIER_SYSTEM_PROMPT
    )
    assert "both evidence and scope" in VERIFIER_SYSTEM_PROMPT
    assert "never infer a\ntarget from directory role" in VERIFIER_SYSTEM_PROMPT
    assert "normalization does not prove qualifier completeness" in (
        VERIFIER_SYSTEM_PROMPT
    )
    assert "material-qualifier test" in VERIFIER_SYSTEM_PROMPT
    assert "member-bound current evidence positively establishes" in (
        VERIFIER_SYSTEM_PROMPT
    )
    assert "do not penalize an absent" in VERIFIER_SYSTEM_PROMPT
    assert "independently usable branches" in VERIFIER_SYSTEM_PROMPT
    assert "never return not_applicable" in VERIFIER_SYSTEM_PROMPT
    assert "host ignores that provisional value" in VERIFIER_SYSTEM_PROMPT


def _settings(*, schema_repairs: int = 0) -> LLMSettings:
    return LLMSettings(
        model="fake-unified-semantic",
        require_response_model=True,
        accepted_effective_model_ids=["fake-unified-semantic"],
        api_key_env="FAKE_API_KEY",
        base_url="https://invalid.example/v1",
        reasoning_wire_policy="offline_mock",
        thinking_mode="disabled",
        schema_repairs=schema_repairs,
        output_limit_retries=0,
        provider_retries=0,
    )


def _ref(kind: str, stable_id: str, digest: str) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id,
        snapshot_digest=SNAPSHOT,
        artifact_path=f"audit/{stable_id}.json",
        artifact_digest=digest,
    )


def _semantics(
    label: str,
    *,
    condition: str | None = None,
    exception: str | None = None,
) -> CandidateSemantics:
    return CandidateSemantics(
        rule=f"Use {label} for plugin registration.",
        normalized_action="use decorator for plugin registration",
        polarity="positive",
        applies_to=["src/**"],
        condition=condition,
        exception=exception,
    )


def _assessment(
    label: str,
    *,
    channel: str,
    raw_status: str,
    normalized_status: str,
    support_ref: ImmutableArtifactRef,
    condition: str | None = None,
    exception: str | None = None,
) -> ChannelAssessment:
    root_id = f"ROOT-{label}"
    assessment_id = make_assessment_id(
        SNAPSHOT, PROFILE, channel, root_id, 0
    )
    payload: dict[str, object] = {
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "assessment_id": assessment_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "channel": channel,
        "candidate_root_id": root_id,
        "candidate_version": 0,
        "parent_candidate_id": None,
        "semantics": _semantics(
            label,
            condition=condition,
            exception=exception,
        ).model_dump(mode="json"),
        "proposed_claim_kind": (
            "normative_declaration"
            if channel == "explicit"
            else "observed_convention"
        ),
        "proposed_strength": "required" if channel == "explicit" else "observed",
        "raw_channel_status": raw_status,
        "raw_policy_version": "local-policy-1",
        "normalized_status": normalized_status,
        "axis_findings": {
            "evidence_sufficiency": "pass",
            "norm_relevance": "pass",
        },
        "typed_reasons": ["local_ambiguity"],
        "support_refs": [support_ref.model_dump(mode="json")],
        "counterevidence_refs": [],
        "coverage_refs": [],
        "lineage_refs": [],
        "extension_refs": [],
        "integrity": "valid",
        "source_authority": (
            "direct_normative" if channel == "explicit" else "current_population"
        ),
        "git_need": {"status": "not_assessed", "judgments": [], "reason": "host_fixture"},
        "provenance": [f"source:{label}"],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return ChannelAssessment.model_validate(payload)


def _case_fixture(
    label: str = "decorator",
    *,
    condition: str | None = None,
    exception: str | None = None,
) -> tuple[
    CandidateCase,
    list[ChannelAssessment],
    CurrentEvidenceBundle,
]:
    first_support = _ref("explicit_evidence", f"EXPLICIT-{label}", "4" * 64)
    second_support = _ref("population_member", f"POP-{label}", "5" * 64)
    assessments = [
        _assessment(
            f"{label}-explicit",
            channel="explicit",
            raw_status="accepted_by_explicit_channel",
            normalized_status="locally_supported",
            support_ref=first_support,
            condition=condition,
            exception=exception,
        ),
        _assessment(
            f"{label}-current",
            channel="implicit_current",
            raw_status="review_by_current_channel",
            normalized_status="locally_uncertain",
            support_ref=second_support,
            condition=condition,
            exception=exception,
        ),
    ]
    members: list[CandidateCaseMember] = []
    for assessment in assessments:
        assessment_ref = _ref(
            "explicit_candidate" if assessment.channel == "explicit" else "norm_hypothesis",
            assessment.assessment_id,
            assessment.artifact_digest,
        )
        members.append(
            CandidateCaseMember(
                member_id=make_member_id(SNAPSHOT, assessment.assessment_id),
                assessment_id=assessment.assessment_id,
                snapshot_digest=SNAPSHOT,
                channel=assessment.channel,
                semantics=assessment.semantics,
                assessment_ref=assessment_ref,
            )
        )
    members.sort(key=lambda item: item.member_id)
    relation = CandidateMemberRelation(
        left_member_id=members[0].member_id,
        right_member_id=members[1].member_id,
        relation="related_variant",
        source="semantic_grouping",
        reason_code="same_registration_decision",
    )
    case_id = make_case_id(SNAPSHOT, [item.member_id for item in members])
    case_payload: dict[str, object] = {
        "case_id": case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "members": [item.model_dump(mode="json") for item in members],
        "relations": [relation.model_dump(mode="json")],
        "grouping_status": "complete",
        "grouping_reason_codes": [],
        "exact_identity_groups": [],
        "required_capabilities": ["current_evidence", "unified_verifier"],
    }
    case_payload["artifact_digest"] = artifact_digest(case_payload)
    case = CandidateCase.model_validate(case_payload)

    content_digest = "6" * 64
    origin_id = make_evidence_origin_id(
        SNAPSHOT, "observation", "src/plugins.py", [4, 7], content_digest
    )
    origin = EvidenceOrigin(
        origin_id=origin_id,
        origin_kind="observation",
        path="src/plugins.py",
        line_start=4,
        line_end=7,
        source_content_digest=content_digest,
        independence_group="src/plugins.py:register",
    )
    observation_ref = _ref("observation", f"OBS-{label}", "7" * 64)
    member_ids = [item.member_id for item in members]
    evidence_id = make_evidence_id(
        origin_id,
        "support",
        member_ids,
        observation_ref.stable_id,
    )
    item = CurrentEvidenceItem(
        evidence_id=evidence_id,
        origin_id=origin_id,
        role="support",
        member_ids=member_ids,
        artifact_ref=observation_ref,
        excerpt_digest="8" * 64,
        description="Both plugin modules use the registration decorator.",
    )
    dependency = CapabilityDependency(
        capability="current_snapshot",
        requirement_mode="required",
        status="complete",
        reason_codes=[],
    )
    bundle_payload: dict[str, object] = {
        "bundle_id": make_current_evidence_bundle_id(SNAPSHOT, PROFILE, case_id),
        "case_id": case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "origins": [origin.model_dump(mode="json")],
        "items": [item.model_dump(mode="json")],
        "denominators": [],
        "population_refs": [],
        "falsification_refs": [],
        "scope_condition_refs": [],
        "dependencies": [dependency.model_dump(mode="json")],
        "integrity": "valid",
        "status": "complete",
        "incomplete_reasons": [],
    }
    bundle_payload["artifact_digest"] = artifact_digest(bundle_payload)
    return case, assessments, CurrentEvidenceBundle.model_validate(bundle_payload)


def _with_ineligible_evidence(
    current: CurrentEvidenceBundle,
    *,
    count: int = 24,
) -> CurrentEvidenceBundle:
    payload = current.model_dump(mode="python", exclude={"artifact_digest"})
    member_id = current.items[0].member_ids[0]
    for ordinal in range(count):
        path = f"src/ineligible_{ordinal:03d}.py"
        source_digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        origin_id = make_evidence_origin_id(
            SNAPSHOT,
            "population_member",
            path,
            [ordinal + 1, ordinal + 1],
            source_digest,
        )
        artifact_ref = _ref(
            "population_member",
            f"POP-INELIGIBLE-{ordinal:03d}",
            hashlib.sha256(f"artifact-{ordinal}".encode("utf-8")).hexdigest(),
        )
        evidence_id = make_evidence_id(
            origin_id,
            "ineligible",
            [member_id],
            artifact_ref.stable_id,
        )
        payload["origins"].append(
            EvidenceOrigin(
                origin_id=origin_id,
                origin_kind="population_member",
                path=path,
                line_start=ordinal + 1,
                line_end=ordinal + 1,
                source_content_digest=source_digest,
                independence_group=f"ineligible:{ordinal:03d}",
            ).model_dump(mode="python")
        )
        payload["items"].append(
            CurrentEvidenceItem(
                evidence_id=evidence_id,
                origin_id=origin_id,
                role="ineligible",
                member_ids=[member_id],
                artifact_ref=artifact_ref,
                excerpt_digest=hashlib.sha256(
                    f"excerpt-{ordinal}".encode("utf-8")
                ).hexdigest(),
                description=(
                    f"VERIFIER_MUST_NOT_SEE_INELIGIBLE_DETAIL_{ordinal:03d}"
                ),
            ).model_dump(mode="python")
        )
    payload["origins"] = sorted(
        payload["origins"], key=lambda item: item["origin_id"]
    )
    payload["items"] = sorted(
        payload["items"], key=lambda item: item["evidence_id"]
    )
    payload["denominators"] = [
        PopulationDenominator(
            member_id=member_id,
            population_id="POPULATION-INELIGIBLE-AUDIT",
            status="complete",
            census_enumeration_status="complete",
            sampling_plan_status="complete",
            sampling_mode="census",
            scanned=count,
            serialized=count,
            matched=count,
            inspected=count,
            uninspected=0,
            support=0,
            counterexample=0,
            unknown=0,
            ineligible=count,
        ).model_dump(mode="python")
    ]
    payload["artifact_digest"] = artifact_digest(payload)
    return CurrentEvidenceBundle.model_validate(payload)


def _git_bundle(case: CandidateCase) -> GitTemporalBundle:
    patch = "@@ -1 +1 @@\n-register(plugin)\n+@register\n"
    slice_item = GitChangeSlice.model_construct(
        item_id="GITITEM-" + "9" * 24,
        case_id=case.case_id,
        snapshot_digest=SNAPSHOT,
        commit_id="a" * 40,
        parent_id="b" * 40,
        commit_parents=["b" * 40],
        commit_time=1,
        commit_message_context_status="complete",
        commit_message_context="replace registration syntax",
        commit_message_context_sha256=hashlib.sha256(
            b"replace registration syntax"
        ).hexdigest(),
        comparison_kind="ordinary",
        change_kind="modification",
        before_path="src/plugins.py",
        after_path="src/plugins.py",
        hunk_header="@@ -1 +1 @@",
        hunk=patch,
        patch_sha256="c" * 64,
        artifact_digest="d" * 64,
    )
    return GitTemporalBundle.model_construct(
        case_id=case.case_id,
        snapshot_digest=SNAPSHOT,
        manifest=GitQueryManifest.model_construct(
            manifest_id="GITMAN-" + "e" * 24,
            artifact_digest="f" * 64,
        ),
        slices=[slice_item],
    )


def _git_bundle_with_items(case: CandidateCase, count: int) -> GitTemporalBundle:
    original = _git_bundle(case)
    slices = [
        original.slices[0].model_copy(
            update={
                "item_id": "GITITEM-" + str(ordinal) * 24,
                "commit_id": str(ordinal) * 40,
                "artifact_digest": str(ordinal + 2) * 64,
            }
        )
        for ordinal in range(1, count + 1)
    ]
    return original.model_copy(update={"slices": slices})


def _request_from_prompt(payload: dict[str, object]) -> dict[str, object]:
    original = payload.get("original_request")
    return original if isinstance(original, dict) else payload


def _valid_response(request: dict[str, object]) -> dict[str, object]:
    stage = request["stage"]
    if stage == "candidate_semantic_group":
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
        span_by_item = {
            item["item_id"]: item["evidence_span_ref"] for item in request["items"]
        }
        return {
            "relations": [
                {
                    "item_id": cell[0],
                    "member_id": cell[1],
                    "relation": "supports_as_written",
                    "affected_facets": ["rule"],
                    "evidence_span": span_by_item[cell[0]],
                    "reason_code": "patch_introduces_pattern",
                    "uncertainty_reason": None,
                }
                for cell in request["required_cells"]
            ]
        }
    assert stage == "unified_case_verify"
    member_ids = request["required_member_ids"]
    evidence_id = request["current_evidence"]["items"][0]["evidence_id"]
    temporal_ids = request["temporal_evidence"].get("item_ids", [])
    evidence_refs = sorted({evidence_id, *temporal_ids[:1]})
    return {
        "member_findings": [
            {
                "member_id": member_id,
                "norm_relevance": "pass",
                "rule_support": "pass",
                "scope_support": "pass",
                "condition_support": "pass",
                "temporal_status": "supported",
                "evidence_sufficiency": "pass",
                "evidence_refs": evidence_refs,
                "closed_local_reason_codes": ["local_ambiguity"],
                "reason_codes": ["current_evidence_supports"],
            }
            for member_id in member_ids
        ],
        "member_relations": [
            {
                "left_member_id": pair[0],
                "right_member_id": pair[1],
                "relation": "compatible",
                "evidence_refs": evidence_refs,
                "reason_code": "same_decision_compatible",
            }
            for pair in request["required_pairs"]
        ],
        "unresolved_conflicts": [],
        "conclusions": [
            {
                "base_member_ids": sorted(member_ids),
                "relation": "compatible",
                "evidence_refs": evidence_refs,
            }
        ],
        "refinement_proposal": None,
    }


class _ScriptedClient:
    def __init__(
        self,
        response: Callable[[dict[str, object], int], dict[str, object]],
    ) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.settings_seen: list[LLMSettings] = []
        self.system_prompts: list[str] = []

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: LLMSettings,
    ) -> ProviderResponse:
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        self.settings_seen.append(settings)
        self.system_prompts.append(system_prompt)
        response = self.response(_request_from_prompt(payload), len(self.calls) - 1)
        return ProviderResponse(
            content=json.dumps(response),
            model=settings.model,
            input_tokens=20,
            output_tokens=20,
            reasoning_tokens=0,
            latency_ms=1,
            finish_reason="stop",
        )


class _MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def read(self, key: str) -> dict[str, object] | None:
        return self.values.get(key)

    def write(self, key: str, record: dict[str, object]) -> None:
        self.values[key] = record


def _record_with_structured_response(
    record: InferenceRecord,
    response: dict[str, object],
) -> InferenceRecord:
    payload = record.model_dump(mode="json")
    payload["structured_response"] = response
    payload["structured_response_digest"] = payload_digest({"response": response})
    payload.pop("record_digest", None)
    payload["record_digest"] = payload_digest(payload)
    return InferenceRecord.model_validate(payload)


def test_candidate_group_is_complete_blind_stable_and_successful() -> None:
    case, _, _ = _case_fixture()
    client = _ScriptedClient(lambda request, _: _valid_response(request))
    settings = _settings(schema_repairs=3)
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert relations is not None and len(relations) == 1
    assert record.stage == "candidate_semantic_group"
    assert record.protocol_version == CANDIDATE_GROUP_PROTOCOL_VERSION
    assert CANDIDATE_GROUP_PROTOCOL_VERSION.endswith("1.6")
    assert CANDIDATE_GROUP_RESPONSE_VERSION.endswith("1.2")
    assert "Different commands" in client.system_prompts[0]
    assert "different configuration keys" in client.system_prompts[0]
    assert "compatible_variant" in client.system_prompts[0]
    assert "mutually exclusive outcomes" in client.system_prompts[0]
    assert record.schema_repairs == 1
    assert client.settings_seen[0].schema_repairs == 1
    prompt = client.calls[0]
    serialized = json.dumps(prompt)
    assert "raw_channel_status" not in serialized
    assert "normalized_status" not in serialized
    assert "accepted_by_explicit_channel" not in serialized
    assert "review_by_current_channel" not in serialized
    assert set(prompt["members"][0]) == {"member_id", "semantics"}
    assert relations[0].relation == "related_variant"
    assert relations[0].reason_code == "model_conflicting_variant"

    same_prompt = build_candidate_semantic_group_prompt(reversed(case.members))
    assert json.loads(same_prompt)["required_pairs"] == prompt["required_pairs"]
    repeated_client = _ScriptedClient(lambda request, _: _valid_response(request))
    _, repeated_record = candidate_semantic_group(
        list(reversed(case.members)),
        llm_settings=settings,
        client=repeated_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert repeated_record.request_ordinal == record.request_ordinal
    assert repeated_record.request_id == record.request_id


def test_unified_request_byte_gates_measure_schema_attached_provider_payloads(
) -> None:
    case, assessments, current = _case_fixture("contractual-byte-gates")
    settings = _settings()

    group_request = build_candidate_semantic_group_prompt(case.members)
    group_raw_bytes = len(group_request.encode("utf-8"))
    assert group_raw_bytes < contractual_request_bytes(
        group_request,
        CandidateSemanticGroupResponse,
    )
    group_client = _ScriptedClient(lambda request, _: _valid_response(request))
    relations, group_record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=group_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        max_request_bytes=group_raw_bytes,
    )
    assert relations is None
    assert group_client.calls == []
    assert group_record.incomplete_reasons == [
        "max_candidate_group_request_bytes"
    ]

    git_bundle = _git_bundle(case)
    temporal_request = build_temporal_evidence_prompt(
        case,
        git_bundle.slices,
    )
    temporal_raw_bytes = len(temporal_request.encode("utf-8"))
    assert temporal_raw_bytes < contractual_request_bytes(
        temporal_request,
        TemporalEvidenceClassificationResponse,
    )
    temporal_client = _ScriptedClient(
        lambda request, _: _valid_response(request)
    )
    matrix, temporal_record = temporal_evidence_classify(
        case,
        git_bundle,
        llm_settings=settings,
        client=temporal_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        max_request_bytes=temporal_raw_bytes,
    )
    assert matrix is None
    assert temporal_client.calls == []
    assert temporal_record.incomplete_reasons == [
        "max_temporal_classifier_request_bytes"
    ]

    verifier_request = build_unified_case_verify_prompt(
        case,
        assessments,
        current,
        None,
        verification_round=0,
    )
    verifier_response_model = _verifier_provider_response_model(
        tuple(
            sorted(
                allowed_verifier_evidence_refs(
                    assessments,
                    current,
                    None,
                )
            )
        )
    )
    verifier_raw_bytes = len(verifier_request.encode("utf-8"))
    assert verifier_raw_bytes < contractual_request_bytes(
        verifier_request,
        verifier_response_model,
    )
    verifier_client = _ScriptedClient(
        lambda request, _: _valid_response(request)
    )
    finding, verifier_record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=verifier_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        max_request_bytes=verifier_raw_bytes,
    )
    assert finding is None
    assert verifier_client.calls == []
    assert verifier_record.incomplete_reasons == [
        "max_unified_verifier_request_bytes"
    ]


@pytest.mark.parametrize(
    ("provider_relation", "artifact_relation", "artifact_reason"),
    [
        (
            "conflicting_variant",
            "related_variant",
            "model_conflicting_variant",
        ),
        (
            "compatible_variant",
            "distinct",
            "model_compatible_variant",
        ),
    ],
)
def test_candidate_group_projects_richer_provider_relation_and_replays_exactly(
    provider_relation: str,
    artifact_relation: str,
    artifact_reason: str,
) -> None:
    case, _, _ = _case_fixture(f"group-relation-{provider_relation}")

    def response(request: dict[str, object], _: int) -> dict[str, object]:
        return {
            "relations": [
                {
                    "left_member_id": pair[0],
                    "right_member_id": pair[1],
                    "relation": provider_relation,
                }
                for pair in request["required_pairs"]
            ]
        }

    settings = _settings()
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=_ScriptedClient(response),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert relations is not None
    assert relations[0].relation == artifact_relation
    assert relations[0].reason_code == artifact_reason
    assert replay_candidate_semantic_group_record(
        record,
        case.members,
    ) == relations


def test_candidate_group_rejects_legacy_ambiguous_related_variant_label() -> None:
    case, _, _ = _case_fixture("group-legacy-related")

    def legacy_response(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        return {
            "relations": [
                {
                    "left_member_id": pair[0],
                    "right_member_id": pair[1],
                    "relation": "related_variant",
                }
                for pair in request["required_pairs"]
            ]
        }

    settings = _settings()
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=_ScriptedClient(legacy_response),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert relations is None
    assert record.terminal_status == "failure"


def test_candidate_group_request_identity_binds_phase_and_exact_pair_subset() -> None:
    case, _, _ = _case_fixture("pair-identity")
    third = case.members[0].model_copy(
        update={"member_id": "MEMBER-" + "f" * 24}
    )
    members = sorted([*case.members, third], key=lambda item: item.member_id)
    first_pairs = [
        (members[0].member_id, members[1].member_id),
        (members[0].member_id, members[2].member_id),
    ]
    second_pairs = [
        (members[0].member_id, members[1].member_id),
        (members[1].member_id, members[2].member_id),
    ]
    settings = _settings()

    _, first = candidate_semantic_group(
        members,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        required_pairs=first_pairs,
        phase="direct",
    )
    _, second = candidate_semantic_group(
        members,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        required_pairs=second_pairs,
        phase="direct",
    )
    _, closure = candidate_semantic_group(
        members,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        required_pairs=first_pairs,
        phase="closure",
    )

    assert len({first.request_id, second.request_id, closure.request_id}) == 3
    assert first.input_manifest == second.input_manifest == closure.input_manifest
    assert first.mode == "blind_pair_relation_direct"
    assert closure.mode == "blind_pair_relation_closure"


@pytest.mark.parametrize("failure_kind", ["missing", "duplicate", "foreign"])
def test_candidate_group_rejects_bad_pair_assignment(failure_kind: str) -> None:
    case, _, _ = _case_fixture(f"group-{failure_kind}")

    def bad(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        relation = dict(response["relations"][0])
        if failure_kind == "missing":
            response["relations"] = []
        elif failure_kind == "duplicate":
            response["relations"] = [relation, relation]
        else:
            relation["right_member_id"] = "MEMBER-" + "f" * 24
            response["relations"] = [relation]
        return response

    settings = _settings()
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=_ScriptedClient(bad),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert relations is None
    assert record.terminal_status == "failure"
    assert record.structured_response is None


def test_candidate_group_discards_foreign_surplus_after_complete_coverage() -> None:
    case, _, _ = _case_fixture("group-foreign-surplus")
    third = case.members[0].model_copy(
        update={"member_id": "MEMBER-" + "f" * 24}
    )
    members = sorted([*case.members, third], key=lambda item: item.member_id)
    required_pair = (members[0].member_id, members[1].member_id)
    surplus_pair = (members[1].member_id, members[2].member_id)

    def surplus(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        response["relations"].append(
            {
                "left_member_id": surplus_pair[0],
                "right_member_id": surplus_pair[1],
                "relation": "distinct",
            }
        )
        return response

    settings = _settings()
    relations, record = candidate_semantic_group(
        members,
        llm_settings=settings,
        client=_ScriptedClient(surplus),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
        required_pairs=[required_pair],
    )

    assert relations is not None and len(relations) == 1
    assert record.terminal_status == "success"
    assert len(record.attempts) == 1
    assert record.structured_response is not None
    assert len(record.structured_response["relations"]) == 1
    assert replay_candidate_semantic_group_record(
        record,
        members,
        expected_pairs=[required_pair],
    ) == relations


def test_candidate_group_rejects_surplus_that_cites_unknown_member() -> None:
    case, _, _ = _case_fixture("group-unknown-surplus")

    def surplus(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        response["relations"].append(
            {
                "left_member_id": request["required_pairs"][0][0],
                "right_member_id": "MEMBER-" + "f" * 24,
                "relation": "distinct",
            }
        )
        return response

    settings = _settings()
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=_ScriptedClient(surplus),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert relations is None
    assert record.terminal_status == "failure"
    assert any(
        "surplus cites foreign members" in reason
        for reason in record.incomplete_reasons
    )


def test_candidate_group_repairs_at_most_once() -> None:
    case, _, _ = _case_fixture("repair")

    def first_invalid(request: dict[str, object], ordinal: int) -> dict[str, object]:
        return {"relations": []} if ordinal == 0 else _valid_response(request)

    settings = _settings(schema_repairs=3)
    client = _ScriptedClient(first_invalid)
    relations, record = candidate_semantic_group(
        case.members,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert relations is not None
    assert len(client.calls) == 2
    assert record.schema_repairs == 1
    assert [item.repair for item in record.attempts] == [False, True]
    assert client.calls[1]["repair"] is True


def test_temporal_classifier_builds_exact_matrix_with_input_span() -> None:
    case, assessments, current = _case_fixture("temporal")
    git_bundle = _git_bundle(case)
    client = _ScriptedClient(lambda request, _: _valid_response(request))
    settings = _settings(schema_repairs=1)
    matrix, record = temporal_evidence_classify(
        case,
        git_bundle,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert matrix is not None
    assert matrix.status == "complete"
    assert matrix.git_manifest_ref is not None
    assert matrix.git_manifest_ref.stable_id == git_bundle.manifest.manifest_id
    assert any(
        item.kind == "git_query_manifest" for item in record.input_manifest
    )
    assert len(matrix.relations) == len(case.members)
    span_ref = client.calls[0]["items"][0]["evidence_span_ref"]
    assert {item.evidence_span for item in matrix.relations} == {span_ref}
    assert client.calls[0]["items"][0]["commit_message_context"] == (
        "replace registration syntax"
    )
    assert client.calls[0]["items"][0]["commit_message_context_status"] == (
        "complete"
    )
    assert record.stage == "temporal_evidence_classify"

    verifier = _ScriptedClient(lambda request, _: _valid_response(request))
    finding, _ = unified_case_verify(
        case,
        assessments,
        current,
        matrix,
        llm_settings=settings,
        client=verifier,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None
    assert git_bundle.slices[0].item_id in finding.member_findings[0].evidence_refs


def test_temporal_classifier_canonicalizes_presentation_lists_and_replays() -> None:
    case, _, _ = _case_fixture("temporal-canonical")
    git_bundle = _git_bundle(case)

    def shuffled(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        response["relations"] = list(reversed(response["relations"]))
        for relation in response["relations"]:
            relation["affected_facets"] = ["scope", "rule", "scope"]
        return response

    settings = _settings()
    matrix, record = temporal_evidence_classify(
        case,
        git_bundle,
        llm_settings=settings,
        client=_ScriptedClient(shuffled),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert matrix is not None
    assert record.protocol_version == TEMPORAL_CLASSIFIER_PROTOCOL_VERSION
    assert TEMPORAL_CLASSIFIER_PROTOCOL_VERSION.endswith("1.2")
    assert all(
        relation.affected_facets == ["rule", "scope"]
        for relation in matrix.relations
    )
    assert record.structured_response is not None
    assert replay_temporal_evidence_record(
        record,
        case,
        git_bundle.slices,
        git_manifest_ref(git_bundle),
    ) == matrix.relations


def test_temporal_cell_cap_and_bisection_recover_complete_matrix() -> None:
    case, _, _ = _case_fixture("temporal-cell-cap")
    temporal_bundle = _git_bundle_with_items(case, 5)
    settings = _settings()
    capped_client = _ScriptedClient(lambda request, _: _valid_response(request))

    matrix, records = temporal_evidence_classify_bounded(
        case,
        temporal_bundle,
        max_batch_items=4,
        max_cells_per_request=4,
        max_total_items=5,
        max_concurrency=1,
        llm_settings=settings,
        client=capped_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert matrix is not None and matrix.status == "complete"
    assert len(records) == 3
    assert [len(call["items"]) for call in capped_client.calls] == [2, 2, 1]
    assert all(len(call["required_cells"]) <= 4 for call in capped_client.calls)

    recovery_bundle = _git_bundle_with_items(case, 3)

    def fail_multi_item(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        if len(request["items"]) > 1:
            response["relations"] = []
        return response

    recovery_client = _ScriptedClient(fail_multi_item)
    recovered, recovery_records = temporal_evidence_classify_bounded(
        case,
        recovery_bundle,
        max_batch_items=3,
        max_cells_per_request=6,
        max_total_items=3,
        max_concurrency=1,
        llm_settings=settings,
        client=recovery_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert recovered is not None and recovered.status == "complete"
    assert len(recovered.relations) == 3 * len(case.members)
    assert len(recovered.inference_refs) == 3
    assert len(recovery_records) == 5
    assert sum(
        record.terminal_status == "failure" for record in recovery_records
    ) == 2


def test_temporal_top_level_batches_run_bounded_parallel_and_merge_stably() -> None:
    case, _, _ = _case_fixture("temporal-parallel")
    temporal_bundle = _git_bundle_with_items(case, 4)
    settings = _settings()
    lock = threading.Lock()
    active = 0
    peak = 0
    completion_order: list[str] = []

    def delayed_response(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        nonlocal active, peak
        item_id = str(request["items"][0]["item_id"])
        delay_ordinal = int(item_id[-1])
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            # Force completion order to differ from planned item order.  The
            # host must still return records in the deterministic batch order.
            time.sleep((5 - delay_ordinal) * 0.02)
            return _valid_response(request)
        finally:
            with lock:
                active -= 1
                completion_order.append(item_id)

    parallel_matrix, parallel_records = temporal_evidence_classify_bounded(
        case,
        temporal_bundle,
        max_batch_items=1,
        max_cells_per_request=2,
        max_total_items=4,
        max_concurrency=3,
        llm_settings=settings,
        client=_ScriptedClient(delayed_response),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    sequential_matrix, sequential_records = temporal_evidence_classify_bounded(
        case,
        temporal_bundle,
        max_batch_items=1,
        max_cells_per_request=2,
        max_total_items=4,
        max_concurrency=1,
        llm_settings=settings,
        client=_ScriptedClient(  # type: ignore[arg-type]
            lambda request, _: _valid_response(request)
        ),
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert 1 < peak <= 3
    assert completion_order != sorted(completion_order)
    assert parallel_matrix == sequential_matrix
    assert [item.request_id for item in parallel_records] == [
        item.request_id for item in sequential_records
    ]
    assert [
        next(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "temporal_evidence_item"
        )
        for record in parallel_records
    ] == sorted(item.item_id for item in temporal_bundle.slices)


def test_temporal_top_level_wave_spans_cases_and_folds_by_frozen_order() -> None:
    fixtures = [_case_fixture(f"temporal-cross-case-{index}") for index in range(4)]
    cases = [fixture[0] for fixture in fixtures]
    work = [(case, _git_bundle_with_items(case, 1)) for case in cases]
    settings = _settings()
    barrier = threading.Barrier(len(cases))
    completed = {case.case_id: threading.Event() for case in cases}
    successor = {
        case.case_id: cases[ordinal + 1].case_id
        for ordinal, case in enumerate(cases[:-1])
    }
    lock = threading.Lock()
    active = 0
    peak = 0
    completion_order: list[str] = []

    def delayed_response(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        nonlocal active, peak
        case_id = str(request["case_id"])
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=10)
            if case_id in successor:
                assert completed[successor[case_id]].wait(timeout=10)
            return _valid_response(request)
        finally:
            with lock:
                active -= 1
                completion_order.append(case_id)
            completed[case_id].set()

    parallel = temporal_evidence_classify_cases_bounded(
        work,
        max_batch_items=1,
        max_cells_per_request=2,
        max_total_items=1,
        max_concurrency=4,
        llm_settings=settings,
        client=_ScriptedClient(delayed_response),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    sequential = temporal_evidence_classify_cases_bounded(
        work,
        max_batch_items=1,
        max_cells_per_request=2,
        max_total_items=1,
        max_concurrency=1,
        llm_settings=settings,
        client=_ScriptedClient(  # type: ignore[arg-type]
            lambda request, _: _valid_response(request)
        ),
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert peak == 4
    assert completion_order == list(reversed([case.case_id for case in cases]))
    assert parallel == sequential
    assert [matrix.case_id for matrix, _ in parallel if matrix is not None] == [
        case.case_id for case in cases
    ]
    assert [
        next(
            ref.stable_id
            for ref in records[0].input_manifest
            if ref.kind == "candidate_case_member"
        )
        for _, records in parallel
    ] == [case.members[0].member_id for case in cases]


def test_temporal_parallel_admission_reserves_complete_bisection_trees() -> None:
    case, _, _ = _case_fixture("temporal-bisection-capacity")
    temporal_bundle = _git_bundle_with_items(case, 4)

    def run_with_call_limit(max_total_calls: int) -> int:
        settings = _settings().model_copy(
            update={
                "max_total_calls": max_total_calls,
                "max_concurrency": 2,
            }
        )
        lock = threading.Lock()
        active = 0
        peak = 0

        def delayed_response(
            request: dict[str, object], _: int
        ) -> dict[str, object]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.04)
                return _valid_response(request)
            finally:
                with lock:
                    active -= 1

        matrix, records = temporal_evidence_classify_bounded(
            case,
            temporal_bundle,
            max_batch_items=2,
            max_cells_per_request=4,
            max_total_items=4,
            max_concurrency=2,
            llm_settings=settings,
            client=_ScriptedClient(delayed_response),  # type: ignore[arg-type]
            budget=InferenceBudget(settings),
            protocol_manifest_digest=MANIFEST,
        )
        assert matrix is not None and matrix.status == "complete"
        assert len(records) == 2
        return peak

    # Each two-item top batch can expand to 2*n-1 == 3 requests.  Five call
    # slots therefore retain stable serial behavior; six admit both trees.
    assert run_with_call_limit(5) == 1
    assert run_with_call_limit(6) == 2


def test_verifier_persists_exact_temporal_span_and_rejects_digest_near_miss() -> None:
    case, assessments, current = _case_fixture("verifier-temporal-span")
    git_bundle = _git_bundle(case)
    settings = _settings()
    matrix, _ = temporal_evidence_classify(
        case,
        git_bundle,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert matrix is not None
    exact_span = matrix.relations[0].evidence_span
    assert exact_span is not None

    def cite(span: str) -> _ScriptedClient:
        def response(request: dict[str, object], _: int) -> dict[str, object]:
            value = _valid_response(request)
            current_id = request["current_evidence"]["items"][0]["evidence_id"]
            refs = sorted({current_id, span})
            for finding in value["member_findings"]:
                finding["evidence_refs"] = refs
            for relation in value["member_relations"]:
                relation["evidence_refs"] = refs
            for conclusion in value["conclusions"]:
                conclusion["evidence_refs"] = refs
            return value

        return _ScriptedClient(response)

    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        matrix,
        llm_settings=settings,
        client=cite(exact_span),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None
    assert record.terminal_status == "success"
    assert all(
        exact_span in item.evidence_refs for item in finding.member_findings
    )

    near_miss = exact_span.rsplit(":", 1)[0] + ":" + "0" * 64
    rejected, rejected_record = unified_case_verify(
        case,
        assessments,
        current,
        matrix,
        llm_settings=settings,
        client=cite(near_miss),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert rejected is None
    assert rejected_record.terminal_status == "failure"


def test_temporal_and_verifier_requests_are_isolated_by_git_cutoff_manifest() -> None:
    case, assessments, current = _case_fixture("temporal-cutoff-cache")
    first_bundle = _git_bundle(case)
    second_bundle = first_bundle.model_copy(
        update={
            "manifest": first_bundle.manifest.model_copy(
                update={
                    "manifest_id": "GITMAN-" + "7" * 24,
                    "artifact_digest": "8" * 64,
                }
            )
        }
    )
    settings = _settings()
    first_matrix, first_record = temporal_evidence_classify(
        case,
        first_bundle,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    second_matrix, second_record = temporal_evidence_classify(
        case,
        second_bundle,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert first_matrix is not None and second_matrix is not None
    assert first_record.request_id != second_record.request_id
    assert inference_cache_key_for_record(
        first_record
    ) != inference_cache_key_for_record(second_record)
    assert first_matrix.matrix_id == second_matrix.matrix_id
    assert first_matrix.artifact_digest != second_matrix.artifact_digest
    with pytest.raises(ValueError, match="replay input manifest differs"):
        replay_temporal_evidence_record(
            first_record,
            case,
            first_bundle.slices,
            git_manifest_ref(second_bundle),
        )

    _, first_verifier = unified_case_verify(
        case,
        assessments,
        current,
        first_matrix,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    _, second_verifier = unified_case_verify(
        case,
        assessments,
        current,
        second_matrix,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert first_verifier.request_id != second_verifier.request_id
    assert inference_cache_key_for_record(
        first_verifier
    ) != inference_cache_key_for_record(second_verifier)


def test_git_line_anchors_are_deterministically_segmented_to_blame_bound() -> None:
    case, _, current = _case_fixture("long-line-anchor")
    origin = current.origins[0].model_copy(
        update={"line_start": 1, "line_end": 450}
    )
    current = current.model_copy(update={"origins": [origin]})

    anchors = _git_anchors(case, current, max_blame_lines=200)
    line_anchors = sorted(
        (item for item in anchors if item.kind == "line"),
        key=lambda item: item.start_line or 0,
    )
    assert [
        (item.start_line, item.end_line) for item in line_anchors
    ] == [(1, 200), (201, 400), (401, 450)]
    assert all(
        item.end_line is not None
        and item.start_line is not None
        and item.end_line - item.start_line + 1 <= 200
        for item in line_anchors
    )
    plan_git_queries(
        "a" * 40,
        anchors,
        limits=GitLimits(max_blame_lines=200),
    )


def test_bounded_temporal_classifier_continues_after_failed_batch() -> None:
    case, _, _ = _case_fixture("temporal-partial")
    original = _git_bundle(case)
    slices = [
        original.slices[0].model_copy(
            update={
                "item_id": "GITITEM-" + str(ordinal) * 24,
                "commit_id": str(ordinal) * 40,
                "artifact_digest": str(ordinal + 3) * 64,
            }
        )
        for ordinal in (1, 2, 3)
    ]
    temporal_bundle = original.model_copy(update={"slices": slices})

    def response(request: dict[str, object], ordinal: int) -> dict[str, object]:
        value = _valid_response(request)
        if ordinal == 1:
            value["relations"] = []
        return value

    settings = _settings()
    matrix, records = temporal_evidence_classify_bounded(
        case,
        temporal_bundle,
        max_batch_items=1,
        max_cells_per_request=2,
        max_total_items=3,
        max_concurrency=1,
        llm_settings=settings,
        client=_ScriptedClient(response),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert matrix is not None
    assert matrix.status == "partial"
    assert matrix.item_ids == sorted(item.item_id for item in slices)
    assert {item.item_id for item in matrix.relations} == {
        slices[0].item_id,
        slices[2].item_id,
    }
    assert len(matrix.inference_refs) == 2
    assert len(records) == 3
    assert records[0].terminal_status == "success"
    assert records[1].terminal_status == "failure"
    assert records[2].terminal_status == "success"
    assert "temporal_batch_incomplete" in matrix.incomplete_reasons
    assert "temporal_batches_unprocessed" not in matrix.incomplete_reasons


@pytest.mark.parametrize(
    "failure_kind", ["missing", "duplicate", "foreign", "illegal"]
)
def test_temporal_classifier_rejects_bad_cells_and_foreign_span(
    failure_kind: str,
) -> None:
    case, _, _ = _case_fixture(f"temporal-{failure_kind}")
    git_bundle = _git_bundle(case)

    def bad(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        if failure_kind == "missing":
            response["relations"] = response["relations"][:-1]
        elif failure_kind == "duplicate":
            response["relations"].append(dict(response["relations"][0]))
        elif failure_kind == "foreign":
            response["relations"][0]["evidence_span"] = "foreign:span"
        else:
            response["relations"][0]["relation"] = "invented_semantics"
        return response

    settings = _settings()
    matrix, record = temporal_evidence_classify(
        case,
        git_bundle,
        llm_settings=settings,
        client=_ScriptedClient(bad),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert matrix is None
    assert record.terminal_status == "failure"


def test_verifier_projection_omits_ineligible_details_but_keeps_full_audit() -> None:
    case, assessments, original = _case_fixture("verify-projection")
    current = _with_ineligible_evidence(original)
    hidden_ids = {
        item.evidence_id for item in current.items if item.role == "ineligible"
    }

    view = build_verifier_current_evidence_view(current)
    prompt = json.loads(
        build_unified_case_verify_prompt(
            case,
            assessments,
            current,
            None,
            verification_round=0,
        )
    )
    projected = prompt["current_evidence"]
    serialized_prompt = json.dumps(prompt, sort_keys=True)
    serialized_full = current.model_dump_json()

    assert len(hidden_ids) == 24
    assert any(item.role == "ineligible" for item in current.items)
    assert current.denominators[0].ineligible == 24
    assert view == projected
    assert projected["source_bundle_id"] == current.bundle_id
    assert projected["source_bundle_digest"] == current.artifact_digest
    assert projected["denominators"][0]["ineligible"] == 24
    assert projected["omitted_detail_item_count"] == 24
    assert {item["role"] for item in projected["items"]} == {"support"}
    assert not hidden_ids & {
        item["evidence_id"] for item in projected["items"]
    }
    assert "VERIFIER_MUST_NOT_SEE_INELIGIBLE_DETAIL" in serialized_full
    assert "VERIFIER_MUST_NOT_SEE_INELIGIBLE_DETAIL" not in serialized_prompt
    assert len(serialized_prompt.encode("utf-8")) < len(
        serialized_full.encode("utf-8")
    )
    assert not hidden_ids & allowed_verifier_evidence_refs(
        assessments,
        current,
        None,
    )

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(lambda request, _: _valid_response(request)),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None
    current_input = next(
        item
        for item in record.input_manifest
        if item.kind == "current_evidence_bundle"
    )
    assert current_input.stable_id == current.bundle_id
    assert current_input.digest == current.artifact_digest
    assert replay_unified_case_verify_record(
        record,
        case,
        assessments,
        current,
        None,
        verification_round=0,
    ) == finding


def test_verifier_citation_and_refinement_closure_excludes_ineligible_ids() -> None:
    case, assessments, original = _case_fixture("verify-hidden-citation")
    current = _with_ineligible_evidence(original, count=1)
    hidden_id = next(
        item.evidence_id for item in current.items if item.role == "ineligible"
    )

    def cite_hidden(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        response["member_findings"][0]["evidence_refs"] = [hidden_id]
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(cite_hidden),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None
    assert record.terminal_status == "success"
    quarantined = finding.member_findings[0]
    assert quarantined.evidence_refs == []
    assert quarantined.reason_codes == [
        "host_quarantined_request_external_evidence"
    ]

    def refine_from_hidden(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        member = request["members"][0]
        semantics = dict(member["semantics"])
        semantics["condition"] = "when the plugin is enabled"
        response["refinement_proposal"] = {
            "parent_member_id": member["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
            "semantics": semantics,
            "evidence_refs": [hidden_id],
            "change_kinds": ["condition"],
        }
        return response

    repair_settings = _settings(schema_repairs=1)
    refined, refinement_record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=repair_settings,
        client=_ScriptedClient(refine_from_hidden),  # type: ignore[arg-type]
        budget=InferenceBudget(repair_settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert refined is not None and refined.refinement_proposals == []
    assert refinement_record.structured_response is not None
    assert refinement_record.structured_response["refinement_quarantine"][
        "reason_code"
    ] == "refinement_foreign_evidence"


def test_unified_verifier_is_blind_per_case_and_has_no_canonical_status() -> None:
    case, assessments, current = _case_fixture("verify")
    client = _ScriptedClient(lambda request, _: _valid_response(request))
    settings = _settings(schema_repairs=1)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None
    assert finding.case_id == case.case_id
    assert len(finding.member_findings) == len(case.members)
    assert finding.member_relations[0].relation == "compatible"
    assert record.stage == "unified_case_verify"

    prompt = client.calls[0]
    assert prompt["case_id"] == case.case_id
    serialized = json.dumps(prompt)
    assert "raw_channel_status" not in serialized
    assert "normalized_status" not in serialized
    assert "accepted_by_explicit_channel" not in serialized
    assert "review_by_current_channel" not in serialized
    assert "canonical_status" not in json.dumps(
        UnifiedCaseVerificationResponse.model_json_schema()
    )
    assert "epistemic_status" not in json.dumps(
        UnifiedCaseVerificationResponse.model_json_schema()
    )
    assert "An evidence path identifies provenance only" in client.system_prompts[0]
    assert "Lint, type-checking, test, coverage, CI" in client.system_prompts[0]
    schema = client.calls[0]["response_json_schema"]
    bounded_finding = schema["$defs"]["BoundedVerifierMemberFinding"]
    assert bounded_finding["properties"]["condition_support"]["enum"] == [
        "pass",
        "fail",
        "uncertain",
    ]
    allowed_refs = allowed_verifier_evidence_refs(assessments, current, None)
    assert set(
        bounded_finding["properties"]["evidence_refs"]["items"]["enum"]
    ) == allowed_refs

    other_case, other_assessments, other_current = _case_fixture("other-case")
    other_finding, _ = unified_case_verify(
        other_case,
        other_assessments,
        other_current,
        None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert other_finding is not None
    assert client.calls[1]["case_id"] == other_case.case_id
    assert client.calls[1]["case_id"] != client.calls[0]["case_id"]


@pytest.mark.parametrize(
    ("condition", "exception", "condition_support"),
    [
        ("When registering a plugin.", None, "not_applicable"),
        (None, "Except generated plugins.", "not_applicable"),
    ],
)
def test_verifier_rejects_qualified_condition_support_shape_mismatch(
    condition: str | None,
    exception: str | None,
    condition_support: str,
) -> None:
    case, assessments, current = _case_fixture(
        f"condition-shape-{condition_support}",
        condition=condition,
        exception=exception,
    )

    def mismatched(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        for member_finding in response["member_findings"]:
            member_finding["condition_support"] = condition_support
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(mismatched),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is None
    assert record.terminal_status == "failure"


def test_verifier_host_normalizes_conditionless_support_axis() -> None:
    case, assessments, current = _case_fixture("condition-shape-host-normalized")

    def mismatched(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        for member_finding in response["member_findings"]:
            member_finding["condition_support"] = "fail"
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(mismatched),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None
    assert record.terminal_status == "success"
    assert {
        item.condition_support for item in finding.member_findings
    } == {"not_applicable"}
    assert all(
        "host_normalized_condition_support_not_applicable" in item.reason_codes
        for item in finding.member_findings
    )


def test_verifier_accepts_supported_qualified_condition_shape() -> None:
    case, assessments, current = _case_fixture(
        "condition-shape-pass",
        condition="When registering a plugin.",
    )

    def supported(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        for member_finding in response["member_findings"]:
            member_finding["condition_support"] = "pass"
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(supported),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None
    assert record.terminal_status == "success"
    assert {
        item.condition_support for item in finding.member_findings
    } == {"pass"}


def test_verifier_terminally_quarantines_only_foreign_member_evidence() -> None:
    case, assessments, current = _case_fixture("foreign-member-evidence")

    def foreign_member(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        response["member_findings"][0]["evidence_refs"] = ["FOREIGN-ref"]
        return response

    settings = _settings(schema_repairs=1)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(foreign_member),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None
    assert record.terminal_status == "success"
    assert [item.repair for item in record.attempts] == [False, True]
    quarantined = finding.member_findings[0]
    sibling = finding.member_findings[1]
    assert quarantined.rule_support == "uncertain"
    assert quarantined.evidence_sufficiency == "uncertain"
    assert quarantined.evidence_refs == []
    assert quarantined.reason_codes == [
        "host_quarantined_request_external_evidence"
    ]
    assert sibling.rule_support == "pass"


@pytest.mark.parametrize(
    ("section", "field", "location", "hint"),
    [
        ("member_findings", "evidence_refs", "member_findings[member_id=", "evidence-reference"),
        ("member_findings", "closed_local_reason_codes", "member_findings[member_id=", "typed_local_reasons"),
        ("member_relations", "evidence_refs", "member_relations[0]", "evidence-reference"),
        ("conclusions", "evidence_refs", "conclusions[0]", "evidence-reference"),
    ],
)
def test_verifier_repair_locates_reference_failure_without_echoing_invalid_value(
    section: str, field: str, location: str, hint: str,
) -> None:
    case, assessments, current = _case_fixture(f"repair-{section}-{field}")
    invalid_value = "UNTRUSTED-value-that-must-not-be-reflected"

    def respond(request: dict[str, object], attempt: int) -> dict[str, object]:
        response = _valid_response(request)
        if attempt == 0:
            response[section][0][field] = [invalid_value]
        return response

    client = _ScriptedClient(respond)
    settings = _settings(schema_repairs=1)
    finding, record = unified_case_verify(
        case, assessments, current, None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None and finding.status == "complete"
    assert [attempt.repair for attempt in record.attempts] == [False, True]
    error = record.attempts[0].validation_errors[0]
    assert location in error and field in error and hint in error
    assert invalid_value not in json.dumps(client.calls[1])
    assert client.calls[1]["original_request"] == {
        key: value for key, value in client.calls[0].items()
        if key != "response_json_schema"
    }
    assert client.calls[1]["response_json_schema"] == (
        client.calls[0]["response_json_schema"]
    )
    assert "conclusions" in client.calls[1]["instruction"]
    assert "typed_local_reasons" in client.calls[1]["instruction"]
    assert all(item.rule_support == "pass" for item in finding.member_findings)
    assert replay_unified_case_verify_record(
        record, case, assessments, current, None, verification_round=0,
    ) == finding


def test_unified_verifier_canonicalizes_lists_pairs_and_replays() -> None:
    case, assessments, current = _case_fixture("verify-canonical")

    def shuffled(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        evidence_id = request["current_evidence"]["items"][0]["evidence_id"]
        response["member_findings"] = list(
            reversed(response["member_findings"])
        )
        for finding in response["member_findings"]:
            finding["evidence_refs"] = [evidence_id, evidence_id]
            finding["closed_local_reason_codes"] = [
                "local_ambiguity",
                "local_ambiguity",
            ]
            finding["reason_codes"] = ["z_reason", "a_reason", "z_reason"]
        relation = response["member_relations"][0]
        relation["left_member_id"], relation["right_member_id"] = (
            relation["right_member_id"],
            relation["left_member_id"],
        )
        relation["evidence_refs"] = [evidence_id, evidence_id]
        member_ids = request["required_member_ids"]
        response["unresolved_conflicts"] = [
            {
                "left_member_id": member_ids[1],
                "right_member_id": member_ids[0],
                "reason_code": "needs_resolution",
            }
        ]
        response["conclusions"][0]["base_member_ids"] = [
            member_ids[1],
            member_ids[0],
            member_ids[1],
        ]
        response["conclusions"][0]["evidence_refs"] = [
            evidence_id,
            evidence_id,
        ]
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(shuffled),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None and finding.status == "complete"
    assert record.protocol_version == UNIFIED_VERIFIER_PROTOCOL_VERSION
    assert UNIFIED_VERIFIER_PROTOCOL_VERSION.endswith("2.3")
    assert record.response_schema_version == UNIFIED_VERIFIER_RESPONSE_VERSION
    assert UNIFIED_VERIFIER_RESPONSE_VERSION.endswith("1.4")
    assert finding.member_relations[0].left_member_id < (
        finding.member_relations[0].right_member_id
    )
    assert finding.member_findings[0].reason_codes == [
        "a_reason",
        "host_normalized_condition_support_not_applicable",
        "z_reason",
    ]
    assert replay_unified_case_verify_record(
        record,
        case,
        assessments,
        current,
        None,
        verification_round=0,
    ) == finding


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("changes_action", "refinement_changes_core_action"),
        ("reverses_polarity", "refinement_reverses_polarity"),
        ("foreign_evidence", "refinement_foreign_evidence"),
        ("no_op", "refinement_has_no_change"),
    ],
)
def test_invalid_optional_refinement_is_quarantined_without_repair_and_cached(
    failure_kind: str,
    expected_reason: str,
) -> None:
    case, assessments, current = _case_fixture(f"refinement-{failure_kind}")

    def invalid_proposal(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        member = request["members"][0]
        semantics = dict(member["semantics"])
        current_id = request["current_evidence"]["items"][0]["evidence_id"]
        evidence_refs = [current_id]
        if failure_kind == "changes_action":
            semantics["normalized_action"] = "use a different registration action"
        elif failure_kind == "reverses_polarity":
            semantics["polarity"] = "negative"
        elif failure_kind == "foreign_evidence":
            evidence_refs = [member["support_ref_ids"][0]]
        response["refinement_proposal"] = {
            "parent_member_id": member["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
            "semantics": semantics,
            "evidence_refs": evidence_refs,
            "change_kinds": ["wording"],
        }
        return response

    settings = _settings(schema_repairs=1)
    cache = _MemoryCache()
    live_client = _ScriptedClient(invalid_proposal)
    live_budget = InferenceBudget(settings, cache=cache)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=live_client,  # type: ignore[arg-type]
        budget=live_budget,
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None and finding.status == "complete"
    assert finding.refinement_proposals == []
    assert len(live_client.calls) == 1
    assert [item.repair for item in record.attempts] == [False]
    assert record.structured_response is not None
    quarantine = record.structured_response["refinement_quarantine"]
    assert quarantine["reason_code"] == expected_reason
    assert quarantine["proposal_digest"] == content_digest(
        quarantine["proposal_json"]
    )
    replayed = replay_unified_case_verify_record(
        record,
        case,
        assessments,
        current,
        None,
        verification_round=0,
    )
    assert replayed == finding

    cached_client = _ScriptedClient(invalid_proposal)
    cached_budget = InferenceBudget(settings, cache=cache)
    cached_finding, cached_record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=cached_client,  # type: ignore[arg-type]
        budget=cached_budget,
        protocol_manifest_digest=MANIFEST,
    )
    assert cached_finding == finding
    assert cached_record == record
    assert cached_client.calls == []
    assert cached_budget.cache_hits == 1


def test_model_supplied_host_quarantine_is_ignored_and_recomputed() -> None:
    case, assessments, current = _case_fixture("refinement-echoed-quarantine")

    def invalid_proposal(
        request: dict[str, object], call_index: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        member = request["members"][0]
        response["refinement_proposal"] = {
            "parent_member_id": member["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
            "semantics": member["semantics"],
            "evidence_refs": [
                request["current_evidence"]["items"][0]["evidence_id"]
            ],
            "change_kinds": ["wording"],
        }
        del call_index
        response["refinement_quarantine"] = {
            "reason_code": "model_forged_reason",
            "proposal_digest": "0" * 64,
        }
        return response

    settings = _settings(schema_repairs=1)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(invalid_proposal),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None and finding.status == "complete"
    assert finding.refinement_proposals == []
    assert record.terminal_status == "success"
    assert record.structured_response is not None
    quarantine = record.structured_response["refinement_quarantine"]
    assert quarantine["reason_code"] == "refinement_has_no_change"
    assert quarantine["proposal_digest"] != "0" * 64


def test_optional_no_op_is_closed_before_repair_can_corrupt_valid_core() -> None:
    case, assessments, current = _case_fixture("no-op-before-core-corruption")

    def no_op_then_foreign_core(
        request: dict[str, object], call_index: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        if call_index == 0:
            member = request["members"][0]
            response["refinement_proposal"] = {
                "parent_member_id": member["member_id"],
                "git_need": {"status": "no_established_need", "questions": []},
                "semantics": member["semantics"],
                "evidence_refs": [
                    request["current_evidence"]["items"][0]["evidence_id"]
                ],
                "change_kinds": ["wording"],
            }
        else:
            response["member_findings"][0]["evidence_refs"] = [
                "EVIDENCE-foreign"
            ]
        return response

    settings = _settings(schema_repairs=1)
    client = _ScriptedClient(no_op_then_foreign_core)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None and finding.status == "complete"
    assert len(client.calls) == 1
    assert record.terminal_status == "success"
    assert record.structured_response is not None
    assert record.structured_response["refinement_quarantine"]["reason_code"] == (
        "refinement_has_no_change"
    )


def test_schema_invalid_optional_refinement_is_replayable_and_cached() -> None:
    case, assessments, current = _case_fixture("refinement-schema-invalid")

    def schema_invalid_proposal(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        response["refinement_proposal"] = {
            "parent_member_id": request["members"][0]["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
        }
        return response

    settings = _settings(schema_repairs=1)
    cache = _MemoryCache()
    live_client = _ScriptedClient(schema_invalid_proposal)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=live_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings, cache=cache),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None and finding.status == "complete"
    assert finding.refinement_proposals == []
    assert len(live_client.calls) == 1
    assert [item.repair for item in record.attempts] == [False]
    assert record.structured_response is not None
    quarantine = record.structured_response["refinement_quarantine"]
    assert quarantine["reason_code"] == "refinement_schema_invalid"
    assert quarantine["proposal_json"] == {
        "parent_member_id": case.members[0].member_id,
        "git_need": {"status": "no_established_need", "questions": []},
    }
    assert quarantine["proposal_digest"] == content_digest(
        quarantine["proposal_json"]
    )
    assert replay_unified_case_verify_record(
        record,
        case,
        assessments,
        current,
        None,
        verification_round=0,
    ) == finding

    cached_client = _ScriptedClient(schema_invalid_proposal)
    cached_budget = InferenceBudget(settings, cache=cache)
    cached_finding, cached_record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=cached_client,  # type: ignore[arg-type]
        budget=cached_budget,
        protocol_manifest_digest=MANIFEST,
    )
    assert cached_finding == finding
    assert cached_record == record
    assert cached_client.calls == []
    assert cached_budget.cache_hits == 1


def test_schema_invalid_refinement_and_foreign_core_are_both_quarantined() -> None:
    case, assessments, current = _case_fixture(
        "schema-invalid-refinement-invalid-core"
    )

    def invalid_proposal_and_core(
        request: dict[str, object], _: int
    ) -> dict[str, object]:
        response = _valid_response(request)
        response["member_findings"][0]["evidence_refs"] = [
            "EVIDENCE-foreign"
        ]
        response["refinement_proposal"] = {
            "parent_member_id": request["members"][0]["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
        }
        return response

    settings = _settings(schema_repairs=1)
    client = _ScriptedClient(invalid_proposal_and_core)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None
    assert len(client.calls) == 2
    assert [item.repair for item in record.attempts] == [False, True]
    assert record.terminal_status == "success"
    assert record.structured_response is not None
    assert record.structured_response["refinement_quarantine"][
        "reason_code"
    ] == "refinement_schema_invalid"
    assert finding.member_findings[0].reason_codes == [
        "host_quarantined_request_external_evidence"
    ]


@pytest.mark.parametrize(
    ("forgery", "expected_error"),
    [
        ("reason", "reason differs from proposal"),
        ("digest", "proposal digest differs"),
    ],
)
def test_forged_quarantine_is_rejected_from_cache_and_saved_replay(
    forgery: str,
    expected_error: str,
) -> None:
    case, assessments, current = _case_fixture(f"refinement-forged-{forgery}")
    settings = _settings()
    cache = _MemoryCache()
    valid_client = _ScriptedClient(lambda request, _: _valid_response(request))
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=valid_client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings, cache=cache),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is not None and record.structured_response is not None

    parent = case.members[0]
    proposal_json = {
        "parent_member_id": parent.member_id,
        "git_need": {"status": "no_established_need", "questions": []},
        "semantics": parent.semantics.model_dump(mode="json"),
        "evidence_refs": [current.items[0].evidence_id],
        "change_kinds": ["wording"],
    }
    forged_response = dict(record.structured_response)
    forged_response["refinement_quarantine"] = {
        "reason_code": (
            "refinement_foreign_evidence"
            if forgery == "reason"
            else "refinement_has_no_change"
        ),
        "proposal_digest": (
            content_digest(proposal_json) if forgery == "reason" else "0" * 64
        ),
        "proposal_json": proposal_json,
    }
    forged_record = _record_with_structured_response(record, forged_response)
    with pytest.raises(ValueError, match=expected_error):
        replay_unified_case_verify_record(
            forged_record,
            case,
            assessments,
            current,
            None,
            verification_round=0,
        )

    cache_key = inference_cache_key_for_record(record)
    cache.values[cache_key] = forged_record.model_dump(mode="json")
    fallback_client = _ScriptedClient(lambda request, _: _valid_response(request))
    cached_budget = InferenceBudget(settings, cache=cache)
    cached_finding, cached_record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=fallback_client,  # type: ignore[arg-type]
        budget=cached_budget,
        protocol_manifest_digest=MANIFEST,
    )
    assert cached_finding == finding
    assert cached_record != forged_record
    assert len(fallback_client.calls) == 1
    assert cached_budget.cache_hits == 0


def test_invalid_refinement_never_masks_invalid_verifier_core() -> None:
    case, assessments, current = _case_fixture("refinement-invalid-core")

    def invalid_core(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        member = request["members"][0]
        response["member_findings"] = response["member_findings"][:-1]
        response["refinement_proposal"] = {
            "parent_member_id": member["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
            "semantics": member["semantics"],
            "evidence_refs": [
                request["current_evidence"]["items"][0]["evidence_id"]
            ],
            "change_kinds": ["wording"],
        }
        return response

    settings = _settings(schema_repairs=1)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(invalid_core),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is None
    assert record.terminal_status == "failure"


def test_optional_refinement_does_not_mask_foreign_core_evidence() -> None:
    case, assessments, current = _case_fixture("refinement-foreign-core")

    def invalid_core(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        member = request["members"][0]
        response["member_findings"][0]["evidence_refs"] = [
            "EVIDENCE-foreign"
        ]
        response["refinement_proposal"] = {
            "parent_member_id": member["member_id"],
            "git_need": {"status": "no_established_need", "questions": []},
            "semantics": member["semantics"],
            "evidence_refs": [
                request["current_evidence"]["items"][0]["evidence_id"]
            ],
            "change_kinds": ["wording"],
        }
        return response

    settings = _settings(schema_repairs=1)
    client = _ScriptedClient(invalid_core)
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )

    assert finding is not None
    assert len(client.calls) == 2
    assert [item.repair for item in record.attempts] == [False, True]
    assert record.terminal_status == "success"
    assert record.structured_response is not None
    assert finding.member_findings[0].reason_codes == [
        "host_quarantined_request_external_evidence"
    ]


@pytest.mark.parametrize(
    "failure_kind", ["missing", "duplicate", "foreign", "illegal"]
)
def test_unified_verifier_rejects_bad_member_assignment_and_foreign_refs(
    failure_kind: str,
) -> None:
    case, assessments, current = _case_fixture(f"verify-{failure_kind}")

    def bad(request: dict[str, object], _: int) -> dict[str, object]:
        response = _valid_response(request)
        if failure_kind == "missing":
            response["member_findings"] = response["member_findings"][:-1]
        elif failure_kind == "duplicate":
            response["member_findings"].append(
                dict(response["member_findings"][0])
            )
        elif failure_kind == "foreign":
            response["conclusions"][0]["evidence_refs"] = ["foreign-evidence"]
        else:
            response["member_relations"][0]["relation"] = "invented_semantics"
        return response

    settings = _settings()
    finding, record = unified_case_verify(
        case,
        assessments,
        current,
        None,
        llm_settings=settings,
        client=_ScriptedClient(bad),  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest=MANIFEST,
    )
    assert finding is None
    assert record.terminal_status == "failure"
