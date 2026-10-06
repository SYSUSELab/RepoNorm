from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.models import SCHEMA_VERSION
from repo_constraint_memory_v3.unified import export as export_module
from repo_constraint_memory_v3.unified.export import (
    _BoundedTreeWriter,
    _validate_bundle_for_export,
    _write_unvalidated_outputs_for_test,
    canonical_json_text,
    path_anchor,
    write_unified_outputs,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
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
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    CanonicalClaim,
    CanonicalEvidenceRef,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    MemberDisposition,
    RefinementRecord,
    TemporalRelationMatrix,
    UNIFIED_CONTRACT_VERSION,
    UNIFIED_POLICY_VERSION,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedCompleteness,
    UnifiedIndex,
    UnifiedMemberFinding,
    UnifiedMemberRelation,
)
from repo_constraint_memory_v3.unified.pipeline import (
    CandidateGroupingExecution,
    _build_unified_funnel,
)
from repo_constraint_memory_v3.unified import publication as publication_module
from repo_constraint_memory_v3.unified.publication import (
    directory_tree_fingerprint,
    rename_directory_noreplace,
)


SNAPSHOT = "sha256:" + "1" * 64
PROFILE = "2" * 64


CLAIM_SPECS = (
    {
        "name": "explicit-default",
        "channel": "explicit",
        "rule": "Run pytest before submitting changes.",
        "scope": "**",
        "status": "accepted",
        "strength": "required",
        "delivery": "default_consider",
        "evidence_kind": "explicit",
        "path": "AGENTS.md",
    },
    {
        "name": "current-default",
        "channel": "implicit_current",
        "rule": "Use decorators for plugin registration.",
        "scope": "src/**",
        "status": "accepted",
        "strength": "observed",
        "delivery": "default_consider",
        "evidence_kind": "current_observation",
        "path": "src/plugins.py",
    },
    {
        "name": "current-reference",
        "channel": "implicit_current",
        "rule": "Prefer the legacy registration helper.",
        "scope": "src/**",
        "status": "accepted",
        "strength": "observed",
        "delivery": "reference_only",
        "evidence_kind": "current_observation",
        "path": "src/legacy.py",
    },
    {
        "name": "needs-review",
        "channel": "explicit",
        "rule": "Migrate every plugin immediately.",
        "scope": "src/**",
        "status": "review",
        "strength": "recommended",
        "delivery": "not_applicable",
        "evidence_kind": "explicit",
        "path": "CONTRIBUTING.md",
    },
    {
        "name": "rejected-rule",
        "channel": "implicit_current",
        "rule": "Never register plugins.",
        "scope": "src/**",
        "status": "rejected",
        "strength": "observed",
        "delivery": "not_applicable",
        "evidence_kind": "current_observation",
        "path": "src/plugins.py",
    },
)


def _ref(kind: str, stable_id: str, digest: str, *, path: str) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id,
        snapshot_digest=SNAPSHOT,
        artifact_path=path,
        artifact_digest=digest,
    )


def _assessment(spec: dict[str, str]) -> ChannelAssessment:
    name = spec["name"]
    channel = spec["channel"]
    assessment_id = make_assessment_id(SNAPSHOT, PROFILE, channel, name, 0)
    semantics = CandidateSemantics(
        rule=spec["rule"],
        normalized_action=f"action for {name}",
        polarity="negative" if name == "rejected-rule" else "positive",
        applies_to=[spec["scope"]],
        condition=None,
        exception=None,
    )
    support = _ref(
        "explicit_evidence" if channel == "explicit" else "population_member",
        f"SOURCE-{name}",
        content_digest(["source", name]),
        path=(
            f"explicit/{name}.json"
            if channel == "explicit"
            else f"current-source/{name}.json"
        ),
    )
    raw_status = spec["status"]
    normalized = {
        "accepted": "locally_supported",
        "review": "locally_uncertain",
        "rejected": "locally_rejected",
    }[raw_status]
    payload = {
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "assessment_id": assessment_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "channel": channel,
        "candidate_root_id": name,
        "candidate_version": 0,
        "parent_candidate_id": None,
        "semantics": semantics,
        "proposed_claim_kind": (
            "normative_declaration"
            if channel == "explicit"
            else "observed_convention"
        ),
        "proposed_strength": spec["strength"],
        "raw_channel_status": raw_status,
        "raw_policy_version": "test-channel-policy-1",
        "normalized_status": normalized,
        "axis_findings": {"evidence_sufficiency": "pass"},
        "typed_reasons": [f"local_{raw_status}"],
        "support_refs": [support],
        "counterevidence_refs": [],
        "coverage_refs": [],
        "lineage_refs": [],
        "extension_refs": [],
        "integrity": "valid",
        "source_authority": (
            "direct_normative" if channel == "explicit" else "current_population"
        ),
        "git_need": {
            "status": "not_assessed", "judgments": [], "reason": "host_fixture",
        },
        "provenance": [f"candidate:{name}"],
    }
    return ChannelAssessment.model_validate(seal_payload(payload))


def _bundle() -> UnifiedBundle:
    assessments = [_assessment(dict(spec)) for spec in CLAIM_SPECS]
    assessment_by_name = {
        assessment.candidate_root_id: assessment for assessment in assessments
    }
    members: list[CandidateCaseMember] = []
    for assessment in assessments:
        members.append(
            CandidateCaseMember(
                member_id=make_member_id(SNAPSHOT, assessment.assessment_id),
                assessment_id=assessment.assessment_id,
                snapshot_digest=SNAPSHOT,
                channel=assessment.channel,
                semantics=assessment.semantics,
                assessment_ref=_ref(
                    "channel_assessment",
                    assessment.assessment_id,
                    assessment.artifact_digest,
                    path=(
                        "canonical/audit/channel-assessments/"
                        f"{assessment.assessment_id}.json"
                    ),
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
            reason_code="export_fixture_distinct",
        )
        for left, right in combinations(members, 2)
    ]
    case_id = make_case_id(SNAPSHOT, [item.member_id for item in members])
    case = CandidateCase.model_validate(
        seal_payload(
            {
                "case_id": case_id,
                "snapshot_digest": SNAPSHOT,
                "profile_digest": PROFILE,
                "members": members,
                "relations": relations,
                "grouping_status": "complete",
                "grouping_reason_codes": [],
                "exact_identity_groups": [],
                "required_capabilities": ["current_evidence", "unified_verifier"],
            }
        )
    )

    member_by_name = {
        assessment.candidate_root_id: next(
            member
            for member in members
            if member.assessment_id == assessment.assessment_id
        )
        for assessment in assessments
    }
    origins: list[EvidenceOrigin] = []
    evidence_items: list[CurrentEvidenceItem] = []
    canonical_evidence: dict[str, CanonicalEvidenceRef] = {}
    for ordinal, spec_value in enumerate(CLAIM_SPECS, start=1):
        spec = dict(spec_value)
        name = spec["name"]
        member = member_by_name[name]
        source_digest = content_digest([spec["path"], ordinal])
        origin_kind = (
            "explicit_span"
            if spec["evidence_kind"] == "explicit"
            else "observation"
        )
        origin_id = make_evidence_origin_id(
            SNAPSHOT,
            origin_kind,
            spec["path"],
            [ordinal, ordinal],
            source_digest,
        )
        origins.append(
            EvidenceOrigin(
                origin_id=origin_id,
                origin_kind=origin_kind,
                path=spec["path"],
                line_start=ordinal,
                line_end=ordinal,
                source_content_digest=source_digest,
                independence_group=f"file:{spec['path']}",
            )
        )
        source_ref = assessment_by_name[name].support_refs[0]
        evidence_id = make_evidence_id(
            origin_id,
            "support",
            [member.member_id],
            source_ref.stable_id,
        )
        description = f"Typed evidence for {name}."
        evidence_items.append(
            CurrentEvidenceItem(
                evidence_id=evidence_id,
                origin_id=origin_id,
                role="support",
                member_ids=[member.member_id],
                artifact_ref=source_ref,
                excerpt_digest=(
                    hashlib.sha256(description.encode("utf-8")).hexdigest()
                    if spec["evidence_kind"] == "explicit"
                    else source_digest
                ),
                description=description,
            )
        )
        canonical_evidence[name] = CanonicalEvidenceRef(
            evidence_id=evidence_id,
            origin_id=origin_id,
            role="support",
            kind=spec["evidence_kind"],
            path=spec["path"],
            line_start=ordinal,
            line_end=ordinal,
            artifact_ref=source_ref,
        )
    current = CurrentEvidenceBundle.model_validate(
        seal_payload(
            {
                "bundle_id": make_current_evidence_bundle_id(
                    SNAPSHOT, PROFILE, case_id
                ),
                "case_id": case_id,
                "snapshot_digest": SNAPSHOT,
                "profile_digest": PROFILE,
                "origins": sorted(origins, key=lambda item: item.origin_id),
                "items": sorted(evidence_items, key=lambda item: item.evidence_id),
                "denominators": [],
                "population_refs": [],
                "falsification_refs": [],
                "scope_condition_refs": [],
                "dependencies": [
                    CapabilityDependency(
                        capability="current_evidence",
                        requirement_mode="required",
                        status="complete",
                        reason_codes=[],
                    )
                ],
                "integrity": "valid",
                "status": "complete",
                "incomplete_reasons": [],
            }
        )
    )

    finding = UnifiedCaseFinding.model_validate(
        seal_payload(
            {
                "finding_id": make_finding_id(SNAPSHOT, PROFILE, case_id, 0),
                "case_id": case_id,
                "snapshot_digest": SNAPSHOT,
                "profile_digest": PROFILE,
                "verification_round": 0,
                "member_findings": [
                    UnifiedMemberFinding(
                        member_id=member.member_id,
                        norm_relevance="pass",
                        rule_support="pass",
                        scope_support="pass",
                        condition_support="not_applicable",
                        temporal_status="not_required",
                        evidence_sufficiency="pass",
                        evidence_refs=[
                            canonical_evidence[
                                next(
                                    assessment.candidate_root_id
                                    for assessment in assessments
                                    if assessment.assessment_id == member.assessment_id
                                )
                            ].evidence_id
                        ],
                        closed_local_reason_codes=[],
                        reason_codes=["export_fixture"],
                    )
                    for member in members
                ],
                "member_relations": [
                    UnifiedMemberRelation(
                        left_member_id=left.member_id,
                        right_member_id=right.member_id,
                        relation="distinct",
                        evidence_refs=[],
                        reason_code="export_fixture_distinct",
                    )
                    for left, right in combinations(members, 2)
                ],
                "unresolved_conflicts": [],
                "conclusions": [],
                "refinement_proposals": [],
                "inference_ref": _ref(
                    "inference_request",
                    "INFER-" + "a" * 24,
                    "a" * 64,
                    path="inference/requests/INFER-" + "a" * 24 + ".json",
                ),
                "status": "complete",
                "failure_reasons": [],
            }
        )
    )

    matrix = TemporalRelationMatrix.model_validate(
        seal_payload(
            {
                "matrix_id": make_temporal_matrix_id(
                    SNAPSHOT,
                    PROFILE,
                    case_id,
                    [],
                    [item.member_id for item in members],
                ),
                "case_id": case_id,
                "snapshot_digest": SNAPSHOT,
                "profile_digest": PROFILE,
                "git_manifest_ref": None,
                "item_ids": [],
                "member_ids": [item.member_id for item in members],
                "relations": [],
                "status": "not_required",
                "incomplete_reasons": [],
                "inference_refs": [],
            }
        )
    )

    refinement = RefinementRecord.model_validate(
        seal_payload(
            {
                "refinement_id": make_refinement_id(
                    SNAPSHOT,
                    case_id,
                    members[0].member_id,
                    [members[1].member_id],
                ),
                "case_id": case_id,
                "snapshot_digest": SNAPSHOT,
                "parent_member_id": members[0].member_id,
                "child_member_ids": [members[1].member_id],
                "round": 1,
                "change_kinds": ["wording"],
                "reacquired_current_bundle_ids": [current.bundle_id],
                "git_bundle_refs": [],
                "finding_ids": [finding.finding_id],
                "status": "complete",
                "reason_codes": ["export_fixture_refinement"],
            }
        )
    )

    claims: list[CanonicalClaim] = []
    dispositions: list[MemberDisposition] = []
    for spec_value in CLAIM_SPECS:
        spec = dict(spec_value)
        assessment = assessment_by_name[spec["name"]]
        member = member_by_name[spec["name"]]
        claim_id = make_canonical_claim_id(
            SNAPSHOT, assessment.semantics.model_dump(mode="json")
        )
        status = spec["status"]
        decision_evidence = canonical_evidence[spec["name"]]
        counterevidence = (
            [decision_evidence.model_copy(update={"role": "counterevidence"})]
            if status == "rejected"
            else []
        )
        claim = CanonicalClaim.model_validate(
            seal_payload(
                {
                    "claim_id": claim_id,
                    "snapshot_digest": SNAPSHOT,
                    "profile_digest": PROFILE,
                    "case_id": case_id,
                    "semantics": assessment.semantics,
                    "epistemic_status": status,
                    "claim_kind": (
                        "normative_declaration"
                        if spec["channel"] == "explicit"
                        else "observed_convention"
                    ),
                    "normative_strength": spec["strength"],
                    "delivery_policy": spec["delivery"],
                    "execution_risk": "read_only",
                    "support": [decision_evidence] if status != "rejected" else [],
                    "counterevidence": counterevidence,
                    "conflicts_with": [],
                    "contributing_member_ids": [member.member_id],
                    "refinement_parent_id": None,
                    "reason_codes": [f"canonical_{status}"],
                    "review_facets": (
                        ["factual_uncertainty"] if status == "review" else []
                    ),
                    "rejection_kind": (
                        "current_snapshot_contradiction"
                        if status == "rejected"
                        else None
                    ),
                    "rejection_basis_refs": (
                        [decision_evidence.evidence_id]
                        if status == "rejected"
                        else []
                    ),
                    "dependency_status": "complete",
                }
            )
        )
        claims.append(claim)
        disposition = {
            "accepted": "adopted_as_support",
            "review": "review",
            "rejected": "rejected",
        }[spec["status"]]
        dispositions.append(
            MemberDisposition(
                member_id=member.member_id,
                case_id=case_id,
                disposition=disposition,
                canonical_claim_ids=[claim_id],
                reason_codes=[f"final_{spec['status']}"],
            )
        )
    claims.sort(key=lambda item: item.claim_id)
    dispositions.sort(key=lambda item: item.member_id)
    funnel = _build_unified_funnel(
        assessments=assessments,
        claims=claims,
        dispositions=dispositions,
        grouping_execution=CandidateGroupingExecution(
            mode="semantic",
            status="complete",
            reason=None,
            planned_pairs=0,
            planned_calls=0,
            terminal_calls=0,
            case_count=1,
        ),
        raw_candidate_events_by_channel={
            "explicit": 2,
            "implicit_current": 3,
        },
    )
    completeness = UnifiedCompleteness(
        structurally_valid=True,
        package_status="complete",
        capabilities=[
            CapabilityDependency(
                capability="unified_export",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            )
        ],
        omitted_possible=False,
        reasons=[],
    )
    bundle_id = make_bundle_id(SNAPSHOT, PROFILE)
    claim_ids = sorted(item.claim_id for item in claims)
    index = UnifiedIndex.model_validate(
        seal_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_version": UNIFIED_CONTRACT_VERSION,
                "policy_version": UNIFIED_POLICY_VERSION,
                "bundle_id": bundle_id,
                "snapshot_digest": SNAPSHOT,
                "snapshot_commit": "a" * 40,
                "profile_digest": PROFILE,
                "channel_assessment_ids": sorted(
                    item.assessment_id for item in assessments
                ),
                "case_ids": [case_id],
                "finding_ids": [finding.finding_id],
                "canonical_claim_ids": claim_ids,
                "accepted_count": 3,
                "review_count": 1,
                "rejected_count": 1,
                "artifact_set_digest": content_digest(
                    [case.artifact_digest, current.artifact_digest]
                ),
            }
        )
    )
    return UnifiedBundle.model_validate(
        seal_payload(
            {
                "index": index,
                "assessments": assessments,
                "cases": [case],
                "current_evidence": [current],
                "git_bundle_refs": [],
                "temporal_matrices": [matrix],
                "findings": [finding],
                "refinements": [refinement],
                "canonical_claims": claims,
                "member_dispositions": dispositions,
                "funnel": funnel,
                "completeness": completeness,
            }
        )
    )


def _with_git_payloads(
    bundle: UnifiedBundle,
) -> tuple[UnifiedBundle, dict[str, object]]:
    manifest_id = "GITMAN-" + "a" * 24
    slice_id = "GITITEM-" + "b" * 24
    manifest_payload: dict[str, object] = {
        "manifest_id": manifest_id,
        "case_id": bundle.cases[0].case_id,
        "snapshot_digest": SNAPSHOT,
        "status": "complete",
    }
    manifest_payload["artifact_digest"] = artifact_digest(manifest_payload)
    slice_payload: dict[str, object] = {
        "item_id": slice_id,
        "case_id": bundle.cases[0].case_id,
        "snapshot_digest": SNAPSHOT,
        "commit_id": "c" * 40,
        "parent_id": "d" * 40,
        "patch": "@@ -1 +1 @@\n-old\n+new\n",
    }
    slice_payload["artifact_digest"] = artifact_digest(slice_payload)
    refs = [
        _ref(
            "git_manifest",
            manifest_id,
            str(manifest_payload["artifact_digest"]),
            path=f"canonical/audit/git/manifests/{manifest_id}.json",
        ),
        _ref(
            "git_change_slice",
            slice_id,
            str(slice_payload["artifact_digest"]),
            path=f"canonical/audit/git/slices/{slice_id}.json",
        ),
    ]
    payload = bundle.model_dump(mode="python", exclude={"artifact_digest"})
    payload["git_bundle_refs"] = refs
    return UnifiedBundle.model_validate(seal_payload(payload)), {
        manifest_id: manifest_payload,
        slice_id: slice_payload,
    }


def _claim_by_rule(bundle: UnifiedBundle, rule: str) -> CanonicalClaim:
    return next(item for item in bundle.canonical_claims if item.semantics.rule == rule)


def _replace_canonical_claim(
    bundle: UnifiedBundle,
    original: CanonicalClaim,
    replacement: CanonicalClaim,
) -> UnifiedBundle:
    claims = sorted(
        [
            replacement if item.claim_id == original.claim_id else item
            for item in bundle.canonical_claims
        ],
        key=lambda item: item.claim_id,
    )
    dispositions = []
    for item in bundle.member_dispositions:
        payload = item.model_dump(mode="python")
        payload["canonical_claim_ids"] = sorted(
            replacement.claim_id if value == original.claim_id else value
            for value in item.canonical_claim_ids
        )
        dispositions.append(MemberDisposition.model_validate(payload))
    index_payload = bundle.index.model_dump(
        mode="python",
        exclude={"artifact_digest"},
    )
    index_payload["canonical_claim_ids"] = sorted(
        item.claim_id for item in claims
    )
    index = UnifiedIndex.model_validate(seal_payload(index_payload))
    payload = bundle.model_dump(mode="python", exclude={"artifact_digest"})
    payload.update(
        {
            "index": index,
            "canonical_claims": claims,
            "member_dispositions": sorted(
                dispositions,
                key=lambda item: item.member_id,
            ),
        }
    )
    return UnifiedBundle.model_validate(seal_payload(payload))


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_text({"unsafe": float("nan")})


def test_writer_emits_exact_audit_and_delivery_safe_consumer_tree(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    output = tmp_path / "unified-run"
    paths = _write_unvalidated_outputs_for_test(
        output,
        bundle=bundle,
        max_artifact_bytes=10_000_000,
    )

    expected = {
        "canonical/claims.json",
        "canonical/completeness.json",
        "canonical/delivery_report.json",
        "canonical/audit/index.json",
        "canonical/audit/member-dispositions.json",
        "canonical/audit/funnel.json",
        "canonical/audit/git/refs.json",
        "consumers/accepted_only/START.md",
        "consumers/accepted_only/catalog.md",
        "consumers/accepted_only/by-path/index.md",
        "consumers/accepted_only/by-path/constraints.md",
        "consumers/accepted_only/by-path/src/constraints.md",
    }
    assert expected <= set(paths)
    audit = output / "canonical" / "audit"
    assert len(list((audit / "channel-assessments").glob("*.json"))) == 5
    assert len(list((audit / "cases").glob("*.json"))) == 1
    assert len(list((audit / "current-evidence").glob("*.json"))) == 1
    assert len(list((audit / "git" / "relations").glob("*.json"))) == 1
    assert len(list((audit / "findings").glob("*.json"))) == 1
    assert len(list((audit / "refinements").glob("*.json"))) == 1
    assert not (output / "unified").exists()
    assert not (output / "memory.json").exists()
    assert not (output / "review.json").exists()
    assert not (output / "by-trigger").exists()

    canonical = json.loads(
        (output / "canonical" / "claims.json").read_text(encoding="utf-8")
    )
    assert {item["epistemic_status"] for item in canonical["claims"]} == {
        "accepted",
        "review",
        "rejected",
    }
    assert len(canonical["claims"]) == 5
    evidence_kinds = {
        evidence["kind"]
        for claim in canonical["claims"]
        for evidence in claim["support"]
    }
    assert evidence_kinds == {"explicit", "current_observation"}

    report = json.loads(
        (output / "canonical" / "delivery_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["delivered_claim_count"] == 2
    assert report["accepted_claim_count"] == 3
    assert report["review_claim_count"] == 1
    assert report["rejected_claim_count"] == 1
    assert report["emitted_consumer_profiles"] == ["accepted_only"]
    reference = _claim_by_rule(bundle, "Prefer the legacy registration helper.")
    review_claim = _claim_by_rule(bundle, "Migrate every plugin immediately.")
    rejected = _claim_by_rule(bundle, "Never register plugins.")
    assert reference.claim_id not in report["delivered_claim_ids"]
    assert review_claim.claim_id not in report["delivered_claim_ids"]
    assert rejected.claim_id not in report["delivered_claim_ids"]

    consumer = output / "consumers" / "accepted_only"
    start = (consumer / "START.md").read_text(encoding="utf-8")
    catalog = (consumer / "catalog.md").read_text(encoding="utf-8")
    root_view = (consumer / "by-path" / "constraints.md").read_text(
        encoding="utf-8"
    )
    source_view = (consumer / "by-path" / "src" / "constraints.md").read_text(
        encoding="utf-8"
    )
    assert "accepted_only" not in start
    manifest = json.loads((consumer / "consumer-manifest.json").read_text(encoding="utf-8"))
    assert manifest["consumer_profile"] == "accepted_only"
    assert "Run pytest before submitting changes" in catalog
    assert reference.semantics.rule.rstrip(".") in catalog
    assert review_claim.semantics.rule.rstrip(".") not in catalog
    assert rejected.semantics.rule.rstrip(".") not in catalog
    assert "Run pytest before submitting changes" in root_view
    assert "Use decorators for plugin registration" in source_view
    for forbidden in (
        reference.semantics.rule,
        review_claim.semantics.rule,
        rejected.semantics.rule,
    ):
        assert forbidden.rstrip(".") not in root_view
        assert forbidden.rstrip(".") not in source_view
    accepted_ids = {
        item.claim_id
        for item in bundle.canonical_claims
        if item.epistemic_status == "accepted"
    }
    assert {
        path.stem for path in (consumer / "claims").glob("*.md")
    } == accepted_ids


def test_streamed_json_writer_is_byte_identical_and_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {
        "z-last": ["plain", "中文", "emoji-😀", {"line": "a\nb"}],
        "a-first": {"truth": True, "nothing": None, "number": 7},
    }
    expected = canonical_json_text(value).encode("utf-8")

    def whole_text_is_forbidden(value: object) -> str:
        del value
        raise AssertionError("streamed JSON writer materialized whole text")

    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.export.canonical_json_text",
        whole_text_is_forbidden,
    )
    root = tmp_path / "streamed"
    writer = _BoundedTreeWriter(root, max_bytes=len(expected))
    path = writer.write_json("nested/value.json", value)
    assert path.read_bytes() == expected
    assert writer.total_bytes == len(expected)
    assert writer.relative_paths == ["nested/value.json"]

    limited_root = tmp_path / "limited"
    limited = _BoundedTreeWriter(limited_root, max_bytes=len(expected) - 1)
    with pytest.raises(ValueError, match="max_unified_artifact_bytes"):
        limited.write_json("value.json", value)
    assert not (limited_root / "value.json").exists()
    assert limited.total_bytes == 0
    assert limited.relative_paths == []


def test_writer_revalidates_nested_artifacts_without_trusting_instances(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    assessment = bundle.assessments[0]
    assessment.typed_reasons = ["z-reason", "a-reason"]
    assessment.artifact_digest = artifact_digest(assessment)
    bundle.artifact_digest = artifact_digest(bundle)

    output = tmp_path / "mutated-nested"
    with pytest.raises(ValueError, match="assessment reasons must be sorted"):
        _write_unvalidated_outputs_for_test(
            output,
            bundle=bundle,
            max_artifact_bytes=10_000_000,
        )
    assert not output.exists()


def test_production_writer_requires_repository_bound_final_validation(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="repository_root"):
        write_unified_outputs(  # type: ignore[call-arg]
            tmp_path / "must-not-publish",
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
        )
    assert not (tmp_path / "must-not-publish").exists()


def test_atomic_directory_publication_never_replaces_a_racing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload.txt").write_text("payload\n", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="destination appeared"):
        rename_directory_noreplace(source, destination)

    assert (source / "payload.txt").read_text(encoding="utf-8") == "payload\n"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []

    second_source = tmp_path / "second-staging"
    second_source.mkdir()
    (second_source / "payload.txt").write_text("second\n", encoding="utf-8")
    symlink_destination = tmp_path / "symlink-destination"
    symlink_destination.symlink_to(destination, target_is_directory=True)

    with pytest.raises(FileExistsError, match="destination appeared"):
        rename_directory_noreplace(second_source, symlink_destination)

    assert second_source.is_dir()
    assert symlink_destination.is_symlink()
    assert symlink_destination.resolve() == destination.resolve()


def test_writer_persists_digest_bound_git_payload_mapping(tmp_path: Path) -> None:
    bundle, git_payloads = _with_git_payloads(_bundle())
    output = tmp_path / "with-git"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=bundle,
        git_artifacts=git_payloads,
        max_artifact_bytes=10_000_000,
    )
    manifest_id = next(
        item.stable_id for item in bundle.git_bundle_refs if item.kind == "git_manifest"
    )
    slice_id = next(
        item.stable_id
        for item in bundle.git_bundle_refs
        if item.kind == "git_change_slice"
    )
    assert (
        output / "canonical" / "audit" / "git" / "manifests" / f"{manifest_id}.json"
    ).is_file()
    assert (
        output / "canonical" / "audit" / "git" / "slices" / f"{slice_id}.json"
    ).is_file()

    incomplete_output = tmp_path / "missing-git"
    with pytest.raises(ValueError, match="mapping differs"):
        _write_unvalidated_outputs_for_test(
            incomplete_output,
            bundle=bundle,
            git_artifacts={manifest_id: git_payloads[manifest_id]},
            max_artifact_bytes=10_000_000,
        )
    assert not incomplete_output.exists()


def test_writer_binds_consumer_metadata_into_persisted_and_caller_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report-bound"
    build_report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "sentinel": "preserved",
    }

    _write_unvalidated_outputs_for_test(
        output,
        bundle=_bundle(),
        max_artifact_bytes=10_000_000,
        build_report=build_report,
    )

    persisted = json.loads(
        (output / "build_report.json").read_text(encoding="utf-8")
    )
    assert persisted == build_report
    consumer_surface = persisted["consumer_surface"]
    assert isinstance(consumer_surface, dict)
    assert consumer_surface["surface_version"] == "static-consumer-surface-3.0"
    assert consumer_surface["manifest_version"] == "static-consumer-manifest-1.0"
    assert consumer_surface["requested_profiles"] == ["accepted_only"]
    assert consumer_surface["emitted_profiles"] == ["accepted_only"]
    manifest = json.loads(
        (
            output
            / "consumers"
            / "accepted_only"
            / "consumer-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert consumer_surface["manifest_digests"] == {
        "accepted_only": manifest["artifact_digest"]
    }


def test_writer_emits_default_and_explicit_experimental_profiles(
    tmp_path: Path,
) -> None:
    output = tmp_path / "two-profiles"
    build_report: dict[str, object] = {}

    _write_unvalidated_outputs_for_test(
        output,
        bundle=_bundle(),
        max_artifact_bytes=10_000_000,
        consumer_profiles=("accepted_plus_review",),
        build_report=build_report,
    )

    assert (output / "consumers" / "accepted_only" / "START.md").is_file()
    experimental = output / "consumers" / "accepted_plus_review"
    assert (experimental / "START.md").is_file()
    report = json.loads(
        (output / "canonical" / "delivery_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["requested_consumer_profiles"] == [
        "accepted_only",
        "accepted_plus_review",
    ]
    assert report["emitted_consumer_profiles"] == [
        "accepted_only",
        "accepted_plus_review",
    ]
    experimental_catalog = (experimental / "catalog.md").read_text(
        encoding="utf-8"
    )
    assert "Migrate every plugin immediately" in experimental_catalog
    assert "Never register plugins" not in experimental_catalog


def test_model_rejects_path_traversal_before_publication(tmp_path: Path) -> None:
    bundle = _bundle()
    claim = bundle.canonical_claims[0]
    support = claim.support[0].model_dump(mode="python")
    support["path"] = "../../outside.py"
    claim_payload = claim.model_dump(mode="python", exclude={"artifact_digest"})
    claim_payload["support"] = [support]
    output = tmp_path / "unsafe"
    with pytest.raises(ValidationError, match="canonical evidence path"):
        CanonicalClaim.model_validate(seal_payload(claim_payload))
    assert not output.exists()
    with pytest.raises(ValueError, match="claim scope"):
        path_anchor("../outside/**")


def test_export_preflight_accepts_review_directory_marker_and_long_provenance_path(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    review = _claim_by_rule(bundle, "Migrate every plugin immediately.")
    semantics = review.semantics.model_copy(update={"applies_to": ["tests/"]})
    review_payload = review.model_dump(mode="python", exclude={"artifact_digest"})
    review_payload.update(
        {
            "claim_id": make_canonical_claim_id(
                SNAPSHOT,
                semantics.model_dump(mode="json"),
            ),
            "semantics": semantics,
            "review_facets": sorted(
                {*review.review_facets, "refinement_or_qualifier_needed"}
            ),
        }
    )
    replacement = CanonicalClaim.model_validate(seal_payload(review_payload))
    bundle = _replace_canonical_claim(bundle, review, replacement)

    output = tmp_path / "review-directory-marker"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=bundle,
        consumer_profiles=("accepted_only", "accepted_plus_review"),
        max_artifact_bytes=20_000_000,
    )

    assert (
        output
        / "consumers"
        / "accepted_plus_review"
        / "by-path"
        / "tests"
        / "constraints.md"
    ).is_file()

    accepted = next(
        item
        for item in bundle.canonical_claims
        if item.epistemic_status == "accepted" and item.support
    )
    support_payload = accepted.support[0].model_dump(mode="python")
    support_payload["path"] = "/".join(
        [f"segment-{ordinal:02d}-" + "x" * 200 for ordinal in range(11)]
        + ["witness.py"]
    )
    claim_payload = accepted.model_dump(mode="python", exclude={"artifact_digest"})
    claim_payload["support"] = [
        CanonicalEvidenceRef.model_validate(support_payload)
    ]
    accepted_replacement = CanonicalClaim.model_validate(
        seal_payload(claim_payload)
    )
    bundle = _replace_canonical_claim(
        bundle,
        accepted,
        accepted_replacement,
    )
    assert _validate_bundle_for_export(bundle) == bundle


def test_writer_is_exclusive_and_preserves_first_publication(tmp_path: Path) -> None:
    bundle = _bundle()
    output = tmp_path / "exclusive"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=bundle,
        max_artifact_bytes=10_000_000,
    )
    first_memory = (output / "canonical" / "claims.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        _write_unvalidated_outputs_for_test(
            output,
            bundle=bundle,
            max_artifact_bytes=10_000_000,
        )
    assert (output / "canonical" / "claims.json").read_bytes() == first_memory

    preexisting = tmp_path / "preexisting"
    preexisting.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _write_unvalidated_outputs_for_test(
            preexisting,
            bundle=bundle,
            max_artifact_bytes=10_000_000,
        )


def test_writer_enforces_total_artifact_byte_bound_without_partial_tree(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded"
    with pytest.raises(ValueError, match="max_unified_artifact_bytes"):
        _write_unvalidated_outputs_for_test(
            output,
            bundle=_bundle(),
            max_artifact_bytes=1_000,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bounded.staging-*"))
    assert not (tmp_path / ".bounded.publish.lock").exists()


def test_writer_publishes_without_complete_stage_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3.unified import validation

    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "prepublish-invalid"
    def forbidden_replay(*_: object, **__: object) -> list[str]:
        raise AssertionError("build publication must not invoke offline replay")

    monkeypatch.setattr(validation, "validate_unified_directory", forbidden_replay)
    write_unified_outputs(
        output,
        bundle=_bundle(),
        max_artifact_bytes=10_000_000,
        repository_name=repository.name,
        repository_root=repository,
    )

    assert (output / "canonical" / "claims.json").is_file()
    assert not list(tmp_path.glob(".prepublish-invalid.staging-*"))


def test_writer_rejects_staging_mutation_after_initial_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "prepublish-mutated"
    real_fingerprint = export_module.directory_tree_fingerprint
    call_count = 0

    def fingerprint_then_mutate(run_directory: Path, **kwargs: object) -> str:
        nonlocal call_count
        result = real_fingerprint(run_directory, **kwargs)
        call_count += 1
        if call_count == 1:
            start = run_directory / "consumers" / "accepted_only" / "START.md"
            start.chmod(0o600)
            start.write_bytes(start.read_bytes() + b"\nmutated after fingerprint\n")
        return result

    monkeypatch.setattr(
        export_module,
        "directory_tree_fingerprint",
        fingerprint_then_mutate,
    )
    with pytest.raises(ValueError, match="staging tree changed"):
        write_unified_outputs(
            output,
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".prepublish-mutated.staging-*"))


def test_writer_runs_external_identity_check_immediately_before_publish(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "prepublication-check-failed"
    called = 0

    def reject_changed_input() -> None:
        nonlocal called
        called += 1
        raise ValueError("repository snapshot changed before publication")

    with pytest.raises(ValueError, match="repository snapshot changed"):
        write_unified_outputs(
            output,
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
            _prepublication_check=reject_changed_input,
        )

    assert called == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".prepublication-check-failed.staging-*"))
    assert not (tmp_path / ".prepublication-check-failed.publish.lock").exists()


def test_production_writer_rejects_repository_and_upstream_overlap(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="target repository"):
        write_unified_outputs(
            repository / "generated-run",
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
        )

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    with pytest.raises(ValueError, match="upstream run"):
        write_unified_outputs(
            upstream / "generated-run",
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
            upstream_directory=upstream,
        )
    assert not (tmp_path / ".prepublish-invalid.publish.lock").exists()


def test_tree_fingerprint_enforces_incremental_entry_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fingerprint-bound"
    root.mkdir()
    for ordinal in range(3):
        (root / f"{ordinal}.txt").write_text("bounded\n", encoding="utf-8")
    monkeypatch.setattr(publication_module, "_MAX_FINGERPRINT_FILES", 2)

    with pytest.raises(ValueError, match="entry-count bound"):
        directory_tree_fingerprint(root, max_bytes=1_000)


def test_upstream_copy_enforces_incremental_entry_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = tmp_path / "bounded-upstream"
    upstream.mkdir()
    for ordinal in range(3):
        (upstream / f"{ordinal}.txt").write_text("bounded\n", encoding="utf-8")
    monkeypatch.setattr(publication_module, "_MAX_FINGERPRINT_FILES", 2)
    repository = tmp_path / "bounded-upstream-repository"
    repository.mkdir()

    with pytest.raises(ValueError, match="upstream tree exceeds entry-count bound"):
        write_unified_outputs(
            tmp_path / "must-not-publish",
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
            upstream_directory=upstream,
        )
    assert not (tmp_path / "must-not-publish").exists()


def test_direct_channel_staging_matches_upstream_import_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    source = _BoundedTreeWriter(upstream, max_bytes=1_000_000)

    def write_channels(writer: _BoundedTreeWriter) -> None:
        writer.write_json("observations/index.json", {"unicode": "中文😀", "rows": [1, 2]})
        writer.write_bytes("agent/stdout.txt", b"first\r\nsecond\n")
        writer.directory("observations/semantic-units/units")

    write_channels(source)
    copied = tmp_path / "copied"
    write_unified_outputs(
        copied, bundle=_bundle(), max_artifact_bytes=10_000_000,
        repository_root=repository, upstream_directory=upstream,
    )
    direct = tmp_path / "direct"
    original_copy = export_module._copy_upstream_tree

    def require_no_import(writer, upstream_directory, **kwargs):
        assert upstream_directory is None
        return original_copy(writer, upstream_directory, **kwargs)

    monkeypatch.setattr(export_module, "_copy_upstream_tree", require_no_import)
    paths = write_unified_outputs(
        direct, bundle=_bundle(), max_artifact_bytes=10_000_000,
        repository_root=repository, channel_writer=write_channels,
    )
    assert directory_tree_fingerprint(direct, max_bytes=10_000_000) == (
        directory_tree_fingerprint(copied, max_bytes=10_000_000)
    )
    for relative in paths:
        assert (direct / relative).read_bytes() == (copied / relative).read_bytes()
    assert (direct / "agent/stdout.txt").read_bytes() == b"first\r\nsecond\n"


def test_direct_channel_staging_is_bounded_and_exclusive(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="max_unified_artifact_bytes"):
        write_unified_outputs(
            destination, bundle=_bundle(), max_artifact_bytes=10,
            repository_root=repository,
            channel_writer=lambda writer: writer.write_json("channel.json", {"x": "large"}),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".run.staging-*"))
    with pytest.raises(ValueError, match="exclusive"):
        write_unified_outputs(
            destination, bundle=_bundle(), max_artifact_bytes=10_000_000,
            repository_root=repository, upstream_directory=tmp_path / "upstream",
            channel_writer=lambda writer: None,
        )
    assert not destination.exists()


def test_upstream_copy_streams_large_utf8_file_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-upstream.jsonl"
    data = ("{\"value\":\"" + "x" * 1_100_000 + "\"}\n") * 2
    source.write_text(data, encoding="utf-8")
    staging = tmp_path / "copy-staging"
    staging.mkdir()
    writer = _BoundedTreeWriter(staging, max_bytes=3_000_000)
    before = source.lstat()
    real_read = export_module.os.read
    requested: list[int] = []

    def bounded_read(descriptor: int, size: int) -> bytes:
        requested.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(export_module.os, "read", bounded_read)
    copied = writer.copy_utf8_file("copied.jsonl", source, before)

    assert copied.read_bytes() == source.read_bytes()
    assert requested and max(requested) <= 1_048_576


def test_staging_defers_sync_until_all_file_types_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    source = tmp_path / "source.json"
    source.write_text('{"source":"中文"}\n')
    writer = _BoundedTreeWriter(tmp_path / "stage", max_bytes=10000)
    synced = []

    def sync(fd: int) -> None:
        synced.append(os.fstat(fd).st_ino)
        assert len(writer.relative_paths) == 3

    monkeypatch.setattr(export_module.os, "fsync", sync)
    text = writer.write_text("text.md", "内容\n", read_only=True)
    encoded = writer.write_json("nested/value.json", {"key": [1, "😀"]})
    copied = writer.copy_utf8_file("copied.json", source, source.stat())
    assert synced == []
    writer.sync_files()
    assert set(synced) == {p.stat().st_ino for p in (text, encoded, copied)}
    assert len(synced) == 3
    assert text.read_text() == "内容\n"
    assert encoded.read_text() == canonical_json_text({"key": [1, "😀"]})
    assert copied.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("replacement", ["symlink", "directory", "fifo"])
def test_staging_sync_rejects_nonregular_replacement(
    tmp_path: Path, replacement: str,
) -> None:
    import os

    writer = _BoundedTreeWriter(tmp_path / "stage", max_bytes=10000)
    path = writer.write_text("value", "content")
    path.unlink()
    if replacement == "symlink":
        source = tmp_path / "source"
        source.write_text("outside")
        path.symlink_to(source)
    elif replacement == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError, match="not a regular file"):
        writer.sync_files()


def test_publication_sync_precedes_fingerprints_and_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original_sync = _BoundedTreeWriter.sync_files
    original_fingerprint = export_module.directory_tree_fingerprint
    original_rename = export_module.rename_directory_noreplace

    def sync(writer: _BoundedTreeWriter) -> None:
        calls.append("sync")
        original_sync(writer)

    def fingerprint(*args, **kwargs):
        calls.append("fingerprint")
        return original_fingerprint(*args, **kwargs)

    def rename(*args, **kwargs):
        calls.append("rename")
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(_BoundedTreeWriter, "sync_files", sync)
    monkeypatch.setattr(export_module, "directory_tree_fingerprint", fingerprint)
    monkeypatch.setattr(export_module, "rename_directory_noreplace", rename)
    progress = []
    _write_unvalidated_outputs_for_test(
        tmp_path / "output", bundle=_bundle(), max_artifact_bytes=10_000_000,
        progress_callback=progress.append,
    )
    assert calls == ["sync", "fingerprint", "fingerprint", "rename"]
    assert [message.split()[1] for message in progress] == [
        "phase=prepare", "phase=write", "phase=sync", "phase=check", "phase=publish",
    ]
    assert " files=" in progress[2] and " bytes=" in progress[2]
    assert all(" elapsed=" in message for message in progress)


def test_publication_sync_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(fd: int) -> None:
        raise OSError("injected sync failure")

    monkeypatch.setattr(export_module.os, "fsync", fail)
    with pytest.raises(OSError, match="injected sync failure"):
        _write_unvalidated_outputs_for_test(
            tmp_path / "failed", bundle=_bundle(), max_artifact_bytes=10_000_000,
        )
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert not (tmp_path / ".failed.publish.lock").exists()


def test_staging_sync_uses_bounded_waves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    writer = _BoundedTreeWriter(tmp_path / "stage", max_bytes=10000)
    for index in range(17):
        writer.write_text(f"{index}.txt", "data")
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    calls = 0
    active = 0
    peak = 0

    def sync(fd: int) -> None:
        nonlocal calls, active, peak
        with lock:
            index = calls
            calls += 1
            active += 1
            peak = max(peak, active)
        if index < 16:
            barrier.wait(timeout=10)
        with lock:
            active -= 1

    monkeypatch.setattr(export_module.os, "fsync", sync)
    writer.sync_files()
    assert calls == 17 and peak == 8 and active == 0


@pytest.mark.parametrize("action", ["raise", "mutate"])
def test_publication_observer_cannot_bypass_final_checks(
    tmp_path: Path, action: str,
) -> None:
    def observe(message: str) -> None:
        if "phase=publish " not in message:
            return
        if action == "raise":
            raise RuntimeError("observer failed")
        staging = next(tmp_path.glob(".observed.staging-*"))
        (staging / "canonical/claims.json").write_text("{}\n")

    expected = RuntimeError if action == "raise" else ValueError
    with pytest.raises(expected):
        _write_unvalidated_outputs_for_test(
            tmp_path / "observed", bundle=_bundle(), max_artifact_bytes=10_000_000,
            progress_callback=observe,
        )
    assert not (tmp_path / "observed").exists()
    assert not list(tmp_path.glob(".observed.staging-*"))
    assert not (tmp_path / ".observed.publish.lock").exists()


def test_other_platforms_keep_writable_descriptor_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setattr(export_module, "_DEFER_FILE_SYNC", False)
    synced = []
    monkeypatch.setattr(export_module.os, "fsync", lambda fd: synced.append(os.fstat(fd).st_ino))
    source = tmp_path / "source"
    source.write_text("source")
    writer = _BoundedTreeWriter(tmp_path / "stage", max_bytes=1000)
    writer.write_text("text", "text", read_only=True)
    assert len(synced) == 1
    writer.write_json("data.json", {"key":1}, read_only=True)
    assert len(synced) == 2
    writer.copy_utf8_file("copy", source, source.stat())
    assert len(synced) == 3
    writer.sync_files()
    assert len(synced) == 3


def test_staging_sync_rejects_replacement_during_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _BoundedTreeWriter(tmp_path / "stage", max_bytes=1000)
    path = writer.write_text("value", "original")
    replacement = tmp_path / "replacement"
    replacement.write_text("other")
    original_open = export_module.os.open
    original_close = export_module.os.close
    opened = []
    closed = []

    def swap_then_open(name, flags):
        replacement.replace(path)
        fd = original_open(name, flags)
        opened.append(fd)
        return fd

    def close(fd: int) -> None:
        closed.append(fd)
        original_close(fd)

    monkeypatch.setattr(export_module.os, "open", swap_then_open)
    monkeypatch.setattr(export_module.os, "close", close)
    with pytest.raises(ValueError, match="changed during sync"):
        writer.sync_files()
    assert len(opened) == 1 and closed == opened
