from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from helpers import config_for_tests
from repo_constraint_memory_v3.builder import protocol_manifest_for_config
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceRecord,
    contractual_request_bytes,
    request_identity,
)
from repo_constraint_memory_v3.models import (
    RepositoryMetadata,
    RepositorySettings,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.semantic.models import PopulationSpec
from repo_constraint_memory_v3.unified.export import (
    _write_unvalidated_outputs_for_test,
    canonical_json_text,
)
from repo_constraint_memory_v3.unified import validation as validation_module
from repo_constraint_memory_v3.unified.git_evidence import (
    GitAnchor,
    GitTemporalBundle,
    _RawChange,
    _make_bundle,
    _make_slice,
    _manifest,
    collect_git_evidence,
)
from repo_constraint_memory_v3.unified.grouping import MemberBlockingAnchors
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    make_case_id,
    make_temporal_matrix_id,
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CapabilityDependency,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    TemporalMemberRelation,
    TemporalRelationMatrix,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedCompleteness,
    UnifiedIndex,
)
from repo_constraint_memory_v3.unified.policy import apply_case_policy
from repo_constraint_memory_v3.unified.pipeline import (
    CandidateGroupingExecution,
    _build_unified_funnel,
    run_unified_layer,
)
from repo_constraint_memory_v3.unified.semantic import (
    TemporalEvidenceClassificationResponse,
    _verifier_provider_response_model,
    allowed_verifier_evidence_refs,
    build_temporal_evidence_prompt,
    build_unified_case_verify_prompt,
    unified_case_verify,
)
from repo_constraint_memory_v3.unified.upstream import make_unified_upstream_inputs
from repo_constraint_memory_v3.unified.validation import (
    _UnifiedOmissionLedger,
    _consumer_repository_name,
    _exhaustive_unified_inference_replay_errors,
    _git_object_replay_errors,
    _inspect_large_json,
    _json_value,
    _refinement_budget_dependency_errors,
    _reference_closure,
    _replay_round_zero_grouping,
    _recompute_policy,
    _snapshot_bytes,
    _snapshot_tree,
    _stream_content_digest,
    validate_unified_directory as _validate_unified_directory,
)

from test_unified_export import PROFILE, SNAPSHOT, _bundle
from test_unified_upstream import (
    PROFILE as UPSTREAM_PROFILE,
    SNAPSHOT as UPSTREAM_SNAPSHOT,
    _artifact as upstream_artifact,
    _decision as upstream_decision,
    _empty_current_source as upstream_empty_current_source,
    _verification as upstream_verification,
)
from test_unified_pipeline import (
    _affirmative_need,
    _NoCallClient,
    _PipelineClient,
    _assessment as pipeline_assessment,
    _run as run_test_pipeline,
)


def validate_unified_directory(*args, **kwargs):
    """Synthetic exporter fixtures intentionally omit channel-local trees."""

    kwargs.setdefault("validate_channels", False)
    return _validate_unified_directory(*args, **kwargs)


def test_tree_snapshot_retains_metadata_and_rejects_late_replacement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload_path = run / "payload.json"
    original = {"value": "a" * 256_000}
    payload_path.write_text(canonical_json_text(original), encoding="utf-8")
    (run / "note.txt").write_text("bounded text", encoding="utf-8")

    tree, errors = _snapshot_tree(run, max_artifact_bytes=1_000_000)

    assert errors == []
    assert tree is not None
    assert not hasattr(tree, "json_values")
    assert all(not isinstance(value, bytes) for value in tree.files.values())
    assert tree.files["payload.json"].size == payload_path.stat().st_size
    read_errors: list[str] = []
    assert _json_value(tree, "payload.json", read_errors) == original
    assert read_errors == []

    replacement = {"value": "b" * 256_000}
    payload_path.write_text(canonical_json_text(replacement), encoding="utf-8")
    assert payload_path.stat().st_size == tree.files["payload.json"].size
    assert _snapshot_bytes(tree, "payload.json", read_errors) is None
    assert "artifact changed after snapshot: payload.json" in read_errors


def test_streamed_model_digest_matches_canonical_nested_upstream_views() -> None:
    current = upstream_empty_current_source()
    observations = SimpleNamespace(
        index=SimpleNamespace(
            snapshot_digest=UPSTREAM_SNAPSHOT,
            observation_set_digest="d" * 64,
            overall="complete",
            incomplete_reasons=[],
        ),
        shards=(),
    )
    _, upstream = upstream_artifact(
        current_source=current,
        observations=observations,
    )

    assert upstream.current_source is not None
    assert upstream.observation_shards is not None
    for value in (upstream.current_source, upstream.observation_shards):
        assert _stream_content_digest(value) == content_digest(
            value.model_dump(mode="json")
        )

    population_spec = PopulationSpec.model_validate(
        {"op": "universe", "universe": "file"}
    )
    assert _stream_content_digest(population_spec) == content_digest(
        population_spec.model_dump(mode="json")
    )
    assert _stream_content_digest([population_spec]) == content_digest(
        [population_spec.model_dump(mode="json")]
    )


def test_streamed_json_scanner_preserves_canonical_and_duplicate_key_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.validation."
        "_STREAMING_JSON_THRESHOLD_BYTES",
        1,
    )
    run = tmp_path / "run"
    unified = run / "canonical" / "audit"
    unified.mkdir(parents=True)
    path = unified / "upstream-inputs.json"
    payload = seal_payload(
        {
            "artifact_kind": "scanner_fixture",
            "current_source": {"items": [1, -0.0, "é", None]},
            "explicit_verification": {"status": "complete"},
            "observation_shards": [],
        }
    )
    path.write_text(canonical_json_text(payload), encoding="utf-8")

    tree, errors = _snapshot_tree(run, max_artifact_bytes=1_000_000)
    assert errors == []
    assert tree is not None
    assert tree.json_content_digests[
        "canonical/audit/upstream-inputs.json"
    ] == payload[
        "artifact_digest"
    ]
    assert tree.json_value_digests["canonical/audit/upstream-inputs.json"][
        "current_source"
    ] == content_digest(payload["current_source"])

    path.write_text('{\n  "a": 1,\n  "a": 2\n}\n', encoding="utf-8")
    _, errors = _snapshot_tree(run, max_artifact_bytes=1_000_000)
    assert any("keys are duplicated" in item for item in errors)


def test_omission_ledger_rejects_index_calls_above_hard_limit() -> None:
    errors: list[str] = []
    ledger = _UnifiedOmissionLedger.from_index(
        SimpleNamespace(
            total_calls=4,
            total_input_tokens=0,
            total_output_tokens=0,
            stages=[],
        ),
        SimpleNamespace(
            max_total_calls=3,
            max_total_input_tokens=100,
            max_total_output_tokens=100,
        ),
        errors,
    )

    assert ledger.max_calls == 3
    assert errors == [
        "inference index total calls exceed configured hard limit"
    ]


def _git(root: Path, *argv: str) -> str:
    environment = {
        "HOME": root.as_posix(),
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ["git", "-C", root.as_posix(), *argv],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Norm4Code Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "README.md").write_text("# Fixture\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--no-gpg-sign", "-m", "fixture snapshot")
    return repository, _git(repository, "rev-parse", "HEAD")


class _FixtureVerificationClient(_PipelineClient):
    def _response(self, request: dict[str, object]) -> dict[str, object]:
        response = super()._response(request)
        if request["stage"] != "unified_case_verify":
            return response
        evidence_by_member: dict[str, list[str]] = {}
        all_evidence: list[str] = []
        for item in request["current_evidence"]["items"]:
            all_evidence.append(item["evidence_id"])
            for member_id in item["member_ids"]:
                evidence_by_member.setdefault(member_id, []).append(
                    item["evidence_id"]
                )
        reasons_by_member = {
            item["member_id"]: sorted(item["typed_local_reasons"])
            for item in request["members"]
        }
        qualifier_by_member = {
            item["member_id"]: bool(
                item["semantics"]["condition"]
                or item["semantics"]["exception"]
            )
            for item in request["members"]
        }
        for item in response["member_findings"]:
            item["evidence_refs"] = sorted(evidence_by_member[item["member_id"]])
            item["closed_local_reason_codes"] = reasons_by_member[item["member_id"]]
            item["condition_support"] = (
                "pass"
                if qualifier_by_member[item["member_id"]]
                else "not_applicable"
            )
        for item in response["member_relations"]:
            item["evidence_refs"] = sorted(
                {
                    *evidence_by_member[item["left_member_id"]],
                    *evidence_by_member[item["right_member_id"]],
                }
            )
        for item in response["conclusions"]:
            item["evidence_refs"] = sorted(all_evidence)
        return response


class _TemporalBisectionClient(_PipelineClient):
    """Force each multi-item temporal request through live bisection."""

    def __init__(self, *, terminal_leaves: bool = False) -> None:
        super().__init__()
        self.terminal_leaves = terminal_leaves

    def _response(self, request: dict[str, object]) -> dict[str, object]:
        response = super()._response(request)
        if (
            request["stage"] == "temporal_evidence_classify"
            and (len(request["items"]) > 1 or self.terminal_leaves)
        ):
            response["relations"] = []
        return response


def _temporal_bisection_replay_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    terminal_leaves: bool = False,
):
    config = config_for_tests()
    assessment = pipeline_assessment(
        "temporal-bisection-replay",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_TemporalBisectionClient(terminal_leaves=terminal_leaves),
        snapshot_commit="e" * 40,
        git_bundle_size=3,
    )
    temporal_records = [
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert len(temporal_records) == 5
    if terminal_leaves:
        assert all(item.terminal_status == "failure" for item in temporal_records)
        assert result.bundle.temporal_matrices[0].status == "partial"
    else:
        assert (
            sum(item.terminal_status == "failure" for item in temporal_records)
            == 2
        )
        assert result.bundle.temporal_matrices[0].status == "complete"
    return result, config.unified, temporal_records


def _policy_bundle(
    repository: Path,
    head: str,
    *,
    partial: bool = False,
    include_git: bool = False,
) -> tuple[UnifiedBundle, dict[str, object], object]:
    base = _bundle()
    llm_settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 0,
            "provider_retries": 0,
            "output_limit_retries": 0,
        }
    )
    finding, inference_record = unified_case_verify(
        base.cases[0],
        base.assessments,
        base.current_evidence[0],
        None,
        llm_settings=llm_settings,
        client=_FixtureVerificationClient(),
        budget=InferenceBudget(llm_settings),
        protocol_manifest_digest=PROFILE,
    )
    assert finding is not None

    assessment_by_id = {item.assessment_id: item for item in base.assessments}
    current_by_case = {item.case_id: item for item in base.current_evidence}
    matrix_by_case = {item.case_id: item for item in base.temporal_matrices}
    claims = []
    dispositions = []
    for case in base.cases:
        result = apply_case_policy(
            case=case,
            assessments=assessment_by_id,
            current_evidence=current_by_case[case.case_id],
            temporal_matrix=matrix_by_case.get(case.case_id),
            finding=finding,
        )
        claims.extend(result.claims)
        dispositions.extend(result.dispositions)
    claims.sort(key=lambda item: item.claim_id)
    dispositions.sort(key=lambda item: item.member_id)

    if partial:
        capabilities = [
            CapabilityDependency(
                capability="canonical_identity_resolution",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            ),
            CapabilityDependency(
                capability="git_augmentative",
                requirement_mode="augmentative",
                status="partial",
                reason_codes=["git_result_budget"],
            )
        ]
        package_status = "partial"
        reasons = ["git_result_budget"]
    else:
        capabilities = [
            CapabilityDependency(
                capability="canonical_identity_resolution",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            ),
            CapabilityDependency(
                capability="unified_validation",
                requirement_mode="required",
                status="complete",
                reason_codes=[],
            )
        ]
        package_status = "complete"
        reasons = []
    completeness = UnifiedCompleteness(
        structurally_valid=True,
        package_status=package_status,
        capabilities=capabilities,
        omitted_possible=partial,
        reasons=reasons,
    )
    counts = {
        status: sum(item.epistemic_status == status for item in claims)
        for status in ("accepted", "review", "rejected")
    }
    git_refs: list[ImmutableArtifactRef] = []
    git_artifacts: dict[str, object] = {}
    git_bundles = []
    if include_git:
        git_bundle = collect_git_evidence(
            repository,
            case_id=base.cases[0].case_id,
            snapshot_digest=SNAPSHOT,
            cutoff_commit=head,
            purpose="augmentative",
            anchors=(),
            required=False,
        )
        git_bundles.append(git_bundle)
        for kind, value, stable_id, digest, directory in (
            (
                "git_temporal_bundle",
                git_bundle,
                git_bundle.bundle_id,
                git_bundle.artifact_digest,
                "bundles",
            ),
            (
                "git_manifest",
                git_bundle.manifest,
                git_bundle.manifest.manifest_id,
                git_bundle.manifest.artifact_digest,
                "manifests",
            ),
        ):
            git_refs.append(
                ImmutableArtifactRef(
                    kind=kind,
                    stable_id=stable_id,
                    snapshot_digest=SNAPSHOT,
                    artifact_path=(
                        f"canonical/audit/git/{directory}/{stable_id}.json"
                    ),
                    artifact_digest=digest,
                )
            )
            git_artifacts[stable_id] = value
        git_refs.sort(key=lambda item: (item.kind, item.stable_id))

    funnel = _build_unified_funnel(
        assessments=base.assessments,
        claims=claims,
        dispositions=dispositions,
        cases=base.cases,
        current_evidence=base.current_evidence,
        git_bundles=git_bundles,
        grouping_execution=CandidateGroupingExecution(
            mode=base.funnel.candidate_grouping_mode,
            status=base.funnel.grouping_status,
            reason=base.funnel.grouping_reason,
            planned_pairs=base.funnel.grouping_planned_pairs,
            planned_calls=base.funnel.grouping_planned_calls,
            terminal_calls=base.funnel.grouping_terminal_calls,
            case_count=base.funnel.grouping_case_count,
        ),
        raw_candidate_events_by_channel={
            channel: values["raw_candidate_events"]
            for channel, values in base.funnel.by_channel.items()
        },
    )
    artifact_digests = sorted(
        [
            *(item.artifact_digest for item in base.assessments),
            *(item.artifact_digest for item in base.cases),
            *(item.artifact_digest for item in base.current_evidence),
            *(item.artifact_digest for item in base.temporal_matrices),
            finding.artifact_digest,
            *(item.artifact_digest for item in claims),
            *(item.artifact_digest for item in git_bundles),
            content_digest(funnel.model_dump(mode="json")),
            content_digest(completeness.model_dump(mode="json")),
        ]
    )
    index_payload = base.index.model_dump(mode="python", exclude={"artifact_digest"})
    index_payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_commit": head,
            "profile_digest": PROFILE,
            "finding_ids": [finding.finding_id],
            "canonical_claim_ids": sorted(item.claim_id for item in claims),
            "accepted_count": counts["accepted"],
            "review_count": counts["review"],
            "rejected_count": counts["rejected"],
            "artifact_set_digest": content_digest(artifact_digests),
        }
    )
    index = UnifiedIndex.model_validate(seal_payload(index_payload))
    payload = base.model_dump(mode="python", exclude={"artifact_digest"})
    payload.update(
        {
            "index": index,
            "git_bundle_refs": git_refs,
            "findings": [finding],
            # The generic exporter fixture carries a display-only refinement
            # record without executable lineage.  Validation fixtures replay
            # the real host contract, so omit that unrelated audit sample.
            "refinements": [],
            "canonical_claims": claims,
            "member_dispositions": dispositions,
            "funnel": funnel,
            "completeness": completeness,
        }
    )
    return (
        UnifiedBundle.model_validate(seal_payload(payload)),
        git_artifacts,
        inference_record,
    )


def _publish(
    tmp_path: Path,
    *,
    partial: bool = False,
    include_git: bool = False,
) -> tuple[Path, Path, UnifiedBundle]:
    repository, head = _repository(tmp_path)
    bundle, git_artifacts, inference_record = _policy_bundle(
        repository, head, partial=partial, include_git=include_git
    )
    output = tmp_path / "run"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=bundle,
        git_artifacts=git_artifacts,
        max_artifact_bytes=20_000_000,
    )
    output.chmod(0o755)
    inference_directory = output / "inference" / "requests"
    inference_directory.mkdir(parents=True)
    inference_path = inference_directory / f"{inference_record.request_id}.json"
    inference_path.write_text(
        canonical_json_text(inference_record), encoding="utf-8"
    )
    inference_path.chmod(0o444)
    inference_directory.chmod(0o555)
    inference_directory.parent.chmod(0o555)
    return output, repository, bundle


def test_consumer_repository_identity_preserves_explicit_config_name(
    tmp_path: Path,
) -> None:
    repository, head = _repository(tmp_path)
    bundle, _, _ = _policy_bundle(repository, head)
    explicit_name = "stable-research-identity"
    config = config_for_tests().model_copy(
        update={"repository": RepositorySettings(name=explicit_name)}
    )
    run = tmp_path / "identity-run"
    run.mkdir()
    report = {
        "repository": RepositoryMetadata(
            name=explicit_name,
            root=repository.as_posix(),
            snapshot_digest=bundle.index.snapshot_digest,
            git_head_for_reference=bundle.index.snapshot_commit,
        ).model_dump(mode="json")
    }
    (run / "build_report.json").write_text(
        canonical_json_text(report), encoding="utf-8"
    )
    tree, snapshot_errors = _snapshot_tree(
        run, max_artifact_bytes=1_000_000
    )
    assert snapshot_errors == []
    assert tree is not None
    errors: list[str] = []
    assert _consumer_repository_name(
        tree,
        repository,
        config,
        bundle.index,
        errors,
        required=True,
    ) == explicit_name
    assert errors == []


class _NoProviderClient:
    def complete_with_settings(self, *args, **kwargs):
        raise AssertionError("upstream fixture must not invoke a provider")

    def complete(self, *args, **kwargs):
        raise AssertionError("upstream fixture must not invoke a provider")


def _publish_with_upstream(
    tmp_path: Path,
) -> tuple[Path, Path, UnifiedBundle]:
    repository, head = _repository(tmp_path)
    decision = upstream_decision()
    verification = upstream_verification(decision)
    config = config_for_tests()
    llm_settings = config.llm.model_copy(
        update={
            "max_retries": 0,
            "backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
        }
    )
    upstream = make_unified_upstream_inputs(
        snapshot_digest=UPSTREAM_SNAPSHOT,
        profile_digest=UPSTREAM_PROFILE,
        snapshot_commit=head,
        explicit_decisions=[decision],
        explicit_verification=verification,
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=1,
    )
    result = run_unified_layer(
        repository_root=repository,
        settings=config.unified,
        llm_settings=llm_settings,
        client=_NoProviderClient(),
        budget=InferenceBudget(llm_settings),
        protocol_manifest_digest=UPSTREAM_PROFILE,
        snapshot_digest=UPSTREAM_SNAPSHOT,
        snapshot_commit=head,
        profile_digest=UPSTREAM_PROFILE,
        explicit_decisions=[decision],
        explicit_verification=verification,
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=1,
    )
    output = tmp_path / "run"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=result.bundle,
        git_artifacts=result.git_artifacts,
        upstream_inputs=upstream,
        max_artifact_bytes=20_000_000,
    )
    return output, repository, result.bundle


def _publish_unsafe_scope_with_upstream(
    tmp_path: Path,
) -> tuple[Path, Path, UnifiedBundle]:
    repository, head = _repository(tmp_path)
    safe = upstream_decision()
    assert safe.raw_candidate is not None
    raw_scope = [
        "*/lib/python*/site-packages/example/*",
        "*\\Lib\\site-packages\\example\\*",
        "testing/*",
    ]
    unsafe = replace(
        safe,
        candidate_fingerprint=content_digest(["unsafe-scope-candidate"]),
        outcome="rejected",
        outcome_stage="semantics",
        reason_code="scope_invalid",
        raw_candidate=safe.raw_candidate.model_copy(
            update={"applies_to": raw_scope}
        ),
        errors=[
            "invalid repository-relative applies_to glob: "
            "*\\Lib\\site-packages\\example\\*"
        ],
        canonical_scope=tuple(raw_scope),
        host_eligible=False,
        verification_disposition=None,
        verification_request_id=None,
        verification_reason_codes=[],
        claim_id=None,
        proposition=None,
    )
    verification = upstream_verification(safe)
    config = config_for_tests()
    llm_settings = config.llm.model_copy(
        update={
            "max_retries": 0,
            "backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
        }
    )
    upstream = make_unified_upstream_inputs(
        snapshot_digest=UPSTREAM_SNAPSHOT,
        profile_digest=UPSTREAM_PROFILE,
        snapshot_commit=head,
        explicit_decisions=[safe, unsafe, unsafe],
        explicit_verification=verification,
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=3,
    )
    result = run_unified_layer(
        repository_root=repository,
        settings=config.unified,
        llm_settings=llm_settings,
        client=_NoProviderClient(),
        budget=InferenceBudget(llm_settings),
        protocol_manifest_digest=UPSTREAM_PROFILE,
        snapshot_digest=UPSTREAM_SNAPSHOT,
        snapshot_commit=head,
        profile_digest=UPSTREAM_PROFILE,
        explicit_decisions=[safe, unsafe, unsafe],
        explicit_verification=verification,
        current_source=None,
        observations=None,
        explicit_discovery_complete=True,
        raw_candidate_events=3,
    )
    output = tmp_path / "run"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=result.bundle,
        git_artifacts=result.git_artifacts,
        upstream_inputs=upstream,
        max_artifact_bytes=20_000_000,
    )
    return output, repository, result.bundle


def _publish_pipeline_refinement(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path, UnifiedBundle]:
    repository, _ = _repository(tmp_path)
    config = config_for_tests(unified={"git_enabled": False})
    protocol = protocol_manifest_for_config(
        config,
        agent_profile_digest="d" * 64,
    )
    profile = protocol.protocol_manifest_digest
    assessment = pipeline_assessment(
        "validated-refinement",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
        profile_digest=profile,
    )
    result = run_test_pipeline(
        monkeypatch,
        repository,
        [assessment],
        client=_PipelineClient(propose_narrowing=True),
        unified_updates={"git_enabled": False},
        profile_digest=profile,
        protocol_manifest_digest=profile,
    )
    output = tmp_path / "refinement-run"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=result.bundle,
        git_artifacts=result.git_artifacts,
        protocol_manifest=protocol,
        inference_records=result.inference_records,
        max_artifact_bytes=20_000_000,
    )
    return output, repository, result.bundle


class _MixedRefinementClient(_PipelineClient):
    """Emit one valid narrowing and one host-invalid no-op proposal."""

    def __init__(self) -> None:
        super().__init__(propose_narrowing=True)

    def _response(self, request: dict[str, object]) -> dict[str, object]:
        response = super()._response(request)
        if (
            request["stage"] == "unified_case_verify"
            and request["verification_round"] == 0
            and request["members"][0]["semantics"]["rule"]
            == "Keep this proposal as a no-op."
        ):
            proposal = response["refinement_proposal"]
            assert proposal is not None
            proposal["semantics"] = dict(request["members"][0]["semantics"])
            proposal["change_kinds"] = ["wording"]
        return response


def _publish_pipeline_mixed_refinement(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path, UnifiedBundle]:
    repository, _ = _repository(tmp_path)
    config = config_for_tests(unified={"git_enabled": False})
    protocol = protocol_manifest_for_config(
        config,
        agent_profile_digest="d" * 64,
    )
    profile = protocol.protocol_manifest_digest
    assessments = [
        pipeline_assessment(
            "valid-refinement",
            channel="implicit_current",
            rule="Register plugins through PluginManager.",
            profile_digest=profile,
        ),
        pipeline_assessment(
            "invalid-refinement",
            channel="implicit_current",
            rule="Keep this proposal as a no-op.",
            profile_digest=profile,
        ),
    ]
    result = run_test_pipeline(
        monkeypatch,
        repository,
        assessments,
        client=_MixedRefinementClient(),
        unified_updates={"git_enabled": False},
        profile_digest=profile,
        protocol_manifest_digest=profile,
    )
    output = tmp_path / "mixed-refinement-run"
    _write_unvalidated_outputs_for_test(
        output,
        bundle=result.bundle,
        git_artifacts=result.git_artifacts,
        protocol_manifest=protocol,
        inference_records=result.inference_records,
        max_artifact_bytes=20_000_000,
    )
    return output, repository, result.bundle


def _rewrite_json(path: Path, value: object) -> None:
    path.chmod(0o644)
    path.write_text(canonical_json_text(value), encoding="utf-8")


def test_snapshot_tree_enforces_width_bound_before_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = tmp_path / "wide-tree"
    run.mkdir()
    for ordinal in range(3):
        (run / f"{ordinal}.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.validation._MAX_FILES",
        2,
    )

    tree, errors = _snapshot_tree(run, max_artifact_bytes=1_000)
    assert tree is None
    assert errors == ["artifact tree exceeds entry-count bound"]


def test_snapshot_tree_rejects_root_replaced_by_symlink_during_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = tmp_path / "lexical-run"
    run.mkdir()
    (run / "artifact.json").write_text("{}\n", encoding="utf-8")
    displaced = tmp_path / "displaced-run"
    real_resolve = Path.resolve
    swapped = False

    def replace_before_resolve(self: Path, *args, **kwargs):
        nonlocal swapped
        if not swapped and self == run.absolute():
            run.rename(displaced)
            # Resolve still reaches the original inode; the lexical path is
            # nevertheless now a symlink and must be rejected independently.
            run.symlink_to(displaced, target_is_directory=True)
            swapped = True
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", replace_before_resolve)
    tree, errors = _snapshot_tree(run, max_artifact_bytes=1_000)

    assert swapped
    assert tree is None
    assert errors == ["run directory changed during safe resolution"]


def test_large_json_scanner_maps_private_copy_not_mutable_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "large.json"
    source.write_text(
        canonical_json_text({"payload": "x" * 10_000}),
        encoding="utf-8",
    )
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_mmap = validation_module.mmap.mmap
    mapped_identities: list[tuple[int, int]] = []

    def record_mapping(descriptor: int, *args, **kwargs):
        info = os.fstat(descriptor)
        mapped_identities.append((info.st_dev, info.st_ino))
        return real_mmap(descriptor, *args, **kwargs)

    monkeypatch.setattr(validation_module.mmap, "mmap", record_mapping)
    metadata, error, digest, _ = _inspect_large_json(
        source,
        remaining=100_000,
        exclude_top_digest=False,
    )

    assert error is None
    assert metadata is not None
    assert digest is None
    assert mapped_identities and mapped_identities[0] != source_identity


def test_valid_complete_tree_replays_without_provider_or_git_history_query(
    tmp_path: Path, monkeypatch
) -> None:
    output, repository, _ = _publish(tmp_path)
    calls: list[tuple[str, ...]] = []
    original = __import__(
        "repo_constraint_memory_v3.unified.validation", fromlist=["BoundedGitRunner"]
    ).BoundedGitRunner.run

    def record(self, repository_root, argv, **kwargs):
        calls.append(tuple(argv))
        return original(self, repository_root, argv, **kwargs)

    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.validation.BoundedGitRunner.run",
        record,
    )
    assert validate_unified_directory(output, repository) == []
    assert calls == [
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "--verify", "HEAD^{commit}"),
        # Recheck HEAD after the artifact replay so a repository change during
        # validation cannot be accepted as the original fixed snapshot.
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "--verify", "HEAD^{commit}"),
    ]


def test_required_current_gap_replays_exact_zero_call_host_finding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from repo_constraint_memory_v3.unified.pipeline import _host_finding

    assessment = pipeline_assessment(
        "required-current-gap-replay",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates={"git_enabled": False},
    )
    case = result.bundle.cases[0]
    current = result.bundle.current_evidence[0]
    reason = "population_query_dependency_partial"
    current_payload = current.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    current_payload.update(
        {
            "dependencies": [
                item.model_copy(
                    update={
                        "status": "partial",
                        "reason_codes": [reason],
                    }
                )
                if item.requirement_mode == "required"
                else item
                for item in current.dependencies
            ],
            "status": "partial",
            "incomplete_reasons": [reason],
        }
    )
    current_payload["artifact_digest"] = artifact_digest(current_payload)
    incomplete = CurrentEvidenceBundle.model_validate(current_payload)
    finding = _host_finding(
        case,
        incomplete,
        profile_digest=result.bundle.index.profile_digest,
        status="not_required",
        reason_code="required_current_capability_incomplete",
        detail_reason_codes=tuple(incomplete.incomplete_reasons),
    )
    bundle = result.bundle.model_copy(
        update={"current_evidence": [incomplete], "findings": [finding]}
    )
    records = [
        item
        for item in result.inference_records
        if item.stage != "unified_case_verify"
    ]
    settings = config_for_tests(unified={"git_enabled": False}).unified
    route = {case.case_id: "not_required_current_incomplete"}

    assert _exhaustive_unified_inference_replay_errors(
        bundle,
        result.git_artifacts,
        records,
        settings=settings,
        round_zero_routes=route,
        git_enabled=False,
        snapshot_commit_available=False,
    ) == []

    wrong_finding = _host_finding(
        case,
        incomplete,
        profile_digest=result.bundle.index.profile_digest,
        status="not_required",
        reason_code="required_current_capability_incomplete",
    )
    tampered = bundle.model_copy(update={"findings": [wrong_finding]})
    errors = _exhaustive_unified_inference_replay_errors(
        tampered,
        result.git_artifacts,
        records,
        settings=settings,
        round_zero_routes=route,
        git_enabled=False,
        snapshot_commit_available=False,
    )
    assert any("required-current gap has no exact host finding" in item for item in errors)


def test_explicit_local_positive_fast_path_replays_exact_reason(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from repo_constraint_memory_v3.unified.pipeline import _host_finding

    assessment = pipeline_assessment(
        "explicit-local-positive-replay",
        channel="explicit",
        rule="Use the registered plugin manager.",
        source_authority="descriptive_typed",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_NoCallClient(),
        unified_updates={"git_enabled": False},
    )
    settings = config_for_tests(unified={"git_enabled": False}).unified

    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=settings,
        round_zero_routes=result.case_routes,
        git_enabled=False,
        snapshot_commit_available=False,
    ) == []

    case = result.bundle.cases[0]
    current = result.bundle.current_evidence[0]
    stale_finding = _host_finding(
        case,
        current,
        profile_digest=result.bundle.index.profile_digest,
        status="not_required",
        reason_code="explicit_high_authority_fast_path",
    )
    stale = result.bundle.model_copy(update={"findings": [stale_finding]})
    errors = _exhaustive_unified_inference_replay_errors(
        stale,
        result.git_artifacts,
        list(result.inference_records),
        settings=settings,
        round_zero_routes=result.case_routes,
        git_enabled=False,
        snapshot_commit_available=False,
    )
    assert any(
        "explicit local-positive fast path has no exact host finding" in item
        for item in errors
    )


def test_temporal_bisection_saved_replay_consumes_and_validates_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, settings, temporal_records = _temporal_bisection_replay_fixture(
        tmp_path,
        monkeypatch,
    )

    def replay(records: list[InferenceRecord]) -> list[str]:
        return _exhaustive_unified_inference_replay_errors(
            result.bundle,
            result.git_artifacts,
            records,
            settings=settings,
            round_zero_routes=result.case_routes,
        )

    saved_records = list(result.inference_records)
    assert replay(saved_records) == []

    singleton_child = next(
        record
        for record in temporal_records
        if record.terminal_status == "success"
        and sum(
            item.kind == "temporal_evidence_item"
            for item in record.input_manifest
        )
        == 1
    )
    missing_errors = replay(
        [
            record
            for record in saved_records
            if record.request_id != singleton_child.request_id
        ]
    )
    assert any(
        "temporal batch has missing or duplicate records" in item
        for item in missing_errors
    )

    tampered_payload = singleton_child.model_dump(mode="json")
    tampered_payload["request_ordinal"] += 1
    tampered_payload["request_id"] = request_identity(
        stage=singleton_child.stage,
        protocol_version=singleton_child.protocol_version,
        mode=singleton_child.mode,
        inputs=singleton_child.input_manifest,
        request_ordinal=tampered_payload["request_ordinal"],
    )
    tampered_payload.pop("record_digest", None)
    tampered_payload["record_digest"] = content_digest(tampered_payload)
    tampered_child = InferenceRecord.model_validate(tampered_payload)
    tampered_errors = replay(
        [
            tampered_child
            if record.request_id == singleton_child.request_id
            else record
            for record in saved_records
        ]
    )
    assert any(
        tampered_child.request_id in item
        and "temporal response replay failed" in item
        for item in tampered_errors
    )


def test_temporal_bisection_replay_preserves_terminal_leaf_gaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, settings, _ = _temporal_bisection_replay_fixture(
        tmp_path,
        monkeypatch,
        terminal_leaves=True,
    )

    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=settings,
        round_zero_routes=result.case_routes,
    ) == []
    matrix = result.bundle.temporal_matrices[0]
    assert "temporal_batch_incomplete" in matrix.incomplete_reasons
    assert "temporal_batches_unprocessed" not in matrix.incomplete_reasons


def test_serial_temporal_bisection_rechecks_budget_for_each_child(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assessment = pipeline_assessment(
        "temporal-serial-budget-child",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    constrained = run_test_pipeline(
        monkeypatch,
        tmp_path / "constrained",
        [assessment],
        client=_TemporalBisectionClient(),
        llm_updates={"max_total_calls": 1, "max_concurrency": 1},
        snapshot_commit="e" * 40,
        git_bundle_size=3,
    )
    temporal_records = [
        item
        for item in constrained.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert [item.usage.calls for item in temporal_records] == [1, 0, 0]
    ledger = _UnifiedOmissionLedger(
        remaining={"unified_case_verify": 1},
        omission_reasons={
            "unified_case_verify": {"max_total_calls": 1}
        },
        max_calls=1,
        max_input_tokens=config_for_tests().llm.max_total_input_tokens,
        max_output_tokens=config_for_tests().llm.max_total_output_tokens,
    )
    assert _exhaustive_unified_inference_replay_errors(
        constrained.bundle,
        constrained.git_artifacts,
        list(constrained.inference_records),
        settings=config_for_tests().unified,
        round_zero_routes=constrained.case_routes,
        omission_ledger=ledger,
        max_concurrency=1,
    ) == []
    assert ledger.finish() == []

    rich = run_test_pipeline(
        monkeypatch,
        tmp_path / "rich",
        [assessment],
        client=_TemporalBisectionClient(),
        llm_updates={"max_total_calls": 100, "max_concurrency": 1},
        snapshot_commit="e" * 40,
        git_bundle_size=3,
    )
    rich_by_id = {
        item.request_id: item
        for item in rich.inference_records
        if item.stage == "temporal_evidence_classify"
    }
    zero_child = next(item for item in temporal_records if item.usage.calls == 0)
    live_child = rich_by_id[zero_child.request_id]
    assert live_child.usage.calls == 1
    tampered_records = [
        live_child if item.request_id == zero_child.request_id else item
        for item in constrained.inference_records
    ]
    tamper_ledger = _UnifiedOmissionLedger(
        remaining={"unified_case_verify": 1},
        omission_reasons={
            "unified_case_verify": {"max_total_calls": 1}
        },
        max_calls=1,
        max_input_tokens=config_for_tests().llm.max_total_input_tokens,
        max_output_tokens=config_for_tests().llm.max_total_output_tokens,
    )
    tamper_errors = _exhaustive_unified_inference_replay_errors(
        constrained.bundle,
        constrained.git_artifacts,
        tampered_records,
        settings=config_for_tests().unified,
        round_zero_routes=constrained.case_routes,
        omission_ledger=tamper_ledger,
        max_concurrency=1,
    )
    assert any(
        "temporal recursive record crosses serial budget closure" in item
        for item in tamper_errors
    )


def test_temporal_request_byte_gate_replays_exact_typed_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unified_updates = {"max_temporal_request_bytes": 1_000}
    config = config_for_tests(unified=unified_updates)
    assessment = pipeline_assessment(
        "temporal-request-byte-gate",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates=unified_updates,
        snapshot_commit="e" * 40,
        git_bundle_size=1,
    )
    temporal_records = [
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert len(temporal_records) == 1
    assert temporal_records[0].terminal_status == "failure"
    assert temporal_records[0].incomplete_reasons == [
        "max_temporal_classifier_request_bytes"
    ]
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
    ) == []

    relaxed = config.unified.model_copy(
        update={"max_temporal_request_bytes": 2_000_000}
    )
    errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=relaxed,
        round_zero_routes=result.case_routes,
    )
    assert any("forged byte-gate skip" in item for item in errors)


def test_temporal_replay_rejects_success_above_saved_byte_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = config_for_tests()
    assessment = pipeline_assessment(
        "temporal-success-byte-bound",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        snapshot_commit="e" * 40,
        git_bundle_size=1,
    )
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
    ) == []

    temporal_record = next(
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    )
    case = result.bundle.cases[0]
    item_ids = {
        item.stable_id
        for item in temporal_record.input_manifest
        if item.kind == "temporal_evidence_item"
    }
    slices = [
        item
        for item in result.git_bundles[0].slices
        if item.item_id in item_ids
    ]
    request = build_temporal_evidence_prompt(case, slices)
    raw_request_bytes = len(request.encode("utf-8"))
    assert raw_request_bytes < contractual_request_bytes(
        request,
        TemporalEvidenceClassificationResponse,
    )
    tightened = config.unified.model_copy(
        update={"max_temporal_request_bytes": raw_request_bytes}
    )
    errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=tightened,
        round_zero_routes=result.case_routes,
    )
    assert any(
        "over-limit temporal request is not the exact" in item
        for item in errors
    )


def test_temporal_replay_derives_top_batches_from_item_and_cell_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unified_updates = {
        "max_temporal_batch_items": 3,
        "max_temporal_cells_per_request": 2,
    }
    config = config_for_tests(unified=unified_updates)
    assessment = pipeline_assessment(
        "temporal-cell-plan-replay",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates=unified_updates,
        snapshot_commit="e" * 40,
        git_bundle_size=3,
    )
    temporal_records = [
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert sorted(
        sum(
            ref.kind == "temporal_evidence_item"
            for ref in record.input_manifest
        )
        for record in temporal_records
    ) == [1, 2]
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
    ) == []


def test_temporal_and_verifier_compact_omissions_replay_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unified_updates = {"max_temporal_batch_items": 1}
    config = config_for_tests(unified=unified_updates)
    assessment = pipeline_assessment(
        "temporal-budget-validator",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={"max_total_calls": 1, "max_concurrency": 1},
        snapshot_commit="e" * 40,
        git_bundle_size=3,
    )
    ledger = _UnifiedOmissionLedger(
        remaining={
            "temporal_evidence_classify": 2,
            "unified_case_verify": 1,
        },
        omission_reasons={
            "temporal_evidence_classify": {"max_total_calls": 2},
            "unified_case_verify": {"max_total_calls": 1},
        },
        max_calls=1,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )

    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=ledger,
        max_concurrency=1,
    ) == []
    assert ledger.finish() == []

    # Final usage reaches the call cap, but deleting the earlier Git bundle
    # must not let the validator move that exhaustion boundary backwards.
    early_gap_ledger = _UnifiedOmissionLedger(
        remaining={
            "temporal_evidence_classify": 2,
            "unified_case_verify": 1,
        },
        omission_reasons={
            "temporal_evidence_classify": {"max_total_calls": 2},
            "unified_case_verify": {"max_total_calls": 1},
        },
        max_calls=1,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    without_git_bundle = {
        key: value
        for key, value in result.git_artifacts.items()
        if not isinstance(value, GitTemporalBundle)
    }
    early_gap_errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        without_git_bundle,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=early_gap_ledger,
        max_concurrency=1,
    )
    assert early_gap_errors
    assert early_gap_ledger.finish()


def test_cross_case_temporal_wave_budget_boundary_replays_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rules = [
        "Alphazoid files invoke frobnicate.",
        "Betamax records preserve quuxification.",
        "Gammadion modules prohibit zorbulation.",
    ]
    assessments = [
        pipeline_assessment(
            f"cross-case-temporal-budget-{ordinal}",
            git_need=_affirmative_need(),
            channel="implicit_current",
            rule=rule,
        )
        for ordinal, rule in enumerate(rules)
    ]
    unified_updates = {"max_temporal_batch_items": 1}
    config = config_for_tests(unified=unified_updates)
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={
            "max_total_calls": 100,
            "max_total_input_tokens": 5,
            "max_concurrency": 2,
        },
        snapshot_commit="e" * 40,
        git_bundle_size=1,
    )
    temporal_records = [
        item
        for item in result.inference_records
        if item.stage == "temporal_evidence_classify"
    ]
    assert len(temporal_records) == 2
    assert sum(
        matrix.status == "complete"
        for matrix in result.bundle.temporal_matrices
    ) == 2
    assert sum(
        matrix.status == "partial"
        for matrix in result.bundle.temporal_matrices
    ) == 1

    ledger = _UnifiedOmissionLedger(
        remaining={
            "temporal_evidence_classify": 1,
            "unified_case_verify": 3,
        },
        omission_reasons={
            "temporal_evidence_classify": {"max_total_input_tokens": 1},
            "unified_case_verify": {"max_total_input_tokens": 3},
        },
        max_calls=100,
        max_input_tokens=5,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=ledger,
        max_concurrency=2,
    ) == []
    assert ledger.finish() == []

    serial_ledger = _UnifiedOmissionLedger(
        remaining={
            "temporal_evidence_classify": 1,
            "unified_case_verify": 3,
        },
        omission_reasons={
            "temporal_evidence_classify": {"max_total_input_tokens": 1},
            "unified_case_verify": {"max_total_input_tokens": 3},
        },
        max_calls=100,
        max_input_tokens=5,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    serial_errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=serial_ledger,
        max_concurrency=1,
    )
    assert any("after global budget closure" in item for item in serial_errors)


def test_parallel_verifier_token_crossing_replays_and_rejects_serial_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rules = [
        "Alphazoid files invoke frobnicate.",
        "Betamax records preserve quuxification.",
        "Gammadion modules prohibit zorbulation.",
        "Deltawave packages require wibblecheck.",
        "Epsilonware tests enforce snorflelint.",
    ]
    assessments = [
        pipeline_assessment(
            f"parallel-budget-{ordinal}",
            channel="implicit_current",
            rule=rule,
        )
        for ordinal, rule in enumerate(rules)
    ]
    unified_updates = {"git_enabled": False}
    config = config_for_tests(unified=unified_updates)
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={
            "max_total_calls": 100,
            "max_total_input_tokens": 5,
            "max_concurrency": 2,
        },
    )
    verifier_records = [
        item
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ]
    assert len(verifier_records) == 2
    exhausted_findings = [
        item
        for item in result.bundle.findings
        if "unified_case_verifier_budget_exhausted" in item.failure_reasons
    ]
    assert len(exhausted_findings) == 3

    ledger = _UnifiedOmissionLedger(
        remaining={"unified_case_verify": 3},
        omission_reasons={
            "unified_case_verify": {"max_total_input_tokens": 3}
        },
        max_calls=100,
        max_input_tokens=5,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=ledger,
        git_enabled=False,
        max_concurrency=2,
    ) == []
    assert ledger.finish() == []

    serial_ledger = _UnifiedOmissionLedger(
        remaining={"unified_case_verify": 3},
        omission_reasons={
            "unified_case_verify": {"max_total_input_tokens": 3}
        },
        max_calls=100,
        max_input_tokens=5,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    serial_errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
        omission_ledger=serial_ledger,
        git_enabled=False,
        max_concurrency=1,
    )
    assert any("after global budget closure" in item for item in serial_errors)


def test_verifier_request_byte_gate_replays_exact_typed_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unified_updates = {
        "git_enabled": False,
        "max_case_input_bytes": 1_000,
    }
    config = config_for_tests(unified=unified_updates)
    assessment = pipeline_assessment(
        "verifier-request-byte-gate",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates=unified_updates,
    )
    verifier_records = [
        item
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ]
    assert len(verifier_records) == 1
    assert verifier_records[0].terminal_status == "failure"
    assert verifier_records[0].incomplete_reasons == [
        "max_unified_verifier_request_bytes"
    ]
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
    ) == []

    relaxed = config.unified.model_copy(
        update={"max_case_input_bytes": 2_000_000}
    )
    errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=relaxed,
        round_zero_routes=result.case_routes,
    )
    assert any(
        "forged byte-gate skip" in item
        for item in errors
    )


def test_verifier_replay_rejects_success_above_saved_byte_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unified_updates = {"git_enabled": False}
    config = config_for_tests(unified=unified_updates)
    assessment = pipeline_assessment(
        "verifier-success-byte-bound",
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=_PipelineClient(),
        unified_updates=unified_updates,
    )
    verifier_records = [
        item
        for item in result.inference_records
        if item.stage == "unified_case_verify"
    ]
    assert len(verifier_records) == 1
    assert verifier_records[0].terminal_status == "success"
    assert _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes=result.case_routes,
    ) == []

    case = result.bundle.cases[0]
    current = next(
        item
        for item in result.bundle.current_evidence
        if item.case_id == case.case_id
    )
    assessments_by_id = {
        item.assessment_id: item for item in result.bundle.assessments
    }
    assessments = [
        assessments_by_id[item.assessment_id] for item in case.members
    ]
    request = build_unified_case_verify_prompt(
        case,
        assessments,
        current,
        None,
        verification_round=0,
    )
    provider_response_model = _verifier_provider_response_model(
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
    raw_request_bytes = len(request.encode("utf-8"))
    assert raw_request_bytes < contractual_request_bytes(
        request,
        provider_response_model,
    )
    tightened = config.unified.model_copy(
        update={"max_case_input_bytes": raw_request_bytes}
    )
    errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=tightened,
        round_zero_routes=result.case_routes,
    )
    assert any(
        "over-limit verifier request is not the exact" in item
        for item in errors
    )


def test_refinement_budget_skip_dependencies_replay_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class ChildNeedClient(_PipelineClient):
        def _response(self, request):
            response = super()._response(request)
            proposal = response.get("refinement_proposal")
            if proposal is not None:
                need = _affirmative_need().judgments[0].model_dump(mode="json")
                need["questions"][0]["evidence_refs"] = proposal["evidence_refs"]
                item = next(item for item in request["current_evidence"]["items"] if item["evidence_id"] == proposal["evidence_refs"][0])
                origin = next(origin for origin in request["current_evidence"]["origins"] if origin["origin_id"] == item["origin_id"])
                need["questions"][0]["path"] = origin["path"]
                proposal["git_need"] = need
            return response

    config = config_for_tests()
    assessment = pipeline_assessment(
        "refinement-budget-dependency",
        git_need=_affirmative_need(),
        channel="implicit_current",
        rule="Register plugins through PluginManager.",
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        [assessment],
        client=ChildNeedClient(propose_narrowing=True),
        llm_updates={"max_total_calls": 2, "max_concurrency": 1},
        snapshot_commit="e" * 40,
        git_bundle_size=1,
    )
    refinement = result.bundle.refinements[0]
    child = next(
        case
        for case in result.bundle.cases
        if case.case_id != refinement.case_id
    )
    ledger = _UnifiedOmissionLedger(
        remaining={"unified_case_verify": 1},
        omission_reasons={
            "unified_case_verify": {"max_total_calls": 1}
        },
        max_calls=2,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    skip_reasons: dict[str, str] = {}
    replay_errors = _exhaustive_unified_inference_replay_errors(
        result.bundle,
        result.git_artifacts,
        list(result.inference_records),
        settings=config.unified,
        round_zero_routes={
            refinement.case_id: result.case_routes[refinement.case_id]
        },
        omission_ledger=ledger,
        max_concurrency=1,
        budget_git_skip_reasons=skip_reasons,
    )
    assert replay_errors == []
    assert ledger.finish() == []
    assert skip_reasons == {child.case_id: "max_total_calls"}
    assert _refinement_budget_dependency_errors(
        result.bundle,
        result.git_artifacts,
        skip_reasons,
    ) == []

    current = next(
        item
        for item in result.bundle.current_evidence
        if item.case_id == child.case_id
    )
    tampered_dependencies = []
    for dependency in current.dependencies:
        if dependency.capability == "git_temporal_evidence":
            payload = dependency.model_dump(mode="python")
            payload["reason_codes"] = [
                "git_not_run_after_inference_budget_exhausted",
                "max_total_input_tokens",
            ]
            dependency = CapabilityDependency.model_validate(payload)
        tampered_dependencies.append(dependency)
    tampered_current = current.model_copy(
        update={"dependencies": tampered_dependencies}
    )
    tampered_bundle = result.bundle.model_copy(
        update={
            "current_evidence": [
                tampered_current if item.case_id == child.case_id else item
                for item in result.bundle.current_evidence
            ]
        }
    )
    tampered_errors = _refinement_budget_dependency_errors(
        tampered_bundle,
        result.git_artifacts,
        skip_reasons,
    )
    assert any(
        "refinement Git budget dependency differs from replay" in item
        for item in tampered_errors
    )

    forged_without_boundary = _refinement_budget_dependency_errors(
        result.bundle,
        result.git_artifacts,
        {},
    )
    assert any(
        "appears without a replayed budget boundary" in item
        for item in forged_without_boundary
    )


def test_grouping_compact_suffix_is_exact_and_rejects_middle_hole(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assessments = [
        pipeline_assessment(
            f"group-budget-{ordinal}",
            channel="explicit",
            rule=f"Require shared registration variant {ordinal}.",
        )
        for ordinal in range(3)
    ]

    def shared_anchors(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
            for item in kwargs["members"]
        }

    unified_updates = {
        "git_enabled": False,
        "max_group_pairs_per_request": 1,
    }
    config = config_for_tests(unified=unified_updates)
    limited = run_test_pipeline(
        monkeypatch,
        tmp_path / "limited-grouping",
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={"max_total_calls": 1, "max_concurrency": 1},
        anchors_for_members=shared_anchors,
    )
    limited_group_records = [
        item
        for item in limited.inference_records
        if item.stage == "candidate_semantic_group"
    ]
    assert len(limited_group_records) == 1
    anchors = shared_anchors(
        members=[member for case in limited.bundle.cases for member in case.members]
    )
    probe = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": 100},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": 100}
        },
        max_calls=1,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    probe_errors, _, _ = _replay_round_zero_grouping(
        limited.bundle,
        list(limited.bundle.assessments),
        anchors,
        limited_group_records,
        config.unified,
        probe,
    )
    assert probe_errors == []
    consumed = 100 - probe.remaining.get("candidate_semantic_group", 0)
    exact = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": consumed},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": consumed}
        },
        max_calls=1,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    exact_errors, _, _ = _replay_round_zero_grouping(
        limited.bundle,
        list(limited.bundle.assessments),
        anchors,
        limited_group_records,
        config.unified,
        exact,
    )
    assert exact_errors == []
    assert exact.finish() == []

    complete = run_test_pipeline(
        monkeypatch,
        tmp_path / "complete-grouping",
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        anchors_for_members=shared_anchors,
    )
    complete_group_records = [
        item
        for item in complete.inference_records
        if item.stage == "candidate_semantic_group"
    ]
    assert len(complete_group_records) >= 3
    hole_ledger = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": 1},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": 1}
        },
        max_calls=1,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    hole_errors, _, _ = _replay_round_zero_grouping(
        complete.bundle,
        list(complete.bundle.assessments),
        anchors,
        [complete_group_records[0], *complete_group_records[2:]],
        config.unified,
        hole_ledger,
    )
    assert any("after global budget closure" in item for item in hole_errors)


def test_singleton_grouping_replay_rejects_foreign_grouping_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assessments = [
        pipeline_assessment(
            f"singleton-replay-{ordinal}",
            channel="explicit",
            rule=f"Require shared registration variant {ordinal}.",
        )
        for ordinal in range(2)
    ]

    def shared_anchors(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
            for item in kwargs["members"]
        }

    semantic = run_test_pipeline(
        monkeypatch,
        tmp_path / "semantic",
        assessments,
        client=_PipelineClient(),
        unified_updates={"git_enabled": False},
        anchors_for_members=shared_anchors,
    )
    grouping_records = [
        record
        for record in semantic.inference_records
        if record.stage == "candidate_semantic_group"
    ]
    assert grouping_records

    singleton_updates = {
        "candidate_grouping_mode": "singleton",
        "git_enabled": False,
    }
    singleton = run_test_pipeline(
        monkeypatch,
        tmp_path / "singleton",
        assessments,
        client=_PipelineClient(),
        unified_updates=singleton_updates,
        anchors_for_members=shared_anchors,
    )
    config = config_for_tests(unified=singleton_updates)
    members = [
        member for case in singleton.bundle.cases for member in case.members
    ]
    anchors = shared_anchors(members=members)

    errors, replayed_cases, replayed_execution = _replay_round_zero_grouping(
        singleton.bundle,
        list(singleton.bundle.assessments),
        anchors,
        grouping_records,
        config.unified,
    )

    assert replayed_cases == tuple(
        sorted(singleton.bundle.cases, key=lambda case: case.case_id)
    )
    assert replayed_execution is not None
    assert replayed_execution.mode == "singleton"
    assert any("orphaned candidate grouping records" in item for item in errors)


def test_grouping_budget_boundary_falls_back_to_stable_serial_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assessments = [
        pipeline_assessment(
            f"parallel-group-budget-{ordinal}",
            channel="explicit",
            rule=f"Require shared registration variant {ordinal}.",
        )
        for ordinal in range(4)
    ]

    def shared_anchors(**kwargs):
        return {
            item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
            for item in kwargs["members"]
        }

    unified_updates = {
        "git_enabled": False,
        "max_group_pairs_per_request": 1,
    }
    config = config_for_tests(unified=unified_updates)
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={"max_total_calls": 2, "max_concurrency": 3},
        anchors_for_members=shared_anchors,
    )
    records = [
        item
        for item in result.inference_records
        if item.stage == "candidate_semantic_group"
    ]
    assert len(records) == 2
    assert [item.terminal_status for item in records].count("success") == 2
    anchors = shared_anchors(
        members=[member for case in result.bundle.cases for member in case.members]
    )
    ledger = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": 4},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": 4}
        },
        max_calls=2,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    errors, _, _ = _replay_round_zero_grouping(
        result.bundle,
        list(result.bundle.assessments),
        anchors,
        records,
        config.unified,
        ledger,
        max_concurrency=3,
    )
    assert errors == []
    assert ledger.finish() == []

    serial_ledger = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": 4},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": 4}
        },
        max_calls=2,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    serial_errors, _, _ = _replay_round_zero_grouping(
        result.bundle,
        list(result.bundle.assessments),
        anchors,
        records,
        config.unified,
        serial_ledger,
        max_concurrency=1,
    )
    assert serial_errors == []
    assert serial_ledger.finish() == []

    over_cap = run_test_pipeline(
        monkeypatch,
        tmp_path / "validator-over-cap",
        assessments,
        client=_PipelineClient(),
        unified_updates=unified_updates,
        llm_updates={"max_total_calls": 3, "max_concurrency": 3},
        anchors_for_members=shared_anchors,
    )
    over_cap_records = [
        item
        for item in over_cap.inference_records
        if item.stage == "candidate_semantic_group"
    ]
    assert len(over_cap_records) == 3
    over_cap_ledger = _UnifiedOmissionLedger(
        remaining={"candidate_semantic_group": 3},
        omission_reasons={
            "candidate_semantic_group": {"max_total_calls": 3}
        },
        max_calls=2,
        max_input_tokens=config.llm.max_total_input_tokens,
        max_output_tokens=config.llm.max_total_output_tokens,
    )
    over_cap_errors, _, _ = _replay_round_zero_grouping(
        over_cap.bundle,
        list(over_cap.bundle.assessments),
        anchors,
        over_cap_records,
        config.unified,
        over_cap_ledger,
        max_concurrency=3,
    )
    assert any(
        "after global budget closure" in item for item in over_cap_errors
    )


def test_validator_does_not_discover_untracked_worktree_state(tmp_path: Path) -> None:
    output, repository, _ = _publish(tmp_path)
    (repository / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert validate_unified_directory(output, repository) == []


def test_finding_closure_accepts_only_exact_saved_temporal_patch_span() -> None:
    """Mirror the Git span citations observed in a real Pluggy verifier run."""

    base = _bundle().model_copy(update={"refinements": []})
    case = base.cases[0]
    item_id = "GITITEM-" + "9" * 24
    exact_span = f"{item_id}:patch:{'a' * 64}"
    manifest_ref = ImmutableArtifactRef(
        kind="git_manifest",
        stable_id="GITMAN-" + "8" * 24,
        snapshot_digest=SNAPSHOT,
        artifact_path=(
            "canonical/audit/git/manifests/GITMAN-" + "8" * 24 + ".json"
        ),
        artifact_digest="7" * 64,
    )
    inference_ref = ImmutableArtifactRef(
        kind="inference_request",
        stable_id="INFER-" + "6" * 24,
        snapshot_digest=SNAPSHOT,
        artifact_path="inference/requests/INFER-" + "6" * 24 + ".json",
        artifact_digest="5" * 64,
    )
    relations = [
        TemporalMemberRelation(
            item_id=item_id,
            member_id=member.member_id,
            relation="supports_as_written",
            affected_facets=["rule"],
            evidence_span=exact_span,
            reason_code="patch_supports_current_rule",
            uncertainty_reason=None,
        )
        for member in case.members
    ]
    matrix = TemporalRelationMatrix.model_validate(
        seal_payload(
            {
                "matrix_id": make_temporal_matrix_id(
                    SNAPSHOT,
                    PROFILE,
                    case.case_id,
                    [item_id],
                    [item.member_id for item in case.members],
                ),
                "case_id": case.case_id,
                "snapshot_digest": SNAPSHOT,
                "profile_digest": PROFILE,
                "git_manifest_ref": manifest_ref,
                "item_ids": [item_id],
                "member_ids": [item.member_id for item in case.members],
                "relations": relations,
                "status": "complete",
                "incomplete_reasons": [],
                "inference_refs": [inference_ref],
            }
        )
    )

    def with_span(span: str) -> UnifiedBundle:
        finding = base.findings[0]
        first = finding.member_findings[0].model_copy(
            update={
                "evidence_refs": sorted(
                    {*finding.member_findings[0].evidence_refs, span}
                )
            }
        )
        finding = finding.model_copy(
            update={
                "member_findings": [first, *finding.member_findings[1:]],
            }
        )
        return base.model_copy(
            update={"temporal_matrices": [matrix], "findings": [finding]}
        )

    git_bundle = GitTemporalBundle.model_construct(
        bundle_id="GITBUNDLE-" + "4" * 24,
        case_id=case.case_id,
        slices=[SimpleNamespace(item_id=item_id)],
    )
    tree = SimpleNamespace(files={})

    exact_errors: list[str] = []
    _reference_closure(
        with_span(exact_span),
        {git_bundle.bundle_id: git_bundle},
        tree,
        exact_errors,
    )
    assert not any("finding evidence closure failed" in item for item in exact_errors)

    foreign_errors: list[str] = []
    near_miss = f"{item_id}:patch:{'b' * 64}"
    _reference_closure(
        with_span(near_miss),
        {git_bundle.bundle_id: git_bundle},
        tree,
        foreign_errors,
    )
    assert any("finding evidence closure failed" in item for item in foreign_errors)


def test_git_object_replay_uses_only_exact_saved_lookups_and_rejects_reseal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, head = _repository(tmp_path)
    git_bundle = collect_git_evidence(
        repository,
        case_id="CASE-exact-object-replay",
        snapshot_digest=SNAPSHOT,
        cutoff_commit=head,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-readme",
                kind="literal",
                path="README.md",
                value="# Fixture",
                origin_ids=["ORIGIN-readme"],
                precise_origin_ids=["ORIGIN-readme"],
            )
        ],
    )
    assert git_bundle.slices

    calls: list[tuple[str, ...]] = []
    original = __import__(
        "repo_constraint_memory_v3.unified.validation", fromlist=["BoundedGitRunner"]
    ).BoundedGitRunner.run

    def record(self, repository_root, argv, **kwargs):
        calls.append(tuple(argv))
        return original(self, repository_root, argv, **kwargs)

    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.validation.BoundedGitRunner.run",
        record,
    )
    assert _git_object_replay_errors(
        repository,
        head,
        {git_bundle.bundle_id: git_bundle},
    ) == []
    forbidden = {"log", "blame", "rev-list", "ls-files"}
    assert not any(argv and argv[0] in forbidden for argv in calls)
    assert not any(
        argv[:2] in {("diff", "--quiet"), ("diff", "--cached")}
        for argv in calls
    )
    assert {argv[0] for argv in calls} <= {
        "merge-base",
        "show",
        "diff-tree",
        "--literal-pathspecs",
    }

    original_slice = git_bundle.slices[0]
    slice_payload = original_slice.model_dump(
        mode="python", exclude={"item_id", "artifact_digest"}
    )
    slice_payload["commit_time"] += 1
    resealed_slice = _make_slice(slice_payload)
    bundle_payload = git_bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    bundle_payload["slices"] = [
        resealed_slice.model_dump(mode="python"),
        *(item.model_dump(mode="python") for item in git_bundle.slices[1:]),
    ]
    resealed_bundle = _make_bundle(bundle_payload)
    errors = _git_object_replay_errors(
        repository,
        head,
        {resealed_bundle.bundle_id: resealed_bundle},
    )
    assert any("strong-slice selection/provenance differs" in item for item in errors)


def test_git_object_replay_skips_unsupported_changes_in_single_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An unsupported peer change must be audited, not crash exact replay."""

    repository, _ = _repository(tmp_path)
    _git(repository, "mv", "README.md", "GUIDE.md")
    _git(repository, "commit", "--no-gpg-sign", "-m", "rename fixture")
    head = _git(repository, "rev-parse", "HEAD")
    git_evidence = __import__(
        "repo_constraint_memory_v3.unified.git_evidence",
        fromlist=["_changes_for_commit"],
    )
    original_changes_for_commit = git_evidence._changes_for_commit

    def with_unsupported_copy(runner, root, metadata, limits):
        comparisons, reason = original_changes_for_commit(
            runner,
            root,
            metadata,
            limits,
        )
        if comparisons is None:
            return comparisons, reason
        replayed = {parent: list(changes) for parent, changes in comparisons.items()}
        parent_id, changes = next(iter(replayed.items()))
        source = changes[0]
        # Match the supported change's identity fields so the retired
        # per-slice comprehension would evaluate _change_kind("C") and crash.
        changes.append(
            _RawChange(
                parent_id=parent_id,
                old_blob_id=source.old_blob_id,
                new_blob_id=source.new_blob_id,
                status="C",
                before_path=source.before_path,
                after_path=source.after_path,
            )
        )
        return replayed, reason

    monkeypatch.setattr(
        git_evidence,
        "_changes_for_commit",
        with_unsupported_copy,
    )
    git_bundle = collect_git_evidence(
        repository,
        case_id="CASE-unsupported-peer-change",
        snapshot_digest=SNAPSHOT,
        cutoff_commit=head,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-readme",
                kind="literal",
                path="GUIDE.md",
                value="# Fixture",
                origin_ids=["ORIGIN-readme"],
                precise_origin_ids=["ORIGIN-readme"],
            )
        ],
    )
    assert "unsupported_change_status" in git_bundle.skipped_change_reasons
    assert _git_object_replay_errors(
        repository,
        head,
        {git_bundle.bundle_id: git_bundle},
    ) == []


def test_truthful_partial_passes_default_and_fails_require_complete(
    tmp_path: Path,
) -> None:
    output, repository, bundle = _publish(tmp_path, partial=True)
    assert any(
        item.epistemic_status == "accepted"
        for item in bundle.canonical_claims
    )
    assert validate_unified_directory(output, repository) == []
    strict = validate_unified_directory(
        output, repository, require_complete=True
    )
    assert strict == ["complete package required but unified package is partial"]


def test_validator_replays_final_funnel_breakdowns_after_coordinated_tamper(
    tmp_path: Path,
) -> None:
    output, repository, _ = _publish(tmp_path)
    path = output / "canonical" / "audit" / "funnel.json"
    funnel = json_load(path)
    channel = funnel["by_channel"]["explicit"]
    assert channel["accepted"] > 0
    channel["accepted"] -= 1
    channel["review"] += 1
    route = funnel["by_evidence_route"]["direct_normative"]
    route["accepted"] -= 1
    route["review"] += 1
    _rewrite_json(path, funnel)

    errors = validate_unified_directory(output, repository)
    assert "unified funnel differs from final disposition replay" in errors


def test_policy_replay_reuses_cross_case_collision_fail_closed_aggregation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assessments = [
        pipeline_assessment(
            "validator-collision-a",
            channel="explicit",
            rule="Run pytest for every change.",
        ),
        pipeline_assessment(
            "validator-collision-b",
            channel="explicit",
            rule="Run pytest for every change.",
        ),
    ]

    def force_separate_cases(plan, relations, **kwargs):
        del relations
        cases = []
        for member in plan.members:
            cases.append(
                CandidateCase.model_validate(
                    seal_payload(
                        {
                            "case_id": make_case_id(
                                member.snapshot_digest, [member.member_id]
                            ),
                            "snapshot_digest": member.snapshot_digest,
                            "profile_digest": kwargs["profile_digest"],
                            "members": [member],
                            "relations": [],
                            "grouping_status": "complete",
                            "grouping_reason_codes": [],
                            "exact_identity_groups": [],
                            "required_capabilities": ["current_evidence"],
                        }
                    )
                )
            )
        return SimpleNamespace(cases=tuple(cases))

    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.pipeline.assemble_candidate_cases",
        force_separate_cases,
    )
    result = run_test_pipeline(
        monkeypatch,
        tmp_path,
        assessments,
        client=_PipelineClient(),
        unified_updates={"git_enabled": False},
    )
    errors: list[str] = []
    _recompute_policy(result.bundle, {}, errors)
    assert errors == []


def test_safe_tree_and_consumer_tampering_are_detected(tmp_path: Path) -> None:
    output, repository, bundle = _publish(tmp_path)
    catalog_path = output / "consumers" / "accepted_only" / "catalog.md"
    catalog_path.chmod(0o644)
    assert validate_unified_directory(output, repository) == []

    case_path = (
        output
        / "canonical"
        / "audit"
        / "cases"
        / f"{bundle.cases[0].case_id}.json"
    )
    case_path.chmod(0o644)
    case_path.write_text("{\n  \"not\": \"canonical spacing\"\n}", encoding="utf-8")
    errors = validate_unified_directory(output, repository)
    assert any("JSON artifact is not canonical" in item for item in errors)

    output2, repository2, _ = _publish(tmp_path / "second")
    catalog2 = output2 / "consumers" / "accepted_only" / "catalog.md"
    catalog2.unlink()
    catalog2.symlink_to(output2 / "consumers" / "accepted_only" / "START.md")
    errors = validate_unified_directory(output2, repository2)
    assert any(
        "artifact is a symlink: consumers/accepted_only/catalog.md" in item
        for item in errors
    )

    output3, repository3, _ = _publish(tmp_path / "third")
    catalog3 = output3 / "consumers" / "accepted_only" / "catalog.md"
    catalog3.chmod(0o644)
    catalog3.write_text(
        catalog3.read_text(encoding="utf-8") + "\nleaked review\n",
        encoding="utf-8",
    )
    errors = validate_unified_directory(output3, repository3)
    assert any("consumer generated bytes differ" in item for item in errors)

    output4, repository4, _ = _publish(tmp_path / "fourth")
    (output4 / "consumers" / "accepted_only" / "START.md").unlink()
    errors = validate_unified_directory(output4, repository4)
    assert any("consumer closed-tree file set differs" in item for item in errors)

    output5, repository5, _ = _publish(tmp_path / "fifth")
    extra = output5 / "consumers" / "accepted_only" / "extra.md"
    extra.write_text("foreign\n", encoding="utf-8")
    errors = validate_unified_directory(output5, repository5)
    assert any("consumer closed-tree file set differs" in item for item in errors)


def json_load(path: Path) -> dict[str, object]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_policy_inference_git_and_snapshot_tampering_are_detected(
    tmp_path: Path,
) -> None:
    output, repository, bundle = _publish(tmp_path, include_git=True)
    assert validate_unified_directory(output, repository) == []

    manifest_ref = next(
        item for item in bundle.git_bundle_refs if item.kind == "git_manifest"
    )
    manifest_path = output / manifest_ref.artifact_path
    manifest = json_load(manifest_path)
    manifest["reason_codes"] = ["tampered"]
    _rewrite_json(manifest_path, manifest)
    errors = validate_unified_directory(output, repository)
    assert any("Git payload" in item or "failed validation" in item for item in errors)

    output_ref, repository_ref, _ = _publish(
        tmp_path / "git-ref", include_git=True
    )
    refs_path = output_ref / "canonical" / "audit" / "git" / "refs.json"
    refs = json_load(refs_path)
    refs["refs"][0]["snapshot_digest"] = "sha256:" + "f" * 64
    _rewrite_json(refs_path, refs)
    errors = validate_unified_directory(output_ref, repository_ref)
    assert any("Git ref snapshot mismatch" in item for item in errors)

    output_orphan, repository_orphan, bundle_orphan = _publish(
        tmp_path / "git-orphan", include_git=True
    )
    manifest_ref = next(
        item
        for item in bundle_orphan.git_bundle_refs
        if item.kind == "git_manifest"
    )
    manifest_payload = json_load(output_orphan / manifest_ref.artifact_path)
    manifest_payload.pop("manifest_id")
    manifest_payload.pop("artifact_digest")
    manifest_payload["purpose"] = "ambiguity_resolving"
    orphan = _manifest(manifest_payload)
    orphan_relative = (
        f"canonical/audit/git/manifests/{orphan.manifest_id}.json"
    )
    (output_orphan / orphan_relative).write_text(
        canonical_json_text(orphan), encoding="utf-8"
    )
    refs_path = output_orphan / "canonical" / "audit" / "git" / "refs.json"
    refs = json_load(refs_path)
    refs["refs"].append(
        {
            "kind": "git_manifest",
            "stable_id": orphan.manifest_id,
            "snapshot_digest": orphan.snapshot_digest,
            "artifact_path": orphan_relative,
            "artifact_digest": orphan.artifact_digest,
        }
    )
    refs["refs"] = sorted(
        refs["refs"], key=lambda item: (item["kind"], item["stable_id"])
    )
    _rewrite_json(refs_path, refs)
    errors = validate_unified_directory(output_orphan, repository_orphan)
    assert any(
        "exact temporal-bundle artifact closure" in item for item in errors
    )

    output2, repository2, bundle2 = _publish(tmp_path / "policy")
    claims_path = output2 / "canonical" / "claims.json"
    wrapper = json_load(claims_path)
    claim = next(
        item for item in wrapper["claims"] if item["epistemic_status"] == "accepted"
    )
    claim["epistemic_status"] = "review"
    claim["delivery_policy"] = "not_applicable"
    claim["review_facets"] = ["factual_uncertainty"]
    claim.update(
        seal_payload(
            {key: value for key, value in claim.items() if key != "artifact_digest"}
        )
    )
    wrapper.update(
        seal_payload(
            {key: value for key, value in wrapper.items() if key != "artifact_digest"}
        )
    )
    _rewrite_json(claims_path, wrapper)
    errors = validate_unified_directory(output2, repository2)
    assert any("frozen policy replay" in item for item in errors)

    output3, repository3, bundle3 = _publish(tmp_path / "inference")
    finding_path = (
        output3
        / "canonical"
        / "audit"
        / "findings"
        / f"{bundle3.findings[0].finding_id}.json"
    )
    finding = json_load(finding_path)
    finding["inference_ref"] = ImmutableArtifactRef(
        kind="inference_request",
        stable_id="INFER-" + "a" * 24,
        snapshot_digest=SNAPSHOT,
        artifact_path="audit/inference/INFER-" + "a" * 24 + ".json",
        artifact_digest="b" * 64,
    ).model_dump(mode="json")
    finding.update(
        seal_payload(
            {key: value for key, value in finding.items() if key != "artifact_digest"}
        )
    )
    _rewrite_json(finding_path, finding)
    errors = validate_unified_directory(output3, repository3)
    assert any("inference record is missing" in item for item in errors)

    (repository3 / "later.txt").write_text("later\n", encoding="utf-8")
    _git(repository3, "add", "later.txt")
    _git(repository3, "commit", "--no-gpg-sign", "-m", "later")
    errors = validate_unified_directory(output3, repository3)
    assert any("HEAD differs" in item for item in errors)


def test_total_artifact_bound_is_enforced(tmp_path: Path) -> None:
    output, repository, _ = _publish(tmp_path)
    errors = validate_unified_directory(
        output, repository, max_artifact_bytes=100
    )
    assert any("artifact byte bound exceeded" in item for item in errors)

    profiled_output, profiled_repository, _ = _publish(tmp_path / "profiled")
    config = config_for_tests()
    config = config.model_copy(
        update={
            "unified": config.unified.model_copy(
                update={"max_unified_artifact_bytes": 1_000}
            )
        }
    )
    (profiled_output / "config.json").write_text(
        canonical_json_text(config),
        encoding="utf-8",
    )
    errors = validate_unified_directory(
        profiled_output,
        profiled_repository,
        max_artifact_bytes=20_000_000,
    )
    assert any(
        "artifact tree bytes exceed unified profile bound" in item
        for item in errors
    )


def test_upstream_envelope_is_bound_and_required_for_channel_validation(
    tmp_path: Path,
) -> None:
    output, repository, _ = _publish_with_upstream(tmp_path)
    assert validate_unified_directory(output, repository) == []

    envelope_path = output / "canonical" / "audit" / "upstream-inputs.json"
    envelope_path.unlink()
    errors = validate_unified_directory(
        output,
        repository,
        validate_channels=True,
    )
    assert any("upstream input envelope is missing" in item for item in errors)

    synthetic, synthetic_repository, _ = _publish(tmp_path / "synthetic")
    assert validate_unified_directory(synthetic, synthetic_repository) == []


def test_streamed_large_upstream_validation_matches_regular_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output, repository, _ = _publish_with_upstream(tmp_path)
    monkeypatch.setattr(
        "repo_constraint_memory_v3.unified.validation."
        "_STREAMING_JSON_THRESHOLD_BYTES",
        1,
    )

    assert validate_unified_directory(output, repository) == []
    envelope_path = output / "canonical" / "audit" / "upstream-inputs.json"
    envelope = json_load(envelope_path)
    envelope["artifact_digest"] = "0" * 64
    _rewrite_json(envelope_path, envelope)
    errors = validate_unified_directory(output, repository)
    assert any(
        "incorrect unified upstream artifact digest" in item for item in errors
    )


def test_upstream_envelope_digest_binding_and_projection_tamper_are_detected(
    tmp_path: Path,
) -> None:
    output, repository, _ = _publish_with_upstream(tmp_path / "digest")
    envelope_path = output / "canonical" / "audit" / "upstream-inputs.json"
    envelope = json_load(envelope_path)
    envelope["raw_candidate_events"] = 2
    _rewrite_json(envelope_path, envelope)
    errors = validate_unified_directory(output, repository)
    assert any("upstream-inputs.json failed validation" in item for item in errors)

    output2, repository2, _ = _publish_with_upstream(tmp_path / "projection")
    envelope_path2 = output2 / "canonical" / "audit" / "upstream-inputs.json"
    envelope2 = json_load(envelope_path2)
    envelope2["explicit_decisions"][0]["canonical_scope"] = ["tests/**"]
    envelope2.update(
        seal_payload(
            {
                key: value
                for key, value in envelope2.items()
                if key != "artifact_digest"
            }
        )
    )
    _rewrite_json(envelope_path2, envelope2)
    errors = validate_unified_directory(output2, repository2)
    assert any("input envelope" in item for item in errors)

    output3, repository3, _ = _publish_with_upstream(tmp_path / "profile")
    envelope_path3 = output3 / "canonical" / "audit" / "upstream-inputs.json"
    envelope3 = json_load(envelope_path3)
    envelope3["profile_digest"] = "f" * 64
    envelope3.update(
        seal_payload(
            {
                key: value
                for key, value in envelope3.items()
                if key != "artifact_digest"
            }
        )
    )
    _rewrite_json(envelope_path3, envelope3)
    errors = validate_unified_directory(output3, repository3)
    assert "unified upstream input profile mismatch" in errors


def test_unsafe_scope_terminal_omission_replays_and_tampering_fails(
    tmp_path: Path,
) -> None:
    output, repository, bundle = _publish_unsafe_scope_with_upstream(
        tmp_path / "valid"
    )
    assert validate_unified_directory(output, repository) == []
    assert bundle.completeness.package_status == "complete"
    assert bundle.funnel.attrition_reasons[
        "channel_assessment_omitted_invalid_scope"
    ] == 1

    output2, repository2, _ = _publish_unsafe_scope_with_upstream(
        tmp_path / "tampered"
    )
    funnel_path = output2 / "canonical" / "audit" / "funnel.json"
    funnel = json_load(funnel_path)
    funnel["attrition_reasons"] = {
        "candidate_event_not_materialized": 1
    }
    explicit = funnel["by_channel"]["explicit"]
    explicit.pop("attrition:channel_assessment_omitted_invalid_scope")
    explicit["attrition:candidate_event_not_materialized"] = 1
    route = funnel["by_evidence_route"]["direct_normative"]
    route.pop("attrition:channel_assessment_omitted_invalid_scope")
    route["attrition:candidate_event_not_materialized"] = 1
    _rewrite_json(funnel_path, funnel)

    errors = validate_unified_directory(output2, repository2)
    assert any(
        "canonical/audit/funnel.json failed validation" in item
        or "upstream omission replay" in item
        for item in errors
    )

    output3, repository3, _ = _publish_unsafe_scope_with_upstream(
        tmp_path / "capability-tampered"
    )
    completeness_path = output3 / "canonical" / "completeness.json"
    completeness = json_load(completeness_path)
    construction = next(
        item
        for item in completeness["state"]["capabilities"]
        if item["capability"] == "channel_assessment_construction"
    )
    construction["requirement_mode"] = "not_required"
    completeness.update(
        seal_payload(
            {
                key: value
                for key, value in completeness.items()
                if key != "artifact_digest"
            }
        )
    )
    _rewrite_json(completeness_path, completeness)

    errors = validate_unified_directory(output3, repository3)
    assert any(
        "channel-assessment completeness differs from upstream replay" in item
        for item in errors
    )


def test_deep_json_is_reported_as_typed_invalid_artifact(tmp_path: Path) -> None:
    output, repository, _ = _publish(tmp_path)
    (output / "deep.json").write_text(
        "[" * 2_000 + "0" + "]" * 2_000,
        encoding="utf-8",
    )
    errors = validate_unified_directory(output, repository)
    assert any("invalid JSON artifact: deep.json" in item for item in errors)


def test_pipeline_refinement_tree_replays_and_lineage_tampering_fails(
    tmp_path: Path, monkeypatch
) -> None:
    output, repository, bundle = _publish_pipeline_refinement(
        tmp_path / "valid", monkeypatch
    )
    assert validate_unified_directory(output, repository) == []

    output2, repository2, bundle2 = _publish_pipeline_refinement(
        tmp_path / "tampered", monkeypatch
    )
    record = bundle2.refinements[0]
    record_path = (
        output2
        / "canonical"
        / "audit"
        / "refinements"
        / f"{record.refinement_id}.json"
    )
    raw = json_load(record_path)
    raw["change_kinds"] = ["condition"]
    raw.update(
        seal_payload(
            {key: value for key, value in raw.items() if key != "artifact_digest"}
        )
    )
    _rewrite_json(record_path, raw)
    errors = validate_unified_directory(output2, repository2)
    assert any("change kinds differ from proposal" in item for item in errors)

    output3, repository3, _ = _publish_pipeline_refinement(
        tmp_path / "policy-tampered", monkeypatch
    )
    claims_path = output3 / "canonical" / "claims.json"
    wrapper = json_load(claims_path)
    claim = wrapper["claims"][0]
    claim["epistemic_status"] = "review"
    claim["delivery_policy"] = "not_applicable"
    claim["review_facets"] = ["factual_uncertainty"]
    claim.update(
        seal_payload(
            {key: value for key, value in claim.items() if key != "artifact_digest"}
        )
    )
    wrapper.update(
        seal_payload(
            {key: value for key, value in wrapper.items() if key != "artifact_digest"}
        )
    )
    _rewrite_json(claims_path, wrapper)
    errors = validate_unified_directory(output3, repository3)
    assert any("canonical claims differ from frozen policy replay" in item for item in errors)


def test_invalid_optional_refinement_is_quarantined_without_partial_capability(
    tmp_path: Path, monkeypatch
) -> None:
    output, repository, bundle = _publish_pipeline_mixed_refinement(
        tmp_path, monkeypatch
    )
    capability = next(
        item
        for item in bundle.completeness.capabilities
        if item.capability == "unified_refinement"
    )
    assert len(
        [finding for finding in bundle.findings if finding.refinement_proposals]
    ) == 1
    assert len(bundle.refinements) == 1
    assert capability.status == "complete"
    assert capability.reason_codes == []
    quarantines = [
        structured["refinement_quarantine"]
        for request_path in (output / "inference" / "requests").glob("*.json")
        if isinstance(
            structured := json_load(request_path).get("structured_response"),
            dict,
        )
        and isinstance(structured.get("refinement_quarantine"), dict)
    ]
    assert len(quarantines) == 1
    assert quarantines[0]["reason_code"] == "refinement_has_no_change"
    assert isinstance(quarantines[0]["proposal_digest"], str)
    assert validate_unified_directory(output, repository) == []
