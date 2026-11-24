from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from helpers import (
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
    copied_fixture,
)
from repo_constraint_memory_v3.agent_runtime import (
    fake_agent_profile,
    stable_json_text,
)
from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.builder import load_build_config
from repo_constraint_memory_v3.inference import InferenceStageCounts
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from repo_constraint_memory_v3.models import BuildConfig, ProviderResponse
from repo_constraint_memory_v3.unified import rebuild as rebuild_module
from repo_constraint_memory_v3.unified import validation as validation_module
from repo_constraint_memory_v3.unified.export import (
    canonical_json_text,
    write_unified_outputs,
)
from repo_constraint_memory_v3.unified.ids import content_digest, seal_payload
from repo_constraint_memory_v3.unified.publication import (
    directory_tree_fingerprint,
)
from repo_constraint_memory_v3.unified.rebuild import (
    _merge_rebuilt_stage_omissions,
    rebuild_unified,
)
from repo_constraint_memory_v3.unified.validation import validate_unified_directory

from test_unified_export import _bundle


@dataclass(frozen=True)
class _FullRun:
    repository: Path
    run: Path
    config: Path
    profile: Path
    package_status: str


def _model_free_config(config: BuildConfig) -> dict[str, object]:
    """Serialize a resolved test config back into the active input contract."""

    value = config.model_dump(mode="json")
    value.pop("execution_model_profile_id")
    value.pop("execution_model_profile_digest")
    llm = value["llm"]
    assert isinstance(llm, dict)
    for key in (
        "provider",
        "model",
        "require_response_model",
        "accepted_effective_model_ids",
        "api_key_env",
        "base_url",
        "temperature",
        "thinking_mode",
        "reasoning_effort",
        "reasoning_wire_policy",
    ):
        llm.pop(key)
    return value


def _model_free_fake_agent_profile() -> dict[str, object]:
    value = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        value.pop(key)
    return value


def _load_offline_build_config(path: Path) -> BuildConfig:
    return load_build_config(
        path,
        model_profile=load_execution_model_profile(
            "offline-deterministic-test",
            allow_offline=True,
        ),
    )


class _OneUncertainExplicitVerifierClient(MockLLMClient):
    """Keep one fixture member verifier-eligible after local verification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emitted_uncertain = False

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") != "explicit_semantic_verify":
            return response
        with self._lock:
            if self._emitted_uncertain:
                return response
            self._emitted_uncertain = True
        body = json.loads(response.content)
        body["results"][0]["coding_relevance"] = "uncertain"
        return response.model_copy(
            update={
                "content": json.dumps(
                    body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            }
        )


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> _FullRun:
    root = tmp_path_factory.mktemp("unified-cli")
    repository = copied_fixture(root)
    config = config_for_tests(
        observation={"enabled": False, "required": False}
    )
    run = root / "full"
    result = build_fixture(
        repository,
        run,
        config=config,
        client=_OneUncertainExplicitVerifierClient(),
    )
    config_path = root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_model_free_config(config), sort_keys=False),
        encoding="utf-8",
    )
    profile_path = root / "profile.json"
    profile_path.write_text(
        stable_json_text(_model_free_fake_agent_profile()), encoding="utf-8"
    )
    return _FullRun(
        repository=repository,
        run=run,
        config=config_path,
        profile=profile_path,
        package_status=result.unified.bundle.completeness.package_status,
    )


def _build_unified_args(full: _FullRun, output: Path) -> list[str]:
    return [
        "build-unified",
        str(full.repository),
        "--config",
        str(full.config),
        "--agent-profile",
        str(full.profile),
        "--upstream-run",
        str(full.run),
        "--out",
        str(output),
        "--mock",
    ]


def _consumer_files(root: Path) -> dict[str, bytes]:
    fixed = {
        relative: (root / relative).read_bytes()
        for relative in (
            "canonical/claims.json",
            "canonical/completeness.json",
            "canonical/delivery_report.json",
        )
    }
    fixed.update(
        {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted((root / "consumers").rglob("*"))
            if path.is_file()
        }
    )
    return fixed


def _rewrite_canonical(path: Path, value: object) -> None:
    path.chmod(0o644)
    path.write_text(canonical_json_text(value), encoding="utf-8")


def test_build_unified_replaces_upstream_unified_stage_omissions() -> None:
    upstream_stages = [
        InferenceStageCounts(
            stage="population_semantic_classify",
            planned=5,
            success=0,
            uncertain=0,
            failure=0,
            omitted=5,
            omission_reasons={"max_total_input_tokens": 5},
        ),
        InferenceStageCounts(
            stage="unified_case_verify",
            planned=7,
            success=0,
            uncertain=0,
            failure=0,
            omitted=7,
            omission_reasons={"max_total_input_tokens": 7},
        ),
    ]

    merged = _merge_rebuilt_stage_omissions(
        upstream_stages,
        {
            "unified_case_verify": (2, {"max_total_input_tokens": 2}),
            "temporal_evidence_classify": (
                3,
                {"max_total_input_tokens": 3},
            ),
        },
    )

    assert merged == {
        "population_semantic_classify": (
            5,
            {"max_total_input_tokens": 5},
        ),
        "temporal_evidence_classify": (
            3,
            {"max_total_input_tokens": 3},
        ),
        "unified_case_verify": (
            2,
            {"max_total_input_tokens": 2},
        ),
    }


def _reseal_artifact(value: dict[str, object]) -> None:
    value.update(
        seal_payload(
            {
                key: item
                for key, item in value.items()
                if key != "artifact_digest"
            }
        )
    )


def _recompute_unified_index(run: Path) -> None:
    digests: list[str] = []
    for directory in (
        "channel-assessments",
        "cases",
        "current-evidence",
        "git/relations",
        "findings",
        "refinements",
        "git/bundles",
    ):
        root = run / "canonical" / "audit" / directory
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            digests.append(value["artifact_digest"])
    claims = json.loads(
        (run / "canonical/claims.json").read_text(encoding="utf-8")
    )
    digests.extend(item["artifact_digest"] for item in claims["claims"])
    funnel = json.loads(
        (run / "canonical/audit/funnel.json").read_text(encoding="utf-8")
    )
    completeness = json.loads(
        (run / "canonical/completeness.json").read_text(encoding="utf-8")
    )
    digests.append(content_digest(funnel))
    digests.append(content_digest(completeness["state"]))
    index_path = run / "canonical/audit/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifact_set_digest"] = content_digest(sorted(digests))
    _reseal_artifact(index)
    _rewrite_canonical(index_path, index)


def test_build_unified_publishes_without_implicit_full_replay(
    full_run: _FullRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated = tmp_path / "isolated"
    expected_exit = 0 if full_run.package_status == "complete" else 1

    def forbidden_replay(*_: object, **__: object) -> list[str]:
        raise AssertionError("build-unified must not invoke full offline replay")

    monkeypatch.setattr(
        validation_module,
        "validate_unified_directory",
        forbidden_replay,
    )

    assert main(_build_unified_args(full_run, isolated)) == expected_exit
    assert _consumer_files(isolated) == _consumer_files(full_run.run)
    assert (
        json.loads(
            (isolated / "canonical/audit/index.json").read_text(encoding="utf-8")
        )[
            "artifact_set_digest"
        ]
        == json.loads(
            (full_run.run / "canonical/audit/index.json").read_text(
                encoding="utf-8"
            )
        )["artifact_set_digest"]
    )
    assert (isolated / "inference/index.json").is_file()
    assert not (isolated / "unified").exists()
    before_validation = directory_tree_fingerprint(
        isolated,
        max_bytes=1_000_000_000,
    )

    root_inference = json.loads(
        (isolated / "inference/index.json").read_text(encoding="utf-8")
    )
    assert {
        path.name
        for path in (isolated / "inference/requests").iterdir()
        if path.is_file()
    } == {
        f"{item['request_id']}.json" for item in root_inference["requests"]
    }
    assert main(
        [
            "validate-unified",
            str(isolated),
            "--repository",
            str(full_run.repository),
        ]
    ) == 0
    assert main(
        ["validate", str(isolated), "--repository", str(full_run.repository)]
    ) == 0
    strict_exit = 0 if full_run.package_status == "complete" else 1
    assert main(
        [
            "validate-unified",
            str(isolated),
            "--repository",
            str(full_run.repository),
            "--require-complete",
        ]
    ) == strict_exit
    assert directory_tree_fingerprint(
        isolated,
        max_bytes=1_000_000_000,
    ) == before_validation

    tampered = tmp_path / "tampered-isolated"
    shutil.copytree(isolated, tampered)
    request = next((tampered / "inference/requests").glob("*.json"))
    value = json.loads(request.read_text(encoding="utf-8"))
    value["request_ordinal"] += 1
    request.write_text(canonical_json_text(value), encoding="utf-8")
    assert main(
        [
            "validate-unified",
            str(tampered),
            "--repository",
            str(full_run.repository),
        ]
    ) == 2


def test_build_unified_emit_consumer_profile_writes_both_views(
    full_run: _FullRun,
    tmp_path: Path,
) -> None:
    isolated = tmp_path / "isolated-plus-review"
    args = _build_unified_args(full_run, isolated)
    args.extend(["--emit-consumer-profile", "accepted_plus_review"])
    expected_exit = 0 if full_run.package_status == "complete" else 1

    assert main(args) == expected_exit
    assert (
        isolated / "canonical/claims.json"
    ).read_bytes() == (
        full_run.run / "canonical/claims.json"
    ).read_bytes()
    assert (
        isolated / "canonical/completeness.json"
    ).read_bytes() == (
        full_run.run / "canonical/completeness.json"
    ).read_bytes()
    report = json.loads(
        (isolated / "canonical/delivery_report.json").read_text(
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
    for profile in report["emitted_consumer_profiles"]:
        manifest = json.loads(
            (
                isolated
                / "consumers"
                / profile
                / "consumer-manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["consumer_profile"] == profile
        assert manifest["emitted_consumer_profiles"] == [
            "accepted_only",
            "accepted_plus_review",
        ]
    assert validate_unified_directory(isolated, full_run.repository) == []


def test_validator_replays_unified_responses_after_coordinated_digest_tamper(
    full_run: _FullRun,
    tmp_path: Path,
) -> None:
    grouping_run = tmp_path / "grouping-tamper"
    shutil.copytree(full_run.run, grouping_run)
    grouping_path = next(
        path
        for path in sorted(
            (grouping_run / "canonical/audit/cases").glob("*.json")
        )
        if any(
            item["source"] == "semantic_grouping"
            for item in json.loads(path.read_text(encoding="utf-8"))["relations"]
        )
    )
    grouping = json.loads(grouping_path.read_text(encoding="utf-8"))
    relation = next(
        item
        for item in grouping["relations"]
        if item["source"] == "semantic_grouping"
    )
    relation["reason_code"] += "_coordinated_tamper"
    _reseal_artifact(grouping)
    _rewrite_canonical(grouping_path, grouping)
    _recompute_unified_index(grouping_run)
    errors = validate_unified_directory(grouping_run, full_run.repository)
    assert any("candidate grouping" in item for item in errors)

    temporal_run = tmp_path / "temporal-tamper"
    shutil.copytree(full_run.run, temporal_run)
    temporal_paths = sorted(
        (temporal_run / "canonical/audit/git/relations").glob("*.json")
    )
    if temporal_paths:
        temporal_path = temporal_paths[0]
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal["relations"][0]["reason_code"] += "_coordinated_tamper"
        _reseal_artifact(temporal)
        _rewrite_canonical(temporal_path, temporal)
        _recompute_unified_index(temporal_run)
        errors = validate_unified_directory(temporal_run, full_run.repository)
        assert any("temporal" in item and "response" in item for item in errors)

    finding_run = tmp_path / "finding-tamper"
    shutil.copytree(full_run.run, finding_run)
    finding_paths = [
        path
        for path in sorted(
            (finding_run / "canonical/audit/findings").glob("*.json")
        )
        if json.loads(path.read_text(encoding="utf-8")).get("inference_ref")
    ]
    assert finding_paths
    finding_path = finding_paths[0]
    finding = json.loads(finding_path.read_text(encoding="utf-8"))
    reason_codes = finding["member_findings"][0]["reason_codes"]
    reason_codes.append("coordinated_tamper")
    reason_codes.sort()
    _reseal_artifact(finding)
    _rewrite_canonical(finding_path, finding)
    _recompute_unified_index(finding_run)
    errors = validate_unified_directory(finding_run, full_run.repository)
    assert any(
        "saved unified finding differs from response" in item
        for item in errors
    )


@pytest.mark.parametrize(
    ("relative_path", "expected_error"),
    [
        ("audit.jsonl", "audit.jsonl differs from deterministic adjudication replay"),
        (
            "evidence_relations.json",
            "evidence_relations.json differs from adjudication replay",
        ),
    ],
)
def test_validator_replays_non_consumer_channel_audit_artifacts(
    full_run: _FullRun,
    tmp_path: Path,
    relative_path: str,
    expected_error: str,
) -> None:
    tampered = tmp_path / relative_path.replace(".", "-")
    shutil.copytree(full_run.run, tampered)
    path = tampered / relative_path
    if relative_path == "audit.jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert records
        records[0]["reason_code"] += "_tampered"
        path.write_text(
            "".join(canonical_json_text(item) for item in records),
            encoding="utf-8",
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["relations"].append(
            {
                "relation": "corroborates",
                "route": "explicit_typed_evidence",
                "source_fingerprint": "0" * 64,
                "target_fingerprint": "1" * 64,
                "reason": "coordinated tamper",
                "effect": "advisory",
            }
        )
        _rewrite_canonical(path, payload)

    assert expected_error in validate_unified_directory(
        tampered, full_run.repository
    )


def test_build_unified_replays_current_source_canonical_consumer_surface(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    full = tmp_path / "current-full"
    full_result = build_fixture(repository, full, config=config)
    implicit_assessment_ids = {
        item.assessment_id
        for item in full_result.unified.bundle.assessments
        if item.channel == "implicit_current"
    }
    assert implicit_assessment_ids
    implicit_case_ids = {
        case.case_id
        for case in full_result.unified.bundle.cases
        if any(
            member.assessment_id in implicit_assessment_ids
            for member in case.members
        )
    }
    assert any(
        claim.case_id in implicit_case_ids
        and claim.epistemic_status == "accepted"
        and claim.normative_strength == "observed"
        for claim in full_result.unified.bundle.canonical_claims
    )

    config_path = tmp_path / "current-config.yaml"
    config_path.write_text(
        yaml.safe_dump(_model_free_config(config), sort_keys=False),
        encoding="utf-8",
    )
    profile_path = tmp_path / "current-profile.json"
    profile_path.write_text(
        stable_json_text(_model_free_fake_agent_profile()), encoding="utf-8"
    )
    isolated = tmp_path / "current-isolated"
    args = [
        "build-unified",
        str(repository),
        "--config",
        str(config_path),
        "--agent-profile",
        str(profile_path),
        "--upstream-run",
        str(full),
        "--out",
        str(isolated),
        "--mock",
    ]
    expected_exit = (
        0
        if full_result.unified.bundle.completeness.package_status == "complete"
        else 1
    )
    assert main(args) == expected_exit
    assert _consumer_files(isolated) == _consumer_files(full)
    for relative in (
        "canonical/claims.json",
        "canonical/audit/member-dispositions.json",
    ):
        assert (isolated / relative).read_bytes() == (full / relative).read_bytes()
    isolated_index = json.loads(
        (isolated / "canonical/audit/index.json").read_text(encoding="utf-8")
    )
    assert (
        isolated_index["artifact_set_digest"]
        == full_result.unified.bundle.index.artifact_set_digest
    )


class _DifferentGroupingResponseClient:
    def __init__(self) -> None:
        self.delegate = MockLLMClient()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = self.delegate.complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "candidate_semantic_group":
            return response
        # Response 1.2 intentionally has no semantically inert free-text
        # rationale. Vary terminal usage instead: this still proves that an
        # isolated rerun replaces the same request ID with a distinct valid
        # record while leaving the typed relation result unchanged.
        return ProviderResponse(
            content=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens + 1,
            reasoning_tokens=response.reasoning_tokens,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
        )

    def complete_with_settings(self, system_prompt, user_prompt, *, settings):
        del settings
        return self.complete(system_prompt, user_prompt)


class _NoCallClient:
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt, user_prompt
        raise AssertionError("compatible unified resume should avoid provider calls")

    def complete_with_settings(self, system_prompt, user_prompt, *, settings):
        del settings
        return self.complete(system_prompt, user_prompt)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("extract_units", "response differs from its request input units"),
        ("verifier_response", "result differs from inference artifacts"),
    ],
)
def test_build_unified_rejects_tampered_channel_response_bindings_before_calls(
    full_run: _FullRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    original_loader = rebuild_module._validated_channel_inference

    def tampered_loader(*args, **kwargs):
        index, records = original_loader(*args, **kwargs)
        changed = list(records)
        if mutation == "extract_units":
            targets = [
                ordinal
                for ordinal, record in enumerate(changed)
                if record.stage == "explicit_extract"
                and record.terminal_status == "success"
            ]
            assert len(targets) >= 2
            left, right = targets[:2]
            left_inputs = changed[left].input_manifest
            right_inputs = changed[right].input_manifest
            changed[left] = changed[left].model_copy(
                update={"input_manifest": right_inputs}
            )
            changed[right] = changed[right].model_copy(
                update={"input_manifest": left_inputs}
            )
        else:
            target = next(
                ordinal
                for ordinal, record in enumerate(changed)
                if record.stage == "explicit_semantic_verify"
                and record.terminal_status == "success"
            )
            response = json.loads(
                json.dumps(changed[target].structured_response)
            )
            judgment = response["results"][0]
            judgment["coding_relevance"] = (
                "fail"
                if judgment["coding_relevance"] != "fail"
                else "pass"
            )
            changed[target] = changed[target].model_copy(
                update={"structured_response": response}
            )
        return index, tuple(changed)

    monkeypatch.setattr(
        rebuild_module,
        "_validated_channel_inference",
        tampered_loader,
    )
    with pytest.raises(
        rebuild_module.UnifiedRebuildError,
        match=expected_error,
    ):
        rebuild_unified(
            full_run.repository,
            _load_offline_build_config(full_run.config),
            fake_agent_profile(),
            full_run.run,
            _NoCallClient(),
            tmp_path / f"tampered-{mutation}",
        )


class _ConcurrentProbeClient(MockLLMClient):
    """Use the normal mock except for an explicit concurrency probe."""

    _PROMPT = "__build_unified_concurrency_probe__"

    def __init__(self, workers: int) -> None:
        self.barrier = threading.Barrier(workers)

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if user_prompt != self._PROMPT:
            return super().complete(system_prompt, user_prompt)
        self.barrier.wait(timeout=5)
        return ProviderResponse(
            content="{}",
            model=self.model,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            latency_ms=1,
            finish_reason="stop",
        )


def test_build_unified_resumes_from_root_inference_tree_without_calls(
    full_run: _FullRun,
    tmp_path: Path,
) -> None:
    output = tmp_path / "resumed"
    result = rebuild_unified(
        full_run.repository,
        _load_offline_build_config(full_run.config),
        fake_agent_profile(),
        full_run.run,
        _NoCallClient(),
        output,
        resume_from=full_run.run,
    )
    assert result.unified.inference_records
    assert _consumer_files(output) == _consumer_files(full_run.run)
    assert not (output / "unified").exists()
    report = json.loads(
        (output / "build_report.json").read_text(encoding="utf-8")
    )
    assert report["llm"]["observed_max_concurrency"] == 0


def test_build_unified_reports_actual_parallel_client_peak(
    full_run: _FullRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = 3
    client = _ConcurrentProbeClient(workers)
    original_run_unified_layer = rebuild_module.run_unified_layer

    def run_with_parallel_probe(**kwargs):
        tracked_client = kwargs["client"]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(
                executor.map(
                    lambda _: tracked_client.complete("", client._PROMPT),
                    range(workers),
                )
            )
        return original_run_unified_layer(**kwargs)

    monkeypatch.setattr(
        rebuild_module,
        "run_unified_layer",
        run_with_parallel_probe,
    )
    output = tmp_path / "parallel-peak"
    rebuild_unified(
        full_run.repository,
        _load_offline_build_config(full_run.config),
        fake_agent_profile(),
        full_run.run,
        client,
        output,
    )

    report = json.loads(
        (output / "build_report.json").read_text(encoding="utf-8")
    )
    assert report["llm"]["observed_max_concurrency"] == workers


def test_isolated_replaces_same_request_id_with_new_valid_response(
    full_run: _FullRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "different-response"
    upstream_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((full_run.run / "inference/requests").glob("*.json"))
    ]
    channel_records = [
        item
        for item in upstream_records
        if item["stage"]
        not in {
            "candidate_semantic_group",
            "temporal_evidence_classify",
            "unified_case_verify",
        }
    ]
    expected_prior_usage = (
        sum(item["usage"]["calls"] for item in channel_records),
        sum(item["usage"]["input_tokens"] for item in channel_records),
        sum(item["usage"]["output_tokens"] for item in channel_records),
    )
    observed_prior_usage: tuple[int, int, int] | None = None
    original_run_unified_layer = rebuild_module.run_unified_layer

    def capture_budget(**kwargs):
        nonlocal observed_prior_usage
        budget = kwargs["budget"]
        observed_prior_usage = (
            budget.calls,
            budget.input_tokens,
            budget.output_tokens,
        )
        return original_run_unified_layer(**kwargs)

    monkeypatch.setattr(rebuild_module, "run_unified_layer", capture_budget)
    result = rebuild_unified(
        full_run.repository,
        _load_offline_build_config(full_run.config),
        fake_agent_profile(),
        full_run.run,
        _DifferentGroupingResponseClient(),
        output,
    )
    assert observed_prior_usage == expected_prior_usage
    assert _consumer_files(output) == _consumer_files(full_run.run)
    assert not (output / "unified").exists()

    upstream_index = json.loads(
        (full_run.run / "inference/index.json").read_text(encoding="utf-8")
    )
    rebuilt_index = json.loads(
        (output / "inference/index.json").read_text(encoding="utf-8")
    )
    upstream_group = {
        item["request_id"]: item
        for item in upstream_index["requests"]
        if item["stage"] == "candidate_semantic_group"
    }
    rebuilt_group = {
        item["request_id"]: item
        for item in rebuilt_index["requests"]
        if item["stage"] == "candidate_semantic_group"
    }
    assert set(rebuilt_group) == set(upstream_group)
    assert any(
        rebuilt_group[request_id]["record_digest"]
        != upstream_group[request_id]["record_digest"]
        for request_id in rebuilt_group
    )
    for request_id, reference in rebuilt_group.items():
        record = json.loads(
            (output / reference["artifact_path"]).read_text(encoding="utf-8")
        )
        assert record["request_id"] == request_id
        assert record["record_digest"] == reference["record_digest"]
    assert len(result.inference_records) == len(rebuilt_index["requests"])

    upstream_report = json.loads(
        (full_run.run / "build_report.json").read_text(encoding="utf-8")
    )
    rebuilt_report = json.loads(
        (output / "build_report.json").read_text(encoding="utf-8")
    )
    assert upstream_report["entrypoint"] == "full"
    assert rebuilt_report["entrypoint"] == "build-unified"
    assert rebuilt_report["llm"]["observed_max_concurrency"] == 1
    assert rebuilt_report["llm"]["record_count"] == len(
        rebuilt_index["requests"]
    )
    assert (
        rebuilt_report["llm"]["inference_set_digest"]
        == rebuilt_index["inference_set_digest"]
    )
    assert rebuilt_report["llm"]["prior_channel_usage"] == {
        "record_count": len(channel_records),
        "calls": expected_prior_usage[0],
        "input_tokens": expected_prior_usage[1],
        "output_tokens": expected_prior_usage[2],
        "reasoning_tokens": (
            sum(item["usage"]["reasoning_tokens"] for item in channel_records)
            if all(
                item["usage"]["reasoning_tokens"] is not None
                for item in channel_records
            )
            else None
        ),
    }
    assert rebuilt_report["unified"]["completeness"] == json.loads(
        (output / "canonical/completeness.json").read_text(encoding="utf-8")
    )["state"]
    assert validate_unified_directory(output, full_run.repository) == []

    tampered_report_run = tmp_path / "tampered-report"
    shutil.copytree(output, tampered_report_run)
    report_path = tampered_report_run / "build_report.json"
    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["llm"]["record_count"] += 1
    _rewrite_canonical(report_path, tampered_report)
    assert any(
        "build_report llm record_count differs" in item
        for item in validate_unified_directory(
            tampered_report_run,
            full_run.repository,
        )
    )


def test_build_unified_rejects_config_profile_envelope_and_snapshot_mismatch(
    full_run: _FullRun,
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(full_run.config.read_text(encoding="utf-8"))
    config["unified"]["git_enabled"] = not config["unified"]["git_enabled"]
    wrong_config = tmp_path / "wrong-config.yaml"
    wrong_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    args = _build_unified_args(full_run, tmp_path / "wrong-config-output")
    args[args.index(str(full_run.config))] = str(wrong_config)
    assert main(args) == 2

    endpoint_config = yaml.safe_load(full_run.config.read_text(encoding="utf-8"))
    endpoint_config["llm"]["base_url"] = "https://other-endpoint.invalid/v1"
    wrong_endpoint = tmp_path / "wrong-endpoint.yaml"
    wrong_endpoint.write_text(
        yaml.safe_dump(endpoint_config, sort_keys=False), encoding="utf-8"
    )
    args = _build_unified_args(full_run, tmp_path / "wrong-endpoint-output")
    args[args.index(str(full_run.config))] = str(wrong_endpoint)
    assert main(args) == 2

    profile = _model_free_fake_agent_profile()
    profile["profile_id"] = "different-profile"
    wrong_profile = tmp_path / "wrong-profile.json"
    wrong_profile.write_text(stable_json_text(profile), encoding="utf-8")
    args = _build_unified_args(full_run, tmp_path / "wrong-profile-output")
    args[args.index(str(full_run.profile))] = str(wrong_profile)
    assert main(args) == 2

    tampered_run = tmp_path / "tampered-upstream"
    shutil.copytree(full_run.run, tampered_run)
    envelope_path = tampered_run / "canonical/audit/upstream-inputs.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["raw_candidate_events"] += 1
    # A valid outer self-digest is insufficient: build-unified must bind the
    # envelope back to the physical channel/inference artifacts it projects.
    _reseal_artifact(envelope)
    _rewrite_canonical(envelope_path, envelope)
    args = _build_unified_args(full_run, tmp_path / "tampered-output")
    args[args.index(str(full_run.run))] = str(tampered_run)
    assert main(args) == 2

    source = full_run.repository / "README.md"
    original = source.read_bytes()
    try:
        source.write_text("# snapshot changed\n", encoding="utf-8")
        assert main(
            _build_unified_args(full_run, tmp_path / "snapshot-output")
        ) == 2
    finally:
        source.write_bytes(original)


def test_validator_rejects_nested_observation_envelope_tamper(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests(
        semantic={"enabled": False, "required": False},
        unified={"git_enabled": False},
    )
    run = tmp_path / "observation-run"
    build_fixture(repository, run, config=config)

    envelope_path = run / "canonical/audit/upstream-inputs.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["observation_shards"] is not None
    envelope["observation_shards"]["observation_set_digest"] = "f" * 64
    # Re-seal the envelope so this reaches the nested channel-replay binding,
    # rather than failing only its outer self-digest.
    _reseal_artifact(envelope)
    _rewrite_canonical(envelope_path, envelope)

    errors = validate_unified_directory(run, repository)
    assert any(
        "upstream observation shards differ from channel replay" in item
        for item in errors
    ), errors


def test_build_unified_rejects_cache_overlap_without_mutating_upstream(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    immutable_parent = tmp_path / "immutable"
    upstream = immutable_parent / "upstream"
    upstream.mkdir(parents=True)
    marker = upstream / "marker.json"
    marker.write_text('{"immutable":true}\n', encoding="utf-8")
    before = marker.read_bytes()

    for cache in (upstream / "cache", immutable_parent):
        with pytest.raises(
            rebuild_module.UnifiedRebuildError,
            match="immutable upstream run",
        ):
            rebuild_module._validate_paths(
                repository,
                upstream,
                tmp_path / f"output-{cache.name}",
                inference_cache=cache,
                resume_from=None,
            )
        assert marker.read_bytes() == before
        assert not (upstream / "cache").exists()

    resolved = rebuild_module._validate_paths(
        repository,
        upstream,
        tmp_path / "resume-output",
        inference_cache=None,
        resume_from=upstream,
    )
    assert resolved[1] == upstream.resolve()
    assert marker.read_bytes() == before


def test_unified_cli_help_and_bounded_upstream_copy(tmp_path: Path) -> None:
    for command in ("build-unified", "validate-unified"):
        with pytest.raises(SystemExit) as exit_signal:
            main([command, "--help"])
        assert exit_signal.value.code == 0

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    repository = tmp_path / "repository"
    repository.mkdir()
    (upstream / "oversized.txt").write_bytes(b"x" * 1_024)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="remaining byte budget"):
        write_unified_outputs(
            output,
            bundle=_bundle(),
            max_artifact_bytes=512,
            repository_root=repository,
            upstream_directory=upstream,
        )
    assert not output.exists()

    regular = tmp_path / "regular.txt"
    regular.write_text("validated upstream", encoding="utf-8")
    symlink_upstream = tmp_path / "symlink-upstream"
    symlink_upstream.mkdir()
    (symlink_upstream / "escape.txt").symlink_to(regular)
    with pytest.raises(ValueError, match="upstream artifact is a symlink"):
        write_unified_outputs(
            tmp_path / "symlink-output",
            bundle=_bundle(),
            max_artifact_bytes=10_000_000,
            repository_root=repository,
            upstream_directory=symlink_upstream,
        )
