from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.agent_runtime import (
    AGENT_RUNNER_REQUEST_SCHEMA_VERSION,
    SOURCE_DISCOVERY_PROMPT_VERSION,
    AgentProfile,
    AgentModelExecutionReceipt,
    AgentRunnerRequest,
    AgentRunnerResult,
    AgentUsage,
    AgentWorkspaceInput,
    SubprocessAgentRuntime,
    agent_request_identity,
    fake_agent_profile,
    load_agent_profile,
    sha256_bytes,
    stable_json_text,
    stable_model_digest,
)
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from repo_constraint_memory_v3.models import AgentStageSettings


_SNAPSHOT = "sha256:" + "1" * 64
_DIGEST = "2" * 64


def _request() -> AgentRunnerRequest:
    profile = fake_agent_profile()
    workspace_inputs = [
        AgentWorkspaceInput(
            path="docs/contributing.rst",
            content_sha256="3" * 64,
            size_bytes=17,
            role="supplement_candidate",
        )
    ]
    workspace_manifest_sha256 = sha256_bytes(
        stable_json_text(
            {
                "inputs": [
                    item.model_dump(mode="json") for item in workspace_inputs
                ]
            }
        ).encode("utf-8")
    )
    payload = {
        "schema_version": AGENT_RUNNER_REQUEST_SCHEMA_VERSION,
        "request_id": "AGENTRUN-" + "0" * 24,
        "attempt_ordinal": 0,
        "stage": "source_discovery",
        "task_context": "task_independent",
        "snapshot_digest": _SNAPSHOT,
        "profile_digest": _DIGEST,
        "execution_model_profile_id": profile.execution_model_profile_id,
        "execution_model_profile_digest": (
            profile.execution_model_profile_digest
        ),
        "framework_id": "test-framework",
        "framework_version": "1.2.3",
        "model_id": profile.model_id,
        "model": profile.model.model_dump(mode="json"),
        "sandbox_profile_id": profile.sandbox_profile_id,
        "sandbox": profile.sandbox.model_dump(mode="json"),
        "prompt_version": SOURCE_DISCOVERY_PROMPT_VERSION,
        "prompt_sha256": "5" * 64,
        "response_schema_sha256": "6" * 64,
        "discovery_sha256": "7" * 64,
        "discovery_view_sha256": "9" * 64,
        "agent_settings_sha256": "8" * 64,
        "workspace_inputs": [
            item.model_dump(mode="json") for item in workspace_inputs
        ],
        "workspace_manifest_sha256": workspace_manifest_sha256,
        "task_path": "TASK.md",
        "discovery_path": "discovery.json",
        "repository_view_path": "repository-view",
        "response_schema_path": "response-schema.json",
        "payload_path": "output/source_supplement.json",
        "trajectory_path": "output/trajectory.json",
        "runner_options": {},
        "limits": AgentStageSettings().model_dump(mode="json"),
    }
    payload["request_id"] = agent_request_identity(payload)
    return AgentRunnerRequest.model_validate(payload)


def _verified_receipt(request: AgentRunnerRequest) -> AgentModelExecutionReceipt:
    return AgentModelExecutionReceipt(
        execution_model_profile_id=request.execution_model_profile_id,
        execution_model_profile_digest=request.execution_model_profile_digest,
        canonical_model_id=request.model.canonical_model_id,
        adapter_model_id=request.model_id,
        reasoning_mode=request.model.reasoning_mode,
        reasoning_effort=request.model.reasoning_effort,
        request_controls_attested=True,
        reasoning_replay_attested=True,
        reported_model_call_count=0,
        model_call_count=0,
        calls=[],
        reasoning_tokens=0,
        reasoning_content_responses=0,
        reasoning_content_bytes=0,
        telemetry_status="not_applicable",
        status="verified",
        violations=[],
    )


def _write_runner(path: Path, body: str) -> Path:
    script = path / "runner.py"
    script.write_text(body, encoding="utf-8")
    return script


def _runtime_profile(script: Path) -> AgentProfile:
    return fake_agent_profile().model_copy(
        update={"runner_command": [sys.executable, script.as_posix()]}
    )


def test_profile_and_usage_reject_ambiguous_or_unsafe_configuration() -> None:
    profile = fake_agent_profile()
    offline_model_profile = load_execution_model_profile(
        "offline-deterministic-test",
        allow_offline=True,
    )

    assert stable_model_digest(profile) == stable_model_digest(
        AgentProfile.model_validate(profile.model_dump(mode="json"))
    )
    assert load_agent_profile(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "agents"
        / "offline_fake.example.yaml",
        model_profile=offline_model_profile,
    ) == profile
    with pytest.raises(ValidationError, match="global configuration paths"):
        AgentProfile.model_validate(
            profile.model_dump(mode="python") | {"environment_keys": ["HOME"]}
        )
    with pytest.raises(ValidationError, match="cannot contain templates"):
        AgentProfile.model_validate(
            profile.model_dump(mode="python")
            | {"runner_command": ["runner", "{request}"]}
        )
    for option_name in ("openai_api_key", "client_secret", "github_token"):
        with pytest.raises(ValidationError, match="secrets must be named"):
            AgentProfile.model_validate(
                profile.model_dump(mode="python")
                | {"runner_options": {option_name: "must-not-be-persisted"}}
            )
    with pytest.raises(ValidationError, match="cannot be embedded"):
        AgentProfile.model_validate(
            profile.model_dump(mode="python")
            | {"runner_command": ["runner", "--token=must-not-be-persisted"]}
        )
    with pytest.raises(ValidationError, match="must be finite"):
        AgentProfile.model_validate(
            profile.model_dump(mode="python")
            | {"runner_options": {"sampling_value": float("nan")}}
        )
    with pytest.raises(ValidationError, match="unknown agent usage"):
        AgentUsage(status="unknown", turns=0)
    with pytest.raises(ValidationError, match="requires every numeric value"):
        AgentUsage(status="known", turns=1)
    with pytest.raises(ValidationError):
        AgentStageSettings(max_output_bytes=10_000_001)
    with pytest.raises(ValidationError):
        AgentStageSettings(max_trajectory_bytes=50_000_001)


def test_profile_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    os.mkfifo(profile_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from repo_constraint_memory_v3.agent_runtime import "
                "load_agent_profile; "
                "from repo_constraint_memory_v3.model_profiles import "
                "load_execution_model_profile; "
                "load_agent_profile("
                "Path(__import__('sys').argv[1]), "
                "model_profile=load_execution_model_profile("
                "'offline-deterministic-test', allow_offline=True))"
            ),
            profile_path.as_posix(),
        ],
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode != 0


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape.json", "/absolute.json", "nested\\windows.json", "a/./b"],
)
def test_request_rejects_unsafe_artifact_paths(unsafe_path: str) -> None:
    payload = _request().model_dump(mode="python")
    payload["payload_path"] = unsafe_path

    with pytest.raises(ValidationError, match="safe relative POSIX"):
        AgentRunnerRequest.model_validate(payload)


def test_request_rejects_workspace_manifest_digest_tamper() -> None:
    payload = _request().model_dump(mode="python")
    payload["workspace_manifest_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="workspace manifest digest"):
        AgentRunnerRequest.model_validate(payload)


def test_request_rejects_stale_source_discovery_prompt_version() -> None:
    payload = _request().model_dump(mode="python")
    payload["prompt_version"] = "source-discovery-agent-1.1"
    payload["request_id"] = agent_request_identity(payload)

    with pytest.raises(ValidationError, match="source-discovery-agent-1.4"):
        AgentRunnerRequest.model_validate(payload)


@pytest.mark.parametrize("unsafe_path", ["../escape.md", "/absolute.md", "a\\b.md"])
def test_workspace_input_rejects_unsafe_repository_paths(
    unsafe_path: str,
) -> None:
    with pytest.raises(ValidationError, match="safe relative POSIX"):
        AgentWorkspaceInput(
            path=unsafe_path,
            content_sha256="3" * 64,
            size_bytes=1,
            role="supplement_candidate",
        )


def test_runner_result_terminal_identity_shape_is_strict() -> None:
    request = _request()
    result = AgentRunnerResult(
        request_id=request.request_id,
        stage=request.stage,
        status="success",
        framework_id=request.framework_id,
        framework_version=request.framework_version,
        execution_model_profile_id=request.execution_model_profile_id,
        execution_model_profile_digest=request.execution_model_profile_digest,
        model_id=request.model_id,
        model_execution=_verified_receipt(request),
        sandbox_profile_id=request.sandbox_profile_id,
        sandbox_status="enforced",
        exit_status="Submitted",
    )

    assert result.usage.status == "unknown"
    with pytest.raises(ValidationError, match="cannot contain errors"):
        AgentRunnerResult.model_validate(
            result.model_dump(mode="python") | {"error_code": "unexpected"}
        )
    with pytest.raises(ValidationError, match="requires error_code"):
        AgentRunnerResult.model_validate(
            result.model_dump(mode="python")
            | {"status": "failed", "exit_status": "ProviderFailure"}
        )


def test_subprocess_runtime_accepts_one_strict_regular_result(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(stable_json_text(request), encoding="utf-8")
    script = _write_runner(
        tmp_path,
        """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text(encoding="utf-8"))
result = {
    "schema_version": "agent-runner-result-1.1",
    "request_id": request["request_id"],
    "stage": request["stage"],
    "status": "success",
    "framework_id": request["framework_id"],
    "framework_version": request["framework_version"],
    "execution_model_profile_id": request["execution_model_profile_id"],
    "execution_model_profile_digest": request["execution_model_profile_digest"],
    "model_id": request["model_id"],
    "model_execution": {
        "schema_version": "agent-model-execution-receipt-1.0",
        "execution_model_profile_id": request["execution_model_profile_id"],
        "execution_model_profile_digest": request["execution_model_profile_digest"],
        "canonical_model_id": request["model"]["canonical_model_id"],
        "adapter_model_id": request["model_id"],
        "reasoning_mode": request["model"]["reasoning_mode"],
        "reasoning_effort": request["model"]["reasoning_effort"],
        "request_controls_attested": True,
        "reasoning_replay_attested": True,
        "reported_model_call_count": 0,
        "model_call_count": 0,
        "calls": [],
        "reasoning_tokens": 0,
        "reasoning_content_responses": 0,
        "reasoning_content_bytes": 0,
        "reasoning_content_retained": False,
        "telemetry_status": "not_applicable",
        "status": "verified",
        "violations": []
    },
    "sandbox_profile_id": request["sandbox_profile_id"],
    "sandbox_status": "enforced",
    "exit_status": "Submitted",
    "usage": {
        "status": "unknown",
        "turns": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "cost_usd": None
    },
    "error_code": None,
    "error_message": None
}
Path(args.result).write_text(json.dumps(result), encoding="utf-8")
print("runner-complete")
""",
    )
    runtime = SubprocessAgentRuntime(_runtime_profile(script))

    outcome = runtime.run(
        request_path,
        result_path,
        cwd=tmp_path,
        timeout_seconds=5,
        max_output_bytes=10_000,
    )

    assert outcome.errors == ()
    assert outcome.exit_code == 0
    assert not outcome.timed_out
    assert outcome.result is not None
    assert outcome.result.request_id == request.request_id
    assert outcome.stdout == b"runner-complete\n"


def test_subprocess_runtime_rejects_symlinked_result(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(stable_json_text(_request()), encoding="utf-8")
    script = _write_runner(
        tmp_path,
        """\
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
target = Path(args.result + ".target")
target.write_text("{}", encoding="utf-8")
os.symlink(target, args.result)
""",
    )
    runtime = SubprocessAgentRuntime(_runtime_profile(script))

    outcome = runtime.run(
        request_path,
        result_path,
        cwd=tmp_path,
        timeout_seconds=5,
        max_output_bytes=10_000,
    )

    assert outcome.result is None
    assert "agent_runner_result_invalid" in outcome.errors


def test_subprocess_runtime_times_out_and_cannot_report_success(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(stable_json_text(_request()), encoding="utf-8")
    script = _write_runner(
        tmp_path,
        """\
import time
time.sleep(30)
""",
    )
    runtime = SubprocessAgentRuntime(_runtime_profile(script))

    outcome = runtime.run(
        request_path,
        result_path,
        cwd=tmp_path,
        timeout_seconds=1,
        max_output_bytes=10_000,
    )

    assert outcome.timed_out
    assert outcome.result is None
    assert "agent_runner_timeout" in outcome.errors
    assert "agent_runner_result_missing" in outcome.errors


@pytest.mark.parametrize(
    ("profile_id", "adapter_key"),
    [
        ("apiyi-qwen3.8-flash-thinking-medium", "OPENAI_API_KEY"),
        ("openrouter-deepseek-v4-flash-0423-nonthinking", "OPENROUTER_API_KEY"),
        ("openrouter-deepseek-v4-flash-0423-gmicloud-fp8-thinking-high", "OPENROUTER_API_KEY"),
        ("openrouter-deepseek-v4-flash-0423-thinking-high", "OPENROUTER_API_KEY"),
        ("openrouter-gemini-2.5-flash-lite-thinking-8192", "OPENROUTER_API_KEY"),
        ("openrouter-glm-5.3-flash-thinking-high", "OPENROUTER_API_KEY"),
        ("openrouter-qwen3.5-flash-02-23-thinking-8192", "OPENROUTER_API_KEY"),
        ("openrouter-qwen3.5-flash-02-23-thinking-enabled", "OPENROUTER_API_KEY"),
        ("apiyi-deepseek-v4-flash-260425-nonthinking", "DEEPSEEK_API_KEY"),
        ("deepseek-flash-nonthinking", "DEEPSEEK_API_KEY"),
    ],
)
def test_real_profile_subprocess_maps_only_selected_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    profile_id: str, adapter_key: str,
) -> None:
    model_profile = load_execution_model_profile(profile_id)
    profile = load_agent_profile(
        Path(__file__).resolve().parents[1]
        / "configs/agents/mini_swe_agent_2_4_6.example.yaml",
        model_profile=model_profile,
    )
    secret = "offline-operator-credential-fixture"
    monkeypatch.setenv("APIYI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-fixture")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unrelated-deepseek-fixture")
    monkeypatch.setenv(model_profile.provider.api_key_env, secret)
    monkeypatch.setenv("UNRELATED_SECRET", "unrelated-secret-fixture")
    script = tmp_path / "credential_probe.py"
    script.write_text(
        "import os, sys\n"
        f"key = {adapter_key!r}\n"
        "assert 'APIYI_API_KEY' not in os.environ\n"
        "assert 'UNRELATED_SECRET' not in os.environ\n"
        "other = 'DEEPSEEK_API_KEY' if key == 'OPENAI_API_KEY' else 'OPENAI_API_KEY'\n"
        "assert other not in os.environ\n"
        f"assert os.environ[key] == {secret!r}\n"
        "print(os.environ[key])\n"
        "print(os.environ[key], file=sys.stderr)\n",
        encoding="utf-8",
    )
    runtime = SubprocessAgentRuntime(profile.model_copy(update={
        "runner_command": [sys.executable, str(script)],
    }))
    outcome = runtime.run(
        tmp_path / "unused-request.json", tmp_path / "unused-result.json",
        cwd=tmp_path, timeout_seconds=10, max_output_bytes=4096,
    )
    # This probe intentionally emits no runner result; it exercises real Popen,
    # environment isolation, credential selection, and captured-output redaction.
    assert outcome.exit_code == 0
    assert outcome.errors == ("agent_runner_result_missing",)
    assert outcome.stdout == b"[REDACTED]\n"
    assert outcome.stderr == b"[REDACTED]\n"
    assert secret not in profile.model_dump_json()
