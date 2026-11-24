"""Framework-agnostic, subprocess-isolated agent runner protocol.

The core owns request construction, process bounds, artifact hashing, and
validation.  A framework-specific runner is an external executable that reads
one request and writes one strict result; the target repository is never passed
to it directly.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import signal
import stat
import string
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Mapping, Protocol

import yaml
from pydantic import Field, field_validator, model_serializer, model_validator

from repo_constraint_memory_v3.models import (
    AgentStageSettings,
    NonEmptyString,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.model_profiles import ExecutionModelProfile


AGENT_PROFILE_SCHEMA_VERSION = "agent-profile-1.1"
AGENT_RUNNER_PROTOCOL_VERSION = "norm4code-agent-runner-1.2"
AGENT_RUNNER_REQUEST_SCHEMA_VERSION = "agent-runner-request-1.1"
AGENT_RUNNER_RESULT_SCHEMA_VERSION = "agent-runner-result-1.1"
AGENT_RUN_ARTIFACT_SCHEMA_VERSION = "agent-run-artifact-1.1"
SOURCE_ACQUISITION_SCHEMA_VERSION = "source-acquisition-1.2"
AGENT_MODEL_EXECUTION_RECEIPT_SCHEMA_VERSION = (
    "agent-model-execution-receipt-1.0"
)
SOURCE_DISCOVERY_PROMPT_VERSION = "source-discovery-agent-1.4"
SOURCE_DISCOVERY_STAGE = "source_discovery"

AgentRunId = Annotated[str, Field(pattern=r"^AGENTRUN-[0-9a-f]{24}$")]


def agent_request_identity(payload: Mapping[str, object]) -> str:
    """Derive an ID from every request field except the ID itself."""

    canonical = dict(payload)
    canonical.pop("request_id", None)
    return f"AGENTRUN-{sha256_bytes(stable_json_text(canonical).encode('utf-8'))[:24]}"


class AgentSandboxPolicy(StrictModel):
    """Sandbox contract for the agent's repository-inspection tools."""

    filesystem_scope: Literal["filtered_workspace"] = "filtered_workspace"
    target_repository_access: Literal["absent"] = "absent"
    network_access: Literal["denied"] = "denied"
    environment_access: Literal["allowlist"] = "allowlist"


class AgentWorkspaceInput(StrictModel):
    path: NonEmptyString
    content_sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    role: Literal["baseline_source", "supplement_candidate"]

    @field_validator("path")
    @classmethod
    def path_is_safe_relative_posix(cls, value: str) -> str:
        path = Path(value)
        if (
            path.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("workspace input path must be safe relative POSIX")
        return value


def stable_json_text(value: object) -> str:
    """Return canonical UTF-8 JSON text for agent trust-boundary artifacts."""

    if isinstance(value, StrictModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_bounded_bytes(path: Path, *, limit: int) -> bytes:
    """Read one non-symlink regular file through a single bounded descriptor."""

    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("path is not a regular file")
        if info.st_size > limit:
            raise ValueError("file exceeds its byte limit")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            value = handle.read(limit + 1)
        if len(value) > limit:
            raise ValueError("file exceeds its byte limit")
        return value
    except OSError as error:
        if isinstance(error, FileNotFoundError):
            raise
        if error.errno == errno.ELOOP:
            raise ValueError("path is not a regular file") from error
        raise ValueError("regular file cannot be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stable_model_digest(value: StrictModel) -> str:
    return sha256_bytes(stable_json_text(value).encode("utf-8"))


class AgentModelBinding(StrictModel):
    """Secret-free mini-swe model settings resolved from one global profile."""

    canonical_model_id: NonEmptyString
    provider_id: NonEmptyString
    provider_protocol: Literal[
        "openai_chat_completions",
        "openai_responses",
        "offline_mock",
    ]
    base_url: NonEmptyString | None
    api_key_env: NonEmptyString | None
    model_id: NonEmptyString
    temperature: int | float
    reasoning_mode: Literal["disabled", "enabled"]
    reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None
    reasoning_max_tokens: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    reasoning_wire_policy: Literal[
        "deepseek_thinking_type",
        "qwen_enable_thinking",
        "openrouter_reasoning",
        "openai_responses_reasoning_effort",
        "offline_mock",
    ]
    require_response_model: bool
    accepted_effective_model_ids: Annotated[
        list[NonEmptyString], Field(min_length=1)
    ]

    @field_validator("accepted_effective_model_ids")
    @classmethod
    def effective_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("accepted effective model IDs must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def reasoning_and_provider_contract(self) -> AgentModelBinding:
        if self.reasoning_mode == "disabled" and (
            self.reasoning_effort is not None
            or self.reasoning_max_tokens is not None
        ):
            raise ValueError("disabled agent reasoning cannot carry a control")
        if (
            self.reasoning_effort is not None
            and self.reasoning_max_tokens is not None
        ):
            raise ValueError("agent reasoning controls are mutually exclusive")
        if (
            self.reasoning_max_tokens is not None
            and self.reasoning_wire_policy != "openrouter_reasoning"
        ):
            raise ValueError("agent reasoning max_tokens requires OpenRouter")
        if (
            self.reasoning_mode == "enabled"
            and self.reasoning_wire_policy != "openrouter_reasoning"
            and self.reasoning_effort is None
        ):
            raise ValueError("enabled agent reasoning requires an effort")
        if self.provider_protocol == "offline_mock":
            if self.base_url is not None or self.api_key_env is not None:
                raise ValueError("offline agent model cannot carry provider settings")
        elif self.base_url is None or self.api_key_env is None:
            raise ValueError("real agent model requires endpoint and credential name")
        return self

    @model_serializer(mode="wrap")
    def omit_unused_reasoning_max_tokens(self, handler):
        payload = handler(self)
        if self.reasoning_max_tokens is None:
            payload.pop("reasoning_max_tokens", None)
        return payload


def agent_model_binding(profile: ExecutionModelProfile) -> AgentModelBinding:
    return AgentModelBinding(
        canonical_model_id=profile.canonical_model_id,
        provider_id=profile.provider.provider_id,
        provider_protocol=profile.provider.protocol,
        base_url=profile.provider.base_url,
        api_key_env=profile.provider.api_key_env,
        model_id=profile.adapter_model_ids.mini_swe_litellm,
        temperature=profile.generation.temperature,
        reasoning_mode=profile.reasoning.mode,
        reasoning_effort=profile.reasoning.effort,
        reasoning_max_tokens=profile.reasoning.max_tokens,
        reasoning_wire_policy=profile.reasoning.wire_policy,
        require_response_model=profile.effective_identity.require_response_model,
        accepted_effective_model_ids=profile.effective_identity.accepted_model_ids,
    )


class AgentProfile(StrictModel):
    """Resolved runtime profile for every agent stage in one build.

    The checked-in YAML intentionally omits every model field.  The bounded
    loader below injects this binding from the one selected execution-model
    profile before the profile is hashed or published.
    """

    schema_version: Literal["agent-profile-1.1"] = AGENT_PROFILE_SCHEMA_VERSION
    profile_id: NonEmptyString
    adapter: Literal["subprocess_runner"] = "subprocess_runner"
    runner_protocol_version: Literal["norm4code-agent-runner-1.2"] = (
        AGENT_RUNNER_PROTOCOL_VERSION
    )
    adapter_id: NonEmptyString
    adapter_version: NonEmptyString
    framework_id: NonEmptyString
    framework_version: NonEmptyString
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    model: AgentModelBinding
    runner_command: Annotated[list[NonEmptyString], Field(min_length=1)]
    runner_options: dict[NonEmptyString, str | int | float | bool] = Field(
        default_factory=dict
    )
    environment_keys: list[NonEmptyString] = Field(default_factory=list)
    sandbox_profile_id: NonEmptyString
    sandbox: AgentSandboxPolicy = Field(default_factory=AgentSandboxPolicy)

    @property
    def model_id(self) -> str:
        """Compatibility accessor for report code while YAML stays model-free."""

        return self.model.model_id

    @field_validator("environment_keys")
    @classmethod
    def environment_keys_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("agent profile environment_keys must be unique")
        forbidden = {"HOME", "CODEX_HOME", "MSWEA_GLOBAL_CONFIG_DIR", "TMPDIR"}
        overlap = forbidden & set(value)
        if overlap:
            raise ValueError(
                "agent profile cannot inherit global configuration paths: "
                f"{sorted(overlap)}"
            )
        return sorted(value)

    @field_validator("runner_command")
    @classmethod
    def runner_command_has_no_templates(cls, value: list[str]) -> list[str]:
        formatter = string.Formatter()
        for argument in value:
            if any(field is not None for _, field, _, _ in formatter.parse(argument)):
                raise ValueError(
                    "runner_command cannot contain templates; the host appends "
                    "--request and --result"
                )
        return value

    @model_validator(mode="after")
    def profile_contains_no_inline_secret_fields(self) -> AgentProfile:
        def secret_bearing_name(raw: str) -> bool:
            normalized = raw.lower().lstrip("-").replace("-", "_")
            compact = normalized.replace("_", "")
            if any(
                marker in compact
                for marker in (
                    "apikey",
                    "password",
                    "secret",
                    "credential",
                    "authorization",
                )
            ):
                return True
            return normalized in {
                "token",
                "access_token",
                "auth_token",
                "bearer_token",
                "refresh_token",
                "github_token",
                "header",
                "headers",
            } or normalized.endswith(
                (
                    "_access_token",
                    "_auth_token",
                    "_bearer_token",
                    "_refresh_token",
                    "_github_token",
                )
            )

        option_names = {
            str(key).lower().replace("-", "_") for key in self.runner_options
        }
        if any(name.startswith("model_") for name in option_names):
            raise ValueError(
                "agent runtime profile cannot contain model runner options"
            )
        if any(secret_bearing_name(name) for name in option_names):
            raise ValueError(
                "agent runner secrets must be named in environment_keys, not "
                "stored in runner_options"
            )
        for argument in self.runner_command[1:]:
            name = argument.split("=", 1)[0]
            if secret_bearing_name(name):
                raise ValueError(
                    "agent runner secrets cannot be embedded in runner_command"
                )
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.runner_options.values()
        ):
            raise ValueError("agent runner numeric options must be finite")
        return self


class AgentRunnerRequest(StrictModel):
    schema_version: Literal["agent-runner-request-1.1"] = (
        AGENT_RUNNER_REQUEST_SCHEMA_VERSION
    )
    request_id: AgentRunId
    attempt_ordinal: Literal[0] = 0
    stage: Literal["source_discovery"]
    task_context: Literal["task_independent"] = "task_independent"
    snapshot_digest: SnapshotDigest
    profile_digest: Sha256Hex
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    framework_id: NonEmptyString
    framework_version: NonEmptyString
    model_id: NonEmptyString
    model: AgentModelBinding
    sandbox_profile_id: NonEmptyString
    sandbox: AgentSandboxPolicy
    prompt_version: Literal["source-discovery-agent-1.4"]
    prompt_sha256: Sha256Hex
    response_schema_sha256: Sha256Hex
    discovery_sha256: Sha256Hex
    discovery_view_sha256: Sha256Hex
    agent_settings_sha256: Sha256Hex
    workspace_inputs: list[AgentWorkspaceInput]
    workspace_manifest_sha256: Sha256Hex
    task_path: NonEmptyString
    discovery_path: NonEmptyString
    repository_view_path: NonEmptyString
    response_schema_path: NonEmptyString
    payload_path: NonEmptyString
    trajectory_path: NonEmptyString
    runner_options: dict[NonEmptyString, str | int | float | bool]
    limits: AgentStageSettings

    @field_validator(
        "task_path",
        "discovery_path",
        "repository_view_path",
        "response_schema_path",
        "payload_path",
        "trajectory_path",
    )
    @classmethod
    def paths_are_safe_relative_posix(cls, value: str) -> str:
        path = Path(value)
        if (
            path.is_absolute()
            or not value
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("agent request path must be safe relative POSIX")
        return value

    @model_validator(mode="after")
    def workspace_manifest_contract(self) -> AgentRunnerRequest:
        payload = {
            "inputs": [item.model_dump(mode="json") for item in self.workspace_inputs]
        }
        expected = sha256_bytes(stable_json_text(payload).encode("utf-8"))
        if self.workspace_manifest_sha256 != expected:
            raise ValueError("agent workspace manifest digest is invalid")
        paths = [item.path for item in self.workspace_inputs]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("agent workspace inputs must be unique and sorted")
        if self.model_id != self.model.model_id:
            raise ValueError("agent request model ID differs from its binding")
        expected_request_id = agent_request_identity(
            self.model_dump(mode="json")
        )
        if self.request_id != expected_request_id:
            raise ValueError("agent request identity is invalid")
        return self


class AgentUsage(StrictModel):
    status: Literal["known", "partial", "unknown"] = "unknown"
    turns: Annotated[int, Field(ge=0)] | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    cached_input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def knowledge_contract(self) -> AgentUsage:
        values = (
            self.turns,
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.cost_usd,
        )
        if self.status == "unknown" and any(item is not None for item in values):
            raise ValueError("unknown agent usage cannot contain numeric values")
        if self.status == "known" and any(item is None for item in values):
            raise ValueError("known agent usage requires every numeric value")
        return self


class AgentModelCallReceipt(StrictModel):
    """Metadata for one terminal provider response, never its hidden text."""

    call_ordinal: Annotated[int, Field(ge=0)]
    configured_model_id: NonEmptyString
    effective_model_id: NonEmptyString
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    cached_input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_content_present: bool = False
    reasoning_content_bytes: Annotated[int, Field(ge=0)] = 0
    finish_reason: str | None = None

    @model_validator(mode="after")
    def reasoning_telemetry_is_consistent(self) -> AgentModelCallReceipt:
        if not self.reasoning_content_present and self.reasoning_content_bytes != 0:
            raise ValueError("reasoning bytes require reasoning content presence")
        if (
            self.reasoning_tokens is not None
            and self.output_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning tokens exceed completion tokens")
        return self


class AgentModelExecutionReceipt(StrictModel):
    """Fail-closed identity and reasoning attestation for one agent episode."""

    schema_version: Literal["agent-model-execution-receipt-1.0"] = (
        AGENT_MODEL_EXECUTION_RECEIPT_SCHEMA_VERSION
    )
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    canonical_model_id: NonEmptyString
    adapter_model_id: NonEmptyString
    reasoning_mode: Literal["disabled", "enabled"]
    reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None
    reasoning_max_tokens: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    request_controls_attested: bool
    reasoning_replay_attested: bool
    reported_model_call_count: Annotated[int, Field(ge=0)] | None = None
    model_call_count: Annotated[int, Field(ge=0)]
    calls: list[AgentModelCallReceipt]
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_content_responses: Annotated[int, Field(ge=0)] = 0
    reasoning_content_bytes: Annotated[int, Field(ge=0)] = 0
    reasoning_content_retained: Literal[False] = False
    telemetry_status: Literal["complete", "partial", "not_applicable"]
    status: Literal["verified", "partial", "violated"]
    violations: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("violations")
    @classmethod
    def violations_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("agent model receipt violations must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def receipt_is_internally_consistent(self) -> AgentModelExecutionReceipt:
        if [item.call_ordinal for item in self.calls] != list(
            range(len(self.calls))
        ):
            raise ValueError("agent model call ordinals must be contiguous")
        if self.model_call_count != len(self.calls):
            raise ValueError("agent model call count differs from call receipts")
        if (
            self.status == "verified"
            and self.reported_model_call_count != self.model_call_count
        ):
            raise ValueError("verified receipt requires complete model-call telemetry")
        if any(item.configured_model_id != self.adapter_model_id for item in self.calls):
            raise ValueError("agent call configured model differs from receipt")
        known_reasoning = all(item.reasoning_tokens is not None for item in self.calls)
        expected_reasoning = (
            sum(item.reasoning_tokens or 0 for item in self.calls)
            if known_reasoning
            else None
        )
        if self.reasoning_tokens != expected_reasoning:
            raise ValueError("agent receipt reasoning-token total is inconsistent")
        if self.reasoning_content_responses != sum(
            item.reasoning_content_present for item in self.calls
        ):
            raise ValueError("agent receipt reasoning response count is inconsistent")
        if self.reasoning_content_bytes != sum(
            item.reasoning_content_bytes for item in self.calls
        ):
            raise ValueError("agent receipt reasoning byte total is inconsistent")
        if self.reasoning_mode == "disabled" and (
            self.reasoning_effort is not None
            or self.reasoning_max_tokens is not None
        ):
            raise ValueError("disabled receipt cannot carry a reasoning control")
        if (
            self.reasoning_effort is not None
            and self.reasoning_max_tokens is not None
        ):
            raise ValueError("receipt reasoning controls are mutually exclusive")
        disabled_violation = self.reasoning_mode == "disabled" and any(
            (item.reasoning_tokens or 0) > 0
            or item.reasoning_content_bytes > 0
            for item in self.calls
        )
        if disabled_violation and self.status != "violated":
            raise ValueError("disabled reasoning observation must violate profile")
        if self.status == "verified" and (
            self.violations
            or not self.request_controls_attested
            or not self.reasoning_replay_attested
        ):
            raise ValueError("verified agent model receipt is not fully attested")
        if self.status == "violated" and not self.violations:
            raise ValueError("violated agent model receipt requires violations")
        if self.status == "partial" and not self.violations:
            raise ValueError("partial agent model receipt requires localized reasons")
        return self

    @model_serializer(mode="wrap")
    def omit_unused_reasoning_max_tokens(self, handler):
        payload = handler(self)
        if self.reasoning_max_tokens is None:
            payload.pop("reasoning_max_tokens", None)
        return payload


class AgentRunnerResult(StrictModel):
    """Untrusted result returned by a framework-specific runner."""

    schema_version: Literal["agent-runner-result-1.1"] = (
        AGENT_RUNNER_RESULT_SCHEMA_VERSION
    )
    request_id: AgentRunId
    stage: Literal["source_discovery"]
    status: Literal["success", "failed"]
    framework_id: NonEmptyString
    framework_version: NonEmptyString
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    model_id: NonEmptyString
    model_execution: AgentModelExecutionReceipt | None = None
    sandbox_profile_id: NonEmptyString
    sandbox_status: Literal["enforced", "failed", "unknown"]
    exit_status: NonEmptyString
    usage: AgentUsage = Field(default_factory=AgentUsage)
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def terminal_contract(self) -> AgentRunnerResult:
        if self.status == "success" and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("successful agent runner result cannot contain errors")
        if self.status == "success" and self.sandbox_status != "enforced":
            raise ValueError("successful agent result requires enforced sandbox")
        if self.status == "success" and (
            self.model_execution is None
            or self.model_execution.status != "verified"
        ):
            raise ValueError(
                "successful agent result requires a verified model receipt"
            )
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed agent runner result requires error_code")
        if self.model_execution is not None and (
            self.model_execution.execution_model_profile_id
            != self.execution_model_profile_id
            or self.model_execution.execution_model_profile_digest
            != self.execution_model_profile_digest
            or self.model_execution.adapter_model_id != self.model_id
        ):
            raise ValueError("agent result model receipt identity differs")
        return self


class AgentRunArtifact(StrictModel):
    """Host-owned immutable record for one bounded agent process."""

    schema_version: Literal["agent-run-artifact-1.1"] = (
        AGENT_RUN_ARTIFACT_SCHEMA_VERSION
    )
    request_id: AgentRunId
    stage: Literal["source_discovery"]
    snapshot_digest: SnapshotDigest
    profile_id: NonEmptyString
    profile_digest: Sha256Hex
    framework_id: NonEmptyString
    framework_version: NonEmptyString
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    model_id: NonEmptyString
    model_execution_receipt_sha256: Sha256Hex | None
    sandbox_profile_id: NonEmptyString
    sandbox_status: Literal["enforced", "failed", "unknown"]
    prompt_version: Literal["source-discovery-agent-1.4"]
    request_sha256: Sha256Hex
    runner_result_sha256: Sha256Hex | None
    status: Literal["success", "failed"]
    process_exit_code: int | None
    timed_out: bool
    duration_ms: Annotated[int, Field(ge=0)]
    exit_status: str | None
    payload_sha256: Sha256Hex | None
    trajectory_sha256: Sha256Hex | None
    stdout_sha256: Sha256Hex
    stderr_sha256: Sha256Hex
    stdout_bytes: Annotated[int, Field(ge=0)]
    stderr_bytes: Annotated[int, Field(ge=0)]
    usage: AgentUsage
    errors: list[NonEmptyString]

    @model_validator(mode="after")
    def completion_contract(self) -> AgentRunArtifact:
        if self.status == "success":
            if self.timed_out or self.process_exit_code != 0 or self.errors:
                raise ValueError("successful agent run has process or validation errors")
            if self.sandbox_status != "enforced":
                raise ValueError("successful agent run requires enforced sandbox")
            if not all(
                (
                    self.runner_result_sha256,
                    self.payload_sha256,
                    self.trajectory_sha256,
                    self.exit_status,
                    self.model_execution_receipt_sha256,
                )
            ):
                raise ValueError("successful agent run is missing terminal artifacts")
        elif not self.errors:
            raise ValueError("failed agent run requires localized errors")
        return self


class SourceAcquisitionArtifact(StrictModel):
    schema_version: Literal["source-acquisition-1.2"] = (
        SOURCE_ACQUISITION_SCHEMA_VERSION
    )
    target_content_admission_policy: Literal[
        "controlled-research-unredacted-target-content-1.0"
    ]
    snapshot_digest: SnapshotDigest
    discovery_sha256: Sha256Hex
    agent_settings_sha256: Sha256Hex
    supplement_sha256: Sha256Hex | None
    agent_profile_sha256: Sha256Hex
    execution_model_profile_digest: Sha256Hex
    execution_model_profile_sha256: Sha256Hex
    model_execution_receipt_sha256: Sha256Hex | None
    agent_request_sha256: Sha256Hex
    agent_run_sha256: Sha256Hex
    source_plan_sha256: Sha256Hex | None
    complete: bool
    incomplete_reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def completion_contract(self) -> SourceAcquisitionArtifact:
        if self.complete and (
            self.supplement_sha256 is None
            or self.model_execution_receipt_sha256 is None
            or self.incomplete_reasons
        ):
            raise ValueError("complete source acquisition has incomplete artifacts")
        if not self.complete and not self.incomplete_reasons:
            raise ValueError("incomplete source acquisition requires reasons")
        return self


@dataclass(frozen=True)
class AgentProcessOutcome:
    result: AgentRunnerResult | None
    exit_code: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    errors: tuple[str, ...]
    duration_ms: int = 0


class AgentRuntime(Protocol):
    profile: AgentProfile

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome: ...


class SubprocessAgentRuntime:
    """Run any conforming framework wrapper without a shell."""

    def __init__(self, profile: AgentProfile) -> None:
        self.profile = profile

    def _environment(self, isolated_root: Path) -> dict[str, str]:
        allowed = {"PATH", "LANG", "LC_ALL", "SYSTEMROOT"}
        allowed.update(self.profile.environment_keys)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in allowed
        }
        model = self.profile.model
        if model.provider_protocol != "offline_mock":
            assert model.api_key_env is not None
            credential = os.environ.get(model.api_key_env)
            if not credential:
                raise ValueError("execution-model credential is unavailable")
            if (
                model.provider_id in {"apiyi", "deepseek"}
                and model.provider_protocol == "openai_chat_completions"
                and model.reasoning_wire_policy == "deepseek_thinking_type"
                and model.model_id.startswith("deepseek/")
            ):
                # LiteLLM's DeepSeek adapter discovers this conventional name.
                # The selected profile credential remains the single source;
                # its value is never placed in request/config/trajectory JSON.
                environment["DEEPSEEK_API_KEY"] = credential
            elif (
                model.provider_id == "apiyi"
                and model.provider_protocol == "openai_chat_completions"
                and model.reasoning_wire_policy == "qwen_enable_thinking"
                and model.model_id.startswith("openai/")
            ):
                # Qwen uses LiteLLM's OpenAI-compatible adapter. Map only the
                # selected profile credential into the trusted runner process;
                # the repository shell still receives its separate allowlist.
                environment["OPENAI_API_KEY"] = credential
            elif (
                model.provider_id == "openrouter"
                and model.provider_protocol == "openai_chat_completions"
                and model.reasoning_wire_policy == "openrouter_reasoning"
                and model.model_id.startswith("openrouter/")
            ):
                environment["OPENROUTER_API_KEY"] = credential
            else:
                raise ValueError(
                    "agent model has no typed process credential adapter"
                )
        isolated_home = isolated_root / "home"
        isolated_tmp = isolated_root / "tmp"
        mini_config = isolated_root / "mini-global-config"
        for path in (isolated_home, isolated_tmp, mini_config):
            path.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "HOME": isolated_home.as_posix(),
                "TMPDIR": isolated_tmp.as_posix(),
                "MSWEA_GLOBAL_CONFIG_DIR": mini_config.as_posix(),
                "MSWEA_CONFIGURED": "true",
                "MSWEA_SILENT_STARTUP": "1",
            }
        )
        return environment

    def _redact_output(self, value: bytes) -> bytes:
        redacted = value
        secret_keys = [*self.profile.environment_keys]
        if self.profile.model.api_key_env is not None:
            secret_keys.append(self.profile.model.api_key_env)
        for key in secret_keys:
            secret = os.environ.get(key)
            if secret and len(secret) >= 4:
                redacted = redacted.replace(
                    secret.encode("utf-8", errors="ignore"),
                    b"[REDACTED]",
                )
        return redacted

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait()

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        if result_path.exists() or result_path.is_symlink():
            return AgentProcessOutcome(
                result=None,
                exit_code=None,
                timed_out=False,
                stdout=b"",
                stderr=b"",
                errors=("agent_runner_result_preexists",),
                duration_ms=0,
            )
        command = [
            *self.profile.runner_command,
            "--request",
            request_path.resolve().as_posix(),
            "--result",
            result_path.resolve().as_posix(),
        ]
        with tempfile.TemporaryDirectory(prefix="norm4code-agent-runtime-") as raw:
            isolated_root = Path(raw)
            errors: list[str] = []
            timed_out = False
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._environment(isolated_root),
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                return AgentProcessOutcome(
                    result=None,
                    exit_code=None,
                    timed_out=False,
                    stdout=b"",
                    stderr=str(error).encode("utf-8", errors="replace")[
                        :max_output_bytes
                    ],
                    errors=("agent_runner_start_failure",),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            assert process.stdout is not None and process.stderr is not None
            stdout_buffer = bytearray()
            stderr_buffer = bytearray()
            output_limit_events: list[str] = []
            output_lock = threading.Lock()

            def drain(
                stream,
                buffer: bytearray,
                error_code: str,
            ) -> None:
                observed = 0
                while True:
                    chunk = stream.read(65_536)
                    if not chunk:
                        break
                    observed += len(chunk)
                    remaining = max_output_bytes - len(buffer)
                    if remaining > 0:
                        buffer.extend(chunk[:remaining])
                    if observed > max_output_bytes:
                        with output_lock:
                            if error_code not in output_limit_events:
                                output_limit_events.append(error_code)
                        break

            readers = (
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout_buffer, "agent_runner_stdout_limit"),
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr_buffer, "agent_runner_stderr_limit"),
                    daemon=True,
                ),
            )
            for reader in readers:
                reader.start()
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    timed_out = True
                    errors.append("agent_runner_timeout")
                    self._terminate_process_group(process)
                    break
                with output_lock:
                    exceeded = bool(output_limit_events)
                if exceeded:
                    self._terminate_process_group(process)
                    break
                time.sleep(0.05)
            exit_code = process.poll()
            for reader in readers:
                reader.join(timeout=2)
            process.stdout.close()
            process.stderr.close()
            with output_lock:
                errors.extend(output_limit_events)
            stdout = self._redact_output(bytes(stdout_buffer))[:max_output_bytes]
            stderr = self._redact_output(bytes(stderr_buffer))[:max_output_bytes]
            duration_ms = int((time.monotonic() - started) * 1000)

        if exit_code != 0:
            errors.append("agent_runner_nonzero_exit")

        result: AgentRunnerResult | None = None
        try:
            result = AgentRunnerResult.model_validate_json(
                _read_regular_bounded_bytes(
                    result_path,
                    limit=max_output_bytes,
                )
            )
        except FileNotFoundError:
            errors.append("agent_runner_result_missing")
        except ValueError:
            errors.append("agent_runner_result_invalid")

        return AgentProcessOutcome(
            result=result,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            errors=tuple(errors),
            duration_ms=duration_ms,
        )


class DeterministicFakeAgentRuntime:
    """Offline test runtime that still exercises the full agent artifact path."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        proposals: tuple[tuple[str, str], ...] = (),
        search_status: Literal["complete", "truncated"] = "complete",
    ) -> None:
        if (
            profile.framework_id != "norm4code-fake-agent"
            or profile.model_id != "deterministic-fake"
            or profile.adapter_id != "norm4code-fake-runner"
            or profile.execution_model_profile_id != "offline-deterministic-test"
        ):
            raise ValueError(
                "deterministic fake runtime requires the explicit fake profile"
            )
        self.profile = profile
        self.proposals = proposals
        self.search_status = search_status

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        del timeout_seconds
        request = AgentRunnerRequest.model_validate_json(
            _read_regular_bounded_bytes(
                request_path,
                limit=100_000_000,
            )
        )
        payload_path = cwd / request.payload_path
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            stable_json_text(
                {
                    "schema_version": "1.2",
                    "snapshot_digest": request.snapshot_digest,
                    "search_status": self.search_status,
                    "proposals": [
                        {
                            "path": path,
                            "compared_baseline_paths": [],
                            "reason": reason,
                        }
                        for path, reason in self.proposals
                    ],
                }
            ),
            encoding="utf-8",
        )
        model_execution = AgentModelExecutionReceipt(
            execution_model_profile_id=self.profile.execution_model_profile_id,
            execution_model_profile_digest=(
                self.profile.execution_model_profile_digest
            ),
            canonical_model_id=self.profile.model.canonical_model_id,
            adapter_model_id=self.profile.model_id,
            reasoning_mode="disabled",
            reasoning_effort=None,
            reasoning_max_tokens=None,
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
        trajectory_path = cwd / request.trajectory_path
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(
            stable_json_text(
                {
                    "trajectory_format": "norm4code-fake-agent-1.0",
                    "info": {"exit_status": "Submitted"},
                    "messages": [],
                    "norm4code_model_execution": model_execution.model_dump(
                        mode="json"
                    ),
                    "norm4code_sanitization": {
                        "schema_version": (
                            "norm4code-agent-trajectory-sanitization-1.0"
                        ),
                        "raw_provider_response_retained": False,
                        "reasoning_content_retained": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        result = AgentRunnerResult(
            request_id=request.request_id,
            stage=request.stage,
            status="success",
            framework_id=self.profile.framework_id,
            framework_version=self.profile.framework_version,
            execution_model_profile_id=(
                self.profile.execution_model_profile_id
            ),
            execution_model_profile_digest=(
                self.profile.execution_model_profile_digest
            ),
            model_id=self.profile.model_id,
            model_execution=model_execution,
            sandbox_profile_id=self.profile.sandbox_profile_id,
            sandbox_status="enforced",
            exit_status="Submitted",
            usage=AgentUsage(status="partial", turns=1),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(stable_json_text(result), encoding="utf-8")
        return AgentProcessOutcome(
            result=result,
            exit_code=0,
            timed_out=False,
            stdout=b"",
            stderr=b"",
            errors=(),
            duration_ms=0,
        )


def fake_agent_profile() -> AgentProfile:
    from repo_constraint_memory_v3.model_profiles import (
        load_execution_model_profile,
    )

    model_profile = load_execution_model_profile(
        "offline-deterministic-test",
        allow_offline=True,
    )
    return AgentProfile(
        profile_id="norm4code-offline-fake",
        framework_id="norm4code-fake-agent",
        framework_version="1.0",
        execution_model_profile_id=model_profile.profile_id,
        execution_model_profile_digest=model_profile.profile_digest,
        model=agent_model_binding(model_profile),
        runner_command=["norm4code-fake-agent"],
        adapter_id="norm4code-fake-runner",
        adapter_version="1.1",
        sandbox_profile_id="filtered-workspace-test-only",
    )


def load_agent_profile(
    path: Path,
    *,
    model_profile: ExecutionModelProfile,
) -> AgentProfile:
    raw_bytes = _read_regular_bounded_bytes(path, limit=1_000_000)
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("agent profile root must be a mapping")
    reserved = {
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    }
    if reserved & set(raw):
        raise ValueError("agent runtime YAML must not contain model settings")
    raw = {
        **raw,
        "execution_model_profile_id": model_profile.profile_id,
        "execution_model_profile_digest": model_profile.profile_digest,
        "model": agent_model_binding(model_profile).model_dump(mode="json"),
    }
    return AgentProfile.model_validate(raw)
