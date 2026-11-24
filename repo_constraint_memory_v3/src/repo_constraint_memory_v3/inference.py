"""Stage-aware, bounded structured inference and canonical replay records."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_serializer,
    model_validator,
)

from repo_constraint_memory_v3 import __version__
from repo_constraint_memory_v3.llm import LLMClient, LLMProviderError
from repo_constraint_memory_v3.models import (
    LLMSettings,
    NonEmptyString,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import payload_digest

INFERENCE_CONTRACT_VERSION = "stage-inference-1.2"
RESOURCE_POLICY_VERSION = "bounded-inference-resource-1.12"
INFERENCE_INDEX_VERSION = "1.5"
INFERENCE_CACHE_KEY_VERSION = "structured-inference-cache-key-1.0"
ENDPOINT_PROFILE_VERSION = "llm-endpoint-profile-1.0"
STRUCTURED_RESPONSE_NORMALIZATION_VERSION = "structured-response-normalization-1.0"
DISABLED_THINKING_REASONING_TOKENS_VIOLATION = (
    "model_profile_violation:thinking_mode_disabled_reasoning_tokens_nonzero"
)
DISABLED_THINKING_REASONING_CONTENT_VIOLATION = (
    "model_profile_violation:thinking_mode_disabled_reasoning_content_nonempty"
)
EFFECTIVE_MODEL_MISMATCH_VIOLATION = (
    "model_profile_violation:effective_model_mismatch"
)
StageName = Literal[
    "explicit_extract",
    "explicit_semantic_verify",
    "current_semantic_map",
    "current_hypothesis_reduce",
    "population_semantic_classify",
    "counterexample_verify",
    "candidate_semantic_group",
    "temporal_evidence_classify",
    "unified_case_verify",
]
LIVE_STAGES = frozenset(
    {
        "explicit_extract",
        "explicit_semantic_verify",
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
        "candidate_semantic_group",
        "temporal_evidence_classify",
        "unified_case_verify",
    }
)
# Parsed provider content for each current-source semantic stage is bounded
# independently of the much wider transport-body guard.  Two megabytes leaves
# substantial headroom above the largest observed 64-hypothesis reducer JSON
# while still preventing unbounded validation work.
MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES = 2_000_000
CURRENT_SOURCE_SEMANTIC_STAGES: frozenset[StageName] = frozenset(
    {
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
    }
)
FUTURE_STAGES: frozenset[str] = frozenset()
STAGE_INPUT_KINDS = {
    "explicit_extract": frozenset({"explicit_unit"}),
    "explicit_semantic_verify": frozenset(
        {"explicit_grounded_candidate"}
    ),
    "current_semantic_map": frozenset({"semantic_packet"}),
    "current_hypothesis_reduce": frozenset(
        {
            "semantic_map_result",
            "semantic_observation",
        }
    ),
    "population_semantic_classify": frozenset(
        {"population_member_packet"}
    ),
    "counterexample_verify": frozenset(
        {
            "norm_hypothesis",
            "population_summary",
            "population_member_packet",
        }
    ),
    "candidate_semantic_group": frozenset({"candidate_case_member"}),
    "temporal_evidence_classify": frozenset(
        {
            "candidate_case_member",
            "git_query_manifest",
            "temporal_evidence_item",
        }
    ),
    "unified_case_verify": frozenset(
        {
            "candidate_case_member",
            "current_evidence_bundle",
            "temporal_evidence_bundle",
        }
    ),
}
T = TypeVar("T", bound=BaseModel)
W = TypeVar("W")
R = TypeVar("R")


class SuccessCache(Protocol):
    def read(self, key: str) -> dict[str, object] | None:
        """Return one untrusted cached inference record, if present."""

    def write(self, key: str, record: dict[str, object]) -> None:
        """Add one immutable terminal-success inference record."""


def _canonical_compact(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_digest(value: BaseModel, field: str) -> str:
    return payload_digest(value.model_dump(mode="json", exclude={field}))


def endpoint_profile_digest(*, provider: str, base_url: str) -> str:
    """Bind transport identity without persisting the configured endpoint."""

    return payload_digest(
        {
            "endpoint_profile_version": ENDPOINT_PROFILE_VERSION,
            "provider": provider,
            "base_url": base_url,
        }
    )


_SECRET_RE = re.compile(
    r"(?i)(bearer\s+|api[_ -]?key\s*[:=]\s*|token\s*[:=]\s*)\S+"
)


def safe_error_summary(value: object) -> str:
    """Bound untrusted errors and remove common credential-shaped fragments."""

    text = " ".join(str(value).split())
    text = _SECRET_RE.sub(r"\1[redacted]", text)
    return (text[:297] + "...") if len(text) > 300 else (text or "unknown_error")


def _disabled_thinking_profile_violations(
    *,
    reasoning_tokens: int | None,
    reasoning_content_bytes: int,
) -> list[str]:
    violations: list[str] = []
    if reasoning_tokens is not None and reasoning_tokens > 0:
        violations.append(DISABLED_THINKING_REASONING_TOKENS_VIOLATION)
    if reasoning_content_bytes > 0:
        violations.append(DISABLED_THINKING_REASONING_CONTENT_VIOLATION)
    return violations


class InferenceInputRef(StrictModel):
    kind: Literal[
        "explicit_unit",
        "explicit_grounded_candidate",
        "semantic_packet",
        "semantic_map_result",
        "semantic_observation",
        "norm_hypothesis",
        "population_member_packet",
        "population_summary",
        "explicit_proposition_context",
        "candidate_case_member",
        "current_evidence_bundle",
        "git_query_manifest",
        "temporal_evidence_item",
        "temporal_evidence_bundle",
    ]
    stable_id: NonEmptyString
    digest: Sha256Hex
    snapshot_digest: SnapshotDigest


class ResolvedModelParameters(StrictModel):
    model: NonEmptyString
    temperature: int | float
    thinking_mode: Literal["disabled", "enabled"]
    reasoning_effort: Literal[
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ] | None = None
    reasoning_max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)

    @model_validator(mode="after")
    def reasoning_contract(self) -> ResolvedModelParameters:
        if self.thinking_mode == "disabled" and (
            self.reasoning_effort is not None
            or self.reasoning_max_tokens is not None
        ):
            raise ValueError("disabled model parameters cannot carry reasoning controls")
        if (
            self.reasoning_effort is not None
            and self.reasoning_max_tokens is not None
        ):
            raise ValueError("model reasoning controls are mutually exclusive")
        return self

    @model_serializer(mode="wrap")
    def omit_unused_reasoning_max_tokens(self, handler):
        payload = handler(self)
        if self.reasoning_max_tokens is None:
            payload.pop("reasoning_max_tokens", None)
        return payload


def resolved_model_parameters_payload(settings: LLMSettings) -> dict[str, object]:
    """Serialize cache- and replay-bearing stage model controls."""

    payload: dict[str, object] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "thinking_mode": settings.thinking_mode,
        "reasoning_effort": settings.reasoning_effort,
        "max_output_tokens": settings.max_output_tokens,
        "timeout_seconds": settings.timeout_seconds,
    }
    if settings.reasoning_max_tokens is not None:
        payload["reasoning_max_tokens"] = settings.reasoning_max_tokens
    return payload


class InferenceUsage(StrictModel):
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    usage_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> InferenceUsage:
        if self.usage_digest != _model_digest(self, "usage_digest"):
            raise ValueError("incorrect inference usage digest")
        return self


class InferenceAttempt(StrictModel):
    attempt_ordinal: int = Field(ge=0)
    schema_attempt: int = Field(ge=0, le=3)
    output_limit_retry: int = Field(ge=0, le=2)
    provider_try: int = Field(ge=0, le=5)
    repair: bool
    parsed: bool
    provider_error: bool
    retryable: bool
    model: NonEmptyString
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    response_content_bytes: int | None = Field(default=None, ge=0)
    reasoning_content_present: bool = False
    reasoning_content_bytes: int = Field(default=0, ge=0)
    latency_ms: int = Field(ge=0)
    finish_reason: str | None
    response_digest: Sha256Hex | None
    validation_errors: list[str]
    attempt_digest: Sha256Hex

    @model_validator(mode="after")
    def invariant_and_digest(self) -> InferenceAttempt:
        if self.parsed and (self.provider_error or self.validation_errors):
            raise ValueError("parsed attempt cannot carry errors")
        if self.provider_error:
            if self.response_digest is not None or self.response_content_bytes is not None:
                raise ValueError("provider error cannot carry response telemetry")
            if (
                self.reasoning_tokens is not None
                or self.reasoning_content_present
                or self.reasoning_content_bytes != 0
            ):
                raise ValueError("provider error cannot carry reasoning telemetry")
        elif self.response_digest is None or self.response_content_bytes is None:
            raise ValueError("provider response attempt requires response telemetry")
        if self.reasoning_tokens is not None and self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens exceed total completion tokens")
        if not self.reasoning_content_present and self.reasoning_content_bytes != 0:
            raise ValueError("reasoning bytes require reasoning content presence")
        if self.attempt_digest != _model_digest(self, "attempt_digest"):
            raise ValueError("incorrect inference attempt digest")
        return self


class InferenceRecord(StrictModel):
    contract_version: Literal["stage-inference-1.2"]
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    request_id: str = Field(pattern=r"^INFER-[0-9a-f]{24}$")
    request_ordinal: int = Field(ge=0)
    stage: StageName
    mode: NonEmptyString | None
    protocol_version: NonEmptyString
    response_schema_version: NonEmptyString
    protocol_manifest_digest: Sha256Hex
    model_parameters: ResolvedModelParameters
    input_manifest: list[InferenceInputRef]
    input_manifest_digest: Sha256Hex
    system_prompt_digest: Sha256Hex
    request_payload_digest: Sha256Hex
    schema_repairs: int = Field(ge=0, le=3)
    output_limit_retries: int = Field(ge=0, le=2)
    provider_retries: int = Field(ge=0, le=5)
    terminal_status: Literal["success", "uncertain", "failure"]
    structured_response: dict[str, object] | None
    structured_response_digest: Sha256Hex | None
    attempts: list[InferenceAttempt]
    usage: InferenceUsage
    incomplete_reasons: list[NonEmptyString]
    record_digest: Sha256Hex

    @model_validator(mode="after")
    def invariant_and_digests(self) -> InferenceRecord:
        if self.stage not in LIVE_STAGES:
            raise ValueError(
                f"stage {self.stage} is not implemented in schema {SCHEMA_VERSION}"
            )
        if any(
            item.snapshot_digest != self.snapshot_digest
            for item in self.input_manifest
        ):
            raise ValueError("inference input snapshot does not match request")
        allowed_kinds = STAGE_INPUT_KINDS[self.stage]
        if any(item.kind not in allowed_kinds for item in self.input_manifest):
            raise ValueError(
                f"inference input kind is invalid for stage {self.stage}"
            )
        if self.request_id != request_identity(
            stage=self.stage,
            protocol_version=self.protocol_version,
            mode=self.mode,
            inputs=self.input_manifest,
            request_ordinal=self.request_ordinal,
        ):
            raise ValueError("inference request ID differs from canonical identity")
        expected_manifest = payload_digest(
            {"inputs": [item.model_dump(mode="json") for item in self.input_manifest]}
        )
        if self.input_manifest_digest != expected_manifest:
            raise ValueError("incorrect inference input-manifest digest")
        if self.terminal_status == "success":
            if self.structured_response is None:
                raise ValueError("successful inference requires a response")
            if self.incomplete_reasons:
                raise ValueError("successful inference cannot be incomplete")
        elif self.structured_response is not None:
            raise ValueError("non-success inference cannot carry a response")
        if self.structured_response is None:
            if self.structured_response_digest is not None:
                raise ValueError("response digest exists without structured response")
        elif self.structured_response_digest != payload_digest(
            {"response": self.structured_response}
        ):
            raise ValueError("incorrect structured-response digest")
        if [item.attempt_ordinal for item in self.attempts] != list(
            range(len(self.attempts))
        ):
            raise ValueError("inference attempts are not canonically ordered")
        max_attempts = (
            (self.schema_repairs + 1)
            * (self.output_limit_retries + 1)
            * (self.provider_retries + 1)
        )
        if len(self.attempts) > max_attempts:
            raise ValueError("inference attempt count exceeds configured bounds")
        if any(
            item.schema_attempt > self.schema_repairs
            or item.output_limit_retry > self.output_limit_retries
            or item.provider_try > self.provider_retries
            for item in self.attempts
        ):
            raise ValueError("inference attempt exceeds declared repair/retry bounds")
        if any(
            item.repair != (item.schema_attempt > 0)
            for item in self.attempts
        ):
            raise ValueError("inference repair flag differs from schema attempt")
        if self.attempts:
            first = self.attempts[0]
            if (
                first.schema_attempt,
                first.output_limit_retry,
                first.provider_try,
            ) != (0, 0, 0):
                raise ValueError("inference attempt trajectory must start at zero")
            for previous, current in zip(self.attempts, self.attempts[1:]):
                same_output_provider_retry = (
                    current.schema_attempt == previous.schema_attempt
                    and current.output_limit_retry
                    == previous.output_limit_retry
                    and current.provider_try == previous.provider_try + 1
                    and previous.provider_error
                    and previous.retryable
                    and not previous.parsed
                )
                prior_is_output_limit = (
                    not previous.provider_error
                    and not previous.parsed
                    and previous.retryable
                    and any(
                        error
                        in {
                            "provider response exceeded max_response_bytes",
                            "provider output exceeded max_output_tokens",
                            "provider output was truncated",
                        }
                        for error in previous.validation_errors
                    )
                )
                next_output_limit_retry = (
                    current.schema_attempt == previous.schema_attempt
                    and current.output_limit_retry
                    == previous.output_limit_retry + 1
                    and current.provider_try == 0
                    and prior_is_output_limit
                )
                next_schema_repair = (
                    current.schema_attempt == previous.schema_attempt + 1
                    and current.output_limit_retry == 0
                    and current.provider_try == 0
                    and not previous.provider_error
                    and not previous.parsed
                    and not previous.retryable
                    and "provider context window was exceeded"
                    not in previous.validation_errors
                )
                if not (
                    same_output_provider_retry
                    or next_output_limit_retry
                    or next_schema_repair
                ):
                    raise ValueError("inference attempt trajectory is not canonical")
        if self.terminal_status == "success" and (
            not self.attempts or not self.attempts[-1].parsed
        ):
            raise ValueError("successful inference must end in a parsed attempt")
        repeated_invalid_marker = "repeated_invalid_response" in (
            self.incomplete_reasons
        )
        repeated_invalid_trajectory = False
        if len(self.attempts) >= 2:
            previous, last = self.attempts[-2:]
            repeated_invalid_trajectory = (
                self.stage == "current_hypothesis_reduce"
                and self.terminal_status == "failure"
                and last.repair
                and not last.parsed
                and not last.provider_error
                and not last.retryable
                and last.response_digest is not None
                and previous.response_digest == last.response_digest
                and not previous.parsed
                and not previous.provider_error
                and not previous.retryable
                and previous.validation_errors == last.validation_errors
                and bool(last.validation_errors)
                and last.schema_attempt == previous.schema_attempt + 1
                and last.output_limit_retry == 0
                and last.provider_try == 0
                and last.schema_attempt < self.schema_repairs
            )
        if repeated_invalid_marker != repeated_invalid_trajectory:
            raise ValueError(
                "repeated-invalid marker differs from terminal attempt proof"
            )
        if self.model_parameters.thinking_mode == "disabled":
            profile_violations: set[str] = set()
            violation_ordinals: list[int] = []
            for attempt in self.attempts:
                violations = _disabled_thinking_profile_violations(
                    reasoning_tokens=attempt.reasoning_tokens,
                    reasoning_content_bytes=attempt.reasoning_content_bytes,
                )
                if not violations:
                    continue
                violation_ordinals.append(attempt.attempt_ordinal)
                profile_violations.update(violations)
                if attempt.parsed or not set(violations) <= set(
                    attempt.validation_errors
                ):
                    raise ValueError(
                        "disabled-thinking profile violation is not fail-closed"
                    )
            if profile_violations:
                if (
                    self.terminal_status != "failure"
                    or violation_ordinals[-1] != len(self.attempts) - 1
                    or not profile_violations <= set(self.incomplete_reasons)
                ):
                    raise ValueError(
                        "disabled-thinking profile violation must be terminal"
                    )
        expected_usage = (
            len(self.attempts),
            sum(item.input_tokens for item in self.attempts),
            sum(item.output_tokens for item in self.attempts),
            (
                sum(item.reasoning_tokens or 0 for item in self.attempts)
                if all(
                    item.provider_error or item.reasoning_tokens is not None
                    for item in self.attempts
                )
                else None
            ),
            sum(item.latency_ms for item in self.attempts),
        )
        actual_usage = (
            self.usage.calls,
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.reasoning_tokens,
            self.usage.latency_ms,
        )
        if actual_usage != expected_usage:
            raise ValueError("inference usage does not equal attempt totals")
        if self.record_digest != _model_digest(self, "record_digest"):
            raise ValueError("incorrect inference record digest")
        return self


def inference_record_model_identity_errors(
    record: InferenceRecord,
    *,
    configured_model: str,
    accepted_effective_model_ids: Sequence[str],
) -> list[str]:
    """Validate configured and response identities without provider access."""

    errors: list[str] = []
    accepted = set(accepted_effective_model_ids)
    if record.model_parameters.model != configured_model:
        errors.append("configured inference model differs")
    for attempt in record.attempts:
        if attempt.provider_error:
            if attempt.model != configured_model:
                errors.append("provider-error attempt configured model differs")
        elif attempt.model not in accepted:
            errors.append("provider effective model is not accepted")
    return errors


def _requires_effective_model_identity(settings: LLMSettings) -> bool:
    """Provider identity is meaningful for real wires, not fixture clients."""

    return settings.reasoning_wire_policy != "offline_mock"


class ProtocolManifest(StrictModel):
    manifest_version: Literal["1.11"]
    package_version: Literal["0.17.1", "0.18.0"]
    schema_version: Literal["4.11-partition-only-required-evidence"]
    target_content_admission_policy: Literal[
        "controlled-research-unredacted-target-content-1.0"
    ]
    provider: NonEmptyString
    endpoint_profile_digest: Sha256Hex
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    agent_profile_digest: Sha256Hex
    repository_settings_digest: Sha256Hex
    discovery_settings_digest: Sha256Hex
    agent_settings_digest: Sha256Hex
    chunking_settings_digest: Sha256Hex
    observation_settings_digest: Sha256Hex
    semantic_settings_digest: Sha256Hex
    unified_settings_digest: Sha256Hex
    agent_stage_protocol_versions: dict[str, NonEmptyString]
    explicit_prompt_version: NonEmptyString
    explicit_response_schema_version: NonEmptyString
    model_parameters: ResolvedModelParameters
    reasoning_wire_policy: Literal[
        "deepseek_thinking_type",
        "qwen_enable_thinking",
        "openrouter_reasoning",
        "openai_responses_reasoning_effort",
        "offline_mock",
    ]
    require_response_model: Literal[True]
    accepted_effective_model_ids: list[NonEmptyString] = Field(
        min_length=1,
        max_length=16,
    )
    parser_contract_version: NonEmptyString
    observation_contract_versions: dict[str, NonEmptyString]
    semantic_unit_contract_version: NonEmptyString
    packet_contract_version: NonEmptyString
    population_spec_contract_version: NonEmptyString
    proposition_contract_version: NonEmptyString
    stage_protocol_versions: dict[str, NonEmptyString]
    inference_contract_version: Literal["stage-inference-1.2"]
    thinking_wire_policy_version: Literal["typed-reasoning-profile-1.1"]
    retry_repair_policy_version: NonEmptyString
    resource_policy_version: Literal["bounded-inference-resource-1.12"]
    schema_repairs: int = Field(ge=0, le=3)
    output_limit_retries: int = Field(ge=0, le=2)
    provider_retries: int = Field(ge=0, le=5)
    provider_retry_delay_seconds: int | float = Field(ge=0, le=60)
    provider_retry_backoff_multiplier: int | float = Field(ge=1, le=10)
    provider_retry_max_delay_seconds: int | float = Field(ge=0, le=600)
    provider_retry_jitter_seconds: int | float = Field(ge=0, le=30)
    resource_limits: dict[str, int]
    protocol_manifest_digest: Sha256Hex

    @model_validator(mode="after")
    def digest_is_valid(self) -> ProtocolManifest:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("protocol manifest version does not match package")
        if (
            not self.accepted_effective_model_ids
            or len(self.accepted_effective_model_ids)
            != len(set(self.accepted_effective_model_ids))
            or self.accepted_effective_model_ids
            != sorted(self.accepted_effective_model_ids)
        ):
            raise ValueError(
                "protocol accepted effective model IDs are not canonical"
            )
        if self.protocol_manifest_digest != _model_digest(
            self, "protocol_manifest_digest"
        ):
            raise ValueError("incorrect protocol-manifest digest")
        return self


def protocol_manifest_for_readonly_replay(
    current_protocol: ProtocolManifest,
    *,
    producer_package_version: str,
) -> ProtocolManifest:
    """Rebuild the identical 4.11 contracts for the two supported producers.

    Only package 0.17.1 and 0.18.0 share this extraction contract.  Callers
    must compare the complete result to the saved manifest, not merely accept
    its producer version.  Build/cache/resume paths keep the current package
    identity and must not call this read-only replay helper.
    """

    if producer_package_version not in {"0.17.1", "0.18.0"}:
        raise ValueError("unsupported extraction producer package version")
    payload = current_protocol.model_dump(mode="json")
    payload["package_version"] = producer_package_version
    del payload["protocol_manifest_digest"]
    payload["protocol_manifest_digest"] = payload_digest(payload)
    return ProtocolManifest.model_validate(payload)


class InferenceRequestRef(StrictModel):
    request_id: str = Field(pattern=r"^INFER-[0-9a-f]{24}$")
    stage: StageName
    artifact_path: str = Field(
        pattern=r"^inference/requests/INFER-[0-9a-f]{24}\.json$"
    )
    artifact_sha256: Sha256Hex
    record_digest: Sha256Hex
    terminal_status: Literal["success", "uncertain", "failure"]


class InferenceStageCounts(StrictModel):
    stage: StageName
    planned: int = Field(ge=0)
    success: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    failure: int = Field(ge=0)
    omitted: int = Field(ge=0)
    omission_reasons: dict[NonEmptyString, int]

    @model_validator(mode="after")
    def algebra(self) -> InferenceStageCounts:
        if (
            self.success + self.uncertain + self.failure + self.omitted
            != self.planned
        ):
            raise ValueError("inference stage counts do not sum to planned")
        if bool(self.omitted) != bool(self.omission_reasons):
            raise ValueError("inference stage omission reasons differ from count")
        if sum(self.omission_reasons.values()) != self.omitted:
            raise ValueError("inference stage omission reason counts differ")
        if list(self.omission_reasons) != sorted(self.omission_reasons):
            raise ValueError("inference stage omission reasons are not canonical")
        if any(count <= 0 for count in self.omission_reasons.values()):
            raise ValueError("inference stage omission reason counts must be positive")
        return self


class InferenceIndex(StrictModel):
    index_version: Literal["1.5"]
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    protocol_manifest: ProtocolManifest
    requests: list[InferenceRequestRef]
    stages: list[InferenceStageCounts]
    total_calls: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_reasoning_tokens: int | None = Field(default=None, ge=0)
    total_latency_ms: int = Field(ge=0)
    total_artifact_bytes: int = Field(ge=0)
    incomplete_reasons: list[NonEmptyString]
    inference_set_digest: Sha256Hex

    @model_validator(mode="after")
    def invariant_and_digest(self) -> InferenceIndex:
        if [item.request_id for item in self.requests] != sorted(
            item.request_id for item in self.requests
        ):
            raise ValueError("inference request references must be sorted")
        if len({item.request_id for item in self.requests}) != len(self.requests):
            raise ValueError("duplicate inference request reference")
        omission_reasons = {
            reason
            for stage in self.stages
            for reason in stage.omission_reasons
        }
        if not omission_reasons <= set(self.incomplete_reasons):
            raise ValueError("inference omission reason is absent from index")
        if self.inference_set_digest != _model_digest(
            self, "inference_set_digest"
        ):
            raise ValueError("incorrect inference-set digest")
        return self


def _identity_response(value: object) -> object:
    return value


@dataclass(frozen=True)
class StageDefinition:
    stage: StageName
    mode: str | None
    protocol_version: str
    response_schema_version: str
    system_prompt: str
    response_model: type[BaseModel]
    validate_response: Callable[[BaseModel], BaseModel]
    repair_prompt: Callable[[str, str, list[str]], str]
    provider_response_model: type[BaseModel] | None = None
    repair_response_model: (
        Callable[[str, list[str]], type[BaseModel]] | None
    ) = None
    terminal_validate_response: Callable[[BaseModel], BaseModel] | None = None
    terminal_parse_response: (
        Callable[[object, Exception], BaseModel] | None
    ) = None
    # Opt in only when terminal_parse_response can isolate a non-essential
    # sub-object and then revalidate every required field. Other stages retain
    # the normal repair-before-terminal behavior.
    terminal_parse_before_repair: bool = False
    validate_raw_response: Callable[[object], object] = _identity_response
    normalize_response: Callable[[object], object] = _identity_response
    max_response_bytes: int | None = None
    max_request_bytes: int | None = None
    stop_on_repeated_invalid_response: bool = False
    bind_response_digest: Callable[[str], None] | None = None
    bind_response_content: Callable[[str], None] | None = None
    validate_record_response: (
        Callable[[BaseModel, InferenceRecord], None] | None
    ) = None

    def __post_init__(self) -> None:
        if self.max_request_bytes is not None and self.max_request_bytes < 1:
            raise ValueError("stage max_request_bytes must be positive")
        if (
            self.stage in CURRENT_SOURCE_SEMANTIC_STAGES
            and self.max_response_bytes
            != MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES
        ):
            raise ValueError(
                "current-source semantic stage must bind the frozen "
                "max_response_bytes"
            )


def _provider_response_model(definition: StageDefinition) -> type[BaseModel]:
    """Return the schema exposed to the provider, if it differs from storage."""

    return definition.provider_response_model or definition.response_model


def _is_repeated_invalid_response(
    attempts: Sequence[InferenceAttempt],
    *,
    response_digest: str,
    validation_errors: Sequence[str],
    repair: bool,
    schema_attempt: int,
    enabled: bool,
    early_stop_available: bool,
) -> bool:
    """Detect an exact invalid repair result already rejected by this call.

    This is deliberately stage-opt-in.  Some cheap stages benefit from using
    every configured repair even after a model repeats itself, while reducers
    can otherwise spend another very large response on byte-identical content.
    Requiring both the response digest and canonical validation errors avoids
    treating the same bytes under a changed host-side repair state as a repeat.
    """

    if not enabled or not repair or not early_stop_available:
        return False
    errors = list(validation_errors)
    if not attempts:
        return False
    previous = attempts[-1]
    return (
        not previous.parsed
        and not previous.provider_error
        and not previous.retryable
        and schema_attempt == previous.schema_attempt + 1
        and previous.response_digest == response_digest
        and previous.validation_errors == errors
    )


@dataclass
class _InferenceCallPermit:
    """Mutable call capacity reserved for one logical stage invocation."""

    budget_token: object
    capacity_calls: int
    remaining_calls: int
    active: bool = False
    released: bool = False


class ReplayBudgetAdmissionError(ValueError):
    """A saved terminal trajectory cannot fit the remaining hard call cap."""


class InferenceBudget:
    """Thread-safe shared call/token/cache/progress state for one build."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        cache: SuccessCache | None = None,
        progress_callback: Callable[[str], None] | None = None,
        initial_calls: int = 0,
        initial_input_tokens: int = 0,
        initial_output_tokens: int = 0,
    ) -> None:
        for label, value in (
            ("initial_calls", initial_calls),
            ("initial_input_tokens", initial_input_tokens),
            ("initial_output_tokens", initial_output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        self.settings = settings
        self.cache = cache
        self.progress_callback = progress_callback
        self._lock = threading.Lock()
        self.calls = initial_calls
        self.input_tokens = initial_input_tokens
        self.output_tokens = initial_output_tokens
        self.logical_calls = initial_calls
        self.logical_input_tokens = initial_input_tokens
        self.logical_output_tokens = initial_output_tokens
        self.cache_hits = 0
        self.exhausted_reasons: set[str] = set()
        self._stage_omission_reasons: dict[str, dict[str, int]] = {}
        self._stage_progress: dict[str, dict[str, int | float]] = {}
        self._permit_token = object()
        self._permit_local = threading.local()
        self._reserved_call_slots = 0

    def plan_stage(self, stage: StageName, count: int) -> None:
        if count < 0:
            raise ValueError("planned inference count must be non-negative")
        message = None
        with self._lock:
            progress = self._stage_progress.setdefault(
                stage,
                {
                    "planned": 0,
                    "done": 0,
                    "success": 0,
                    "failure": 0,
                    "omitted": 0,
                    "cache_hits": 0,
                    "retries": 0,
                    "repairs": 0,
                    "elapsed_seconds": 0.0,
                    "started": time.monotonic(),
                },
            )
            progress["planned"] = int(progress["planned"]) + count
            if self.progress_callback is not None and count:
                message = self._progress_message(stage, progress)
        if message is not None:
            self.progress_callback(message)

    @staticmethod
    def _progress_message(
        stage: str,
        progress: Mapping[str, int | float],
    ) -> str:
        elapsed = max(time.monotonic() - float(progress["started"]), 0.0)
        return (
            f"inference stage={stage} planned={int(progress['planned'])} "
            f"done={int(progress['done'])} success={int(progress['success'])} "
            f"failure={int(progress['failure'])} "
            f"omitted={int(progress['omitted'])} "
            f"cache_hits={int(progress['cache_hits'])} "
            f"retries={int(progress['retries'])} "
            f"repairs={int(progress['repairs'])} elapsed={elapsed:.1f}s"
        )

    def record_terminal(
        self,
        record: InferenceRecord,
        *,
        cache_hit: bool = False,
        replay_hit: bool = False,
    ) -> None:
        message = None
        with self._lock:
            permit = getattr(self._permit_local, "call_permit", None)
            if cache_hit or replay_hit:
                if (
                    self._valid_active_permit_locked(permit)
                    and record.usage.calls > permit.remaining_calls
                ):
                    raise ReplayBudgetAdmissionError(
                        "replayed inference usage exceeds reserved call permit"
                    )
                if (
                    not self._valid_active_permit_locked(permit)
                    and self.logical_calls
                    + self._reserved_call_slots
                    + record.usage.calls
                    > self.settings.max_total_calls
                ):
                    raise ReplayBudgetAdmissionError(
                        "replayed inference usage exceeds remaining call budget"
                    )
            progress = self._stage_progress.setdefault(
                record.stage,
                {
                    "planned": 0,
                    "done": 0,
                    "success": 0,
                    "failure": 0,
                    "omitted": 0,
                    "cache_hits": 0,
                    "retries": 0,
                    "repairs": 0,
                    "elapsed_seconds": 0.0,
                    "started": time.monotonic(),
                },
            )
            progress["done"] = int(progress["done"]) + 1
            terminal_key = (
                "success" if record.terminal_status == "success" else "failure"
            )
            progress[terminal_key] = int(progress[terminal_key]) + 1
            if cache_hit or replay_hit:
                # A success-only cache avoids a provider call, but the cached
                # logical result still owns the usage that was admitted by the
                # frozen run budget.  Counting it here makes resume and offline
                # replay stop at the same global budget boundary instead of
                # silently granting a second token/call allowance.
                if self._valid_active_permit_locked(permit):
                    permit.remaining_calls -= record.usage.calls
                    self._reserved_call_slots -= record.usage.calls
                self.logical_calls += record.usage.calls
                self.logical_input_tokens += record.usage.input_tokens
                self.logical_output_tokens += record.usage.output_tokens
            retries = (
                0
                if cache_hit or replay_hit
                else sum(
                    item.provider_try > 0 or item.output_limit_retry > 0
                    for item in record.attempts
                )
            )
            repairs = (
                0
                if cache_hit or replay_hit
                else sum(item.repair for item in record.attempts)
            )
            progress["retries"] = int(progress["retries"]) + retries
            progress["repairs"] = int(progress["repairs"]) + repairs
            if cache_hit:
                self.cache_hits += 1
                progress["cache_hits"] = int(progress["cache_hits"]) + 1
            progress["elapsed_seconds"] = max(
                time.monotonic() - float(progress["started"]),
                0.0,
            )
            done = int(progress["done"])
            accounted = done + int(progress["omitted"])
            planned = int(progress["planned"])
            if self.progress_callback is not None and (
                done == 1
                or done % 10 == 0
                or (planned and accounted == planned)
            ):
                message = self._progress_message(record.stage, progress)
        if message is not None:
            self.progress_callback(message)

    @property
    def stage_progress(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            result: dict[str, dict[str, int | float]] = {}
            for stage, values in sorted(self._stage_progress.items()):
                result[stage] = {
                    key: item
                    for key, item in values.items()
                    if key not in {"started", "elapsed_seconds"}
                }
                done = int(values["done"]) + int(values["omitted"])
                planned = int(values["planned"])
                elapsed = float(values["elapsed_seconds"])
                if done < planned:
                    elapsed = max(
                        time.monotonic() - float(values["started"]),
                        0.0,
                    )
                result[stage]["elapsed_seconds"] = round(elapsed, 6)
            return result

    def record_omitted(
        self,
        stage: StageName,
        count: int,
        *,
        reason: str,
    ) -> None:
        """Close planned logical requests without materializing fake records."""

        if count < 0:
            raise ValueError("omitted inference count must be non-negative")
        if not count:
            return
        message = None
        with self._lock:
            progress = self._stage_progress.setdefault(
                stage,
                {
                    "planned": 0,
                    "done": 0,
                    "success": 0,
                    "failure": 0,
                    "omitted": 0,
                    "cache_hits": 0,
                    "retries": 0,
                    "repairs": 0,
                    "elapsed_seconds": 0.0,
                    "started": time.monotonic(),
                },
            )
            if int(progress["done"]) + int(progress["omitted"]) + count > int(
                progress["planned"]
            ):
                raise ValueError("omitted inference count exceeds stage plan")
            progress["omitted"] = int(progress["omitted"]) + count
            self.exhausted_reasons.add(safe_error_summary(reason))
            bounded_reason = safe_error_summary(reason)
            stage_reasons = self._stage_omission_reasons.setdefault(stage, {})
            stage_reasons[bounded_reason] = (
                stage_reasons.get(bounded_reason, 0) + count
            )
            progress["elapsed_seconds"] = max(
                time.monotonic() - float(progress["started"]),
                0.0,
            )
            if self.progress_callback is not None:
                message = self._progress_message(stage, progress)
        if message is not None:
            self.progress_callback(message)

    def _terminal_exhaustion_reason_locked(
        self,
        *,
        include_reservations: bool = True,
    ) -> str | None:
        accounted_calls = self.logical_calls + (
            self._reserved_call_slots if include_reservations else 0
        )
        if accounted_calls >= self.settings.max_total_calls:
            return "max_total_calls"
        if self.logical_input_tokens >= self.settings.max_total_input_tokens:
            return "max_total_input_tokens"
        if self.logical_output_tokens >= self.settings.max_total_output_tokens:
            return "max_total_output_tokens"
        return None

    def terminal_exhaustion_reason(self) -> str | None:
        """Return the deterministic reason no further provider call may start."""

        with self._lock:
            permit = getattr(self._permit_local, "call_permit", None)
            if self._valid_active_permit_locked(permit):
                # A whole worst-case trajectory was atomically admitted before
                # this worker started.  Other jobs in the same wave may cross
                # a token threshold first, but cannot revoke this permit.
                return None
            reason = self._terminal_exhaustion_reason_locked()
            if reason is not None:
                # Unlike terminal usage accounting or worker cleanup, an
                # explicit non-permit boundary query means downstream work is
                # deciding whether it can continue.  Persist that real closure
                # without exposing concurrent completion order.
                self.exhausted_reasons.add(reason)
            return reason

    def finalize_usage_overrun(self) -> str | None:
        """Persist a deterministic terminal token overshoot, if any.

        Provider usage is known only after a response.  Equality is a valid
        completed boundary; a strict overshoot is partial even when no later
        request asks the budget whether it may continue.  Calls can never
        overshoot and are guarded separately.
        """

        with self._lock:
            permit = getattr(self._permit_local, "call_permit", None)
            if self._valid_active_permit_locked(permit):
                return None
            if self.logical_calls > self.settings.max_total_calls:
                raise ValueError("logical inference calls exceed hard limit")
            reason: str | None = None
            if self.logical_input_tokens > self.settings.max_total_input_tokens:
                reason = "max_total_input_tokens"
            elif (
                self.logical_output_tokens
                > self.settings.max_total_output_tokens
            ):
                reason = "max_total_output_tokens"
            if reason is not None:
                self.exhausted_reasons.add(reason)
            return reason

    def reserve_call(self) -> str | None:
        with self._lock:
            permit = getattr(self._permit_local, "call_permit", None)
            if self._valid_active_permit_locked(permit):
                if permit.remaining_calls <= 0:
                    self.exhausted_reasons.add("max_total_calls")
                    return "max_total_calls"
                permit.remaining_calls -= 1
                self._reserved_call_slots -= 1
                self.calls += 1
                self.logical_calls += 1
                return None
            exhausted = self._terminal_exhaustion_reason_locked()
            if exhausted is not None:
                self.exhausted_reasons.add(exhausted)
                return exhausted
            self.calls += 1
            self.logical_calls += 1
            return None

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.logical_input_tokens += input_tokens
            self.logical_output_tokens += output_tokens

    @property
    def max_provider_calls_per_request(self) -> int:
        """Return the exact configured worst-case call trajectory size."""

        return max_provider_calls_per_request(self.settings)

    def _valid_active_permit_locked(
        self, permit: object
    ) -> bool:
        return (
            isinstance(permit, _InferenceCallPermit)
            and permit.budget_token is self._permit_token
            and permit.active
            and not permit.released
        )

    def reserve_call_permits(
        self,
        logical_request_capacities: Sequence[int],
    ) -> tuple[_InferenceCallPermit, ...] | None:
        """Atomically reserve full trajectories for one deterministic wave.

        Returning ``None`` asks the caller to retain the old stable serial
        semantics.  No partial wave is ever reserved.
        """

        capacities = list(logical_request_capacities)
        if not capacities or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in capacities
        ):
            raise ValueError("permit logical-request capacities must be positive")
        call_capacities = [
            value * self.max_provider_calls_per_request for value in capacities
        ]
        requested = sum(call_capacities)
        with self._lock:
            if (
                self._terminal_exhaustion_reason_locked(
                    include_reservations=False
                )
                is not None
                or self.logical_calls + self._reserved_call_slots + requested
                > self.settings.max_total_calls
            ):
                return None
            permits = tuple(
                _InferenceCallPermit(
                    budget_token=self._permit_token,
                    capacity_calls=capacity,
                    remaining_calls=capacity,
                )
                for capacity in call_capacities
            )
            self._reserved_call_slots += requested
            return permits

    def _release_call_permit(self, permit: _InferenceCallPermit) -> None:
        with self._lock:
            if permit.budget_token is not self._permit_token:
                raise ValueError("call permit belongs to another budget")
            if permit.released:
                return
            self._reserved_call_slots -= permit.remaining_calls
            permit.remaining_calls = 0
            permit.active = False
            permit.released = True

    @contextmanager
    def activate_call_permit(
        self, permit: _InferenceCallPermit
    ) -> Iterator[None]:
        """Bind one pre-admitted trajectory only to the current worker."""

        prior = getattr(self._permit_local, "call_permit", None)
        if prior is not None:
            raise ValueError("nested inference call permits are forbidden")
        with self._lock:
            if (
                permit.budget_token is not self._permit_token
                or permit.released
                or permit.active
            ):
                raise ValueError("inference call permit is invalid or already active")
            permit.active = True
        self._permit_local.call_permit = permit
        try:
            yield
        finally:
            del self._permit_local.call_permit
            self._release_call_permit(permit)

    def release_call_permits(
        self, permits: Sequence[_InferenceCallPermit]
    ) -> None:
        """Idempotently release permits left unused after worker failure."""

        for permit in permits:
            self._release_call_permit(permit)

    @property
    def stage_omissions(self) -> dict[str, tuple[int, dict[str, int]]]:
        """Return exact omitted logical-call counts and their typed reasons."""

        with self._lock:
            return {
                stage: (
                    int(values["omitted"]),
                    dict(
                        sorted(
                            self._stage_omission_reasons.get(stage, {}).items()
                        )
                    ),
                )
                for stage, values in sorted(self._stage_progress.items())
                if int(values["omitted"])
            }


@dataclass(frozen=True)
class BudgetedStageRunResult(Generic[R]):
    """Terminal outcomes plus an auditable compact omitted-request suffix."""

    outcomes: tuple[R, ...]
    planned: int
    omitted: int
    exhaustion_reason: str | None


def max_provider_calls_per_request(settings: LLMSettings | object) -> int:
    """Return the frozen worst-case provider attempts for one request."""

    return (
        (int(getattr(settings, "schema_repairs")) + 1)
        * (int(getattr(settings, "output_limit_retries")) + 1)
        * (int(getattr(settings, "provider_retries")) + 1)
    )


def _run_pre_admitted_wave(
    wave: Sequence[W],
    invoke: Callable[[W], R],
    *,
    budget: InferenceBudget,
    max_concurrency: int,
    thread_name_prefix: str,
    max_logical_requests_per_item: Callable[[W], int] | None = None,
) -> list[R] | None:
    """Run one complete permitted wave, or request stable serial fallback."""

    items = list(wave)
    if len(items) <= 1 or max_concurrency <= 1:
        return None
    logical_capacities = [
        (
            max_logical_requests_per_item(item)
            if max_logical_requests_per_item is not None
            else 1
        )
        for item in items
    ]
    permits = budget.reserve_call_permits(logical_capacities)
    if permits is None:
        return None

    def invoke_with_permit(
        value: tuple[W, _InferenceCallPermit],
    ) -> R:
        item, permit = value
        with budget.activate_call_permit(permit):
            return invoke(item)

    try:
        outcomes = run_bounded_stage_calls(
            list(zip(items, permits)),
            invoke_with_permit,
            max_concurrency=max_concurrency,
            thread_name_prefix=thread_name_prefix,
        )
    finally:
        # Worker contexts normally release their own unused slots.  This
        # idempotent cleanup also covers executor failure before every worker
        # enters its context.
        budget.release_call_permits(permits)
    budget.finalize_usage_overrun()
    return outcomes


def run_call_permitted_stage_calls(
    work: Sequence[W],
    invoke: Callable[[W], R],
    *,
    budget: InferenceBudget,
    max_concurrency: int,
    thread_name_prefix: str,
    max_logical_requests_per_item: Callable[[W], int] | None = None,
) -> list[R]:
    """Run deterministic waves without changing terminal-record coverage.

    This variant is for stages whose existing contract materializes one
    terminal record for every planned item.  A full-capacity wave is parallel;
    otherwise every item is invoked serially in plan order, including the
    legacy zero-attempt failures after exhaustion.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    items = list(work)
    outcomes: list[R] = []
    for offset in range(0, len(items), max_concurrency):
        wave = items[offset : offset + max_concurrency]
        admitted = _run_pre_admitted_wave(
            wave,
            invoke,
            budget=budget,
            max_concurrency=max_concurrency,
            thread_name_prefix=thread_name_prefix,
            max_logical_requests_per_item=max_logical_requests_per_item,
        )
        if admitted is not None:
            outcomes.extend(admitted)
            continue
        for item in wave:
            outcomes.append(invoke(item))
            budget.finalize_usage_overrun()
    return outcomes


def run_budgeted_stage_calls(
    work: Sequence[W],
    invoke: Callable[[W], R],
    *,
    stage: StageName,
    budget: InferenceBudget,
    max_concurrency: int,
    thread_name_prefix: str,
    max_logical_requests_per_item: Callable[[W], int] | None = None,
) -> BudgetedStageRunResult[R]:
    """Run bounded waves and compactly close the suffix after exhaustion.

    A multi-item wave runs concurrently only after the shared call budget can
    reserve every item's complete worst-case trajectory.  Near the call-budget
    boundary, the same planned wave falls back to stable serial execution so
    thread completion order can never choose which logical request survives.
    Token boundaries are checked between waves (or serial items), while a fully
    admitted concurrent trajectory cannot be revoked by a sibling response.
    """

    items = list(work)
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    budget.plan_stage(stage, len(items))
    outcomes: list[R] = []
    exhaustion_reason: str | None = None
    next_item = 0
    while next_item < len(items):
        exhaustion_reason = budget.terminal_exhaustion_reason()
        if exhaustion_reason is not None:
            break
        wave = items[next_item : next_item + max_concurrency]
        admitted = _run_pre_admitted_wave(
            wave,
            invoke,
            budget=budget,
            max_concurrency=max_concurrency,
            thread_name_prefix=thread_name_prefix,
            max_logical_requests_per_item=max_logical_requests_per_item,
        )
        if admitted is not None:
            outcomes.extend(admitted)
            next_item += len(wave)
            continue

        # Insufficient capacity for every complete trajectory: preserve the
        # original per-item budget semantics in canonical plan order.
        for item in wave:
            exhaustion_reason = budget.terminal_exhaustion_reason()
            if exhaustion_reason is not None:
                break
            outcomes.append(invoke(item))
            budget.finalize_usage_overrun()
            next_item += 1
        if exhaustion_reason is not None:
            break
    omitted = len(items) - len(outcomes)
    if omitted:
        exhaustion_reason = (
            exhaustion_reason
            or budget.terminal_exhaustion_reason()
            or "stage_execution_closed"
        )
        budget.record_omitted(stage, omitted, reason=exhaustion_reason)
    return BudgetedStageRunResult(
        outcomes=tuple(outcomes),
        planned=len(items),
        omitted=omitted,
        exhaustion_reason=exhaustion_reason if omitted else None,
    )


def run_bounded_stage_calls(
    work: Sequence[W],
    invoke: Callable[[W], R],
    *,
    max_concurrency: int,
    thread_name_prefix: str,
) -> list[R]:
    """Execute independent calls concurrently while preserving planned order."""

    items = list(work)
    if len(items) <= 1 or max_concurrency == 1:
        return [invoke(item) for item in items]
    workers = min(max_concurrency, len(items))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        return list(executor.map(invoke, items))


def input_manifest_digest(inputs: Sequence[InferenceInputRef]) -> str:
    return payload_digest(
        {"inputs": [item.model_dump(mode="json") for item in inputs]}
    )


def request_identity(
    *,
    stage: StageName,
    protocol_version: str,
    mode: str | None = None,
    inputs: Sequence[InferenceInputRef],
    request_ordinal: int,
) -> str:
    digest = payload_digest(
        {
            "contract_version": INFERENCE_CONTRACT_VERSION,
            "stage": stage,
            "protocol_version": protocol_version,
            "mode": mode,
            "request_ordinal": request_ordinal,
            "input_manifest": [
                item.model_dump(mode="json") for item in inputs
            ],
        }
    )
    return f"INFER-{digest[:24]}"


def inference_cache_key(
    *,
    snapshot_digest: str,
    request_id: str,
    stage: StageName,
    mode: str | None,
    protocol_version: str,
    response_schema_version: str,
    protocol_manifest_digest: str,
    model_parameters: Mapping[str, object],
    input_manifest_digest_value: str,
    system_prompt_digest: str,
    request_payload_digest: str,
) -> str:
    """Bind reusable output to every semantic and transport-shaping input."""

    return payload_digest(
        {
            "cache_key_version": INFERENCE_CACHE_KEY_VERSION,
            "normalization_version": STRUCTURED_RESPONSE_NORMALIZATION_VERSION,
            "snapshot_digest": snapshot_digest,
            "request_id": request_id,
            "stage": stage,
            "mode": mode,
            "protocol_version": protocol_version,
            "response_schema_version": response_schema_version,
            "protocol_manifest_digest": protocol_manifest_digest,
            "model_parameters": dict(model_parameters),
            "input_manifest_digest": input_manifest_digest_value,
            "system_prompt_digest": system_prompt_digest,
            "request_payload_digest": request_payload_digest,
        }
    )


def inference_cache_key_for_record(record: InferenceRecord) -> str:
    return inference_cache_key(
        snapshot_digest=record.snapshot_digest,
        request_id=record.request_id,
        stage=record.stage,
        mode=record.mode,
        protocol_version=record.protocol_version,
        response_schema_version=record.response_schema_version,
        protocol_manifest_digest=record.protocol_manifest_digest,
        model_parameters=record.model_parameters.model_dump(mode="json"),
        input_manifest_digest_value=record.input_manifest_digest,
        system_prompt_digest=record.system_prompt_digest,
        request_payload_digest=record.request_payload_digest,
    )


def _attempt(
    *,
    ordinal: int,
    schema_attempt: int,
    output_limit_retry: int,
    provider_try: int,
    repair: bool,
    parsed: bool,
    provider_error: bool,
    retryable: bool,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None,
    response_content_bytes: int | None,
    reasoning_content_present: bool,
    reasoning_content_bytes: int,
    latency_ms: int,
    finish_reason: str | None,
    response_digest: str | None,
    errors: Sequence[str],
) -> InferenceAttempt:
    payload: dict[str, object] = {
        "attempt_ordinal": ordinal,
        "schema_attempt": schema_attempt,
        "output_limit_retry": output_limit_retry,
        "provider_try": provider_try,
        "repair": repair,
        "parsed": parsed,
        "provider_error": provider_error,
        "retryable": retryable,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_content_bytes": response_content_bytes,
        "reasoning_content_present": reasoning_content_present,
        "reasoning_content_bytes": reasoning_content_bytes,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "response_digest": response_digest,
        "validation_errors": [safe_error_summary(item) for item in errors],
    }
    payload["attempt_digest"] = payload_digest(payload)
    return InferenceAttempt.model_validate(payload)


def _usage(attempts: Sequence[InferenceAttempt]) -> InferenceUsage:
    reasoning_tokens = (
        sum(item.reasoning_tokens or 0 for item in attempts)
        if all(
            item.provider_error or item.reasoning_tokens is not None
            for item in attempts
        )
        else None
    )
    payload: dict[str, object] = {
        "calls": len(attempts),
        "input_tokens": sum(item.input_tokens for item in attempts),
        "output_tokens": sum(item.output_tokens for item in attempts),
        "reasoning_tokens": reasoning_tokens,
        "latency_ms": sum(item.latency_ms for item in attempts),
    }
    payload["usage_digest"] = payload_digest(payload)
    return InferenceUsage.model_validate(payload)


def _record(payload: dict[str, object]) -> InferenceRecord:
    payload.pop("record_digest", None)
    payload["record_digest"] = payload_digest(payload)
    return InferenceRecord.model_validate(payload)


@lru_cache(maxsize=128)
def _response_json_schema(
    response_model: type[BaseModel],
) -> dict[str, object]:
    """Cache immutable model schemas used by request planning and execution."""

    return response_model.model_json_schema(mode="validation")


def attach_response_json_schema(
    request_payload: str,
    response_model: type[BaseModel],
) -> str:
    """Attach the canonical response schema to one JSON request object."""

    value = json.loads(request_payload)
    if not isinstance(value, dict):
        raise ValueError("structured inference request payload must be a JSON object")
    if "response_json_schema" in value:
        raise ValueError("request payload already defines response_json_schema")
    value["response_json_schema"] = _response_json_schema(response_model)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def contractual_request_bytes(
    request_payload: str,
    response_model: type[BaseModel],
) -> int:
    """Measure the exact JSON request passed to a structured provider."""

    return len(
        attach_response_json_schema(request_payload, response_model).encode(
            "utf-8"
        )
    )


def make_skipped_inference_record(
    *,
    definition: StageDefinition,
    inputs: Sequence[InferenceInputRef],
    request_ordinal: int,
    request_payload: str,
    settings: LLMSettings,
    protocol_manifest_digest: str,
    snapshot_digest: str,
    reason: str,
    budget: InferenceBudget | None = None,
) -> InferenceRecord:
    """Record a bounded pre-provider terminal failure without fabricating a call."""

    ordered_inputs = list(inputs)
    usage = _usage(())
    contractual_request = attach_response_json_schema(
        request_payload,
        _provider_response_model(definition),
    )
    request_id = request_identity(
        stage=definition.stage,
        protocol_version=definition.protocol_version,
        mode=definition.mode,
        inputs=ordered_inputs,
        request_ordinal=request_ordinal,
    )
    model_parameters = resolved_model_parameters_payload(settings)
    manifest_digest = input_manifest_digest(ordered_inputs)
    system_digest = _digest_text(definition.system_prompt)
    request_digest = _digest_text(contractual_request)
    record = _record(
        {
            "contract_version": INFERENCE_CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": snapshot_digest,
            "request_id": request_id,
            "request_ordinal": request_ordinal,
            "stage": definition.stage,
            "mode": definition.mode,
            "protocol_version": definition.protocol_version,
            "response_schema_version": definition.response_schema_version,
            "protocol_manifest_digest": protocol_manifest_digest,
            "model_parameters": model_parameters,
            "input_manifest": [
                item.model_dump(mode="json") for item in ordered_inputs
            ],
            "input_manifest_digest": manifest_digest,
            "system_prompt_digest": system_digest,
            "request_payload_digest": request_digest,
            "schema_repairs": settings.schema_repairs,
            "output_limit_retries": settings.output_limit_retries,
            "provider_retries": settings.provider_retries,
            "terminal_status": "failure",
            "structured_response": None,
            "structured_response_digest": None,
            "attempts": [],
            "usage": usage.model_dump(mode="json"),
            "incomplete_reasons": [safe_error_summary(reason)],
        }
    )
    if budget is not None:
        budget.record_terminal(record)
    return record


def error_strings(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [
            safe_error_summary(
                ".".join(str(part) for part in item["loc"]) + ": " + item["msg"]
            )
            for item in error.errors(include_url=False)
        ]
    return [safe_error_summary(error)]


def _decode_json_object(content: str) -> dict[str, object]:
    """Accept strict JSON or one complete fenced JSON object, never prose scraping."""

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        lines = stripped.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip().casefold() not in {"```", "```json"}
            or lines[-1].strip() != "```"
        ):
            raise
        value = json.loads("\n".join(lines[1:-1]))
    if not isinstance(value, dict):
        raise ValueError("structured inference response must be one JSON object")
    return value


def _retry_delay_seconds(
    settings: LLMSettings,
    *,
    request_id: str,
    provider_try: int,
    retry_after_seconds: float | None,
) -> float:
    base = float(settings.provider_retry_delay_seconds)
    multiplier = float(settings.provider_retry_backoff_multiplier)
    maximum = float(settings.provider_retry_max_delay_seconds)
    jitter_limit = float(settings.provider_retry_jitter_seconds)
    exponential = base * (multiplier ** provider_try)
    deterministic_fraction = int(
        hashlib.sha256(
            f"{request_id}:{provider_try}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    ) / 0xFFFFFFFF
    delay = exponential + jitter_limit * deterministic_fraction
    if retry_after_seconds is not None:
        delay = max(delay, retry_after_seconds)
    return min(max(delay, 0.0), maximum)


def run_structured_inference(
    *,
    definition: StageDefinition,
    inputs: Sequence[InferenceInputRef],
    request_ordinal: int,
    request_payload: str,
    client: LLMClient,
    settings: LLMSettings,
    protocol_manifest_digest: str,
    snapshot_digest: str,
    budget: InferenceBudget,
) -> tuple[BaseModel | None, InferenceRecord]:
    """Run one logical request, recording every bounded terminal outcome."""

    if definition.stage not in LIVE_STAGES:
        raise ValueError(
            f"stage {definition.stage} is not live in schema {SCHEMA_VERSION}"
        )
    contractual_request = attach_response_json_schema(
        request_payload,
        _provider_response_model(definition),
    )
    ordered_inputs = list(inputs)
    request_id = request_identity(
        stage=definition.stage,
        protocol_version=definition.protocol_version,
        mode=definition.mode,
        inputs=ordered_inputs,
        request_ordinal=request_ordinal,
    )
    model_parameters = resolved_model_parameters_payload(settings)
    manifest_digest = input_manifest_digest(ordered_inputs)
    system_digest = _digest_text(definition.system_prompt)
    request_digest = _digest_text(contractual_request)
    cache_key = inference_cache_key(
        snapshot_digest=snapshot_digest,
        request_id=request_id,
        stage=definition.stage,
        mode=definition.mode,
        protocol_version=definition.protocol_version,
        response_schema_version=definition.response_schema_version,
        protocol_manifest_digest=protocol_manifest_digest,
        model_parameters=model_parameters,
        input_manifest_digest_value=manifest_digest,
        system_prompt_digest=system_digest,
        request_payload_digest=request_digest,
    )
    common: dict[str, object] = {
        "contract_version": INFERENCE_CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "request_id": request_id,
        "request_ordinal": request_ordinal,
        "stage": definition.stage,
        "mode": definition.mode,
        "protocol_version": definition.protocol_version,
        "response_schema_version": definition.response_schema_version,
        "protocol_manifest_digest": protocol_manifest_digest,
        "model_parameters": model_parameters,
        "input_manifest": [
            item.model_dump(mode="json") for item in ordered_inputs
        ],
        "input_manifest_digest": manifest_digest,
        "system_prompt_digest": system_digest,
        "request_payload_digest": request_digest,
        "schema_repairs": settings.schema_repairs,
        "output_limit_retries": settings.output_limit_retries,
        "provider_retries": settings.provider_retries,
    }

    if budget.cache is not None:
        cached_value = budget.cache.read(cache_key)
        if cached_value is not None:
            try:
                cached_record = InferenceRecord.model_validate(cached_value)
                if (
                    cached_record.terminal_status != "success"
                    or cached_record.structured_response is None
                    or inference_cache_key_for_record(cached_record) != cache_key
                    or cached_record.schema_repairs != settings.schema_repairs
                    or cached_record.output_limit_retries
                    != settings.output_limit_retries
                    or cached_record.provider_retries != settings.provider_retries
                    or (
                        _requires_effective_model_identity(settings)
                        and inference_record_model_identity_errors(
                            cached_record,
                            configured_model=settings.model,
                            accepted_effective_model_ids=(
                                settings.accepted_effective_model_ids
                            ),
                        )
                    )
                ):
                    raise ValueError("cached inference identity is invalid")
                cached_parsed = definition.response_model.model_validate(
                    definition.normalize_response(
                        cached_record.structured_response
                    )
                )
                try:
                    cached_parsed = definition.validate_response(cached_parsed)
                except (ValidationError, ValueError, TypeError):
                    if definition.terminal_validate_response is None:
                        raise
                    cached_parsed = definition.terminal_validate_response(
                        cached_parsed
                    )
                if (
                    cached_parsed.model_dump(mode="json")
                    != cached_record.structured_response
                ):
                    raise ValueError("cached response is not canonically normalized")
                if definition.validate_record_response is not None:
                    definition.validate_record_response(
                        cached_parsed,
                        cached_record,
                    )
            except (ValidationError, ValueError, TypeError):
                cached_parsed = None
            if cached_parsed is not None:
                try:
                    budget.record_terminal(cached_record, cache_hit=True)
                except ReplayBudgetAdmissionError:
                    # A production success cache never decides coverage.  An
                    # oversized saved trajectory is a strict miss; stable
                    # serial execution may still use the remaining live calls.
                    rejected = getattr(
                        budget.cache,
                        "record_admission_rejected",
                        None,
                    )
                    if callable(rejected):
                        rejected()
                    cached_parsed = None
                else:
                    budget.finalize_usage_overrun()
                    return cached_parsed, cached_record
        # Offline validation may expose an explicit terminal-replay method.
        # Production caches intentionally do not implement it and remain
        # success-only.  Exact replay of a saved failure is necessary because
        # its attempts may have consumed the logical budget that closes the
        # following omitted suffix.
        read_terminal = getattr(budget.cache, "read_terminal", None)
        terminal_value = (
            read_terminal(cache_key) if callable(read_terminal) else None
        )
        if terminal_value is not None:
            try:
                terminal_record = InferenceRecord.model_validate(terminal_value)
                if (
                    terminal_record.terminal_status == "success"
                    or terminal_record.structured_response is not None
                    or inference_cache_key_for_record(terminal_record) != cache_key
                    or terminal_record.schema_repairs != settings.schema_repairs
                    or terminal_record.output_limit_retries
                    != settings.output_limit_retries
                    or terminal_record.provider_retries != settings.provider_retries
                    or (
                        _requires_effective_model_identity(settings)
                        and inference_record_model_identity_errors(
                            terminal_record,
                            configured_model=settings.model,
                            accepted_effective_model_ids=(
                                settings.accepted_effective_model_ids
                            ),
                        )
                    )
                ):
                    raise ValueError("terminal replay identity is invalid")
            except (ValidationError, ValueError, TypeError):
                terminal_record = None
            if terminal_record is not None:
                try:
                    budget.record_terminal(terminal_record, replay_hit=True)
                except ReplayBudgetAdmissionError:
                    # Terminal replay is the validator's exact saved
                    # trajectory, so crossing the hard cap is artifact-invalid
                    # rather than a reason to alter it with live inference.
                    raise
                else:
                    budget.finalize_usage_overrun()
                    return None, terminal_record
    attempts: list[InferenceAttempt] = []
    prompt = contractual_request
    invalid_content = ""
    last_errors: list[str] = []
    parsed_result: BaseModel | None = None
    terminal = "failure"
    incomplete: list[str] = []
    stop = False
    for schema_attempt in range(settings.schema_repairs + 1):
        repair = schema_attempt > 0
        if repair:
            repair_response_model = (
                definition.repair_response_model(
                    invalid_content,
                    last_errors,
                )
                if definition.repair_response_model is not None
                else _provider_response_model(definition)
            )
            prompt = attach_response_json_schema(
                definition.repair_prompt(
                    request_payload, invalid_content, last_errors
                ),
                repair_response_model,
            )
        schema_response_invalid = False
        prompt_bytes = (
            len(prompt.encode("utf-8"))
            if definition.max_request_bytes is not None else 0
        )
        for output_limit_retry in range(settings.output_limit_retries + 1):
            retry_output_limit = False
            for provider_try in range(settings.provider_retries + 1):
                if (
                    definition.max_request_bytes is not None
                    and prompt_bytes > definition.max_request_bytes
                ):
                    # The prompt includes the actual dynamic schema, including
                    # repair wrappers. An unissued request is not an attempt
                    # and must not reserve a physical call or fabricate usage.
                    reason = f"max_{definition.stage}_request_bytes"
                    last_errors = [reason]
                    incomplete.append(reason)
                    stop = True
                    break
                exhausted = budget.reserve_call()
                if exhausted is not None:
                    last_errors = [f"inference budget exhausted: {exhausted}"]
                    incomplete.append(exhausted)
                    stop = True
                    break
                try:
                    bounded_complete = getattr(client, "complete_with_settings", None)
                    if callable(bounded_complete):
                        response = bounded_complete(
                            definition.system_prompt,
                            prompt,
                            settings=settings,
                        )
                    else:
                        response = client.complete(definition.system_prompt, prompt)
                except LLMProviderError as error:
                    attempt = _attempt(
                        ordinal=len(attempts),
                        schema_attempt=schema_attempt,
                        output_limit_retry=output_limit_retry,
                        provider_try=provider_try,
                        repair=repair,
                        parsed=False,
                        provider_error=True,
                        retryable=error.retryable,
                        model=settings.model,
                        input_tokens=0,
                        output_tokens=0,
                        reasoning_tokens=None,
                        response_content_bytes=None,
                        reasoning_content_present=False,
                        reasoning_content_bytes=0,
                        latency_ms=error.latency_ms,
                        finish_reason=None,
                        response_digest=None,
                        errors=[safe_error_summary(error)],
                    )
                    attempts.append(attempt)
                    last_errors = attempt.validation_errors
                    if error.retryable and provider_try < settings.provider_retries:
                        delay = _retry_delay_seconds(
                            settings,
                            request_id=request_id,
                            provider_try=provider_try,
                            retry_after_seconds=error.retry_after_seconds,
                        )
                        if delay:
                            time.sleep(delay)
                        continue
                    stop = True
                    break
                budget.record_usage(response.input_tokens, response.output_tokens)
                raw_digest = _digest_text(response.content)
                profile_violations = (
                    (
                        []
                        if not _requires_effective_model_identity(settings)
                        or response.model in settings.accepted_effective_model_ids
                        else [EFFECTIVE_MODEL_MISMATCH_VIOLATION]
                    )
                    + (
                        _disabled_thinking_profile_violations(
                            reasoning_tokens=response.reasoning_tokens,
                            reasoning_content_bytes=(
                                response.reasoning_content_bytes
                            ),
                        )
                        if settings.thinking_mode == "disabled"
                        else []
                    )
                )
                if profile_violations:
                    # The response may be syntactically valid, but it belongs to
                    # a different effective model profile.  It must neither be
                    # parsed nor repaired/retried into homogeneous results.
                    last_errors = profile_violations
                    attempts.append(
                        _attempt(
                            ordinal=len(attempts),
                            schema_attempt=schema_attempt,
                            output_limit_retry=output_limit_retry,
                            provider_try=provider_try,
                            repair=repair,
                            parsed=False,
                            provider_error=False,
                            retryable=False,
                            model=response.model,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            reasoning_tokens=response.reasoning_tokens,
                            response_content_bytes=response.response_content_bytes,
                            reasoning_content_present=(
                                response.reasoning_content_present
                            ),
                            reasoning_content_bytes=(
                                response.reasoning_content_bytes
                            ),
                            latency_ms=response.latency_ms,
                            finish_reason=response.finish_reason,
                            response_digest=raw_digest,
                            errors=profile_violations,
                        )
                    )
                    stop = True
                    break
                finish_reason = (response.finish_reason or "").casefold().replace(
                    "-", "_"
                )
                context_limit = (
                    "context_length" in finish_reason
                    or "context_window" in finish_reason
                )
                output_limit = finish_reason in {"length", "max_tokens"}
                response_byte_limit = (
                    definition.max_response_bytes is not None
                    and response.response_content_bytes is not None
                    and response.response_content_bytes
                    > definition.max_response_bytes
                )
                reported_output_limit = (
                    response.output_tokens > settings.max_output_tokens
                )
                resource_output_limit = (
                    response_byte_limit or reported_output_limit or output_limit
                )
                if resource_output_limit or context_limit:
                    last_errors = [
                        (
                            "provider context window was exceeded"
                            if context_limit
                            else (
                                "provider response exceeded max_response_bytes"
                                if response_byte_limit
                                else (
                                    "provider output exceeded max_output_tokens"
                                    if reported_output_limit
                                    else "provider output was truncated"
                                )
                            )
                        )
                    ]
                    can_retry_output = (
                        resource_output_limit
                        and not context_limit
                        and output_limit_retry < settings.output_limit_retries
                    )
                    attempts.append(
                        _attempt(
                            ordinal=len(attempts),
                            schema_attempt=schema_attempt,
                            output_limit_retry=output_limit_retry,
                            provider_try=provider_try,
                            repair=repair,
                            parsed=False,
                            provider_error=False,
                            retryable=can_retry_output,
                            model=response.model,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            reasoning_tokens=response.reasoning_tokens,
                            response_content_bytes=response.response_content_bytes,
                            reasoning_content_present=(
                                response.reasoning_content_present
                            ),
                            reasoning_content_bytes=(
                                response.reasoning_content_bytes
                            ),
                            latency_ms=response.latency_ms,
                            finish_reason=response.finish_reason,
                            response_digest=raw_digest,
                            errors=last_errors,
                        )
                    )
                    # A bounded output-limit retry repeats this exact prompt.
                    # Truncated content is never reflected into schema repair.
                    if can_retry_output:
                        retry_output_limit = True
                    else:
                        stop = True
                    break
                if definition.bind_response_digest is not None:
                    definition.bind_response_digest(raw_digest)
                if definition.bind_response_content is not None:
                    definition.bind_response_content(response.content)
                invalid_content = response.content
                validation_error: Exception | None = None
                normalized: object | None = None
                repeated_invalid_response = False
                try:
                    decoded = _decode_json_object(response.content)
                    trusted_raw = definition.validate_raw_response(decoded)
                    normalized = definition.normalize_response(trusted_raw)
                    parsed = definition.response_model.model_validate(
                        normalized
                    )
                except (ValidationError, ValueError, TypeError) as error:
                    repeated_invalid_response = _is_repeated_invalid_response(
                        attempts,
                        response_digest=raw_digest,
                        validation_errors=error_strings(error),
                        repair=repair,
                        schema_attempt=schema_attempt,
                        enabled=definition.stop_on_repeated_invalid_response,
                        early_stop_available=(
                            schema_attempt < settings.schema_repairs
                        ),
                    )
                    if (
                        (
                            definition.terminal_parse_before_repair
                            or schema_attempt == settings.schema_repairs
                            or repeated_invalid_response
                        )
                        and normalized is not None
                        and definition.terminal_parse_response is not None
                    ):
                        try:
                            parsed = definition.terminal_parse_response(
                                normalized,
                                error,
                            )
                            if (
                                schema_attempt == settings.schema_repairs
                                or repeated_invalid_response
                            ) and definition.terminal_validate_response is not None:
                                parsed = definition.terminal_validate_response(
                                    parsed
                                )
                            else:
                                parsed = definition.validate_response(parsed)
                        except (
                            ValidationError,
                            ValueError,
                            TypeError,
                        ) as terminal_error:
                            validation_error = terminal_error
                    else:
                        validation_error = error
                else:
                    try:
                        parsed = definition.validate_response(parsed)
                    except (ValidationError, ValueError, TypeError) as error:
                        repeated_invalid_response = _is_repeated_invalid_response(
                            attempts,
                            response_digest=raw_digest,
                            validation_errors=error_strings(error),
                            repair=repair,
                            schema_attempt=schema_attempt,
                            enabled=definition.stop_on_repeated_invalid_response,
                            early_stop_available=(
                                schema_attempt < settings.schema_repairs
                            ),
                        )
                        if (
                            (
                                schema_attempt == settings.schema_repairs
                                or repeated_invalid_response
                            )
                            and definition.terminal_validate_response is not None
                        ):
                            try:
                                parsed = definition.terminal_validate_response(
                                    parsed
                                )
                            except (
                                ValidationError,
                                ValueError,
                                TypeError,
                            ) as terminal_error:
                                validation_error = terminal_error
                        else:
                            validation_error = error
                if validation_error is not None:
                    last_errors = error_strings(validation_error)
                    # A terminal parser/validator may replace the error that
                    # originally triggered fallback.  Early closure is valid
                    # only when the final canonical error saved on this
                    # attempt still exactly repeats the immediately preceding
                    # invalid attempt.
                    repeated_invalid_response = _is_repeated_invalid_response(
                        attempts,
                        response_digest=raw_digest,
                        validation_errors=last_errors,
                        repair=repair,
                        schema_attempt=schema_attempt,
                        enabled=definition.stop_on_repeated_invalid_response,
                        early_stop_available=(
                            schema_attempt < settings.schema_repairs
                        ),
                    )
                    attempts.append(
                        _attempt(
                            ordinal=len(attempts),
                            schema_attempt=schema_attempt,
                            output_limit_retry=output_limit_retry,
                            provider_try=provider_try,
                            repair=repair,
                            parsed=False,
                            provider_error=False,
                            retryable=False,
                            model=response.model,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            reasoning_tokens=response.reasoning_tokens,
                            response_content_bytes=response.response_content_bytes,
                            reasoning_content_present=(
                                response.reasoning_content_present
                            ),
                            reasoning_content_bytes=(
                                response.reasoning_content_bytes
                            ),
                            latency_ms=response.latency_ms,
                            finish_reason=response.finish_reason,
                            response_digest=raw_digest,
                            errors=last_errors,
                        )
                    )
                    if repeated_invalid_response:
                        incomplete.append("repeated_invalid_response")
                        stop = True
                    else:
                        schema_response_invalid = True
                    break
                attempts.append(
                    _attempt(
                        ordinal=len(attempts),
                        schema_attempt=schema_attempt,
                        output_limit_retry=output_limit_retry,
                        provider_try=provider_try,
                        repair=repair,
                        parsed=True,
                        provider_error=False,
                        retryable=False,
                        model=response.model,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        reasoning_tokens=response.reasoning_tokens,
                        response_content_bytes=response.response_content_bytes,
                        reasoning_content_present=response.reasoning_content_present,
                        reasoning_content_bytes=response.reasoning_content_bytes,
                        latency_ms=response.latency_ms,
                        finish_reason=response.finish_reason,
                        response_digest=raw_digest,
                        errors=[],
                    )
                )
                parsed_result = parsed
                terminal = "success"
                stop = True
                break
            if stop or schema_response_invalid:
                break
            if retry_output_limit:
                continue
        if stop:
            break
    if terminal != "success":
        incomplete.extend(last_errors or ["structured_response_failed"])
    response_payload = (
        parsed_result.model_dump(mode="json") if parsed_result is not None else None
    )
    record_payload = {
        **common,
        "terminal_status": terminal,
        "structured_response": response_payload,
        "structured_response_digest": (
            payload_digest({"response": response_payload})
            if response_payload is not None
            else None
        ),
        "attempts": [item.model_dump(mode="json") for item in attempts],
        "usage": _usage(attempts).model_dump(mode="json"),
        "incomplete_reasons": sorted(set(incomplete)),
    }
    record = _record(record_payload)
    if len(_canonical_compact(record).encode("utf-8")) > (
        settings.max_inference_artifact_bytes
    ):
        record_payload.update(
            {
                "terminal_status": "failure",
                "structured_response": None,
                "structured_response_digest": None,
                "incomplete_reasons": sorted(
                    set(incomplete) | {"max_inference_artifact_bytes"}
                ),
            }
        )
        record = _record(record_payload)
        parsed_result = None
    if (
        parsed_result is not None
        and record.terminal_status == "success"
        and definition.validate_record_response is not None
    ):
        definition.validate_record_response(parsed_result, record)
    if (
        parsed_result is not None
        and record.terminal_status == "success"
        and budget.cache is not None
    ):
        budget.cache.write(
            cache_key,
            record.model_dump(mode="json"),
        )
    budget.record_terminal(record)
    # Direct or refinement invocations may not be followed by another stage
    # boundary query.  Finalize a strict token overshoot here when execution is
    # serial; permitted workers defer settlement until the full wave joins.
    budget.finalize_usage_overrun()
    return parsed_result, record


def make_protocol_manifest(
    *,
    settings: LLMSettings,
    execution_model_profile_id: str,
    execution_model_profile_digest: str,
    agent_profile_digest: str,
    repository_settings_digest: str,
    discovery_settings_digest: str,
    agent_settings_digest: str,
    chunking_settings_digest: str,
    observation_settings_digest: str,
    semantic_settings_digest: str,
    unified_settings_digest: str,
    target_content_admission_policy: str,
    agent_stage_protocol_versions: Mapping[str, str],
    explicit_prompt_version: str,
    explicit_response_schema_version: str,
    parser_contract_version: str,
    observation_contract_versions: Mapping[str, str],
    semantic_unit_contract_version: str,
    packet_contract_version: str,
    population_spec_contract_version: str,
    proposition_contract_version: str,
    stage_protocol_versions: Mapping[str, str] | None = None,
    additional_resource_limits: Mapping[str, int] | None = None,
) -> ProtocolManifest:
    payload: dict[str, object] = {
        "manifest_version": "1.11",
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "target_content_admission_policy": target_content_admission_policy,
        "provider": settings.provider,
        "endpoint_profile_digest": endpoint_profile_digest(
            provider=settings.provider,
            base_url=settings.base_url,
        ),
        "execution_model_profile_id": execution_model_profile_id,
        "execution_model_profile_digest": execution_model_profile_digest,
        "agent_profile_digest": agent_profile_digest,
        "repository_settings_digest": repository_settings_digest,
        "discovery_settings_digest": discovery_settings_digest,
        "agent_settings_digest": agent_settings_digest,
        "chunking_settings_digest": chunking_settings_digest,
        "observation_settings_digest": observation_settings_digest,
        "semantic_settings_digest": semantic_settings_digest,
        "unified_settings_digest": unified_settings_digest,
        "agent_stage_protocol_versions": dict(
            sorted(agent_stage_protocol_versions.items())
        ),
        "explicit_prompt_version": explicit_prompt_version,
        "explicit_response_schema_version": explicit_response_schema_version,
        "model_parameters": resolved_model_parameters_payload(settings),
        "reasoning_wire_policy": settings.reasoning_wire_policy,
        "require_response_model": settings.require_response_model,
        "accepted_effective_model_ids": (
            settings.accepted_effective_model_ids
        ),
        "parser_contract_version": parser_contract_version,
        "observation_contract_versions": dict(
            sorted(observation_contract_versions.items())
        ),
        "semantic_unit_contract_version": semantic_unit_contract_version,
        "packet_contract_version": packet_contract_version,
        "population_spec_contract_version": population_spec_contract_version,
        "proposition_contract_version": proposition_contract_version,
        "stage_protocol_versions": dict(
            sorted((stage_protocol_versions or {}).items())
        ),
        "inference_contract_version": INFERENCE_CONTRACT_VERSION,
        "thinking_wire_policy_version": "typed-reasoning-profile-1.1",
        "retry_repair_policy_version": (
            "schema-repair-provider-output-limit-retry-1.4"
        ),
        "resource_policy_version": RESOURCE_POLICY_VERSION,
        "schema_repairs": settings.schema_repairs,
        "output_limit_retries": settings.output_limit_retries,
        "provider_retries": settings.provider_retries,
        "provider_retry_delay_seconds": settings.provider_retry_delay_seconds,
        "provider_retry_backoff_multiplier": (
            settings.provider_retry_backoff_multiplier
        ),
        "provider_retry_max_delay_seconds": (
            settings.provider_retry_max_delay_seconds
        ),
        "provider_retry_jitter_seconds": settings.provider_retry_jitter_seconds,
        "resource_limits": {
            "max_concurrency": settings.max_concurrency,
            "max_provider_response_bytes": (
                settings.max_provider_response_bytes
            ),
            "max_total_calls": settings.max_total_calls,
            "max_total_input_tokens": settings.max_total_input_tokens,
            "max_total_output_tokens": settings.max_total_output_tokens,
            "max_inference_artifact_bytes": settings.max_inference_artifact_bytes,
            **dict(sorted((additional_resource_limits or {}).items())),
        },
    }
    payload["protocol_manifest_digest"] = payload_digest(payload)
    return ProtocolManifest.model_validate(payload)


def make_inference_index(
    *,
    snapshot_digest: str,
    protocol_manifest: ProtocolManifest,
    records: Sequence[InferenceRecord],
    artifact_sha256_by_id: Mapping[str, str],
    total_artifact_bytes: int,
    incomplete_reasons: Sequence[str] = (),
    stage_omissions: Mapping[str, tuple[int, Mapping[str, int]]] | None = None,
) -> InferenceIndex:
    ordered = sorted(records, key=lambda item: item.request_id)
    omissions = dict(stage_omissions or {})
    for stage, (count, reasons) in omissions.items():
        if stage not in LIVE_STAGES:
            raise ValueError(f"unknown omitted inference stage: {stage}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("inference stage omission count must be positive")
        if not reasons:
            raise ValueError("omitted inference stage requires a typed reason")
        if sum(reasons.values()) != count:
            raise ValueError("inference stage omission reason counts differ")
    refs = [
        InferenceRequestRef(
            request_id=item.request_id,
            stage=item.stage,
            artifact_path=f"inference/requests/{item.request_id}.json",
            artifact_sha256=artifact_sha256_by_id[item.request_id],
            record_digest=item.record_digest,
            terminal_status=item.terminal_status,
        )
        for item in ordered
    ]
    stage_names = sorted({item.stage for item in records} | set(omissions))
    stages = [
        InferenceStageCounts(
            stage=stage,
            planned=(
                sum(item.stage == stage for item in records)
                + omissions.get(stage, (0, {}))[0]
            ),
            success=sum(
                item.stage == stage and item.terminal_status == "success"
                for item in records
            ),
            uncertain=sum(
                item.stage == stage and item.terminal_status == "uncertain"
                for item in records
            ),
            failure=sum(
                item.stage == stage and item.terminal_status == "failure"
                for item in records
            ),
            omitted=omissions.get(stage, (0, {}))[0],
            omission_reasons=dict(
                sorted(omissions.get(stage, (0, {}))[1].items())
            ),
        )
        for stage in stage_names
    ]
    payload: dict[str, object] = {
        "index_version": INFERENCE_INDEX_VERSION,
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "protocol_manifest": protocol_manifest.model_dump(mode="json"),
        "requests": [item.model_dump(mode="json") for item in refs],
        "stages": [item.model_dump(mode="json") for item in stages],
        "total_calls": sum(item.usage.calls for item in records),
        "total_input_tokens": sum(item.usage.input_tokens for item in records),
        "total_output_tokens": sum(item.usage.output_tokens for item in records),
        "total_reasoning_tokens": (
            sum(item.usage.reasoning_tokens or 0 for item in records)
            if all(item.usage.reasoning_tokens is not None for item in records)
            else None
        ),
        "total_latency_ms": sum(item.usage.latency_ms for item in records),
        "total_artifact_bytes": total_artifact_bytes,
        "incomplete_reasons": sorted(
            set(incomplete_reasons)
            | {
                reason
                for _, reasons in omissions.values()
                for reason in reasons
            }
            | {
                reason
                for item in records
                for reason in item.incomplete_reasons
            }
        ),
    }
    payload["inference_set_digest"] = payload_digest(payload)
    return InferenceIndex.model_validate(payload)
