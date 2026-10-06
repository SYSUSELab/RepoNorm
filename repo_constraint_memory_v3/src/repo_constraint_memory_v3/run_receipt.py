"""Independent command-level timing and provider-cache observation receipts."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from repo_constraint_memory_v3 import __version__
from repo_constraint_memory_v3.agent_runtime import AgentModelExecutionReceipt
from repo_constraint_memory_v3.models import NonEmptyString, Sha256Hex, StrictModel


RUN_RECEIPT_SCHEMA_VERSION = "norm4code-command-run-receipt-1.0"
CachedInputSource = Literal[
    "source_agent_receipt",
    "core_openai_chat",
    "offline_mock",
]


def _canonical_default(value: object) -> object:
    if isinstance(value, StrictModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _canonical_text(value: object) -> str:
    if isinstance(value, StrictModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_default,
    ) + "\n"


def _digest(value: dict[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key != "receipt_digest"}
    return hashlib.sha256(_canonical_text(payload).encode("utf-8")).hexdigest()


class CachedInputTelemetry(StrictModel):
    """Provider prompt-cache tokens plus the denominator needed to interpret them."""

    source: CachedInputSource
    status: Literal["complete", "partial", "unavailable", "not_applicable"]
    tokens: int | None = Field(default=None, ge=0)
    provider_request_count: int | None = Field(default=None, ge=0)
    provider_response_count: int = Field(ge=0)
    responses_with_value: int = Field(ge=0)

    @model_validator(mode="after")
    def telemetry_is_interpretable(self) -> CachedInputTelemetry:
        if self.responses_with_value > self.provider_response_count:
            raise ValueError("cached-input reported responses exceed all responses")
        if (
            self.provider_request_count is not None
            and self.provider_response_count > self.provider_request_count
        ):
            raise ValueError("provider responses exceed provider requests")
        if (self.tokens is None) != (self.responses_with_value == 0):
            raise ValueError("cached-input tokens require at least one reported response")
        if self.status == "complete" and (
            self.provider_request_count is None
            or self.provider_response_count != self.provider_request_count
            or self.responses_with_value != self.provider_response_count
        ):
            raise ValueError("complete cached-input telemetry requires every request")
        if self.status == "partial" and self.responses_with_value == 0:
            raise ValueError("partial cached-input telemetry requires a reported value")
        if self.status == "unavailable" and self.responses_with_value != 0:
            raise ValueError("unavailable cached-input telemetry cannot report values")
        if self.status == "not_applicable" and (
            self.source != "offline_mock"
            or self.tokens is not None
            or self.provider_request_count != 0
            or self.provider_response_count != 0
        ):
            raise ValueError("not-applicable telemetry is reserved for offline mock runs")
        return self


def unavailable_cached_input(source: CachedInputSource) -> CachedInputTelemetry:
    return CachedInputTelemetry(
        source=source,
        status="unavailable",
        provider_request_count=None,
        provider_response_count=0,
        responses_with_value=0,
    )


def cached_input_from_agent_receipt(
    receipt: AgentModelExecutionReceipt,
) -> CachedInputTelemetry:
    if receipt.telemetry_status == "not_applicable":
        return CachedInputTelemetry(
            source="offline_mock",
            status="not_applicable",
            provider_request_count=0,
            provider_response_count=0,
            responses_with_value=0,
        )
    responses = len(receipt.calls)
    reported = [
        item.cached_input_tokens
        for item in receipt.calls
        if item.cached_input_tokens is not None
    ]
    requests = receipt.reported_model_call_count
    if not reported:
        status = "unavailable"
        tokens = None
    elif requests == responses == len(reported):
        status = "complete"
        tokens = sum(reported)
    else:
        status = "partial"
        tokens = sum(reported)
    return CachedInputTelemetry(
        source="source_agent_receipt",
        status=status,
        tokens=tokens,
        provider_request_count=requests,
        provider_response_count=responses,
        responses_with_value=len(reported),
    )


class CommandRunReceipt(StrictModel):
    schema_version: Literal["norm4code-command-run-receipt-1.0"] = (
        RUN_RECEIPT_SCHEMA_VERSION
    )
    package_version: NonEmptyString = __version__
    command: Literal["discover", "build", "build-current-source", "build-unified"]
    phase: Literal["running", "terminal"]
    outcome: Literal["running", "complete", "valid_partial", "failed", "interrupted"]
    exit_code: int | None = None
    started_at: NonEmptyString
    finished_at: NonEmptyString | None = None
    wall_seconds: float | None = Field(default=None, ge=0)
    wall_clock: Literal["time.monotonic_ns"] = "time.monotonic_ns"
    peak_rss_kib: int | None = Field(default=None, ge=0)
    self_peak_rss_kib: int | None = Field(default=None, ge=0)
    children_peak_rss_kib: int | None = Field(default=None, ge=0)
    peak_rss_scope: Literal[
        "maximum_of_cli_process_and_reaped_children_high_watermarks"
    ] = "maximum_of_cli_process_and_reaped_children_high_watermarks"
    repository: NonEmptyString
    output: NonEmptyString
    execution_model_profile_id: NonEmptyString | None = None
    execution_model_profile_digest: Sha256Hex | None = None
    provider_cached_input: CachedInputTelemetry | None = None
    receipt_digest: Sha256Hex

    @model_validator(mode="after")
    def receipt_is_consistent(self) -> CommandRunReceipt:
        terminal_fields = (
            self.finished_at,
            self.wall_seconds,
            self.peak_rss_kib,
            self.self_peak_rss_kib,
            self.children_peak_rss_kib,
        )
        if self.phase == "running":
            if self.outcome != "running" or self.exit_code is not None:
                raise ValueError("running receipt has terminal outcome")
            if any(item is not None for item in terminal_fields):
                raise ValueError("running receipt has terminal measurements")
            if self.provider_cached_input is not None:
                raise ValueError("running receipt has terminal provider telemetry")
        else:
            if self.outcome == "running" or any(
                item is None for item in terminal_fields
            ):
                raise ValueError("terminal receipt is missing measurements")
            if self.outcome == "interrupted":
                if self.exit_code is not None:
                    raise ValueError("interrupted receipt cannot invent an exit code")
            elif self.exit_code is None:
                raise ValueError("terminal receipt requires an exit code")
            if self.provider_cached_input is None:
                raise ValueError("terminal receipt requires cached-input status")
            if self.peak_rss_kib != max(
                self.self_peak_rss_kib or 0,
                self.children_peak_rss_kib or 0,
            ):
                raise ValueError("peak RSS differs from its documented components")
        if (self.execution_model_profile_id is None) != (
            self.execution_model_profile_digest is None
        ):
            raise ValueError("execution-model receipt identity is incomplete")
        if self.receipt_digest != _digest(self.model_dump(mode="json")):
            raise ValueError("command run receipt digest is invalid")
        return self


def _rss_kib(value: int | float) -> int:
    # getrusage reports bytes on macOS and KiB on Linux/BSD.
    return max(round(value / 1024), 0) if sys.platform == "darwin" else max(round(value), 0)


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_atomically(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".run-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _receipt(**payload: object) -> CommandRunReceipt:
    value: dict[str, object] = {
        "schema_version": RUN_RECEIPT_SCHEMA_VERSION,
        "package_version": __version__,
        "exit_code": None,
        "finished_at": None,
        "wall_seconds": None,
        "wall_clock": "time.monotonic_ns",
        "peak_rss_kib": None,
        "self_peak_rss_kib": None,
        "children_peak_rss_kib": None,
        "peak_rss_scope": (
            "maximum_of_cli_process_and_reaped_children_high_watermarks"
        ),
        "execution_model_profile_id": None,
        "execution_model_profile_digest": None,
        "provider_cached_input": None,
    }
    value.update(payload)
    value["receipt_digest"] = _digest(value)
    return CommandRunReceipt.model_validate(value)


class CommandRunRecorder:
    """Persist a crash-visible start marker and one atomically finalized receipt."""

    def __init__(
        self,
        path: Path,
        *,
        command: str,
        repository: Path,
        output: Path,
    ) -> None:
        self.path = path
        self.command = command
        self.repository = repository.resolve()
        self.output = output.resolve()
        self.started_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        self.started_ns = time.monotonic_ns()
        running = _receipt(
            command=command,
            phase="running",
            outcome="running",
            started_at=self.started_at,
            repository=str(self.repository),
            output=str(self.output),
        )
        _write_exclusive(self.path, _canonical_text(running))

    def finish(
        self,
        *,
        exit_code: int | None,
        interrupted: bool,
        execution_model_profile_id: str | None,
        execution_model_profile_digest: str | None,
        provider_cached_input: CachedInputTelemetry,
    ) -> CommandRunReceipt:
        finished_ns = time.monotonic_ns()
        finished_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self_peak = _rss_kib(self_usage.ru_maxrss)
        children_peak = _rss_kib(children_usage.ru_maxrss)
        if interrupted:
            outcome = "interrupted"
        elif exit_code == 0:
            outcome = "complete"
        elif exit_code == 1 and self.command.startswith("build"):
            outcome = "valid_partial"
        else:
            outcome = "failed"
        receipt = _receipt(
            command=self.command,
            phase="terminal",
            outcome=outcome,
            exit_code=exit_code,
            started_at=self.started_at,
            finished_at=finished_at,
            wall_seconds=round((finished_ns - self.started_ns) / 1_000_000_000, 9),
            peak_rss_kib=max(self_peak, children_peak),
            self_peak_rss_kib=self_peak,
            children_peak_rss_kib=children_peak,
            repository=str(self.repository),
            output=str(self.output),
            execution_model_profile_id=execution_model_profile_id,
            execution_model_profile_digest=execution_model_profile_digest,
            provider_cached_input=provider_cached_input,
        )
        _replace_atomically(self.path, _canonical_text(receipt))
        return receipt


def load_command_run_receipt(path: Path) -> CommandRunReceipt:
    return CommandRunReceipt.model_validate_json(path.read_text(encoding="utf-8"))
