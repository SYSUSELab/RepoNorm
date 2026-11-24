"""Frozen Qwen/DeepSeek transport for CodeWiki's sync and agentic API paths.

Receipts contain identity, counters and hashes, never credentials or message text.
All wire requests pass the same admission gate, including upstream repair calls.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI, OpenAI, DefaultHttpxClient, DefaultAsyncHttpxClient
from pydantic import BaseModel, ConfigDict, Field

from repo_constraint_memory_v3.model_profiles import ExecutionModelProfile


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


class AdapterError(RuntimeError):
    """A sanitized terminal error; never include provider bodies or credentials."""


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    # No added generation limits. Explicit values are for diagnostic probes.
    max_requests: int | None = Field(default=None, ge=1)
    max_request_bytes: int | None = Field(default=None, ge=1)
    max_total_request_bytes: int | None = Field(default=None, ge=1)
    max_reserved_output_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)


class CallReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["codewiki-profile-call-1.0"] = "codewiki-profile-call-1.0"
    execution_kind: Literal["real_provider", "offline_mock"] = "real_provider"
    call_id: int
    profile_id: str
    profile_digest: str
    canonical_model: str
    configured_model: str
    wire_model: str
    endpoint_digest: str
    thinking_mode: str
    thinking_effort: str | None
    wire_parameters: dict[str, Any]
    assistant_reasoning_replay: list[dict[str, Any]] = Field(default_factory=list)
    request_sha256: str
    request_bytes: int
    output_token_limit: int
    started_at: str
    ended_at: str | None = None
    wall_seconds: float | None = None
    status: str = "running"
    http_status: int | None = None
    response_sha256: str | None = None
    effective_model: str | None = None
    effective_version: str | None = None
    version_status: str = "unavailable"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    reasoning_content_bytes: int | None = None
    finish_reason: str | None = None
    wire_attestation: Literal["host_transport_observed"] = "host_transport_observed"


def integer(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


class ProfileRuntime:
    @staticmethod
    def validate_profile(profile: ExecutionModelProfile) -> None:
        if (profile.profile_kind != "real"
                or profile.provider.protocol != "openai_chat_completions"
                or profile.reasoning.wire_policy not in {
                    "qwen_enable_thinking", "deepseek_thinking_type",
                }):
            raise AdapterError("unsupported_execution_profile")

    def __init__(
        self,
        profile: ExecutionModelProfile,
        api_key: str,
        receipt_dir: Path,
        limits: Limits,
        *,
        sync_transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        execution_kind: Literal["real_provider", "offline_mock"] = "real_provider",
    ) -> None:
        self.validate_profile(profile)
        if not api_key:
            raise AdapterError("credential_unavailable")
        self.profile = profile
        self.execution_kind = execution_kind
        self.limits = limits
        self.receipt_dir = receipt_dir
        receipt_dir.mkdir(parents=True, exist_ok=False)
        self._key = api_key
        self._lock = threading.Lock()
        self.calls: list[CallReceipt] = []
        self.failure: str | None = None
        self.request_bytes = 0
        self.reserved_output_tokens = 0
        self._starts: dict[int, float] = {}
        self._sync_transport = sync_transport
        self._async_transport = async_transport
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def wire(self) -> dict[str, Any]:
        enabled = self.profile.reasoning.mode == "enabled"
        fields: dict[str, Any] = {
            "model": self.profile.adapter_model_ids.core_openai_chat,
            "temperature": self.profile.generation.temperature,
        }
        if self.profile.reasoning.wire_policy == "qwen_enable_thinking":
            fields.update(enable_thinking=enabled, preserve_thinking=True)
        else:
            fields["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if enabled:
            fields["reasoning_effort"] = self.profile.reasoning.effort
        return fields

    def _save(self, receipt: CallReceipt) -> None:
        target = self.receipt_dir / f"{receipt.call_id:05d}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    def stop(self, reason: str) -> None:
        self.failure = self.failure or reason
        raise AdapterError(self.failure)

    def prepare(self, request: httpx.Request) -> tuple[httpx.Request, CallReceipt]:
        with self._lock:
            if self.failure:
                raise AdapterError(self.failure)
            expected_url = self.profile.provider.base_url + "/chat/completions"
            if request.method != "POST" or str(request.url) != expected_url:
                self.stop("unexpected_provider_route")
            try:
                payload = json.loads(request.content)
                if not isinstance(payload, dict):
                    raise ValueError
            except (ValueError, TypeError):
                self.stop("invalid_request_json")
            if payload.get("stream"):
                self.stop("streaming_not_supported")
            for key, value in self.wire.items():
                if key in payload and (payload[key] != value or type(payload[key]) is not type(value)):
                    # Temperature may be represented as 1 or 1.0 by SDKs.
                    if key != "temperature" or type(payload[key]) not in (int, float) or payload[key] != value:
                        self.stop("conflicting_wire_parameters")
                payload[key] = value
            foreign_reasoning_fields = {
                "thinking", "reasoning", "reasoning_effort", "enable_thinking", "preserve_thinking",
            } - self.wire.keys()
            if foreign_reasoning_fields.intersection(payload):
                self.stop("conflicting_wire_parameters")
            token_keys = [k for k in ("max_tokens", "max_completion_tokens") if k in payload]
            tokens = integer(payload[token_keys[0]]) if len(token_keys) == 1 else None
            if not tokens or (self.limits.max_output_tokens is not None
                              and tokens > self.limits.max_output_tokens):
                self.stop("invalid_output_token_limit")
            body = json_bytes(payload)
            if any(limit is not None and value > limit for value, limit in (
                (len(self.calls) + 1, self.limits.max_requests),
                (len(body), self.limits.max_request_bytes),
                (self.request_bytes + len(body), self.limits.max_total_request_bytes),
                (self.reserved_output_tokens + tokens, self.limits.max_reserved_output_tokens),
            )):
                self.stop("resource_budget_exhausted")
            headers = dict(request.headers)
            headers.pop("content-length", None)
            bounded = httpx.Request(
                request.method, request.url, headers=headers, content=body,
                extensions=request.extensions,
            )
            call = CallReceipt(
                execution_kind=self.execution_kind,
                call_id=len(self.calls) + 1, profile_id=self.profile.profile_id,
                profile_digest=self.profile.profile_digest,
                canonical_model=self.profile.canonical_model_id,
                configured_model=self.profile.adapter_model_ids.core_openai_chat,
                wire_model=payload["model"],
                endpoint_digest=digest(self.profile.provider.base_url.encode()),
                thinking_mode=self.profile.reasoning.mode,
                thinking_effort=self.profile.reasoning.effort,
                wire_parameters=self.wire, request_sha256=digest(body),
                assistant_reasoning_replay=[
                    {"message_index": index, "bytes": len(message["reasoning_content"].encode()),
                     "sha256": digest(message["reasoning_content"].encode())}
                    for index, message in enumerate(payload.get("messages", []))
                    if message.get("role") == "assistant"
                    and isinstance(message.get("reasoning_content"), str)
                ],
                request_bytes=len(body), output_token_limit=tokens, started_at=now(),
            )
            self.calls.append(call)
            self._starts[call.call_id] = time.monotonic()
            self.request_bytes += len(body)
            self.reserved_output_tokens += tokens
            self._save(call)
            return bounded, call

    def finish(self, call: CallReceipt, response: httpx.Response) -> None:
        call.http_status = response.status_code
        call.response_sha256 = digest(response.content)
        try:
            if response.status_code != 200:
                call.status = "provider_http_error"
                return  # Let native SDK retries and CodeWiki cache repair handle HTTP errors.
            payload = response.json()
            raw_model = payload.get("model")
            # Never persist an arbitrary provider string (it could echo a secret).
            if raw_model not in self.profile.effective_identity.accepted_model_ids:
                self.stop("effective_model_missing_or_mismatch")
            call.effective_model = raw_model
            # The Qwen profile admits an alias, not a pinned version. Do not
            # substitute system_fingerprint or the configured name for a version.
            version = payload.get("model_version")
            if isinstance(version, str) and 0 < len(version) <= 128 and self._key not in version:
                call.effective_version = version
                call.version_status = "provider_reported"
            usage = payload.get("usage") or {}
            call.prompt_tokens = integer(usage.get("prompt_tokens"))
            call.completion_tokens = integer(usage.get("completion_tokens"))
            call.cached_input_tokens = integer((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
            call.reasoning_tokens = integer((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            choices = payload["choices"]
            if len(choices) != 1:
                self.stop("invalid_response_choices")
            message = choices[0]["message"]
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                self.stop("invalid_reasoning_content")
            call.reasoning_content_bytes = len(reasoning.encode()) if reasoning is not None else None
            finish_reason = choices[0].get("finish_reason")
            if finish_reason in {"stop", "length", "tool_calls", "content_filter", "function_call"}:
                call.finish_reason = finish_reason
            if self.profile.reasoning.mode == "disabled" and (
                (call.reasoning_tokens or 0) > 0 or (call.reasoning_content_bytes or 0) > 0
            ):
                self.stop("unexpected_reasoning")
            call.status = "success"
        except AdapterError:
            call.status = self.failure
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            call.status = "invalid_provider_response"
            self.stop(call.status)
        finally:
            call.ended_at = now()
            call.wall_seconds = time.monotonic() - self._starts[call.call_id]
            self._save(call)

    def transport_failure(self, call: CallReceipt) -> None:
        if call.status == "running":
            call.status = "provider_transport_error"
            call.ended_at = now()
            call.wall_seconds = time.monotonic() - self._starts[call.call_id]
            self._save(call)

    def summary(self) -> dict[str, Any]:
        responses = [c for c in self.calls if c.http_status is not None]
        def aggregate(field: str) -> dict[str, Any]:
            values = [getattr(c, field) for c in responses if getattr(c, field) is not None]
            return {
                "status": "unavailable" if not values else "complete" if len(values) == len(responses) else "partial",
                "known_total": sum(values) if values else None,
                "responses_with_value": len(values),
                "responses": len(responses), "requests": len(self.calls),
            }
        return {
            "requests": len(self.calls), "responses": len(responses),
            "http_200_responses": sum(c.http_status == 200 for c in responses),
            "successful_calls": sum(c.status == "success" for c in self.calls),
            "failure": self.failure, "request_bytes": self.request_bytes,
            "reserved_output_tokens": self.reserved_output_tokens,
            "prompt_tokens": aggregate("prompt_tokens"),
            "completion_tokens": aggregate("completion_tokens"),
            "cached_input_tokens": aggregate("cached_input_tokens"),
            "reasoning_tokens": aggregate("reasoning_tokens"),
            "reasoning_content_bytes": aggregate("reasoning_content_bytes"),
        }

    def create_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self._key, base_url=self.profile.provider.base_url,
                **({"timeout": self.limits.timeout_seconds}
                   if self.limits.timeout_seconds is not None else {}),
                http_client=DefaultHttpxClient(
                    transport=SyncTransport(self, self._sync_transport or httpx.HTTPTransport(retries=0)),
                    **({"timeout": self.limits.timeout_seconds}
                       if self.limits.timeout_seconds is not None else {}),
                    follow_redirects=False, trust_env=False,
                ),
            )
        return self._sync_client

    def create_model(self, config: Any) -> Any:
        from pydantic_ai.providers.openai import OpenAIProvider
        from codewiki.src.be.llm_services import CachingOpenAIModel
        if any(name != self.wire["model"] for name in (
            config.main_model, config.cluster_model, config.fallback_model,
        )):
            self.stop("conflicting_model_roles")
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._key, base_url=self.profile.provider.base_url,
                **({"timeout": self.limits.timeout_seconds}
                   if self.limits.timeout_seconds is not None else {}),
                http_client=DefaultAsyncHttpxClient(
                    transport=AsyncTransport(self, self._async_transport or httpx.AsyncHTTPTransport(retries=0)),
                    **({"timeout": self.limits.timeout_seconds}
                       if self.limits.timeout_seconds is not None else {}),
                    follow_redirects=False, trust_env=False,
                ),
            )
        return CachingOpenAIModel(
            self.wire["model"], provider=OpenAIProvider(openai_client=self._async_client),
            prompt_caching=config.prompt_caching,
            cache_registry_key=config.llm_base_url or "",
            settings={"max_tokens": config.max_tokens},
            profile={"openai_chat_thinking_field": "reasoning_content",
                     "openai_chat_send_back_thinking_parts": "field"},
        )

    async def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
        if self._async_client is not None:
            await self._async_client.close()


class SyncTransport(httpx.BaseTransport):
    def __init__(self, runtime: ProfileRuntime, inner: httpx.BaseTransport) -> None:
        self.runtime, self.inner = runtime, inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request, call = self.runtime.prepare(request)
        try:
            response = self.inner.handle_request(request)
            response.read()
            self.runtime.finish(call, response)
            return response
        except AdapterError:
            raise
        except Exception:
            self.runtime.transport_failure(call)
            raise httpx.ConnectError("provider_transport_error", request=request) from None

    def close(self) -> None:
        self.inner.close()


class AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, runtime: ProfileRuntime, inner: httpx.AsyncBaseTransport) -> None:
        self.runtime, self.inner = runtime, inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request, call = self.runtime.prepare(request)
        try:
            response = await self.inner.handle_async_request(request)
            await response.aread()
            self.runtime.finish(call, response)
            return response
        except AdapterError:
            raise
        except Exception:
            self.runtime.transport_failure(call)
            raise httpx.ConnectError("provider_transport_error", request=request) from None

    async def aclose(self) -> None:
        await self.inner.aclose()
