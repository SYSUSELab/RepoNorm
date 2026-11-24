"""Task-local mini-swe-agent call budgets and provider response telemetry.

Install once before creating workers. Enter ``task_telemetry`` inside each worker.
No prompts, responses, credential values or exception messages are persisted.
LiteLLM completion attempts are observable; its internal HTTP retries are not.
This module replaces, and must not be combined with, logical_call_budget hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from threading import RLock
from time import monotonic
from typing import Any, Callable, Iterator


PROTOCOL = "task-llm-telemetry-1.1"
METRICS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")


class LogicalLLMBudgetExceeded(RuntimeError):
    """The next logical call would exceed the task budget."""


class ModelIdentityViolation(RuntimeError):
    """A request or response violates the configured model/reasoning contract."""


class TelemetryPersistenceError(OSError):
    """A receipt could not be persisted; retrying the model would lose evidence."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_number(*values: Any) -> int | None:
    return next((number for value in values if (number := _number(value)) is not None), None)


def _response_record(
    response: Any,
    expected_model: str,
    expected_provider: str | None = None,
    expected_reasoning_mode: str = "disabled",
    accepted_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    usage = _get(response, "usage")
    prompt_details = _get(usage, "prompt_tokens_details")
    completion_details = _get(usage, "completion_tokens_details")
    tokens = {
        "input_tokens": _first_number(_get(usage, "prompt_tokens"), _get(usage, "input_tokens")),
        "output_tokens": _first_number(_get(usage, "completion_tokens"), _get(usage, "output_tokens")),
        "cached_input_tokens": _first_number(
            _get(prompt_details, "cached_tokens"), _get(usage, "prompt_cache_hit_tokens")
        ),
        "reasoning_tokens": _first_number(
            _get(completion_details, "reasoning_tokens"), _get(usage, "reasoning_tokens")
        ),
    }
    model = _get(response, "model")
    model = model if isinstance(model, str) and model else None
    reasoning_present = False
    for choice in _get(response, "choices", []) or []:
        message = _get(choice, "message")
        fields = _get(message, "provider_specific_fields")
        for name in ("reasoning_content", "reasoning", "thinking", "thinking_blocks", "reasoning_details"):
            if _get(message, name) or _get(fields, name):
                reasoning_present = True
    provider = _get(response, "provider")
    violations = []
    if expected_provider is not None and provider != expected_provider:
        violations.append("effective_provider_missing" if provider is None else "effective_provider_mismatch")
    if model is None:
        violations.append("effective_model_missing")
    elif model not in (accepted_models or (expected_model,)):
        violations.append("effective_model_mismatch")
    if expected_reasoning_mode == "disabled":
        if (tokens["reasoning_tokens"] or 0) > 0:
            violations.append("nonthinking_reasoning_tokens")
        if reasoning_present:
            violations.append("nonthinking_reasoning_content")
    elif expected_reasoning_mode != "enabled":
        violations.append("configured_reasoning_mode_invalid")
    return {
        "effective_model": model,
        "provider": provider,
        "sdk_provider": _get(_get(response, "_hidden_params"), "custom_llm_provider"),
        "provider_identity_source": "response.provider; not upstream weight attestation",
        "reasoning_details_present": any(bool(_get(_get(choice, "message"), "reasoning_details")
            or _get(_get(_get(choice, "message"), "provider_specific_fields"), "reasoning_details"))
            for choice in _get(response, "choices", []) or []),
        "effective_model_version": _get(response, "model_version"),
        "system_fingerprint": _get(response, "system_fingerprint"),
        "identity_source": "litellm_normalized_response.model",
        "tokens": tokens,
        "cache_creation_input_tokens": _first_number(
            _get(prompt_details, "cache_creation_input_tokens"),
            _get(prompt_details, "cache_write_tokens"),
        ),
        "reasoning_content_present": reasoning_present,
        "violations": violations,
    }


def _metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [attempt["response"] for attempt in attempts if attempt.get("response") is not None]
    result = {}
    for metric in METRICS:
        values = [response["tokens"][metric] for response in responses if response["tokens"][metric] is not None]
        # No-response failures can have consumed tokens. Preserve their denominator.
        status = "unavailable" if not values else "complete" if len(values) == len(attempts) else "partial"
        result[metric] = {
            "value": sum(values) if values else None,
            "status": status,
            "responses_with_value": len(values),
            "responses_total": len(responses),
            "completion_attempts_total": len(attempts),
        }
    return result


def _wire_record(kwargs: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Inspect only configured SDK arguments, never claim to observe HTTP bytes."""
    violations = []
    for key in ("model", "api_base", "extra_body", "drop_params"):
        if key not in kwargs or kwargs[key] != expected.get(key):
            violations.append(f"sdk_request_{key}_mismatch")
    if "temperature" in expected:
        if kwargs.get("temperature") != expected["temperature"]:
            violations.append("sdk_request_temperature_mismatch")
    elif "temperature" in kwargs:
        violations.append("sdk_request_temperature_unexpected")
    if ("max_tokens" in kwargs or "max_tokens" in expected) and (
        kwargs.get("max_tokens") != expected.get("max_tokens")
    ):
        violations.append("sdk_request_max_tokens_mismatch")
    for key in ("timeout", "max_retries", "num_retries"):
        if key in expected and kwargs.get(key) != expected[key]:
            violations.append(f"sdk_request_{key}_mismatch")
    if kwargs.get("drop_params") is not False:
        violations.append("sdk_request_drop_params_not_false")
    extra_body = kwargs.get("extra_body")
    extra_body = extra_body if isinstance(extra_body, dict) else {}
    reasoning = extra_body.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    expected_extra = expected.get("extra_body") or {}
    if "reasoning_effort" in kwargs or (
        "reasoning_effort" in extra_body and extra_body.get("reasoning_effort") != expected_extra.get("reasoning_effort")
    ):
        violations.append("sdk_request_reasoning_effort_present")
    endpoint = kwargs.get("api_base")
    return {
        "source": "configured_litellm_completion_kwargs; not independent HTTP wire attestation",
        "model": kwargs.get("model"),
        "endpoint_digest": sha256(endpoint.encode()).hexdigest() if isinstance(endpoint, str) else None,
        "temperature": kwargs.get("temperature", extra_body.get("temperature")),
        "temperature_transport": "top_level" if "temperature" in kwargs else "extra_body",
        "thinking_type": _get(_get(extra_body, "thinking"), "type"),
        "enable_thinking": extra_body.get("enable_thinking"),
        "thinking_budget": extra_body.get("thinking_budget"),
        "reasoning_enabled": reasoning.get("enabled"),
        "reasoning_effort": reasoning.get("effort", extra_body.get("reasoning_effort")),
        "reasoning_max_tokens": reasoning.get("max_tokens"),
        "max_tokens": kwargs.get("max_tokens"),
        **{key: kwargs.get(key) for key in ("timeout", "max_retries", "num_retries") if key in expected},
        "provider_routing": deepcopy(extra_body.get("provider")),
        "drop_params": kwargs.get("drop_params"),
        "reasoning_effort_present": "reasoning_effort" in kwargs or "reasoning_effort" in extra_body,
        "status": "violation" if violations else "matched",
        "violations": violations,
    }


@dataclass
class TaskTelemetry:
    limit: int
    expected_model: str
    accepted_models: tuple[str, ...] = ()
    receipt_callback: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)
    expected_wire: dict[str, Any] | None = field(default=None, repr=False)
    expected_provider: str | None = None
    expected_reasoning_mode: str = "disabled"
    expected_reasoning_effort: str | None = None
    expected_reasoning_max_tokens: int | None = None
    prompt_cache_policy: str | None = None
    reasoning_content_replay_compat: str | None = None
    reasoning_content_replay_injections: int = 0
    started_at: str = field(default_factory=_utc)
    ended_at: str | None = None
    exhausted: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=monotonic, repr=False)
    _elapsed: float | None = field(default=None, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def begin(self, kind: str) -> dict[str, Any]:
        with self._lock:
            if len(self.calls) >= self.limit:
                self.exhausted = True
                self.notify()
                raise LogicalLLMBudgetExceeded(f"max_logical_llm_calls={self.limit} exhausted")
            call = {
                "index": len(self.calls) + 1, "kind": kind, "started_at": _utc(),
                "ended_at": None, "elapsed_seconds": None, "status": "running",
                "error_type": None, "completion_attempts": [],
                "sdk_request_checks": [],
            }
            self.calls.append(call)
            self.notify()
            return call

    def notify(self) -> None:
        """Persist a detached snapshot. Persistence failures stop further work."""
        if self.receipt_callback is not None:
            try:
                self.receipt_callback(self.receipt())
            except Exception as error:
                raise TelemetryPersistenceError("Could not persist task telemetry") from error

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            calls = deepcopy(self.calls)
            attempts = [attempt for call in calls for attempt in call["completion_attempts"]]
            responses = [attempt["response"] for attempt in attempts if attempt.get("response") is not None]
            request_checks = [check for call in calls for check in call["sdk_request_checks"]]
            violations = [{"call": call["index"], "attempt": attempt["index"], "reason": reason}
                          for call in calls for attempt in call["completion_attempts"]
                          for reason in (attempt.get("response") or {}).get("violations", [])]
            violations.extend({"call": call["index"], "sdk_request_check": index, "reason": reason}
                              for call in calls for index, check in enumerate(call["sdk_request_checks"], 1)
                              for reason in check["violations"])
            kinds = {}
            for kind in ("solver", "compaction"):
                subset = [call for call in calls if call["kind"] == kind]
                selected = [attempt for call in subset for attempt in call["completion_attempts"]]
                kinds[kind] = {
                    "logical_calls": len(subset),
                    "succeeded": sum(call["status"] == "succeeded" for call in subset),
                    "failed": sum(call["status"] == "failed" for call in subset),
                    "tokens": _metrics(selected),
                }
            return {
                "protocol": PROTOCOL, "limit": self.limit, "total": len(calls),
                "by_kind": {kind: value["logical_calls"] for kind, value in kinds.items()},
                "exhausted": self.exhausted, "started_at": self.started_at, "ended_at": self.ended_at,
                "elapsed_seconds": self._elapsed if self._elapsed is not None else monotonic() - self._started,
                "succeeded": sum(call["status"] == "succeeded" for call in calls),
                "failed": sum(call["status"] == "failed" for call in calls),
                "tokens": _metrics(attempts), "by_component": kinds,
                "usage_scope": "Observed LiteLLM responses only; unobserved internal SDK retries may consume additional tokens. Completeness is relative to recorded completion attempts.",
                "sdk_completion_attempts": len(attempts),
                "sdk_completion_retries": sum(max(0, len(call["completion_attempts"]) - 1) for call in calls),
                "provider_responses": len(responses),
                "actual_http_requests": {"value": None, "status": "unavailable"},
                "internal_sdk_retries": {"value": None, "status": "unavailable"},
                "expected_model": self.expected_model,
                "accepted_models": list(self.accepted_models or (self.expected_model,)),
                "expected_provider": self.expected_provider,
                "expected_reasoning_mode": self.expected_reasoning_mode,
                "expected_reasoning_effort": self.expected_reasoning_effort,
                "expected_reasoning_max_tokens": self.expected_reasoning_max_tokens,
                "prompt_cache_policy": self.prompt_cache_policy,
                "reasoning_content_replay_compat": self.reasoning_content_replay_compat,
                "reasoning_content_replay_injections": self.reasoning_content_replay_injections,
                "effective_models": sorted({response["effective_model"] for response in responses if response["effective_model"]}),
                "effective_providers": sorted({response["provider"] for response in responses if isinstance(response.get("provider"), str)}),
                "effective_model_versions": sorted({response["effective_model_version"] for response in responses if isinstance(response.get("effective_model_version"), str)}),
                "responses_with_model_version": sum(bool(response.get("effective_model_version")) for response in responses),
                "identity_status": "violation" if violations else "observed" if responses else "unavailable",
                "identity_source": "litellm_normalized_response.model; not independent wire attestation",
                "configured_sdk_request_attestation": {
                    "status": "violation" if any(check["violations"] for check in request_checks)
                              else "matched" if request_checks else "unavailable",
                    "checks": len(request_checks),
                    "source": "configured_litellm_completion_kwargs; not independent HTTP wire attestation",
                },
                "violations": violations,
                "calls": calls,
            }


_CURRENT: ContextVar[TaskTelemetry | None] = ContextVar("n4c_task_telemetry", default=None)
_CALL: ContextVar[dict[str, Any] | None] = ContextVar("n4c_task_logical_call", default=None)
_INSTALL_LOCK = RLock()


@contextmanager
def task_telemetry(
    limit: int = 200,
    expected_model: str = "deepseek-v4-flash-260425",
    receipt_callback: Callable[[dict[str, Any]], None] | None = None,
    expected_wire: dict[str, Any] | None = None,
    expected_provider: str | None = None,
    expected_reasoning_mode: str = "disabled",
    expected_reasoning_effort: str | None = None,
    expected_reasoning_max_tokens: int | None = None,
    prompt_cache_policy: str | None = None,
    reasoning_content_replay_compat: str | None = None,
    accepted_models: tuple[str, ...] | list[str] | None = None,
) -> Iterator[TaskTelemetry]:
    if prompt_cache_policy is not None:
        from dashscope_profile import CACHE_POLICY, ENDPOINT
        if (prompt_cache_policy != CACHE_POLICY or not expected_wire
                or expected_wire.get('api_base') != ENDPOINT
                or expected_wire.get('model') != 'openai/qwen3.5-flash'):
            raise ValueError('Invalid DashScope cache binding')
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not expected_model:
        raise ValueError("expected_model must be nonempty")
    accepted_models = tuple(accepted_models or (expected_model,))
    if (
        expected_model not in accepted_models
        or len(set(accepted_models)) != len(accepted_models)
        or any(not isinstance(model, str) or not model for model in accepted_models)
    ):
        raise ValueError("accepted_models must be unique nonempty strings containing expected_model")
    if expected_reasoning_mode not in {"disabled", "enabled"}:
        raise ValueError("expected_reasoning_mode must be disabled or enabled")
    if expected_reasoning_max_tokens is not None and (
        not isinstance(expected_reasoning_max_tokens, int)
        or isinstance(expected_reasoning_max_tokens, bool)
        or expected_reasoning_max_tokens <= 0
    ):
        raise ValueError("expected_reasoning_max_tokens must be a positive integer")
    if expected_reasoning_mode == "disabled" and (
        expected_reasoning_effort is not None
        or expected_reasoning_max_tokens is not None
    ):
        raise ValueError("disabled reasoning cannot carry a control")
    if (
        expected_reasoning_effort is not None
        and expected_reasoning_max_tokens is not None
    ):
        raise ValueError("reasoning effort and max_tokens are mutually exclusive")
    state = TaskTelemetry(
        limit=limit,
        expected_model=expected_model,
        accepted_models=accepted_models,
        receipt_callback=receipt_callback,
        expected_wire=deepcopy(expected_wire),
        expected_provider=expected_provider,
        expected_reasoning_mode=expected_reasoning_mode,
        expected_reasoning_effort=expected_reasoning_effort,
        expected_reasoning_max_tokens=expected_reasoning_max_tokens,
        prompt_cache_policy=prompt_cache_policy,
        reasoning_content_replay_compat=reasoning_content_replay_compat,
    )
    token = _CURRENT.set(state)
    call_token = _CALL.set(None)
    try:
        state.notify()
        yield state
    finally:
        try:
            with state._lock:
                state.ended_at = _utc()
                state._elapsed = monotonic() - state._started
                state.notify()
        finally:
            _CALL.reset(call_token)
            _CURRENT.reset(token)


def install_hooks(model_class: Any = None, completion_module: Any = None) -> None:
    """Install idempotent hooks; optional arguments permit dependency-free tests."""
    if model_class is None or completion_module is None:
        from cooperbench.agents.mini_swe_agent_v2.models import litellm_model
        model_class = model_class or litellm_model.LitellmModel
        completion_module = completion_module or litellm_model.litellm
    with _INSTALL_LOCK:
        if getattr(model_class, "_n4c_logical_call_budget_installed", False):
            raise RuntimeError("Remove legacy logical_call_budget hooks before installing task telemetry")
        if getattr(model_class, "_n4c_task_telemetry_installed", False):
            return

        original_prepare = getattr(model_class, "_prepare_messages_for_api", None)
        if original_prepare is not None:
            @wraps(original_prepare)
            def prepare_messages_for_api(self: Any, messages: list[dict]) -> list[dict]:
                prepared = original_prepare(self, messages)
                state = _CURRENT.get()
                if state is None or state.reasoning_content_replay_compat != "agicto_space_for_empty_tool_reasoning":
                    return prepared
                # AGICTO's DeepSeek-compatible thinking endpoint requires the
                # assistant reasoning field to survive every continued round,
                # including synthetic compaction summaries.  The
                # endpoint accepts an empty value, but LiteLLM drops empty
                # extension fields while normalizing the next request.  A
                # single space is semantically empty yet remains on the wire.
                for message in prepared:
                    if (
                        message.get("role") == "assistant"
                        and not message.get("reasoning_content")
                    ):
                        message["reasoning_content"] = " "
                        with state._lock:
                            state.reasoning_content_replay_injections += 1
                            state.notify()
                return prepared

            model_class._prepare_messages_for_api = prepare_messages_for_api

        def logical_wrapper(original: Any, kind: str) -> Any:
            @wraps(original)
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                state = _CURRENT.get()
                if state is None:
                    return original(self, *args, **kwargs)
                call = state.begin(kind)
                token = _CALL.set(call)
                started = monotonic()
                try:
                    result = original(self, *args, **kwargs)
                except BaseException as error:
                    with state._lock:
                        call.update(status="failed", error_type=type(error).__name__)
                    raise
                else:
                    with state._lock:
                        call["status"] = "succeeded"
                    return result
                finally:
                    with state._lock:
                        call.update(ended_at=_utc(), elapsed_seconds=monotonic() - started)
                    try:
                        state.notify()
                    finally:
                        _CALL.reset(token)
            return wrapped

        if not getattr(completion_module, "_n4c_task_telemetry_installed", False):
            original_completion = completion_module.completion

            @wraps(original_completion)
            def completion(*args: Any, **kwargs: Any) -> Any:
                state, call = _CURRENT.get(), _CALL.get()
                if state is None or call is None:
                    return original_completion(*args, **kwargs)
                if state.prompt_cache_policy is not None:
                    from dashscope_profile import cache_messages
                    kwargs['messages'] = cache_messages(kwargs['messages'])
                if state.expected_wire is not None:
                    check = _wire_record(kwargs, state.expected_wire)
                    with state._lock:
                        call["sdk_request_checks"].append(check)
                        state.notify()
                    if check["violations"]:
                        raise ModelIdentityViolation("Configured SDK request violates model profile: " + ", ".join(check["violations"]))
                with state._lock:
                    attempt = {"index": len(call["completion_attempts"]) + 1,
                               "started_at": _utc(), "ended_at": None, "elapsed_seconds": None,
                               "status": "running", "error_type": None, "response": None}
                    call["completion_attempts"].append(attempt)
                    state.notify()
                started = monotonic()
                try:
                    response = original_completion(*args, **kwargs)
                    record = _response_record(
                        response,
                        state.expected_model,
                        state.expected_provider,
                        state.expected_reasoning_mode,
                        state.accepted_models,
                    )
                    with state._lock:
                        attempt.update(response=record, status="succeeded")
                        state.notify()
                    if record["violations"]:
                        raise ModelIdentityViolation(
                            "Provider response violates model profile: " + ", ".join(record["violations"])
                        )
                    return response
                except BaseException as error:
                    with state._lock:
                        attempt.update(status="failed", error_type=type(error).__name__)
                    raise
                finally:
                    with state._lock:
                        attempt.update(ended_at=_utc(), elapsed_seconds=monotonic() - started)
                        state.notify()

            completion_module.completion = completion
            completion_module._n4c_task_telemetry_installed = True
        model_class.query = logical_wrapper(model_class.query, "solver")
        model_class.summarize_context = logical_wrapper(model_class.summarize_context, "compaction")
        model_class.abort_exceptions = list(model_class.abort_exceptions) + [
            ModelIdentityViolation, TelemetryPersistenceError,
        ]
        model_class._n4c_task_telemetry_installed = True
