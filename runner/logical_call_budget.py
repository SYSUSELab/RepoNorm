"""A harness-neutral budget for logical LLM calls.

A logical call is one top-level request made by an agent component. Provider
retries performed inside that request are deliberately not counted again.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Iterator


PROTOCOL = "logical-llm-call-1.0"


class LogicalLLMBudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured logical budget."""


@dataclass
class LogicalCallState:
    limit: int
    receipt_callback: Callable[[dict], None] | None = field(default=None, repr=False)
    total: int = 0
    by_kind: Counter[str] = field(default_factory=Counter)
    exhausted: bool = False
    provider_responses: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    responses_with_usage: int = 0
    responses_with_model: int = 0
    responses_with_cache_read_field: int = 0
    effective_models: set[str] = field(default_factory=set)
    calls_with_reasoning_tokens_nonzero: int = 0
    calls_with_reasoning_content: int = 0

    def reserve(self, kind: str) -> None:
        if self.total >= self.limit:
            self.exhausted = True
            self.notify()
            raise LogicalLLMBudgetExceeded(
                f"max_logical_llm_calls={self.limit} exhausted"
            )
        self.total += 1
        self.by_kind[kind] += 1
        self.notify()

    def notify(self) -> None:
        if self.receipt_callback:
            self.receipt_callback(self.receipt())

    def record_response(self, response: object) -> None:
        """Record one successful provider response before trajectory compaction."""
        try:
            value = response.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            value = response if isinstance(response, dict) else {}
        self.provider_responses += 1
        model = value.get("model")
        if model:
            self.responses_with_model += 1
            self.effective_models.add(str(model))
        usage = value.get("usage") or {}
        if usage:
            self.responses_with_usage += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in prompt_details and prompt_details.get("cached_tokens") is not None:
            self.responses_with_cache_read_field += 1
            self.cache_read_tokens += int(prompt_details.get("cached_tokens") or 0)
        completion_details = usage.get("completion_tokens_details") or {}
        if completion_details.get("reasoning_tokens"):
            self.calls_with_reasoning_tokens_nonzero += 1
        message = ((value.get("choices") or [{}])[0].get("message") or {})
        if message.get("reasoning_content") not in (None, ""):
            self.calls_with_reasoning_content += 1
        self.notify()

    def receipt(self) -> dict[str, object]:
        return {
            "protocol": PROTOCOL,
            "limit": self.limit,
            "total": self.total,
            "by_kind": dict(sorted(self.by_kind.items())),
            "exhausted": self.exhausted,
            "provider_usage": {
                "successful_responses": self.provider_responses,
                "responses_with_usage": self.responses_with_usage,
                "responses_with_model": self.responses_with_model,
                "responses_with_cache_read_field": self.responses_with_cache_read_field,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "effective_models": sorted(self.effective_models),
                "calls_with_reasoning_tokens_nonzero": self.calls_with_reasoning_tokens_nonzero,
                "calls_with_reasoning_content": self.calls_with_reasoning_content,
            },
        }


_CURRENT: ContextVar[LogicalCallState | None] = ContextVar(
    "logical_llm_call_budget", default=None
)
_INSTALL_LOCK = Lock()


def reserve(kind: str) -> None:
    state = _CURRENT.get()
    if state is not None:
        state.reserve(kind)


def record_response(response: object) -> None:
    state = _CURRENT.get()
    if state is not None:
        state.record_response(response)


@contextmanager
def logical_call_budget(limit: int, receipt_callback: Callable[[dict], None] | None = None) -> Iterator[LogicalCallState]:
    if limit <= 0:
        raise ValueError("max_logical_llm_calls must be a positive integer")
    state = LogicalCallState(limit=limit, receipt_callback=receipt_callback)
    token = _CURRENT.set(state)
    try:
        state.notify()
        yield state
    finally:
        try:
            state.notify()
        finally:
            _CURRENT.reset(token)


def install_mini_swe_agent_hooks() -> None:
    """Count mini-swe-agent solver and compaction requests exactly once."""
    from cooperbench.agents.mini_swe_agent_v2.models.litellm_model import (
        LitellmModel,
    )

    with _INSTALL_LOCK:
        if getattr(LitellmModel, "_n4c_logical_call_budget_installed", False):
            return

        original_query = LitellmModel.query
        original_raw_query = LitellmModel._query
        original_summarize = LitellmModel.summarize_context

        def raw_query(self, *args, **kwargs):
            response = original_raw_query(self, *args, **kwargs)
            record_response(response)
            return response

        def query(self, *args, **kwargs):
            reserve("solver")
            return original_query(self, *args, **kwargs)

        def summarize_context(self, *args, **kwargs):
            reserve("condenser")
            message = original_summarize(self, *args, **kwargs)
            record_response(((message.get("extra") or {}).get("response") or {}))
            return message

        LitellmModel._query = raw_query
        LitellmModel.query = query
        LitellmModel.summarize_context = summarize_context
        LitellmModel._n4c_logical_call_budget_installed = True
