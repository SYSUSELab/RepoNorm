from __future__ import annotations

from repo_constraint_memory_v3.agent_runtime import (
    AgentModelCallReceipt,
    AgentModelExecutionReceipt,
)
from repo_constraint_memory_v3.run_receipt import cached_input_from_agent_receipt


def _agent_receipt(cached: list[int | None]) -> AgentModelExecutionReceipt:
    calls = [
        AgentModelCallReceipt(
            call_ordinal=index,
            configured_model_id="openai/test-model",
            effective_model_id="test-model",
            input_tokens=10,
            cached_input_tokens=value,
            output_tokens=2,
            reasoning_tokens=0,
        )
        for index, value in enumerate(cached)
    ]
    return AgentModelExecutionReceipt(
        execution_model_profile_id="test-profile",
        execution_model_profile_digest="1" * 64,
        canonical_model_id="test-model",
        adapter_model_id="openai/test-model",
        reasoning_mode="disabled",
        reasoning_effort=None,
        request_controls_attested=True,
        reasoning_replay_attested=True,
        reported_model_call_count=len(calls),
        model_call_count=len(calls),
        calls=calls,
        reasoning_tokens=0,
        telemetry_status="complete",
        status="verified",
    )


def test_source_cached_input_receipt_preserves_complete_and_partial_coverage() -> None:
    complete = cached_input_from_agent_receipt(_agent_receipt([3, 4]))
    partial = cached_input_from_agent_receipt(_agent_receipt([3, None]))

    assert complete.model_dump(mode="json") == {
        "source": "source_agent_receipt",
        "status": "complete",
        "tokens": 7,
        "provider_request_count": 2,
        "provider_response_count": 2,
        "responses_with_value": 2,
    }
    assert partial.model_dump(mode="json") == {
        "source": "source_agent_receipt",
        "status": "partial",
        "tokens": 3,
        "provider_request_count": 2,
        "provider_response_count": 2,
        "responses_with_value": 1,
    }
