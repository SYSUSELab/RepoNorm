"""Offline wire-level tests for the optional Norm4Code profile adapter."""

import asyncio
import hashlib
import json

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

from integration.runtime import AdapterError, Limits, ProfileRuntime
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from codewiki.src.be.llm_services import call_llm, create_fallback_models
from codewiki.src.config import Config


KEY = "test-secret-key-must-not-be-in-receipts"
PROMPT = "private prompt must not be persisted"
MODEL = "qwen3.8-flash"


def completion(*, usage=True, model=MODEL, message=None):
    payload = {
        "id": "chatcmpl-test", "created": 1, "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message":
            message or {"role": "assistant", "content": "done"}}],
    }
    if model is not None:
        payload["model"] = model
    if usage:
        payload["usage"] = {
            "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        }
    return payload


@pytest.fixture
def setup_runtime(tmp_path):
    runtimes = []

    def setup(handler, *, limits=None, nonthinking=False, profile_id=None):
        profile = load_execution_model_profile(
            profile_id or ("apiyi-qwen3.8-flash-nonthinking" if nonthinking else None)
        )
        transport = httpx.MockTransport(handler)
        runtime = ProfileRuntime(
            profile, KEY, tmp_path / f"receipts-{len(runtimes)}",
            limits or Limits(max_output_tokens=64),
            sync_transport=transport, async_transport=transport,
        )
        runtimes.append(runtime)
        config = Config.from_cli(
            repo_path=str(tmp_path / "repo"), output_dir=str(tmp_path / "docs"),
            llm_base_url=profile.provider.base_url, llm_api_key=KEY,
            main_model=profile.canonical_model_id, cluster_model=profile.canonical_model_id,
            fallback_model=profile.canonical_model_id,
            max_tokens=64, prompt_caching=False,
        )
        config.execution_profile_adapter = runtime
        return runtime, config

    yield setup
    for runtime in runtimes:
        asyncio.run(runtime.close())


def assert_wire(payload):
    assert payload["model"] == MODEL
    assert payload["temperature"] == 1
    assert payload["enable_thinking"] is True
    assert payload["preserve_thinking"] is True
    assert payload["reasoning_effort"] == "medium"
    assert payload.get("max_tokens", payload.get("max_completion_tokens")) == 64


def test_sync_calls_share_frozen_wire_and_private_receipts(setup_runtime):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    runtime, config = setup_runtime(handler)
    assert call_llm(PROMPT, config, config.cluster_model) == "done"
    assert call_llm(PROMPT, config, config.main_model) == "done"
    for payload in captured:
        assert_wire(payload)
    assert runtime.summary()["prompt_tokens"]["known_total"] == 20
    assert runtime.summary()["reserved_output_tokens"] == 128
    for path in runtime.receipt_dir.glob("*.json"):
        text = path.read_text()
        assert KEY not in text
        assert PROMPT not in text
        receipt = json.loads(text)
        assert receipt["effective_model"] == MODEL
        assert receipt["effective_version"] is None
        assert receipt["version_status"] == "unavailable"
        assert receipt["profile_digest"] == runtime.profile.profile_digest


def test_agent_tool_roundtrip_uses_single_model_and_frozen_wire(setup_runtime):
    captured = []
    tool_calls = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        if len(captured) == 1:
            reply = completion(message={
                "role": "assistant", "content": None,
                "reasoning_content": "fixture private reasoning before the tool",
                "tool_calls": [{"id": "tool-1", "type": "function", "function": {
                    "name": "read_fact", "arguments": "{}",
                }}],
            })
            reply["choices"][0]["finish_reason"] = "tool_calls"
        else:
            reply = completion()
        return httpx.Response(200, json=reply)

    runtime, config = setup_runtime(handler)
    model = create_fallback_models(config)
    assert not isinstance(model, FallbackModel)
    agent = Agent(model)

    @agent.tool_plain
    def read_fact() -> str:
        """Read an offline fixture fact."""
        tool_calls.append(True)
        return "fixture fact"

    assert asyncio.run(agent.run(PROMPT)).output == "done"
    assert tool_calls == [True]
    assert len(captured) == 2
    for payload in captured:
        assert_wire(payload)
    assert any(m["role"] == "tool" for m in captured[1]["messages"])
    assistant = next(m for m in captured[1]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "fixture private reasoning before the tool"
    replay = runtime.calls[1].assistant_reasoning_replay
    assert len(replay) == 1
    reasoning_bytes = assistant["reasoning_content"].encode()
    assert replay[0]["bytes"] == len(reasoning_bytes)
    assert replay[0]["sha256"] == hashlib.sha256(reasoning_bytes).hexdigest()
    assert "fixture private reasoning" not in (runtime.receipt_dir / "00002.json").read_text()
    assert runtime.summary()["requests"] == 2
    assert runtime.summary()["reserved_output_tokens"] == 128


@pytest.mark.parametrize("effective_model", [None, "another-model", KEY])
@pytest.mark.parametrize("agentic", [False, True])
def test_identity_failure_stops_later_requests(setup_runtime, effective_model, agentic):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=completion(model=effective_model))

    runtime, config = setup_runtime(handler)
    model = create_fallback_models(config) if agentic else None
    for _ in range(2):
        with pytest.raises(Exception):
            if agentic:
                asyncio.run(Agent(model).run(PROMPT))
            else:
                call_llm(PROMPT, config)
    assert len(captured) == 1
    assert runtime.failure == "effective_model_missing_or_mismatch"
    assert runtime.calls[0].effective_model is None
    assert runtime.calls[0].prompt_tokens is None
    assert KEY not in next(runtime.receipt_dir.glob("*.json")).read_text()


@pytest.mark.parametrize("source", ["tokens", "content"])
def test_nonthinking_rejects_reasoning(setup_runtime, source):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        payload = completion(usage=source == "tokens")
        if source == "content":
            payload["choices"][0]["message"]["reasoning_content"] = "reasoning"
        return httpx.Response(200, json=payload)

    runtime, config = setup_runtime(handler, nonthinking=True)
    with pytest.raises(Exception):
        call_llm(PROMPT, config)
    assert runtime.failure == "unexpected_reasoning"
    assert captured[0]["enable_thinking"] is False
    assert "reasoning_effort" not in captured[0]


@pytest.mark.parametrize("budget", ["requests", "output", "bytes"])
def test_total_budgets_account_for_multiple_requests(setup_runtime, budget):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=completion())

    limits = Limits(max_output_tokens=64, max_requests=2,
                    max_reserved_output_tokens=128)
    runtime, config = setup_runtime(handler, limits=limits)
    call_llm(PROMPT, config)
    if budget == "bytes":
        runtime.limits = limits.model_copy(update={
            "max_total_request_bytes": runtime.request_bytes * 2,
            "max_requests": 10, "max_reserved_output_tokens": 640,
        })
    elif budget == "output":
        runtime.limits = limits.model_copy(update={"max_requests": 10})
    else:
        runtime.limits = limits.model_copy(update={"max_reserved_output_tokens": 640})
    call_llm(PROMPT, config)
    with pytest.raises(Exception):
        call_llm(PROMPT, config)
    assert len(captured) == 2
    assert runtime.failure == "resource_budget_exhausted"
    assert runtime.reserved_output_tokens == 128


def test_generation_can_exceed_old_request_and_reserved_token_caps(setup_runtime):
    runtime, config = setup_runtime(
        lambda _: httpx.Response(200, json=completion()), limits=Limits(),
    )
    config.max_tokens = 32768
    for _ in range(201):
        call_llm(PROMPT, config)
    assert len(runtime.calls) == 201
    assert runtime.reserved_output_tokens == 201 * 32768
    assert runtime.failure is None


@pytest.mark.parametrize("agentic", [False, True])
def test_native_sdk_retries_transient_http_error(setup_runtime, agentic):
    captured = []
    def handler(request):
        captured.append(request)
        return httpx.Response(500 if len(captured) == 1 else 200,
                              json={"error": {"message": "temporary"}} if len(captured) == 1
                              else completion())
    runtime, config = setup_runtime(handler)
    if agentic:
        assert asyncio.run(Agent(create_fallback_models(config)).run(PROMPT)).output == "done"
    else:
        assert call_llm(PROMPT, config) == "done"
    assert [call.status for call in runtime.calls] == ["provider_http_error", "success"]
    assert runtime.failure is None
    assert runtime.summary()["prompt_tokens"]["status"] == "partial"


def test_native_cache_marker_repair_survives_http_rejection(setup_runtime):
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            return httpx.Response(400, json={"error": {"message": "unsupported marker"}})
        return httpx.Response(200, json=completion())
    runtime, config = setup_runtime(handler)
    config.prompt_caching = True
    assert asyncio.run(Agent(create_fallback_models(config)).run(PROMPT)).output == "done"
    assert "cache_control" in json.dumps(captured[0])
    assert "cache_control" not in json.dumps(captured[1])
    assert runtime.failure is None


def test_missing_usage_remains_partial_or_unavailable(setup_runtime):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion(usage=len(calls) == 1))

    runtime, config = setup_runtime(handler)
    call_llm(PROMPT, config)
    call_llm(PROMPT, config)
    assert runtime.calls[1].prompt_tokens is None
    summary = runtime.summary()
    assert summary["prompt_tokens"] == {
        "status": "partial", "known_total": 10, "responses_with_value": 1,
        "responses": 2, "requests": 2,
    }
    assert summary["reasoning_content_bytes"]["status"] == "unavailable"
    assert summary["reasoning_content_bytes"]["known_total"] is None


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_provider_failures_have_no_fake_usage_or_secret_receipts(setup_runtime, failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "transport":
            raise httpx.ConnectError(KEY + PROMPT, request=request)
        return httpx.Response(400, json={"error": {"message": KEY + PROMPT}})

    runtime, config = setup_runtime(handler)
    for _ in range(2):
        with pytest.raises(Exception):
            call_llm(PROMPT, config)
    expected_requests = 6 if failure == "transport" else 2
    assert len(requests) == expected_requests
    assert runtime.calls[0].prompt_tokens is None
    assert runtime.calls[0].completion_tokens is None
    summary = runtime.summary()
    assert summary["prompt_tokens"]["known_total"] is None
    assert summary["prompt_tokens"]["status"] == "unavailable"
    assert summary["responses"] == (2 if failure == "http" else 0)
    assert summary["http_200_responses"] == 0
    text = next(runtime.receipt_dir.glob("*.json")).read_text()
    assert KEY not in text
    assert PROMPT not in text


def test_conflicting_model_rejected_before_wire(setup_runtime):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=completion())

    runtime, config = setup_runtime(handler)
    with pytest.raises(Exception):
        call_llm(PROMPT, config, model="another-model")
    assert runtime.failure == "conflicting_wire_parameters"
    assert captured == []


def test_conflicting_roles_cannot_build_agent_fallback(setup_runtime):
    runtime, config = setup_runtime(lambda _: httpx.Response(200, json=completion()))
    config.fallback_model = "another-model"
    with pytest.raises(AdapterError, match="conflicting_model_roles"):
        create_fallback_models(config)


@pytest.mark.parametrize("profile_id,model,enabled", [
    (f"apiyi-deepseek-v4-flash-{version}-{mode}", f"deepseek-v4-flash-{version}",
     mode == "thinking-high")
    for version in ("260425", "ga-260731")
    for mode in ("nonthinking", "thinking-high")
] + [("deepseek-flash-nonthinking", "deepseek-flash", False)])
def test_deepseek_profile_sync_and_agent_tool_roundtrip(setup_runtime, profile_id, model, enabled):
    from integration.run import mock_response, probe
    captured = []
    def handler(request):
        expected_url = runtime.profile.provider.base_url + "/chat/completions"
        assert str(request.url) == expected_url
        assert request.headers["authorization"] == f"Bearer {KEY}"
        captured.append(json.loads(request.content))
        return mock_response(request)
    runtime, config = setup_runtime(handler, profile_id=profile_id)
    asyncio.run(probe(runtime, config))
    assert len(captured) == 3
    for payload, receipt in zip(captured, runtime.calls):
        assert payload["model"] == model
        assert payload["thinking"] == {"type": "enabled" if enabled else "disabled"}
        assert payload["temperature"] == runtime.profile.generation.temperature
        assert "enable_thinking" not in payload
        assert "preserve_thinking" not in payload
        if enabled:
            assert payload["reasoning_effort"] == "high"
        else:
            assert "reasoning_effort" not in payload
        assert receipt.profile_id == profile_id
        assert receipt.profile_digest == runtime.profile.profile_digest
        assert receipt.effective_model == model
    assert bool(runtime.calls[-1].assistant_reasoning_replay) == enabled
    if not enabled:
        assert runtime.summary()["reasoning_tokens"]["known_total"] == 0


@pytest.mark.parametrize("source", ["tokens", "content"])
@pytest.mark.parametrize("profile_id,model", [
    ("apiyi-deepseek-v4-flash-ga-260731-nonthinking", "deepseek-v4-flash-ga-260731"),
    ("deepseek-flash-nonthinking", "deepseek-flash"),
])
def test_deepseek_nonthinking_rejects_provider_reasoning(setup_runtime, source, profile_id, model):
    def handler(request):
        payload = completion(model=model, usage=source == "tokens")
        if source == "content":
            payload["choices"][0]["message"]["reasoning_content"] = "unexpected"
        return httpx.Response(200, json=payload)
    runtime, config = setup_runtime(handler, profile_id=profile_id)
    with pytest.raises(Exception):
        call_llm(PROMPT, config)
    assert runtime.failure == "unexpected_reasoning"


@pytest.mark.parametrize("profile_id", [
    "apiyi-deepseek-v4-flash-ga-260731-nonthinking", "deepseek-flash-nonthinking",
])
def test_deepseek_profile_cannot_fall_back_to_another_model(setup_runtime, profile_id):
    runtime, config = setup_runtime(
        lambda request: httpx.Response(200, json=completion(model="deepseek-v4-flash-260425")),
        profile_id=profile_id,
    )
    with pytest.raises(Exception):
        call_llm(PROMPT, config)
    assert runtime.failure == "effective_model_missing_or_mismatch"


@pytest.mark.parametrize("field,value", [
    ("enable_thinking", False), ("preserve_thinking", True),
    ("reasoning_effort", "high"), ("thinking", {"type": "enabled"}),
])
def test_deepseek_nonthinking_rejects_conflicting_wire_fields(setup_runtime, field, value):
    runtime, config = setup_runtime(
        lambda request: pytest.fail("conflict must not be sent"),
        profile_id="apiyi-deepseek-v4-flash-ga-260731-nonthinking",
    )
    with pytest.raises(Exception):
        runtime.create_client().chat.completions.create(
            model=config.main_model, messages=[{"role": "user", "content": PROMPT}],
            max_tokens=64, extra_body={field: value},
        )
    assert runtime.failure == "conflicting_wire_parameters"
    assert not runtime.calls
