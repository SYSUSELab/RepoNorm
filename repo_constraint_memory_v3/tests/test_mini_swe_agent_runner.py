from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from repo_constraint_memory_v3.agent_runtime import (
    SubprocessAgentRuntime, agent_model_binding, load_agent_profile,
)
from repo_constraint_memory_v3.integrations.agent_runners.mini_swe_agent import (
    FINALIZER_COMMAND,
    SUCCESS_SUBMISSION,
    RunnerFailure,
    _load_trajectory,
    _runner_options,
)
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from test_agent_runtime import _request


def _qwen_request(mode="thinking-medium"):
    profile = load_execution_model_profile(f"apiyi-qwen3.8-flash-{mode}")
    return _request().model_copy(update={
        "model": agent_model_binding(profile),
        "model_id": profile.adapter_model_ids.mini_swe_litellm,
        "execution_model_profile_id": profile.profile_id,
        "execution_model_profile_digest": profile.profile_digest,
    })


@pytest.mark.parametrize("mode", ["nonthinking", "thinking-high"])
def test_deepseek_mini_wire_is_preserved(mode):
    profile = load_execution_model_profile(f"apiyi-deepseek-v4-flash-260425-{mode}")
    request = _request().model_copy(update={"model": agent_model_binding(profile)})
    kwargs = _runner_options(request)["model_kwargs"]
    assert kwargs["extra_body"] == {
        "thinking": {"type": "disabled" if mode == "nonthinking" else "enabled"}
    }
    assert kwargs.get("reasoning_effort") == (None if mode == "nonthinking" else "high")
    assert kwargs["drop_params"] is False


@pytest.mark.parametrize("mode", ["nonthinking", "thinking-medium"])
def test_qwen_mini_wire_and_reasoning_replay_use_actual_http_transport(
    monkeypatch, tmp_path: Path, mode,
):
    # Optional external runner packages are exercised entirely offline.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    module = pytest.importorskip("minisweagent.models.litellm_model")
    request = _qwen_request(mode)
    thinking = mode == "thinking-medium"
    profile = load_agent_profile(
        Path(__file__).resolve().parents[1]
        / "configs/agents/mini_swe_agent_2_4_6.example.yaml",
        model_profile=load_execution_model_profile(request.execution_model_profile_id),
    )
    monkeypatch.setenv("APIYI_API_KEY", "offline-test-key")
    runtime_environment = SubprocessAgentRuntime(profile)._environment(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", runtime_environment["OPENAI_API_KEY"])
    bodies = []

    def handle(http_request):
        bodies.append(json.loads(http_request.content))
        return httpx.Response(200, json={
            "id": "offline-call", "object": "chat.completion", "created": 1,
            "model": "qwen3.8-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Inspect the source.",
                **({"reasoning_content": "private reasoning marker"} if thinking else {}),
                "tool_calls": [{"id": "tool-1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'
                }}],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 3 if thinking else 0}},
        })

    client = openai.OpenAI(
        base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    model = module.LitellmModel(
        model_name=request.model_id,
        model_kwargs=_runner_options(request)["model_kwargs"],
        cost_tracking="ignore_errors",
    )
    first = model.query([{"role": "user", "content": "Inspect repository"}], client=client)
    model.query([
        {"role": "user", "content": "Inspect repository"}, first,
        {"role": "tool", "tool_call_id": "tool-1", "content": "/repository"},
    ], client=client)
    assert len(bodies) == 2
    for body in bodies:
        assert body["model"] == "qwen3.8-flash"
        assert body["enable_thinking"] is thinking
        if thinking:
            assert body["reasoning_effort"] == "medium"
            assert body["preserve_thinking"] is True
        else:
            assert "reasoning_effort" not in body
            assert "preserve_thinking" not in body
        assert "thinking" not in body
    if thinking:
        assert bodies[1]["messages"][1]["reasoning_content"] == "private reasoning marker"
    else:
        assert not bodies[1]["messages"][1].get("reasoning_content")
    assert "extra" not in bodies[1]["messages"][1]
    assert _runner_options(request)["model_kwargs"]["drop_params"] is False


def test_qwen_mini_receipt_attests_controls_and_removes_private_reasoning(tmp_path: Path):
    request = _qwen_request()
    model = {"model_name": request.model_id,
             "model_kwargs": _runner_options(request)["model_kwargs"]}
    environment = {"wrapper_args": ["--test-wrapper"], "cwd": "/test-cwd",
                   "executable": "/test-bwrap", "env": {"PAGER": "cat"}}
    trajectory = {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "mini_version": request.framework_version, "exit_status": "Submitted",
            "submission": SUCCESS_SUBMISSION,
            "model_stats": {"api_calls": 1, "instance_cost": 0},
            "config": {"model": deepcopy(model),
                       "agent_type": "minisweagent.agents.default.DefaultAgent",
                       "environment_type": "minisweagent.environments.extra.bubblewrap.BubblewrapEnvironment",
                       "environment": environment},
        },
        "messages": [{
            "role": "assistant", "reasoning_content": "private reasoning marker",
            "reasoning": "private reasoning marker",
            "reasoning_details": [{"text": "private reasoning marker"}],
            "extra": {"actions": [{"command": FINALIZER_COMMAND}], "response": {
                "model": "qwen3.8-flash",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "reasoning_content": "private reasoning marker"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "completion_tokens_details": {"reasoning_tokens": 3}},
            }},
        }],
    }
    path = tmp_path / "trajectory.json"

    def load():
        path.write_text(json.dumps(trajectory), encoding="utf-8")
        return _load_trajectory(
            path, request, expected_wrapper_args=environment["wrapper_args"],
            expected_cwd=environment["cwd"], expected_executable=environment["executable"],
            expected_environment=environment["env"], expected_model_config=model,
        )

    sanitized, _, _, receipt = load()
    assert receipt.status == "verified"
    assert receipt.reasoning_effort == "medium"
    assert receipt.request_controls_attested and receipt.reasoning_replay_attested
    assert "private reasoning marker" not in json.dumps(sanitized)
    for control in ("enable_thinking", "reasoning_effort", "preserve_thinking"):
        trajectory["info"]["config"]["model"] = deepcopy(model)
        del trajectory["info"]["config"]["model"]["model_kwargs"]["extra_body"][control]
        with pytest.raises(RunnerFailure, match="model does not match"):
            load()
    trajectory["info"]["config"]["model"] = deepcopy(model)
    trajectory["messages"][0].pop("reasoning_content")
    _, _, _, receipt = load()
    assert receipt.status == "violated"
    assert "provider_reasoning_replay_mismatch" in receipt.violations


def test_official_deepseek_mini_wire():
    profile = load_execution_model_profile("deepseek-flash-nonthinking")
    request = _request().model_copy(update={"model": agent_model_binding(profile)})
    kwargs = _runner_options(request)["model_kwargs"]
    assert kwargs == {
        "api_base": "https://api.deepseek.com", "drop_params": False,
        "temperature": 0, "extra_body": {"thinking": {"type": "disabled"}},
    }
    for provider, wire, model in [
        ("unsupported", "deepseek_thinking_type", "deepseek/deepseek-flash"),
        ("deepseek", "qwen_enable_thinking", "openai/qwen3.8-flash"),
        ("deepseek", "deepseek_thinking_type", "openai/deepseek-flash"),
    ]:
        unsupported = request.model_copy(update={"model": request.model.model_copy(update={
            "provider_id": provider, "reasoning_wire_policy": wire, "model_id": model,
        })})
        with pytest.raises(RunnerFailure):
            _runner_options(unsupported)


def test_official_deepseek_mini_actual_http_transport(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "official-test-key")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    module = pytest.importorskip("minisweagent.models.litellm_model")
    profile = load_execution_model_profile("deepseek-flash-nonthinking")
    request = _request().model_copy(update={"model": agent_model_binding(profile)})
    bodies = []

    def handle(http_request):
        assert str(http_request.url) == "https://api.deepseek.com/chat/completions"
        assert http_request.headers["authorization"] == "Bearer official-test-key"
        bodies.append(json.loads(http_request.content))
        return httpx.Response(200, json={
            "id": "offline-call", "object": "chat.completion", "created": 1,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Inspect source",
                "tool_calls": [{"id": "tool-1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'
                }}],
            }}], "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    client = HTTPHandler(client=http_client)
    model = module.LitellmModel(
        model_name=profile.adapter_model_ids.mini_swe_litellm,
        model_kwargs=_runner_options(request)["model_kwargs"],
        cost_tracking="ignore_errors",
    )
    try:
        model.query([{"role": "user", "content": "Inspect source"}], client=client)
    finally:
        http_client.close()
    body, = bodies
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert "enable_thinking" not in body


def test_openrouter_mini_actual_http_transport(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "official-test-key")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    module = pytest.importorskip("minisweagent.models.litellm_model")
    profile = load_execution_model_profile("openrouter-deepseek-v4-flash-0423-nonthinking")
    request = _request().model_copy(update={"model": agent_model_binding(profile)})
    bodies = []

    def handle(http_request):
        assert str(http_request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert http_request.headers["authorization"] == "Bearer official-test-key"
        bodies.append(json.loads(http_request.content))
        return httpx.Response(200, json={
            "id": "offline-call", "object": "chat.completion", "created": 1,
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Inspect source",
                "tool_calls": [{"id": "tool-1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'
                }}],
            }}], "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    client = HTTPHandler(client=http_client)
    model = module.LitellmModel(
        model_name=profile.adapter_model_ids.mini_swe_litellm,
        model_kwargs=_runner_options(request)["model_kwargs"],
        cost_tracking="ignore_errors",
    )
    try:
        model.query([{"role": "user", "content": "Inspect source"}], client=client)
    finally:
        http_client.close()
    body, = bodies
    assert body["model"] == "deepseek/deepseek-v4-flash"
    assert body["reasoning"] == {"enabled": False}
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "enable_thinking" not in body


def test_openrouter_thinking_high_mini_runner_options():
    profile = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-thinking-high"
    )
    request = _request().model_copy(update={"model": agent_model_binding(profile)})

    options = _runner_options(request)

    assert options["model_kwargs"]["extra_body"] == {
        "reasoning": {"effort": "high"}
    }
    assert "reasoning_effort" not in options["model_kwargs"]


@pytest.mark.parametrize(
    ("profile_id", "expected_reasoning"),
    (
        (
            "openrouter-deepseek-v4-flash-0423-gmicloud-fp8-thinking-high",
            {"effort": "high"},
        ),
        (
            "openrouter-qwen3.5-flash-02-23-thinking-enabled",
            {"enabled": True},
        ),
        (
            "openrouter-qwen3.5-flash-02-23-thinking-8192",
            {"max_tokens": 8192},
        ),
        (
            "openrouter-gemini-2.5-flash-lite-thinking-8192",
            {"max_tokens": 8192},
        ),
        (
            "openrouter-glm-5.3-flash-thinking-high",
            {"effort": "high"},
        ),
    ),
)
def test_openrouter_default_and_budget_reasoning_mini_runner_options(
    profile_id: str,
    expected_reasoning: dict[str, object],
):
    profile = load_execution_model_profile(profile_id)
    request = _request().model_copy(update={"model": agent_model_binding(profile)})

    options = _runner_options(request)

    assert options["model_kwargs"]["extra_body"] == {
        "reasoning": expected_reasoning
    }
    assert "reasoning_effort" not in options["model_kwargs"]


@pytest.mark.parametrize("field,value", [("reasoning_details", [{"type": "reasoning.text", "text": "hidden"}]), ("reasoning", "hidden")])
def test_openrouter_mini_rejects_reasoning_extensions(field, value):
    from repo_constraint_memory_v3.integrations.agent_runners.mini_swe_agent import _response_call_receipts
    profile = load_execution_model_profile("openrouter-deepseek-v4-flash-0423-nonthinking")
    request = _request().model_copy(update={"model": agent_model_binding(profile)})
    trajectory = {"messages": [{"role": "assistant", "extra": {"response": {
        "model": profile.canonical_model_id,
        "choices": [{"message": {field: value}}],
        "usage": {},
    }}}]}
    _, violations, _ = _response_call_receipts(trajectory, request)
    assert "disabled_reasoning_content_nonempty" in violations
