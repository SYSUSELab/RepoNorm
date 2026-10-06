from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from repo_constraint_memory_v3.model_profiles import (
    DEFAULT_EXECUTION_MODEL_PROFILE_ID,
    MAX_EXECUTION_MODEL_PROFILE_BYTES,
    MAX_EXECUTION_MODEL_PROFILE_DEPTH,
    MAX_EXECUTION_MODEL_PROFILE_NODES,
    OFFLINE_EXECUTION_MODEL_PROFILE_ID,
    ExecutionModelProfile,
    builtin_profile_ids,
    load_execution_model_profile,
    load_profile_artifact,
    stable_profile_text,
)


PUBLIC_PROFILE_IDS = (
    "apiyi-deepseek-v4-flash-260425-nonthinking",
    "apiyi-deepseek-v4-flash-260425-thinking-high",
    "apiyi-deepseek-v4-flash-ga-260731-nonthinking",
    "apiyi-deepseek-v4-flash-ga-260731-thinking-high",
    "apiyi-glm-5.3-flash-thinking-high",
    "apiyi-qwen3.8-flash-nonthinking",
    "apiyi-qwen3.8-flash-thinking-medium",
    "deepseek-flash-nonthinking",
    "openrouter-deepseek-v4-flash-0423-gmicloud-fp8-thinking-high",
    "openrouter-deepseek-v4-flash-0423-nonthinking",
    "openrouter-deepseek-v4-flash-0423-thinking-high",
    "openrouter-gemini-2.5-flash-lite-thinking-8192",
    "openrouter-glm-5.3-flash-thinking-high",
    "openrouter-qwen3.5-flash-02-23-thinking-8192",
    "openrouter-qwen3.5-flash-02-23-thinking-enabled",
)


def _default_payload() -> dict[str, object]:
    return load_execution_model_profile().model_dump(mode="python")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_default_profile_is_qwen38_flash_medium() -> None:
    profile = load_execution_model_profile()

    assert DEFAULT_EXECUTION_MODEL_PROFILE_ID == "apiyi-qwen3.8-flash-thinking-medium"
    assert profile.profile_id == DEFAULT_EXECUTION_MODEL_PROFILE_ID
    assert profile.profile_kind == "real"
    assert profile.canonical_model_id == "qwen3.8-flash"
    assert profile.provider.model_dump() == {
        "provider_id": "apiyi",
        "protocol": "openai_chat_completions",
        "base_url": "https://vip.apiyi.com/v1",
        "api_key_env": "APIYI_API_KEY",
    }
    assert profile.adapter_model_ids.model_dump() == {
        "core_openai_chat": "qwen3.8-flash",
        "mini_swe_litellm": "openai/qwen3.8-flash",
        "downstream_codex": "qwen3.8-flash",
    }
    assert profile.generation.temperature == 1
    assert profile.reasoning.mode == "enabled"
    assert profile.reasoning.effort == "medium"
    assert profile.reasoning.wire_policy == "qwen_enable_thinking"
    assert profile.effective_identity.require_response_model is True
    assert profile.effective_identity.accepted_model_ids == [
        "qwen3.8-flash"
    ]
    assert re.fullmatch(r"[0-9a-f]{64}", profile.profile_digest)

    with pytest.raises(ValidationError, match="frozen"):
        profile.profile_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("profile_id", "canonical_model", "reasoning_mode", "reasoning_effort"),
    (
        (
            "apiyi-deepseek-v4-flash-260425-nonthinking",
            "deepseek-v4-flash-260425",
            "disabled",
            None,
        ),
        (
            "apiyi-deepseek-v4-flash-260425-thinking-high",
            "deepseek-v4-flash-260425",
            "enabled",
            "high",
        ),
        (
            "apiyi-deepseek-v4-flash-ga-260731-nonthinking",
            "deepseek-v4-flash-ga-260731",
            "disabled",
            None,
        ),
        (
            "apiyi-deepseek-v4-flash-ga-260731-thinking-high",
            "deepseek-v4-flash-ga-260731",
            "enabled",
            "high",
        ),
    ),
)
def test_all_deepseek_profiles_bind_one_model_and_explicit_reasoning(
    profile_id: str,
    canonical_model: str,
    reasoning_mode: str,
    reasoning_effort: str | None,
) -> None:
    profile = load_execution_model_profile(profile_id)

    assert profile.profile_id == profile_id
    assert profile.canonical_model_id == canonical_model
    assert profile.provider.provider_id == "apiyi"
    assert profile.provider.protocol == "openai_chat_completions"
    assert profile.provider.base_url == "https://vip.apiyi.com/v1"
    assert profile.provider.api_key_env == "APIYI_API_KEY"
    assert profile.adapter_model_ids.core_openai_chat == canonical_model
    assert profile.adapter_model_ids.mini_swe_litellm == f"deepseek/{canonical_model}"
    assert profile.adapter_model_ids.downstream_codex == canonical_model
    assert profile.generation.temperature == 0
    assert profile.reasoning.mode == reasoning_mode
    assert profile.reasoning.effort == reasoning_effort
    assert profile.reasoning.wire_policy == "deepseek_thinking_type"
    assert profile.effective_identity.require_response_model is True
    assert profile.effective_identity.accepted_model_ids == [canonical_model]


def test_public_lookup_is_exact_and_digests_separate_all_profiles() -> None:
    assert builtin_profile_ids() == PUBLIC_PROFILE_IDS
    assert builtin_profile_ids(include_offline=True) == tuple(sorted((
        *PUBLIC_PROFILE_IDS,
        OFFLINE_EXECUTION_MODEL_PROFILE_ID,
    )))
    assert len(
        {
            load_execution_model_profile(profile_id).profile_digest
            for profile_id in PUBLIC_PROFILE_IDS
        }
    ) == len(PUBLIC_PROFILE_IDS)


def test_offline_profile_is_hidden_gated_and_provider_free() -> None:
    assert OFFLINE_EXECUTION_MODEL_PROFILE_ID not in builtin_profile_ids()
    with pytest.raises(ValueError, match="test-only"):
        load_execution_model_profile(OFFLINE_EXECUTION_MODEL_PROFILE_ID)

    profile = load_execution_model_profile(
        OFFLINE_EXECUTION_MODEL_PROFILE_ID,
        allow_offline=True,
    )
    assert profile.profile_kind == "offline_test"
    assert profile.canonical_model_id == "deterministic-fake"
    assert profile.provider.provider_id == "offline"
    assert profile.provider.protocol == "offline_mock"
    assert profile.provider.base_url is None
    assert profile.provider.api_key_env is None
    assert set(profile.adapter_model_ids.model_dump().values()) == {
        "deterministic-fake"
    }
    assert profile.reasoning.model_dump() == {
        "mode": "disabled",
        "effort": None,
        "wire_policy": "offline_mock",
    }
    assert profile.effective_identity.accepted_model_ids == [
        "deterministic-fake"
    ]


def test_digest_excludes_only_api_key_environment_name() -> None:
    profile = load_execution_model_profile()

    renamed_credential = profile.model_copy(
        update={
            "provider": profile.provider.model_copy(
                update={"api_key_env": "ALTERNATE_API_KEY"}
            )
        }
    )
    assert renamed_credential.profile_digest == profile.profile_digest
    assert stable_profile_text(renamed_credential) != stable_profile_text(profile)
    assert "api_key_env" not in profile.method_payload()["provider"]

    changed_provider = profile.model_copy(
        update={
            "provider": profile.provider.model_copy(
                update={"provider_id": "apiyi-alternate"}
            )
        }
    )
    changed_endpoint = profile.model_copy(
        update={
            "provider": profile.provider.model_copy(
                update={"base_url": "https://alternate.apiyi.example/v1"}
            )
        }
    )
    changed_adapter = profile.model_copy(
        update={
            "adapter_model_ids": profile.adapter_model_ids.model_copy(
                update={"mini_swe_litellm": "deepseek/alternate-wire-id"}
            )
        }
    )
    changed_generation = profile.model_copy(
        update={
            "generation": profile.generation.model_copy(update={"temperature": 0.25})
        }
    )
    changed_reasoning = profile.model_copy(
        update={
            "reasoning": profile.reasoning.model_copy(
                update={"mode": "enabled", "effort": "high"}
            )
        }
    )
    changed_allowlist = profile.model_copy(
        update={
            "effective_identity": profile.effective_identity.model_copy(
                update={
                    "accepted_model_ids": [
                        profile.canonical_model_id,
                        "deepseek-v4-flash-260425-provider-revision",
                    ]
                }
            )
        }
    )
    changed_profile_id = profile.model_copy(update={"profile_id": "replacement"})

    assert {
        changed_provider.profile_digest,
        changed_endpoint.profile_digest,
        changed_adapter.profile_digest,
        changed_generation.profile_digest,
        changed_reasoning.profile_digest,
        changed_allowlist.profile_digest,
        changed_profile_id.profile_digest,
    }.isdisjoint({profile.profile_digest})


def test_profile_model_is_strict_and_rejects_weakened_identity() -> None:
    payload = _default_payload()
    payload["generation"]["temperature"] = "0"  # type: ignore[index]
    with pytest.raises(ValidationError):
        ExecutionModelProfile.model_validate(payload)

    payload = _default_payload()
    payload["effective_identity"]["require_response_model"] = False  # type: ignore[index]
    with pytest.raises(ValidationError):
        ExecutionModelProfile.model_validate(payload)

    payload = _default_payload()
    payload["effective_identity"]["accepted_model_ids"] = ["some-other-model"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="canonical model ID"):
        ExecutionModelProfile.model_validate(payload)

    payload = _default_payload()
    payload["unexpected"] = "field"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExecutionModelProfile.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "effort"),
    (("disabled", "high"), ("enabled", None)),
)
def test_reasoning_mode_and_effort_cannot_be_inconsistent(
    mode: str,
    effort: str | None,
) -> None:
    payload = _default_payload()
    payload["reasoning"]["mode"] = mode  # type: ignore[index]
    payload["reasoning"]["effort"] = effort  # type: ignore[index]
    with pytest.raises(ValidationError):
        ExecutionModelProfile.model_validate(payload)


@pytest.mark.parametrize(
    "effort",
    ("none", "low", "medium", "high", "xhigh", "max"),
)
def test_enabled_reasoning_accepts_bounded_future_efforts(effort: str) -> None:
    payload = load_execution_model_profile(PUBLIC_PROFILE_IDS[0]).model_dump(mode="python")
    payload["reasoning"]["mode"] = "enabled"  # type: ignore[index]
    payload["reasoning"]["effort"] = effort  # type: ignore[index]

    profile = ExecutionModelProfile.model_validate(payload)

    assert profile.reasoning.effort == effort


def test_future_gpt_responses_profile_is_schema_only_adapter_extension() -> None:
    payload = _default_payload()
    payload["profile_id"] = "apiyi-gpt-5.6-luna-thinking-high"
    payload["canonical_model_id"] = "gpt-5.6-luna"
    payload["provider"]["protocol"] = "openai_responses"  # type: ignore[index]
    payload["adapter_model_ids"] = {
        "core_openai_chat": "gpt-5.6-luna",
        "mini_swe_litellm": "openai/gpt-5.6-luna",
        "downstream_codex": "gpt-5.6-luna",
    }
    payload["reasoning"] = {
        "mode": "enabled",
        "effort": "high",
        "wire_policy": "openai_responses_reasoning_effort",
    }
    payload["effective_identity"]["accepted_model_ids"] = [  # type: ignore[index]
        "gpt-5.6-luna"
    ]

    profile = ExecutionModelProfile.model_validate(payload)

    assert profile.provider.protocol == "openai_responses"
    assert profile.reasoning.wire_policy == "openai_responses_reasoning_effort"
    assert profile.adapter_model_ids.downstream_codex == "gpt-5.6-luna"


@pytest.mark.parametrize(
    "base_url",
    (
        "http://vip.apiyi.com/v1",
        "https://vip.apiyi.com/v1/",
        "https://user:secret@vip.apiyi.com/v1",
        "https://vip.apiyi.com/v1?model=floating",
    ),
)
def test_real_endpoint_must_be_canonical_credential_free_https(
    base_url: str,
) -> None:
    payload = _default_payload()
    payload["provider"]["base_url"] = base_url  # type: ignore[index]
    with pytest.raises(ValidationError):
        ExecutionModelProfile.model_validate(payload)


def test_external_profile_rejects_duplicate_keys(tmp_path: Path) -> None:
    source = yaml.safe_dump(_default_payload(), sort_keys=False)
    source = source.replace(
        "  provider_id: apiyi\n",
        "  provider_id: apiyi\n  provider_id: shadow-provider\n",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="bounded YAML"):
        load_execution_model_profile(path)


@pytest.mark.parametrize(
    "source",
    (
        "first: &shared value\nsecond: *shared\n",
        "base: &base\n  value: one\nmerged:\n  <<: *base\n",
    ),
)
def test_external_profile_rejects_aliases_anchors_and_merges(
    tmp_path: Path,
    source: str,
) -> None:
    path = tmp_path / "alias.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="aliases and anchors"):
        load_execution_model_profile(path)


def test_external_profile_rejects_excessive_yaml_depth(tmp_path: Path) -> None:
    lines = [f"{'  ' * depth}level_{depth}:" for depth in range(
        MAX_EXECUTION_MODEL_PROFILE_DEPTH + 1
    )]
    lines.append(
        f"{'  ' * (MAX_EXECUTION_MODEL_PROFILE_DEPTH + 1)}value"
    )
    path = tmp_path / "deep.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="depth limit"):
        load_execution_model_profile(path)


def test_external_profile_rejects_excessive_yaml_nodes(tmp_path: Path) -> None:
    source = "items:\n" + "".join(
        f"  - item-{index}\n"
        for index in range(MAX_EXECUTION_MODEL_PROFILE_NODES)
    )
    path = tmp_path / "wide.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="node limit"):
        load_execution_model_profile(path)


def test_external_profile_rejects_excessive_bytes_and_symlink(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b"x" * (MAX_EXECUTION_MODEL_PROFILE_BYTES + 1))
    with pytest.raises(ValueError, match="byte limit"):
        load_execution_model_profile(oversized)

    target = tmp_path / "target.yaml"
    _write_yaml(target, _default_payload())
    link = tmp_path / "link.yaml"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        load_execution_model_profile(link)


def test_saved_profile_artifact_requires_exact_canonical_json(
    tmp_path: Path,
) -> None:
    profile = load_execution_model_profile()
    artifact = tmp_path / "execution-model-profile.json"
    artifact.write_text(stable_profile_text(profile), encoding="utf-8")
    assert load_profile_artifact(artifact) == profile

    artifact.write_text(
        json.dumps(profile.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        load_profile_artifact(artifact)


def test_saved_profile_artifact_rejects_duplicate_keys(tmp_path: Path) -> None:
    profile = load_execution_model_profile()
    duplicate = stable_profile_text(profile).replace(
        "{\n",
        f'{{\n  "profile_id": "{profile.profile_id}",\n',
        1,
    )
    artifact = tmp_path / "duplicate.json"
    artifact.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_profile_artifact(artifact)


@pytest.mark.parametrize("effort", ["none", "high", "max"])
def test_qwen_rejects_remapped_or_unsupported_effort(effort: str) -> None:
    payload = _default_payload()
    payload["reasoning"]["effort"] = effort
    with pytest.raises(ValidationError, match="Qwen thinking requires"):
        ExecutionModelProfile.model_validate(payload)


def test_qwen_nonthinking_changes_only_profile_identity_and_reasoning() -> None:
    thinking = load_execution_model_profile()
    nonthinking = load_execution_model_profile("apiyi-qwen3.8-flash-nonthinking")
    expected = thinking.model_dump(mode="json")
    expected["profile_id"] = nonthinking.profile_id
    expected["reasoning"]["mode"] = "disabled"
    expected["reasoning"]["effort"] = None
    assert nonthinking.model_dump(mode="json") == expected
    assert nonthinking.profile_digest != thinking.profile_digest
    assert nonthinking.generation.temperature == 1


def test_official_deepseek_flash_profile() -> None:
    profile = load_execution_model_profile("deepseek-flash-nonthinking")
    assert profile.provider.model_dump() == {
        "provider_id": "deepseek", "protocol": "openai_chat_completions",
        "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY",
    }
    assert profile.canonical_model_id == "deepseek-flash"
    assert profile.adapter_model_ids.core_openai_chat == "deepseek-flash"
    assert profile.adapter_model_ids.mini_swe_litellm == "deepseek/deepseek-flash"
    assert profile.effective_identity.accepted_model_ids == ["deepseek-flash"]
    assert profile.reasoning.mode == "disabled"
    assert profile.reasoning.effort is None
    assert profile.reasoning.wire_policy == "deepseek_thinking_type"
    assert profile.generation.temperature == 0


def test_official_deepseek_core_http_transport():
    import httpx
    from repo_constraint_memory_v3.builder import load_build_config
    from repo_constraint_memory_v3.llm import OpenAICompatibleClient

    profile = load_execution_model_profile("deepseek-flash-nonthinking")
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs/hardened.yaml",
        model_profile=profile,
    )
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps({
            "model": "deepseek-flash",
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }).encode()))
    client = OpenAICompatibleClient(settings=config.llm, api_key="official-test-key", model_profile=profile)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        response = client.complete("Return JSON", "Return ok")
    finally:
        client._client.close()
    assert response.model == "deepseek-flash"
    request, = captured
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer official-test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert "enable_thinking" not in payload


@pytest.mark.parametrize("reasoning_field", [None, "reasoning", "reasoning_details"])
def test_openrouter_core_http_transport(reasoning_field):
    import httpx
    from repo_constraint_memory_v3.builder import load_build_config
    from repo_constraint_memory_v3.llm import OpenAICompatibleClient

    profile = load_execution_model_profile("openrouter-deepseek-v4-flash-0423-nonthinking")
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs/hardened.yaml",
        model_profile=profile,
    )
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps({
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"message": {"content": '{"ok": true}', **({reasoning_field: [{"text": "hidden"}]} if reasoning_field else {})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }).encode()))
    client = OpenAICompatibleClient(settings=config.llm, api_key="official-test-key", model_profile=profile)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        response = client.complete("Return JSON", "Return ok")
    finally:
        client._client.close()
    assert response.model == "deepseek/deepseek-v4-flash"
    from repo_constraint_memory_v3.inference import _disabled_thinking_profile_violations
    violations = _disabled_thinking_profile_violations(
        reasoning_tokens=response.reasoning_tokens, reasoning_content_bytes=response.reasoning_content_bytes,
    )
    assert bool(violations) is (reasoning_field is not None)
    request, = captured
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer official-test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "deepseek/deepseek-v4-flash"
    assert payload["reasoning"] == {"enabled": False}
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert "enable_thinking" not in payload


def test_openrouter_thinking_high_profile_changes_only_reasoning_identity():
    nonthinking = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-nonthinking"
    )
    thinking = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-thinking-high"
    )
    expected = nonthinking.model_dump(mode="json")
    expected["profile_id"] = thinking.profile_id
    expected["reasoning"].update(mode="enabled", effort="high")

    assert thinking.model_dump(mode="json") == expected
    assert thinking.profile_digest != nonthinking.profile_digest
    assert thinking.provider.api_key_env == "OPENROUTER_API_KEY"


def test_openrouter_gmicloud_profile_has_an_independent_identity():
    original = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-thinking-high"
    )
    gmicloud = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-gmicloud-fp8-thinking-high"
    )
    expected = original.model_dump(mode="json")
    expected["profile_id"] = gmicloud.profile_id

    assert gmicloud.model_dump(mode="json") == expected
    assert gmicloud.profile_digest != original.profile_digest


def test_openrouter_thinking_high_core_http_transport():
    import httpx
    from repo_constraint_memory_v3.builder import load_build_config
    from repo_constraint_memory_v3.llm import OpenAICompatibleClient

    profile = load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-thinking-high"
    )
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs/hardened.yaml",
        model_profile=profile,
    )
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps({
            "model": "deepseek/deepseek-v4-flash",
            "choices": [{"message": {
                "content": '{"ok": true}',
                "reasoning": "hidden reasoning",
            }, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }).encode()))

    client = OpenAICompatibleClient(
        settings=config.llm,
        api_key="official-test-key",
        model_profile=profile,
    )
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        response = client.complete("Return JSON", "Return ok")
    finally:
        client._client.close()

    assert response.model == "deepseek/deepseek-v4-flash"
    assert response.reasoning_tokens == 2
    assert response.reasoning_content_bytes > 0
    request, = captured
    payload = json.loads(request.content)
    assert payload["reasoning"] == {"effort": "high"}
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize(
    ("profile_id", "model_id", "expected_reasoning"),
    (
        (
            "openrouter-qwen3.5-flash-02-23-thinking-enabled",
            "qwen/qwen3.5-flash-02-23",
            {"enabled": True},
        ),
        (
            "openrouter-qwen3.5-flash-02-23-thinking-8192",
            "qwen/qwen3.5-flash-02-23",
            {"max_tokens": 8192},
        ),
        (
            "openrouter-gemini-2.5-flash-lite-thinking-8192",
            "google/gemini-2.5-flash-lite",
            {"max_tokens": 8192},
        ),
        (
            "openrouter-glm-5.3-flash-thinking-high",
            "z-ai/glm-5.3-flash",
            {"effort": "high"},
        ),
    ),
)
def test_openrouter_default_and_budget_reasoning_core_http_transport(
    profile_id: str,
    model_id: str,
    expected_reasoning: dict[str, object],
):
    import httpx
    from repo_constraint_memory_v3.builder import load_build_config
    from repo_constraint_memory_v3.llm import OpenAICompatibleClient

    profile = load_execution_model_profile(profile_id)
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs/hardened.yaml",
        model_profile=profile,
    )
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps({
            "model": model_id,
            "choices": [{"message": {
                "content": '{"ok": true}',
                "reasoning": "hidden reasoning",
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7},
        }).encode()))

    client = OpenAICompatibleClient(
        settings=config.llm,
        api_key="official-test-key",
        model_profile=profile,
    )
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        response = client.complete("Return JSON", "Return ok")
    finally:
        client._client.close()

    assert response.model == model_id
    request, = captured
    payload = json.loads(request.content)
    assert payload["model"] == model_id
    assert payload["reasoning"] == expected_reasoning
    assert "reasoning_effort" not in payload
    assert "thinking" not in payload


def test_reasoning_max_tokens_is_openrouter_only_and_excludes_effort() -> None:
    payload = _default_payload()
    payload["reasoning"]["effort"] = None
    payload["reasoning"]["max_tokens"] = 8192
    with pytest.raises(ValidationError, match="requires the OpenRouter wire"):
        ExecutionModelProfile.model_validate(payload)

    payload = load_execution_model_profile(
        "openrouter-gemini-2.5-flash-lite-thinking-8192"
    ).model_dump(mode="python")
    payload["reasoning"]["effort"] = "medium"
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ExecutionModelProfile.model_validate(payload)


def test_optional_reasoning_budget_does_not_relabel_existing_profile_digests() -> None:
    assert load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-nonthinking"
    ).profile_digest == "b4cc4737aef3d7732a4628bd853f2252d38549eafebef75a1fa252a3dfc85c85"
    assert load_execution_model_profile(
        "openrouter-deepseek-v4-flash-0423-thinking-high"
    ).profile_digest == "5d1eed78e45b0121d77b8083350fde29ead8a0035091db9eb5472cd3d73dd7bb"


@pytest.mark.parametrize(
    ("profile_id", "expected_max_tokens"),
    (
        ("openrouter-qwen3.5-flash-02-23-thinking-enabled", None),
        ("openrouter-qwen3.5-flash-02-23-thinking-8192", 8192),
        ("openrouter-gemini-2.5-flash-lite-thinking-8192", 8192),
        ("openrouter-glm-5.3-flash-thinking-high", None),
    ),
)
def test_run_summary_records_only_configured_reasoning_budget(
    profile_id: str,
    expected_max_tokens: int | None,
) -> None:
    from repo_constraint_memory_v3.builder import load_build_config
    from repo_constraint_memory_v3.exporters import execution_model_run_summary

    profile = load_execution_model_profile(profile_id)
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs/hardened.yaml",
        model_profile=profile,
    )
    receipt = SimpleNamespace(
        calls=[],
        status="verified",
        adapter_model_id=profile.adapter_model_ids.mini_swe_litellm,
        model_call_count=0,
        reasoning_tokens=0,
        reasoning_content_responses=0,
        reasoning_content_bytes=0,
        request_controls_attested=True,
        reasoning_replay_attested=True,
        telemetry_status="not_applicable",
        violations=[],
    )
    source_acquisition = SimpleNamespace(
        model_profile=profile,
        model_execution=receipt,
    )

    summary = execution_model_run_summary(config, source_acquisition, [])

    assert summary["reasoning"].get("max_tokens") == expected_max_tokens
    assert ("max_tokens" in summary["reasoning"]) is (
        expected_max_tokens is not None
    )
