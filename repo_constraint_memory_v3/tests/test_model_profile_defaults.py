from __future__ import annotations

from pathlib import Path

import yaml

from repo_constraint_memory_v3.agent_runtime import load_agent_profile
from repo_constraint_memory_v3.builder import load_build_config
from repo_constraint_memory_v3.model_profiles import (
    DEFAULT_EXECUTION_MODEL_PROFILE_ID,
    load_execution_model_profile,
)
from repo_constraint_memory_v3.models import (
    LLMSettings,
    ObservationSettings,
    SemanticSettings,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIGS = (
    PACKAGE_ROOT / "configs" / "hardened.yaml",
    PACKAGE_ROOT / "configs" / "openai_agents_python.yaml",
)
AGENT_RUNTIME_PROFILE = (
    PACKAGE_ROOT
    / "configs"
    / "agents"
    / "mini_swe_agent_2_4_6.example.yaml"
)


def test_semantic_classification_batch_default_and_active_profiles_are_16() -> None:
    profile = load_execution_model_profile()
    assert SemanticSettings().max_classification_members_per_request == 16
    assert {
        load_build_config(path, model_profile=profile)
        .semantic.max_classification_members_per_request
        for path in ACTIVE_CONFIGS
    } == {16}


def test_hardened_full_run_keeps_the_frozen_generation_boundaries() -> None:
    profile = load_execution_model_profile()
    config = load_build_config(ACTIVE_CONFIGS[0], model_profile=profile)

    assert config.llm.timeout_seconds == 600
    assert config.llm.max_output_tokens == 64_000
    assert config.agent.stages["source_discovery"].timeout_seconds == 900
    assert load_build_config(ACTIVE_CONFIGS[1], model_profile=profile).llm.timeout_seconds == 800


def test_active_configs_preserve_coverage_with_authorized_call_budget() -> None:
    profile = load_execution_model_profile()
    for path in ACTIVE_CONFIGS:
        config = load_build_config(path, model_profile=profile)
        assert (
            config.semantic.target_population_members_per_hypothesis,
            config.semantic.max_classified_population_members_per_hypothesis,
            config.semantic.max_total_classified_population_members,
            config.semantic.max_classification_calls_per_hypothesis,
            config.semantic.max_total_classification_calls,
            config.semantic.max_classification_request_bytes_per_hypothesis,
            config.semantic.max_total_classification_request_bytes,
        ) == (15, 20, 1_000, 32, 256, 16_000_000, 64_000_000)
        assert (
            config.llm.max_total_calls,
            config.llm.max_total_input_tokens,
            config.llm.max_total_output_tokens,
        ) == (15_000, 100_000_000, 20_000_000)


def test_active_configs_use_twenty_four_provider_workers_and_four_observation_workers() -> None:
    profile = load_execution_model_profile()

    assert {
        load_build_config(path, model_profile=profile).llm.max_concurrency
        for path in ACTIVE_CONFIGS
    } == {24}
    assert ObservationSettings().max_workers == 4
    assert {
        load_build_config(path, model_profile=profile).observation.max_workers
        for path in ACTIVE_CONFIGS
    } == {4}


def test_active_configs_resolve_the_one_new_default_profile() -> None:
    model_profile = load_execution_model_profile()
    configs = [
        load_build_config(path, model_profile=model_profile)
        for path in ACTIVE_CONFIGS
    ]
    profile = load_agent_profile(
        AGENT_RUNTIME_PROFILE,
        model_profile=model_profile,
    )

    assert model_profile.profile_id == DEFAULT_EXECUTION_MODEL_PROFILE_ID
    assert model_profile.canonical_model_id == "qwen3.8-flash"
    assert {
        (
            config.execution_model_profile_id,
            config.execution_model_profile_digest,
            config.llm.model,
            tuple(config.llm.accepted_effective_model_ids),
            config.llm.api_key_env,
            config.llm.thinking_mode,
            config.llm.reasoning_effort,
            config.llm.reasoning_wire_policy,
        )
        for config in configs
    } == {
        (
            DEFAULT_EXECUTION_MODEL_PROFILE_ID,
            model_profile.profile_digest,
            "qwen3.8-flash",
            ("qwen3.8-flash",),
            "APIYI_API_KEY",
            "enabled",
            "medium",
            "qwen_enable_thinking",
        )
    }
    assert profile.profile_id == "mini-swe-agent-2.4.6-runtime"
    assert profile.model_id == "openai/qwen3.8-flash"
    assert profile.execution_model_profile_digest == model_profile.profile_digest
    assert profile.model.reasoning_mode == "enabled"
    assert profile.model.reasoning_effort == "medium"
    assert profile.runner_protocol_version == "norm4code-agent-runner-1.2"
    assert profile.adapter_version == "1.4"


def test_active_yaml_files_are_model_free() -> None:
    forbidden_build_keys = {
        "provider",
        "model",
        "require_response_model",
        "accepted_effective_model_ids",
        "api_key_env",
        "base_url",
        "temperature",
        "thinking_mode",
        "reasoning_effort",
        "reasoning_wire_policy",
    }
    for path in ACTIVE_CONFIGS:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert forbidden_build_keys.isdisjoint(raw["llm"])

    raw_agent = yaml.safe_load(AGENT_RUNTIME_PROFILE.read_text(encoding="utf-8"))
    assert "model" not in raw_agent
    assert "model_id" not in raw_agent
    assert "execution_model_profile_id" not in raw_agent
    assert "execution_model_profile_digest" not in raw_agent
    assert raw_agent.get("environment_keys", []) == []


def test_general_llm_defaults_use_twenty_four_workers_and_conservative_retries() -> None:
    settings = LLMSettings(
        model="test-model",
        require_response_model=True,
        accepted_effective_model_ids=["test-model"],
        api_key_env="TEST_API_KEY",
        base_url="https://example.invalid/v1",
        reasoning_wire_policy="offline_mock",
    )

    assert settings.timeout_seconds == 600
    assert settings.max_concurrency == 24
    assert (
        settings.provider_retries,
        settings.provider_retry_delay_seconds,
        settings.provider_retry_backoff_multiplier,
        settings.provider_retry_max_delay_seconds,
        settings.provider_retry_jitter_seconds,
    ) == (2, 1.0, 2.0, 30.0, 0.5)
