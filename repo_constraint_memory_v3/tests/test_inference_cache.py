from __future__ import annotations

from pathlib import Path

import pytest

from helpers import CONFIG_PATH, build_fixture, config_for_tests, copied_fixture
from repo_constraint_memory_v3.builder import (
    load_build_config,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.llm import LLMProviderError, MockLLMClient
from repo_constraint_memory_v3.inference_cache import SuccessInferenceCache
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile


class _CountingMockClient:
    def __init__(self) -> None:
        self.calls = 0
        self.mock = MockLLMClient()

    def complete(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        return self.mock.complete(system_prompt, user_prompt)


class _ExplodingClient:
    def complete(self, system_prompt: str, user_prompt: str):
        del system_prompt, user_prompt
        raise AssertionError("a reusable successful inference called the provider")


def test_cache_admission_rejection_reclassifies_hit_as_miss(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    cache = SuccessInferenceCache(
        None,
        resume_from=None,
        repository_root=repository,
        max_entry_bytes=10_000,
        max_resume_entries=10,
    )
    cache.metrics.hits = 1

    cache.record_admission_rejected()

    assert cache.metrics.hits == 0
    assert cache.metrics.misses == 1


def _assert_replayed_semantics_are_identical(first, second) -> None:
    assert first.inference_records == second.inference_records
    assert first.current_source == second.current_source
    assert first.traceability == second.traceability
    assert first.memory == second.memory


def test_external_success_cache_replays_without_provider_calls(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    cache = tmp_path / "inference-cache"
    progress: list[str] = []
    first_client = _CountingMockClient()
    first = build_fixture(
        repository,
        tmp_path / "first-run",
        client=first_client,
        inference_cache=cache,
    )
    second = build_fixture(
        repository,
        tmp_path / "second-run",
        client=_ExplodingClient(),
        inference_cache=cache,
        progress_callback=progress.append,
    )

    assert first_client.calls == len(first.inference_records)
    assert second.build_report["llm"]["this_run_provider_calls"] == 0
    assert second.build_report["llm"]["this_run_cache_hits"] == len(
        second.inference_records
    )
    assert second.build_report["llm"]["cache"]["hits"] == len(
        second.inference_records
    )
    assert all(
        stage["retries"] == 0 and stage["repairs"] == 0
        for stage in second.build_report["llm"]["stage_progress"].values()
    )
    assert any("cache_hits=" in message for message in progress)
    assert not list(cache.rglob("*.tmp"))
    _assert_replayed_semantics_are_identical(first, second)
    assert validate_run_directory(
        tmp_path / "second-run",
        repository_root=repository,
    ) == []


def test_prior_run_is_a_success_only_resume_source(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    first_run = tmp_path / "first-run"
    first = build_fixture(repository, first_run)
    second = build_fixture(
        repository,
        tmp_path / "resumed-run",
        client=_ExplodingClient(),
        resume_from=first_run,
    )

    assert second.build_report["llm"]["this_run_provider_calls"] == 0
    assert second.build_report["llm"]["cache"]["resume_entries"] == len(
        first.inference_records
    )
    _assert_replayed_semantics_are_identical(first, second)
    assert validate_run_directory(
        tmp_path / "resumed-run",
        repository_root=repository,
    ) == []


def test_failed_inference_is_retried_and_never_cached(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    cache = tmp_path / "inference-cache"
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={
                    "max_concurrency": 1,
                    "provider_retries": 0,
                    "schema_repairs": 0,
                }
            )
        }
    )

    class FailFirstThenMock(_CountingMockClient):
        def complete(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if self.calls == 1:
                raise LLMProviderError("terminal fixture failure", retryable=False)
            return self.mock.complete(system_prompt, user_prompt)

    first = build_fixture(
        repository,
        tmp_path / "partial-run",
        config=config,
        client=FailFirstThenMock(),
        inference_cache=cache,
    )
    failed = {
        record.request_id
        for record in first.inference_records
        if record.terminal_status == "failure"
    }
    assert failed

    second_client = _CountingMockClient()
    second = build_fixture(
        repository,
        tmp_path / "recovered-run",
        config=config,
        client=second_client,
        inference_cache=cache,
    )
    recovered = {
        record.request_id
        for record in second.inference_records
        if record.terminal_status == "success"
    }
    assert failed <= recovered
    assert second_client.calls >= len(failed)
    assert second.build_report["llm"]["this_run_cache_hits"] > 0


@pytest.mark.parametrize(
    ("enabled_profile_id", "effort"),
    [
        ("apiyi-deepseek-v4-flash-260425-thinking-high", "high"),
        ("apiyi-qwen3.8-flash-thinking-medium", "medium"),
    ],
)
def test_thinking_mode_is_part_of_success_cache_identity(
    enabled_profile_id: str, effort: str
) -> None:
    disabled_profile = load_execution_model_profile(
        "apiyi-deepseek-v4-flash-260425-nonthinking"
    )
    enabled_profile = load_execution_model_profile(enabled_profile_id)
    disabled = protocol_manifest_for_config(
        load_build_config(CONFIG_PATH, model_profile=disabled_profile),
        agent_profile_digest="0" * 64,
    )
    enabled = protocol_manifest_for_config(
        load_build_config(CONFIG_PATH, model_profile=enabled_profile),
        agent_profile_digest="0" * 64,
    )

    assert disabled.protocol_manifest_digest != enabled.protocol_manifest_digest
    assert disabled.execution_model_profile_digest != (
        enabled.execution_model_profile_digest
    )
    assert (disabled.model_parameters.thinking_mode, disabled.model_parameters.reasoning_effort) == (
        "disabled",
        None,
    )
    assert (enabled.model_parameters.thinking_mode, enabled.model_parameters.reasoning_effort) == (
        "enabled",
        effort,
    )


def test_endpoint_profile_is_part_of_success_cache_identity(tmp_path: Path) -> None:
    del tmp_path
    base_profile = load_execution_model_profile(
        "apiyi-deepseek-v4-flash-260425-nonthinking"
    )
    other_endpoint_profile = base_profile.model_copy(
        update={
            "profile_id": "alternate-endpoint-test",
            "provider": base_profile.provider.model_copy(
                update={"base_url": "https://other-endpoint.invalid/v1"}
            )
        }
    )
    first = protocol_manifest_for_config(
        load_build_config(CONFIG_PATH, model_profile=base_profile),
        agent_profile_digest="0" * 64,
    )
    second = protocol_manifest_for_config(
        load_build_config(CONFIG_PATH, model_profile=other_endpoint_profile),
        agent_profile_digest="0" * 64,
    )

    assert first.endpoint_profile_digest != second.endpoint_profile_digest
    assert first.protocol_manifest_digest != second.protocol_manifest_digest


def test_inference_cache_must_be_outside_target_repository(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the target repository"):
        build_fixture(
            repository,
            tmp_path / "run",
            inference_cache=repository / ".unsafe-cache",
        )


def test_resume_run_and_output_must_be_disjoint(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    prior = tmp_path / "prior-run"
    build_fixture(repository, prior)
    with pytest.raises(ValueError, match="outside the generated run"):
        build_fixture(
            repository,
            prior / "nested-output",
            resume_from=prior,
        )
