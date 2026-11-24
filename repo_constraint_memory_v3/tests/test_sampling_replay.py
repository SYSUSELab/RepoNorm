from __future__ import annotations

from pathlib import Path

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.run_validation import validate_run_directory


class _ExplodingClient:
    def complete(
        self, system_prompt: str, user_prompt: str
    ) -> ProviderResponse:
        del system_prompt, user_prompt
        raise AssertionError("a cached sampled build called the provider")


def test_sampled_population_cache_and_offline_replay_preserve_frozen_plan(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests(
        semantic={
            "max_classified_population_members_per_hypothesis": 1,
            "max_total_classified_population_members": 1,
        },
        unified={
            "candidate_grouping_mode": "singleton",
            "git_enabled": False,
        },
    )
    cache = tmp_path / "inference-cache"
    live_run = tmp_path / "live-run"
    live = build_fixture(
        repository,
        live_run,
        config=config,
        client=ParameterizedSemanticClient(
            category="sample-replay-identity",
            rule="Contributors preserve the sampled fixture convention.",
        ),
        inference_cache=cache,
    )

    assert live.current_source is not None
    sampled = [
        population
        for population in live.current_source.populations
        if population.sampling_plan.mode == "deterministic_stratified_sample"
    ]
    assert sampled
    assert all(
        population.sampling_plan.sampling_plan_status == "complete"
        and population.sampling_plan.census_enumeration_status == "complete"
        and population.sampling_plan.matched_count
        == population.sampling_plan.inspected_count
        + population.sampling_plan.uninspected_count
        and population.sampling_plan.uninspected_count > 0
        for population in sampled
    )

    cached_run = tmp_path / "cached-run"
    cached = build_fixture(
        repository,
        cached_run,
        config=config,
        client=_ExplodingClient(),
        inference_cache=cache,
    )

    assert cached.current_source is not None
    assert cached.current_source.populations == live.current_source.populations
    assert cached.current_source.index.population_matched_members == (
        live.current_source.index.population_matched_members
    )
    assert cached.current_source.index.population_inspected_members == (
        live.current_source.index.population_inspected_members
    )
    assert cached.current_source.index.population_uninspected_members == (
        live.current_source.index.population_uninspected_members
    )
    assert validate_run_directory(live_run, repository_root=repository) == []
    assert validate_run_directory(cached_run, repository_root=repository) == []
