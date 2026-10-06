from __future__ import annotations

from pathlib import Path

import pytest

from repo_constraint_memory_v3.observations import run_observations

from helpers import (
    config_for_tests,
    copied_observation_fixture,
    discover_fixture,
)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("max_ast_nodes_per_file", 1, "max_ast_nodes_per_file"),
        ("max_entities_per_file", 1, "max_entities_per_file"),
        ("max_relations_per_file", 1, "max_relations_per_file"),
        ("max_symbol_chars", 16, "max_symbol_chars"),
        ("max_file_artifact_bytes", 1000, "max_file_artifact_bytes"),
        (
            "max_total_observation_bytes",
            1000,
            "max_total_observation_bytes",
        ),
    ],
)
def test_semantic_and_serialized_limits_are_auditable_partial(
    tmp_path: Path,
    field: str,
    value: int,
    reason: str,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests(observation={field: value})
    discovery = discover_fixture(repository, config)
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert bundle.snapshot is not None
    assert bundle.snapshot.status == "partial"
    assert any(
        reason in incomplete
        for shard in bundle.shards
        for incomplete in shard.incomplete_reasons
    )
    assert bundle.index.overall == "partial"


def test_syntax_encoding_and_parse_failure_limit_are_visible(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    (repository / "broken_a.py").write_text("def broken(:\n", encoding="utf-8")
    (repository / "broken_b.py").write_text("def also_broken(:\n", encoding="utf-8")
    (repository / "invalid.py").write_bytes(
        b"def value():\n    return '" + bytes([255]) + b"'\n"
    )
    config = config_for_tests(
        observation={"max_parse_failures": 1, "max_workers": 1}
    )
    discovery = discover_fixture(repository, config)
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=tmp_path / "cache",
    ).bundle
    reasons = {
        reason
        for shard in bundle.shards
        for reason in shard.incomplete_reasons
    }
    assert "syntax_error" in reasons
    assert "max_parse_failures" in reasons
    assert "invalid_utf8" in reasons


def test_total_entity_and_relation_limits_are_partial(tmp_path: Path) -> None:
    repository = copied_observation_fixture(tmp_path)
    for field in ("max_total_entities", "max_total_relations"):
        config = config_for_tests(observation={field: 1})
        discovery = discover_fixture(repository, config)
        bundle = run_observations(
            repository,
            discovery,
            config.observation,
        ).bundle
        assert bundle.snapshot is not None
        assert bundle.snapshot.status == "partial"
        assert any(
            field in reason
            for shard in bundle.shards
            for reason in shard.incomplete_reasons
        )
