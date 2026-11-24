from __future__ import annotations

from pathlib import Path

import json
import pytest

import repo_constraint_memory_v3.observations.python_snapshot as python_snapshot
from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.observations.cache import (
    CACHE_SCHEMA_VERSION,
    ParseCache,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_text,
    payload_digest,
)
from repo_constraint_memory_v3.observations.models import (
    ParsedPythonDescriptor,
)
from repo_constraint_memory_v3.observations.python_snapshot import (
    parser_provenance,
)
from repo_constraint_memory_v3.observations.semantic_units import (
    build_semantic_units,
)

from helpers import (
    config_for_tests,
    copied_observation_fixture,
    discover_fixture,
)


_FORGED_TARGET = "CACHE_DESCRIPTOR_MUST_NOT_REACH_PROVIDER"


def _cache_for(repository: Path, config, directory: Path) -> ParseCache:
    settings = config.observation
    return ParseCache(
        directory,
        repository_root=repository,
        parser=parser_provenance(),
        semantic_limits={
            "max_ast_nodes_per_file": settings.max_ast_nodes_per_file,
            "max_entities_per_file": settings.max_entities_per_file,
            "max_relations_per_file": settings.max_relations_per_file,
            "max_symbol_chars": settings.max_symbol_chars,
        },
        max_entry_bytes=settings.max_cache_entry_bytes,
    )


def _forged_descriptor(content_sha256: str) -> ParsedPythonDescriptor:
    return ParsedPythonDescriptor(
        content_sha256=content_sha256,
        ast_node_count=1,
        ast_fingerprint="f" * 64,
        entities=[],
        relations=[
            {
                "kind": "references",
                "source_entity_index": -1,
                "target": _FORGED_TARGET,
                "alias": None,
                "start_line": 1,
                "start_column": 0,
                "end_line": 1,
                "end_column": 1,
                "dynamic": False,
            }
        ],
    )


def test_cache_cold_warm_corrupt_and_path_binding_are_deterministic(
    tmp_path: Path,
) -> None:
    assert CACHE_SCHEMA_VERSION == "1.1"
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    cache = tmp_path / "cache"

    cold = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )
    warm = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )
    assert cold.bundle == warm.bundle
    assert cold.processing_metrics["cache"]["writes"] > 0
    assert warm.processing_metrics["cache"]["hits"] > 0
    duplicate_ids = {
        item.path: item.file_id
        for item in cold.bundle.shards
        if item.path in {
            "src/pkg/duplicate_a.py",
            "src/pkg/duplicate_b.py",
        }
    }
    assert len(set(duplicate_ids.values())) == 2
    cache_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in cache.rglob("*.json")
    )
    assert "must not enter cache" not in cache_text
    assert '"literal"' not in cache_text

    entry = next(cache.rglob("*.json"))
    assert (
        json.loads(entry.read_text(encoding="utf-8"))["schema_version"]
        == "1.1"
    )
    entry.write_text('{"corrupt":true}\n', encoding="utf-8")
    recovered = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )
    assert recovered.bundle == cold.bundle
    assert recovered.processing_metrics["cache"]["rejected"] >= 1


def test_worker_count_does_not_change_artifacts_and_cache_must_be_external(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    serial_settings = config.observation.model_copy(
        update={"max_workers": 1}
    )
    parallel_settings = config.observation.model_copy(
        update={"max_workers": 4}
    )
    serial = run_observations(
        repository,
        discovery,
        serial_settings,
    )
    parallel = run_observations(
        repository,
        discovery,
        parallel_settings,
    )
    assert serial.bundle == parallel.bundle
    assert serial.processing_metrics["observed_max_workers"] <= 1
    assert parallel.processing_metrics["observed_max_workers"] <= 4

    with pytest.raises(ValueError, match="outside"):
        run_observations(
            repository,
            discovery,
            config.observation,
            cache_directory=repository / ".cache",
        )


def test_warm_cache_hit_still_requires_a_fresh_complete_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    cache = tmp_path / "cache"
    run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )
    calls: list[str] = []
    real_parse = python_snapshot.parse_python_descriptor

    def tracking_parse(content: bytes, **kwargs):
        calls.append(kwargs["content_sha256"])
        return real_parse(content, **kwargs)

    monkeypatch.setattr(
        python_snapshot,
        "parse_python_descriptor",
        tracking_parse,
    )
    warm = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )

    assert warm.processing_metrics["cache"]["hits"] > 0
    assert len(calls) >= warm.processing_metrics["cache"]["hits"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "9.9"),
        ("cache_key", "0" * 64),
        ("entry_digest", "0" * 64),
    ],
)
def test_cache_version_key_and_digest_mismatch_are_rejected_and_recomputed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    cache = tmp_path / "cache"
    expected = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    ).bundle
    entry = next(cache.rglob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload[field] = value
    entry.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )
    assert result.bundle == expected
    assert result.processing_metrics["cache"]["rejected"] >= 1


@pytest.mark.parametrize("tamper", ["content_identity", "semantic_limit"])
def test_resealed_cache_descriptor_identity_and_limits_are_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    cache = tmp_path / "cache"
    expected = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    ).bundle
    entry = next(cache.rglob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    if tamper == "content_identity":
        payload["descriptor"]["content_sha256"] = "f" * 64
    else:
        payload["descriptor"]["ast_node_count"] = (
            config.observation.max_ast_nodes_per_file + 1
        )
    payload["entry_digest"] = payload_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "entry_digest"
        }
    )
    entry.write_text(canonical_json_text(payload), encoding="utf-8")

    result = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache,
    )

    assert result.bundle == expected
    assert result.processing_metrics["cache"]["rejected"] >= 1


def test_resealed_structurally_valid_descriptor_is_rejected_by_fresh_parse(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    expected = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    expected_semantic_units = build_semantic_units(
        repository,
        discovery,
        expected,
        config.semantic,
    )
    target = next(
        item
        for item in discovery.manifest.files
        if item.path == "src/pkg/core.py"
    )
    cache_directory = tmp_path / "forged-cache"
    cache = _cache_for(repository, config, cache_directory)
    forged = _forged_descriptor(target.content_sha256)
    forged_key = cache.key_for(target.content_sha256)
    cache.write(forged_key, forged)

    recovered = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache_directory,
    )
    recovered_semantic_units = build_semantic_units(
        repository,
        discovery,
        recovered.bundle,
        config.semantic,
    )

    assert recovered.bundle == expected
    assert recovered_semantic_units == expected_semantic_units
    assert recovered.processing_metrics["cache"]["hits"] == 0
    assert recovered.processing_metrics["cache"]["rejected"] == 1
    assert _FORGED_TARGET not in "\n".join(
        artifact.model_dump_json()
        for artifact in recovered.bundle.query_artifacts().values()
    )
    assert _FORGED_TARGET not in "\n".join(
        unit.model_dump_json() for unit in recovered_semantic_units.units
    )
    forged_path = cache_directory / forged_key[:2] / f"{forged_key}.json"
    assert _FORGED_TARGET in forged_path.read_text(encoding="utf-8")


def test_syntax_invalid_source_cannot_be_promoted_by_valid_forged_cache(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    broken_path = repository / "src/pkg/forged_syntax.py"
    broken_path.write_text("def broken(:\n", encoding="utf-8")
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    target = next(
        item
        for item in discovery.manifest.files
        if item.path == "src/pkg/forged_syntax.py"
    )
    cache_directory = tmp_path / "syntax-forged-cache"
    cache = _cache_for(repository, config, cache_directory)
    forged = _forged_descriptor(target.content_sha256)
    cache.write(cache.key_for(target.content_sha256), forged)

    result = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=cache_directory,
    )
    shard = next(
        item
        for item in result.bundle.shards
        if item.path == "src/pkg/forged_syntax.py"
    )
    semantic_units = build_semantic_units(
        repository,
        discovery,
        result.bundle,
        config.semantic,
    )

    assert shard.parse_status == "failed"
    assert shard.incomplete_reasons == ["syntax_error"]
    assert result.processing_metrics["cache"]["hits"] == 0
    assert result.processing_metrics["cache"]["rejected"] == 0
    assert all(
        unit.path != "src/pkg/forged_syntax.py"
        for unit in semantic_units.units
    )
    assert _FORGED_TARGET not in "\n".join(
        artifact.model_dump_json()
        for artifact in result.bundle.query_artifacts().values()
    )
