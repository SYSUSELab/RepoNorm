from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from helpers import (
    config_for_tests,
    copied_semantic_fixture,
    discover_fixture,
)
from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.observations.semantic_units import (
    build_semantic_units,
    reconstruct_packet_bytes,
)


def _bundle(
    tmp_path: Path,
    *,
    broken: bool = False,
    observation=None,
    semantic=None,
):
    repository = copied_semantic_fixture(tmp_path, include_broken=broken)
    config = config_for_tests(observation=observation, semantic=semantic)
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=None,
    ).bundle
    units = build_semantic_units(
        repository,
        discovery,
        observations,
        config.semantic,
    )
    return repository, discovery, observations, units, config


def test_primary_ownership_partitions_every_parsed_file(tmp_path: Path) -> None:
    repository, _, observations, bundle, _ = _bundle(tmp_path)
    expected = sum(
        len(shard.entities)
        for shard in observations.shards
        if shard.parse_status == "complete"
    )
    assert bundle.index.eligible_primary_universe_count == expected
    by_path = defaultdict(list)
    for unit in bundle.units:
        by_path[unit.path].extend(
            (item.start_utf8_byte, item.end_utf8_byte)
            for item in unit.owned_source_ranges
        )
        if unit.decorator_site_ids:
            assert any(
                ref.kind == "site"
                and ref.stable_id in unit.decorator_site_ids
                for ref in unit.context_refs
            )
    for path, ranges in by_path.items():
        content = (repository / path).read_bytes()
        cursor = 0
        for start, end in sorted(ranges):
            assert start == cursor
            assert end > start
            cursor = end
        assert cursor == len(content)

    entities = {
        entity.entity_id: entity
        for shard in observations.shards
        for entity in shard.entities
    }
    alpha = (repository / "src/alpha/core.py").read_bytes()

    def owned(unit) -> bytes:
        return b"".join(
            alpha[item.start_utf8_byte : item.end_utf8_byte]
            for item in unit.owned_source_ranges
        )

    alpha_units = {
        entities[unit.entity_id].qualified_name: unit
        for unit in bundle.units
        if unit.path == "src/alpha/core.py"
    }
    assert {unit.component_key for unit in bundle.units} == {
        "module:alpha",
        "module:beta",
    }
    module_bytes = owned(alpha_units["alpha.core"])
    assert b"MODULE_FLAG = True" in module_bytes
    assert b'AFTER_DEFINITIONS = "module-owned"' in module_bytes
    assert b"class Resource" not in module_bytes
    class_bytes = owned(alpha_units["Resource"])
    assert b'@marker("component")' in class_bytes
    compute_bytes = owned(alpha_units["Resource.compute"])
    nested_bytes = owned(alpha_units["Resource.compute.nested"])
    assert b"def nested" not in compute_bytes
    assert b'return nested("done")' in compute_bytes
    assert b"def nested" in nested_bytes

    unit_ids = {unit.unit_id for unit in bundle.units}
    observation_ids = {
        "file": {
            shard.file_id for shard in observations.shards
        },
        "entity": {
            entity.entity_id
            for shard in observations.shards
            for entity in shard.entities
        },
        "site": {
            site.site_id
            for shard in observations.shards
            for site in shard.sites
        },
        "relation": {
            relation.relation_id
            for shard in observations.shards
            for relation in shard.relations
        },
        "clone": {
            cluster.cluster_id
            for cluster in observations.clone_clusters.clusters
        },
    }
    for unit in bundle.units:
        if unit.parent_unit_id is not None:
            assert unit.parent_unit_id in unit_ids
        for ref in unit.context_refs:
            if ref.kind in {"parent", "child"}:
                assert ref.stable_id in observation_ids["entity"]
            else:
                assert ref.stable_id in observation_ids[ref.kind]
    cloned = [
        unit
        for unit in bundle.units
        if unit.path in {"src/beta/clone.py", "src/beta/worker.py"}
    ]
    assert cloned
    assert all(unit.clone_group_id is not None for unit in cloned)


def test_segmentation_reconstructs_packets_without_source_duplication(
    tmp_path: Path,
) -> None:
    repository, _, _, bundle, _ = _bundle(
        tmp_path,
        semantic={
            "max_unit_source_bytes": 80,
            "max_packet_bytes": 4_000,
        },
    )
    assert any(len(unit.segments) > 1 for unit in bundle.units)
    for unit in bundle.units:
        content = (repository / unit.path).read_bytes()
        for packet in unit.packets:
            raw = reconstruct_packet_bytes(packet, unit=unit, content=content)
            assert len(raw) == packet.reconstructed_bytes
            assert b"untrusted_data" in raw
        assert "source_text" not in unit.model_dump(mode="json")


def test_unit_limit_and_parse_failure_are_truthful_partial(tmp_path: Path) -> None:
    _, _, _, limited, _ = _bundle(
        tmp_path,
        semantic={"max_units": 2},
    )
    assert limited.index.status == "partial"
    assert limited.index.materialized_primary_count == len(limited.units) == 2
    assert limited.index.omitted_count > 0
    assert limited.index.eligible_primary_universe_count == (
        limited.index.materialized_primary_count + limited.index.omitted_count
    )
    assert sum(limited.index.component_universe_counts.values()) == (
        limited.index.eligible_primary_universe_count
    )
    assert sum(limited.index.component_selected_counts.values()) == len(
        limited.units
    )
    assert limited.index.component_selected_counts == {
        "module:alpha": 1,
        "module:beta": 1,
    }
    selected_unit_ids = {unit.unit_id for unit in limited.units}
    assert all(
        unit.parent_unit_id is None or unit.parent_unit_id in selected_unit_ids
        for unit in limited.units
    )
    assert "max_units" in limited.index.incomplete_reasons

    other = tmp_path / "other"
    other.mkdir()
    _, _, _, broken, _ = _bundle(other, broken=True)
    assert broken.index.status == "partial"
    assert any(
        item.parse_status == "failed" and item.status == "failed"
        for item in broken.index.file_dependencies
    )
    _, _, _, complete, _ = _bundle(tmp_path / "no-input")
    no_input = [
        unit for unit in complete.units if unit.path.endswith("/trivia.py")
    ]
    assert len(no_input) == 1
    assert no_input[0].unit_kind == "module"
    assert no_input[0].input_status == "no_input"
    assert not no_input[0].segments
    assert complete.index.no_input_count >= 1

    _, _, _, skipped, _ = _bundle(
        tmp_path / "skipped",
        observation={"max_python_file_bytes": 16},
    )
    assert skipped.index.status == "partial"
    assert not skipped.units
    assert all(
        item.parse_status == "skipped" and item.status == "partial"
        for item in skipped.index.file_dependencies
    )

    _, _, _, packet_limited, _ = _bundle(
        tmp_path / "packet-limited",
        semantic={"max_total_packet_bytes": 1},
    )
    assert packet_limited.index.status == "partial"
    assert any(
        reason.startswith("max_total_packet_bytes")
        for reason in packet_limited.index.incomplete_reasons
    )
    assert packet_limited.index.failed_count > 0


def test_unsplittable_physical_lines_are_not_truncated(tmp_path: Path) -> None:
    _, _, _, bundle, _ = _bundle(
        tmp_path,
        semantic={
            "max_unit_source_bytes": 10,
            "max_packet_bytes": 4_000,
        },
    )
    assert bundle.index.status == "partial"
    affected = [
        unit
        for unit in bundle.units
        if any(
            reason.startswith("unsplittable_physical_line")
            for reason in unit.incomplete_reasons
        )
    ]
    assert affected
    assert all(unit.input_status in {"partial", "failed"} for unit in affected)


def test_unit_and_packet_ids_ignore_observation_cache_and_workers(
    tmp_path: Path,
) -> None:
    repository = copied_semantic_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    serial = config.observation.model_copy(update={"max_workers": 1})
    parallel = config.observation.model_copy(update={"max_workers": 4})
    first_observations = run_observations(
        repository,
        discovery,
        serial,
        cache_directory=tmp_path / "cache",
    ).bundle
    second_observations = run_observations(
        repository,
        discovery,
        parallel,
        cache_directory=tmp_path / "cache",
    ).bundle
    first = build_semantic_units(
        repository, discovery, first_observations, config.semantic
    )
    second = build_semantic_units(
        repository, discovery, second_observations, config.semantic
    )
    assert first == second
