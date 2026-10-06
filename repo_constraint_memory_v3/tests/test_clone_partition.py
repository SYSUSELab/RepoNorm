from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from helpers import (
    config_for_tests,
    copied_observation_fixture,
    discover_fixture,
)
from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.observations.clones import observe_clone_clusters
from repo_constraint_memory_v3.observations.models import (
    PythonCloneClustersArtifact,
)
from repo_constraint_memory_v3.observations.semantic_units import (
    build_semantic_units,
)
from repo_constraint_memory_v3.semantic.classify import _fact_index


def test_overlapping_exact_clone_relations_form_one_disjoint_component(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    sources = {
        "src/pkg/overlap_a.py": (
            "def overlap_unique(value):\n"
            "    return value + 7\n"
        ),
        "src/pkg/overlap_b.py": (
            "def overlap_unique(value):\n"
            "    return value + 7\n"
        ),
        "src/pkg/overlap_c.py": (
            "# AST-equivalent, but not content-equivalent.\n\n"
            "def overlap_unique( value ):\n"
            "    return value+7\n"
        ),
    }
    for relative, content in sources.items():
        (repository / relative).write_text(content, encoding="utf-8")

    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
        cache_directory=None,
    ).bundle
    assert observations.clone_clusters is not None
    assert observations.clone_clusters.query_version == "1.1"
    assert observations.snapshot is not None
    assert observe_clone_clusters(
        observations.snapshot,
        tuple(reversed(observations.shards)),
    ) == observations.clone_clusters
    clone_entry = next(
        item
        for item in observations.index.query_plan
        if item.name == "python_clone_clusters"
    )
    assert clone_entry.version == "1.1"

    path_by_id = {item.file_id: item.path for item in observations.shards}
    target_paths = set(sources)
    target_ids = {
        file_id
        for file_id, path in path_by_id.items()
        if path in target_paths
    }
    containing = [
        cluster
        for cluster in observations.clone_clusters.clusters
        if target_ids.intersection(cluster.member_file_ids)
    ]
    assert len(containing) == 1
    cluster = containing[0]
    assert {path_by_id[item] for item in cluster.member_file_ids} == target_paths
    assert set(cluster.match_basis) == {"exact_ast", "exact_content"}

    all_members = [
        member
        for item in observations.clone_clusters.clusters
        for member in item.member_file_ids
    ]
    assert len(all_members) == len(set(all_members))

    semantic_units = build_semantic_units(
        repository,
        discovery,
        observations,
        config.semantic,
    )
    target_units = [
        item for item in semantic_units.units if item.path in target_paths
    ]
    assert {item.path for item in target_units} == target_paths
    assert {item.clone_group_id for item in target_units} == {
        cluster.cluster_id
    }

    hypothesis = SimpleNamespace(
        independence_spec=SimpleNamespace(mode="clone_aware_observation")
    )
    facts = _fact_index(
        repository,
        discovery,
        observations,
        semantic_units,
        hypothesis,  # type: ignore[arg-type]
        config.semantic,
        required_observation_ids=target_ids,
    )
    assert {facts[item].clone_group for item in target_ids} == {
        cluster.cluster_id
    }
    assert {
        facts[item].clone_independence_group for item in target_ids
    } == {f"clone:{cluster.cluster_id}"}

    payload = observations.clone_clusters.model_dump(mode="json")
    payload["clusters"] = [
        cluster.model_dump(mode="json"),
        cluster.model_dump(mode="json"),
    ]
    with pytest.raises(ValidationError, match="disjoint membership"):
        PythonCloneClustersArtifact.model_validate(payload)
