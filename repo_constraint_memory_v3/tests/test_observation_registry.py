from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.models import (
    ObservationQuerySettings,
    ObservationSettings,
)
from repo_constraint_memory_v3.observations.models import ParserProvenance
from repo_constraint_memory_v3.observations.python_snapshot import (
    SnapshotBuild,
    observe_python_snapshot,
)
from repo_constraint_memory_v3.observations.registry import (
    QueryRegistryError,
    QuerySpec,
    resolve_query_plan,
)
from repo_constraint_memory_v3.run_validation import validate_run_directory

from helpers import (
    build_fixture,
    config_for_tests,
    copied_fixture,
)


def test_strict_models_reject_extra_fields_and_type_coercion() -> None:
    with pytest.raises(ValidationError):
        ParserProvenance.model_validate(
            {
                "implementation": "CPython",
                "python_version": "3.12",
                "ast_contract_version": "1.0",
                "extra": True,
            }
        )
    with pytest.raises(ValidationError):
        ObservationSettings.model_validate(
            {
                "enabled": True,
                "queries": [
                    {"name": "python_snapshot", "required": "true"}
                ],
            }
        )
    with pytest.raises(ValidationError, match="unique"):
        ObservationSettings(
            queries=[
                ObservationQuerySettings(
                    name="python_snapshot",
                    required=True,
                ),
                ObservationQuerySettings(
                    name="python_snapshot",
                    required=False,
                ),
            ]
        )


def test_registry_rejects_unknown_missing_dependency_and_cycle() -> None:
    with pytest.raises(QueryRegistryError, match="unknown"):
        resolve_query_plan(["not_registered"])
    with pytest.raises(QueryRegistryError, match="missing dependencies"):
        resolve_query_plan(["python_clone_clusters"])
    cyclic = {
        "a": QuerySpec("a", "1.0", ("b",)),
        "b": QuerySpec("b", "1.0", ("a",)),
    }
    with pytest.raises(QueryRegistryError, match="cycle"):
        resolve_query_plan(["a", "b"], registry=cyclic)
    assert [item.name for item in resolve_query_plan(
        [
            "python_snapshot",
            "python_clone_clusters",
        ]
    )] == [
        "python_snapshot",
        "python_clone_clusters",
    ]
    assert resolve_query_plan(["python_snapshot"])[0].version == "1.4"
    assert resolve_query_plan(
        ["python_snapshot", "python_clone_clusters"]
    )[1].version == "1.1"


def test_optional_partial_query_does_not_fail_build_and_disabled_has_no_tree(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    optional = config_for_tests(
        observation={
            "queries": [
                ObservationQuerySettings(
                    name="python_snapshot",
                    required=False,
                )
            ],
            "max_ast_nodes_per_file": 1,
        },
    )
    optional_result = build_fixture(
        repository,
        tmp_path / "optional-run",
        config=optional,
    )
    # Optional observation degradation does not fail the package.  Related
    # variants alone no longer force temporal Git evidence, so this fixture has
    # no required Git capability despite not being a Git repository.
    assert optional_result.build_report["complete"]
    assert optional_result.build_report["unified"]["completeness"][
        "package_status"
    ] == "complete"
    assert optional_result.memory.coverage.observation.overall == "complete"
    assert optional_result.memory.coverage.observation.queries[0].status == "partial"
    assert validate_run_directory(
        tmp_path / "optional-run",
        repository_root=repository,
    ) == []

    disabled = config_for_tests(
        observation={"enabled": False, "queries": []},
    )
    disabled_result = build_fixture(
        repository,
        tmp_path / "disabled-run",
        config=disabled,
    )
    assert disabled_result.build_report["complete"]
    assert disabled_result.build_report["unified"]["completeness"][
        "package_status"
    ] == "complete"
    assert disabled_result.memory.coverage.observation.mode == "not_run"
    assert not (tmp_path / "disabled-run" / "observations").exists()


def test_unknown_query_is_a_build_input_error(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests(
        observation={
            "queries": [
                ObservationQuerySettings(
                    name="unknown_query",
                    required=True,
                )
            ]
        }
    )
    with pytest.raises(QueryRegistryError, match="unknown"):
        build_fixture(repository, tmp_path / "run", config=config)


def test_failed_dependency_marks_dependents_not_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    from helpers import discover_fixture

    discovery = discover_fixture(repository, config)
    base = observe_python_snapshot(
        repository,
        discovery,
        config.observation,
        cache_directory=None,
    )
    failed = base.artifact.model_copy(
        update={
            "status": "failed",
            "processed_file_count": 0,
            "failed_file_count": base.artifact.eligible_file_count,
            "entity_count": 0,
            "site_count": 0,
            "relation_count": 0,
            "incomplete_reasons": ["forced_query_failure"],
        }
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.observations.engine.observe_python_snapshot",
        lambda *args, **kwargs: SnapshotBuild(
            artifact=failed,
            shards=(),
            cache_metrics={
                "hits": 0,
                "misses": 0,
                "rejected": 0,
                "writes": 0,
            },
            observed_max_workers=0,
        ),
    )
    from repo_constraint_memory_v3.observations import run_observations

    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert bundle.index.query_plan[0].status == "failed"
    assert all(
        item.status == "not_run"
        for item in bundle.index.query_plan[1:]
    )
    assert all(
        item.artifact is None
        for item in bundle.index.query_plan[1:]
    )
