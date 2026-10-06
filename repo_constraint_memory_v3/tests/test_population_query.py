from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from helpers import config_for_tests, copied_semantic_fixture, discover_fixture
from repo_constraint_memory_v3.discovery import discover_repository
from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.observations.canonical import (
    model_payload_digest,
    payload_digest,
)
from repo_constraint_memory_v3.observations.models import (
    MAX_RECORDED_PYTHON_MANIFEST_GAPS,
)
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.semantic.models import PopulationSpec
from repo_constraint_memory_v3.semantic.queries import (
    count_population_members_for_admission,
    evaluate_population,
    population_query_facts,
    population_value_catalog,
)
from repo_constraint_memory_v3.semantic import queries as queries_module
from repo_constraint_memory_v3.semantic.reduce import (
    HypothesisDraft,
    _preflight_population,
)
from repo_constraint_memory_v3.semantic.validation import (
    _population_root_schedule,
)


def _observations(tmp_path: Path):
    repository = copied_semantic_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository, discovery, config.observation
    ).bundle
    return observations, config


@pytest.mark.parametrize("universe", ["file", "entity", "site", "relation"])
def test_all_universes_are_deterministic(tmp_path: Path, universe: str) -> None:
    observations, config = _observations(tmp_path)
    spec = {"op": "universe", "universe": universe}
    first = evaluate_population(spec, observations, config.semantic)
    second = evaluate_population(spec, observations, config.semantic)
    assert first == second
    assert first.status == "complete"
    assert first.members == sorted(
        first.members, key=lambda item: item.observation_id
    )


def test_snapshot_query_context_reuses_facts_and_census_but_not_resource_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, config = _observations(tmp_path)
    original_facts = queries_module._facts
    original_evaluate = queries_module._evaluate
    counts = {"facts": 0, "evaluate": 0}

    def facts(bundle):
        counts["facts"] += 1
        return original_facts(bundle)

    def evaluate(expression, state):
        counts["evaluate"] += 1
        return original_evaluate(expression, state)

    monkeypatch.setattr(queries_module, "_facts", facts)
    monkeypatch.setattr(queries_module, "_evaluate", evaluate)
    spec = {"op": "universe", "universe": "entity"}
    result = evaluate_population(spec, observations, config.semantic)
    admission = count_population_members_for_admission(
        spec, observations, config.semantic, applies_to=["**"],
    )
    assert admission.matched_count == result.matched_count
    assert counts == {"facts": 1, "evaluate": 1}
    assert evaluate_population(spec, observations, config.semantic) == result
    assert result.scanned_count == len(observations.population_query_context.graph.facts["entity"])

    catalog = population_value_catalog(observations, max_values_per_field=2)
    expected_catalog = json.loads(json.dumps(catalog))
    catalog["universes"].clear()
    assert population_value_catalog(observations, max_values_per_field=2) == expected_catalog
    population_query_facts(observations, [item.observation_id for item in result.members])
    assert counts == {"facts": 1, "evaluate": 1}
    capped = config.semantic.model_copy(update={"max_population_members": 1})
    limited = evaluate_population(spec, observations, capped)
    assert limited.matched_count == result.matched_count
    assert limited.scanned_count == result.scanned_count
    assert limited.serialized_count == 1
    assert limited.omitted_count == result.matched_count - 1
    for byte_limit in (1, 128):
        with pytest.raises(ValueError, match="max_population_artifact_bytes"):
            evaluate_population(spec, observations, config.semantic.model_copy(
                update={"max_population_artifact_bytes": byte_limit},
            ))
    with pytest.raises(TypeError):
        observations.population_query_context.graph.facts["entity"]["changed"] = None


def test_query_context_is_owned_by_each_bundle_and_catalog_keeps_smallest_values(
    tmp_path: Path,
) -> None:
    observations, _ = _observations(tmp_path)
    separate = type(observations)(
        index=observations.index, snapshot=observations.snapshot,
        clone_clusters=observations.clone_clusters, shards=observations.shards,
    )
    assert separate.population_query_context is not observations.population_query_context
    for bound in (1, 2, 128, 2):
        catalog = population_value_catalog(observations, max_values_per_field=bound)
        for universe, payload in catalog["universes"].items():
            for field, field_payload in payload["fields"].items():
                values = set()
                for fact in observations.population_query_context.graph.facts[universe].values():
                    value = fact.values.get(field)
                    candidates = value if isinstance(value, tuple) else (value,)
                    values.update(item for item in candidates if isinstance(item, (str, bool)))
                ordered = sorted(values, key=lambda item: (str(type(item)), str(item)))
                assert field_payload["values"] == ordered[:bound]
                assert field_payload["truncated"] == (len(ordered) > bound)
                assert field_payload["distinct_count"] == (
                    None if len(ordered) > bound else len(ordered)
                )


def test_shared_query_graph_keeps_parallel_scan_state_independent(tmp_path: Path) -> None:
    observations, config = _observations(tmp_path)
    specs = [{"op": "universe", "universe": name} for name in (
        "file", "entity", "site", "relation",
    )]
    expected = [evaluate_population(spec, observations, config.semantic) for spec in specs]
    jobs = specs * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(
            lambda spec: evaluate_population(spec, observations, config.semantic), jobs,
        ))
    assert actual == expected * 8


def test_filters_traversal_and_set_algebra(tmp_path: Path) -> None:
    observations, config = _observations(tmp_path)
    functions = {
        "op": "filter",
        "source": {"op": "universe", "universe": "entity"},
        "predicates": [
            {"field": "kind", "operator": "in", "value": ["function", "method"]}
        ],
    }
    alpha = {
        "op": "filter",
        "source": {"op": "universe", "universe": "entity"},
        "predicates": [
            {"field": "path", "operator": "path_under", "value": "src/alpha"}
        ],
    }
    intersection = {
        "op": "intersection",
        "operands": [functions, alpha],
    }
    result = evaluate_population(intersection, observations, config.semantic)
    assert result.members
    assert all(item.path.startswith("src/alpha/") for item in result.members)

    children = evaluate_population(
        {
            "op": "traverse",
            "source": alpha,
            "traversal": "descendant",
        },
        observations,
        config.semantic,
    )
    assert children.members

    decorators = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "site"},
            "predicates": [
                {
                    "field": "decorator_symbols",
                    "operator": "has",
                    "value": "marker",
                }
            ],
        },
        observations,
        config.semantic,
    )
    assert decorators.matched_count >= 2

    all_entities = {"op": "universe", "universe": "entity"}
    union = evaluate_population(
        {"op": "union", "operands": [functions, functions]},
        observations,
        config.semantic,
    )
    function_result = evaluate_population(
        functions, observations, config.semantic
    )
    assert union.members == function_result.members
    difference = evaluate_population(
        {
            "op": "difference",
            "left": all_entities,
            "right": functions,
        },
        observations,
        config.semantic,
    )
    assert {
        item.observation_id for item in difference.members
    }.isdisjoint(
        item.observation_id for item in function_result.members
    )

    calls = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "relation"},
            "predicates": [
                {"field": "kind", "operator": "eq", "value": "calls"}
            ],
        },
        observations,
        config.semantic,
    )
    assert calls.members
    called_entities = evaluate_population(
        {
            "op": "traverse",
            "source": functions,
            "traversal": "relation",
            "relation_kind": "calls",
            "direction": "outgoing",
        },
        observations,
        config.semantic,
    )
    assert called_entities.source_universe == "entity"

    implementation_files = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {
                    "field": "source_form",
                    "operator": "eq",
                    "value": "implementation",
                },
                {
                    "field": "component",
                    "operator": "in",
                    "value": ["module:alpha", "module:beta"],
                },
            ],
        },
        observations,
        config.semantic,
    )
    assert implementation_files.members

    recursive_python = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {
                    "field": "path",
                    "operator": "path_glob",
                    "value": ["src/alpha/**/*.py"],
                }
            ],
        },
        observations,
        config.semantic,
    )
    assert recursive_python.members
    assert all(item.path.startswith("src/alpha/") for item in recursive_python.members)


def test_population_spec_rejects_uncompilable_path_glob() -> None:
    with pytest.raises(ValidationError, match="invalid character class"):
        PopulationSpec.model_validate(
            {
                "op": "filter",
                "source": {"op": "universe", "universe": "file"},
                "predicates": [
                    {
                        "field": "path",
                        "operator": "path_glob",
                        "value": ["src/[z-a].py"],
                    }
                ],
            }
        )


def test_empty_result_is_complete_and_evaluation_is_pure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observations, config = _observations(tmp_path)

    def fail_filesystem(*args, **kwargs):
        del args, kwargs
        raise AssertionError("population evaluation accessed the filesystem")

    monkeypatch.setattr(Path, "read_bytes", fail_filesystem)
    spec = {
        "op": "filter",
        "source": {"op": "universe", "universe": "entity"},
        "predicates": [
            {
                "field": "qualified_name",
                "operator": "eq",
                "value": "definitely.missing",
            }
        ],
    }
    first = evaluate_population(spec, observations, config.semantic)
    second = evaluate_population(spec, observations, config.semantic)
    assert first == second
    assert first.status == "complete"
    assert first.matched_count == 0
    assert first.members == []
    assert first.member_set_digest is not None


def test_limits_propagate_partial_without_negative_inference(tmp_path: Path) -> None:
    observations, config = _observations(tmp_path)
    limited = config.semantic.model_copy(update={"max_population_members": 1})
    result = evaluate_population(
        {"op": "universe", "universe": "entity"},
        observations,
        limited,
    )
    assert result.status == "partial"
    assert result.omitted_count > 0
    assert result.member_set_digest is None
    assert "max_population_members" in result.incomplete_reasons


def test_admission_count_uses_full_known_set_not_serialized_prefix(
    tmp_path: Path,
) -> None:
    observations, config = _observations(tmp_path)
    limited = config.semantic.model_copy(update={"max_population_members": 1})
    spec = {"op": "universe", "universe": "entity"}

    serialized = evaluate_population(spec, observations, limited)
    admission = count_population_members_for_admission(
        spec,
        observations,
        limited,
        applies_to=["**/*.py"],
    )

    assert serialized.status == "partial"
    assert len(serialized.members) == 1
    assert admission.matched_count == serialized.matched_count
    assert admission.in_scope_member_count == serialized.matched_count


def test_noncomplete_known_files_do_not_consume_provider_admission_cap(
    tmp_path: Path,
) -> None:
    observations, config = _observations(tmp_path)
    template = observations.shards[0].model_dump(mode="json")
    synthetic = []
    for ordinal in range(513):
        shard = json.loads(json.dumps(template))
        shard["file_id"] = f"PYFILE-{ordinal:024x}"
        shard["path"] = f"quarantined/member_{ordinal:04d}.py"
        shard["content_sha256"] = f"{ordinal:064x}"
        shard["parse_status"] = "skipped"
        shard["parser"] = None
        shard["ast_fingerprint"] = None
        shard["ast_node_count"] = None
        shard["entities"] = []
        shard["sites"] = []
        shard["relations"] = []
        shard["incomplete_reasons"] = ["source_quarantined"]
        shard["shard_digest"] = payload_digest(
            {key: value for key, value in shard.items() if key != "shard_digest"}
        )
        synthetic.append(type(observations.shards[0]).model_validate(shard))
    expanded = type(observations)(
        index=observations.index,
        snapshot=observations.snapshot,
        clone_clusters=observations.clone_clusters,
        shards=(*observations.shards, *synthetic),
    )

    admission = count_population_members_for_admission(
        {"op": "universe", "universe": "file"},
        expanded,
        config.semantic,
        applies_to=["**/*.py"],
    )

    assert admission.in_scope_member_count > 512
    assert admission.provider_eligible_in_scope_member_count == len(
        observations.shards
    )
    assert admission.provider_eligible_in_scope_member_count <= 512

    provider_draft_payload = json.loads(
        MockLLMClient()
        .complete(
            "",
            json.dumps(
                {
                    "stage": "current_hypothesis_reduce",
                    "mode": "component_partition",
                    "inputs": [{}],
                    "accounted_input_ids": ["OBS-1"],
                }
            ),
        )
        .content
    )["hypotheses"][0]
    provider_draft_payload["superpopulation_spec"] = {
        "op": "universe",
        "universe": "file",
    }
    provider_draft_payload["applies_to"] = ["**/*.py"]
    provider_draft = HypothesisDraft.model_validate(provider_draft_payload)
    preflight_cost = _preflight_population(
        provider_draft,
        observations=expanded,
        semantic_map=SimpleNamespace(observations=()),
        semantic_units=SimpleNamespace(units=()),
        settings=config.semantic,
    )

    assert preflight_cost == admission.provider_eligible_in_scope_member_count

    scheduled = _population_root_schedule(
        [
                SimpleNamespace(
                    hypothesis_id="HYP-source-quarantine-cap",
                    candidate_root_id="ROOT-source-quarantine-cap",
                component_keys=("component",),
                superpopulation_spec=provider_draft.superpopulation_spec,
                predicate_mode="semantic",
                applies_to=provider_draft.applies_to,
            )
        ],
        observations=expanded,
        semantic_settings=config.semantic,
    )
    assert scheduled[0].semantic_member_cost == (
        admission.provider_eligible_in_scope_member_count
    )
    assert scheduled[0].matched_member_count == admission.matched_count


def test_query_local_partiality_preserves_exact_file_denominator(
    tmp_path: Path,
) -> None:
    repository = copied_semantic_fixture(tmp_path, include_broken=True)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository, discovery, config.observation
    ).bundle
    assert observations.index.overall == "partial"
    incomplete_shard = next(
        shard for shard in observations.shards if shard.parse_status != "complete"
    )

    file_result = evaluate_population(
        {"op": "universe", "universe": "file"},
        observations,
        config.semantic,
    )
    assert file_result.status == "complete"
    assert file_result.dependency_file_ids == []
    assert file_result.member_set_digest is not None
    assert incomplete_shard.file_id in {
        item.file_id for item in file_result.members
    }

    entity_result = evaluate_population(
        {"op": "universe", "universe": "entity"},
        observations,
        config.semantic,
    )
    assert entity_result.status == "partial"
    assert entity_result.member_set_digest is None
    assert entity_result.dependency_file_ids == [incomplete_shard.file_id]
    assert "population_query_dependency_partial" in (
        entity_result.incomplete_reasons
    )

    excluded = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "entity"},
            "predicates": [
                {
                    "field": "path",
                    "operator": "path_under",
                    "value": "src/alpha",
                }
            ],
        },
        observations,
        config.semantic,
    )
    assert excluded.status == "complete"
    assert excluded.dependency_file_ids == []


def test_query_local_may_affect_follows_frozen_set_and_traverse_algebra(
    tmp_path: Path,
) -> None:
    repository = copied_semantic_fixture(tmp_path, include_broken=True)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository, discovery, config.observation
    ).bundle
    incomplete_file_id = next(
        shard.file_id
        for shard in observations.shards
        if shard.parse_status != "complete"
    )
    excluded = {
        "op": "filter",
        "source": {"op": "universe", "universe": "entity"},
        "predicates": [
            {
                "field": "path",
                "operator": "path_under",
                "value": "src/alpha",
            }
        ],
    }
    semantic_unknown = {
        "op": "filter",
        "source": {"op": "universe", "universe": "entity"},
        "predicates": [
            {"field": "kind", "operator": "eq", "value": "function"}
        ],
    }
    cases = [
        (
            {"op": "union", "operands": [excluded, semantic_unknown]},
            "partial",
        ),
        (
            {"op": "intersection", "operands": [excluded, semantic_unknown]},
            "complete",
        ),
        (
            {"op": "difference", "left": excluded, "right": semantic_unknown},
            "partial",
        ),
        (
            {"op": "traverse", "source": excluded, "traversal": "descendant"},
            "complete",
        ),
        (
            {
                "op": "traverse",
                "source": excluded,
                "traversal": "relation",
                "relation_kind": "calls",
                "direction": "outgoing",
            },
            "partial",
        ),
    ]
    for spec, expected_status in cases:
        result = evaluate_population(spec, observations, config.semantic)
        assert result.status == expected_status
        assert result.dependency_file_ids == (
            [incomplete_file_id] if expected_status == "partial" else []
        )


def test_unrepresented_manifest_python_path_is_typed_and_query_local(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "manifest-gap-repository"
    (repository / "src").mkdir(parents=True)
    (repository / "omitted").mkdir()
    (repository / "src/kept.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "omitted/large.py").write_text(
        "VALUE = " + repr("x" * 512) + "\n",
        encoding="utf-8",
    )
    config = config_for_tests()
    config = config.model_copy(
        update={
            "discovery": config.discovery.model_copy(
                update={"max_manifest_file_bytes": 128}
            )
        }
    )
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert observations.snapshot is not None
    snapshot = observations.snapshot

    assert snapshot.status == "partial"
    assert snapshot.eligible_file_count == len(observations.shards) + 1
    assert snapshot.skipped_file_count == 1
    assert [item.model_dump(mode="json") for item in snapshot.unrepresented_python_files] == [
        {
            "path": "omitted/large.py",
            "reason": "oversized_manifest_file",
            "reason_detail": None,
            "component_key": "path:omitted",
        }
    ]
    assert snapshot.unrepresented_python_file_count == 1
    assert snapshot.unrepresented_python_files_omitted_count == 0
    assert snapshot.artifact_digest == model_payload_digest(
        snapshot,
        digest_field="artifact_digest",
    )

    for universe in ("file", "entity"):
        result = evaluate_population(
            {"op": "universe", "universe": universe},
            observations,
            config.semantic,
        )
        assert result.status == "partial"
        assert result.dependency_file_ids == []
        assert result.dependency_paths == ["omitted/large.py"]
        assert result.member_set_digest is None

    disjoint = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "path_under", "value": "src"}
            ],
        },
        observations,
        config.semantic,
    )
    assert disjoint.status == "complete"
    assert disjoint.dependency_paths == []
    assert disjoint.member_set_digest is not None

    component_disjoint = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "entity"},
            "predicates": [
                {"field": "component", "operator": "eq", "value": "path:src"}
            ],
        },
        observations,
        config.semantic,
    )
    assert component_disjoint.status == "complete"
    assert component_disjoint.dependency_paths == []

    # A self-consistent rewrite cannot change the result of the pure replay.
    # This mirrors the offline validator's saved-vs-replayed population check.
    tampered_payload = evaluate_population(
        {"op": "universe", "universe": "file"},
        observations,
        config.semantic,
    ).model_dump(mode="json")
    tampered_payload.update(
        {
            "status": "complete",
            "dependency_paths": [],
            "incomplete_reasons": [],
            "member_set_digest": payload_digest(
                {
                    "universe": "file",
                    "members": tampered_payload["members"],
                }
            ),
        }
    )
    tampered_payload["result_digest"] = payload_digest(
        {
            key: value
            for key, value in tampered_payload.items()
            if key != "result_digest"
        }
    )
    tampered = type(disjoint).model_validate(tampered_payload)
    replayed = evaluate_population(
        {"op": "universe", "universe": "file"},
        observations,
        config.semantic,
    )
    assert tampered != replayed
    assert replayed.dependency_paths == ["omitted/large.py"]


def test_manifest_gap_recording_overflow_is_bounded_and_fail_closed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "many-manifest-gaps"
    (repository / "src").mkdir(parents=True)
    (repository / "src/kept.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = config_for_tests()
    gap_count = MAX_RECORDED_PYTHON_MANIFEST_GAPS + 3
    candidate_paths = [
        *(f"missing/member_{ordinal:05d}.py" for ordinal in range(gap_count)),
        "src/kept.py",
    ]
    discovery = discover_repository(
        repository,
        config.discovery,
        candidate_paths=candidate_paths,
    )
    observations = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert observations.snapshot is not None
    snapshot = observations.snapshot

    assert snapshot.unrepresented_python_file_count == gap_count
    assert len(snapshot.unrepresented_python_files) == (
        MAX_RECORDED_PYTHON_MANIFEST_GAPS
    )
    assert snapshot.unrepresented_python_files_omitted_count == 3

    # Even a path filter disjoint from every recorded gap cannot prove that the
    # bounded omitted suffix is disjoint, so the evaluator fails closed.
    result = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "path_under", "value": "src"}
            ],
        },
        observations,
        config.semantic,
    )
    assert result.status == "partial"
    assert result.dependency_paths == []
    assert result.dependency_paths_omitted_count == 3
    assert "population_query_unrepresented_path_overflow" in (
        result.incomplete_reasons
    )


def test_unrepresented_package_marker_propagates_cross_file_metadata_gap(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "package-marker-gap"
    (repository / "pkg").mkdir(parents=True)
    (repository / "pkg/x.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "pkg/__init__.py").write_bytes(b"PACKAGE = 1\0binary")
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert observations.snapshot is not None
    assert [
        item.path for item in observations.snapshot.unrepresented_python_files
    ] == ["pkg/__init__.py"]

    # The absent marker changes x.py's module/package/component metadata even
    # though the marker itself cannot pass this exact path predicate.
    result = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "eq", "value": "pkg/x.py"},
                {"field": "module_name", "operator": "eq", "value": "pkg.x"},
            ],
        },
        observations,
        config.semantic,
    )
    assert result.status == "partial"
    assert result.dependency_paths == ["pkg/__init__.py"]
    assert result.member_set_digest is None


def test_unrepresented_module_makes_direct_relation_resolution_partial(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "relation-resolution-gap"
    repository.mkdir()
    (repository / "a.py").write_text("import foo\n", encoding="utf-8")
    (repository / "foo.py").write_bytes(b"VALUE = 1\0binary")
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert observations.snapshot is not None
    assert [
        item.path for item in observations.snapshot.unrepresented_python_files
    ] == ["foo.py"]

    result = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "relation"},
            "predicates": [
                {"field": "path", "operator": "eq", "value": "a.py"},
                {
                    "field": "resolution",
                    "operator": "eq",
                    "value": "exact_local_module",
                },
            ],
        },
        observations,
        config.semantic,
    )

    assert result.status == "partial"
    assert result.dependency_paths == ["foo.py"]
    assert result.member_set_digest is None


def test_noncomplete_file_attribute_filter_is_partial_but_metadata_is_exact(
    tmp_path: Path,
) -> None:
    repository = copied_semantic_fixture(tmp_path)
    encoded = repository / "src/alpha/generated_invalid.py"
    encoded.write_bytes(b"# generated file\nVALUE = '\xff'\n")
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    observations = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    shard = next(
        item for item in observations.shards if item.path == "src/alpha/generated_invalid.py"
    )
    assert shard.parse_status == "failed"
    assert shard.incomplete_reasons == ["invalid_utf8"]

    all_files = evaluate_population(
        {"op": "universe", "universe": "file"},
        observations,
        config.semantic,
    )
    assert all_files.status == "complete"
    assert all_files.dependency_file_ids == []

    generated = evaluate_population(
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "attributes", "operator": "has", "value": "generated"}
            ],
        },
        observations,
        config.semantic,
    )
    assert generated.status == "partial"
    assert generated.dependency_file_ids == [shard.file_id]
    assert generated.dependency_paths == []

    disjoint = evaluate_population(
        {
            "op": "filter",
            "source": {
                "op": "filter",
                "source": {"op": "universe", "universe": "file"},
                "predicates": [
                    {"field": "path", "operator": "path_under", "value": "src/beta"}
                ],
            },
            "predicates": [
                {"field": "attributes", "operator": "has", "value": "generated"}
            ],
        },
        observations,
        config.semantic,
    )
    assert disjoint.status == "complete"
    assert disjoint.dependency_file_ids == []


@pytest.mark.parametrize(
    "spec",
    [
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "regex", "value": ".*"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "path_under", "value": "../escape"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "path_under", "value": "/absolute"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {
                    "field": "path",
                    "operator": "path_under",
                    "value": "https://example.invalid/repo",
                }
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "command", "operator": "eq", "value": "exec()"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path_under", "operator": "path_under", "value": "tests"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "entity"},
            "predicates": [
                {"field": "file_path", "operator": "eq", "value": "src/pkg.py"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "has", "value": ".py"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "attributes", "operator": "eq", "value": "testing"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "in", "value": ["**/*.py"]}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "path", "operator": "path_under", "value": "tests/**"}
            ],
        },
        {
            "op": "filter",
            "source": {"op": "universe", "universe": "file"},
            "predicates": [
                {"field": "role", "operator": "path_glob", "value": "**/*.py"}
            ],
        },
        {
            "op": "union",
            "operands": [
                {"op": "universe", "universe": "file"},
                {"op": "universe", "universe": "entity"},
            ],
        },
        {"op": "members", "ids": ["PYENT-000000000000000000000000"]},
    ],
)
def test_unsafe_or_incompatible_queries_are_rejected(
    tmp_path: Path,
    spec,
) -> None:
    observations, config = _observations(tmp_path)
    with pytest.raises((ValidationError, ValueError)):
        evaluate_population(spec, observations, config.semantic)


def test_query_complexity_limits_are_enforced(tmp_path: Path) -> None:
    observations, config = _observations(tmp_path)
    nested = {"op": "universe", "universe": "entity"}
    for _ in range(4):
        nested = {
            "op": "filter",
            "source": nested,
            "predicates": [
                {"field": "kind", "operator": "eq", "value": "function"}
            ],
        }
    with pytest.raises(ValueError, match="depth"):
        evaluate_population(
            nested,
            observations,
            config.semantic.model_copy(update={"max_population_query_depth": 3}),
        )
    with pytest.raises(ValueError, match="nodes"):
        evaluate_population(
            nested,
            observations,
            config.semantic.model_copy(update={"max_population_query_nodes": 3}),
        )
    with pytest.raises(ValueError, match="literal"):
        evaluate_population(
            {
                "op": "filter",
                "source": {"op": "universe", "universe": "entity"},
                "predicates": [
                    {
                        "field": "kind",
                        "operator": "in",
                        "value": ["function", "method"],
                    }
                ],
            },
            observations,
            config.semantic.model_copy(
                update={"max_population_literal_values": 1}
            ),
        )
    with pytest.raises(ValueError, match="artifact"):
        evaluate_population(
            {
                "op": "filter",
                "source": {"op": "universe", "universe": "entity"},
                "predicates": [
                    {
                        "field": "qualified_name",
                        "operator": "eq",
                        "value": "x" * 2_000,
                    }
                ],
            },
            observations,
            config.semantic.model_copy(
                update={"max_population_artifact_bytes": 1_000}
            ),
        )


def test_population_catalog_bounds_high_cardinality_without_exact_set(
    tmp_path: Path,
) -> None:
    observations, _ = _observations(tmp_path)
    first = population_value_catalog(observations, max_values_per_field=2)
    second = population_value_catalog(observations, max_values_per_field=2)
    assert first == second
    path_field = first["universes"]["file"]["fields"]["path"]
    assert len(path_field["values"]) == 2
    assert path_field["values"] == sorted(path_field["values"])
    assert path_field["truncated"] is True
    assert path_field["distinct_count"] is None
    assert path_field["distinct_count_lower_bound"] == 3
    with pytest.raises(ValueError, match="positive"):
        population_value_catalog(observations, max_values_per_field=0)


def test_population_query_facts_are_typed_bounded_and_accounted(
    tmp_path: Path,
) -> None:
    observations, config = _observations(tmp_path)
    files = evaluate_population(
        {"op": "universe", "universe": "file"},
        observations,
        config.semantic,
    )
    requested = [item.observation_id for item in files.members[:2]]
    facts = population_query_facts(
        observations,
        [*requested, "PYFILE-000000000000000000000000"],
        max_facts=1,
    )
    assert facts["requested_ref_count"] == 3
    assert facts["matched_fact_count"] == 2
    assert facts["serialized_fact_count"] == 1
    assert facts["omitted_fact_count"] == 1
    assert facts["unknown_refs"] == ["PYFILE-000000000000000000000000"]
    assert facts["facts"][0]["universe"] == "file"
    assert facts["facts"][0]["observation_id"] == sorted(requested)[0]
    assert facts["facts"][0]["values"]["path"]
