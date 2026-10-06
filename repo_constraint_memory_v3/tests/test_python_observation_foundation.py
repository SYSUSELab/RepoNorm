from __future__ import annotations

import ast
from pathlib import Path

import pytest

from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.observations.cache import ParseCache
from repo_constraint_memory_v3.observations.ids import (
    entity_id,
    file_id,
    site_id,
)
from repo_constraint_memory_v3.observations.models import SourceLocation
from repo_constraint_memory_v3.observations.python_snapshot import (
    AST_CONTRACT_VERSION,
    parser_provenance,
)

from helpers import (
    config_for_tests,
    copied_observation_fixture,
    discover_fixture,
)


def _bundle(tmp_path: Path):
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    result = run_observations(
        repository,
        discovery,
        config.observation,
    )
    return repository, discovery, result.bundle


def test_snapshot_roles_modules_entities_relations_and_locations(
    tmp_path: Path,
) -> None:
    repository, discovery, bundle = _bundle(tmp_path)
    assert bundle.index.overall == "complete"
    assert bundle.snapshot is not None
    assert bundle.snapshot.query_version == "1.4"
    assert bundle.snapshot.parser.ast_contract_version == (
        "stdlib-ast-structural-1.2"
    )
    assert bundle.snapshot.eligible_file_count == len(bundle.shards)
    assert not (repository / "OBSERVATION_EXECUTED").exists()

    by_path = {item.path: item for item in bundle.shards}
    assert by_path["src/pkg/core.py"].role.location_role == "production"
    assert by_path["tests/test_core.py"].role.location_role == "test"
    assert by_path["examples/demo.py"].role.location_role == "example"
    assert (
        by_path["migrations/versions/001_initial.py"].role.location_role
        == "migration"
    )
    assert (
        by_path["third_party/library.py"].role.location_role == "third_party"
    )
    assert by_path["namespace/tool.py"].role.location_role == "unknown"
    assert (
        by_path["tests/examples/test_conflict.py"].role.classification_status
        == "ambiguous"
    )
    assert "generated" in by_path["src/pkg/generated.py"].role.attributes
    assert "deprecated" in by_path["deprecated/old.py"].role.attributes
    assert by_path["src/pkg/core.pyi"].source_form == "stub"
    assert by_path["src/pkg/core.py"].module.resolution_status == "exact"
    assert by_path["namespace/tool.py"].module.resolution_status == "ambiguous"
    decorator_sites = [
        site
        for site in by_path["tests/test_core.py"].sites
        if site.kind == "decorator"
    ]
    assert decorator_sites
    assert any(
        site.decorator_chain == ["pt", "mark", "slow"]
        and site.decorator_call is False
        and site.decorator_dynamic is False
        for site in decorator_sites
    )
    assert any(
        site.decorator_chain == ["pt", "mark", "integration"]
        and site.decorator_call is True
        for site in decorator_sites
    )
    assert any(site.decorator_dynamic is True for site in decorator_sites)

    core = by_path["src/pkg/core.py"]
    kinds = {item.kind for item in core.entities}
    assert {
        "module",
        "class",
        "method",
        "async_method",
        "function",
        "async_function",
        "nested_function",
    } <= kinds
    repeated = [
        item for item in core.entities if item.qualified_name == "repeated"
    ]
    assert [item.lexical_ordinal for item in repeated] == [1, 2]
    assert len({item.entity_id for item in repeated}) == 2
    relation_kinds = {item.kind for item in core.relations}
    assert {
        "contains",
        "defines",
        "imports",
        "imports_from",
        "references",
        "inherits",
        "calls",
    } <= relation_kinds
    local_helper_imports = [
        item
        for item in core.relations
        if item.kind == "imports_from"
        and item.target.rendered
        in {"pkg.helpers:helper", "pkg.helpers:identity"}
    ]
    assert len(local_helper_imports) == 2
    assert len({item.relation_id for item in local_helper_imports}) == 2
    assert len({item.target.file_id for item in local_helper_imports}) == 1
    resolutions = {item.resolution for item in core.relations}
    assert {
        "exact_local_entity",
        "exact_local_module",
        "external",
        "ambiguous",
        "unresolved",
    } <= resolutions
    dynamic_getattr = [
        item
        for item in core.relations
        if item.kind == "calls" and item.target.rendered == "getattr"
    ]
    assert dynamic_getattr and dynamic_getattr[0].resolution == "unresolved"
    serialized_shards = "\n".join(
        shard.model_dump_json() for shard in bundle.shards
    )
    assert "not persisted in observations" not in serialized_shards
    assert "must not enter cache" not in serialized_shards

    manifest = {item.path: item for item in discovery.manifest.files}
    source = (repository / "src/pkg/core.py").read_bytes()
    assert manifest["src/pkg/core.py"].content_sha256 == core.content_sha256
    for site in core.sites:
        location = site.location
        assert location.parser_query_version == "python_snapshot:1.4"
        assert site.site_id == site_id(
            core.path,
            core.content_sha256,
            site.kind,
            location.start_line,
            location.start_utf8_byte_column,
            location.end_line,
            location.end_utf8_byte_column,
        )
        lines = source.splitlines(keepends=True)
        start = sum(map(len, lines[: location.start_line - 1])) + (
            location.start_utf8_byte_column
        )
        end = sum(map(len, lines[: location.end_line - 1])) + (
            location.end_utf8_byte_column
        )
        assert start <= end <= len(source)


def test_snapshot_admits_python_credential_shaped_content(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    reference_path = repository / "src/pkg/credential_reference.py"
    reference_path.write_text(
        "api_key = AZURE_API_KEY\n"
        "client = Client(api_key=load_api_key())\n",
        encoding="utf-8",
    )
    (repository / "src/pkg/credential_literal.py").write_text(
        'API_KEY = "abcdefghijklmnopqrstuvwxyz"\n',
        encoding="utf-8",
    )
    (repository / "src/pkg/credential_token.py").write_text(
        'token = "sk-abcdefghijklmnopqrstuvwxyz123456"\n',
        encoding="utf-8",
    )
    (repository / "src/pkg/credential_comment.py").write_text(
        '# API_KEY = "abcdefghijklmnopqrstuvwxyz"\nvalue = 1\n',
        encoding="utf-8",
    )
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    by_path = {item.path: item for item in bundle.shards}

    for path in (
        "src/pkg/credential_reference.py",
        "src/pkg/credential_literal.py",
        "src/pkg/credential_token.py",
        "src/pkg/credential_comment.py",
    ):
        assert by_path[path].parse_status == "complete"
        assert by_path[path].incomplete_reasons == []


def test_deep_python_ast_is_typed_incomplete_instead_of_crashing(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    deep_path = repository / "src/pkg/deep_attribute.py"
    deep_path.write_text(
        "value = root" + ".attribute" * 600 + "\n",
        encoding="utf-8",
    )
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    shard = next(item for item in bundle.shards if item.path == "src/pkg/deep_attribute.py")

    assert shard.parse_status == "failed"
    assert shard.incomplete_reasons == ["parser_recursionerror"]


def test_ids_have_documented_stability_contract(tmp_path: Path) -> None:
    path = "src/pkg/core.py"
    first_hash = "1" * 64
    second_hash = "2" * 64
    assert entity_id(path, "function", "target", 1) == entity_id(
        path,
        "function",
        "target",
        1,
    )
    assert file_id(path, first_hash) != file_id(path, second_hash)
    assert site_id(path, first_hash, "definition", 1, 0, 2, 4) != site_id(
        path,
        second_hash,
        "definition",
        2,
        0,
        3,
        4,
    )
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    before = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    entity_before = next(
        item
        for shard in before.shards
        if shard.path == path
        for item in shard.entities
        if item.qualified_name == "ambiguous_reference"
    )
    core_path = repository / path
    core_path.write_text(
        "\n" + core_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    changed_discovery = discover_fixture(repository, config)
    after = run_observations(
        repository,
        changed_discovery,
        config.observation,
    ).bundle
    entity_after = next(
        item
        for shard in after.shards
        if shard.path == path
        for item in shard.entities
        if item.qualified_name == "ambiguous_reference"
    )
    assert entity_before.entity_id == entity_after.entity_id
    assert entity_before.definition_site_id != entity_after.definition_site_id


def test_snapshot_upgrade_changes_parse_cache_identity_and_location_contract(
    tmp_path: Path,
) -> None:
    current_parser = parser_provenance()
    legacy_parser = current_parser.model_copy(
        update={"ast_contract_version": "stdlib-ast-structural-1.1"}
    )
    cache_kwargs = {
        "directory": None,
        "repository_root": tmp_path,
        "semantic_limits": {"max_ast_nodes_per_file": 1},
        "max_entry_bytes": 1_024,
    }
    current_cache = ParseCache(parser=current_parser, **cache_kwargs)
    legacy_cache = ParseCache(parser=legacy_parser, **cache_kwargs)

    assert AST_CONTRACT_VERSION == "stdlib-ast-structural-1.2"
    assert current_cache.key_for("0" * 64) != legacy_cache.key_for("0" * 64)

    location_payload = {
        "path": "src/example.py",
        "source_content_sha256": "1" * 64,
        "start_line": 1,
        "start_utf8_byte_column": 0,
        "end_line": 1,
        "end_utf8_byte_column": 1,
        "excerpt_sha256": "2" * 64,
        "parser_query_version": "python_snapshot:1.3",
    }
    with pytest.raises(ValueError, match="python_snapshot:1.4"):
        SourceLocation.model_validate(location_payload)


def test_exact_clones_are_generic_facts_not_norms(
    tmp_path: Path,
) -> None:
    _, _, bundle = _bundle(tmp_path)
    assert bundle.clone_clusters is not None
    by_id = {item.file_id: item.path for item in bundle.shards}
    clusters = [
        (
            {by_id[file_id] for file_id in item.member_file_ids},
            set(item.match_basis),
        )
        for item in bundle.clone_clusters.clusters
    ]
    assert (
        {"src/pkg/duplicate_a.py", "src/pkg/duplicate_b.py"},
        {"exact_content", "exact_ast"},
    ) in clusters
    assert (
        {"src/pkg/format_a.py", "src/pkg/format_b.py"},
        {"exact_ast"},
    ) in clusters
    payloads = [
        artifact.model_dump(mode="json")
        for artifact in bundle.query_artifacts().values()
    ]
    assert all(
        key not in payload
        for payload in payloads
        for key in ("rule", "claim_kind", "normative_strength")
    )
    assert "observed_convention" not in str(payloads)


def test_discovery_exclusions_are_not_rescanned_by_observations(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    config = config.model_copy(
        update={
            "discovery": config.discovery.model_copy(
                update={"exclude": ["third_party/**"]}
            )
        }
    )
    discovery = discover_fixture(repository, config)
    assert any(
        item.path == "third_party/library.py"
        for item in discovery.manifest.files
    )
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    assert "third_party/library.py" not in {
        item.path for item in bundle.shards
    }


def test_observation_never_executes_imports_or_mutates_target_tree(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    before = {
        path.relative_to(repository).as_posix(): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file()
    }
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    run_observations(repository, discovery, config.observation)
    after = {
        path.relative_to(repository).as_posix(): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (repository / "OBSERVATION_EXECUTED").exists()


def test_dependent_queries_reuse_one_shared_parse_per_unique_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    calls = 0
    original = ast.parse

    def tracked_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "repo_constraint_memory_v3.observations.python_snapshot.ast.parse",
        tracked_parse,
    )
    bundle = run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle
    unique_complete_content = {
        shard.content_sha256
        for shard in bundle.shards
        if shard.parse_status == "complete"
    }
    # Descriptor parsing remains content-deduplicated and shared by dependent
    # queries; there is no credential-specific preflight parse.
    assert calls == len(unique_complete_content)
