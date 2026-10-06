from __future__ import annotations

import json
from pathlib import Path

from repo_constraint_memory_v3.run_validation import validate_run_directory

from helpers import (
    build_fixture,
    config_for_tests,
    copied_fixture,
    copied_observation_fixture,
)


def _rewrite_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_full_observation_tree_replays_and_cold_warm_workers_match(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)
    serial = config_for_tests(observation={"max_workers": 1})
    parallel = config_for_tests(observation={"max_workers": 4})
    first = tmp_path / "first"
    second = tmp_path / "second"
    cache = tmp_path / "cache"
    build_fixture(
        repository,
        first,
        config=serial,
        observation_cache=cache,
    )
    build_fixture(
        repository,
        second,
        config=parallel,
        observation_cache=cache,
    )
    assert validate_run_directory(first, repository_root=repository) == []
    assert validate_run_directory(second, repository_root=repository) == []
    first_files = {
        path.relative_to(first / "observations").as_posix(): path.read_bytes()
        for path in (first / "observations").rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second / "observations").as_posix(): path.read_bytes()
        for path in (second / "observations").rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_tampered_shard_digest_id_location_missing_and_orphan_are_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)

    run = tmp_path / "tampered-location"
    build_fixture(repository, run)
    shard_path = next((run / "observations/python-files").glob("*.json"))
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if shard["sites"]:
        shard["sites"][0]["location"]["excerpt_sha256"] = "0" * 64
    else:
        shard["file_id"] = "PYFILE-" + "0" * 24
    _rewrite_json(shard_path, shard)
    assert validate_run_directory(run, repository_root=repository)

    missing = tmp_path / "missing"
    build_fixture(repository, missing)
    next((missing / "observations/python-files").glob("*.json")).unlink()
    assert any(
        "missing" in error
        for error in validate_run_directory(missing, repository_root=repository)
    )

    orphan = tmp_path / "orphan"
    build_fixture(repository, orphan)
    (orphan / "observations/queries/orphan.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    assert any(
        "unexpected/orphan" in error
        for error in validate_run_directory(orphan, repository_root=repository)
    )


def test_observation_coverage_tamper_and_disabled_artifact_are_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "coverage"
    build_fixture(repository, run)
    # Schema 4 keeps consumer memory deliberately narrow; observation
    # coverage remains in the auditable observation index.
    observation_index_path = run / "observations" / "index.json"
    observation_index = json.loads(
        observation_index_path.read_text(encoding="utf-8")
    )
    observation_index["observation_set_digest"] = "0" * 64
    _rewrite_json(observation_index_path, observation_index)
    errors = validate_run_directory(run, repository_root=repository)
    assert any("digest" in error or "coverage" in error for error in errors)

    disabled = config_for_tests(
        observation={"enabled": False, "queries": []},
    )
    disabled_run = tmp_path / "disabled"
    build_fixture(repository, disabled_run, config=disabled)
    (disabled_run / "observations").mkdir()
    assert any(
        "disabled" in error
        for error in validate_run_directory(
            disabled_run,
            repository_root=repository,
        )
    )


def test_query_version_digest_and_clone_membership_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_observation_fixture(tmp_path)

    version_run = tmp_path / "version"
    build_fixture(repository, version_run)
    snapshot_path = (
        version_run / "observations/queries/python_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["query_version"] = "9.9"
    _rewrite_json(snapshot_path, snapshot)
    assert validate_run_directory(version_run, repository_root=repository)

    digest_run = tmp_path / "digest"
    build_fixture(repository, digest_run)
    clone_path = (
        digest_run / "observations/queries/python_clone_clusters.json"
    )
    clones = json.loads(clone_path.read_text(encoding="utf-8"))
    clones["artifact_digest"] = "0" * 64
    _rewrite_json(clone_path, clones)
    assert validate_run_directory(digest_run, repository_root=repository)

    membership_run = tmp_path / "membership"
    build_fixture(repository, membership_run)
    clone_path = (
        membership_run / "observations/queries/python_clone_clusters.json"
    )
    clones = json.loads(clone_path.read_text(encoding="utf-8"))
    assert clones["clusters"]
    clones["clusters"][0]["member_file_ids"][0] = "PYFILE-" + "0" * 24
    _rewrite_json(clone_path, clones)
    errors = validate_run_directory(
        membership_run,
        repository_root=repository,
    )
    assert any("clone" in error or "replay" in error for error in errors)
