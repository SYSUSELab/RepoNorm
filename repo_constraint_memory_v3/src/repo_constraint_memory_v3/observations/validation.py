"""Strict observation-tree validation and cache-free deterministic replay."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ValidationError

from repo_constraint_memory_v3.models import (
    DiscoveryArtifact,
    ObservationCoverage,
    ObservationSettings,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    canonical_json_text,
    model_payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.observations.engine import run_observations
from repo_constraint_memory_v3.observations.ids import (
    clone_cluster_id,
    entity_id,
    file_id,
    relation_id,
    site_id,
    symbol_ref_identity,
)
from repo_constraint_memory_v3.observations.models import (
    ObservationIndex,
    PythonCloneClustersArtifact,
    PythonFileShard,
    PythonSnapshotArtifact,
)
from repo_constraint_memory_v3.observations.python_snapshot import (
    safe_read_manifest_file,
    source_location,
)
from repo_constraint_memory_v3.observations.registry import resolve_query_plan

_QUERY_MODELS: dict[str, type[BaseModel]] = {
    "python_snapshot": PythonSnapshotArtifact,
    "python_clone_clusters": PythonCloneClustersArtifact,
}


def _read_model(
    path: Path,
    model: type[BaseModel],
    errors: list[str],
) -> tuple[BaseModel | None, bytes | None]:
    relative = path.as_posix()
    if path.is_symlink() or not path.is_file():
        errors.append(f"observation artifact is missing or a symlink: {relative}")
        return None, None
    try:
        content = path.read_bytes()
        value = model.model_validate_json(content)
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        errors.append(f"observation artifact schema/read error {relative}: {error}")
        return None, None
    if content != canonical_json_bytes(value):
        errors.append(f"observation artifact is not canonical JSON: {relative}")
    return value, content


def _actual_files(root: Path, errors: list[str]) -> set[str]:
    actual: set[str] = set()
    for current, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        for dirname in list(dirnames):
            child = current_path / dirname
            if child.is_symlink():
                errors.append(
                    "observations contains a symlink directory: "
                    f"{child.relative_to(root).as_posix()}"
                )
                dirnames.remove(dirname)
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append(
                    f"observations contains a symlink file: {relative}"
                )
            else:
                actual.add(f"observations/{relative}")
    return actual


def _validate_shard(
    shard: PythonFileShard,
    *,
    repository_root: Path,
    discovery: DiscoveryArtifact,
    errors: list[str],
) -> None:
    if shard.file_id != file_id(shard.path, shard.content_sha256):
        errors.append(f"{shard.path}: file ID does not replay")
    if shard.shard_digest != model_payload_digest(
        shard,
        digest_field="shard_digest",
    ):
        errors.append(f"{shard.path}: shard digest does not replay")
    manifest = {item.path: item for item in discovery.manifest.files}
    item = manifest.get(shard.path)
    if item is None:
        errors.append(f"{shard.path}: shard path is absent from discovery")
        return
    if (
        item.content_sha256 != shard.content_sha256
        or item.size_bytes != shard.size_bytes
        or item.role != "python_source"
    ):
        errors.append(f"{shard.path}: shard metadata disagrees with discovery")
        return
    content = safe_read_manifest_file(repository_root, item)
    for entity in shard.entities:
        if entity.entity_id != entity_id(
            shard.path,
            entity.kind,
            entity.qualified_name,
            entity.lexical_ordinal,
        ):
            errors.append(f"{shard.path}: entity ID does not replay")
    for site in shard.sites:
        location = site.location
        expected_site_id = site_id(
            shard.path,
            shard.content_sha256,
            site.kind,
            location.start_line,
            location.start_utf8_byte_column,
            location.end_line,
            location.end_utf8_byte_column,
        )
        if site.site_id != expected_site_id:
            errors.append(f"{shard.path}: site ID does not replay")
        try:
            replayed_location = source_location(
                path=shard.path,
                content_sha256=shard.content_sha256,
                content=content,
                start_line=location.start_line,
                start_column=location.start_utf8_byte_column,
                end_line=location.end_line,
                end_column=location.end_utf8_byte_column,
            )
        except ValueError as error:
            errors.append(f"{shard.path}: source location cannot replay: {error}")
        else:
            if replayed_location != location:
                errors.append(f"{shard.path}: source location hash does not replay")
    for relation in shard.relations:
        expected_relation_id = relation_id(
            relation.kind,
            symbol_ref_identity(
                relation.source.rendered,
                entity_id=relation.source.entity_id,
                file_id=relation.source.file_id,
            ),
            symbol_ref_identity(
                relation.target.rendered,
                entity_id=relation.target.entity_id,
                file_id=relation.target.file_id,
            ),
            relation.site_id,
        )
        if relation.relation_id != expected_relation_id:
            errors.append(f"{shard.path}: relation ID does not replay")


def validate_observation_tree(
    run_directory: Path,
    *,
    repository_root: Path,
    discovery: DiscoveryArtifact,
    settings: ObservationSettings,
    coverage: ObservationCoverage,
    semantic_units_enabled: bool = False,
) -> list[str]:
    errors: list[str] = []
    observations = run_directory / "observations"
    if observations.is_symlink() or not observations.is_dir():
        return ["observations/ is missing or a symlink"]
    index_value, index_content = _read_model(
        observations / "index.json",
        ObservationIndex,
        errors,
    )
    if index_value is None or index_content is None:
        return errors
    assert isinstance(index_value, ObservationIndex)
    index = index_value
    if index.snapshot_digest != discovery.repository.snapshot_digest:
        errors.append("observation index snapshot does not match discovery")
    if index.observation_set_digest != model_payload_digest(
        index,
        digest_field="observation_set_digest",
    ):
        errors.append("observation index digest does not replay")
    if coverage.observation_set_digest != index.observation_set_digest:
        errors.append("memory observation digest differs from observation index")
    if coverage.overall != index.overall:
        errors.append("memory observation overall differs from observation index")
    coverage_rows = [
        (
            item.name,
            item.version,
            item.required,
            item.status,
            item.incomplete_reasons,
        )
        for item in coverage.queries
    ]
    index_rows = [
        (
            item.name,
            item.version,
            item.required,
            item.status,
            item.incomplete_reasons,
        )
        for item in index.query_plan
    ]
    if coverage_rows != index_rows:
        errors.append("memory observation query coverage differs from index")
    try:
        expected_plan = resolve_query_plan(item.name for item in settings.queries)
    except ValueError as error:
        errors.append(f"observation config/registry error: {error}")
        return errors
    required_by_name = {item.name: item.required for item in settings.queries}
    expected_plan_rows = [
        (
            item.name,
            item.version,
            item.dependencies,
            required_by_name[item.name],
        )
        for item in expected_plan
    ]
    actual_plan_rows = [
        (
            item.name,
            item.version,
            tuple(item.dependencies),
            item.required,
        )
        for item in index.query_plan
    ]
    if actual_plan_rows != expected_plan_rows:
        errors.append("observation index query plan disagrees with registry/config")
    required_incomplete = any(
        item.required and item.status != "complete" for item in index.query_plan
    )
    if index.overall != ("partial" if required_incomplete else "complete"):
        errors.append("observation index overall status is inconsistent")

    artifacts: dict[str, BaseModel] = {}
    artifact_contents: dict[str, bytes] = {}
    expected_files = {"observations/index.json"}
    for entry in index.query_plan:
        if entry.artifact is None:
            if entry.status != "not_run":
                errors.append(
                    f"{entry.name}: missing artifact is not marked not_run"
                )
            continue
        expected_files.add(entry.artifact.path)
        path = run_directory / entry.artifact.path
        artifact, content = _read_model(
            path,
            _QUERY_MODELS[entry.name],
            errors,
        )
        if artifact is None or content is None:
            continue
        artifacts[entry.name] = artifact
        artifact_contents[entry.name] = content
        if sha256_hex(content) != entry.artifact.sha256:
            errors.append(f"{entry.name}: query artifact hash mismatch")
        if getattr(artifact, "snapshot_digest") != index.snapshot_digest:
            errors.append(f"{entry.name}: query artifact snapshot mismatch")
        if getattr(artifact, "status") != entry.status:
            errors.append(f"{entry.name}: query status differs from index")
        if getattr(artifact, "artifact_digest") != model_payload_digest(
            artifact,
            digest_field="artifact_digest",
        ):
            errors.append(f"{entry.name}: query artifact digest does not replay")

    shards: list[PythonFileShard] = []
    shard_contents: dict[str, bytes] = {}
    for reference in index.file_shards:
        expected_files.add(reference.artifact_path)
        expected_path = (
            f"observations/python-files/{reference.file_id}.json"
        )
        if reference.artifact_path != expected_path:
            errors.append(f"{reference.path}: shard path/file ID mismatch")
        shard_value, content = _read_model(
            run_directory / reference.artifact_path,
            PythonFileShard,
            errors,
        )
        if shard_value is None or content is None:
            continue
        assert isinstance(shard_value, PythonFileShard)
        shard = shard_value
        shards.append(shard)
        shard_contents[shard.file_id] = content
        if (
            shard.file_id != reference.file_id
            or shard.path != reference.path
            or shard.parse_status != reference.parse_status
        ):
            errors.append(f"{reference.path}: shard reference mismatch")
        if sha256_hex(content) != reference.artifact_sha256:
            errors.append(f"{reference.path}: shard artifact hash mismatch")
        _validate_shard(
            shard,
            repository_root=repository_root,
            discovery=discovery,
            errors=errors,
        )
    snapshot = artifacts.get("python_snapshot")
    if isinstance(snapshot, PythonSnapshotArtifact):
        if snapshot.file_shards != index.file_shards:
            errors.append("python snapshot shard refs differ from observation index")

    clone_artifact = artifacts.get("python_clone_clusters")
    if isinstance(clone_artifact, PythonCloneClustersArtifact):
        for cluster in clone_artifact.clusters:
            if cluster.cluster_id != clone_cluster_id(
                cluster.member_file_ids,
                cluster.match_basis,
            ):
                errors.append("clone cluster ID does not replay")
    actual_files = _actual_files(observations, errors)
    if semantic_units_enabled:
        actual_files = {
            item
            for item in actual_files
            if not item.startswith("observations/semantic-units/")
        }
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    if missing:
        errors.append(f"missing observation artifacts: {missing}")
    if extra:
        errors.append(f"unexpected/orphan observation artifacts: {extra}")

    try:
        replayed = run_observations(
            repository_root,
            discovery,
            settings,
            cache_directory=None,
        ).bundle
    except (OSError, ValueError) as error:
        errors.append(f"cannot replay observations without cache: {error}")
        return errors
    if canonical_json_text(replayed.index) != index_content.decode("utf-8"):
        errors.append("observation index does not replay without cache")
    replayed_artifacts = replayed.query_artifacts()
    if set(replayed_artifacts) != set(artifacts):
        errors.append("replayed observation query artifact set differs")
    for name in sorted(set(replayed_artifacts) & set(artifacts)):
        if canonical_json_bytes(replayed_artifacts[name]) != artifact_contents[name]:
            errors.append(f"{name}: query artifact does not replay without cache")
    replayed_shards = {item.file_id: item for item in replayed.shards}
    if set(replayed_shards) != set(shard_contents):
        errors.append("replayed observation shard set differs")
    for shard_id in sorted(set(replayed_shards) & set(shard_contents)):
        if canonical_json_bytes(replayed_shards[shard_id]) != shard_contents[shard_id]:
            errors.append(f"{shard_id}: shard does not replay without cache")
    return errors
