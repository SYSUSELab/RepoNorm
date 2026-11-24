"""Deterministic orchestration of the generic observation query plan."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from pydantic import BaseModel

from repo_constraint_memory_v3.models import (
    DiscoveryArtifact,
    ObservationSettings,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    model_payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.observations.clones import (
    observe_clone_clusters,
)
from repo_constraint_memory_v3.observations.models import (
    ObservationIndex,
    ObservationQueryIndexEntry,
    PythonCloneClustersArtifact,
    PythonFileShard,
    PythonFileShardRef,
    PythonSnapshotArtifact,
    QueryArtifactRef,
)
from repo_constraint_memory_v3.observations.python_snapshot import (
    observe_python_snapshot,
    parser_provenance,
)
from repo_constraint_memory_v3.observations.registry import (
    QuerySpec,
    resolve_query_plan,
)

if TYPE_CHECKING:
    from repo_constraint_memory_v3.semantic.queries import PopulationQueryContext


@dataclass(frozen=True)
class ObservationBundle:
    index: ObservationIndex
    snapshot: PythonSnapshotArtifact | None
    clone_clusters: PythonCloneClustersArtifact | None
    shards: tuple[PythonFileShard, ...]

    @cached_property
    def population_query_context(self) -> PopulationQueryContext:
        # The lazy graph belongs to this fixed snapshot bundle and is released
        # with it; independent runs and replay bundles never share this state.
        from repo_constraint_memory_v3.semantic.queries import PopulationQueryContext

        return PopulationQueryContext(self)

    def query_artifacts(self) -> dict[str, BaseModel]:
        candidates: dict[str, BaseModel | None] = {
            "python_snapshot": self.snapshot,
            "python_clone_clusters": self.clone_clusters,
        }
        return {
            name: artifact
            for name, artifact in candidates.items()
            if artifact is not None
        }


@dataclass(frozen=True)
class ObservationRunResult:
    bundle: ObservationBundle
    processing_metrics: dict[str, object]


def _query_path(name: str) -> str:
    return f"observations/queries/{name}.json"


def _query_ref(name: str, artifact: BaseModel) -> QueryArtifactRef:
    return QueryArtifactRef(
        path=_query_path(name),
        sha256=sha256_hex(canonical_json_bytes(artifact)),
    )


def _entry(
    spec: QuerySpec,
    *,
    required: bool,
    artifact: BaseModel | None,
) -> ObservationQueryIndexEntry:
    if artifact is None:
        return ObservationQueryIndexEntry(
            name=spec.name,  # type: ignore[arg-type]
            version=spec.version,
            dependencies=list(spec.dependencies),
            required=required,
            status="not_run",
            artifact=None,
            eligible_count=0,
            processed_count=0,
            skipped_count=0,
            failure_count=0,
            incomplete_reasons=["dependency_artifact_unavailable"],
            limitations=[],
        )
    status = getattr(artifact, "status")
    if spec.name == "python_snapshot":
        eligible = artifact.eligible_file_count
        processed = artifact.processed_file_count
        skipped = artifact.skipped_file_count
        failures = artifact.failed_file_count
    elif spec.name == "python_clone_clusters":
        eligible = artifact.eligible_file_count
        processed = artifact.parsed_file_count
        skipped = artifact.excluded_or_failed_file_count
        failures = 0
    else:
        raise ValueError(f"unsupported observation query: {spec.name}")
    return ObservationQueryIndexEntry(
        name=spec.name,  # type: ignore[arg-type]
        version=spec.version,
        dependencies=list(spec.dependencies),
        required=required,
        status=status,
        artifact=_query_ref(spec.name, artifact),
        eligible_count=eligible,
        processed_count=processed,
        skipped_count=skipped,
        failure_count=failures,
        incomplete_reasons=list(artifact.incomplete_reasons),
        limitations=list(artifact.limitations),
    )


def run_observations(
    repository_root: Path,
    discovery: DiscoveryArtifact,
    settings: ObservationSettings,
    *,
    cache_directory: Path | None = None,
) -> ObservationRunResult:
    """Execute one validated, static observation plan against one snapshot."""

    required_by_name = {item.name: item.required for item in settings.queries}
    plan = resolve_query_plan(required_by_name)
    snapshot: PythonSnapshotArtifact | None = None
    shards: tuple[PythonFileShard, ...] = ()
    clones: PythonCloneClustersArtifact | None = None
    cache_metrics: dict[str, int] = {
        "hits": 0,
        "misses": 0,
        "rejected": 0,
        "writes": 0,
    }
    observed_workers = 0

    for spec in plan:
        if spec.name == "python_snapshot":
            built = observe_python_snapshot(
                repository_root,
                discovery,
                settings,
                cache_directory=cache_directory,
            )
            shards = built.shards
            refs = [
                PythonFileShardRef(
                    file_id=shard.file_id,
                    path=shard.path,
                    artifact_path=(
                        f"observations/python-files/{shard.file_id}.json"
                    ),
                    artifact_sha256=sha256_hex(canonical_json_bytes(shard)),
                    parse_status=shard.parse_status,
                )
                for shard in shards
            ]
            provisional = built.artifact.model_copy(
                update={
                    "file_shards": refs,
                    "artifact_digest": "0" * 64,
                }
            )
            snapshot = provisional.model_copy(
                update={
                    "artifact_digest": model_payload_digest(
                        provisional,
                        digest_field="artifact_digest",
                    )
                }
            )
            cache_metrics = built.cache_metrics
            observed_workers = built.observed_max_workers
        elif snapshot is None or snapshot.status == "failed":
            continue
        elif spec.name == "python_clone_clusters":
            clones = observe_clone_clusters(snapshot, shards)

    artifact_by_name: Mapping[str, BaseModel | None] = {
        "python_snapshot": snapshot,
        "python_clone_clusters": clones,
    }
    query_entries = [
        _entry(
            spec,
            required=required_by_name[spec.name],
            artifact=artifact_by_name[spec.name],
        )
        for spec in plan
    ]
    required_incomplete = [
        entry
        for entry in query_entries
        if entry.required and entry.status != "complete"
    ]
    provisional_index = ObservationIndex(
        schema_version="1.0",
        index_version="1.0",
        snapshot_digest=discovery.repository.snapshot_digest,
        language_scope="python",
        parser=parser_provenance(),
        query_plan=query_entries,
        file_shards=list(snapshot.file_shards) if snapshot else [],
        overall="partial" if required_incomplete else "complete",
        incomplete_reasons=sorted(
            {
                f"observation:{entry.name}:{reason}"
                for entry in required_incomplete
                for reason in (
                    entry.incomplete_reasons or [entry.status]
                )
            }
        ),
        limitations=sorted(
            {
                limitation
                for entry in query_entries
                for limitation in entry.limitations
            }
        ),
        observation_set_digest="0" * 64,
    )
    index = provisional_index.model_copy(
        update={
            "observation_set_digest": model_payload_digest(
                provisional_index,
                digest_field="observation_set_digest",
            )
        }
    )
    bundle = ObservationBundle(
        index=index,
        snapshot=snapshot,
        clone_clusters=clones,
        shards=shards,
    )
    return ObservationRunResult(
        bundle=bundle,
        processing_metrics={
            "cache": cache_metrics,
            "observed_max_workers": observed_workers,
            "query_count": len(query_entries),
            "file_role_counts": dict(
                sorted(
                    {
                        role: sum(
                            shard.role.location_role == role
                            for shard in shards
                        )
                        for role in {
                            shard.role.location_role for shard in shards
                        }
                    }.items()
                )
            ),
            "entity_count": sum(len(shard.entities) for shard in shards),
            "site_count": sum(len(shard.sites) for shard in shards),
            "relation_count": sum(len(shard.relations) for shard in shards),
        },
    )
