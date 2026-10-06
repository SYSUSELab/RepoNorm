"""Strict schemas for bounded, replayable Python repository observations."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from repo_constraint_memory_v3.models import (
    NonEmptyString,
    PositiveInt,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)

OBSERVATION_SCHEMA_VERSION = "1.0"
PYTHON_SNAPSHOT_QUERY_VERSION = "1.4"
PYTHON_CLONE_QUERY_VERSION = "1.1"

PythonFileObservationId = Annotated[
    str, Field(pattern=r"^PYFILE-[0-9a-f]{24}$")
]
PythonEntityId = Annotated[str, Field(pattern=r"^PYENT-[0-9a-f]{24}$")]
PythonSiteId = Annotated[str, Field(pattern=r"^PYSITE-[0-9a-f]{24}$")]
PythonRelationId = Annotated[str, Field(pattern=r"^PYREL-[0-9a-f]{24}$")]
PythonCloneClusterId = Annotated[
    str, Field(pattern=r"^PYCLONE-[0-9a-f]{24}$")
]
MAX_RECORDED_PYTHON_MANIFEST_GAPS = 4_096

QueryStatus = Literal["complete", "partial", "failed", "not_run"]
ParseStatus = Literal["complete", "partial", "failed", "skipped"]
PythonManifestGapReason = Literal[
    "missing",
    "symlink",
    "not_regular_file",
    "oversized_manifest_file",
    "manifest_file_count_budget",
    "manifest_total_bytes_budget",
    "read_error",
    "binary_content",
]
ResolutionStatus = Literal[
    "exact_local_entity",
    "exact_local_module",
    "external",
    "ambiguous",
    "unresolved",
]


class ParserProvenance(StrictModel):
    implementation: NonEmptyString
    python_version: Annotated[str, Field(pattern=r"^\d+\.\d+$")]
    ast_contract_version: NonEmptyString


class SourceLocation(StrictModel):
    path: Annotated[str, Field(min_length=1, max_length=4_096)]
    source_content_sha256: Sha256Hex
    start_line: PositiveInt
    start_utf8_byte_column: Annotated[int, Field(ge=0)]
    end_line: PositiveInt
    end_utf8_byte_column: Annotated[int, Field(ge=0)]
    excerpt_sha256: Sha256Hex
    parser_query_version: Literal["python_snapshot:1.4"]

    @model_validator(mode="after")
    def endpoint_order(self) -> SourceLocation:
        if (self.end_line, self.end_utf8_byte_column) < (
            self.start_line,
            self.start_utf8_byte_column,
        ):
            raise ValueError("source location end precedes start")
        return self


class FileRoleObservation(StrictModel):
    role_version: Literal["python-file-role-1.0"]
    location_role: Literal[
        "production",
        "test",
        "example",
        "migration",
        "third_party",
        "unknown",
    ]
    classification_status: Literal["exact", "ambiguous", "unknown"]
    attributes: list[Literal["generated", "deprecated"]]
    basis: list[NonEmptyString]

    @model_validator(mode="after")
    def canonical_lists(self) -> FileRoleObservation:
        if self.attributes != sorted(set(self.attributes)):
            raise ValueError("file role attributes must be unique and sorted")
        if self.basis != sorted(set(self.basis)):
            raise ValueError("file role basis must be unique and sorted")
        return self


class ModuleIdentity(StrictModel):
    module_name: str | None
    package_parts: list[NonEmptyString]
    source_root: str | None
    resolution_status: Literal["exact", "ambiguous", "unknown"]
    basis: list[NonEmptyString]


class PythonSite(StrictModel):
    site_id: PythonSiteId
    kind: Literal[
        "definition",
        "decorator",
        "import",
        "reference",
        "inheritance",
        "call",
    ]
    location: SourceLocation
    decorator_chain: list[NonEmptyString] | None = None
    decorator_symbols: list[NonEmptyString] | None = None
    decorator_call: bool | None = None
    decorator_dynamic: bool | None = None

    @model_validator(mode="after")
    def decorator_fields_are_scoped(self) -> PythonSite:
        values = (
            self.decorator_chain,
            self.decorator_symbols,
            self.decorator_call,
            self.decorator_dynamic,
        )
        if self.kind == "decorator" and any(value is None for value in values):
            raise ValueError("decorator site requires structured decorator fields")
        if self.kind != "decorator" and any(value is not None for value in values):
            raise ValueError("non-decorator site cannot carry decorator fields")
        return self


class PythonEntity(StrictModel):
    entity_id: PythonEntityId
    file_id: PythonFileObservationId
    kind: Literal[
        "module",
        "class",
        "function",
        "async_function",
        "method",
        "async_method",
        "nested_class",
        "nested_function",
        "nested_async_function",
    ]
    name: NonEmptyString
    qualified_name: NonEmptyString
    lexical_ordinal: PositiveInt
    parent_entity_id: PythonEntityId | None
    definition_site_id: PythonSiteId | None
    decorator_site_ids: list[PythonSiteId]


class PythonSymbolRef(StrictModel):
    rendered: Annotated[str, Field(min_length=1, max_length=4096)]
    entity_id: PythonEntityId | None = None
    file_id: PythonFileObservationId | None = None


class PythonRelation(StrictModel):
    relation_id: PythonRelationId
    kind: Literal[
        "contains",
        "defines",
        "imports",
        "imports_from",
        "references",
        "inherits",
        "calls",
    ]
    source: PythonSymbolRef
    target: PythonSymbolRef
    site_id: PythonSiteId | None
    resolution: ResolutionStatus
    alias: str | None = None


class PythonFileShard(StrictModel):
    schema_version: Literal["1.0"]
    file_id: PythonFileObservationId
    path: NonEmptyString
    content_sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    source_form: Literal["implementation", "stub"]
    role: FileRoleObservation
    module: ModuleIdentity
    parse_status: ParseStatus
    parser: ParserProvenance | None
    ast_fingerprint: Sha256Hex | None
    ast_node_count: Annotated[int, Field(ge=0)] | None
    entities: list[PythonEntity]
    sites: list[PythonSite]
    relations: list[PythonRelation]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    shard_digest: Sha256Hex

    @model_validator(mode="after")
    def references_and_uniqueness(self) -> PythonFileShard:
        entity_ids = [item.entity_id for item in self.entities]
        site_ids = [item.site_id for item in self.sites]
        relation_ids = [item.relation_id for item in self.relations]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Python entity IDs must be unique within a shard")
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("Python site IDs must be unique within a shard")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("Python relation IDs must be unique within a shard")
        known_entities = set(entity_ids)
        known_sites = set(site_ids)
        for entity in self.entities:
            if entity.file_id != self.file_id:
                raise ValueError("entity file_id does not match shard")
            if (
                entity.parent_entity_id is not None
                and entity.parent_entity_id not in known_entities
            ):
                raise ValueError("entity parent is not present in shard")
            if (
                entity.definition_site_id is not None
                and entity.definition_site_id not in known_sites
            ):
                raise ValueError("entity definition site is not present in shard")
            if not set(entity.decorator_site_ids) <= known_sites:
                raise ValueError("entity decorator site is not present in shard")
        for relation in self.relations:
            if relation.site_id is not None and relation.site_id not in known_sites:
                raise ValueError("relation site is not present in shard")
        if self.parse_status == "complete" and self.incomplete_reasons:
            raise ValueError("complete shard cannot have incomplete reasons")
        return self


class PythonFileShardRef(StrictModel):
    file_id: PythonFileObservationId
    path: NonEmptyString
    artifact_path: Annotated[
        str, Field(pattern=r"^observations/python-files/PYFILE-[0-9a-f]{24}\.json$")
    ]
    artifact_sha256: Sha256Hex
    parse_status: ParseStatus


class PythonManifestGap(StrictModel):
    """One safe Python path enumerated but not materialized by discovery."""

    path: NonEmptyString
    reason: PythonManifestGapReason
    reason_detail: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    component_key: NonEmptyString

    @model_validator(mode="after")
    def bounded_reason_detail(self) -> PythonManifestGap:
        pure = PurePosixPath(self.path)
        if (
            pure.is_absolute()
            or self.path != pure.as_posix()
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.suffix.casefold() not in {".py", ".pyi"}
        ):
            raise ValueError("Python manifest gap path is not a safe source path")
        if self.reason == "read_error":
            if not self.reason_detail:
                raise ValueError("read-error Python manifest gap requires detail")
        elif self.reason_detail is not None:
            raise ValueError("only read-error Python manifest gap may have detail")
        return self


class PythonSnapshotArtifact(StrictModel):
    schema_version: Literal["1.0"]
    query_name: Literal["python_snapshot"]
    query_version: Literal["1.4"]
    snapshot_digest: SnapshotDigest
    status: Literal["complete", "partial", "failed"]
    parser: ParserProvenance
    eligible_file_count: Annotated[int, Field(ge=0)]
    processed_file_count: Annotated[int, Field(ge=0)]
    skipped_file_count: Annotated[int, Field(ge=0)]
    failed_file_count: Annotated[int, Field(ge=0)]
    entity_count: Annotated[int, Field(ge=0)]
    site_count: Annotated[int, Field(ge=0)]
    relation_count: Annotated[int, Field(ge=0)]
    file_shards: list[PythonFileShardRef]
    unrepresented_python_file_count: Annotated[int, Field(ge=0)]
    unrepresented_python_files: Annotated[
        list[PythonManifestGap],
        Field(max_length=MAX_RECORDED_PYTHON_MANIFEST_GAPS),
    ]
    unrepresented_python_files_omitted_count: Annotated[int, Field(ge=0)]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def canonical_manifest_gaps(self) -> PythonSnapshotArtifact:
        ordered = sorted(
            self.unrepresented_python_files,
            key=lambda item: (item.path, item.reason, item.reason_detail or ""),
        )
        if self.unrepresented_python_files != ordered:
            raise ValueError("unrepresented Python files are not canonical")
        paths = [item.path for item in self.unrepresented_python_files]
        if len(paths) != len(set(paths)):
            raise ValueError("unrepresented Python file paths must be unique")
        if self.unrepresented_python_file_count != (
            len(self.unrepresented_python_files)
            + self.unrepresented_python_files_omitted_count
        ):
            raise ValueError("unrepresented Python file count is inconsistent")
        if self.status == "complete" and self.unrepresented_python_file_count:
            raise ValueError("complete Python snapshot cannot have manifest gaps")
        return self


class PythonCloneCluster(StrictModel):
    cluster_id: PythonCloneClusterId
    member_file_ids: Annotated[list[PythonFileObservationId], Field(min_length=2)]
    match_basis: Annotated[
        list[Literal["exact_ast", "exact_content"]], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def canonical_membership(self) -> PythonCloneCluster:
        if self.member_file_ids != sorted(set(self.member_file_ids)):
            raise ValueError("clone cluster members are not canonical")
        if self.match_basis != sorted(set(self.match_basis)):
            raise ValueError("clone cluster match basis is not canonical")
        return self


class PythonCloneClustersArtifact(StrictModel):
    schema_version: Literal["1.0"]
    query_name: Literal["python_clone_clusters"]
    query_version: Literal["1.1"]
    snapshot_digest: SnapshotDigest
    status: Literal["complete", "partial", "failed"]
    upstream_status: Literal["complete", "partial", "failed"]
    eligible_file_count: Annotated[int, Field(ge=0)]
    parsed_file_count: Annotated[int, Field(ge=0)]
    unclustered_file_count: Annotated[int, Field(ge=0)]
    excluded_or_failed_file_count: Annotated[int, Field(ge=0)]
    clusters: list[PythonCloneCluster]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def disjoint_canonical_clusters(self) -> PythonCloneClustersArtifact:
        keys = [
            (tuple(item.member_file_ids), tuple(item.match_basis))
            for item in self.clusters
        ]
        if keys != sorted(keys):
            raise ValueError("clone clusters are not canonical")
        seen: set[str] = set()
        for cluster in self.clusters:
            overlap = seen.intersection(cluster.member_file_ids)
            if overlap:
                raise ValueError("clone clusters must have disjoint membership")
            seen.update(cluster.member_file_ids)
        return self


class QueryArtifactRef(StrictModel):
    path: Annotated[
        str,
        Field(
            pattern=(
                r"^observations/queries/"
                r"(python_snapshot|python_clone_clusters)\.json$"
            )
        ),
    ]
    sha256: Sha256Hex


class ObservationQueryIndexEntry(StrictModel):
    name: Literal[
        "python_snapshot",
        "python_clone_clusters",
    ]
    version: NonEmptyString
    dependencies: list[NonEmptyString]
    required: bool
    status: QueryStatus
    artifact: QueryArtifactRef | None
    eligible_count: Annotated[int, Field(ge=0)]
    processed_count: Annotated[int, Field(ge=0)]
    skipped_count: Annotated[int, Field(ge=0)]
    failure_count: Annotated[int, Field(ge=0)]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]


class ObservationIndex(StrictModel):
    schema_version: Literal["1.0"]
    index_version: Literal["1.0"]
    snapshot_digest: SnapshotDigest
    language_scope: Literal["python"]
    parser: ParserProvenance
    query_plan: list[ObservationQueryIndexEntry]
    file_shards: list[PythonFileShardRef]
    overall: Literal["complete", "partial"]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    observation_set_digest: Sha256Hex

    @model_validator(mode="after")
    def unique_query_and_shard_refs(self) -> ObservationIndex:
        query_names = [item.name for item in self.query_plan]
        shard_ids = [item.file_id for item in self.file_shards]
        shard_paths = [item.path for item in self.file_shards]
        if len(query_names) != len(set(query_names)):
            raise ValueError("observation query names must be unique")
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("observation shard IDs must be unique")
        if len(shard_paths) != len(set(shard_paths)):
            raise ValueError("observation shard paths must be unique")
        return self


class EntityDescriptor(StrictModel):
    descriptor_index: Annotated[int, Field(ge=0)]
    parent_index: Annotated[int, Field(ge=0)] | None
    kind: NonEmptyString
    name: NonEmptyString
    qualified_name: NonEmptyString
    lexical_ordinal: PositiveInt
    start_line: PositiveInt
    start_column: Annotated[int, Field(ge=0)]
    end_line: PositiveInt
    end_column: Annotated[int, Field(ge=0)]
    decorators: list["DecoratorDescriptor"]


class DecoratorDescriptor(StrictModel):
    start_line: PositiveInt
    start_column: Annotated[int, Field(ge=0)]
    end_line: PositiveInt
    end_column: Annotated[int, Field(ge=0)]
    chain: list[NonEmptyString]
    symbols: list[NonEmptyString]
    called: bool
    dynamic: bool


class RelationDescriptor(StrictModel):
    kind: Literal["imports", "imports_from", "references", "inherits", "calls"]
    source_entity_index: Annotated[int, Field(ge=-1)]
    target: NonEmptyString
    alias: str | None
    start_line: PositiveInt
    start_column: Annotated[int, Field(ge=0)]
    end_line: PositiveInt
    end_column: Annotated[int, Field(ge=0)]
    dynamic: bool


class ParsedPythonDescriptor(StrictModel):
    content_sha256: Sha256Hex
    ast_node_count: Annotated[int, Field(ge=0)]
    ast_fingerprint: Sha256Hex
    entities: list[EntityDescriptor]
    relations: list[RelationDescriptor]


class ParseCacheEntry(StrictModel):
    schema_version: Literal["1.1"]
    cache_key: Sha256Hex
    parser: ParserProvenance
    descriptor: ParsedPythonDescriptor
    entry_digest: Sha256Hex
