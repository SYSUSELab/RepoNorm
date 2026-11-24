"""Deterministic semantic-unit ownership, segmentation, and packet manifests."""

from __future__ import annotations

import hashlib
import io
import json
import token
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import Field, ValidationError, model_validator

from repo_constraint_memory_v3.models import (
    NonEmptyString,
    SemanticSettings,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    model_payload_digest,
    payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.observations.models import (
    PythonEntity,
    PythonFileShard,
    SourceLocation,
)
from repo_constraint_memory_v3.observations.python_snapshot import (
    safe_read_manifest_file,
)
from repo_constraint_memory_v3.models import DiscoveryArtifact
from repo_constraint_memory_v3.selection import (
    FAIR_SELECTION_POLICY_VERSION,
    component_fair_stable_select,
)

SEMANTIC_UNIT_CONTRACT_VERSION = "semantic-unit-1.0"
PACKET_CONTRACT_VERSION = "semantic-packet-1.0"
SEMANTIC_UNIT_INDEX_VERSION = "1.1"
SEMANTIC_UNIT_SELECTION_VERSION = (
    FAIR_SELECTION_POLICY_VERSION + "+parent-closure-1.1"
)

SemanticUnitId = Annotated[str, Field(pattern=r"^SEMUNIT-[0-9a-f]{24}$")]
SemanticSegmentId = Annotated[str, Field(pattern=r"^SEMSEG-[0-9a-f]{24}$")]
SemanticRangeId = Annotated[str, Field(pattern=r"^SEMRANGE-[0-9a-f]{24}$")]
PacketId = Annotated[str, Field(pattern=r"^PACKET-[0-9a-f]{24}$")]


class SemanticSourceRange(StrictModel):
    range_id: SemanticRangeId
    start_utf8_byte: int = Field(ge=0)
    end_utf8_byte: int = Field(ge=0)
    content_sha256: Sha256Hex

    @model_validator(mode="after")
    def ordered(self) -> SemanticSourceRange:
        if self.end_utf8_byte <= self.start_utf8_byte:
            raise ValueError("semantic source range must be non-empty")
        return self


class SemanticInputSegment(StrictModel):
    segment_id: SemanticSegmentId
    primary_unit_id: SemanticUnitId
    ordinal: int = Field(ge=0)
    source_ranges: list[SemanticSourceRange]
    source_bytes: int = Field(ge=1)
    segment_digest: Sha256Hex

    @model_validator(mode="after")
    def canonical(self) -> SemanticInputSegment:
        if self.source_bytes != sum(
            item.end_utf8_byte - item.start_utf8_byte
            for item in self.source_ranges
        ):
            raise ValueError("semantic segment byte count is inconsistent")
        if self.segment_digest != model_payload_digest(
            self, digest_field="segment_digest"
        ):
            raise ValueError("incorrect semantic segment digest")
        return self


class PacketContextRef(StrictModel):
    kind: Literal[
        "file",
        "entity",
        "site",
        "relation",
        "clone",
        "parent",
        "child",
    ]
    stable_id: NonEmptyString


class SemanticPacketManifest(StrictModel):
    packet_id: PacketId
    segment_id: SemanticSegmentId
    primary_unit_id: SemanticUnitId
    input_range_ids: list[SemanticRangeId]
    context_refs: list[PacketContextRef]
    reconstructed_bytes: int = Field(ge=1)
    packet_digest: Sha256Hex
    incomplete_reasons: list[NonEmptyString]


class SemanticUnit(StrictModel):
    contract_version: Literal["semantic-unit-1.0"]
    snapshot_digest: SnapshotDigest
    unit_id: SemanticUnitId
    unit_kind: Literal[
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
    input_status: Literal["ready", "no_input", "partial", "failed"]
    file_id: NonEmptyString
    entity_id: NonEmptyString
    parent_unit_id: SemanticUnitId | None
    path: NonEmptyString
    role: NonEmptyString
    source_form: Literal["implementation", "stub"]
    module_name: str | None
    component_key: NonEmptyString
    owned_source_ranges: list[SemanticSourceRange]
    definition_site_id: str | None
    decorator_site_ids: list[str]
    parent_entity_id: str | None
    child_entity_ids: list[str]
    context_refs: list[PacketContextRef]
    clone_group_id: str | None
    segments: list[SemanticInputSegment]
    packets: list[SemanticPacketManifest]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def invariant_and_digest(self) -> SemanticUnit:
        if self.input_status == "ready" and (
            not self.segments
            or len(self.segments) != len(self.packets)
            or self.incomplete_reasons
        ):
            raise ValueError("ready semantic unit has incomplete inputs")
        if self.input_status == "no_input" and (
            self.segments or self.packets or self.incomplete_reasons
        ):
            raise ValueError("no-input semantic unit cannot carry packets or failures")
        if self.input_status in {"partial", "failed"} and not self.incomplete_reasons:
            raise ValueError("incomplete semantic unit requires a reason")
        if [item.ordinal for item in self.segments] != list(
            range(len(self.segments))
        ):
            raise ValueError("semantic input segments are not ordered")
        if [item.segment_id for item in self.segments] != [
            item.segment_id for item in self.packets
        ]:
            raise ValueError("semantic packets do not cover segments in order")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect semantic-unit artifact digest")
        return self


class SemanticFileDependency(StrictModel):
    file_id: NonEmptyString
    path: NonEmptyString
    parse_status: Literal["complete", "partial", "failed", "skipped"]
    eligible_primary_count: int = Field(ge=0)
    materialized_primary_count: int = Field(ge=0)
    status: Literal["complete", "partial", "failed"]
    reasons: list[NonEmptyString]


class SemanticUnitRef(StrictModel):
    unit_id: SemanticUnitId
    artifact_path: str = Field(
        pattern=r"^observations/semantic-units/units/SEMUNIT-[0-9a-f]{24}\.json$"
    )
    artifact_sha256: Sha256Hex
    artifact_digest: Sha256Hex
    input_status: Literal["ready", "no_input", "partial", "failed"]


class SemanticUnitIndex(StrictModel):
    index_version: Literal["1.1"]
    contract_version: Literal["semantic-unit-1.0"]
    packet_contract_version: Literal["semantic-packet-1.0"]
    snapshot_digest: SnapshotDigest
    observation_set_digest: Sha256Hex
    required: bool
    status: Literal["complete", "partial", "failed"]
    eligible_primary_universe_count: int = Field(ge=0)
    eligible_primary_universe_digest: Sha256Hex
    selection_policy_version: Literal[
        "component-fair-stable-hash-1.0+parent-closure-1.1"
    ]
    selection_seed_digest: Sha256Hex
    selected_primary_digest: Sha256Hex
    omitted_primary_digest: Sha256Hex
    component_universe_counts: dict[NonEmptyString, int]
    component_selected_counts: dict[NonEmptyString, int]
    planned_primary_count: int = Field(ge=0)
    materialized_primary_count: int = Field(ge=0)
    input_segment_count: int = Field(ge=0)
    packet_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    no_input_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    total_owned_source_bytes: int = Field(ge=0)
    total_packet_bytes: int = Field(ge=0)
    file_dependencies: list[SemanticFileDependency]
    units: list[SemanticUnitRef]
    unit_set_digest: Sha256Hex
    packet_set_digest: Sha256Hex
    resource_use: dict[str, int]
    incomplete_reasons: list[NonEmptyString]
    limitations: list[NonEmptyString]
    index_digest: Sha256Hex

    @model_validator(mode="after")
    def algebra_and_digest(self) -> SemanticUnitIndex:
        if self.planned_primary_count != self.eligible_primary_universe_count:
            raise ValueError("planned semantic units do not equal eligible universe")
        if self.materialized_primary_count + self.omitted_count != (
            self.planned_primary_count
        ):
            raise ValueError("materialized plus omitted semantic units is inconsistent")
        if sum(self.component_universe_counts.values()) != (
            self.eligible_primary_universe_count
        ):
            raise ValueError("semantic-unit component universe counts differ")
        if sum(self.component_selected_counts.values()) != (
            self.materialized_primary_count
        ):
            raise ValueError("semantic-unit component selection counts differ")
        if set(self.component_selected_counts) != set(
            self.component_universe_counts
        ) or any(
            self.component_selected_counts[key]
            > self.component_universe_counts[key]
            for key in self.component_universe_counts
        ):
            raise ValueError("semantic-unit component selection ledger is invalid")
        if (
            self.ready_count
            + self.no_input_count
            + self.partial_count
            + self.failed_count
            != self.materialized_primary_count
        ):
            raise ValueError("semantic-unit status counts are inconsistent")
        if self.input_segment_count != self.packet_count:
            raise ValueError("every semantic input segment requires one packet")
        if self.status == "complete" and (
            self.partial_count
            or self.failed_count
            or self.omitted_count
            or self.incomplete_reasons
        ):
            raise ValueError("complete semantic-unit index has incomplete work")
        if self.index_digest != model_payload_digest(
            self, digest_field="index_digest"
        ):
            raise ValueError("incorrect semantic-unit index digest")
        return self


@dataclass(frozen=True)
class SemanticUnitBundle:
    index: SemanticUnitIndex
    units: tuple[SemanticUnit, ...]


def _stable_id(prefix: str, payload: object) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _line_starts(content: bytes) -> list[int]:
    starts = [0]
    for index, value in enumerate(content):
        if value == 10:
            starts.append(index + 1)
    return starts


def _offset(location: SourceLocation, starts: Sequence[int], *, end: bool) -> int:
    line = location.end_line if end else location.start_line
    column = (
        location.end_utf8_byte_column
        if end
        else location.start_utf8_byte_column
    )
    if line - 1 >= len(starts):
        raise ValueError("source location line is outside file content")
    return starts[line - 1] + column


def _entity_envelopes(
    shard: PythonFileShard,
    content: bytes,
) -> dict[str, tuple[int, int]]:
    sites = {item.site_id: item for item in shard.sites}
    starts = _line_starts(content)
    envelopes: dict[str, tuple[int, int]] = {}
    for entity in shard.entities:
        if entity.kind == "module":
            envelopes[entity.entity_id] = (0, len(content))
            continue
        if entity.definition_site_id is None:
            continue
        definition = sites[entity.definition_site_id].location
        start = _offset(definition, starts, end=False)
        end = _offset(definition, starts, end=True)
        for site_id in entity.decorator_site_ids:
            decorator = sites[site_id].location
            # AST decorator locations begin after the physical ``@`` token.
            # Ownership begins at the decorator's line start so neither the
            # sigil nor indentation leaks into the lexical parent.
            start = min(start, starts[decorator.start_line - 1])
        if not (0 <= start < end <= len(content)):
            raise ValueError("entity envelope is outside source content")
        envelopes[entity.entity_id] = (start, end)
    return envelopes


def _subtract(
    envelope: tuple[int, int],
    children: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    cursor, end = envelope
    result: list[tuple[int, int]] = []
    for child_start, child_end in sorted(children):
        if child_start < cursor or child_end > end or child_end <= child_start:
            raise ValueError("child entity envelope is not properly nested")
        if cursor < child_start:
            result.append((cursor, child_start))
        cursor = child_end
    if cursor < end:
        result.append((cursor, end))
    return result


def _range(start: int, end: int, content: bytes) -> SemanticSourceRange:
    return SemanticSourceRange(
        range_id=_stable_id("SEMRANGE", [start, end, sha256_hex(content[start:end])]),
        start_utf8_byte=start,
        end_utf8_byte=end,
        content_sha256=sha256_hex(content[start:end]),
    )


def _has_non_trivia(content: bytes, ranges: Sequence[SemanticSourceRange]) -> bool:
    for item in ranges:
        fragment = content[item.start_utf8_byte : item.end_utf8_byte]
        try:
            tokens = tokenize.tokenize(io.BytesIO(fragment).readline)
            if any(
                tok.type
                not in {
                    token.ENDMARKER,
                    token.ENCODING,
                    token.NEWLINE,
                    token.INDENT,
                    token.DEDENT,
                    tokenize.NL,
                    tokenize.COMMENT,
                }
                for tok in tokens
            ):
                return True
        except (IndentationError, SyntaxError, tokenize.TokenError):
            if any(
                line.strip() and not line.lstrip().startswith(b"#")
                for line in fragment.splitlines()
            ):
                return True
    return False


def _split_ranges(
    content: bytes,
    owned: Sequence[SemanticSourceRange],
    max_bytes: int,
) -> tuple[list[list[SemanticSourceRange]], list[str]]:
    pieces: list[SemanticSourceRange] = []
    reasons: list[str] = []
    for item in owned:
        start = item.start_utf8_byte
        end = item.end_utf8_byte
        cursor = start
        while cursor < end:
            budget_end = min(cursor + max_bytes, end)
            if budget_end < end:
                newline = content.rfind(b"\n", cursor, budget_end)
                if newline < cursor:
                    line_end = content.find(b"\n", cursor, end)
                    line_end = end if line_end < 0 else line_end + 1
                    if line_end - cursor > max_bytes:
                        reasons.append(
                            f"unsplittable_physical_line:{cursor}:{line_end}"
                        )
                        cursor = line_end
                        continue
                    budget_end = line_end
                else:
                    budget_end = newline + 1
            pieces.append(_range(cursor, budget_end, content))
            cursor = budget_end
    segments: list[list[SemanticSourceRange]] = []
    current: list[SemanticSourceRange] = []
    current_bytes = 0
    for piece in pieces:
        size = piece.end_utf8_byte - piece.start_utf8_byte
        if current and current_bytes + size > max_bytes:
            segments.append(current)
            current = []
            current_bytes = 0
        current.append(piece)
        current_bytes += size
    if current:
        segments.append(current)
    return segments, reasons


def _component_key(shard: PythonFileShard) -> str:
    if shard.module.resolution_status == "exact" and shard.module.package_parts:
        return f"module:{shard.module.package_parts[0]}"
    parts = PurePosixPath(shard.path).parts
    if parts:
        return f"path:{parts[0]}"
    return "unknown"


def _context_refs(
    shard: PythonFileShard,
    entity: PythonEntity,
    children: Sequence[PythonEntity],
    clone_id: str | None,
) -> list[PacketContextRef]:
    refs: set[tuple[str, str]] = {("file", shard.file_id)}
    refs.add(("entity", entity.entity_id))
    if entity.parent_entity_id is not None:
        refs.add(("parent", entity.parent_entity_id))
    refs.update(("child", item.entity_id) for item in children)
    if entity.definition_site_id is not None:
        refs.add(("site", entity.definition_site_id))
    refs.update(("site", item) for item in entity.decorator_site_ids)
    for relation in shard.relations:
        if relation.source.entity_id == entity.entity_id:
            refs.add(("relation", relation.relation_id))
            if relation.site_id is not None:
                refs.add(("site", relation.site_id))
            if relation.target.entity_id is not None:
                refs.add(("entity", relation.target.entity_id))
    if clone_id is not None:
        refs.add(("clone", clone_id))
    return [
        PacketContextRef(kind=kind, stable_id=stable_id)  # type: ignore[arg-type]
        for kind, stable_id in sorted(refs)
    ]


def reconstruct_packet_bytes(
    packet: SemanticPacketManifest,
    *,
    unit: SemanticUnit,
    content: bytes,
) -> bytes:
    ranges = {
        item.range_id: item
        for segment in unit.segments
        for item in segment.source_ranges
    }
    payload = {
        "protocol": PACKET_CONTRACT_VERSION,
        "repository_text_trust": "untrusted_data",
        "primary_unit_id": unit.unit_id,
        "segment_id": packet.segment_id,
        "path": unit.path,
        "component_key": unit.component_key,
        "source": [
            {
                "range_id": range_id,
                "text": content[
                    ranges[range_id].start_utf8_byte :
                    ranges[range_id].end_utf8_byte
                ].decode("utf-8", errors="strict"),
            }
            for range_id in packet.input_range_ids
        ],
        "context_refs": [
            item.model_dump(mode="json") for item in packet.context_refs
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _segments_and_packets(
    *,
    unit_id: str,
    path: str,
    component_key: str,
    content: bytes,
    owned: Sequence[SemanticSourceRange],
    contexts: Sequence[PacketContextRef],
    settings: SemanticSettings,
    remaining_packet_bytes: int,
) -> tuple[
    list[SemanticInputSegment],
    list[SemanticPacketManifest],
    list[str],
    int,
]:
    grouped, reasons = _split_ranges(
        content, owned, settings.max_unit_source_bytes
    )
    segments: list[SemanticInputSegment] = []
    packets: list[SemanticPacketManifest] = []
    used = 0
    for ordinal, ranges in enumerate(grouped):
        segment_payload: dict[str, object] = {
            "segment_id": _stable_id(
                "SEMSEG",
                [
                    unit_id,
                    ordinal,
                    [item.range_id for item in ranges],
                ],
            ),
            "primary_unit_id": unit_id,
            "ordinal": ordinal,
            "source_ranges": [item.model_dump(mode="json") for item in ranges],
            "source_bytes": sum(
                item.end_utf8_byte - item.start_utf8_byte for item in ranges
            ),
        }
        segment_payload["segment_digest"] = payload_digest(segment_payload)
        segment = SemanticInputSegment.model_validate(segment_payload)
        selected = list(contexts[: settings.max_context_refs_per_packet])
        if len(selected) != len(contexts):
            reasons.append(f"context_limit:{segment.segment_id}")
        provisional = SemanticPacketManifest(
            packet_id=_stable_id(
                "PACKET",
                [
                    segment.segment_id,
                    [item.model_dump(mode="json") for item in selected],
                ],
            ),
            segment_id=segment.segment_id,
            primary_unit_id=unit_id,
            input_range_ids=[item.range_id for item in ranges],
            context_refs=selected,
            reconstructed_bytes=1,
            packet_digest="0" * 64,
            incomplete_reasons=[],
        )
        holder = SemanticUnit.model_construct(
            unit_id=unit_id,
            path=path,
            component_key=component_key,
            segments=[segment],
        )
        while True:
            raw = reconstruct_packet_bytes(
                provisional, unit=holder, content=content
            )
            if len(raw) <= settings.max_packet_bytes or not selected:
                break
            selected.pop()
            reasons.append(f"packet_context_budget:{segment.segment_id}")
            provisional = provisional.model_copy(
                update={
                    "packet_id": _stable_id(
                        "PACKET",
                        [
                            segment.segment_id,
                            [
                                item.model_dump(mode="json")
                                for item in selected
                            ],
                        ],
                    ),
                    "context_refs": selected,
                }
            )
        if len(raw) > settings.max_packet_bytes:
            reasons.append(f"packet_unsplittable:{segment.segment_id}")
            continue
        if used + len(raw) > remaining_packet_bytes:
            reasons.append(f"max_total_packet_bytes:{segment.segment_id}")
            continue
        packet = provisional.model_copy(
            update={
                "reconstructed_bytes": len(raw),
                "packet_digest": sha256_hex(raw),
                "incomplete_reasons": sorted(
                    reason
                    for reason in reasons
                    if segment.segment_id in reason
                ),
            }
        )
        segments.append(segment)
        packets.append(packet)
        used += len(raw)
    return segments, packets, sorted(set(reasons)), used


def build_semantic_units(
    repository_root: Path,
    discovery: DiscoveryArtifact,
    observations: ObservationBundle,
    settings: SemanticSettings,
) -> SemanticUnitBundle:
    """Build the complete eligible universe and bounded canonical unit plan."""

    root = repository_root.resolve()
    manifest_by_path = {item.path: item for item in discovery.manifest.files}
    clone_by_file = {
        member: cluster.cluster_id
        for cluster in (
            observations.clone_clusters.clusters
            if observations.clone_clusters is not None
            else []
        )
        for member in cluster.member_file_ids
    }
    eligible_specs: list[
        tuple[PythonFileShard, PythonEntity, bytes, list[SemanticSourceRange]]
    ] = []
    dependencies: list[SemanticFileDependency] = []
    eligible_count_by_file: dict[str, int] = defaultdict(int)
    preliminary_errors: dict[str, list[str]] = defaultdict(list)
    for shard in sorted(observations.shards, key=lambda item: item.path):
        if shard.parse_status != "complete":
            dependencies.append(
                SemanticFileDependency(
                    file_id=shard.file_id,
                    path=shard.path,
                    parse_status=shard.parse_status,
                    eligible_primary_count=0,
                    materialized_primary_count=0,
                    status=(
                        "failed" if shard.parse_status == "failed" else "partial"
                    ),
                    reasons=list(shard.incomplete_reasons),
                )
            )
            continue
        item = manifest_by_path.get(shard.path)
        if item is None:
            preliminary_errors[shard.file_id].append("manifest_file_missing")
            continue
        try:
            content = safe_read_manifest_file(root, item)
            envelopes = _entity_envelopes(shard, content)
        except (OSError, ValueError) as error:
            preliminary_errors[shard.file_id].append(
                f"source_reconstruction:{type(error).__name__}"
            )
            continue
        children_by_parent: dict[str, list[PythonEntity]] = defaultdict(list)
        for entity in shard.entities:
            if entity.parent_entity_id is not None:
                children_by_parent[entity.parent_entity_id].append(entity)
        for entity in sorted(
            shard.entities, key=lambda value: value.lexical_ordinal
        ):
            envelope = envelopes.get(entity.entity_id)
            if envelope is None:
                preliminary_errors[shard.file_id].append(
                    f"entity_envelope_missing:{entity.entity_id}"
                )
                continue
            child_envelopes = [
                envelopes[child.entity_id]
                for child in children_by_parent[entity.entity_id]
                if child.entity_id in envelopes
            ]
            try:
                owned = [
                    _range(start, end, content)
                    for start, end in _subtract(envelope, child_envelopes)
                    if end > start
                ]
            except ValueError as error:
                preliminary_errors[shard.file_id].append(
                    f"ownership:{entity.entity_id}:{safe_reason(error)}"
                )
                continue
            eligible_specs.append((shard, entity, content, owned))
            eligible_count_by_file[shard.file_id] += 1

    universe_rows = [
        {
            "file_id": shard.file_id,
            "entity_id": entity.entity_id,
            "path": shard.path,
            "kind": entity.kind,
            "owned": [
                [item.start_utf8_byte, item.end_utf8_byte, item.content_sha256]
                for item in owned
            ],
        }
        for shard, entity, _, owned in eligible_specs
    ]
    universe_digest = payload_digest({"eligible_primary_units": universe_rows})

    # Freeze a hash rank inside every component, then select one ready entity
    # from each component per round.  "Ready" means that its eligible parent
    # has already been selected.  This topological round-robin preserves the
    # parent closure without allowing one child's multi-ancestor closure to
    # consume another component's first-round allocation.  The final
    # materialization order remains canonical source order below.
    fair_order = component_fair_stable_select(
        eligible_specs,
        limit=len(eligible_specs),
        seed={
            "snapshot_digest": discovery.repository.snapshot_digest,
            "purpose": "semantic_unit_selection",
            "selection_policy": SEMANTIC_UNIT_SELECTION_VERSION,
        },
        key=lambda item: item[1].entity_id,
        component=lambda item: _component_key(item[0]),
    )
    spec_by_entity = {item[1].entity_id: item for item in eligible_specs}
    fair_rank = {
        item[1].entity_id: ordinal
        for ordinal, item in enumerate(fair_order.selected)
    }
    specs_by_component: dict[
        str,
        list[tuple[PythonFileShard, PythonEntity, bytes, list[SemanticSourceRange]]],
    ] = defaultdict(list)
    for item in eligible_specs:
        specs_by_component[_component_key(item[0])].append(item)
    for items in specs_by_component.values():
        items.sort(key=lambda item: fair_rank[item[1].entity_id])

    selected_entity_ids: set[str] = set()
    selection_limit = min(settings.max_units, len(eligible_specs))
    while len(selected_entity_ids) < selection_limit:
        advanced = False
        for component_key in sorted(specs_by_component):
            ready: PythonEntity | None = None
            for _, entity, _, _ in specs_by_component[component_key]:
                if entity.entity_id in selected_entity_ids:
                    continue
                parent_spec = spec_by_entity.get(entity.parent_entity_id or "")
                if (
                    parent_spec is None
                    or parent_spec[1].entity_id in selected_entity_ids
                ):
                    ready = entity
                    break
            if ready is None:
                continue
            selected_entity_ids.add(ready.entity_id)
            advanced = True
            if len(selected_entity_ids) == selection_limit:
                break
        if not advanced:
            raise ValueError("semantic-unit parent graph contains a cycle")
    materialized_specs = [
        item for item in eligible_specs if item[1].entity_id in selected_entity_ids
    ]
    omitted_specs = [
        item for item in eligible_specs if item[1].entity_id not in selected_entity_ids
    ]
    omitted = len(omitted_specs)
    selected_primary_digest = payload_digest(
        {"selected_entity_ids": sorted(selected_entity_ids)}
    )
    omitted_primary_digest = payload_digest(
        {
            "omitted_entity_ids": sorted(
                item[1].entity_id for item in omitted_specs
            )
        }
    )
    component_universe_counts = dict(
        sorted(
            Counter(_component_key(item[0]) for item in eligible_specs).items()
        )
    )
    component_selected_counts = {
        component: sum(
            _component_key(item[0]) == component for item in materialized_specs
        )
        for component in component_universe_counts
    }
    unit_id_by_entity = {
        entity.entity_id: _stable_id(
            "SEMUNIT",
            [
                discovery.repository.snapshot_digest,
                shard.file_id,
                entity.entity_id,
            ],
        )
        for shard, entity, _, _ in eligible_specs
    }
    units: list[SemanticUnit] = []
    total_packet_bytes = 0
    for shard, entity, content, owned in materialized_specs:
        unit_id = unit_id_by_entity[entity.entity_id]
        children = sorted(
            (
                item
                for item in shard.entities
                if item.parent_entity_id == entity.entity_id
            ),
            key=lambda item: item.lexical_ordinal,
        )
        contexts = _context_refs(
            shard, entity, children, clone_by_file.get(shard.file_id)
        )
        reasons: list[str] = []
        segments: list[SemanticInputSegment] = []
        packets: list[SemanticPacketManifest] = []
        if _has_non_trivia(content, owned):
            segments, packets, reasons, used = _segments_and_packets(
                unit_id=unit_id,
                path=shard.path,
                component_key=_component_key(shard),
                content=content,
                owned=owned,
                contexts=contexts,
                settings=settings,
                remaining_packet_bytes=max(
                    settings.max_total_packet_bytes - total_packet_bytes, 0
                ),
            )
            total_packet_bytes += used
            status = "ready" if not reasons else "partial"
            if not segments:
                status = "failed"
                reasons.append("no_reconstructable_packet")
        else:
            status = "no_input"
        payload: dict[str, object] = {
            "contract_version": SEMANTIC_UNIT_CONTRACT_VERSION,
            "snapshot_digest": discovery.repository.snapshot_digest,
            "unit_id": unit_id,
            "unit_kind": entity.kind,
            "input_status": status,
            "file_id": shard.file_id,
            "entity_id": entity.entity_id,
            "parent_unit_id": (
                unit_id_by_entity.get(entity.parent_entity_id)
                if entity.parent_entity_id is not None
                else None
            ),
            "path": shard.path,
            "role": shard.role.location_role,
            "source_form": shard.source_form,
            "module_name": shard.module.module_name,
            "component_key": _component_key(shard),
            "owned_source_ranges": [
                item.model_dump(mode="json") for item in owned
            ],
            "definition_site_id": entity.definition_site_id,
            "decorator_site_ids": list(entity.decorator_site_ids),
            "parent_entity_id": entity.parent_entity_id,
            "child_entity_ids": [item.entity_id for item in children],
            "context_refs": [item.model_dump(mode="json") for item in contexts],
            "clone_group_id": clone_by_file.get(shard.file_id),
            "segments": [item.model_dump(mode="json") for item in segments],
            "packets": [item.model_dump(mode="json") for item in packets],
            "incomplete_reasons": sorted(set(reasons)),
            "limitations": [
                "packets_contain_structural_context_only",
                "semantic_packets_feed_current_source_map",
            ],
        }
        payload["artifact_digest"] = payload_digest(payload)
        units.append(SemanticUnit.model_validate(payload))

    materialized_by_file: dict[str, int] = defaultdict(int)
    for unit in units:
        materialized_by_file[unit.file_id] += 1
    existing_dependency_ids = {item.file_id for item in dependencies}
    for shard in sorted(observations.shards, key=lambda item: item.path):
        if shard.file_id in existing_dependency_ids:
            continue
        reasons = preliminary_errors.get(shard.file_id, [])
        eligible = eligible_count_by_file.get(shard.file_id, 0)
        materialized = materialized_by_file.get(shard.file_id, 0)
        status: Literal["complete", "partial", "failed"]
        if reasons and not eligible:
            status = "failed"
        elif reasons or materialized != eligible:
            status = "partial"
        else:
            status = "complete"
        dependencies.append(
            SemanticFileDependency(
                file_id=shard.file_id,
                path=shard.path,
                parse_status=shard.parse_status,
                eligible_primary_count=eligible,
                materialized_primary_count=materialized,
                status=status,
                reasons=sorted(set(reasons)),
            )
        )
    refs = [
        SemanticUnitRef(
            unit_id=unit.unit_id,
            artifact_path=(
                f"observations/semantic-units/units/{unit.unit_id}.json"
            ),
            artifact_sha256=sha256_hex(canonical_json_bytes(unit)),
            artifact_digest=unit.artifact_digest,
            input_status=unit.input_status,
        )
        for unit in sorted(units, key=lambda item: item.unit_id)
    ]
    incomplete = {
        reason
        for unit in units
        for reason in unit.incomplete_reasons
    }
    incomplete.update(
        reason
        for dependency in dependencies
        if dependency.status != "complete"
        for reason in (dependency.reasons or [f"file_{dependency.status}"])
    )
    if omitted:
        incomplete.add("max_units")
    ready = sum(item.input_status == "ready" for item in units)
    no_input = sum(item.input_status == "no_input" for item in units)
    partial = sum(item.input_status == "partial" for item in units)
    failed = sum(item.input_status == "failed" for item in units)
    status = "complete"
    if failed == len(units) and units:
        status = "failed"
    elif incomplete or omitted or partial or failed:
        status = "partial"
    packet_digests = [
        packet.packet_digest for unit in units for packet in unit.packets
    ]
    payload = {
        "index_version": SEMANTIC_UNIT_INDEX_VERSION,
        "contract_version": SEMANTIC_UNIT_CONTRACT_VERSION,
        "packet_contract_version": PACKET_CONTRACT_VERSION,
        "snapshot_digest": discovery.repository.snapshot_digest,
        "observation_set_digest": observations.index.observation_set_digest,
        "required": settings.required,
        "status": status,
        "eligible_primary_universe_count": len(eligible_specs),
        "eligible_primary_universe_digest": universe_digest,
        "selection_policy_version": SEMANTIC_UNIT_SELECTION_VERSION,
        "selection_seed_digest": fair_order.seed_digest,
        "selected_primary_digest": selected_primary_digest,
        "omitted_primary_digest": omitted_primary_digest,
        "component_universe_counts": component_universe_counts,
        "component_selected_counts": component_selected_counts,
        "planned_primary_count": len(eligible_specs),
        "materialized_primary_count": len(units),
        "input_segment_count": sum(len(item.segments) for item in units),
        "packet_count": sum(len(item.packets) for item in units),
        "ready_count": ready,
        "no_input_count": no_input,
        "partial_count": partial,
        "failed_count": failed,
        "omitted_count": omitted,
        "total_owned_source_bytes": sum(
            source.end_utf8_byte - source.start_utf8_byte
            for unit in units
            for source in unit.owned_source_ranges
        ),
        "total_packet_bytes": total_packet_bytes,
        "file_dependencies": [
            item.model_dump(mode="json")
            for item in sorted(dependencies, key=lambda value: value.path)
        ],
        "units": [item.model_dump(mode="json") for item in refs],
        "unit_set_digest": payload_digest(
            {"unit_artifact_digests": [item.artifact_digest for item in refs]}
        ),
        "packet_set_digest": payload_digest(
            {"packet_digests": packet_digests}
        ),
        "resource_use": {
            "eligible_primary_units": len(eligible_specs),
            "materialized_primary_units": len(units),
            "omitted_primary_units": omitted,
            "input_segments": sum(len(item.segments) for item in units),
            "packet_bytes": total_packet_bytes,
            "context_refs": sum(
                len(packet.context_refs)
                for unit in units
                for packet in unit.packets
            ),
        },
        "incomplete_reasons": sorted(incomplete),
        "limitations": [
            "semantic_units_are_inputs_not_norm_claims",
            "semantic_packets_feed_current_source_map",
        ],
    }
    payload["index_digest"] = payload_digest(payload)
    index = SemanticUnitIndex.model_validate(payload)
    return SemanticUnitBundle(
        index=index,
        units=tuple(sorted(units, key=lambda item: item.unit_id)),
    )


def safe_reason(error: Exception) -> str:
    return " ".join(str(error).split())[:200] or type(error).__name__


def validate_semantic_unit_tree(
    run_directory: Path,
    *,
    repository_root: Path,
    discovery: DiscoveryArtifact,
    observations: ObservationBundle,
    settings: SemanticSettings,
) -> list[str]:
    """Strict-parse and cache-free replay the semantic-unit subtree."""

    errors: list[str] = []
    root = run_directory / "observations" / "semantic-units"
    if root.is_symlink() or not root.is_dir():
        return ["observations/semantic-units/ is missing or a symlink"]
    index_path = root / "index.json"
    try:
        if index_path.is_symlink() or not index_path.is_file():
            raise OSError("index is missing or a symlink")
        index_bytes = index_path.read_bytes()
        index = SemanticUnitIndex.model_validate_json(index_bytes)
        if index_bytes != canonical_json_bytes(index):
            errors.append("semantic-unit index is not canonical JSON")
    except (OSError, ValidationError, UnicodeDecodeError) as error:
        return [f"semantic-unit index error: {error}"]
    expected_files = {"index.json"}
    units: list[SemanticUnit] = []
    for reference in index.units:
        relative = f"units/{reference.unit_id}.json"
        expected_files.add(relative)
        path = root / relative
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("unit artifact is missing or a symlink")
            content = path.read_bytes()
            unit = SemanticUnit.model_validate_json(content)
            if content != canonical_json_bytes(unit):
                errors.append(f"{relative} is not canonical JSON")
            if unit.unit_id != reference.unit_id:
                errors.append(f"{relative} unit ID does not match index")
            if sha256_hex(content) != reference.artifact_sha256:
                errors.append(f"{relative} artifact hash does not match index")
            if unit.artifact_digest != reference.artifact_digest:
                errors.append(f"{relative} artifact digest does not match index")
            units.append(unit)
        except (OSError, ValidationError, UnicodeDecodeError) as error:
            errors.append(f"semantic-unit artifact error {relative}: {error}")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(
                "semantic-unit tree contains a symlink: "
                f"{path.relative_to(root).as_posix()}"
            )
        elif path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if expected_files != actual_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"missing semantic-unit artifacts: {missing}")
        if extra:
            errors.append(f"unexpected semantic-unit artifacts: {extra}")
    try:
        replayed = build_semantic_units(
            repository_root,
            discovery,
            observations,
            settings,
        )
    except (OSError, ValueError, UnicodeDecodeError) as error:
        errors.append(f"semantic-unit replay failed: {error}")
        return errors
    if index != replayed.index:
        errors.append("semantic-unit index differs from cache-free replay")
    if tuple(sorted(units, key=lambda item: item.unit_id)) != replayed.units:
        errors.append("semantic-unit artifacts differ from cache-free replay")
    manifest = {item.path: item for item in discovery.manifest.files}
    for unit in units:
        item = manifest.get(unit.path)
        if item is None:
            errors.append(f"{unit.unit_id}: source path is outside manifest")
            continue
        try:
            content = safe_read_manifest_file(repository_root, item)
        except OSError as error:
            errors.append(f"{unit.unit_id}: cannot reconstruct source: {error}")
            continue
        for packet in unit.packets:
            raw = reconstruct_packet_bytes(packet, unit=unit, content=content)
            if (
                len(raw) != packet.reconstructed_bytes
                or sha256_hex(raw) != packet.packet_digest
            ):
                errors.append(f"{packet.packet_id}: packet does not reconstruct")
    return errors
