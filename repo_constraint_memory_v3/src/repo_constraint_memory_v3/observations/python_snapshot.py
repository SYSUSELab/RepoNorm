"""Bounded Python parsing, path binding, locations, entities, and relations."""

from __future__ import annotations

import ast
import builtins
import hashlib
import platform
import stat
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from repo_constraint_memory_v3.discovery import (
    DiscoveryError,
    matches_glob,
    sha256_bytes,
    valid_relative_posix_path,
)
from repo_constraint_memory_v3.models import (
    DiscoveryArtifact,
    ManifestFile,
    ObservationSettings,
)
from repo_constraint_memory_v3.observations.cache import ParseCache
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    model_payload_digest,
)
from repo_constraint_memory_v3.observations.file_roles import (
    observe_file_role,
    observe_module_identity,
)
from repo_constraint_memory_v3.observations.ids import (
    entity_id,
    file_id,
    relation_id,
    site_id,
    symbol_ref_identity,
)
from repo_constraint_memory_v3.observations.models import (
    DecoratorDescriptor,
    EntityDescriptor,
    FileRoleObservation,
    ModuleIdentity,
    MAX_RECORDED_PYTHON_MANIFEST_GAPS,
    ParsedPythonDescriptor,
    ParserProvenance,
    PythonEntity,
    PythonFileShard,
    PythonManifestGap,
    PythonRelation,
    PythonSite,
    PythonSnapshotArtifact,
    PythonSymbolRef,
    RelationDescriptor,
    SourceLocation,
)
AST_CONTRACT_VERSION = "stdlib-ast-structural-1.2"
SNAPSHOT_LIMITATION = (
    "Static syntax observations do not model runtime dispatch, dynamic imports, "
    "monkey patching, control flow, data flow, or a complete call graph."
)
MANIFEST_GAP_LIMITATION = (
    "Python paths omitted by the safe workspace manifest retain path-level "
    "dependency metadata but no source or AST facts."
)
_PYTHON_MANIFEST_GAP_REASONS = frozenset(
    {
        "missing",
        "symlink",
        "not_regular_file",
        "oversized_manifest_file",
        "manifest_file_count_budget",
        "manifest_total_bytes_budget",
        "binary_content",
    }
)
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True)
class SnapshotBuild:
    artifact: PythonSnapshotArtifact
    shards: tuple[PythonFileShard, ...]
    cache_metrics: dict[str, int]
    observed_max_workers: int


class DescriptorLimitError(ValueError):
    """A bounded semantic parser limit was exceeded."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def parser_provenance() -> ParserProvenance:
    return ParserProvenance(
        implementation=platform.python_implementation(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        ast_contract_version=AST_CONTRACT_VERSION,
    )


def safe_read_manifest_file(root: Path, item: ManifestFile) -> bytes:
    if not valid_relative_posix_path(item.path):
        raise DiscoveryError(f"unsafe observation path: {item.path}")
    absolute = root.resolve().joinpath(*PurePosixPath(item.path).parts)
    info = absolute.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise DiscoveryError(
            f"observation input is not a regular file: {item.path}"
        )
    content = absolute.read_bytes()
    if len(content) != item.size_bytes or sha256_bytes(content) != item.content_sha256:
        raise DiscoveryError(f"observation input changed after discovery: {item.path}")
    return content


def _endpoints(node: ast.AST) -> tuple[int, int, int, int]:
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_column = getattr(node, "end_col_offset", None)
    if None in {line, column, end_line, end_column}:
        raise DescriptorLimitError("missing_source_endpoint")
    return int(line), int(column), int(end_line), int(end_column)


def _chain(node: ast.AST) -> tuple[list[str], bool]:
    called = isinstance(node, ast.Call)
    target = node.func if called else node
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
        return list(reversed(parts)), called
    return list(reversed(parts)), called


def _target_text(node: ast.AST) -> tuple[str, bool]:
    chain, _ = _chain(node)
    if chain:
        return ".".join(chain), not isinstance(node, (ast.Name, ast.Attribute))
    return type(node).__name__, True


class _DescriptorVisitor(ast.NodeVisitor):
    def __init__(self, max_symbol_chars: int) -> None:
        self.max_symbol_chars = max_symbol_chars
        self.entities: list[EntityDescriptor] = []
        self.relations: list[RelationDescriptor] = []
        self.stack: list[int] = []
        self.kind_stack: list[str] = []
        self.ordinals: Counter[tuple[str, str]] = Counter()

    def _bounded(self, value: str) -> str:
        if len(value) > self.max_symbol_chars:
            raise DescriptorLimitError("max_symbol_chars")
        return value

    def _source_index(self) -> int:
        return self.stack[-1] if self.stack else -1

    def _relation(
        self,
        *,
        kind: str,
        node: ast.AST,
        target: str,
        alias: str | None = None,
        dynamic: bool = False,
    ) -> None:
        start_line, start_column, end_line, end_column = _endpoints(node)
        self.relations.append(
            RelationDescriptor(
                kind=kind,  # type: ignore[arg-type]
                source_entity_index=self._source_index(),
                target=self._bounded(target),
                alias=self._bounded(alias) if alias else None,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
                dynamic=dynamic,
            )
        )

    def _definition(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        parent_index = self.stack[-1] if self.stack else None
        parent_qname = (
            self.entities[parent_index].qualified_name
            if parent_index is not None
            else ""
        )
        qualified_name = (
            f"{parent_qname}.{node.name}" if parent_qname else node.name
        )
        if isinstance(node, ast.ClassDef):
            kind = "nested_class" if self.stack else "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            if self.kind_stack and self.kind_stack[-1] in {"class", "nested_class"}:
                kind = "async_method"
            elif self.stack:
                kind = "nested_async_function"
            else:
                kind = "async_function"
        elif self.kind_stack and self.kind_stack[-1] in {"class", "nested_class"}:
            kind = "method"
        elif self.stack:
            kind = "nested_function"
        else:
            kind = "function"
        key = (kind, qualified_name)
        self.ordinals[key] += 1
        start_line, start_column, end_line, end_column = _endpoints(node)
        decorators: list[DecoratorDescriptor] = []
        for decorator in node.decorator_list:
            d_start_line, d_start_column, d_end_line, d_end_column = _endpoints(
                decorator
            )
            chain, called = _chain(decorator)
            symbols = sorted(
                {
                    child.id
                    for child in ast.walk(decorator)
                    if isinstance(child, ast.Name)
                }
                | {
                    child.attr
                    for child in ast.walk(decorator)
                    if isinstance(child, ast.Attribute)
                }
            )
            decorators.append(
                DecoratorDescriptor(
                    start_line=d_start_line,
                    start_column=d_start_column,
                    end_line=d_end_line,
                    end_column=d_end_column,
                    chain=[self._bounded(part) for part in chain],
                    symbols=[self._bounded(part) for part in symbols],
                    called=called,
                    dynamic=not bool(chain) or (called and len(chain) == 1),
                )
            )
        descriptor_index = len(self.entities)
        self.entities.append(
            EntityDescriptor(
                descriptor_index=descriptor_index,
                parent_index=parent_index,
                kind=kind,
                name=self._bounded(node.name),
                qualified_name=self._bounded(qualified_name),
                lexical_ordinal=self.ordinals[key],
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
                decorators=decorators,
            )
        )
        self.stack.append(descriptor_index)
        self.kind_stack.append(kind)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                target, dynamic = _target_text(base)
                self._relation(
                    kind="inherits",
                    node=base,
                    target=target,
                    dynamic=dynamic,
                )
        for statement in node.body:
            self.visit(statement)
        self.kind_stack.pop()
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._definition(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._definition(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._definition(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._relation(
                kind="imports",
                node=node,
                target=alias.name,
                alias=alias.asname,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self._relation(
                kind="imports_from",
                node=node,
                target=f"{module}:{alias.name}",
                alias=alias.asname,
                dynamic=alias.name == "*",
            )

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self._relation(
                kind="references",
                node=node,
                target=node.id,
            )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        target, dynamic = _target_text(node.func)
        self._relation(
            kind="calls",
            node=node.func,
            target=target,
            dynamic=dynamic or target in {"getattr", "__import__"},
        )
        self.generic_visit(node)


def parse_python_descriptor(
    content: bytes,
    *,
    content_sha256: str,
    settings: ObservationSettings,
) -> ParsedPythonDescriptor:
    text = content.decode("utf-8", errors="strict")
    tree = ast.parse(text)
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > settings.max_ast_nodes_per_file:
        raise DescriptorLimitError("max_ast_nodes_per_file")
    visitor = _DescriptorVisitor(settings.max_symbol_chars)
    visitor.visit(tree)
    if len(visitor.entities) > settings.max_entities_per_file:
        raise DescriptorLimitError("max_entities_per_file")
    if len(visitor.relations) > settings.max_relations_per_file:
        raise DescriptorLimitError("max_relations_per_file")
    fingerprint = hashlib.sha256(
        ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
    ).hexdigest()
    return ParsedPythonDescriptor(
        content_sha256=content_sha256,
        ast_node_count=node_count,
        ast_fingerprint=fingerprint,
        entities=visitor.entities,
        relations=visitor.relations,
    )


def _line_starts(content: bytes) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    return tuple(starts)


def _excerpt_bytes(
    content: bytes,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
    line_starts: tuple[int, ...] | None = None,
) -> bytes:
    starts = _line_starts(content) if line_starts is None else line_starts
    if not (1 <= start_line <= len(starts)) or not (1 <= end_line <= len(starts)):
        raise ValueError("source location line is outside the snapshot")
    start = starts[start_line - 1] + start_column
    end = starts[end_line - 1] + end_column
    if start < 0 or end < start or end > len(content):
        raise ValueError("source location byte endpoint is outside the snapshot")
    return content[start:end]


def source_location(
    *,
    path: str,
    content_sha256: str,
    content: bytes,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
    line_starts: tuple[int, ...] | None = None,
) -> SourceLocation:
    excerpt = _excerpt_bytes(
        content,
        start_line,
        start_column,
        end_line,
        end_column,
        line_starts,
    )
    return SourceLocation(
        path=path,
        source_content_sha256=content_sha256,
        start_line=start_line,
        start_utf8_byte_column=start_column,
        end_line=end_line,
        end_utf8_byte_column=end_column,
        excerpt_sha256=sha256_bytes(excerpt),
        parser_query_version="python_snapshot:1.4",
    )


def _minimal_shard(
    item: ManifestFile,
    *,
    role: FileRoleObservation,
    module: ModuleIdentity,
    status: str,
    reason: str,
    parser: ParserProvenance | None,
) -> PythonFileShard:
    provisional = PythonFileShard(
        schema_version="1.0",
        file_id=file_id(item.path, item.content_sha256),
        path=item.path,
        content_sha256=item.content_sha256,
        size_bytes=item.size_bytes,
        source_form="stub" if item.suffix == ".pyi" else "implementation",
        role=role,
        module=module,
        parse_status=status,  # type: ignore[arg-type]
        parser=parser,
        ast_fingerprint=None,
        ast_node_count=None,
        entities=[],
        sites=[],
        relations=[],
        incomplete_reasons=[reason],
        limitations=[SNAPSHOT_LIMITATION],
        shard_digest="0" * 64,
    )
    return provisional.model_copy(
        update={
            "shard_digest": model_payload_digest(
                provisional,
                digest_field="shard_digest",
            )
        }
    )


def _resolution(
    target: str,
    *,
    dynamic: bool,
    local_entities: dict[str, list[PythonEntity]],
    import_aliases: set[str],
    local_modules: dict[str, str],
) -> tuple[str, str | None, str | None]:
    if dynamic or target == "*":
        return "unresolved", None, None
    first = target.split(".", 1)[0].split(":", 1)[0]
    matches = local_entities.get(first, [])
    if len(matches) == 1:
        return "exact_local_entity", matches[0].entity_id, None
    if len(matches) > 1:
        return "ambiguous", None, None
    if first in local_modules:
        return "exact_local_module", None, local_modules[first]
    if first in import_aliases or first in _BUILTIN_NAMES:
        return "external", None, None
    return "unresolved", None, None


def bind_descriptor(
    item: ManifestFile,
    *,
    content: bytes,
    descriptor: ParsedPythonDescriptor,
    role: FileRoleObservation,
    module: ModuleIdentity,
    parser: ParserProvenance,
    local_modules: dict[str, str],
) -> PythonFileShard:
    current_file_id = file_id(item.path, item.content_sha256)
    line_starts = _line_starts(content)
    module_name = module.module_name or PurePosixPath(item.path).stem
    module_entity = PythonEntity(
        entity_id=entity_id(item.path, "module", module_name, 1),
        file_id=current_file_id,
        kind="module",
        name=module_name,
        qualified_name=module_name,
        lexical_ordinal=1,
        parent_entity_id=None,
        definition_site_id=None,
        decorator_site_ids=[],
    )
    sites: list[PythonSite] = []
    entities: list[PythonEntity] = [module_entity]
    descriptor_ids: dict[int, str] = {}
    for entity in descriptor.entities:
        current_id = entity_id(
            item.path,
            entity.kind,
            entity.qualified_name,
            entity.lexical_ordinal,
        )
        descriptor_ids[entity.descriptor_index] = current_id
        definition_id = site_id(
            item.path,
            item.content_sha256,
            "definition",
            entity.start_line,
            entity.start_column,
            entity.end_line,
            entity.end_column,
        )
        sites.append(
            PythonSite(
                site_id=definition_id,
                kind="definition",
                location=source_location(
                    path=item.path,
                    content_sha256=item.content_sha256,
                    content=content,
                    line_starts=line_starts,
                    start_line=entity.start_line,
                    start_column=entity.start_column,
                    end_line=entity.end_line,
                    end_column=entity.end_column,
                ),
            )
        )
        current_decorators: list[str] = []
        for decorator in entity.decorators:
            current_site_id = site_id(
                item.path,
                item.content_sha256,
                "decorator",
                decorator.start_line,
                decorator.start_column,
                decorator.end_line,
                decorator.end_column,
            )
            sites.append(
                PythonSite(
                    site_id=current_site_id,
                    kind="decorator",
                    location=source_location(
                        path=item.path,
                        content_sha256=item.content_sha256,
                        content=content,
                        line_starts=line_starts,
                        start_line=decorator.start_line,
                        start_column=decorator.start_column,
                        end_line=decorator.end_line,
                        end_column=decorator.end_column,
                    ),
                    decorator_chain=decorator.chain,
                    decorator_symbols=decorator.symbols,
                    decorator_call=decorator.called,
                    decorator_dynamic=decorator.dynamic,
                )
            )
            current_decorators.append(current_site_id)
        entities.append(
            PythonEntity(
                entity_id=current_id,
                file_id=current_file_id,
                kind=entity.kind,  # type: ignore[arg-type]
                name=entity.name,
                qualified_name=entity.qualified_name,
                lexical_ordinal=entity.lexical_ordinal,
                parent_entity_id=(
                    descriptor_ids[entity.parent_index]
                    if entity.parent_index is not None
                    else module_entity.entity_id
                ),
                definition_site_id=definition_id,
                decorator_site_ids=current_decorators,
            )
        )

    entities_by_name: dict[str, list[PythonEntity]] = defaultdict(list)
    for entity in entities[1:]:
        entities_by_name[entity.name].append(entity)
    import_aliases: set[str] = set()
    for relation in descriptor.relations:
        if relation.kind == "imports":
            import_aliases.add(relation.alias or relation.target.split(".", 1)[0])
        elif relation.kind == "imports_from":
            imported = relation.target.split(":", 1)[-1]
            import_aliases.add(relation.alias or imported)

    relations: list[PythonRelation] = []
    for entity in entities[1:]:
        parent_id = entity.parent_entity_id or module_entity.entity_id
        for kind in ("defines", "contains"):
            relations.append(
                PythonRelation(
                    relation_id=relation_id(
                        kind,
                        parent_id,
                        entity.entity_id,
                        entity.definition_site_id,
                    ),
                    kind=kind,  # type: ignore[arg-type]
                    source=PythonSymbolRef(
                        rendered=parent_id,
                        entity_id=parent_id,
                        file_id=current_file_id,
                    ),
                    target=PythonSymbolRef(
                        rendered=entity.qualified_name,
                        entity_id=entity.entity_id,
                        file_id=current_file_id,
                    ),
                    site_id=entity.definition_site_id,
                    resolution="exact_local_entity",
                )
            )
    relation_site_kind = {
        "imports": "import",
        "imports_from": "import",
        "references": "reference",
        "inherits": "inheritance",
        "calls": "call",
    }
    known_site_ids = {site.site_id for site in sites}
    entities_by_id = {entity.entity_id: entity for entity in entities}
    for descriptor_relation in descriptor.relations:
        current_site_id = site_id(
            item.path,
            item.content_sha256,
            relation_site_kind[descriptor_relation.kind],
            descriptor_relation.start_line,
            descriptor_relation.start_column,
            descriptor_relation.end_line,
            descriptor_relation.end_column,
        )
        if current_site_id not in known_site_ids:
            known_site_ids.add(current_site_id)
            sites.append(
                PythonSite(
                    site_id=current_site_id,
                    kind=relation_site_kind[descriptor_relation.kind],  # type: ignore[arg-type]
                    location=source_location(
                        path=item.path,
                        content_sha256=item.content_sha256,
                        content=content,
                        line_starts=line_starts,
                        start_line=descriptor_relation.start_line,
                        start_column=descriptor_relation.start_column,
                        end_line=descriptor_relation.end_line,
                        end_column=descriptor_relation.end_column,
                    ),
                )
            )
        source_entity = (
            module_entity
            if descriptor_relation.source_entity_index == -1
            else entities_by_id[
                descriptor_ids[descriptor_relation.source_entity_index]
            ]
        )
        if descriptor_relation.kind in {"imports", "imports_from"}:
            resolution = (
                "exact_local_module"
                if descriptor_relation.target.split(":", 1)[0].lstrip(".")
                in local_modules
                else "external"
            )
            target_entity_id = None
            target_file_id = local_modules.get(
                descriptor_relation.target.split(":", 1)[0].lstrip(".")
            )
            if descriptor_relation.dynamic:
                resolution = "unresolved"
                target_file_id = None
        else:
            resolution, target_entity_id, target_file_id = _resolution(
                descriptor_relation.target,
                dynamic=descriptor_relation.dynamic,
                local_entities=entities_by_name,
                import_aliases=import_aliases,
                local_modules=local_modules,
            )
        target_identity = symbol_ref_identity(
            descriptor_relation.target,
            entity_id=target_entity_id,
            file_id=target_file_id,
        )
        relations.append(
            PythonRelation(
                relation_id=relation_id(
                    descriptor_relation.kind,
                    source_entity.entity_id,
                    target_identity,
                    current_site_id,
                ),
                kind=descriptor_relation.kind,
                source=PythonSymbolRef(
                    rendered=source_entity.qualified_name,
                    entity_id=source_entity.entity_id,
                    file_id=current_file_id,
                ),
                target=PythonSymbolRef(
                    rendered=descriptor_relation.target,
                    entity_id=target_entity_id,
                    file_id=target_file_id,
                ),
                site_id=current_site_id,
                resolution=resolution,  # type: ignore[arg-type]
                alias=descriptor_relation.alias,
            )
        )
    entities.sort(key=lambda value: value.entity_id)
    sites.sort(key=lambda value: value.site_id)
    relations.sort(key=lambda value: value.relation_id)
    provisional = PythonFileShard(
        schema_version="1.0",
        file_id=current_file_id,
        path=item.path,
        content_sha256=item.content_sha256,
        size_bytes=item.size_bytes,
        source_form="stub" if item.suffix == ".pyi" else "implementation",
        role=role,
        module=module,
        parse_status="complete",
        parser=parser,
        ast_fingerprint=descriptor.ast_fingerprint,
        ast_node_count=descriptor.ast_node_count,
        entities=entities,
        sites=sites,
        relations=relations,
        incomplete_reasons=[],
        limitations=[SNAPSHOT_LIMITATION],
        shard_digest="0" * 64,
    )
    return provisional.model_copy(
        update={
            "shard_digest": model_payload_digest(
                provisional,
                digest_field="shard_digest",
            )
        }
    )


def _artifact_for_shards(
    *,
    discovery: DiscoveryArtifact,
    parser: ParserProvenance,
    shards: Iterable[PythonFileShard],
) -> PythonSnapshotArtifact:
    ordered = sorted(shards, key=lambda item: item.path)
    manifest_gaps, manifest_gap_count = _python_manifest_gaps(discovery)
    manifest_gap_omitted = manifest_gap_count - len(manifest_gaps)
    reasons = sorted(
        {
            *(
                f"{item.path}:{reason}"
                for item in ordered
                for reason in item.incomplete_reasons
            ),
            *(
                "manifest_gap:"
                f"{item.path}:"
                f"{item.reason}"
                + (f":{item.reason_detail}" if item.reason_detail else "")
                for item in manifest_gaps
            ),
            *(
                [
                    "manifest_gap:recording_limit:"
                    f"{manifest_gap_omitted}"
                ]
                if manifest_gap_omitted
                else []
            ),
        }
    )
    processed = [item for item in ordered if item.parse_status == "complete"]
    failed = [
        item for item in ordered if item.parse_status in {"failed", "partial"}
    ]
    skipped = [item for item in ordered if item.parse_status == "skipped"]
    status = "partial" if reasons else "complete"
    provisional = PythonSnapshotArtifact(
        schema_version="1.0",
        query_name="python_snapshot",
        query_version="1.4",
        snapshot_digest=discovery.repository.snapshot_digest,
        status=status,
        parser=parser,
        eligible_file_count=len(ordered) + manifest_gap_count,
        processed_file_count=len(processed),
        skipped_file_count=len(skipped) + manifest_gap_count,
        failed_file_count=len(failed),
        entity_count=sum(len(item.entities) for item in processed),
        site_count=sum(len(item.sites) for item in processed),
        relation_count=sum(len(item.relations) for item in processed),
        file_shards=[],
        unrepresented_python_file_count=manifest_gap_count,
        unrepresented_python_files=manifest_gaps,
        unrepresented_python_files_omitted_count=manifest_gap_omitted,
        incomplete_reasons=reasons,
        limitations=[
            SNAPSHOT_LIMITATION,
            *([MANIFEST_GAP_LIMITATION] if manifest_gaps else []),
        ],
        artifact_digest="0" * 64,
    )
    return provisional.model_copy(
        update={
            "artifact_digest": model_payload_digest(
                provisional,
                digest_field="artifact_digest",
            )
        }
    )


def _python_manifest_gaps(
    discovery: DiscoveryArtifact,
) -> tuple[list[PythonManifestGap], int]:
    """Normalize only true, safe, in-scope manifest omissions for Python."""

    represented_paths = {item.path for item in discovery.manifest.files}
    candidates: list[tuple[str, str, str | None]] = []
    candidate_count = 0
    for skipped in discovery.manifest.skipped:
        path = skipped.path
        pure = PurePosixPath(path)
        if (
            path in represented_paths
            or not valid_relative_posix_path(path)
            or pure.suffix.casefold() not in {".py", ".pyi"}
            or any(matches_glob(path, pattern) for pattern in discovery.limits.exclude)
        ):
            continue
        reason = skipped.reason
        detail: str | None = None
        if reason.startswith("read_error:"):
            reason, _, detail = reason.partition(":")
            if not detail:
                continue
        elif reason not in _PYTHON_MANIFEST_GAP_REASONS:
            # Baseline scope exclusions (ignored/generated/secret paths) and
            # later explicit-source skips are not Python observation omissions.
            continue
        candidate_count += 1
        if len(candidates) < MAX_RECORDED_PYTHON_MANIFEST_GAPS:
            candidates.append((path, reason, detail))

    all_python_paths = represented_paths | {path for path, _, _ in candidates}
    gaps: list[PythonManifestGap] = []
    for path, reason, detail in sorted(candidates):
        module = observe_module_identity(path, all_paths=all_python_paths)
        if module.resolution_status == "exact" and module.package_parts:
            component_key = f"module:{module.package_parts[0]}"
        else:
            parts = PurePosixPath(path).parts
            component_key = f"path:{parts[0]}" if parts else "unknown"
        gaps.append(
            PythonManifestGap(
                path=path,
                reason=reason,  # type: ignore[arg-type]
                reason_detail=detail,
                component_key=component_key,
            )
        )
    return gaps, candidate_count


def observe_python_snapshot(
    repository_root: Path,
    discovery: DiscoveryArtifact,
    settings: ObservationSettings,
    *,
    cache_directory: Path | None,
) -> SnapshotBuild:
    root = repository_root.resolve()
    parser = parser_provenance()
    eligible = sorted(
        (
            item
            for item in discovery.manifest.files
            if item.role == "python_source"
            and not any(
                matches_glob(item.path, pattern)
                for pattern in discovery.limits.exclude
            )
        ),
        key=lambda item: item.path,
    )
    all_paths = {item.path for item in discovery.manifest.files}
    src_roots = {
        PurePosixPath(item.path).parts[0]
        for item in eligible
        if PurePosixPath(item.path).parts
        and PurePosixPath(item.path).parts[0] == "src"
    }
    unique_src = len(src_roots) == 1
    contents: dict[str, bytes] = {}
    roles: dict[str, FileRoleObservation] = {}
    modules: dict[str, ModuleIdentity] = {}
    preliminary: dict[str, PythonFileShard] = {}
    total_bytes = 0
    selected: list[ManifestFile] = []
    for index, item in enumerate(eligible):
        package_chain = any(
            (
                PurePosixPath(*PurePosixPath(item.path).parts[:position])
                / "__init__.py"
            ).as_posix()
            in all_paths
            for position in range(1, len(PurePosixPath(item.path).parts))
        )
        module = observe_module_identity(item.path, all_paths=all_paths)
        modules[item.path] = module
        try:
            content = safe_read_manifest_file(root, item)
            text = content.decode("utf-8", errors="strict")
            header_lines = text.splitlines()[:5]
        except UnicodeDecodeError:
            header_lines = []
            role = observe_file_role(
                item.path,
                header_lines=header_lines,
                package_chain=package_chain,
                unique_src_root=unique_src
                and PurePosixPath(item.path).parts[0] == "src",
            )
            roles[item.path] = role
            preliminary[item.path] = _minimal_shard(
                item,
                role=role,
                module=module,
                status="failed",
                reason="invalid_utf8",
                parser=parser,
            )
            continue
        role = observe_file_role(
            item.path,
            header_lines=header_lines,
            package_chain=package_chain,
            unique_src_root=unique_src
            and PurePosixPath(item.path).parts[0] == "src",
        )
        roles[item.path] = role
        if index >= settings.max_python_files:
            preliminary[item.path] = _minimal_shard(
                item,
                role=role,
                module=module,
                status="skipped",
                reason="max_python_files",
                parser=None,
            )
            continue
        if item.size_bytes > settings.max_python_file_bytes:
            preliminary[item.path] = _minimal_shard(
                item,
                role=role,
                module=module,
                status="skipped",
                reason="max_python_file_bytes",
                parser=None,
            )
            continue
        if total_bytes + item.size_bytes > settings.max_total_python_bytes:
            preliminary[item.path] = _minimal_shard(
                item,
                role=role,
                module=module,
                status="skipped",
                reason="max_total_python_bytes",
                parser=None,
            )
            continue
        total_bytes += item.size_bytes
        contents[item.path] = content
        selected.append(item)

    semantic_limits = {
        "max_ast_nodes_per_file": settings.max_ast_nodes_per_file,
        "max_entities_per_file": settings.max_entities_per_file,
        "max_relations_per_file": settings.max_relations_per_file,
        "max_symbol_chars": settings.max_symbol_chars,
    }
    cache = ParseCache(
        cache_directory,
        repository_root=root,
        parser=parser,
        semantic_limits=semantic_limits,
        max_entry_bytes=settings.max_cache_entry_bytes,
    )
    items_by_hash: dict[str, list[ManifestFile]] = defaultdict(list)
    for item in selected:
        items_by_hash[item.content_sha256].append(item)
    descriptors: dict[str, ParsedPythonDescriptor] = {}
    parse_errors: dict[str, str] = {}
    peak_lock = threading.Lock()
    active = 0
    peak = 0

    def parse_group(content_hash: str) -> tuple[str, ParsedPythonDescriptor | None, str | None]:
        nonlocal active, peak
        key = cache.key_for(content_hash)
        representative = sorted(items_by_hash[content_hash], key=lambda item: item.path)[0]
        with peak_lock:
            active += 1
            peak = max(peak, active)
        try:
            descriptor = parse_python_descriptor(
                contents[representative.path],
                content_sha256=content_hash,
                settings=settings,
            )
        except UnicodeDecodeError:
            return content_hash, None, "invalid_utf8"
        except SyntaxError:
            return content_hash, None, "syntax_error"
        except (ValueError, RecursionError) as error:
            reason = (
                error.reason
                if isinstance(error, DescriptorLimitError)
                else f"parser_{type(error).__name__.casefold()}"
            )
            return content_hash, None, reason
        finally:
            with peak_lock:
                active -= 1
        cached = cache.read(
            key,
            expected_content_sha256=content_hash,
            expected_descriptor=descriptor,
        )
        if cached is not None:
            return content_hash, cached, None
        cache.write(key, descriptor)
        return content_hash, descriptor, None

    pending_hashes = sorted(items_by_hash)
    failure_count = 0
    while pending_hashes:
        if failure_count >= settings.max_parse_failures:
            for content_hash in pending_hashes:
                parse_errors[content_hash] = "max_parse_failures"
            break
        wave = pending_hashes[: settings.max_workers]
        pending_hashes = pending_hashes[settings.max_workers :]
        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            results = list(executor.map(parse_group, wave))
        for content_hash, descriptor, error in results:
            if descriptor is not None:
                descriptors[content_hash] = descriptor
            elif error is not None:
                parse_errors[content_hash] = error
                if error != "max_parse_failures":
                    failure_count += 1

    eligible_by_path = {item.path: item for item in eligible}
    local_modules: dict[str, str] = {}
    local_module_forms: dict[str, str] = {}
    for path, module in sorted(modules.items()):
        if module.module_name is None or module.resolution_status != "exact":
            continue
        item = eligible_by_path[path]
        source_form = "stub" if item.suffix == ".pyi" else "implementation"
        previous = local_module_forms.get(module.module_name)
        if previous == "implementation" and source_form == "stub":
            continue
        local_modules[module.module_name] = file_id(path, item.content_sha256)
        local_module_forms[module.module_name] = source_form
    shards: list[PythonFileShard] = []
    total_entities = 0
    total_relations = 0
    total_artifact_bytes = 0
    for item in eligible:
        if item.path in preliminary:
            shard = preliminary[item.path]
        elif item.content_sha256 in parse_errors:
            reason = parse_errors[item.content_sha256]
            shard = _minimal_shard(
                item,
                role=roles[item.path],
                module=modules[item.path],
                status="skipped" if reason == "max_parse_failures" else "failed",
                reason=reason,
                parser=parser,
            )
        else:
            descriptor = descriptors[item.content_sha256]
            shard = bind_descriptor(
                item,
                content=contents[item.path],
                descriptor=descriptor,
                role=roles[item.path],
                module=modules[item.path],
                parser=parser,
                local_modules=local_modules,
            )
            if total_entities + len(shard.entities) > settings.max_total_entities:
                shard = _minimal_shard(
                    item,
                    role=roles[item.path],
                    module=modules[item.path],
                    status="partial",
                    reason="max_total_entities",
                    parser=parser,
                )
            elif total_relations + len(shard.relations) > settings.max_total_relations:
                shard = _minimal_shard(
                    item,
                    role=roles[item.path],
                    module=modules[item.path],
                    status="partial",
                    reason="max_total_relations",
                    parser=parser,
                )
            elif len(canonical_json_bytes(shard)) > settings.max_file_artifact_bytes:
                shard = _minimal_shard(
                    item,
                    role=roles[item.path],
                    module=modules[item.path],
                    status="partial",
                    reason="max_file_artifact_bytes",
                    parser=parser,
                )
            if shard.parse_status == "complete":
                total_entities += len(shard.entities)
                total_relations += len(shard.relations)
        shard_size = len(canonical_json_bytes(shard))
        if (
            total_artifact_bytes + shard_size
            > settings.max_total_observation_bytes
            and shard.parse_status == "complete"
        ):
            shard = _minimal_shard(
                item,
                role=roles[item.path],
                module=modules[item.path],
                status="partial",
                reason="max_total_observation_bytes",
                parser=parser,
            )
            shard_size = len(canonical_json_bytes(shard))
        total_artifact_bytes += shard_size
        shards.append(shard)
    artifact = _artifact_for_shards(
        discovery=discovery,
        parser=parser,
        shards=shards,
    )
    return SnapshotBuild(
        artifact=artifact,
        shards=tuple(shards),
        cache_metrics={
            "hits": cache.metrics.hits,
            "misses": cache.metrics.misses,
            "rejected": cache.metrics.rejected,
            "writes": cache.metrics.writes,
        },
        observed_max_workers=peak,
    )
