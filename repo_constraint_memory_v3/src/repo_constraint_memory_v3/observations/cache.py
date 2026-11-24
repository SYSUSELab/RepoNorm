"""Untrusted, content-addressed parse cache with exclusive immutable entries."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    canonical_json_text,
    model_payload_digest,
    payload_digest,
)
from repo_constraint_memory_v3.observations.models import (
    ParseCacheEntry,
    ParsedPythonDescriptor,
    ParserProvenance,
)

CACHE_SCHEMA_VERSION = "1.1"

_ENTITY_KINDS = frozenset(
    {
        "class",
        "function",
        "async_function",
        "method",
        "async_method",
        "nested_class",
        "nested_function",
        "nested_async_function",
    }
)


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    rejected: int = 0
    writes: int = 0


class ParseCache:
    """Read and add immutable descriptors without trusting persisted content."""

    def __init__(
        self,
        directory: Path | None,
        *,
        repository_root: Path,
        parser: ParserProvenance,
        semantic_limits: dict[str, int],
        max_entry_bytes: int,
    ) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self.parser = parser
        self.semantic_limits = dict(sorted(semantic_limits.items()))
        self.max_entry_bytes = max_entry_bytes
        self.metrics = CacheMetrics()
        self._lock = threading.Lock()
        if self.directory is not None:
            repository_root = repository_root.resolve()
            if self.directory == repository_root or repository_root in self.directory.parents:
                raise ValueError(
                    "observation cache must be outside the target repository"
                )
            if self.directory.exists() and self.directory.is_symlink():
                raise ValueError("observation cache directory must not be a symlink")
            self.directory.mkdir(parents=True, exist_ok=True)

    def key_for(self, content_sha256: str) -> str:
        return payload_digest(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "parser": self.parser.model_dump(mode="json"),
                "semantic_limits": self.semantic_limits,
                "max_entry_bytes": self.max_entry_bytes,
                "content_sha256": content_sha256,
            }
        )

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / key[:2] / f"{key}.json"

    def _descriptor_within_semantic_limits(
        self,
        descriptor: ParsedPythonDescriptor,
    ) -> bool:
        try:
            max_nodes = self.semantic_limits["max_ast_nodes_per_file"]
            max_entities = self.semantic_limits["max_entities_per_file"]
            max_relations = self.semantic_limits["max_relations_per_file"]
            max_symbol_chars = self.semantic_limits["max_symbol_chars"]
        except KeyError:
            return False
        if (
            descriptor.ast_node_count > max_nodes
            or len(descriptor.entities) > max_entities
            or len(descriptor.relations) > max_relations
        ):
            return False
        for ordinal, entity in enumerate(descriptor.entities):
            if (
                entity.descriptor_index != ordinal
                or entity.kind not in _ENTITY_KINDS
                or (
                    entity.parent_index is not None
                    and not 0 <= entity.parent_index < ordinal
                )
            ):
                return False
            symbols = [entity.name, entity.qualified_name]
            for decorator in entity.decorators:
                symbols.extend(decorator.chain)
                symbols.extend(decorator.symbols)
            if any(len(value) > max_symbol_chars for value in symbols):
                return False
        for relation in descriptor.relations:
            if not -1 <= relation.source_entity_index < len(
                descriptor.entities
            ):
                return False
            if len(relation.target) > max_symbol_chars or (
                relation.alias is not None
                and len(relation.alias) > max_symbol_chars
            ):
                return False
        return True

    def read(
        self,
        key: str,
        *,
        expected_content_sha256: str,
        expected_descriptor: ParsedPythonDescriptor,
    ) -> ParsedPythonDescriptor | None:
        if self.directory is None:
            with self._lock:
                self.metrics.misses += 1
            return None
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.metrics.misses += 1
            return None
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("cache entry is not a regular file")
            content = path.read_bytes()
            if len(content) > self.max_entry_bytes:
                raise ValueError("cache entry exceeds configured byte limit")
            entry = ParseCacheEntry.model_validate_json(content)
            if entry.cache_key != key:
                raise ValueError("cache entry key mismatch")
            if (
                entry.descriptor.content_sha256 != expected_content_sha256
                or self.key_for(entry.descriptor.content_sha256) != key
            ):
                raise ValueError("cache descriptor content identity mismatch")
            if entry.descriptor != expected_descriptor:
                raise ValueError("cache descriptor differs from fresh parse")
            if entry.parser != self.parser:
                raise ValueError("cache parser provenance mismatch")
            if not self._descriptor_within_semantic_limits(entry.descriptor):
                raise ValueError("cache descriptor exceeds semantic limits")
            if entry.entry_digest != model_payload_digest(
                entry,
                digest_field="entry_digest",
            ):
                raise ValueError("cache entry digest mismatch")
            if content != canonical_json_bytes(entry):
                raise ValueError("cache entry is not canonical JSON")
        except (OSError, UnicodeDecodeError, ValidationError, ValueError):
            with self._lock:
                self.metrics.rejected += 1
            return None
        with self._lock:
            self.metrics.hits += 1
        return entry.descriptor

    def write(self, key: str, descriptor: ParsedPythonDescriptor) -> None:
        if self.directory is None:
            return
        if self.key_for(descriptor.content_sha256) != key:
            raise ValueError("cache descriptor content identity mismatch")
        if not self._descriptor_within_semantic_limits(descriptor):
            raise ValueError("cache descriptor exceeds semantic limits")
        provisional = ParseCacheEntry(
            schema_version=CACHE_SCHEMA_VERSION,
            cache_key=key,
            parser=self.parser,
            descriptor=descriptor,
            entry_digest="0" * 64,
        )
        entry = provisional.model_copy(
            update={
                "entry_digest": model_payload_digest(
                    provisional,
                    digest_field="entry_digest",
                )
            }
        )
        content = canonical_json_text(entry).encode("utf-8")
        if len(content) > self.max_entry_bytes:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor_fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return
        try:
            with os.fdopen(descriptor_fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        with self._lock:
            self.metrics.writes += 1
