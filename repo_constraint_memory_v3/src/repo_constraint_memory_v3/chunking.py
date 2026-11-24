"""Deterministic, lossless, line-addressable evidence units and batches."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Sequence

from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.models import EvidenceUnit
from repo_constraint_memory_v3.spans import physical_line_spans

CHUNKER_VERSION = "v3.4-make-phony-logical-blocks-1"

_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
_RST_UNDERLINE_RE = re.compile(r"^\s*[=\-~^\"'`:+*#<>_]{3,}\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_TOML_INI_SECTION_RE = re.compile(r"^\s*\[\[?[^\]]+\]\]?\s*(?:[#;].*)?$")
_MAKE_TARGET_RE = re.compile(r"^[A-Za-z0-9_.%/@+,-][^:=#]*:(?!=)")
_MAKE_PHONY_RE = re.compile(r"^\.PHONY\s*:\s*(?P<targets>[^#]*?)(?:\s+#.*)?$")


def _make_target_names(line: str) -> set[str]:
    match = _MAKE_TARGET_RE.match(line)
    if match is None:
        return set()
    target_text = line[: line.find(":")]
    return {item for item in target_text.split() if item}


def _paired_with_immediately_preceding_phony(
    lines: Sequence[str],
    index: int,
) -> bool:
    if index == 0:
        return False
    phony = _MAKE_PHONY_RE.match(lines[index - 1].rstrip("\r\n"))
    if phony is None:
        return False
    declared = set(phony.group("targets").split())
    return bool(declared & _make_target_names(lines[index]))


@dataclass(frozen=True)
class UnitBatch:
    batch_id: str
    units: tuple[EvidenceUnit, ...]
    total_chars: int


@dataclass(frozen=True)
class _Chunk:
    line_start: int
    line_end: int
    content: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _primary_boundaries(path: str, lines: Sequence[str]) -> list[int]:
    if not lines:
        return [0]
    suffix = PurePosixPath(path).suffix.casefold()
    name = PurePosixPath(path).name.casefold()
    boundaries = {0}
    if suffix in {".md", ".markdown", ".mdx", ".rst"}:
        for index, line in enumerate(lines):
            if _MARKDOWN_HEADING_RE.match(line):
                boundaries.add(index)
            if (
                index + 1 < len(lines)
                and line.strip()
                and _RST_UNDERLINE_RE.match(lines[index + 1])
            ):
                boundaries.add(index)
    elif suffix in {".toml", ".ini", ".cfg"}:
        for index, line in enumerate(lines):
            if _TOML_INI_SECTION_RE.match(line):
                boundaries.add(index)
    elif name in {"makefile", "gnumakefile"}:
        for index, line in enumerate(lines):
            if _MAKE_TARGET_RE.match(line) and not (
                _paired_with_immediately_preceding_phony(lines, index)
            ):
                boundaries.add(index)
    return sorted(boundaries)


def _preferred_breaks(lines: Sequence[str], start: int, end: int) -> set[int]:
    result: set[int] = set()
    for index in range(start + 1, end):
        if not lines[index - 1].strip():
            result.add(index)
        if _LIST_ITEM_RE.match(lines[index]):
            result.add(index)
    return result


def _split_line_range(
    lines: Sequence[str],
    start: int,
    end: int,
    maximum: int,
) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    preferred = _preferred_breaks(lines, start, end)
    cursor = start
    while cursor < end:
        if len(lines[cursor]) > maximum:
            # A physical line is the smallest addressable span. Keep it intact
            # even when it exceeds the preferred unit size; the builder reports
            # a bounded incomplete outcome if it cannot fit the batch budget.
            chunks.append(
                _Chunk(
                    line_start=cursor + 1,
                    line_end=cursor + 1,
                    content=lines[cursor],
                )
            )
            cursor += 1
            continue
        char_count = 0
        furthest = cursor
        while furthest < end and char_count + len(lines[furthest]) <= maximum:
            char_count += len(lines[furthest])
            furthest += 1
        if furthest == end:
            chosen_end = end
        else:
            choices = [
                boundary
                for boundary in preferred
                if cursor < boundary <= furthest
            ]
            chosen_end = max(choices) if choices else furthest
        if chosen_end <= cursor:
            chosen_end = cursor + 1
        content = "".join(lines[cursor:chosen_end])
        if content:
            chunks.append(
                _Chunk(
                    line_start=cursor + 1,
                    line_end=chosen_end,
                    content=content,
                )
            )
        cursor = chosen_end
    return chunks


def default_scope(path: str) -> list[str]:
    pure = PurePosixPath(path)
    name = pure.name.casefold()
    if name in {"agents.md", "claude.md"} and pure.parent != PurePosixPath("."):
        return [f"{pure.parent.as_posix()}/**"]
    if pure.parent == PurePosixPath("tests") and name.startswith("readme"):
        return ["tests/**"]
    if name in {
        "mypy.ini",
        "pyproject.toml",
        "pyrightconfig.json",
        "ruff.toml",
    }:
        return ["**/*.py"]
    if name == "pytest.ini":
        return ["tests/**"]
    return ["**"]


def chunk_source(source: SourceFile, *, max_unit_chars: int) -> list[EvidenceUnit]:
    if max_unit_chars < 1:
        raise ValueError("max_unit_chars must be positive")
    if not source.content:
        return []
    lines = source.content.splitlines(keepends=True) or [source.content]
    boundaries = _primary_boundaries(source.path, lines)
    boundaries.append(len(lines))
    chunks: list[_Chunk] = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if start < end:
            chunks.extend(_split_line_range(lines, start, end, max_unit_chars))

    units: list[EvidenceUnit] = []
    for chunk in chunks:
        unit_content_hash = _sha256_text(chunk.content)
        identity = {
            "chunker_version": CHUNKER_VERSION,
            "content_sha256": unit_content_hash,
            "line_end": chunk.line_end,
            "line_start": chunk.line_start,
            "path": source.path,
        }
        units.append(
            EvidenceUnit(
                unit_id=f"UNIT-{_sha256_text(_canonical_json(identity))[:24]}",
                path=source.path,
                source_kind=source.source_kind,
                discovery_origin=source.discovery_origin,  # type: ignore[arg-type]
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                default_scope=default_scope(source.path),
                content=chunk.content,
                content_sha256=unit_content_hash,
                source_content_sha256=source.content_sha256,
                spans=physical_line_spans(
                    chunk.content,
                    source_content_sha256=source.content_sha256,
                    line_start=chunk.line_start,
                ),
            )
        )
    return units


def chunk_sources(
    sources: Iterable[SourceFile],
    *,
    max_unit_chars: int,
) -> list[EvidenceUnit]:
    units = [
        unit
        for source in sources
        for unit in chunk_source(source, max_unit_chars=max_unit_chars)
    ]
    return sorted(
        units,
        key=lambda unit: (unit.path, unit.line_start, unit.line_end, unit.unit_id),
    )


def batch_units(
    units: Iterable[EvidenceUnit],
    *,
    max_batch_chars: int,
    max_units_per_batch: int,
) -> list[UnitBatch]:
    if max_batch_chars < 1 or max_units_per_batch < 1:
        raise ValueError("batch limits must be positive")
    # Keep the deterministic baseline ahead of the agent supplement and make
    # every request source-local.  Adding a supplemental path or another unit
    # to one source must not change the sampled request for an unrelated file.
    origin_rank = {"deterministic": 0, "agent_supplement": 1}
    ordered = sorted(
        units,
        key=lambda unit: (
            origin_rank[unit.discovery_origin],
            unit.path,
            unit.line_start,
            unit.line_end,
            unit.unit_id,
        ),
    )
    batches: list[UnitBatch] = []
    current: list[EvidenceUnit] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        identity = _canonical_json([unit.unit_id for unit in current])
        batches.append(
            UnitBatch(
                batch_id=f"BATCH-{_sha256_text(identity)[:24]}",
                units=tuple(current),
                total_chars=current_chars,
            )
        )
        current = []
        current_chars = 0

    for unit in ordered:
        unit_chars = len(unit.content)
        if unit_chars > max_batch_chars:
            raise ValueError(
                f"unit {unit.unit_id} exceeds max_batch_chars; "
                "max_batch_chars must be >= max_unit_chars"
            )
        if current and (
            current[-1].discovery_origin != unit.discovery_origin
            or current[-1].path != unit.path
            or current_chars + unit_chars > max_batch_chars
            or len(current) >= max_units_per_batch
        ):
            flush()
        current.append(unit)
        current_chars += unit_chars
    flush()
    return batches
