"""Shared deterministic bounds for explicit evidence selections.

The ordinary evidence budget stays deliberately small.  A larger selection is
accepted only when the host can prove that it is one complete structured
assignment whose physical representation is necessarily multiline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Sequence

try:  # pragma: no cover - exercised only on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_EVIDENCE_LINE_LIMIT = 12
MAX_LOGICAL_EVIDENCE_LINES = 64

# EditorConfig intentionally stays on the ordinary limit: unlike the
# ConfigParser-style formats below, its values do not have continuation lines.
_INI_CFG_NAMES = {".flake8", ".coveragerc"}
_INI_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_.-]+)\s*[:=](?P<value>.*)$"
)
_SECTION_RE = re.compile(r"^\s*\[\[?[^\]]+\]\]?\s*(?:[#;].*)?$")
_TOML_KEY_PART = r'(?:[A-Za-z0-9_-]+|"[^"\n]+"|\'[^\'\n]+\')'
_TOML_ARRAY_START_RE = re.compile(
    rf"^\s*{_TOML_KEY_PART}(?:\s*\.\s*{_TOML_KEY_PART})*\s*=\s*\["
)
_TOML_ASSIGNMENT_RE = re.compile(
    rf"^\s*{_TOML_KEY_PART}(?:\s*\.\s*{_TOML_KEY_PART})*\s*="
)
_TOML_ARRAY_END_RE = re.compile(r"^\s*\]\s*(?:#.*)?$")


class EvidenceRangePolicyError(ValueError):
    """One evidence range exceeds the bounded host-verifiable policy."""


@dataclass(frozen=True)
class EvidenceRangeShape:
    """The deterministic shape that justified one evidence selection."""

    kind: Literal["ordinary", "ini_continuation", "toml_multiline_array"]
    line_count: int

    @property
    def uses_logical_block_exception(self) -> bool:
        return self.kind != "ordinary"


def _indent_width(line: str) -> int:
    prefix = line[: len(line) - len(line.lstrip(" \t"))]
    # Tabs are legal in some INI readers.  A fixed expansion makes the
    # comparison deterministic without attempting to execute a parser.
    return len(prefix.expandtabs(8))


def _is_ini_cfg(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        pure.suffix.casefold() in {".ini", ".cfg"}
        or pure.name.casefold() in _INI_CFG_NAMES
    )


def _complete_ini_continuation(
    unit_lines: Sequence[str],
    start_index: int,
    end_index_exclusive: int,
) -> bool:
    selected = unit_lines[start_index:end_index_exclusive]
    assignment = _INI_ASSIGNMENT_RE.match(selected[0])
    if assignment is None or selected[0].lstrip().startswith(("#", ";")):
        return False
    assignment_indent = _indent_width(selected[0])
    saw_value_line = False
    for line in selected[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_RE.match(line):
            return False
        nested_assignment = _INI_ASSIGNMENT_RE.match(line)
        if (
            nested_assignment is not None
            and _indent_width(line) <= assignment_indent
        ):
            return False
        if _indent_width(line) <= assignment_indent:
            return False
        if not stripped.startswith(("#", ";")):
            saw_value_line = True
    if not saw_value_line:
        return False

    # A long selection must end at the end of the continuation value rather
    # than truncating a still-indented tail.  Blank lines are skipped because
    # ConfigParser-style values may continue after them.
    for following in unit_lines[end_index_exclusive:]:
        if not following.strip():
            continue
        if _SECTION_RE.match(following):
            break
        following_assignment = _INI_ASSIGNMENT_RE.match(following)
        if (
            following_assignment is not None
            and _indent_width(following) <= assignment_indent
        ):
            break
        if _indent_width(following) > assignment_indent:
            return False
        break
    return bool(selected[-1].strip())


def _complete_toml_multiline_array(selected: Sequence[str]) -> bool:
    if not _TOML_ARRAY_START_RE.match(selected[0]):
        return False
    if not _TOML_ARRAY_END_RE.match(selected[-1]):
        return False
    for line in selected[1:-1]:
        if _SECTION_RE.match(line) or _TOML_ASSIGNMENT_RE.match(line):
            return False
    try:
        parsed = tomllib.loads("".join(selected))
    except (tomllib.TOMLDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and len(parsed) == 1


def validate_evidence_range_shape(
    *,
    path: str,
    unit_lines: Sequence[str],
    start_index: int,
    end_index_exclusive: int,
) -> EvidenceRangeShape:
    """Validate and classify one inclusive/exclusive evidence selection.

    Callers pass the largest immutable line context available. Extraction first
    checks the complete unit, then invokes full-source grounding before success
    caching; later adjudication and replay invoke that same grounding path.
    """

    if not 0 <= start_index < end_index_exclusive <= len(unit_lines):
        raise EvidenceRangePolicyError("evidence indices are outside the unit")
    line_count = end_index_exclusive - start_index
    if line_count <= DEFAULT_EVIDENCE_LINE_LIMIT:
        return EvidenceRangeShape(kind="ordinary", line_count=line_count)
    if line_count > MAX_LOGICAL_EVIDENCE_LINES:
        raise EvidenceRangePolicyError(
            f"evidence range has {line_count} physical lines; the absolute "
            f"structured-block maximum is {MAX_LOGICAL_EVIDENCE_LINES}"
        )

    selected = unit_lines[start_index:end_index_exclusive]
    if _is_ini_cfg(path) and _complete_ini_continuation(
        unit_lines,
        start_index,
        end_index_exclusive,
    ):
        return EvidenceRangeShape(
            kind="ini_continuation",
            line_count=line_count,
        )
    if (
        PurePosixPath(path).suffix.casefold() == ".toml"
        and _complete_toml_multiline_array(selected)
    ):
        return EvidenceRangeShape(
            kind="toml_multiline_array",
            line_count=line_count,
        )
    raise EvidenceRangePolicyError(
        f"evidence range has {line_count} physical lines; ordinary evidence "
        f"is limited to {DEFAULT_EVIDENCE_LINE_LIMIT}, and a longer range "
        "must be exactly one complete INI/CFG continuation assignment or "
        "TOML multiline array"
    )
