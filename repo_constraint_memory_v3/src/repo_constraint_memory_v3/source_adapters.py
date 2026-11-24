"""Typed, non-executing source adapters for recovered evidence spans."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.evidence_validation import (
    EvidenceRangePolicyError,
    MAX_LOGICAL_EVIDENCE_LINES,
    validate_evidence_range_shape,
)
from repo_constraint_memory_v3.models import (
    AdapterKind,
    CandidateEvidenceRange,
    Evidence,
    EvidenceSpan,
    EvidenceType,
    EvidenceUnit,
)
from repo_constraint_memory_v3.spans import SpanRecoveryError, recover_span_range

_HARD_MODAL_RE = re.compile(
    r"\b(must|required|shall|do not|don't|never)\b",
    re.IGNORECASE,
)
_SOFT_MODAL_RE = re.compile(
    r"\b(should|recommended|recommend|prefer|preferred)\b",
    re.IGNORECASE,
)
_NEGATED_MODAL_RE = re.compile(
    r"\b(?:not\s+(?:required|recommended|necessary)|need\s+not|"
    r"does\s+not\s+need\s+to|optional)\b",
    re.IGNORECASE,
)
_HARD_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:always\s+|never\s+)?"
    r"(?:add|avoid|check|document|ensure|follow|keep|maintain|preserve|"
    r"read|re-?read|report|run|test|update|use|validate|write)\b"
)
_SOFT_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?"
    r"(?:consider|please|prefer|try)\b"
)
_DIRECT_POLICY_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+)?(?:all\s+)?"
    r"(?:commands?|contributors?|developers?|maintainers?|changes?|code|tests?)\s+"
    r"(?:are\s+(?:expected|required)\s+to|need(?:s)?\s+to|"
    r"use|run|include|preserve|follow|require)\b"
)
_DOCUMENTED_COMMAND_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)`[^`\n]+`\s*"
    r"(?:[-:—]\s*)(?:run|execute|test|check|build|validate)\b"
)
_CONTRIBUTOR_CONTEXT_RE = re.compile(
    r"\b(contribut(?:e|or|ing)|developer|maintainer|pull request|"
    r"coding style|development|testing|documentation|change)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_MARKDOWN_HEADING_RE = re.compile(
    r"^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$"
)
_RST_HEADING_RE = re.compile(r"^\s*[=\-~^\"'`:+*#<>_]{3,}\s*$")
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_TEST_HEADING_RE = re.compile(r"\b(?:pytest|tests?|testing)\b", re.IGNORECASE)
_PROSE_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|\d+[.)]|#\.)\s+(?P<body>.*?)\s*$"
)
_ASSIGNMENT_RE = re.compile(
    r"^\s*[\"']?([A-Za-z0-9_.-]+)[\"']?\s*[:=]\s*(.*?)\s*$"
)
_INI_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:[#;].*)?$")
_TOML_KEY_PART = r'(?:[A-Za-z0-9_-]+|"[^"\n]+"|\'[^\'\n]+\')'
_TOML_ASSIGNMENT_RE = re.compile(
    rf"^\s*(?P<key>{_TOML_KEY_PART}(?:\s*\.\s*{_TOML_KEY_PART})*)"
    r"\s*=\s*(?P<value>.*?)\s*$"
)
_TOML_SECTION_RE = re.compile(r"^\s*\[\[?(.+?)\]\]?\s*(?:#.*)?$")
_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.%/@+,-][^:=#]*):(?!=)")
_WORKFLOW_LITERAL_RE = re.compile(r"^\s*(?:-\s*)?(run|uses):\s*(.*?)\s*$")
_WORKFLOW_PREDICATE_RE = re.compile(r"^\s*(on|if):\s*(.*?)\s*$")
_YAML_SEQUENCE_ITEM_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_YAML_MAPPING_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.-]+):\s*(?P<value>.*?)\s*$"
)
_YAML_CONFIG_MAPPING_RE = re.compile(
    r"^(?P<indent>\s*)(?P<sequence>-\s+)?"
    r"(?P<key>[A-Za-z0-9_.-]+|\"[^\"\n]+\"|'[^'\n]+'):\s*"
    r"(?P<value>.*?)\s*$"
)
_MARKER_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")
_RUFF_KEYS = {
    "line-length",
    "target-version",
    "select",
    "ignore",
    "extend-select",
    "extend-ignore",
}
_MAX_EXAMPLE_CONTEXT_LINES = 32


class AdapterError(ValueError):
    """Recovered text is not an active literal for the selected adapter."""


@dataclass(frozen=True)
class GroundedEvidence:
    evidence: Evidence
    semantic_flags: frozenset[str]
    contributor_context: bool = False
    hard_modal: bool = False
    soft_modal: bool = False
    hard_directive: bool = False
    soft_directive: bool = False
    example_context: bool = False


# Private, run-local successful computations only. Canonical unit/range values
# bind mutable DTOs; SourceFile is immutable and remains owned by the build.
GroundingMemo = dict[tuple[int, str, str], GroundedEvidence]


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def adapter_for(path: str, source_kind: str) -> tuple[AdapterKind, EvidenceType]:
    pure = PurePosixPath(path)
    name = pure.name.casefold()
    suffix = pure.suffix.casefold()
    lowered = path.casefold()
    if lowered.startswith(".github/workflows/") and suffix in {
        ".yaml",
        ".yml",
        ".json",
    }:
        return "workflow", "workflow_step"
    if name in {
        "makefile",
        "gnumakefile",
        "justfile",
        "taskfile",
        "taskfile.yml",
        "taskfile.yaml",
    } or source_kind == "command":
        return "make_just_task", "command_definition"
    if suffix == ".toml":
        return "toml", "configuration_entry"
    if suffix in {".ini", ".cfg"} or name in {
        ".editorconfig",
        ".flake8",
        ".coveragerc",
    }:
        return "ini_cfg", "configuration_entry"
    if suffix in {".yaml", ".yml", ".json"}:
        return "yaml_json", "configuration_entry"
    if suffix in {".md", ".markdown", ".mdx", ".rst", ".txt"} or source_kind in {
        "instruction",
        "readme",
        "documentation",
    }:
        return "prose", "direct_declaration"
    raise AdapterError(f"no typed source adapter for {path}")


def _heading_before(source: SourceFile, line_number: int) -> str:
    hierarchy = _heading_hierarchy_before(source, line_number)
    return hierarchy[-1] if hierarchy else ""


def _rst_adornment(line: str) -> str | None:
    """Return one valid repeated RST/Setext adornment character."""

    stripped = line.strip()
    if len(stripped) < 3 or _RST_HEADING_RE.match(line) is None:
        return None
    return stripped[0] if len(set(stripped)) == 1 else None


def _heading_hierarchy_before(
    source: SourceFile,
    line_number: int,
) -> tuple[str, ...]:
    """Recover the active Markdown/RST heading hierarchy before one line.

    Markdown heading levels are explicit.  RST heading levels are determined
    by the document's first-seen adornment order, as required by RST's local
    convention.  Fenced Markdown content is skipped so repository examples do
    not masquerade as section structure.
    """

    lines = source.content.splitlines()
    limit = min(max(line_number - 1, 0), len(lines))
    hierarchy: dict[int, str] = {}
    rst_levels: dict[str, int] = {}
    fence: tuple[str, int] | None = None
    suffix = PurePosixPath(source.path).suffix.casefold()

    def set_heading(level: int, title: str) -> None:
        for existing in [item for item in hierarchy if item >= level]:
            del hierarchy[existing]
        hierarchy[level] = title.strip()

    for index in range(limit):
        line = lines[index]
        fence_match = _FENCE_RE.match(line)
        if fence_match is not None:
            marker = fence_match.group("fence")
            marker_key = marker[0]
            if fence is None:
                fence = (marker_key, len(marker))
            elif marker_key == fence[0] and len(marker) >= fence[1]:
                fence = None
            continue
        if fence is not None:
            continue

        markdown = _MARKDOWN_HEADING_RE.match(line)
        if markdown is not None:
            set_heading(
                len(markdown.group("marks")),
                markdown.group("title"),
            )
            continue

        if index + 1 >= limit or not line.strip():
            continue
        adornment = _rst_adornment(lines[index + 1])
        if adornment is None or len(lines[index + 1].strip()) < len(line.strip()):
            continue
        if suffix in {".md", ".markdown", ".mdx"} and adornment in {"=", "-"}:
            level = 1 if adornment == "=" else 2
        else:
            level = rst_levels.setdefault(adornment, len(rst_levels) + 1)
        set_heading(level, line.strip())

    return tuple(hierarchy[level] for level in sorted(hierarchy))


def _test_section_context(source: SourceFile, line_number: int) -> bool:
    """Return whether the active typed heading hierarchy is test-specific."""

    return any(
        _TEST_HEADING_RE.search(heading) is not None
        for heading in _heading_hierarchy_before(source, line_number)
    )


def _context_window(source: SourceFile, line_start: int, line_end: int) -> str:
    lines = source.content.splitlines()
    start = max(line_start - 4, 0)
    end = min(line_end + 3, len(lines))
    return "\n".join(lines[start:end])


def _example_parent(line: str) -> tuple[int, bool] | None:
    """Return the indentation and list shape of an ``Example(s):`` label."""

    expanded = line.expandtabs(4)
    match = _PROSE_LIST_ITEM_RE.match(expanded)
    if match is None:
        indent = len(expanded) - len(expanded.lstrip())
        body = expanded.strip()
        is_list = False
    else:
        indent = len(match.group("indent"))
        body = match.group("body").strip()
        is_list = True
    # Common Markdown/RST emphasis is presentation, not part of the label.
    normalized = body.strip("*_` ").casefold()
    if normalized not in {"example:", "examples:"}:
        return None
    return indent, is_list


def _local_label_boundary(line: str, child_indent: int) -> bool:
    """Return whether a same/outer-level prose label starts a new local block."""

    expanded = line.expandtabs(4)
    if _PROSE_LIST_ITEM_RE.match(expanded) is not None:
        return False
    indent = len(expanded) - len(expanded.lstrip())
    stripped = expanded.strip()
    if indent > child_indent or not stripped.endswith(":"):
        return False
    # A colon inside an ordinary sentence is not a structural boundary.  A
    # terminal colon, by contrast, introduces the following Markdown/RST block.
    return bool(stripped[:-1].strip())


def _nested_example_context(
    source: SourceFile,
    line_start: int,
    line_end: int,
) -> bool:
    """Detect selected list items nested under a nearby example label.

    The backwards search is deliberately bounded and stops at a section/fence
    boundary.  This makes the flag depend on local source syntax without
    treating an earlier, unrelated ``Examples:`` paragraph as global context.
    """

    lines = source.content.splitlines()
    selected_start = max(line_start - 1, 0)
    selected_end = min(line_end, len(lines))
    for child_index in range(selected_start, selected_end):
        child = _PROSE_LIST_ITEM_RE.match(lines[child_index].expandtabs(4))
        if child is None:
            continue
        child_indent = len(child.group("indent"))
        lower_bound = max(0, child_index - _MAX_EXAMPLE_CONTEXT_LINES)
        for parent_index in range(child_index - 1, lower_bound - 1, -1):
            parent_line = lines[parent_index]
            parent = _example_parent(parent_line)
            if parent is not None:
                parent_indent, parent_is_list = parent
                if child_indent > parent_indent or (
                    not parent_is_list and child_indent >= parent_indent
                ):
                    return True
                break
            intervening_list = _PROSE_LIST_ITEM_RE.match(
                parent_line.expandtabs(4)
            )
            if (
                intervening_list is not None
                and len(intervening_list.group("indent")) < child_indent
            ):
                # This is the child's nearer syntactic list parent.  An older
                # Example(s) sibling/ancestor cannot label the selected item.
                break
            if _local_label_boundary(parent_line, child_indent):
                break
            stripped = parent_line.strip()
            if (
                _HEADING_RE.match(parent_line)
                or _RST_HEADING_RE.match(parent_line)
                or stripped.startswith("```")
                or stripped.startswith("~~~")
            ):
                break
    return False


def _toml_section_and_assignment(
    source: SourceFile,
    line_start: int,
    quote: str,
) -> tuple[str, str, str]:
    lines = source.content.splitlines()
    section = ""
    for line in lines[: max(line_start - 1, 0)]:
        section_match = _TOML_SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1).strip()

    quote_lines = quote.splitlines()
    assignments: list[tuple[int, str, str]] = []
    for index, line in enumerate(quote_lines):
        section_match = _TOML_SECTION_RE.match(line)
        if section_match:
            if assignments:
                raise AdapterError(
                    "selected TOML span crosses a configuration section boundary"
                )
            section = section_match.group(1).strip()
            continue
        assignment = _TOML_ASSIGNMENT_RE.match(line)
        if assignment and not line.lstrip().startswith(("#", ";")):
            assignments.append(
                (
                    index,
                    assignment.group("key").replace(" ", "").strip("\"'"),
                    assignment.group("value").strip(),
                )
            )
    if not assignments:
        raise AdapterError("selected TOML span has no active assignment")
    if len(assignments) != 1:
        raise AdapterError(
            "selected TOML evidence range must contain exactly one active assignment"
        )
    assignment_index, key, value = assignments[0]
    # Preserve the body of a selected multiline array/string in its typed
    # identity. Multiple peer assignments and section crossings were rejected
    # above, so these lines can only belong to this selected logical value.
    continuation = [
        line.strip()
        for line in quote_lines[assignment_index + 1 :]
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if continuation:
        value = "\n".join(filter(None, (value, *continuation)))
    if not value:
        raise AdapterError("selected TOML assignment has no literal value")
    return section, key, value


def _yaml_section_and_assignments(
    quote: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Recover one local YAML/JSON mapping without crossing list siblings.

    Sequence mappings such as ``- id: hook`` are active assignments in their
    own right.  Only syntax inside the selected range is used, so a selection
    can never inherit ``args`` or another value from a preceding sibling.
    """

    lines = quote.splitlines()
    mappings: list[tuple[int, int, bool, str, str]] = []
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        match = _YAML_CONFIG_MAPPING_RE.match(line)
        if match is None:
            continue
        key = match.group("key").strip("\"'")
        mappings.append(
            (
                index,
                len(match.group("indent").expandtabs(8)),
                match.group("sequence") is not None,
                key,
                match.group("value").strip().rstrip(","),
            )
        )

    sequence_roots = [item for item in mappings if item[2]]
    if sequence_roots:
        sequence_counts: dict[int, int] = {}
        for _, indent, _, _, _ in sequence_roots:
            sequence_counts[indent] = sequence_counts.get(indent, 0) + 1
        if any(count > 1 for count in sequence_counts.values()):
            raise AdapterError(
                "selected YAML evidence range crosses a sequence-item boundary"
            )
        root_indent = min(item[1] for item in sequence_roots)
        root_items = [item for item in sequence_roots if item[1] == root_indent]
        if len(root_items) != 1:
            raise AdapterError(
                "selected YAML evidence range crosses a sequence-item boundary"
            )
        root_index = root_items[0][0]
        mappings = [
            item
            for item in mappings
            if item[0] >= root_index and item[1] >= root_indent
        ]

    mapping_literal_entries = [
        (key, value)
        for _, _, _, key, value in mappings
        if value not in {"", "|", ">", "|-", ">-", "|+", ">+"}
    ]
    literal_entries = list(mapping_literal_entries)
    # Scalar list values belong to their selected parent assignment. They are
    # not sibling mapping objects, so a cohesive list may contain many values.
    for index, line in enumerate(lines):
        item = _YAML_SEQUENCE_ITEM_RE.match(line)
        if item is None or _YAML_CONFIG_MAPPING_RE.match(line) is not None:
            continue
        item_indent = len(line.expandtabs(8)) - len(line.lstrip().expandtabs(8))
        parent = next(
            (
                key
                for parent_index, parent_indent, _, key, value in reversed(
                    mappings
                )
                if parent_index < index
                and parent_indent < item_indent
                and not value
            ),
            "",
        )
        value = item.group(1).strip()
        if parent and value and not value.startswith("#"):
            literal_entries.append((parent, value))
    if not literal_entries:
        raise AdapterError("selected YAML/JSON span has no active assignment")
    if not sequence_roots and (
        len(mapping_literal_entries) > 1
        or len({key for key, _ in literal_entries}) != 1
    ):
        raise AdapterError(
            "selected YAML/JSON evidence range must contain exactly one "
            "active assignment"
        )

    first_sequence_index = sequence_roots[0][0] if sequence_roots else len(lines)
    context_keys = [
        key
        for index, _, is_sequence, key, value in mappings
        if index < first_sequence_index and not value and not is_sequence
    ]
    section = ".".join(context_keys)
    if sequence_roots:
        section = ".".join(filter(None, (section, "sequence_item")))
    return section, literal_entries


def _ini_section_and_assignment(
    source: SourceFile,
    line_start: int,
    quote: str,
) -> tuple[str, str, str]:
    lines = source.content.splitlines()
    section = ""
    for line in lines[: max(line_start - 1, 0)]:
        section_match = _INI_SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1).strip()

    quote_lines = quote.splitlines()
    active: tuple[int, int, str, str] | None = None
    for index, line in enumerate(quote_lines):
        section_match = _INI_SECTION_RE.match(line)
        if section_match:
            if active is not None:
                raise AdapterError(
                    "selected INI/CFG span crosses a configuration section boundary"
                )
            section = section_match.group(1).strip()
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if not assignment or line.lstrip().startswith(("#", ";")):
            continue
        indent = len(line) - len(line.lstrip())
        if active is not None and indent <= active[1]:
            raise AdapterError(
                "selected INI/CFG evidence range must contain exactly one "
                "active assignment"
            )
        if active is not None:
            # Assignment-like text nested inside a continuation is literal
            # continuation content, not a second peer assignment.
            continue
        key = assignment.group(1)
        value = assignment.group(2).strip()
        active = (index, indent, key, value)
    if active is None:
        raise AdapterError("selected INI/CFG span has no active assignment")

    index, assignment_indent, key, value = active
    continuation: list[str] = []
    for following in quote_lines[index + 1 :]:
        if not following.strip() or following.lstrip().startswith(("#", ";")):
            continue
        indent = len(following) - len(following.lstrip())
        if indent <= assignment_indent:
            raise AdapterError(
                "selected INI/CFG evidence range contains text outside its "
                "active assignment"
            )
        continuation.append(following.strip())
    if continuation:
        value = "\n".join(filter(None, (value, *continuation)))
    if not value:
        raise AdapterError("selected INI/CFG assignment has no literal value")

    selected_end = max(line_start - 1, 0) + len(quote_lines)
    for following in lines[selected_end:]:
        if not following.strip() or following.lstrip().startswith(("#", ";")):
            continue
        if _INI_SECTION_RE.match(following):
            break
        if len(following) - len(following.lstrip()) > assignment_indent:
            raise AdapterError(
                "selected INI/CFG evidence range truncates its active assignment"
            )
        break
    return section, key, value


def _expand_truncated_ini_assignment(
    source: SourceFile,
    unit: EvidenceUnit,
    spans: list[EvidenceSpan],
) -> list[EvidenceSpan]:
    """Expand a selected INI assignment to its bounded physical atom.

    Models occasionally select only the assignment line (or an initial prefix)
    of a ConfigParser-style continuation value.  The immutable source makes
    the missing tail deterministic.  We expand only when the selection starts
    at one assignment and remains inside that assignment; all other shapes are
    left for the normal fail-closed adapter checks.
    """

    first = spans[0].text
    assignment = _ASSIGNMENT_RE.match(first)
    if assignment is None or first.lstrip().startswith(("#", ";")):
        return spans
    assignment_indent = len(first) - len(first.lstrip())

    for line in (span.text for span in spans[1:]):
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        if _INI_SECTION_RE.match(line):
            return spans
        indent = len(line) - len(line.lstrip())
        if indent <= assignment_indent:
            return spans

    source_lines = source.content.splitlines(keepends=True) or [source.content]
    assignment_index = spans[0].line_number - 1
    selected_end_index = spans[-1].line_number - 1
    if not 0 <= assignment_index <= selected_end_index < len(source_lines):
        return spans

    logical_end_index = assignment_index
    for index in range(assignment_index + 1, len(source_lines)):
        line = source_lines[index]
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        if _INI_SECTION_RE.match(line):
            break
        indent = len(line) - len(line.lstrip())
        if indent <= assignment_indent:
            break
        logical_end_index = index

    if logical_end_index <= selected_end_index:
        return spans

    spans_by_line = {span.line_number: span for span in unit.spans}
    expanded: list[EvidenceSpan] = []
    for line_number in range(assignment_index + 1, logical_end_index + 2):
        span = spans_by_line.get(line_number)
        if span is None:
            raise AdapterError(
                "truncated INI/CFG assignment cannot be expanded within its "
                "evidence unit"
            )
        expanded.append(span)
    return expanded


def _marker_names(source: SourceFile, adapter: AdapterKind) -> list[str]:
    text = source.content
    results: list[str] = []
    if adapter == "toml":
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10
                import tomli as tomllib  # type: ignore[no-redef]

            parsed = tomllib.loads(text)
            values = (
                parsed.get("tool", {})
                .get("pytest", {})
                .get("ini_options", {})
                .get("markers", [])
            )
            if isinstance(values, str):
                values = values.splitlines()
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        match = _MARKER_NAME_RE.match(value)
                        if match:
                            results.append(match.group(1))
        except (ValueError, TypeError):
            return []
    else:
        in_pytest = False
        collecting = False
        for line in text.splitlines():
            section = _INI_SECTION_RE.match(line)
            if section:
                in_pytest = section.group(1).casefold() in {
                    "pytest",
                    "tool:pytest",
                }
                collecting = False
                continue
            if not in_pytest:
                continue
            assignment = _ASSIGNMENT_RE.match(line)
            if assignment:
                collecting = assignment.group(1).casefold() == "markers"
                if collecting and assignment.group(2).strip():
                    match = _MARKER_NAME_RE.match(assignment.group(2))
                    if match:
                        results.append(match.group(1))
                continue
            if collecting and line[:1].isspace():
                match = _MARKER_NAME_RE.match(line)
                if match:
                    results.append(match.group(1))
            elif line.strip():
                collecting = False
    return sorted(set(results))


def _derive_prose(
    source: SourceFile,
    quote: str,
    line_start: int,
    line_end: int,
) -> tuple[str, set[str], bool, bool, bool, bool, bool]:
    if not quote.strip() or all(
        not line.strip() or _HEADING_RE.match(line)
        for line in quote.splitlines()
    ):
        raise AdapterError("selected prose span is empty or heading-only")
    heading = _heading_before(source, line_start)
    context = _context_window(source, line_start, line_end)
    name = PurePosixPath(source.path).name.casefold()
    contributor_context = (
        source.source_kind == "instruction"
        or "contribut" in name
        or name in {"developers.md", "development.md"}
        or bool(_CONTRIBUTOR_CONTEXT_RE.search(heading))
        or bool(_CONTRIBUTOR_CONTEXT_RE.search(context))
    )
    modal_quote = _NEGATED_MODAL_RE.sub("", quote)
    hard = bool(_HARD_MODAL_RE.search(modal_quote))
    soft = bool(_SOFT_MODAL_RE.search(modal_quote))
    documented_command_recipe = bool(
        _DOCUMENTED_COMMAND_DIRECTIVE_RE.search(quote)
    )
    hard_directive = bool(
        _HARD_DIRECTIVE_RE.search(quote)
        or _DIRECT_POLICY_RE.search(quote)
    )
    soft_directive = bool(_SOFT_DIRECTIVE_RE.search(quote))
    flags = {
        f"heading:{_normal(heading)}" if heading else "heading:none",
        f"modal:{'hard' if hard else 'soft' if soft else 'none'}",
        (
            "directive:hard"
            if hard_directive
            else "directive:soft"
            if soft_directive
            else "directive:none"
        ),
        f"contributor_context:{str(contributor_context).lower()}",
    }
    if documented_command_recipe:
        flags.add("documented_command_recipe")
    if _nested_example_context(source, line_start, line_end):
        flags.add("example_context")
    if _test_section_context(source, line_start):
        flags.add("test_section_context")
    return (
        f"declaration:{_normal(quote)}",
        flags,
        contributor_context,
        hard,
        soft,
        hard_directive,
        soft_directive,
    )


def _derive_configuration(
    source: SourceFile,
    adapter: AdapterKind,
    quote: str,
    line_start: int,
) -> tuple[str, set[str]]:
    if not quote.strip() or all(
        not line.strip() or line.lstrip().startswith(("#", ";"))
        for line in quote.splitlines()
    ):
        raise AdapterError("selected configuration span is inactive")
    if adapter == "ini_cfg":
        section, key, value = _ini_section_and_assignment(
            source,
            line_start,
            quote,
        )
        entries = [(key, value)]
    elif adapter == "toml":
        section, key, value = _toml_section_and_assignment(
            source,
            line_start,
            quote,
        )
        entries = [(key, value)]
    else:
        section, entries = _yaml_section_and_assignments(quote)
        key, value = entries[0]
    section_normal = _normal(section)
    key_normal = key.casefold()
    flags = {
        f"config_section:{section_normal}",
        *(f"config_key:{item_key.casefold()}" for item_key, _ in entries),
    }
    is_pytest_markers = key_normal == "markers" and (
        "pytest" in section_normal
        or PurePosixPath(source.path).name.casefold()
        in {"pytest.ini", "setup.cfg"}
    )
    if is_pytest_markers:
        flags.add("pytest_marker_registration")
        flags.update(f"pytest_marker:{name}" for name in _marker_names(source, adapter))
    if (
        "ruff" in section_normal
        or PurePosixPath(source.path).name.casefold() == "ruff.toml"
    ) and key_normal in _RUFF_KEYS:
        flags.add("allowlisted_enforcement")
    normalized_entries = "|".join(
        f"{item_key.casefold()}={_normal(item_value)}"
        for item_key, item_value in entries
    )
    semantic_key = f"{adapter}:{section_normal or '<root>'}.{normalized_entries}"
    return semantic_key, flags


def _derive_command(quote: str) -> tuple[str, set[str]]:
    lines = quote.splitlines()
    target_name = ""
    command = ""
    for line in lines:
        target = _MAKE_TARGET_RE.match(line)
        if target:
            target_name = target.group(1).strip()
            continue
        if target_name and line.startswith(("\t", "    ")):
            candidate = line.strip().lstrip("@")
            if candidate and not candidate.startswith("#"):
                command = candidate
                break
    if not target_name or not command:
        raise AdapterError(
            "selected command span must contain an active target and body"
        )
    flags = {f"command_target:{_normal(target_name)}", f"command:{_normal(command)}"}
    return f"command:{_normal(target_name)}:{_normal(command)}", flags


def _taskfile_task_and_command(
    lines: list[str],
) -> tuple[str, str] | None:
    mappings: list[tuple[int, int, str, str]] = []
    for index, line in enumerate(lines):
        match = _YAML_MAPPING_RE.match(line)
        if match is None or line.lstrip().startswith("#"):
            continue
        mappings.append(
            (
                index,
                len(match.group("indent")),
                match.group("key"),
                match.group("value").strip(),
            )
        )
    for index, indent, key, value in mappings:
        if key.casefold() not in {"cmd", "cmds"}:
            continue
        parent = next(
            (
                parent_key
                for parent_index, parent_indent, parent_key, _ in reversed(mappings)
                if parent_index < index
                and parent_indent < indent
                and parent_key.casefold() != "tasks"
            ),
            "",
        )
        command = value
        if not command:
            for following in lines[index + 1 :]:
                if not following.strip() or following.lstrip().startswith("#"):
                    continue
                following_indent = len(following) - len(following.lstrip())
                if following_indent <= indent:
                    break
                item = _YAML_SEQUENCE_ITEM_RE.match(following)
                if item is None:
                    continue
                command = item.group(1).strip()
                if command.casefold().startswith("cmd:"):
                    command = command.split(":", 1)[1].strip()
                if command:
                    break
        if parent and command:
            return parent, command
    return None


def taskfile_range_has_active_task_body(lines: list[str]) -> bool:
    """Return whether selected Taskfile YAML contains a task and command."""

    return _taskfile_task_and_command(lines) is not None


def _derive_taskfile(quote: str) -> tuple[str, set[str]]:
    parsed = _taskfile_task_and_command(quote.splitlines())
    if parsed is None:
        raise AdapterError(
            "selected Taskfile span must contain a task and active command body"
        )
    target_name, command = parsed
    flags = {
        f"command_target:{_normal(target_name)}",
        f"command:{_normal(command)}",
    }
    return f"command:{_normal(target_name)}:{_normal(command)}", flags


def _indented_block_value(lines: list[str], index: int) -> str:
    base_indent = len(lines[index]) - len(lines[index].lstrip())
    values: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        stripped = line.strip()
        if not stripped.startswith("#"):
            values.append(stripped)
    return "\n".join(values)


def _derive_workflow(source: SourceFile, quote: str) -> tuple[str, set[str]]:
    lines = quote.splitlines()
    actions: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = _WORKFLOW_LITERAL_RE.match(line)
        if match is None:
            continue
        key = match.group(1).casefold()
        command = match.group(2).strip()
        if key == "run" and command in {"|", ">", "|-", ">-", "|+", ">+"}:
            command = _indented_block_value(lines, index)
        if command:
            actions.append((key, command))

    predicates: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = _WORKFLOW_PREDICATE_RE.match(line)
        if match is None:
            continue
        key = match.group(1).casefold()
        predicate = match.group(2).strip()
        if predicate in {"", "|", ">", "|-", ">-", "|+", ">+"}:
            predicate = _indented_block_value(lines, index)
        if predicate:
            predicates.append((key, predicate))

    if actions or predicates:
        flags = {
            *(f"workflow_key:{key}" for key, _ in actions),
            *(f"workflow_key:{key}" for key, _ in predicates),
            *(f"command:{_normal(value)}" for _, value in actions),
            *(
                f"workflow_predicate:{_normal(value)}"
                for _, value in predicates
            ),
        }
        parts = [
            *(f"{key}:{_normal(value)}" for key, value in predicates),
            *(f"{key}:{_normal(value)}" for key, value in actions),
        ]
        return (
            f"workflow:{_normal(source.path)}:literal:{'|'.join(parts)}",
            flags,
        )
    for index, line in enumerate(lines):
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment is None or assignment.group(2).strip():
            continue
        key = assignment.group(1).casefold()
        base_indent = len(line) - len(line.lstrip())
        values: list[str] = []
        for child in lines[index + 1 :]:
            if not child.strip():
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= base_indent:
                break
            item = _YAML_SEQUENCE_ITEM_RE.match(child)
            if item is None:
                break
            value = item.group(1).strip()
            if value and not value.startswith("#"):
                values.append(value)
        if values:
            normalized = ",".join(_normal(value) for value in values)
            flags = {
                f"workflow_key:{key}",
                "workflow_sequence_literal",
            }
            return (
                f"workflow:{_normal(source.path)}:sequence:{key}:{normalized}",
                flags,
            )
    for line in lines:
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment is None or line.lstrip().startswith("#"):
            continue
        key = assignment.group(1).casefold()
        value = assignment.group(2).strip()
        if value:
            flags = {
                f"workflow_key:{key}",
                "workflow_auxiliary_literal",
            }
            return (
                f"workflow:{_normal(source.path)}:literal:{key}:{_normal(value)}",
                flags,
            )
    raise AdapterError(
        "selected workflow span has no active literal"
    )


def ground_evidence_range(
    selection: CandidateEvidenceRange,
    *,
    units_by_id: Mapping[str, EvidenceUnit],
    sources_by_path: Mapping[str, SourceFile],
    allowed_unit_ids: set[str],
    _memo: GroundingMemo | None = None,
) -> GroundedEvidence:
    if selection.unit_id not in allowed_unit_ids:
        raise AdapterError("evidence unit is outside the current extraction batch")
    unit = units_by_id.get(selection.unit_id)
    if unit is None:
        raise AdapterError("evidence references an unknown unit")
    source = sources_by_path.get(unit.path)
    if source is None:
        raise AdapterError("evidence source was not loaded from the source plan")
    memo_key = None
    if _memo is not None:
        memo_key = (id(source), unit.model_dump_json(), selection.model_dump_json())
        previous = _memo.get(memo_key)
        if previous is not None:
            return previous
    adapter, evidence_type = adapter_for(unit.path, unit.source_kind)
    try:
        spans = recover_span_range(
            selection,
            unit,
            max_lines=MAX_LOGICAL_EVIDENCE_LINES,
        )
    except SpanRecoveryError as error:
        raise AdapterError(str(error)) from error
    if adapter == "ini_cfg":
        spans = _expand_truncated_ini_assignment(source, unit, spans)
    source_lines = source.content.splitlines(keepends=True) or [source.content]
    try:
        validate_evidence_range_shape(
            path=unit.path,
            unit_lines=source_lines,
            start_index=spans[0].line_number - 1,
            end_index_exclusive=spans[-1].line_number,
        )
    except EvidenceRangePolicyError as error:
        raise AdapterError(str(error)) from error
    quote = "".join(span.text for span in spans)
    if not quote.strip():
        raise AdapterError("selected evidence is blank")
    semantic_flags: set[str]
    contributor_context = False
    hard_modal = False
    soft_modal = False
    hard_directive = False
    soft_directive = False
    if adapter == "prose":
        (
            semantic_key,
            semantic_flags,
            contributor_context,
            hard_modal,
            soft_modal,
            hard_directive,
            soft_directive,
        ) = _derive_prose(
            source,
            quote,
            spans[0].line_number,
            spans[-1].line_number,
        )
    elif adapter in {"toml", "ini_cfg", "yaml_json"}:
        semantic_key, semantic_flags = _derive_configuration(
            source,
            adapter,
            quote,
            spans[0].line_number,
        )
    elif adapter == "make_just_task":
        if PurePosixPath(unit.path).name.casefold() in {
            "taskfile.yml",
            "taskfile.yaml",
        }:
            semantic_key, semantic_flags = _derive_taskfile(quote)
        else:
            semantic_key, semantic_flags = _derive_command(quote)
    else:
        semantic_key, semantic_flags = _derive_workflow(source, quote)
    evidence = Evidence(
        type=evidence_type,
        adapter=adapter,
        semantic_key=semantic_key,
        path=unit.path,
        span_ids=[span.span_id for span in spans],
        line_start=spans[0].line_number,
        line_end=spans[-1].line_number,
        quote=quote,
        quote_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        source_content_sha256=unit.source_content_sha256,
        source_kind=unit.source_kind,
        discovery_origin=unit.discovery_origin,
    )
    grounded = GroundedEvidence(
        evidence=evidence,
        semantic_flags=frozenset(semantic_flags),
        contributor_context=contributor_context,
        hard_modal=hard_modal,
        soft_modal=soft_modal,
        hard_directive=hard_directive,
        soft_directive=soft_directive,
        example_context="example_context" in semantic_flags,
    )
    if _memo is not None and memo_key is not None:
        _memo[memo_key] = grounded
    return grounded
