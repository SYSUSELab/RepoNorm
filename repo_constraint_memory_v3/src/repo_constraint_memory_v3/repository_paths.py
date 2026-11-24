"""Shared lexical contracts for repository and generated consumer paths."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


MAX_PHYSICAL_PATH_COMPONENT_BYTES = 255
MAX_CONSUMER_ANCHOR_RELATIVE_BYTES = 2_048
_CONSUMER_RESERVED_COMPONENTS = frozenset({"index.md", "constraints.md"})
_GLOB_COMPONENT = re.compile(r"[*?\[]")


def is_safe_display_text(value: str) -> bool:
    """Return whether untrusted text has an exact safe UTF-8 Markdown form."""

    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return not any(
        (ord(character) < 32 and character not in {"\n", "\t"})
        or ord(character) == 127
        for character in value
    )


def is_canonical_repository_path(value: str) -> bool:
    """Return whether ``value`` has one safe canonical POSIX spelling."""

    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or "://" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        bool(path.parts)
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and all(
            len(part.encode("utf-8")) <= MAX_PHYSICAL_PATH_COMPONENT_BYTES
            for part in path.parts
        )
    )


def escape_consumer_path_component(component: str) -> str:
    """Return the reversible collision-free physical anchor component."""

    escaped = component.replace("%", "%25")
    if component.casefold() in _CONSUMER_RESERVED_COMPONENTS:
        escaped = f"%{ord(escaped[0]):02X}{escaped[1:]}"
    return escaped


def is_consumer_routable_repository_path(value: str) -> bool:
    """Check escaped component and complete consumer-relative path bounds."""

    if not is_canonical_repository_path(value):
        return False
    literal_anchor: list[str] = []
    for component in PurePosixPath(value).parts:
        if _GLOB_COMPONENT.search(component) is not None:
            break
        literal_anchor.append(component)
    escaped = [escape_consumer_path_component(item) for item in literal_anchor]
    if any(
        len(item.encode("utf-8")) > MAX_PHYSICAL_PATH_COMPONENT_BYTES
        for item in escaped
    ):
        return False
    relative = PurePosixPath(
        "by-path", *escaped, "constraints.md"
    ).as_posix()
    return len(relative.encode("utf-8")) <= MAX_CONSUMER_ANCHOR_RELATIVE_BYTES


def is_nonaccepted_directory_marker(value: str) -> bool:
    """Recognize one canonical consumer-routable path plus one trailing slash."""

    return value.endswith("/") and is_consumer_routable_repository_path(value[:-1])


__all__ = [
    "MAX_CONSUMER_ANCHOR_RELATIVE_BYTES",
    "MAX_PHYSICAL_PATH_COMPONENT_BYTES",
    "escape_consumer_path_component",
    "is_canonical_repository_path",
    "is_consumer_routable_repository_path",
    "is_nonaccepted_directory_marker",
    "is_safe_display_text",
]
