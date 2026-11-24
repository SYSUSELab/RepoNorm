"""Bounded target-input integrity checks for the controlled research profile."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import yaml

try:  # pragma: no cover - Python 3.10 compatibility
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

TARGET_CONTENT_ADMISSION_POLICY_VERSION = (
    "controlled-research-unredacted-target-content-1.0"
)
_MAX_STRUCTURED_VALUES = 100_000
_MAX_STRUCTURED_CONTENT_CHARS = 4_000_000
_MAX_STRUCTURED_NESTING = 256


class _JsonObjectPairs(list[tuple[str, object]]):
    """Preserve JSON member order so duplicate keys remain observable."""


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _bounded_delimiter_depth(content: str) -> bool:
    """Reject adversarial JSON/TOML nesting before recursive parsers."""

    depth = 0
    quote: str | None = None
    triple = False
    escaped = False
    index = 0
    while index < len(content):
        character = content[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote is not None:
            if character == "\\" and quote == '"':
                escaped = True
            elif triple and content.startswith(quote * 3, index):
                quote = None
                triple = False
                index += 3
                continue
            elif not triple and character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            if content.startswith(character * 3, index):
                quote = character
                triple = True
                index += 3
                continue
            quote = character
        elif character in "[{(":
            depth += 1
            if depth > _MAX_STRUCTURED_NESTING:
                return False
        elif character in "]})":
            depth = max(depth - 1, 0)
        index += 1
    return True


def _parse_yaml_nodes(content: str) -> list[yaml.nodes.Node]:
    depth = 0
    event_count = 0
    for event in yaml.parse(content, Loader=yaml.SafeLoader):
        event_count += 1
        if event_count > _MAX_STRUCTURED_VALUES:
            raise ValueError("structured event limit exceeded")
        if isinstance(
            event,
            (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent),
        ):
            depth += 1
            if depth > _MAX_STRUCTURED_NESTING:
                raise ValueError("structured nesting limit exceeded")
        elif isinstance(
            event,
            (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent),
        ):
            depth -= 1
    return list(yaml.compose_all(content, Loader=yaml.SafeLoader))


def _validate_structured_value(value: object) -> None:
    """Validate bounded JSON/TOML structure without inspecting values."""

    pending = [value]
    visited: set[int] = set()
    examined = 0
    while pending:
        current = pending.pop()
        examined += 1
        if examined > _MAX_STRUCTURED_VALUES:
            raise ValueError("structured value limit exceeded")
        if isinstance(current, _JsonObjectPairs):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            seen_keys: set[str] = set()
            for key, child in current:
                if key in seen_keys:
                    raise ValueError("duplicate structured key")
                seen_keys.add(key)
                pending.append(child)
        elif isinstance(current, dict):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(current)


def _validate_yaml_structure(nodes: list[yaml.nodes.Node]) -> None:
    """Validate bounded YAML structure and scalar-key uniqueness."""

    pending: list[yaml.nodes.Node] = list(nodes)
    visited: set[int] = set()
    examined = 0
    loader = yaml.SafeLoader("")
    try:
        while pending:
            node = pending.pop()
            identity = id(node)
            if identity in visited:
                continue
            visited.add(identity)
            examined += 1
            if examined > _MAX_STRUCTURED_VALUES:
                raise ValueError("structured value limit exceeded")
            if isinstance(node, yaml.nodes.MappingNode):
                seen_keys: set[tuple[str, str]] = set()
                for key_node, value_node in node.value:
                    if not isinstance(key_node, yaml.nodes.ScalarNode):
                        raise ValueError("complex structured key is unsupported")
                    key_value = loader.construct_object(key_node, deep=True)
                    key = (key_node.tag, repr(key_value))
                    if key in seen_keys:
                        raise ValueError("duplicate structured key")
                    seen_keys.add(key)
                    pending.extend((key_node, value_node))
            elif isinstance(node, yaml.nodes.SequenceNode):
                pending.extend(node.value)
    finally:
        loader.dispose()


def structured_content_failure_reason(
    content: str,
    *,
    path: str | None = None,
) -> str | None:
    """Return a typed integrity failure for supported structured sources.

    Credential names, token-shaped strings, private-key-shaped text, comments,
    and literal values do not affect admission. JSON, TOML, and YAML still
    fail closed on malformed input, duplicate keys, excessive nesting or value
    count, and the bounded whole-file character limit.
    """

    if path is None:
        return None
    pure = PurePosixPath(path)
    suffix = pure.suffix.casefold()
    if pure.name.casefold() == "taskfile":
        suffix = ".yaml"
    if suffix not in {".json", ".toml", ".yaml", ".yml"}:
        return None
    if len(content) > _MAX_STRUCTURED_CONTENT_CHARS:
        return "structured_content_unavailable"
    try:
        if suffix == ".json":
            if not _bounded_delimiter_depth(content):
                raise ValueError("structured nesting limit exceeded")
            parsed: object = json.loads(
                content,
                object_pairs_hook=_JsonObjectPairs,
                parse_constant=_reject_json_constant,
            )
            _validate_structured_value(parsed)
        elif suffix == ".toml":
            if not _bounded_delimiter_depth(content):
                raise ValueError("structured nesting limit exceeded")
            _validate_structured_value(tomllib.loads(content))
        else:
            _validate_yaml_structure(_parse_yaml_nodes(content))
    except (RecursionError, ValueError, TypeError, yaml.YAMLError):
        return "structured_content_unavailable"
    return None
