"""Static allowlist and deterministic dependency planning for observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class QuerySpec:
    name: str
    version: str
    dependencies: tuple[str, ...] = ()


QUERY_REGISTRY: dict[str, QuerySpec] = {
    "python_snapshot": QuerySpec("python_snapshot", "1.4"),
    "python_clone_clusters": QuerySpec(
        "python_clone_clusters",
        "1.1",
        ("python_snapshot",),
    ),
}


class QueryRegistryError(ValueError):
    """The configured static query graph is invalid."""


def resolve_query_plan(
    configured_names: Iterable[str],
    *,
    registry: Mapping[str, QuerySpec] = QUERY_REGISTRY,
) -> tuple[QuerySpec, ...]:
    names = list(configured_names)
    if len(names) != len(set(names)):
        raise QueryRegistryError("observation query names must be unique")
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise QueryRegistryError(f"unknown observation queries: {unknown}")
    enabled = set(names)
    for name in names:
        missing = sorted(set(registry[name].dependencies) - enabled)
        if missing:
            raise QueryRegistryError(
                f"observation query {name} is missing dependencies: {missing}"
            )

    remaining = set(names)
    ordered: list[QuerySpec] = []
    completed: set[str] = set()
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if set(registry[name].dependencies) <= completed
        )
        if not ready:
            raise QueryRegistryError(
                "observation query dependency cycle detected"
            )
        for name in ready:
            for dependency in registry[name].dependencies:
                if dependency not in registry:
                    raise QueryRegistryError(
                        f"query {name} has unknown dependency {dependency}"
                    )
            ordered.append(registry[name])
            completed.add(name)
            remaining.remove(name)
    return tuple(ordered)
