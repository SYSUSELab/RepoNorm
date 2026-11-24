"""Deterministic component-fair selection over a frozen finite universe.

The selector is intentionally semantic-free.  Callers provide stable item and
component identities, while this module derives a snapshot-bound hash order
inside each component and folds components in round-robin order.  Provider
completion order, input iteration order, and process hash randomization never
participate in the result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar


FAIR_SELECTION_POLICY_VERSION = "component-fair-stable-hash-1.0"

_T = TypeVar("_T")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FairSelectionResult(Generic[_T]):
    """One complete deterministic selection ledger."""

    selected: tuple[_T, ...]
    omitted: tuple[_T, ...]
    selected_keys: tuple[str, ...]
    omitted_keys: tuple[str, ...]
    universe_count: int
    selected_count: int
    omitted_count: int
    seed_digest: str
    universe_digest: str
    selected_digest: str
    omitted_digest: str
    component_universe_counts: dict[str, int]
    component_selected_counts: dict[str, int]


def component_fair_stable_select(
    values: Sequence[_T],
    *,
    limit: int,
    seed: object,
    key: Callable[[_T], str],
    component: Callable[[_T], str],
) -> FairSelectionResult[_T]:
    """Select at most ``limit`` unique values with stable component fairness.

    Components are visited lexicographically in repeated rounds.  Within a
    component, values are ordered by SHA-256 over the frozen seed, component,
    and item identity, with the identity itself as the collision tie-break.
    The returned selected tuple is the exact fair planning order.
    """

    if limit < 0:
        raise ValueError("fair selection limit must be non-negative")
    by_key: dict[str, _T] = {}
    component_by_key: dict[str, str] = {}
    for value in values:
        item_key = key(value)
        item_component = component(value)
        if not item_key or not item_component:
            raise ValueError("fair selection identities must be non-empty")
        if item_key in by_key:
            raise ValueError("fair selection item identities must be unique")
        by_key[item_key] = value
        component_by_key[item_key] = item_component

    seed_digest = _canonical_digest(
        {
            "policy": FAIR_SELECTION_POLICY_VERSION,
            "seed": seed,
        }
    )
    buckets: dict[str, list[str]] = {}
    for item_key, item_component in component_by_key.items():
        buckets.setdefault(item_component, []).append(item_key)
    for item_component, item_keys in buckets.items():
        item_keys.sort(
            key=lambda item_key: (
                _canonical_digest(
                    {
                        "seed_digest": seed_digest,
                        "component": item_component,
                        "item_key": item_key,
                    }
                ),
                item_key,
            )
        )

    ordered_keys: list[str] = []
    active = sorted(buckets)
    offset = 0
    while active:
        next_active: list[str] = []
        for item_component in active:
            ordered_keys.append(buckets[item_component][offset])
            if offset + 1 < len(buckets[item_component]):
                next_active.append(item_component)
        active = next_active
        offset += 1

    selected_keys = tuple(ordered_keys[:limit])
    omitted_keys = tuple(ordered_keys[limit:])
    component_universe_counts = {
        item_component: len(item_keys)
        for item_component, item_keys in sorted(buckets.items())
    }
    component_selected_counts = dict.fromkeys(sorted(buckets), 0)
    for item_key in selected_keys:
        component_selected_counts[component_by_key[item_key]] += 1
    canonical_universe_keys = tuple(sorted(by_key))
    return FairSelectionResult(
        selected=tuple(by_key[item_key] for item_key in selected_keys),
        omitted=tuple(by_key[item_key] for item_key in omitted_keys),
        selected_keys=selected_keys,
        omitted_keys=omitted_keys,
        universe_count=len(by_key),
        selected_count=len(selected_keys),
        omitted_count=len(omitted_keys),
        seed_digest=seed_digest,
        universe_digest=_canonical_digest(
            {
                "policy": FAIR_SELECTION_POLICY_VERSION,
                "item_keys": canonical_universe_keys,
            }
        ),
        selected_digest=_canonical_digest(
            {
                "policy": FAIR_SELECTION_POLICY_VERSION,
                "selected_keys": selected_keys,
            }
        ),
        omitted_digest=_canonical_digest(
            {
                "policy": FAIR_SELECTION_POLICY_VERSION,
                "omitted_keys": omitted_keys,
            }
        ),
        component_universe_counts=component_universe_counts,
        component_selected_counts=component_selected_counts,
    )


__all__ = [
    "FAIR_SELECTION_POLICY_VERSION",
    "FairSelectionResult",
    "component_fair_stable_select",
]
