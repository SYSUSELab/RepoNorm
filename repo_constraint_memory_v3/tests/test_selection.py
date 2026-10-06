from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pytest

from repo_constraint_memory_v3.selection import (
    FAIR_SELECTION_POLICY_VERSION,
    FairSelectionResult,
    component_fair_stable_select,
)


@dataclass(frozen=True)
class _Item:
    item_id: str
    component: str


def _select(
    items: list[_Item], *, limit: int = 5
) -> FairSelectionResult[_Item]:
    return component_fair_stable_select(
        items,
        limit=limit,
        seed={"snapshot_digest": "sha256:fixture", "purpose": "test"},
        key=lambda item: item.item_id,
        component=lambda item: item.component,
    )


def test_component_fair_selection_is_input_order_independent_and_accounted() -> None:
    items = [
        *(_Item(f"alpha-{ordinal}", "alpha") for ordinal in range(5)),
        *(_Item(f"beta-{ordinal}", "beta") for ordinal in range(2)),
        _Item("gamma-0", "gamma"),
    ]

    forward = _select(items)
    reverse = _select(list(reversed(items)))
    interleaved = _select(items[::2] + items[1::2])

    assert FAIR_SELECTION_POLICY_VERSION == "component-fair-stable-hash-1.0"
    assert forward == reverse == interleaved
    assert forward.universe_count == 8
    assert forward.selected_count == 5
    assert forward.omitted_count == 3
    assert forward.selected_count + forward.omitted_count == forward.universe_count
    assert set(forward.selected_keys).isdisjoint(forward.omitted_keys)
    assert set(forward.selected_keys) | set(forward.omitted_keys) == {
        item.item_id for item in items
    }
    assert forward.component_universe_counts == {
        "alpha": 5,
        "beta": 2,
        "gamma": 1,
    }
    # Lexicographic component rounds guarantee that no non-empty component is
    # starved before a second member is selected from a larger component.
    assert forward.component_selected_counts == {
        "alpha": 2,
        "beta": 2,
        "gamma": 1,
    }
    assert all(
        len(digest) == 64
        for digest in (
            forward.seed_digest,
            forward.universe_digest,
            forward.selected_digest,
            forward.omitted_digest,
        )
    )


def test_component_fair_selection_zero_limit_preserves_complete_denominator() -> None:
    items = [_Item("alpha", "a"), _Item("beta", "b")]

    result = _select(items, limit=0)

    assert result.selected == ()
    assert result.selected_keys == ()
    assert result.omitted_keys == tuple(item.item_id for item in result.omitted)
    assert set(result.omitted_keys) == {"alpha", "beta"}
    assert result.component_selected_counts == {"a": 0, "b": 0}
    assert result.universe_count == result.omitted_count == 2


@pytest.mark.parametrize("limit", [0, 1, 7, 71, 1000])
def test_active_bucket_plan_matches_full_round_reference(limit: int) -> None:
    items = [
        *(_Item(f"wide-{ordinal}", "wide") for ordinal in range(100)),
        *(_Item(f"single-{ordinal}", f"single-{ordinal:03d}") for ordinal in range(40)),
        *(_Item(f"medium-{ordinal}", "medium") for ordinal in range(7)),
    ]
    result = _select(items, limit=limit)

    def digest(value: object) -> str:
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    buckets = {}
    for item in items:
        buckets.setdefault(item.component, []).append(item.item_id)
    for component, keys in buckets.items():
        keys.sort(key=lambda key: (digest({
            "seed_digest": result.seed_digest, "component": component, "item_key": key,
        }), key))
    expected = tuple(
        buckets[component][ordinal]
        for ordinal in range(max(map(len, buckets.values())))
        for component in sorted(buckets)
        if ordinal < len(buckets[component])
    )
    assert result.selected_keys == expected[:limit]
    assert result.omitted_keys == expected[limit:]
    by_key = {item.item_id: item for item in items}
    assert result.selected == tuple(by_key[key] for key in expected[:limit])
    assert result.omitted == tuple(by_key[key] for key in expected[limit:])
    assert result.component_selected_counts == {
        component: sum(by_key[key].component == component for key in expected[:limit])
        for component in sorted(buckets)
    }
    assert result.selected_digest == digest({
        "policy": FAIR_SELECTION_POLICY_VERSION, "selected_keys": expected[:limit],
    })
    assert result.omitted_digest == digest({
        "policy": FAIR_SELECTION_POLICY_VERSION, "omitted_keys": expected[limit:],
    })


@pytest.mark.parametrize(
    "items",
    [
        [_Item("duplicate", "a"), _Item("duplicate", "a")],
        [_Item("duplicate", "a"), _Item("duplicate", "b")],
    ],
)
def test_component_fair_selection_rejects_ambiguous_item_identity(
    items: list[_Item],
) -> None:
    with pytest.raises(ValueError, match="item identities must be unique"):
        _select(items)
