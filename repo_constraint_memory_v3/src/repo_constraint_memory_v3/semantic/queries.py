"""Pure evaluator for the safe PopulationSpec algebra."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from heapq import heappush, heapreplace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from repo_constraint_memory_v3.discovery import matches_glob
from repo_constraint_memory_v3.models import SemanticSettings
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    payload_digest,
)
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.observations.semantic_units import (
    _component_key,
)
from repo_constraint_memory_v3.semantic.models import (
    DifferenceSpec,
    FilterPredicate,
    FilterSpec,
    POPULATION_SPEC_VERSION,
    PopulationExpression,
    PopulationMemberRef,
    PopulationResult,
    PopulationSpec,
    SetSpec,
    TraverseSpec,
    UniverseSpec,
)

_SCALAR_FIELDS: dict[str, set[str]] = {
    "file": {
        "path",
        "component",
        "role",
        "source_form",
        "module_name",
        "file_id",
    },
    "entity": {
        "kind",
        "name",
        "qualified_name",
        "parent_entity_id",
        "file_id",
        "path",
        "component",
    },
    "site": {
        "kind",
        "decorator_call",
        "decorator_dynamic",
        "file_id",
        "path",
        "component",
        "site_id",
    },
    "relation": {
        "kind",
        "resolution",
        "source_entity_id",
        "source_file_id",
        "target_entity_id",
        "target_file_id",
        "site_id",
        "relation_id",
        "path",
        "component",
    },
}
_COLLECTION_FIELDS: dict[str, set[str]] = {
    "file": {"attributes"},
    "entity": set(),
    "site": {"decorator_symbols"},
    "relation": set(),
}
_FIELDS = {
    universe: _SCALAR_FIELDS[universe] | _COLLECTION_FIELDS[universe]
    for universe in _SCALAR_FIELDS
}
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class _Fact:
    universe: str
    observation_id: str
    file_id: str | None
    path: str | None
    values: Mapping[str, object]


@dataclass(frozen=True)
class _FactGraph:
    facts: Mapping[str, Mapping[str, _Fact]]
    parents: Mapping[str, str | None]
    children: Mapping[str, frozenset[str]]
    relations: tuple[object, ...]


@dataclass
class _EvaluationState:
    facts: Mapping[str, Mapping[str, _Fact]]
    parents: Mapping[str, str | None]
    children: Mapping[str, frozenset[str]]
    relations: tuple[object, ...]
    scanned: int = 0


class PopulationQueryContext:
    """Snapshot-owned facts with bounded memoization and independent scans.

    Retain only the most recent census and value-catalog bound.  These are
    host computations over one immutable ObservationBundle, never persistent
    cache entries or replacements for query validation/resource checks.
    """

    def __init__(self, observations: ObservationBundle) -> None:
        self.graph = _facts(observations)
        self.fact_by_id = MappingProxyType(
            {
                fact.observation_id: fact
                for universe in sorted(self.graph.facts)
                for fact in self.graph.facts[universe].values()
            }
        )
        self.shard_complete = MappingProxyType(
            {
                shard.file_id: shard.parse_status == "complete"
                for shard in observations.shards
            }
        )
        self.last_evaluation: tuple[bytes, str, frozenset[str], int] | None = None
        self.last_catalog: tuple[int, dict[str, object]] | None = None

    def evaluate(
        self, expression: PopulationExpression,
    ) -> tuple[str, frozenset[str], int]:
        identity = canonical_json_bytes(expression)
        previous = self.last_evaluation
        if previous is not None and previous[0] == identity:
            return previous[1:]
        graph = self.graph
        state = _EvaluationState(
            graph.facts, graph.parents, graph.children, graph.relations,
        )
        universe, members = _evaluate(expression, state)
        result = (universe, frozenset(members), state.scanned)
        self.last_evaluation = (identity, *result)
        return result


@dataclass(frozen=True)
class PopulationAdmissionCount:
    """Exact known-member cardinality used before semantic LLM admission.

    ``in_scope_member_count`` is exact for the supplied immutable observation
    bundle.  It intentionally does not claim that an incomplete observation
    bundle covers repository members that were never observed.
    """

    source_universe: str
    matched_count: int
    in_scope_member_count: int
    provider_eligible_in_scope_member_count: int


@dataclass(frozen=True)
class _CatalogMaximum:
    order: tuple[str, str]
    value: str | bool

    def __lt__(self, other: _CatalogMaximum) -> bool:
        # A bounded maximum heap retains the smallest catalog values without
        # rescanning all 128 retained values for each incoming distinct item.
        return self.order > other.order


def _facts(bundle: ObservationBundle) -> _FactGraph:
    facts: dict[str, dict[str, _Fact]] = {
        "file": {},
        "entity": {},
        "site": {},
        "relation": {},
    }
    parents: dict[str, str | None] = {}
    children: dict[str, set[str]] = {}
    relations: list[object] = []
    for shard in bundle.shards:
        component = _component_key(shard)
        facts["file"][shard.file_id] = _Fact(
            "file",
            shard.file_id,
            shard.file_id,
            shard.path,
            {
                "path": shard.path,
                "component": component,
                "role": shard.role.location_role,
                "source_form": shard.source_form,
                "attributes": tuple(shard.role.attributes),
                "module_name": shard.module.module_name,
                "file_id": shard.file_id,
            },
        )
        for entity in shard.entities:
            facts["entity"][entity.entity_id] = _Fact(
                "entity",
                entity.entity_id,
                shard.file_id,
                shard.path,
                {
                    "kind": entity.kind,
                    "name": entity.name,
                    "qualified_name": entity.qualified_name,
                    "parent_entity_id": entity.parent_entity_id,
                    "file_id": shard.file_id,
                    "path": shard.path,
                    "component": component,
                },
            )
            parents[entity.entity_id] = entity.parent_entity_id
            if entity.parent_entity_id is not None:
                children.setdefault(entity.parent_entity_id, set()).add(
                    entity.entity_id
                )
        for site in shard.sites:
            facts["site"][site.site_id] = _Fact(
                "site",
                site.site_id,
                shard.file_id,
                shard.path,
                {
                    "kind": site.kind,
                    "decorator_symbols": tuple(site.decorator_symbols or ()),
                    "decorator_call": site.decorator_call,
                    "decorator_dynamic": site.decorator_dynamic,
                    "file_id": shard.file_id,
                    "path": shard.path,
                    "component": component,
                    "site_id": site.site_id,
                },
            )
        for relation in shard.relations:
            facts["relation"][relation.relation_id] = _Fact(
                "relation",
                relation.relation_id,
                shard.file_id,
                shard.path,
                {
                    "kind": relation.kind,
                    "resolution": relation.resolution,
                    "source_entity_id": relation.source.entity_id,
                    "source_file_id": relation.source.file_id,
                    "target_entity_id": relation.target.entity_id,
                    "target_file_id": relation.target.file_id,
                    "site_id": relation.site_id,
                    "relation_id": relation.relation_id,
                    "path": shard.path,
                    "component": component,
                },
            )
            relations.append(relation)
    return _FactGraph(
        facts=MappingProxyType(
            {
                universe: MappingProxyType(
                    {
                        key: _Fact(
                            fact.universe, fact.observation_id, fact.file_id,
                            fact.path, MappingProxyType(dict(fact.values)),
                        )
                        for key, fact in members.items()
                    }
                )
                for universe, members in facts.items()
            }
        ),
        parents=MappingProxyType(parents),
        children=MappingProxyType(
            {key: frozenset(value) for key, value in children.items()}
        ),
        relations=tuple(relations),
    )


def _contains_relation_traverse(expression: PopulationExpression) -> bool:
    if isinstance(expression, TraverseSpec):
        return expression.traversal == "relation" or _contains_relation_traverse(
            expression.source
        )
    if isinstance(expression, FilterSpec):
        return _contains_relation_traverse(expression.source)
    if isinstance(expression, SetSpec):
        return any(_contains_relation_traverse(item) for item in expression.operands)
    if isinstance(expression, DifferenceSpec):
        return _contains_relation_traverse(
            expression.left
        ) or _contains_relation_traverse(expression.right)
    return False


def _known_missing_shard_values(shard: object) -> dict[str, object]:
    """Return fields whose values do not depend on a successful AST parse."""

    return {
        "path": getattr(shard, "path"),
        "component": _component_key(shard),
        "file_id": getattr(shard, "file_id"),
        "source_file_id": getattr(shard, "file_id"),
    }


def _file_membership_values(
    expression: PopulationExpression,
    shard: object,
) -> frozenset[bool]:
    """Return possible membership of one represented non-complete file.

    File identity and path/module/role metadata are materialized in the shard.
    Role attributes may depend on a decoded source header, so they remain
    unknown for a non-complete shard even though its file membership is exact.
    """

    if isinstance(expression, UniverseSpec):
        return frozenset({expression.universe == "file"})
    if isinstance(expression, FilterSpec):
        values = _file_membership_values(expression.source, shard)
        known = _known_missing_shard_values(shard)
        known.update(
            {
                "role": getattr(shard, "role").location_role,
                "source_form": getattr(shard, "source_form"),
                "module_name": getattr(shard, "module").module_name,
            }
        )
        for predicate in expression.predicates:
            predicate_values = (
                frozenset({False, True})
                if predicate.field == "attributes"
                else frozenset(
                    {
                        _matches(known[predicate.field], predicate)
                        if predicate.field in known
                        else False,
                    }
                )
            )
            values = frozenset(
                left and right
                for left in values
                for right in predicate_values
            )
        return values
    if isinstance(expression, SetSpec):
        operands = [
            _file_membership_values(item, shard)
            for item in expression.operands
        ]
        values = operands[0]
        for operand in operands[1:]:
            if expression.op == "union":
                values = frozenset(
                    left or right for left in values for right in operand
                )
            else:
                values = frozenset(
                    left and right for left in values for right in operand
                )
        return values
    if isinstance(expression, DifferenceSpec):
        left = _file_membership_values(expression.left, shard)
        right = _file_membership_values(expression.right, shard)
        return frozenset(
            left_value and not right_value
            for left_value in left
            for right_value in right
        )
    # Valid traversal expressions produce entities, not files.  Keep this
    # helper fail closed if a future contract adds another file traversal.
    if isinstance(expression, TraverseSpec):
        return frozenset({False, True})
    raise TypeError("unsupported population expression")


def _missing_shard_may_affect(
    expression: PopulationExpression,
    shard: object,
) -> bool:
    """Conservatively over-approximate one incomplete shard's query effect.

    Relation traversal is handled by the caller as a repository-wide dependency.
    Lexical traversal remains within its source shard, so its uncertainty follows
    the source expression.  Unknown semantic fields can never prove exclusion.
    """

    if isinstance(expression, UniverseSpec):
        return expression.universe != "file"
    if isinstance(expression, FilterSpec):
        if not _missing_shard_may_affect(expression.source, shard):
            return False
        known = _known_missing_shard_values(shard)
        return all(
            predicate.field not in known
            or _matches(known[predicate.field], predicate)
            for predicate in expression.predicates
        )
    if isinstance(expression, TraverseSpec):
        if expression.traversal == "relation":
            return True
        return _missing_shard_may_affect(expression.source, shard)
    if isinstance(expression, SetSpec):
        affected = [
            _missing_shard_may_affect(item, shard)
            for item in expression.operands
        ]
        return any(affected) if expression.op == "union" else all(affected)
    if isinstance(expression, DifferenceSpec):
        # Either operand can change the final set: unknown left facts can be
        # added and unknown right facts can remove otherwise selected facts.
        return _missing_shard_may_affect(
            expression.left, shard
        ) or _missing_shard_may_affect(expression.right, shard)
    raise TypeError("unsupported population expression")


def _known_manifest_gap_values(gap: object) -> dict[str, object]:
    return {
        "path": getattr(gap, "path"),
        "component": getattr(gap, "component_key"),
    }


def _manifest_gap_may_affect(
    expression: PopulationExpression,
    gap: object,
) -> bool:
    """Conservatively localize an enumerated, unmaterialized Python path."""

    if isinstance(expression, UniverseSpec):
        return True
    if isinstance(expression, FilterSpec):
        if not _manifest_gap_may_affect(expression.source, gap):
            return False
        known = _known_manifest_gap_values(gap)
        return all(
            predicate.field not in known
            or _matches(known[predicate.field], predicate)
            for predicate in expression.predicates
        )
    if isinstance(expression, TraverseSpec):
        if expression.traversal == "relation":
            return True
        return _manifest_gap_may_affect(expression.source, gap)
    if isinstance(expression, SetSpec):
        affected = [
            _manifest_gap_may_affect(item, gap)
            for item in expression.operands
        ]
        return any(affected) if expression.op == "union" else all(affected)
    if isinstance(expression, DifferenceSpec):
        return _manifest_gap_may_affect(
            expression.left, gap
        ) or _manifest_gap_may_affect(expression.right, gap)
    raise TypeError("unsupported population expression")


def population_dependency_file_ids(
    spec: PopulationSpec | Mapping[str, object],
    observations: ObservationBundle,
    settings: SemanticSettings,
) -> tuple[str, ...]:
    """Return the exact incomplete shards that may affect one denominator.

    This analysis is pure and fail-closed.  It deliberately ignores a
    hypothesis's ``applies_to`` scope because the population member-set digest
    covers the canonical PopulationSpec itself, including out-of-scope members.
    """

    parsed = (
        spec
        if isinstance(spec, PopulationSpec)
        else PopulationSpec.model_validate(spec)
    )
    universe = validate_population_spec(parsed, settings)
    incomplete = tuple(
        sorted(
            (
                shard
                for shard in observations.shards
                if shard.parse_status != "complete"
            ),
            key=lambda item: (item.path, item.file_id),
        )
    )
    if not incomplete:
        return ()
    if universe == "file":
        return tuple(
            sorted(
                shard.file_id
                for shard in incomplete
                if len(_file_membership_values(parsed.root, shard)) > 1
            )
        )
    if _contains_relation_traverse(parsed.root):
        # A missing relation can connect an incomplete shard to a fact in any
        # path, so outer path filters cannot safely localize this dependency.
        return tuple(sorted(shard.file_id for shard in incomplete))
    return tuple(
        sorted(
            shard.file_id
            for shard in incomplete
            if _missing_shard_may_affect(parsed.root, shard)
        )
    )


def population_dependency_paths(
    spec: PopulationSpec | Mapping[str, object],
    observations: ObservationBundle,
    settings: SemanticSettings,
) -> tuple[str, ...]:
    """Return unrepresented Python paths that may affect one denominator."""

    parsed = (
        spec
        if isinstance(spec, PopulationSpec)
        else PopulationSpec.model_validate(spec)
    )
    universe = validate_population_spec(parsed, settings)
    snapshot = observations.snapshot
    if snapshot is None:
        return ()
    gaps = tuple(snapshot.unrepresented_python_files)
    if not gaps:
        return ()
    package_markers = {
        gap.path
        for gap in gaps
        if PurePosixPath(gap.path).name.casefold()
        in {"__init__.py", "__init__.pyi"}
    }
    # A missing package marker changes module/package/component metadata and
    # local relation resolution for represented siblings.  Its effect is not
    # confined to whether the marker itself passes an outer path filter.
    if universe == "relation" or _contains_relation_traverse(parsed.root):
        # An omitted Python module can change resolution and target_file_id on
        # an import relation owned by a represented sibling.  Therefore direct
        # relation-universe queries are cross-file dependent even when their
        # outer path filter excludes the omitted path itself.
        return tuple(sorted(gap.path for gap in gaps))
    return tuple(
        sorted(
            {
                *package_markers,
                *(
                    gap.path
                    for gap in gaps
                    if _manifest_gap_may_affect(parsed.root, gap)
                ),
            }
        )
    )


def _shape(expression: PopulationExpression) -> tuple[str, int, int, int]:
    """Return universe, nodes, depth, and bounded literal count."""

    if isinstance(expression, UniverseSpec):
        return expression.universe, 1, 1, 0
    if isinstance(expression, FilterSpec):
        universe, nodes, depth, literals = _shape(expression.source)
        for predicate in expression.predicates:
            if predicate.field not in _FIELDS[universe]:
                raise ValueError(
                    f"unknown {universe} population field: {predicate.field}"
                )
            if predicate.operator == "has":
                if predicate.field not in _COLLECTION_FIELDS[universe]:
                    raise ValueError(
                        "has operator requires a collection field; "
                        f"{universe}.{predicate.field} is scalar"
                    )
            elif predicate.field in _COLLECTION_FIELDS[universe]:
                raise ValueError(
                    f"{predicate.operator} operator requires a scalar field; "
                    f"{universe}.{predicate.field} is a collection"
                )
            if predicate.operator == "path_under" and predicate.field != "path":
                raise ValueError("path_under operator is valid only for the path field")
            if predicate.operator == "path_glob" and predicate.field != "path":
                raise ValueError("path_glob operator is valid only for the path field")
            literal_values = (
                predicate.value
                if isinstance(predicate.value, list)
                else [predicate.value]
            )
            if predicate.operator in {"in", "path_under"} and any(
                isinstance(item, str) and any(char in item for char in _GLOB_CHARS)
                for item in literal_values
            ):
                raise ValueError(
                    f"{predicate.operator} uses exact literals and cannot contain globs"
                )
            literals += (
                len(predicate.value)
                if isinstance(predicate.value, list)
                else 1
            )
        return universe, nodes + 1, depth + 1, literals
    if isinstance(expression, TraverseSpec):
        universe, nodes, depth, literals = _shape(expression.source)
        if universe != "entity":
            raise ValueError("traversal operands must use the entity universe")
        return "entity", nodes + 1, depth + 1, literals
    if isinstance(expression, SetSpec):
        shapes = [_shape(item) for item in expression.operands]
        universes = {item[0] for item in shapes}
        if len(universes) != 1:
            raise ValueError("set operands use incompatible universes")
        return (
            shapes[0][0],
            1 + sum(item[1] for item in shapes),
            1 + max(item[2] for item in shapes),
            sum(item[3] for item in shapes),
        )
    if isinstance(expression, DifferenceSpec):
        left = _shape(expression.left)
        right = _shape(expression.right)
        if left[0] != right[0]:
            raise ValueError("difference operands use incompatible universes")
        return (
            left[0],
            1 + left[1] + right[1],
            1 + max(left[2], right[2]),
            left[3] + right[3],
        )
    raise TypeError("unsupported population expression")


def validate_population_spec(
    spec: PopulationSpec | Mapping[str, object],
    settings: SemanticSettings,
) -> str:
    """Validate the bounded universe/field/operator semantics without executing."""

    parsed = (
        spec
        if isinstance(spec, PopulationSpec)
        else PopulationSpec.model_validate(spec)
    )
    universe, nodes, depth, literals = _shape(parsed.root)
    if nodes > settings.max_population_query_nodes:
        raise ValueError("PopulationSpec exceeds max_population_query_nodes")
    if depth > settings.max_population_query_depth:
        raise ValueError("PopulationSpec exceeds max_population_query_depth")
    if literals > settings.max_population_literal_values:
        raise ValueError("PopulationSpec exceeds max_population_literal_values")
    spec_payload = parsed.root.model_dump(mode="json")
    if len(canonical_json_bytes(spec_payload)) > (
        settings.max_population_artifact_bytes
    ):
        raise ValueError("PopulationSpec exceeds max_population_artifact_bytes")
    return universe


def _matches(value: object, predicate: FilterPredicate) -> bool:
    expected = predicate.value
    if predicate.operator == "eq":
        return value == expected
    if predicate.operator == "in":
        assert isinstance(expected, list)
        return value in expected
    if predicate.operator == "path_under":
        assert isinstance(expected, str)
        if not isinstance(value, str):
            return False
        path = PurePosixPath(value)
        base = PurePosixPath(expected)
        return path == base or base in path.parents
    if predicate.operator == "path_glob":
        if not isinstance(value, str):
            return False
        patterns = expected if isinstance(expected, list) else [expected]
        return any(
            isinstance(pattern, str) and matches_glob(value, pattern)
            for pattern in patterns
        )
    if predicate.operator == "has":
        assert isinstance(expected, str)
        return isinstance(value, (list, tuple, set)) and expected in value
    raise ValueError(f"unknown population predicate: {predicate.operator}")


def population_query_contract() -> dict[str, object]:
    """Return the exact bounded field/operator contract shown to the planner."""

    return {
        "version": POPULATION_SPEC_VERSION,
        "universes": {
            universe: {
                "scalar_fields": sorted(_SCALAR_FIELDS[universe]),
                "collection_fields": sorted(_COLLECTION_FIELDS[universe]),
            }
            for universe in sorted(_FIELDS)
        },
        "operators": {
            "eq": "one exact scalar literal",
            "in": "one scalar against a non-empty exact literal list",
            "has": "one member of a collection field",
            "path_under": "path at or below one concrete repository directory",
            "path_glob": "path matching one or more safe repository globs",
        },
    }


def population_value_catalog(
    observations: ObservationBundle,
    *,
    max_values_per_field: int = 128,
) -> dict[str, object]:
    """Build a bounded deterministic catalog of values available to a query.

    The catalog intentionally does not compute an exact distinct count after a
    field exceeds its bound.  Keeping an exact count would require retaining an
    unbounded set for high-cardinality identifiers.  Instead, it retains the
    lexicographically smallest values and reports a deterministic lower bound.
    """

    if max_values_per_field < 1:
        raise ValueError("max_values_per_field must be positive")

    context = observations.population_query_context
    previous = context.last_catalog
    if previous is not None and previous[0] == max_values_per_field:
        return deepcopy(previous[1])
    state = context.graph
    catalog: dict[str, object] = {}
    for universe in sorted(state.facts):
        values_by_field: dict[str, object] = {}
        facts = state.facts[universe].values()
        for field in sorted(_FIELDS[universe]):
            values: set[str | bool] = set()
            truncated = False
            maximum_heap: list[_CatalogMaximum] = []
            for fact in facts:
                value = fact.values.get(field)
                candidates: Iterable[object]
                if isinstance(value, (str, bool)):
                    candidates = (value,)
                elif isinstance(value, (tuple, list, set)):
                    candidates = value
                else:
                    candidates = ()
                for item in candidates:
                    if not isinstance(item, (str, bool)) or item in values:
                        continue
                    ranked = _CatalogMaximum((str(type(item)), str(item)), item)
                    if len(values) < max_values_per_field:
                        values.add(item)
                        heappush(maximum_heap, ranked)
                        continue
                    truncated = True
                    if ranked.order < maximum_heap[0].order:
                        largest = heapreplace(maximum_heap, ranked)
                        values.remove(largest.value)
                        values.add(item)
            ordered = sorted(values, key=lambda item: (str(type(item)), str(item)))
            values_by_field[field] = {
                "distinct_count": None if truncated else len(ordered),
                "distinct_count_lower_bound": (
                    max_values_per_field + 1 if truncated else len(ordered)
                ),
                "values": ordered,
                "truncated": truncated,
            }
        catalog[universe] = {
            "fact_count": len(state.facts[universe]),
            "fields": values_by_field,
        }
    result = {
        "contract_version": POPULATION_SPEC_VERSION,
        "universes": catalog,
    }
    context.last_catalog = (max_values_per_field, result)
    return deepcopy(result)


def population_query_facts(
    observations: ObservationBundle,
    observation_refs: Sequence[str],
    *,
    max_facts: int = 512,
) -> dict[str, object]:
    """Return bounded typed facts that may be referenced by witness bindings."""

    if max_facts < 1:
        raise ValueError("max_facts must be positive")
    context = observations.population_query_context
    state = context.graph
    requested_refs = sorted(set(observation_refs))
    fact_by_id = context.fact_by_id
    matched_refs = [item for item in requested_refs if item in fact_by_id]
    serialized_refs = matched_refs[:max_facts]
    facts: list[dict[str, object]] = []
    for observation_id in serialized_refs:
        fact = fact_by_id[observation_id]
        facts.append(
            {
                "universe": fact.universe,
                "observation_id": fact.observation_id,
                "file_id": fact.file_id,
                "path": fact.path,
                "values": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in sorted(fact.values.items())
                },
            }
        )
    return {
        "requested_ref_count": len(requested_refs),
        "matched_fact_count": len(matched_refs),
        "serialized_fact_count": len(facts),
        "omitted_fact_count": len(matched_refs) - len(facts),
        "unknown_refs": sorted(set(requested_refs) - set(matched_refs)),
        "universe_fact_counts": {
            universe: len(state.facts[universe])
            for universe in sorted(state.facts)
        },
        "facts": facts,
    }


def _evaluate(
    expression: PopulationExpression,
    state: _EvaluationState,
) -> tuple[str, set[str]]:
    if isinstance(expression, UniverseSpec):
        values = set(state.facts[expression.universe])
        state.scanned += len(values)
        return expression.universe, values
    if isinstance(expression, FilterSpec):
        universe, source = _evaluate(expression.source, state)
        selected = {
            item
            for item in source
            if all(
                _matches(state.facts[universe][item].values.get(predicate.field), predicate)
                for predicate in expression.predicates
            )
        }
        return universe, selected
    if isinstance(expression, TraverseSpec):
        _, source = _evaluate(expression.source, state)
        selected: set[str] = set()
        if expression.traversal == "parent":
            selected.update(
                parent
                for item in source
                if (parent := state.parents.get(item)) is not None
            )
        elif expression.traversal == "child":
            selected.update(
                child
                for item in source
                for child in state.children.get(item, ())
            )
        elif expression.traversal == "ancestor":
            for item in source:
                parent = state.parents.get(item)
                while parent is not None:
                    selected.add(parent)
                    parent = state.parents.get(parent)
        elif expression.traversal == "descendant":
            pending = list(source)
            while pending:
                children = state.children.get(pending.pop(), set())
                selected.update(children)
                pending.extend(children)
        else:
            for relation in state.relations:
                if relation.kind != expression.relation_kind:
                    continue
                source_id = relation.source.entity_id
                target_id = relation.target.entity_id
                if (
                    expression.direction == "outgoing"
                    and source_id in source
                    and target_id is not None
                ):
                    selected.add(target_id)
                elif (
                    expression.direction == "incoming"
                    and target_id in source
                    and source_id is not None
                ):
                    selected.add(source_id)
        return "entity", selected & set(state.facts["entity"])
    if isinstance(expression, SetSpec):
        evaluated = [_evaluate(item, state) for item in expression.operands]
        universe = evaluated[0][0]
        if expression.op == "union":
            return universe, set().union(*(item[1] for item in evaluated))
        result = set(evaluated[0][1])
        for _, values in evaluated[1:]:
            result.intersection_update(values)
        return universe, result
    if isinstance(expression, DifferenceSpec):
        universe, left = _evaluate(expression.left, state)
        _, right = _evaluate(expression.right, state)
        return universe, left - right
    raise TypeError("unsupported population expression")


def evaluate_population(
    spec: PopulationSpec | Mapping[str, object],
    observations: ObservationBundle,
    settings: SemanticSettings,
) -> PopulationResult:
    """Evaluate one bounded spec using only the supplied serialized facts."""

    parsed = (
        spec
        if isinstance(spec, PopulationSpec)
        else PopulationSpec.model_validate(spec)
    )
    validate_population_spec(parsed, settings)
    universe, nodes, depth, literals = _shape(parsed.root)
    spec_payload = parsed.root.model_dump(mode="json")
    spec_digest = payload_digest(
        {"version": POPULATION_SPEC_VERSION, "spec": spec_payload}
    )
    context = observations.population_query_context
    state = context.graph
    evaluated_universe, member_ids, scanned = context.evaluate(parsed.root)
    assert evaluated_universe == universe
    ordered = sorted(member_ids)
    omitted = max(len(ordered) - settings.max_population_members, 0)
    serialized_ids = ordered[: settings.max_population_members]
    members = [
        PopulationMemberRef(
            universe=universe,  # type: ignore[arg-type]
            observation_id=item,
            file_id=state.facts[universe][item].file_id,
            path=state.facts[universe][item].path,
        )
        for item in serialized_ids
    ]
    dependency_file_ids = list(
        population_dependency_file_ids(parsed, observations, settings)
    )
    dependency_paths = list(
        population_dependency_paths(parsed, observations, settings)
    )
    dependency_paths_omitted_count = (
        observations.snapshot.unrepresented_python_files_omitted_count
        if observations.snapshot is not None
        else 0
    )
    reasons: list[str] = []
    snapshot = observations.snapshot
    if snapshot is None or snapshot.status == "failed":
        reasons.append("population_query_dependency_unavailable")
    elif (
        dependency_file_ids
        or dependency_paths
        or dependency_paths_omitted_count
    ):
        reasons.append("population_query_dependency_partial")
        if dependency_paths_omitted_count:
            reasons.append("population_query_unrepresented_path_overflow")
    if omitted:
        reasons.append("max_population_members")
    status = "complete" if not reasons else "partial"
    member_digest = (
        payload_digest(
            {
                "universe": universe,
                "members": [item.model_dump(mode="json") for item in members],
            }
        )
        if status == "complete"
        else None
    )
    base: dict[str, object] = {
        "spec_version": POPULATION_SPEC_VERSION,
        "spec": spec_payload,
        "spec_digest": spec_digest,
        "snapshot_digest": observations.index.snapshot_digest,
        "observation_set_digest": observations.index.observation_set_digest,
        "status": status,
        "source_universe": universe,
        "scanned_count": scanned,
        "matched_count": len(ordered),
        "serialized_count": len(members),
        "omitted_count": omitted,
        "members": [item.model_dump(mode="json") for item in members],
        "dependency_file_ids": dependency_file_ids,
        "dependency_paths": dependency_paths,
        "dependency_paths_omitted_count": dependency_paths_omitted_count,
        "member_set_digest": member_digest,
        "incomplete_reasons": reasons,
        "limitations": [
            "enumeration_only_no_semantic_classification",
            "missing_members_are_not_negative_evidence",
        ],
    }
    resource = {
        "query_nodes": nodes,
        "query_depth": depth,
        "literal_values": literals,
        "scanned": scanned,
        "serialized": len(members),
        "artifact_bytes": 0,
    }
    base["resource_use"] = resource
    base["result_digest"] = "0" * 64
    artifact_bytes = len(canonical_json_bytes(base))
    while resource["artifact_bytes"] != artifact_bytes:
        resource["artifact_bytes"] = artifact_bytes
        artifact_bytes = len(canonical_json_bytes(base))
    if artifact_bytes > settings.max_population_artifact_bytes:
        raise ValueError("population result exceeds max_population_artifact_bytes")
    base["result_digest"] = payload_digest(
        {key: value for key, value in base.items() if key != "result_digest"}
    )
    return PopulationResult.model_validate(base)


def count_population_members_for_admission(
    spec: PopulationSpec | Mapping[str, object],
    observations: ObservationBundle,
    settings: SemanticSettings,
    *,
    applies_to: Sequence[str],
) -> PopulationAdmissionCount:
    """Count every known matching member without artifact serialization caps.

    Population artifact limits can truncate ``PopulationResult.members`` even
    though the safe evaluator has already resolved the complete set of known
    IDs.  Admission must therefore count directly from that set; using the
    serialized prefix would understate the expensive semantic classification
    denominator.
    """

    if not applies_to:
        raise ValueError("population admission requires at least one scope pattern")
    parsed = (
        spec
        if isinstance(spec, PopulationSpec)
        else PopulationSpec.model_validate(spec)
    )
    validate_population_spec(parsed, settings)
    context = observations.population_query_context
    universe, member_ids, _ = context.evaluate(parsed.root)
    facts = context.graph.facts[universe]
    shard_complete = context.shard_complete
    matching = [
        fact
        for fact in (facts[item] for item in member_ids)
        if fact.path is not None
        and any(matches_glob(fact.path, pattern) for pattern in applies_to)
    ]
    return PopulationAdmissionCount(
        source_universe=universe,
        matched_count=len(member_ids),
        in_scope_member_count=len(matching),
        provider_eligible_in_scope_member_count=sum(
            fact.file_id is not None and shard_complete.get(fact.file_id, False)
            for fact in matching
        ),
    )
