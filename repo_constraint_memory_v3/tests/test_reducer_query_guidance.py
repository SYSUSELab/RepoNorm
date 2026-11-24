"""Executable prompt examples and observed invalid DSL shapes stay aligned."""

import json
import re

import pytest

from repo_constraint_memory_v3.models import SemanticSettings
from repo_constraint_memory_v3.semantic.reduce import REDUCE_SYSTEM_PROMPT
from repo_constraint_memory_v3.semantic.queries import validate_population_spec


def test_reducer_query_examples_validate_against_the_actual_language() -> None:
    examples = re.findall(r"```json\n(.*?)\n```", REDUCE_SYSTEM_PROMPT, re.DOTALL)
    assert len(examples) == 3
    assert [validate_population_spec(json.loads(example), SemanticSettings())
            for example in examples] == ["entity", "file", "entity"]


@pytest.mark.parametrize("field,operator,value", [
    ("name", "path_glob", "*adapter*"),
    ("qualified_name", "path_glob", "*adapter*"),
    ("name", "has", "type_adapter"),
    ("path", "path_eq", "src/adapter.py"),
    ("name", "matches", "Adapter.*"),
])
def test_observed_illegal_predicates_are_not_relaxed_by_guidance(field, operator, value) -> None:
    with pytest.raises(ValueError):
        validate_population_spec({
            "op": "filter", "source": {"op": "universe", "universe": "entity"},
            "predicates": [{"field": field, "operator": operator, "value": value}],
        }, SemanticSettings())


@pytest.mark.parametrize("spec", [
    {"op": "union", "operands": [{"op": "universe", "universe": "entity"}]},
    {"op": "intersection", "operands": [
        {"op": "universe", "universe": "entity"},
        {"op": "universe", "universe": "file"},
    ]},
    {"op": "traverse", "source": {"op": "universe", "universe": "entity"},
     "traversal": "parent", "relation_kind": "contains", "direction": "incoming"},
])
def test_observed_illegal_combinations_still_fail_closed(spec) -> None:
    with pytest.raises(ValueError):
        validate_population_spec(spec, SemanticSettings())
