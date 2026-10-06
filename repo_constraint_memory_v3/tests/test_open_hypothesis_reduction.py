from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from helpers import (
    DelayedMockClient,
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.inference import (
    InferenceRecord,
    _is_repeated_invalid_response,
    request_identity,
)
from repo_constraint_memory_v3.llm import LLMProviderError, MockLLMClient
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.semantic.reduce import (
    MAX_REDUCE_REPAIR_INVALID_RESPONSE_BYTES,
    MAX_REDUCE_SELECTION_CANDIDATES,
    MAX_REDUCE_SELECTION_PROMPT_BYTES,
    HypothesisDraft,
    ProviderReductionResponse,
    REDUCE_PROTOCOL_VERSION,
    REDUCE_RESPONSE_SCHEMA_VERSION,
    REDUCE_SYSTEM_PROMPT,
    ReductionSelectionResponse,
    ReductionCallResult,
    ReductionResponse,
    _ReduceInput,
    _ReductionRepairController,
    _effective_output_hypothesis_limit,
    _normalize_response,
    _partition,
    _preflight_population,
    _population_context,
    _validate_population_context_references,
    _quarantine_reason,
    _repair_prompt,
    _validate_cardinality_selection_attempt_binding,
    _validate_raw_response,
    _validate_response,
)


def _editable_reducer_response(
    response: ProviderResponse, request: dict[str, object]
) -> dict[str, object]:
    return json.loads(response.content)


class _RecordingReductionClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.reduction_requests: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "current_hypothesis_reduce":
            self.reduction_requests.append(request)
        return super().complete(system_prompt, user_prompt)


def test_reducer_prompt_contrastively_separates_scope_from_witness() -> None:
    assert "governed-change test" in REDUCE_SYSTEM_PROMPT
    assert "test is evidence for production behavior" in REDUCE_SYSTEM_PROMPT
    assert "test files themselves are direct change targets" in (
        REDUCE_SYSTEM_PROMPT
    )
    assert "may likewise be both witness and scope" in REDUCE_SYSTEM_PROMPT
    assert "Directory role, filename" in REDUCE_SYSTEM_PROMPT
    assert "material-qualifier test" in REDUCE_SYSTEM_PROMPT
    assert "supplied evidence does not establish a material one" in (
        REDUCE_SYSTEM_PROMPT
    )
    assert "independently usable branches" in REDUCE_SYSTEM_PROMPT
    assert "rather than splitting one atomic input" in REDUCE_SYSTEM_PROMPT


class _LosslessReductionClient:
    """Emit traceable partitions and optional exact-identity evidence variants."""

    def __init__(
        self,
        *,
        alias_component_witnesses: bool = False,
        reverse_alias_duplicates: bool = False,
        invalidate_alias_duplicate: bool = False,
    ) -> None:
        self.mock = MockLLMClient()
        self.alias_component_witnesses = alias_component_witnesses
        self.reverse_alias_duplicates = reverse_alias_duplicates
        self.invalidate_alias_duplicate = invalidate_alias_duplicate
        self.partition_rules: set[str] = set()
        self.reduction_requests: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") != "current_hypothesis_reduce":
            response = self.mock.complete(system_prompt, user_prompt)
            if (
                self.alias_component_witnesses
                and request.get("stage") == "current_semantic_map"
            ):
                value = _editable_reducer_response(response, request)
                packet_by_id = {
                    item["packet_id"]: item for item in request["packets"]
                }
                for result in value["results"]:
                    ranges = [
                        item["range_id"]
                        for item in packet_by_id[result["packet_id"]]["source"]
                    ]
                    if ranges:
                        for observation in result["observations"]:
                            observation["evidence_refs"] = [ranges[0]]
                return ProviderResponse(
                    content=json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                )
            return response
        self.reduction_requests.append(request)
        mode = request["mode"]
        response = self.mock.complete(system_prompt, user_prompt)
        value = _editable_reducer_response(response, request)
        if mode == "component_partition":
            owners_by_alias: dict[str, set[str]] = {}
            for item in request["inputs"]:
                observation_id = str(item.get("observation_id", ""))
                for evidence_ref in item.get("evidence_refs", []):
                    alias = str(evidence_ref)
                    if alias.startswith("SEMRANGE-") and observation_id:
                        owners_by_alias.setdefault(alias, set()).add(
                            observation_id
                        )
            aliases = sorted(owners_by_alias)
            direct_duplicates: list[dict[str, object]] = []
            for hypothesis in value.get("hypotheses", []):
                hypothesis["rule"] = (
                    "Preserve reducer branch "
                    + str(request["accounted_input_ids"][0])
                )
                if (
                    self.alias_component_witnesses
                    and aliases
                ):
                    hypothesis["witness_refs"] = [aliases[0]]
                    direct = json.loads(json.dumps(hypothesis))
                    direct["witness_refs"] = sorted(
                        owners_by_alias[aliases[0]]
                    )
                    direct["category"] = "lossless-direct-duplicate"
                    if self.invalidate_alias_duplicate:
                        direct["explicit_context_refs"] = [
                            "UNKNOWN-EXPLICIT-CONTEXT"
                        ]
                    direct_duplicates.append(direct)
                self.partition_rules.add(str(hypothesis["rule"]))
            if self.reverse_alias_duplicates:
                value["hypotheses"] = [
                    *direct_duplicates,
                    *value["hypotheses"],
                ]
            else:
                value["hypotheses"].extend(direct_duplicates)
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="lossless-reduction-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _AlwaysInvalidPartitionPopulationClient(ParameterizedSemanticClient):
    """Return invalid, malformed, then terminally quarantinable reduction output."""

    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.partition_calls = 0
        self.target_input_ids = None
        self.target_lock = Lock()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        value = _editable_reducer_response(response, request)
        if request.get("stage") == "current_hypothesis_reduce":
            input_ids = tuple(request["accounted_input_ids"])
            with self.target_lock:
                if self.target_input_ids is None:
                    self.target_input_ids = input_ids
            if input_ids != self.target_input_ids:
                return response
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
        ):
            self.partition_calls += 1
            if self.partition_calls == 2:
                return ProviderResponse(
                    content="{not-json",
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                )
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
            and value.get("hypotheses")
        ):
            invalid = json.loads(json.dumps(value["hypotheses"][0]))
            invalid["superpopulation_spec"] = {
                "op": "filter",
                "source": {"op": "universe", "universe": "file"},
                "predicates": [
                    {
                        "field": "file_path",
                        "operator": "has",
                        "value": ".py",
                    }
                ],
            }
            value["hypotheses"].append(invalid)
        return ProviderResponse(
            content=json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _AlwaysRawMalformedPopulationClient(ParameterizedSemanticClient):
    """Keep one valid sibling while a raw PopulationSpec exhausts repairs."""

    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.partition_calls = 0
        self.target_input_ids = None
        self.target_lock = Lock()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        value = _editable_reducer_response(response, request)
        if request.get("stage") == "current_hypothesis_reduce":
            input_ids = tuple(request["accounted_input_ids"])
            with self.target_lock:
                if self.target_input_ids is None:
                    self.target_input_ids = input_ids
            if input_ids != self.target_input_ids:
                return response
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
        ):
            self.partition_calls += 1
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
            and value.get("hypotheses")
        ):
            malformed = json.loads(json.dumps(value["hypotheses"][0]))
            malformed["superpopulation_spec"] = {
                "op": "filter",
                "source": {"op": "universe", "universe": "file"},
                "predicates": [
                    {
                        "field": "path",
                        "operator": "totally_unknown_operator",
                        "value": "src",
                    }
                ],
            }
            value["hypotheses"].append(malformed)
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _StructuralPartitionPopulationClient(ParameterizedSemanticClient):
    """Turn a mock partition hypothesis into a fully structural predicate."""

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        value = _editable_reducer_response(response, request)
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
        ):
            for hypothesis in value.get("hypotheses", []):
                superpopulation = hypothesis["superpopulation_spec"]
                hypothesis["predicate_mode"] = "structural"
                hypothesis["structural_predicate"] = {
                    "eligible_spec": superpopulation,
                    "support_spec": superpopulation,
                }
                hypothesis["semantic_predicate"] = None
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _EvidenceRangeWitnessAliasClient(ParameterizedSemanticClient):
    target_summary = "The fetch_document packet supplies the alias fixture."
    companion_summary = (
        "The same exact range also supports a distinct companion interpretation."
    )
    shared_range = False

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        value = _editable_reducer_response(response, request)
        if request.get("stage") == "current_semantic_map":
            packet_by_id = {
                item["packet_id"]: item for item in request["packets"]
            }
            for result in value["results"]:
                packet = packet_by_id[result["packet_id"]]
                ranges = [item["range_id"] for item in packet["source"]]
                if ranges:
                    for observation in result["observations"]:
                        observation["evidence_refs"] = [ranges[0]]
                # One source-owned function packet, independent of thread arrival
                # and request-local transport aliases. No shared mutation occurs.
                if (
                    packet["path"] == "src/aurora/api.py"
                    and len(packet["source"]) == 1
                    and packet["source"][0]["text"].startswith(
                        "def fetch_document("
                    )
                ):
                    assert len(result["observations"]) == 1
                    result["observations"][0]["summary"] = self.target_summary
                    if self.shared_range:
                        companion = json.loads(
                            json.dumps(result["observations"][0])
                        )
                        companion["summary"] = self.companion_summary
                        result["observations"].append(companion)
        if (
            request.get("stage") == "current_hypothesis_reduce"
            and request.get("mode") == "component_partition"
            and value.get("hypotheses")
        ):
            target_summary = (
                self.companion_summary if self.shared_range else self.target_summary
            )
            targets = [
                item for item in request["inputs"]
                if item.get("summary") == target_summary
            ]
            if targets:
                assert len(targets) == 1
                # Only this partition's own canonical evidence may be injected.
                # A companion in another request cannot affect this response.
                value["hypotheses"][0]["witness_refs"][0] = (
                    targets[0]["evidence_refs"][0]
                )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _SharedEvidenceRangeWitnessAliasClient(_EvidenceRangeWitnessAliasClient):
    shared_range = True


class _CardinalitySelectionClient(ParameterizedSemanticClient):
    """Emit one bounded over-cap response, then select original ordinals only."""

    def __init__(
        self,
        *,
        overflow_count: int,
        selected_ordinals: list[int],
        invalid_population_ordinal: int | None = None,
        structural_ordinal: int | None = None,
        overflow_mode: str | None = None,
    ) -> None:
        super().__init__(
            category="bounded-cardinality-selection",
            rule="A cardinality-selection fixture draft.",
        )
        self.overflow_count = overflow_count
        self.selected_ordinals = selected_ordinals
        self.invalid_population_ordinal = invalid_population_ordinal
        self.structural_ordinal = structural_ordinal
        self.overflow_mode = overflow_mode or "component_partition"
        self.all_partitions = overflow_mode is not None
        self.overflow_input_ids = None
        self.target_lock = Lock()
        self.reducer_calls = 0
        self.initial_reducer_requests: list[dict[str, object]] = []
        self.initial_reducer_schemas: list[dict[str, object]] = []
        self.selection_requests: list[dict[str, object]] = []
        self.selection_schemas: list[dict[str, object]] = []
        self.overflow_drafts: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        if payload.get("repair_type") == "bounded_cardinality_selection":
            self.reducer_calls += 1
            self.selection_requests.append(payload)
            self.selection_schemas.append(payload["response_json_schema"])
            content = json.dumps(
                {
                    "selected_hypothesis_ordinals": self.selected_ordinals,
                    # All three values are hostile provider claims and must be
                    # discarded before the ordinal-only response is validated.
                    "hypothesis_ordinals": [999],
                    "host_population_repairs": [{"forged": True}],
                    "quarantined_hypotheses": [{"forged": True}],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            return ProviderResponse(
                content=content,
                model="cardinality-selection-fake",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1,
            )

        response = super().complete(system_prompt, user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        self.reducer_calls += 1
        self.initial_reducer_requests.append(request)
        self.initial_reducer_schemas.append(payload["response_json_schema"])
        if request.get("mode") != self.overflow_mode:
            return response
        input_ids = tuple(request["accounted_input_ids"])
        with self.target_lock:
            if self.overflow_input_ids is None:
                self.overflow_input_ids = input_ids
        if not self.all_partitions and input_ids != self.overflow_input_ids:
            return response
        value = _editable_reducer_response(response, request)
        if not value.get("hypotheses"):
            return response
        original = value["hypotheses"][0]
        hypotheses = []
        for ordinal in range(self.overflow_count):
            draft = json.loads(json.dumps(original))
            draft["rule"] = f"Preserve exact overflow draft {ordinal}."
            if ordinal == self.invalid_population_ordinal:
                draft["superpopulation_spec"] = {
                    "op": "filter",
                    "source": {"op": "universe", "universe": "file"},
                    "predicates": [
                        {
                            "field": "path",
                            "operator": "has",
                            "value": ".py",
                        }
                    ],
                }
            if ordinal == self.structural_ordinal:
                draft["predicate_mode"] = "structural"
                draft["structural_predicate"] = {
                    "eligible_spec": json.loads(
                        json.dumps(draft["superpopulation_spec"])
                    ),
                    "support_spec": json.loads(
                        json.dumps(draft["superpopulation_spec"])
                    ),
                }
                draft["semantic_predicate"] = None
            hypotheses.append(draft)
        value["outcome"] = "hypotheses"
        value["hypotheses"] = hypotheses
        self.overflow_drafts = json.loads(json.dumps(hypotheses))
        value["reason_code"] = "bounded_over_cardinality_fixture"
        value["hypothesis_ordinals"] = [999]
        value["host_population_repairs"] = [{"forged": True}]
        value["quarantined_hypotheses"] = [{"forged": True}]
        value["host_cardinality_selection"] = {"forged": True}
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="cardinality-selection-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


class _RepeatedInvalidReductionClient(ParameterizedSemanticClient):
    """Repeat one exact invalid reducer response across every repair prompt."""

    def __init__(self, *, content: str | None = None) -> None:
        super().__init__(
            category="repeated-invalid-response",
            rule="A repeated invalid response must stop deterministically.",
        )
        self.reducer_calls = 0
        self.content = content or json.dumps(
            {
                "outcome": "hypotheses",
                "hypotheses": [],
                "reason_code": "empty_hypothesis_repeat",
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") == "current_hypothesis_reduce":
            self.reducer_calls += 1
            return ProviderResponse(
                content=self.content,
                model="repeated-invalid-reducer-fake",
                input_tokens=10,
                output_tokens=1,
                latency_ms=1,
                finish_reason="stop",
            )
        return super().complete(system_prompt, user_prompt)


class _HostFieldEchoReductionClient(ParameterizedSemanticClient):
    """Echo forged host metadata on otherwise valid reducer responses."""

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = _editable_reducer_response(response, request)
        value["hypothesis_ordinals"] = [999]
        value["host_population_repairs"] = [{"forged": True}]
        value["quarantined_hypotheses"] = [{"forged": True}]
        value["host_cardinality_selection"] = {"forged": True}
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="host-field-echo-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


def test_partition_reduction_accounts_actual_calls_without_category_dispatch(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RecordingReductionClient(
        category="metadata-shape-never-named-by-production",
        rule="Fixture participants expose their local metadata shape.",
    )
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=client,
    )
    assert result.current_source is not None
    reduction = result.current_source.reduction
    assert reduction.index.status == "complete"
    assert {item.mode for item in reduction.call_results} == {
        "component_partition",
    }
    assert reduction.index.planned_calls == len(reduction.call_results)
    assert reduction.index.generated_candidate_events == len(reduction.events)
    assert reduction.index.unique_candidate_roots == len(reduction.hypotheses)
    assert reduction.index.cancelled_stages == ["component_merge", "repository_merge"]
    assert reduction.hypotheses
    assert {
        item.category for item in reduction.hypotheses
    } == {"metadata-shape-never-named-by-production"}
    assert client.reduction_requests
    assert all(
        "response_constraints" not in request
        and request["hypothesis_budget"]["max_output_hypotheses"]
        == min(
            request["hypothesis_budget"][
                "configured_max_output_hypotheses"
            ],
            request["hypothesis_budget"]["non_expansion_source_count"],
        )
        and request["hypothesis_budget"]["non_expansion_basis"]
        == (
            "atomic_partition_inputs"
            if request["mode"] == "component_partition"
            else "embedded_child_hypotheses"
        )
        and "each_witness_query_fact_inside_superpopulation"
        in request["response_preflight_contract"]["required_before_emit"]
        and "applies_to_narrowly_supported_independent_of_witness_path"
        in request["response_preflight_contract"]["required_before_emit"]
        and "semantic_superpopulation_or_structural_support_has_target_scope_member"
        in request["response_preflight_contract"]["required_before_emit"]
        and "each_witness_path_covered_by_applies_to"
        not in request["response_preflight_contract"]["required_before_emit"]
        and request["accounted_input_ids"]
        for request in client.reduction_requests
    )
    assert "evidence path is allowed to\ndiffer from `applies_to`" in (
        REDUCE_SYSTEM_PROMPT
    )
    assert "exact typed literals, not synonyms, aliases, or a\ntype hierarchy" in (
        REDUCE_SYSTEM_PROMPT
    )
    for request in client.reduction_requests:
        context = request["population_context"]
        assert context["context_version"] == "population-planning-context-1.3"
        assert context["detail_level"] == "witness_grounding"
        expected_witnesses = set(request["accounted_input_ids"])
        bindings = context["witness_bindings"]
        assert {item["witness_id"] for item in bindings} == expected_witnesses
        fact_by_id = {
            item["observation_id"]: item for item in context["query_facts"]
        }
        for binding in bindings:
            assert binding["query_fact_refs"]
            assert set(binding["query_fact_refs"]) <= set(fact_by_id)
            if binding["entity_id"] is not None:
                entity_fact = fact_by_id[binding["entity_id"]]
                assert entity_fact["path"] == binding["path"]
                assert entity_fact["values"]["kind"] == binding["unit_kind"]
        assert "observed_value_catalog" not in context


def test_every_partition_request_respects_the_item_and_byte_caps(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    for ordinal in range(5):
        package = repository / "src" / f"bounded_package_{ordinal}"
        package.mkdir()
        (package / "__init__.py").write_text(
            "def preserve_local_shape() -> int:\n    return 1\n",
            encoding="utf-8",
        )
    client = _RecordingReductionClient(
        category="bounded-partition-regression",
        rule="Contributors preserve the observed fixture convention.",
    )
    config = config_for_tests(
        semantic={"max_reduce_items_per_request": 2}
    )
    result = build_fixture(
        repository,
        tmp_path / "bounded-partition-run",
        config=config,
        client=client,
    )

    assert result.current_source is not None
    assert result.current_source.reduction.index.status == "complete"
    assert client.reduction_requests
    assert {item["mode"] for item in client.reduction_requests} == {"component_partition"}
    assert all(len(item["accounted_input_ids"]) <= 2 for item in client.reduction_requests)
    assert all(len(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
               <= config.semantic.max_reduce_request_bytes for item in client.reduction_requests)


def test_reduction_repair_repeats_exact_top_level_accounting_contract() -> None:
    # Evidence-range aliases are valid provider-visible witness bindings and
    # must receive the same targeted repair facts as observation IDs.
    witness_id = "SEMRANGE-" + "a" * 24
    prompt = _repair_prompt(
        json.dumps(
            {
                "stage": "current_hypothesis_reduce",
                "mode": "component_partition",
                "accounted_input_ids": ["HYP-a", "HYP-b"],
                "population_context": {
                    "witness_bindings": [
                        {
                            "witness_id": witness_id,
                            "path": "src/pkg.py",
                            "query_fact_refs": ["PYFILE-a"],
                        }
                    ]
                },
            }
        ),
        '{"invalid":true}',
        [f"witness {witness_id} is outside the superpopulation"],
    )
    repair = json.loads(prompt)
    assert repair["original_request"]["mode"] == "component_partition"
    assert repair["original_request"]["accounted_input_ids"] == [
        "HYP-a",
        "HYP-b",
    ]
    assert repair["invalid_response"] == {
        "content_is_untrusted": True,
        "digest": repair["invalid_response"]["digest"],
        "original_bytes": len('{"invalid":true}'),
        "excerpt": '{"invalid":true}',
        "excerpt_truncated": False,
    }
    assert repair["validation_errors"] == [
        f"witness {witness_id} is outside the superpopulation"
    ]
    assert repair["targeted_witness_context"]["bindings"] == [
        {
            "witness_id": witness_id,
            "path": "src/pkg.py",
            "paths": None,
            "candidate_query_fact_refs": ["PYFILE-a"],
            "candidate_query_fact_ref_count": 1,
        }
    ]
    assert any(
        "candidate_query_fact_refs" in item
        for item in repair["repair_directives"]
    )
    assert "no more than 64 hypotheses" in repair["instruction"]
    assert "within the supplied input boundary" in repair["instruction"]
    assert "untrusted data" in repair["instruction"]


def test_reduction_repair_retains_bounded_untrusted_response_context() -> None:
    invalid = "head:" + ("中" * 20_000) + ":tail"
    repair = json.loads(
        _repair_prompt(
            json.dumps(
                {
                    "stage": "current_hypothesis_reduce",
                    "mode": "component_partition",
                    "hypothesis_budget": {"max_output_hypotheses": 7},
                    "accounted_input_ids": ["HYP-a"],
                }
            ),
            invalid,
            ["hypotheses must contain at most 7 items"],
        )
    )
    reflected = repair["invalid_response"]
    assert reflected["content_is_untrusted"] is True
    assert reflected["excerpt_truncated"] is True
    assert reflected["original_bytes"] == len(invalid.encode("utf-8"))
    assert (
        len(reflected["excerpt"].encode("utf-8"))
        <= MAX_REDUCE_REPAIR_INVALID_RESPONSE_BYTES
    )
    assert reflected["excerpt"].startswith("head:")
    assert reflected["excerpt"].endswith(":tail")
    assert "no more than 7 hypotheses" in repair["instruction"]


def _draft_fixture_input(index: int, hypothesis_count: int) -> _ReduceInput:
    hypotheses = [
        {
            "rule": f"Preserve child {index} hypothesis {ordinal}.",
            "category": "partition-draft-fixture",
            "proposed_claim_kind": "observed_convention",
            "proposed_strength": "observed",
            "applies_to": ["**/*.py"],
            "condition": "When modifying the represented fixture population.",
            "witness_refs": [f"SEMOBS-{index:024x}"],
            "explicit_context_refs": [],
            "superpopulation_spec": {
                "op": "universe",
                "universe": "file",
            },
            "predicate_mode": "semantic",
            "structural_predicate": None,
            "semantic_predicate": {
                "eligibility_question": "Is this fixture member eligible?",
                "support_question": "Does this fixture member support the rule?",
                "allowed_reason_codes": ["fixture_reason"],
                "expected_evidence_form": "A fixture fact.",
                "ambiguity_instruction": "Return unknown if ambiguous.",
                "repository_specificity_test": "Use only fixture evidence.",
                "protocol_version": "semantic-member-classification-1.0",
            },
            "independence_spec": {"mode": "clone_aware_observation"},
            "limitations": [],
        }
        for ordinal in range(hypothesis_count)
    ]
    payload = {
        "mode": "component_partition",
        "accounted_input_ids": [f"SOURCE-{index}"],
        "outcome": "hypotheses" if hypotheses else "no_hypothesis",
        "hypotheses": hypotheses,
        "hypothesis_ordinals": list(range(hypothesis_count)),
        "reason_code": "fixture_hypotheses" if hypotheses else "fixture_empty",
        "host_population_repairs": [],
        "quarantined_hypotheses": [],
        "host_cardinality_selection": None,
        "host_transport": None,
    }
    return _ReduceInput(
        stable_id=f"CHILD-{index}",
        digest=payload_digest(payload),
        payload=payload,
        ref_kind="semantic_observation",
        )








def test_effective_reducer_output_limit_is_non_expanding_but_schema_stays_wide() -> None:
    settings = config_for_tests(
        semantic={"max_hypotheses_per_component": 8}
    ).semantic
    observations = [
        _ReduceInput(
            stable_id=f"OBS-{index}",
            digest=f"{index:064x}",
            payload={"observation_id": f"OBS-{index}"},
            ref_kind="semantic_observation",
        )
        for index in range(12)
    ]

    assert _effective_output_hypothesis_limit(
        "component_partition", observations[:5], settings
    ) == 5
    assert _effective_output_hypothesis_limit(
        "component_partition", observations, settings
    ) == 8
    assert ProviderReductionResponse.model_json_schema()["properties"][
        "hypotheses"
    ]["maxItems"] == 64








def test_singleton_partitions_reach_final_admission_without_merge_gaps(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "stalled-survivor-run"
    inference_cache = tmp_path / "stalled-survivor-inference-cache"
    config = config_for_tests(
        semantic={"max_reduce_items_per_request": 1, "max_hypotheses_per_component": 1}
    )
    client = _LosslessReductionClient()
    result = build_fixture(
        repository,
        output,
        config=config,
        client=client,
        inference_cache=inference_cache,
    )

    assert result.current_source is not None
    reduction = result.current_source.reduction
    assert reduction.index.status == "complete"
    assert reduction.hypotheses
    assert len(reduction.hypotheses) > config.semantic.max_hypotheses_per_component
    for hypothesis in reduction.hypotheses:
        expected_origins = {event.inference_id for event in reduction.events
                            if event.winning_root_id == hypothesis.candidate_root_id}
        assert set(hypothesis.origin_inference_ids) == expected_origins
    assert {item.rule for item in reduction.hypotheses} == (
        client.partition_rules
    )
    assert {request["mode"] for request in client.reduction_requests} == {"component_partition"}
    assert validate_run_directory(output, repository_root=repository) == []

    replay_client = _LosslessReductionClient()
    replayed = build_fixture(
        repository,
        tmp_path / "stalled-survivor-replay-run",
        config=config,
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.reduction_requests == []
    assert replayed.current_source is not None
    assert replayed.current_source.reduction == reduction


def test_total_hypothesis_cap_records_final_suffix_as_cost_omitted_and_replays(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "total-hypothesis-cap-run"
    inference_cache = tmp_path / "total-hypothesis-cap-cache"
    config = config_for_tests(
        semantic={
            "max_reduce_items_per_request": 1,
            "max_total_hypotheses": 1,
        }
    )
    client = _LosslessReductionClient()
    result = build_fixture(
        repository,
        output,
        config=config,
        client=client,
        inference_cache=inference_cache,
    )

    assert result.current_source is not None
    reduction = result.current_source.reduction
    assert len(client.partition_rules) > config.semantic.max_total_hypotheses
    assert len(reduction.hypotheses) == 1
    cap_omitted = [
        item
        for item in reduction.events
        if "max_total_hypotheses" in item.reason_codes
    ]
    assert cap_omitted
    assert all(item.normalization == "cost_omitted" for item in cap_omitted)
    assert not any(
        item.normalization == "invalid"
        and "not_promoted_by_repository_reduce" in item.reason_codes
        for item in reduction.events
    )
    assert reduction.index.status == "partial"
    assert "max_total_hypotheses" in reduction.index.incomplete_reasons
    assert reduction.index.generated_candidate_events == len(reduction.events)
    assert reduction.index.cost_omitted_candidate_events == sum(
        item.normalization == "cost_omitted" for item in reduction.events
    )
    assert reduction.index.invalid_candidate_events == sum(
        item.normalization == "invalid" for item in reduction.events
    )
    assert validate_run_directory(output, repository_root=repository) == []

    replay_client = _LosslessReductionClient()
    replayed = build_fixture(
        repository,
        tmp_path / "total-hypothesis-cap-replay-run",
        config=config,
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.reduction_requests == []
    assert replayed.current_source is not None
    assert replayed.current_source.reduction == reduction


def test_total_hypothesis_cap_counts_unique_identities_before_duplicates(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests(
        semantic={
            "max_reduce_items_per_request": 4,
            "max_total_hypotheses": 2,
        }
    )
    reductions = []
    response_category_orders: list[list[str]] = []
    for reverse in (False, True):
        client = _LosslessReductionClient(
            alias_component_witnesses=True,
            reverse_alias_duplicates=reverse,
        )
        result = build_fixture(
            repository,
            tmp_path / f"unique-before-cap-{reverse}",
            config=config,
            client=client,
        )
        assert result.current_source is not None
        reduction = result.current_source.reduction
        reductions.append(reduction)
        partition_record = next(
            record
            for record in result.inference_records
            if record.stage == "current_hypothesis_reduce"
            and record.mode == "component_partition"
            and record.structured_response is not None
            and any(
                item["category"] == "lossless-direct-duplicate"
                for item in record.structured_response["hypotheses"]
            )
        )
        response_category_orders.append(
            [
                str(item["category"])
                for item in partition_record.structured_response["hypotheses"]
            ]
        )
        assert reduction.index.unique_candidate_roots == 2
        assert len(reduction.hypotheses) == 2
        assert any(
            item.normalization == "merged_duplicate"
            for item in reduction.events
        )
        assert any(
            item.normalization == "cost_omitted"
            and item.reason_codes == ["max_total_hypotheses"]
            for item in reduction.events
        )
    assert response_category_orders[0][:2] == list(
        reversed(response_category_orders[1][:2])
    )
    assert response_category_orders[0][2:] == response_category_orders[1][2:]
    assert {
        item.hypothesis_id for item in reductions[0].hypotheses
    } == {
        item.hypothesis_id for item in reductions[1].hypotheses
    }
    assert reductions[0].index.unique_candidate_roots == (
        reductions[1].index.unique_candidate_roots
    )


def test_invalid_duplicate_payload_cannot_poison_valid_identity(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _LosslessReductionClient(
        alias_component_witnesses=True,
        invalidate_alias_duplicate=True,
    )
    result = build_fixture(
        repository,
        tmp_path / "invalid-duplicate-payload-run",
        config=config_for_tests(),
        client=client,
    )

    assert result.current_source is not None
    reduction = result.current_source.reduction
    invalid_events = [
        item
        for item in reduction.events
        if item.normalization == "invalid"
        and item.reason_codes == ["invalid_evidence_reference"]
    ]
    assert invalid_events
    assert all(item.winning_root_id is None for item in invalid_events)
    for invalid_event in invalid_events:
        assert any(
            item.draft_digest == invalid_event.draft_digest
            and item.normalization in {"unique", "merged_duplicate"}
            and item.winning_root_id is not None
            for item in reduction.events
        )
    assert reduction.hypotheses








def test_reducer_host_supplies_bookkeeping_before_schema_validation() -> None:
    normalized = _normalize_response(
        {
            "outcome": "no_hypothesis",
            "hypotheses": [],
            "reason_code": "no_repository_hypothesis",
        },
        mode="component_partition",
        input_ids=["HYP-a", "HYP-b"],
    )
    response = ReductionResponse.model_validate(normalized)
    assert response.mode == "component_partition"
    assert response.accounted_input_ids == ["HYP-a", "HYP-b"]


@pytest.mark.parametrize(
    ("predicate_mode", "inactive_field", "repair_code"),
    [
        (
            "semantic",
            "structural_predicate",
            "inactive_structural_predicate_defaulted_null",
        ),
        (
            "structural",
            "semantic_predicate",
            "inactive_semantic_predicate_defaulted_null",
        ),
    ],
)
def test_reducer_defaults_only_the_omitted_inactive_predicate_branch(
    predicate_mode: str,
    inactive_field: str,
    repair_code: str,
) -> None:
    draft = _mock_valid_reducer_draft()
    draft["predicate_mode"] = predicate_mode
    if predicate_mode == "structural":
        draft["structural_predicate"] = {
            "eligible_spec": draft["superpopulation_spec"],
            "support_spec": draft["superpopulation_spec"],
        }
    draft.pop(inactive_field)

    normalized = _normalize_response(
        {
            "outcome": "hypotheses",
            "hypotheses": [draft],
            "reason_code": "candidate",
        },
        mode="component_partition",
        input_ids=["OBS-1"],
    )
    response = ReductionResponse.model_validate(normalized)

    assert getattr(response.hypotheses[0], inactive_field) is None
    assert [
        item.repair_code for item in response.host_population_repairs
    ] == [repair_code]


def test_reducer_provider_schema_excludes_and_live_output_discards_host_fields() -> None:
    provider_properties = ProviderReductionResponse.model_json_schema()[
        "properties"
    ]
    assert set(provider_properties) == {
        "outcome",
        "hypotheses",
        "reason_code",
    }
    assert provider_properties["hypotheses"]["maxItems"] == 64
    assert ReductionSelectionResponse.model_json_schema()["properties"][
        "selected_hypothesis_ordinals"
    ]["maxItems"] == 64
    assert MAX_REDUCE_SELECTION_CANDIDATES == 128
    assert MAX_REDUCE_SELECTION_PROMPT_BYTES == 131_072
    assert ReductionSelectionResponse.model_validate(
        {"selected_hypothesis_ordinals": list(range(64))}
    ).selected_hypothesis_ordinals == list(range(64))
    with pytest.raises(ValueError):
        ReductionSelectionResponse.model_validate(
            {"selected_hypothesis_ordinals": list(range(65))}
        )
    assert {
        "mode",
        "accounted_input_ids",
        "hypothesis_ordinals",
        "host_population_repairs",
        "quarantined_hypotheses",
        "host_cardinality_selection",
    } <= set(ReductionResponse.model_json_schema()["properties"])

    sanitized = _validate_raw_response(
        {
            "outcome": "no_hypothesis",
            "hypotheses": [],
            "reason_code": "none",
            "hypothesis_ordinals": [999],
            "host_population_repairs": [{"forged": True}],
            "quarantined_hypotheses": [{"forged": True}],
            "host_cardinality_selection": {"forged": True},
        }
    )
    assert sanitized == {
        "outcome": "no_hypothesis",
        "hypotheses": [],
        "reason_code": "none",
    }
    semantic_extra = dict(sanitized)
    semantic_extra["unrecognized_semantic_field"] = True
    normalized = _normalize_response(
        semantic_extra,
        mode="component_partition",
        input_ids=["HYP-a"],
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ReductionResponse.model_validate(normalized)


def test_valid_live_reducer_host_echo_is_recomputed_without_repair(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "host-field-echo-run",
        config=config_for_tests(),
        client=_HostFieldEchoReductionClient(
            category="host-field-echo-regression",
            rule="Valid semantic output survives forged host metadata.",
        ),
    )

    assert result.current_source is not None
    reducer_records = [
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
    ]
    assert reducer_records
    assert all(
        len(record.attempts) == 1
        and record.attempts[0].parsed
        and record.terminal_status == "success"
        for record in reducer_records
    )
    for record in reducer_records:
        assert record.structured_response is not None
        response = ReductionResponse.model_validate(
            record.structured_response
        )
        assert response.hypothesis_ordinals == list(
            range(len(response.hypotheses))
        )
        assert response.host_population_repairs == []
        assert response.quarantined_hypotheses == []
        assert response.host_cardinality_selection is None


def _mock_valid_reducer_draft() -> dict[str, object]:
    response = MockLLMClient().complete(
        "",
        json.dumps(
            {
                "stage": "current_hypothesis_reduce",
                "mode": "component_partition",
                "inputs": [{}],
                "accounted_input_ids": ["OBS-1"],
            }
        ),
    )
    return json.loads(response.content)["hypotheses"][0]


@pytest.mark.parametrize(
    "invalid_selection, expected_error",
    [
        ([0, 0], "sorted and unique"),
        ([17], "out of range"),
        (list(range(MAX_REDUCE_SELECTION_CANDIDATES + 1)), "at most 128"),
    ],
)
def test_cardinality_selection_rejects_every_illegal_ordinal_boundary(
    invalid_selection: list[int],
    expected_error: str,
) -> None:
    settings = config_for_tests(
        semantic={
            "max_hypotheses_per_component": 8,
        }
    ).semantic
    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
    )
    original = _mock_valid_reducer_draft()
    normalized = controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Preserve exact overflow draft {ordinal}.",
                }
                for ordinal in range(17)
            ],
            "reason_code": "configured_limit_fixture",
        }
    )
    assert ReductionResponse.model_validate(normalized)
    assert controller.response_model_for_repair(
        "",
        ["reducer response exceeds effective request hypothesis limit 8"],
    ) is ReductionSelectionResponse
    with pytest.raises(ValueError, match=expected_error):
        controller.normalize(
            {"selected_hypothesis_ordinals": invalid_selection}
        )


def test_overselected_cardinality_response_repairs_before_terminal_prefix() -> None:
    settings = config_for_tests(
        semantic={
            "max_hypotheses_per_component": 8,
        }
    ).semantic
    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
    )
    original = _mock_valid_reducer_draft()
    controller.bind_response_digest("a" * 64)
    normalized = controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Preserve terminal prefix draft {ordinal}.",
                }
                for ordinal in range(17)
            ],
            "reason_code": "terminal_prefix_fixture",
        }
    )
    assert len(ReductionResponse.model_validate(normalized).hypotheses) == 17
    assert controller.response_model_for_repair(
        "",
        ["reducer response exceeds effective request hypothesis limit 8"],
    ) is ReductionSelectionResponse

    overselected = {
        "selected_hypothesis_ordinals": list(range(9)),
    }
    ordinary_normalized = controller.normalize(overselected)
    assert ordinary_normalized == overselected
    with pytest.raises(ValueError) as selection_error:
        ReductionResponse.model_validate(ordinary_normalized)

    terminal = controller.terminal_parse_response(
        ordinary_normalized,
        selection_error.value,
    )
    assert terminal.hypothesis_ordinals == list(range(8))
    assert terminal.reason_code == (
        "overselected_cardinality_suffix_quarantined_after_repair"
    )
    provenance = terminal.host_cardinality_selection
    assert provenance is not None
    assert provenance.selected_source_ordinals == list(range(8))
    assert provenance.cardinality_not_selected_source_ordinals == list(
        range(8, 17)
    )
    assert provenance.cardinality_not_selected_source_draft_digests == [
        provenance.source_draft_digests[ordinal]
        for ordinal in range(8, 17)
    ]


def test_post_selection_generic_repair_reflects_reconstructed_drafts() -> None:
    settings = config_for_tests(
        semantic={
            "max_hypotheses_per_component": 8,
        }
    ).semantic
    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
    )
    original = _mock_valid_reducer_draft()
    controller.bind_response_digest("a" * 64)
    normalized = controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Preserve reconstructed draft {ordinal}.",
                }
                for ordinal in range(17)
            ],
            "reason_code": "post_selection_context_fixture",
        }
    )
    assert ReductionResponse.model_validate(normalized)
    assert controller.response_model_for_repair(
        "",
        ["reducer response exceeds effective request hypothesis limit 8"],
    ) is ReductionSelectionResponse
    reconstructed = controller.normalize(
        {"selected_hypothesis_ordinals": [0, 1]}
    )
    assert ReductionResponse.model_validate(reconstructed)

    repair = json.loads(
        controller.repair_prompt(
            json.dumps(
                {
                    "stage": "current_hypothesis_reduce",
                    "mode": "component_partition",
                    "hypothesis_budget": {"max_output_hypotheses": 8},
                    "accounted_input_ids": ["HYP-a"],
                }
            ),
            '{"selected_hypothesis_ordinals":[0,1]}',
            ["synthetic post-selection population failure"],
        )
    )
    reflected = json.loads(repair["invalid_response"]["excerpt"])
    assert [item["rule"] for item in reflected["hypotheses"]] == [
        "Preserve reconstructed draft 0.",
        "Preserve reconstructed draft 1.",
    ]
    assert "selected_hypothesis_ordinals" not in reflected


def test_all_population_invalid_overflow_drafts_fail_closed_before_selection() -> None:
    settings = config_for_tests(
        semantic={"max_hypotheses_per_component": 8}
    ).semantic

    def reject_draft(_draft: HypothesisDraft) -> None:
        raise ValueError(
            "complete superpopulation is empty despite required witness_refs"
        )

    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
        validate_overflow_draft=reject_draft,
    )
    original = _mock_valid_reducer_draft()
    controller.bind_response_digest("a" * 64)
    normalized = controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Invalid overflow draft {ordinal}.",
                }
                for ordinal in range(17)
            ],
            "reason_code": "all_invalid_overflow_fixture",
        }
    )
    response = ReductionResponse.model_validate(normalized)
    assert response.outcome == "no_hypothesis"
    assert response.hypotheses == []
    assert response.reason_code == (
        "all_overflow_hypotheses_preflight_excluded"
    )
    provenance = response.host_cardinality_selection
    assert provenance is not None
    assert provenance.source_candidate_count == 17
    assert provenance.selected_source_ordinals == []
    assert provenance.cardinality_not_selected_source_ordinals == []
    assert len(provenance.preflight_excluded_hypotheses) == 17
    assert {
        item.reason_code
        for item in provenance.preflight_excluded_hypotheses
    } == {"population_preflight_empty"}
    assert controller.selection_state is None


def test_preflight_excluded_ordinal_cannot_be_selected() -> None:
    settings = config_for_tests(
        semantic={"max_hypotheses_per_component": 8}
    ).semantic

    def reject_one(draft: HypothesisDraft) -> None:
        if str(draft.rule).endswith("9."):
            raise ValueError(
                "witness SEMOBS-"
                + "a" * 24
                + " is outside the superpopulation"
            )

    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
        validate_overflow_draft=reject_one,
    )
    original = _mock_valid_reducer_draft()
    controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Selectable overflow draft {ordinal}.",
                }
                for ordinal in range(17)
            ],
            "reason_code": "excluded_ordinal_fixture",
        }
    )
    assert controller.response_model_for_repair(
        "",
        ["reducer response exceeds effective request hypothesis limit 8"],
    ) is ReductionSelectionResponse
    selection_prompt = json.loads(
        controller.repair_prompt("{}", "{}", ["cardinality"])
    )
    assert 9 not in [item["ordinal"] for item in selection_prompt["candidates"]]
    assert selection_prompt["source_candidate_count"] == 17
    assert selection_prompt["excluded_candidate_count"] == 1
    with pytest.raises(ValueError, match="failed host population preflight"):
        controller.normalize({"selected_hypothesis_ordinals": [9]})


def test_selection_prompt_retains_the_128_candidate_boundary() -> None:
    settings = config_for_tests().semantic
    controller = _ReductionRepairController(
        mode="component_partition",
        input_ids=["HYP-a"],
        settings=settings,
    )
    original = _mock_valid_reducer_draft()
    controller.bind_response_digest("a" * 64)
    controller.normalize(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    **json.loads(json.dumps(original)),
                    "rule": f"Preserve candidate boundary draft {ordinal}.",
                }
                for ordinal in range(128)
            ],
            "reason_code": "selection_candidate_boundary_fixture",
        }
    )
    assert controller.response_model_for_repair(
        "",
        [
            "hypotheses: List should have at most 64 items after validation, "
            "not 128"
        ],
    ) is ReductionSelectionResponse
    prompt = controller.repair_prompt("{}", "{}", ["cardinality"])
    selection = json.loads(prompt)
    assert selection["source_candidate_count"] == 128
    assert selection["candidate_count"] == 128
    assert [item["ordinal"] for item in selection["candidates"]] == list(
        range(128)
    )
    assert len(prompt.encode("utf-8")) <= MAX_REDUCE_SELECTION_PROMPT_BYTES


def test_partial_population_preflight_skips_provider_member_admission(
    monkeypatch,
) -> None:
    draft = HypothesisDraft.model_validate(_mock_valid_reducer_draft())
    settings = config_for_tests().semantic
    observed_admission: list[object] = []

    def admission_count(*_args, **_kwargs):
        value = SimpleNamespace(
            in_scope_member_count=513,
            provider_eligible_in_scope_member_count=7,
        )
        observed_admission.append(value)
        return value

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce."
        "count_population_members_for_admission",
        admission_count,
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce.evaluate_population",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="partial",
            matched_count=513,
        ),
    )

    cost = _preflight_population(
        draft,
        observations=SimpleNamespace(),
        semantic_map=SimpleNamespace(observations=()),
        semantic_units=SimpleNamespace(units=()),
        settings=settings,
    )

    assert observed_admission == []
    assert cost == 0


def test_evidence_witness_may_be_outside_target_applies_to(
    monkeypatch,
) -> None:
    payload = _mock_valid_reducer_draft()
    payload["applies_to"] = ["src/**/*.py"]
    draft = HypothesisDraft.model_validate(payload)
    witness_id = draft.witness_refs[0]

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce."
        "count_population_members_for_admission",
        lambda *_args, **_kwargs: SimpleNamespace(
            in_scope_member_count=1,
            provider_eligible_in_scope_member_count=1,
        ),
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce.evaluate_population",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="partial",
            matched_count=1,
        ),
    )

    cost = _preflight_population(
        draft,
        observations=SimpleNamespace(),
        semantic_map=SimpleNamespace(
            observations=(
                SimpleNamespace(
                    observation_id=witness_id,
                    primary_unit_id="UNIT-outside",
                ),
            )
        ),
        semantic_units=SimpleNamespace(
            units=(
                SimpleNamespace(
                    unit_id="UNIT-outside",
                    path="tests/outside.py",
                ),
            )
        ),
        settings=config_for_tests().semantic,
    )

    assert cost == 0


def test_complete_semantic_population_requires_a_target_scope_member(
    monkeypatch,
) -> None:
    payload = _mock_valid_reducer_draft()
    payload["applies_to"] = ["src/**/*.py"]
    draft = HypothesisDraft.model_validate(payload)

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce."
        "count_population_members_for_admission",
        lambda *_args, **_kwargs: SimpleNamespace(
            in_scope_member_count=0,
            provider_eligible_in_scope_member_count=0,
        ),
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce.evaluate_population",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="complete",
            matched_count=1,
            members=(
                SimpleNamespace(
                    observation_id="FILE-test-witness",
                    path="tests/test_witness.py",
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError, match="no deterministic target-scope member"
    ) as raised:
        _preflight_population(
            draft,
            observations=SimpleNamespace(),
            semantic_map=SimpleNamespace(observations=()),
            semantic_units=SimpleNamespace(units=()),
            settings=config_for_tests().semantic,
        )
    assert _quarantine_reason(raised.value) == (
        "population_preflight_target_scope_empty"
    )


def test_complete_semantic_population_keeps_off_scope_witness_with_target_member(
    monkeypatch,
) -> None:
    payload = _mock_valid_reducer_draft()
    payload["applies_to"] = ["src/**/*.py"]
    draft = HypothesisDraft.model_validate(payload)
    witness_id = draft.witness_refs[0]
    members = (
        SimpleNamespace(
            observation_id="FILE-test-witness",
            path="tests/test_witness.py",
        ),
        SimpleNamespace(
            observation_id="FILE-production-target",
            path="src/package/runtime.py",
        ),
    )

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce."
        "count_population_members_for_admission",
        lambda *_args, **_kwargs: SimpleNamespace(
            in_scope_member_count=1,
            provider_eligible_in_scope_member_count=1,
        ),
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce.evaluate_population",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="complete",
            matched_count=2,
            members=members,
        ),
    )

    cost = _preflight_population(
        draft,
        observations=SimpleNamespace(),
        semantic_map=SimpleNamespace(
            observations=(
                SimpleNamespace(
                    observation_id=witness_id,
                    primary_unit_id="UNIT-test-witness",
                    evidence_refs=("FILE-test-witness",),
                ),
            )
        ),
        semantic_units=SimpleNamespace(
            units=(
                SimpleNamespace(
                    unit_id="UNIT-test-witness",
                    path="tests/test_witness.py",
                    file_id="FILE-test-witness",
                    entity_id=None,
                ),
            )
        ),
        settings=config_for_tests().semantic,
    )

    assert cost == 1


def test_complete_structural_population_requires_in_scope_support(
    monkeypatch,
) -> None:
    payload = _mock_valid_reducer_draft()
    payload["applies_to"] = ["src/**/*.py"]
    payload["predicate_mode"] = "structural"
    payload["semantic_predicate"] = None
    payload["structural_predicate"] = {
        "eligible_spec": payload["superpopulation_spec"],
        "support_spec": payload["superpopulation_spec"],
    }
    draft = HypothesisDraft.model_validate(payload)
    witness_id = draft.witness_refs[0]
    population = SimpleNamespace(
        status="complete",
        matched_count=1,
        members=(
            SimpleNamespace(
                observation_id="FILE-test-witness",
                path="tests/test_witness.py",
            ),
        ),
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.reduce.evaluate_population",
        lambda *_args, **_kwargs: population,
    )

    with pytest.raises(
        ValueError, match="structural support has no deterministic target-scope member"
    ) as raised:
        _preflight_population(
            draft,
            observations=SimpleNamespace(),
            semantic_map=SimpleNamespace(
                observations=(
                    SimpleNamespace(
                        observation_id=witness_id,
                        primary_unit_id="UNIT-test-witness",
                        evidence_refs=("FILE-test-witness",),
                    ),
                )
            ),
            semantic_units=SimpleNamespace(
                units=(
                    SimpleNamespace(
                        unit_id="UNIT-test-witness",
                        path="tests/test_witness.py",
                        file_id="FILE-test-witness",
                        entity_id=None,
                    ),
                )
            ),
            settings=config_for_tests().semantic,
        )
    assert _quarantine_reason(raised.value) == (
        "population_preflight_target_scope_empty"
    )


def test_over_cardinality_reducer_selects_exact_drafts_and_replays_cache(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "cardinality-selection-run"
    inference_cache = tmp_path / "cardinality-selection-cache"
    client = _CardinalitySelectionClient(
        overflow_count=65,
        selected_ordinals=[1, 64],
    )
    result = build_fixture(
        repository,
        output,
        config=config_for_tests(semantic={"max_hypotheses_per_component": 2}),
        client=client,
        inference_cache=inference_cache,
    )

    assert result.current_source is not None
    assert (
        result.current_source.reduction.index.index_version
        == "current-hypothesis-reduction-index-1.13"
    )
    partition_result = next(
        item
        for item in result.current_source.reduction.call_results
        if tuple(item.input_ids) == client.overflow_input_ids
    )
    assert partition_result.host_cardinality_selection is not None
    partition_record = next(
        item
        for item in result.inference_records
        if item.request_id == partition_result.inference_id
    )
    assert partition_record.terminal_status == "success"
    assert partition_record.protocol_version == REDUCE_PROTOCOL_VERSION
    assert (
        partition_record.response_schema_version
        == REDUCE_RESPONSE_SCHEMA_VERSION
    )
    assert [item.parsed for item in partition_record.attempts] == [False, True]
    assert partition_record.structured_response is not None
    assert partition_record.structured_response["hypothesis_ordinals"] == [
        1,
        64,
    ]
    assert partition_record.structured_response["host_population_repairs"] == []
    assert partition_record.structured_response["quarantined_hypotheses"] == []
    assert partition_record.structured_response["hypotheses"] == [
        client.overflow_drafts[ordinal]
        for ordinal in [1, 64]
    ]
    provenance = partition_record.structured_response[
        "host_cardinality_selection"
    ]
    assert provenance is not None
    assert provenance["source_candidate_count"] == 65
    assert provenance["selected_source_ordinals"] == [1, 64]
    assert provenance["preflight_excluded_hypotheses"] == []
    assert provenance["cardinality_not_selected_source_ordinals"] == [
        ordinal for ordinal in range(65) if ordinal not in {1, 64}
    ]
    assert provenance["source_response_digest"] == (
        partition_record.attempts[0].response_digest
    )
    assert provenance["selected_source_draft_digests"] == [
        provenance["source_draft_digests"][ordinal]
        for ordinal in [1, 64]
    ]
    assert partition_result.host_cardinality_selection == (
        ReductionResponse.model_validate(
            partition_record.structured_response
        ).host_cardinality_selection
    )
    tampered = json.loads(
        json.dumps(partition_record.structured_response)
    )
    tampered["host_cardinality_selection"][
        "selected_source_ordinals"
    ] = [0, 64]
    with pytest.raises(ValueError, match="selected draft digests differ"):
        ReductionResponse.model_validate(tampered)
    incomplete_partition = json.loads(
        json.dumps(partition_record.structured_response)
    )
    incomplete_provenance = incomplete_partition[
        "host_cardinality_selection"
    ]
    incomplete_provenance[
        "cardinality_not_selected_source_ordinals"
    ].pop()
    incomplete_provenance[
        "cardinality_not_selected_source_draft_digests"
    ].pop()
    incomplete_provenance["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in incomplete_provenance.items()
            if key != "artifact_digest"
        }
    )
    with pytest.raises(ValueError, match="do not cover"):
        ReductionResponse.model_validate(incomplete_partition)
    tampered_source_attempt = json.loads(
        json.dumps(partition_record.structured_response)
    )
    source_provenance = tampered_source_attempt[
        "host_cardinality_selection"
    ]
    source_provenance["source_response_digest"] = "f" * 64
    source_provenance["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in source_provenance.items()
            if key != "artifact_digest"
        }
    )
    parsed_tampered_source = ReductionResponse.model_validate(
        tampered_source_attempt
    )
    with pytest.raises(ValueError, match="overflow inference attempt"):
        _validate_cardinality_selection_attempt_binding(
            parsed_tampered_source,
            partition_record,
            effective_limit=2,
        )
    assert {
        item.rule for item in result.current_source.reduction.hypotheses
    } >= {
        "Preserve exact overflow draft 1.",
        "Preserve exact overflow draft 64.",
    }
    partition_events = [
        item
        for item in result.current_source.reduction.events
        if item.inference_id == partition_record.request_id
    ]
    assert len(partition_events) == 65
    assert sum(
        item.normalization == "selection_omitted"
        for item in partition_events
    ) == 63
    assert result.current_source.reduction.index.status == "partial"
    assert "cardinality_not_selected" in (
        result.current_source.reduction.index.incomplete_reasons
    )
    assert (
        result.current_source.reduction.index.selection_omitted_candidate_events
        >= 63
    )
    assert {
        item.source_response_ordinal
        for item in partition_events
        if item.normalization != "selection_omitted"
    } == {1, 64}
    assert len(client.selection_requests) == 1
    selection = client.selection_requests[0]
    assert selection["candidate_count"] == 65
    assert selection["max_selected_hypotheses"] == 2
    assert [item["ordinal"] for item in selection["candidates"]] == list(
        range(65)
    )
    assert len(
        json.dumps(
            selection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ) <= MAX_REDUCE_SELECTION_PROMPT_BYTES
    assert all(
        set(schema["properties"]) == {"selected_hypothesis_ordinals"}
        for schema in client.selection_schemas
    )
    assert all(
        set(schema["properties"])
        == {"outcome", "hypotheses", "reason_code"}
        for schema in client.initial_reducer_schemas
    )
    assert validate_run_directory(output, repository_root=repository) == []

    replay_client = _CardinalitySelectionClient(
        overflow_count=65,
        selected_ordinals=[0],
    )
    replayed = build_fixture(
        repository,
        tmp_path / "cardinality-selection-replay-run",
        config=config_for_tests(semantic={"max_hypotheses_per_component": 2}),
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.reducer_calls == 0
    assert replayed.current_source is not None
    assert replayed.current_source.reduction == result.current_source.reduction


def test_terminal_overselection_prefix_replays_live_cache_and_offline(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "terminal-overselection-run"
    inference_cache = tmp_path / "terminal-overselection-cache"
    client = _CardinalitySelectionClient(
        overflow_count=65,
        selected_ordinals=[0, 1, 2],
    )
    result = build_fixture(
        repository,
        output,
        config=config_for_tests(semantic={"max_hypotheses_per_component": 2}),
        client=client,
        inference_cache=inference_cache,
    )

    assert result.current_source is not None
    partition_record = next(
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    )
    assert partition_record.terminal_status == "success"
    assert [attempt.parsed for attempt in partition_record.attempts] == [
        False,
        False,
        True,
    ]
    assert len(client.selection_requests) == 2
    assert all(
        request["max_selected_hypotheses"] == 2
        for request in client.selection_requests
    )
    assert all(
        schema["properties"]["selected_hypothesis_ordinals"]["maxItems"]
        == 64
        for schema in client.selection_schemas
    )
    assert partition_record.structured_response is not None
    response = ReductionResponse.model_validate(
        partition_record.structured_response
    )
    assert response.hypothesis_ordinals == [0, 1]
    assert response.hypotheses == [
        HypothesisDraft.model_validate(client.overflow_drafts[index])
        for index in [0, 1]
    ]
    assert response.reason_code == (
        "overselected_cardinality_suffix_quarantined_after_repair"
    )
    provenance = response.host_cardinality_selection
    assert provenance is not None
    assert provenance.selected_source_ordinals == [0, 1]
    assert provenance.cardinality_not_selected_source_ordinals == list(
        range(2, 65)
    )
    assert provenance.cardinality_not_selected_source_draft_digests == [
        provenance.source_draft_digests[index]
        for index in range(2, 65)
    ]
    assert "cardinality_not_selected" in (
        result.current_source.reduction.index.incomplete_reasons
    )
    assert validate_run_directory(output, repository_root=repository) == []

    replay_output = tmp_path / "terminal-overselection-cache-replay-run"
    replay_client = _CardinalitySelectionClient(
        overflow_count=65,
        selected_ordinals=[63, 64],
    )
    replayed = build_fixture(
        repository,
        replay_output,
        config=config_for_tests(semantic={"max_hypotheses_per_component": 2}),
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.reducer_calls == 0
    assert replayed.current_source is not None
    assert replayed.current_source.reduction == result.current_source.reduction
    assert validate_run_directory(
        replay_output,
        repository_root=repository,
    ) == []


def test_partition_effective_limit_selects_overflow_and_replays(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "partition-effective-limit-run"
    inference_cache = tmp_path / "partition-effective-limit-cache"
    client = _CardinalitySelectionClient(
        overflow_count=17,
        selected_ordinals=[0],
        overflow_mode="component_partition",
    )
    result = build_fixture(
        repository,
        output,
        config=config_for_tests(),
        client=client,
        inference_cache=inference_cache,
    )

    assert result.current_source is not None
    partition_requests = [
        request
        for request in client.initial_reducer_requests
        if request["mode"] == "component_partition"
    ]
    assert partition_requests
    assert all(
        request["hypothesis_budget"]["max_output_hypotheses"]
        == len(request["accounted_input_ids"])
        for request in partition_requests
    )
    partition_records = [
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
    ]
    assert len(partition_records) == len(partition_requests)
    assert all(
        record.terminal_status == "success"
        and [attempt.parsed for attempt in record.attempts] == [False, True]
        and record.structured_response is not None
        and record.structured_response["host_cardinality_selection"]
        is not None
        for record in partition_records
    )
    assert all(
        schema["properties"]["hypotheses"]["maxItems"] == 64
        for schema in client.initial_reducer_schemas
    )
    assert validate_run_directory(output, repository_root=repository) == []

    replay_client = _CardinalitySelectionClient(
        overflow_count=17,
        selected_ordinals=[1],
        overflow_mode="component_partition",
    )
    replayed = build_fixture(
        repository,
        tmp_path / "partition-effective-limit-replay-run",
        config=config_for_tests(),
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.reducer_calls == 0
    assert replayed.current_source is not None
    assert replayed.current_source.reduction == result.current_source.reduction


def test_invalid_selection_fails_closed_and_salvages_with_configured_small_cap(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _CardinalitySelectionClient(
        overflow_count=17,
        selected_ordinals=[0, 0],
    )
    result = build_fixture(
        repository,
        tmp_path / "invalid-cardinality-selection-run",
        config=config_for_tests(
            semantic={
                "max_hypotheses_per_component": 8,
                }
        ),
        client=client,
    )

    assert result.current_source is not None
    reduction = result.current_source.reduction
    assert reduction.index.status == "partial"
    assert "hypothesis_reducer_failure" in (
        reduction.index.incomplete_reasons
    )
    assert client.overflow_input_ids is not None
    # The fixture injects overflow into the first request that reaches the
    # client; concurrent arrival order need not match the saved record order.
    overflow_records = [
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    ]
    assert len(overflow_records) == 1
    partition_record = overflow_records[0]
    assert partition_record.terminal_status == "failure"
    assert len(partition_record.attempts) == 3
    assert all(
        not item.parsed for item in partition_record.attempts
    )
    assert len(client.selection_requests) == 2
    assert all(
        request["max_selected_hypotheses"]
        == min(8, len(client.overflow_input_ids))
        for request in client.selection_requests
    )
    assert not any(
        item.rule.startswith("Preserve exact overflow draft")
        for item in reduction.hypotheses
    )


def test_selection_repair_is_not_used_beyond_its_bounded_candidate_cap(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _CardinalitySelectionClient(
        overflow_count=MAX_REDUCE_SELECTION_CANDIDATES + 1,
        selected_ordinals=[0],
    )
    result = build_fixture(
        repository,
        tmp_path / "unbounded-cardinality-run",
        config=config_for_tests(),
        client=client,
    )

    assert result.current_source is not None
    assert client.selection_requests == []
    assert client.overflow_input_ids is not None
    overflow_records = [
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    ]
    assert len(overflow_records) == 1
    partition_record = overflow_records[0]
    assert partition_record.terminal_status == "failure"
    assert "hypothesis_reducer_failure" in (
        result.current_source.reduction.index.incomplete_reasons
    )


def test_unselected_invalid_overflow_draft_is_excluded_before_selection(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "invalid-unselected-overflow-run"
    client = _CardinalitySelectionClient(
        overflow_count=33,
        selected_ordinals=[0],
        invalid_population_ordinal=32,
    )
    result = build_fixture(
        repository,
        output,
        config=config_for_tests(
            semantic={
                "max_hypotheses_per_component": 32,
                }
        ),
        client=client,
    )

    assert result.current_source is not None
    assert len(client.selection_requests) == 1
    selection_request = client.selection_requests[0]
    assert selection_request["source_candidate_count"] == 33
    assert selection_request["candidate_count"] == 32
    assert selection_request["excluded_candidate_count"] == 1
    assert selection_request["excluded_candidates"] == [
        {
            "ordinal": 32,
            "draft_digest": selection_request["excluded_candidates"][0][
                "draft_digest"
            ],
            "reason_code": "population_spec_contract_invalid",
        }
    ]
    assert [item["ordinal"] for item in selection_request["candidates"]] == list(
        range(32)
    )
    partition_record = next(
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    )
    assert partition_record.terminal_status == "success"
    assert [item.parsed for item in partition_record.attempts] == [False, True]
    assert partition_record.structured_response is not None
    assert partition_record.structured_response["hypotheses"] == [
        client.overflow_drafts[0]
    ]
    provenance = partition_record.structured_response[
        "host_cardinality_selection"
    ]
    assert provenance["source_candidate_count"] == 33
    assert provenance["selected_source_ordinals"] == [0]
    exclusions = provenance["preflight_excluded_hypotheses"]
    assert len(exclusions) == 1
    assert exclusions[0]["hypothesis_ordinal"] == 32
    assert exclusions[0]["reason_code"] == "population_spec_contract_invalid"
    assert exclusions[0]["draft"] == client.overflow_drafts[32]
    assert exclusions[0]["draft_digest"] == provenance[
        "source_draft_digests"
    ][32]
    assert provenance["cardinality_not_selected_source_ordinals"] == list(
        range(1, 32)
    )
    partition_events = [
        item
        for item in result.current_source.reduction.events
        if item.inference_id == partition_record.request_id
    ]
    assert len(partition_events) == 33
    excluded_event = next(
        item for item in partition_events if item.ordinal == 32
    )
    assert excluded_event.normalization == "invalid"
    assert excluded_event.reason_codes[-1] == "cardinality_preflight_excluded"
    assert sum(
        item.normalization == "selection_omitted"
        for item in partition_events
    ) == 31
    assert validate_run_directory(output, repository_root=repository) == []


def test_cardinality_selection_keeps_wide_population_drafts_eligible(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _CardinalitySelectionClient(
        overflow_count=33,
        selected_ordinals=[0],
        structural_ordinal=0,
    )
    result = build_fixture(
        repository,
        tmp_path / "cardinality-cost-exclusion-run",
        config=config_for_tests(
            semantic={
                "max_hypotheses_per_component": 32,
                    "max_classified_population_members_per_hypothesis": 2,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    partition_record = next(
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    )
    assert partition_record.terminal_status == "success"
    assert partition_record.structured_response is not None
    assert partition_record.structured_response["host_cardinality_selection"] is not None
    partition_events = [
        item
        for item in result.current_source.reduction.events
        if item.inference_id == partition_record.request_id
    ]
    assert not any(
        item.normalization == "cost_omitted" for item in partition_events
    )
    assert sum(
        item.normalization == "selection_omitted"
        for item in partition_events
    ) == 32


@pytest.mark.parametrize(
    ("invalid_population_ordinal", "expected_invalid_events"),
    [(None, 0), (32, 1)],
)
def test_wide_overflow_candidates_still_use_cardinality_selection(
    tmp_path: Path,
    invalid_population_ordinal: int | None,
    expected_invalid_events: int,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / (
        "all-cost-overflow-excluded-run"
        if invalid_population_ordinal is None
        else "mixed-overflow-excluded-run"
    )
    client = _CardinalitySelectionClient(
        overflow_count=33,
        selected_ordinals=[0],
        invalid_population_ordinal=invalid_population_ordinal,
    )
    result = build_fixture(
        repository,
        output,
        config=config_for_tests(
            semantic={
                "max_hypotheses_per_component": 32,
                    "max_classified_population_members_per_hypothesis": 1,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    reduction = result.current_source.reduction
    partition_record = next(
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.mode == "component_partition"
        and tuple(item.stable_id for item in record.input_manifest)
        == client.overflow_input_ids
    )
    assert partition_record.terminal_status == "success"
    assert len(partition_record.attempts) == 2
    assert [item.parsed for item in partition_record.attempts] == [False, True]
    assert len(client.selection_requests) == 1
    assert partition_record.structured_response is not None
    response = ReductionResponse.model_validate(
        partition_record.structured_response
    )
    assert response.outcome == "hypotheses"
    provenance = response.host_cardinality_selection
    assert provenance is not None
    assert provenance.source_response_digest == (
        partition_record.attempts[0].response_digest
    )
    assert provenance.source_candidate_count == 33
    assert provenance.selected_source_ordinals == [0]
    assert len(provenance.preflight_excluded_hypotheses) == (
        expected_invalid_events
    )
    with pytest.raises(ValueError, match="does not exceed"):
        _validate_cardinality_selection_attempt_binding(
            response,
            partition_record,
            effective_limit=33,
        )
    partition_events = [
        item
        for item in reduction.events
        if item.inference_id == partition_record.request_id
    ]
    assert len(partition_events) == 33
    assert sum(
        item.normalization == "invalid" for item in partition_events
    ) == expected_invalid_events
    assert not any(
        item.normalization == "cost_omitted" for item in partition_events
    )
    assert sum(
        item.normalization == "selection_omitted"
        for item in partition_events
    ) == 32 - expected_invalid_events
    assert reduction.index.generated_candidate_events == len(reduction.events)
    assert reduction.index.cost_omitted_candidate_events == sum(
        item.normalization == "cost_omitted" for item in reduction.events
    )
    assert validate_run_directory(output, repository_root=repository) == []


def test_reducer_stops_after_one_exact_invalid_repair_repeat(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RepeatedInvalidReductionClient()
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"schema_repairs": 3})
        }
    )
    result = build_fixture(
        repository,
        tmp_path / "repeated-invalid-reducer-run",
        config=config,
        client=client,
    )

    reducer_records = [
        item
        for item in result.inference_records
        if item.stage == "current_hypothesis_reduce"
    ]
    assert reducer_records
    assert client.reducer_calls == 2 * len(reducer_records)
    assert all(item.terminal_status == "failure" for item in reducer_records)
    assert all(
        [attempt.schema_attempt for attempt in item.attempts] == [0, 1]
        for item in reducer_records
    )
    assert all(
        "repeated_invalid_response" in item.incomplete_reasons
        for item in reducer_records
    )
    first_attempt = reducer_records[0].attempts[0]
    intervening_attempt = first_attempt.model_copy(
        update={"response_digest": "b" * 64}
    )
    assert not _is_repeated_invalid_response(
        [first_attempt, intervening_attempt],
        response_digest=first_attempt.response_digest or "",
        validation_errors=first_attempt.validation_errors,
        repair=True,
        schema_attempt=2,
        enabled=True,
        early_stop_available=True,
    )

    exemplar = reducer_records[0].model_dump(mode="json")

    def redigest(value: dict[str, object]) -> dict[str, object]:
        for attempt in value["attempts"]:
            attempt["attempt_digest"] = payload_digest(
                {
                    key: item
                    for key, item in attempt.items()
                    if key != "attempt_digest"
                }
            )
        value["record_digest"] = payload_digest(
            {
                key: item
                for key, item in value.items()
                if key != "record_digest"
            }
        )
        return value

    deleted = json.loads(json.dumps(exemplar))
    deleted["incomplete_reasons"].remove("repeated_invalid_response")
    with pytest.raises(ValueError, match="repeated-invalid marker"):
        InferenceRecord.model_validate(redigest(deleted))

    wrong_digest = json.loads(json.dumps(exemplar))
    wrong_digest["attempts"][-1]["response_digest"] = "f" * 64
    with pytest.raises(ValueError, match="repeated-invalid marker"):
        InferenceRecord.model_validate(redigest(wrong_digest))

    wrong_errors = json.loads(json.dumps(exemplar))
    wrong_errors["attempts"][-1]["validation_errors"].append(
        "different canonical error"
    )
    with pytest.raises(ValueError, match="repeated-invalid marker"):
        InferenceRecord.model_validate(redigest(wrong_errors))

    nonrepair = json.loads(json.dumps(exemplar))
    nonrepair["attempts"][-1]["repair"] = False
    with pytest.raises(ValueError):
        InferenceRecord.model_validate(redigest(nonrepair))

    wrong_stage = json.loads(json.dumps(exemplar))
    wrong_stage["stage"] = "current_semantic_map"
    wrong_stage["input_manifest"] = []
    wrong_stage["input_manifest_digest"] = payload_digest({"inputs": []})
    wrong_stage["request_id"] = request_identity(
        stage="current_semantic_map",
        protocol_version=wrong_stage["protocol_version"],
        mode=wrong_stage["mode"],
        inputs=[],
        request_ordinal=wrong_stage["request_ordinal"],
    )
    with pytest.raises(ValueError, match="repeated-invalid marker"):
        InferenceRecord.model_validate(redigest(wrong_stage))


def test_exact_repeat_on_last_allowed_repair_is_normal_terminal_failure(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RepeatedInvalidReductionClient()
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"schema_repairs": 1})
        }
    )
    result = build_fixture(
        repository,
        tmp_path / "last-repair-repeat-run",
        config=config,
        client=client,
    )

    reducer_records = [
        item
        for item in result.inference_records
        if item.stage == "current_hypothesis_reduce"
    ]
    assert reducer_records
    assert all(len(item.attempts) == 2 for item in reducer_records)
    assert all(
        "repeated_invalid_response" not in item.incomplete_reasons
        for item in reducer_records
    )


def test_terminal_fallback_error_change_does_not_claim_exact_repeat(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RepeatedInvalidReductionClient(content="{}")
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"schema_repairs": 3})
        }
    )
    result = build_fixture(
        repository,
        tmp_path / "changed-terminal-error-repeat-run",
        config=config,
        client=client,
    )

    reducer_records = [
        item
        for item in result.inference_records
        if item.stage == "current_hypothesis_reduce"
    ]
    assert reducer_records
    assert all(len(item.attempts) == 4 for item in reducer_records)
    assert all(
        "repeated_invalid_response" not in item.incomplete_reasons
        for item in reducer_records
    )


def test_reducer_applies_only_audited_equivalent_population_repairs() -> None:
    normalized = _normalize_response(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    "superpopulation_spec": {
                        "op": "filter",
                        "source": {"op": "universe", "universe": "entity"},
                        "predicates": [
                            {
                                "field": "file_path",
                                "operator": "in",
                                "value": ["src/pkg/mod.py", "src/pkg/mod.py"],
                            },
                            {
                                "field": "path_under",
                                "operator": "eq",
                                "value": "src/pkg/**",
                            },
                        ],
                    },
                    "structural_predicate": None,
                }
            ],
            "reason_code": "candidate",
        },
        mode="component_partition",
        input_ids=["HYP-a"],
    )
    hypothesis = normalized["hypotheses"][0]
    predicates = hypothesis["superpopulation_spec"]["predicates"]
    assert predicates == [
        {"field": "path", "operator": "eq", "value": "src/pkg/mod.py"},
        {"field": "path", "operator": "path_under", "value": "src/pkg"},
    ]
    assert {
        item["repair_code"] for item in normalized["host_population_repairs"]
    } == {
        "field_alias_file_path_to_path",
        "field_operator_path_under_normalized",
        "literal_list_deduplicated",
        "path_under_suffix_removed",
        "singleton_in_to_eq",
    }


def test_reducer_does_not_silently_repair_semantic_population_errors() -> None:
    normalized = _normalize_response(
        {
            "outcome": "hypotheses",
            "hypotheses": [
                {
                    "superpopulation_spec": {
                        "op": "filter",
                        "source": {"op": "universe", "universe": "file"},
                        "predicates": [
                            {"field": "path", "operator": "has", "value": ".py"},
                            {
                                "field": "path",
                                "operator": "in",
                                "value": ["**/*.py"],
                            },
                        ],
                    },
                    "structural_predicate": None,
                }
            ],
            "reason_code": "candidate",
        },
        mode="component_partition",
        input_ids=["HYP-a"],
    )
    predicates = normalized["hypotheses"][0]["superpopulation_spec"][
        "predicates"
    ]
    assert predicates[0]["operator"] == "has"
    assert predicates[1] == {
        "field": "path",
        "operator": "in",
        "value": ["**/*.py"],
    }
    assert normalized["host_population_repairs"] == []


def test_reducer_canonicalizes_only_complete_unique_accounting_metadata() -> None:
    settings = config_for_tests().semantic
    reordered = ReductionResponse(
        mode="component_partition",
        accounted_input_ids=["HYP-b", "HYP-a"],
        outcome="no_hypothesis",
        hypotheses=[],
        reason_code="no_repository_hypothesis",
    )
    normalized = _validate_response(
        reordered,
        mode="component_partition",
        input_ids=["HYP-a", "HYP-b"],
        settings=settings,
    )
    assert normalized.accounted_input_ids == ["HYP-a", "HYP-b"]

    missing = reordered.model_copy(
        update={"accounted_input_ids": ["HYP-a"]}
    )
    with pytest.raises(ValueError, match="coverage differs"):
        _validate_response(
            missing,
            mode="component_partition",
            input_ids=["HYP-a", "HYP-b"],
            settings=settings,
        )

    duplicate = reordered.model_copy(
        update={"accounted_input_ids": ["HYP-a", "HYP-a"]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_response(
            duplicate,
            mode="component_partition",
            input_ids=["HYP-a", "HYP-b"],
            settings=settings,
        )


def test_raw_malformed_population_is_quarantined_without_losing_valid_sibling(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _AlwaysRawMalformedPopulationClient(
        category="raw-population-contract-regression",
        rule="The valid sibling survives malformed population output.",
    )
    config = config_for_tests()
    config = config.model_copy(
        update={"llm": config.llm.model_copy(update={"schema_repairs": 2})}
    )
    result = build_fixture(
        repository,
        tmp_path / "raw-quarantine-run",
        config=config,
        client=client,
    )
    assert client.partition_calls == 3
    assert result.current_source is not None
    assert client.target_input_ids is not None
    target_results = [
        item
        for item in result.current_source.reduction.call_results
        if item.mode == "component_partition"
        and tuple(item.input_ids) == client.target_input_ids
    ]
    assert len(target_results) == 1
    partition_result = target_results[0]
    assert len(partition_result.hypothesis_digests) == 1
    assert len(partition_result.quarantined_hypotheses) == 1
    partition_record = next(
        item
        for item in result.inference_records
        if item.request_id == partition_result.inference_id
    )
    assert [item.schema_attempt for item in partition_record.attempts] == [0, 1, 2]
    assert [item.parsed for item in partition_record.attempts] == [False, False, True]
    quarantined = partition_result.quarantined_hypotheses[0]
    assert quarantined.hypothesis_ordinal == 1
    assert quarantined.reason_code == "population_spec_contract_invalid"
    assert (
        quarantined.draft["superpopulation_spec"]["predicates"][0]["operator"]
        == "totally_unknown_operator"
    )
    assert result.current_source.reduction.index.unique_candidate_roots == 1


def test_partition_semantic_population_over_inspection_cap_is_sampled(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RecordingReductionClient(
        category="semantic-selectivity-regression",
        rule="A broad semantic population must be narrowed before admission.",
    )
    config = config_for_tests(
        semantic={"max_classified_population_members_per_hypothesis": 1}
    )
    config = config.model_copy(
        update={"llm": config.llm.model_copy(update={"schema_repairs": 2})}
    )
    result = build_fixture(
        repository,
        tmp_path / "semantic-selectivity-quarantine-run",
        config=config,
        client=client,
    )

    assert result.current_source is not None
    partition_results = [
        item
        for item in result.current_source.reduction.call_results
        if item.mode == "component_partition"
    ]
    quarantined = [
        item
        for result_item in partition_results
        for item in result_item.quarantined_hypotheses
    ]
    assert not quarantined
    assert result.current_source.reduction.hypotheses
    assert not any(
        item.normalization == "cost_omitted"
        for item in result.current_source.reduction.events
    )
    population = next(
        item
        for item in result.current_source.populations
        if item.route == "current_semantic_population"
    )
    assert population.sampling_plan.mode == "deterministic_stratified_sample"
    assert population.sampling_plan.provider_inspected_count == 1
    assert population.sampling_plan.provider_uninspected_count > 0
    partition_requests = [
        item
        for item in client.reduction_requests
        if item["mode"] == "component_partition"
    ]
    assert partition_requests
    planning = partition_requests[0]["hypothesis_budget"][
        "semantic_population_planning"
    ]
    assert planning == {
        "soft_target_in_scope_members_per_hypothesis": 15,
        "wide_population_action": (
            "retain_supported_hypothesis_for_downstream_"
            "deterministic_bounded_inspection"
        ),
        "structural_predicates_exempt": True,
        "counting_basis": (
            "provider-eligible known PopulationSpec matches whose path is "
            "inside the hypothesis applies_to scope"
        ),
    }


def test_structural_population_is_exempt_from_semantic_member_admission_cap(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "structural-selectivity-exemption-run",
        config=config_for_tests(
            semantic={
                "max_classified_population_members_per_hypothesis": 1,
                "max_total_classified_population_members": 1,
            }
        ),
        client=_StructuralPartitionPopulationClient(
            category="structural-selectivity-regression",
            rule="The deterministic structural population remains admissible.",
        ),
    )

    assert result.current_source is not None
    assert result.current_source.reduction.hypotheses
    assert all(
        item.predicate_mode == "structural"
        for item in result.current_source.reduction.hypotheses
    )
    assert all(
        item.sampling_plan.mode == "census"
        and item.sampling_plan.provider_eligible_count == 0
        for item in result.current_source.populations
    )


def test_run_wide_semantic_inspection_cap_is_audited_after_reduction(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "semantic-total-admission-run",
        config=config_for_tests(
            semantic={
                "max_classified_population_members_per_hypothesis": 1,
                "max_total_classified_population_members": 1,
            }
        ),
        client=ParameterizedSemanticClient(
            category="semantic-total-admission-regression",
            rule="Total admission is deterministic and explicitly audited.",
        ),
    )

    assert result.current_source is not None
    assert result.current_source.reduction.hypotheses
    assert not any(
        item.normalization == "cost_omitted"
        for item in result.current_source.reduction.events
    )
    schedule = result.current_source.index.population_schedule
    assert sum(item.classified_member_limit for item in schedule) <= 1
    assert sum(item.provider_unallocated_count for item in schedule) > 0


def test_unique_evidence_range_witness_alias_is_repaired_and_audited(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "witness-alias-run",
        client=_EvidenceRangeWitnessAliasClient(
            category="witness-alias-regression",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    targets = [
        item for item in result.current_source.semantic_map.observations
        if item.summary == _EvidenceRangeWitnessAliasClient.target_summary
    ]
    assert len(targets) == 1
    evidence_ref = targets[0].evidence_refs[0]
    expected_owners = {
        item.observation_id
        for item in result.current_source.semantic_map.observations
        if evidence_ref in item.evidence_refs
    }
    partition_event = next(
        item
        for item in result.current_source.reduction.events
        if item.mode == "component_partition"
        and "unique_witness_evidence_alias_repaired" in item.reason_codes
    )
    hypothesis = next(
        item
        for item in result.current_source.reduction.hypotheses
        if item.candidate_root_id == partition_event.winning_root_id
    )
    assert expected_owners == {targets[0].observation_id}
    assert len(hypothesis.witness_ref_repairs) == 1
    assert {
        (item.source_evidence_ref, item.target_observation_ref)
        for item in hypothesis.witness_ref_repairs
    } == {(evidence_ref, targets[0].observation_id)}
    repair = hypothesis.witness_ref_repairs[0]
    assert repair.source_evidence_ref.startswith("SEMRANGE-")
    assert repair.target_observation_ref in hypothesis.witness_refs
    assert "unique_witness_evidence_alias_repaired" in (
        partition_event.reason_codes
    )


def test_shared_evidence_range_witness_alias_expands_all_owners_and_is_audited(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "shared-witness-alias-run",
        client=_SharedEvidenceRangeWitnessAliasClient(
            category="shared-witness-alias-regression",
            rule="Contributors preserve both interpretations of a shared range.",
        ),
    )
    assert result.current_source is not None
    targets = [
        item for item in result.current_source.semantic_map.observations
        if item.summary == _EvidenceRangeWitnessAliasClient.target_summary
    ]
    assert len(targets) == 1
    evidence_ref = targets[0].evidence_refs[0]
    expected_owners = {
        item.observation_id
        for item in result.current_source.semantic_map.observations
        if evidence_ref in item.evidence_refs
    }
    partition_event = next(
        item
        for item in result.current_source.reduction.events
        if item.mode == "component_partition"
        and "shared_witness_evidence_alias_expanded" in item.reason_codes
    )
    hypothesis = next(
        item
        for item in result.current_source.reduction.hypotheses
        if item.candidate_root_id == partition_event.winning_root_id
    )
    companions = [
        item for item in result.current_source.semantic_map.observations
        if item.summary == _EvidenceRangeWitnessAliasClient.companion_summary
    ]
    assert len(companions) == 1
    assert companions[0].evidence_refs == targets[0].evidence_refs
    assert expected_owners == {
        targets[0].observation_id, companions[0].observation_id
    }
    assert len(expected_owners) == 2
    repairs = hypothesis.witness_ref_repairs
    assert len(repairs) == 2
    assert {
        (item.source_evidence_ref, item.target_observation_ref)
        for item in repairs
    } == {(evidence_ref, owner) for owner in expected_owners}
    assert {
        item.repair_code for item in repairs
    } == {"shared_evidence_range_to_observation"}
    assert {
        item.target_observation_ref for item in repairs
    } <= set(hypothesis.witness_refs)
    assert "shared_witness_evidence_alias_expanded" in (
        partition_event.reason_codes
    )


def test_quarantined_invalid_population_is_retained_in_lineage_and_export(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "quarantined-run"
    inference_cache = tmp_path / "inference-cache"
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"schema_repairs": 2})
        }
    )
    client = _AlwaysInvalidPartitionPopulationClient(
        category="invalid-population-regression",
        rule="The invalid population remains auditable after repair exhaustion.",
    )
    result = build_fixture(
        repository,
        output,
        config=config,
        client=client,
        inference_cache=inference_cache,
    )
    assert client.partition_calls == 3
    assert result.current_source is not None
    reduction = result.current_source.reduction
    partition_result = next(
        item
        for item in reduction.call_results
        if tuple(item.input_ids) == client.target_input_ids
    )
    partition_record = next(
        item
        for item in result.inference_records
        if item.request_id == partition_result.inference_id
    )
    assert [item.schema_attempt for item in partition_record.attempts] == [
        0,
        1,
        2,
    ]
    assert [item.parsed for item in partition_record.attempts] == [
        False,
        False,
        True,
    ]
    assert len(partition_result.hypothesis_digests) == 1
    assert len(partition_result.quarantined_hypotheses) == 1
    quarantined = partition_result.quarantined_hypotheses[0]
    assert quarantined.hypothesis_ordinal == 1
    assert quarantined.reason_code == "population_spec_contract_invalid"
    assert quarantined.draft_digest
    assert quarantined.draft["superpopulation_spec"]["op"] == "filter"
    assert [
        item.repair_code for item in quarantined.host_population_repairs
    ] == ["field_alias_file_path_to_path"]
    event = next(
        item
        for item in reduction.events
        if item.inference_id == partition_result.inference_id
        and item.draft_digest == quarantined.draft_digest
    )
    assert event.normalization == "invalid"
    assert event.ordinal == quarantined.hypothesis_ordinal
    assert event.reason_codes == [
        "population_spec_contract_invalid",
        "population_repair_exhausted",
    ]
    assert reduction.index.generated_candidate_events == len(reduction.events)
    assert reduction.index.invalid_candidate_events == sum(
        item.normalization == "invalid" for item in reduction.events
    )

    exported = (
        output
        / "current-source"
        / "hypotheses"
        / "reduction-results"
        / f"{partition_result.result_id}.json"
    )
    parsed = ReductionCallResult.model_validate_json(
        exported.read_text(encoding="utf-8")
    )
    assert parsed == partition_result
    assert ReductionCallResult.model_validate(
        parsed.model_dump(mode="json")
    ) == parsed
    replay_client = _AlwaysInvalidPartitionPopulationClient(
        category="invalid-population-regression",
        rule="The invalid population remains auditable after repair exhaustion.",
    )
    replayed = build_fixture(
        repository,
        tmp_path / "quarantined-replay-run",
        config=config,
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.partition_calls == 0
    assert replayed.current_source is not None
    replayed_partition_result = next(
        item
        for item in replayed.current_source.reduction.call_results
        if item.result_id == partition_result.result_id
    )
    assert replayed_partition_result == partition_result
    assert replayed.current_source.reduction.events == reduction.events
    assert REDUCE_RESPONSE_SCHEMA_VERSION == (
        "current-hypothesis-reduce-response-1.10"
    )
    assert ReductionResponse.model_json_schema()["properties"]["hypotheses"][
        "maxItems"
    ] == 64


def test_population_context_has_deterministic_byte_cap_and_coverage(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RecordingReductionClient(
        category="bounded-context",
        rule="Population planning context remains bounded.",
    )
    result = build_fixture(repository, tmp_path / "bounded-run", client=client)
    assert result.current_source is not None
    assert result.observation is not None
    assert result.semantic_units is not None
    partition_request = next(
        request
        for request in client.reduction_requests
        if request["mode"] == "component_partition"
    )
    inputs = tuple(
        _ReduceInput(
            stable_id=stable_id,
            digest="0" * 64,
            payload=payload,
            ref_kind="semantic_observation",
        )
        for stable_id, payload in zip(
            partition_request["accounted_input_ids"],
            partition_request["inputs"],
        )
    )
    byte_limit = 1_024
    first = _population_context(
        mode="component_partition",
        inputs=inputs,
        observations=result.observation,
        semantic_map=result.current_source.semantic_map,
        semantic_units=result.semantic_units,
        byte_limit=byte_limit,
    )
    second = _population_context(
        mode="component_partition",
        inputs=inputs,
        observations=result.observation,
        semantic_map=result.current_source.semantic_map,
        semantic_units=result.semantic_units,
        byte_limit=byte_limit,
    )
    assert first == second
    assert first["coverage"]["status"] == "complete"
    assert first["coverage"]["context_bytes"] > byte_limit
    _validate_population_context_references(
        first, inputs=inputs, snapshot_digest=result.current_source.index.snapshot_digest,
    )

    observation = result.current_source.semantic_map.observations[0]
    atomic_input = _ReduceInput(
        stable_id=observation.observation_id,
        digest=observation.artifact_digest,
        payload=observation.model_dump(mode="json"),
        ref_kind="semantic_observation",
    )
    witness_context = _population_context(
        mode="component_partition",
        inputs=(atomic_input,),
        observations=result.observation,
        semantic_map=result.current_source.semantic_map,
        semantic_units=result.semantic_units,
        byte_limit=200_000,
    )
    assert witness_context["detail_level"] == "witness_grounding"
    assert witness_context["coverage"]["status"] == "complete"
    assert "observed_value_catalog" not in witness_context
    assert [
        item["witness_id"] for item in witness_context["witness_bindings"]
    ] == [observation.observation_id]
    binding = witness_context["witness_bindings"][0]
    fact_by_id = {
        item["observation_id"]: item
        for item in witness_context["query_facts"]
    }
    assert set(binding["query_fact_refs"]) <= set(fact_by_id)
    if binding["entity_id"] is not None:
        assert fact_by_id[binding["entity_id"]]["values"]["kind"] == (
            binding["unit_kind"]
        )
    _validate_population_context_references(
        witness_context,
        inputs=(atomic_input,),
        snapshot_digest=result.current_source.index.snapshot_digest,
    )
    repair = json.loads(
        _repair_prompt(
            json.dumps(
                {
                    "stage": "current_hypothesis_reduce",
                    "mode": "component_partition",
                    "accounted_input_ids": [atomic_input.stable_id],
                    "population_context": witness_context,
                }
            ),
            '{"invalid":true}',
            [
                "witness "
                f"{observation.observation_id} is outside the superpopulation"
            ],
        )
    )
    targeted = repair["targeted_witness_context"]["bindings"]
    assert targeted == [
        {
            "witness_id": observation.observation_id,
            "path": binding["path"],
            "paths": None,
            "candidate_query_fact_refs": binding["query_fact_refs"],
            "candidate_query_fact_ref_count": len(binding["query_fact_refs"]),
        }
    ]
    missing_binding = json.loads(json.dumps(witness_context))
    missing_binding["witness_bindings"] = []
    with pytest.raises(ValueError, match="does not bind every input witness"):
        _validate_population_context_references(
            missing_binding,
            inputs=(atomic_input,),
            snapshot_digest=result.current_source.index.snapshot_digest,
        )


def test_extension_evidence_and_fake_output_add_unseen_category_only(
    tmp_path: Path,
) -> None:
    base_repository = copied_current_source_fixture(tmp_path / "base")
    (base_repository / "src/extensions/extra_pattern.py").unlink()
    base = build_fixture(
        base_repository,
        tmp_path / "base-run",
        client=ParameterizedSemanticClient(
            category="fixture-base-pattern",
            rule="Contributors preserve the base fixture pattern.",
        ),
    )

    extended_repository = copied_current_source_fixture(tmp_path / "extended")
    extended = build_fixture(
        extended_repository,
        tmp_path / "extended-run",
        client=ParameterizedSemanticClient(
            category="fourth-unseen-timestamp-pattern",
            rule="Contributors use the repository timestamp serializer.",
        ),
    )
    assert base.current_source is not None
    assert extended.current_source is not None
    assert {item.category for item in base.current_source.hypotheses} == {
        "fixture-base-pattern"
    }
    assert {
        item.category for item in extended.current_source.hypotheses
    } == {"fourth-unseen-timestamp-pattern"}
    assert any(
        "src/extensions/extra_pattern.py" == item.path
        for item in extended.observation.shards  # type: ignore[union-attr]
    )
    assert all(
        item.epistemic_status != "accepted"
        for item in extended.current_source.review_items
    )


def test_current_source_independent_calls_use_bounded_concurrency(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests(
            semantic={
                "max_map_packets_per_request": 1,
                "max_reduce_items_per_request": 2,
                "max_classification_members_per_request": 4,
            "max_falsification_items_per_request": 1,
        }
    )
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"max_concurrency": 4})
        }
    )
    client = DelayedMockClient(delay_seconds=0.01)
    result = build_fixture(
        repository,
        tmp_path / "parallel-run",
        config=config,
        client=client,
    )
    second_client = DelayedMockClient(delay_seconds=0.01)
    second = build_fixture(
        repository,
        tmp_path / "parallel-run-second",
        config=config,
        client=second_client,
    )

    assert result.current_source is not None
    assert result.current_source == second.current_source
    assert result.traceability == second.traceability
    assert result.inference_records == second.inference_records
    assert 2 <= result.build_report["llm"]["observed_max_concurrency"] <= 4
    records_by_stage: dict[str, int] = {}
    for record in result.inference_records:
        records_by_stage[record.stage] = records_by_stage.get(record.stage, 0) + 1
    for stage in (
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
    ):
        assert records_by_stage[stage] > 1
        assert 2 <= client.peak_in_flight_by_stage[stage] <= 4
        assert 2 <= second_client.peak_in_flight_by_stage[stage] <= 4


def test_partition_packing_uses_complete_schema_attached_utf8_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full required facts above 64 KiB survive; two large inputs split stably."""
    from repo_constraint_memory_v3.inference import InferenceBudget, attach_response_json_schema
    from repo_constraint_memory_v3.semantic import reduce as reducer

    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "packing-substrate")
    assert result.current_source and result.observation and result.semantic_units
    semantic_map = result.current_source.semantic_map
    unit_by_id = {item.unit_id: item for item in result.semantic_units.units}
    by_file = {}
    for observation in semantic_map.observations:
        by_file.setdefault(unit_by_id[observation.primary_unit_id].file_id, observation)
    selected = sorted(by_file.values(), key=lambda item: item.observation_id)[:2]
    assert len(selected) == 2
    inputs = tuple(_ReduceInput(
        stable_id=item.observation_id, digest=item.artifact_digest,
        payload=item.model_dump(mode="json"), ref_kind="semantic_observation",
    ) for item in selected)
    original_facts = reducer.population_query_facts

    def large_facts(*args, **kwargs):
        payload = original_facts(*args, **kwargs)
        for fact in payload["facts"]:
            if fact["universe"] == "file":
                fact["values"]["module_name"] = "界" * 35_000
        return payload

    monkeypatch.setattr(reducer, "population_query_facts", large_facts)
    settings = config_for_tests().semantic.model_copy(update={"max_reduce_request_bytes": 200_000})
    common = dict(
        mode="component_partition", component_key="packing-fixture",
        snapshot_digest=semantic_map.index.snapshot_digest, settings=settings,
        coverage={}, observations=result.observation, semantic_map=semantic_map,
        semantic_units=result.semantic_units,
    )
    prepare = lambda selected: reducer._prepare_reduce_request(inputs=selected, **common)
    partitions = reducer._partition(inputs, settings, prepare_request=prepare)
    assert [part for part, _ in partitions] == [(inputs[0],), (inputs[1],)]
    assert partitions == reducer._partition(inputs, settings, prepare_request=prepare)
    assert prepare(inputs).request_bytes > settings.max_reduce_request_bytes

    class NoHypothesisClient:
        requests: list[str] = []

        def complete(self, system_prompt, user_prompt):
            self.requests.append(user_prompt)
            assert len(user_prompt.encode("utf-8")) <= 200_000
            return ProviderResponse(content=json.dumps({
                "outcome": "no_hypothesis", "hypotheses": [], "reason_code": "no_rule",
            }), model="packing-mock", input_tokens=1, output_tokens=1, latency_ms=0)

    client = NoHypothesisClient()
    llm_settings = config_for_tests().llm
    budget = InferenceBudget(llm_settings)
    budget.plan_stage("current_hypothesis_reduce", len(partitions))
    for ordinal, (part, prepared) in enumerate(partitions):
        context = json.loads(prepared.request)["population_context"]
        assert context["coverage"]["status"] == "complete"
        assert context["coverage"]["context_bytes"] > 65_536
        assert not prepared.context_invalid
        assert prepared.request_bytes == len(attach_response_json_schema(
            prepared.request, ProviderReductionResponse,
        ).encode("utf-8"))
        response, record, _ = reducer._call(
            inputs=part, ordinal=ordinal, llm_settings=llm_settings,
            client=client, budget=budget, protocol_manifest_digest="a" * 64,
            prepared_request=prepared, **common,
        )
        assert response is not None and record.terminal_status == "success"
        assert client.requests[-1] == attach_response_json_schema(prepared.request, ProviderReductionResponse)

    # A truly non-fitting singleton is accounted without provider execution.
    small_settings = settings.model_copy(update={"max_reduce_request_bytes": 80_000})
    small_common = {**common, "settings": small_settings}
    oversized = reducer._prepare_reduce_request(inputs=inputs[:1], **small_common)
    assert oversized.request_bytes > small_settings.max_reduce_request_bytes
    response, record, call_result = reducer._call(
        inputs=inputs[:1], ordinal=3, llm_settings=llm_settings,
        client=client, budget=budget, protocol_manifest_digest="a" * 64,
        prepared_request=oversized, **small_common,
    )
    assert response is None and not record.attempts
    assert "max_reduce_request_bytes" in call_result.reason_codes
    assert len(client.requests) == 2


def test_population_context_rejects_missing_duplicate_dangling_and_wrong_snapshot(
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    from repo_constraint_memory_v3.semantic import reduce as reducer

    result = build_fixture(copied_current_source_fixture(tmp_path), tmp_path / "context-bindings")
    assert result.current_source and result.observation and result.semantic_units
    observation = result.current_source.semantic_map.observations[0]
    inputs = (_ReduceInput(stable_id=observation.observation_id,
        digest=observation.artifact_digest, payload=observation.model_dump(mode="json"),
        ref_kind="semantic_observation"),)
    snapshot = result.current_source.index.snapshot_digest
    context = reducer._population_context(mode="component_partition", inputs=inputs,
        observations=result.observation, semantic_map=result.current_source.semantic_map,
        semantic_units=result.semantic_units, byte_limit=200_000)
    broken = []
    missing = deepcopy(context); missing["witness_bindings"] = []; broken.append(missing)
    duplicate = deepcopy(context); duplicate["query_facts"].append(duplicate["query_facts"][0]); broken.append(duplicate)
    dangling = deepcopy(context); dangling["witness_bindings"][0]["query_fact_refs"].append("MISSING"); broken.append(dangling)
    wrong = deepcopy(context); wrong["snapshot_digest"] = "sha256:" + "f" * 64; broken.append(wrong)
    for payload in broken:
        with pytest.raises(ValueError):
            reducer._validate_population_context_references(payload, inputs=inputs, snapshot_digest=snapshot)


@pytest.mark.parametrize("failure_kind", ["full_repair", "local_repair_fallback", "selection"])
def test_every_reducer_repair_respects_the_final_total_request_cap(
    tmp_path: Path, failure_kind: str,
) -> None:
    from copy import deepcopy
    from repo_constraint_memory_v3.inference import InferenceBudget
    from repo_constraint_memory_v3.semantic import reduce as reducer

    result = build_fixture(copied_current_source_fixture(tmp_path), tmp_path / "repair-cap-substrate")
    assert result.current_source and result.observation and result.semantic_units
    semantic_map = result.current_source.semantic_map
    inputs = tuple(_ReduceInput(stable_id=item.observation_id,
        digest=item.artifact_digest, payload=item.model_dump(mode="json"),
        ref_kind="semantic_observation") for item in semantic_map.observations[:2])
    assert len(inputs) == 2
    settings = config_for_tests().semantic
    common = dict(mode="component_partition", component_key="repair-cap-fixture",
        snapshot_digest=semantic_map.index.snapshot_digest, settings=settings,
        coverage={}, observations=result.observation, semantic_map=semantic_map,
        semantic_units=result.semantic_units)
    initial = reducer._prepare_reduce_request(inputs=inputs, **common)
    settings = settings.model_copy(update={"max_reduce_request_bytes": initial.request_bytes + 32})
    common["settings"] = settings
    prepared = reducer._prepare_reduce_request(inputs=inputs, **common)
    assert prepared.request_bytes <= settings.max_reduce_request_bytes
    draft = _mock_valid_reducer_draft()
    if failure_kind == "full_repair":
        content = '{"hypotheses":'
    elif failure_kind == "local_repair_fallback":
        bad = deepcopy(draft)
        bad.pop("superpopulation_spec")
        content = json.dumps({"outcome": "hypotheses", "hypotheses": [draft, bad], "reason_code": "bad_leaf"})
    else:
        content = json.dumps({"outcome": "hypotheses", "hypotheses": [
            {**deepcopy(draft), "rule": f"Preserve selection candidate {ordinal}."}
            for ordinal in range(65)
        ], "reason_code": "overflow"})

    class OnceClient:
        calls = 0

        def complete(self, system_prompt, user_prompt):
            self.calls += 1
            assert self.calls == 1, "over-limit repair must be provider-free"
            assert len(user_prompt.encode()) <= settings.max_reduce_request_bytes
            return ProviderResponse(content=content, model="repair-cap-mock",
                input_tokens=1, output_tokens=1, latency_ms=0)

    client = OnceClient()
    llm_settings = config_for_tests().llm
    response, record, call_result = reducer._call(inputs=inputs, ordinal=0,
        llm_settings=llm_settings, client=client, budget=InferenceBudget(llm_settings),
        protocol_manifest_digest="b" * 64, prepared_request=prepared, **common)
    assert response is None and client.calls == len(record.attempts) == 1
    assert "max_current_hypothesis_reduce_request_bytes" in call_result.reason_codes


def test_missing_matched_query_fact_cannot_be_filtered_into_complete_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3.inference import InferenceBudget
    from repo_constraint_memory_v3.semantic import reduce as reducer

    result = build_fixture(copied_current_source_fixture(tmp_path), tmp_path / "missing-fact-substrate")
    assert result.current_source and result.observation and result.semantic_units
    semantic_map = result.current_source.semantic_map
    observation = semantic_map.observations[0]
    inputs = (_ReduceInput(stable_id=observation.observation_id,
        digest=observation.artifact_digest, payload=observation.model_dump(mode="json"),
        ref_kind="semantic_observation"),)
    real_facts = reducer.population_query_facts

    def missing_fact(*args, **kwargs):
        value = real_facts(*args, **kwargs)
        removed = value["facts"].pop(0)
        value["matched_fact_count"] -= 1
        value["serialized_fact_count"] -= 1
        value["unknown_refs"].append(removed["observation_id"])
        return value

    monkeypatch.setattr(reducer, "population_query_facts", missing_fact)
    common = dict(mode="component_partition", component_key="missing-fact-fixture",
        snapshot_digest=semantic_map.index.snapshot_digest, settings=config_for_tests().semantic,
        coverage={}, observations=result.observation, semantic_map=semantic_map,
        semantic_units=result.semantic_units)
    prepared = reducer._prepare_reduce_request(inputs=inputs, **common)
    assert prepared.context_invalid
    context = json.loads(prepared.request)["population_context"]
    assert context["coverage"]["status"] == "partial"
    assert context["coverage"]["omitted_query_fact_count"] == 0
    assert context["coverage"]["unknown_query_fact_ref_count"] == 1

    class NoCalls:
        def complete(self, *args):
            pytest.fail("missing required query fact reached provider")

    llm_settings = config_for_tests().llm
    response, record, call_result = reducer._call(inputs=inputs, ordinal=0,
        llm_settings=llm_settings, client=NoCalls(), budget=InferenceBudget(llm_settings),
        protocol_manifest_digest="c" * 64, prepared_request=prepared, **common)
    assert response is None and not record.attempts
    assert "population_context_reference_incomplete" in call_result.reason_codes


def test_final_admission_does_not_semantically_merge_distinct_partition_rules(
    tmp_path: Path,
) -> None:
    rules = {"Preserve local fixture behavior.", "Maintain local fixture behavior."}

    class NearDuplicateClient(ParameterizedSemanticClient):
        ordinal = 0

        def complete(self, system_prompt, user_prompt):
            response = super().complete(system_prompt, user_prompt)
            request = json.loads(user_prompt)
            if request.get("stage") != "current_hypothesis_reduce":
                return response
            assert request["mode"] == "component_partition"
            value = json.loads(response.content)
            for draft in value["hypotheses"]:
                draft["rule"] = sorted(rules)[self.ordinal % 2]
                self.ordinal += 1
            return response.model_copy(update={"content": json.dumps(value)})

    result = build_fixture(copied_current_source_fixture(tmp_path), tmp_path / "near-duplicates",
        config=config_for_tests(semantic={"max_reduce_items_per_request": 1}),
        client=NearDuplicateClient(category="open-category", rule="Temporary fixture rule."))
    assert result.current_source
    reduction = result.current_source.reduction
    assert {item.rule for item in reduction.hypotheses} == rules
    assert len(reduction.hypotheses) == 2
    for hypothesis in reduction.hypotheses:
        events = [event for event in reduction.events if event.winning_root_id == hypothesis.candidate_root_id]
        assert set(hypothesis.origin_inference_ids) == {event.inference_id for event in events}
        assert len(events) > 1


def test_exact_full_draft_duplicates_keep_all_partition_lineage_and_one_allocation(
    tmp_path: Path,
) -> None:
    """Two atomic observations of one range yield identical full draft payloads."""
    from copy import deepcopy

    class ExactDuplicateClient(ParameterizedSemanticClient):
        anchor_packet = None
        emitted_drafts: list[dict[str, object]] = []

        def complete(self, system_prompt, user_prompt):
            response = super().complete(system_prompt, user_prompt)
            payload = json.loads(user_prompt)
            request = payload.get("original_request", payload)
            value = json.loads(response.content)
            if request.get("stage") == "current_semantic_map":
                packets = {item["packet_id"]: item for item in request["packets"]}
                for result in value["results"]:
                    if self.anchor_packet is None and result.get("observations"):
                        self.anchor_packet = result["packet_id"]
                    if result["packet_id"] == self.anchor_packet:
                        original = result["observations"][0]
                        original["evidence_refs"] = [packets[result["packet_id"]]["source"][0]["range_id"]]
                        result["observations"] = [
                            {**deepcopy(original), "summary": f"Exact duplicate range interpretation {ordinal}."}
                            for ordinal in range(2)
                        ]
                    else:
                        result["observations"] = []
                        result["outcome"] = "no_observation"
                empty_ids = [item["packet_id"] for item in value["results"] if not item["observations"]]
                value["results"] = [item for item in value["results"] if item["observations"]]
                if empty_ids:
                    value.setdefault("empty_results", []).append({"packet_ids": empty_ids, "outcome": "no_observation", "reason_code": "fixture_empty"})
            elif request.get("stage") == "current_hypothesis_reduce":
                assert request["mode"] == "component_partition"
                atomic = request["inputs"][0]
                if not str(atomic.get("summary", "")).startswith("Exact duplicate range"):
                    value = {"outcome": "no_hypothesis", "hypotheses": [], "reason_code": "no_candidate"}
                else:
                    assert len(value["hypotheses"]) == 1
                    draft = value["hypotheses"][0]
                    draft["witness_refs"] = [atomic["evidence_refs"][0]]
                    self.emitted_drafts.append(deepcopy(draft))
            return response.model_copy(update={"content": json.dumps(value)})

    client = ExactDuplicateClient(category="exact-draft-fixture", rule="Preserve the exact fixture convention.")
    result = build_fixture(copied_current_source_fixture(tmp_path), tmp_path / "exact-duplicate-partitions",
        config=config_for_tests(semantic={"max_reduce_items_per_request": 1}), client=client)
    assert result.current_source
    current = result.current_source
    reduction = current.reduction
    assert len(client.emitted_drafts) == 2
    assert client.emitted_drafts[0] == client.emitted_drafts[1]
    assert len(reduction.hypotheses) == 1
    hypothesis = reduction.hypotheses[0]
    assert reduction.index.generated_candidate_events == len(reduction.events) == 2
    assert reduction.index.normalized_unique_candidate_events == 1
    assert reduction.index.duplicate_merged_events == 1
    assert {event.normalization for event in reduction.events} == {"unique", "merged_duplicate"}
    assert {event.winning_root_id for event in reduction.events} == {hypothesis.candidate_root_id}
    assert set(hypothesis.origin_inference_ids) == {event.inference_id for event in reduction.events}
    assert len(hypothesis.origin_inference_ids) == 2
    assert len(current.index.population_schedule) == len(current.populations) == 1
    assert current.index.population_schedule[0].hypothesis_id == hypothesis.hypothesis_id
    assert {pair.hypothesis_id for plan in current.index.classification_transport_plans
            for request in plan.requests for pair in request.pairs} == {hypothesis.hypothesis_id}
