from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
from pydantic import ValidationError

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.inference import InferenceBudget, attach_response_json_schema
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.observations.semantic_units import (
    reconstruct_packet_bytes,
)
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.semantic.map import (
    MapObservationDraft,
    SemanticMapResponse,
    _map_provider_response_model,
    _MapRequest,
    _map_request,
    _packet_work,
    run_semantic_map,
    _normalize_response,
    _terminal_validate_response,
    _validate_response,
)


def test_compact_map_empty_groups_require_exact_coverage() -> None:
    request = _MapRequest(
        payload="{}",
        provider_model=_map_provider_response_model(("p0", "p1"), ("p0", "p1", "r2")),
        canonical_by_alias={"p0": "PACKET-a", "p1": "PACKET-b", "r2": "EVIDENCE-a"},
        packet_aliases=frozenset({"p0", "p1"}),
    )
    response = {
        "results": [],
        "empty_results": [{"packet_ids": ["p0", "p1"], "outcome": "no_observation"}],
    }
    expanded = SemanticMapResponse.model_validate(request.expand(response))
    assert [item.packet_id for item in expanded.results] == ["PACKET-a", "PACKET-b"]
    assert all(item.reason_code == "model_reason_omitted" for item in expanded.results)
    for ids in (["p0"], ["p0", "p0"], ["p0", "PACKET-b"]):
        response["empty_results"][0]["packet_ids"] = ids
        with pytest.raises(ValueError, match="coverage"):
            request.expand(response)


@pytest.mark.parametrize("malformed", [
    {"results": [{"packet_id": "p0", "outcome": "no_observation",
                  "observations": [], "reason_code": "ordinary_source"}],
     "empty_results": []},
    {"results": [{"packet_id": "p0", "outcome": "observations",
                  "reason_code": "invariant"}], "empty_results": []},
    {"results": [], "empty_results": [{"packet_ids": ["p0"],
      "outcome": "no_observation", "reason_code": ""}]},
    {"results": [], "empty_results": [{"packet_ids": ["p0"],
      "outcome_group": "no_observation", "reason_code": "ordinary_source"}]},
])
def test_map_rejects_observed_malformed_group_shapes(malformed: dict) -> None:
    model = _map_provider_response_model(("p0",), ("p0", "r1"))
    with pytest.raises(ValidationError):
        model.model_validate(malformed)
    valid = model.model_validate({
        "results": [],
        "empty_results": [{"packet_ids": ["p0"], "outcome": "no_observation",
                           "reason_code": "ordinary_source"}],
    })
    assert valid.empty_results[0].packet_ids == ["p0"]


def test_compact_map_does_not_treat_guessed_canonical_ids_as_wire_refs() -> None:
    request = _MapRequest(
        payload="{}",
        provider_model=_map_provider_response_model(("p0",), ("p0", "r1")),
        canonical_by_alias={"p0": "PACKET-a", "r1": "EVIDENCE-a"},
        packet_aliases=frozenset({"p0"}),
    )
    observation = {
        "category": "boundary", "summary": "Keep EVIDENCE-a text unchanged.",
        "contributor_relevance": "actionable", "evidence_refs": ["EVIDENCE-a"],
        "related_unit_ids": [], "uncertainty": None,
    }
    value = request.expand({
        "results": [{"packet_id": "p0", "outcome": "observations", "observations": [observation]}],
        "empty_results": [],
    })
    draft = value["results"][0]["observations"][0]
    assert draft["summary"] == observation["summary"]
    assert draft["evidence_refs"] == ["unknown-map-reference:EVIDENCE-a"]


def test_short_alias_is_rechecked_against_its_exact_packet_domain() -> None:
    request = _MapRequest(
        payload="{}",
        provider_model=_map_provider_response_model(
            ("p0", "p1"), ("p0", "p1", "r2", "r3")
        ),
        canonical_by_alias={
            "p0": "PACKET-a", "p1": "PACKET-b",
            "r2": "EVIDENCE-a", "r3": "EVIDENCE-b",
        },
        packet_aliases=frozenset({"p0", "p1"}),
    )
    observation = {
        "category": "boundary", "summary": "A contributor invariant.",
        "contributor_relevance": "actionable", "evidence_refs": ["r2"],
        "related_unit_ids": [], "uncertainty": None,
    }
    response = SemanticMapResponse.model_validate(request.expand({
        "results": [{
            "packet_id": "p0", "outcome": "observations",
            "observations": [observation, {**observation, "evidence_refs": ["r3"]}],
        }],
        "empty_results": [{"packet_ids": ["p1"], "outcome": "uncertain"}],
    }))
    work = [
        SimpleNamespace(
            packet=SimpleNamespace(packet_id=f"PACKET-{suffix}"),
            unit=SimpleNamespace(unit_id=f"UNIT-{suffix}"),
            allowed_refs=frozenset({f"EVIDENCE-{suffix}"}),
        )
        for suffix in ("a", "b")
    ]
    settings = config_for_tests().semantic
    with pytest.raises(ValueError, match="unsupplied evidence"):
        _validate_response(response, work, settings)
    terminal = _terminal_validate_response(response, work, settings)
    assert terminal.results[0].outcome == "observations"
    assert len(terminal.results[0].observations) == 1
    assert terminal.results[0].observations[0].evidence_refs == ["EVIDENCE-a"]
    assert terminal.results[1].outcome == "uncertain"


def test_map_short_ids_preserve_source_and_reuse_checked_file_bytes(tmp_path: Path, monkeypatch) -> None:
    from test_semantic_units import _bundle
    from repo_constraint_memory_v3.semantic import map as map_module

    repository, discovery, _, units, _ = _bundle(tmp_path)
    reads: list[str] = []
    original_read = map_module.safe_read_manifest_file

    def checked_read(root, metadata):
        reads.append(metadata.path)
        return original_read(root, metadata)

    monkeypatch.setattr(map_module, "safe_read_manifest_file", checked_read)
    work = _packet_work(repository, discovery, units)
    assert len(reads) == len(set(reads))
    assert len(work) > len(reads)
    source = json.loads(work[0].rendered)
    source["source"][0]["text"] += "\n# Literal " + work[0].packet.packet_id
    changed = replace(work[0], rendered=json.dumps(source).encode())
    request = _map_request([changed], discovery.repository.snapshot_digest)
    packet = json.loads(request.payload)["packets"][0]
    assert packet["source"][0]["text"] == source["source"][0]["text"]
    assert request.canonical_by_alias[packet["packet_id"]] == changed.packet.packet_id
    assert request.canonical_by_alias[packet["source"][0]["range_id"]] == source["source"][0]["range_id"]
    assert len(set(request.canonical_by_alias.values())) == len(request.canonical_by_alias)


def test_map_schema_attached_request_gate_rejects_bare_payload_that_fits(tmp_path: Path) -> None:
    from test_semantic_units import _bundle

    repository, discovery, _, units, config = _bundle(tmp_path)
    work = _packet_work(repository, discovery, units)
    requests = [_map_request([item], discovery.repository.snapshot_digest) for item in work]
    bare_limit = max(len(request.payload.encode()) for request in requests)
    assert any(len(attach_response_json_schema(request.payload, request.provider_model).encode()) > bare_limit for request in requests)

    class EmptyClient:
        def complete(self, system_prompt, user_prompt):
            payload = json.loads(user_prompt)
            return ProviderResponse(
                content=json.dumps({"results": [], "empty_results": [{
                    "packet_ids": [packet["packet_id"] for packet in payload["packets"]],
                    "outcome": "no_observation", "reason_code": "inspected_empty",
                }]}),
                model="offline-test", input_tokens=0, output_tokens=0, latency_ms=0,
            )

    result = run_semantic_map(
        repository_root=repository, discovery=discovery, semantic_units=units,
        semantic_settings=config.semantic.model_copy(update={"max_map_request_bytes": bare_limit}),
        llm_settings=config.llm, client=EmptyClient(), budget=InferenceBudget(config.llm),
        protocol_manifest_digest="a" * 64,
    )
    rejected = [record for record in result.inference_records if "max_map_request_bytes" in record.incomplete_reasons]
    assert rejected
    assert all(not record.attempts for record in rejected)
    assert result.index.packet_failure > 0


def test_compact_map_mixed_terminal_results_round_trip_through_success_cache(tmp_path: Path) -> None:
    from test_semantic_units import _bundle
    from test_compact_reducer_transport import _MemoryCache

    repository, discovery, _, units, config = _bundle(tmp_path)
    settings = config.llm.model_copy(update={"schema_repairs": 0})
    cache = _MemoryCache()

    class MixedClient:
        calls = 0
        request_bytes = 0

        def complete(self, system_prompt, user_prompt):
            self.calls += 1
            self.request_bytes += len(user_prompt.encode("utf-8"))
            payload = json.loads(user_prompt)
            results = []
            empty = []
            for ordinal, packet in enumerate(payload["packets"]):
                if ordinal % 3:
                    empty.append({"packet_ids": [packet["packet_id"]],
                                  "outcome": "no_observation" if ordinal % 3 == 1 else "uncertain",
                                  "reason_code": "explicit_empty_outcome"})
                else:
                    results.append({"packet_id": packet["packet_id"], "outcome": "observations",
                                    "reason_code": "inspect", "observations": [{
                                        "category": "boundary", "summary": "A source-bound contributor invariant.",
                                        "contributor_relevance": "actionable", "related_unit_ids": [],
                                        "evidence_refs": ["foreign_ref"], "uncertainty": None,
                                    }]})
            return ProviderResponse(content=json.dumps({"results": results, "empty_results": empty}),
                                    model="offline-test", input_tokens=1, output_tokens=1, latency_ms=0)

    client = MixedClient()

    def run():
        return run_semantic_map(
            repository_root=repository, discovery=discovery, semantic_units=units,
            semantic_settings=config.semantic, llm_settings=settings, client=client,
            budget=InferenceBudget(settings, cache=cache), protocol_manifest_digest="a" * 64,
        )

    first = run()
    assert first.index.resource_use["planned_request_bytes"] == client.request_bytes
    assert first.index.resource_use["source_packet_bytes"] == sum(
        len(item.rendered) for item in _packet_work(repository, discovery, units)
    )
    assert "request_bytes" not in first.index.resource_use
    assert first.index.packet_no_observation > 0
    assert first.index.packet_uncertain > 0
    assert first.index.packet_failure == 0
    assert any(item.reason_codes == ["host_quarantined_request_external_observation"] for item in first.results)
    count = client.calls
    second = run()
    assert client.calls == count
    assert first == second


def test_map_observation_schema_requires_contributor_relevance() -> None:
    with pytest.raises(ValidationError):
        MapObservationDraft(
            category="plain_definition",
            summary="Defines an ordinary helper.",
            contributor_relevance="none",
            evidence_refs=["PYENT-example"],
            related_unit_ids=[],
            uncertainty=None,
        )


def test_map_provider_schema_binds_request_local_refs_and_packet_ids() -> None:
    schema = _map_provider_response_model(
        ("PACKET-a", "PACKET-b"),
        ("EVIDENCE-a", "EVIDENCE-b"),
    ).model_json_schema()
    observation = schema["$defs"]["BoundedMapObservationDraft"]
    packet = schema["$defs"]["BoundedMapPacketDraft"]

    assert observation["properties"]["evidence_refs"]["items"]["enum"] == [
        "EVIDENCE-a",
        "EVIDENCE-b",
    ]
    assert packet["properties"]["packet_id"]["enum"] == [
        "PACKET-a",
        "PACKET-b",
    ]


def test_map_normalizes_missing_reason_and_terminally_isolates_foreign_ref() -> None:
    normalized = _normalize_response(
        {
            "results": [
                {
                    "packet_id": "PACKET-a",
                    "outcome": "observations",
                    "observations": [
                        {
                            "category": "boundary",
                            "summary": "A supported observation.",
                            "contributor_relevance": "actionable",
                            "evidence_refs": ["FOREIGN-ref"],
                            "related_unit_ids": [],
                            "uncertainty": None,
                        }
                    ],
                }
            ]
        }
    )
    assert normalized["results"][0]["reason_code"] == "model_reason_omitted"
    response = SemanticMapResponse.model_validate(normalized)
    work = [
        SimpleNamespace(
            packet=SimpleNamespace(packet_id="PACKET-a"),
            unit=SimpleNamespace(unit_id="UNIT-a"),
            allowed_refs=frozenset({"EVIDENCE-a", "UNIT-a"}),
        )
    ]
    settings = config_for_tests().semantic
    with pytest.raises(ValueError, match="unsupplied evidence"):
        _validate_response(response, work, settings)

    terminal = _terminal_validate_response(response, work, settings)
    assert terminal.results[0].outcome == "uncertain"
    assert terminal.results[0].observations == []
    assert terminal.results[0].reason_code == (
        "host_quarantined_request_external_observation"
    )


def test_semantic_map_covers_every_packet_and_reconstructs_exact_ranges(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=ParameterizedSemanticClient(
            category="boundary-adaptation",
            rule="Contributors preserve the fixture boundary adaptation pattern.",
        ),
    )
    assert result.current_source is not None
    assert result.semantic_units is not None
    semantic_map = result.current_source.semantic_map
    packets = [
        packet for unit in result.semantic_units.units for packet in unit.packets
    ]
    assert semantic_map.index.packet_count == len(packets)
    assert {item.packet_id for item in semantic_map.results} == {
        item.packet_id for item in packets
    }
    assert all(item.outcome != "failure" for item in semantic_map.results)

    unit = next(item for item in result.semantic_units.units if item.packets)
    packet = unit.packets[0]
    source = (repository / unit.path).read_bytes()
    reconstructed = json.loads(
        reconstruct_packet_bytes(packet, unit=unit, content=source)
    )
    ranges = {
        item.range_id: item
        for segment in unit.segments
        for item in segment.source_ranges
    }
    assert [item["range_id"] for item in reconstructed["source"]] == (
        packet.input_range_ids
    )
    for item in reconstructed["source"]:
        source_range = ranges[item["range_id"]]
        assert item["text"].encode() == source[
            source_range.start_utf8_byte : source_range.end_utf8_byte
        ]


def test_semantic_map_observation_cap_preserves_complete_selection_ledger(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(semantic={"max_total_observations": 2}),
        client=ParameterizedSemanticClient(
            category="bounded-map-selection",
            rule="Contributors preserve the bounded fixture convention.",
        ),
    )

    assert result.current_source is not None
    semantic_map = result.current_source.semantic_map
    index = semantic_map.index
    assert index.observation_candidate_count > 2
    assert index.observation_count == len(semantic_map.observations) == 2
    assert index.omitted_observation_count == (
        index.observation_candidate_count - index.observation_count
    )
    assert sum(index.component_observation_candidate_counts.values()) == (
        index.observation_candidate_count
    )
    assert sum(index.component_observation_selected_counts.values()) == (
        index.observation_count
    )
    assert set(index.component_observation_selected_counts) == set(
        index.component_observation_candidate_counts
    )
    assert "max_total_observations" in index.incomplete_reasons
    assert index.status == "partial"
    assert validate_run_directory(run, repository_root=repository) == []


class _BrokenMapClient:
    def __init__(self) -> None:
        self.mock = MockLLMClient()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "current_semantic_map":
            return ProviderResponse(
                content='{"results":[]}',
                model="broken-map-fake",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )
        return self.mock.complete(system_prompt, user_prompt)


class _ReorderedMapClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_semantic_map":
            return response
        value = json.loads(response.content)
        value["results"].reverse()
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="reordered-map-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_map_canonicalizes_complete_unique_results_to_manifest_order(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=_ReorderedMapClient(
            category="reordered-map-fixture",
            rule="Contributors preserve the fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.semantic_map.index.status == "complete"
    assert validate_run_directory(run, repository_root=repository) == []


def test_map_schema_failure_is_visible_and_cannot_become_no_observation(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(repository, run, client=_BrokenMapClient())
    assert result.current_source is not None
    semantic_map = result.current_source.semantic_map
    assert semantic_map.index.status == "partial"
    assert semantic_map.index.packet_failure == semantic_map.index.packet_count
    assert semantic_map.index.packet_no_observation == 0
    assert not result.build_report["complete"]
    assert validate_run_directory(run, repository_root=repository) == []
    assert any(
        "unified package is partial" in item
        for item in validate_run_directory(
            run,
            repository_root=repository,
            require_complete=True,
        )
    )


def test_optional_map_failure_remains_a_valid_visible_partial_run(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(semantic={"required": False}),
        client=_BrokenMapClient(),
    )
    assert result.current_source is not None
    assert result.current_source.index.status == "partial"
    assert not result.build_report["complete"]
    assert result.build_report["unified"]["completeness"][
        "package_status"
    ] == "partial"
    assert validate_run_directory(run, repository_root=repository) == []
