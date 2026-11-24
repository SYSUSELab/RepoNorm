from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.adjudication import prepare_candidate_decisions
from repo_constraint_memory_v3.chunking import batch_units, chunk_sources
from repo_constraint_memory_v3.discovery import SourceFile, load_source_file
from repo_constraint_memory_v3.extraction import (
    PROMPT_VERSION,
    ResponseContractError,
    _repair_prompt,
    _terminal_validate_response,
    _validate_recorded_response,
    _validate_response,
    build_batch_prompt,
    extract_batch,
    stable_batch_request_ordinal,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceInputRef,
    request_identity,
)
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.models import (
    BatchExtractionResponse,
    CandidateClaim,
    ChunkingSettings,
    EvidenceUnit,
    ProviderResponse,
)
from repo_constraint_memory_v3.spans import recover_span_range

from helpers import (
    RetryThenMock,
    config_for_tests,
    copied_fixture,
    discover_fixture,
)


def _agent_unit(tmp_path: Path):
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    metadata = next(
        item for item in discovery.baseline_sources if item.path == "AGENTS.md"
    )
    source = load_source_file(repository, metadata, origin="deterministic")
    units = chunk_sources([source], max_unit_chars=4_000)
    return source, units[0]


def _configuration_response(
    unit: EvidenceUnit,
    *,
    start: int,
    end: int,
) -> BatchExtractionResponse:
    return BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "configuration_fact",
                    "claims": [
                        {
                            "rule": "The repository configures this tool setting.",
                            "claim_kind": "enforced_configuration",
                            "normative_strength": "observed",
                            "applies_to": ["**/*.py"],
                            "condition": None,
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[start].span_id,
                                    "span_end_id": unit.spans[end].span_id,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )


def _typed_invalid_json_case():
    content = (
        "{\n"
        '  "buildCommand": "pip install -r requirements-docs.txt && mkdocs build",\n'
        '  "outputDirectory": "site"\n'
        "}\n"
    )
    source = SourceFile(
        path="vercel.json",
        source_kind="configuration",
        discovery_origin="agent_supplement",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=1,
    )[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "configuration_fact",
                    "claims": [
                        {
                            "rule": "Vercel builds the documentation site with "
                            "the configured build command.",
                            "claim_kind": "enforced_configuration",
                            "normative_strength": "observed",
                            "applies_to": ["docs/**"],
                            "condition": None,
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[1].span_id,
                                    "span_end_id": unit.spans[1].span_id,
                                }
                            ],
                        },
                        {
                            "rule": "Vercel builds the documentation site and "
                            "publishes the site directory.",
                            "claim_kind": "enforced_configuration",
                            "normative_strength": "observed",
                            "applies_to": ["docs/**"],
                            "condition": None,
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[1].span_id,
                                    "span_end_id": unit.spans[2].span_id,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    return source, unit, batch, response


def _range_invalid_mkdocs_case():
    content = (
        "strict: true\n"
        + "".join(
            f"# Additional MkDocs context line {index}.\n"
            for index in range(2, 15)
        )
    )
    source = SourceFile(
        path="mkdocs.yml",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=1,
    )[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "configuration_fact",
                    "claims": [
                        {
                            "rule": "MkDocs uses every setting in this "
                            "configuration block.",
                            "claim_kind": "enforced_configuration",
                            "normative_strength": "observed",
                            "applies_to": ["docs/**"],
                            "condition": None,
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[13].span_id,
                                }
                            ],
                        },
                        {
                            "rule": "MkDocs strict mode is enabled.",
                            "claim_kind": "enforced_configuration",
                            "normative_strength": "observed",
                            "applies_to": ["docs/**"],
                            "condition": None,
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[0].span_id,
                                }
                            ],
                        },
                    ],
                }
            ]
        }
    )
    return source, unit, batch, response


class _RepeatedTypedInvalidClient:
    def __init__(self, response: BatchExtractionResponse) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt, user_prompt
        self.calls += 1
        return ProviderResponse(
            content=self.response.model_dump_json(),
            model="typed-invalid-fake",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


def test_typed_grounding_failure_is_quarantined_only_on_final_repair() -> None:
    source, unit, batch, response = _typed_invalid_json_case()
    sources_by_path = {source.path: source}

    with pytest.raises(
        ResponseContractError,
        match="typed evidence preflight failed.*exactly one active assignment",
    ):
        _validate_response(
            response,
            batch,
            sources_by_path=sources_by_path,
        )
    assert _terminal_validate_response(
        response,
        batch,
        sources_by_path=sources_by_path,
    ) == response

    settings = config_for_tests().llm.model_copy(
        update={
            "model": "typed-invalid-fake",
            "schema_repairs": 2,
            "output_limit_retries": 0,
            "provider_retries": 0,
        }
    )
    client = _RepeatedTypedInvalidClient(response)
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert client.calls == 3
    assert [item.parsed for item in outcome.attempts] == [False, False, True]
    assert [item.schema_attempt for item in outcome.attempts] == [0, 1, 2]
    assert all(
        "typed evidence preflight failed" in item.validation_errors[0]
        for item in outcome.attempts[:-1]
    )

    prepared = prepare_candidate_decisions(
        [
            (
                unit.unit_id,
                outcome.results[0].reason_code,
                candidate,
            )
            for candidate in outcome.results[0].claims
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path=sources_by_path,
        manifest_paths={source.path, "docs/index.md"},
    )
    assert len(prepared) == 2
    assert prepared[0].host_eligible is True
    assert prepared[1].outcome == "review"
    assert prepared[1].outcome_stage == "grounding"
    assert prepared[1].host_eligible is False
    assert any(
        "exactly one active assignment" in error
        for error in prepared[1].errors
    )

    class ExactSuccessCache:
        def read(self, key: str):
            del key
            return outcome.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class ExplodingClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise AssertionError("terminal success cache reached the provider")

    cache_budget = InferenceBudget(settings, cache=ExactSuccessCache())
    cached = extract_batch(
        batch,
        ExplodingClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
        budget=cache_budget,
    )
    assert cached.inference_record == outcome.inference_record
    assert cached.results == outcome.results
    assert cache_budget.cache_hits == 1
    assert cache_budget.calls == 0


def test_typed_grounding_terminal_quarantine_requires_a_repair() -> None:
    source, _, batch, response = _typed_invalid_json_case()
    client = _RepeatedTypedInvalidClient(response)
    outcome = extract_batch(
        batch,
        client,
        sources_by_path={source.path: source},
        model_name="typed-invalid-fake",
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
    )

    assert not outcome.succeeded
    assert client.calls == 1
    assert outcome.attempts[0].parsed is False
    assert "typed evidence preflight failed" in outcome.errors[0]


@pytest.mark.parametrize("malformation", ["foreign_unit", "foreign_span", "reversed"])
def test_terminal_quarantine_preserves_structural_evidence_failures(
    malformation: str,
) -> None:
    source, unit, batch, response = _typed_invalid_json_case()
    payload = response.model_dump(mode="json")
    evidence = payload["results"][0]["claims"][0]["evidence"][0]
    if malformation == "foreign_unit":
        evidence["unit_id"] = "UNIT-" + "f" * 24
        expected = "unit outside its result"
    elif malformation == "foreign_span":
        evidence["span_start_id"] = "SPAN-" + "f" * 24
        expected = "span outside the request"
    else:
        evidence["span_start_id"] = unit.spans[2].span_id
        evidence["span_end_id"] = unit.spans[1].span_id
        expected = "span endpoints are reversed"
    malformed = BatchExtractionResponse.model_validate(payload)

    with pytest.raises(ResponseContractError, match=expected):
        _terminal_validate_response(
            malformed,
            batch,
            sources_by_path={source.path: source},
        )


@pytest.mark.parametrize(
    ("malformation", "expected"),
    [
        ("duplicate_unit", "duplicate unit results"),
        ("missing_unit", "unit coverage mismatch"),
        ("missing_source", "has no immutable full source"),
        ("source_digest", "does not match its immutable full source"),
        ("source_kind", "source kind differs from its immutable full source"),
    ],
)
def test_terminal_quarantine_preserves_response_and_source_identity_failures(
    malformation: str,
    expected: str,
) -> None:
    source, _, batch, response = _range_invalid_mkdocs_case()
    payload = response.model_dump(mode="json")
    sources_by_path = {source.path: source}
    if malformation == "duplicate_unit":
        payload["results"].append(payload["results"][0])
    elif malformation == "missing_unit":
        payload["results"] = []
    elif malformation == "missing_source":
        sources_by_path = {}
    elif malformation == "source_digest":
        sources_by_path = {
            source.path: replace(source, content_sha256="f" * 64)
        }
    else:
        sources_by_path = {
            source.path: replace(source, source_kind="workflow")
        }
    malformed = BatchExtractionResponse.model_validate(payload)

    with pytest.raises(ResponseContractError, match=expected):
        _terminal_validate_response(
            malformed,
            batch,
            sources_by_path=sources_by_path,
        )


def test_terminal_range_quarantine_is_candidate_local_and_replayable() -> None:
    source, unit, batch, response = _range_invalid_mkdocs_case()
    sources_by_path = {source.path: source}

    with pytest.raises(
        ResponseContractError,
        match="14 physical lines.*ordinary evidence is limited to 12",
    ):
        _validate_response(
            response,
            batch,
            sources_by_path=sources_by_path,
        )
    assert _terminal_validate_response(
        response,
        batch,
        sources_by_path=sources_by_path,
    ) == response

    settings = config_for_tests().llm.model_copy(
        update={
            "model": "range-invalid-fake",
            "schema_repairs": 2,
            "output_limit_retries": 0,
            "provider_retries": 0,
        }
    )
    client = _RepeatedTypedInvalidClient(response)
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert client.calls == 3
    assert [item.parsed for item in outcome.attempts] == [False, False, True]
    assert _validate_recorded_response(
        response,
        outcome.inference_record,
        batch,
        sources_by_path=sources_by_path,
    ) == response

    prepared = prepare_candidate_decisions(
        [
            (
                unit.unit_id,
                outcome.results[0].reason_code,
                candidate,
            )
            for candidate in outcome.results[0].claims
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path=sources_by_path,
        manifest_paths={source.path, "docs/index.md"},
    )
    assert len(prepared) == 2
    assert prepared[0].outcome == "review"
    assert prepared[0].outcome_stage == "grounding"
    assert prepared[0].host_eligible is False
    assert any("14 physical lines" in error for error in prepared[0].errors)
    assert prepared[1].host_eligible is True

    class ExactSuccessCache:
        def read(self, key: str):
            del key
            return outcome.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class ExplodingClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise AssertionError("terminal range cache reached the provider")

    cache_budget = InferenceBudget(settings, cache=ExactSuccessCache())
    cached = extract_batch(
        batch,
        ExplodingClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
        budget=cache_budget,
    )
    assert cached.inference_record == outcome.inference_record
    assert cached.results == outcome.results
    assert cache_budget.cache_hits == 1
    assert cache_budget.calls == 0


def test_terminal_range_quarantine_requires_a_repair() -> None:
    source, _, batch, response = _range_invalid_mkdocs_case()
    client = _RepeatedTypedInvalidClient(response)
    outcome = extract_batch(
        batch,
        client,
        sources_by_path={source.path: source},
        model_name="range-invalid-fake",
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
    )

    assert not outcome.succeeded
    assert client.calls == 1
    assert outcome.attempts[0].parsed is False
    assert "14 physical lines" in outcome.errors[0]


def test_terminal_range_quarantine_catches_full_ini_expansion() -> None:
    content = "[flake8]\nexclude =\n" + "".join(
        f"    generated/path-{index},\n" for index in range(1, 65)
    )
    source = SourceFile(
        path=".flake8",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=1,
    )[0]
    # The provider selects one physical line, but deterministic INI grounding
    # expands it to the complete 65-line assignment before shape validation.
    response = _configuration_response(unit, start=1, end=1)

    assert _terminal_validate_response(
        response,
        batch,
        sources_by_path={source.path: source},
    ) == response

    prepared = prepare_candidate_decisions(
        [(unit.unit_id, "configuration_fact", response.results[0].claims[0])],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={source.path},
    )
    assert prepared[0].outcome == "review"
    assert prepared[0].outcome_stage == "grounding"
    assert prepared[0].host_eligible is False
    assert any("65 physical lines" in error for error in prepared[0].errors)


def test_terminal_range_quarantine_isolates_header_only_ini_evidence() -> None:
    content = "[flake8]\nmax-line-length = 88\n"
    source = SourceFile(
        path=".flake8",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=1,
    )[0]
    response = _configuration_response(unit, start=0, end=0)

    with pytest.raises(ResponseContractError, match="must contain its assignment"):
        _validate_response(
            response,
            batch,
            sources_by_path={source.path: source},
        )
    assert _terminal_validate_response(
        response,
        batch,
        sources_by_path={source.path: source},
    ) == response

    prepared = prepare_candidate_decisions(
        [(unit.unit_id, "configuration_fact", response.results[0].claims[0])],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={source.path},
    )
    assert prepared[0].outcome == "review"
    assert prepared[0].outcome_stage == "grounding"
    assert prepared[0].host_eligible is False
    assert any("assignment" in error for error in prepared[0].errors)


def test_span_ids_are_stable_and_soft_wrap_recovers_exact_source(
    tmp_path: Path,
) -> None:
    source, unit = _agent_unit(tmp_path)
    repeated = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]

    outcome = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path={source.path: source},
        model_name="mock",
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
    )

    assert outcome.succeeded
    assert [item.span_id for item in unit.spans] == [
        item.span_id for item in repeated.spans
    ]
    claim = outcome.results[0].claims[0]
    assert set(claim.evidence[0].model_dump()) == {
        "unit_id",
        "span_start_id",
        "span_end_id",
    }
    selected = recover_span_range(claim.evidence[0], unit)
    quote = "".join(item.text for item in selected)
    assert quote == (
        "Contributors must keep public API changes\n"
        "documented and covered by focused tests.\n"
    )


def test_duplicate_line_text_has_distinct_span_ids(tmp_path: Path) -> None:
    _, unit = _agent_unit(tmp_path)
    first = unit.spans[0]
    duplicate_id = unit.spans[-1].model_copy(
        update={"text": first.text}
    )
    from repo_constraint_memory_v3.spans import make_span_id

    assert make_span_id(
        unit.source_content_sha256,
        first.line_number,
        first.text,
    ) != make_span_id(
        unit.source_content_sha256,
        duplicate_id.line_number,
        duplicate_id.text,
    )


def test_extraction_contract_rejects_overlong_range_before_grounding() -> None:
    content = "".join(f"setting_{index} = true\n" for index in range(13))
    source = SourceFile(
        path="pyproject.toml",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "configuration_fact",
                    "claims": [
                        {
                            "rule": "The repository has thirteen settings.",
                            "claim_kind": "operational_fact",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "repository configuration",
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[-1].span_id,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    with pytest.raises(
        ResponseContractError,
        match=(
            r"results\[0\]\.claims\[0\]\.evidence\[0\].*"
            r"13 physical lines.*limited to 12"
        ),
    ):
        _validate_response(
            response,
            batch,
            sources_by_path={source.path: source},
        )


def test_extraction_contract_accepts_one_long_ini_continuation_block() -> None:
    content = (
        "[report]\n"
        "exclude_lines =\n"
        + "".join(f"    pattern-{index}\n" for index in range(15))
        + "precision = 2\n"
    )
    source = SourceFile(
        path=".coveragerc",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="1" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    validated = _validate_response(
        _configuration_response(unit, start=1, end=16),
        batch,
        sources_by_path={source.path: source},
    )

    assert validated.results[0].claims[0].evidence[0].span_end_id == (
        unit.spans[16].span_id
    )


def test_extraction_contract_repairs_truncated_ini_atom_deterministically() -> None:
    content = (
        "[pytest]\n"
        "filterwarnings =\n"
        "    error\n"
        "    ignore:legacy warning\n"
        "addopts = -ra\n"
    )
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="9" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    validated = _validate_response(
        _configuration_response(unit, start=1, end=1),
        batch,
        sources_by_path={source.path: source},
    )

    assert validated.results[0].claims[0].evidence[0].span_start_id == (
        unit.spans[1].span_id
    )


def test_extraction_contract_rejects_long_ini_range_crossing_assignment() -> None:
    content = (
        "[report]\n"
        "exclude_lines =\n"
        + "".join(f"    pattern-{index}\n" for index in range(15))
        + "precision = 2\n"
    )
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="2" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    with pytest.raises(ResponseContractError, match="complete INI/CFG"):
        _validate_response(
            _configuration_response(unit, start=1, end=17),
            batch,
            sources_by_path={source.path: source},
        )


def test_extraction_contract_accepts_one_long_toml_multiline_array() -> None:
    content = (
        "[tool.ruff.lint]\n"
        "select = [\n"
        + "".join(f'    "RULE{index}",\n' for index in range(13))
        + "]\n"
        + "ignore = []\n"
    )
    source = SourceFile(
        path="pyproject.toml",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="3" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    validated = _validate_response(
        _configuration_response(unit, start=1, end=15),
        batch,
        sources_by_path={source.path: source},
    )

    assert len(validated.results[0].claims[0].evidence) == 1


def test_extraction_contract_accepts_yaml_sequence_mapping_assignment() -> None:
    content = "hooks:\n  - id: trailing-whitespace\n"
    source = SourceFile(
        path=".pre-commit-config.yaml",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="8" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    validated = _validate_response(
        _configuration_response(unit, start=1, end=1),
        batch,
        sources_by_path={source.path: source},
    )

    assert len(validated.results[0].claims[0].evidence) == 1


def test_extraction_retains_trigger_only_workflow_claim_for_host_adjudication() -> None:
    content = "on: [pull_request]\n"
    source = SourceFile(
        path=".github/workflows/tests.yml",
        source_kind="workflow",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="9" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "workflow_fact",
                    "claims": [
                        {
                            "rule": "Pull requests run the test jobs.",
                            "claim_kind": "workflow_fact",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "on pull requests",
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[0].span_id,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    validated = _validate_response(
        response,
        batch,
        sources_by_path={source.path: source},
    )

    assert validated == response


def test_extraction_contract_accepts_workflow_predicate_and_action_ranges() -> None:
    content = (
        "if: github.event.pull_request.draft == false\n"
        "run: python -m pytest\n"
    )
    source = SourceFile(
        path=".github/workflows/tests.yml",
        source_kind="workflow",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="a" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "workflow_fact",
                    "claims": [
                        {
                            "rule": "Non-draft pull requests run pytest.",
                            "claim_kind": "workflow_fact",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "when the pull request is not a draft",
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[0].span_id,
                                },
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[1].span_id,
                                    "span_end_id": unit.spans[1].span_id,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    )

    validated = _validate_response(
        response,
        batch,
        sources_by_path={source.path: source},
    )

    assert len(validated.results[0].claims[0].evidence) == 2


def test_extraction_contract_rejects_more_than_64_structured_lines() -> None:
    content = "exclude_lines =\n" + "".join(
        f"    pattern-{index}\n" for index in range(64)
    )
    source = SourceFile(
        path=".coveragerc",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="4" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=1)[0]

    with pytest.raises(
        ResponseContractError,
        match="65 physical lines.*maximum is 64",
    ):
        _validate_response(
            _configuration_response(unit, start=0, end=64),
            batch,
            sources_by_path={source.path: source},
        )


def test_extraction_contract_rejects_make_declaration_without_recipe() -> None:
    content = ".PHONY: tests\ntests:\n\tuv run pytest\n"
    source = SourceFile(
        path="Makefile",
        source_kind="command",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    units = chunk_sources([source], max_unit_chars=4_000)
    declaration = units[0]
    assert len(units) == 1
    assert declaration.content == content
    batch = batch_units(
        units,
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": declaration.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "command_fact",
                    "claims": [
                        {
                            "rule": "The tests target is phony.",
                            "claim_kind": "command_recipe",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "tests target",
                            "evidence": [
                                {
                                    "unit_id": declaration.unit_id,
                                    "span_start_id": declaration.spans[0].span_id,
                                    "span_end_id": declaration.spans[0].span_id,
                                }
                            ],
                        }
                    ],
                },
            ]
        }
    )

    with pytest.raises(ResponseContractError, match="target and at least one"):
        _validate_response(
            response,
            batch,
            sources_by_path={source.path: source},
        )


def test_make_chunker_only_pairs_matching_immediate_phony_target() -> None:
    content = (
        ".PHONY: test\n"
        "test:\n"
        "\tuv run pytest\n"
        ".PHONY: lint\n"
        "# An intervening line prevents pairing.\n"
        "lint:\n"
        "\tuv run ruff check .\n"
    )
    source = SourceFile(
        path="Makefile",
        source_kind="command",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="5" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )

    units = chunk_sources([source], max_unit_chars=4_000)

    assert units[0].content == ".PHONY: test\ntest:\n\tuv run pytest\n"
    assert units[1].content == (
        ".PHONY: lint\n# An intervening line prevents pairing.\n"
    )
    assert units[2].content == "lint:\n\tuv run ruff check .\n"


def test_chunking_default_isolates_each_explicit_unit() -> None:
    assert ChunkingSettings().max_units_per_batch == 1


def test_extraction_repair_includes_only_a_bounded_invalid_response_excerpt(
    tmp_path: Path,
) -> None:
    _, unit = _agent_unit(tmp_path)
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=1,
    )[0]
    invalid = "SENSITIVE_INVALID_RESPONSE_BODY"
    error = "results[0].claims[0].evidence[0]: bad range"

    prompt = json.loads(
        _repair_prompt(
            batch,
            build_batch_prompt(batch),
            invalid,
            [error],
        )
    )

    assert prompt["original_request"]["batch_id"] == batch.batch_id
    assert prompt["invalid_response_sha256"] == hashlib.sha256(
        invalid.encode("utf-8")
    ).hexdigest()
    assert prompt["invalid_response_bytes"] == len(invalid.encode("utf-8"))
    assert prompt["invalid_response_excerpt"] == invalid
    assert prompt["invalid_response_excerpt_truncated"] is False
    assert prompt["validation_errors"] == [error]
    assert "exactly one result" in prompt["instruction"]
    assert "copy each unit_id exactly" in prompt["instruction"]
    assert "truncated INI/CFG" in prompt["instruction"]
    assert "never invent an action" in prompt["instruction"]
    assert "omit that candidate" in prompt["instruction"]
    assert "discarding valid sibling candidates" in prompt["instruction"]

    oversized = "界" * 10_000
    oversized_prompt = json.loads(
        _repair_prompt(
            batch,
            build_batch_prompt(batch),
            oversized,
            [error],
        )
    )
    assert oversized_prompt["invalid_response_excerpt_bytes"] <= 16 * 1024
    assert oversized_prompt["invalid_response_excerpt_truncated"] is True
    assert oversized_prompt["invalid_response_sha256"] == hashlib.sha256(
        oversized.encode("utf-8")
    ).hexdigest()
    assert "merely to evade" in prompt["instruction"]


def test_extraction_contract_rejects_taskfile_task_without_command() -> None:
    content = "tasks:\n  test:\n    desc: Run tests\n"
    source = SourceFile(
        path="Taskfile.yml",
        source_kind="command",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]
    response = BatchExtractionResponse.model_validate(
        {
            "results": [
                {
                    "unit_id": unit.unit_id,
                    "classification": "contains_claims",
                    "reason_code": "command_fact",
                    "claims": [
                        {
                            "rule": "Use the test task.",
                            "claim_kind": "command_recipe",
                            "normative_strength": "observed",
                            "applies_to": ["**"],
                            "condition": "test task",
                            "evidence": [
                                {
                                    "unit_id": unit.unit_id,
                                    "span_start_id": unit.spans[0].span_id,
                                    "span_end_id": unit.spans[-1].span_id,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    with pytest.raises(ResponseContractError, match="active task"):
        _validate_response(
            response,
            batch,
            sources_by_path={source.path: source},
        )


def test_agent_units_do_not_perturb_deterministic_extraction_batches() -> None:
    deterministic = SourceFile(
        path="z-config.toml",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="1" * 64,
        size_bytes=9,
        content="x = true\n",
    )
    supplemental = SourceFile(
        path="a-extra.toml",
        source_kind="configuration",
        discovery_origin="agent_supplement",
        reason="test",
        referenced_by=(),
        content_sha256="2" * 64,
        size_bytes=9,
        content="y = true\n",
    )
    deterministic_units = chunk_sources(
        [deterministic], max_unit_chars=4_000
    )
    baseline = batch_units(
        deterministic_units,
        max_batch_chars=8_000,
        max_units_per_batch=8,
    )
    supplemented = batch_units(
        [
            *deterministic_units,
            *chunk_sources([supplemental], max_unit_chars=4_000),
        ],
        max_batch_chars=8_000,
        max_units_per_batch=8,
    )

    assert supplemented[: len(baseline)] == baseline
    assert supplemented[-1].units[0].discovery_origin == "agent_supplement"


def test_unrelated_earlier_source_does_not_perturb_extraction_request() -> None:
    original = SourceFile(
        path="z-instructions.md",
        source_kind="instruction",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="3" * 64,
        size_bytes=29,
        content="Contributors must run tests.\n",
    )
    earlier = SourceFile(
        path="a-instructions.md",
        source_kind="instruction",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="4" * 64,
        size_bytes=29,
        content="Contributors must run Ruff.\n",
    )
    original_units = chunk_sources([original], max_unit_chars=4_000)
    baseline = batch_units(
        original_units,
        max_batch_chars=8_000,
        max_units_per_batch=8,
    )
    supplemented = batch_units(
        [
            *original_units,
            *chunk_sources([earlier], max_unit_chars=4_000),
        ],
        max_batch_chars=8_000,
        max_units_per_batch=8,
    )

    original_batch = next(
        batch for batch in supplemented if batch.units[0].path == original.path
    )
    assert baseline[0] == original_batch
    assert stable_batch_request_ordinal(baseline[0]) == (
        stable_batch_request_ordinal(original_batch)
    )
    snapshot = "sha256:" + "0" * 64
    request_ids = [
        request_identity(
            stage="explicit_extract",
            protocol_version=PROMPT_VERSION,
            mode=None,
            inputs=[
                InferenceInputRef(
                    kind="explicit_unit",
                    stable_id=unit.unit_id,
                    digest=unit.content_sha256,
                    snapshot_digest=snapshot,
                )
                for unit in batch.units
            ],
            request_ordinal=stable_batch_request_ordinal(batch),
        )
        for batch in (baseline[0], original_batch)
    ]
    assert request_ids[0] == request_ids[1]
    assert all(len({unit.path for unit in batch.units}) == 1 for batch in supplemented)


def test_discriminated_results_reject_invalid_reason_and_unknown_fields() -> None:
    unit_id = "UNIT-" + "a" * 24
    with pytest.raises(ValidationError):
        BatchExtractionResponse.model_validate(
            {
                "results": [
                    {
                        "unit_id": unit_id,
                        "classification": "no_claims",
                        "reason_code": "direct_norm",
                        "claims": [],
                    }
                ]
            }
        )
    with pytest.raises(ValidationError, match="extra"):
        CandidateClaim.model_validate(
            {
                "rule": "Contributors must test changes.",
                "claim_kind": "normative_declaration",
                "normative_strength": "required",
                "applies_to": ["**"],
                "condition": None,
                "evidence": [
                    {
                        "unit_id": unit_id,
                        "span_start_id": "SPAN-" + "b" * 24,
                        "span_end_id": "SPAN-" + "b" * 24,
                    }
                ],
                "evidence_type": "direct_declaration",
            }
        )


def test_legal_no_claims_needs_no_schema_repair(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    metadata = next(
        item for item in discovery.baseline_sources if item.path == "README.md"
    )
    source = load_source_file(repository, metadata, origin="deterministic")
    unit = chunk_sources([source], max_unit_chars=4_000)[0]
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]

    outcome = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path={source.path: source},
        model_name="mock",
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
    )

    assert outcome.succeeded
    assert outcome.results[0].classification == "no_claims"
    assert len(outcome.attempts) == 1
    assert not outcome.attempts[0].repair
    assert "quote" not in json.dumps(
        outcome.results[0].model_dump(mode="json")
    )


def test_live_grounding_is_reused_but_replay_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3 import source_adapters

    source, unit = _agent_unit(tmp_path)
    batch = batch_units([unit], max_batch_chars=8_000, max_units_per_batch=2)[0]
    original = source_adapters._derive_prose
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source_adapters, "_derive_prose", counted)
    outcome = extract_batch(
        batch, MockLLMClient(), sources_by_path={source.path: source},
        model_name="mock", schema_repairs=1, provider_retries=0,
        provider_retry_delay_seconds=0,
    )
    assert outcome.succeeded and outcome.grounding_memo
    first_calls = calls
    assert first_calls == len(outcome.grounding_memo) > 0
    candidates = [
        (result.unit_id, result.reason_code, candidate)
        for result in outcome.results for candidate in result.claims
    ]
    arguments = {
        "units_by_id": {unit.unit_id: unit},
        "sources_by_path": {source.path: source},
        "manifest_paths": {source.path},
    }
    reused = prepare_candidate_decisions(
        candidates, **arguments, _grounding_memo=outcome.grounding_memo,
    )
    assert calls == first_calls
    independently_grounded = prepare_candidate_decisions(candidates, **arguments)
    assert reused == independently_grounded
    assert calls > first_calls
    before_replay = calls
    _validate_recorded_response(
        BatchExtractionResponse(results=list(outcome.results)),
        outcome.inference_record, batch, sources_by_path={source.path: source},
    )
    assert calls > before_replay
    # Equal replacement objects are independently rebound rather than trusted
    # from a prior source object's process-local identity.
    before_replacement = calls
    prepare_candidate_decisions(
        candidates, **{**arguments, "sources_by_path": {source.path: replace(source)}},
        _grounding_memo=outcome.grounding_memo,
    )
    assert calls > before_replacement


def test_mock_abstains_when_generated_rule_would_exceed_schema_limit() -> None:
    unit_id = "UNIT-" + "a" * 24
    span_id = "SPAN-" + "b" * 24
    response = MockLLMClient().complete(
        "",
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "batch_id": "BATCH-" + "c" * 24,
                "units": [
                    {
                        "unit_id": unit_id,
                        "path": "AGENTS.md",
                        "source_kind": "instruction",
                        "discovery_origin": "deterministic",
                        "default_scope": ["**"],
                        "annotated_text": "",
                        "spans": [
                            {
                                "span_id": span_id,
                                "line_number": 1,
                                "text": (
                                    "Contributors must "
                                    + "test changes carefully " * 30
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
    )
    parsed = BatchExtractionResponse.model_validate_json(response.content)

    assert parsed.results[0].classification == "no_claims"
    assert parsed.results[0].claims == []


def test_invalid_schema_gets_one_bounded_repair(tmp_path: Path) -> None:
    source, unit = _agent_unit(tmp_path)
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]

    class InvalidThenMock:
        def __init__(self) -> None:
            self.calls = 0
            self.mock = MockLLMClient()

        def complete(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if self.calls == 1:
                return ProviderResponse(
                    content='{"results":[]}',
                    model="invalid",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                )
            return self.mock.complete(system_prompt, user_prompt)

    client = InvalidThenMock()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path={source.path: source},
        model_name="mock",
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
    )
    assert outcome.succeeded
    assert client.calls == 2
    assert [item.repair for item in outcome.attempts] == [False, True]


def test_retryable_provider_failure_is_bounded(tmp_path: Path) -> None:
    source, unit = _agent_unit(tmp_path)
    batch = batch_units(
        [unit],
        max_batch_chars=8_000,
        max_units_per_batch=2,
    )[0]
    client = RetryThenMock()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path={source.path: source},
        model_name="mock",
        schema_repairs=0,
        provider_retries=1,
        provider_retry_delay_seconds=0,
    )
    assert outcome.succeeded
    assert client.calls == 2
    assert outcome.attempts[0].provider_error
    assert outcome.attempts[1].provider_try == 1
