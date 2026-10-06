from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import (
    SequenceClient,
    build_fixture,
    config_for_tests,
    copied_fixture,
    discover_fixture,
)
from repo_constraint_memory_v3.chunking import batch_units, chunk_sources
from repo_constraint_memory_v3.builder import load_build_config
from repo_constraint_memory_v3.discovery import load_source_file
from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.extraction import (
    MAX_OUTPUT_TOKENS as MAX_EXTRACTION_OUTPUT_TOKENS,
    MAX_RESPONSE_BYTES as MAX_EXTRACTION_RESPONSE_BYTES,
    MAX_TIMEOUT_SECONDS as MAX_EXTRACTION_TIMEOUT_SECONDS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_batch_prompt,
    extract_batch,
)
from repo_constraint_memory_v3.explicit_verification import (
    SYSTEM_PROMPT as EXPLICIT_VERIFIER_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.inference import (
    DISABLED_THINKING_REASONING_CONTENT_VIOLATION,
    DISABLED_THINKING_REASONING_TOKENS_VIOLATION,
    FUTURE_STAGES,
    LIVE_STAGES,
    STAGE_INPUT_KINDS,
    InferenceBudget,
    InferenceInputRef,
    InferenceRecord,
    attach_response_json_schema,
    max_provider_calls_per_request,
    request_identity,
    run_call_permitted_stage_calls,
    run_budgeted_stage_calls,
)
from repo_constraint_memory_v3.llm import (
    LLMProviderError,
    MockLLMClient,
    OpenAICompatibleClient,
)
from repo_constraint_memory_v3.models import BatchExtractionResponse, ProviderResponse
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.semantic.classify import CLASSIFY_SYSTEM_PROMPT
from repo_constraint_memory_v3.semantic.falsification import (
    FALSIFICATION_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.semantic.map import MAP_SYSTEM_PROMPT
from repo_constraint_memory_v3.semantic.reduce import REDUCE_SYSTEM_PROMPT
from repo_constraint_memory_v3.unified.semantic import (
    GROUP_SYSTEM_PROMPT,
    TEMPORAL_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)


def _batch(tmp_path: Path, *, two: bool = False):
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    metadata = discovery.baseline_sources[: 2 if two else 1]
    sources = [
        load_source_file(repository, item, origin="deterministic")
        for item in metadata
    ]
    units = chunk_sources(sources, max_unit_chars=20_000)
    batch = batch_units(
        units,
        max_batch_chars=40_000,
        max_units_per_batch=4,
    )[0]
    return batch, config, {source.path: source for source in sources}


def test_explicit_prompt_declares_strict_claim_contract() -> None:
    assert "`reason_code`" in SYSTEM_PROMPT
    assert "Never use a `reason`" in SYSTEM_PROMPT
    assert "`workflow_fact`" in SYSTEM_PROMPT
    assert all(
        value in SYSTEM_PROMPT
        for value in ("`required`", "`recommended`", "`observed`")
    )
    assert "necessary activation context" in SYSTEM_PROMPT
    assert "Configuration facts normally use null" in SYSTEM_PROMPT
    assert "repository-specific" in SYSTEM_PROMPT
    assert "including imperatives" in SYSTEM_PROMPT
    assert "ordinary package metadata" in SYSTEM_PROMPT
    assert "not automatically eligible" in SYSTEM_PROMPT
    assert "smallest sufficient evidence" in SYSTEM_PROMPT
    assert "Never cite a section header as a separate range" in SYSTEM_PROMPT
    assert "fully entailed" in SYSTEM_PROMPT
    assert "do not combine unrelated" in SYSTEM_PROMPT
    assert "not the location of the evidence file" in SYSTEM_PROMPT
    assert "`**/*.py` or a narrower Python glob" in SYSTEM_PROMPT
    assert "test-only rules" in SYSTEM_PROMPT
    assert "configuration names excluded directories" in SYSTEM_PROMPT
    assert "CI Python-version matrix" in SYSTEM_PROMPT
    assert "local/path dependency topology" in SYSTEM_PROMPT
    assert "circular wrappers" in SYSTEM_PROMPT
    assert "observed behavior" in SYSTEM_PROMPT
    assert "governed-change test" in SYSTEM_PROMPT
    assert "material-qualifier test" in SYSTEM_PROMPT
    assert "independently usable branches" in SYSTEM_PROMPT
    assert "generic lifecycle context" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "prompt",
    [
        SYSTEM_PROMPT,
        EXPLICIT_VERIFIER_SYSTEM_PROMPT,
        REDUCE_SYSTEM_PROMPT,
        FALSIFICATION_SYSTEM_PROMPT,
        VERIFIER_SYSTEM_PROMPT,
    ],
)
def test_scope_and_qualifier_prompts_do_not_embed_audited_repository_cases(
    prompt: str,
) -> None:
    lowered = " ".join(prompt.casefold().split())
    assert "material-qualifier test" in lowered
    assert "section label or provenance context" in lowered
    assert "generic lifecycle context" in lowered
    for leaked_case in ("pluggy", "marshmallow", "downstream", "--list"):
        assert leaked_case not in lowered


def test_ini_section_only_evidence_is_repaired_before_grounding() -> None:
    content = "[pytest]\nminversion=8.0\n"
    encoded = content.encode("utf-8")
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    batch = batch_units([unit], max_batch_chars=2_000, max_units_per_batch=1)[0]
    section_span, assignment_span = unit.spans

    claim = {
        "rule": "pytest minimum version is 8.0.",
        "claim_kind": "enforced_configuration",
        "normative_strength": "observed",
        "applies_to": ["**"],
        "condition": "pytest",
        "evidence": [
            {
                "unit_id": unit.unit_id,
                "span_start_id": section_span.span_id,
                "span_end_id": section_span.span_id,
            },
            {
                "unit_id": unit.unit_id,
                "span_start_id": assignment_span.span_id,
                "span_end_id": assignment_span.span_id,
            },
        ],
    }
    invalid = {
        "results": [
            {
                "unit_id": unit.unit_id,
                "classification": "contains_claims",
                "reason_code": "configuration_fact",
                "claims": [claim],
            }
        ]
    }
    repaired = json.loads(json.dumps(invalid))
    repaired["results"][0]["claims"][0]["evidence"] = [claim["evidence"][1]]

    settings = config_for_tests().llm.model_copy(
        update={"schema_repairs": 1, "provider_retries": 0}
    )
    outcome = extract_batch(
        batch,
        SequenceClient([json.dumps(invalid), json.dumps(repaired)]),
        sources_by_path={source.path: source},
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert outcome.inference_record.schema_repairs == 1
    assert [item.repair for item in outcome.attempts] == [False, True]
    assert len(outcome.results[0].claims[0].evidence) == 1


def test_cross_assignment_config_evidence_is_repaired_before_success_cache() -> None:
    content = (
        "[testenv:release]\n"
        "description = perform a release\n"
        "commands = python scripts/release.py {posargs}\n"
    )
    encoded = content.encode("utf-8")
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        content=content,
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    batch = batch_units([unit], max_batch_chars=2_000, max_units_per_batch=1)[0]
    base_claim = {
        "rule": "The release environment runs the repository release script.",
        "claim_kind": "command_recipe",
        "normative_strength": "observed",
        "applies_to": ["**"],
        "condition": "when performing a release",
        "evidence": [
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[1].span_id,
                "span_end_id": unit.spans[2].span_id,
            }
        ],
    }
    invalid = {
        "results": [
            {
                "unit_id": unit.unit_id,
                "classification": "contains_claims",
                "reason_code": "configuration_fact",
                "claims": [base_claim],
            }
        ]
    }
    repaired = json.loads(json.dumps(invalid))
    repaired["results"][0]["claims"][0]["evidence"][0].update(
        {
            "span_start_id": unit.spans[2].span_id,
            "span_end_id": unit.spans[2].span_id,
        }
    )
    settings = config_for_tests().llm.model_copy(
        update={"schema_repairs": 1, "provider_retries": 0}
    )

    outcome = extract_batch(
        batch,
        SequenceClient([json.dumps(invalid), json.dumps(repaired)]),
        sources_by_path={source.path: source},
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert outcome.inference_record.schema_repairs == 1
    assert any(
        "exactly one active assignment" in error
        for error in outcome.attempts[0].validation_errors
    )


def test_truncated_ini_continuation_is_host_validated_before_success_cache() -> None:
    content = (
        "[tox]\n"
        "isolated_build = true\n"
        "\n"
        "[testenv]\n"
        "commands=\n"
        "  {env:_PLUGGY_TOX_CMD:pytest} {posargs}\n"
        "  coverage: coverage combine\n"
        "  coverage: coverage report -m\n"
        "  coverage: coverage xml\n"
    )
    encoded = content.encode("utf-8")
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="Pluggy regression",
        referenced_by=(),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        content=content,
    )
    unit = next(
        item
        for item in chunk_sources([source], max_unit_chars=1_000)
        if item.line_start == 4
    )
    batch = batch_units([unit], max_batch_chars=2_000, max_units_per_batch=1)[0]
    claim = {
        "rule": "Pytest with positional arguments is the default test command.",
        "claim_kind": "command_recipe",
        "normative_strength": "observed",
        "applies_to": ["tests/**"],
        "condition": "when running the default tox test environment",
        "evidence": [
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[1].span_id,
                "span_end_id": unit.spans[2].span_id,
            }
        ],
    }
    invalid = {
        "results": [
            {
                "unit_id": unit.unit_id,
                "classification": "contains_claims",
                "reason_code": "command_fact",
                "claims": [claim],
            }
        ]
    }
    settings = config_for_tests().llm.model_copy(
        update={"schema_repairs": 1, "provider_retries": 0}
    )

    outcome = extract_batch(
        batch,
        SequenceClient([json.dumps(invalid)]),
        sources_by_path={source.path: source},
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert [item.repair for item in outcome.attempts] == [False]
    assert outcome.attempts[0].validation_errors == []
    # The immutable model response remains auditable. Full-source grounding
    # deterministically expands this range when materializing typed evidence.
    assert outcome.results[0].claims[0].evidence[0].span_end_id == (
        unit.spans[2].span_id
    )


def test_every_live_prompt_declares_json_transport_contract() -> None:
    prompts = (
        SYSTEM_PROMPT,
        MAP_SYSTEM_PROMPT,
        REDUCE_SYSTEM_PROMPT,
        CLASSIFY_SYSTEM_PROMPT,
        FALSIFICATION_SYSTEM_PROMPT,
        GROUP_SYSTEM_PROMPT,
        TEMPORAL_SYSTEM_PROMPT,
        VERIFIER_SYSTEM_PROMPT,
    )
    # The explicit semantic verifier has its own contract test in its module.
    assert len(prompts) == len(LIVE_STAGES - {"explicit_semantic_verify"})
    assert all("json" in prompt.casefold() for prompt in prompts)


def test_response_schema_is_attached_to_every_live_stage(tmp_path: Path) -> None:
    class ContractAssertingClient:
        def __init__(self) -> None:
            self.mock = MockLLMClient()
            self.stages: set[str] = set()

        def complete(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> ProviderResponse:
            payload = json.loads(user_prompt)
            schema = payload["response_json_schema"]
            assert schema["type"] == "object"
            request = payload.get("original_request", payload)
            self.stages.add(
                request.get("stage", "explicit_extract")
            )
            return self.mock.complete(system_prompt, user_prompt)

    repository = copied_fixture(tmp_path)
    client = ContractAssertingClient()
    result = build_fixture(repository, tmp_path / "run", client=client)
    assert result.build_report["complete"]
    assert client.stages == LIVE_STAGES - {"temporal_evidence_classify"}


def test_response_schema_attachment_is_canonical_and_collision_safe() -> None:
    first = attach_response_json_schema(
        '{"stage":"example"}',
        InferenceRecord,
    )
    second = attach_response_json_schema(
        '{ "stage": "example" }',
        InferenceRecord,
    )
    assert first == second
    assert "response_json_schema" in json.loads(first)
    with pytest.raises(ValueError, match="already defines"):
        attach_response_json_schema(
            '{"response_json_schema":{}}',
            InferenceRecord,
        )


def test_request_identity_and_record_digests_are_stable(tmp_path: Path) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    first = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=config.llm.model_copy(update={"provider_retries": 0}),
    )
    second = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=config.llm.model_copy(update={"provider_retries": 0}),
    )
    assert first.inference_record == second.inference_record
    assert first.inference_record.request_id.startswith("INFER-")
    assert first.inference_record.terminal_status == "success"
    assert first.inference_record.attempts[0].attempt_digest
    assert first.inference_record.usage.usage_digest

    tampered = first.inference_record.model_dump(mode="json")
    tampered["record_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(tampered)

    for future_stage in FUTURE_STAGES:
        future = first.inference_record.model_dump(mode="json")
        future["stage"] = future_stage
        future["record_digest"] = payload_digest(
            {
                key: value
                for key, value in future.items()
                if key != "record_digest"
            }
        )
        with pytest.raises(ValidationError, match="not implemented"):
            InferenceRecord.model_validate(future)

    unknown = first.inference_record.model_dump(mode="json")
    unknown["unexpected"] = True
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(unknown)

    coerced = first.inference_record.model_dump(mode="json")
    coerced["request_ordinal"] = "0"
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(coerced)

    attempt_tamper = first.inference_record.model_dump(mode="json")
    attempt_tamper["attempts"][0]["attempt_digest"] = "0" * 64
    attempt_tamper["record_digest"] = payload_digest(
        {
            key: value
            for key, value in attempt_tamper.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="attempt digest"):
        InferenceRecord.model_validate(attempt_tamper)

    repair_tamper = first.inference_record.model_dump(mode="json")
    repair_tamper["attempts"][0]["schema_attempt"] = 1
    repair_tamper["attempts"][0]["repair"] = False
    repair_tamper["attempts"][0]["attempt_digest"] = payload_digest(
        {
            key: value
            for key, value in repair_tamper["attempts"][0].items()
            if key != "attempt_digest"
        }
    )
    repair_tamper["record_digest"] = payload_digest(
        {
            key: value
            for key, value in repair_tamper.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="repair flag"):
        InferenceRecord.model_validate(repair_tamper)

    retry_bound_tamper = first.inference_record.model_dump(mode="json")
    retry_bound_tamper["attempts"][0]["provider_try"] = 1
    retry_bound_tamper["attempts"][0]["attempt_digest"] = payload_digest(
        {
            key: value
            for key, value in retry_bound_tamper["attempts"][0].items()
            if key != "attempt_digest"
        }
    )
    retry_bound_tamper["record_digest"] = payload_digest(
        {
            key: value
            for key, value in retry_bound_tamper.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="declared repair/retry bounds"):
        InferenceRecord.model_validate(retry_bound_tamper)

    parsed_then_retry = first.inference_record.model_dump(mode="json")
    second_attempt = json.loads(json.dumps(parsed_then_retry["attempts"][0]))
    second_attempt["attempt_ordinal"] = 1
    second_attempt["output_limit_retry"] = 1
    second_attempt["attempt_digest"] = payload_digest(
        {
            key: value
            for key, value in second_attempt.items()
            if key != "attempt_digest"
        }
    )
    parsed_then_retry["attempts"].append(second_attempt)
    usage = parsed_then_retry["usage"]
    usage["calls"] = 2
    usage["input_tokens"] *= 2
    usage["output_tokens"] *= 2
    if usage["reasoning_tokens"] is not None:
        usage["reasoning_tokens"] *= 2
    usage["latency_ms"] *= 2
    usage["usage_digest"] = payload_digest(
        {key: value for key, value in usage.items() if key != "usage_digest"}
    )
    parsed_then_retry["record_digest"] = payload_digest(
        {
            key: value
            for key, value in parsed_then_retry.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="trajectory is not canonical"):
        InferenceRecord.model_validate(parsed_then_retry)

    wrong_kind = first.inference_record.model_dump(mode="json")
    wrong_kind["input_manifest"][0]["kind"] = "semantic_packet"
    wrong_kind["input_manifest_digest"] = payload_digest(
        {"inputs": wrong_kind["input_manifest"]}
    )
    wrong_inputs = [
        InferenceInputRef.model_validate(item)
        for item in wrong_kind["input_manifest"]
    ]
    wrong_kind["request_id"] = request_identity(
        stage="explicit_extract",
        protocol_version=wrong_kind["protocol_version"],
        mode=wrong_kind["mode"],
        inputs=wrong_inputs,
        request_ordinal=wrong_kind["request_ordinal"],
    )
    wrong_kind["record_digest"] = payload_digest(
        {
            key: value
            for key, value in wrong_kind.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="input kind"):
        InferenceRecord.model_validate(wrong_kind)


def test_cache_hit_rejects_coordinated_retry_bound_mismatch(tmp_path: Path) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "provider_retries": 0,
            "schema_repairs": 0,
            "output_limit_retries": 1,
        }
    )
    first = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    tampered = first.inference_record.model_dump(mode="json")
    tampered["output_limit_retries"] = 2
    tampered["record_digest"] = payload_digest(
        {
            key: value
            for key, value in tampered.items()
            if key != "record_digest"
        }
    )
    assert InferenceRecord.model_validate(tampered).output_limit_retries == 2

    class CoordinatedCache:
        def read(self, key: str):
            del key
            return tampered

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            return MockLLMClient().complete(system_prompt, user_prompt)

    client = CountingClient()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        budget=InferenceBudget(settings, cache=CoordinatedCache()),
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert client.calls == 1
    assert outcome.inference_record.output_limit_retries == 1


def test_real_profile_cache_rejects_resealed_wrong_effective_model(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    configured_model = "fixed-real-model"
    settings = config.llm.model_copy(
        update={
            "model": configured_model,
            "accepted_effective_model_ids": [configured_model],
            "reasoning_wire_policy": "deepseek_thinking_type",
            "provider_retries": 0,
            "schema_repairs": 0,
        }
    )
    valid_content = MockLLMClient().complete(
        SYSTEM_PROMPT,
        attach_response_json_schema(
            build_batch_prompt(batch),
            BatchExtractionResponse,
        ),
    ).content

    class CorrectIdentityClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            self.calls += 1
            return ProviderResponse(
                content=valid_content,
                model=configured_model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )

    first_client = CorrectIdentityClient()
    first = extract_batch(
        batch,
        first_client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert first.succeeded
    tampered = first.inference_record.model_dump(mode="json")
    attempt = tampered["attempts"][0]
    attempt["model"] = "different-effective-model"
    attempt["attempt_digest"] = payload_digest(
        {
            key: value
            for key, value in attempt.items()
            if key != "attempt_digest"
        }
    )
    tampered["record_digest"] = payload_digest(
        {
            key: value
            for key, value in tampered.items()
            if key != "record_digest"
        }
    )
    assert InferenceRecord.model_validate(tampered)

    class CoordinatedCache:
        def read(self, key: str):
            del key
            return tampered

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    replay_client = CorrectIdentityClient()
    replay = extract_batch(
        batch,
        replay_client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        budget=InferenceBudget(settings, cache=CoordinatedCache()),
        llm_settings=settings,
    )

    assert replay.succeeded
    assert replay_client.calls == 1
    assert replay.inference_record.attempts[-1].model == configured_model


def test_default_mock_completes_all_live_stages(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")
    assert {item.stage for item in result.inference_records} == (
        LIVE_STAGES - {"temporal_evidence_classify"}
    )
    assert all(
        item.terminal_status == "success" for item in result.inference_records
    )


def test_reordered_explicit_results_are_canonicalized(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path, two=True)
    valid = json.loads(
        MockLLMClient().complete("", json.dumps({
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch.batch_id,
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "path": unit.path,
                    "source_kind": unit.source_kind,
                    "discovery_origin": unit.discovery_origin,
                    "default_scope": unit.default_scope,
                    "annotated_text": unit.content,
                    "spans": [span.model_dump(mode="json") for span in unit.spans],
                }
                for unit in batch.units
            ],
        })).content
    )
    reordered = extract_batch(
        batch,
        SequenceClient(
            [json.dumps({"results": list(reversed(valid["results"]))})]
        ),
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        request_ordinal=0,
        llm_settings=config.llm.model_copy(
            update={"schema_repairs": 0, "provider_retries": 0}
        ),
    )
    assert reordered.succeeded
    assert [item.unit_id for item in reordered.results] == [
        item.unit_id for item in batch.units
    ]


def test_missing_and_ungrounded_explicit_results_fail_strictly(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path, two=True)
    valid = json.loads(
        MockLLMClient().complete("", json.dumps({
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch.batch_id,
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "path": unit.path,
                    "source_kind": unit.source_kind,
                    "discovery_origin": unit.discovery_origin,
                    "default_scope": unit.default_scope,
                    "annotated_text": unit.content,
                    "spans": [span.model_dump(mode="json") for span in unit.spans],
                }
                for unit in batch.units
            ],
        })).content
    )
    invalid_values = [{"results": valid["results"][:-1]}]
    ungrounded = json.loads(json.dumps(valid))
    claim_result = next(
        (item for item in ungrounded["results"] if item["claims"]),
        None,
    )
    if claim_result is not None:
        claim_result["claims"][0]["evidence"][0]["span_start_id"] = (
            "SPAN-" + "0" * 24
        )
        invalid_values.append(ungrounded)
    for index, invalid in enumerate(invalid_values):
        outcome = extract_batch(
            batch,
            SequenceClient([json.dumps(invalid)]),
            sources_by_path=sources_by_path,
            model_name=config.llm.model,
            schema_repairs=0,
            provider_retries=0,
            provider_retry_delay_seconds=0,
            request_ordinal=index,
            llm_settings=config.llm.model_copy(
                update={"schema_repairs": 0, "provider_retries": 0}
            ),
        )
        assert not outcome.succeeded
        assert outcome.inference_record.terminal_status == "failure"


def test_invalid_raw_output_is_not_serialized_and_call_budget_is_visible(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 1,
            "provider_retries": 0,
            "max_total_calls": 1,
        }
    )
    raw = '{"secret":"api_key=do-not-store"}'
    outcome = extract_batch(
        batch,
        SequenceClient([raw]),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    serialized = json.dumps(outcome.inference_record.model_dump(mode="json"))
    assert "do-not-store" not in serialized
    assert "max_total_calls" in outcome.inference_record.incomplete_reasons
    assert outcome.inference_record.terminal_status == "failure"


def test_three_bounded_schema_repairs_can_recover_a_logical_request(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={"schema_repairs": 3, "provider_retries": 0}
    )
    prompt = attach_response_json_schema(
        build_batch_prompt(batch),
        BatchExtractionResponse,
    )
    valid = MockLLMClient().complete(SYSTEM_PROMPT, prompt).content
    outcome = extract_batch(
        batch,
        SequenceClient(["{}", "{}", "{}", valid]),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=3,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert outcome.succeeded
    assert [item.schema_attempt for item in outcome.attempts] == [0, 1, 2, 3]
    assert [item.repair for item in outcome.attempts] == [
        False,
        True,
        True,
        True,
    ]


def test_one_complete_fenced_json_object_is_accepted(tmp_path: Path) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    valid = MockLLMClient().complete(
        SYSTEM_PROMPT,
        attach_response_json_schema(
            build_batch_prompt(batch),
            BatchExtractionResponse,
        ),
    ).content
    outcome = extract_batch(
        batch,
        SequenceClient([f"```json\n{valid}\n```"]),
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=config.llm.model_copy(
            update={"schema_repairs": 0, "provider_retries": 0}
        ),
    )
    assert outcome.succeeded
    assert len(outcome.attempts) == 1


def test_retry_backoff_is_exponential_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    sleeps: list[float] = []

    class RetryTwice:
        def __init__(self) -> None:
            self.calls = 0
            self.mock = MockLLMClient()

        def complete(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if self.calls <= 2:
                raise LLMProviderError("temporary", retryable=True)
            return self.mock.complete(system_prompt, user_prompt)

    client = RetryTwice()
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 0,
            "provider_retries": 2,
            "provider_retry_delay_seconds": 2,
            "provider_retry_backoff_multiplier": 2,
            "provider_retry_max_delay_seconds": 5,
            "provider_retry_jitter_seconds": 0,
        }
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.inference.time.sleep",
        sleeps.append,
    )
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=2,
        provider_retry_delay_seconds=2,
        llm_settings=settings,
    )
    assert outcome.succeeded
    assert client.calls == 3
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    ("finish_reason", "expected_error", "expected_calls"),
    [
        ("length", "provider output was truncated", 2),
        ("max_tokens", "provider output was truncated", 2),
        (
            "context_length_exceeded",
            "provider context window was exceeded",
            1,
        ),
    ],
)
def test_output_limit_finish_reason_has_only_bounded_non_schema_retry(
    tmp_path: Path,
    finish_reason: str,
    expected_error: str,
    expected_calls: int,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    valid = MockLLMClient().complete(
        SYSTEM_PROMPT,
        attach_response_json_schema(
            build_batch_prompt(batch),
            BatchExtractionResponse,
        ),
    ).content

    class TruncatedThenValid:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            self.calls += 1
            return ProviderResponse(
                content=valid,
                model="truncation-fake",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                finish_reason=finish_reason,
            )

    client = TruncatedThenValid()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=config.llm.model_copy(
            update={"schema_repairs": 1, "provider_retries": 0}
        ),
    )
    assert not outcome.succeeded
    assert client.calls == expected_calls
    assert len(outcome.attempts) == expected_calls
    assert all(item.finish_reason == finish_reason for item in outcome.attempts)
    assert all(item.validation_errors == [expected_error] for item in outcome.attempts)
    assert not any(item.repair for item in outcome.attempts)
    assert [item.output_limit_retry for item in outcome.attempts] == list(
        range(expected_calls)
    )
    assert outcome.inference_record.terminal_status == "failure"
    assert outcome.inference_record.structured_response is None


def test_output_limit_retry_repeats_exact_prompt_and_preserves_telemetry(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    valid = MockLLMClient().complete(
        SYSTEM_PROMPT,
        attach_response_json_schema(
            build_batch_prompt(batch),
            BatchExtractionResponse,
        ),
    ).content

    class LengthThenValid:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return ProviderResponse(
                    content="{}",
                    model="length-then-valid",
                    input_tokens=11,
                    output_tokens=8_000,
                    reasoning_tokens=7_900,
                    reasoning_content_present=True,
                    reasoning_content_bytes=31_600,
                    latency_ms=20,
                    finish_reason="length",
                )
            return ProviderResponse(
                content=valid,
                model="length-then-valid",
                input_tokens=11,
                output_tokens=100,
                reasoning_tokens=0,
                latency_ms=2,
                finish_reason="stop",
            )

    client = LengthThenValid()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=config.llm.model_copy(
            update={
                "thinking_mode": "enabled",
                "reasoning_effort": "high",
                "schema_repairs": 1,
                "provider_retries": 0,
                "output_limit_retries": 1,
            }
        ),
    )

    assert outcome.succeeded
    assert outcome.inference_record.model_parameters.thinking_mode == (
        "enabled"
    )
    assert client.prompts[0] == client.prompts[1]
    assert [item.output_limit_retry for item in outcome.attempts] == [0, 1]
    assert [item.schema_attempt for item in outcome.attempts] == [0, 0]
    assert [item.repair for item in outcome.attempts] == [False, False]
    assert outcome.inference_record.usage.calls == 2
    assert outcome.inference_record.usage.output_tokens == 8_100
    assert outcome.inference_record.usage.reasoning_tokens == 7_900
    first = outcome.inference_record.attempts[0]
    assert first.reasoning_content_present
    assert first.reasoning_content_bytes == 31_600
    assert first.response_content_bytes == 2
    relabeled = outcome.inference_record.model_dump(mode="json")
    relabeled["model_parameters"]["thinking_mode"] = "disabled"
    relabeled["model_parameters"]["reasoning_effort"] = None
    relabeled["record_digest"] = payload_digest(
        {
            key: value
            for key, value in relabeled.items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValidationError, match="profile violation"):
        InferenceRecord.model_validate(relabeled)


@pytest.mark.parametrize(
    (
        "reasoning_tokens",
        "reasoning_content_present",
        "reasoning_content_bytes",
        "reason",
    ),
    [
        (
            1,
            False,
            0,
            DISABLED_THINKING_REASONING_TOKENS_VIOLATION,
        ),
        (
            0,
            True,
            len("trace"),
            DISABLED_THINKING_REASONING_CONTENT_VIOLATION,
        ),
    ],
)
def test_disabled_thinking_profile_violation_is_terminal_and_not_cached(
    tmp_path: Path,
    reasoning_tokens: int,
    reasoning_content_present: bool,
    reasoning_content_bytes: int,
    reason: str,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    valid = MockLLMClient().complete(
        SYSTEM_PROMPT,
        attach_response_json_schema(
            build_batch_prompt(batch),
            BatchExtractionResponse,
        ),
    ).content

    class ViolatingClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
            del system_prompt, user_prompt
            self.calls += 1
            return ProviderResponse(
                content=valid,
                model="violating-fake",
                input_tokens=3,
                output_tokens=10,
                reasoning_tokens=reasoning_tokens,
                reasoning_content_present=reasoning_content_present,
                reasoning_content_bytes=reasoning_content_bytes,
                latency_ms=1,
                finish_reason="stop",
            )

    class RecordingCache:
        def __init__(self) -> None:
            self.writes: list[tuple[str, dict[str, object]]] = []

        def read(self, key: str):
            del key
            return None

        def write(self, key: str, record: dict[str, object]) -> None:
            self.writes.append((key, record))

    settings = config.llm.model_copy(
        update={
            "thinking_mode": "disabled",
            "schema_repairs": 1,
            "provider_retries": 1,
            "output_limit_retries": 1,
            "provider_retry_delay_seconds": 0,
        }
    )
    client = ViolatingClient()
    cache = RecordingCache()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=0,
        budget=InferenceBudget(settings, cache=cache),
        llm_settings=settings,
    )

    assert not outcome.succeeded
    assert client.calls == 1
    assert outcome.inference_record.terminal_status == "failure"
    assert outcome.inference_record.structured_response is None
    assert outcome.inference_record.incomplete_reasons == [reason]
    assert len(outcome.attempts) == 1
    assert not outcome.attempts[0].parsed
    assert not outcome.attempts[0].retryable
    assert outcome.attempts[0].validation_errors == [reason]
    assert cache.writes == []


def test_disabled_thinking_without_reasoning_telemetry_succeeds(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "thinking_mode": "disabled",
            "schema_repairs": 0,
            "provider_retries": 0,
        }
    )
    outcome = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert outcome.succeeded
    assert outcome.inference_record.terminal_status == "success"
    assert outcome.inference_record.attempts[-1].reasoning_tokens == 0


def test_output_limit_retry_respects_call_budget(tmp_path: Path) -> None:
    batch, config, sources_by_path = _batch(tmp_path)

    class AlwaysLength:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            self.calls += 1
            return ProviderResponse(
                content="{}",
                model="always-length",
                input_tokens=1,
                output_tokens=10,
                reasoning_tokens=None,
                latency_ms=1,
                finish_reason="length",
            )

    settings = config.llm.model_copy(
        update={
            "schema_repairs": 0,
            "provider_retries": 0,
            "output_limit_retries": 1,
            "max_total_calls": 1,
        }
    )
    client = AlwaysLength()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )

    assert not outcome.succeeded
    assert client.calls == 1
    assert len(outcome.attempts) == 1
    assert outcome.inference_record.usage.reasoning_tokens is None
    assert "max_total_calls" in outcome.inference_record.incomplete_reasons


def test_extraction_resolves_hard_stage_bounds(tmp_path: Path) -> None:
    batch, config, sources_by_path = _batch(tmp_path)

    class SettingsCapture:
        def __init__(self) -> None:
            self.settings = []
            self.mock = MockLLMClient()

        def complete_with_settings(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings,
        ):
            self.settings.append(settings)
            return self.mock.complete(system_prompt, user_prompt)

    configured = config.llm.model_copy(
        update={"max_output_tokens": 65_536, "timeout_seconds": 800}
    )
    client = SettingsCapture()
    outcome = extract_batch(
        batch,
        client,
        sources_by_path=sources_by_path,
        model_name=configured.model,
        schema_repairs=configured.schema_repairs,
        provider_retries=configured.provider_retries,
        provider_retry_delay_seconds=configured.provider_retry_delay_seconds,
        llm_settings=configured,
    )

    assert outcome.succeeded
    assert len(client.settings) == 1
    assert client.settings[0].max_output_tokens == MAX_EXTRACTION_OUTPUT_TOKENS
    assert client.settings[0].timeout_seconds == MAX_EXTRACTION_TIMEOUT_SECONDS
    assert outcome.inference_record.model_parameters.max_output_tokens == (
        MAX_EXTRACTION_OUTPUT_TOKENS
    )
    assert MAX_EXTRACTION_RESPONSE_BYTES == 128 * 1024


@pytest.mark.parametrize(
    (
        "wire_policy",
        "thinking_mode",
        "reasoning_effort",
        "reasoning_tokens",
        "reasoning_content",
    ),
    [
        ("deepseek_thinking_type", "disabled", None, 0, None),
        ("qwen_enable_thinking", "disabled", None, 0, None),
        ("qwen_enable_thinking", "enabled", "medium", 3, "think"),
        ("deepseek_thinking_type", "enabled", "high", None, None),
        ("deepseek_thinking_type", "enabled", "high", 3, "think"),
    ],
)
def test_openai_transport_binds_thinking_and_reasoning_telemetry(
    wire_policy: str,
    thinking_mode: str,
    reasoning_effort: str | None,
    reasoning_tokens: int | None,
    reasoning_content: str | None,
) -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "thinking_mode": thinking_mode,
            "reasoning_effort": reasoning_effort,
            "reasoning_wire_policy": wire_policy,
        }
    )
    client = OpenAICompatibleClient(settings=settings, api_key="test-key")
    client._client.close()

    class StubResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        def raise_for_status(self) -> None:
            return None

        def iter_raw(self, *, chunk_size: int):
            del chunk_size
            usage = {"prompt_tokens": 3, "completion_tokens": 5}
            message = {"content": '{"ok":true}'}
            if reasoning_tokens is not None:
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": reasoning_tokens
                }
            if reasoning_content is not None:
                message["reasoning_content"] = reasoning_content
            yield json.dumps(
                {
                    "choices": [
                        {"message": message, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                    "model": "transport-fixture",
                }
            ).encode("utf-8")

    class StubHTTPClient:
        def __init__(self) -> None:
            self.payload = None

        def stream(self, *args, **kwargs):
            del args
            self.payload = kwargs["json"]
            return StubResponse()

    stub = StubHTTPClient()
    client._client = stub  # type: ignore[assignment]
    response = client.complete("system", "user")

    if wire_policy == "qwen_enable_thinking":
        assert "thinking" not in stub.payload
        assert stub.payload["enable_thinking"] is (thinking_mode == "enabled")
        assert stub.payload["preserve_thinking"] is True
    else:
        assert stub.payload["thinking"] == {"type": thinking_mode}
        assert "enable_thinking" not in stub.payload
    if thinking_mode == "disabled":
        assert "reasoning_effort" not in stub.payload
    else:
        assert stub.payload["reasoning_effort"] == reasoning_effort
    assert "reasoning" not in stub.payload
    assert response.reasoning_tokens == reasoning_tokens
    assert response.reasoning_content_present is (reasoning_content is not None)
    assert response.reasoning_content_bytes == len(
        (reasoning_content or "").encode("utf-8")
    )
    assert response.response_content_bytes == len(response.content.encode("utf-8"))


def test_openai_transport_preserves_null_content_length_stop_for_output_retry() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "provider_retries": 0,
            "output_limit_retries": 1,
            "reasoning_wire_policy": "deepseek_thinking_type",
        }
    )
    client = OpenAICompatibleClient(settings=settings, api_key="test-key")
    client._client.close()

    class StubResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback

        def raise_for_status(self) -> None:
            return None

        def iter_raw(self, *, chunk_size: int):
            del chunk_size
            yield json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning_content": "bounded hidden trace",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 9,
                        "completion_tokens_details": {"reasoning_tokens": 9},
                    },
                    "model": "transport-fixture",
                }
            ).encode("utf-8")

    class StubHTTPClient:
        def stream(self, *args, **kwargs):
            del args, kwargs
            return StubResponse()

    client._client = StubHTTPClient()  # type: ignore[assignment]
    response = client.complete("system", "user")

    assert response.content == ""
    assert response.finish_reason == "length"
    assert response.reasoning_tokens == 9
    assert response.reasoning_content_present
    assert response.reasoning_content_bytes == len("bounded hidden trace")


def test_explicit_semantic_verify_stage_uses_only_grounded_candidates() -> None:
    assert "explicit_semantic_verify" in LIVE_STAGES
    assert STAGE_INPUT_KINDS["explicit_semantic_verify"] == frozenset(
        {"explicit_grounded_candidate"}
    )
    reference = InferenceInputRef(
        kind="explicit_grounded_candidate",
        stable_id="EXPL-example",
        digest="0" * 64,
        snapshot_digest="sha256:" + "1" * 64,
    )
    assert reference.kind == "explicit_grounded_candidate"


def test_reported_token_and_artifact_limits_stop_remaining_work(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)

    class TokenReportingClient:
        def __init__(self) -> None:
            self.mock = MockLLMClient()

        def complete(self, system_prompt: str, user_prompt: str):
            response = self.mock.complete(system_prompt, user_prompt)
            return response.model_copy(update={"input_tokens": 2})

    token_settings = config.llm.model_copy(
        update={
            "provider_retries": 0,
            "schema_repairs": 0,
            "max_total_input_tokens": 1,
        }
    )
    budget = InferenceBudget(token_settings)
    first = extract_batch(
        batch,
        TokenReportingClient(),
        sources_by_path=sources_by_path,
        model_name=token_settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        request_ordinal=0,
        budget=budget,
        llm_settings=token_settings,
    )
    second = extract_batch(
        batch,
        TokenReportingClient(),
        sources_by_path=sources_by_path,
        model_name=token_settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        request_ordinal=1,
        budget=budget,
        llm_settings=token_settings,
    )
    assert first.succeeded
    assert "max_total_input_tokens" in budget.exhausted_reasons
    assert not second.succeeded
    assert "max_total_input_tokens" in second.inference_record.incomplete_reasons

    artifact_settings = config.llm.model_copy(
        update={
            "provider_retries": 0,
            "schema_repairs": 0,
            "max_inference_artifact_bytes": 1_000,
        }
    )
    artifact = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=artifact_settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=artifact_settings,
    )
    assert not artifact.succeeded
    assert artifact.inference_record.structured_response is None
    assert (
        "max_inference_artifact_bytes"
        in artifact.inference_record.incomplete_reasons
    )


def test_inference_budget_preserves_validated_prior_consumption() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "max_total_calls": 3,
            "max_total_input_tokens": 5,
            "max_total_output_tokens": 7,
        }
    )
    budget = InferenceBudget(
        settings,
        initial_calls=2,
        initial_input_tokens=4,
        initial_output_tokens=6,
    )
    assert budget.reserve_call() is None
    assert budget.calls == 3
    assert budget.reserve_call() == "max_total_calls"
    assert budget.input_tokens == 4
    assert budget.output_tokens == 6

    assert (
        InferenceBudget(
            settings,
            initial_input_tokens=settings.max_total_input_tokens,
        ).reserve_call()
        == "max_total_input_tokens"
    )
    assert (
        InferenceBudget(
            settings,
            initial_output_tokens=settings.max_total_output_tokens,
        ).reserve_call()
        == "max_total_output_tokens"
    )

    for field in (
        "initial_calls",
        "initial_input_tokens",
        "initial_output_tokens",
    ):
        with pytest.raises(ValueError, match="non-negative integer"):
            InferenceBudget(settings, **{field: -1})
        with pytest.raises(ValueError, match="non-negative integer"):
            InferenceBudget(settings, **{field: True})


@pytest.mark.parametrize("config_name", ["hardened", "openai_agents_python"])
@pytest.mark.parametrize("concurrency", [12, 24])
def test_active_retry_budget_admits_full_temporal_waves(
    config_name: str, concurrency: int,
) -> None:
    config = load_build_config(
        Path(__file__).resolve().parents[1] / "configs" / f"{config_name}.yaml"
    )
    settings = config.llm
    assert settings.max_concurrency == 24
    assert settings.provider_retries == 1
    assert settings.schema_repairs == 2
    assert settings.output_limit_retries == 1
    assert settings.max_total_calls == 15_000
    assert max_provider_calls_per_request(settings) == 12
    tree_requests = 2 * config.unified.max_temporal_batch_items - 1
    old_budget = InferenceBudget(
        settings.model_copy(update={"provider_retries": 4}), initial_calls=6_000
    )
    assert old_budget.reserve_call_permits([tree_requests] * concurrency) is None

    budget = InferenceBudget(settings, initial_calls=6_000)
    barrier = threading.Barrier(concurrency)

    def invoke(item: int) -> int:
        assert budget.reserve_call() is None
        barrier.wait(timeout=10)
        return item

    result = run_budgeted_stage_calls(
        list(range(concurrency * 2)),
        invoke,
        stage="temporal_evidence_classify",
        budget=budget,
        max_concurrency=concurrency,
        thread_name_prefix="active-temporal-budget",
        max_logical_requests_per_item=lambda _: tree_requests,
    )
    assert result.outcomes == tuple(range(concurrency * 2))
    assert result.omitted == 0
    assert budget.calls == 6_000 + concurrency * 2
    # Reservations are released; they are not charged as performed calls.
    required = concurrency * tree_requests * 12
    exhausted = InferenceBudget(
        settings, initial_calls=settings.max_total_calls - required + 1
    )
    assert exhausted.reserve_call_permits([tree_requests] * concurrency) is None


def test_budgeted_stage_compactly_omits_twenty_thousand_request_suffix() -> None:
    settings = config_for_tests().llm.model_copy(
        update={"max_total_calls": 2, "max_concurrency": 1}
    )
    budget = InferenceBudget(settings, initial_calls=1)
    invoked: list[int] = []

    def invoke(item: int) -> int:
        invoked.append(item)
        assert budget.reserve_call() is None
        return item

    result = run_budgeted_stage_calls(
        list(range(20_000)),
        invoke,
        stage="population_semantic_classify",
        budget=budget,
        max_concurrency=1,
        thread_name_prefix="budget-stress",
    )

    assert invoked == [0]
    assert result.outcomes == (0,)
    assert result.planned == 20_000
    assert result.omitted == 19_999
    assert result.exhaustion_reason == "max_total_calls"
    assert budget.stage_omissions == {
        "population_semantic_classify": (
            19_999,
            {"max_total_calls": 19_999},
        )
    }
    assert result.planned == len(result.outcomes) + result.omitted


def test_parallel_wave_is_atomic_across_token_crossing_and_keeps_plan_order() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 3,
            "max_total_input_tokens": 1,
            "max_concurrency": 2,
        }
    )
    budget = InferenceBudget(settings)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def invoke(item: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            assert budget.reserve_call() is None
            barrier.wait(timeout=5)
            # Finish in the opposite order from the canonical input plan.
            if item == 0:
                time.sleep(0.04)
            budget.record_usage(input_tokens=2, output_tokens=0)
            return item
        finally:
            with lock:
                active -= 1

    result = run_budgeted_stage_calls(
        [0, 1, 2],
        invoke,
        stage="unified_case_verify",
        budget=budget,
        max_concurrency=2,
        thread_name_prefix="token-crossing",
    )

    assert peak == 2
    assert result.outcomes == (0, 1)
    assert result.omitted == 1
    assert result.exhaustion_reason == "max_total_input_tokens"
    assert budget.calls == budget.logical_calls == 2


def test_exact_final_budget_usage_is_complete_until_a_later_boundary_query() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 1,
        }
    )
    budget = InferenceBudget(settings)

    assert budget.reserve_call() is None
    budget.record_usage(input_tokens=0, output_tokens=0)
    assert budget.exhausted_reasons == set()

    assert budget.terminal_exhaustion_reason() == "max_total_calls"
    assert budget.exhausted_reasons == {"max_total_calls"}


@pytest.mark.parametrize("reverse_finish", [False, True])
def test_parallel_final_token_overshoot_has_stable_single_reason(
    reverse_finish: bool,
) -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 2,
            "max_total_input_tokens": 5,
            "max_total_output_tokens": 5,
            "max_concurrency": 2,
        }
    )
    budget = InferenceBudget(settings)
    barrier = threading.Barrier(2)

    def invoke(item: int) -> int:
        assert budget.reserve_call() is None
        barrier.wait(timeout=5)
        if (item == 0) == reverse_finish:
            time.sleep(0.04)
        if item == 0:
            budget.record_usage(input_tokens=6, output_tokens=0)
        else:
            budget.record_usage(input_tokens=0, output_tokens=6)
        return item

    outcomes = run_call_permitted_stage_calls(
        [0, 1],
        invoke,
        budget=budget,
        max_concurrency=2,
        thread_name_prefix="stable-overrun",
    )

    assert outcomes == [0, 1]
    assert budget.exhausted_reasons == {"max_total_input_tokens"}


def test_call_permit_reservation_uses_full_retry_repair_trajectory() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 1,
            "output_limit_retries": 1,
            "provider_retries": 1,
            "max_total_calls": 15,
        }
    )
    budget = InferenceBudget(settings)

    assert max_provider_calls_per_request(settings) == 8
    assert budget.reserve_call_permits([1, 1]) is None
    permit = budget.reserve_call_permits([1])
    assert permit is not None
    budget.release_call_permits(permit)


def test_exceptional_permit_context_releases_all_unconsumed_capacity() -> None:
    settings = config_for_tests().llm.model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 2,
        }
    )
    budget = InferenceBudget(settings)
    permits = budget.reserve_call_permits([1, 1])
    assert permits is not None

    with pytest.raises(RuntimeError, match="worker failed"):
        with budget.activate_call_permit(permits[0]):
            raise RuntimeError("worker failed")
    budget.release_call_permits(permits)

    replacement = budget.reserve_call_permits([1, 1])
    assert replacement is not None
    budget.release_call_permits(replacement)


def test_live_and_cached_records_consume_equal_logical_permit_budget(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 4,
        }
    )
    first = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=0,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert first.succeeded

    class ExactSuccessCache:
        def read(self, key: str):
            del key
            return first.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class ExplodingClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise AssertionError("cached success reached the provider")

    live_budget = InferenceBudget(settings)
    live_permit = live_budget.reserve_call_permits([1])
    assert live_permit is not None
    with live_budget.activate_call_permit(live_permit[0]):
        live = extract_batch(
            batch,
            MockLLMClient(),
            sources_by_path=sources_by_path,
            model_name=settings.model,
            schema_repairs=0,
            provider_retries=0,
            provider_retry_delay_seconds=0,
            llm_settings=settings,
            budget=live_budget,
        )

    cache_budget = InferenceBudget(settings, cache=ExactSuccessCache())
    cache_permit = cache_budget.reserve_call_permits([1])
    assert cache_permit is not None
    with cache_budget.activate_call_permit(cache_permit[0]):
        cached = extract_batch(
            batch,
            ExplodingClient(),
            sources_by_path=sources_by_path,
            model_name=settings.model,
            schema_repairs=0,
            provider_retries=0,
            provider_retry_delay_seconds=0,
            llm_settings=settings,
            budget=cache_budget,
        )

    assert live.inference_record == cached.inference_record
    assert live_budget.calls == live_budget.logical_calls == 1
    assert cache_budget.calls == 0
    assert cache_budget.logical_calls == live_budget.logical_calls
    assert cache_budget.logical_input_tokens == live_budget.logical_input_tokens
    assert cache_budget.logical_output_tokens == live_budget.logical_output_tokens


def test_replay_record_cannot_exceed_its_reserved_call_trajectory(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    record_settings = config.llm.model_copy(
        update={
            "schema_repairs": 1,
            "output_limit_retries": 0,
            "provider_retries": 0,
        }
    )

    class InvalidClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            return ProviderResponse(
                content="{}",
                model=record_settings.model,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    failed = extract_batch(
        batch,
        InvalidClient(),
        sources_by_path=sources_by_path,
        model_name=record_settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=record_settings,
    )
    assert failed.inference_record.usage.calls == 2

    replay_settings = record_settings.model_copy(
        update={"schema_repairs": 0, "max_total_calls": 4}
    )
    budget = InferenceBudget(replay_settings)
    permit = budget.reserve_call_permits([1])
    assert permit is not None
    with pytest.raises(ValueError, match="exceeds reserved call permit"):
        with budget.activate_call_permit(permit[0]):
            budget.record_terminal(failed.inference_record, replay_hit=True)
    assert budget.logical_calls == 0
    assert budget.stage_progress == {}


def test_serial_oversized_success_cache_falls_back_to_live_remaining_calls(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 1,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 3,
        }
    )

    class RepairThenValidClient:
        def __init__(self) -> None:
            self.calls = 0
            self.mock = MockLLMClient()

        def complete(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if self.calls == 1:
                return ProviderResponse(
                    content="{}",
                    model=settings.model,
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1,
                )
            return self.mock.complete(system_prompt, user_prompt)

    original = extract_batch(
        batch,
        RepairThenValidClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert original.succeeded
    assert original.inference_record.usage.calls == 2

    class ExactSuccessCache:
        def read(self, key: str):
            del key
            return original.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    budget = InferenceBudget(
        settings,
        cache=ExactSuccessCache(),
        initial_calls=2,
    )
    replayed = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
        budget=budget,
    )

    assert replayed.succeeded
    assert replayed.inference_record.usage.calls == 1
    assert budget.cache_hits == 0
    assert budget.calls == budget.logical_calls == 3
    assert budget.terminal_exhaustion_reason() == "max_total_calls"


def test_oversized_offline_terminal_replay_is_hard_invalid(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 1,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 3,
        }
    )

    class InvalidClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            return ProviderResponse(
                content="{}",
                model=settings.model,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    failed = extract_batch(
        batch,
        InvalidClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert failed.inference_record.usage.calls == 2

    class ExactTerminalCache:
        def read(self, key: str):
            del key
            return None

        def read_terminal(self, key: str):
            del key
            return failed.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    budget = InferenceBudget(
        settings,
        cache=ExactTerminalCache(),
        initial_calls=2,
    )
    with pytest.raises(
        ValueError,
        match="replayed inference usage exceeds remaining call budget",
    ):
        extract_batch(
            batch,
            MockLLMClient(),
            sources_by_path=sources_by_path,
            model_name=settings.model,
            schema_repairs=1,
            provider_retries=0,
            provider_retry_delay_seconds=0,
            llm_settings=settings,
            budget=budget,
        )
    assert budget.calls == budget.logical_calls == 2


def test_concurrent_nonpermit_replays_atomically_preserve_hard_call_cap(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={
            "schema_repairs": 1,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "max_total_calls": 5,
        }
    )

    class InvalidClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            return ProviderResponse(
                content="{}",
                model=settings.model,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    failed = extract_batch(
        batch,
        InvalidClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=1,
        provider_retries=0,
        provider_retry_delay_seconds=0,
        llm_settings=settings,
    )
    assert failed.inference_record.usage.calls == 2

    budget = InferenceBudget(settings, initial_calls=2)
    budget.plan_stage("explicit_extract", 2)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def replay() -> None:
        barrier.wait(timeout=5)
        try:
            budget.record_terminal(failed.inference_record, replay_hit=True)
        except ValueError as error:
            outcome = str(error)
        else:
            outcome = "accepted"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=replay) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert outcomes.count("accepted") == 1
    assert outcomes.count(
        "replayed inference usage exceeds remaining call budget"
    ) == 1
    assert budget.logical_calls == 4
    assert budget.logical_calls <= settings.max_total_calls
    assert budget.stage_progress["explicit_extract"]["done"] == 1


def test_cached_success_debits_logical_budget_without_provider_cost(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    first = extract_batch(
        batch,
        MockLLMClient(),
        sources_by_path=sources_by_path,
        model_name=config.llm.model,
        schema_repairs=config.llm.schema_repairs,
        provider_retries=config.llm.provider_retries,
        provider_retry_delay_seconds=config.llm.provider_retry_delay_seconds,
        llm_settings=config.llm,
    )
    assert first.succeeded

    class ExactSuccessCache:
        def read(self, key: str):
            del key
            return first.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class ExplodingClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise AssertionError("cached success reached the provider")

    settings = config.llm.model_copy(
        update={
            "max_total_calls": first.inference_record.usage.calls,
        }
    )
    budget = InferenceBudget(settings, cache=ExactSuccessCache())
    replayed = extract_batch(
        batch,
        ExplodingClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=settings.provider_retry_delay_seconds,
        llm_settings=settings,
        budget=budget,
    )

    assert replayed.inference_record == first.inference_record
    assert budget.calls == 0
    assert budget.input_tokens == 0
    assert budget.logical_calls == first.inference_record.usage.calls
    assert budget.terminal_exhaustion_reason() == "max_total_calls"


def test_offline_terminal_failure_replay_preserves_consumed_budget(
    tmp_path: Path,
) -> None:
    batch, config, sources_by_path = _batch(tmp_path)
    settings = config.llm.model_copy(
        update={"schema_repairs": 1, "provider_retries": 0}
    )

    class InvalidTokenClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            return ProviderResponse(
                content="{}",
                model=settings.model,
                input_tokens=7,
                output_tokens=1,
                latency_ms=1,
            )

    failed = extract_batch(
        batch,
        InvalidTokenClient(),
        sources_by_path=sources_by_path,
        model_name=settings.model,
        schema_repairs=settings.schema_repairs,
        provider_retries=settings.provider_retries,
        provider_retry_delay_seconds=settings.provider_retry_delay_seconds,
        llm_settings=settings,
    )
    assert not failed.succeeded
    assert failed.inference_record.usage.calls == 2
    assert failed.inference_record.usage.input_tokens == 14

    class ExactTerminalReplayCache:
        def read(self, key: str):
            del key
            return None

        def read_terminal(self, key: str):
            del key
            return failed.inference_record.model_dump(mode="json")

        def write(self, key: str, record: dict[str, object]) -> None:
            del key, record

    class ExplodingClient:
        def complete(self, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            raise AssertionError("terminal replay reached the provider")

    replay_settings = settings.model_copy(
        update={"max_total_calls": 100, "max_total_input_tokens": 14}
    )
    budget = InferenceBudget(
        replay_settings,
        cache=ExactTerminalReplayCache(),
    )
    replayed = extract_batch(
        batch,
        ExplodingClient(),
        sources_by_path=sources_by_path,
        model_name=replay_settings.model,
        schema_repairs=replay_settings.schema_repairs,
        provider_retries=replay_settings.provider_retries,
        provider_retry_delay_seconds=(
            replay_settings.provider_retry_delay_seconds
        ),
        llm_settings=replay_settings,
        budget=budget,
    )

    assert replayed.inference_record == failed.inference_record
    assert budget.calls == 0
    assert budget.logical_calls == 2
    assert budget.logical_input_tokens == 14
    assert budget.terminal_exhaustion_reason() == "max_total_input_tokens"
