from __future__ import annotations

import gzip
import json

import httpx
import pytest

from helpers import CONFIG_PATH, config_for_tests
from repo_constraint_memory_v3.builder import (
    load_build_config,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.inference import (
    CURRENT_SOURCE_SEMANTIC_STAGES,
    MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    InferenceBudget,
    InferenceInputRef,
    StageDefinition,
    run_structured_inference,
)
from repo_constraint_memory_v3.llm import OpenAICompatibleClient
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from repo_constraint_memory_v3.models import LLMSettings, StrictModel


_TRANSPORT_MODEL_PROFILE = load_execution_model_profile()


def _transport_settings() -> LLMSettings:
    return load_build_config(
        CONFIG_PATH,
        model_profile=_TRANSPORT_MODEL_PROFILE,
    ).llm


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.yielded_chunks = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded_chunks += 1
            yield chunk


class _BooleanResponse(StrictModel):
    ok: bool


class _CacheSpy:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, object]]] = []

    def read(self, key: str) -> None:
        del key
        return None

    def write(self, key: str, record: dict[str, object]) -> None:
        self.writes.append((key, record))


def _provider_body(
    content: str,
    *,
    effective_model_id: str,
    cached_input_tokens: int | None = None,
) -> bytes:
    usage: dict[str, object] = {"prompt_tokens": 3, "completion_tokens": 5}
    if cached_input_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_input_tokens}
    return json.dumps(
        {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "model": effective_model_id,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _transport_client(
    *,
    settings,
    handler,
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(
        settings=settings,
        api_key="test-key",
        model_profile=_TRANSPORT_MODEL_PROFILE,
    )
    client._client.close()
    client._client = httpx.Client(  # type: ignore[assignment]
        transport=httpx.MockTransport(handler),
        timeout=settings.timeout_seconds,
    )
    return client


def test_openai_transport_accepts_response_near_frozen_wire_limit() -> None:
    limit = 1_000_000
    settings = _transport_settings().model_copy(
        update={"max_provider_response_bytes": limit}
    )
    empty_size = len(
        _provider_body("", effective_model_id=settings.model)
    )
    content = "x" * (limit - empty_size)
    body = _provider_body(content, effective_model_id=settings.model)
    assert len(body) == limit
    stream = _ChunkStream(
        [body[offset : offset + 197_123] for offset in range(0, len(body), 197_123)]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            200,
            request=request,
            headers={"Content-Encoding": "identity"},
            stream=stream,
        )

    client = _transport_client(settings=settings, handler=handler)
    try:
        response = client.complete("system", "user")
    finally:
        client.close()

    assert response.content == content
    assert stream.yielded_chunks == len(stream.chunks)


def test_openai_transport_reports_cached_input_with_explicit_coverage() -> None:
    settings = _transport_settings()
    bodies = [
        _provider_body(
            '{"ok":true}',
            effective_model_id=settings.model,
            cached_input_tokens=2,
        ),
        _provider_body('{"ok":true}', effective_model_id=settings.model),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            stream=_ChunkStream([bodies.pop(0)]),
        )

    client = _transport_client(settings=settings, handler=handler)
    try:
        client.complete("system", "first")
        client.complete("system", "second")
        telemetry = client.cached_input_telemetry()
    finally:
        client.close()

    assert telemetry == {
        "source": "core_openai_chat",
        "status": "partial",
        "tokens": 2,
        "provider_request_count": 2,
        "provider_response_count": 2,
        "responses_with_value": 1,
    }


def test_oversized_stream_is_retryable_terminal_failure_and_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _transport_settings().model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 1,
            "provider_retry_delay_seconds": 0,
            "provider_retry_max_delay_seconds": 0,
            "provider_retry_jitter_seconds": 0,
            "max_provider_response_bytes": 1_000_000,
        }
    )
    streams: list[_ChunkStream] = []

    def forbidden_json_decode(value: object) -> object:
        del value
        raise AssertionError("oversized provider body reached JSON decoding")

    class NoDecode:
        loads = staticmethod(forbidden_json_decode)

    monkeypatch.setattr("repo_constraint_memory_v3.llm.json", NoDecode)

    def handler(request: httpx.Request) -> httpx.Response:
        stream = _ChunkStream([b"x" * 400_000 for _ in range(4)])
        streams.append(stream)
        return httpx.Response(200, request=request, stream=stream)

    cache = _CacheSpy()
    budget = InferenceBudget(settings, cache=cache)
    definition = StageDefinition(
        stage="current_semantic_map",
        mode="transport_limit_fixture",
        protocol_version="transport-limit-fixture-1.0",
        response_schema_version="transport-limit-response-1.0",
        system_prompt="Return JSON.",
        response_model=_BooleanResponse,
        validate_response=lambda value: value,
        repair_prompt=lambda request, response, errors: request,
        max_response_bytes=MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    )
    client = _transport_client(settings=settings, handler=handler)
    try:
        parsed, record = run_structured_inference(
            definition=definition,
            inputs=[
                InferenceInputRef(
                    kind="semantic_packet",
                    stable_id="PACKET-fixture",
                    digest="2" * 64,
                    snapshot_digest="sha256:" + "1" * 64,
                )
            ],
            request_ordinal=0,
            request_payload='{"fixture":true}',
            client=client,
            settings=settings,
            protocol_manifest_digest="3" * 64,
            snapshot_digest="sha256:" + "1" * 64,
            budget=budget,
        )
    finally:
        client.close()

    assert parsed is None
    assert record.terminal_status == "failure"
    assert len(record.attempts) == 2
    assert all(attempt.provider_error for attempt in record.attempts)
    assert all(attempt.retryable for attempt in record.attempts)
    assert record.incomplete_reasons == [
        "LLM provider response exceeded max_provider_response_bytes"
    ]
    assert len(streams) == 2
    assert all(stream.yielded_chunks == 3 for stream in streams)
    assert cache.writes == []


def test_compressed_response_is_rejected_before_inflation_decode_or_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 1_000_000
    settings = _transport_settings().model_copy(
        update={
            "schema_repairs": 0,
            "output_limit_retries": 0,
            "provider_retries": 0,
            "provider_retry_delay_seconds": 0,
            "provider_retry_max_delay_seconds": 0,
            "provider_retry_jitter_seconds": 0,
            "max_provider_response_bytes": limit,
        }
    )
    inflated_body = _provider_body(
        "x" * (limit + 100_000),
        effective_model_id=settings.model,
    )
    compressed_body = gzip.compress(inflated_body)
    assert len(compressed_body) < limit
    assert len(inflated_body) > limit
    stream = _ChunkStream([compressed_body])

    def forbidden_json_decode(value: object) -> object:
        del value
        raise AssertionError("compressed provider body reached JSON decoding")

    class NoDecode:
        loads = staticmethod(forbidden_json_decode)

    monkeypatch.setattr("repo_constraint_memory_v3.llm.json", NoDecode)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            200,
            request=request,
            headers={"Content-Encoding": "gzip"},
            stream=stream,
        )

    cache = _CacheSpy()
    budget = InferenceBudget(settings, cache=cache)
    definition = StageDefinition(
        stage="current_semantic_map",
        mode="compressed_transport_fixture",
        protocol_version="compressed-transport-fixture-1.0",
        response_schema_version="compressed-transport-response-1.0",
        system_prompt="Return JSON.",
        response_model=_BooleanResponse,
        validate_response=lambda value: value,
        repair_prompt=lambda request, response, errors: request,
        max_response_bytes=MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    )
    client = _transport_client(settings=settings, handler=handler)
    try:
        parsed, record = run_structured_inference(
            definition=definition,
            inputs=[
                InferenceInputRef(
                    kind="semantic_packet",
                    stable_id="PACKET-compressed-fixture",
                    digest="4" * 64,
                    snapshot_digest="sha256:" + "1" * 64,
                )
            ],
            request_ordinal=0,
            request_payload='{"fixture":true}',
            client=client,
            settings=settings,
            protocol_manifest_digest="3" * 64,
            snapshot_digest="sha256:" + "1" * 64,
            budget=budget,
        )
    finally:
        client.close()

    assert parsed is None
    assert record.terminal_status == "failure"
    assert len(record.attempts) == 1
    assert record.attempts[0].provider_error
    assert record.attempts[0].retryable
    assert record.incomplete_reasons == [
        "LLM provider returned unsupported Content-Encoding"
    ]
    assert stream.yielded_chunks == 0
    assert cache.writes == []


@pytest.mark.parametrize("stage", sorted(CURRENT_SOURCE_SEMANTIC_STAGES))
def test_current_source_stage_definitions_require_frozen_response_cap(stage) -> None:
    definition = StageDefinition(
        stage=stage,
        mode="response-cap-fixture",
        protocol_version="response-cap-fixture-1.0",
        response_schema_version="response-cap-response-1.0",
        system_prompt="Return JSON.",
        response_model=_BooleanResponse,
        validate_response=lambda value: value,
        repair_prompt=lambda request, response, errors: request,
        max_response_bytes=MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    )
    assert definition.max_response_bytes == 2_000_000

    with pytest.raises(ValueError, match="frozen max_response_bytes"):
        StageDefinition(
            stage=stage,
            mode="response-cap-fixture",
            protocol_version="response-cap-fixture-1.0",
            response_schema_version="response-cap-response-1.0",
            system_prompt="Return JSON.",
            response_model=_BooleanResponse,
            validate_response=lambda value: value,
            repair_prompt=lambda request, response, errors: request,
            max_response_bytes=MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES - 1,
        )


def test_protocol_manifest_binds_wire_and_current_source_response_caps() -> None:
    config = config_for_tests()
    manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )

    assert config.llm.max_provider_response_bytes == 32_000_000
    assert manifest.resource_limits["max_provider_response_bytes"] == 32_000_000
    assert {
        key: manifest.resource_limits[key]
        for key in (
            "max_current_semantic_map_response_bytes",
            "max_current_hypothesis_reduce_response_bytes",
            "max_population_semantic_classify_response_bytes",
            "max_counterexample_verify_response_bytes",
        )
    } == {
        "max_current_semantic_map_response_bytes": 2_000_000,
        "max_current_hypothesis_reduce_response_bytes": 2_000_000,
        "max_population_semantic_classify_response_bytes": 2_000_000,
        "max_counterexample_verify_response_bytes": 2_000_000,
    }
    assert MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES > 141_670
