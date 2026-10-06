from __future__ import annotations

import json
import threading
from pathlib import Path

import repo_constraint_memory_v3.cli as cli_module
import repo_constraint_memory_v3.current_source_run as current_source_run_module
from helpers import (
    CONFIG_PATH,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.agent_runtime import (
    DeterministicFakeAgentRuntime,
    fake_agent_profile,
)
from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.current_source_run import (
    _read_bounded_bytes,
    _read_inference_tree,
    build_current_source_only,
    current_source_inference_record_bound,
    validate_current_source_only_run,
)
from repo_constraint_memory_v3.exporters import (
    execution_model_inference_errors,
    stable_json_text,
)
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.model_profiles import (
    ExecutionModelProfile,
    load_execution_model_profile,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.observations.canonical import (
    payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.source_acquisition import acquire_sources


_CURRENT_STAGES = {
    "current_semantic_map",
    "current_hypothesis_reduce",
    "population_semantic_classify",
    "counterexample_verify",
}
_FORBIDDEN_OUTPUTS = {
    "audit.jsonl",
    "build_report.json",
    "canonical",
    "consumers",
    "explicit_verification.json",
    "traceability.json",
    "unified",
}


def _offline_model_profile() -> ExecutionModelProfile:
    return load_execution_model_profile(
        "offline-deterministic-test",
        allow_offline=True,
    )


class _StageSpyClient:
    def __init__(self) -> None:
        self.delegate = MockLLMClient()
        self.stages: list[str] = []
        self.lock = threading.Lock()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        stage = str(request.get("stage") or "explicit_extract")
        with self.lock:
            self.stages.append(stage)
        return self.delegate.complete(system_prompt, user_prompt)


class _BrokenMapClient:
    def __init__(self) -> None:
        self.delegate = MockLLMClient()

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
        return self.delegate.complete(system_prompt, user_prompt)


class _ExplodingClient:
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt, user_prompt
        raise AssertionError("a cached current-source inference called the provider")


def _build_direct_run(
    tmp_path: Path,
    *,
    client=None,
    config=None,
):
    repository = copied_current_source_fixture(tmp_path)
    config = config or config_for_tests()
    profile = fake_agent_profile()
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "direct-acquisition",
    )
    run = tmp_path / "direct-current-source-run"
    result = build_current_source_only(
        repository,
        config,
        acquisition,
        client or MockLLMClient(),
        run,
    )
    return repository, run, result


def test_cli_current_source_only_has_no_explicit_calls_or_delivery_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    profile_path = tmp_path / "profile.json"
    profile_value = profile.model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        profile_value.pop(key)
    profile_path.write_text(stable_json_text(profile_value), encoding="utf-8")
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "acquisition",
    )
    run = tmp_path / "current-source-run"
    spy = _StageSpyClient()
    monkeypatch.setattr(cli_module, "MockLLMClient", lambda: spy)

    assert main(
        [
            "build-current-source",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile_path),
            "--acquisition",
            str(acquisition.directory),
            "--out",
            str(run),
            "--mock",
        ]
    ) == 0

    assert spy.stages
    assert set(spy.stages) <= _CURRENT_STAGES
    assert "explicit_extract" not in spy.stages
    assert "explicit_semantic_verify" not in spy.stages
    manifest = json.loads((run / "channel_run.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "current_source_only"
    assert manifest["explicit_extract_status"] == "not_run"
    assert manifest["explicit_semantic_verify_status"] == "not_run"
    assert manifest["explicit_context_mode"] == "none"
    assert manifest["execution_model_profile_id"] == (
        config.execution_model_profile_id
    )
    assert manifest["execution_model_profile_digest"] == (
        config.execution_model_profile_digest
    )
    assert {item.name for item in (run / "model").iterdir()} == {
        "profile.json",
        "execution-receipt.json",
    }
    assert not (_FORBIDDEN_OUTPUTS & {item.name for item in run.iterdir()})
    assert main(
        [
            "validate-current-source",
            str(run),
            "--repository",
            str(repository),
        ]
    ) == 0

    with (run / "current-source" / "index.json").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write("\n")
    assert main(
        [
            "validate-current-source",
            str(run),
            "--repository",
            str(repository),
        ]
    ) == 1


def test_current_source_only_cli_help_returns_zero() -> None:
    for argv in (
        ["build-current-source", "--help"],
        ["validate-current-source", "--help"],
    ):
        try:
            main(argv)
        except SystemExit as exit_signal:
            assert exit_signal.code == 0
        else:
            raise AssertionError("argparse help must exit cleanly")


def test_current_source_model_artifact_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    repository, run, _ = _build_direct_run(tmp_path)
    receipt = run / "model" / "execution-receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b"\n")

    errors = validate_current_source_only_run(
        run,
        repository_root=repository,
    )
    assert errors
    assert any(
        "model" in item.lower()
        or "canonical" in item.lower()
        or "source_acquisition" in item
        for item in errors
    )


def test_partial_inference_reasons_round_trip_through_channel_manifest(
    tmp_path: Path,
) -> None:
    repository, run, result = _build_direct_run(
        tmp_path,
        client=_BrokenMapClient(),
    )

    assert not result.run_manifest.complete
    assert result.inference_index.incomplete_reasons
    assert any(
        record.terminal_status == "failure"
        for record in result.inference_records
    )
    assert set(result.inference_index.incomplete_reasons) <= set(
        result.run_manifest.incomplete_reasons
    )
    assert validate_current_source_only_run(
        run,
        repository_root=repository,
    ) == []
    assert "current-source-only run is incomplete" in (
        validate_current_source_only_run(
            run,
            repository_root=repository,
            require_complete=True,
        )
    )


def test_manifest_python_gap_round_trips_and_population_tamper_replays(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    gap_path = repository / "omitted/manifest_gap.py"
    gap_path.parent.mkdir()
    config = config_for_tests()
    gap_path.write_bytes(b"VALUE = 1\0untrusted-binary-tail")
    profile = fake_agent_profile()
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "gap-acquisition",
    )
    run = tmp_path / "gap-current-source-run"
    result = build_current_source_only(
        repository,
        config,
        acquisition,
        MockLLMClient(),
        run,
    )

    assert result.observation.snapshot is not None
    assert [
        item.path
        for item in result.observation.snapshot.unrepresented_python_files
    ] == ["omitted/manifest_gap.py"]
    assert not result.run_manifest.complete
    matching = [
        population
        for population in result.current_source.populations
        if population.population_result.dependency_paths
    ]
    assert matching
    assert all(
        population.population_result.dependency_paths
        == ["omitted/manifest_gap.py"]
        for population in matching
    )
    assert validate_current_source_only_run(
        run,
        repository_root=repository,
    ) == []

    population_path = (
        run
        / "current-source/populations"
        / f"{matching[0].population_id}.json"
    )
    payload = json.loads(population_path.read_text(encoding="utf-8"))
    payload["population_result"]["dependency_paths"] = []
    payload["population_result"]["result_digest"] = payload_digest(
        {
            key: value
            for key, value in payload["population_result"].items()
            if key != "result_digest"
        }
    )
    payload["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_digest"
        }
    )
    population_path.write_text(stable_json_text(payload), encoding="utf-8")
    errors = validate_current_source_only_run(
        run,
        repository_root=repository,
    )
    assert "current-source populations differs from recorded replay" in errors


def test_map_reduce_records_can_exceed_provider_budget_before_compact_suffix(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={"max_total_calls": 1, "max_concurrency": 1}
            )
        }
    )
    profile = fake_agent_profile()
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "partial-acquisition",
    )
    run = tmp_path / "partial-current-source-run"
    result = build_current_source_only(
        repository,
        config,
        acquisition,
        MockLLMClient(),
        run,
    )

    assert len(result.inference_records) > config.llm.max_total_calls
    assert result.inference_index.total_calls == config.llm.max_total_calls
    assert not result.run_manifest.complete
    assert validate_current_source_only_run(
        run,
        repository_root=repository,
    ) == []


def test_logical_record_bound_is_independent_of_provider_call_budget() -> None:
    config = config_for_tests()
    constrained = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"max_total_calls": 1})
        }
    )

    assert current_source_inference_record_bound(constrained) == (
        current_source_inference_record_bound(config)
    )
    settings = config.semantic
    expected = (
            settings.max_total_packet_bytes
            + 5 * (settings.max_total_observations + settings.max_units)
            + settings.max_total_classified_population_members
            + settings.max_total_falsification_calls
        )
    assert current_source_inference_record_bound(constrained) == expected


def test_multi_record_tree_uses_per_record_and_aggregate_bounds(
    tmp_path: Path,
) -> None:
    repository, _, baseline = _build_direct_run(tmp_path)
    baseline_sizes = [
        len(stable_json_text(record).encode("utf-8"))
        for record in baseline.inference_records
    ]
    assert len(baseline_sizes) > 1
    baseline_index_bytes = len(
        stable_json_text(baseline.inference_index).encode("utf-8")
    )
    per_record_limit = max(
        1_000,
        max([*baseline_sizes, baseline_index_bytes]) + 1_024,
    )
    config = config_for_tests()
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={"max_inference_artifact_bytes": per_record_limit}
            )
        }
    )
    profile = fake_agent_profile()
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "bounded-acquisition",
    )
    run = tmp_path / "bounded-current-source-run"
    result = build_current_source_only(
        repository,
        config,
        acquisition,
        MockLLMClient(),
        run,
    )

    assert all(
        len(stable_json_text(record).encode("utf-8")) <= per_record_limit
        for record in result.inference_records
    )
    assert result.inference_index.total_artifact_bytes > per_record_limit
    assert validate_current_source_only_run(
        run,
        repository_root=repository,
    ) == []


def test_success_cache_can_emit_multiple_logical_records_without_calls(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    acquisition = acquire_sources(
        repository,
        config,
        profile,
        _offline_model_profile(),
        DeterministicFakeAgentRuntime(profile),
        tmp_path / "cache-acquisition",
    )
    cache = tmp_path / "inference-cache"
    first = build_current_source_only(
        repository,
        config,
        acquisition,
        MockLLMClient(),
        tmp_path / "first-current-source-run",
        inference_cache=cache,
    )
    assert len(first.inference_records) > 1
    assert all(
        record.terminal_status == "success"
        for record in first.inference_records
    )

    second_run = tmp_path / "cached-current-source-run"
    second = build_current_source_only(
        repository,
        config,
        acquisition,
        _ExplodingClient(),
        second_run,
        inference_cache=cache,
    )

    assert second.inference_records == first.inference_records
    assert validate_current_source_only_run(
        second_run,
        repository_root=repository,
    ) == []


def test_validator_rejects_coordinated_current_method_tamper(
    tmp_path: Path,
) -> None:
    repository, run, _ = _build_direct_run(tmp_path)
    request_path = next((run / "inference" / "requests").glob("*.json"))
    record = json.loads(request_path.read_text(encoding="utf-8"))
    record["system_prompt_digest"] = "0" * 64
    record["record_digest"] = payload_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )
    request_text = stable_json_text(record)
    request_path.write_text(request_text, encoding="utf-8")

    index_path = run / "inference" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    reference = next(
        item
        for item in index["requests"]
        if item["request_id"] == record["request_id"]
    )
    reference["artifact_sha256"] = sha256_hex(request_text.encode("utf-8"))
    reference["record_digest"] = record["record_digest"]
    index["inference_set_digest"] = payload_digest(
        {
            key: value
            for key, value in index.items()
            if key != "inference_set_digest"
        }
    )
    index_path.write_text(stable_json_text(index), encoding="utf-8")

    manifest_path = run / "channel_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inference_set_digest"] = index["inference_set_digest"]
    manifest["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in manifest.items()
            if key != "artifact_digest"
        }
    )
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")

    errors = validate_current_source_only_run(
        run,
        repository_root=repository,
    )
    assert any("current-stage system prompt differs" in item for item in errors)


def test_host_skipped_reducer_reason_is_indexed_and_record_tamper_fails_replay(
    tmp_path: Path,
) -> None:
    config = config_for_tests(semantic={"max_reduce_request_bytes": 1})
    repository, run, result = _build_direct_run(tmp_path, config=config)
    skipped = next(
        record
        for record in result.inference_records
        if record.stage == "current_hypothesis_reduce"
        and record.attempts == []
        and record.incomplete_reasons
        == ["max_reduce_request_bytes"]
    )
    assert "max_reduce_request_bytes" in (
        result.current_source.reduction.index.incomplete_reasons
    )
    assert "max_reduce_request_bytes" in (
        result.current_source.index.incomplete_reasons
    )
    assert validate_current_source_only_run(
        run,
        repository_root=repository,
    ) == []

    request_path = run / "inference" / "requests" / f"{skipped.request_id}.json"
    record = json.loads(request_path.read_text(encoding="utf-8"))
    tamper_reason = "coordinated_inference_record_tamper"
    record["incomplete_reasons"] = sorted(
        {*record["incomplete_reasons"], tamper_reason}
    )
    record["record_digest"] = payload_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )
    request_text = stable_json_text(record)
    request_path.write_text(request_text, encoding="utf-8")

    index_path = run / "inference" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    reference = next(
        item
        for item in index["requests"]
        if item["request_id"] == record["request_id"]
    )
    reference["artifact_sha256"] = sha256_hex(request_text.encode("utf-8"))
    reference["record_digest"] = record["record_digest"]
    index["total_artifact_bytes"] = sum(
        len(path.read_bytes())
        for path in (run / "inference" / "requests").glob("*.json")
    )
    index["incomplete_reasons"] = sorted(
        {*index["incomplete_reasons"], tamper_reason}
    )
    index["inference_set_digest"] = payload_digest(
        {
            key: value
            for key, value in index.items()
            if key != "inference_set_digest"
        }
    )
    index_path.write_text(stable_json_text(index), encoding="utf-8")

    manifest_path = run / "channel_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inference_set_digest"] = index["inference_set_digest"]
    manifest["incomplete_reasons"] = sorted(
        {*manifest["incomplete_reasons"], tamper_reason}
    )
    manifest["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in manifest.items()
            if key != "artifact_digest"
        }
    )
    manifest_path.write_text(stable_json_text(manifest), encoding="utf-8")

    errors = validate_current_source_only_run(
        run,
        repository_root=repository,
    )
    assert (
        "current-source inference records differs from recorded replay"
        in errors
    )


def test_real_profile_rejects_effective_model_tamper_offline(
    tmp_path: Path,
) -> None:
    _, _, result = _build_direct_run(tmp_path)
    profile = load_execution_model_profile(
        "apiyi-deepseek-v4-flash-260425-nonthinking"
    )
    original = result.inference_records[0]
    record = original.model_copy(
        update={
            "model_parameters": original.model_parameters.model_copy(
                update={"model": profile.adapter_model_ids.core_openai_chat}
            )
        }
    )
    errors = execution_model_inference_errors([record], profile)
    assert any("provider effective model is not accepted" in item for item in errors)


def test_validator_fails_closed_for_unknown_current_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, run, result = _build_direct_run(tmp_path)
    stage = result.inference_records[0].stage
    monkeypatch.delitem(current_source_run_module._CURRENT_STAGE_METHODS, stage)

    errors = validate_current_source_only_run(
        run,
        repository_root=repository,
    )
    assert any("unknown current-stage method" in item for item in errors)


def test_inference_tree_rejects_records_beyond_logical_plan_bound(
    tmp_path: Path,
) -> None:
    _, run, result = _build_direct_run(tmp_path)
    assert len(result.inference_records) > 1
    errors: list[str] = []

    index, records = _read_inference_tree(
        run,
        snapshot_digest=result.run_manifest.snapshot_digest,
        protocol_manifest=result.protocol_manifest,
        max_record_bytes=config_for_tests().llm.max_inference_artifact_bytes,
        max_logical_records=len(result.inference_records) - 1,
        errors=errors,
    )

    assert index is not None
    assert records == ()
    assert any("logical-record bound" in item for item in errors)


def test_bounded_inference_reader_rejects_oversize_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "oversize.json"
    artifact.write_bytes(b"1234")

    try:
        _read_bounded_bytes(artifact, max_bytes=3)
    except ValueError as error:
        assert "exceeds configured byte bound" in str(error)
    else:
        raise AssertionError("oversize inference artifact must be rejected")
