"""Deterministic channel audit exporters and execution-model summaries."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable, Iterable, Protocol

from pydantic import BaseModel

from repo_constraint_memory_v3.inference import (
    InferenceIndex,
    InferenceRecord,
    endpoint_profile_digest,
    inference_record_model_identity_errors,
)
from repo_constraint_memory_v3.models import (
    AuditRecord,
    BuildConfig,
    EvidenceRelation,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.model_profiles import (
    ExecutionModelProfile,
    stable_profile_text,
)
from repo_constraint_memory_v3.observations import (
    ObservationBundle,
)
from repo_constraint_memory_v3.observations.semantic_units import (
    SemanticUnitBundle,
)
from repo_constraint_memory_v3.semantic.validation import (
    CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
    CurrentSourceBundle,
)
from repo_constraint_memory_v3.traceability import TraceabilityArtifact
from repo_constraint_memory_v3.explicit_verification import (
    ExplicitVerificationBundle,
)
from repo_constraint_memory_v3.source_acquisition import SourceAcquisitionBundle


class ChannelArtifactWriter(Protocol):
    """The bounded staging sink owned by the final package exporter."""

    root: Path

    def write_json(self, relative: str, value: object) -> Path: ...

    def write_text(self, relative: str, text: str) -> Path: ...

    def write_bytes(self, relative: str, value: bytes) -> Path: ...


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def stable_json_text(value: object) -> str:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def execution_model_config_errors(
    config: BuildConfig,
    profile: ExecutionModelProfile,
) -> list[str]:
    """Bind every resolved model setting back to the one saved profile."""

    if profile.provider.protocol == "offline_mock":
        expected_api_key_env = "NORM4CODE_OFFLINE_UNUSED_KEY"
        expected_base_url = "https://offline.invalid/v1"
    else:
        expected_api_key_env = profile.provider.api_key_env
        expected_base_url = profile.provider.base_url
    expected = {
        "execution_model_profile_id": profile.profile_id,
        "execution_model_profile_digest": profile.profile_digest,
        "llm.provider": "openai_compatible",
        "llm.model": profile.adapter_model_ids.core_openai_chat,
        "llm.require_response_model": (
            profile.effective_identity.require_response_model
        ),
        "llm.accepted_effective_model_ids": (
            profile.effective_identity.accepted_model_ids
        ),
        "llm.api_key_env": expected_api_key_env,
        "llm.base_url": expected_base_url,
        "llm.temperature": profile.generation.temperature,
        "llm.thinking_mode": profile.reasoning.mode,
        "llm.reasoning_effort": profile.reasoning.effort,
        "llm.reasoning_max_tokens": profile.reasoning.max_tokens,
        "llm.reasoning_wire_policy": profile.reasoning.wire_policy,
    }
    actual = {
        "execution_model_profile_id": config.execution_model_profile_id,
        "execution_model_profile_digest": config.execution_model_profile_digest,
        "llm.provider": config.llm.provider,
        "llm.model": config.llm.model,
        "llm.require_response_model": config.llm.require_response_model,
        "llm.accepted_effective_model_ids": (
            config.llm.accepted_effective_model_ids
        ),
        "llm.api_key_env": config.llm.api_key_env,
        "llm.base_url": config.llm.base_url,
        "llm.temperature": config.llm.temperature,
        "llm.thinking_mode": config.llm.thinking_mode,
        "llm.reasoning_effort": config.llm.reasoning_effort,
        "llm.reasoning_max_tokens": config.llm.reasoning_max_tokens,
        "llm.reasoning_wire_policy": config.llm.reasoning_wire_policy,
    }
    return [
        f"execution-model config field differs from profile: {field}"
        for field in expected
        if actual[field] != expected[field]
    ]


def execution_model_inference_errors(
    records: Iterable[InferenceRecord],
    profile: ExecutionModelProfile,
) -> list[str]:
    """Re-attest configured and provider-reported model identity offline.

    A response attempt stores the provider-reported effective identity.
    Provider-error attempts have no response identity and retain the configured
    wire model.  Treating those cases separately prevents a configured-name
    fallback from masquerading as provider evidence during replay.
    """

    errors: list[str] = []
    configured_model = profile.adapter_model_ids.core_openai_chat
    for record in records:
        request_id = record.request_id
        parameters = record.model_parameters
        if profile.profile_kind == "real":
            errors.extend(
                f"{request_id}: {item}"
                for item in inference_record_model_identity_errors(
                    record,
                    configured_model=configured_model,
                    accepted_effective_model_ids=(
                        profile.effective_identity.accepted_model_ids
                    ),
                )
            )
        elif parameters.model != configured_model:
            errors.append(
                f"{request_id}: configured inference model differs from profile"
            )
        if parameters.temperature != profile.generation.temperature:
            errors.append(
                f"{request_id}: inference temperature differs from profile"
            )
        if parameters.thinking_mode != profile.reasoning.mode:
            errors.append(
                f"{request_id}: inference thinking mode differs from profile"
            )
        if parameters.reasoning_effort != profile.reasoning.effort:
            errors.append(
                f"{request_id}: inference reasoning effort differs from profile"
            )
        if parameters.reasoning_max_tokens != profile.reasoning.max_tokens:
            errors.append(
                f"{request_id}: inference reasoning max_tokens differs from profile"
            )
    return errors


def execution_model_run_summary(
    config: BuildConfig,
    source_acquisition: SourceAcquisitionBundle,
    records: Iterable[InferenceRecord],
) -> dict[str, object]:
    """Return one secret-free configured/effective identity run report."""

    profile = source_acquisition.model_profile
    ordered_records = tuple(records)
    attempts = tuple(
        attempt
        for record in ordered_records
        for attempt in record.attempts
    )
    response_attempts = tuple(
        attempt for attempt in attempts if not attempt.provider_error
    )
    core_errors = execution_model_inference_errors(ordered_records, profile)
    profile_violations = sorted(
        {
            reason
            for record in ordered_records
            for reason in record.incomplete_reasons
            if reason.startswith("model_profile_violation:")
        }
        | {
            reason
            for attempt in attempts
            for reason in attempt.validation_errors
            if reason.startswith("model_profile_violation:")
        }
    )
    agent_receipt = source_acquisition.model_execution
    agent_effective_models = sorted(
        {item.effective_model_id for item in agent_receipt.calls}
    )
    core_effective_models = sorted(
        {item.model for item in response_attempts}
    )
    if profile.profile_kind == "offline_test":
        core_status = "not_applicable"
        homogeneity_status = "offline_test"
    else:
        core_status = (
            "violated"
            if core_errors or profile_violations
            else ("verified" if response_attempts else "not_observed")
        )
        homogeneity_status = (
            "violated"
            if agent_receipt.status != "verified" or core_status == "violated"
            else ("verified" if core_status == "verified" else "configured_only")
        )
    reasoning_identity: dict[str, object] = {
        "mode": profile.reasoning.mode,
        "effort": profile.reasoning.effort,
        "wire_policy": profile.reasoning.wire_policy,
    }
    if profile.reasoning.max_tokens is not None:
        reasoning_identity["max_tokens"] = profile.reasoning.max_tokens
    return {
        "profile_id": profile.profile_id,
        "profile_digest": profile.profile_digest,
        "canonical_model_id": profile.canonical_model_id,
        "provider_id": profile.provider.provider_id,
        "endpoint_profile_digest": endpoint_profile_digest(
            provider=config.llm.provider,
            base_url=config.llm.base_url,
        ),
        "reasoning": reasoning_identity,
        "effective_identity_policy": {
            "require_response_model": (
                profile.effective_identity.require_response_model
            ),
            "accepted_model_ids": (
                profile.effective_identity.accepted_model_ids
            ),
        },
        "configured_adapter_model_ids": {
            "core": profile.adapter_model_ids.core_openai_chat,
            "source_agent": profile.adapter_model_ids.mini_swe_litellm,
            "downstream_agent": profile.adapter_model_ids.downstream_codex,
        },
        "source_agent": {
            "status": agent_receipt.status,
            "configured_model_id": agent_receipt.adapter_model_id,
            "effective_model_ids": agent_effective_models,
            "model_call_count": agent_receipt.model_call_count,
            "reasoning_tokens": agent_receipt.reasoning_tokens,
            "reasoning_content_responses": (
                agent_receipt.reasoning_content_responses
            ),
            "reasoning_content_bytes": agent_receipt.reasoning_content_bytes,
            "request_controls_attested": (
                agent_receipt.request_controls_attested
            ),
            "reasoning_replay_attested": (
                agent_receipt.reasoning_replay_attested
            ),
            "telemetry_status": agent_receipt.telemetry_status,
            "violations": agent_receipt.violations,
        },
        "core": {
            "status": core_status,
            "configured_model_id": config.llm.model,
            "effective_model_ids": core_effective_models,
            "record_count": len(ordered_records),
            "attempt_count": len(attempts),
            "response_attempt_count": len(response_attempts),
            "reasoning_tokens": (
                sum(record.usage.reasoning_tokens or 0 for record in ordered_records)
                if ordered_records
                and all(
                    record.usage.reasoning_tokens is not None
                    for record in ordered_records
                )
                else None
            ),
            "reasoning_content_responses": sum(
                item.reasoning_content_present for item in response_attempts
            ),
            "reasoning_content_bytes": sum(
                item.reasoning_content_bytes for item in response_attempts
            ),
            "profile_violations": profile_violations,
            "validation_errors": sorted(set(core_errors)),
        },
        "homogeneity_status": homogeneity_status,
    }




def _audit_text(records: Iterable[AuditRecord]) -> str:
    ordered = sorted(
        records,
        key=lambda item: (
            item.path,
            item.line_start,
            item.line_end,
            item.unit_id,
        ),
    )
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in ordered
    )


def evidence_relations_payload(
    snapshot_digest: str,
    relations: Iterable[EvidenceRelation],
) -> dict[str, object]:
    ordered = sorted(
        relations,
        key=lambda item: (
            item.relation,
            item.route,
            item.target_fingerprint,
            item.reason,
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "relations": [item.model_dump(mode="json") for item in ordered],
    }


def explicit_verification_payload(
    snapshot_digest: str,
    verification: ExplicitVerificationBundle,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_digest": snapshot_digest,
        "plan": verification.plan.model_dump(mode="json"),
        "result": verification.result.model_dump(mode="json"),
    }


def _write_current_source_tree(
    output_directory: Path,
    *,
    current_source: CurrentSourceBundle,
    snapshot_digest: str,
    paths: dict[str, Path],
    write_json: Callable[[Path, object], None] | None = None,
) -> None:
    """Write the shared immutable current-source subtree."""

    if write_json is None:
        def write_json(path: Path, value: object) -> None:
            _write_text_exclusive(path, stable_json_text(value))

    current_root = output_directory / "current-source"
    current_index_path = current_root / "index.json"
    write_json(
        current_index_path,
        current_source.index,
    )
    paths["current_source_index"] = current_index_path
    map_index_path = current_root / "semantic-observations" / "index.json"
    write_json(
        map_index_path,
        current_source.semantic_map.index,
    )
    paths["semantic_map_index"] = map_index_path
    result_ref_by_packet = {
        item.packet_id: item for item in current_source.semantic_map.index.results
    }
    for result in current_source.semantic_map.results:
        reference = result_ref_by_packet[result.packet_id]
        path = (
            current_root
            / "semantic-observations"
            / "units"
            / f"{reference.result_id}.json"
        )
        write_json(path, result)
        paths[f"semantic_map_{reference.result_id}"] = path
    for observation_item in current_source.semantic_map.observations:
        path = (
            current_root
            / "semantic-observations"
            / "observations"
            / f"{observation_item.observation_id}.json"
        )
        write_json(path, observation_item)
        paths[f"semantic_observation_{observation_item.observation_id}"] = path
    reduction_index_path = current_root / "hypotheses" / "index.json"
    write_json(
        reduction_index_path,
        current_source.reduction.index,
    )
    paths["hypothesis_reduction_index"] = reduction_index_path
    for result in current_source.reduction.call_results:
        path = (
            current_root
            / "hypotheses"
            / "reduction-results"
            / f"{result.result_id}.json"
        )
        write_json(path, result)
        paths[f"reduction_result_{result.result_id}"] = path
    for hypothesis in current_source.hypotheses:
        path = current_root / "hypotheses" / f"{hypothesis.hypothesis_id}.json"
        write_json(path, hypothesis)
        paths[f"hypothesis_{hypothesis.hypothesis_id}"] = path
    for population in current_source.populations:
        path = current_root / "populations" / f"{population.population_id}.json"
        write_json(path, population)
        paths[f"population_{population.population_id}"] = path
    for packet in current_source.member_packets:
        path = current_root / "member-packets" / f"{packet.packet_id}.json"
        write_json(path, packet)
        paths[f"member_packet_{packet.packet_id}"] = path
    for falsification in current_source.falsifications:
        path = (
            current_root
            / "falsifications"
            / f"{falsification.falsification_id}.json"
        )
        write_json(path, falsification)
        paths[f"falsification_{falsification.falsification_id}"] = path
    for refinement in current_source.refinements:
        path = current_root / "refinements" / f"{refinement.refinement_id}.json"
        write_json(path, refinement)
        paths[f"refinement_{refinement.refinement_id}"] = path
    for trace in current_source.traces:
        path = current_root / "lineage" / f"{trace.trace_id}.json"
        write_json(path, trace)
        paths[f"current_trace_{trace.trace_id}"] = path
    events_path = current_root / "lineage" / "events.json"
    write_json(
        events_path,
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": snapshot_digest,
            "events": [
                item.model_dump(mode="json")
                for item in current_source.reduction.events
            ],
            "event_traces": [
                item.model_dump(mode="json")
                for item in current_source.event_traces
            ],
        },
    )
    paths["current_events"] = events_path
    propositions_path = current_root / "propositions.json"
    write_json(
        propositions_path,
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": snapshot_digest,
            "propositions": [
                item.model_dump(mode="json")
                for item in current_source.propositions
            ],
        },
    )
    paths["current_propositions"] = propositions_path
    policy_path = current_root / "policy-decisions.json"
    write_json(
        policy_path,
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": snapshot_digest,
            "policy_version": CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
            "decisions": [
                item.model_dump(mode="json")
                for item in current_source.policy_decisions
            ],
        },
    )
    paths["current_policy_decisions"] = policy_path


def write_build_outputs(
    output_directory: Path,
    *,
    snapshot_digest: str,
    audit_records: Iterable[AuditRecord],
    explicit_verification: ExplicitVerificationBundle,
    evidence_relations: Iterable[EvidenceRelation],
    observation: ObservationBundle | None,
    semantic_units: SemanticUnitBundle | None,
    current_source: CurrentSourceBundle | None,
    traceability: TraceabilityArtifact,
    inference_records: Iterable[InferenceRecord],
    source_acquisition: SourceAcquisitionBundle,
    config: BuildConfig,
    writer: ChannelArtifactWriter,
) -> dict[str, Path]:
    """Write channel evidence directly into a final package staging sink.

    The final exporter owns canonical/consumer, build-report and inference
    artifacts.  No superseded memory, review, catalog or path views are rendered.
    Full builds supply the bounded final-tree writer; independent build-unified
    continues to import its separately authenticated upstream tree.
    """

    inference_records = tuple(inference_records)
    model_config_errors = execution_model_config_errors(
        config, source_acquisition.model_profile,
    )
    model_config_errors.extend(execution_model_inference_errors(
        inference_records, source_acquisition.model_profile,
    ))
    if model_config_errors:
        raise ValueError("; ".join(model_config_errors))
    if writer.root != output_directory:
        raise ValueError("channel staging writer root does not match output")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    def write_json(path: Path, value: object) -> None:
        writer.write_json(path.relative_to(output_directory).as_posix(), value)

    def write_text(path: Path, value: str) -> None:
        writer.write_text(path.relative_to(output_directory).as_posix(), value)

    json_files = {
        "explicit_verification": ("explicit_verification.json", explicit_verification_payload(
            snapshot_digest, explicit_verification,
        )),
        "relations": ("evidence_relations.json", evidence_relations_payload(
            snapshot_digest, evidence_relations,
        )),
        "traceability": ("traceability.json", traceability),
        "discovery": ("discovery.json", source_acquisition.discovery),
        "supplement": ("source_supplement.json", source_acquisition.supplement),
        "source_acquisition": ("source_acquisition.json", source_acquisition.acquisition),
        "config": ("config.json", config),
        "model_execution_receipt": ("model/execution-receipt.json", source_acquisition.model_execution),
        "agent_profile": ("agent/profile.json", source_acquisition.profile),
        "agent_request": ("agent/request.json", source_acquisition.request),
        "agent_runner_result": ("agent/runner-result.json", source_acquisition.runner_result),
        "agent_run": ("agent/run.json", source_acquisition.agent_run),
    }
    for key, (relative, value) in json_files.items():
        path = output_directory / relative
        write_json(path, value)
        paths[key] = path
    text_files = {
        "audit": ("audit.jsonl", _audit_text(audit_records)),
        "execution_model_profile": ("model/profile.json", stable_profile_text(source_acquisition.model_profile)),
        "agent_task": ("agent/task.md", source_acquisition.task_text),
        "agent_response_schema": ("agent/response-schema.json", source_acquisition.response_schema_text),
    }
    for key, (relative, value) in text_files.items():
        path = output_directory / relative
        write_text(path, value)
        paths[key] = path
    binary_files = {
        "agent_trajectory": ("agent/trajectory.json", source_acquisition.trajectory),
        "agent_stdout": ("agent/stdout.txt", source_acquisition.stdout),
        "agent_stderr": ("agent/stderr.txt", source_acquisition.stderr),
    }
    for key, (relative, value) in binary_files.items():
        path = output_directory / relative
        writer.write_bytes(relative, value)
        paths[key] = path
    if observation is not None:
        index_path = output_directory / "observations" / "index.json"
        write_json(index_path, observation.index)
        paths["observation_index"] = index_path
        for name, artifact in sorted(observation.query_artifacts().items()):
            artifact_path = output_directory / "observations" / "queries" / f"{name}.json"
            write_json(artifact_path, artifact)
            paths[f"observation_query_{name}"] = artifact_path
        for shard in observation.shards:
            shard_path = output_directory / "observations" / "python-files" / f"{shard.file_id}.json"
            write_json(shard_path, shard)
            paths[f"observation_shard_{shard.file_id}"] = shard_path
        if semantic_units is not None:
            semantic_root = output_directory / "observations" / "semantic-units"
            semantic_index_path = semantic_root / "index.json"
            write_json(semantic_index_path, semantic_units.index)
            paths["semantic_unit_index"] = semantic_index_path
            (semantic_root / "units").mkdir(parents=True, exist_ok=False)
            for unit in semantic_units.units:
                unit_path = semantic_root / "units" / f"{unit.unit_id}.json"
                write_json(unit_path, unit)
                paths[f"semantic_unit_{unit.unit_id}"] = unit_path
    if current_source is not None:
        _write_current_source_tree(
            output_directory,
            current_source=current_source,
            snapshot_digest=snapshot_digest,
            paths=paths,
            write_json=write_json,
        )
    return paths


def write_current_source_only_outputs(
    output_directory: Path,
    *,
    run_manifest: BaseModel,
    config: BuildConfig,
    observation: ObservationBundle,
    semantic_units: SemanticUnitBundle,
    current_source: CurrentSourceBundle,
    inference_index: InferenceIndex,
    inference_records: Iterable[InferenceRecord],
    source_acquisition: SourceAcquisitionBundle,
) -> dict[str, Path]:
    """Write a non-delivery current-source-only channel run."""

    inference_records = tuple(inference_records)
    model_config_errors = execution_model_config_errors(
        config,
        source_acquisition.model_profile,
    )
    model_config_errors.extend(
        execution_model_inference_errors(
            inference_records,
            source_acquisition.model_profile,
        )
    )
    if model_config_errors:
        raise ValueError("; ".join(model_config_errors))

    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    top_level_files: dict[str, tuple[str, object]] = {
        "channel_run": ("channel_run.json", run_manifest),
        "config": ("config.json", config),
        "discovery": ("discovery.json", source_acquisition.discovery),
        "supplement": (
            "source_supplement.json",
            source_acquisition.supplement,
        ),
        "source_acquisition": (
            "source_acquisition.json",
            source_acquisition.acquisition,
        ),
    }
    for key, (relative, value) in top_level_files.items():
        path = output_directory / relative
        _write_text_exclusive(path, stable_json_text(value))
        paths[key] = path

    model_files = {
        "execution_model_profile": (
            "profile.json",
            stable_profile_text(source_acquisition.model_profile),
        ),
        "model_execution_receipt": (
            "execution-receipt.json",
            stable_json_text(source_acquisition.model_execution),
        ),
    }
    for key, (relative, text) in model_files.items():
        path = output_directory / "model" / relative
        _write_text_exclusive(path, text)
        paths[key] = path

    agent_text_files = {
        "agent_profile": ("profile.json", stable_json_text(source_acquisition.profile)),
        "agent_request": ("request.json", stable_json_text(source_acquisition.request)),
        "agent_runner_result": (
            "runner-result.json",
            stable_json_text(source_acquisition.runner_result),
        ),
        "agent_run": ("run.json", stable_json_text(source_acquisition.agent_run)),
        "agent_task": ("task.md", source_acquisition.task_text),
        "agent_response_schema": (
            "response-schema.json",
            source_acquisition.response_schema_text,
        ),
    }
    for key, (relative, text) in agent_text_files.items():
        path = output_directory / "agent" / relative
        _write_text_exclusive(path, text)
        paths[key] = path
    agent_binary_files = {
        "agent_trajectory": ("trajectory.json", source_acquisition.trajectory),
        "agent_stdout": ("stdout.txt", source_acquisition.stdout),
        "agent_stderr": ("stderr.txt", source_acquisition.stderr),
    }
    for key, (relative, value) in agent_binary_files.items():
        path = output_directory / "agent" / relative
        _write_bytes_exclusive(path, value)
        paths[key] = path

    inference_index_path = output_directory / "inference" / "index.json"
    _write_text_exclusive(inference_index_path, stable_json_text(inference_index))
    paths["inference_index"] = inference_index_path
    (output_directory / "inference" / "requests").mkdir(
        parents=True,
        exist_ok=False,
    )
    for record in sorted(inference_records, key=lambda item: item.request_id):
        request_path = (
            output_directory
            / "inference"
            / "requests"
            / f"{record.request_id}.json"
        )
        _write_text_exclusive(request_path, stable_json_text(record))
        paths[f"inference_request_{record.request_id}"] = request_path

    observation_index_path = output_directory / "observations" / "index.json"
    _write_text_exclusive(
        observation_index_path,
        stable_json_text(observation.index),
    )
    paths["observation_index"] = observation_index_path
    for name, artifact in sorted(observation.query_artifacts().items()):
        artifact_path = (
            output_directory / "observations" / "queries" / f"{name}.json"
        )
        _write_text_exclusive(artifact_path, stable_json_text(artifact))
        paths[f"observation_query_{name}"] = artifact_path
    for shard in observation.shards:
        shard_path = (
            output_directory
            / "observations"
            / "python-files"
            / f"{shard.file_id}.json"
        )
        _write_text_exclusive(shard_path, stable_json_text(shard))
        paths[f"observation_shard_{shard.file_id}"] = shard_path
    semantic_index_path = (
        output_directory / "observations" / "semantic-units" / "index.json"
    )
    _write_text_exclusive(
        semantic_index_path,
        stable_json_text(semantic_units.index),
    )
    paths["semantic_unit_index"] = semantic_index_path
    (output_directory / "observations" / "semantic-units" / "units").mkdir(
        parents=True,
        exist_ok=False,
    )
    for unit in semantic_units.units:
        unit_path = (
            output_directory
            / "observations"
            / "semantic-units"
            / "units"
            / f"{unit.unit_id}.json"
        )
        _write_text_exclusive(unit_path, stable_json_text(unit))
        paths[f"semantic_unit_{unit.unit_id}"] = unit_path

    _write_current_source_tree(
        output_directory,
        current_source=current_source,
        snapshot_digest=current_source.index.snapshot_digest,
        paths=paths,
    )
    return paths


def _write_text_exclusive(path: Path, text: str) -> None:
    """Atomically publish one new UTF-8 artifact without overwriting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite generated artifact: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite generated artifact: {path}"
            ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    """Atomically publish one new binary artifact without overwriting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite generated artifact: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite generated artifact: {path}"
            ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
