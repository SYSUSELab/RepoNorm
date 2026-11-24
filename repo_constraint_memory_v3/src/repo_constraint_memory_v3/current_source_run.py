"""Standalone fixed-snapshot current-source channel builds and replay validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import ValidationError, model_validator

from repo_constraint_memory_v3 import __version__
from repo_constraint_memory_v3.builder import (
    _TrackingClient,
    _verify_discovery,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.exporters import (
    execution_model_config_errors,
    execution_model_inference_errors,
    execution_model_run_summary,
    stable_json_text,
    write_current_source_only_outputs,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceIndex,
    InferenceRecord,
    ProtocolManifest,
    make_inference_index,
    protocol_manifest_for_readonly_replay,
)
from repo_constraint_memory_v3.inference_cache import SuccessInferenceCache
from repo_constraint_memory_v3.llm import LLMClient
from repo_constraint_memory_v3.models import (
    BuildConfig,
    NonEmptyString,
    ObservationCoverage,
    ObservationQueryCoverage,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    model_payload_digest,
    payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.observations.engine import (
    ObservationBundle,
    run_observations,
)
from repo_constraint_memory_v3.observations.semantic_units import (
    SemanticUnitBundle,
    build_semantic_units,
    validate_semantic_unit_tree,
)
from repo_constraint_memory_v3.observations.validation import (
    validate_observation_tree,
)
from repo_constraint_memory_v3.semantic.validation import (
    CurrentSourceBundle,
    validate_current_source_tree,
    run_current_source_pipeline,
)
from repo_constraint_memory_v3.semantic.classify import (
    CLASSIFY_PROTOCOL_VERSION,
    CLASSIFY_RESPONSE_SCHEMA_VERSION,
    CLASSIFY_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.semantic.falsification import (
    FALSIFICATION_PROTOCOL_VERSION,
    FALSIFICATION_RESPONSE_SCHEMA_VERSION,
    FALSIFICATION_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.semantic.map import (
    MAP_PROTOCOL_VERSION,
    MAP_RESPONSE_SCHEMA_VERSION,
    MAP_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.semantic.reduce import (
    REDUCE_PROTOCOL_VERSION,
    REDUCE_RESPONSE_SCHEMA_VERSION,
    REDUCE_SYSTEM_PROMPT,
)
from repo_constraint_memory_v3.source_acquisition import (
    SourceAcquisitionBundle,
    SourceAcquisitionError,
    load_source_acquisition,
    validate_acquisition_for_build,
)


CURRENT_SOURCE_ONLY_RUN_VERSION = "current-source-only-run-1.0"
CURRENT_SOURCE_INFERENCE_STAGES = frozenset(
    {
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
    }
)
_CURRENT_STAGE_METHODS = {
    "current_semantic_map": (
        MAP_PROTOCOL_VERSION,
        MAP_RESPONSE_SCHEMA_VERSION,
        MAP_SYSTEM_PROMPT,
        frozenset({"packet_map"}),
    ),
    "current_hypothesis_reduce": (
        REDUCE_PROTOCOL_VERSION,
        REDUCE_RESPONSE_SCHEMA_VERSION,
        REDUCE_SYSTEM_PROMPT,
        frozenset({"component_partition"}),
    ),
    "population_semantic_classify": (
        CLASSIFY_PROTOCOL_VERSION,
        CLASSIFY_RESPONSE_SCHEMA_VERSION,
        CLASSIFY_SYSTEM_PROMPT,
        frozenset({"inspected_member_classification"}),
    ),
    "counterexample_verify": (
        FALSIFICATION_PROTOCOL_VERSION,
        FALSIFICATION_RESPONSE_SCHEMA_VERSION,
        FALSIFICATION_SYSTEM_PROMPT,
        frozenset({"candidate_falsification"}),
    ),
}


class CurrentSourceOnlyRun(StrictModel):
    run_version: Literal["current-source-only-run-1.0"]
    run_kind: Literal["current_source_only"]
    schema_version: Literal["4.11-partition-only-required-evidence"]
    package_version: Literal["0.17.1", "0.18.0"]
    snapshot_digest: SnapshotDigest
    complete: bool
    explicit_extract_status: Literal["not_run"]
    explicit_semantic_verify_status: Literal["not_run"]
    explicit_context_mode: Literal["none"]
    protocol_manifest_digest: Sha256Hex
    agent_profile_digest: Sha256Hex
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    execution_model: dict[str, object]
    observation_set_digest: Sha256Hex
    semantic_unit_index_digest: Sha256Hex
    current_source_index_digest: Sha256Hex
    inference_set_digest: Sha256Hex
    incomplete_reasons: list[str]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def versions_and_digest(self) -> CurrentSourceOnlyRun:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("current-source-only run version does not match package")
        if self.incomplete_reasons != sorted(set(self.incomplete_reasons)):
            raise ValueError("current-source-only incomplete reasons are not canonical")
        if self.artifact_digest != model_payload_digest(
            self,
            digest_field="artifact_digest",
        ):
            raise ValueError("incorrect current-source-only run digest")
        return self


@dataclass(frozen=True)
class CurrentSourceOnlyResult:
    source_acquisition: SourceAcquisitionBundle
    run_manifest: CurrentSourceOnlyRun
    observation: ObservationBundle
    semantic_units: SemanticUnitBundle
    current_source: CurrentSourceBundle
    inference_index: InferenceIndex
    inference_records: tuple[InferenceRecord, ...]
    protocol_manifest: ProtocolManifest
    output_paths: dict[str, Path]


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_external_paths(
    repository_root: Path,
    acquisition: SourceAcquisitionBundle,
    output_directory: Path,
    *,
    observation_cache: Path | None,
    inference_cache: Path | None,
    resume_from: Path | None,
) -> None:
    resolved_root = repository_root.resolve()
    resolved_acquisition = acquisition.directory.resolve()
    resolved_output = output_directory.resolve()
    if output_directory.is_symlink():
        raise ValueError("generated run directory cannot be a symlink")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: {output_directory}"
        )
    if _paths_overlap(resolved_output, resolved_root):
        raise ValueError("generated run directory must be outside the target repository")
    if _paths_overlap(resolved_acquisition, resolved_root):
        raise ValueError("source acquisition must be outside the target repository")
    if _paths_overlap(resolved_output, resolved_acquisition):
        raise ValueError("generated run directory must be outside source acquisition")
    for label, path in (
        ("observation cache", observation_cache),
        ("inference cache", inference_cache),
        ("resume run", resume_from),
    ):
        if path is None:
            continue
        resolved = path.resolve()
        if _paths_overlap(resolved, resolved_root):
            raise ValueError(f"{label} must be outside the target repository")
        if _paths_overlap(resolved, resolved_output):
            raise ValueError(f"{label} must be outside the generated run directory")
        if _paths_overlap(resolved, resolved_acquisition):
            raise ValueError(f"{label} must be outside source acquisition")


def _observation_coverage(observation: ObservationBundle) -> ObservationCoverage:
    return ObservationCoverage(
        mode="foundational",
        overall=observation.index.overall,
        queries=[
            ObservationQueryCoverage(
                name=item.name,
                version=item.version,
                required=item.required,
                status=item.status,
                incomplete_reasons=item.incomplete_reasons,
            )
            for item in observation.index.query_plan
        ],
        observation_set_digest=observation.index.observation_set_digest,
    )


def build_current_source_only(
    repository_root: Path,
    config: BuildConfig,
    acquisition: SourceAcquisitionBundle,
    client: LLMClient,
    output_directory: Path,
    *,
    observation_cache: Path | None = None,
    inference_cache: Path | None = None,
    resume_from: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> CurrentSourceOnlyResult:
    """Build the current-source channel without invoking either explicit stage."""

    if not config.observation.enabled or not config.semantic.enabled:
        raise ValueError(
            "build-current-source requires observation.enabled and semantic.enabled"
        )
    _validate_external_paths(
        repository_root,
        acquisition,
        output_directory,
        observation_cache=observation_cache,
        inference_cache=inference_cache,
        resume_from=resume_from,
    )
    validate_acquisition_for_build(acquisition, config)
    model_config_errors = execution_model_config_errors(
        config,
        acquisition.model_profile,
    )
    if model_config_errors:
        raise ValueError("; ".join(model_config_errors))
    resolved_root = repository_root.resolve()
    discovery = _verify_discovery(resolved_root, config, acquisition.discovery)
    protocol_manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest=acquisition.acquisition.agent_profile_sha256,
    )
    if (
        protocol_manifest.execution_model_profile_id
        != acquisition.model_profile.profile_id
        or protocol_manifest.execution_model_profile_digest
        != acquisition.model_profile.profile_digest
    ):
        raise ValueError(
            "protocol execution-model profile differs from acquisition"
        )
    success_cache = (
        SuccessInferenceCache(
            inference_cache,
            resume_from=resume_from,
            repository_root=resolved_root,
            max_entry_bytes=config.llm.max_inference_artifact_bytes,
            max_resume_entries=config.llm.max_total_calls,
        )
        if inference_cache is not None or resume_from is not None
        else None
    )
    budget = InferenceBudget(
        config.llm,
        cache=success_cache,
        progress_callback=progress_callback,
    )
    tracking_client = _TrackingClient(client)
    observation = run_observations(
        resolved_root,
        discovery,
        config.observation,
        cache_directory=observation_cache,
    ).bundle
    semantic_units = build_semantic_units(
        resolved_root,
        discovery,
        observation,
        config.semantic,
    )
    current_source = run_current_source_pipeline(
        repository_root=resolved_root,
        discovery=discovery,
        observations=observation,
        semantic_units=semantic_units,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        client=tracking_client,
        budget=budget,
        protocol_manifest_digest=protocol_manifest.protocol_manifest_digest,
        explicit_context=(),
    )
    # Reject a snapshot that changed after the initial verification even if the
    # changed file was not needed by a particular semantic packet.
    _verify_discovery(resolved_root, config, acquisition.discovery)
    inference_records = tuple(current_source.inference_records)
    if any(
        item.stage not in CURRENT_SOURCE_INFERENCE_STAGES
        for item in inference_records
    ):
        raise ValueError("current-source-only run contains a non-current inference stage")
    indexed_inference_ids = current_source.index.inference_ids
    recorded_inference_ids = {item.request_id for item in inference_records}
    if (
        len(indexed_inference_ids) != len(set(indexed_inference_ids))
        or set(indexed_inference_ids) != recorded_inference_ids
    ):
        raise ValueError(
            "current-source inference IDs differ from recorded inference set"
        )
    inference_text_by_id = {
        item.request_id: stable_json_text(item) for item in inference_records
    }
    budget.finalize_usage_overrun()
    inference_index = make_inference_index(
        snapshot_digest=discovery.repository.snapshot_digest,
        protocol_manifest=protocol_manifest,
        records=inference_records,
        artifact_sha256_by_id={
            request_id: sha256_hex(text.encode("utf-8"))
            for request_id, text in inference_text_by_id.items()
        },
        total_artifact_bytes=sum(
            len(text.encode("utf-8")) for text in inference_text_by_id.values()
        ),
        incomplete_reasons=sorted(budget.exhausted_reasons),
        stage_omissions=budget.stage_omissions,
    )
    incomplete_reasons = sorted(
        {
            *observation.index.incomplete_reasons,
            *semantic_units.index.incomplete_reasons,
            *current_source.index.incomplete_reasons,
            *inference_index.incomplete_reasons,
        }
    )
    complete = (
        observation.index.overall == "complete"
        and semantic_units.index.status == "complete"
        and current_source.index.status == "complete"
        and not incomplete_reasons
    )
    manifest_payload = {
        "run_version": CURRENT_SOURCE_ONLY_RUN_VERSION,
        "run_kind": "current_source_only",
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "snapshot_digest": discovery.repository.snapshot_digest,
        "complete": complete,
        "explicit_extract_status": "not_run",
        "explicit_semantic_verify_status": "not_run",
        "explicit_context_mode": "none",
        "protocol_manifest_digest": protocol_manifest.protocol_manifest_digest,
        "agent_profile_digest": acquisition.acquisition.agent_profile_sha256,
        "execution_model_profile_id": acquisition.model_profile.profile_id,
        "execution_model_profile_digest": acquisition.model_profile.profile_digest,
        "execution_model": execution_model_run_summary(
            config,
            acquisition,
            inference_records,
        ),
        "observation_set_digest": observation.index.observation_set_digest,
        "semantic_unit_index_digest": semantic_units.index.index_digest,
        "current_source_index_digest": current_source.index.index_digest,
        "inference_set_digest": inference_index.inference_set_digest,
        "incomplete_reasons": incomplete_reasons,
    }
    manifest_payload["artifact_digest"] = payload_digest(manifest_payload)
    run_manifest = CurrentSourceOnlyRun.model_validate(manifest_payload)
    output_paths = write_current_source_only_outputs(
        output_directory,
        run_manifest=run_manifest,
        config=config,
        observation=observation,
        semantic_units=semantic_units,
        current_source=current_source,
        inference_index=inference_index,
        inference_records=inference_records,
        source_acquisition=acquisition,
    )
    return CurrentSourceOnlyResult(
        source_acquisition=acquisition,
        run_manifest=run_manifest,
        observation=observation,
        semantic_units=semantic_units,
        current_source=current_source,
        inference_index=inference_index,
        inference_records=inference_records,
        protocol_manifest=protocol_manifest,
        output_paths=output_paths,
    )


def _read_bounded_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read one untrusted artifact without crossing its configured byte bound."""

    if max_bytes < 0:
        raise ValueError("artifact byte bound is exhausted")
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"artifact exceeds configured byte bound: {path.name}")
    return content


def current_source_inference_record_bound(config: BuildConfig) -> int:
    """Return a provider-call-independent bound for logical current records.

    This remains a conservative tree bound independent of provider cost:
    cache hits and map/reduce terminal jobs still emit records, while package
    Current classification/falsification may compactly omit an exhausted suffix.
    The deterministic stage-plan universes therefore remain the safe upper
    bound rather than ``max_total_calls``.
    """

    if not config.semantic.enabled:
        return 0
    settings = config.semantic
    # Every packet has at least one reconstructed byte and map batches are
    # non-empty, so the total packet-byte budget also bounds map records.
    map_records = settings.max_total_packet_bytes
    # Initial reducer inputs are bounded by observations plus one fallback map
    # result per unit.  Recursive partition/component/repository waves consume
    # at most five non-empty jobs per initial input, including the two cap=1
    # stall rounds.
    reduce_input_universe = (
        settings.max_total_observations + settings.max_units
    )
    reduce_records = 5 * reduce_input_universe
    # Classification batches are non-empty and the member universe is globally
    # bounded across roots and refinements.
    classification_records = settings.max_total_classified_population_members
    # Falsification calls have an independent run-wide hard bound.
    falsification_records = settings.max_total_falsification_calls
    return (
        map_records
        + reduce_records
        + classification_records
        + falsification_records
    )


def validate_current_inference_record_method(
    record: InferenceRecord,
    *,
    protocol_manifest: ProtocolManifest,
    errors: list[str],
) -> None:
    """Bind every replayed current-stage record to the resolved run method."""

    method = _CURRENT_STAGE_METHODS.get(record.stage)
    if method is None:
        errors.append(
            f"{record.request_id}: unknown current-stage method {record.stage}"
        )
        return
    protocol_version, response_schema_version, system_prompt, allowed_modes = method
    if (
        protocol_manifest.stage_protocol_versions.get(record.stage)
        != protocol_version
    ):
        errors.append(
            f"{record.request_id}: protocol manifest omits current stage method"
        )
    if record.protocol_version != protocol_version:
        errors.append(f"{record.request_id}: current-stage protocol version differs")
    if record.response_schema_version != response_schema_version:
        errors.append(f"{record.request_id}: current-stage response schema differs")
    if record.mode not in allowed_modes:
        errors.append(f"{record.request_id}: current-stage mode differs")
    if record.system_prompt_digest != hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest():
        errors.append(f"{record.request_id}: current-stage system prompt differs")
    if record.model_parameters != protocol_manifest.model_parameters:
        errors.append(f"{record.request_id}: current-stage model bounds differ")
    if record.schema_repairs != protocol_manifest.schema_repairs:
        errors.append(
            f"{record.request_id}: current-stage schema-repair bound differs"
        )
    if record.output_limit_retries != protocol_manifest.output_limit_retries:
        errors.append(
            f"{record.request_id}: current-stage output-limit-retry bound differs"
        )
    if record.provider_retries != protocol_manifest.provider_retries:
        errors.append(
            f"{record.request_id}: current-stage provider-retry bound differs"
        )


def _read_inference_tree(
    run_directory: Path,
    *,
    snapshot_digest: str,
    protocol_manifest: ProtocolManifest,
    max_record_bytes: int,
    max_logical_records: int,
    errors: list[str],
) -> tuple[InferenceIndex | None, tuple[InferenceRecord, ...]]:
    max_total_artifact_bytes = max_record_bytes * max_logical_records
    root = run_directory / "inference"
    if root.is_symlink() or not root.is_dir():
        errors.append("inference/ is missing or a symlink")
        return None, ()
    index_path = root / "index.json"
    try:
        if index_path.is_symlink() or not index_path.is_file():
            raise OSError("index is missing or a symlink")
        content = _read_bounded_bytes(
            index_path,
            max_bytes=max_record_bytes,
        )
        index = InferenceIndex.model_validate_json(content)
        if content != canonical_json_bytes(index):
            errors.append("inference/index.json is not canonical JSON")
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as error:
        errors.append(f"inference/index.json error: {error}")
        return None, ()
    if index.snapshot_digest != snapshot_digest:
        errors.append("inference index snapshot differs from channel run")
    if index.protocol_manifest != protocol_manifest:
        errors.append("inference protocol manifest differs from current config")
    if len(index.requests) > max_logical_records:
        errors.append(
            "inference request count exceeds configured logical-record bound"
        )
        return index, ()
    if index.total_artifact_bytes > max_total_artifact_bytes:
        errors.append(
            "inference artifact bytes exceed configured record-tree bound"
        )
        return index, ()
    request_root = root / "requests"
    if request_root.is_symlink() or not request_root.is_dir():
        errors.append("inference/requests/ is missing or a symlink")
        return index, ()
    expected_files = {f"{item.request_id}.json" for item in index.requests}
    actual_files: set[str] = set()
    records: list[InferenceRecord] = []
    bytes_by_id: dict[str, bytes] = {}
    entry_count = 0
    for path in request_root.iterdir():
        entry_count += 1
        if entry_count > max_logical_records:
            errors.append(
                "inference request tree exceeds configured logical-record bound"
            )
            return index, ()
        if path.is_symlink() or not path.is_file():
            errors.append(f"unsafe inference request artifact: {path.name}")
            continue
        actual_files.add(path.name)
    total_bytes = 0
    for file_name in sorted(expected_files & actual_files):
        path = request_root / file_name
        try:
            content = _read_bounded_bytes(
                path,
                max_bytes=max_record_bytes,
            )
            total_bytes += len(content)
            if total_bytes > max_total_artifact_bytes:
                errors.append(
                    "inference artifact bytes exceed configured record-tree bound"
                )
                return index, ()
            record = InferenceRecord.model_validate_json(content)
            if content != canonical_json_bytes(record):
                errors.append(f"inference request is not canonical: {path.name}")
            if path.name != f"{record.request_id}.json":
                errors.append(f"inference request filename mismatch: {path.name}")
            if record.snapshot_digest != snapshot_digest:
                errors.append(f"{record.request_id}: snapshot mismatch")
            if record.protocol_manifest_digest != protocol_manifest.protocol_manifest_digest:
                errors.append(f"{record.request_id}: protocol manifest mismatch")
            if record.stage not in CURRENT_SOURCE_INFERENCE_STAGES:
                errors.append(f"{record.request_id}: non-current inference stage")
            validate_current_inference_record_method(
                record,
                protocol_manifest=protocol_manifest,
                errors=errors,
            )
            records.append(record)
            bytes_by_id[record.request_id] = content
        except (OSError, UnicodeDecodeError, ValidationError, ValueError) as error:
            errors.append(f"inference request error {path.name}: {error}")
    if expected_files != actual_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"missing inference request artifacts: {missing}")
        if extra:
            errors.append(f"unexpected inference request artifacts: {extra}")
    if len({item.request_id for item in records}) != len(records):
        errors.append("duplicate inference request IDs")
    refs = {item.request_id: item for item in index.requests}
    for record in records:
        reference = refs.get(record.request_id)
        content = bytes_by_id[record.request_id]
        if reference is None:
            continue
        if (
            reference.stage != record.stage
            or reference.terminal_status != record.terminal_status
            or reference.record_digest != record.record_digest
            or reference.artifact_sha256 != sha256_hex(content)
        ):
            errors.append(f"{record.request_id}: inference index reference mismatch")
    try:
        replayed_index = make_inference_index(
            snapshot_digest=snapshot_digest,
            protocol_manifest=protocol_manifest,
            records=records,
            artifact_sha256_by_id={
                item.request_id: sha256_hex(bytes_by_id[item.request_id])
                for item in records
            },
            total_artifact_bytes=sum(len(bytes_by_id[item.request_id]) for item in records),
            incomplete_reasons=index.incomplete_reasons,
            stage_omissions={
                item.stage: (item.omitted, item.omission_reasons)
                for item in index.stages
                if item.omitted
            },
        )
    except (KeyError, ValueError) as error:
        errors.append(f"cannot replay inference index: {error}")
    else:
        if replayed_index != index:
            errors.append("inference index differs from request replay")
    actual_root = {item.name for item in root.iterdir()}
    if actual_root != {"index.json", "requests"}:
        errors.append("unexpected inference tree entries")
    return index, tuple(sorted(records, key=lambda item: item.request_id))


def validate_current_source_only_run(
    run_directory: Path,
    *,
    repository_root: Path,
    require_complete: bool = False,
) -> list[str]:
    """Provider-free validation for one current-source-only channel run."""

    errors: list[str] = []
    if run_directory.is_symlink():
        return ["run directory cannot be a symlink"]
    run_directory = run_directory.resolve()
    repository_root = repository_root.resolve()
    if not run_directory.is_dir():
        return [f"run directory does not exist: {run_directory}"]
    if not repository_root.is_dir():
        return [f"expected repository does not exist: {repository_root}"]
    expected_top_level = {
        "agent",
        "channel_run.json",
        "config.json",
        "current-source",
        "discovery.json",
        "inference",
        "model",
        "observations",
        "source_acquisition.json",
        "source_supplement.json",
    }
    actual_top_level = {item.name for item in run_directory.iterdir()}
    for item in run_directory.iterdir():
        if item.is_symlink():
            errors.append(f"top-level artifact is a symlink: {item.name}")
    if actual_top_level != expected_top_level:
        missing = sorted(expected_top_level - actual_top_level)
        extra = sorted(actual_top_level - expected_top_level)
        if missing:
            errors.append(f"missing current-source-only artifacts: {missing}")
        if extra:
            errors.append(f"unexpected current-source-only artifacts: {extra}")
    try:
        manifest_path = run_directory / "channel_run.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise OSError("manifest is missing or a symlink")
        manifest_bytes = manifest_path.read_bytes()
        manifest = CurrentSourceOnlyRun.model_validate_json(manifest_bytes)
        if manifest_bytes != canonical_json_bytes(manifest):
            errors.append("channel_run.json is not canonical JSON")
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        return [*errors, f"channel_run.json error: {error}"]
    try:
        config_path = run_directory / "config.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise OSError("config is missing or a symlink")
        config_bytes = config_path.read_bytes()
        config = BuildConfig.model_validate_json(config_bytes)
        if config_bytes != canonical_json_bytes(config):
            errors.append("config.json is not canonical JSON")
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        return [*errors, f"config.json error: {error}"]
    if not config.observation.enabled or not config.semantic.enabled:
        errors.append("current-source-only config disables a required channel stage")
        return errors
    try:
        acquisition = load_source_acquisition(
            run_directory,
            expected_stage_settings=config.agent.stages["source_discovery"],
        )
        validate_acquisition_for_build(acquisition, config)
    except SourceAcquisitionError as error:
        errors.extend(error.errors)
        return sorted(set(errors))
    errors.extend(execution_model_config_errors(config, acquisition.model_profile))
    try:
        discovery = _verify_discovery(
            repository_root,
            config,
            acquisition.discovery,
        )
    except (OSError, ValueError) as error:
        errors.append(f"snapshot replay failed: {error}")
        return sorted(set(errors))
    protocol_manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest=acquisition.acquisition.agent_profile_sha256,
    )
    protocol_manifest = protocol_manifest_for_readonly_replay(
        protocol_manifest,
        producer_package_version=manifest.package_version,
    )
    if manifest.snapshot_digest != discovery.repository.snapshot_digest:
        errors.append("channel run snapshot differs from discovery")
    if manifest.protocol_manifest_digest != protocol_manifest.protocol_manifest_digest:
        errors.append("channel run protocol manifest differs from config")
    if manifest.agent_profile_digest != acquisition.acquisition.agent_profile_sha256:
        errors.append("channel run agent profile differs from acquisition")
    if (
        manifest.execution_model_profile_id != acquisition.model_profile.profile_id
        or manifest.execution_model_profile_digest
        != acquisition.model_profile.profile_digest
    ):
        errors.append("channel run execution-model profile differs from acquisition")
    if (
        protocol_manifest.execution_model_profile_id
        != acquisition.model_profile.profile_id
        or protocol_manifest.execution_model_profile_digest
        != acquisition.model_profile.profile_digest
    ):
        errors.append("channel protocol execution-model profile differs from acquisition")
    inference_index, inference_records = _read_inference_tree(
        run_directory,
        snapshot_digest=manifest.snapshot_digest,
        protocol_manifest=protocol_manifest,
        max_record_bytes=config.llm.max_inference_artifact_bytes,
        max_logical_records=current_source_inference_record_bound(config),
        errors=errors,
    )
    errors.extend(
        execution_model_inference_errors(
            inference_records,
            acquisition.model_profile,
        )
    )
    if manifest.execution_model != execution_model_run_summary(
        config,
        acquisition,
        inference_records,
    ):
        errors.append("channel run execution model summary is inconsistent")
    try:
        observation = run_observations(
            repository_root,
            discovery,
            config.observation,
            cache_directory=None,
        ).bundle
    except (OSError, ValueError) as error:
        errors.append(f"observation replay failed: {error}")
        return sorted(set(errors))
    errors.extend(
        validate_observation_tree(
            run_directory,
            repository_root=repository_root,
            discovery=discovery,
            settings=config.observation,
            coverage=_observation_coverage(observation),
            semantic_units_enabled=True,
        )
    )
    errors.extend(
        validate_semantic_unit_tree(
            run_directory,
            repository_root=repository_root,
            discovery=discovery,
            observations=observation,
            settings=config.semantic,
        )
    )
    try:
        semantic_units = build_semantic_units(
            repository_root,
            discovery,
            observation,
            config.semantic,
        )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"semantic-unit replay failed: {error}")
        return sorted(set(errors))
    current_errors, current_source = validate_current_source_tree(
        run_directory,
        repository_root=repository_root,
        discovery=discovery,
        observations=observation,
        semantic_units=semantic_units,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        protocol_manifest_digest=protocol_manifest.protocol_manifest_digest,
        inference_records=inference_records,
        explicit_context=(),
        expected_stage_omissions=(
            {
                item.stage: (item.omitted, item.omission_reasons)
                for item in inference_index.stages
                if item.stage in CURRENT_SOURCE_INFERENCE_STAGES and item.omitted
            }
            if inference_index is not None
            else None
        ),
    )
    errors.extend(current_errors)
    if manifest.observation_set_digest != observation.index.observation_set_digest:
        errors.append("channel run observation digest differs from replay")
    if manifest.semantic_unit_index_digest != semantic_units.index.index_digest:
        errors.append("channel run semantic-unit digest differs from replay")
    if inference_index is not None and (
        manifest.inference_set_digest != inference_index.inference_set_digest
    ):
        errors.append("channel run inference digest differs from replay")
    if current_source is not None and (
        manifest.current_source_index_digest != current_source.index.index_digest
    ):
        errors.append("channel run current-source digest differs from replay")
    if current_source is not None:
        indexed_inference_ids = current_source.index.inference_ids
        recorded_inference_ids = {item.request_id for item in inference_records}
        if (
            len(indexed_inference_ids) != len(set(indexed_inference_ids))
            or set(indexed_inference_ids) != recorded_inference_ids
        ):
            errors.append(
                "current-source inference IDs differ from recorded inference set"
            )
    expected_incomplete = sorted(
        {
            *observation.index.incomplete_reasons,
            *semantic_units.index.incomplete_reasons,
            *(
                current_source.index.incomplete_reasons
                if current_source is not None
                else []
            ),
            *(
                inference_index.incomplete_reasons
                if inference_index is not None
                else []
            ),
        }
    )
    expected_complete = (
        observation.index.overall == "complete"
        and semantic_units.index.status == "complete"
        and current_source is not None
        and current_source.index.status == "complete"
        and not expected_incomplete
    )
    if manifest.incomplete_reasons != expected_incomplete:
        errors.append("channel run incomplete reasons differ from replay")
    if manifest.complete != expected_complete:
        errors.append("channel run complete status differs from replay")
    if require_complete and not manifest.complete:
        errors.append("current-source-only run is incomplete")
    return sorted(set(errors))
