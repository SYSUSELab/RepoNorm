"""Shared isolated entry point for rebuilding the canonical unified layer."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from repo_constraint_memory_v3.agent_runtime import (
    AgentProfile,
    stable_model_digest,
)
from repo_constraint_memory_v3.adjudication import candidate_fingerprint
from repo_constraint_memory_v3.builder import (
    _TrackingClient,
    _explicit_verification_inputs,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.discovery import (
    DiscoveryArtifact,
    DiscoveryError,
    discover_repository,
)
from repo_constraint_memory_v3.explicit_verification import (
    EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
    EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION,
    SYSTEM_PROMPT as EXPLICIT_VERIFICATION_SYSTEM_PROMPT,
    ExplicitVerificationBatch,
    ExplicitVerificationPlan,
    ExplicitVerificationResponse,
    ExplicitVerificationResult,
    _verification_llm_settings,
    build_explicit_verification_prompt,
    finalize_explicit_verification,
    plan_explicit_verification,
    stable_verification_request_ordinal,
    validate_explicit_verification_response,
)
from repo_constraint_memory_v3.extraction import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    SYSTEM_PROMPT as EXPLICIT_EXTRACTION_SYSTEM_PROMPT,
    extraction_llm_settings,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceIndex,
    InferenceInputRef,
    InferenceRecord,
    InferenceStageCounts,
    ProtocolManifest,
    attach_response_json_schema,
    request_identity,
    resolved_model_parameters_payload,
)
from repo_constraint_memory_v3.inference_cache import SuccessInferenceCache
from repo_constraint_memory_v3.llm import LLMClient
from repo_constraint_memory_v3.models import (
    BatchExtractionResponse,
    BuildConfig,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.observations.models import (
    ObservationIndex,
    PythonFileShard,
)
from repo_constraint_memory_v3.exporters import (
    execution_model_config_errors,
    execution_model_inference_errors,
    execution_model_run_summary,
)
from repo_constraint_memory_v3.current_source_run import (
    validate_current_inference_record_method,
)
from repo_constraint_memory_v3.source_acquisition import (
    SourceAcquisitionBundle,
    SourceAcquisitionError,
    load_source_acquisition,
    validate_acquisition_for_build,
)
from repo_constraint_memory_v3.run_validation import (
    _inference_logical_record_bound,
)
from repo_constraint_memory_v3.unified.export import (
    make_export_inference_index,
    write_unified_outputs,
)
from repo_constraint_memory_v3.unified.models import UnifiedBundle
from repo_constraint_memory_v3.unified.pipeline import (
    UnifiedRunResult,
    run_unified_layer,
)
from repo_constraint_memory_v3.unified.publication import (
    consume_initial_empty_directory,
    directory_tree_fingerprint,
)
from repo_constraint_memory_v3.unified.upstream import (
    MAX_UNIFIED_UPSTREAM_BYTES,
    UnifiedUpstreamInputs,
    _read_regular_bounded,
    _reject_duplicate_keys,
    _reject_non_finite_constant,
    _runtime_inputs,
    load_unified_upstream_artifact,
)
from repo_constraint_memory_v3.semantic.models import (
    FalsificationResult,
    NormHypothesis,
    PopulationArtifact,
    PopulationMemberPacket,
    RefinementRecord,
    SemanticObservation,
)
from repo_constraint_memory_v3.semantic.validation import (
    CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
    CurrentSourceIndex,
    CurrentSourcePolicyDecision,
)
_UNIFIED_INFERENCE_STAGES = {
    "candidate_semantic_group",
    "temporal_evidence_classify",
    "unified_case_verify",
}
_BUDGET_REASON_CODES = {
    "max_total_calls",
    "max_total_input_tokens",
    "max_total_output_tokens",
}
_CURRENT_INFERENCE_STAGES = {
    "current_semantic_map",
    "current_hypothesis_reduce",
    "population_semantic_classify",
    "counterexample_verify",
}
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _merge_rebuilt_stage_omissions(
    upstream_stages: Sequence[InferenceStageCounts],
    replacement_omissions: Mapping[str, tuple[int, Mapping[str, int]]],
) -> dict[str, tuple[int, dict[str, int]]]:
    """Keep channel omissions while replacing all unified-stage omissions."""

    combined = {
        item.stage: (item.omitted, dict(item.omission_reasons))
        for item in upstream_stages
        if item.omitted and item.stage not in _UNIFIED_INFERENCE_STAGES
    }
    for stage, (count, reasons) in replacement_omissions.items():
        previous_count, previous_reasons = combined.get(stage, (0, {}))
        merged_reasons = dict(previous_reasons)
        for reason, reason_count in reasons.items():
            merged_reasons[reason] = (
                merged_reasons.get(reason, 0) + reason_count
            )
        combined[stage] = (
            previous_count + count,
            dict(sorted(merged_reasons.items())),
        )
    return combined


class UnifiedRebuildError(ValueError):
    """The upstream run or caller-authorized bindings are invalid."""

    def __init__(self, errors: str | list[str] | tuple[str, ...]) -> None:
        values = [errors] if isinstance(errors, str) else list(errors)
        self.errors = tuple(sorted(set(values)))
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class UnifiedRebuildResult:
    upstream_inputs: UnifiedUpstreamInputs
    unified: UnifiedRunResult
    inference_records: tuple[InferenceRecord, ...]
    output_paths: dict[str, Path]

    @property
    def bundle(self) -> UnifiedBundle:
        return self.unified.bundle

    @property
    def complete(self) -> bool:
        return self.bundle.completeness.package_status == "complete"


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_paths(
    repository_root: Path,
    upstream_run: Path,
    output_directory: Path,
    *,
    inference_cache: Path | None,
    resume_from: Path | None,
) -> tuple[Path, Path, Path]:
    repository = repository_root.resolve()
    if repository_root.is_symlink() or not repository.is_dir():
        raise UnifiedRebuildError("repository must be a regular directory")
    if upstream_run.is_symlink():
        raise UnifiedRebuildError("upstream run cannot be a symlink")
    try:
        upstream = upstream_run.resolve(strict=True)
    except OSError as error:
        raise UnifiedRebuildError(f"upstream run is unavailable: {error}") from error
    if not upstream.is_dir():
        raise UnifiedRebuildError("upstream run must be a directory")
    if output_directory.is_symlink():
        raise UnifiedRebuildError("unified output cannot be a symlink")
    output = output_directory.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise UnifiedRebuildError(
            "unified output must not exist or must be an empty directory"
        )
    for label, value in (("upstream run", upstream), ("unified output", output)):
        if _overlaps(repository, value):
            raise UnifiedRebuildError(
                f"{label} must be outside the target repository"
            )
    if _overlaps(upstream, output):
        raise UnifiedRebuildError(
            "unified output must be outside the upstream run"
        )
    for label, value in (
        ("inference cache", inference_cache),
        ("resume run", resume_from),
    ):
        if value is None:
            continue
        resolved = value.resolve()
        if _overlaps(repository, resolved) or _overlaps(output, resolved):
            raise UnifiedRebuildError(
                f"{label} must be outside repository and output"
            )
        if label == "inference cache" and _overlaps(upstream, resolved):
            raise UnifiedRebuildError(
                "inference cache must be outside the immutable upstream run"
            )
    return repository, upstream, output


def _load_json_model(path: Path, model: type[BuildConfig | AgentProfile]):
    data = _read_regular_bounded(path, max_bytes=10_000_000)
    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
        return model.model_validate(payload)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
        ValueError,
    ) as error:
        raise UnifiedRebuildError(
            f"invalid upstream binding artifact {path.name}: {error}"
        ) from error


def _load_json_object(path: Path, *, max_bytes: int) -> dict[str, object]:
    data = _read_regular_bounded(path, max_bytes=max_bytes)
    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise UnifiedRebuildError(
            f"invalid upstream binding artifact {path.name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise UnifiedRebuildError(
            f"invalid upstream binding artifact {path.name}: expected object"
        )
    return payload


def _load_artifact_model(
    path: Path,
    model: type[_ModelT],
    *,
    max_bytes: int,
) -> _ModelT:
    payload = _load_json_object(path, max_bytes=max_bytes)
    try:
        return model.model_validate(payload)
    except (ValidationError, ValueError) as error:
        raise UnifiedRebuildError(
            f"invalid upstream channel artifact {path}: {error}"
        ) from error


def _require_closed_directory(
    directory: Path,
    *,
    expected_files: set[str],
    expected_directories: frozenset[str] | set[str] = frozenset(),
) -> None:
    try:
        with os.scandir(directory) as entries:
            actual_files: set[str] = set()
            actual_directories: set[str] = set()
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    actual_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    actual_directories.add(entry.name)
                else:
                    raise UnifiedRebuildError(
                        f"unsafe upstream channel entry: {entry.path}"
                    )
    except FileNotFoundError:
        if not expected_files and not expected_directories:
            return
        raise UnifiedRebuildError(
            f"cannot enumerate upstream channel directory {directory}: missing"
        )
    except OSError as error:
        raise UnifiedRebuildError(
            f"cannot enumerate upstream channel directory {directory}: {error}"
        ) from error
    if (
        actual_files != expected_files
        or actual_directories != expected_directories
    ):
        raise UnifiedRebuildError(
            f"upstream channel directory differs from envelope: {directory}"
        )


def _validate_upstream_channel_bindings(
    upstream: Path,
    artifact: UnifiedUpstreamInputs,
    channel_records: tuple[InferenceRecord, ...],
    *,
    max_entry_bytes: int,
    build_report: Mapping[str, object],
    discovery: DiscoveryArtifact,
) -> None:
    """Bind the isolated envelope to saved channel artifacts without replay.

    This deliberately performs no repository observation, extraction, semantic,
    canonical, or export replay.  It only parses the exact persisted DTOs that
    the envelope projects and verifies their inference/reference commitments.
    """

    verification_payload = _load_json_object(
        upstream / "explicit_verification.json",
        max_bytes=max_entry_bytes,
    )
    if set(verification_payload) != {
        "schema_version",
        "snapshot_digest",
        "plan",
        "result",
    }:
        raise UnifiedRebuildError(
            "upstream explicit-verification wrapper has unexpected fields"
        )
    if (
        verification_payload.get("schema_version") != SCHEMA_VERSION
        or verification_payload.get("snapshot_digest")
        != artifact.snapshot_digest
    ):
        raise UnifiedRebuildError(
            "upstream explicit-verification identity mismatch"
        )
    try:
        verification_plan = ExplicitVerificationPlan.model_validate(
            verification_payload.get("plan")
        )
        verification_result = ExplicitVerificationResult.model_validate(
            verification_payload.get("result")
        )
    except (ValidationError, ValueError) as error:
        raise UnifiedRebuildError(
            f"invalid upstream explicit-verification artifact: {error}"
        ) from error
    current = artifact.current_source
    current_root = upstream / "current-source"
    if current is None:
        if current_root.exists():
            raise UnifiedRebuildError(
                "unified upstream omits existing current-source artifacts"
            )
    else:
        if _load_artifact_model(
            current_root / "index.json",
            CurrentSourceIndex,
            max_bytes=max_entry_bytes,
        ) != current.index:
            raise UnifiedRebuildError(
                "unified upstream current-source index differs from channel artifact"
            )

        current_files: tuple[tuple[str, type[BaseModel], Sequence[BaseModel]], ...] = (
            (
                "semantic-observations/observations",
                SemanticObservation,
                current.semantic_observations,
            ),
            ("hypotheses", NormHypothesis, current.hypotheses),
            ("populations", PopulationArtifact, current.populations),
            ("member-packets", PopulationMemberPacket, current.member_packets),
            ("falsifications", FalsificationResult, current.falsifications),
            ("refinements", RefinementRecord, current.refinements),
        )
        id_fields = (
            "observation_id",
            "hypothesis_id",
            "population_id",
            "packet_id",
            "falsification_id",
            "refinement_id",
        )
        for (directory, model, expected_items), id_field in zip(
            current_files,
            id_fields,
            strict=True,
        ):
            expected_names = {
                f"{getattr(item, id_field)}.json" for item in expected_items
            }
            if directory != "hypotheses":
                _require_closed_directory(
                    current_root / directory,
                    expected_files=expected_names,
                )
            for expected in expected_items:
                stable_id = getattr(expected, id_field)
                actual = _load_artifact_model(
                    current_root / directory / f"{stable_id}.json",
                    model,
                    max_bytes=max_entry_bytes,
                )
                if actual != expected:
                    raise UnifiedRebuildError(
                        "unified upstream current-source DTO differs from "
                        f"channel artifact: {directory}/{stable_id}.json"
                    )

        policy_payload = _load_json_object(
            current_root / "policy-decisions.json",
            max_bytes=max_entry_bytes,
        )
        expected_policy_payload = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": artifact.snapshot_digest,
            "policy_version": CURRENT_SOURCE_EXPERIMENTAL_POLICY_VERSION,
            "decisions": [
                item.model_dump(mode="json")
                for item in current.policy_decisions
            ],
        }
        if policy_payload != expected_policy_payload:
            raise UnifiedRebuildError(
                "unified upstream current-source policy differs from channel artifact"
            )

        failed_population_hypotheses = current.index.resource_use.get(
            "failed_population_hypotheses",
            0,
        )
        count_bindings = {
            "semantic observations": (
                current.index.semantic_observations,
                len(current.semantic_observations),
            ),
            "hypothesis versions": (
                current.index.hypothesis_versions,
                len(current.hypotheses),
            ),
            "unique candidate roots": (
                current.index.unique_candidate_roots,
                len(current.policy_decisions),
            ),
            "materialized populations": (
                current.index.complete_populations
                + current.index.partial_populations
                + current.index.failed_populations
                - failed_population_hypotheses,
                len(current.populations),
            ),
            "classified members": (
                current.index.classified_members,
                sum(len(item.members) for item in current.populations),
            ),
            "refinements": (
                current.index.refinements,
                len(current.refinements),
            ),
            "completed falsifications": (
                current.index.completed_falsifications,
                sum(item.status == "complete" for item in current.falsifications),
            ),
            "experimental dispositions": (
                current.index.experimental_accept_candidates
                + current.index.experimental_review
                + current.index.experimental_rejected,
                len(current.policy_decisions),
            ),
        }
        mismatches = [
            label
            for label, (recorded, projected) in count_bindings.items()
            if recorded != projected
        ]
        if mismatches:
            raise UnifiedRebuildError(
                "unified upstream current-source collection counts differ: "
                + ", ".join(mismatches)
            )

    observations = artifact.observation_shards
    observation_index_path = upstream / "observations" / "index.json"
    if observations is None:
        if observation_index_path.exists():
            raise UnifiedRebuildError(
                "unified upstream omits existing observation artifacts"
            )
    else:
        observation_index = _load_artifact_model(
            observation_index_path,
            ObservationIndex,
            max_bytes=max_entry_bytes,
        )
        _require_closed_directory(
            upstream / "observations" / "python-files",
            expected_files={
                f"{item.file_id}.json" for item in observation_index.file_shards
            },
        )
        shards: list[PythonFileShard] = []
        for reference in observation_index.file_shards:
            shard_path = upstream / reference.artifact_path
            shard_data = _read_regular_bounded(
                shard_path,
                max_bytes=max_entry_bytes,
            )
            if hashlib.sha256(shard_data).hexdigest() != reference.artifact_sha256:
                raise UnifiedRebuildError(
                    f"upstream observation shard ref mismatch: {reference.file_id}"
                )
            try:
                shard = PythonFileShard.model_validate_json(shard_data)
            except (ValidationError, ValueError) as error:
                raise UnifiedRebuildError(
                    f"invalid upstream observation shard {reference.file_id}: {error}"
                ) from error
            if (
                shard.file_id != reference.file_id
                or shard.path != reference.path
                or shard.parse_status != reference.parse_status
            ):
                raise UnifiedRebuildError(
                    f"upstream observation shard identity mismatch: {reference.file_id}"
                )
            shards.append(shard)
        if (
            observation_index.snapshot_digest != observations.snapshot_digest
            or observation_index.observation_set_digest
            != observations.observation_set_digest
            or observation_index.overall != observations.overall
            or observation_index.incomplete_reasons
            != observations.incomplete_reasons
            or shards != observations.shards
        ):
            raise UnifiedRebuildError(
                "unified upstream observations differ from channel artifacts"
            )

    current_record_ids = {
        item.request_id
        for item in channel_records
        if item.stage in _CURRENT_INFERENCE_STAGES
    }
    if current is not None and set(current.index.inference_ids) != current_record_ids:
        raise UnifiedRebuildError(
            "unified upstream current-source inference references differ"
        )
    if current is None and current_record_ids:
        raise UnifiedRebuildError(
            "unified upstream omits current-source inference records"
        )

    raw_candidates: list[tuple[str, str, object]] = []
    for record in channel_records:
        if record.stage != "explicit_extract":
            continue
        record_unit_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "explicit_unit"
        ]
        if record.terminal_status != "success":
            continue
        try:
            response = BatchExtractionResponse.model_validate(
                record.structured_response
            )
        except (ValidationError, ValueError) as error:
            raise UnifiedRebuildError(
                f"invalid upstream explicit extraction response: {error}"
            ) from error
        response_unit_ids = [item.unit_id for item in response.results]
        if (
            len(response_unit_ids) != len(set(response_unit_ids))
            or len(record_unit_ids) != len(set(record_unit_ids))
            or set(response_unit_ids) != set(record_unit_ids)
        ):
            raise UnifiedRebuildError(
                "upstream explicit extraction response differs from its "
                "request input units"
            )
        for result in response.results:
            for candidate in result.claims:
                raw_candidates.append(
                    (
                        result.unit_id,
                        candidate_fingerprint(result.unit_id, candidate),
                        candidate,
                    )
                )
    envelope_candidates = [
        (item.unit_id, item.candidate_fingerprint, item.raw_candidate)
        for item in artifact.explicit_decisions
        if item.raw_candidate is not None
    ]
    candidate_key = lambda item: (item[0], item[1])
    if sorted(raw_candidates, key=candidate_key) != sorted(
        envelope_candidates,
        key=candidate_key,
    ):
        raise UnifiedRebuildError(
            "unified upstream explicit candidates differ from inference artifacts"
        )
    try:
        grounded = _explicit_verification_inputs(
            item.to_runtime() for item in artifact.explicit_decisions
        )
        rebuilt_verification_plan = plan_explicit_verification(grounded)
    except (ValidationError, ValueError) as error:
        raise UnifiedRebuildError(
            f"invalid unified upstream grounded candidates: {error}"
        ) from error
    if rebuilt_verification_plan != verification_plan:
        raise UnifiedRebuildError(
            "unified upstream explicit verification plan differs from grounded "
            "candidates"
        )

    expected_verification_records: dict[
        str, tuple[ExplicitVerificationBatch, list[InferenceInputRef]]
    ] = {}
    request_by_root: dict[str, str] = {}
    for batch in rebuilt_verification_plan.batches:
        inputs = [
            InferenceInputRef(
                kind="explicit_grounded_candidate",
                stable_id=item.candidate_root_id,
                digest=item.candidate_digest,
                snapshot_digest=artifact.snapshot_digest,
            )
            for item in batch.candidates
        ]
        ordinal = stable_verification_request_ordinal(batch)
        request_id = request_identity(
            stage="explicit_semantic_verify",
            protocol_version=EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
            mode="grounded_candidate_axes",
            inputs=inputs,
            request_ordinal=ordinal,
        )
        expected_verification_records[request_id] = (batch, inputs)
        request_by_root.update(
            {item.candidate_root_id: request_id for item in batch.candidates}
        )
    verification_records = {
        item.request_id: item
        for item in channel_records
        if item.stage == "explicit_semantic_verify"
    }
    if set(verification_records) != set(expected_verification_records):
        raise UnifiedRebuildError(
            "upstream explicit verifier request set differs from grounded plan"
        )

    responses: dict[str, ExplicitVerificationResponse] = {}
    failures: dict[str, str] = {}
    for request_id, (batch, inputs) in expected_verification_records.items():
        record = verification_records[request_id]
        if (
            record.request_ordinal != stable_verification_request_ordinal(batch)
            or record.input_manifest != inputs
            or record.request_payload_digest
            != hashlib.sha256(
                attach_response_json_schema(
                    build_explicit_verification_prompt(batch),
                    ExplicitVerificationResponse,
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise UnifiedRebuildError(
                f"upstream explicit verifier request differs from plan: {request_id}"
            )
        if record.terminal_status == "success":
            try:
                response = validate_explicit_verification_response(
                    ExplicitVerificationResponse.model_validate(
                        record.structured_response
                    ),
                    batch,
                )
            except (ValidationError, ValueError, TypeError) as error:
                raise UnifiedRebuildError(
                    f"invalid upstream explicit verifier response: {error}"
                ) from error
            responses[batch.batch_id] = response
        else:
            failures[batch.batch_id] = "structured_inference_failure"
    rebuilt_verification_result = finalize_explicit_verification(
        rebuilt_verification_plan,
        responses_by_batch=responses,
        failure_codes_by_batch=failures,
    )
    if (
        rebuilt_verification_result != verification_result
        or verification_result != artifact.explicit_verification
    ):
        raise UnifiedRebuildError(
            "unified upstream explicit verification result differs from inference "
            "artifacts"
        )

    for decision in artifact.explicit_decisions:
        if decision.raw_candidate is None:
            continue
        root_id = f"EXPL-{decision.candidate_fingerprint[:24]}"
        expected_request = request_by_root.get(root_id)
        if decision.verification_request_id != expected_request:
            raise UnifiedRebuildError(
                "unified upstream explicit verification request binding mismatch"
            )

    processing = build_report.get("processing")
    adjudication = build_report.get("adjudication_funnel")
    if not isinstance(processing, Mapping) or not isinstance(adjudication, Mapping):
        raise UnifiedRebuildError(
            "upstream build report lacks channel binding counts"
        )
    dropped = processing.get("dropped_unit_count")
    failed = processing.get("failed_unit_count")
    explicit_events = adjudication.get("raw_candidate_events")
    if (
        isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or isinstance(failed, bool)
        or not isinstance(failed, int)
        or isinstance(explicit_events, bool)
        or not isinstance(explicit_events, int)
    ):
        raise UnifiedRebuildError(
            "upstream build report channel binding counts are invalid"
        )
    if explicit_events != len(raw_candidates):
        raise UnifiedRebuildError(
            "upstream explicit event count differs from inference artifacts"
        )
    expected_discovery_complete = discovery.complete and dropped == 0 and failed == 0
    if artifact.explicit_discovery_complete != expected_discovery_complete:
        raise UnifiedRebuildError(
            "unified upstream explicit discovery completeness differs from channel"
        )
    current_events = current.index.candidate_events if current is not None else 0
    if artifact.raw_candidate_events != explicit_events + current_events:
        raise UnifiedRebuildError(
            "unified upstream raw candidate count differs from channel artifacts"
        )


def _channel_record_method_errors(
    records: Sequence[InferenceRecord],
    *,
    config: BuildConfig,
    protocol: ProtocolManifest,
) -> list[str]:
    """Apply the existing frozen stage-method identities to channel records."""

    manifest = protocol
    errors: list[str] = []
    extraction_settings = extraction_llm_settings(config.llm)
    verification_settings = _verification_llm_settings(config.llm)
    for record in records:
        if record.protocol_manifest_digest != manifest.protocol_manifest_digest:
            errors.append(f"{record.request_id}: protocol-manifest digest differs")
            continue
        if record.stage in _CURRENT_INFERENCE_STAGES:
            validate_current_inference_record_method(
                record,
                protocol_manifest=manifest,
                errors=errors,
            )
            continue
        if record.stage == "explicit_extract":
            expected = (
                PROMPT_VERSION,
                RESPONSE_SCHEMA_VERSION,
                None,
                EXPLICIT_EXTRACTION_SYSTEM_PROMPT,
                extraction_settings,
            )
        elif record.stage == "explicit_semantic_verify":
            expected = (
                EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
                EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION,
                "grounded_candidate_axes",
                EXPLICIT_VERIFICATION_SYSTEM_PROMPT,
                verification_settings,
            )
        else:
            errors.append(f"{record.request_id}: unknown channel-stage method")
            continue
        protocol_version, response_version, mode, system_prompt, settings = expected
        if record.protocol_version != protocol_version:
            errors.append(f"{record.request_id}: channel protocol version differs")
        if record.response_schema_version != response_version:
            errors.append(f"{record.request_id}: channel response schema differs")
        if record.mode != mode:
            errors.append(f"{record.request_id}: channel mode differs")
        if record.system_prompt_digest != hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest():
            errors.append(f"{record.request_id}: channel system prompt differs")
        if record.model_parameters.model_dump(
            mode="json"
        ) != resolved_model_parameters_payload(settings):
            errors.append(f"{record.request_id}: channel model bounds differ")
        if (
            record.schema_repairs != settings.schema_repairs
            or record.output_limit_retries != settings.output_limit_retries
            or record.provider_retries != settings.provider_retries
        ):
            errors.append(f"{record.request_id}: channel retry bounds differ")
    return errors


def _usage(records: tuple[InferenceRecord, ...]) -> dict[str, int | None]:
    return {
        "record_count": len(records),
        "calls": sum(item.usage.calls for item in records),
        "input_tokens": sum(item.usage.input_tokens for item in records),
        "output_tokens": sum(item.usage.output_tokens for item in records),
        "reasoning_tokens": (
            sum(item.usage.reasoning_tokens or 0 for item in records)
            if all(item.usage.reasoning_tokens is not None for item in records)
            else None
        ),
    }


def _rebuilt_build_report(
    upstream_report: dict[str, object],
    *,
    unified: UnifiedRunResult,
    inference_index: InferenceIndex,
    combined_records: tuple[InferenceRecord, ...],
    channel_records: tuple[InferenceRecord, ...],
    budget: InferenceBudget,
    initial_calls: int,
    observed_max_concurrency: int,
    success_cache: SuccessInferenceCache | None,
    config: BuildConfig,
    source_acquisition: SourceAcquisitionBundle,
) -> dict[str, object]:
    """Replace run-dependent fields while preserving validated channel facts."""

    report = deepcopy(upstream_report)
    bundle = unified.bundle
    delivered = [
        item
        for item in bundle.canonical_claims
        if item.epistemic_status == "accepted"
        and item.delivery_policy != "reference_only"
    ]
    canonical_counts = {
        "accepted": bundle.index.accepted_count,
        "review": bundle.index.review_count,
        "rejected": bundle.index.rejected_count,
        "delivered": len(delivered),
    }
    report["entrypoint"] = "build-unified"
    report["snapshot_digest"] = bundle.index.snapshot_digest
    report["unified"] = {
        "bundle_id": bundle.index.bundle_id,
        "profile_digest": bundle.index.profile_digest,
        "case_count": len(bundle.cases),
        "assessment_count": len(bundle.assessments),
        "case_routes": dict(sorted(unified.case_routes.items())),
        "candidate_grouping": unified.grouping_execution.report_metadata(),
        "git_case_count": sum(
            bool(item.manifest.commands) for item in unified.git_bundles
        ),
        "git_command_count": sum(
            len(item.manifest.command_results) for item in unified.git_bundles
        ),
        "temporal_matrix_count": len(bundle.temporal_matrices),
        "finding_count": len(bundle.findings),
        "refinement_count": len(bundle.refinements),
        "canonical_counts": canonical_counts,
        "completeness": bundle.completeness.model_dump(mode="json"),
    }

    outcomes = report.get("outcomes")
    if not isinstance(outcomes, dict):
        outcomes = {}
        report["outcomes"] = outcomes
    outcomes["merged_propositions"] = canonical_counts
    outcomes["exported_review_items"] = sum(
        item.epistemic_status in {"review", "rejected"}
        for item in bundle.canonical_claims
    )
    adjudication = report.get("adjudication_funnel")
    if isinstance(adjudication, dict):
        adjudication["merged_accepted_propositions"] = canonical_counts["accepted"]
        adjudication["delivered_propositions"] = canonical_counts["delivered"]
    report["scope_audit"] = {
        "global": sum(
            item.semantics.applies_to == ["**"]
            for item in bundle.canonical_claims
        ),
        "local": sum(
            item.semantics.applies_to != ["**"]
            for item in bundle.canonical_claims
        ),
        "origin_channels": sorted(
            {item.channel for item in bundle.assessments}
        ),
    }

    cache_metrics = (
        success_cache.metrics if success_cache is not None else None
    )
    unified_records = tuple(
        item
        for item in combined_records
        if item.stage in _UNIFIED_INFERENCE_STAGES
    )
    report["llm"] = {
        "batch_wall_seconds": round(
            sum(item.usage.latency_ms for item in unified_records) / 1000,
            6,
        ),
        "observed_max_concurrency": observed_max_concurrency,
        "attempt_count": sum(len(item.attempts) for item in combined_records),
        "schema_repairs": sum(
            attempt.repair
            for record in combined_records
            for attempt in record.attempts
        ),
        "provider_retries": sum(
            attempt.provider_try > 0
            for record in combined_records
            for attempt in record.attempts
        ),
        "output_limit_retries": sum(
            attempt.output_limit_retry > 0
            for record in combined_records
            for attempt in record.attempts
        ),
        "input_tokens": inference_index.total_input_tokens,
        "output_tokens": inference_index.total_output_tokens,
        "reasoning_tokens": inference_index.total_reasoning_tokens,
        "record_count": len(inference_index.requests),
        "inference_set_digest": inference_index.inference_set_digest,
        "this_run_provider_calls": budget.calls - initial_calls,
        "this_run_cache_hits": budget.cache_hits,
        "stage_progress": budget.stage_progress,
        "prior_channel_usage": _usage(channel_records),
        "this_run_unified_usage": {
            **_usage(unified_records),
            "provider_calls": budget.calls - initial_calls,
            "cache_hits": budget.cache_hits,
        },
        "cache": {
            "hits": cache_metrics.hits if cache_metrics is not None else 0,
            "misses": cache_metrics.misses if cache_metrics is not None else 0,
            "rejected": cache_metrics.rejected if cache_metrics is not None else 0,
            "writes": cache_metrics.writes if cache_metrics is not None else 0,
            "resume_entries": (
                cache_metrics.resume_entries if cache_metrics is not None else 0
            ),
        },
    }
    report["execution_model"] = execution_model_run_summary(
        config,
        source_acquisition,
        combined_records,
    )
    report["complete"] = bundle.completeness.package_status == "complete"
    report["incomplete_reasons"] = sorted(bundle.completeness.reasons)
    return report


def _validated_channel_inference(
    upstream: Path,
    *,
    expected_snapshot_digest: str,
    expected_profile_digest: str,
    max_entry_bytes: int,
    max_records: int,
) -> tuple[InferenceIndex, tuple[InferenceRecord, ...]]:
    raw_index = _read_regular_bounded(
        upstream / "inference" / "index.json",
        max_bytes=max_entry_bytes,
    )
    try:
        index_payload = json.loads(
            raw_index.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
        index = InferenceIndex.model_validate(index_payload)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
        ValueError,
    ) as error:
        raise UnifiedRebuildError(
            f"invalid upstream inference index: {error}"
        ) from error
    if index.snapshot_digest != expected_snapshot_digest:
        raise UnifiedRebuildError("upstream inference snapshot mismatch")
    if (
        index.protocol_manifest.protocol_manifest_digest
        != expected_profile_digest
    ):
        raise UnifiedRebuildError("upstream inference profile mismatch")
    if len(index.requests) > max_records:
        raise UnifiedRebuildError("upstream inference record bound exceeded")

    records: list[InferenceRecord] = []
    for reference in index.requests:
        path = upstream / reference.artifact_path
        data = _read_regular_bounded(path, max_bytes=max_entry_bytes)
        try:
            payload = json.loads(
                data.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite_constant,
            )
            record = InferenceRecord.model_validate(payload)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValidationError,
            ValueError,
        ) as error:
            raise UnifiedRebuildError(
                f"invalid upstream inference record {reference.request_id}: {error}"
            ) from error
        if (
            record.request_id != reference.request_id
            or record.stage != reference.stage
            or record.terminal_status != reference.terminal_status
            or record.record_digest != reference.record_digest
            or hashlib.sha256(data).hexdigest() != reference.artifact_sha256
        ):
            raise UnifiedRebuildError(
                f"upstream inference ref mismatch: {reference.request_id}"
            )
        records.append(record)
    channel = tuple(
        item for item in records if item.stage not in _UNIFIED_INFERENCE_STAGES
    )
    return index, channel


def _current_discovery(
    repository_root: Path,
    config: BuildConfig,
) -> DiscoveryArtifact:
    return discover_repository(
        repository_root,
        config.discovery,
        repository_name=config.repository.name,
    )


def rebuild_unified(
    repository_root: Path,
    config: BuildConfig,
    agent_profile: AgentProfile,
    upstream_run: Path,
    client: LLMClient,
    output_directory: Path,
    *,
    inference_cache: Path | None = None,
    resume_from: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    consumer_profiles: Sequence[str] = ("accepted_only",),
) -> UnifiedRebuildResult:
    """Load bound channel inputs and rerun the shared canonical pipeline.

    The loader performs the schema, snapshot, profile, protocol, inference, and
    reference checks required by this build operation.  Full artifact replay is
    an independent ``validate`` operation and does not gate rebuilding or
    publication.
    """

    repository, upstream, output = _validate_paths(
        repository_root,
        upstream_run,
        output_directory,
        inference_cache=inference_cache,
        resume_from=resume_from,
    )
    output_preflight = consume_initial_empty_directory(output)
    try:
        upstream_fingerprint = directory_tree_fingerprint(
            upstream,
            max_bytes=config.unified.max_unified_artifact_bytes,
        )
    except ValueError as error:
        raise UnifiedRebuildError(
            f"invalid upstream tree: {error}"
        ) from error
    supplied_agent_digest = stable_model_digest(agent_profile)
    protocol = protocol_manifest_for_config(
        config,
        agent_profile_digest=supplied_agent_digest,
    )
    upstream_config = _load_json_model(upstream / "config.json", BuildConfig)
    if upstream_config != config:
        raise UnifiedRebuildError("build-unified config differs from upstream config")
    upstream_profile = _load_json_model(
        upstream / "agent" / "profile.json", AgentProfile
    )
    if upstream_profile != agent_profile:
        raise UnifiedRebuildError(
            "build-unified agent profile differs from upstream profile"
        )

    before = _current_discovery(repository, config)
    artifact = load_unified_upstream_artifact(
        upstream / "canonical" / "audit" / "upstream-inputs.json",
        expected_snapshot_digest=before.repository.snapshot_digest,
        expected_profile_digest=protocol.protocol_manifest_digest,
        max_bytes=min(
            MAX_UNIFIED_UPSTREAM_BYTES,
            config.unified.max_unified_artifact_bytes,
        ),
    )
    if artifact.snapshot_commit != before.repository.git_head_for_reference:
        raise UnifiedRebuildError(
            "unified upstream commit differs from the current repository"
        )
    upstream_report = _load_json_object(
        upstream / "build_report.json",
        max_bytes=min(
            50_000_000,
            config.unified.max_unified_artifact_bytes,
        ),
    )
    source_acquisition = load_source_acquisition(
        upstream,
        expected_stage_settings=config.agent.stages["source_discovery"],
    )
    try:
        validate_acquisition_for_build(source_acquisition, config)
    except SourceAcquisitionError as error:
        raise UnifiedRebuildError(list(error.errors)) from error
    model_config_errors = execution_model_config_errors(
        config,
        source_acquisition.model_profile,
    )
    if model_config_errors:
        raise UnifiedRebuildError(model_config_errors)
    if (
        protocol.execution_model_profile_id
        != source_acquisition.model_profile.profile_id
        or protocol.execution_model_profile_digest
        != source_acquisition.model_profile.profile_digest
    ):
        raise UnifiedRebuildError(
            "build-unified protocol execution-model profile differs from acquisition"
        )
    if source_acquisition.discovery != before:
        raise UnifiedRebuildError(
            "upstream source acquisition differs from current repository discovery"
        )

    upstream_inference, channel_records = _validated_channel_inference(
        upstream,
        expected_snapshot_digest=artifact.snapshot_digest,
        expected_profile_digest=artifact.profile_digest,
        max_entry_bytes=config.llm.max_inference_artifact_bytes,
        max_records=_inference_logical_record_bound(config),
    )
    if upstream_inference.protocol_manifest != protocol:
        raise UnifiedRebuildError(
            "upstream inference protocol manifest differs from current config"
        )
    record_identity_errors = execution_model_inference_errors(
        channel_records,
        source_acquisition.model_profile,
    )
    record_identity_errors.extend(
        _channel_record_method_errors(
            channel_records,
            config=config,
            protocol=protocol,
        )
    )
    if record_identity_errors:
        raise UnifiedRebuildError(record_identity_errors)
    _validate_upstream_channel_bindings(
        upstream,
        artifact,
        channel_records,
        max_entry_bytes=config.unified.max_unified_artifact_bytes,
        build_report=upstream_report,
        discovery=before,
    )
    initial_calls = sum(item.usage.calls for item in channel_records)
    initial_input_tokens = sum(item.usage.input_tokens for item in channel_records)
    initial_output_tokens = sum(
        item.usage.output_tokens for item in channel_records
    )

    success_cache = (
        SuccessInferenceCache(
            inference_cache,
            resume_from=resume_from,
            repository_root=repository,
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
        initial_calls=initial_calls,
        initial_input_tokens=initial_input_tokens,
        initial_output_tokens=initial_output_tokens,
    )
    runtime = _runtime_inputs(artifact)
    tracking_client = (
        client if isinstance(client, _TrackingClient) else _TrackingClient(client)
    )
    unified = run_unified_layer(
        repository_root=repository,
        settings=config.unified,
        llm_settings=config.llm,
        client=tracking_client,
        budget=budget,
        protocol_manifest_digest=protocol.protocol_manifest_digest,
        current_source_required=(
            config.semantic.enabled and config.semantic.required
        ),
        **runtime,
    )
    if (
        not unified.bundle.completeness.structurally_valid
        or unified.bundle.completeness.package_status == "invalid"
    ):
        raise UnifiedRebuildError(
            [
                "unified pipeline produced a structurally invalid package",
                *unified.bundle.completeness.reasons,
            ]
        )

    after = _current_discovery(repository, config)
    if after != before:
        raise UnifiedRebuildError(
            "repository snapshot changed during build-unified"
        )
    records_by_id = {item.request_id: item for item in channel_records}
    for record in unified.inference_records:
        if record.request_id in records_by_id:
            raise UnifiedRebuildError(
                f"channel/unified inference ID collision: {record.request_id}"
            )
        records_by_id[record.request_id] = record
    combined_records = tuple(
        records_by_id[key] for key in sorted(records_by_id)
    )
    channel_incomplete_reasons = {
        reason
        for record in channel_records
        for reason in record.incomplete_reasons
        if reason in _BUDGET_REASON_CODES
    }
    budget.finalize_usage_overrun()
    inference_incomplete_reasons = sorted(
        channel_incomplete_reasons | budget.exhausted_reasons
    )
    combined_stage_omissions = _merge_rebuilt_stage_omissions(
        upstream_inference.stages,
        budget.stage_omissions,
    )
    rebuilt_inference_index = make_export_inference_index(
        unified.bundle,
        protocol_manifest=protocol,
        inference_records=combined_records,
        incomplete_reasons=inference_incomplete_reasons,
        stage_omissions=combined_stage_omissions,
    )
    build_report = _rebuilt_build_report(
        upstream_report,
        unified=unified,
        inference_index=rebuilt_inference_index,
        combined_records=combined_records,
        channel_records=channel_records,
        budget=budget,
        initial_calls=initial_calls,
        observed_max_concurrency=tracking_client.peak_in_flight,
        success_cache=success_cache,
        config=config,
        source_acquisition=source_acquisition,
    )
    def recheck_publication_inputs() -> None:
        try:
            current = _current_discovery(repository, config)
            current_upstream_fingerprint = directory_tree_fingerprint(
                upstream,
                max_bytes=config.unified.max_unified_artifact_bytes,
            )
        except (DiscoveryError, OSError, ValueError) as error:
            raise UnifiedRebuildError(
                f"build-unified input changed before publication: {error}"
            ) from error
        if current != before:
            raise UnifiedRebuildError(
                "repository snapshot changed before build-unified publication"
            )
        if current_upstream_fingerprint != upstream_fingerprint:
            raise UnifiedRebuildError(
                "upstream run changed before build-unified publication"
            )

    output_paths = write_unified_outputs(
        output,
        progress_callback=progress_callback,
        bundle=unified.bundle,
        git_artifacts=unified.git_artifacts,
        max_artifact_bytes=config.unified.max_unified_artifact_bytes,
        repository_root=repository,
        repository_name=config.repository.name,
        consumer_profiles=consumer_profiles,
        upstream_directory=upstream,
        upstream_inputs=artifact,
        protocol_manifest=protocol,
        inference_records=combined_records,
        inference_incomplete_reasons=inference_incomplete_reasons,
        inference_stage_omissions=combined_stage_omissions,
        build_report=build_report,
        _destination_preflight=output_preflight,
        _prepublication_check=recheck_publication_inputs,
    )
    return UnifiedRebuildResult(
        upstream_inputs=artifact,
        unified=unified,
        inference_records=combined_records,
        output_paths=output_paths,
    )


__all__ = [
    "UnifiedRebuildError",
    "UnifiedRebuildResult",
    "rebuild_unified",
]
