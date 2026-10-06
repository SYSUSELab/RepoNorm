"""Shared channel replay helpers for schema-4 fixed-snapshot validation."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path

from pydantic import ValidationError

from repo_constraint_memory_v3.adjudication import (
    prepare_candidate_decisions,
)
from repo_constraint_memory_v3.builder import (
    _explicit_verification_inputs,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.chunking import batch_units, chunk_sources
from repo_constraint_memory_v3.current_source_run import (
    CURRENT_SOURCE_INFERENCE_STAGES,
    current_source_inference_record_bound,
    validate_current_inference_record_method,
)
from repo_constraint_memory_v3.discovery import DiscoveryError, load_source_file
from repo_constraint_memory_v3.extraction import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    _validate_recorded_response,
    build_batch_prompt,
    extraction_llm_settings,
    order_results_by_source_plan,
    stable_batch_request_ordinal,
)
from repo_constraint_memory_v3.exporters import (
    stable_json_text,
)
from repo_constraint_memory_v3.explicit_verification import (
    EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
    EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION,
    SYSTEM_PROMPT as EXPLICIT_VERIFICATION_SYSTEM_PROMPT,
    ExplicitVerificationBundle,
    ExplicitVerificationResponse,
    _verification_llm_settings,
    build_explicit_verification_prompt,
    finalize_explicit_verification,
    plan_explicit_verification,
    stable_verification_request_ordinal,
    validate_explicit_verification_response,
)
from repo_constraint_memory_v3.models import (
    BatchExtractionResponse,
    BuildConfig,
    ContainsClaimsResult,
    DiscoveryArtifact,
    NoClaimsResult,
    SourceSupplement,
    UncertainResult,
)
from repo_constraint_memory_v3.inference import (
    InferenceIndex,
    InferenceInputRef,
    InferenceRecord,
    attach_response_json_schema,
    input_manifest_digest,
    make_inference_index,
    protocol_manifest_for_readonly_replay,
    request_identity,
    resolved_model_parameters_payload,
)
from repo_constraint_memory_v3.observations.canonical import sha256_hex
from repo_constraint_memory_v3.source_plan import validate_source_plan
from repo_constraint_memory_v3.unified.semantic import (
    validate_unified_inference_record_method,
)

_UNIT_RESULT_TYPES = (
    ContainsClaimsResult,
    NoClaimsResult,
    UncertainResult,
)
_MAX_VALIDATION_ARTIFACT_BYTES = 100_000_000


def _inference_logical_record_bound(config: BuildConfig) -> int:
    """Bound all logical records independently of provider-call attempts."""

    explicit_extract_records = config.chunking.max_batches
    # Every verifier job is rooted in one valid extracted candidate.  A success
    # record must fit the configured artifact bound, including when replayed
    # from cache, which conservatively bounds candidate count even if each
    # candidate occupied only one byte.
    explicit_verify_records = (
        explicit_extract_records * config.llm.max_inference_artifact_bytes
    )
    # One failed temporal batch can be bisected down to singleton items.  For
    # n items the full binary recovery tree has at most 2n-1 logical records,
    # independent of the initial item/cell batch size.  Each refinement child
    # can execute the same temporal route plus its own round-one verifier.
    temporal_records_per_case = (
        2 * config.unified.max_temporal_items_per_case - 1
    )
    records_per_case = (
        temporal_records_per_case
        + 1
        + config.unified.max_refinement_children
        * (temporal_records_per_case + 1)
    )
    # Direct and closure grouping share max_group_pairs.  In the smallest legal
    # batch each pair can produce its own independently replayable record, so
    # candidate-member count is not a sound record bound for two-phase runs.
    effective_case_bound = (
        config.unified.max_candidate_members
        if config.unified.candidate_grouping_mode == "singleton"
        else config.unified.max_cases
    )
    unified_records = (
        config.unified.max_group_pairs
        + effective_case_bound * records_per_case
    )
    return (
        explicit_extract_records
        + explicit_verify_records
        + current_source_inference_record_bound(config)
        + unified_records
    )


def _read_required(
    path: Path,
    errors: list[str],
    *,
    max_bytes: int = _MAX_VALIDATION_ARTIFACT_BYTES,
) -> str | None:
    relative = path.name
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("artifact is not a regular file")
        if info.st_size > max_bytes:
            raise ValueError("artifact exceeds its byte limit")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("artifact exceeds its byte limit")
        return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        detail = "artifact is a symlink" if (
            isinstance(error, OSError) and error.errno == errno.ELOOP
        ) else str(error)
        errors.append(f"cannot read {relative}: {detail}")
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json_model(
    path: Path,
    model,
    errors: list[str],
    *,
    max_bytes: int = _MAX_VALIDATION_ARTIFACT_BYTES,
):
    text = _read_required(path, errors, max_bytes=max_bytes)
    if text is None:
        return None, None
    try:
        value = model.model_validate_json(text)
    except ValidationError as error:
        errors.append(f"{path.name} schema error: {error}")
        return None, text
    if text != stable_json_text(value):
        errors.append(f"{path.name} is not in canonical stable JSON form")
    return value, text


def _explicit_replay_inputs(
    repository_root: Path,
    discovery: DiscoveryArtifact,
    supplement: SourceSupplement,
    config: BuildConfig,
):
    plan = validate_source_plan(discovery, supplement, config.discovery)
    sources = [
        load_source_file(repository_root, metadata, origin=origin)
        for metadata, origin in plan.sources
    ]
    all_units = chunk_sources(
        sources,
        max_unit_chars=config.chunking.max_unit_chars,
    )
    eligible = []
    for unit in all_units:
        if len(unit.content) > config.chunking.max_batch_chars:
            continue
        eligible.append(unit)
    eligible = eligible[: config.chunking.max_units]
    batches = batch_units(
        eligible,
        max_batch_chars=config.chunking.max_batch_chars,
        max_units_per_batch=config.chunking.max_units_per_batch,
    )[: config.chunking.max_batches]
    return sources, all_units, batches


def _validate_inference_tree(
    run_directory: Path,
    *,
    repository_root: Path,
    discovery: DiscoveryArtifact,
    supplement: SourceSupplement,
    config: BuildConfig,
    agent_profile_digest: str,
    errors: list[str],
) -> tuple[
    InferenceIndex | None,
    list[InferenceRecord],
    dict[str, object],
    list[object],
    list[object],
    ExplicitVerificationBundle | None,
]:
    root = run_directory / "inference"
    if root.is_symlink() or not root.is_dir():
        errors.append("inference/ is missing or a symlink")
        return None, [], {}, [], [], None
    root_names = {item.name for item in root.iterdir()}
    if root_names != {"index.json", "requests"}:
        errors.append(
            "inference tree has missing or unexpected entries: "
            f"{sorted(root_names)}"
        )
    index, _ = _strict_json_model(
        root / "index.json",
        InferenceIndex,
        errors,
        max_bytes=config.llm.max_inference_artifact_bytes,
    )
    if index is None:
        return None, [], {}, [], [], None
    assert isinstance(index, InferenceIndex)
    expected_protocol = protocol_manifest_for_config(
        config,
        agent_profile_digest=agent_profile_digest,
    )
    expected_protocol = protocol_manifest_for_readonly_replay(
        expected_protocol,
        producer_package_version=index.protocol_manifest.package_version,
    )
    if index.protocol_manifest != expected_protocol:
        errors.append("inference protocol manifest differs from resolved config")
    if index.snapshot_digest != discovery.repository.snapshot_digest:
        errors.append("inference index snapshot differs from discovery")
    max_logical_records = _inference_logical_record_bound(config)
    max_record_bytes = config.llm.max_inference_artifact_bytes
    max_tree_bytes = max_logical_records * max_record_bytes
    if len(index.requests) > max_logical_records:
        errors.append(
            "inference request count exceeds configured logical-record bound"
        )
        return index, [], {}, [], [], None
    if index.total_artifact_bytes > max_tree_bytes:
        errors.append(
            "inference artifact bytes exceed configured record-tree bound"
        )
        return index, [], {}, [], [], None
    try:
        sources, all_units, batches = _explicit_replay_inputs(
            repository_root,
            discovery,
            supplement,
            config,
        )
    except (OSError, ValueError, DiscoveryError) as error:
        errors.append(f"cannot reconstruct explicit inference plan: {error}")
        return index, [], {}, [], [], None
    sources_by_path = {item.path: item for item in sources}
    expected_by_id = {}
    for batch in batches:
        ordinal = stable_batch_request_ordinal(batch)
        inputs = [
            InferenceInputRef(
                kind="explicit_unit",
                stable_id=unit.unit_id,
                digest=unit.content_sha256,
                snapshot_digest=discovery.repository.snapshot_digest,
            )
            for unit in batch.units
        ]
        expected_id = request_identity(
            stage="explicit_extract",
            protocol_version=PROMPT_VERSION,
            mode=None,
            inputs=inputs,
            request_ordinal=ordinal,
        )
        expected_by_id[expected_id] = (ordinal, batch, inputs)
    requests = root / "requests"
    if requests.is_symlink() or not requests.is_dir():
        errors.append("inference/requests/ is missing or a symlink")
        return index, [], {}, sources, all_units, None
    actual_files: set[str] = set()
    request_entry_count = 0
    for path in requests.iterdir():
        request_entry_count += 1
        if request_entry_count > max_logical_records:
            errors.append(
                "inference request tree exceeds configured logical-record bound"
            )
            return index, [], {}, sources, all_units, None
        if path.is_symlink() or not path.is_file():
            errors.append(
                f"inference request artifact is not a regular file: {path.name}"
            )
        else:
            actual_files.add(path.name)
    refs_by_id = {item.request_id: item for item in index.requests}
    expected_files = {f"{item}.json" for item in refs_by_id}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"missing inference request artifacts: {missing}")
        if extra:
            errors.append(f"unexpected inference request artifacts: {extra}")
    records: list[InferenceRecord] = []
    artifact_sha: dict[str, str] = {}
    artifact_bytes: dict[str, int] = {}
    results_by_unit: dict[str, object] = {}
    for request_id in sorted(refs_by_id):
        path = requests / f"{request_id}.json"
        record, text = _strict_json_model(
            path,
            InferenceRecord,
            errors,
            max_bytes=config.llm.max_inference_artifact_bytes,
        )
        if record is None or text is None:
            continue
        assert isinstance(record, InferenceRecord)
        records.append(record)
        encoded_record = text.encode("utf-8")
        artifact_sha[request_id] = sha256_hex(encoded_record)
        artifact_bytes[request_id] = len(encoded_record)
        if record.request_id != request_id:
            errors.append(f"{request_id}: inference identity differs from plan")
        if record.protocol_manifest_digest != (
            expected_protocol.protocol_manifest_digest
        ):
            errors.append(f"{request_id}: protocol-manifest digest differs")
        if record.stage in CURRENT_SOURCE_INFERENCE_STAGES:
            if not config.semantic.enabled:
                errors.append(
                    f"{request_id}: current-stage record exists while disabled"
                )
            validate_current_inference_record_method(
                record,
                protocol_manifest=expected_protocol,
                errors=errors,
            )
            continue
        if record.stage in {
            "candidate_semantic_group",
            "temporal_evidence_classify",
            "unified_case_verify",
        }:
            validate_unified_inference_record_method(
                record,
                protocol_manifest=expected_protocol,
                errors=errors,
            )
            continue
        if record.stage == "explicit_semantic_verify":
            continue
        if record.stage != "explicit_extract":
            errors.append(f"{request_id}: unknown inference-stage method")
            continue
        expected = expected_by_id.get(request_id)
        if expected is None:
            errors.append(f"{request_id}: explicit request is outside source plan")
            continue
        ordinal, batch, inputs = expected
        if record.request_ordinal != ordinal:
            errors.append(f"{request_id}: explicit ordinal differs from plan")
        if record.protocol_version != PROMPT_VERSION:
            errors.append(f"{request_id}: prompt protocol version differs")
        if record.response_schema_version != RESPONSE_SCHEMA_VERSION:
            errors.append(f"{request_id}: response schema version differs")
        if record.input_manifest != inputs:
            errors.append(f"{request_id}: input manifest differs from explicit plan")
        if record.input_manifest_digest != input_manifest_digest(inputs):
            errors.append(f"{request_id}: input-manifest digest differs")
        extraction_settings = extraction_llm_settings(config.llm)
        expected_extract_model_parameters = resolved_model_parameters_payload(
            extraction_settings
        )
        if record.model_parameters.model_dump(mode="json") != (
            expected_extract_model_parameters
        ):
            errors.append(f"{request_id}: explicit extraction model bounds differ")
        if record.schema_repairs != extraction_settings.schema_repairs:
            errors.append(
                f"{request_id}: explicit extraction schema-repair bound differs"
            )
        if record.output_limit_retries != (
            extraction_settings.output_limit_retries
        ):
            errors.append(
                f"{request_id}: explicit extraction output-limit-retry bound differs"
            )
        if record.provider_retries != extraction_settings.provider_retries:
            errors.append(
                f"{request_id}: explicit extraction provider-retry bound differs"
            )
        if record.system_prompt_digest != hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest():
            errors.append(f"{request_id}: system-prompt digest differs")
        if record.request_payload_digest != hashlib.sha256(
            attach_response_json_schema(
                build_batch_prompt(batch),
                BatchExtractionResponse,
            ).encode("utf-8")
        ).hexdigest():
            errors.append(f"{request_id}: request-payload digest differs")
        if record.terminal_status == "success":
            try:
                response = record.structured_response
                parsed = _validate_recorded_response(
                    BatchExtractionResponse.model_validate(response),
                    record,
                    batch,
                    sources_by_path=sources_by_path,
                )
            except (ValidationError, ValueError, TypeError) as error:
                errors.append(f"{request_id}: response replay failed: {error}")
            else:
                results_by_unit.update(
                    {item.unit_id: item for item in parsed.results}
                )
    # Request artifacts are traversed by identity above for canonical tree
    # validation. Recover the live builder's source-plan order before any
    # downstream adjudication: request IDs are hashes and are not an ordering
    # contract for candidate relations.
    results_by_unit = order_results_by_source_plan(results_by_unit, batches)

    explicit_record_ids = {
        item.request_id for item in records if item.stage == "explicit_extract"
    }
    if explicit_record_ids != set(expected_by_id):
        errors.append("explicit inference request set differs from source plan")

    verification_bundle: ExplicitVerificationBundle | None = None
    try:
        units_by_id = {item.unit_id: item for item in all_units}
        extraction_request_by_unit = {
            item.stable_id: record.request_id
            for record in records
            if record.stage == "explicit_extract"
            for item in record.input_manifest
            if item.kind == "explicit_unit"
        }
        raw_candidates = [
            (result.unit_id, result.reason_code, candidate)
            for result in results_by_unit.values()
            if isinstance(result, _UNIT_RESULT_TYPES)
            for candidate in result.claims
        ]
        prepared = prepare_candidate_decisions(
            raw_candidates,
            units_by_id=units_by_id,
            sources_by_path={item.path: item for item in sources},
            manifest_paths={item.path for item in discovery.manifest.files},
            snapshot_digest=discovery.repository.snapshot_digest,
            inference_request_by_unit=extraction_request_by_unit,
        )
        verification_plan = plan_explicit_verification(
            _explicit_verification_inputs(prepared)
        )
    except (TypeError, ValueError) as error:
        errors.append(f"cannot reconstruct explicit verification plan: {error}")
        verification_plan = None

    if verification_plan is not None:
        verify_expected_by_id: dict[str, tuple[object, list[InferenceInputRef]]] = {}
        for batch in verification_plan.batches:
            inputs = [
                InferenceInputRef(
                    kind="explicit_grounded_candidate",
                    stable_id=item.candidate_root_id,
                    digest=item.candidate_digest,
                    snapshot_digest=discovery.repository.snapshot_digest,
                )
                for item in batch.candidates
            ]
            expected_id = request_identity(
                stage="explicit_semantic_verify",
                protocol_version=EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
                mode="grounded_candidate_axes",
                inputs=inputs,
                request_ordinal=stable_verification_request_ordinal(batch),
            )
            verify_expected_by_id[expected_id] = (batch, inputs)
        verify_records = {
            item.request_id: item
            for item in records
            if item.stage == "explicit_semantic_verify"
        }
        if set(verify_records) != set(verify_expected_by_id):
            errors.append(
                "explicit verification inference request set differs from "
                "grounded candidate plan"
            )
        responses: dict[str, ExplicitVerificationResponse] = {}
        failures: dict[str, str] = {}
        ordered_verify_records: list[InferenceRecord] = []
        stage_settings = _verification_llm_settings(config.llm)
        expected_model_parameters = resolved_model_parameters_payload(
            stage_settings
        )
        for request_id, (untyped_batch, inputs) in verify_expected_by_id.items():
            batch = untyped_batch
            record = verify_records.get(request_id)
            if record is None:
                failures[batch.batch_id] = "structured_inference_failure"
                continue
            ordered_verify_records.append(record)
            if record.request_ordinal != stable_verification_request_ordinal(
                batch
            ):
                errors.append(f"{request_id}: verifier ordinal differs from plan")
            if record.protocol_version != EXPLICIT_VERIFICATION_PROTOCOL_VERSION:
                errors.append(f"{request_id}: verifier protocol version differs")
            if record.response_schema_version != (
                EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION
            ):
                errors.append(f"{request_id}: verifier response schema differs")
            if record.mode != "grounded_candidate_axes":
                errors.append(f"{request_id}: verifier mode differs")
            if record.input_manifest != inputs:
                errors.append(f"{request_id}: verifier input manifest differs")
            if record.model_parameters.model_dump(mode="json") != (
                expected_model_parameters
            ):
                errors.append(f"{request_id}: verifier model bounds differ")
            if record.schema_repairs != stage_settings.schema_repairs:
                errors.append(f"{request_id}: verifier schema-repair bound differs")
            if record.output_limit_retries != stage_settings.output_limit_retries:
                errors.append(
                    f"{request_id}: verifier output-limit-retry bound differs"
                )
            if record.provider_retries != stage_settings.provider_retries:
                errors.append(f"{request_id}: verifier provider-retry bound differs")
            if record.system_prompt_digest != hashlib.sha256(
                EXPLICIT_VERIFICATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest():
                errors.append(f"{request_id}: verifier system prompt differs")
            if record.request_payload_digest != hashlib.sha256(
                attach_response_json_schema(
                    build_explicit_verification_prompt(batch),
                    ExplicitVerificationResponse,
                ).encode("utf-8")
            ).hexdigest():
                errors.append(f"{request_id}: verifier request payload differs")
            if record.terminal_status == "success":
                try:
                    parsed = validate_explicit_verification_response(
                        ExplicitVerificationResponse.model_validate(
                            record.structured_response
                        ),
                        batch,
                    )
                except (ValidationError, ValueError, TypeError) as error:
                    errors.append(
                        f"{request_id}: verifier response replay failed: {error}"
                    )
                    failures[batch.batch_id] = "structured_inference_failure"
                else:
                    responses[batch.batch_id] = parsed
            else:
                failures[batch.batch_id] = "structured_inference_failure"
        verification_result = finalize_explicit_verification(
            verification_plan,
            responses_by_batch=responses,
            failure_codes_by_batch=failures,
        )
        verification_bundle = ExplicitVerificationBundle(
            plan=verification_plan,
            result=verification_result,
            inference_records=tuple(
                sorted(
                    ordered_verify_records,
                    key=lambda item: item.request_ordinal,
                )
            ),
        )
    for record in records:
        reference = refs_by_id.get(record.request_id)
        if reference is None:
            continue
        if (
            reference.record_digest != record.record_digest
            or reference.artifact_sha256 != artifact_sha[record.request_id]
            or reference.terminal_status != record.terminal_status
            or reference.stage != record.stage
        ):
            errors.append(
                f"{record.request_id}: inference index reference mismatch"
            )
    if index.total_calls != sum(item.usage.calls for item in records):
        errors.append("inference index call total is inconsistent")
    if index.total_calls > config.llm.max_total_calls:
        errors.append("inference index calls exceed configured hard limit")
    if index.total_input_tokens != sum(
        item.usage.input_tokens for item in records
    ):
        errors.append("inference index input-token total is inconsistent")
    if index.total_output_tokens != sum(
        item.usage.output_tokens for item in records
    ):
        errors.append("inference index output-token total is inconsistent")
    expected_reasoning_tokens = (
        sum(item.usage.reasoning_tokens or 0 for item in records)
        if all(item.usage.reasoning_tokens is not None for item in records)
        else None
    )
    if index.total_reasoning_tokens != expected_reasoning_tokens:
        errors.append("inference index reasoning-token total is inconsistent")
    if index.total_latency_ms != sum(
        item.usage.latency_ms for item in records
    ):
        errors.append("inference index latency total is inconsistent")
    if index.total_artifact_bytes != sum(
        artifact_bytes.get(item.request_id, 0) for item in records
    ):
        errors.append("inference index artifact-byte total is inconsistent")
    if sum(artifact_bytes.values()) > max_tree_bytes:
        errors.append(
            "inference artifact bytes exceed configured record-tree bound"
        )
    inference_incomplete_reasons = {
        reason
        for record in records
        for reason in record.incomplete_reasons
    }
    if index.total_input_tokens > config.llm.max_total_input_tokens:
        inference_incomplete_reasons.add("max_total_input_tokens")
    elif index.total_output_tokens > config.llm.max_total_output_tokens:
        inference_incomplete_reasons.add("max_total_output_tokens")
    reconstructed_index = make_inference_index(
        snapshot_digest=discovery.repository.snapshot_digest,
        protocol_manifest=expected_protocol,
        records=records,
        artifact_sha256_by_id=artifact_sha,
        total_artifact_bytes=sum(artifact_bytes.values()),
        incomplete_reasons=sorted(inference_incomplete_reasons),
        stage_omissions={
            item.stage: (item.omitted, item.omission_reasons)
            for item in index.stages
            if item.omitted
        },
    )
    if index != reconstructed_index:
        errors.append("inference index differs from canonical record replay")
    return (
        index,
        records,
        results_by_unit,
        sources,
        all_units,
        verification_bundle,
    )


def validate_run_directory(
    run_directory: Path,
    *,
    repository_root: Path,
    require_complete: bool = False,
) -> list[str]:
    """Validate the schema-4.11 canonical unified package.

    The unified validator reuses the narrow channel replay helpers above.
    There is one public
    acceptance contract for complete builds and ``validate-unified``.
    """

    from repo_constraint_memory_v3.unified.validation import (
        validate_unified_directory,
    )

    return validate_unified_directory(
        run_directory,
        repository_root,
        require_complete=require_complete,
    )
