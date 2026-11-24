"""Request-bound wire provenance and whole-draft local reducer repair."""

from __future__ import annotations

import json
from collections import Counter
from typing import Annotated, Callable, Sequence

from pydantic import Field, ValidationError

from repo_constraint_memory_v3.inference import (
    InferenceRecord,
    _decode_json_object,
    _is_repeated_invalid_response,
    attach_response_json_schema,
    error_strings,
)
from repo_constraint_memory_v3.models import StrictModel
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.semantic.reduce import (
    MAX_REDUCE_OUTPUT_HYPOTHESES,
    MAX_REDUCE_REPAIR_INVALID_RESPONSE_BYTES,
    HostReducerTransport,
    HypothesisDraft,
    ProviderReductionResponse,
    ReducerTransportStep,
    ReductionResponse,
    ReductionSelectionResponse,
    _ReduceInput,
    _ReductionRepairController,
    _bounded_validation_errors,
    _validate_raw_response,
)


class LocalDraftReplacement(StrictModel):
    ordinal: int = Field(ge=0, lt=MAX_REDUCE_OUTPUT_HYPOTHESES)
    draft: HypothesisDraft


class LocalDraftRepairResponse(StrictModel):
    replacements: Annotated[
        list[LocalDraftReplacement], Field(min_length=1, max_length=MAX_REDUCE_OUTPUT_HYPOTHESES)
    ]


class ReducerWireController:
    """Compose compact transport with the existing bounded selection controller.

    Canonical cache responses never pass through the provider expansion route.
    Saved wire text binds every expansion to actual attempt bytes; a new
    controller independently reconstructs it during cache/validator validation.
    """

    def __init__(
        self,
        inner: _ReductionRepairController,
        inputs: Sequence[_ReduceInput],
        validate: Callable[[ReductionResponse], ReductionResponse],
        terminal_validate: Callable[[ReductionResponse], ReductionResponse],
    ) -> None:
        self.inner = inner
        self.inputs = tuple(inputs)
        self.validate = validate
        self.terminal_validate = terminal_validate
        self.provider_model = ProviderReductionResponse
        self.steps: list[ReducerTransportStep] = []
        self.last_normalized: dict[str, object] | None = None
        self.last_expanded: dict[str, object] | None = None
        self.local_source: dict[str, object] | None = None
        self.failed_ordinals: tuple[int, ...] = ()
        self.local_source_digest: str | None = None
        self.original_request = "{}"
        self.live_validated_response: dict[str, object] | None = None

    def bind_content(self, content: str) -> None:
        self.last_normalized = None
        self.last_expanded = None
        self.live_validated_response = None
        digest = self.inner.current_response_digest
        if digest is None:
            raise ValueError("reducer source content lacks response digest")
        self.steps.append(ReducerTransportStep(raw_response=content, response_digest=digest))

    def raw(self, value: object) -> object:
        return _validate_raw_response(value)

    def _expand(self, value: object) -> object:
        if self.local_source is not None:
            # Validate the envelope/ordinals without parsing the replacement
            # draft yet: malformed leaf drafts must reach terminal quarantine.
            if not isinstance(value, dict) or set(value) != {"replacements"}:
                raise ValueError("local reducer repair requires replacements only")
            replacements = value["replacements"]
            if not isinstance(replacements, list) or any(
                not isinstance(item, dict) or set(item) != {"ordinal", "draft"}
                or type(item["ordinal"]) is not int
                for item in replacements
            ):
                raise ValueError("local reducer repair has invalid replacement structure")
            ordinals = [item["ordinal"] for item in replacements]
            if len(ordinals) != len(set(ordinals)) or set(ordinals) != set(self.failed_ordinals):
                counts = Counter(ordinals)
                missing = sorted(set(self.failed_ordinals) - counts.keys())
                unexpected = sorted(counts.keys() - set(self.failed_ordinals))
                duplicates = sorted(ordinal for ordinal, count in counts.items() if count > 1)
                raise ValueError(
                    "local reducer repair must cover each failed ordinal exactly once; "
                    f"required={list(self.failed_ordinals)}; missing={missing}; "
                    f"unexpected={unexpected[:16]} (count={len(unexpected)}); "
                    f"duplicates={duplicates[:16]} (count={len(duplicates)})"
                )
            source = dict(self.local_source)
            hypotheses = list(source["hypotheses"])
            for replacement in replacements:
                hypotheses[replacement["ordinal"]] = replacement["draft"]
            source["hypotheses"] = hypotheses
            return source
        return value

    def _attach(self, value: object) -> object:
        if not isinstance(value, dict) or "hypotheses" not in value:
            return value
        if not self.steps:
            raise ValueError("compact reducer response lacks bound wire source")
        return {**value, "host_transport": HostReducerTransport(
            steps=self.steps
        ).model_dump(mode="json")}

    def normalize(self, value: object) -> object:
        # Only cache/replay can supply host-owned fields; raw() removes them
        # before live normalization. Provenance is checked independently below.
        if isinstance(value, dict) and value.get("host_transport") is not None:
            return self.inner.normalize(value)
        if self.inner.selection_active:
            normalized = self.inner.normalize(value)
        else:
            expanded = self._expand(value)
            if isinstance(expanded, dict):
                self.last_expanded = expanded
            normalized = self.inner.normalize(expanded)
        if isinstance(normalized, dict):
            self.last_normalized = normalized
        return self._attach(normalized)

    def response_model_for_repair(self, invalid: str, errors: list[str]) -> type[StrictModel]:
        model = self.inner.response_model_for_repair(invalid, errors)
        if model is ReductionSelectionResponse:
            self.local_source = None
            return model
        fallback_model = (
            LocalDraftRepairResponse if self.local_source is not None
            else self.provider_model
        )
        source = self.last_normalized
        raw_source = self.last_expanded
        if source is None or raw_source is None:
            return fallback_model
        hypotheses = source.get("hypotheses")
        if (
            not isinstance(hypotheses, list)
            or not 1 < len(hypotheses) <= self.inner.effective_output_limit
            or source.get("outcome") != "hypotheses"
            or self.inner.validate_overflow_draft is None
        ):
            return fallback_model
        # Prove all non-draft fields before splitting. A damaged envelope never
        # gets repaired by guessing that its siblings are safe.
        try:
            ReductionResponse.model_validate({
                **source, "outcome": "no_hypothesis", "hypotheses": [],
                "hypothesis_ordinals": [], "host_population_repairs": [],
            })
        except (ValidationError, ValueError, TypeError):
            return fallback_model
        failed: list[int] = []
        for ordinal, raw_draft in enumerate(hypotheses):
            try:
                draft = HypothesisDraft.model_validate(raw_draft)
                self.inner.validate_overflow_draft(draft)
            except (ValidationError, ValueError, TypeError):
                failed.append(ordinal)
        if not failed or len(failed) == len(hypotheses):
            return fallback_model
        failed_payload = [{"ordinal": ordinal, "draft": hypotheses[ordinal]} for ordinal in failed]
        if len(json.dumps(failed_payload, ensure_ascii=False).encode("utf-8")) > MAX_REDUCE_REPAIR_INVALID_RESPONSE_BYTES:
            return fallback_model
        # Freeze raw semantic fields, not a regenerated projection. Population
        # normalization is repeated for replacements and unchanged siblings.
        previous_state = (
            self.local_source, self.failed_ordinals, self.local_source_digest
        )
        self.local_source = {
            "outcome": raw_source["outcome"],
            "hypotheses": raw_source["hypotheses"],
            "reason_code": raw_source["reason_code"],
        }
        self.failed_ordinals = tuple(failed)
        self.local_source_digest = self.inner.current_response_digest
        request = self.repair_prompt(self.original_request, invalid, errors)
        if len(attach_response_json_schema(request, LocalDraftRepairResponse).encode("utf-8")) > self.inner.settings.max_reduce_request_bytes:
            self.local_source, self.failed_ordinals, self.local_source_digest = previous_state
            return fallback_model
        return LocalDraftRepairResponse

    def repair_prompt(self, original: str, invalid: str, errors: list[str]) -> str:
        if self.local_source is None:
            return self.inner.repair_prompt(original, invalid, errors)
        validation_errors, omitted = _bounded_validation_errors(errors)
        request = {
            "stage": "current_hypothesis_reduce", "repair": True,
            "repair_type": "local_draft_replacement",
            "original_request": json.loads(original),
            "source_response_digest": self.local_source_digest,
            "fixed_draft_digests": {
                str(ordinal): payload_digest(draft)
                for ordinal, draft in enumerate(self.local_source["hypotheses"])
                if ordinal not in self.failed_ordinals
            },
            "replacements_required": [
                {"ordinal": ordinal, "draft": self.local_source["hypotheses"][ordinal]}
                for ordinal in self.failed_ordinals
            ],
            "validation_errors": validation_errors,
            "validation_errors_omitted": omitted,
            "instruction": (
                "Return one JSON object with only the replacements array; each "
                "item has ordinal and draft, exactly as in the attached "
                "response_json_schema. Cover every ordinal listed in "
                "replacements_required exactly once, including still-failing "
                "ordinals from prior attempts; do not renumber, omit, or duplicate "
                "them. Each replacement is a complete draft, not a field patch. "
                "If no supported correction is available, return that required "
                "draft unchanged; the host keeps it invalid and applies terminal "
                "quarantine only at the existing repair boundary. The host preserves all "
                "other drafts exactly; do not regenerate them. Repair population "
                "and structure from supplied facts; never widen applies_to merely "
                "to include witnesses."
            ),
        }
        return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def terminal_parse(self, value: object, error: Exception) -> ReductionResponse:
        semantic_value = (
            {key: item for key, item in value.items() if key != "host_transport"}
            if isinstance(value, dict) else value
        )
        parsed = self.inner.terminal_parse_response(semantic_value, error)
        return ReductionResponse.model_validate(self._attach(parsed.model_dump(mode="json")))

    def validate_live(self, value: ReductionResponse) -> ReductionResponse:
        validated = self.validate(value)
        if self.steps:
            self.live_validated_response = validated.model_dump(mode="json")
        return validated

    def terminal_validate_live(self, value: ReductionResponse) -> ReductionResponse:
        validated = self.terminal_validate(value)
        if self.steps:
            self.live_validated_response = validated.model_dump(mode="json")
        return validated

    def validate_record(self, value: ReductionResponse, record: InferenceRecord) -> None:
        transport = value.host_transport
        if transport is None:
            raise ValueError("reducer response lacks request-bound wire provenance")
        attempts = [attempt for attempt in record.attempts if (
            not attempt.provider_error and not attempt.retryable
        )]
        if [step.response_digest for step in transport.steps] != [
            attempt.response_digest for attempt in attempts
        ]:
            raise ValueError("reducer compact source trace differs from inference attempts")
        if (
            self.live_validated_response == value.model_dump(mode="json")
            and transport.steps == self.steps
        ):
            # This exact live expansion already passed the full preflight.
            # A fresh cache/replay controller has no such trusted local result.
            return
        inner = _ReductionRepairController(
            mode=self.inner.mode, input_ids=self.inner.input_ids,
            settings=self.inner.settings,
            effective_output_limit=self.inner.effective_output_limit,
            validate_overflow_draft=self.inner.validate_overflow_draft,
        )
        replay = ReducerWireController(inner, self.inputs, self.validate, self.terminal_validate)
        replay.original_request = self.original_request
        normalized: object = None
        parsed: ReductionResponse | None = None
        for ordinal, (step, attempt) in enumerate(zip(transport.steps, attempts)):
            normalized = None
            inner.bind_response_digest(step.response_digest)
            replay.bind_content(step.raw_response)
            error: Exception | None = None
            try:
                normalized = replay.normalize(replay.raw(_decode_json_object(step.raw_response)))
                parsed = ReductionResponse.model_validate(normalized)
                parsed = self.validate(parsed)
            except (ValidationError, ValueError, TypeError) as failure:
                error = failure
            if error is not None and attempt.parsed:
                terminal_allowed = (
                    attempt.schema_attempt == record.schema_repairs
                    or _is_repeated_invalid_response(
                        record.attempts[:record.attempts.index(attempt)],
                        response_digest=step.response_digest,
                        validation_errors=error_strings(error),
                        repair=attempt.repair,
                        schema_attempt=attempt.schema_attempt,
                        enabled=True,
                        early_stop_available=(attempt.schema_attempt < record.schema_repairs),
                    )
                )
                if not terminal_allowed:
                    raise ValueError("reducer quarantine precedes the terminal repair boundary")
                if normalized is None:
                    raise ValueError("terminal compact reducer trace cannot be parsed")
                parsed = replay.terminal_parse(normalized, error)
                parsed = self.terminal_validate(parsed)
                error = None
            if ordinal < len(attempts) - 1:
                if error is None or error_strings(error) != attempt.validation_errors:
                    raise ValueError("reducer compact trace failure differs from source attempt")
                replay.response_model_for_repair(step.raw_response, error_strings(error))
            elif error is not None or parsed is None or not attempt.parsed:
                raise ValueError("reducer compact trace lacks successful terminal expansion")
        if parsed is None or parsed.model_dump(mode="json") != value.model_dump(mode="json"):
            raise ValueError("reducer compact expansion differs from canonical response")
