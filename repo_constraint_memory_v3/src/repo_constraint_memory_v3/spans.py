"""Stable physical-line spans and deterministic exact evidence recovery."""

from __future__ import annotations

import hashlib

from repo_constraint_memory_v3.models import (
    CandidateEvidenceRange,
    EvidenceSpan,
    EvidenceUnit,
)


class SpanRecoveryError(ValueError):
    """A model-selected range cannot be recovered from its evidence unit."""


def make_span_id(
    source_content_sha256: str,
    line_number: int,
    exact_line_text: str,
) -> str:
    payload = (
        source_content_sha256
        + "\0"
        + str(line_number)
        + "\0"
        + exact_line_text
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"SPAN-{digest[:24]}"


def physical_line_spans(
    content: str,
    *,
    source_content_sha256: str,
    line_start: int,
) -> list[EvidenceSpan]:
    lines = content.splitlines(keepends=True)
    if not lines and content:
        lines = [content]
    return [
        EvidenceSpan(
            span_id=make_span_id(
                source_content_sha256,
                line_start + offset,
                text,
            ),
            line_number=line_start + offset,
            text=text,
        )
        for offset, text in enumerate(lines)
    ]


def recover_span_range(
    selection: CandidateEvidenceRange,
    unit: EvidenceUnit,
    *,
    max_lines: int = 12,
) -> list[EvidenceSpan]:
    if selection.unit_id != unit.unit_id:
        raise SpanRecoveryError("evidence range references a different unit")
    positions = {span.span_id: index for index, span in enumerate(unit.spans)}
    try:
        start = positions[selection.span_start_id]
        end = positions[selection.span_end_id]
    except KeyError as error:
        raise SpanRecoveryError(f"unknown span endpoint: {error.args[0]}") from error
    if end < start:
        raise SpanRecoveryError("span endpoints are reversed")
    selected = unit.spans[start : end + 1]
    if len(selected) > max_lines:
        raise SpanRecoveryError(
            f"evidence range covers {len(selected)} lines; maximum is {max_lines}"
        )
    return selected


def render_unit_for_prompt(unit: EvidenceUnit) -> str:
    return "".join(
        f"{span.span_id} L{span.line_number}: {span.text}"
        + ("" if span.text.endswith("\n") else "\n")
        for span in unit.spans
    ).rstrip("\n")
