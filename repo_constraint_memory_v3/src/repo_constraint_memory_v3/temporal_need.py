"""Bounded model-owned historical questions; the host only validates bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from repo_constraint_memory_v3.models import StrictModel

GIT_NEED_PROTOCOL_VERSION = "git-need-assessment-1.0"

GitNeedText = Annotated[str, Field(min_length=1, max_length=800)]

GIT_NEED_PROMPT = """
Assess git_need for this exact candidate and current snapshot in this existing
judgment. Use affirmative only for a specific historical question, a concrete
information gap in cited current evidence, and how the answer will be used by
an existing currentness/scope/condition/replacement judgment or evidence record.
Each question names a supplied current path and legal current evidence_refs.
Set blocking_axis to an existing uncertain fact axis, material ambiguity, or
explicit unresolved conflict/question only when its resolution is a prerequisite
for acceptance; otherwise use null (optional augmentation). An established fail
is a resolved negative fact, not an unresolved axis that history can override.
A documented A-to-B migration with an unresolved candidate-specific reversal
relationship can qualify if the question and use are concrete. Migration words
alone, legacy/deprecated words, generic uncertainty, conflicting sources without
a historical question, background, confidence, checking just in case, and facts
already answered by current sources do not qualify. Use no_established_need or
undetermined with questions=[] for those cases. Neither state asserts fact axes
pass nor demands proof that history is absolutely unnecessary. Never emit Git
commands. For a refinement proposal assess the proposed child independently;
do not inherit the parent's need.
"""


class GitNeedQuestion(StrictModel):
    question: GitNeedText
    current_gap: GitNeedText
    answer_use: GitNeedText
    path: Annotated[str, Field(min_length=1, max_length=1024)]
    evidence_refs: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1, max_length=16)]
    blocking_axis: Annotated[str, Field(min_length=1, max_length=80)] | None

    @field_validator("question", "current_gap", "answer_use")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Git need explanatory fields must be nonblank")
        return value


class GitNeedJudgment(StrictModel):
    status: Literal["affirmative", "no_established_need", "undetermined"]
    questions: Annotated[list[GitNeedQuestion], Field(max_length=4)]

    @model_validator(mode="after")
    def affirmative_has_questions(self) -> GitNeedJudgment:
        if bool(self.questions) != (self.status == "affirmative"):
            raise ValueError("only affirmative Git need carries concrete questions")
        return self


class GitNeedAssessment(StrictModel):
    """Contained in a digest-bound candidate/version/snapshot artifact."""

    status: Literal["affirmative", "no_established_need", "undetermined", "not_assessed", "invalid"]
    judgments: Annotated[list[GitNeedJudgment], Field(max_length=100_000)]
    reason: Annotated[str, Field(min_length=1, max_length=160)] | None

    @model_validator(mode="after")
    def truthful_state(self) -> GitNeedAssessment:
        if self.status in {"not_assessed", "invalid"}:
            if self.judgments or self.reason is None:
                raise ValueError("host Git need state requires reason and no judgments")
        else:
            if not self.judgments or self.reason is not None:
                raise ValueError("normal Git need state requires semantic judgments")
            statuses = {item.status for item in self.judgments}
            expected = "affirmative" if "affirmative" in statuses else "undetermined" if "undetermined" in statuses else "no_established_need"
            if self.status != expected:
                raise ValueError("Git need aggregate disagrees with judgments")
            payloads = [item.model_dump_json() for item in self.judgments]
            if payloads != sorted(set(payloads)):
                raise ValueError("Git need judgments must be exactly deduplicated and stable")
        return self


def unassessed_git_need(reason: str, *, invalid: bool = False) -> GitNeedAssessment:
    return GitNeedAssessment(status="invalid" if invalid else "not_assessed", judgments=[], reason=reason)


def aggregate_git_needs(judgments: Sequence[GitNeedJudgment]) -> GitNeedAssessment:
    if not judgments:
        raise ValueError("unexecuted Git need must retain a host-only state")
    unique = {value.model_dump_json(): value for value in judgments}
    values = [unique[key] for key in sorted(unique)]
    statuses = {value.status for value in values}
    status = "affirmative" if "affirmative" in statuses else "undetermined" if "undetermined" in statuses else "no_established_need"
    return GitNeedAssessment(status=status, judgments=values, reason=None)


def validate_git_need(
    judgment: GitNeedJudgment,
    *,
    allowed_refs: set[str],
    allowed_paths: set[str],
    unresolved_axes: set[str],
    reference_paths: Mapping[str, set[str]],
) -> None:
    for question in judgment.questions:
        if not set(question.evidence_refs) <= allowed_refs:
            raise ValueError("Git need cites unsupplied current evidence")
        if len(question.evidence_refs) != len(set(question.evidence_refs)):
            raise ValueError("Git need repeats a current evidence reference")
        if question.path not in allowed_paths:
            raise ValueError("Git need path is not bound to supplied current evidence")
        if not any(question.path in reference_paths.get(ref, set()) for ref in question.evidence_refs):
            raise ValueError("Git need path is not bound to its cited current evidence")
        if question.blocking_axis is not None and question.blocking_axis not in unresolved_axes:
            raise ValueError("required Git need must bind an existing unresolved fact axis")


def git_need_required(need: GitNeedAssessment) -> bool:
    return any(question.blocking_axis is not None for judgment in need.judgments for question in judgment.questions)
