"""Explicit evidence adjudication and delivery gate."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable, Mapping

from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.models import (
    AcceptanceBasis,
    Adjudication,
    CandidateClaim,
    ClaimKind,
    Constraint,
    DeliveryPolicy,
    Evidence,
    EvidenceRelation,
    EvidenceUnit,
    ExecutionRisk,
    NormativeStrength,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.propositions import (
    PropositionEnvelope,
    normalize_explicit_proposition,
    validate_explicit_proposition_route,
)
from repo_constraint_memory_v3.source_adapters import (
    GroundingMemo,
    AdapterError,
    GroundedEvidence,
    ground_evidence_range,
)

_PLACEHOLDER_CONDITION = {
    "always",
    "general",
    "n/a",
    "na",
    "none",
    "not applicable",
}
_SUBJECT_MODAL_PREFIX_RE = re.compile(
    r"^(?:all\s+)?(?:contributors?|developers?|maintainers?|reviewers?|"
    r"changes?|code|tests?|files?|modules?|packages?|pull requests?)\s+"
    r"(?:must|required(?:\s+to)?|shall|should|are\s+recommended\s+to)\s+",
    re.IGNORECASE,
)
_SUBJECT_NEGATIVE_DIRECTIVE_PREFIX_RE = re.compile(
    r"^(?:all\s+)?(?:contributors?|developers?|maintainers?|reviewers?|"
    r"changes?|code|tests?|files?|modules?|packages?|pull requests?)\s+"
    r"(?:do\s+not|don't|never)\s+",
    re.IGNORECASE,
)
_PUNCTUATION_RE = re.compile(r"[^\w./*?-]+", re.UNICODE)
_EXTERNAL_ACTION_RE = re.compile(
    r"\b(?:publish(?:es|ed|ing)?|deploy(?:s|ed|ing)?|"
    r"upload(?:s|ed|ing)?|promot(?:e|es|ed|ing))\b|"
    r"\b(?:git\s+push|docker\s+(?:buildx\s+)?push)\b|"
    r"\bpush(?:es|ed|ing)?\s+(?:the\s+)?"
    r"(?:tag|image|artifact|package|commits?|changes?)\b|"
    r"\b(?:creat(?:e|es|ed|ing)|cut(?:s|ting)?|publish(?:es|ed|ing)?|"
    r"mak(?:e|es|ing))\s+(?:a\s+)?release\b|"
    r"\breleas(?:e|es|ed|ing)\s+(?:the\s+)?"
    r"(?:package|artifact|version)\b|"
    r"\binstall(?:s|ed|ing)?\s+(?:the\s+)?"
    r"(?:dependencies|packages?|requirements?)\b|"
    r"\b(?:uv\s+sync|(?:uv\s+pip|pip3?|python\s+-m\s+pip)\s+install)\b",
    re.IGNORECASE,
)
_SECURITY_REPORT_RE = re.compile(
    r"\b(?:report|submit|disclose)\b.{0,80}\b(?:security|vulnerabilit(?:y|ies))\b",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_MUTATION_RE = re.compile(
    r"--fix(?:\b|=)|--in-place\b|"
    r"--inline-snapshot(?:=|\s+)(?:create|fix)\b|"
    r"\bupdate[- ]snapshots?\b",
    re.IGNORECASE,
)
_PRE_COMMIT_ALL_FILES_RE = re.compile(
    r"\bpre-commit\s+run\b[^\n`]*(?:-a\b|--all-files\b)",
    re.IGNORECASE,
)
_PRE_COMMIT_FULL_SUITE_RE = re.compile(
    r"\b(?:full|complete|entire)\s+pre-commit(?:\s+(?:suite|hooks?|checks?))?\b|"
    r"\bpre-commit\b.{0,80}\b(?:all[- ]files?|full|complete|entire)\b",
    re.IGNORECASE | re.DOTALL,
)
_HIGH_RISK_REPOSITORY_SCRIPT_RE = re.compile(
    r"\bpython(?:3(?:\.\d+)?)?\s+"
    r"(?:scripts?|tools?)/"
    r"(?:[A-Za-z0-9_-]+[_-])?"
    r"(?:release|publish|deploy|bump(?:[_-]version)?|"
    r"regen(?:erate)?|generate)"
    r"(?:[_-][A-Za-z0-9_-]+)?\.py\b",
    re.IGNORECASE,
)
_FORMATTER_TOOL_RE = re.compile(
    r"\b(?:ruff\s+format|ruff-format|black|prettier|autoflake|pyupgrade)\b",
    re.IGNORECASE,
)
_GENERIC_FORMAT_ACTION_RE = re.compile(r"\bformat\b", re.IGNORECASE)
_NON_WRITING_FORMATTER_MODE_RE = re.compile(r"--(?:check|diff)\b", re.IGNORECASE)
_CONFIG_MUTATING_HOOK_RE = re.compile(
    r"\bid\s*:\s*(?:ruff-format|pyupgrade|autoflake)\b|"
    r"\bhook\b.{0,80}\b(?:formats?|rewrites?|modifies|updates?)\b",
    re.IGNORECASE | re.DOTALL,
)
_COMMAND_REPOSITORY_ACTION_RE = re.compile(
    r"\b(?:build|regen(?:erate)?|update[- ]snapshots?)\b",
    re.IGNORECASE,
)
_DIRECT_REPOSITORY_ACTION_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:always\s+|please\s+)?"
    r"(?:(?:run|execute|invoke)\s+)?`?"
    r"(?:(?:uv|hatch|poetry)\s+(?:run\s+)?)?"
    r"(?:build|format|ruff\s+format|black|prettier|autoflake|pyupgrade|"
    r"regen(?:erate)?|rewrite|update\s+(?:the\s+)?snapshots?)\b|"
    r"\b(?:must|shall|should|required\s+to)\s+"
    r"(?:build|format|regen(?:erate)?|rewrite|update\s+(?:the\s+)?snapshots?)\b"
)
_BOUNDED_TOOL_RE = re.compile(
    r"\b(?:pytest|mypy|pyright|type[- ]?check(?:ing)?|lint(?:ing)?|coverage)\b|"
    r"\bruff\s+check\b|--check\b",
    re.IGNORECASE,
)
_DIRECT_BOUNDED_ACTION_RE = re.compile(
    r"\b(?:run|execute|invoke|pass|check|validate)\s+(?:the\s+)?"
    r"(?:tests?|test\s+suite|pytest|mypy|pyright|type[- ]?checks?|"
    r"lint(?:er|ing)?|coverage(?:\s+check)?)\b|"
    r"\b(?:tests?|test\s+suite|pytest|mypy|pyright|type[- ]?checks?|"
    r"lint(?:er|ing)?|coverage)\s+"
    r"(?:must|shall|should|required\s+to|needs?\s+to)\s+"
    r"(?:pass|run|succeed|complete)\b",
    re.IGNORECASE,
)
_DIRECT_READ_ONLY_ACTION_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:always\s+|please\s+)?"
    r"(?:read|re-?read|review|consult|inspect)\b"
)
_NEGATED_RULE_FORCE_RE = re.compile(
    r"\b(?:not\s+(?:required|recommended|necessary)|need\s+not|"
    r"does\s+not\s+need\s+to|optional)\b",
    re.IGNORECASE,
)
_HARD_RULE_FORCE_RE = re.compile(
    r"\b(?:must|required(?:\s+to)?|shall|do\s+not|don't|never)\b",
    re.IGNORECASE,
)
_SOFT_RULE_FORCE_RE = re.compile(
    r"\b(?:should|recommended|recommend|prefer|preferred)\b",
    re.IGNORECASE,
)
_OBSERVED_RULE_UPGRADE_RE = re.compile(
    r"\b(?:must|required(?:\s+to)?|shall|should|recommended)\b",
    re.IGNORECASE,
)
_HARD_RULE_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:always\s+|never\s+)?"
    r"(?:add|avoid|check|document|ensure|follow|keep|maintain|preserve|"
    r"read|re-?read|report|run|test|update|use|validate|write)\b"
)
_SOFT_RULE_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:[-*+]\s+|\d+[.)]\s+)?"
    r"(?:consider|please|prefer|try)\b"
)
_WORKFLOW_ACTIVATION_ASSERTION_RE = re.compile(
    r"\b(?:when|whenever|if|only\s+(?:if|when)|conditioned\s+on|"
    r"trigger(?:ed|s)?\s+(?:by|on)|"
    r"(?:on|for)\s+(?:push(?:es)?|pull[ _-]?requests?|workflow[ _-]?dispatch|"
    r"releases?|tags?|branches?|schedules?|merges?|code\s+changes?)|"
    r"(?:only\s+)?(?:on|for|under)\s+(?:windows|ubuntu|linux|macos|"
    r"python\s*\d|the\s+[a-z0-9_-]+\s+job|matrix\b))\b",
    re.IGNORECASE,
)
_WORKFLOW_ACTION_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:-\s*)?(?P<key>run|uses):\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
_TEMPLATE_PACKAGE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])your(?:[ _-]*)(?:pkg|package|project|module|app|repo)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TEST_ARTIFACT_RE = re.compile(
    r"\bpytest(?:\.[A-Za-z_][A-Za-z0-9_.]*)?\b|"
    r"\btests\b|"
    r"\b(?:each|every|a)\s+test\b|"
    r"\btests?\s+(?:functions?|modules?|files?|code|data|names?|cases?|"
    r"suites?|fixtures?|directories|directory|layouts?|structures?)\b|"
    r"\b(?:unit|integration|functional|acceptance|e2e|end[- ]to[- ]end|"
    r"property[- ]based|example[- ]based)\s+tests?\b|"
    r"\b(?:fixtures?|assertions?|parametri[sz](?:e|ed|ation)|conftest\.py)\b|"
    r"\btest_[A-Za-z0-9_<>.-]+",
    re.IGNORECASE,
)
_PRODUCTION_CODE_TEST_OBLIGATION_RE = re.compile(
    r"\b(?:all\s+)?(?:(?:production|source|library|application|async)\s+)?"
    r"code\s+(?:(?:must|shall|should|needs?\s+to|is|are)\s+)?"
    r"(?:be\s+)?(?:tested|covered)\b|"
    r"\bcoverage\s+(?:of|for)\s+(?:all\s+)?"
    r"(?:(?:production|source|library|application|async)\s+)?code\b",
    re.IGNORECASE,
)
_TEST_ROOT_COMPONENT_RE = re.compile(
    r"^(?:test|tests|testing|[a-z0-9]+_tests?)$",
    re.IGNORECASE,
)
EXPLICIT_ADJUDICATION_VERSION = "explicit-host-adjudication-1.12"


@dataclass
class CandidateDecision:
    unit_id: str
    candidate_fingerprint: str
    outcome: str
    outcome_stage: str
    reason_code: str
    raw_candidate: CandidateClaim | None
    errors: list[str] = field(default_factory=list)
    grounded: list[GroundedEvidence] = field(default_factory=list)
    relations: list[EvidenceRelation] = field(default_factory=list)
    derived_kind: ClaimKind | None = None
    derived_strength: NormativeStrength | None = None
    acceptance_basis: list[AcceptanceBasis] = field(default_factory=list)
    semantic_core: str | None = None
    canonical_scope: tuple[str, ...] = ()
    canonical_condition: str = ""
    condition_source: str = "unprocessed"
    host_eligible: bool = False
    verification_disposition: str | None = None
    verification_request_id: str | None = None
    verification_reason_codes: list[str] = field(default_factory=list)
    verification_technical_reason_code: str | None = None
    claim_id: str | None = None
    proposition: PropositionEnvelope | None = None


@dataclass(frozen=True)
class AdjudicationResult:
    claims: tuple[Constraint, ...]
    decisions: tuple[CandidateDecision, ...]
    evidence_relations: tuple[EvidenceRelation, ...]


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_fingerprint(unit_id: str, candidate: CandidateClaim | None) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": unit_id,
        "candidate": (
            candidate.model_dump(mode="json") if candidate is not None else None
        ),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def semantic_core(rule: str, kind: ClaimKind) -> str:
    normalized = _normal(rule)
    if kind == "normative_declaration":
        normalized = _SUBJECT_NEGATIVE_DIRECTIVE_PREFIX_RE.sub(
            "not ", normalized
        )
        normalized = _SUBJECT_MODAL_PREFIX_RE.sub("", normalized)
        normalized = re.sub(r"^(?:always|please)\s+", "", normalized)
        normalized = re.sub(r"\b(?:using|with)\b", "via", normalized)
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def valid_repository_glob(pattern: str) -> bool:
    if not pattern or pattern.startswith(("/", "\\")) or "\\" in pattern:
        return False
    parts = PurePosixPath(pattern).parts
    return ".." not in parts and "." not in parts


def _literal_anchor(pattern: str) -> str:
    meta = min(
        [position for token in ("*", "?", "[") if (position := pattern.find(token)) >= 0]
        or [len(pattern)]
    )
    prefix = pattern[:meta].rstrip("/")
    return prefix


def _scope_exists_or_is_creatable(pattern: str, manifest_paths: set[str]) -> bool:
    if pattern == "**":
        return True
    anchor = _literal_anchor(pattern)
    if not anchor:
        return True
    if anchor in manifest_paths or any(
        path == anchor or path.startswith(anchor + "/") for path in manifest_paths
    ):
        return True
    parent = PurePosixPath(anchor).parent
    if parent == PurePosixPath("."):
        return True
    parent_text = parent.as_posix()
    return any(
        path == parent_text or path.startswith(parent_text + "/")
        for path in manifest_paths
    )


def _scope_stays_within(pattern: str, directory: str) -> bool:
    return pattern == f"{directory}/**" or pattern.startswith(f"{directory}/")


def _manifest_test_roots(manifest_paths: set[str]) -> tuple[str, ...]:
    """Return concrete manifest directories that conventionally contain tests."""

    roots: set[str] = set()
    for path in manifest_paths:
        parts = PurePosixPath(path).parts
        for index, component in enumerate(parts[:-1]):
            normalized = component.casefold().replace("-", "_")
            if _TEST_ROOT_COMPONENT_RE.fullmatch(normalized) is not None:
                roots.add(PurePosixPath(*parts[: index + 1]).as_posix())
                break
    return tuple(sorted(roots))


def _precisely_targets_test_artifact(text: str) -> bool:
    """Distinguish a test artifact from an obligation on production code."""

    if (
        _PRODUCTION_CODE_TEST_OBLIGATION_RE.search(text) is not None
        and re.search(r"\btests?\s+code\b", text, re.IGNORECASE) is None
    ):
        return False
    return _TEST_ARTIFACT_RE.search(text) is not None


def _test_section_scope_errors(
    candidate: CandidateClaim,
    grounded: list[GroundedEvidence],
    scope: tuple[str, ...],
    manifest_paths: set[str],
) -> list[str]:
    """Fail closed on broad scopes for typed test-artifact declarations."""

    if not grounded or not all(
        "test_section_context" in item.semantic_flags for item in grounded
    ):
        return []
    if not _precisely_targets_test_artifact(candidate.rule):
        return []
    if not any(
        _precisely_targets_test_artifact(item.evidence.quote)
        for item in grounded
    ):
        return []

    test_roots = _manifest_test_roots(manifest_paths)
    if not test_roots:
        return [
            "test-section declaration targets test artifacts but the manifest "
            "has no concrete test root"
        ]
    return [
        "test-section declaration targets test artifacts outside manifest "
        f"test roots: {pattern}"
        for pattern in scope
        if not any(_scope_stays_within(pattern, root) for root in test_roots)
    ]


def _template_token_key(value: str) -> str:
    return re.sub(r"[ _-]+", "", value.casefold())


def _unbound_template_package_tokens(
    candidate: CandidateClaim,
    grounded: Iterable[GroundedEvidence],
    manifest_paths: set[str],
) -> list[str]:
    """Find obvious ``your_package``-style tokens with no repository binding."""

    candidate_text = "\n".join(
        [
            candidate.rule,
            candidate.condition or "",
            *candidate.applies_to,
            *(item.evidence.quote for item in grounded),
        ]
    )
    occurrences = {
        match.group(0).casefold()
        for match in _TEMPLATE_PACKAGE_TOKEN_RE.finditer(candidate_text)
    }
    if not occurrences:
        return []
    manifest_components = {
        component.casefold()
        for path in manifest_paths
        for component in PurePosixPath(path).parts
    }
    python_module_stems = {
        PurePosixPath(path).stem.casefold()
        for path in manifest_paths
        if PurePosixPath(path).suffix.casefold() in {".py", ".pyi"}
    }
    bound_keys = {
        _template_token_key(item)
        for item in manifest_components | python_module_stems
    }
    return sorted(
        token
        for token in occurrences
        if _template_token_key(token) not in bound_keys
    )


def _validate_scope_condition(
    candidate: CandidateClaim,
    unit: EvidenceUnit,
    *,
    manifest_paths: set[str],
    evidence_type: str,
    semantic_flags: set[str],
    grounded: list[GroundedEvidence],
) -> tuple[tuple[str, ...], str, str, list[str]]:
    errors: list[str] = []
    scope = tuple(sorted(set(candidate.applies_to)))
    for pattern in scope:
        if not valid_repository_glob(pattern):
            errors.append(f"invalid repository-relative applies_to glob: {pattern}")
        elif not _scope_exists_or_is_creatable(pattern, manifest_paths):
            errors.append(f"applies_to has no manifest anchor: {pattern}")
    name = PurePosixPath(unit.path).name.casefold()
    parent = PurePosixPath(unit.path).parent
    if name in {"agents.md", "claude.md"} and parent != PurePosixPath("."):
        directory = parent.as_posix()
        for pattern in scope:
            if not _scope_stays_within(pattern, directory):
                errors.append(
                    f"nested instruction scope escapes {directory}: {pattern}"
                )
    if evidence_type == "direct_declaration":
        errors.extend(
            _test_section_scope_errors(
                candidate,
                grounded,
                scope,
                manifest_paths,
            )
        )
    condition = ""
    condition_source = "none"
    if evidence_type == "direct_declaration":
        condition = (candidate.condition or "").strip()
        condition_source = "model" if condition else "none"
        if condition.casefold() in _PLACEHOLDER_CONDITION:
            condition = ""
            condition_source = "none"
    elif evidence_type == "configuration_entry":
        # A section is evidence provenance, not necessarily an activation
        # predicate.  The rule and applies_to carry the tool and path context.
        condition_source = "typed_unconditional"
    elif evidence_type == "command_definition":
        targets = sorted(
            flag[len("command_target:") :]
            for flag in semantic_flags
            if flag.startswith("command_target:")
            and flag[len("command_target:") :] not in {"", "<root>"}
        )
        if len(targets) != 1:
            errors.append(
                "command evidence must identify exactly one canonical target"
            )
            condition_source = "typed_unconditional"
        else:
            condition = f"command target: {targets[0]}"
            condition_source = "typed_context"
    elif evidence_type == "workflow_step":
        predicates = sorted(
            flag[len("workflow_predicate:") :]
            for flag in semantic_flags
            if flag.startswith("workflow_predicate:")
            and flag[len("workflow_predicate:") :] not in {"", "<root>"}
        )
        actions = sorted(
            flag[len("command:") :]
            for flag in semantic_flags
            if flag.startswith("command:")
            and flag[len("command:") :] not in {"", "<root>"}
        )
        if not actions:
            errors.append(
                "workflow evidence must identify an exact typed run/uses action"
            )
        if predicates:
            condition = "workflow predicate: " + " and ".join(predicates)
            condition_source = "typed_context"
        elif actions:
            # A precise run/uses literal can support an unconditional fact about
            # what this workflow contains.  It cannot support a trigger or job
            # activation claim unless that predicate is selected separately.
            sampled_condition = (candidate.condition or "").strip()
            if (
                sampled_condition
                and sampled_condition.casefold() not in _PLACEHOLDER_CONDITION
            ):
                errors.append(
                    "action-only workflow evidence cannot support a model-supplied "
                    "activation condition"
                )
            activation_text = "\n".join(
                [candidate.rule, sampled_condition]
            )
            if _WORKFLOW_ACTIVATION_ASSERTION_RE.search(activation_text):
                errors.append(
                    "action-only workflow evidence cannot support trigger or "
                    "activation wording"
                )
            condition_source = "typed_action_only"
        else:
            condition_source = "typed_context_missing"
    return scope, _normal(condition), condition_source, errors


def _derive_semantics(
    candidate: CandidateClaim,
    grounded: list[GroundedEvidence],
) -> tuple[
    ClaimKind | None,
    NormativeStrength | None,
    list[AcceptanceBasis],
    list[str],
]:
    errors: list[str] = []
    types = {item.evidence.type for item in grounded}
    if len(types) != 1:
        return None, None, [], ["one candidate mixes incompatible evidence types"]
    evidence_type = next(iter(types))
    if evidence_type == "direct_declaration":
        if any(item.example_context for item in grounded):
            errors.append("prose is nested under a typed example context")
        if all(
            "documented_command_recipe" in item.semantic_flags
            for item in grounded
        ):
            return (
                "command_recipe",
                "observed",
                ["typed_repository_literal"],
                errors,
            )
        if not all(item.contributor_context for item in grounded):
            errors.append("prose lacks contributor/source-section context")
        hard = any(item.hard_modal for item in grounded)
        soft = any(item.soft_modal for item in grounded)
        hard_directive = any(item.hard_directive for item in grounded)
        soft_directive = any(item.soft_directive for item in grounded)
        if not hard and not soft and not hard_directive and not soft_directive:
            errors.append("prose has no direct normative or directive language")
        basis: list[AcceptanceBasis] = []
        if hard or soft:
            basis.append("direct_modal_text")
        if hard_directive or soft_directive:
            basis.append("direct_instruction_text")
        return (
            "normative_declaration",
            (
                "required"
                if hard or hard_directive
                else "recommended"
                if soft or soft_directive
                else None
            ),
            basis,
            errors,
        )
    if evidence_type == "configuration_entry":
        flags = set().union(*(item.semantic_flags for item in grounded))
        if "allowlisted_enforcement" in flags:
            return (
                "enforced_configuration",
                "observed",
                ["executable_enforcement"],
                errors,
            )
        return (
            "operational_fact",
            "observed",
            ["typed_repository_literal"],
            errors,
        )
    if evidence_type == "command_definition":
        return (
            "command_recipe",
            "observed",
            ["typed_repository_literal"],
            errors,
        )
    if evidence_type == "workflow_step":
        return (
            "workflow_fact",
            "observed",
            ["typed_repository_literal"],
            errors,
        )
    return None, None, [], ["unsupported evidence type"]


def _rule_normative_force(
    rule: str,
    *,
    include_directives: bool,
) -> NormativeStrength:
    """Classify only force expressed by model-authored rule wording."""

    text = _NEGATED_RULE_FORCE_RE.sub("", rule)
    if _HARD_RULE_FORCE_RE.search(text) or (
        include_directives and _HARD_RULE_DIRECTIVE_RE.search(text)
    ):
        return "required"
    if _SOFT_RULE_FORCE_RE.search(text) or (
        include_directives and _SOFT_RULE_DIRECTIVE_RE.search(text)
    ):
        return "recommended"
    return "observed"


def _force_compatibility_errors(
    candidate: CandidateClaim,
    *,
    kind: ClaimKind,
    evidence_strength: NormativeStrength,
) -> list[str]:
    """Reject model wording that strengthens the typed evidence semantics."""

    if kind not in {
        "normative_declaration",
        "operational_fact",
        "enforced_configuration",
        "workflow_fact",
    }:
        return []
    if kind == "normative_declaration":
        rule_strength = _rule_normative_force(
            candidate.rule,
            include_directives=True,
        )
    else:
        upgrade = _OBSERVED_RULE_UPGRADE_RE.search(candidate.rule)
        if upgrade is None:
            rule_strength = "observed"
        elif upgrade.group(0).casefold() in {"should", "recommended"}:
            rule_strength = "recommended"
        else:
            rule_strength = "required"
    rank = {"observed": 0, "recommended": 1, "required": 2}
    if rank[rule_strength] <= rank[evidence_strength]:
        return []
    return [
        "model rule wording is stronger than grounded evidence: "
        f"rule={rule_strength}, evidence={evidence_strength}"
    ]


def _workflow_executable_action_text(evidence: Iterable[Evidence]) -> str:
    actions: list[str] = []
    block_markers = {"|", ">", "|-", ">-", "|+", ">+"}
    for item in evidence:
        if item.adapter != "workflow":
            continue
        lines = item.quote.splitlines()
        for index, line in enumerate(lines):
            match = _WORKFLOW_ACTION_RE.match(line)
            if match is None:
                continue
            value = match.group("value").strip()
            if value not in block_markers:
                if value:
                    actions.append(value)
                continue
            base_indent = len(match.group("indent").expandtabs(4))
            block: list[str] = []
            for following in lines[index + 1 :]:
                if not following.strip():
                    continue
                expanded = following.expandtabs(4)
                indent = len(expanded) - len(expanded.lstrip())
                if indent <= base_indent:
                    break
                block.append(expanded.strip())
            actions.extend(block)
    return "\n".join(actions)


def _has_repository_mutation(kind: ClaimKind, text: str) -> bool:
    if (
        _EXPLICIT_MUTATION_RE.search(text)
        or _PRE_COMMIT_ALL_FILES_RE.search(text)
        or _PRE_COMMIT_FULL_SUITE_RE.search(text)
        or _HIGH_RISK_REPOSITORY_SCRIPT_RE.search(text)
    ):
        return True
    formatter = _FORMATTER_TOOL_RE.search(text) is not None
    formatter_is_check = _NON_WRITING_FORMATTER_MODE_RE.search(text) is not None
    if kind in {"command_recipe", "workflow_fact"}:
        if (
            formatter or _GENERIC_FORMAT_ACTION_RE.search(text) is not None
        ) and not formatter_is_check:
            return True
        return _COMMAND_REPOSITORY_ACTION_RE.search(text) is not None
    if kind in {"operational_fact", "enforced_configuration"}:
        return bool(
            _CONFIG_MUTATING_HOOK_RE.search(text)
            or (
                formatter
                and not formatter_is_check
                and re.search(r"\b(?:hook|invokes?|runs?)\b", text, re.IGNORECASE)
            )
        )
    if kind == "normative_declaration":
        if formatter and not formatter_is_check:
            return _DIRECT_REPOSITORY_ACTION_RE.search(text) is not None
        return _DIRECT_REPOSITORY_ACTION_RE.search(text) is not None
    return False


def _has_bounded_check(kind: ClaimKind, text: str) -> bool:
    if kind in {"command_recipe", "workflow_fact"}:
        return _BOUNDED_TOOL_RE.search(text) is not None
    if kind == "normative_declaration":
        return _DIRECT_BOUNDED_ACTION_RE.search(text) is not None
    if kind in {"operational_fact", "enforced_configuration"}:
        # Configuration is passive unless its exact literal identifies a
        # validation hook or explicit non-writing check invocation.
        return bool(
            re.search(
                r"\bid\s*:\s*(?:ruff|mypy|pyright|pytest|check-[a-z0-9_-]+)\b|"
                r"\b(?:ruff\s+check|pytest|mypy|pyright)\b|--check\b",
                text,
                re.IGNORECASE,
            )
        )
    return False


def classify_execution_risk(
    kind: ClaimKind,
    rule: str,
    evidence: Iterable[Evidence],
) -> ExecutionRisk:
    evidence_items = list(evidence)
    if kind == "workflow_fact":
        # A predicate such as ``if: ... release ...`` activates the fact but is
        # not an executable action.  Likewise, sampled rule wording is not
        # authoritative for side effects.  Only exact run/uses evidence drives
        # workflow execution risk.
        combined = _workflow_executable_action_text(evidence_items)
    else:
        combined = "\n".join(
            [rule, *(getattr(item, "quote", "") for item in evidence_items)]
        )
    if _EXTERNAL_ACTION_RE.search(combined) or _SECURITY_REPORT_RE.search(combined):
        return "external_side_effect"
    if _has_repository_mutation(kind, combined):
        return "repository_wide_mutation"
    if _has_bounded_check(kind, combined):
        return "bounded_check"
    if kind == "normative_declaration" and _DIRECT_READ_ONLY_ACTION_RE.search(
        combined
    ):
        return "read_only"
    if kind in {"operational_fact", "enforced_configuration"}:
        return "read_only"
    return "scoped_mutation"


def delivery_policy_for(
    condition: str | None,
    risk: ExecutionRisk,
) -> DeliveryPolicy:
    if risk in {"repository_wide_mutation", "external_side_effect"}:
        return "reference_only"
    if condition:
        return "condition_matched_only"
    return "default_consider"


def _proposition_payload(decision: CandidateDecision) -> dict[str, object]:
    assert decision.derived_kind is not None
    assert decision.derived_strength is not None
    assert decision.semantic_core is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_kind": decision.derived_kind,
        "semantic_core": decision.semantic_core,
        "normative_strength": decision.derived_strength,
        "applies_to": list(decision.canonical_scope),
        "condition": decision.canonical_condition,
    }


def claim_id_for_proposition(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_stable_json(dict(payload)).encode("utf-8")).hexdigest()
    return f"RCM3-{digest[:24]}"


def prepare_candidate_decisions(
    candidates: Iterable[tuple[str, str, CandidateClaim]],
    *,
    units_by_id: Mapping[str, EvidenceUnit],
    sources_by_path: Mapping[str, SourceFile],
    manifest_paths: set[str],
    snapshot_digest: str = "sha256:" + "0" * 64,
    inference_request_by_unit: Mapping[str, str] | None = None,
    _grounding_memo: GroundingMemo | None = None,
) -> tuple[CandidateDecision, ...]:
    """Ground and type every explicit candidate without merging or accepting it."""

    decisions: list[CandidateDecision] = []
    for unit_id, reason_code, candidate in candidates:
        fingerprint = candidate_fingerprint(unit_id, candidate)
        decision = CandidateDecision(
            unit_id=unit_id,
            candidate_fingerprint=fingerprint,
            outcome="review",
            outcome_stage="grounding",
            reason_code=reason_code,
            raw_candidate=candidate,
        )
        unit = units_by_id.get(unit_id)
        if unit is None:
            decision.errors.append("candidate result references an unknown unit")
            decisions.append(decision)
            continue
        grounded: list[GroundedEvidence] = []
        for selection in candidate.evidence:
            try:
                grounded.append(
                    ground_evidence_range(
                        selection,
                        units_by_id=units_by_id,
                        sources_by_path=sources_by_path,
                        allowed_unit_ids={unit_id},
                        _memo=_grounding_memo,
                    )
                )
            except AdapterError as error:
                decision.errors.append(str(error))
        decision.grounded = grounded
        if len(grounded) != len(candidate.evidence):
            decisions.append(decision)
            continue
        unbound_tokens = _unbound_template_package_tokens(
            candidate,
            grounded,
            manifest_paths,
        )
        decision.errors.extend(
            f"unbound template package token: {token}"
            for token in unbound_tokens
        )
        kind, strength, basis, semantic_errors = _derive_semantics(
            candidate,
            grounded,
        )
        decision.derived_kind = kind
        decision.derived_strength = strength
        decision.acceptance_basis = basis
        decision.outcome_stage = "semantics"
        decision.errors.extend(semantic_errors)
        if kind is None or strength is None:
            decisions.append(decision)
            continue
        decision.errors.extend(
            _force_compatibility_errors(
                candidate,
                kind=kind,
                evidence_strength=strength,
            )
        )
        semantic_flags = set().union(
            *(item.semantic_flags for item in grounded)
        )
        scope, condition, condition_source, scope_errors = _validate_scope_condition(
            candidate,
            unit,
            manifest_paths=manifest_paths,
            evidence_type=grounded[0].evidence.type,
            semantic_flags=semantic_flags,
            grounded=grounded,
        )
        decision.canonical_scope = scope
        decision.canonical_condition = condition
        decision.condition_source = condition_source
        decision.semantic_core = semantic_core(candidate.rule, kind)
        decision.errors.extend(scope_errors)
        if decision.errors:
            decisions.append(decision)
            continue
        try:
            decision.proposition = normalize_explicit_proposition(
                snapshot_digest=snapshot_digest,
                unit=unit,
                candidate=candidate,
                derived_kind=kind,
                derived_strength=strength,
                canonical_scope=scope,
                canonical_condition=condition,
                grounded_evidence=[item.evidence for item in grounded],
                inference_request_id=(
                    inference_request_by_unit.get(unit_id)
                    if inference_request_by_unit is not None
                    else None
                ),
            )
            validate_explicit_proposition_route(decision.proposition)
        except ValueError as error:
            decision.errors.append(str(error))
            decisions.append(decision)
            continue
        decision.host_eligible = True
        decisions.append(decision)

    return tuple(decisions)


def finalize_candidate_decisions(
    prepared_decisions: Iterable[CandidateDecision],
) -> AdjudicationResult:
    """Apply the deterministic acceptance/merge/delivery policy to prepared roots."""

    decisions = list(prepared_decisions)
    for decision in decisions:
        verifier_eligible = (
            decision.verification_disposition == "eligible"
            and decision.verification_technical_reason_code is None
            and "verified_eligible" in decision.verification_reason_codes
        )
        if decision.host_eligible and verifier_eligible and not decision.errors:
            decision.outcome = "accepted"
            decision.outcome_stage = "merge"
        elif (
            decision.host_eligible
            and decision.verification_disposition is None
            and not decision.errors
        ):
            decision.outcome = "review"
            decision.outcome_stage = "explicit_semantic_verify"
            decision.errors.append("explicit_verification:not_run")

    valid = [item for item in decisions if item.outcome == "accepted"]
    groups: dict[tuple[object, ...], list[CandidateDecision]] = defaultdict(list)
    for decision in valid:
        groups[
            (
                decision.derived_kind,
                decision.semantic_core,
                decision.canonical_scope,
                decision.canonical_condition,
            )
        ].append(decision)

    # Record conservative same-core relationships before claims copy their
    # evidence relations.  These are audit hints only; they do not implement
    # semantic conflict or authority adjudication.
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            if (
                left.derived_kind == right.derived_kind
                and left.semantic_core == right.semantic_core
                and (
                    left.canonical_scope != right.canonical_scope
                    or left.canonical_condition != right.canonical_condition
                )
            ):
                relation_kind = (
                    "refines_scope"
                    if left.canonical_scope != right.canonical_scope
                    else "possible_duplicate"
                )
                for source, target in ((left, right), (right, left)):
                    source.relations.append(
                        EvidenceRelation(
                            relation=relation_kind,
                            route="proposition_merge_v1",
                            target_fingerprint=target.candidate_fingerprint,
                            reason=(
                                "Semantic core matches but scope or condition differs; "
                                "the candidates were not merged."
                            ),
                        )
                    )

    claims: list[Constraint] = []
    for group in groups.values():
        strengths = {item.derived_strength for item in group}
        if len(strengths) != 1:
            for item in group:
                item.outcome = "review"
                item.outcome_stage = "merge"
                item.errors.append(
                    "same proposition has conflicting normative strengths"
                )
            continue
        representative = min(
            group,
            key=lambda item: (
                item.raw_candidate.rule if item.raw_candidate else "",
                item.candidate_fingerprint,
            ),
        )
        payload = _proposition_payload(representative)
        claim_id = claim_id_for_proposition(payload)
        proposition_digest = hashlib.sha256(
            _stable_json(payload).encode("utf-8")
        ).hexdigest()
        evidence_by_key = {
            (
                grounded.evidence.path,
                grounded.evidence.line_start,
                grounded.evidence.line_end,
                grounded.evidence.type,
                grounded.evidence.quote_sha256,
                grounded.evidence.discovery_origin,
            ): grounded.evidence
            for item in group
            for grounded in item.grounded
        }
        evidence = sorted(
            evidence_by_key.values(),
            key=lambda item: (
                item.path,
                item.line_start,
                item.line_end,
                item.type,
                item.quote_sha256,
                item.discovery_origin,
            ),
        )
        relations = sorted(
            {
                (
                    relation.relation,
                    relation.route,
                    relation.target_fingerprint,
                    relation.reason,
                ): relation
                for item in group
                for relation in item.relations
            }.values(),
            key=lambda item: (
                item.relation,
                item.route,
                item.target_fingerprint,
                item.reason,
            ),
        )
        assert representative.raw_candidate is not None
        assert representative.derived_kind is not None
        assert representative.derived_strength is not None
        assert representative.proposition is not None
        validate_explicit_proposition_route(representative.proposition)
        risk = classify_execution_risk(
            representative.derived_kind,
            representative.raw_candidate.rule,
            evidence,
        )
        condition = representative.canonical_condition or None
        policy = delivery_policy_for(
            condition,
            risk,
        )
        claim = Constraint(
            id=claim_id,
            epistemic_status="accepted",
            origin_channels=["explicit"],
            rule=representative.raw_candidate.rule,
            claim_kind=representative.derived_kind,
            normative_strength=representative.derived_strength,
            acceptance_basis=sorted(set(representative.acceptance_basis)),
            adjudication=Adjudication(
                route="explicit_typed_evidence",
                policy_version="1.9",
                sufficiency="passed",
                reasons=[
                    "span_grounded",
                    "typed_adapter_valid",
                    "scope_condition_valid",
                    *(
                        ["explicit_semantic_verified"]
                        if representative.verification_disposition == "eligible"
                        else []
                    ),
                    *(
                        ["condition_derived_from_typed_context"]
                        if representative.condition_source == "typed_context"
                        else []
                    ),
                ],
            ),
            applies_to=list(representative.canonical_scope),
            condition=condition,
            delivery_policy=policy,
            execution_risk=risk,
            proposition_key=f"sha256:{proposition_digest}",
            evidence=evidence,
            evidence_relations=relations,
            conflicts=[],
        )
        claims.append(claim)
        for item in group:
            item.claim_id = claim_id

    all_relations = sorted(
        {
            (
                relation.relation,
                relation.route,
                relation.target_fingerprint,
                relation.reason,
            ): relation
            for decision in decisions
            for relation in decision.relations
        }.values(),
        key=lambda item: (
            item.relation,
            item.route,
            item.target_fingerprint,
            item.reason,
        ),
    )
    return AdjudicationResult(
        claims=tuple(sorted(claims, key=lambda item: item.id)),
        decisions=tuple(decisions),
        evidence_relations=tuple(all_relations),
    )


def adjudicate_candidates(
    candidates: Iterable[tuple[str, str, CandidateClaim]],
    *,
    units_by_id: Mapping[str, EvidenceUnit],
    sources_by_path: Mapping[str, SourceFile],
    manifest_paths: set[str],
    snapshot_digest: str = "sha256:" + "0" * 64,
    inference_request_by_unit: Mapping[str, str] | None = None,
) -> AdjudicationResult:
    """Run host adjudication only; unverified candidates remain in review."""

    prepared = prepare_candidate_decisions(
        candidates,
        units_by_id=units_by_id,
        sources_by_path=sources_by_path,
        manifest_paths=manifest_paths,
        snapshot_digest=snapshot_digest,
        inference_request_by_unit=inference_request_by_unit,
    )
    return finalize_candidate_decisions(prepared)
