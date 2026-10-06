"""Deterministic, bounded construction of unified ``CandidateCase`` objects.

The functions in this module deliberately separate deterministic blocking from
the (supplied) semantic relation matrix.  They do not inspect channel outcomes
or evidence verdicts, and they never call a provider.  Resource-suppressed or
budget-exhausted candidate relations remain explicitly incomplete so that a
candidate cannot become deliverable merely because a bound hid a possible
relation.  Host-typed, mutually exclusive explicit decisions are instead a
deterministic reason that broad lexical overlap never proposed a relation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from itertools import combinations
from pathlib import PurePosixPath
from typing import Iterable, Literal, Mapping, Sequence

from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    canonical_bytes,
    make_case_id,
    stable_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
)

GROUPING_POLICY_VERSION = "candidate-grouping-1.7"

CandidateGroupingMode = Literal["semantic", "singleton"]

_TOKEN_RE = re.compile(r"[^\W_][\w.-]*", re.UNICODE)
_LEXICAL_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "using",
        "when",
        "with",
    }
)

ExactIdentity = tuple[str, str, tuple[str, ...], str | None, str | None]
MemberPair = tuple[str, str]


class GroupingValidationError(ValueError):
    """The host rejected an incomplete or foreign semantic relation matrix."""


@dataclass(frozen=True)
class GroupingLimits:
    """All resource choices required by deterministic candidate grouping."""

    max_candidate_members: int
    max_block_members: int
    max_group_pairs: int
    max_group_pairs_per_request: int
    max_group_members_per_request: int
    max_group_request_bytes: int
    max_case_members: int
    max_case_input_bytes: int
    max_anchors_per_member: int
    max_paths_per_member: int
    max_tokens_per_member: int
    max_anchor_bytes: int
    lexical_min_shared_tokens: int

    def __post_init__(self) -> None:
        values = {
            "max_candidate_members": self.max_candidate_members,
            "max_block_members": self.max_block_members,
            "max_group_pairs": self.max_group_pairs,
            "max_group_pairs_per_request": self.max_group_pairs_per_request,
            "max_group_members_per_request": self.max_group_members_per_request,
            "max_group_request_bytes": self.max_group_request_bytes,
            "max_case_members": self.max_case_members,
            "max_case_input_bytes": self.max_case_input_bytes,
            "max_anchors_per_member": self.max_anchors_per_member,
            "max_paths_per_member": self.max_paths_per_member,
            "max_tokens_per_member": self.max_tokens_per_member,
            "max_anchor_bytes": self.max_anchor_bytes,
            "lexical_min_shared_tokens": self.lexical_min_shared_tokens,
        }
        invalid = sorted(name for name, value in values.items() if value < 1)
        if invalid:
            raise ValueError(
                "grouping limits must be positive: " + ", ".join(invalid)
            )
        if self.max_case_members > self.max_block_members:
            raise ValueError("max_case_members must be <= max_block_members")
        if self.max_group_members_per_request < 2:
            raise ValueError("max_group_members_per_request must be >= 2")
        if self.max_group_pairs_per_request > self.max_group_pairs:
            raise ValueError("per-request grouping pairs exceed the global budget")
        if self.max_group_members_per_request > self.max_candidate_members:
            raise ValueError("per-request grouping members exceed candidate members")


@dataclass(frozen=True)
class MemberBlockingAnchors:
    """Exact, host-supplied anchors for one candidate member.

    These values must already have been acquired from the fixed snapshot.  This
    module only canonicalizes and bounds them; it does not read the repository.
    """

    symbols: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    workflow_actions: tuple[str, ...] = ()
    decision_slots: tuple[str, ...] = ()
    literals: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()


def _decision_slot_namespace(value: str) -> str | None:
    """Return the comparable namespace of one host-derived decision slot.

    A negative lexical decision is safe only when both slots use a format the
    host can interpret and belong to the same decision namespace.  In
    particular, a config entry and a command may constrain the same behavior,
    and settings in different config adapters may supersede one another; those
    cross-namespace pairs must therefore remain eligible for semantic review.
    """

    if value.startswith("configuration:"):
        parts = value.split(":", 3)
        if (
            len(parts) == 4
            and parts[1] in {"toml", "ini_cfg", "yaml_json"}
            and parts[2]
            and parts[3]
        ):
            return f"configuration:{parts[1]}"
        return None
    if value.startswith("command-target:"):
        target = value.removeprefix("command-target:")
        return "command-target" if target else None
    if value.startswith("workflow-action:"):
        payload = value.removeprefix("workflow-action:")
        try:
            path, start, end = payload.rsplit(":", 2)
            line_start = int(start)
            line_end = int(end)
        except (TypeError, ValueError):
            return None
        if (
            not path
            or line_start < 1
            or line_end < line_start
        ):
            return None
        return f"workflow-action:{path}"
    return None


@dataclass(frozen=True)
class GroupingBlock:
    block_id: str
    kind: str
    key_digest: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroupingBatch:
    """One deterministic, independently replayable semantic-pair request."""

    phase: str
    member_ids: tuple[str, ...]
    pairs: tuple[MemberPair, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"direct", "closure"}:
            raise ValueError("grouping batch phase must be direct or closure")
        if not self.pairs:
            raise ValueError("grouping batch must contain at least one pair")
        if self.member_ids != tuple(sorted(set(self.member_ids))):
            raise ValueError("grouping batch members must be canonical")
        if self.pairs != tuple(sorted(set(self.pairs))):
            raise ValueError("grouping batch pairs must be canonical")
        endpoints = {member_id for pair in self.pairs for member_id in pair}
        if endpoints != set(self.member_ids) or any(
            left >= right for left, right in self.pairs
        ):
            raise ValueError("grouping batch members and pairs differ")


@dataclass(frozen=True)
class GroupingBatchPlan:
    """Bounded batches plus pairs that could not fit even alone."""

    phase: str
    batches: tuple[GroupingBatch, ...]
    omitted_pairs: tuple[MemberPair, ...]


@dataclass(frozen=True)
class IncompleteGroupingComponent:
    member_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class GroupingPlan:
    snapshot_digest: str
    members: tuple[CandidateCaseMember, ...]
    blocks: tuple[GroupingBlock, ...]
    exact_identity_groups: tuple[tuple[str, ...], ...]
    semantic_pairs: tuple[MemberPair, ...]
    incomplete_components: tuple[IncompleteGroupingComponent, ...]
    _raw_components: tuple[tuple[str, ...], ...] = field(repr=False)
    _max_group_pairs: int = field(repr=False)
    _max_group_pairs_per_request: int = field(repr=False)
    _max_group_members_per_request: int = field(repr=False)
    _max_group_request_bytes: int = field(repr=False)
    _max_case_members: int = field(repr=False)
    _max_case_input_bytes: int = field(repr=False)


@dataclass(frozen=True)
class GroupingResult:
    plan: GroupingPlan
    cases: tuple[CandidateCase, ...]
    semantic_relations: tuple[CandidateMemberRelation, ...]


class _UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self._parent = {item: item for item in members}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != item:
            parent = self._parent[item]
            self._parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self._parent[high] = low

    def components(self) -> tuple[tuple[str, ...], ...]:
        grouped: dict[str, list[str]] = {}
        for item in sorted(self._parent):
            grouped.setdefault(self.find(item), []).append(item)
        return tuple(sorted((tuple(values) for values in grouped.values())))


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_rule(value: str) -> str:
    """Normalize presentation-only rule differences for exact identity."""

    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError("candidate rule must not normalize to an empty string")
    return normalized


def normalize_polarity(value: str) -> str:
    normalized = _normalize_text(value)
    if normalized not in {"positive", "negative"}:
        raise ValueError(f"unsupported candidate polarity: {value!r}")
    return normalized


def normalize_scope(values: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize scope spelling without case-folding repository paths."""

    normalized: set[str] = set()
    for value in values:
        scope = unicodedata.normalize("NFKC", value).strip()
        while scope.startswith("./"):
            scope = scope[2:]
        scope = re.sub(r"/{2,}", "/", scope)
        if scope.endswith("/") and scope != "/":
            scope = scope.rstrip("/")
        if not scope:
            raise ValueError("candidate scope must not normalize to empty")
        normalized.add(scope)
    if not normalized:
        raise ValueError("candidate scope must not be empty")
    return tuple(sorted(normalized))


def normalize_condition(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    return normalized or None


def exact_candidate_identity(semantics: CandidateSemantics) -> ExactIdentity:
    """Return exact identity without erasing a stated exception."""

    return (
        normalize_rule(semantics.rule),
        normalize_polarity(semantics.polarity),
        normalize_scope(semantics.applies_to),
        normalize_condition(semantics.condition),
        normalize_condition(semantics.exception),
    )


def _canonical_pair(left: str, right: str) -> MemberPair:
    if left == right:
        raise ValueError("a member cannot be paired with itself")
    return (left, right) if left < right else (right, left)


def _canonical_anchor(value: str, *, kind: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if kind == "command":
        normalized = " ".join(normalized.split())
    elif kind == "path":
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = re.sub(r"/{2,}", "/", normalized).rstrip("/")
    return normalized


def _bounded_values(
    values: Iterable[str],
    *,
    kind: str,
    maximum: int,
    max_bytes: int,
) -> tuple[tuple[str, ...], bool]:
    canonical = sorted(
        {
            item
            for value in values
            if (item := _canonical_anchor(value, kind=kind))
        }
    )
    oversized = any(len(item.encode("utf-8")) > max_bytes for item in canonical)
    admissible = [
        item for item in canonical if len(item.encode("utf-8")) <= max_bytes
    ]
    truncated = len(admissible) > maximum
    return tuple(admissible[:maximum]), oversized or truncated


def _lexical_tokens(semantics: CandidateSemantics) -> tuple[str, ...]:
    text = " ".join(
        item
        for item in (semantics.rule, semantics.condition, semantics.exception)
        if item
    )
    return tuple(
        sorted(
            {
                token
                for token in _TOKEN_RE.findall(_normalize_text(text))
                if len(token) >= 2 and token not in _LEXICAL_STOP_WORDS
            }
        )
    )


def _qualified_parent_namespaces(values: Iterable[str]) -> frozenset[str]:
    """Return immediate parents of strict dotted identifier tokens.

    This deliberately rejects prose punctuation such as ``e.g.`` while
    accepting repository-independent Python-shaped names such as
    ``GuardrailsResponse.llm_response``.  Values may come from the frozen
    candidate semantics or from bounded host-supplied symbol anchors; no
    repository text is read here.
    """

    parents: set[str] = set()
    for value in values:
        for token in _TOKEN_RE.findall(_normalize_text(value)):
            parts = token.split(".")
            if len(parts) < 2 or any(not part.isidentifier() for part in parts):
                continue
            parents.add(".".join(parts[:-1]))
    return frozenset(parents)


def _component_input_bytes(
    component: Sequence[str], members_by_id: Mapping[str, CandidateCaseMember]
) -> int:
    payload = [
        {
            "member_id": member_id,
            "semantics": members_by_id[member_id].semantics.model_dump(mode="json"),
        }
        for member_id in component
    ]
    return len(canonical_bytes(payload))


def _group_request_input_bytes(
    component: Sequence[str],
    pairs: Sequence[MemberPair],
    members_by_id: Mapping[str, CandidateCaseMember],
    *,
    phase: str,
) -> int:
    """Measure the exact schema-attached semantic-group provider request."""

    from repo_constraint_memory_v3.inference import contractual_request_bytes
    from repo_constraint_memory_v3.unified.semantic import (
        CandidateSemanticGroupResponse,
    )

    payload = {
        "stage": "candidate_semantic_group",
        "phase": phase,
        "members": [
            {
                "member_id": member_id,
                "semantics": members_by_id[member_id].semantics.model_dump(
                    mode="json"
                ),
            }
            for member_id in component
        ],
        "required_pairs": [list(pair) for pair in sorted(pairs)],
    }
    return contractual_request_bytes(
        canonical_bytes(payload).decode("utf-8"),
        CandidateSemanticGroupResponse,
    )


def plan_grouping_batches(
    plan: GroupingPlan,
    pairs: Sequence[MemberPair],
    *,
    phase: str,
) -> GroupingBatchPlan:
    """Pack the complete pair graph under exact member, pair, and byte bounds.

    Every admissible pair is still judged exactly once.  The planner grows a
    deterministic dense vertex set from residual graph degree, then places as
    many induced edges as fit.  This avoids repeatedly serializing almost the
    same members when lexicographically adjacent pairs have unrelated second
    endpoints.  A pair is omitted only when its exact standalone request is
    already too large, preserving the existing fail-closed component scope.
    """

    if phase not in {"direct", "closure"}:
        raise ValueError("grouping batch phase must be direct or closure")
    canonical_pairs = tuple(sorted(set(pairs)))
    known_members = {item.member_id for item in plan.members}
    if any(
        left >= right or left not in known_members or right not in known_members
        for left, right in canonical_pairs
    ):
        raise GroupingValidationError("grouping batch pair is foreign")
    members_by_id = {item.member_id: item for item in plan.members}
    batches: list[GroupingBatch] = []
    omitted: list[MemberPair] = []

    def fits(candidate_pairs: Sequence[MemberPair], member_ids: set[str]) -> bool:
        return (
            len(candidate_pairs) <= plan._max_group_pairs_per_request
            and len(member_ids) <= plan._max_group_members_per_request
            and _group_request_input_bytes(
                sorted(member_ids),
                candidate_pairs,
                members_by_id,
                phase=phase,
            )
            <= plan._max_group_request_bytes
        )

    residual: set[MemberPair] = set()
    adjacency: dict[str, set[str]] = {member_id: set() for member_id in known_members}
    for pair in canonical_pairs:
        if not fits([pair], set(pair)):
            omitted.append(pair)
            continue
        residual.add(pair)
        left, right = pair
        adjacency[left].add(right)
        adjacency[right].add(left)

    # The heap is updated lazily after edge removals.  Its ordering and every
    # tie-break depend only on the frozen pair graph, never on provider output
    # or runtime timing.
    degree_heap: list[tuple[int, str]] = []
    for member_id in sorted(known_members):
        if adjacency[member_id]:
            heappush(degree_heap, (-len(adjacency[member_id]), member_id))

    def highest_degree_member(excluded: set[str]) -> str | None:
        skipped: list[tuple[int, str]] = []
        selected: str | None = None
        while degree_heap:
            negative_degree, member_id = heappop(degree_heap)
            degree = len(adjacency[member_id])
            if degree == 0 or negative_degree != -degree:
                continue
            if member_id in excluded:
                skipped.append((negative_degree, member_id))
                continue
            selected = member_id
            skipped.append((negative_degree, member_id))
            break
        for item in skipped:
            heappush(degree_heap, item)
        return selected

    while residual:
        selected_members: set[str] = set()
        frontier_scores: dict[str, int] = {}
        induced_count = 0

        while (
            len(selected_members) < plan._max_group_members_per_request
            and induced_count < plan._max_group_pairs_per_request
        ):
            frontier = [
                member_id
                for member_id, score in frontier_scores.items()
                if score > 0
                and member_id not in selected_members
                and adjacency[member_id]
            ]
            if frontier:
                member_id = min(
                    frontier,
                    key=lambda item: (
                        -frontier_scores[item],
                        -len(adjacency[item]),
                        item,
                    ),
                )
            else:
                member_id = highest_degree_member(selected_members)
                if member_id is None:
                    break

            edges_to_selected = sum(
                neighbor in selected_members
                for neighbor in adjacency[member_id]
            )
            selected_members.add(member_id)
            frontier_scores.pop(member_id, None)
            induced_count += edges_to_selected
            for neighbor in adjacency[member_id]:
                if neighbor not in selected_members:
                    frontier_scores[neighbor] = frontier_scores.get(neighbor, 0) + 1

        induced_pairs = sorted(
            (left, right)
            for left in selected_members
            for right in adjacency[left]
            if left < right and right in selected_members
        )
        chosen_pairs: list[MemberPair] = []
        chosen_members: set[str] = set()
        for pair in induced_pairs:
            if len(chosen_pairs) >= plan._max_group_pairs_per_request:
                break
            proposed_pairs = [*chosen_pairs, pair]
            proposed_members = {*chosen_members, *pair}
            if fits(proposed_pairs, proposed_members):
                chosen_pairs.append(pair)
                chosen_members.update(pair)

        # Standalone preflight above guarantees progress: the highest-degree
        # seed has a residual neighbor, both enter the bounded selected set,
        # and that edge fits when the batch is otherwise empty.
        if not chosen_pairs:
            raise GroupingValidationError(
                "grouping graph packer made no progress after standalone preflight"
            )
        canonical_chosen = tuple(sorted(chosen_pairs))
        batches.append(
            GroupingBatch(
                phase=phase,
                member_ids=tuple(sorted(chosen_members)),
                pairs=canonical_chosen,
            )
        )
        for left, right in canonical_chosen:
            residual.remove((left, right))
            adjacency[left].remove(right)
            adjacency[right].remove(left)
            if adjacency[left]:
                heappush(degree_heap, (-len(adjacency[left]), left))
            if adjacency[right]:
                heappush(degree_heap, (-len(adjacency[right]), right))

    batches.sort(key=lambda item: (item.pairs, item.member_ids))
    return GroupingBatchPlan(
        phase=phase,
        batches=tuple(batches),
        omitted_pairs=tuple(sorted(omitted)),
    )


def plan_candidate_grouping(
    members: Sequence[CandidateCaseMember],
    *,
    anchors_by_member: Mapping[str, MemberBlockingAnchors],
    limits: GroupingLimits,
    mode: CandidateGroupingMode = "semantic",
) -> GroupingPlan:
    """Create a deterministic blind blocking plan for supplied case members."""

    if mode not in {"semantic", "singleton"}:
        raise ValueError("unknown candidate grouping mode")
    ordered_members = tuple(sorted(members, key=lambda item: item.member_id))
    if not ordered_members:
        raise ValueError("candidate grouping requires at least one member")
    if len(ordered_members) > limits.max_candidate_members:
        raise ValueError("candidate member count exceeds the configured run limit")
    member_ids = [item.member_id for item in ordered_members]
    if len(member_ids) != len(set(member_ids)):
        raise ValueError("candidate grouping members must be unique")
    snapshots = {item.snapshot_digest for item in ordered_members}
    if len(snapshots) != 1:
        raise ValueError("candidate grouping cannot mix snapshots")
    unknown_anchor_members = set(anchors_by_member) - set(member_ids)
    if unknown_anchor_members:
        raise ValueError(
            "blocking anchors reference unknown members: "
            + ", ".join(sorted(unknown_anchor_members))
        )
    if mode == "singleton":
        return GroupingPlan(
            snapshot_digest=next(iter(snapshots)),
            members=ordered_members,
            blocks=(),
            exact_identity_groups=(),
            semantic_pairs=(),
            incomplete_components=(),
            _raw_components=tuple((member_id,) for member_id in member_ids),
            _max_group_pairs=limits.max_group_pairs,
            _max_group_pairs_per_request=limits.max_group_pairs_per_request,
            _max_group_members_per_request=limits.max_group_members_per_request,
            _max_group_request_bytes=limits.max_group_request_bytes,
            _max_case_members=limits.max_case_members,
            _max_case_input_bytes=limits.max_case_input_bytes,
        )

    union = _UnionFind(member_ids)
    members_by_id = {item.member_id: item for item in ordered_members}
    reasons_by_member: dict[str, set[str]] = {
        member_id: set() for member_id in member_ids
    }
    raw_blocks: dict[tuple[str, str], set[str]] = {}
    typed_decisions_by_member: dict[str, frozenset[str]] = {}
    qualified_parents_by_member: dict[str, frozenset[str]] = {}

    identities: dict[ExactIdentity, list[str]] = {}
    for member in ordered_members:
        identities.setdefault(exact_candidate_identity(member.semantics), []).append(
            member.member_id
        )
    exact_groups = tuple(
        sorted(tuple(sorted(values)) for values in identities.values() if len(values) > 1)
    )
    exact_group_by_member = {
        member_id: group
        for group in exact_groups
        for member_id in group
    }

    def same_exact_group(pair: MemberPair) -> bool:
        return pair[0] in exact_group_by_member and (
            exact_group_by_member[pair[0]] == exact_group_by_member.get(pair[1])
        )

    exact_pairs = {
        pair
        for group in exact_groups
        if len(group) <= limits.max_case_members
        for pair in combinations(group, 2)
    }
    for identity, values in identities.items():
        if len(values) > 1:
            raw_blocks[("exact_identity", stable_id("IDENTITY", identity))] = set(
                values
            )

    tokens_by_member: dict[str, tuple[str, ...]] = {}
    for group in exact_groups:
        if len(group) > limits.max_case_members:
            for member_id in group:
                reasons_by_member[member_id].add("grouping_case_member_limit")
    for member in ordered_members:
        member_id = member.member_id
        supplied = anchors_by_member.get(member_id, MemberBlockingAnchors())
        bounded_symbols: tuple[str, ...] = ()
        anchor_fields = (
            ("symbol", supplied.symbols, limits.max_anchors_per_member),
            ("config_key", supplied.config_keys, limits.max_anchors_per_member),
            ("command", supplied.commands, limits.max_anchors_per_member),
            (
                "workflow_action",
                supplied.workflow_actions,
                limits.max_anchors_per_member,
            ),
            (
                "decision_slot",
                supplied.decision_slots,
                limits.max_anchors_per_member,
            ),
            ("literal", supplied.literals, limits.max_anchors_per_member),
        )
        typed_decisions: set[str] = set()
        for kind, values, maximum in anchor_fields:
            bounded, truncated = _bounded_values(
                values,
                kind=kind,
                maximum=maximum,
                max_bytes=limits.max_anchor_bytes,
            )
            if truncated:
                reasons_by_member[member_id].add(f"blocking_{kind}_limit")
            for value in bounded:
                raw_blocks.setdefault((kind, value), set()).add(member_id)
                if kind == "decision_slot" and not truncated:
                    typed_decisions.add(value)
            if kind == "symbol":
                bounded_symbols = bounded
            if kind == "decision_slot" and truncated:
                # A bounded prefix cannot prove two candidates disjoint: a
                # shared slot may be hidden in the omitted suffix.  Keep the
                # strong anchors that fit, but abstain from the negative gate.
                typed_decisions.clear()
        typed_decisions_by_member[member_id] = frozenset(typed_decisions)

        # A path locates evidence or scopes a rule; it is not itself evidence
        # that two rules express the same decision.  In particular, one
        # contributor guide commonly contains dozens of unrelated rules, and
        # using that shared evidence path as a hyperedge turns them into one
        # unreviewable component.  Keep deterministic path bounds/provenance
        # diagnostics, but only typed anchors and direct lexical pairs may
        # create semantic-grouping edges.
        _, paths_truncated = _bounded_values(
            (*supplied.paths, *member.semantics.applies_to),
            kind="path",
            maximum=limits.max_paths_per_member,
            max_bytes=limits.max_anchor_bytes,
        )
        if paths_truncated:
            reasons_by_member[member_id].add("blocking_path_limit")

        tokens = _lexical_tokens(member.semantics)
        if len(tokens) > limits.max_tokens_per_member:
            reasons_by_member[member_id].add("blocking_token_limit")
        tokens_by_member[member_id] = tokens[: limits.max_tokens_per_member]
        qualified_parents_by_member[member_id] = _qualified_parent_namespaces(
            (*tokens_by_member[member_id], *bounded_symbols)
        )

    candidate_pairs: set[MemberPair] = set()
    pair_budget_exhausted = False

    def shares_same_narrow_file_scope(pair: MemberPair) -> bool:
        """Return whether both rules have the same single concrete file scope."""

        left_scopes = tuple(members_by_id[pair[0]].semantics.applies_to)
        right_scopes = tuple(members_by_id[pair[1]].semantics.applies_to)
        if left_scopes != right_scopes or len(left_scopes) != 1:
            return False
        scope = left_scopes[0]
        return (
            not any(marker in scope for marker in ("*", "?", "[", "]"))
            and bool(PurePosixPath(scope).suffix)
        )

    def qualifies_for_high_df_parent_fallback(
        pair: MemberPair,
        *,
        retained_shared_count: int,
        high_document_frequency_tokens: frozenset[str],
    ) -> bool:
        """Recover only a qualified-symbol edge lost specifically to DF pruning.

        The fallback cannot create an edge in a corpus where no shared token
        was suppressed.  It also requires the surviving shared token to name
        the immediate parent namespace of a strict dotted identifier on at
        least one side.  This keeps the scope-local ``Class.member`` case while
        rejecting broad same-file topic words such as ``PII`` or ``guardrail``.
        """

        if (
            not high_document_frequency_tokens
            or retained_shared_count < 1
            or not shares_same_narrow_file_scope(pair)
        ):
            return False
        left, right = pair
        shared = set(tokens_by_member[left]) & set(tokens_by_member[right])
        shared_high_df = shared & high_document_frequency_tokens
        if (
            retained_shared_count >= limits.lexical_min_shared_tokens
            or retained_shared_count + len(shared_high_df)
            < limits.lexical_min_shared_tokens
        ):
            return False
        shared_retained = shared - high_document_frequency_tokens
        qualified_parents = (
            qualified_parents_by_member[left]
            | qualified_parents_by_member[right]
        )
        return bool(shared_retained & qualified_parents)

    def explicit_typed_decisions_are_disjoint(pair: MemberPair) -> bool:
        """Identify lexical-only pairs that denote separate typed decisions.

        This is deliberately a negative lexical gate, not a semantic verdict.
        It applies only when both candidates are explicit and both expose a
        bounded host-derived configuration, command-target, or workflow-action
        slot.  Full proposition keys include values or command bodies and are
        deliberately not used for this negative decision.  Cross-channel pairs
        and pairs with an untyped side remain eligible for semantic comparison.
        A shared decision slot is also retained by the strong-anchor planner
        before lexical overlap is considered.
        """

        left, right = pair
        if (
            members_by_id[left].channel != "explicit"
            or members_by_id[right].channel != "explicit"
        ):
            return False
        left_typed = typed_decisions_by_member[left]
        right_typed = typed_decisions_by_member[right]
        if not left_typed or not right_typed:
            return False
        left_namespaces = {
            namespace
            for value in left_typed
            if (namespace := _decision_slot_namespace(value)) is not None
        }
        right_namespaces = {
            namespace
            for value in right_typed
            if (namespace := _decision_slot_namespace(value)) is not None
        }
        # Missing/malformed identities, mixed decision families, and
        # cross-family comparisons all remain semantic candidates.  Only two
        # wholly comparable, single-namespace slot sets can prove separation.
        if (
            len(left_namespaces) != 1
            or len(right_namespaces) != 1
            or left_namespaces != right_namespaces
            or any(
                _decision_slot_namespace(value) is None
                for value in (*left_typed, *right_typed)
            )
        ):
            return False
        return left_typed.isdisjoint(right_typed)

    # Strong typed anchors propose direct pairs.  They no longer become
    # transitive semantic hyperedges before the model has judged those pairs.
    for (kind, _), postings in sorted(raw_blocks.items()):
        if kind == "exact_identity" or len(postings) < 2:
            continue
        if len(postings) > limits.max_block_members:
            # Suppress this high-frequency posting rather than expanding its
            # clique.  Every affected member is explicitly incomplete, so the
            # suppression cannot masquerade as a distinct judgment.
            for member_id in postings:
                reasons_by_member[member_id].add("grouping_block_member_limit")
            continue
        for pair in combinations(sorted(postings), 2):
            if same_exact_group(pair) or pair in candidate_pairs:
                continue
            if len(candidate_pairs) >= limits.max_group_pairs:
                pair_budget_exhausted = True
                break
            candidate_pairs.add(pair)
        if pair_budget_exhausted:
            break

    # Count pair overlap through token postings.  This work is independent of
    # the configured overlap threshold, so raising the threshold can only
    # remove candidate pairs; it can never trigger combinatorial C(tokens, k)
    # expansion.  A common single token likewise creates no edge when k > 1.
    shared_counts: dict[MemberPair, int] = {}
    high_document_frequency_tokens: set[str] = set()
    if not pair_budget_exhausted:
        token_postings: dict[str, list[str]] = {}
        for member_id, tokens in tokens_by_member.items():
            for token in tokens:
                token_postings.setdefault(token, []).append(member_id)
        for token, postings in sorted(token_postings.items()):
            # A token used by most of a sufficiently large candidate corpus is
            # normative/template boilerplate, not a useful blocking key.  The
            # absolute guard preserves existing behavior for small repositories;
            # exact identity and every typed strong anchor remain unaffected.
            # Suppress before combinations are expanded so ubiquitous wording
            # cannot manufacture quadratic provider work.
            if (
                len(ordered_members) >= 32
                and len(postings) > limits.max_block_members
                and len(postings) * 2 > len(ordered_members)
            ):
                high_document_frequency_tokens.add(token)
                raw_blocks[
                    (
                        "lexical_high_document_frequency",
                        stable_id(
                            "LEXDF",
                            [token, len(postings), len(ordered_members)],
                        ),
                    )
                ] = set(postings)
                continue
            for pair in combinations(sorted(postings), 2):
                # Different explicit typed decisions do not become a possible
                # semantic relation merely because their prose shares broad
                # tool/topic words (for example, two independent mypy keys).
                # Apply this before accounting planner work so a large config
                # table cannot exhaust the pair budget with suppressed pairs.
                if explicit_typed_decisions_are_disjoint(pair):
                    continue
                if pair not in shared_counts:
                    if len(shared_counts) >= limits.max_group_pairs:
                        pair_budget_exhausted = True
                        break
                    shared_counts[pair] = 0
                shared_counts[pair] += 1
            if pair_budget_exhausted:
                break

    if not pair_budget_exhausted:
        frozen_high_df_tokens = frozenset(high_document_frequency_tokens)
        for pair, count in sorted(shared_counts.items()):
            lexical_match = count >= limits.lexical_min_shared_tokens
            qualified_parent_match = (
                not lexical_match
                and qualifies_for_high_df_parent_fallback(
                    pair,
                    retained_shared_count=count,
                    high_document_frequency_tokens=frozen_high_df_tokens,
                )
            )
            if (
                not (lexical_match or qualified_parent_match)
                or same_exact_group(pair)
            ):
                continue
            if pair not in candidate_pairs and len(candidate_pairs) >= (
                limits.max_group_pairs
            ):
                pair_budget_exhausted = True
                break
            candidate_pairs.add(pair)
            raw_blocks.setdefault(
                ("lexical", stable_id("LEXPAIR", [pair, count])), set()
            ).update(pair)

    if pair_budget_exhausted:
        # The unexamined suffix can still contain a related pair.  Preserve all
        # members in one typed-incomplete component instead of silently
        # treating that suffix as distinct.
        for member_id in member_ids[1:]:
            union.union(member_ids[0], member_id)
        for member_id in member_ids:
            reasons_by_member[member_id].add("grouping_pair_budget")
    else:
        for group in exact_groups:
            for member_id in group[1:]:
                union.union(group[0], member_id)
        for pair in sorted(candidate_pairs):
            union.union(*pair)

    raw_components = union.components()
    incomplete_reason_by_component: dict[tuple[str, ...], set[str]] = {}
    for component in raw_components:
        reasons = set().union(*(reasons_by_member[item] for item in component))
        if reasons:
            incomplete_reason_by_component[component] = reasons

    incomplete_members = {
        member_id
        for component in incomplete_reason_by_component
        for member_id in component
    }
    semantic_pairs = sorted(
        pair
        for pair in candidate_pairs
        if not set(pair) & incomplete_members
    )

    blocks = tuple(
        GroupingBlock(
            block_id=stable_id("GBLOCK", [kind, key, sorted(postings)]),
            kind=kind,
            key_digest=stable_id("BLOCKKEY", [kind, key]),
            member_ids=tuple(sorted(postings)),
        )
        for (kind, key), postings in sorted(raw_blocks.items())
        if len(postings) > 1
    )
    incomplete_components = tuple(
        IncompleteGroupingComponent(
            member_ids=component,
            reason_codes=tuple(
                sorted({"grouping_incomplete", *incomplete_reason_by_component[component]})
            ),
        )
        for component in raw_components
        if component in incomplete_reason_by_component
    )
    return GroupingPlan(
        snapshot_digest=next(iter(snapshots)),
        members=ordered_members,
        blocks=blocks,
        exact_identity_groups=exact_groups,
        semantic_pairs=tuple(sorted(semantic_pairs)),
        incomplete_components=incomplete_components,
        _raw_components=raw_components,
        _max_group_pairs=limits.max_group_pairs,
        _max_group_pairs_per_request=limits.max_group_pairs_per_request,
        _max_group_members_per_request=limits.max_group_members_per_request,
        _max_group_request_bytes=limits.max_group_request_bytes,
        _max_case_members=limits.max_case_members,
        _max_case_input_bytes=limits.max_case_input_bytes,
    )


def validate_semantic_relation_matrix(
    plan: GroupingPlan,
    relations: Sequence[CandidateMemberRelation],
) -> tuple[CandidateMemberRelation, ...]:
    """Require exactly one blind semantic result for every planned pair."""

    expected = set(plan.semantic_pairs)
    actual: dict[MemberPair, CandidateMemberRelation] = {}
    for relation in relations:
        pair = (relation.left_member_id, relation.right_member_id)
        if pair in actual:
            raise GroupingValidationError(
                f"duplicate semantic grouping relation: {pair[0]} / {pair[1]}"
            )
        if relation.source != "semantic_grouping":
            raise GroupingValidationError(
                f"semantic matrix cell has invalid source: {pair[0]} / {pair[1]}"
            )
        actual[pair] = relation
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing=" + ",".join(f"{left}/{right}" for left, right in missing)
            )
        if extra:
            details.append(
                "extra=" + ",".join(f"{left}/{right}" for left, right in extra)
            )
        raise GroupingValidationError(
            "semantic grouping relation matrix mismatch: " + "; ".join(details)
        )
    return tuple(actual[pair] for pair in sorted(actual))


def resolve_direct_grouping(
    plan: GroupingPlan,
    relations: Sequence[CandidateMemberRelation],
    *,
    failed_pairs: Sequence[MemberPair] = (),
    failure_reason_code: str = "candidate_semantic_group_failed",
) -> tuple[GroupingPlan, tuple[MemberPair, ...]]:
    """Resolve direct cells into local potential components and closure work.

    A failed direct cell is an unknown possible non-distinct edge.  It joins
    only its endpoints (plus already known non-distinct edges), so one failed
    batch cannot automatically quarantine an entire transitive blocking graph.
    Complete, bounded non-distinct components receive all missing closure cells
    under the run-wide pair budget.
    """

    expected = set(plan.semantic_pairs)
    failed = set(failed_pairs)
    if any(left >= right for left, right in failed) or not failed <= expected:
        raise GroupingValidationError("failed direct grouping pair is foreign")
    successful_direct = expected - failed
    direct_plan = replace(plan, semantic_pairs=tuple(sorted(successful_direct)))
    validated = validate_semantic_relation_matrix(direct_plan, relations)
    relation_by_pair = {
        (item.left_member_id, item.right_member_id): item for item in validated
    }

    member_ids = [item.member_id for item in plan.members]
    members_by_id = {item.member_id: item for item in plan.members}
    potential_union = _UnionFind(member_ids)
    for group in plan.exact_identity_groups:
        for member_id in group[1:]:
            potential_union.union(group[0], member_id)
    for incomplete in plan.incomplete_components:
        for member_id in incomplete.member_ids[1:]:
            potential_union.union(incomplete.member_ids[0], member_id)
    for pair, relation in relation_by_pair.items():
        if relation.relation != "distinct":
            potential_union.union(*pair)
    for pair in failed:
        potential_union.union(*pair)
    components = potential_union.components()
    component_by_member = {
        member_id: component
        for component in components
        for member_id in component
    }

    reasons_by_component: dict[tuple[str, ...], set[str]] = {}
    for incomplete in plan.incomplete_components:
        component = component_by_member[incomplete.member_ids[0]]
        reasons_by_component.setdefault(component, set()).update(
            incomplete.reason_codes
        )
    for left, _ in failed:
        component = component_by_member[left]
        reasons_by_component.setdefault(component, set()).update(
            {"grouping_incomplete", failure_reason_code}
        )

    exact_pairs = {
        pair
        for group in plan.exact_identity_groups
        if len(group) <= plan._max_case_members
        for pair in combinations(group, 2)
    }
    closure_pairs: set[MemberPair] = set()
    # Failed direct cells still consumed bounded provider work.  The global
    # pair ceiling covers the whole direct+closure protocol, not only cells
    # that happened to succeed.
    planned_pair_count = len(expected)
    for component in components:
        reasons = reasons_by_component.setdefault(component, set())
        if len(component) > plan._max_case_members:
            reasons.update({"grouping_incomplete", "grouping_case_member_limit"})
        if _component_input_bytes(component, members_by_id) > (
            plan._max_case_input_bytes
        ):
            reasons.update({"grouping_incomplete", "grouping_case_byte_limit"})
        if "grouping_incomplete" in reasons:
            continue
        additions = {
            pair
            for pair in combinations(component, 2)
            if pair not in exact_pairs and pair not in successful_direct
        }
        if planned_pair_count + len(additions) > plan._max_group_pairs:
            reasons.update({"grouping_incomplete", "grouping_pair_budget"})
            continue
        closure_pairs.update(additions)
        planned_pair_count += len(additions)

    incomplete_components = tuple(
        IncompleteGroupingComponent(
            member_ids=component,
            reason_codes=tuple(sorted(reasons)),
        )
        for component, reasons in sorted(reasons_by_component.items())
        if "grouping_incomplete" in reasons
    )
    resolved_plan = replace(
        plan,
        semantic_pairs=tuple(sorted(successful_direct | closure_pairs)),
        incomplete_components=incomplete_components,
        _raw_components=components,
    )
    return resolved_plan, tuple(sorted(closure_pairs))


def mark_grouping_pairs_incomplete(
    plan: GroupingPlan,
    failed_pairs: Sequence[MemberPair],
    *,
    reason_code: str,
) -> GroupingPlan:
    """Remove failed closure cells and quarantine only their final components."""

    failed = set(failed_pairs)
    expected = set(plan.semantic_pairs)
    if any(left >= right for left, right in failed) or not failed <= expected:
        raise GroupingValidationError("incomplete grouping pair is foreign")
    component_by_member = {
        member_id: component
        for component in plan._raw_components
        for member_id in component
    }
    existing = {
        item.member_ids: set(item.reason_codes)
        for item in plan.incomplete_components
    }
    for left, right in failed:
        component = component_by_member[left]
        if component_by_member[right] != component:
            raise GroupingValidationError(
                "closure grouping pair spans final components"
            )
        existing.setdefault(component, set()).update(
            {"grouping_incomplete", reason_code}
        )
    return replace(
        plan,
        semantic_pairs=tuple(sorted(expected - failed)),
        incomplete_components=tuple(
            IncompleteGroupingComponent(
                member_ids=component,
                reason_codes=tuple(sorted(reasons)),
            )
            for component, reasons in sorted(existing.items())
        ),
    )


def mark_grouping_components_incomplete(
    plan: GroupingPlan,
    member_components: Sequence[Sequence[str]],
    *,
    reason_code: str,
) -> GroupingPlan:
    """Fail closed for provider/resource failures without fabricating relations."""

    known_components = set(plan._raw_components)
    requested = {tuple(sorted(component)) for component in member_components}
    if not requested <= known_components:
        raise GroupingValidationError("incomplete grouping component is foreign")
    existing = {item.member_ids: item for item in plan.incomplete_components}
    for component in requested:
        reasons = set(existing.get(component).reason_codes if component in existing else ())
        reasons.update({"grouping_incomplete", reason_code})
        existing[component] = IncompleteGroupingComponent(
            member_ids=component,
            reason_codes=tuple(sorted(reasons)),
        )
    requested_component_by_member = {
        member_id: component
        for component in requested
        for member_id in component
    }

    def belongs_to_failed_component(pair: MemberPair) -> bool:
        left_component = requested_component_by_member.get(pair[0])
        return left_component is not None and left_component == (
            requested_component_by_member.get(pair[1])
        )

    return replace(
        plan,
        semantic_pairs=tuple(
            pair
            for pair in plan.semantic_pairs
            if not belongs_to_failed_component(pair)
        ),
        incomplete_components=tuple(existing[key] for key in sorted(existing)),
    )


def _exact_relation(left: str, right: str) -> CandidateMemberRelation:
    return CandidateMemberRelation(
        left_member_id=left,
        right_member_id=right,
        relation="equivalent",
        source="exact_identity",
        reason_code="normalized_exact_identity",
    )


def _build_case(
    *,
    snapshot_digest: str,
    profile_digest: str,
    member_ids: Sequence[str],
    members_by_id: Mapping[str, CandidateCaseMember],
    relations: Sequence[CandidateMemberRelation],
    exact_groups: Sequence[Sequence[str]],
    grouping_status: str,
    reason_codes: Iterable[str],
    required_capabilities: Iterable[str],
) -> CandidateCase:
    canonical_members = sorted(member_ids)
    payload: dict[str, object] = {
        "case_id": make_case_id(snapshot_digest, canonical_members),
        "snapshot_digest": snapshot_digest,
        "profile_digest": profile_digest,
        "members": [members_by_id[item] for item in canonical_members],
        "relations": sorted(
            relations,
            key=lambda item: (item.left_member_id, item.right_member_id),
        ),
        "grouping_status": grouping_status,
        "grouping_reason_codes": sorted(set(reason_codes)),
        "exact_identity_groups": sorted(
            [sorted(group) for group in exact_groups if len(group) > 1]
        ),
        "required_capabilities": sorted(set(required_capabilities)),
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CandidateCase.model_validate(payload)


def assemble_candidate_cases(
    plan: GroupingPlan,
    semantic_relations: Sequence[CandidateMemberRelation],
    *,
    profile_digest: str,
    required_capabilities: Iterable[str] = ("current_evidence",),
) -> GroupingResult:
    """Validate the matrix and form connected, lossless candidate cases."""

    validated = validate_semantic_relation_matrix(plan, semantic_relations)
    members_by_id = {item.member_id: item for item in plan.members}
    exact_pairs = {
        pair
        for group in plan.exact_identity_groups
        if len(group) <= plan._max_case_members
        for pair in combinations(group, 2)
    }
    exact_relations = {
        pair: _exact_relation(*pair) for pair in sorted(exact_pairs)
    }
    semantic_by_pair = {
        (item.left_member_id, item.right_member_id): item for item in validated
    }
    incomplete_by_component = {
        item.member_ids: item for item in plan.incomplete_components
    }

    cases: list[CandidateCase] = []
    for raw_component in plan._raw_components:
        incomplete = incomplete_by_component.get(raw_component)
        raw_set = set(raw_component)
        raw_exact_groups = [
            tuple(item for item in group if item in raw_set)
            for group in plan.exact_identity_groups
            if len(set(group) & raw_set) > 1
        ]
        if incomplete is not None:
            # Retain exact and successfully judged direct/closure cells.  The
            # incomplete case does not fabricate quadratic uncertain cells for
            # failed or omitted pairs.
            component_relations = [
                relation
                for pair, relation in exact_relations.items()
                if set(pair) <= raw_set
            ] + [
                relation
                for pair, relation in semantic_by_pair.items()
                if set(pair) <= raw_set
            ]
            cases.append(
                _build_case(
                    snapshot_digest=plan.snapshot_digest,
                    profile_digest=profile_digest,
                    member_ids=raw_component,
                    members_by_id=members_by_id,
                    relations=component_relations,
                    exact_groups=raw_exact_groups,
                    grouping_status="incomplete",
                    reason_codes=incomplete.reason_codes,
                    required_capabilities=required_capabilities,
                )
            )
            continue

        semantic_union = _UnionFind(raw_component)
        for pair in exact_pairs:
            if set(pair) <= raw_set:
                semantic_union.union(*pair)
        for pair, relation in semantic_by_pair.items():
            if set(pair) <= raw_set and relation.relation != "distinct":
                # ``uncertain`` co-locates members for adjudication but remains
                # an explicit relation; it is never collapsed into equivalence.
                semantic_union.union(*pair)

        for component in semantic_union.components():
            component_set = set(component)
            case_reason_codes: set[str] = set()
            if len(component) > plan._max_case_members:
                case_reason_codes.update(
                    {"grouping_incomplete", "grouping_case_member_limit"}
                )
            if _component_input_bytes(component, members_by_id) > (
                plan._max_case_input_bytes
            ):
                case_reason_codes.update(
                    {"grouping_incomplete", "grouping_case_byte_limit"}
                )
            if "grouping_incomplete" in case_reason_codes:
                # The direct relation table is bounded.  Do not materialize
                # quadratic closure merely to prove that an already oversized
                # case is incomplete.
                relations = [
                    relation
                    for pair, relation in sorted(exact_relations.items())
                    if set(pair) <= component_set
                ] + [
                    relation
                    for pair, relation in sorted(semantic_by_pair.items())
                    if set(pair) <= component_set
                ]
            else:
                relations = []
                missing_pairs: list[MemberPair] = []
                for pair in combinations(component, 2):
                    relation = exact_relations.get(pair) or semantic_by_pair.get(pair)
                    if relation is None:
                        missing_pairs.append(pair)
                        continue
                    relations.append(relation)
                if missing_pairs:
                    # Direct blocking edges can form a semantic chain.  A
                    # complete case requires closure judgments for every pair,
                    # which the current one-round runner cannot obtain safely.
                    # Preserve the connected chain as explicitly incomplete
                    # instead of splitting it or fabricating relation cells.
                    case_reason_codes.update(
                        {
                            "grouping_incomplete",
                            "grouping_relation_closure_missing",
                        }
                    )
            if any(item.relation == "uncertain" for item in relations):
                case_reason_codes.add("uncertain_relation")
            exact_groups = [
                tuple(item for item in group if item in component_set)
                for group in raw_exact_groups
                if len(set(group) & component_set) > 1
            ]
            cases.append(
                _build_case(
                    snapshot_digest=plan.snapshot_digest,
                    profile_digest=profile_digest,
                    member_ids=component,
                    members_by_id=members_by_id,
                    relations=relations,
                    exact_groups=exact_groups,
                    grouping_status=(
                        "incomplete"
                        if "grouping_incomplete" in case_reason_codes
                        else "complete"
                    ),
                    reason_codes=case_reason_codes,
                    required_capabilities=required_capabilities,
                )
            )

    cases.sort(key=lambda item: item.case_id)
    assigned = [member.member_id for case in cases for member in case.members]
    expected_members = sorted(item.member_id for item in plan.members)
    if sorted(assigned) != expected_members or len(assigned) != len(set(assigned)):
        raise GroupingValidationError(
            "candidate case assembly did not assign each member exactly once"
        )
    return GroupingResult(
        plan=plan,
        cases=tuple(cases),
        semantic_relations=validated,
    )


def group_candidate_members(
    members: Sequence[CandidateCaseMember],
    semantic_relations: Sequence[CandidateMemberRelation],
    *,
    anchors_by_member: Mapping[str, MemberBlockingAnchors],
    limits: GroupingLimits,
    profile_digest: str,
    required_capabilities: Iterable[str] = ("current_evidence",),
    mode: CandidateGroupingMode = "semantic",
) -> GroupingResult:
    """Convenience wrapper for deterministic planning and case assembly."""

    plan = plan_candidate_grouping(
        members,
        anchors_by_member=anchors_by_member,
        limits=limits,
        mode=mode,
    )
    return assemble_candidate_cases(
        plan,
        semantic_relations,
        profile_digest=profile_digest,
        required_capabilities=required_capabilities,
    )


__all__ = [
    "CandidateGroupingMode",
    "GROUPING_POLICY_VERSION",
    "GroupingBatch",
    "GroupingBatchPlan",
    "GroupingBlock",
    "GroupingLimits",
    "GroupingPlan",
    "GroupingResult",
    "GroupingValidationError",
    "IncompleteGroupingComponent",
    "MemberBlockingAnchors",
    "assemble_candidate_cases",
    "exact_candidate_identity",
    "group_candidate_members",
    "mark_grouping_components_incomplete",
    "mark_grouping_pairs_incomplete",
    "normalize_condition",
    "normalize_polarity",
    "normalize_rule",
    "normalize_scope",
    "plan_candidate_grouping",
    "plan_grouping_batches",
    "resolve_direct_grouping",
    "validate_semantic_relation_matrix",
]
