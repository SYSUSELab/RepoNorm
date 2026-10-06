"""Exact, replayable clone grouping over shared Python snapshot facts."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from repo_constraint_memory_v3.observations.canonical import model_payload_digest
from repo_constraint_memory_v3.observations.ids import clone_cluster_id
from repo_constraint_memory_v3.observations.models import (
    PythonCloneCluster,
    PythonCloneClustersArtifact,
    PythonFileShard,
    PythonSnapshotArtifact,
)

CLONE_LIMITATION = (
    "Exact clone clusters are disjoint connected components over exact-content "
    "and exact-AST membership; match_basis is the union of predicates in the "
    "component, not a claim that every member pair matches every predicate. "
    "Membership is a structural fact, not an independent support count or a "
    "requirement that members be changed together."
)

CloneMatchBasis = Literal["exact_ast", "exact_content"]


def _clone_components(
    groups: tuple[
        tuple[dict[str, list[str]], CloneMatchBasis],
        ...,
    ],
) -> list[tuple[tuple[str, ...], set[CloneMatchBasis]]]:
    """Merge overlapping exact-match groups into canonical disjoint components."""

    qualifying: list[tuple[tuple[str, ...], CloneMatchBasis]] = []
    for grouped_members, basis in groups:
        for members in grouped_members.values():
            unique_members = tuple(sorted(set(members)))
            if len(unique_members) >= 2:
                qualifying.append((unique_members, basis))
    qualifying.sort(key=lambda item: (item[0], item[1]))

    grouped_member_ids = {
        member
        for members, _ in qualifying
        for member in members
    }
    parent = {member: member for member in grouped_member_ids}

    def find(member: str) -> str:
        root = member
        while parent[root] != root:
            root = parent[root]
        while parent[member] != member:
            next_member = parent[member]
            parent[member] = root
            member = next_member
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for members, _ in qualifying:
        anchor = members[0]
        for member in members[1:]:
            union(anchor, member)

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for member in sorted(parent):
        members_by_root[find(member)].add(member)
    basis_by_root: dict[str, set[CloneMatchBasis]] = defaultdict(set)
    for members, basis in qualifying:
        basis_by_root[find(members[0])].add(basis)

    return sorted(
        (
            (tuple(sorted(members)), basis_by_root[root])
            for root, members in members_by_root.items()
        ),
        key=lambda item: (item[0], sorted(item[1])),
    )


def observe_clone_clusters(
    snapshot: PythonSnapshotArtifact,
    shards: tuple[PythonFileShard, ...],
) -> PythonCloneClustersArtifact:
    parsed = [item for item in shards if item.parse_status == "complete"]
    content_groups: dict[str, list[str]] = defaultdict(list)
    ast_groups: dict[str, list[str]] = defaultdict(list)
    for shard in parsed:
        content_groups[shard.content_sha256].append(shard.file_id)
        assert shard.ast_fingerprint is not None
        ast_groups[shard.ast_fingerprint].append(shard.file_id)
    components = _clone_components(
        (
            (content_groups, "exact_content"),
            (ast_groups, "exact_ast"),
        )
    )
    clusters = [
        PythonCloneCluster(
            cluster_id=clone_cluster_id(members, basis),
            member_file_ids=list(members),
            match_basis=sorted(basis),
        )
        for members, basis in components
    ]
    clustered = {
        member for cluster in clusters for member in cluster.member_file_ids
    }
    status = "complete" if snapshot.status == "complete" else "partial"
    reasons = (
        []
        if status == "complete"
        else ["dependency:python_snapshot:partial"]
    )
    provisional = PythonCloneClustersArtifact(
        schema_version="1.0",
        query_name="python_clone_clusters",
        query_version="1.1",
        snapshot_digest=snapshot.snapshot_digest,
        status=status,
        upstream_status=snapshot.status,
        eligible_file_count=snapshot.eligible_file_count,
        parsed_file_count=len(parsed),
        unclustered_file_count=sum(
            shard.file_id not in clustered for shard in parsed
        ),
        excluded_or_failed_file_count=(
            snapshot.eligible_file_count - len(parsed)
        ),
        clusters=clusters,
        incomplete_reasons=reasons,
        limitations=[CLONE_LIMITATION],
        artifact_digest="0" * 64,
    )
    return provisional.model_copy(
        update={
            "artifact_digest": model_payload_digest(
                provisional,
                digest_field="artifact_digest",
            )
        }
    )
