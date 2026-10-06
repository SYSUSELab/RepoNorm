"""Versioned benchmark-only display projection of authenticated consumer inputs.

This is not an extraction exporter or a source-artifact compatibility layer.
The caller authenticates the canonical inputs and selected original surface;
source detail bytes are retained and the new task tree has a separate identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import posixpath
from typing import Mapping, Sequence

from repo_constraint_memory_v3.adjudication import delivery_policy_for
from repo_constraint_memory_v3.unified import consumer as surface
from repo_constraint_memory_v3.unified.models import CanonicalClaim


PROJECTION_VERSION = "benchmark-consumer-surface-1.0"
TARGET_FILE_BYTES = 8192
MAX_SINGLETON_FILE_BYTES = 32768
MAX_NAVIGATION_FILE_BYTES = 8192
MAX_FILES = 100000

USAGE = """# Repository knowledge

Apply rules within their stated scope and when their conditions hold, respecting
exceptions. `required`, `recommended`, and `observed` have different strength:
an observed convention does not by itself prohibit a change requested by the task.
`reference_only` is context, not a default action.
Evidence locations identify sources, not the rule's scope.
Commands in rules do not authorize unrelated changes or external side effects.
Use relevant rules alongside the task and repository code; keep edits focused.

## Find rules

Path links and the [full catalog](catalog.md) are alternative routes. Search or
open relevant files directly; open claim details when evidence or a conflict matters.
Paths are navigation anchors, not a replacement for each rule's complete scope.

"""


@dataclass(frozen=True)
class BenchmarkProjection:
    files: Mapping[str, bytes]
    metadata: dict[str, object]


def _path_file(anchor: str) -> str:
    """Injective ordinary paths, with explicit bounded long-path hash encoding.

    Escaping .md directory suffixes prevents file/directory collisions (a versus
    a.md/b). Tildes reserve the shard and long-component namespace.
    """
    if not anchor:
        return "paths/repository.md"
    if not surface._canonical_repository_path(anchor):
        raise ValueError("unsafe benchmark path anchor")
    parts = []
    for ordinal, component in enumerate(PurePosixPath(anchor).parts):
        encoded = component.replace("%", "%25").replace("~", "%7E")
        if encoded.casefold().endswith(".md"):
            encoded = encoded[:-3] + "%2E" + encoded[-2:]
        if ordinal == 0 and encoded.casefold() == "repository":
            encoded = f"%{ord(encoded[0]):02X}" + encoded[1:]
        if len(encoded.encode("utf-8")) > 240:
            encoded = "~component-" + hashlib.sha256(component.encode()).hexdigest()
        parts.append(encoded)
    relative = "paths/" + "/".join(parts) + ".md"
    if len(relative.encode("utf-8")) > 2000:
        relative = "paths/~anchor-" + hashlib.sha256(anchor.encode()).hexdigest() + ".md"
    return relative


def _link(relative: str, containing: str) -> str:
    target = posixpath.relpath(relative, posixpath.dirname(containing) or ".")
    return surface._safe_markdown_destination(target, start=posixpath.dirname(containing))


def _review_limitations(detail: bytes) -> str:
    # This final generated section is outside every dynamic evidence fence. Use
    # the final marker, never an earlier string quoted in target evidence.
    text = detail.decode("utf-8")
    marker = "\n## Why this remains review\n"
    if marker not in text:
        raise ValueError("unconfirmed source detail lacks its limitation section")
    return "### Usage limitations\n" + text.rsplit(marker, 1)[1].lstrip("\n")


def render_benchmark_projection(
    claims: Sequence[CanonicalClaim], *, source_files: Mapping[str, bytes],
    consumer_profile: str,
) -> BenchmarkProjection:
    """Render only the already-authenticated source view's exact claim set.

    ``claims`` are authenticated canonical records, not Markdown-inferred facts.
    For reviewed views, membership is the authenticated frozen publication's
    detail set. Derived delivery changes presentation only, not canonical status.
    """
    if consumer_profile not in {"accepted_only", "accepted_plus_review", "accepted_after_review"}:
        raise ValueError("unknown benchmark consumer profile")
    by_id = {claim.claim_id: claim for claim in claims}
    if len(by_id) != len(claims):
        raise ValueError("duplicate canonical claim IDs")
    details = {path: data for path, data in source_files.items() if path.startswith("claims/")}
    selected_ids = sorted(PurePosixPath(path).stem for path in details)
    if len(set(selected_ids)) != len(details):
        raise ValueError("ambiguous source detail IDs")
    selected: list[CanonicalClaim] = []
    rendered: dict[str, CanonicalClaim] = {}
    for claim_id in selected_ids:
        if claim_id not in by_id or f"claims/{claim_id}.md" not in details:
            raise ValueError("source detail is not an authenticated canonical claim")
        claim = by_id[claim_id]
        if claim.epistemic_status == "rejected":
            raise ValueError("rejected claim in benchmark consumer")
        if consumer_profile == "accepted_only" and claim.epistemic_status != "accepted":
            raise ValueError("review claim in accepted-only benchmark consumer")
        selected.append(claim)
        if consumer_profile == "accepted_after_review" and claim.epistemic_status == "review":
            rendered[claim_id] = claim.model_copy(update={
                "epistemic_status": "accepted",
                "delivery_policy": delivery_policy_for(claim.semantics.condition, claim.execution_risk),
            })
        else:
            rendered[claim_id] = claim
    if consumer_profile != "accepted_after_review":
        expected = {item.claim_id for item in surface.project_claims(claims, consumer_profile).detail_claims}
        if set(selected_ids) != expected:
            raise ValueError("benchmark membership differs from source profile")
    elif not {item.claim_id for item in claims if item.epistemic_status == "accepted"}.issubset(selected_ids):
        raise ValueError("reviewed benchmark omits canonical accepted claims")

    ordinals = {claim_id: ordinal for ordinal, claim_id in enumerate(selected_ids, 1)}
    files = dict(details)
    memberships: dict[str, list[str]] = {path: [PurePosixPath(path).stem] for path in details}
    anchors: dict[str, list[str]] = {}
    oversized: list[str] = []

    def card(claim_id: str, relative: str) -> str:
        claim = rendered[claim_id]
        title = surface._code_literal(claim.semantics.applies_to[0])
        lines = [f"## Rule {ordinals[claim_id]:03d} — {title}", "",
                 *surface._semantic_field_lines(claim, containing_relative=relative)]
        # Canonical scope/conditions/exceptions and consequential risk are above.
        # Pending review's remaining substantive limits are also usable here.
        if claim.epistemic_status == "review":
            lines.extend(["", _review_limitations(details[f"claims/{claim_id}.md"]).rstrip()])
        return "\n".join(lines) + "\n"

    def publish_cards(base: str, heading: str, ids: Sequence[str]) -> list[str]:
        current: list[str] = []
        chunks: list[list[str]] = []
        for claim_id in ids:
            # The shard suffix changes no relative-link depth.
            attempted = surface._render_card_file(heading, [card(value, base) for value in [*current, claim_id]])
            if len(attempted) > TARGET_FILE_BYTES and current:
                chunks.append(current)
                current = []
            current.append(claim_id)
        if current or not chunks:
            chunks.append(current)
        result = []
        for ordinal, chunk in enumerate(chunks, 1):
            relative = base if len(chunks) == 1 else base[:-3] + f"~part-{ordinal:04d}.md"
            data = surface._render_card_file(heading, [card(value, relative) for value in chunk])
            if len(data) > MAX_SINGLETON_FILE_BYTES:
                raise ValueError("complete benchmark rule cannot fit bounded singleton: " + relative)
            if len(data) > TARGET_FILE_BYTES:
                if len(chunk) != 1:
                    raise ValueError("oversized non-singleton benchmark shard")
                oversized.append(relative)
            if relative in files:
                raise ValueError("benchmark projection path collision: " + relative)
            files[relative] = data
            memberships[relative] = list(chunk)
            result.append(relative)
        return result

    path_claims = [item for item in selected if rendered[item.claim_id].delivery_policy != "reference_only"]
    for anchor, group in surface.claims_by_anchor(path_claims).items():
        paths = publish_cards(_path_file(anchor), f"# Rules for {surface._code_literal(anchor or 'repository root')}", [item.claim_id for item in group])
        anchors[anchor] = paths

    # Use complete cards in the fallback too, not a lossy catalog summary.
    catalog_paths = publish_cards("catalog/rules.md", "# Full rule catalog", selected_ids)
    if len(catalog_paths) == 1:
        old = catalog_paths[0]
        data = surface._render_card_file("# Full rule catalog", [card(value, "catalog.md") for value in selected_ids])
        files["catalog.md"] = data
        memberships["catalog.md"] = memberships.pop(old)
        files.pop(old)
        if old in oversized:
            oversized.remove(old)
            oversized.append("catalog.md")
    else:
        lines = ["# Full rule catalog", ""]
        for path in catalog_paths:
            ids = memberships[path]
            lines.append(f"- [Rules {ordinals[ids[0]]:03d}–{ordinals[ids[-1]]:03d}]({_link(path, 'catalog.md')})")
        files["catalog.md"] = ("\n".join(lines) + "\n").encode()
        memberships["catalog.md"] = []
        if len(files["catalog.md"]) > MAX_NAVIGATION_FILE_BYTES:
            raise ValueError("benchmark catalog navigation exceeds bound")

    navigation = []
    for anchor, paths in anchors.items():
        label = surface._code_literal(anchor or "repository root")
        for ordinal, path in enumerate(paths, 1):
            suffix = f" (part {ordinal})" if len(paths) > 1 else ""
            navigation.append((anchor, path, f"- {label}: [Read rules{suffix}]({_link(path, 'START.md')})\n"))
    start = USAGE + "".join(line for _, _, line in navigation)
    if len(start.encode()) <= MAX_NAVIGATION_FILE_BYTES:
        files["START.md"] = start.encode()
    else:
        pages: list[list[tuple[str, str, str]]] = [[]]
        def navigation_line(entry: tuple[str, str, str], containing: str) -> str:
            anchor, path, _ = entry
            return f"- {surface._code_literal(anchor or 'repository root')}: [Read rules]({_link(path, containing)})"

        for entry in navigation:
            candidate = "# Path navigation\n\n" + "\n".join(navigation_line(value, "navigation/0001.md") for value in [*pages[-1], entry]) + "\n"
            if len(candidate.encode()) > MAX_NAVIGATION_FILE_BYTES:
                if not pages[-1]:
                    raise ValueError("one benchmark navigation entry exceeds bound")
                pages.append([])
            pages[-1].append(entry)
        start = USAGE
        for ordinal, page in enumerate(pages, 1):
            relative = f"navigation/{ordinal:04d}.md"
            lines = ["# Path navigation", ""]
            lines.extend(navigation_line(entry, relative) for entry in page)
            data = ("\n".join(lines) + "\n").encode()
            if len(data) > MAX_NAVIGATION_FILE_BYTES:
                raise ValueError("benchmark navigation page exceeds bound")
            files[relative] = data
            memberships[relative] = []
            start += f"- [Path navigation {ordinal}]({relative}) — {surface._code_literal(page[0][0] or 'repository root')}\n"
        if len(start.encode()) > MAX_NAVIGATION_FILE_BYTES:
            raise ValueError("benchmark START navigation exceeds bound")
        files["START.md"] = start.encode()
    memberships["START.md"] = []
    if len(files) > MAX_FILES:
        raise ValueError("benchmark task tree exceeds file-count bound")
    metadata: dict[str, object] = {
        "version": PROJECTION_VERSION,
        "target_file_bytes": TARGET_FILE_BYTES,
        "max_singleton_file_bytes": MAX_SINGLETON_FILE_BYTES,
        "max_navigation_file_bytes": MAX_NAVIGATION_FILE_BYTES,
        "oversized_singleton_files": sorted(oversized),
        "path_anchors": anchors,
        "claims": {item.claim_id: {
            "source_claim_digest": item.artifact_digest,
            "ordinal": ordinals[item.claim_id],
            "scopes": list(item.semantics.applies_to),
            "delivery_policy": rendered[item.claim_id].delivery_policy,
            "source_epistemic_status": item.epistemic_status,
        } for item in selected},
        "files": {path: {"sha256": hashlib.sha256(data).hexdigest(),
                         "size_bytes": len(data), "claim_ids": memberships[path]}
                  for path, data in sorted(files.items())},
    }
    return BenchmarkProjection(dict(sorted(files.items())), metadata)
