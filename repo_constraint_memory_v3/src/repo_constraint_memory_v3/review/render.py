"""Deterministic reviewed surfaces; canonical records are never reclassified."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from repo_constraint_memory_v3.adjudication import delivery_policy_for
from repo_constraint_memory_v3.unified import consumer as surface
from repo_constraint_memory_v3.unified.ids import seal_payload
from repo_constraint_memory_v3.unified.models import CanonicalClaim
from .models import ReviewConsumerManifest, ReviewDecision, ReviewSession


@dataclass(frozen=True)
class ReviewConsumerPlan:
    files: Mapping[str, bytes]
    directories: tuple[str, ...]
    manifest: ReviewConsumerManifest
    profile: str = "accepted_after_review"
    publication_directory: Path | None = None


@dataclass(frozen=True)
class ReviewProjection:
    """Consumer acceptance and delivery, separate from the source claim's status."""
    claim: CanonicalClaim
    decision: ReviewDecision | None

    @property
    def acceptance_source(self) -> str:
        if self.decision is None:
            return "automatic_verification"
        return "developer_confirmation" if self.decision.reviewer_kind == "developer" else "model_review"

    @property
    def delivery_policy(self) -> str:
        if self.decision is None:
            return self.claim.delivery_policy
        return delivery_policy_for(self.claim.semantics.condition, self.claim.execution_risk)


def _fields(item: ReviewProjection, relative: str | None) -> list[str]:
    claim = item.claim
    code = surface._code_literal
    lines: list[str] = []
    lines.extend(surface._labeled_scalar("Rule", claim.semantics.rule))
    lines.append("- **Applies to:**")
    lines.extend(f"  - {code(value)}" for value in claim.semantics.applies_to)
    if claim.semantics.condition is not None:
        lines.extend(surface._labeled_scalar("Condition", claim.semantics.condition))
    if claim.semantics.exception is not None:
        lines.extend(surface._labeled_scalar("Exception", claim.semantics.exception))
    lines.append(f"- **Normative strength:** {code(claim.normative_strength)}")
    if item.delivery_policy == "reference_only":
        lines.append(f"- **Use:** {code(item.delivery_policy)}")
    if claim.execution_risk in {"repository_wide_mutation", "external_side_effect"}:
        lines.append(f"- **Execution risk:** {code(claim.execution_risk)}")
    if relative is not None:
        lines.append(f"- **Detail:** [Open claim details]({surface._detail_link(relative, claim.claim_id)})")
    return lines


def _card(item: ReviewProjection, relative: str) -> str:
    text = "\n".join(["## Rule", "", *_fields(item, relative)]) + "\n"
    if len(text.encode("utf-8")) > surface.MAX_RENDERED_COMPACT_CARD_BYTES:
        raise ValueError("reviewed compact card exceeds frozen byte bound")
    return text


def _catalog_entry(item: ReviewProjection, relative: str) -> str:
    # The delivery override is a rendering projection, never a canonical rewrite.
    return surface.render_catalog_entry(
        item.claim.model_copy(update={"delivery_policy": item.delivery_policy}),
        containing_relative=relative, review_pending=False,
    )


def _catalog(items: list[ReviewProjection]) -> dict[str, bytes]:
    cards = [_catalog_entry(item, "catalog.md") for item in items]
    whole = surface._render_card_file("# Claim catalog", cards)
    if len(whole) <= surface.MAX_CATALOG_FILE_BYTES:
        return {"catalog.md": whole}
    chunks: list[list[ReviewProjection]] = [[]]
    for item in items:
        path = f"catalog/{len(chunks):04d}.md"
        attempted = surface._render_card_file("# Claim catalog", [_catalog_entry(value, path) for value in [*chunks[-1], item]])
        if len(attempted) > surface.MAX_CATALOG_FILE_BYTES:
            if not chunks[-1]:
                raise ValueError("review catalog single card exceeds shard bound")
            chunks.append([])
        chunks[-1].append(item)
    if len(chunks) > surface.MAX_CATALOG_SHARDS:
        raise ValueError("review catalog exceeds shard count bound")
    files = {}
    links = ["# Claim catalog", ""]
    offset = 0
    for ordinal, chunk in enumerate(chunks, 1):
        path = f"catalog/{ordinal:04d}.md"
        data = surface._render_card_file("# Claim catalog", [_catalog_entry(item, path) for item in chunk])
        if len(data) > surface.MAX_CATALOG_FILE_BYTES:
            raise ValueError("review catalog shard exceeds byte bound")
        files[path] = data
        scope = chunk[0].claim.semantics.applies_to[0]
        preview = scope[:80] + ("…" if len(scope) > 80 else "")
        links.append(f"- [Rules {offset + 1}–{offset + len(chunk)}]({path}); scope preview: {surface._code_literal(preview)}")
        offset += len(chunk)
    files["catalog.md"] = ("\n".join(links) + "\n").encode("utf-8")
    if len(files["catalog.md"]) > surface.MAX_CATALOG_FILE_BYTES:
        raise ValueError("review catalog index exceeds byte bound")
    return files


def render_review_consumer(inputs: surface._StandaloneRenderInputs, session: ReviewSession) -> ReviewConsumerPlan:
    if (
        session.canonical_claims_digest != inputs.canonical.artifact_digest
        or session.completeness_digest != inputs.completeness.artifact_digest
        or session.snapshot_digest != inputs.canonical.snapshot_digest
        or session.snapshot_commit != inputs.canonical.snapshot_commit
        or session.claim_digests != {
            claim.claim_id: claim.artifact_digest
            for claim in inputs.canonical.claims
            if claim.epistemic_status == "review"
        }
    ):
        raise ValueError("review projection differs from bound source claims")
    for claim in inputs.canonical.claims:
        for scope in claim.semantics.applies_to:
            surface.claim_path_anchor(claim, scope)
    latest = session.latest_decisions()
    items = [ReviewProjection(claim, latest.get(claim.claim_id) if claim.epistemic_status == "review" else None) for claim in inputs.canonical.claims if claim.epistemic_status == "accepted" or (claim.epistemic_status == "review" and claim.claim_id in latest and latest[claim.claim_id].decision == "accept")]
    item_by_id = {item.claim.claim_id: item for item in items}
    paths: dict[str, list[ReviewProjection]] = {}
    for item in items:
        for scope in item.claim.semantics.applies_to:
            anchor = surface.claim_path_anchor(item.claim, scope)
            if item.delivery_policy != "reference_only":
                group = paths.setdefault(anchor, [])
                if item not in group:
                    group.append(item)
    paths = dict(sorted(paths.items()))
    current_items, currents = surface._current_evidence_maps(inputs.bundle.current_evidence)
    git = surface._git_payload_by_id(inputs.git_artifacts)
    files = _catalog(items)
    files["by-path/index.md"] = surface._render_path_index({anchor: [item.claim for item in group] for anchor, group in paths.items()})
    for anchor, group in paths.items():
        path = surface.path_view_relative(anchor)
        files[path] = surface._render_card_file(f"# Constraints for {surface._code_literal(anchor or '.')}", [_card(item, path) for item in group])
    for item in items:
        claim = item.claim
        current = currents.get(claim.case_id)
        local = current_items.get(claim.case_id)
        if current is None or local is None:
            raise ValueError("review projection claim lacks case-local evidence")
        lines = ["# Rule details", "", *_fields(item, None), f"- **Acceptance source:** {surface._code_literal(item.acceptance_source)}", ""]
        for title, refs, counter in [("Supporting evidence", claim.support, False), ("Counterevidence", claim.counterevidence, True)]:
            groups = surface._group_evidence(refs, current_items=local, git_items=git)
            if not counter:
                groups = surface._attach_population_support(groups, surface._population_support(claim, current))
            lines.extend(surface._render_evidence_groups(title, groups, counterevidence=counter))
        if item.decision and item.decision.evidence:
            lines.extend(["## Review evidence locations", ""])
            for evidence in item.decision.evidence:
                lines.append(f"- {surface._code_literal(f'{evidence.path}:{evidence.start_line}-{evidence.end_line}')}")
            lines.append("")
        conflicts = [item_by_id[value] for value in claim.conflicts_with if value in item_by_id]
        if conflicts:
            lines.extend(["## Material conflicts", ""])
            for conflict in conflicts:
                lines.extend(surface._render_conflict_entry(conflict.claim))
            lines.append("")
        files[f"claims/{claim.claim_id}.md"] = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    status = inputs.completeness.state.package_status
    if status not in {"partial", "complete"} or not inputs.completeness.state.structurally_valid:
        raise ValueError("invalid source package cannot be reviewed for delivery")
    progress = session.progress()
    lines = [
        "# Repository knowledge", "",
        "- [Catalog](catalog.md): browse rules and their scopes.",
        "- [By path](by-path/index.md): browse rules associated with repository paths.",
        "- Follow a rule's detail link for sources, evidence, and limitations.", "",
    ]
    if status == "partial":
        lines.extend(["This collection may omit repository conventions.", ""])
    files["START.md"] = "\n".join(lines).encode("utf-8")
    manifest = ReviewConsumerManifest.model_validate(seal_payload({"schema_version": "4.11-partition-only-required-evidence", "max_catalog_file_bytes": surface.MAX_CATALOG_FILE_BYTES, "max_rendered_compact_card_bytes": surface.MAX_RENDERED_COMPACT_CARD_BYTES, "repository": session.repository, "snapshot_digest": session.snapshot_digest, "snapshot_commit": session.snapshot_commit, "acquisition_profile_digest": inputs.canonical.profile_digest, "canonical_claims_digest": session.canonical_claims_digest, "completeness_digest": session.completeness_digest, "source_tree_digest": session.source_tree_digest, "review_session_id": session.session_id, "review_decisions_digest": session.artifact_digest, "package_status": status, "canonical_accepted_count": sum(claim.epistemic_status == "accepted" for claim in inputs.canonical.claims), "canonical_review_count": len(session.claim_digests), "review_accepted_count": progress["accepted"], "review_rejected_count": progress["rejected"], "review_unprocessed_count": progress["unprocessed"], "exposed_accepted_count": len(items), "delivered_claim_count": sum(item.delivery_policy != "reference_only" for item in items), "catalog_card_count": len(items), "detail_file_count": len(items), "markdown_file_count": len(files), "task_tree_digest": surface.consumer_tree_digest(files), "format_version": "review-consumer-manifest-1.0", "consumer_surface_version": "review-consumer-surface-2.0", "consumer_profile": "accepted_after_review"}))
    files["consumer-manifest.json"] = surface.canonical_json_text(manifest).encode("utf-8")
    return ReviewConsumerPlan(dict(sorted(files.items())), surface._tree_directories(files), manifest)
