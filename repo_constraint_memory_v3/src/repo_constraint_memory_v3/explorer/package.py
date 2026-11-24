"""Project canonical packages and generated consumer detail cards into a local catalog."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from ..unified import consumer
from ..unified.consumer import CanonicalClaimsFile, CanonicalCompletenessFile
from ..unified.models import CurrentEvidenceBundle


CLAIM_PATH = re.compile(r"claims/(CLAIM-[0-9a-f]{24})\.md")
FIELD = re.compile(r"^- \*\*(.+?):\*\* *(.*)$")
FENCE = re.compile(r"^( *)(`{3,})text$")
CODE = re.compile(r"^(`+)(.*?)\1$")


def _json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _literal(value: str) -> str:
    match = CODE.fullmatch(value)
    if not match:
        return value
    content = match[2]
    # The producer pads both sides if the literal starts or ends in a space or backtick.
    if content.startswith(" ") and content.endswith(" "):
        return content[1:-1]
    return content


def _card(claim_id: str, source: str, profile: str) -> dict:
    """Read the consumer renderer's headings, labeled fields, lists and text fences."""
    fields: dict[str, list[str]] = {}
    sections = []
    field_values = None
    lines = iter(source.splitlines())
    for line in lines:
        if not line.strip() or line == "# Rule details":
            continue
        fence = FENCE.fullmatch(line)
        if fence:
            body = []
            for content in lines:
                if content == fence[1] + fence[2]:
                    break
                body.append(content.removeprefix(fence[1]))
            else:
                raise ValueError(f"Unclosed text block in {claim_id}")
            text = "\n".join(body)
            kind = "quote"
        elif line.startswith("## "):
            sections.append({"title": line[3:], "blocks": []})
            continue
        elif not sections and (match := FIELD.match(line)):
            field_values = fields.setdefault(match[1], [])
            if match[2]:
                field_values.append(_literal(match[2]))
            continue
        elif line.startswith("### "):
            text, kind = line[4:], "heading"
        elif match := FIELD.match(line):
            text, kind = (match[1] + " " + _literal(match[2])).strip(), "label"
        else:
            text, kind = _literal(line.strip().removeprefix("- ")), "text"
        if sections:
            sections[-1]["blocks"].append({"kind": kind, "text": text})
        elif field_values is not None:
            field_values.append(text)
    if not fields.get("Rule") or not fields.get("Applies to"):
        raise ValueError(f"Missing rule or scope in {claim_id}")
    status = "Accepted" if profile in {"accepted_only", "accepted_after_review", "accepted_plus_review"} else "Included"
    if "Status" in fields:
        status = "\n".join(fields.pop("Status"))
        if status == "Unconfirmed (review)":
            status = "Review"
    strength = fields.pop("Normative strength", None) or fields.pop("Proposed normative strength", [])
    return {
        "id": claim_id, "rule": "\n".join(fields.pop("Rule")), "scopes": fields.pop("Applies to"),
        "strength": "\n".join(strength), "status": status, "fields": fields, "sections": sections,
    }


def _canonical(root: Path) -> dict:
    directory = root / "canonical"
    claims = CanonicalClaimsFile.model_validate_json((directory / "claims.json").read_bytes())
    complete = CanonicalCompletenessFile.model_validate_json((directory / "completeness.json").read_bytes())
    if (claims.snapshot_digest, claims.snapshot_commit, claims.profile_digest) != (
        complete.snapshot_digest, complete.snapshot_commit, complete.profile_digest
    ):
        raise ValueError("Claims and completeness belong to different runs")
    audit = directory / "audit"
    bundles = [
        CurrentEvidenceBundle.model_validate_json(file.read_bytes())
        for file in (audit / "current-evidence").glob("*.json")
    ]
    by_case, _ = consumer._current_evidence_maps(bundles)
    git = consumer._git_payload_by_id({
        file.stem: _json(file) for file in (audit / "git/slices").glob("*.json")
    })
    rows = []
    for claim in claims.claims:
        semantics = claim.semantics
        fields = {
            "Condition": [semantics.condition] if semantics.condition else [],
            "Exception": [semantics.exception] if semantics.exception else [],
            "Delivery policy": [claim.delivery_policy], "Execution risk": [claim.execution_risk],
            "Claim kind": [claim.claim_kind], "Review facets": claim.review_facets,
            "Rejection kind": [claim.rejection_kind] if claim.rejection_kind else [],
            "Decision reasons": claim.reason_codes, "Conflicts with": claim.conflicts_with,
        }
        sections = []
        for title, refs in [("Supporting evidence", claim.support), ("Counterevidence", claim.counterevidence)]:
            blocks = []
            groups = consumer._group_evidence(refs, current_items=by_case.get(claim.case_id, {}), git_items=git)
            for group in groups:
                blocks.append({"kind": "heading", "text": group.evidence_kind.replace("_", " ").capitalize()})
                blocks.append({"kind": "quote", "text": group.content})
                blocks.extend({"kind": "text", "text": location} for location in group.locations)
            if blocks:
                sections.append({"title": title, "blocks": blocks})
        rows.append({
            "id": claim.claim_id, "rule": semantics.rule, "scopes": semantics.applies_to,
            "status": claim.epistemic_status.title(), "strength": claim.normative_strength,
            "fields": {key: value for key, value in fields.items() if value}, "sections": sections,
        })
    manifest = root / "consumers/accepted_only/consumer-manifest.json"
    repository = _json(manifest).get("repository") if manifest.is_file() else None
    return {
        "name": repository or root.name, "view": "Canonical package", "commit": claims.snapshot_commit,
        "coverage": complete.state.package_status, "reasons": complete.state.reasons, "claims": rows,
    }


def _released_metadata(path: Path) -> dict:
    manifest = path.parent.parent / "manifest.json"
    if path.parent.name not in {"reponorm", "accepted_after_review"} or not manifest.is_file():
        return {}
    data = _json(manifest)
    if data.get("schema") != "reponorm-knowledge-release-1.0":
        return {}
    for item in data["packages"]:
        if item["archive"] == path.relative_to(manifest.parent).as_posix():
            if item["method"] != "reponorm":
                raise ValueError("Select a RepoNorm knowledge archive")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Archive differs from its release manifest")
            return {"repository": item["snapshot_id"], "consumer_profile": item["view"],
                    "snapshot_commit": item["snapshot_commit"], "package_status": item["state"]["package_status"],
                    "detail_file_count": item["readable_entries"]}
    return {}


def load_package(path: Path) -> dict:
    path = path.resolve()
    if (path / "canonical").is_dir():
        return _canonical(path)
    metadata = {}
    if path.is_dir() and (path / "START.md").is_file():
        manifest = path / "consumer-manifest.json"
        if manifest.is_file():
            metadata = _json(manifest)
        cards = {p.stem: p.read_text(encoding="utf-8") for p in sorted((path / "claims").glob("*.md"))
                 if CLAIM_PATH.fullmatch(p.relative_to(path).as_posix())}
    elif path.is_file() and path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                if "START.md" not in archive.namelist():
                    raise ValueError("Archive has no RepoNorm consumer view at its root")
                if "consumer-manifest.json" in archive.namelist():
                    metadata = json.loads(archive.read("consumer-manifest.json"))
                else:
                    metadata = _released_metadata(path)
                cards = {}
                for entry in sorted(archive.infolist(), key=lambda item: item.filename):
                    if match := CLAIM_PATH.fullmatch(entry.filename):
                        if match[1] in cards:
                            raise ValueError(f"Duplicate claim {match[1]}")
                        cards[match[1]] = archive.read(entry).decode("utf-8")
        except BadZipFile as error:
            raise ValueError(f"Invalid ZIP archive: {path.name}") from error
    else:
        raise ValueError("Expected a build package directory, consumer view, or RepoNorm ZIP archive")
    if not isinstance(metadata, dict):
        raise ValueError("Invalid consumer manifest")
    expected = metadata.get("detail_file_count")
    if expected is not None and expected != len(cards):
        raise ValueError(f"Expected {expected} claim details, found {len(cards)}")
    if not cards and expected is None:
        raise ValueError("No RepoNorm claim details or empty-view manifest found")
    for field in ("consumer_profile", "package_status", "repository", "snapshot_commit"):
        if metadata.get(field) is not None and not isinstance(metadata[field], str):
            raise ValueError(f"Invalid consumer metadata field: {field}")
    profile = metadata.get("consumer_profile") or "consumer_view"
    return {
        "name": metadata.get("repository") or path.stem, "view": profile.replace("_", " ").capitalize(),
        "commit": metadata.get("snapshot_commit"), "coverage": metadata.get("package_status") or "unreported",
        "reasons": [], "claims": [_card(key, content, profile) for key, content in cards.items()],
    }
