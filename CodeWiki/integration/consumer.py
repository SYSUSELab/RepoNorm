"""Export CodeWiki Markdown, with explicit opt-in for incomplete generation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from integration.runtime import AdapterError, digest, now
from repo_constraint_memory_v3.model_profiles import load_profile_artifact
from repo_constraint_memory_v3.unified.publication import rename_directory_noreplace


# Frozen bytes identify the host-only page in immutable v1 exports. Never used
# for new publication, and never used to filter native documents by filename.
LEGACY_START = """# Repository Documentation

Start with [overview.md](overview.md) to locate relevant module documents.
Read the documents relevant to your task as needed.

This generated documentation is reference material and may contain errors or
omissions. Consult the repository source when necessary.
"""


def _generation_admitted(receipt: dict, command: dict, *, partial: bool) -> bool:
    if receipt.get("mode") != "generate" or receipt.get("snapshot_unchanged") is not True:
        return False
    if partial:
        return (receipt.get("status") == "failed"
                and receipt.get("error") == "IncompleteDocumentationError"
                and command.get("status") == "failed" and command.get("exit_code") == 1
                and receipt.get("provider", {}).get("failure") is None)
    return (receipt.get("status") == "complete"
            and command.get("status") == "complete" and command.get("exit_code") == 0)


def _partial_coverage(tree: dict, docs: Path) -> dict:
    from codewiki.src.be.module_naming import find_missing_module_docs
    return {"status": "partial", "missing_modules": find_missing_module_docs(tree, str(docs)),
            "semantic_quality": "not_assessed", "authorization": "explicit_allow_partial"}


def _source_markdown_names(run_dir: Path) -> set[str]:
    docs = run_dir / "docs"
    if docs.is_symlink() or not docs.is_dir():
        raise AdapterError("consumer_docs_missing_or_symlink")
    for path in docs.rglob("*.md"):
        if path.parent != docs and "temp" not in path.relative_to(docs).parts:
            raise AdapterError("consumer_nested_document_unsupported")
    return {path.name for path in docs.glob("*.md")}


def _verified_consumer(
    run_dir: Path, repository: Path, *, expected_commit: str | None = None,
) -> tuple[Path, dict[str, bytes]]:
    """Verify original-byte mapping, including the narrowly supported v1 wrapper.

    Omitting expected_commit retains the generator's registered target default.
    Coding-task hosts supply their independently frozen initial task commit.
    """
    from integration.run import snapshot
    package = run_dir / "consumer"
    if package.is_symlink() or (package / "manifest.json").is_symlink():
        raise AdapterError("consumer_symlink")
    manifest = json.loads((package / "manifest.json").read_text())
    version = manifest.get("schema_version")
    if version not in {"codewiki-consumer-1.0", "codewiki-consumer-2.0", "codewiki-consumer-2.1"}:
        raise AdapterError("consumer_schema_mismatch")
    legacy = version == "codewiki-consumer-1.0"
    partial = version == "codewiki-consumer-2.1"
    if (manifest.get("knowledge_directory") != "knowledge"
            or manifest.get("source_authentication") != "export_time_hashes_not_generation_time_attestation"
            or manifest.get("content_policy") != (
                "original_markdown_bytes_plus_navigation_only" if legacy else "original_markdown_bytes_only")
            or (legacy and manifest.get("entrypoint") != "START.md")
            or (not legacy and "entrypoint" in manifest)):
        raise AdapterError("consumer_projection_contract_mismatch")
    current = snapshot(repository) if expected_commit is None else snapshot(repository, expected=expected_commit)
    if manifest["snapshot"] != current:
        raise AdapterError("consumer_snapshot_mismatch")
    knowledge = package / "knowledge"
    if knowledge.is_symlink() or not knowledge.is_dir():
        raise AdapterError("consumer_symlink")
    hashes = manifest["knowledge_sha256"]
    if set(p.name for p in knowledge.iterdir()) != set(hashes):
        raise AdapterError("consumer_tree_changed")
    for name, expected in hashes.items():
        path = knowledge / name
        if (Path(name).name != name or not name.endswith(".md") or path.is_symlink()
                or not path.is_file() or digest(path.read_bytes()) != expected):
            raise AdapterError("consumer_content_changed")
    source_names = _source_markdown_names(run_dir)
    if "overview.md" not in source_names:
        raise AdapterError("consumer_overview_missing")
    if legacy and any(name.casefold() == "start.md" for name in source_names):
        raise AdapterError("consumer_legacy_start_collision")
    required_sources = {
        "receipt.json", "command.json", "execution-profile.json",
        "docs/module_tree.json", "docs/metadata.json",
    } | {"docs/" + name for name in source_names}
    if set(manifest["source_sha256"]) != required_sources:
        raise AdapterError("consumer_source_inventory_changed")
    sources = {}
    for name, expected in manifest["source_sha256"].items():
        path = run_dir / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or path.is_symlink() or not path.is_file()):
            raise AdapterError("consumer_source_changed")
        data = path.read_bytes()
        if digest(data) != expected:
            raise AdapterError("consumer_source_changed")
        sources[name] = data
    original = {name: sources["docs/" + name] for name in sorted(source_names)}
    projected = {**original, "START.md": LEGACY_START.encode()} if legacy else original
    if set(hashes) != set(projected):
        raise AdapterError("consumer_projection_inventory_mismatch")
    for name, content in projected.items():
        if hashes[name] != digest(content) or (knowledge / name).read_bytes() != content:
            raise AdapterError("consumer_projection_content_mismatch")
    receipt = json.loads(sources["receipt.json"])
    command = json.loads(sources["command.json"])
    profile = load_profile_artifact(run_dir / "execution-profile.json")
    if (not _generation_admitted(receipt, command, partial=partial)
            or receipt.get("snapshot") != current
            or receipt.get("target") != manifest.get("target")
            or profile.profile_id != manifest.get("profile_id")
            or any(profile.profile_digest != document.get("profile_digest")
                   for document in (manifest, receipt, command))):
        raise AdapterError("consumer_source_identity_mismatch")
    if partial:
        tree = json.loads(sources["docs/module_tree.json"])
        coverage = _partial_coverage(tree, run_dir / "docs")
        metadata = json.loads(sources["docs/metadata.json"])
        if (not coverage["missing_modules"] or manifest.get("coverage") != coverage
                or metadata.get("generation_info", {}).get("commit_id") != current["commit"]):
            raise AdapterError("consumer_partial_coverage_mismatch")
    return knowledge, original


def verify_consumer(run_dir: Path, repository: Path, *, expected_commit: str | None = None) -> Path:
    """Host preflight for immutable v1, v2.0 and explicit partial v2.1 trees."""
    return _verified_consumer(run_dir, repository, expected_commit=expected_commit)[0]


def original_markdown(
    run_dir: Path, repository: Path, *, expected_commit: str | None = None,
) -> dict[str, bytes]:
    """Return verified native documents for a new, separate host projection.

    V1's exact authenticated host START is omitted; v2.x native START filenames
    are retained. Source inventory and every source-to-consumer byte mapping
    are checked before returning anything. This never rewrites a source run.
    """
    return _verified_consumer(run_dir, repository, expected_commit=expected_commit)[1]


def export_consumer(run_dir: Path, repository: Path, *, expected_commit: str | None = None,
                    allow_partial: bool = False) -> Path:
    """Publish consumer/{knowledge,manifest.json}; only knowledge is agent-facing."""
    from integration.run import snapshot
    from codewiki.src.be.module_naming import find_missing_module_docs

    run_dir = run_dir.resolve(strict=True)
    destination = run_dir / "consumer"
    if destination.exists() or destination.is_symlink():
        raise AdapterError("consumer_destination_exists")
    source_bytes: dict[str, bytes] = {}

    def read(relative: str) -> bytes:
        path = run_dir / relative
        if path.is_symlink() or not path.is_file():
            raise AdapterError("consumer_source_missing_or_symlink")
        data = path.read_bytes()
        source_bytes[relative] = data
        return data

    receipt = json.loads(read("receipt.json"))
    command = json.loads(read("command.json"))
    partial = allow_partial and _generation_admitted(receipt, command, partial=True)
    if not _generation_admitted(receipt, command, partial=partial):
        raise AdapterError("consumer_requires_successful_generation")
    current = snapshot(repository) if expected_commit is None else snapshot(repository, expected=expected_commit)
    if current != receipt.get("snapshot"):
        raise AdapterError("consumer_snapshot_mismatch")
    read("execution-profile.json")
    profile = load_profile_artifact(run_dir / "execution-profile.json")
    if (profile.profile_digest != receipt.get("profile_digest")
            or profile.profile_digest != command.get("profile_digest")):
        raise AdapterError("consumer_profile_mismatch")
    docs = run_dir / "docs"
    source_names = _source_markdown_names(run_dir)
    tree = json.loads(read("docs/module_tree.json"))
    metadata = json.loads(read("docs/metadata.json"))
    if (not isinstance(tree, dict)
            or metadata.get("generation_info", {}).get("commit_id") != current["commit"]
            or (not partial and find_missing_module_docs(tree, str(docs)))):
        raise AdapterError("consumer_documentation_incomplete")
    coverage = _partial_coverage(tree, docs) if partial else None
    if partial and not coverage["missing_modules"]:
        raise AdapterError("consumer_partial_coverage_mismatch")
    markdown = {}
    for name in sorted(source_names):
        path = docs / name
        content = read("docs/" + path.name)
        if not content.decode("utf-8").strip():
            raise AdapterError("consumer_empty_document")
        markdown[path.name] = content
    if "overview.md" not in markdown:
        raise AdapterError("consumer_overview_missing")
    projected = markdown
    stage = Path(tempfile.mkdtemp(prefix=".consumer-", dir=run_dir))
    try:
        knowledge = stage / "knowledge"
        knowledge.mkdir()
        for name, content in projected.items():
            (knowledge / name).write_bytes(content)
        manifest = {
            "schema_version": "codewiki-consumer-2.1" if partial else "codewiki-consumer-2.0", "created_at": now(),
            "source_run": str(run_dir), "target": receipt["target"],
            "snapshot": current, "profile_id": profile.profile_id,
            "profile_digest": profile.profile_digest,
            "knowledge_directory": "knowledge",
            "source_sha256": {name: digest(data) for name, data in source_bytes.items()},
            "knowledge_sha256": {name: digest(data) for name, data in projected.items()},
            "content_policy": "original_markdown_bytes_only",
            "source_authentication": "export_time_hashes_not_generation_time_attestation",
        }
        if partial:
            manifest["coverage"] = coverage
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name, content in projected.items():
            if (knowledge / name).read_bytes() != content:
                raise AdapterError("consumer_projection_changed")
        if any((run_dir / name).read_bytes() != data for name, data in source_bytes.items()):
            raise AdapterError("consumer_source_changed")
        if set(path.name for path in docs.glob("*.md")) != set(markdown):
            raise AdapterError("consumer_source_changed")
        after = snapshot(repository) if expected_commit is None else snapshot(repository, expected=expected_commit)
        if after != current:
            raise AdapterError("consumer_snapshot_changed")
        for path in knowledge.iterdir():
            path.chmod(0o444)
        knowledge.chmod(0o555)
        (stage / "manifest.json").chmod(0o444)
        rename_directory_noreplace(stage, destination)
    finally:
        if stage.exists():
            if (stage / "knowledge").exists():
                (stage / "knowledge").chmod(0o755)
            shutil.rmtree(stage)
    return destination / "knowledge"
