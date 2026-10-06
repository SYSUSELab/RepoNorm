"""Trusted pre-task handoff; this module runs outside the coding-agent workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from repo_constraint_memory_v3.unified import consumer as surface
from repo_constraint_memory_v3.unified.models import CanonicalClaim
from repo_constraint_memory_v3.unified.publication import (
    directory_tree_fingerprint, rename_directory_noreplace,
)

if __package__:
    from .benchmark_projection import (
        BenchmarkProjection, PROJECTION_VERSION, render_benchmark_projection,
    )
else:
    # The external runner and direct script CLI import this file from the
    # evaluation directory. Select that explicit layout, not an ImportError fallback.
    from benchmark_projection import (
        BenchmarkProjection, PROJECTION_VERSION, render_benchmark_projection,
    )


def prepare_consumer_handoff(
    view_directory: Path, *, source_run: Path, repository_root: Path,
    output_directory: Path, review_directory: Path | None = None,
    benchmark: bool = False,
) -> dict[str, object]:
    """Authenticate initial binding, then atomically expose only read-only Markdown.

    The returned receipt belongs to the host. Do not put it in the agent mount.
    Call once before task execution; normal task edits require no further checks.
    This bounded authentication deliberately does not run full offline validation.
    """
    source = surface._resolve_input_directory(source_run, label="source run")
    repository = surface._resolve_input_directory(repository_root, label="repository")
    view = surface._resolve_input_directory(view_directory, label="selected consumer")
    if output_directory.is_symlink():
        raise ValueError("task output is a symlink")
    destination = output_directory.resolve()
    protected = [source, repository, view]
    if review_directory is not None:
        review_directory = surface._resolve_input_directory(review_directory, label="review session")
        protected.append(review_directory)
    if any(surface._overlaps(destination, path) for path in protected):
        raise ValueError("task output overlaps a host input")
    if surface._overlaps(source, repository):
        raise ValueError("source run overlaps repository")
    if destination.exists():
        raise ValueError("task output must be absent")

    authenticated_claims: tuple[CanonicalClaim, ...] | None = None
    authenticated_claims_digest: str | None = None
    authenticated_inputs: surface._StandaloneRenderInputs | None = None

    def authenticate() -> dict[str, object]:
        nonlocal authenticated_claims, authenticated_claims_digest, authenticated_inputs
        if review_directory is not None:
            from repo_constraint_memory_v3.review.service import preflight_review_consumer
            return preflight_review_consumer(
                view, review_directory=review_directory,
                repository_root=repository, source_run=source,
            ).model_dump(mode="json")
        inputs = surface._load_standalone_inputs(source)
        errors = surface._repository_snapshot_errors(inputs, repository)
        manifest, manifest_errors = surface._view_manifest(view)
        errors.extend(manifest_errors)
        if manifest is None:
            raise ValueError("invalid selected consumer: " + "; ".join(errors))
        profiles = (
            tuple(inputs.delivery_report.requested_consumer_profiles)
            if inputs.delivery_report.consumer_manifest_digests.get(manifest.consumer_profile)
            == manifest.artifact_digest else (manifest.consumer_profile,)
        )
        plan = surface._render_standalone_plan(
            inputs, profile=manifest.consumer_profile, requested_profiles=profiles,
        )
        errors.extend(surface._compare_consumer_directory(view, plan))
        errors.extend(surface._source_files_unchanged(
            source, inputs.authenticated_files, inputs.authenticated_directories,
        ))
        errors.extend(surface._repository_snapshot_errors(inputs, repository))
        if errors:
            raise ValueError("invalid selected consumer: " + "; ".join(sorted(set(errors))))
        # Reuse the already authenticated typed records for this display-only
        # projection. The second authentication refreshes these before exact
        # regeneration; no additional source-tree load is needed per render.
        authenticated_claims = tuple(inputs.canonical.claims)
        authenticated_claims_digest = inputs.canonical.artifact_digest
        authenticated_inputs = inputs
        return manifest.model_dump(mode="json")

    from repo_constraint_memory_v3.unified.validation import _snapshot_tree, _snapshot_bytes, _tree_unchanged_errors
    tree, errors = _snapshot_tree(view, max_artifact_bytes=1_000_000_000)
    if tree is None or errors:
        raise ValueError("invalid selected tree: " + "; ".join(errors))
    manifest = authenticate()
    source_task_digest = surface._consumer_tree_digest_from_metadata({
        relative: (metadata.sha256, metadata.size)
        for relative, metadata in tree.files.items() if relative.endswith(".md")
    })
    if source_task_digest != manifest["task_tree_digest"]:
        raise ValueError("selected Markdown tree differs from authenticated manifest")
    if _tree_unchanged_errors(tree):
        raise ValueError("selected consumer changed during authentication")
    source_files = {}
    for relative in sorted(tree.files):
        if relative.endswith(".md"):
            data = _snapshot_bytes(tree, relative, errors)
            if data is None or errors:
                raise ValueError("selected consumer changed during read: " + "; ".join(errors))
            source_files[relative] = data

    def projection() -> BenchmarkProjection:
        nonlocal authenticated_claims, authenticated_claims_digest, authenticated_inputs
        if authenticated_claims is None:
            # Review preflight authenticates its own inputs but returns only a
            # manifest. Load its typed canonical records once; both independent
            # review preflights must still bind this exact canonical digest.
            inputs = surface._load_standalone_inputs(source)
            if inputs.completeness.artifact_digest != manifest["completeness_digest"]:
                raise ValueError("benchmark completeness differs from selected manifest")
            if surface._source_files_unchanged(source, inputs.authenticated_files, inputs.authenticated_directories):
                raise ValueError("benchmark projection source changed during loading")
            authenticated_claims = tuple(inputs.canonical.claims)
            authenticated_claims_digest = inputs.canonical.artifact_digest
            authenticated_inputs = inputs
        if authenticated_claims_digest != manifest["canonical_claims_digest"]:
            raise ValueError("benchmark projection source differs from selected manifest")
        result = render_benchmark_projection(
            authenticated_claims, source_files=source_files,
            consumer_profile=str(manifest["consumer_profile"]),
        )
        assert authenticated_inputs is not None
        if surface._source_files_unchanged(
            source, authenticated_inputs.authenticated_files,
            authenticated_inputs.authenticated_directories,
        ):
            raise ValueError("benchmark projection source changed during rendering")
        return result

    projected = projection() if benchmark else None
    task_files = projected.files if projected is not None else source_files
    task_digest = surface.consumer_tree_digest(task_files)
    before = directory_tree_fingerprint(view, max_bytes=1_000_000_000)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.handoff-", dir=destination.parent))
    try:
        for relative, data in sorted(task_files.items()):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o444)
        if not (staging / "START.md").is_file():
            raise ValueError("selected consumer has no START.md")
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        staging.chmod(0o555)
        staged_tree, staged_errors = _snapshot_tree(staging, max_artifact_bytes=1_000_000_000)
        if staged_tree is None or staged_errors:
            raise ValueError("invalid task staging: " + "; ".join(staged_errors))
        staged_digest = surface._consumer_tree_digest_from_metadata({
            relative: (metadata.sha256, metadata.size)
            for relative, metadata in staged_tree.files.items()
        })
        if staged_digest != task_digest:
            raise ValueError("task staging differs from authenticated Markdown tree")
        digest = directory_tree_fingerprint(staging, max_bytes=1_000_000_000)
        if authenticate() != manifest or directory_tree_fingerprint(view, max_bytes=1_000_000_000) != before:
            raise ValueError("host inputs changed during handoff")
        if benchmark and projection() != projected:
            raise ValueError("benchmark projection changed during handoff regeneration")
        if directory_tree_fingerprint(staging, max_bytes=1_000_000_000) != digest:
            raise ValueError("task staging changed during handoff")
        rename_directory_noreplace(staging, destination)
        receipt = {
            "schema_version": "task-consumer-handoff-2.0" if benchmark else "task-consumer-handoff-1.0",
            "source_run": str(source), "selected_consumer": str(view),
            "initial_repository": str(repository), "task_directory": str(destination),
            "source_manifest": manifest, "task_tree_digest": task_digest,
            "task_tree_fingerprint": digest,
        }
        if projected is not None:
            receipt.update({
                "projection_version": PROJECTION_VERSION,
                "source_task_tree_digest": source_task_digest,
                "projection": projected.metadata,
            })
        return receipt
    finally:
        if staging.exists():
            staging.chmod(0o755)
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", type=Path)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-directory", type=Path)
    parser.add_argument("--benchmark", action="store_true", help="publish separately versioned benchmark display, without a skill or collection warning")
    args = parser.parse_args()
    receipt = prepare_consumer_handoff(
        args.view, source_run=args.source_run, repository_root=args.repository,
        output_directory=args.output, review_directory=args.review_directory,
        benchmark=args.benchmark,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
