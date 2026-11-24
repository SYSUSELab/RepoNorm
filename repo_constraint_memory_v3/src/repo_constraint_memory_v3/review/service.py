"""Provider-free review sessions, immutable publication and read-only replay."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal
from uuid import uuid4

from repo_constraint_memory_v3.repository_paths import is_canonical_repository_path
from repo_constraint_memory_v3.unified import consumer as surface
from repo_constraint_memory_v3.unified.ids import content_digest, seal_payload
from repo_constraint_memory_v3.unified.publication import directory_tree_fingerprint, rename_directory_noreplace
from .models import ReviewConsumerManifest, ReviewDecision, ReviewEvidence, ReviewSession
from .render import ReviewConsumerPlan, render_review_consumer

MAX_SESSION_BYTES = 64_000_000


class ReviewDecisionConflictError(ValueError):
    """A decision changed after the caller began reviewing this claim."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_session(path: Path) -> ReviewSession:
    if path.is_symlink() or not path.is_file():
        raise ValueError("review session is not a regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_SESSION_BYTES + 1)
    if len(data) > MAX_SESSION_BYTES:
        raise ValueError("review session exceeds resource bound")
    session = ReviewSession.model_validate_json(data)
    if data != surface.canonical_json_text(session).encode("utf-8"):
        raise ValueError("review session is not canonical JSON")
    return session


def _write_session(path: Path, session: ReviewSession) -> None:
    data = surface.canonical_json_text(session).encode("utf-8")
    if len(data) > MAX_SESSION_BYTES:
        raise ValueError("review session exceeds resource bound")
    if path.is_symlink():
        raise ValueError("review session cannot be a symlink")
    fd, name = tempfile.mkstemp(prefix=".session-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _locked(directory: Path, repository_root: Path) -> Iterator[None]:
    session = _read_session(directory / "session.json")
    repository = _repository(session, repository_root)
    source = surface._resolve_input_directory(Path(session.source_run), label="review source run")
    if any(surface._overlaps(directory, protected) for protected in (repository, source)):
        raise ValueError("review directory overlaps a protected input")
    lock = directory / ".review.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _tree_digest(inputs: surface._StandaloneRenderInputs) -> str:
    return content_digest({"files": inputs.authenticated_files, "directories": sorted(inputs.authenticated_directories)})


def _check_unchanged(source: Path, repository: Path, inputs: surface._StandaloneRenderInputs) -> None:
    errors = surface._source_files_unchanged(source, inputs.authenticated_files, inputs.authenticated_directories)
    errors.extend(surface._repository_snapshot_errors(inputs, repository))
    if errors:
        raise ValueError("review source/snapshot changed: " + "; ".join(errors))


def _bound_inputs(session: ReviewSession, repository: Path, review_directory: Path | None = None) -> surface._StandaloneRenderInputs:
    source = surface._resolve_input_directory(Path(session.source_run), label="review source run")
    if surface._overlaps(source, repository):
        raise ValueError("review source overlaps repository")
    if review_directory is not None and any(surface._overlaps(review_directory, protected) for protected in (source, repository)):
        raise ValueError("review directory overlaps a protected input")
    inputs = surface._load_standalone_inputs(source)
    errors = surface._repository_snapshot_errors(inputs, repository)
    if errors:
        raise ValueError("invalid review snapshot binding: " + "; ".join(errors))
    expected = {item.claim_id: item.artifact_digest for item in inputs.canonical.claims if item.epistemic_status == "review"}
    if (session.snapshot_digest != inputs.canonical.snapshot_digest or session.snapshot_commit != inputs.canonical.snapshot_commit or session.canonical_claims_digest != inputs.canonical.artifact_digest or session.completeness_digest != inputs.completeness.artifact_digest or session.repository != inputs.repository_name or session.source_tree_digest != _tree_digest(inputs) or session.claim_digests != expected):
        raise ValueError("review session differs from source run/claim identity")
    return inputs


def _safe_directory(path: Path) -> Path:
    return surface._resolve_input_directory(path, label="review directory")


def _repository(session: ReviewSession, supplied: Path | None) -> Path:
    return surface._resolve_input_directory(supplied or Path(session.repository_root), label="review repository")


def create_review(source_run: Path, *, repository_root: Path, output_directory: Path) -> ReviewSession:
    source = surface._resolve_input_directory(source_run, label="source run")
    repository = surface._resolve_input_directory(repository_root, label="target repository")
    if output_directory.is_symlink():
        raise ValueError("review output is a symlink")
    destination = output_directory.resolve(strict=False)
    if any(surface._overlaps(left, right) for left, right in [(source, repository), (destination, source), (destination, repository)]):
        raise ValueError("review output/source overlaps a protected input")
    if destination.exists():
        raise ValueError("new review requires a fresh output directory")
    inputs = surface._load_standalone_inputs(source)
    errors = surface._repository_snapshot_errors(inputs, repository)
    if errors:
        raise ValueError("invalid review repository binding: " + "; ".join(errors))
    session = ReviewSession.model_validate(seal_payload({"format_version": "developer-review-1.0", "session_id": "REVIEW-" + uuid4().hex, "source_run": str(source), "repository_root": str(repository), "repository": inputs.repository_name, "snapshot_digest": inputs.canonical.snapshot_digest, "snapshot_commit": inputs.canonical.snapshot_commit, "canonical_claims_digest": inputs.canonical.artifact_digest, "completeness_digest": inputs.completeness.artifact_digest, "source_tree_digest": _tree_digest(inputs), "created_at": _now(), "claim_digests": {claim.claim_id: claim.artifact_digest for claim in inputs.canonical.claims if claim.epistemic_status == "review"}, "decisions": []}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _write_session(staging / "session.json", session)
        fingerprint = directory_tree_fingerprint(staging, max_bytes=MAX_SESSION_BYTES)
        _check_unchanged(source, repository, inputs)
        if _read_session(staging / "session.json") != session:
            raise ValueError("review session staging changed")
        if directory_tree_fingerprint(staging, max_bytes=MAX_SESSION_BYTES) != fingerprint:
            raise ValueError("review session staging tree changed")
        rename_directory_noreplace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return session


def load_review(review_directory: Path, *, repository_root: Path | None = None) -> ReviewSession:
    directory = _safe_directory(review_directory)
    session = _read_session(directory / "session.json")
    repository = _repository(session, repository_root)
    inputs = _bound_inputs(session, repository, directory)
    _check_unchanged(Path(session.source_run), repository, inputs)
    return session


def review_queue(review_directory: Path, *, repository_root: Path) -> dict[str, object]:
    directory = _safe_directory(review_directory)
    session = _read_session(directory / "session.json")
    repository = _repository(session, repository_root)
    inputs = _bound_inputs(session, repository, directory)
    current_items, currents = surface._current_evidence_maps(inputs.bundle.current_evidence)
    findings = surface._findings_by_case(inputs.bundle.findings)
    git = surface._git_payload_by_id(inputs.git_artifacts)
    claims = []
    for claim in inputs.canonical.claims:
        if claim.epistemic_status != "review":
            continue
        row = claim.model_dump(mode="json")
        row["review_explanations"] = [surface._REVIEW_EXPLANATIONS[value] for value in claim.review_facets]
        current = currents.get(claim.case_id)
        local = current_items.get(claim.case_id)
        if current is None or local is None:
            raise ValueError("review claim lacks current evidence")
        row["detail_markdown"] = surface._render_detail(claim, exposed_ids={value.claim_id for value in inputs.canonical.claims}, claim_by_id={value.claim_id: value for value in inputs.canonical.claims}, current_items=local, current=current, findings=findings.get(claim.case_id, ()), git_items=git).decode("utf-8")
        row["decision"] = session.latest_decisions().get(claim.claim_id)
        if row["decision"] is not None:
            row["decision"] = row["decision"].model_dump(mode="json")
        claims.append(row)
    _check_unchanged(Path(session.source_run), repository, inputs)
    return {"session": session.model_dump(mode="json"), "claims": claims, "progress": session.progress(), "completeness": inputs.completeness.state.model_dump(mode="json")}


def _check_evidence(repository: Path, evidence: list[ReviewEvidence], *, bind: bool = False) -> list[ReviewEvidence]:
    bound = []
    for reference in evidence:
        path = repository.joinpath(*Path(reference.path).parts)
        if any(part.is_symlink() for part in [path, *path.parents] if part != repository.parent):
            raise ValueError("review evidence cannot traverse a symlink")
        if not path.is_file() or not path.resolve().is_relative_to(repository):
            raise ValueError("review evidence does not refer to a repository file")
        with path.open("rb") as handle:
            data = handle.read(8_000_001)
        if len(data) > 8_000_000:
            raise ValueError("review evidence file exceeds read bound")
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeError as error:
            raise ValueError("review evidence must be UTF-8 text") from error
        if reference.end_line > len(lines):
            raise ValueError("review evidence line is outside repository file")
        digest = hashlib.sha256(data).hexdigest()
        if reference.content_digest != digest and (not bind or reference.content_digest is not None):
            raise ValueError("review evidence content differs from bound repository file")
        bound.append(reference.model_copy(update={"content_digest": digest}))
    return bound


def _execution_bytes(directory: Path, reference: str) -> bytes:
    if not is_canonical_repository_path(reference):
        raise ValueError("execution reference must be a safe relative path")
    path = directory / reference
    if any(part.is_symlink() for part in [path, *path.parents] if part.is_relative_to(directory)):
        raise ValueError("execution reference traverses a symlink")
    if not path.is_file() or not path.resolve().is_relative_to(directory):
        raise ValueError("execution reference does not bind a review artifact")
    with path.open("rb") as handle:
        data = handle.read(MAX_SESSION_BYTES + 1)
    if len(data) > MAX_SESSION_BYTES:
        raise ValueError("execution artifact exceeds resource bound")
    try:
        text = data.decode("utf-8")
        if path.suffix == ".json" and text != surface.canonical_json_text(json.loads(text)):
            raise ValueError("execution JSON must use canonical presentation")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("execution artifact is not valid UTF-8/JSON") from error
    return data


def _execution_artifacts(directory: Path, session: ReviewSession) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    for item in session.decisions:
        if item.execution_ref is None:
            continue
        data = _execution_bytes(directory, item.execution_ref)
        if hashlib.sha256(data).hexdigest() != item.execution_digest:
            raise ValueError("review execution artifact differs from decision digest")
        if item.execution_ref not in files:
            total += len(data)
        files[item.execution_ref] = data
        if total > MAX_SESSION_BYTES:
            raise ValueError("review execution closure exceeds resource bound")
    return files


def save_decision(review_directory: Path, *, repository_root: Path, claim_id: str, decision: Literal["accept", "reject"], reviewer_kind: Literal["developer", "model"] = "developer", reviewer_id: str = "local-developer", rationale: str = "", evidence: list[ReviewEvidence] | None = None, execution_ref: str | None = None, expected_version: int | None = None) -> ReviewSession:
    if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
        raise ValueError("expected decision version must be a nonnegative integer")
    directory = _safe_directory(review_directory)
    with _locked(directory, repository_root):
        session = _read_session(directory / "session.json")
        repository = _repository(session, repository_root)
        inputs = _bound_inputs(session, repository, directory)
        if claim_id not in session.claim_digests:
            raise ValueError("only this session's canonical review claims can be decided")
        latest = session.latest_decisions().get(claim_id)
        if expected_version is not None and expected_version != (latest.version if latest else 0):
            raise ReviewDecisionConflictError("review decision changed during this review")
        execution_digest = hashlib.sha256(_execution_bytes(directory, execution_ref)).hexdigest() if execution_ref is not None else None
        item = ReviewDecision(claim_id=claim_id, claim_digest=session.claim_digests[claim_id], decision=decision, reviewer_kind=reviewer_kind, reviewer_id=reviewer_id, rationale=rationale, evidence=evidence or [], execution_ref=execution_ref, execution_digest=execution_digest, decided_at=_now(), version=1 if latest is None else latest.version + 1)
        item = item.model_copy(update={"evidence": _check_evidence(repository, item.evidence, bind=True)})
        payload = session.model_dump(mode="json")
        payload["decisions"].append(item.model_dump(mode="json"))
        updated = ReviewSession.model_validate(seal_payload(payload))
        _check_unchanged(Path(session.source_run), repository, inputs)
        _write_session(directory / "session.json", updated)
        return updated



def save_model_decisions(review_directory: Path, *, repository_root: Path, decisions: list[ReviewDecision]) -> tuple[ReviewSession, list[str]]:
    """Atomically import a model batch without replacing existing decisions.

    The importer validates individual results first. This transaction rechecks
    their provenance against one bound source/snapshot under the session lock.
    Invalid input aborts the transaction; concurrent decisions are preserved.
    """
    directory = _safe_directory(review_directory)
    with _locked(directory, repository_root):
        session = _read_session(directory / "session.json")
        repository = _repository(session, repository_root)
        inputs = _bound_inputs(session, repository, directory)
        latest = session.latest_decisions()
        conflicts: list[str] = []
        additions: list[ReviewDecision] = []
        seen: set[str] = set()
        for supplied in decisions:
            item = ReviewDecision.model_validate(supplied.model_dump(mode="json"))
            if item.claim_id in seen:
                raise ValueError("model batch contains duplicate claim IDs")
            seen.add(item.claim_id)
            if item.reviewer_kind != "model" or item.version != 1:
                raise ValueError("model batch requires initial model decisions")
            if session.claim_digests.get(item.claim_id) != item.claim_digest:
                raise ValueError("model decision differs from session claim binding")
            if item.claim_id in latest:
                conflicts.append(item.claim_id)
                continue
            data = _execution_bytes(directory, item.execution_ref)
            if hashlib.sha256(data).hexdigest() != item.execution_digest:
                raise ValueError("review execution artifact differs from decision digest")
            additions.append(item.model_copy(update={"evidence": _check_evidence(repository, item.evidence, bind=True)}))
        payload = session.model_dump(mode="json")
        payload["decisions"].extend(item.model_dump(mode="json") for item in additions)
        updated = ReviewSession.model_validate(seal_payload(payload))
        _check_unchanged(Path(session.source_run), repository, inputs)
        if additions:
            _write_session(directory / "session.json", updated)
        return updated, conflicts


def _publication_plan(
    plan: ReviewConsumerPlan,
    session: ReviewSession,
    executions: dict[str, bytes],
) -> ReviewConsumerPlan:
    files = {"consumer/" + path: data for path, data in plan.files.items()}
    files["decisions.json"] = surface.canonical_json_text(session).encode("utf-8")
    files.update({"audit/" + path: data for path, data in executions.items()})
    directories = {"consumer", *("consumer/" + value for value in plan.directories)}
    for relative in files:
        directories.update(str(parent) for parent in Path(relative).parents if str(parent) != ".")
    return ReviewConsumerPlan(files, tuple(sorted(directories)), plan.manifest)


def _next_publication(directory: Path) -> Path:
    parent = directory / "publications"
    if parent.is_symlink():
        raise ValueError("review publications directory is a symlink")
    existing = []
    if parent.exists():
        for ordinal, item in enumerate(parent.iterdir()):
            if ordinal >= 100_000:
                raise ValueError("review publication count exceeds resource bound")
            if item.name.isascii() and item.name.isdigit() and len(item.name) <= 12:
                existing.append(int(item.name))
    return parent / f"{max(existing, default=0) + 1:06d}"


def publish_review(review_directory: Path, *, repository_root: Path, output_directory: Path | None = None) -> ReviewConsumerPlan:
    directory = _safe_directory(review_directory)
    with _locked(directory, repository_root):
        session = _read_session(directory / "session.json")
        repository = _repository(session, repository_root)
        inputs = _bound_inputs(session, repository, directory)
        for decision in session.latest_decisions().values():
            _check_evidence(repository, decision.evidence)
        executions = _execution_artifacts(directory, session)
        plan = render_review_consumer(inputs, session)
        destination_arg = output_directory or _next_publication(directory)
        if destination_arg.is_symlink():
            raise ValueError("review publication output is a symlink")
        destination = destination_arg.resolve(strict=False)
        source = Path(session.source_run)
        if destination.exists() or surface._overlaps(destination, source) or surface._overlaps(destination, repository) or directory == destination or directory.is_relative_to(destination):
            raise ValueError("review publication requires a fresh unprotected destination")
        if any((parent / "decisions.json").exists() and (parent / "consumer").is_dir() for parent in destination.parents):
            raise ValueError("review publication cannot modify an existing publication")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        try:
            _write_session(staging / "decisions.json", session)
            (staging / "decisions.json").chmod(0o444)
            for reference, data in executions.items():
                target = staging / "audit" / reference
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o444)
            consumer_dir = staging / "consumer"
            consumer_dir.mkdir()
            surface.write_consumer_view(plan, consumer_dir, read_only=True)
            fingerprint = directory_tree_fingerprint(staging, max_bytes=1_000_000_000)
            errors = surface._compare_consumer_directory(
                staging, _publication_plan(plan, session, executions)
            )
            if errors:
                raise ValueError("invalid review projection: " + "; ".join(errors))
            _check_unchanged(source, repository, inputs)
            if _read_session(directory / "session.json").artifact_digest != session.artifact_digest:
                raise ValueError("review decisions changed during publication")
            if directory_tree_fingerprint(staging, max_bytes=1_000_000_000) != fingerprint:
                raise ValueError("review publication staging changed")
            rename_directory_noreplace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return replace(plan, publication_directory=destination)


def _frozen_publication(view_directory: Path, review_directory: Path) -> tuple[Path, Path, ReviewSession]:
    view = surface._resolve_input_directory(view_directory, label="review consumer")
    if view.name != "consumer":
        raise ValueError("review consumer must be the publication's consumer directory")
    directory = _safe_directory(review_directory)
    active = _read_session(directory / "session.json")
    frozen = _read_session(view.parent / "decisions.json")
    excluded = {"decisions", "artifact_digest"}
    if active.model_dump(exclude=excluded) != frozen.model_dump(exclude=excluded):
        raise ValueError("publication belongs to a different review session")
    if active.decisions[:len(frozen.decisions)] != frozen.decisions:
        raise ValueError("publication decisions differ from this session's history")
    return view, directory, frozen


def preflight_review_consumer(
    view_directory: Path,
    *,
    review_directory: Path,
    repository_root: Path,
    source_run: Path | None = None,
) -> ReviewConsumerManifest:
    """Authenticate the selected projection and snapshot without source replay.

    Only the returned view may be handed to a consumer. The decision ledger and
    execution closure are host inputs, never consumer inputs.
    """
    view, directory, frozen = _frozen_publication(view_directory, review_directory)
    repository = _repository(frozen, repository_root)
    source = surface._resolve_input_directory(Path(frozen.source_run), label="review source run")
    if source_run is not None and surface._resolve_input_directory(source_run, label="authorized source run") != source:
        raise ValueError("review publication differs from authorized source run")
    before = directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000)
    inputs = _bound_inputs(frozen, repository, directory)
    executions = _execution_artifacts(view.parent / "audit", frozen)
    for decision in frozen.latest_decisions().values():
        _check_evidence(repository, decision.evidence)
    plan = render_review_consumer(inputs, frozen)
    errors = surface._compare_consumer_directory(
        view.parent, _publication_plan(plan, frozen, executions)
    )
    _check_unchanged(source, repository, inputs)
    if directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000) != before:
        errors.append("review publication changed during preflight")
    if errors:
        raise ValueError("invalid review consumer: " + "; ".join(sorted(set(errors))))
    return plan.manifest


def validate_review_consumer(view_directory: Path, *, review_directory: Path, repository_root: Path, require_complete: bool = False) -> list[str]:
    """Replay source once, then regenerate from the immutable decision snapshot."""
    from repo_constraint_memory_v3.unified.validation import INCOMPLETE_PACKAGE_ERROR, validate_unified_directory
    try:
        view, directory, frozen = _frozen_publication(view_directory, review_directory)
        repository = _repository(frozen, repository_root)
        source = Path(frozen.source_run)
        before = directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000)
        source_before = directory_tree_fingerprint(source, max_bytes=1_000_000_000)
        errors = validate_unified_directory(source, repository, require_complete=False)
        if errors:
            return ["invalid review source: " + error for error in errors]
        inputs = _bound_inputs(frozen, repository, directory)
        executions = _execution_artifacts(view.parent / "audit", frozen)
        for decision in frozen.latest_decisions().values():
            _check_evidence(repository, decision.evidence)
        plan = render_review_consumer(inputs, frozen)
        errors = surface._compare_consumer_directory(
            view.parent, _publication_plan(plan, frozen, executions)
        )
        _check_unchanged(source, repository, inputs)
        if directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000) != before or directory_tree_fingerprint(source, max_bytes=1_000_000_000) != source_before:
            errors.append("review source or publication changed during validation")
        if require_complete and inputs.completeness.state.package_status != "complete":
            errors.append(INCOMPLETE_PACKAGE_ERROR)
        return sorted(set(errors))
    except (ValueError, OSError) as error:
        return [str(error)]
