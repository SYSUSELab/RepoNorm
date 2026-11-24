"""Bounded, deterministic explicit-source discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from repo_constraint_memory_v3.models import (
    DiscoveryArtifact,
    DiscoverySettings,
    ManifestFile,
    ManifestSection,
    RepositoryMetadata,
    SCHEMA_VERSION,
    SkippedPath,
    SourceKind,
    SourceMetadata,
)
from repo_constraint_memory_v3.safety import structured_content_failure_reason

_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
    "venv",
}
_BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".db",
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".sqlite",
    ".tar",
    ".tgz",
    ".tiff",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}
_LOCKFILE_NAMES = {
    "bun.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "package-lock.json",
    "pdm.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}
_PRIVATE_KEY_NAMES = {"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
_SUPPORTED_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".markdown",
    ".md",
    ".mdx",
    ".rst",
    ".text",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*]\(([^)\s]+)(?:\s+[^)]*)?\)")


class DiscoveryError(RuntimeError):
    """The repository cannot be safely discovered."""


class UnsupportedRepositoryError(DiscoveryError):
    """No eligible Python source signal exists in the manifest."""


@dataclass(frozen=True)
class WorkspaceManifest:
    root: Path
    section: ManifestSection
    snapshot_digest: str
    complete: bool
    incomplete_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SourceFile:
    path: str
    source_kind: SourceKind
    discovery_origin: str
    reason: str
    referenced_by: tuple[str, ...]
    content_sha256: str
    size_bytes: int
    content: str


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_relative_posix_path(path: str) -> bool:
    """Accept only canonical, non-escaping repository-relative POSIX paths."""

    if not path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    return (
        not pure.is_absolute()
        and path == pure.as_posix()
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def _is_secret_filename(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if name in _PRIVATE_KEY_NAMES:
        return True
    return name.endswith((".key", ".p12", ".pfx", ".pem"))


def _baseline_skip_reason(path: str) -> str | None:
    pure = PurePosixPath(path)
    folded_directories = {part.casefold() for part in pure.parts[:-1]}
    if folded_directories & _IGNORED_DIRECTORY_NAMES:
        return "excluded_directory"
    name = pure.name.casefold()
    if name in _LOCKFILE_NAMES or name.endswith((".lock", ".min.css", ".min.js")):
        return "lock_or_generated_file"
    if pure.suffix.casefold() in _BINARY_SUFFIXES:
        return "binary_extension"
    if _is_secret_filename(path):
        return "sensitive_filename"
    return None


def _fallback_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                paths.append(relative)
            elif dirname.casefold() not in _IGNORED_DIRECTORY_NAMES:
                retained.append(dirname)
        dirnames[:] = retained
        for filename in sorted(filenames):
            paths.append((current_path / filename).relative_to(root).as_posix())
    return paths


def enumerate_workspace_paths(
    root: Path,
    *,
    candidate_paths: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Enumerate current files without reading Git history or repository code."""

    if candidate_paths is not None:
        return "provided", list(candidate_paths)
    if (root / ".git").exists():
        process = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if process.returncode != 0:
            message = process.stderr.decode("utf-8", errors="replace").strip()
            raise DiscoveryError(f"git ls-files failed: {message}")
        try:
            decoded = process.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise DiscoveryError("git returned a non-UTF-8 path") from error
        return "git", [path for path in decoded.split("\0") if path]
    return "walk", _fallback_paths(root)


def is_supported_explicit_path(path: str) -> bool:
    """Return whether supported text can be proposed as an explicit source."""

    pure = PurePosixPath(path)
    name = pure.name.casefold()
    if pure.suffix.casefold() in {".py", ".pyi"}:
        return False
    return (
        pure.suffix.casefold() in _SUPPORTED_TEXT_SUFFIXES
        or name in {".editorconfig", ".flake8", ".coveragerc"}
        or name in {"makefile", "gnumakefile", "justfile"}
        or name in {"taskfile.yml", "taskfile.yaml"}
    )


def source_kind_for_path(path: str) -> SourceKind | None:
    """Classify a supported path without inferring any constraint."""

    if not is_supported_explicit_path(path):
        return None
    pure = PurePosixPath(path)
    folded_name = pure.name.casefold()
    folded_path = path.casefold()
    if folded_name in {"agents.md", "claude.md"}:
        return "instruction"
    if folded_path.startswith(".github/pull_request_template/"):
        return "instruction"
    if folded_path == ".github/copilot-instructions.md":
        return "instruction"
    if len(pure.parts) == 1 and folded_name == "security.md":
        return "instruction"
    if len(pure.parts) == 1 and folded_name.startswith(
        ("contributing", "development", "styleguide", "coding")
    ):
        return "instruction"
    if len(pure.parts) == 1 and folded_name.startswith("readme"):
        return "readme"
    if (
        folded_path.startswith(".github/workflows/")
        and pure.suffix.casefold() in {".yml", ".yaml"}
    ):
        return "workflow"
    if folded_name in {
        "makefile",
        "gnumakefile",
        "justfile",
        "taskfile.yml",
        "taskfile.yaml",
    }:
        return "command"
    if pure.suffix.casefold() in {".md", ".markdown", ".mdx", ".rst", ".text", ".txt"}:
        return "documentation"
    return "configuration"


def _manifest_role(path: str) -> str:
    if PurePosixPath(path).suffix.casefold() in {".py", ".pyi"}:
        return "python_source"
    if is_supported_explicit_path(path):
        return "supported_text"
    return "other"


def build_workspace_manifest(
    root: Path,
    settings: DiscoverySettings,
    *,
    candidate_paths: Sequence[str] | None = None,
) -> WorkspaceManifest:
    """Hash a stable, bounded manifest of safe current-workspace files."""

    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise DiscoveryError(f"repository root is not a directory: {root}")
    mode, raw_paths = enumerate_workspace_paths(
        resolved_root,
        candidate_paths=candidate_paths,
    )
    files: dict[str, ManifestFile] = {}
    skipped: list[SkippedPath] = []
    incomplete_reasons: set[str] = set()
    total_bytes = 0

    for raw_path in sorted(set(raw_paths)):
        path = raw_path.removeprefix("./")
        if not valid_relative_posix_path(path):
            skipped.append(SkippedPath(path=raw_path or "<empty>", reason="unsafe_path"))
            continue
        skip_reason = _baseline_skip_reason(path)
        if skip_reason is not None:
            skipped.append(SkippedPath(path=path, reason=skip_reason))
            continue
        absolute = resolved_root.joinpath(*PurePosixPath(path).parts)
        try:
            metadata = absolute.lstat()
        except FileNotFoundError:
            skipped.append(SkippedPath(path=path, reason="missing"))
            continue
        if stat.S_ISLNK(metadata.st_mode):
            skipped.append(SkippedPath(path=path, reason="symlink"))
            continue
        if not stat.S_ISREG(metadata.st_mode):
            skipped.append(SkippedPath(path=path, reason="not_regular_file"))
            continue
        if metadata.st_size > settings.max_manifest_file_bytes:
            skipped.append(SkippedPath(path=path, reason="oversized_manifest_file"))
            incomplete_reasons.add("max_manifest_file_bytes_exceeded")
            continue
        if len(files) >= settings.max_manifest_files:
            skipped.append(SkippedPath(path=path, reason="manifest_file_count_budget"))
            incomplete_reasons.add("max_manifest_files_exceeded")
            continue
        if total_bytes + metadata.st_size > settings.max_manifest_total_bytes:
            skipped.append(SkippedPath(path=path, reason="manifest_total_bytes_budget"))
            incomplete_reasons.add("max_manifest_total_bytes_exceeded")
            continue
        try:
            content = absolute.read_bytes()
        except OSError as error:
            skipped.append(
                SkippedPath(path=path, reason=f"read_error:{type(error).__name__}")
            )
            incomplete_reasons.add("manifest_read_error")
            continue
        if b"\0" in content[:8192]:
            skipped.append(SkippedPath(path=path, reason="binary_content"))
            continue
        files[path] = ManifestFile(
            path=path,
            content_sha256=sha256_bytes(content),
            size_bytes=len(content),
            suffix=PurePosixPath(path).suffix.casefold(),
            role=_manifest_role(path),  # type: ignore[arg-type]
        )
        total_bytes += len(content)

    ordered_files = [files[path] for path in sorted(files)]
    file_paths = [item.path for item in ordered_files]
    digest_items = [(item.path, item.content_sha256) for item in ordered_files]
    section = ManifestSection(
        mode=mode,  # type: ignore[arg-type]
        file_list_sha256=sha256_bytes(canonical_json_bytes(file_paths)),
        files=ordered_files,
        skipped=sorted(skipped, key=lambda item: (item.path, item.reason)),
    )
    return WorkspaceManifest(
        root=resolved_root,
        section=section,
        snapshot_digest=f"sha256:{sha256_bytes(canonical_json_bytes(digest_items))}",
        complete=not incomplete_reasons,
        incomplete_reasons=tuple(sorted(incomplete_reasons)),
    )


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    result = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    result.append("(?:.*/)?")
                    index += 1
                else:
                    result.append(".*")
                continue
            result.append("[^/]*")
        elif character == "?":
            result.append("[^/]")
        elif character == "[":
            closing = pattern.find("]", index + 1)
            if closing != -1:
                result.append(pattern[index : closing + 1])
                index = closing
            else:
                result.append(r"\[")
        else:
            result.append(re.escape(character))
        index += 1
    result.append("$")
    return re.compile("".join(result))


def matches_glob(path: str, pattern: str) -> bool:
    try:
        return bool(_glob_regex(pattern).fullmatch(path))
    except re.error as error:
        raise ValueError("glob pattern contains an invalid character class") from error


def _is_excluded(path: str, patterns: Iterable[str]) -> bool:
    return any(matches_glob(path, pattern) for pattern in patterns)


def _is_deterministic_registry_source(path: str) -> tuple[SourceKind, str, int] | None:
    pure = PurePosixPath(path)
    name = pure.name
    folded_name = name.casefold()
    folded_path = path.casefold()
    if folded_name in {"agents.md", "claude.md"}:
        return "instruction", "registry:nested_instruction", 0
    if len(pure.parts) == 1 and folded_name.startswith("readme"):
        return "readme", "registry:root_readme", 1
    if len(pure.parts) == 1 and folded_name.startswith(
        ("contributing", "development", "styleguide", "coding")
    ):
        return "instruction", "registry:root_contributor_document", 0
    if len(pure.parts) == 1 and folded_name == "security.md":
        return "instruction", "registry:root_security_document", 1
    if folded_path == ".github/copilot-instructions.md":
        return "instruction", "registry:copilot_instructions", 0
    if folded_path.startswith(".github/pull_request_template/"):
        return "instruction", "registry:pull_request_template", 1
    if (
        folded_path.startswith(".github/workflows/")
        and pure.suffix.casefold() in {".yml", ".yaml"}
    ):
        return "workflow", "registry:workflow", 4
    if len(pure.parts) != 1:
        return None
    if folded_name in {
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "pytest.ini",
        "ruff.toml",
        "mypy.ini",
        "pyrightconfig.json",
        ".editorconfig",
        ".flake8",
        ".coveragerc",
    }:
        return "configuration", "registry:root_configuration", 2
    if folded_name in {
        "makefile",
        "gnumakefile",
        "justfile",
        "taskfile.yml",
        "taskfile.yaml",
    }:
        return "command", "registry:root_command", 3
    return None


def _normalize_link_target(source_path: str, raw_target: str) -> str | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "/", "\\")):
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target.casefold().endswith((".md", ".markdown", ".rst")):
        return None
    joined = PurePosixPath(source_path).parent / PurePosixPath(target)
    parts: list[str] = []
    for part in joined.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        return None
    normalized = PurePosixPath(*parts).as_posix()
    if not valid_relative_posix_path(normalized):
        return None
    if not normalized.startswith("docs/"):
        return None
    return normalized


def _read_utf8(root: Path, item: ManifestFile) -> str:
    content = root.joinpath(*PurePosixPath(item.path).parts).read_bytes()
    if sha256_bytes(content) != item.content_sha256:
        raise DiscoveryError(f"file changed during discovery: {item.path}")
    return content.decode("utf-8", errors="strict").replace("\r\n", "\n")


def _git_head_for_reference(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return None
    value = process.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def discover_repository(
    root: Path,
    settings: DiscoverySettings,
    *,
    repository_name: str | None = None,
    candidate_paths: Sequence[str] | None = None,
) -> DiscoveryArtifact:
    """Build the complete deterministic discovery artifact."""

    manifest = build_workspace_manifest(
        root,
        settings,
        candidate_paths=candidate_paths,
    )
    files_by_path = {item.path: item for item in manifest.section.files}
    if not any(item.role == "python_source" for item in manifest.section.files):
        raise UnsupportedRepositoryError(
            "unsupported repository: no eligible .py/.pyi file exists in the manifest"
        )

    incomplete_reasons = set(manifest.incomplete_reasons)
    skipped = list(manifest.section.skipped)
    excluded = {
        item.path
        for item in manifest.section.files
        if _is_excluded(item.path, settings.exclude)
    }
    registry: dict[str, tuple[SourceKind, str, int, set[str]]] = {}
    for item in manifest.section.files:
        if item.path in excluded:
            continue
        selected = _is_deterministic_registry_source(item.path)
        if selected is not None:
            kind, reason, priority = selected
            registry[item.path] = (kind, reason, priority, set())

    decoded: dict[str, str] = {}

    def load_candidate(path: str, *, baseline: bool) -> str | None:
        if path in decoded:
            return decoded[path]
        item = files_by_path[path]
        if item.size_bytes > settings.max_source_file_bytes:
            skipped.append(SkippedPath(path=path, reason="oversized_source"))
            if baseline:
                incomplete_reasons.add("oversized_baseline_source")
            return None
        try:
            value = _read_utf8(manifest.root, item)
        except UnicodeDecodeError:
            skipped.append(SkippedPath(path=path, reason="invalid_utf8"))
            if baseline:
                incomplete_reasons.add("invalid_utf8_baseline_source")
            return None
        decoded[path] = value
        return value

    if settings.include_linked_docs_one_hop:
        linked_by: dict[str, set[str]] = {}
        for source_path in sorted(registry):
            if PurePosixPath(source_path).suffix.casefold() not in {
                ".md",
                ".markdown",
                ".mdx",
                ".rst",
            }:
                continue
            content = load_candidate(source_path, baseline=True)
            if content is None:
                continue
            for match in _MARKDOWN_LINK_RE.finditer(content):
                target = _normalize_link_target(source_path, match.group(1))
                if target is not None:
                    linked_by.setdefault(target, set()).add(source_path)
        for target, referrers in sorted(linked_by.items()):
            if target in excluded:
                continue
            if target not in files_by_path:
                skipped.append(
                    SkippedPath(path=target, reason="linked_target_unavailable")
                )
                incomplete_reasons.add("linked_target_unavailable")
                continue
            if source_kind_for_path(target) != "documentation":
                continue
            if target in registry:
                kind, reason, priority, existing = registry[target]
                registry[target] = (
                    kind,
                    reason,
                    priority,
                    existing | referrers,
                )
            else:
                registry[target] = (
                    "documentation",
                    "one_hop_local_docs_link",
                    5,
                    set(referrers),
                )

    eligible_baseline: list[tuple[int, SourceMetadata]] = []
    for path, (kind, reason, priority, referrers) in sorted(registry.items()):
        content = load_candidate(path, baseline=True)
        if content is None:
            continue
        structural_failure = structured_content_failure_reason(
            content,
            path=path,
        )
        if structural_failure is not None:
            skipped.append(
                SkippedPath(
                    path=path,
                    reason=structural_failure,
                )
            )
            incomplete_reasons.add("structured_baseline_source_unavailable")
            continue
        item = files_by_path[path]
        eligible_baseline.append(
            (
                priority,
                SourceMetadata(
                    path=path,
                    source_kind=kind,
                    reason=reason,
                    content_sha256=item.content_sha256,
                    size_bytes=item.size_bytes,
                    referenced_by=sorted(referrers),
                ),
            )
        )

    selected_baseline: list[SourceMetadata] = []
    source_bytes = 0
    for _, source in sorted(eligible_baseline, key=lambda value: (value[0], value[1].path)):
        if len(selected_baseline) >= settings.max_source_files:
            skipped.append(
                SkippedPath(path=source.path, reason="source_file_count_budget")
            )
            incomplete_reasons.add("max_source_files_exceeded")
            continue
        if source_bytes + source.size_bytes > settings.max_total_source_bytes:
            skipped.append(
                SkippedPath(path=source.path, reason="source_total_bytes_budget")
            )
            incomplete_reasons.add("max_total_source_bytes_exceeded")
            continue
        selected_baseline.append(source)
        source_bytes += source.size_bytes
    selected_baseline.sort(key=lambda item: item.path)
    registry_paths = set(registry)

    supplement_candidates: list[SourceMetadata] = []
    for item in manifest.section.files:
        path = item.path
        if (
            path in excluded
            or path in registry_paths
            or not is_supported_explicit_path(path)
        ):
            continue
        if item.size_bytes > settings.max_source_file_bytes:
            skipped.append(
                SkippedPath(path=path, reason="oversized_supplement_candidate")
            )
            continue
        content = load_candidate(path, baseline=False)
        if content is None:
            continue
        structural_failure = structured_content_failure_reason(
            content,
            path=path,
        )
        if structural_failure is not None:
            skipped.append(
                SkippedPath(
                    path=path,
                    reason=structural_failure,
                )
            )
            continue
        kind = source_kind_for_path(path)
        if kind is None:
            continue
        supplement_candidates.append(
            SourceMetadata(
                path=path,
                source_kind=kind,
                reason="supported_text_candidate",
                content_sha256=item.content_sha256,
                size_bytes=item.size_bytes,
                referenced_by=[],
            )
        )

    section = manifest.section.model_copy(
        update={
            "skipped": sorted(
                {(item.path, item.reason): item for item in skipped}.values(),
                key=lambda item: (item.path, item.reason),
            )
        }
    )
    resolved_root = manifest.root
    return DiscoveryArtifact(
        schema_version=SCHEMA_VERSION,
        repository=RepositoryMetadata(
            name=(repository_name.strip() if repository_name else resolved_root.name),
            root=resolved_root.as_posix(),
            snapshot_digest=manifest.snapshot_digest,
            git_head_for_reference=_git_head_for_reference(resolved_root),
        ),
        language_scope="python",
        limits=settings,
        manifest=section,
        baseline_sources=selected_baseline,
        supplement_candidates=sorted(
            supplement_candidates,
            key=lambda item: item.path,
        ),
        complete=not incomplete_reasons,
        incomplete_reasons=sorted(incomplete_reasons),
    )


def load_source_file(
    root: Path,
    metadata: SourceMetadata,
    *,
    origin: str,
) -> SourceFile:
    """Read one already-admitted source and bind it to its manifest hash."""

    if not valid_relative_posix_path(metadata.path):
        raise DiscoveryError(f"unsafe selected source path: {metadata.path}")
    absolute = root.resolve().joinpath(*PurePosixPath(metadata.path).parts)
    try:
        stat_result = absolute.lstat()
    except FileNotFoundError as error:
        raise DiscoveryError(f"selected source disappeared: {metadata.path}") from error
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISREG(stat_result.st_mode):
        raise DiscoveryError(f"selected source is not a regular file: {metadata.path}")
    content_bytes = absolute.read_bytes()
    if len(content_bytes) != metadata.size_bytes:
        raise DiscoveryError(f"selected source size changed: {metadata.path}")
    if sha256_bytes(content_bytes) != metadata.content_sha256:
        raise DiscoveryError(f"selected source content changed: {metadata.path}")
    try:
        content = content_bytes.decode("utf-8", errors="strict").replace("\r\n", "\n")
    except UnicodeDecodeError as error:
        raise DiscoveryError(f"selected source is not UTF-8: {metadata.path}") from error
    return SourceFile(
        path=metadata.path,
        source_kind=metadata.source_kind,
        discovery_origin=origin,
        reason=metadata.reason,
        referenced_by=tuple(metadata.referenced_by),
        content_sha256=metadata.content_sha256,
        size_bytes=metadata.size_bytes,
        content=content,
    )
