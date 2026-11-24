"""Untrusted success-only structured-inference cache and run resume source."""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, model_validator

from repo_constraint_memory_v3.models import Sha256Hex, StrictModel
from repo_constraint_memory_v3.observations.canonical import (
    canonical_json_bytes,
    model_payload_digest,
    payload_digest,
)

INFERENCE_CACHE_SCHEMA_VERSION = "success-only-inference-cache-1.0"


class InferenceCacheEntry(StrictModel):
    schema_version: Literal["success-only-inference-cache-1.0"]
    cache_key: Sha256Hex
    inference_record: dict[str, object]
    entry_digest: Sha256Hex

    @model_validator(mode="after")
    def digests_are_valid(self) -> InferenceCacheEntry:
        if self.entry_digest != model_payload_digest(
            self,
            digest_field="entry_digest",
        ):
            raise ValueError("inference-cache entry digest mismatch")
        return self


@dataclass
class InferenceCacheMetrics:
    hits: int = 0
    misses: int = 0
    rejected: int = 0
    writes: int = 0
    resume_entries: int = 0


class SuccessInferenceCache:
    """Reuse only strict terminal successes and revalidate them at every hit."""

    def __init__(
        self,
        directory: Path | None,
        *,
        resume_from: Path | None,
        repository_root: Path,
        max_entry_bytes: int,
        max_resume_entries: int,
    ) -> None:
        declared_directory = (
            directory.absolute() if directory is not None else None
        )
        declared_resume = (
            resume_from.absolute() if resume_from is not None else None
        )
        self.directory = (
            declared_directory.resolve()
            if declared_directory is not None
            else None
        )
        self.resume_from = (
            declared_resume.resolve() if declared_resume is not None else None
        )
        self.max_entry_bytes = max_entry_bytes
        self.max_resume_entries = max_resume_entries
        self.metrics = InferenceCacheMetrics()
        self._lock = threading.Lock()
        self._resume_by_key: dict[str, dict[str, object]] = {}
        resolved_root = repository_root.resolve()
        if self.directory is not None:
            self._validate_external_directory(
                self.directory,
                declared_directory=declared_directory,
                repository_root=resolved_root,
                label="inference cache",
                create=True,
            )
        if self.resume_from is not None:
            self._validate_external_directory(
                self.resume_from,
                declared_directory=declared_resume,
                repository_root=resolved_root,
                label="resume run",
                create=False,
            )
            self._load_resume_run()

    @staticmethod
    def _validate_external_directory(
        directory: Path,
        *,
        declared_directory: Path,
        repository_root: Path,
        label: str,
        create: bool,
    ) -> None:
        if (
            directory == repository_root
            or repository_root in directory.parents
            or declared_directory == repository_root
            or repository_root in declared_directory.parents
        ):
            raise ValueError(f"{label} must be outside the target repository")
        if declared_directory.exists() and declared_directory.is_symlink():
            raise ValueError(f"{label} directory must not be a symlink")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise ValueError(f"{label} directory does not exist: {directory}")

    def _load_resume_run(self) -> None:
        assert self.resume_from is not None
        request_root = self.resume_from / "inference" / "requests"
        if request_root.is_symlink() or not request_root.is_dir():
            raise ValueError("resume run has no regular inference/requests directory")
        resolved_request_root = request_root.resolve()
        if self.resume_from not in resolved_request_root.parents:
            raise ValueError("resume inference directory escapes the resume run")
        request_root = resolved_request_root
        paths = sorted(request_root.glob("INFER-*.json"))
        if len(paths) > self.max_resume_entries:
            raise ValueError("resume run exceeds configured inference-entry limit")

        # Imported lazily so inference.py can depend only on the cache protocol.
        from repo_constraint_memory_v3.inference import (
            InferenceRecord,
            inference_cache_key_for_record,
        )

        conflicts: set[str] = set()
        for path in paths:
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("resume inference artifact is not a regular file")
                content = path.read_bytes()
                if len(content) > self.max_entry_bytes:
                    raise ValueError("resume inference artifact exceeds cache limit")
                record = InferenceRecord.model_validate_json(content)
                if content != canonical_json_bytes(record):
                    raise ValueError("resume inference artifact is not canonical JSON")
                if (
                    record.terminal_status != "success"
                    or record.structured_response is None
                ):
                    continue
                key = inference_cache_key_for_record(record)
                previous = self._resume_by_key.get(key)
                record_payload = record.model_dump(mode="json")
                if previous is not None and previous != record_payload:
                    conflicts.add(key)
                    self._resume_by_key.pop(key, None)
                    continue
                if key not in conflicts:
                    self._resume_by_key[key] = record_payload
            except (OSError, UnicodeDecodeError, ValidationError, ValueError):
                self.metrics.rejected += 1
        self.metrics.rejected += len(conflicts)
        self.metrics.resume_entries = len(self._resume_by_key)

    def _path(self, key: str, *, create_parent: bool = False) -> Path:
        assert self.directory is not None
        shard = self.directory / key[:2]
        if create_parent:
            shard.mkdir(parents=False, exist_ok=True)
        elif not shard.exists():
            return shard / f"{key}.json"
        if (
            shard.is_symlink()
            or not shard.is_dir()
            or shard.resolve().parent != self.directory
        ):
            raise ValueError("inference-cache shard directory is unsafe")
        path = shard / f"{key}.json"
        if path.exists() and path.is_symlink():
            raise ValueError("inference-cache entry must not be a symlink")
        return path

    def _read_external(self, key: str) -> dict[str, object] | None:
        if self.directory is None:
            return None
        try:
            path = self._path(key)
            if not path.exists():
                return None
            if path.is_symlink() or not path.is_file():
                raise ValueError("inference-cache entry is not a regular file")
            content = path.read_bytes()
            if len(content) > self.max_entry_bytes:
                raise ValueError("inference-cache entry exceeds configured limit")
            entry = InferenceCacheEntry.model_validate_json(content)
            if entry.cache_key != key:
                raise ValueError("inference-cache key mismatch")
            if content != canonical_json_bytes(entry):
                raise ValueError("inference-cache entry is not canonical JSON")
        except (OSError, UnicodeDecodeError, ValidationError, ValueError):
            with self._lock:
                self.metrics.rejected += 1
            return None
        return dict(entry.inference_record)

    def read(self, key: str) -> dict[str, object] | None:
        response = self._read_external(key)
        if response is None:
            response = self._resume_by_key.get(key)
            response = dict(response) if response is not None else None
        with self._lock:
            if response is None:
                self.metrics.misses += 1
            else:
                self.metrics.hits += 1
        return response

    def record_admission_rejected(self) -> None:
        """Reclassify a syntactic hit that cannot fit the logical budget."""

        with self._lock:
            if self.metrics.hits <= 0:
                raise ValueError("cache admission rejection has no prior hit")
            self.metrics.hits -= 1
            self.metrics.misses += 1

    def write(self, key: str, record: dict[str, object]) -> None:
        if self.directory is None:
            return
        payload = {
            "schema_version": INFERENCE_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "inference_record": record,
        }
        entry = InferenceCacheEntry.model_validate(
            {**payload, "entry_digest": payload_digest(payload)}
        )
        content = canonical_json_bytes(entry)
        if len(content) > self.max_entry_bytes:
            return
        try:
            path = self._path(key, create_parent=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{key}.",
                suffix=".tmp",
                dir=path.parent,
            )
        except (OSError, ValueError):
            with self._lock:
                self.metrics.rejected += 1
            return
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_path, path)
        except FileExistsError:
            return
        except OSError:
            with self._lock:
                self.metrics.rejected += 1
            return
        finally:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        with self._lock:
            self.metrics.writes += 1
