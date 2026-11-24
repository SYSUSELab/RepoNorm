"""Fail-closed atomic publication primitives for generated directory trees."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_MAX_FINGERPRINT_FILES = 100_000


@dataclass(frozen=True)
class _PublicationPreflight:
    destination: Path


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
    )


def consume_initial_empty_directory(destination: Path) -> _PublicationPreflight:
    """Remove only the same empty directory observed at initial preflight.

    Call this before a long build or validation.  A destination that appears
    afterwards is then protected by the final atomic no-replace rename instead
    of being mistaken for the operator's original empty spelling.
    """

    lexical = destination.absolute()
    if lexical.is_symlink():
        raise ValueError("publication destination is a symlink")
    target = lexical.resolve(strict=False)
    try:
        before = target.lstat()
    except FileNotFoundError:
        return _PublicationPreflight(destination=target)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("publication destination is not a regular directory")
    if os.name != "posix":
        if any(target.iterdir()):
            raise ValueError("publication destination directory is not empty")
        after = target.lstat()
        if not _same_entry(before, after):
            raise ValueError("publication destination changed during preflight")
        os.rmdir(target)
        return _PublicationPreflight(destination=target)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if not _same_entry(before, opened):
            raise ValueError("publication destination changed during preflight")
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise ValueError("publication destination directory is not empty")
        after = target.lstat()
        if not _same_entry(before, after):
            raise ValueError("publication destination changed during preflight")
        os.rmdir(target)
        return _PublicationPreflight(destination=target)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _bounded_tree_paths(
    root: Path,
    *,
    excluded_top_level: frozenset[str] = frozenset(),
    label: str = "staging tree",
    max_entries: int | None = None,
) -> tuple[Path, ...]:
    """Collect a deterministic tree order behind an explicit entry bound."""

    bound = _MAX_FINGERPRINT_FILES if max_entries is None else max_entries
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
        raise ValueError("tree entry bound must be positive")
    target = root.absolute()
    paths: list[Path] = []
    pending = [target]
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(target)
        collected: list[tuple[str, Path, bool]] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if (
                        not relative_directory.parts
                        and entry.name in excluded_top_level
                    ):
                        continue
                    if len(paths) + len(collected) >= bound:
                        raise ValueError(f"{label} exceeds entry-count bound")
                    collected.append(
                        (
                            entry.name,
                            Path(entry.path),
                            entry.is_dir(follow_symlinks=False),
                        )
                    )
        except OSError as error:
            raise ValueError(f"{label} changed during traversal") from error
        collected.sort(key=lambda item: item[0])
        for _, path, is_directory in collected:
            paths.append(path)
            if is_directory:
                pending.append(path)
    paths.sort(key=lambda item: item.as_posix())
    return tuple(paths)


def directory_tree_fingerprint(root: Path, *, max_bytes: int) -> str:
    """Hash a closed regular-file tree and reject concurrent mutations.

    Permission bits are intentionally excluded, matching consumer artifact
    identity.  Paths, directory membership, file sizes, and file bytes are
    authenticated; symlinks and non-regular entries fail closed.
    """

    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
    ):
        raise ValueError("tree fingerprint byte bound must be positive")
    target = root.absolute()
    try:
        root_before = target.lstat()
    except OSError as error:
        raise ValueError("staging tree is unavailable") from error
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("staging tree is not a regular directory")

    paths = _bounded_tree_paths(target)
    signatures: dict[Path, os.stat_result] = {target: root_before}
    items: list[tuple[object, ...]] = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(target).as_posix()
        try:
            before = path.lstat()
        except OSError as error:
            raise ValueError(f"staging entry is unavailable: {relative}") from error
        signatures[path] = before
        if stat.S_ISDIR(before.st_mode):
            items.append(("directory", relative))
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"staging entry is not a regular file: {relative}")
        total_bytes += before.st_size
        if total_bytes > max_bytes:
            raise ValueError("staging tree exceeds byte bound")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        digest = hashlib.sha256()
        observed = 0
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or not _same_entry(before, opened):
                raise ValueError(f"staging file changed during open: {relative}")
            remaining = max_bytes - total_bytes + before.st_size
            while True:
                chunk = os.read(descriptor, min(1_048_576, remaining + 1))
                if not chunk:
                    break
                observed += len(chunk)
                if observed > before.st_size or observed > remaining:
                    raise ValueError(f"staging file grew during read: {relative}")
                digest.update(chunk)
            after_read = os.fstat(descriptor)
            if (
                not _same_entry(before, after_read)
                or after_read.st_size != observed
                or after_read.st_mtime_ns != before.st_mtime_ns
            ):
                raise ValueError(f"staging file changed during read: {relative}")
        finally:
            if descriptor is not None:
                os.close(descriptor)
        items.append(("file", relative, observed, digest.hexdigest()))

    for path, expected in signatures.items():
        relative = "." if path == target else path.relative_to(target).as_posix()
        try:
            current = path.lstat()
        except OSError as error:
            raise ValueError(f"staging tree changed after read: {relative}") from error
        if (
            not _same_entry(expected, current)
            or current.st_size != expected.st_size
            or current.st_mtime_ns != expected.st_mtime_ns
        ):
            raise ValueError(f"staging tree changed after read: {relative}")
    payload = json.dumps(
        items,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raise_rename_error(error_number: int, destination: Path) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "publication destination appeared before atomic rename",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a sibling directory without replacing any target.

    Python's portable ``os.rename`` may replace an empty directory or symlink
    created after a caller's existence check.  Linux ``renameat2``, Darwin
    ``renamex_np``, and Windows ``MoveFileExW`` provide no-replace semantics.
    Platforms without such a primitive fail closed instead of silently
    weakening the publication contract.
    """

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        move_file.restype = ctypes.c_int
        if move_file(str(source), str(destination), 0):
            return
        error_number = ctypes.get_last_error()
        if error_number in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
            raise FileExistsError(
                error_number,
                "publication destination appeared before atomic rename",
                str(destination),
            )
        raise OSError(
            error_number,
            ctypes.FormatError(error_number),
            str(destination),
        )

    if os.name != "posix":
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            str(destination),
        )

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is not None:
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            if renamex_np(
                os.fsencode(source),
                os.fsencode(destination),
                _RENAME_EXCL,
            ) == 0:
                return
            _raise_rename_error(ctypes.get_errno(), destination)

    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        ) == 0:
            return
        _raise_rename_error(ctypes.get_errno(), destination)

    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory publication is unavailable",
        str(destination),
    )


__all__ = [
    "_bounded_tree_paths",
    "consume_initial_empty_directory",
    "directory_tree_fingerprint",
    "rename_directory_noreplace",
]
