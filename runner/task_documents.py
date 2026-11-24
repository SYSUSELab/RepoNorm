"""Host-only checks for immutable CodeWiki documents inside the solver repo.

No script output from this module is injected into the coding conversation.
An unmounted image probe prevents a bind mount from hiding original content;
the mounted tree is then checked independently of the unchanged source snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from four_arm_common import ROOT, SNAPSHOT_SCRIPT, command, digest
from task_knowledge import CODEWIKI_DESTINATION, _tree

DOCUMENT_PREFIX = "docs/codewiki"

UNMOUNTED_CHECK_SCRIPT = r'''
import os
from pathlib import Path
docs = Path("docs")
if docs.is_symlink() or (docs.exists() and not docs.is_dir()):
    raise ValueError("CodeWiki parent is not a safe directory")
if os.path.lexists("docs/codewiki"):
    raise ValueError("CodeWiki destination already exists in the base image")
'''

MOUNTED_CHECK_SCRIPT = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path
expected = json.loads(sys.argv[1])
docs = Path("docs")
root = docs / "codewiki"
if docs.is_symlink() or not docs.is_dir() or root.is_symlink() or not root.is_dir():
    raise ValueError("Unsafe mounted document path")
actual = {}
for path in root.iterdir():
    if path.is_symlink() or not path.is_file():
        raise ValueError("Nonregular mounted document")
    actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise ValueError("Mounted CodeWiki content mismatch")
tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "docs/codewiki"]).split(b"\0")
if any(tracked):
    raise ValueError("CodeWiki path overlaps tracked source")
gitdir = Path(".git")
if gitdir.is_symlink() or not gitdir.is_dir():
    raise ValueError("Expected standalone task Git directory")
info = gitdir / "info"
if info.is_symlink() or (info.exists() and not info.is_dir()):
    raise ValueError("Unsafe Git info directory")
info.mkdir(exist_ok=True)
exclude = info / "exclude"
if exclude.is_symlink() or (exclude.exists() and not exclude.is_file()):
    raise ValueError("Unsafe Git exclude file")
before = exclude.read_bytes() if exclude.exists() else b""
entry = b"/docs/codewiki/"
if entry not in before.splitlines():
    with exclude.open("ab") as stream:
        stream.write(b"\n" + entry + b"\n")
print(json.dumps({"document_sha256": actual, "git_exclude": "/docs/codewiki/"}, sort_keys=True))
'''


def verify_unmounted_image(image: str, expected: dict) -> dict:
    """Read the frozen image without knowledge mounts before starting the solver."""
    output = command([
        "docker", "run", "--rm", "--network", "none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "-w", "/workspace/repo",
        "--entrypoint", "python", image, "-c", UNMOUNTED_CHECK_SCRIPT + SNAPSHOT_SCRIPT,
    ])
    actual = json.loads(output)
    if actual != expected:
        raise ValueError("Unmounted CodeWiki base image differs from knowledge snapshot")
    return {"source_worktree_digest": digest(actual), "destination_absent": True}


def protect_mounted_documents(container: str, handoff: dict, inspect: dict) -> dict:
    """Authenticate only the exact extra tree and exclude it from normal Git add."""
    source = Path(handoff["knowledge_directory"]).resolve(strict=True)
    matches = [m for m in inspect["Mounts"] if m["Destination"] == CODEWIKI_DESTINATION]
    if (len(matches) != 1 or matches[0]["RW"] or matches[0].get("Type") != "bind"
            or Path(matches[0]["Source"]).resolve() != source):
        raise ValueError("CodeWiki readonly bind mount was not applied exactly")
    return json.loads(command([
        "docker", "exec", "-w", "/workspace/repo", container, "python", "-c",
        MOUNTED_CHECK_SCRIPT, json.dumps(_tree(source), sort_keys=True),
    ]))


def assert_no_document_patch(patch: str) -> None:
    """Use Git's parser, including quoted names/renames, before any filtering."""
    if not patch.strip():
        return
    # git apply's numstat emits the destination of a rename/copy, not both
    # names. Parse the reverse too, so moving a delivery document out also fails.
    for direction in ([], ["--reverse"]):
        result = subprocess.run(["git", "apply", *direction, "--numstat", "-z"], input=patch.encode(),
                                cwd=ROOT, capture_output=True, timeout=30, check=True)
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            fields = record.split(b"\t", 2)
            if len(fields) != 3 or not fields[2]:
                raise ValueError("Unparseable patch path receipt")
            raw = fields[2]
            name = raw.decode("utf-8", errors="surrogateescape")
            if name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError("Unsafe submission path")
            if name == DOCUMENT_PREFIX or name.startswith(DOCUMENT_PREFIX + "/"):
                raise ValueError("Submitted patch contains readonly CodeWiki delivery documents")
