"""Authenticate staged public knowledge views before each task mount."""
from __future__ import annotations

import hashlib

import importlib

import json

import os

from pathlib import Path

import sys

import tempfile

from typing import Any

ARMS = ("raw", "codewiki", "accepted_only", "accepted_after_review")

SCHEMA = "cooperbench-task-knowledge-2.0"

DELIVERY = "benchmark-knowledge-delivery-1.0"

CODEWIKI_DESTINATION = "/workspace/repo/docs/codewiki"

class KnowledgeError(ValueError):
    pass

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise KnowledgeError(f"missing or symlink tree: {path}")
    result = {}
    entries = 0
    total_bytes = 0
    for root, directories, files in os.walk(path, followlinks=False):
        entries += len(directories) + len(files)
        if entries > 100_000:
            raise KnowledgeError("knowledge entry resource limit")
        for name in directories + files:
            entry = Path(root) / name
            if entry.is_symlink():
                raise KnowledgeError(f"symlink in knowledge: {entry}")
        for name in files:
            entry = Path(root) / name
            if not entry.is_file():
                raise KnowledgeError(f"nonregular knowledge: {entry}")
            hasher = hashlib.sha256()
            with entry.open("rb") as stream:
                while chunk := stream.read(1_048_576):
                    total_bytes += len(chunk)
                    if total_bytes > 1_000_000_000:
                        raise KnowledgeError("knowledge byte resource limit")
                    hasher.update(chunk)
            result[entry.relative_to(path).as_posix()] = hasher.hexdigest()
    return result

def verify_staged(receipt: dict) -> None:
    """Check exact staged trees immediately around the runner's container binding.

    The host owns this receipt and its digest fields; never expose it in mounts.
    This is a local content check, not source authentication or artifact replay.
    """
    if receipt["arm"] == "raw":
        if receipt.get("knowledge_directory") is not None or receipt.get("skill_directory") is not None:
            raise KnowledgeError("raw handoff must not expose knowledge or skill")
        return
    if receipt.get("schema_version") != SCHEMA or receipt.get("delivery_version") != DELIVERY:
        raise KnowledgeError("unsupported staged delivery; prepare a fresh handoff")
    if receipt.get("skill_directory") is not None or receipt.get("skill_tree_sha256") is not None:
        raise KnowledgeError("benchmark must not receive a skill")
    tree = _tree(Path(receipt["knowledge_directory"]))
    if not tree or any(not name.endswith(".md") for name in tree):
        raise KnowledgeError("staged knowledge must contain only Markdown")
    if _digest(tree) != receipt.get("knowledge_tree_sha256"):
        raise KnowledgeError("staged knowledge changed")
    if receipt["arm"] == "codewiki":
        if receipt.get("container_destination") != CODEWIKI_DESTINATION:
            raise KnowledgeError("CodeWiki destination mismatch")
        if tree != receipt.get("host_receipt", {}).get("original_markdown_sha256"):
            raise KnowledgeError("CodeWiki original Markdown mapping mismatch")
    else:
        if receipt["arm"] not in {"accepted_only", "accepted_after_review"}:
            raise KnowledgeError("unknown staged knowledge arm")
        if "START.md" not in tree or receipt.get("container_destination") != "/knowledge":
            raise KnowledgeError("Norm4Code requires START at its frozen destination")
