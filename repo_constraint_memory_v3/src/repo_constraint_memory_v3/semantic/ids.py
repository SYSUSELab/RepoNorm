"""Stable identifiers for current-source semantic artifacts."""

from __future__ import annotations

from repo_constraint_memory_v3.observations.canonical import payload_digest


def semantic_id(prefix: str, payload: object) -> str:
    """Return a stable, content-derived identifier with a bounded prefix."""

    return f"{prefix}-{payload_digest(payload)[:24]}"
