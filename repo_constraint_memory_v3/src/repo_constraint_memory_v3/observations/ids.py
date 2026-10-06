"""Centralized, versioned identifiers for Python observation artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

FILE_CONTRACT_VERSION = "python-file-id-1.0"
ENTITY_CONTRACT_VERSION = "python-entity-id-1.0"
SITE_CONTRACT_VERSION = "python-site-id-1.0"
RELATION_CONTRACT_VERSION = "python-relation-id-1.1"
CLONE_CONTRACT_VERSION = "python-clone-id-1.0"


def _identifier(prefix: str, payload: Iterable[object]) -> str:
    encoded = json.dumps(
        list(payload),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def file_id(path: str, content_sha256: str) -> str:
    return _identifier("PYFILE", [FILE_CONTRACT_VERSION, path, content_sha256])


def entity_id(
    path: str,
    kind: str,
    qualified_name: str,
    lexical_ordinal: int,
) -> str:
    return _identifier(
        "PYENT",
        [
            ENTITY_CONTRACT_VERSION,
            path,
            kind,
            qualified_name,
            lexical_ordinal,
        ],
    )


def site_id(
    path: str,
    content_sha256: str,
    node_kind: str,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
) -> str:
    return _identifier(
        "PYSITE",
        [
            SITE_CONTRACT_VERSION,
            path,
            content_sha256,
            node_kind,
            start_line,
            start_column,
            end_line,
            end_column,
        ],
    )


def relation_id(
    kind: str,
    source_ref: str,
    target_ref: str,
    site_ref: str | None,
) -> str:
    return _identifier(
        "PYREL",
        [RELATION_CONTRACT_VERSION, kind, source_ref, target_ref, site_ref],
    )


def symbol_ref_identity(
    rendered: str,
    *,
    entity_id: str | None,
    file_id: str | None,
) -> str:
    if entity_id is not None:
        return entity_id
    if file_id is not None:
        return f"{file_id}:{rendered}"
    return rendered


def clone_cluster_id(
    member_file_ids: Iterable[str],
    match_basis: Iterable[str],
) -> str:
    return _identifier(
        "PYCLONE",
        [
            CLONE_CONTRACT_VERSION,
            sorted(member_file_ids),
            sorted(match_basis),
        ],
    )
