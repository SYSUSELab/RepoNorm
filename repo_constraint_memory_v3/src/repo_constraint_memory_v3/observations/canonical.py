"""Canonical serialization and digest helpers for observation artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from pydantic import BaseModel


def jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def canonical_json_text(value: object) -> str:
    return (
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def payload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_hex(encoded)


def model_payload_digest(model: BaseModel, *, digest_field: str) -> str:
    payload = model.model_dump(mode="json", exclude={digest_field})
    return payload_digest(payload)
