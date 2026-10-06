from __future__ import annotations

import pytest

from repo_constraint_memory_v3.safety import (
    TARGET_CONTENT_ADMISSION_POLICY_VERSION,
    structured_content_failure_reason,
)


def test_controlled_research_content_policy_is_frozen() -> None:
    assert TARGET_CONTENT_ADMISSION_POLICY_VERSION == (
        "controlled-research-unredacted-target-content-1.0"
    )


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("settings.json", '{"service":{"api_key":"abcdefghijklmnop"}}'),
        ("settings.toml", 'service.api_key = "abcdefghijklmnop"\n'),
        ("settings.yaml", "service: {api_key: abcdefghijklmnop}\n"),
        ("Taskfile", "env: {API_KEY: abcdefghijklmnop}\n"),
        (
            "settings.yaml",
            "private_key: |\n  -----BEGIN PRIVATE KEY-----\n  material\n",
        ),
    ],
)
def test_valid_structured_credential_values_are_admitted(
    path: str,
    content: str,
) -> None:
    assert structured_content_failure_reason(content, path=path) is None


def test_nonstructured_content_is_not_scanned_for_credentials() -> None:
    content = "API_KEY = abcdefghijklmnop\n"

    assert structured_content_failure_reason(content, path="AGENTS.md") is None
    assert structured_content_failure_reason(content) is None


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            "settings.json",
            '{"api_key":"abcdefghijklmnop","api_key":"second-value"}',
        ),
        ("settings.yaml", "api_key: abcdefghijklmnop\napi_key: second-value\n"),
        ("settings.yaml", "1: first\n01: second\n"),
        ("settings.yaml", "? [a, b]\n: first\n? [a, b]\n: second\n"),
        ("settings.toml", 'api_key = "first-value"\napi_key = "second-value"\n'),
        ("settings.json", '{"api_key":"abcdefghijklmnop",}'),
        ("settings.json", '{"value":NaN}'),
        ("settings.yaml", "{api_key: abcdefghijklmnop,\n"),
        ("settings.toml", 'api_key = "unterminated\n'),
    ],
)
def test_duplicate_or_malformed_structured_content_fails_closed(
    path: str,
    content: str,
) -> None:
    assert structured_content_failure_reason(content, path=path) == (
        "structured_content_unavailable"
    )


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("settings.json", "[" * 400 + "0" + "]" * 400),
        ("settings.toml", "value = " + "[" * 400 + "0" + "]" * 400),
        ("settings.yaml", "- " * 400 + "value\n"),
    ],
)
def test_deep_structured_content_fails_closed(
    path: str,
    content: str,
) -> None:
    assert structured_content_failure_reason(content, path=path) == (
        "structured_content_unavailable"
    )


def test_oversized_structured_content_fails_closed() -> None:
    content = '{"value":"' + "x" * 4_000_000 + '"}'

    assert structured_content_failure_reason(
        content,
        path="settings.json",
    ) == "structured_content_unavailable"


def test_structured_value_count_fails_closed() -> None:
    content = "[" + ",".join("0" for _ in range(100_001)) + "]"

    assert structured_content_failure_reason(
        content,
        path="settings.json",
    ) == "structured_content_unavailable"
