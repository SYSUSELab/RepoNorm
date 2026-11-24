"""Frozen execution-model profiles shared by every real-model consumer.

Profile files are a trust boundary: an explicitly supplied path can contain
untrusted YAML.  Loading therefore has fixed byte, event, nesting, alias, and
mapping-key limits before Pydantic validates the normalized contract.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from yaml.events import AliasEvent, CollectionEndEvent, CollectionStartEvent
from yaml.nodes import MappingNode


EXECUTION_MODEL_PROFILE_SCHEMA_VERSION = "execution-model-profile-1.0"
DEFAULT_EXECUTION_MODEL_PROFILE_ID = (
    "apiyi-qwen3.8-flash-thinking-medium"
)
OFFLINE_EXECUTION_MODEL_PROFILE_ID = "offline-deterministic-test"

# These bounds apply to both checked-in and caller-supplied profiles.  Built-in
# resources are trusted source files, but they must not silently bypass the
# serialized contract enforced for an external profile.
MAX_EXECUTION_MODEL_PROFILE_BYTES = 64 * 1024
MAX_EXECUTION_MODEL_PROFILE_DEPTH = 16
MAX_EXECUTION_MODEL_PROFILE_NODES = 512
MAX_BUILTIN_EXECUTION_MODEL_PROFILES = 64
MAX_EFFECTIVE_MODEL_IDS = 16

_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_BUILTIN_FILENAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}\.yaml$")

NonEmptyString = Annotated[str, Field(min_length=1, max_length=512)]
ProfileId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=_PROFILE_ID_PATTERN.pattern),
]
EnvironmentKey = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=_ENVIRONMENT_KEY_PATTERN.pattern,
    ),
]


def _has_surrounding_whitespace(value: str) -> bool:
    return value != value.strip()


class _StrictProfileModel(BaseModel):
    """Reject coercion and mutation at the normalized-profile boundary."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ProviderBinding(_StrictProfileModel):
    provider_id: NonEmptyString
    protocol: Literal[
        "openai_chat_completions",
        "openai_responses",
        "offline_mock",
    ]
    base_url: NonEmptyString | None = None
    api_key_env: EnvironmentKey | None = None

    @field_validator("provider_id")
    @classmethod
    def provider_id_is_canonical(cls, value: str) -> str:
        if _has_surrounding_whitespace(value):
            raise ValueError("provider_id cannot contain surrounding whitespace")
        return value

    @field_validator("base_url")
    @classmethod
    def endpoint_is_canonical_https(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _has_surrounding_whitespace(value) or value.endswith("/"):
            raise ValueError(
                "provider base_url must be canonical and omit a trailing slash"
            )
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("real provider base_url must be a credential-free HTTPS URL")
        return value

    @model_validator(mode="after")
    def live_provider_requires_endpoint_and_credential(self) -> ProviderBinding:
        if self.protocol == "offline_mock":
            if self.base_url is not None or self.api_key_env is not None:
                raise ValueError(
                    "offline model profile cannot contain endpoint or credential settings"
                )
        elif self.base_url is None or self.api_key_env is None:
            raise ValueError("real model profile requires base_url and api_key_env")
        return self


class AdapterModelIds(_StrictProfileModel):
    core_openai_chat: NonEmptyString
    mini_swe_litellm: NonEmptyString
    downstream_codex: NonEmptyString

    @field_validator(
        "core_openai_chat",
        "mini_swe_litellm",
        "downstream_codex",
    )
    @classmethod
    def wire_model_id_is_canonical(cls, value: str) -> str:
        if _has_surrounding_whitespace(value):
            raise ValueError("wire model ID cannot contain surrounding whitespace")
        return value


class GenerationProfile(_StrictProfileModel):
    temperature: Annotated[int | float, Field(ge=0, le=2)] = 0

    @field_validator("temperature")
    @classmethod
    def temperature_is_finite(cls, value: int | float) -> int | float:
        if not math.isfinite(value):
            raise ValueError("temperature must be finite")
        return value


class ReasoningProfile(_StrictProfileModel):
    mode: Literal["disabled", "enabled"]
    effort: (
        Literal["none", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    max_tokens: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    wire_policy: Literal[
        "deepseek_thinking_type",
        "qwen_enable_thinking",
        "openrouter_reasoning",
        "openai_chat_reasoning_effort",
        "openai_responses_reasoning_effort",
        "offline_mock",
    ]

    @model_validator(mode="after")
    def mode_and_effort_are_explicit(self) -> ReasoningProfile:
        if self.mode == "disabled" and (
            self.effort is not None or self.max_tokens is not None
        ):
            raise ValueError("disabled reasoning cannot carry a control")
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("reasoning effort and max_tokens are mutually exclusive")
        if self.max_tokens is not None and self.wire_policy != "openrouter_reasoning":
            raise ValueError("reasoning max_tokens requires the OpenRouter wire")
        if (
            self.mode == "enabled"
            and self.wire_policy != "openrouter_reasoning"
            and self.effort is None
        ):
            raise ValueError("enabled reasoning requires an explicit effort")
        if self.wire_policy == "offline_mock" and self.mode != "disabled":
            raise ValueError("offline mock reasoning must be disabled")
        if (
            self.wire_policy == "qwen_enable_thinking"
            and self.mode == "enabled"
            and self.effort not in {"low", "medium", "xhigh"}
        ):
            raise ValueError("Qwen thinking requires low, medium, or xhigh effort")
        return self

    @model_serializer(mode="wrap")
    def omit_unused_max_tokens(self, handler):
        payload = handler(self)
        if self.max_tokens is None:
            payload.pop("max_tokens", None)
        return payload


def openrouter_reasoning_payload(
    *,
    mode: Literal["disabled", "enabled"],
    effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None,
    max_tokens: int | None,
) -> dict[str, object]:
    """Render the exact mutually exclusive OpenRouter reasoning control."""

    if mode == "disabled":
        return {"enabled": False}
    if max_tokens is not None:
        return {"max_tokens": max_tokens}
    if effort is not None:
        return {"effort": effort}
    return {"enabled": True}


class EffectiveIdentityPolicy(_StrictProfileModel):
    # Every supported profile is homogeneous only when the provider attests its
    # effective model.  This literal prevents a custom profile from weakening
    # the receipt boundary.
    require_response_model: Literal[True] = True
    accepted_model_ids: Annotated[
        list[NonEmptyString],
        Field(min_length=1, max_length=MAX_EFFECTIVE_MODEL_IDS),
    ]

    @field_validator("accepted_model_ids")
    @classmethod
    def accepted_ids_are_canonical(cls, value: list[str]) -> list[str]:
        if any(_has_surrounding_whitespace(item) for item in value):
            raise ValueError(
                "accepted effective model IDs cannot contain surrounding whitespace"
            )
        if len(value) != len(set(value)):
            raise ValueError("accepted effective model IDs must be unique")
        # Treat this field as a bounded set and serialize it canonically.
        return sorted(value)


class ExecutionModelProfile(_StrictProfileModel):
    """One immutable provider/model/reasoning identity for a complete run."""

    schema_version: Literal["execution-model-profile-1.0"] = (
        EXECUTION_MODEL_PROFILE_SCHEMA_VERSION
    )
    profile_id: ProfileId
    profile_kind: Literal["real", "offline_test"] = "real"
    canonical_model_id: NonEmptyString
    provider: ProviderBinding
    adapter_model_ids: AdapterModelIds
    generation: GenerationProfile = Field(default_factory=GenerationProfile)
    reasoning: ReasoningProfile
    effective_identity: EffectiveIdentityPolicy

    @field_validator("canonical_model_id")
    @classmethod
    def canonical_model_id_has_no_padding(cls, value: str) -> str:
        if _has_surrounding_whitespace(value):
            raise ValueError(
                "canonical model ID cannot contain surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def profile_kind_matches_provider_and_reasoning(self) -> ExecutionModelProfile:
        offline_provider = self.provider.protocol == "offline_mock"
        offline_wire = self.reasoning.wire_policy == "offline_mock"
        if self.profile_kind == "offline_test":
            if not offline_provider or not offline_wire:
                raise ValueError(
                    "offline profile requires offline provider and reasoning wire"
                )
            if any(
                model_id != self.canonical_model_id
                for model_id in (
                    self.adapter_model_ids.core_openai_chat,
                    self.adapter_model_ids.mini_swe_litellm,
                    self.adapter_model_ids.downstream_codex,
                )
            ):
                raise ValueError(
                    "offline profile adapter IDs must equal its canonical model ID"
                )
        elif offline_provider or offline_wire:
            raise ValueError("real profile cannot use an offline provider or wire")
        elif (
            self.reasoning.wire_policy in {
                "deepseek_thinking_type", "qwen_enable_thinking",
                "openrouter_reasoning", "openai_chat_reasoning_effort",
            }
            and self.provider.protocol != "openai_chat_completions"
        ):
            raise ValueError(
                "Selected thinking wire requires Chat Completions transport"
            )
        elif (
            self.reasoning.wire_policy == "openai_responses_reasoning_effort"
            and self.provider.protocol != "openai_responses"
        ):
            raise ValueError(
                "OpenAI reasoning-effort wire requires Responses transport"
            )
        if self.canonical_model_id not in self.effective_identity.accepted_model_ids:
            raise ValueError(
                "canonical model ID must be an accepted effective model ID"
            )
        return self

    def method_payload(self) -> dict[str, object]:
        """Return every identity-bearing field except the credential variable name."""

        payload = self.model_dump(mode="json")
        provider = dict(payload["provider"])
        # This is the sole non-method field: credential values are never in a
        # profile, and merely renaming their environment slot cannot split arms.
        provider.pop("api_key_env", None)
        payload["provider"] = provider
        return payload

    @property
    def profile_digest(self) -> str:
        return hashlib.sha256(
            _canonical_compact(self.method_payload()).encode("utf-8")
        ).hexdigest()


def _canonical_compact(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_profile_text(profile: ExecutionModelProfile) -> str:
    return json.dumps(
        profile.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate and unhashable mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    if not isinstance(node, MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            "expected a mapping node",
            node.start_mark,
        )
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not permitted in model profiles",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "unhashable mapping key is not permitted in a model profile",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key is not permitted: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_yaml_event_bounds(text: str) -> None:
    depth = 0
    event_count = 0
    try:
        events = yaml.parse(text, Loader=yaml.SafeLoader)
        for event in events:
            event_count += 1
            if event_count > MAX_EXECUTION_MODEL_PROFILE_NODES:
                raise ValueError("model profile exceeds its YAML node limit")
            if isinstance(event, AliasEvent) or getattr(event, "anchor", None):
                raise ValueError("YAML aliases and anchors are not permitted")
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > MAX_EXECUTION_MODEL_PROFILE_DEPTH:
                    raise ValueError("model profile exceeds its YAML depth limit")
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
    except yaml.YAMLError as error:
        raise ValueError("model profile is not valid bounded YAML") from error


def _read_regular_bounded_bytes(
    path: Path,
    *,
    limit: int = MAX_EXECUTION_MODEL_PROFILE_BYTES,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("model profile path is not a regular file")
        if info.st_size > limit:
            raise ValueError("model profile exceeds its byte limit")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            value = handle.read(limit + 1)
        if len(value) > limit:
            raise ValueError("model profile exceeds its byte limit")
        return value
    except OSError as error:
        if isinstance(error, FileNotFoundError):
            raise
        if error.errno == errno.ELOOP:
            raise ValueError("model profile path is not a regular file") from error
        raise ValueError("model profile cannot be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse_profile_bytes(value: bytes) -> ExecutionModelProfile:
    if len(value) > MAX_EXECUTION_MODEL_PROFILE_BYTES:
        raise ValueError("model profile exceeds its byte limit")
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("model profile is not valid UTF-8") from error
    _validate_yaml_event_bounds(text)
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("model profile is not valid bounded YAML") from error
    if not isinstance(raw, dict):
        raise ValueError("model profile root must be a mapping")
    return ExecutionModelProfile.model_validate(raw)


def _builtin_profile_resources() -> dict[str, object]:
    root = resources.files("repo_constraint_memory_v3.model_profile_data")
    items = sorted(
        (
            item
            for item in root.iterdir()
            if item.name.endswith(".yaml") and item.is_file()
        ),
        key=lambda item: item.name,
    )
    if len(items) > MAX_BUILTIN_EXECUTION_MODEL_PROFILES:
        raise ValueError("too many built-in execution-model profiles")
    builtins: dict[str, object] = {}
    for item in items:
        if not _BUILTIN_FILENAME_PATTERN.fullmatch(item.name):
            raise ValueError("invalid built-in execution-model profile filename")
        profile_id = item.name.removesuffix(".yaml")
        if profile_id in builtins:
            raise ValueError("duplicate built-in execution-model profile ID")
        builtins[profile_id] = item
    return builtins


def builtin_profile_ids(*, include_offline: bool = False) -> tuple[str, ...]:
    values = sorted(_builtin_profile_resources())
    if not include_offline:
        values = [item for item in values if item != OFFLINE_EXECUTION_MODEL_PROFILE_ID]
    return tuple(values)


def load_execution_model_profile(
    selector: str | Path | None = None,
    *,
    allow_offline: bool = False,
) -> ExecutionModelProfile:
    """Load a built-in profile ID or one explicitly supplied bounded YAML file."""

    selected: str | Path = (
        DEFAULT_EXECUTION_MODEL_PROFILE_ID if selector is None else selector
    )
    selected_builtin_id: str | None = None
    if isinstance(selected, Path):
        profile = _parse_profile_bytes(_read_regular_bounded_bytes(selected))
    else:
        builtins = _builtin_profile_resources()
        if selected in builtins:
            selected_builtin_id = selected
            resource = builtins[selected]
            profile = _parse_profile_bytes(resource.read_bytes())  # type: ignore[attr-defined]
        else:
            try:
                profile_bytes = _read_regular_bounded_bytes(Path(selected))
            except FileNotFoundError as error:
                raise ValueError(
                    f"unknown execution-model profile: {selected}"
                ) from error
            profile = _parse_profile_bytes(profile_bytes)
    if selected_builtin_id is not None and profile.profile_id != selected_builtin_id:
        raise ValueError("built-in model profile ID differs from its filename")
    if profile.profile_kind == "offline_test" and not allow_offline:
        raise ValueError("offline execution-model profile is test-only")
    return profile


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"saved model profile contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"saved model profile contains non-finite number: {value}")


def load_profile_artifact(path: Path) -> ExecutionModelProfile:
    """Load a saved canonical JSON profile through the same bounded path policy."""

    value = _read_regular_bounded_bytes(path)
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("saved model profile is not valid UTF-8 JSON") from error
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("saved model profile is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("saved model profile root must be an object")
    profile = ExecutionModelProfile.model_validate(raw)
    if text != stable_profile_text(profile):
        raise ValueError("saved model profile is not canonical JSON")
    return profile
