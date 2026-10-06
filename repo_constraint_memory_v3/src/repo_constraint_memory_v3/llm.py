"""Bounded OpenAI-compatible transport and deterministic offline mock."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from repo_constraint_memory_v3.discovery import matches_glob
from repo_constraint_memory_v3.models import LLMSettings, ProviderResponse
from repo_constraint_memory_v3.model_profiles import (
    ExecutionModelProfile,
    openrouter_reasoning_payload,
)


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        """Return one JSON response without tools or side effects."""


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        latency_ms: int = 0,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.latency_ms = max(latency_ms, 0)
        self.retry_after_seconds = retry_after_seconds


class _EffectiveModelViolation(ValueError):
    """Internal signal for a provider response outside the frozen profile."""


@dataclass
class OpenAICompatibleClient:
    """One-attempt OpenAI-compatible client with a bounded connection pool."""

    settings: LLMSettings
    api_key: str
    model_profile: ExecutionModelProfile | None = None
    _client: httpx.Client = field(init=False, repr=False)
    _telemetry_lock: threading.Lock = field(
        init=False,
        repr=False,
        default_factory=threading.Lock,
    )
    _provider_request_count: int = field(init=False, repr=False, default=0)
    _provider_response_count: int = field(init=False, repr=False, default=0)
    _cached_input_response_count: int = field(init=False, repr=False, default=0)
    _cached_input_tokens: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._validate_profile_settings(self.settings)
        self._client = httpx.Client(
            timeout=self.settings.timeout_seconds,
            limits=httpx.Limits(
                max_connections=self.settings.max_concurrency,
                max_keepalive_connections=self.settings.max_concurrency,
            ),
        )

    def _validate_profile_settings(self, settings: LLMSettings) -> None:
        if self.model_profile is None:
            return
        profile = self.model_profile
        expected = {
            "model": profile.adapter_model_ids.core_openai_chat,
            "require_response_model": (
                profile.effective_identity.require_response_model
            ),
            "accepted_effective_model_ids": (
                profile.effective_identity.accepted_model_ids
            ),
            "base_url": profile.provider.base_url,
            "api_key_env": profile.provider.api_key_env,
            "temperature": profile.generation.temperature,
            "thinking_mode": profile.reasoning.mode,
            "reasoning_effort": profile.reasoning.effort,
            "reasoning_max_tokens": profile.reasoning.max_tokens,
            "reasoning_wire_policy": profile.reasoning.wire_policy,
        }
        actual = {
            "model": settings.model,
            "require_response_model": settings.require_response_model,
            "accepted_effective_model_ids": (
                settings.accepted_effective_model_ids
            ),
            "base_url": settings.base_url,
            "api_key_env": settings.api_key_env,
            "temperature": settings.temperature,
            "thinking_mode": settings.thinking_mode,
            "reasoning_effort": settings.reasoning_effort,
            "reasoning_max_tokens": settings.reasoning_max_tokens,
            "reasoning_wire_policy": settings.reasoning_wire_policy,
        }
        if actual != expected:
            raise ValueError(
                "resolved LLM settings differ from execution-model profile"
            )

    def close(self) -> None:
        self._client.close()

    def _record_provider_request(self) -> None:
        with self._telemetry_lock:
            self._provider_request_count += 1

    def _record_provider_response(self, cached_input_tokens: int | None) -> None:
        with self._telemetry_lock:
            self._provider_response_count += 1
            if cached_input_tokens is not None:
                self._cached_input_response_count += 1
                self._cached_input_tokens += cached_input_tokens

    def cached_input_telemetry(self) -> dict[str, object]:
        """Return prompt-cache totals without fabricating missing provider fields."""

        with self._telemetry_lock:
            requests = self._provider_request_count
            responses = self._provider_response_count
            reported = self._cached_input_response_count
            tokens = self._cached_input_tokens
        if reported == 0 and requests > 0:
            status = "unavailable"
            total: int | None = None
        elif requests == responses == reported:
            status = "complete"
            total = tokens if reported else None
        else:
            status = "partial"
            total = tokens
        return {
            "source": "core_openai_chat",
            "status": status,
            "tokens": total,
            "provider_request_count": requests,
            "provider_response_count": responses,
            "responses_with_value": reported,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        return self.complete_with_settings(
            system_prompt,
            user_prompt,
            settings=self.settings,
        )

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: LLMSettings,
    ) -> ProviderResponse:
        """Resolve stage-specific token/time bounds without changing credentials."""

        self._validate_profile_settings(settings)
        if settings.reasoning_wire_policy not in {
            "deepseek_thinking_type", "qwen_enable_thinking", "openrouter_reasoning"
        }:
            raise ValueError(
                "OpenAI chat client does not implement the selected reasoning wire"
            )

        started = time.monotonic()
        self._record_provider_request()

        def elapsed_ms() -> int:
            return max(round((time.monotonic() - started) * 1000), 0)

        request_payload: dict[str, object] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if settings.reasoning_wire_policy == "openrouter_reasoning":
            request_payload["reasoning"] = openrouter_reasoning_payload(
                mode=settings.thinking_mode,
                effort=settings.reasoning_effort,
                max_tokens=settings.reasoning_max_tokens,
            )
        elif settings.reasoning_wire_policy == "qwen_enable_thinking":
            request_payload["enable_thinking"] = settings.thinking_mode == "enabled"
            request_payload["preserve_thinking"] = True
            if settings.thinking_mode == "enabled":
                request_payload["reasoning_effort"] = settings.reasoning_effort
        elif settings.thinking_mode == "disabled":
            request_payload["thinking"] = {"type": "disabled"}
        else:
            request_payload["thinking"] = {"type": "enabled"}
            request_payload["reasoning_effort"] = settings.reasoning_effort

        response_body = bytearray()
        response_body_complete = False
        try:
            with self._client.stream(
                "POST",
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                timeout=settings.timeout_seconds,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    # A decoded iterator can inflate one compressed chunk
                    # before the host gets a chance to enforce its byte cap.
                    # Require an unencoded response and count raw wire bytes.
                    "Accept-Encoding": "identity",
                },
                json=request_payload,
            ) as response:
                response.raise_for_status()
                content_encoding = response.headers.get(
                    "Content-Encoding",
                    "",
                ).strip().casefold()
                if content_encoding not in {"", "identity"}:
                    raise LLMProviderError(
                        "LLM provider returned unsupported Content-Encoding",
                        retryable=True,
                        latency_ms=elapsed_ms(),
                    )
                for chunk in response.iter_raw(chunk_size=64 * 1024):
                    if (
                        len(response_body) + len(chunk)
                        > settings.max_provider_response_bytes
                    ):
                        raise LLMProviderError(
                            "LLM provider response exceeded "
                            "max_provider_response_bytes",
                            retryable=True,
                            latency_ms=elapsed_ms(),
                        )
                    response_body.extend(chunk)
                response_body_complete = True
        except httpx.TimeoutException as error:
            raise LLMProviderError(
                "LLM request timed out",
                retryable=True,
                latency_ms=elapsed_ms(),
            ) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            retry_after = error.response.headers.get("Retry-After")
            try:
                retry_after_seconds = (
                    max(float(retry_after), 0.0)
                    if retry_after is not None
                    else None
                )
            except ValueError:
                retry_after_seconds = None
            raise LLMProviderError(
                f"LLM provider returned HTTP {status}",
                retryable=status == 429 or status >= 500,
                latency_ms=elapsed_ms(),
                retry_after_seconds=retry_after_seconds,
            ) from error
        except httpx.HTTPError as error:
            raise LLMProviderError(
                f"LLM transport failed: {type(error).__name__}",
                retryable=True,
                latency_ms=elapsed_ms(),
            ) from error
        telemetry_recorded = False
        try:
            payload = json.loads(response_body)
            choice = payload["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("choice is not an object")
            finish_reason_value = choice.get("finish_reason")
            finish_reason = (
                finish_reason_value
                if isinstance(finish_reason_value, str)
                else None
            )
            normalized_finish_reason = (finish_reason or "").casefold().replace(
                "-", "_"
            )
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            content_value = message["content"]
            if isinstance(content_value, str):
                content = content_value
            elif content_value is None and normalized_finish_reason in {
                "length",
                "max_tokens",
            }:
                # Some OpenAI-compatible gateways omit content entirely on a
                # length stop. Preserve the typed finish reason so the common
                # output-limit retry axis, not provider/schema retry, owns it.
                content = ""
            elif isinstance(content_value, list):
                parts = [
                    item["text"]
                    for item in content_value
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ]
                if len(parts) != len(content_value):
                    raise TypeError("message content parts are not all text")
                content = "".join(parts)
            else:
                raise TypeError("message content is not text")
            usage = payload.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")
            prompt_details = usage.get("prompt_tokens_details")
            raw_cached_input_tokens = (
                prompt_details.get("cached_tokens")
                if isinstance(prompt_details, dict)
                else None
            )
            cached_input_tokens = (
                raw_cached_input_tokens
                if isinstance(raw_cached_input_tokens, int)
                and not isinstance(raw_cached_input_tokens, bool)
                and raw_cached_input_tokens >= 0
                else None
            )
            self._record_provider_response(cached_input_tokens)
            telemetry_recorded = True
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            completion_details = usage.get("completion_tokens_details")
            raw_reasoning_tokens = (
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else None
            )
            reasoning_tokens = (
                raw_reasoning_tokens
                if isinstance(raw_reasoning_tokens, int)
                and not isinstance(raw_reasoning_tokens, bool)
                and raw_reasoning_tokens >= 0
                else None
            )
            reasoning_content_value = message.get("reasoning_content")
            if reasoning_content_value is None:
                reasoning_content_present = False
                reasoning_content_bytes = 0
            elif isinstance(reasoning_content_value, str):
                reasoning_content_present = True
                reasoning_content_bytes = len(
                    reasoning_content_value.encode("utf-8")
                )
            else:
                raise TypeError("reasoning content is not text")
            if settings.reasoning_wire_policy == "openrouter_reasoning":
                for field in ("reasoning", "reasoning_details"):
                    value = message.get(field)
                    if value:
                        reasoning_content_present = True
                        reasoning_content_bytes += len(
                            json.dumps(value, ensure_ascii=False).encode("utf-8")
                        )
            raw_model = payload.get("model")
            if self.model_profile is not None:
                identity = self.model_profile.effective_identity
                if not isinstance(raw_model, str) or not raw_model:
                    if identity.require_response_model:
                        raise _EffectiveModelViolation(
                            "model_profile_violation:effective_model_missing"
                        )
                    model = settings.model
                elif raw_model not in identity.accepted_model_ids:
                    raise _EffectiveModelViolation(
                        "model_profile_violation:effective_model_mismatch"
                    )
                else:
                    model = raw_model
            else:
                model = raw_model or settings.model
                if not isinstance(model, str):
                    model = settings.model
        except _EffectiveModelViolation as error:
            raise LLMProviderError(
                str(error),
                retryable=False,
                latency_ms=elapsed_ms(),
            ) from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            if response_body_complete and not telemetry_recorded:
                self._record_provider_response(None)
            raise LLMProviderError(
                "LLM provider response has an invalid shape",
                retryable=True,
                latency_ms=elapsed_ms(),
            ) from error
        return ProviderResponse(
            content=content,
            model=model,
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            reasoning_tokens=reasoning_tokens,
            response_content_bytes=len(content.encode("utf-8")),
            reasoning_content_present=reasoning_content_present,
            reasoning_content_bytes=reasoning_content_bytes,
            latency_ms=elapsed_ms(),
            finish_reason=finish_reason,
        )


_PROSE_MODAL_RE = re.compile(
    r"\b(must|required|shall|should|recommended|recommend|prefer|preferred|"
    r"do not|don't|never)\b",
    re.IGNORECASE,
)
_SOFT_MODAL_RE = re.compile(
    r"\b(should|recommended|recommend|prefer|preferred)\b",
    re.IGNORECASE,
)
_SUBJECT_RE = re.compile(
    r"\b(contributors?|developers?|maintainers?|reviewers?|changes?|code|"
    r"tests?|files?|modules?|packages?|pull requests?|commits?|documentation|"
    r"additions?|contents?|news fragments?|public api)\b",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"^\s*[\"']?([A-Za-z0-9_.-]+)[\"']?\s*[:=]\s*(.*?)\s*$"
)
_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.%/@+,-][^:=#]*):(?!=)")
_WORKFLOW_RE = re.compile(r"^\s*(?:-\s*)?(run|uses):\s*(.*?)\s*$")
_WORKFLOW_PREDICATE_RE = re.compile(r"^(?P<indent>\s*)(?:on|if):\s*(.*?)\s*$")
_CONFIG_KEYS = {
    "addopts",
    "extend-ignore",
    "extend-select",
    "ignore",
    "line-length",
    "markers",
    "requires-python",
    "select",
    "target-version",
    "testpaths",
}


class MockLLMClient:
    """Deterministic schema-4.11 stage-aware smoke mock; never network-backed."""

    model = "deterministic-fake"

    def cached_input_telemetry(self) -> dict[str, object]:
        return {
            "source": "offline_mock",
            "status": "not_applicable",
            "tokens": None,
            "provider_request_count": 0,
            "provider_response_count": 0,
            "responses_with_value": 0,
        }

    @staticmethod
    def _range(
        unit: dict[str, object],
        start: int,
        end: int | None = None,
    ) -> dict[str, object]:
        spans = unit["spans"]
        assert isinstance(spans, list)
        last = start if end is None else end
        return {
            "unit_id": unit["unit_id"],
            "span_start_id": spans[start]["span_id"],
            "span_end_id": spans[last]["span_id"],
        }

    @staticmethod
    def _condition(text: str) -> str | None:
        stripped = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", text).strip()
        leading = re.match(
            r"^((?:When|Before|After|Unless)\b[^,:;.]{1,160})[,;:]",
            stripped,
            re.IGNORECASE,
        )
        if leading:
            return leading.group(1).strip()
        trailing = re.search(
            r"\b(before|after|when|unless)\b([^.;]{1,140})[.;]?$",
            stripped,
            re.IGNORECASE,
        )
        if trailing:
            return (
                f"{trailing.group(1).title()} {trailing.group(2).strip()}"
            ).strip()
        return None

    def _prose_claims(
        self,
        unit: dict[str, object],
    ) -> list[dict[str, object]]:
        spans = unit["spans"]
        assert isinstance(spans, list)
        claims: list[dict[str, object]] = []
        consumed: set[int] = set()
        for index, span in enumerate(spans):
            if index in consumed:
                continue
            raw = str(span["text"])
            if not _PROSE_MODAL_RE.search(raw) or not _SUBJECT_RE.search(raw):
                continue
            end = index
            combined = raw.rstrip("\r\n")
            while (
                end + 1 < len(spans)
                and end - index < 2
                and not re.search(r"[.!?;:]\s*$", combined)
                and str(spans[end + 1]["text"]).strip()
                and not str(spans[end + 1]["text"]).lstrip().startswith("#")
            ):
                end += 1
                consumed.add(end)
                combined += "\n" + str(spans[end]["text"]).rstrip("\r\n")
            rule = " ".join(
                re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line).strip()
                for line in combined.splitlines()
            ).strip()
            claims.append(
                {
                    "rule": rule,
                    "claim_kind": "normative_declaration",
                    "normative_strength": (
                        "recommended"
                        if _SOFT_MODAL_RE.search(combined)
                        else "required"
                    ),
                    "applies_to": list(unit["default_scope"]),
                    "condition": self._condition(rule),
                    "evidence": [self._range(unit, index, end)],
                }
            )
        return claims

    def _configuration_claims(
        self,
        unit: dict[str, object],
    ) -> list[dict[str, object]]:
        spans = unit["spans"]
        assert isinstance(spans, list)
        path = str(unit["path"])
        claims: list[dict[str, object]] = []
        for index, span in enumerate(spans):
            line = str(span["text"]).rstrip("\r\n")
            match = _ASSIGNMENT_RE.match(line)
            if (
                match is None
                or line.lstrip().startswith(("#", ";"))
                or match.group(1).casefold() not in _CONFIG_KEYS
            ):
                continue
            key = match.group(1).casefold()
            end = index
            if key == "markers" and "[" in line and "]" not in line:
                while end + 1 < len(spans) and end - index < 11:
                    end += 1
                    if "]" in str(spans[end]["text"]):
                        break
            if key == "markers":
                rule = "The repository registers pytest marker names in its configuration."
                scope = ["tests/**"]
                condition = "When inspecting pytest marker configuration."
                kind = "operational_fact"
            elif key in {
                "line-length",
                "target-version",
                "select",
                "ignore",
                "extend-select",
                "extend-ignore",
            } and ("ruff" in path.casefold() or path.endswith("pyproject.toml")):
                rule = f"The repository tool configuration sets `{line.strip()}`."
                scope = ["**/*.py"]
                condition = "When Ruff processes Python files."
                kind = "enforced_configuration"
            else:
                rule = f"The repository configuration contains `{line.strip()}`."
                scope = list(unit["default_scope"])
                condition = "When using the corresponding configured tool."
                kind = "operational_fact"
            claims.append(
                {
                    "rule": rule,
                    "claim_kind": kind,
                    "normative_strength": "observed",
                    "applies_to": scope,
                    "condition": condition,
                    "evidence": [self._range(unit, index, end)],
                }
            )
        return claims

    def _command_claims(
        self,
        unit: dict[str, object],
    ) -> list[dict[str, object]]:
        spans = unit["spans"]
        assert isinstance(spans, list)
        claims: list[dict[str, object]] = []
        target: tuple[str, int] | None = None
        for index, span in enumerate(spans):
            line = str(span["text"]).rstrip("\r\n")
            match = _MAKE_TARGET_RE.match(line)
            if match:
                target = (match.group(1).strip(), index)
                continue
            if target is None or not line.startswith(("\t", "    ")):
                continue
            command = line.strip().lstrip("@")
            if not command or command.startswith("#"):
                continue
            name, start = target
            claims.append(
                {
                    "rule": f"The `{name}` target runs `{command}`.",
                    "claim_kind": "command_recipe",
                    "normative_strength": "observed",
                    "applies_to": ["**"],
                    "condition": f"When invoking the `{name}` command target.",
                    "evidence": [self._range(unit, start, index)],
                }
            )
        return claims

    def _workflow_claims(
        self,
        unit: dict[str, object],
    ) -> list[dict[str, object]]:
        spans = unit["spans"]
        assert isinstance(spans, list)
        claims: list[dict[str, object]] = []
        predicate_range: dict[str, object] | None = None
        for index, span in enumerate(spans):
            line = str(span["text"]).rstrip("\r\n")
            predicate = _WORKFLOW_PREDICATE_RE.match(line)
            if predicate is None:
                continue
            end = index
            if not predicate.group(2).strip():
                base_indent = len(predicate.group("indent").expandtabs(4))
                for following_index, following_span in enumerate(
                    spans[index + 1 :],
                    start=index + 1,
                ):
                    following = str(following_span["text"]).rstrip("\r\n")
                    if not following.strip():
                        continue
                    expanded = following.expandtabs(4)
                    indent = len(expanded) - len(expanded.lstrip())
                    if indent <= base_indent:
                        break
                    end = following_index
                    if end - index + 1 >= 12:
                        break
            predicate_range = self._range(unit, index, end)
            break
        for index, span in enumerate(spans):
            line = str(span["text"]).rstrip("\r\n")
            match = _WORKFLOW_RE.match(line)
            if match is None or match.group(2) in {"", "|", ">"}:
                continue
            claims.append(
                {
                    "rule": (
                        f"The workflow {match.group(1)} step uses "
                        f"`{match.group(2).strip()}`."
                    ),
                    "claim_kind": "workflow_fact",
                    "normative_strength": "observed",
                    "applies_to": ["**"],
                    "condition": "When the cited workflow step runs.",
                    "evidence": [
                        *([predicate_range] if predicate_range is not None else []),
                        self._range(unit, index),
                    ],
                }
            )
        return claims

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt
        payload = json.loads(user_prompt)
        if payload.get("repair_type") == "local_draft_replacement":
            return self._response({
                "replacements": [
                    {"ordinal": item["ordinal"], "draft": item["draft"]}
                    for item in payload["replacements_required"]
                ],
            })
        request = payload.get("original_request", payload)
        stage = request.get("stage")
        if stage == "explicit_semantic_verify":
            return self._response(
                {
                    "results": [
                        {
                            "candidate_root_id": item["candidate_root_id"],
                            "coding_relevance": "pass",
                            "evidence_entailment": "pass",
                            "scope_condition_support": "pass",
                            "ambiguity": "none",
                            "git_need": {"status": "no_established_need", "questions": []},
                        }
                        for item in request["candidates"]
                    ]
                }
            )
        if stage == "current_semantic_map":
            results = []
            for packet in request["packets"]:
                allowed = list(packet["allowed_evidence_refs"])
                results.append(
                    {
                        "packet_id": packet["packet_id"],
                        "outcome": "observations",
                        "observations": [
                            {
                                "category": "mock_repository_pattern",
                                "summary": (
                                    "This packet participates in a repository-local "
                                    "implementation pattern requiring population review."
                                ),
                                "contributor_relevance": "actionable",
                                "evidence_refs": [allowed[0]],
                                "related_unit_ids": [
                                    packet["primary_unit_id"]
                                ],
                                "uncertainty": None,
                            }
                        ],
                        "reason_code": "mock_observation",
                    }
                )
            return self._response({"results": results, "empty_results": []})
        if stage == "current_hypothesis_reduce":
            witness = request["accounted_input_ids"][0]
            hypotheses = [
                {
                    "rule": (
                        "Contributors should preserve the repository-local "
                        "implementation pattern represented by the supplied population."
                    ),
                    "category": "mock_repository_pattern",
                    "proposed_claim_kind": "observed_convention",
                    "proposed_strength": "observed",
                    "applies_to": ["**/*.py"],
                    "condition": (
                        "When modifying members of the represented Python population."
                    ),
                    "witness_refs": [witness],
                    "explicit_context_refs": [],
                    "superpopulation_spec": {
                        "op": "universe",
                        "universe": "entity",
                    },
                    "predicate_mode": "semantic",
                    "structural_predicate": None,
                    "semantic_predicate": {
                        "eligibility_question": (
                            "Is this member relevant to the supplied repository-local pattern?"
                        ),
                        "support_question": (
                            "Does this member preserve the supplied repository-local pattern?"
                        ),
                        "allowed_reason_codes": [
                            "mock_support",
                            "mock_counterexample",
                            "mock_ineligible",
                            "mock_unknown",
                        ],
                        "expected_evidence_form": (
                            "A supplied observation fact or source identity."
                        ),
                        "ambiguity_instruction": (
                            "Return unknown whenever supplied context is insufficient."
                        ),
                        "repository_specificity_test": (
                            "Judge only the supplied repository pattern, not generic Python."
                        ),
                        "protocol_version": (
                            "semantic-member-classification-1.0"
                        ),
                    },
                    "independence_spec": {
                        "mode": "clone_aware_observation"
                    },
                    "limitations": ["mock_semantic_quality_not_established"],
                }
            ]
            outcome = "hypotheses" if hypotheses else "no_hypothesis"
            return self._response(
                {
                    "mode": request["mode"],
                    "accounted_input_ids": request["accounted_input_ids"],
                    "outcome": outcome,
                    "hypotheses": hypotheses,
                    "reason_code": (
                        "mock_hypothesis"
                        if hypotheses
                        else "mock_no_hypothesis"
                    ),
                }
            )
        if stage == "population_semantic_classify":
            labels = ("support", "counterexample", "ineligible", "unknown")
            results = []
            reason_aliases = {
                value: key for key, value in request["reason_catalog"].items()
            }
            for index, item in enumerate(request["members"]):
                label = labels[index % len(labels)]
                packet = item["packet"]
                path = packet["observation_ref"].get("path")
                patterns = request["roots"][item["root_ref"]]["hypothesis"]["applies_to"]
                if (
                    path is None
                    or not any(
                        matches_glob(path, pattern)
                        for pattern in patterns
                    )
                ):
                    label = "ineligible"
                results.append(
                    {
                        "member_ordinal": item["member_ordinal"],
                        "label": label,
                        "evidence_context": {
                            "support": "production",
                            "counterexample": "production",
                            "ineligible": "irrelevant",
                            "unknown": "unknown",
                        }[label],
                        "reason_code": reason_aliases[f"mock_{label}"],
                        "reason_summary": (
                            f"Deterministic smoke classification: {label}."
                        ),
                        "evidence_refs": [packet["evidence_refs"][0]],
                    }
                )
            return self._response({"results": results})
        if stage == "counterexample_verify":
            allowed_refs = request["allowed_evidence_refs"]
            return self._response(
                {
                    "reviewed_member_ids": request["reviewed_member_ids"],
                    "verdict": "supported_as_written",
                    "evidence_refs": [allowed_refs[0]],
                    "issue_interpretations": [
                        {
                            "member_id": member_id,
                            "interpretation": "not_counterexample",
                            "rationale_summary": (
                                "The deterministic mock found that the supplied "
                                "issue does not refute or qualify the rule."
                            ),
                        }
                        for member_id in request["reviewed_member_ids"]
                    ],
                    "repository_specificity": "pass",
                    "contributor_value": "pass",
                    "scope_condition_fit": "pass",
                    "evidence_sufficiency": "pass",
                    "proposed_applies_to": None,
                    "proposed_condition": None,
                    "exception_summary": None,
                    "alternative_explanation": None,
                    "proposed_population_spec": None,
                    "unresolved_questions": [],
                    "git_need": {"status": "no_established_need", "questions": []},
                    "rationale_summary": (
                        "The deterministic smoke verifier found no material objection."
                    ),
                }
            )
        if stage == "candidate_semantic_group":
            by_id = {item["member_id"]: item for item in request["members"]}
            relations = []
            for left_id, right_id in request["required_pairs"]:
                left = by_id[left_id]["semantics"]
                right = by_id[right_id]["semantics"]
                if left == right:
                    relation = "equivalent"
                elif (
                    left["normalized_action"] == right["normalized_action"]
                    or set(left["normalized_action"].split())
                    & set(right["normalized_action"].split())
                ):
                    relation = "conflicting_variant"
                else:
                    relation = "distinct"
                relations.append(
                    {
                        "left_member_id": left_id,
                        "right_member_id": right_id,
                        "relation": relation,
                    }
                )
            return self._response({"relations": relations})
        if stage == "temporal_evidence_classify":
            return self._response(
                {
                    "relations": [
                        {
                            "item_id": item_id,
                            "member_id": member_id,
                            "relation": "context_only",
                            "affected_facets": [],
                            "evidence_span": None,
                            "reason_code": "mock_temporal_context_only",
                            "uncertainty_reason": None,
                        }
                        for item_id, member_id in request["required_cells"]
                    ]
                }
            )
        if stage == "unified_case_verify":
            current = request["current_evidence"]
            support_by_member: dict[str, list[str]] = {}
            for item in current["items"]:
                if item["role"] != "support":
                    continue
                for member_id in item["member_ids"]:
                    support_by_member.setdefault(member_id, []).append(
                        item["evidence_id"]
                    )
            temporal = request["temporal_evidence"]
            temporal_by_member: dict[str, list[str]] = {}
            for item in temporal.get("relations", []):
                temporal_by_member.setdefault(item["member_id"], []).append(
                    item["relation"]
                )
            member_findings = []
            by_member = {item["member_id"]: item for item in request["members"]}
            for member_id in request["required_member_ids"]:
                member = by_member[member_id]
                refs = sorted(set(support_by_member.get(member_id, [])))
                has_support = bool(refs)
                local_axes = member["local_axis_findings"]
                local_fail = any(value == "fail" for value in local_axes.values())
                temporal_values = temporal_by_member.get(member_id, [])
                adverse = any(
                    value
                    in {
                        "contradicts_as_written",
                        "qualifies_scope_or_condition",
                        "retires_or_replaces",
                        "uncertain",
                    }
                    for value in temporal_values
                )
                axis = "pass" if has_support and not local_fail else "uncertain"
                closable = [
                    reason
                    for reason in member["typed_local_reasons"]
                    if not any(
                        marker in reason
                        for marker in (
                            "technical",
                            "missing",
                            "incomplete",
                            "failed",
                            "invalid",
                        )
                    )
                ]
                member_findings.append(
                    {
                        "member_id": member_id,
                        "norm_relevance": axis,
                        "rule_support": axis,
                        "scope_support": axis,
                        "condition_support": axis,
                        "temporal_status": (
                            "adverse_unresolved"
                            if adverse
                            else "current"
                            if temporal_values
                            else "not_required"
                        ),
                        "evidence_sufficiency": axis,
                        "evidence_refs": refs,
                        "closed_local_reason_codes": sorted(set(closable)),
                        "reason_codes": [
                            "mock_unified_axes_passed"
                            if axis == "pass"
                            else "mock_unified_evidence_uncertain"
                        ],
                    }
                )
            member_relations = []
            for left_id, right_id in request["required_pairs"]:
                left = by_member[left_id]["semantics"]
                right = by_member[right_id]["semantics"]
                if left == right:
                    relation = "equivalent"
                elif (
                    left["normalized_action"] == right["normalized_action"]
                    and left["polarity"] != right["polarity"]
                ):
                    relation = "conflicting"
                elif left["normalized_action"] == right["normalized_action"]:
                    relation = "compatible"
                else:
                    relation = "distinct"
                refs = sorted(
                    set(
                        support_by_member.get(left_id, [])
                        + support_by_member.get(right_id, [])
                    )
                )
                member_relations.append(
                    {
                        "left_member_id": left_id,
                        "right_member_id": right_id,
                        "relation": relation,
                        "evidence_refs": refs,
                        "reason_code": f"mock_{relation}",
                    }
                )
            conclusions = [
                {
                    "base_member_ids": [member_id],
                    "relation": "distinct",
                    "evidence_refs": sorted(
                        set(support_by_member.get(member_id, []))
                    ),
                }
                for member_id in request["required_member_ids"]
                if support_by_member.get(member_id)
            ]
            return self._response(
                {
                    "member_findings": member_findings,
                    "member_relations": member_relations,
                    "unresolved_conflicts": [],
                    "conclusions": conclusions,
                    "refinement_proposal": None,
                }
            )
        if request.get("prompt_version") != (
            "explicit-coding-knowledge-extraction-3.20"
        ):
            raise LLMProviderError(
                "mock stage is not implemented in schema 4.11",
                retryable=False,
            )
        results: list[dict[str, object]] = []
        for unit in request["units"]:
            text = "".join(str(item["text"]) for item in unit["spans"])
            if "UNCERTAIN_SCOPE" in text:
                results.append(
                    {
                        "unit_id": unit["unit_id"],
                        "classification": "uncertain",
                        "reason_code": "scope_unclear",
                        "claims": [],
                    }
                )
                continue
            kind = str(unit["source_kind"])
            if kind in {"instruction", "readme", "documentation"}:
                claims = self._prose_claims(unit)
                reason = "direct_norm"
            elif kind == "configuration":
                claims = self._configuration_claims(unit)
                reason = "configuration_fact"
            elif kind == "command":
                claims = self._command_claims(unit)
                reason = "command_fact"
            elif kind == "workflow":
                claims = self._workflow_claims(unit)
                reason = "workflow_fact"
            else:
                claims = []
                reason = "configuration_fact"
            claims = [
                claim
                for claim in claims
                if len(str(claim.get("rule", ""))) <= 500
            ]
            if claims:
                results.append(
                    {
                        "unit_id": unit["unit_id"],
                        "classification": "contains_claims",
                        "reason_code": reason,
                        "claims": claims,
                    }
                )
            else:
                results.append(
                    {
                        "unit_id": unit["unit_id"],
                        "classification": "no_claims",
                        "reason_code": (
                            "product_behavior"
                            if kind in {"readme", "documentation"}
                            else "no_actionable_repository_knowledge"
                        ),
                        "claims": [],
                    }
                )
        return self._response({"results": results})

    def _response(self, payload: dict[str, object]) -> ProviderResponse:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return ProviderResponse(
            content=content,
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            latency_ms=0,
        )
