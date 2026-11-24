from __future__ import annotations

import hashlib
import json
import threading

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.explicit_verification import (
    EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
    MAX_CANDIDATES_PER_BATCH,
    MAX_CANDIDATE_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    ExplicitVerificationCoverage,
    ExplicitVerificationJudgment,
    ExplicitVerificationLimits,
    ExplicitVerificationPlan,
    ExplicitVerificationResponse,
    SYSTEM_PROMPT,
    _repair_prompt,
    apply_explicit_verification_policy,
    build_explicit_verification_prompt,
    finalize_explicit_verification,
    make_grounded_candidate_input,
    plan_explicit_verification,
    run_explicit_verification,
    stable_verification_request_ordinal,
    validate_explicit_verification_response,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceInputRef,
    request_identity,
)
from repo_constraint_memory_v3.inference import attach_response_json_schema
from repo_constraint_memory_v3.models import (
    CandidateClaim,
    Evidence,
    LLMSettings,
    ProviderResponse,
)
from repo_constraint_memory_v3.source_adapters import GroundedEvidence


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate(
    label: str,
    *,
    quote: str | None = None,
    discovery_origin: str = "deterministic",
    path: str = "AGENTS.md",
):
    quote = quote or f"Contributors must run {label}.\n"
    unit_id = f"UNIT-{_digest('unit:' + path + ':' + label)[:24]}"
    span_id = f"SPAN-{_digest('span:' + path + ':' + label)[:24]}"
    evidence = Evidence(
        type="direct_declaration",
        adapter="prose",
        semantic_key=f"instruction:{label}",
        path=path,
        span_ids=[span_id],
        line_start=1,
        line_end=1,
        quote=quote,
        quote_sha256=_digest(quote),
        source_content_sha256=_digest(quote),
        source_kind="instruction",
        discovery_origin=discovery_origin,
    )
    grounded = GroundedEvidence(
        evidence=evidence,
        semantic_flags=frozenset({"instruction_section:testing", "prose"}),
        contributor_context=True,
        hard_modal=True,
    )
    candidate = CandidateClaim(
        rule=f"Contributors must run {label}.",
        claim_kind="normative_declaration",
        normative_strength="required",
        applies_to=["**"],
        condition=None,
        evidence=[
            {
                "unit_id": unit_id,
                "span_start_id": span_id,
                "span_end_id": span_id,
            }
        ],
    )
    fingerprint = _digest(f"candidate:{path}:{label}")
    return make_grounded_candidate_input(
        unit_id=unit_id,
        candidate_fingerprint=fingerprint,
        candidate=candidate,
        grounded=[grounded],
        host_claim_kind="normative_declaration",
        host_normative_strength="required",
        applies_to=["**"],
        condition=None,
    )


def _judgment(
    candidate,
    *,
    coding_relevance: str = "pass",
    evidence_entailment: str = "pass",
    scope_condition_support: str = "pass",
    ambiguity: str = "none",
) -> ExplicitVerificationJudgment:
    return ExplicitVerificationJudgment(
        git_need={"status": "no_established_need", "questions": []},
        candidate_root_id=candidate.candidate_root_id,
        coding_relevance=coding_relevance,
        evidence_entailment=evidence_entailment,
        scope_condition_support=scope_condition_support,
        ambiguity=ambiguity,
    )


def test_model_output_is_root_id_four_axes_and_git_need() -> None:
    properties = ExplicitVerificationJudgment.model_json_schema()["properties"]
    assert set(properties) == {
        "candidate_root_id",
        "coding_relevance",
        "evidence_entailment",
        "scope_condition_support",
        "ambiguity",
        "git_need",
    }
    candidate = _candidate("tests")
    payload = _judgment(candidate).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExplicitVerificationJudgment.model_validate(
            {**payload, "rationale": "looks good"}
        )
    with pytest.raises(ValidationError):
        ExplicitVerificationJudgment.model_validate(
            {**payload, "coding_relevance": True}
        )


def test_verifier_requires_array_wrapper_for_single_judgment() -> None:
    judgment = _judgment(_candidate("wrapper")).model_dump(mode="json")
    response = ExplicitVerificationResponse.model_validate({"results": [judgment]})
    assert len(response.results) == 1
    for malformed in (
        judgment,
        {"results": judgment},
        {"results": [judgment], "candidate_root_id": judgment["candidate_root_id"]},
    ):
        with pytest.raises(ValidationError):
            ExplicitVerificationResponse.model_validate(malformed)


def test_verifier_prompt_separates_enforcement_from_inventory_and_root_scope() -> None:
    assert EXPLICIT_VERIFICATION_PROTOCOL_VERSION == (
        "explicit-semantic-verification-1.15"
    )
    assert "Fail only when the proposition" in SYSTEM_PROMPT
    assert "either relevance or irrelevance, use uncertain" in SYSTEM_PROMPT
    assert "Do not fail merely because" in SYSTEM_PROMPT
    assert "supported Python runtime" in SYSTEM_PROMPT
    assert "minimum version of pytest" in SYSTEM_PROMPT
    assert "dependency/group inventory" in SYSTEM_PROMPT
    assert "merely enumerating configured categories is inventory" in SYSTEM_PROMPT
    assert "cannot support a repository-root" in SYSTEM_PROMPT
    assert "Fail unsupported scope" in SYSTEM_PROMPT
    assert "Markdown presentation" in SYSTEM_PROMPT
    assert "placeholder paths" in SYSTEM_PROMPT
    assert "`**/*.py` or a narrower Python glob" in SYSTEM_PROMPT
    assert "test-only rule" in SYSTEM_PROMPT
    assert "CI Python version matrix" in SYSTEM_PROMPT
    assert "local/path dependency topology" in SYSTEM_PROMPT
    assert "circular wrappers" in SYSTEM_PROMPT
    assert "example_context" in SYSTEM_PROMPT
    assert "observed/descriptive" in SYSTEM_PROMPT
    assert "every implementation surface explicitly named" in SYSTEM_PROMPT
    assert "generated-file or synchronized-artifact rules" in SYSTEM_PROMPT
    assert "exact `run:`/`uses:` evidence may support an unconditional" in SYSTEM_PROMPT
    assert "copy its candidate_root_id byte-for-byte" in SYSTEM_PROMPT
    assert "governed-change test" in SYSTEM_PROMPT
    assert "material-qualifier test" in SYSTEM_PROMPT
    assert "positively establishes a conflicting in-scope case" in SYSTEM_PROMPT
    assert "do not penalize an absent condition" in SYSTEM_PROMPT
    assert "independently usable branches" in SYSTEM_PROMPT
    assert "rule's core" in SYSTEM_PROMPT
    assert "Do not use this axis to judge" in SYSTEM_PROMPT
    assert "applies_to, condition, or exception" in SYSTEM_PROMPT
    assert "those belong only to" in SYSTEM_PROMPT
    assert "triggers and exceptions only under" in SYSTEM_PROMPT
    assert "Observed or descriptive strength does not by itself" in SYSTEM_PROMPT


def test_grounded_input_requires_exact_digests_and_canonical_fields() -> None:
    candidate = _candidate("pytest")
    assert candidate.evidence[0].evidence.quote == "Contributors must run pytest.\n"
    assert candidate.evidence[0].semantic_flags == [
        "instruction_section:testing",
        "prose",
    ]
    assert candidate.evidence[0].example_context is False

    tampered = candidate.model_dump(mode="json")
    tampered["rule"] = "Contributors must skip tests."
    with pytest.raises(ValidationError, match="candidate digest"):
        type(candidate).model_validate(tampered)

    tampered_evidence = candidate.evidence[0].model_dump(mode="json")
    tampered_evidence["evidence"]["quote"] = "different\n"
    with pytest.raises(ValidationError, match="quote digest"):
        type(candidate.evidence[0]).model_validate(tampered_evidence)


def test_planner_is_stable_bounded_and_covers_every_grounded_root() -> None:
    candidates = [_candidate(f"check-{index}") for index in range(17)]
    first = plan_explicit_verification(reversed(candidates))
    second = plan_explicit_verification(candidates)

    assert first == second
    assert first.plan_version == "explicit-semantic-verification-plan-1.3"
    assert first.planned_calls == 17
    assert [len(item.candidates) for item in first.batches] == [1] * 17
    assert first.planned_candidates == 17
    assert not first.omissions
    assert all(
        len(item.candidates) <= MAX_CANDIDATES_PER_BATCH
        and item.request_bytes <= MAX_REQUEST_BYTES
        and item.request_bytes
        == len(
            attach_response_json_schema(
                build_explicit_verification_prompt(item),
                ExplicitVerificationResponse,
            ).encode("utf-8")
        )
        for item in first.batches
    )
    assert all(
        len(candidate.model_dump_json().encode("utf-8")) <= MAX_CANDIDATE_BYTES
        for batch in first.batches
        for candidate in batch.candidates
    )
    assert '"stage":"explicit_semantic_verify"' in (
        build_explicit_verification_prompt(first.batches[0])
    )


def test_agent_candidates_do_not_perturb_deterministic_verifier_batches() -> None:
    deterministic = [_candidate(f"baseline-{index}") for index in range(10)]
    baseline = plan_explicit_verification(deterministic)
    supplemented = plan_explicit_verification(
        [
            *deterministic,
            _candidate(
                "agent-extra",
                discovery_origin="agent_supplement",
            ),
        ]
    )

    assert supplemented.batches[: len(baseline.batches)] == baseline.batches
    assert all(
        item.evidence[0].evidence.discovery_origin == "agent_supplement"
        for item in supplemented.batches[-1].candidates
    )


def test_unrelated_earlier_source_does_not_perturb_verifier_request() -> None:
    original = [_candidate(f"z-{index}", path="z-instructions.md") for index in range(3)]
    baseline = plan_explicit_verification(original)
    supplemented = plan_explicit_verification(
        [
            _candidate("a-extra", path="a-instructions.md"),
            *original,
        ]
    )

    baseline_batch = baseline.batches[0]
    supplemented_batch = next(
        batch
        for batch in supplemented.batches
        if batch.candidates[0].evidence[0].evidence.path
        == "z-instructions.md"
    )
    assert baseline_batch.batch_id == supplemented_batch.batch_id
    assert build_explicit_verification_prompt(baseline_batch) == (
        build_explicit_verification_prompt(supplemented_batch)
    )
    snapshot = "sha256:" + "0" * 64
    request_ids = [
        request_identity(
            stage="explicit_semantic_verify",
            protocol_version=EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
            mode="grounded_candidate_axes",
            inputs=[
                InferenceInputRef(
                    kind="explicit_grounded_candidate",
                    stable_id=item.candidate_root_id,
                    digest=item.candidate_digest,
                    snapshot_digest=snapshot,
                )
                for item in batch.candidates
            ],
            request_ordinal=stable_verification_request_ordinal(batch),
        )
        for batch in (baseline_batch, supplemented_batch)
    ]
    assert request_ids[0] == request_ids[1]
    assert all(
        len({item.evidence[0].evidence.path for item in batch.candidates}) == 1
        for batch in supplemented.batches
    )


def test_candidate_with_multiple_evidence_paths_fails_closed() -> None:
    first = _candidate("one", path="AGENTS.md")
    second = _candidate("two", path="CONTRIBUTING.md")
    mixed = first.model_copy(
        update={"evidence": [first.evidence[0], second.evidence[0]]}
    )

    with pytest.raises(ValueError, match="cannot mix evidence source paths"):
        plan_explicit_verification([mixed])


def test_planner_counts_the_attached_response_schema_at_request_boundary() -> None:
    candidates = [
        _candidate(f"near-boundary-{index}", quote="x" * 15_000)
        for index in range(4)
    ]
    plan = plan_explicit_verification(candidates)

    assert plan.planned_candidates == len(candidates)
    assert not plan.omissions
    for batch in plan.batches:
        actual_prompt = attach_response_json_schema(
            build_explicit_verification_prompt(batch),
            ExplicitVerificationResponse,
        )
        assert len(actual_prompt.encode("utf-8")) == batch.request_bytes
        assert batch.request_bytes <= MAX_REQUEST_BYTES


def test_planner_rejects_duplicate_roots_instead_of_silently_deduplicating() -> None:
    candidate = _candidate("duplicate")
    with pytest.raises(ValueError, match="must be unique"):
        plan_explicit_verification([candidate, candidate])


def test_oversized_candidate_is_explicitly_omitted() -> None:
    normal = _candidate("normal")
    oversized = _candidate("oversized", quote="x" * (MAX_CANDIDATE_BYTES + 100))
    plan = plan_explicit_verification([oversized, normal])

    assert plan.planned_candidates == 1
    assert len(plan.omissions) == 1
    assert plan.omissions[0].candidate_root_id == oversized.candidate_root_id
    assert plan.omissions[0].reason_code == "candidate_byte_limit_exceeded"
    assigned = {
        item.candidate_root_id
        for batch in plan.batches
        for item in batch.candidates
    } | {item.candidate_root_id for item in plan.omissions}
    assert assigned == {normal.candidate_root_id, oversized.candidate_root_id}


def test_response_coverage_is_atomic_for_one_candidate_batch() -> None:
    plan = plan_explicit_verification(
        [_candidate("one"), _candidate("two"), _candidate("three")]
    )
    batch = plan.batches[0]
    judgment = _judgment(batch.candidates[0])
    response = ExplicitVerificationResponse(results=[judgment])
    validated = validate_explicit_verification_response(response, batch)
    assert [item.candidate_root_id for item in validated.results] == [
        item.candidate_root_id for item in batch.candidates
    ]

    missing = ExplicitVerificationResponse(results=[])
    with pytest.raises(ValueError, match="coverage differs"):
        validate_explicit_verification_response(missing, batch)
    with pytest.raises(ValidationError, match="duplicate root IDs"):
        ExplicitVerificationResponse(results=[judgment, judgment])


def test_repair_prompt_requires_exact_root_and_full_regeneration() -> None:
    candidate = _candidate("repair")
    batch = plan_explicit_verification([candidate]).batches[0]
    original = build_explicit_verification_prompt(batch)
    repair = json.loads(
        _repair_prompt(original, "not valid", ["candidate root differs"])
    )

    assert repair["original_request"] == json.loads(original)
    assert "exactly one result" in repair["instruction"]
    assert "candidate_root_id byte-for-byte" in repair["instruction"]
    assert "Do not copy or continue" in repair["instruction"]
    assert "not valid" not in json.dumps(repair)


@pytest.mark.parametrize(
    (
        "axis_updates",
        "disposition",
        "reason",
    ),
    [
        ({}, "eligible", "verified_eligible"),
        ({"coding_relevance": "fail"}, "rejected", "coding_relevance_failed"),
        (
            {"evidence_entailment": "fail"},
            "rejected",
            "evidence_entailment_failed",
        ),
        (
            {"coding_relevance": "uncertain"},
            "review",
            "coding_relevance_uncertain",
        ),
        (
            {"evidence_entailment": "uncertain"},
            "review",
            "evidence_entailment_uncertain",
        ),
        (
            {"scope_condition_support": "fail"},
            "review",
            "scope_condition_support_failed",
        ),
        (
            {"scope_condition_support": "uncertain"},
            "review",
            "scope_condition_support_uncertain",
        ),
        ({"ambiguity": "material"}, "review", "material_ambiguity"),
        ({"ambiguity": "uncertain"}, "review", "ambiguity_uncertain"),
    ],
)
def test_policy_is_fixed_and_never_directly_accepts(
    axis_updates: dict[str, str],
    disposition: str,
    reason: str,
) -> None:
    candidate = _candidate("policy")
    judgment = _judgment(candidate, **axis_updates)
    result = apply_explicit_verification_policy(
        candidate.candidate_root_id,
        judgment,
    )
    assert result.disposition == disposition
    assert result.reason_codes == [reason]
    assert result.disposition != "accepted"


def test_relevance_or_entailment_failure_overrides_review_axes() -> None:
    candidate = _candidate("override")
    judgment = _judgment(
        candidate,
        coding_relevance="fail",
        scope_condition_support="uncertain",
        ambiguity="material",
    )
    result = apply_explicit_verification_policy(
        candidate.candidate_root_id,
        judgment,
    )
    assert result.disposition == "rejected"
    assert result.reason_codes == [
        "coding_relevance_failed",
        "scope_condition_support_uncertain",
        "material_ambiguity",
    ]


def test_semantic_rejection_does_not_make_coverage_incomplete() -> None:
    candidates = [_candidate("eligible"), _candidate("noise")]
    plan = plan_explicit_verification(candidates)
    by_label = {
        item.rule: item
        for batch in plan.batches
        for item in batch.candidates
    }
    responses = {
        batch.batch_id: ExplicitVerificationResponse(
            results=[
                _judgment(
                    batch.candidates[0],
                    coding_relevance=(
                        "fail"
                        if "noise" in batch.candidates[0].rule
                        else "pass"
                    ),
                )
            ]
        )
        for batch in plan.batches
    }
    result = finalize_explicit_verification(
        plan,
        responses_by_batch=responses,
    )

    assert by_label
    assert result.coverage.status == "complete"
    assert result.coverage.verified_roots == 2
    assert result.coverage.policy_eligible == 1
    assert result.coverage.policy_rejected == 1
    assert result.coverage.policy_review == 0
    assert not result.coverage.incomplete_reasons


def test_batch_failure_and_omission_become_review_and_incomplete() -> None:
    normal = _candidate("normal-failure")
    oversized = _candidate("large-failure", quote="z" * (MAX_CANDIDATE_BYTES + 100))
    plan = plan_explicit_verification([normal, oversized])
    batch = plan.batches[0]
    result = finalize_explicit_verification(
        plan,
        responses_by_batch={},
        failure_codes_by_batch={batch.batch_id: "provider_failure"},
    )

    assert result.coverage.status == "failed"
    assert result.coverage.technical_failure_roots == 1
    assert result.coverage.omitted_roots == 1
    assert result.coverage.policy_review == 2
    assert {item.reason_codes[0] for item in result.decisions} == {
        "technical_missing",
        "resource_omitted",
    }


def test_mixed_batch_success_and_failure_is_partial() -> None:
    candidates = [_candidate(f"mixed-{index}") for index in range(9)]
    plan = plan_explicit_verification(candidates)
    failed = plan.batches[-1]
    responses = {
        batch.batch_id: ExplicitVerificationResponse(
            results=[_judgment(batch.candidates[0])]
        )
        for batch in plan.batches[:-1]
    }
    result = finalize_explicit_verification(
        plan,
        responses_by_batch=responses,
        failure_codes_by_batch={failed.batch_id: "schema_failure"},
    )

    assert result.coverage.status == "partial"
    assert result.coverage.successful_batches == 8
    assert result.coverage.failed_batches == 1
    assert result.coverage.policy_eligible == 8
    assert result.coverage.policy_review == 1


def test_plan_result_and_coverage_reject_tampering() -> None:
    candidate = _candidate("tamper")
    plan = plan_explicit_verification([candidate])
    tampered_plan = plan.model_dump(mode="json")
    tampered_plan["planned_request_bytes"] += 1
    with pytest.raises(ValidationError, match="planned byte count"):
        ExplicitVerificationPlan.model_validate(tampered_plan)

    with pytest.raises(ValidationError, match="planned/omitted"):
        ExplicitVerificationCoverage(
            status="complete",
            grounded_roots=1,
            planned_roots=0,
            verified_roots=0,
            technical_failure_roots=0,
            omitted_roots=0,
            planned_batches=0,
            successful_batches=0,
            failed_batches=0,
            policy_eligible=0,
            policy_review=0,
            policy_rejected=1,
            incomplete_reasons=[],
        )


class _VerifierClient:
    def __init__(self) -> None:
        self.settings_seen: list[LLMSettings] = []

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: LLMSettings,
    ) -> ProviderResponse:
        del system_prompt
        self.settings_seen.append(settings)
        payload = json.loads(user_prompt)
        assert payload["stage"] == "explicit_semantic_verify"
        content = {
            "results": [
                {
                    "candidate_root_id": item["candidate_root_id"],
                    "coding_relevance": "pass",
                    "evidence_entailment": "pass",
                    "scope_condition_support": "pass",
                    "ambiguity": "none",
                    "git_need": {"status": "no_established_need", "questions": []},
                }
                for item in payload["candidates"]
            ]
        }
        return ProviderResponse(
            content=json.dumps(content),
            model=settings.model,
            input_tokens=100,
            output_tokens=40,
            latency_ms=1,
            finish_reason="stop",
        )


def test_run_uses_dedicated_stage_and_hard_llm_ceilings() -> None:
    settings = LLMSettings(
        model="fake-verifier",
        require_response_model=True,
        accepted_effective_model_ids=["fake-verifier"],
        api_key_env="FAKE_API_KEY",
        base_url="https://invalid.example/v1",
        reasoning_wire_policy="offline_mock",
        max_output_tokens=65_536,
        timeout_seconds=800,
        schema_repairs=3,
        max_concurrency=8,
        provider_retries=0,
    )
    client = _VerifierClient()
    bundle = run_explicit_verification(
        candidates=[_candidate("runner")],
        snapshot_digest="sha256:" + "1" * 64,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest="2" * 64,
    )

    assert bundle.result.coverage.status == "complete"
    assert bundle.result.coverage.policy_eligible == 1
    assert len(bundle.inference_records) == 1
    assert bundle.inference_records[0].stage == "explicit_semantic_verify"
    assert {
        item.kind for item in bundle.inference_records[0].input_manifest
    } == {"explicit_grounded_candidate"}
    assert len(client.settings_seen) == 1
    assert client.settings_seen[0].max_output_tokens == 12_288
    assert client.settings_seen[0].timeout_seconds == 300
    assert client.settings_seen[0].schema_repairs == 1
    assert client.settings_seen[0].max_concurrency == 8


@pytest.mark.parametrize("workers", [2, 12, 24])
def test_verifier_batch_concurrency_matches_global_limit(workers: int) -> None:
    class ConcurrentVerifierClient(_VerifierClient):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(workers, timeout=10)
            self.lock = threading.Lock()
            self.active = 0
            self.peak_active = 0

        def complete_with_settings(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings: LLMSettings,
        ) -> ProviderResponse:
            with self.lock:
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
            try:
                # Rendezvous proves simultaneous requests without timing sleeps.
                self.barrier.wait()
                response = super().complete_with_settings(
                    system_prompt, user_prompt, settings=settings
                )
                self.barrier.wait()
                return response
            finally:
                with self.lock:
                    self.active -= 1

    settings = LLMSettings(
        model="fake-verifier",
        require_response_model=True,
        accepted_effective_model_ids=["fake-verifier"],
        api_key_env="FAKE_API_KEY",
        base_url="https://invalid.example/v1",
        reasoning_wire_policy="offline_mock",
        max_concurrency=workers,
        provider_retries=0,
    )
    client = ConcurrentVerifierClient()
    bundle = run_explicit_verification(
        candidates=[_candidate(f"parallel-{index}") for index in range(2 * workers)],
        snapshot_digest="sha256:" + "1" * 64,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest="2" * 64,
        limits=ExplicitVerificationLimits(max_candidates_per_batch=1),
    )

    assert bundle.result.coverage.status == "complete"
    assert bundle.result.coverage.successful_batches == 2 * workers
    assert bundle.result.coverage.policy_eligible == 2 * workers
    assert len(client.settings_seen) == 2 * workers
    assert {item.max_concurrency for item in client.settings_seen} == {workers}
    assert client.peak_active == workers
    assert client.active == 0


def test_oversized_provider_content_is_terminal_without_repair_or_retention() -> None:
    class OversizedResponseClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_with_settings(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            settings: LLMSettings,
        ) -> ProviderResponse:
            del system_prompt, user_prompt
            self.calls += 1
            return ProviderResponse(
                content="x" * (MAX_RESPONSE_BYTES + 1),
                model=settings.model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=1,
                finish_reason="stop",
            )

    settings = LLMSettings(
        model="fake-verifier",
        require_response_model=True,
        accepted_effective_model_ids=["fake-verifier"],
        api_key_env="FAKE_API_KEY",
        base_url="https://invalid.example/v1",
        reasoning_wire_policy="offline_mock",
        schema_repairs=3,
        provider_retries=0,
    )
    client = OversizedResponseClient()
    bundle = run_explicit_verification(
        candidates=[_candidate("oversized-response")],
        snapshot_digest="sha256:" + "1" * 64,
        llm_settings=settings,
        client=client,  # type: ignore[arg-type]
        budget=InferenceBudget(settings),
        protocol_manifest_digest="2" * 64,
    )

    record = bundle.inference_records[0]
    assert client.calls == 2
    assert record.terminal_status == "failure"
    assert len(record.attempts) == 2
    assert [item.output_limit_retry for item in record.attempts] == [0, 1]
    assert all(
        item.validation_errors
        == ["provider response exceeded max_response_bytes"]
        for item in record.attempts
    )
    assert record.structured_response is None
    assert bundle.result.coverage.status == "failed"
