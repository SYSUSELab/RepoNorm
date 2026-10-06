from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.inference import InferenceBudget
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.semantic.models import (
    FalsificationResult,
    PopulationMemberRef,
)
from repo_constraint_memory_v3.semantic.falsification import (
    FALSIFICATION_PROTOCOL_VERSION,
    FALSIFICATION_RESPONSE_SCHEMA_VERSION,
    FALSIFICATION_SYSTEM_PROMPT,
    FalsificationAdmissionLedger,
    FalsificationCostNotAdmitted,
    FalsificationResponse,
    run_falsification,
)
from repo_constraint_memory_v3.semantic.validation import _current_proposition
from repo_constraint_memory_v3.unified.inputs import (
    _falsification_capability,
    build_current_source_assessments,
)


def test_falsification_prompt_contrastively_checks_governed_scope() -> None:
    assert "governed-change test" in FALSIFICATION_SYSTEM_PROMPT
    assert "behavioral\n  witness" in FALSIFICATION_SYSTEM_PROMPT
    assert "test construction directly governs test files" in (
        FALSIFICATION_SYSTEM_PROMPT
    )
    assert "both witness and scope" in FALSIFICATION_SYSTEM_PROMPT
    assert "never infer it from directory role" in FALSIFICATION_SYSTEM_PROMPT
    assert "material-qualifier test" in FALSIFICATION_SYSTEM_PROMPT
    assert "even when the hypothesis declares no condition" in (
        FALSIFICATION_SYSTEM_PROMPT
    )
    assert "positively establishes a conflicting in-scope member" in (
        FALSIFICATION_SYSTEM_PROMPT
    )
    assert "do not penalize an absent" in FALSIFICATION_SYSTEM_PROMPT


class _RecordingSemanticClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.falsification_requests: list[dict[str, object]] = []
        self.falsification_request_bytes: list[int] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "counterexample_verify":
            self.falsification_requests.append(request)
            self.falsification_request_bytes.append(
                len(user_prompt.encode("utf-8"))
            )
        return super().complete(system_prompt, user_prompt)


class _LongRationaleFalsificationClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.falsification_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        self.falsification_calls += 1
        value = json.loads(response.content)
        value["rationale_summary"] = "R" * 1_600
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="long-falsification-rationale-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _MisclassifyingNegativeTestClient(ParameterizedSemanticClient):
    """Model the classifier mistake that the independent verifier must repair."""

    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.misclassified_member_ids: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        stage = request.get("stage")
        value = json.loads(response.content)
        if stage == "population_semantic_classify":
            from repo_constraint_memory_v3.semantic.classification_transport import expand_classification_request
            request = expand_classification_request(request)
            request_members = {
                item["member_ordinal"]: item for item in request["members"]
            }
            for item in value["results"]:
                request_member = request_members[item["member_ordinal"]]
                source_context = json.dumps(
                    request_member["packet"]["fact"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if "pytest.raises" in source_context:
                    item.update(
                        {
                            "label": "counterexample",
                            "evidence_context": "production",
                            "reason_code": next(
                                ref for ref, code in request["reason_catalog"].items()
                                if code == "mock_counterexample"
                            ),
                            "reason_summary": (
                                "The classifier mistakes the deliberate failing "
                                "example for permitted production behavior."
                            ),
                        }
                    )
                    self.misclassified_member_ids.append(
                        request_member["member_id"]
                    )
                else:
                    item.update(
                        {
                            "label": "support",
                            "evidence_context": "production",
                            "reason_code": next(
                                ref for ref, code in request["reason_catalog"].items()
                                if code == "mock_support"
                            ),
                            "reason_summary": (
                                "The supplied member independently supports the rule."
                            ),
                        }
                    )
        elif stage == "counterexample_verify":
            reviewed = list(request["reviewed_member_ids"])
            value.update(
                {
                    "reviewed_member_ids": reviewed,
                    "verdict": "supported_as_written",
                    "git_need": {"status": "no_established_need", "questions": []},
                    "evidence_refs": [request["allowed_evidence_refs"][0]],
                    "issue_interpretations": [
                        {
                            "member_id": member_id,
                            "interpretation": "confirming_negative_test",
                            "rationale_summary": (
                                "The member deliberately violates the rule inside "
                                "pytest.raises and therefore confirms enforcement."
                            ),
                        }
                        for member_id in reviewed
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
                    "rationale_summary": (
                        "The apparent contradiction is a confirming negative test."
                    ),
                }
            )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="misclassified-negative-test-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _AllSupportedPopulationClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        for item in value["results"]:
            item.update(
                {
                    "label": "support",
                    "evidence_context": "production",
                    "reason_code": next(
                        ref for ref, code in request["reason_catalog"].items()
                        if code == "mock_support"
                    ),
                    "reason_summary": (
                        "The supplied member independently supports the rule."
                    ),
                }
            )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="all-supported-population-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _AllIneligiblePopulationClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "population_semantic_classify":
            return response
        value = json.loads(response.content)
        for item in value["results"]:
            item.update(
                {
                    "label": "ineligible",
                    "evidence_context": "irrelevant",
                    "reason_code": next(
                        ref for ref, code in request["reason_catalog"].items()
                        if code == "irrelevant_member"
                    ),
                    "reason_summary": (
                        "The supplied member is outside the semantic predicate."
                    ),
                }
            )
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="all-ineligible-population-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _ContradictorySupportedFalsificationClient(ParameterizedSemanticClient):
    """Return a verdict that contradicts its own per-member interpretation."""

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        value = json.loads(response.content)
        value["verdict"] = "supported_as_written"
        value["issue_interpretations"] = [
            {
                "member_id": member_id,
                "interpretation": "unresolved",
                "rationale_summary": (
                    "The issue remains unresolved despite the claimed verdict."
                ),
            }
            for member_id in request["reviewed_member_ids"]
        ]
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="contradictory-supported-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_provider_rationale_1600_is_cached_replayed_and_aggregated_to_800(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    inference_cache = tmp_path / "long-rationale-cache"
    first_run = tmp_path / "long-rationale-first"
    first_client = _LongRationaleFalsificationClient(
        category="long-rationale-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    first = build_fixture(
        repository,
        first_run,
        client=first_client,
        inference_cache=inference_cache,
    )

    assert first.current_source is not None
    records = [
        record
        for record in first.inference_records
        if record.stage == "counterexample_verify"
        and record.terminal_status == "success"
    ]
    assert records
    assert first_client.falsification_calls == len(records)
    assert all(
        record.protocol_version == FALSIFICATION_PROTOCOL_VERSION
        and record.response_schema_version
        == FALSIFICATION_RESPONSE_SCHEMA_VERSION
        for record in records
    )
    assert all(
        len(record.structured_response["rationale_summary"]) == 1_600
        for record in records
        if record.structured_response is not None
    )
    assert all(
        len(item.rationale_summary) == 800
        for item in first.current_source.falsifications
        if item.inference_ids
    )
    assert FALSIFICATION_PROTOCOL_VERSION == "counterexample-falsification-1.11"
    assert FALSIFICATION_RESPONSE_SCHEMA_VERSION == (
        "counterexample-falsification-response-1.5"
    )
    assert FalsificationResponse.model_json_schema()["properties"][
        "rationale_summary"
    ]["maxLength"] == 1_600
    assert FalsificationResult.model_json_schema()["properties"][
        "rationale_summary"
    ]["maxLength"] == 800
    overlong = dict(records[0].structured_response or {})
    overlong["rationale_summary"] = "R" * 1_601
    with pytest.raises(ValidationError):
        FalsificationResponse.model_validate(overlong)
    assert validate_run_directory(first_run, repository_root=repository) == []

    replay_run = tmp_path / "long-rationale-replay"
    replay_client = _LongRationaleFalsificationClient(
        category="long-rationale-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    replayed = build_fixture(
        repository,
        replay_run,
        client=replay_client,
        inference_cache=inference_cache,
    )
    assert replay_client.falsification_calls == 0
    assert replayed.current_source is not None
    assert replayed.current_source.falsifications == (
        first.current_source.falsifications
    )
    assert validate_run_directory(replay_run, repository_root=repository) == []


def test_complete_high_quality_candidate_enters_experimental_accept_only(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=_AllSupportedPopulationClient(
            category="experimental-accept-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.review_items[0].epistemic_status == "review"
    assert result.current_source.policy_decisions[0].experimental_disposition == (
        "accept_candidate"
    )
    assert result.current_source.index.root_accepted == 0
    assert result.current_source.index.experimental_accept_candidates == 1
    assert result.current_source.index.index_version == "current-source-index-1.10"
    decision = result.current_source.policy_decisions[0]
    assert decision.policy_version == "current-source-experimental-policy-1.4"
    assert decision.reason_codes == [
        "complete_population_and_falsification",
        "local_quality_axes_passed",
        "pre_calibration_non_delivery",
    ]
    hypothesis = next(
        item
        for item in result.current_source.hypotheses
        if item.hypothesis_id == decision.hypothesis_id
    )
    population = next(
        item
        for item in result.current_source.populations
        if item.hypothesis_id == decision.hypothesis_id
    )
    proposition = result.current_source.propositions[0]
    evidence_keys = [
        (item.kind, item.stable_id, item.snapshot_digest)
        for item in proposition.evidence_refs
    ]
    assert evidence_keys == sorted(set(evidence_keys))
    assert {
        ("semantic_observation", item, hypothesis.snapshot_digest)
        for item in hypothesis.witness_refs
    } <= set(evidence_keys)
    assert {
        (
            f"observation_{member.observation_ref.universe}",
            member.observation_ref.observation_id,
            hypothesis.snapshot_digest,
        )
        for member in population.members
        if member.label == "support"
    } <= set(evidence_keys)

    template = population.members[0]
    synthetic_members = [
        template.model_copy(
            update={
                "observation_ref": PopulationMemberRef(
                    universe=universe,
                    observation_id=f"{universe}-support",
                    file_id=None,
                    path=None,
                )
            }
        )
        for universe in ("file", "entity", "site", "relation")
    ]
    synthetic_population = population.model_copy(
        update={"members": [*synthetic_members, synthetic_members[0]]}
    )
    synthetic_proposition = _current_proposition(
        hypothesis=hypothesis,
        population=synthetic_population,
        falsification=result.current_source.falsifications[0],
    )
    synthetic_evidence_keys = [
        (item.kind, item.stable_id, item.snapshot_digest)
        for item in synthetic_proposition.evidence_refs
    ]
    assert synthetic_evidence_keys == sorted(set(synthetic_evidence_keys))
    assert {
        item.kind
        for item in synthetic_proposition.evidence_refs
        if item.kind.startswith("observation_")
    } == {
        "observation_file",
        "observation_entity",
        "observation_site",
        "observation_relation",
    }
    assert all(
        "implicit_current" not in claim.origin_channels
        for claim in result.memory.claims
    )
    policy = json.loads(
        (run / "current-source" / "policy-decisions.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["decisions"][0]["experimental_disposition"] == (
        "accept_candidate"
    )
    assert validate_run_directory(run, repository_root=repository) == []


def test_complete_zero_support_population_is_rejected_without_falsification(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "zero-support-run",
        client=_AllIneligiblePopulationClient(
            category="zero-support-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )

    assert result.current_source is not None
    assert result.current_source.populations[0].status == "complete"
    assert result.current_source.populations[0].support_count == 0
    assert len(result.current_source.falsifications) == 1
    deterministic = result.current_source.falsifications[0]
    assert deterministic.status == "complete"
    assert deterministic.verdict == "reject_insufficient"
    assert deterministic.inference_ids == []
    assert deterministic.execution.planned_calls == 0
    assert deterministic.execution.planned_items == 0
    assert _falsification_capability(
        result.current_source.populations[0], deterministic
    ) == ("complete", ())
    assert "no_population_support_falsification_not_required" in (
        deterministic.rationale_summary
    )
    assert not any(
        record.stage == "counterexample_verify"
        for record in result.inference_records
    )
    assert result.current_source.review_items[0].epistemic_status == "rejected"
    assert result.current_source.review_items[0].reason_codes == [
        "no_population_support"
    ]
    assert result.current_source.policy_decisions[0].experimental_disposition == (
        "rejected"
    )
    assert result.current_source.index.resource_use[
        "falsification_deterministic_rejections"
    ] == 1
    assert result.current_source.index.resource_use[
        "falsification_terminal_inference_record_hypotheses"
    ] == 0


def test_exhausted_falsification_closes_ten_thousand_issues_without_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    config = config_for_tests()
    baseline = build_fixture(
        repository,
        tmp_path / "baseline-budget-falsification",
        config=config,
        client=ParameterizedSemanticClient(
            category="budget-falsification-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert baseline.current_source is not None
    population = baseline.current_source.populations[0]
    hypothesis = next(
        item
        for item in baseline.current_source.hypotheses
        if item.hypothesis_id == population.hypothesis_id
    )
    template_member = population.members[0]
    template_packet = next(
        item
        for item in baseline.current_source.member_packets
        if item.observation_ref.observation_id
        == template_member.observation_ref.observation_id
    )
    members = []
    packets = []
    for ordinal in range(10_000):
        observation_id = f"OBS-STRESS-{ordinal:08d}"
        member_id = f"POPMEMBER-{ordinal:024x}"
        observation_ref = template_member.observation_ref.model_copy(
            update={"observation_id": observation_id}
        )
        members.append(
            template_member.model_copy(
                update={
                    "member_id": member_id,
                    "observation_ref": observation_ref,
                    "label": "unknown",
                    "classification_basis": "technical_unknown",
                    "inference_ref": None,
                }
            )
        )
        packets.append(
            template_packet.model_copy(
                update={
                    "packet_id": f"MEMPACKET-{ordinal:024x}",
                    "observation_ref": observation_ref,
                }
            )
        )
    synthetic_population = population.model_copy(
        update={
            "members": members,
            "support_count": 0,
            "counterexample_count": 0,
            "ineligible_count": 0,
            "unknown_count": len(members),
            "semantic_unknown_count": 0,
            "technical_unknown_count": len(members),
            "status": "partial",
        }
    )

    def forbidden_json(*args, **kwargs):
        del args, kwargs
        raise AssertionError("exhausted falsification serialized a payload")

    class ForbiddenPayloadJson:
        loads = staticmethod(json.loads)
        dumps = staticmethod(forbidden_json)

    monkeypatch.setattr(
        "repo_constraint_memory_v3.semantic.falsification.json",
        ForbiddenPayloadJson,
    )
    budget = InferenceBudget(
        config.llm,
        initial_input_tokens=config.llm.max_total_input_tokens,
    )
    built = run_falsification(
        hypothesis=hypothesis,
        population=synthetic_population,
        member_packets=packets,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        client=object(),  # type: ignore[arg-type]
        budget=budget,
        protocol_manifest_digest=(
            baseline.protocol_manifest.protocol_manifest_digest
        ),
    )

    assert built.inference_records == ()
    assert built.result.status == "partial"
    assert built.result.execution.planned_items == 10_000
    assert built.result.execution.omitted_items == 10_000
    assert built.result.execution.terminal_items == 0
    assert not built.result.execution.request_plan_complete
    assert built.result.execution.closure_reason == (
        "falsification_not_eligible:population_incomplete"
    )
    assert "counterexample_verify" not in budget.stage_progress


@pytest.mark.parametrize(
    ("semantic_update", "expected_reason"),
    [
        (
            {
                "max_falsification_items_per_request": 1,
                "max_falsification_calls_per_hypothesis": 1,
            },
            "max_falsification_calls_per_hypothesis",
        ),
        (
            {
                "max_falsification_items_per_request": 1,
                "max_falsification_request_bytes": 10_000,
                "max_falsification_request_bytes_per_hypothesis": 10_000,
            },
            "max_falsification_request_bytes_per_hypothesis",
        ),
    ],
)
def test_falsification_cost_rejection_preserves_complete_issue_denominator(
    tmp_path: Path,
    semantic_update: dict[str, int],
    expected_reason: str,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    baseline = build_fixture(
        repository,
        tmp_path / "baseline-cost-admission",
        client=ParameterizedSemanticClient(
            category=f"falsification-cost-{expected_reason}",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert baseline.current_source is not None
    population = baseline.current_source.populations[0]
    hypothesis = next(
        item
        for item in baseline.current_source.hypotheses
        if item.hypothesis_id == population.hypothesis_id
    )
    packets = [
        item
        for item in baseline.current_source.member_packets
        if item.hypothesis_id == hypothesis.hypothesis_id
    ]
    issue_ids = [
        item.member_id
        for item in population.members
        if item.label in {"counterexample", "unknown"}
    ]
    assert len(issue_ids) > 1
    config = config_for_tests(semantic=semantic_update)
    client = _RecordingSemanticClient(
        category="forbidden-falsification-provider-call",
        rule="A rejected job must never reach the provider.",
    )
    budget = InferenceBudget(config.llm)

    built = run_falsification(
        hypothesis=hypothesis,
        population=population,
        member_packets=packets,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        client=client,
        budget=budget,
        protocol_manifest_digest=(
            baseline.protocol_manifest.protocol_manifest_digest
        ),
    )

    result = built.result
    closure = result.execution.closure_reason
    assert result.status == "partial"
    assert result.verdict == "uncertain"
    assert built.inference_records == ()
    assert client.falsification_requests == []
    assert "counterexample_verify" not in budget.stage_progress
    assert result.reviewed_member_ids == []
    assert not result.execution.request_plan_complete
    assert result.execution.planned_calls == 0
    assert result.execution.terminal_calls == 0
    assert result.execution.omitted_calls == 0
    assert result.execution.planned_items == len(issue_ids)
    assert result.execution.terminal_items == 0
    assert result.execution.omitted_items == len(issue_ids)
    assert closure is not None
    stable_reason = f"falsification_cost_not_admitted:{expected_reason}"
    assert closure == stable_reason
    assert result.admission_estimate is not None
    assert result.admission_estimate.status == "not_admitted"
    assert result.admission_estimate.reason_code == expected_reason
    assert result.admission_estimate.estimated_calls > 0
    assert result.admission_estimate.estimated_request_bytes > 0
    detail_reason = next(
        item
        for item in result.incomplete_reasons
        if item.startswith(f"{stable_reason}:calls=")
    )
    assert result.incomplete_reasons == [stable_reason, detail_reason]
    assert result.unresolved_questions == [detail_reason]
    capability_status, capability_reasons = _falsification_capability(population, result)
    assert capability_status == "partial"
    assert "falsification_cost_not_admitted" in capability_reasons
    assert "falsification_execution_incomplete" in capability_reasons
    assert _falsification_capability(
        population,
        result.model_copy(update={
            "status": "complete", "admission_estimate": None, "incomplete_reasons": [],
        }),
    )[0] == "partial"
    assert _falsification_capability(population, None)[0] == "unavailable"
    assert _falsification_capability(
        population.model_copy(update={"status": "partial"}), None
    ) == ("not_required", ())
    assert _falsification_capability(
        population,
        result.model_copy(update={"status": "failed", "admission_estimate": None}),
    )[0] == "failed"
    # Missing required evidence remains a valid coverage gap in adaptation.
    assessments = build_current_source_assessments(
        current_source=replace(baseline.current_source, falsifications=()),
        snapshot_digest=hypothesis.snapshot_digest,
        profile_digest=baseline.unified.bundle.index.profile_digest,
    )
    assert assessments and all(item.integrity == "valid" for item in assessments)
    assert all("falsification_missing" in item.typed_reasons for item in assessments)


@pytest.mark.parametrize(
    ("semantic_update", "planned_calls", "planned_request_bytes", "reason"),
    [
        (
            {
                "max_falsification_calls_per_hypothesis": 1,
                "max_total_falsification_calls": 1,
            },
            1,
            100,
            "max_total_falsification_calls",
        ),
        (
            {
                "max_falsification_request_bytes_per_hypothesis": 2_000_000,
                "max_total_falsification_request_bytes": 2_000_000,
            },
            1,
            1_500_000,
            "max_total_falsification_request_bytes",
        ),
    ],
)
def test_run_wide_falsification_ledger_rejects_second_complete_job_atomically(
    semantic_update: dict[str, int],
    planned_calls: int,
    planned_request_bytes: int,
    reason: str,
) -> None:
    settings = config_for_tests(semantic=semantic_update).semantic
    ledger = FalsificationAdmissionLedger(settings)
    ledger.admit(
        planned_calls=planned_calls,
        planned_request_bytes=planned_request_bytes,
    )

    with pytest.raises(FalsificationCostNotAdmitted) as caught:
        ledger.admit(
            planned_calls=planned_calls,
            planned_request_bytes=planned_request_bytes,
        )

    assert caught.value.reason_code == reason
    assert caught.value.planned_calls == planned_calls
    assert caught.value.planned_request_bytes == planned_request_bytes
    assert ledger.admitted_hypotheses == 1
    assert ledger.admitted_calls == planned_calls
    assert ledger.admitted_request_bytes == planned_request_bytes
    assert ledger.not_admitted_hypotheses == 1
    assert ledger.not_admitted_calls == planned_calls
    assert ledger.not_admitted_request_bytes == planned_request_bytes


@pytest.fixture(scope="module")
def cost_omitted_falsification_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> FalsificationResult:
    tmp_path = tmp_path_factory.mktemp("falsification-cost-schema")
    repository = copied_current_source_fixture(tmp_path)
    baseline = build_fixture(
        repository,
        tmp_path / "baseline",
        client=ParameterizedSemanticClient(
            category="falsification-cost-schema",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert baseline.current_source is not None
    population = baseline.current_source.populations[0]
    hypothesis = next(
        item
        for item in baseline.current_source.hypotheses
        if item.hypothesis_id == population.hypothesis_id
    )
    packets = [
        item
        for item in baseline.current_source.member_packets
        if item.hypothesis_id == hypothesis.hypothesis_id
    ]
    config = config_for_tests(
        semantic={
            "max_falsification_items_per_request": 1,
            "max_falsification_calls_per_hypothesis": 1,
        }
    )
    built = run_falsification(
        hypothesis=hypothesis,
        population=population,
        member_packets=packets,
        semantic_settings=config.semantic,
        llm_settings=config.llm,
        client=_RecordingSemanticClient(
            category="forbidden-falsification-schema-call",
            rule="A rejected job must never reach the provider.",
        ),
        budget=InferenceBudget(config.llm),
        protocol_manifest_digest=(
            baseline.protocol_manifest.protocol_manifest_digest
        ),
    )
    assert built.result.admission_estimate is not None
    return built.result


def _redigest_falsification(payload: dict[str, object]) -> None:
    payload["artifact_digest"] = payload_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "artifact_digest"
        }
    )


def test_falsification_admission_estimate_rejects_unknown_gate_reason(
    cost_omitted_falsification_result: FalsificationResult,
) -> None:
    payload = cost_omitted_falsification_result.model_dump(mode="json")
    estimate = payload["admission_estimate"]
    assert isinstance(estimate, dict)
    estimate["reason_code"] = "unfrozen_falsification_gate"
    _redigest_falsification(payload)

    with pytest.raises(ValidationError, match="Input should be"):
        FalsificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "tamper",
    ["execution_closure", "stable_reason", "estimated_calls"],
)
def test_falsification_admission_estimate_must_match_preplan_closure(
    cost_omitted_falsification_result: FalsificationResult,
    tamper: str,
) -> None:
    payload = cost_omitted_falsification_result.model_dump(mode="json")
    estimate = payload["admission_estimate"]
    execution = payload["execution"]
    incomplete_reasons = payload["incomplete_reasons"]
    assert isinstance(estimate, dict)
    assert isinstance(execution, dict)
    assert isinstance(incomplete_reasons, list)
    stable_reason = (
        "falsification_cost_not_admitted:" + str(estimate["reason_code"])
    )
    if tamper == "execution_closure":
        execution["closure_reason"] = (
            "falsification_cost_not_admitted:"
            "max_total_falsification_request_bytes"
        )
    elif tamper == "stable_reason":
        payload["incomplete_reasons"] = [
            item for item in incomplete_reasons if item != stable_reason
        ]
    else:
        estimate["estimated_calls"] = int(estimate["estimated_calls"]) + 1
    _redigest_falsification(payload)

    with pytest.raises(
        ValidationError,
        match="admission estimate requires a partial pre-inference result",
    ):
        FalsificationResult.model_validate(payload)


def test_supported_verdict_with_unresolved_members_fails_closed(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "contradictory-supported-run",
        client=_ContradictorySupportedFalsificationClient(
            category="contradictory-falsification-regression",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )

    assert result.current_source is not None
    assert result.current_source.falsifications[0].status == "partial"
    assert result.current_source.policy_decisions[0].experimental_disposition == (
        "review"
    )
    assert all(
        record.terminal_status == "failure"
        for record in result.inference_records
        if record.stage == "counterexample_verify"
    )


def test_cost_omitted_falsification_remains_required_review_and_replays(tmp_path: Path) -> None:
    from repo_constraint_memory_v3.unified.ids import artifact_digest

    repository = copied_current_source_fixture(tmp_path)
    output = tmp_path / "required-falsification-gap"
    result = build_fixture(
        repository, output,
        config=config_for_tests(semantic={
            "max_falsification_items_per_request": 1,
            "max_falsification_calls_per_hypothesis": 1,
        }),
        client=ParameterizedSemanticClient(
            category="required-falsification-gap",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    omitted = [item for item in result.current_source.falsifications
               if item.admission_estimate is not None]
    assert omitted
    gap_ids = {item.falsification_id for item in omitted}
    currents = [item for item in result.unified.bundle.current_evidence
                if any(ref.stable_id in gap_ids for ref in item.falsification_refs)]
    assert currents
    case_ids = {item.case_id for item in currents}
    for current in currents:
        dependency = next(item for item in current.dependencies
                          if item.capability == "current_falsification")
        assert dependency.requirement_mode == "required"
        assert dependency.status == "partial"
        assert "falsification_execution_incomplete" in dependency.reason_codes
    claims = [item for item in result.unified.bundle.canonical_claims if item.case_id in case_ids]
    assert claims and all(item.epistemic_status == "review" for item in claims)
    assert all(item.review_facets == ["technical_or_coverage_gap"] for item in claims)
    findings = [item for item in result.unified.bundle.findings if item.case_id in case_ids]
    assert findings and all(item.inference_ref is None for item in findings)
    assert validate_run_directory(output, repository_root=repository) == []
    assert validate_run_directory(output, repository_root=repository, require_complete=True)

    current = currents[0]
    path = output / "canonical/audit/current-evidence" / f"{current.bundle_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dependency in payload["dependencies"]:
        if dependency["capability"] == "current_falsification":
            dependency.update(status="complete", reason_codes=[])
    payload["status"] = "complete"
    payload["incomplete_reasons"] = []
    payload["artifact_digest"] = artifact_digest({
        key: value for key, value in payload.items() if key != "artifact_digest"
    })
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    errors = validate_run_directory(output, repository_root=repository)
    assert any("round-zero current evidence differs from replay" in item for item in errors)


def test_falsifier_can_recover_a_classifier_negative_test_mistake(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    tests = repository / "tests"
    tests.mkdir()
    (tests / "test_public_error_translation.py").write_text(
        """import pytest

from aurora.api import fetch_document
from aurora.errors import AuroraPublicError


def test_empty_key_is_translated_at_the_public_boundary() -> None:
    with pytest.raises(AuroraPublicError):
        fetch_document(\"\")
""",
        encoding="utf-8",
    )
    client = _MisclassifyingNegativeTestClient(
        category="public-error-boundary",
        rule=(
            "Public API functions translate InternalStorageError into "
            "AuroraPublicError."
        ),
    )
    result = build_fixture(repository, tmp_path / "run", client=client)

    assert result.current_source is not None
    population = result.current_source.populations[0]
    falsification = result.current_source.falsifications[0]
    decision = result.current_source.policy_decisions[0]
    assert client.misclassified_member_ids
    assert {
        item.member_id
        for item in population.members
        if item.label == "counterexample"
        and item.evidence_context == "production"
    } == set(client.misclassified_member_ids)
    assert set(falsification.reviewed_member_ids) == set(
        client.misclassified_member_ids
    )
    assert {
        item.interpretation for item in falsification.issue_interpretations
    } == {"confirming_negative_test"}
    assert falsification.status == "complete"
    assert falsification.verdict == "supported_as_written"
    assert falsification.evidence_refs
    assert decision.experimental_disposition == "accept_candidate"
    assert result.current_source.review_items[0].epistemic_status == "review"
    assert result.current_source.index.root_accepted == 0
    assert result.current_source.index.experimental_accept_candidates == 1
    assert all(
        "implicit_current" not in claim.origin_channels
        for claim in result.memory.claims
    )


class _MissingReviewedIdsClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        value = json.loads(response.content)
        value.pop("reviewed_member_ids", None)
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="missing-reviewed-ids-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_falsifier_host_supplies_reviewed_member_ids(tmp_path: Path) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_MissingReviewedIdsClient(
            category="host-bookkeeping-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.falsifications[0].status == "complete"
    assert all(
        record.terminal_status == "success"
        for record in result.inference_records
        if record.stage == "counterexample_verify"
    )


def test_falsifier_covers_every_issue_and_omits_generator_prompts(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    client = _RecordingSemanticClient(
        category="independent-verifier-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    result = build_fixture(repository, tmp_path / "run", client=client)
    assert result.current_source is not None
    population = result.current_source.populations[0]
    falsification = result.current_source.falsifications[0]
    issue_ids = [
        item.member_id
        for item in population.members
        if item.label in {"counterexample", "unknown"}
    ]
    assert falsification.status == "complete"
    assert falsification.reviewed_member_ids == issue_ids
    assert [
        member_id
        for request in client.falsification_requests
        for member_id in request["reviewed_member_ids"]  # type: ignore[index]
    ] == issue_ids
    assert all(
        "expected_counterexamples" not in request["hypothesis"]  # type: ignore[operator]
        and "falsification_questions" not in request["hypothesis"]  # type: ignore[operator]
        for request in client.falsification_requests
    )
    assert all(
        len(request["population_summary"]["issue_source_packets"])  # type: ignore[index]
        == len(request["reviewed_member_ids"])  # type: ignore[arg-type]
        and request["population_summary"]["support_source_packets"]  # type: ignore[index]
        for request in client.falsification_requests
    )
    assert all(
        item.epistemic_status != "accepted"
        for item in result.current_source.review_items
    )


def test_falsifier_partitions_by_request_bytes_without_losing_issues(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    request_byte_limit = 70_000
    client = _RecordingSemanticClient(
        category="byte-bounded-verifier-fixture",
        rule="Contributors preserve the observed fixture convention.",
    )
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config_for_tests(
            semantic={
                "max_falsification_items_per_request": 128,
                "max_falsification_request_bytes": request_byte_limit,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    population = result.current_source.populations[0]
    falsification = result.current_source.falsifications[0]
    issue_ids = [
        item.member_id
        for item in population.members
        if item.label in {"counterexample", "unknown"}
    ]
    records = [
        record
        for record in result.inference_records
        if record.stage == "counterexample_verify"
    ]
    reviewed_ids = [
        member_id
        for request in client.falsification_requests
        for member_id in request["reviewed_member_ids"]  # type: ignore[index]
    ]
    assert len(client.falsification_requests) > 1
    # Provider calls may execute concurrently; the frozen aggregate below owns
    # canonical member order while the observed call set proves exact coverage.
    assert len(reviewed_ids) == len(set(reviewed_ids))
    assert set(reviewed_ids) == set(issue_ids)
    assert falsification.reviewed_member_ids == issue_ids
    assert falsification.status == "complete"
    assert len(records) == len(client.falsification_requests)
    assert all(record.terminal_status == "success" for record in records)
    assert all(
        size <= request_byte_limit
        for size in client.falsification_request_bytes
    )


class _NarrowThenSupportClient(_RecordingSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.parent_hypothesis_id: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        hypothesis_id = request["hypothesis"]["hypothesis_id"]
        if self.parent_hypothesis_id is None:
            self.parent_hypothesis_id = hypothesis_id
        value = json.loads(response.content)
        if hypothesis_id == self.parent_hypothesis_id:
            value["verdict"] = "narrow_scope"
            value["proposed_applies_to"] = ["src/aurora/**"]
            value["rationale_summary"] = "The evidence supports a narrower scope."
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="narrow-then-support-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _ConditionThenSupportClient(_NarrowThenSupportClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        hypothesis_id = request["hypothesis"]["hypothesis_id"]
        if hypothesis_id != self.parent_hypothesis_id:
            return response
        value = json.loads(response.content)
        value["verdict"] = "narrow_condition"
        value["proposed_applies_to"] = None
        value["proposed_condition"] = "when the fixture plugin is enabled"
        value["rationale_summary"] = "The evidence supports a narrower condition."
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="condition-then-support-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_refinement_preserves_ancestry_and_reruns_complete_population(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_NarrowThenSupportClient(
            category="bounded-refinement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert len(result.current_source.refinements) == 1
    refinement = result.current_source.refinements[0]
    parent = next(
        item
        for item in result.current_source.hypotheses
        if item.hypothesis_id == refinement.parent_hypothesis_id
    )
    child = next(
        item
        for item in result.current_source.hypotheses
        if item.hypothesis_id == refinement.child_hypothesis_id
    )
    parent_population = next(
        item
        for item in result.current_source.populations
        if item.hypothesis_id == parent.hypothesis_id
    )
    child_population = next(
        item
        for item in result.current_source.populations
        if item.hypothesis_id == child.hypothesis_id
    )
    assert child.parent_hypothesis_id == parent.hypothesis_id
    assert child.refinement_round == parent.refinement_round + 1
    assert child.applies_to == ["src/aurora/**"]
    assert {
        item.observation_ref.observation_id for item in parent_population.members
    } <= {
        item.observation_ref.observation_id for item in child_population.members
    }
    assert all(
        item.label == "ineligible"
        for item in child_population.members
        if not (item.observation_ref.path or "").startswith("src/aurora/")
    )
    assert len(refinement.carried_member_dispositions) == len(
        parent_population.members
    )
    assert len(result.current_source.falsifications) == 2
    decision = result.current_source.policy_decisions[0]
    assert decision.hypothesis_id == child.hypothesis_id
    assert decision.experimental_disposition == "review"
    assert decision.reason_codes == [
        "post_falsification_refinement",
        "refinement_requires_calibration",
    ]
    assert result.current_source.index.experimental_accept_candidates == 0
    assert result.current_source.index.experimental_review == 1


def test_sampled_refinement_carries_parent_inspected_members(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "sampled-refinement-run",
        config=config_for_tests(
            semantic={
                "max_classified_population_members_per_hypothesis": 1,
                "max_total_classified_population_members": 2,
            }
        ),
        client=_ConditionThenSupportClient(
            category="sampled-refinement-fixture",
            rule="Contributors preserve the sampled fixture convention.",
        ),
    )

    assert result.current_source is not None
    current = result.current_source
    assert len(current.refinements) == 1
    refinement = current.refinements[0]
    parent_population = next(
        item
        for item in current.populations
        if item.hypothesis_id == refinement.parent_hypothesis_id
    )
    child_population = next(
        item
        for item in current.populations
        if item.hypothesis_id == refinement.child_hypothesis_id
    )
    assert parent_population.sampling_plan.provider_inspected_count == 1
    assert child_population.sampling_plan.required_inspected_member_ids
    parent_observation_ids = {
        item.observation_ref.observation_id
        for item in parent_population.members
    }
    child_required_observation_ids = {
        item.observation_ref.observation_id
        for item in child_population.members
        if item.member_id
        in child_population.sampling_plan.required_inspected_member_ids
    }
    assert child_required_observation_ids <= parent_observation_ids
    assert child_required_observation_ids
    assert {
        item.observation_ref.observation_id
        for item in parent_population.members
    } <= {
        item.observation_ref.observation_id
        for item in child_population.members
    }


def test_refinement_falsification_reuses_run_wide_cost_ledger(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    client = _NarrowThenSupportClient(
        category="falsification-shared-admission-ledger",
        rule="Contributors preserve the observed fixture convention.",
    )
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(
            semantic={
                # The parent consumes the sole complete falsification job.
                # Its child is independently planned, then must abstain before
                # any second provider call under the shared run-wide ledger.
                "max_falsification_items_per_request": 10_000,
                "max_falsification_calls_per_hypothesis": 1,
                "max_total_falsification_calls": 1,
            }
        ),
        client=client,
    )

    assert result.current_source is not None
    current = result.current_source
    parent = next(
        item for item in current.hypotheses if item.parent_hypothesis_id is None
    )
    child = next(
        item for item in current.hypotheses if item.parent_hypothesis_id is not None
    )
    assert child.parent_hypothesis_id == parent.hypothesis_id
    assert len(client.falsification_requests) == 1
    assert client.falsification_requests[0]["hypothesis"]["hypothesis_id"] == (
        parent.hypothesis_id
    )
    parent_result = next(
        item for item in current.falsifications if item.hypothesis_id == parent.hypothesis_id
    )
    child_result = next(
        item for item in current.falsifications if item.hypothesis_id == child.hypothesis_id
    )
    assert parent_result.status == "complete"
    assert child_result.status == "partial"
    assert child_result.inference_ids == []
    assert child_result.admission_estimate is not None
    assert child_result.admission_estimate.reason_code == (
        "max_total_falsification_calls"
    )
    assert child_result.execution.request_plan_complete is False
    assert child_result.execution.planned_calls == 0
    assert current.index.status == "partial"
    assert current.index.resource_use["falsification_admitted_hypotheses"] == 1
    assert current.index.resource_use[
        "falsification_cost_not_admitted_hypotheses"
    ] == 1
    assert current.index.resource_use["falsification_admitted_calls"] == 1
    assert current.index.resource_use[
        "falsification_cost_not_admitted_calls"
    ] == 1
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


def test_global_discovery_gap_does_not_block_candidate_local_refinement(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        config=config_for_tests(
            semantic={"required": False, "max_units": 2}
        ),
        client=_NarrowThenSupportClient(
            category="partial-refinement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.index.status == "partial"
    assert len(result.current_source.refinements) == 1
    assert len(result.current_source.falsifications) == 2
    assert all(
        item.status == "complete"
        for item in result.current_source.falsifications
    )
    coverage = result.memory.coverage.current_source.falsification
    assert (
        coverage.planned_hypotheses,
        coverage.completed,
        coverage.refined,
        coverage.rejected,
        coverage.uncertain,
        coverage.failed,
    ) == (2, 1, 1, 0, 0, 0)
    # The channel-local semantic stage is optional, but schema 4 still exposes
    # its partial denominator as a required unified current-evidence gap.
    # Independently accepted claims remain usable under the default
    # integrity validator; only the all-capabilities completeness gate fails.
    assert not result.build_report["complete"]
    assert result.build_report["unified"]["completeness"][
        "package_status"
    ] == "partial"
    assert validate_run_directory(run, repository_root=repository) == []


def test_refinement_population_budget_exhaustion_is_terminal_and_visible(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    baseline = build_fixture(repository, tmp_path / "baseline")
    assert baseline.current_source is not None
    member_count = len(baseline.current_source.populations[0].members)
    config = config_for_tests(
        semantic={
                "max_population_members": member_count,
                "max_population_members_per_hypothesis": member_count,
                "max_total_population_members": member_count,
                "max_classified_population_members_per_hypothesis": member_count,
                "max_total_classified_population_members": member_count,
            }
    )
    run = tmp_path / "limited"
    result = build_fixture(
        repository,
        run,
        config=config,
        client=_NarrowThenSupportClient(
            category="bounded-refinement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.index.status == "partial"
    assert result.current_source.index.failed_populations == 1
    assert "refinement_population_budget_exhausted" in (
        result.current_source.index.incomplete_reasons
    )
    assert result.current_source.review_items[0].epistemic_status == "hypothesis"
    assert not result.build_report["complete"]
    assert validate_run_directory(run, repository_root=repository) == []
    assert validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )


class _DisagreeingFalsifierClient(ParameterizedSemanticClient):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.verifier_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        value = json.loads(response.content)
        if self.verifier_calls % 2:
            value["verdict"] = "reject_insufficient"
            value["rationale_summary"] = "This batch contains a material objection."
        self.verifier_calls += 1
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="disagreeing-falsifier-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_multi_batch_material_disagreement_is_partial_and_uncertain(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config_for_tests(
            semantic={"max_falsification_items_per_request": 1}
        ),
        client=_DisagreeingFalsifierClient(
            category="verifier-disagreement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    falsification = result.current_source.falsifications[0]
    assert len(falsification.inference_ids) > 1
    assert falsification.status == "partial"
    assert falsification.verdict == "uncertain"
    assert "falsification_material_disagreement" in (
        falsification.incomplete_reasons
    )
    assert result.current_source.review_items[0].epistemic_status == "hypothesis"


class _NoOpRefinementClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        value = json.loads(response.content)
        value["verdict"] = "narrow_scope"
        value["proposed_applies_to"] = request["hypothesis"]["applies_to"]
        value["rationale_summary"] = "The proposed change is intentionally a no-op."
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="no-op-refinement-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_no_op_refinement_is_rejected_with_terminal_reason(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        client=_NoOpRefinementClient(
            category="no-op-refinement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert not result.current_source.refinements
    assert result.current_source.index.status == "partial"
    review = result.current_source.review_items[0]
    assert review.epistemic_status == "rejected"
    assert review.reason_codes[0].startswith("invalid_refinement:")


class _TwoRoundRefinementClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "counterexample_verify":
            return response
        value = json.loads(response.content)
        scope = request["hypothesis"]["applies_to"]
        proposed = (
            ["src/**"]
            if scope == ["**/*.py"]
            else (
                ["src/aurora/**"]
                if scope == ["src/**"]
                else ["src/aurora/api.py"]
            )
        )
        value["verdict"] = "narrow_scope"
        value["proposed_applies_to"] = proposed
        value["rationale_summary"] = "The evidence supports another bounded scope."
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="two-round-refinement-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_two_refinement_rounds_reuse_each_frozen_child_population_once(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config_for_tests(semantic={"max_refinement_rounds": 2}),
        client=_TwoRoundRefinementClient(
            category="two-round-refinement-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    assert result.current_source is not None
    assert len(result.current_source.hypotheses) == 3
    assert len(result.current_source.populations) == 3
    assert len(result.current_source.falsifications) == 3
    assert len(result.current_source.refinements) == 2
    assert sorted(
        item.refinement_round for item in result.current_source.hypotheses
    ) == [
        0,
        1,
        2,
    ]
    assert result.current_source.review_items[0].reason_codes == [
        "refinement_round_limit_reached"
    ]
