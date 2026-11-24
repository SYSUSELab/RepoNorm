from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import (
    ParameterizedSemanticClient,
    build_fixture,
    config_for_tests,
    copied_current_source_fixture,
)
from repo_constraint_memory_v3.exporters import stable_json_text
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.observations.canonical import (
    payload_digest,
    sha256_hex,
)
from repo_constraint_memory_v3.run_validation import validate_run_directory


class _ZeroHypothesisClient(ParameterizedSemanticClient):
    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = super().complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") != "current_hypothesis_reduce":
            return response
        value = json.loads(response.content)
        value["outcome"] = "no_hypothesis"
        value["hypotheses"] = []
        value["reason_code"] = "complete_no_hypothesis"
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="zero-hypothesis-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class _OneRefinementClient(ParameterizedSemanticClient):
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
        if hypothesis_id != self.parent_hypothesis_id:
            return response
        value = json.loads(response.content)
        value["verdict"] = "narrow_scope"
        value["proposed_applies_to"] = ["src/aurora/**"]
        value["rationale_summary"] = "The evidence supports a narrower scope."
        return ProviderResponse(
            content=json.dumps(value, sort_keys=True, separators=(",", ":")),
            model="one-refinement-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_current_source_candidate_can_enter_canonical_delivery_views(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    semantic_run = tmp_path / "semantic"
    semantic = build_fixture(repository, semantic_run)
    explicit_run = tmp_path / "explicit"
    explicit = build_fixture(
        repository,
        explicit_run,
        config=config_for_tests(
            observation={"enabled": False, "queries": []}
        ),
    )
    assert semantic.current_source is not None
    # The channel-local experimental disposition remains audit-only.  The
    # schema-4 unified policy independently admits a sufficiently grounded
    # current-source convention as accepted + observed.
    assert semantic.current_source.review_items
    semantic_ids = {
        item.claim_id
        for item in semantic.unified.bundle.canonical_claims
        if item.epistemic_status == "accepted"
    }
    explicit_ids = {
        item.claim_id
        for item in explicit.unified.bundle.canonical_claims
        if item.epistemic_status == "accepted"
    }
    current_ids = semantic_ids - explicit_ids
    assert len(current_ids) == 1
    current_claim = next(
        item
        for item in semantic.unified.bundle.canonical_claims
        if item.claim_id in current_ids
    )
    assert current_claim.epistemic_status == "accepted"
    assert current_claim.normative_strength == "observed"
    assert current_claim.delivery_policy != "reference_only"
    assert any(
        current_claim.claim_id in path.read_text(encoding="utf-8")
        for path in (
            semantic_run / "consumers" / "accepted_only" / "by-path"
        ).rglob("constraints.md")
    )
    assert validate_run_directory(semantic_run, repository_root=repository) == []


def test_complete_zero_hypothesis_is_not_an_incomplete_or_accepted_result(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    result = build_fixture(
        repository,
        run,
        client=_ZeroHypothesisClient(
            category="unused-zero-result",
            rule="This rule must not be materialized.",
        ),
    )
    assert result.current_source is not None
    assert result.current_source.index.status == "complete"
    assert result.current_source.index.unique_candidate_roots == 0
    assert not result.current_source.hypotheses
    assert not result.current_source.review_items
    assert result.traceability.current_source.root_dispositions is not None
    assert result.traceability.current_source.root_dispositions.total == 0
    assert validate_run_directory(run, repository_root=repository) == []


def test_orphan_current_source_directory_is_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    (run / "current-source" / "history").mkdir()
    assert any(
        "unexpected current-source directories" in item
        for item in validate_run_directory(run, repository_root=repository)
    )


def _rewrite_with_digest(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value["artifact_digest"] = payload_digest(
        {key: item for key, item in value.items() if key != "artifact_digest"}
    )
    path.write_text(stable_json_text(value), encoding="utf-8")


@pytest.mark.parametrize("artifact", ["hypothesis", "population", "falsification"])
def test_semantic_artifact_tamper_is_detected_after_local_digest_repair(
    tmp_path: Path,
    artifact: str,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / artifact
    build_fixture(repository, run)
    root = run / "current-source"
    if artifact == "hypothesis":
        path = next((root / "hypotheses").glob("HYP-*.json"))
        _rewrite_with_digest(
            path,
            lambda value: value.__setitem__(
                "rule", value["rule"] + " Coordinated tamper."
            ),
        )
    elif artifact == "population":
        path = next((root / "populations").glob("POP-*.json"))

        def mutate_population(value):
            member = value["members"][0]
            member["reason_summary"] += " Coordinated tamper."
            member["artifact_digest"] = payload_digest(
                {
                    key: item
                    for key, item in member.items()
                    if key != "artifact_digest"
                }
            )

        _rewrite_with_digest(path, mutate_population)
    else:
        path = next((root / "falsifications").glob("FALSIFY-*.json"))
        _rewrite_with_digest(
            path,
            lambda value: value.__setitem__(
                "rationale_summary",
                value["rationale_summary"] + " Coordinated tamper.",
            ),
        )
    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "current-source" in item and (
            "differs from recorded replay" in item
            or "reference mismatch" in item
        )
        for item in errors
    )


def test_refinement_lineage_tamper_is_detected_after_digest_repair(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(
        repository,
        run,
        client=_OneRefinementClient(
            category="refinement-tamper-fixture",
            rule="Contributors preserve the observed fixture convention.",
        ),
    )
    path = next(
        (run / "current-source" / "refinements").glob("REFINE-*.json")
    )

    def mutate(value):
        value["carried_member_dispositions"][0]["reason_code"] = (
            "coordinated_tamper"
        )

    _rewrite_with_digest(path, mutate)
    assert any(
        "current-source refinements differs from recorded replay" in item
        for item in validate_run_directory(run, repository_root=repository)
    )


def test_experimental_policy_tamper_is_detected_after_digest_repair(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    path = run / "current-source" / "policy-decisions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    decision = value["decisions"][0]
    decision["experimental_disposition"] = "rejected"
    decision["reason_codes"] = ["coordinated_tamper"]
    decision["artifact_digest"] = payload_digest(
        {
            key: item
            for key, item in decision.items()
            if key != "artifact_digest"
        }
    )
    path.write_text(stable_json_text(value), encoding="utf-8")
    assert any(
        "current-source policy decisions differs from recorded replay" in item
        for item in validate_run_directory(run, repository_root=repository)
    )


def test_full_validator_rejects_coordinated_current_method_tamper(
    tmp_path: Path,
) -> None:
    repository = copied_current_source_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    request_path = next(
        path
        for path in (run / "inference" / "requests").glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["stage"]
        == "current_semantic_map"
    )
    record = json.loads(request_path.read_text(encoding="utf-8"))
    record["system_prompt_digest"] = "0" * 64
    record["record_digest"] = payload_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )
    request_text = stable_json_text(record)
    request_path.write_text(request_text, encoding="utf-8")

    index_path = run / "inference" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    reference = next(
        item
        for item in index["requests"]
        if item["request_id"] == record["request_id"]
    )
    reference["artifact_sha256"] = sha256_hex(request_text.encode("utf-8"))
    reference["record_digest"] = record["record_digest"]
    index["inference_set_digest"] = payload_digest(
        {
            key: value
            for key, value in index.items()
            if key != "inference_set_digest"
        }
    )
    index_path.write_text(stable_json_text(index), encoding="utf-8")

    assert any(
        "current-stage system prompt differs" in item
        for item in validate_run_directory(run, repository_root=repository)
    )
