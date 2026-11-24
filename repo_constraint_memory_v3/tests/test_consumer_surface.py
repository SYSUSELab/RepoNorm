from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from repo_constraint_memory_v3.unified import consumer as consumer_module
from repo_constraint_memory_v3.unified.consumer import (
    MAX_CATALOG_FILE_BYTES,
    MAX_RENDERED_COMPACT_CARD_BYTES,
    ConsumerManifest,
    _compare_consumer_directory,
    build_consumer_artifacts,
    consumer_tree_digest,
    make_canonical_claims_file,
    make_canonical_completeness_file,
    path_anchor,
    path_view_relative,
    project_claims,
    render_claim_card,
    render_consumer_view,
)
from repo_constraint_memory_v3.unified.ids import (
    content_digest,
    make_canonical_claim_id,
    make_case_id,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
    make_member_id,
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateSemantics,
    CanonicalClaim,
    CanonicalEvidenceRef,
    CapabilityDependency,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    PopulationDenominator,
    UnifiedBundle,
    UnifiedCompleteness,
    UnifiedIndex,
)


SNAPSHOT = "sha256:" + "1" * 64
PROFILE = "2" * 64
COMMIT = "3" * 40


def _pair(
    token: str,
    *,
    status: str = "accepted",
    delivery: str = "default_consider",
    scopes: tuple[str, ...] = ("src/pkg.py",),
    rule: str | None = None,
    condition: str | None = None,
    exception: str | None = None,
    risk: str = "bounded_check",
    facets: tuple[str, ...] = (),
    evidence: tuple[tuple[str, str, str, str], ...] | None = None,
    population: tuple[int, int, int, int] | None = None,
    dependency_status: str | None = None,
) -> tuple[CanonicalClaim, CurrentEvidenceBundle]:
    assessment_id = "ASSESS-" + content_digest({"token": token})[:24]
    member_id = make_member_id(SNAPSHOT, assessment_id)
    case_id = make_case_id(SNAPSHOT, [member_id])
    semantics = CandidateSemantics(
        rule=rule or f"Rule for {token}.",
        normalized_action=f"normalized action for {token}",
        polarity="positive",
        applies_to=sorted(scopes),
        condition=condition,
        exception=exception,
    )
    evidence = evidence or (
        (
            "counterevidence" if status == "rejected" else "support",
            "current_observation" if status == "rejected" else "explicit",
            f"docs/{token}.md",
            f"Evidence for {token}.",
        ),
    )
    origins: list[EvidenceOrigin] = []
    items: list[CurrentEvidenceItem] = []
    support: list[CanonicalEvidenceRef] = []
    counter: list[CanonicalEvidenceRef] = []
    for ordinal, (role, kind, path, description) in enumerate(evidence, start=1):
        origin_kind = "explicit_span" if kind == "explicit" else "population_member"
        source_digest = content_digest({"path": path, "description": description})
        origin_id = make_evidence_origin_id(
            SNAPSHOT,
            origin_kind,
            path,
            [ordinal, ordinal],
            source_digest,
        )
        artifact_ref = ImmutableArtifactRef(
            kind="explicit_evidence" if kind == "explicit" else "population_member",
            stable_id=f"SOURCE-{content_digest([token, ordinal])[:24]}",
            snapshot_digest=SNAPSHOT,
            artifact_path=f"canonical/audit/source/{token}-{ordinal}.json",
            artifact_digest=content_digest({"source": token, "ordinal": ordinal}),
        )
        evidence_id = make_evidence_id(
            origin_id,
            role,
            [member_id],
            artifact_ref.stable_id,
        )
        origins.append(
            EvidenceOrigin(
                origin_id=origin_id,
                origin_kind=origin_kind,
                path=path,
                line_start=ordinal,
                line_end=ordinal,
                source_content_digest=source_digest,
                independence_group=f"file:{path}",
            )
        )
        items.append(
            CurrentEvidenceItem(
                evidence_id=evidence_id,
                origin_id=origin_id,
                role=role,
                member_ids=[member_id],
                artifact_ref=artifact_ref,
                excerpt_digest=(
                    hashlib.sha256(description.encode("utf-8")).hexdigest()
                    if kind == "explicit"
                    else content_digest({"fact": description})
                ),
                description=description,
            )
        )
        canonical = CanonicalEvidenceRef(
            evidence_id=evidence_id,
            origin_id=origin_id,
            role=role,
            kind=kind,
            path=path,
            line_start=ordinal,
            line_end=ordinal,
            artifact_ref=artifact_ref,
        )
        (support if role == "support" else counter).append(canonical)
    claim_payload = {
        "claim_id": make_canonical_claim_id(
            SNAPSHOT, semantics.model_dump(mode="json")
        ),
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "case_id": case_id,
        "semantics": semantics,
        "epistemic_status": status,
        "claim_kind": "normative_declaration",
        "normative_strength": "required",
        "delivery_policy": delivery if status == "accepted" else "not_applicable",
        "execution_risk": risk,
        "support": sorted(support, key=lambda item: item.evidence_id),
        "counterevidence": sorted(counter, key=lambda item: item.evidence_id),
        "conflicts_with": [],
        "contributing_member_ids": [member_id],
        "refinement_parent_id": None,
        "reason_codes": [f"fixture_{status}"],
        "review_facets": sorted(facets) if status == "review" else [],
        "rejection_kind": (
            "current_snapshot_contradiction" if status == "rejected" else None
        ),
        "rejection_basis_refs": (
            [item.evidence_id for item in counter] if status == "rejected" else []
        ),
        "dependency_status": dependency_status
        or ("incomplete" if status == "review" else "complete"),
    }
    claim = CanonicalClaim.model_validate(seal_payload(claim_payload))
    denominators = []
    if population is not None:
        support_count, counter_count, unknown, ineligible = population
        denominators.append(
            PopulationDenominator(
                member_id=member_id,
                population_id=f"POP-{token}",
                status="complete",
                census_enumeration_status="complete",
                sampling_plan_status="complete",
                sampling_mode="census",
                scanned=support_count + counter_count + unknown + ineligible,
                serialized=support_count + counter_count + unknown + ineligible,
                matched=support_count + counter_count + unknown + ineligible,
                inspected=support_count + counter_count + unknown + ineligible,
                uninspected=0,
                support=support_count,
                counterexample=counter_count,
                unknown=unknown,
                ineligible=ineligible,
            )
        )
    current_payload = {
        "bundle_id": make_current_evidence_bundle_id(SNAPSHOT, PROFILE, case_id),
        "case_id": case_id,
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "origins": sorted(origins, key=lambda item: item.origin_id),
        "items": sorted(items, key=lambda item: item.evidence_id),
        "denominators": denominators,
        "population_refs": [],
        "falsification_refs": [],
        "scope_condition_refs": [],
        "dependencies": [
            CapabilityDependency(
                capability="current_evidence",
                requirement_mode="required",
                status="partial" if status == "review" else "complete",
                reason_codes=["fixture_gap"] if status == "review" else [],
            )
        ],
        "integrity": "valid",
        "status": "partial" if status == "review" else "complete",
        "incomplete_reasons": ["fixture_gap"] if status == "review" else [],
    }
    current = CurrentEvidenceBundle.model_validate(seal_payload(current_payload))
    return claim, current


def _bundle(
    pairs: list[tuple[CanonicalClaim, CurrentEvidenceBundle]],
    *,
    partial: bool = False,
) -> UnifiedBundle:
    claims = sorted((item[0] for item in pairs), key=lambda item: item.claim_id)
    current = sorted((item[1] for item in pairs), key=lambda item: item.bundle_id)
    completeness = UnifiedCompleteness(
        structurally_valid=True,
        package_status="partial" if partial else "complete",
        capabilities=[
            CapabilityDependency(
                capability="unified_export",
                requirement_mode="required",
                status="partial" if partial else "complete",
                reason_codes=["fixture_gap"] if partial else [],
            )
        ],
        omitted_possible=partial,
        reasons=["fixture_gap"] if partial else [],
    )
    index = UnifiedIndex.model_construct(
        snapshot_digest=SNAPSHOT,
        snapshot_commit=COMMIT,
        profile_digest=PROFILE,
    )
    return UnifiedBundle.model_construct(
        index=index,
        current_evidence=current,
        findings=[],
        canonical_claims=claims,
        completeness=completeness,
    )


def _replace_claim(claim: CanonicalClaim, **updates: object) -> CanonicalClaim:
    payload = claim.model_dump(mode="python", exclude={"artifact_digest"})
    payload.update(updates)
    return CanonicalClaim.model_validate(seal_payload(payload))


def _render(
    pairs: list[tuple[CanonicalClaim, CurrentEvidenceBundle]],
    *,
    profile: str = "accepted_only",
    partial: bool = False,
):
    bundle = _bundle(pairs, partial=partial)
    return render_consumer_view(
        bundle=bundle,
        repository_name="configured-repository-name",
        profile=profile,
        canonical=make_canonical_claims_file(bundle),
        completeness=make_canonical_completeness_file(bundle),
        git_artifacts=None,
    )


def _materialize(root: Path, plan) -> None:
    for relative in plan.directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative, data in plan.files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_minimized_start_conditional_cards_and_manifest_binding() -> None:
    accepted = _pair(
        "accepted",
        delivery="reference_only",
        scopes=("src/**/*.py",),
        exception="except generated modules",
        risk="repository_wide_mutation",
    )
    review = _pair(
        "review",
        status="review",
        delivery="not_applicable",
        scopes=("tests/**/*.py",),
        facets=("factual_uncertainty", "technical_or_coverage_gap"),
    )
    plan = _render([accepted, review], profile="accepted_plus_review", partial=True)
    start = plan.files["START.md"].decode("utf-8")
    catalog = plan.files["catalog.md"].decode("utf-8")

    assert "may omit repository conventions" in start
    assert "Unconfirmed (review)" in start
    for noise in ("Snapshot", "profile", "validation", "accepted /", "reference_only"):
        assert noise not in start
    assert "Snapshot digest" not in start
    assert "profile digest" not in start.lower()
    assert "Canonical accepted" not in start
    assert accepted[0].semantics.rule not in start

    assert catalog.count("## Rule") == 2
    assert accepted[0].semantics.rule in catalog
    assert review[0].semantics.rule in catalog
    assert "Unconfirmed (review)" in catalog
    assert "- **Use:** `reference_only`" in catalog
    assert "- **Execution risk:** `repository_wide_mutation`" in catalog
    assert "- **Proposed normative strength:** `required`" in catalog
    assert "Read applicability and details" in catalog
    detail = plan.files[f"claims/{accepted[0].claim_id}.md"].decode()
    assert "- **Exception:** except generated modules" in detail
    assert "Delivery policy" not in catalog
    assert "not_applicable" not in catalog
    clean_start = _render([accepted]).files["START.md"].decode()
    assert "review" not in clean_start
    assert "may omit" not in clean_start

    manifest = ConsumerManifest.model_validate_json(
        plan.files["consumer-manifest.json"]
    )
    assert manifest.consumer_profile == "accepted_plus_review"
    assert manifest.catalog_card_count == 2
    assert manifest.task_tree_digest == consumer_tree_digest(plan.files)
    assert manifest.markdown_file_count == len(
        [item for item in plan.files if item.endswith(".md")]
    )


def test_exact_a_r_d_profile_sets_and_no_status_leakage() -> None:
    default = _pair("projection-default", scopes=("src/default.py",))
    conditional = _pair(
        "projection-condition",
        delivery="condition_matched_only",
        scopes=("tests/**/*.py",),
        condition="when changing tests",
    )
    reference = _pair(
        "projection-reference",
        delivery="reference_only",
        scopes=("docs/**/*.md",),
    )
    review = _pair(
        "projection-review",
        status="review",
        delivery="not_applicable",
        scopes=("src/review.py",),
        facets=("factual_uncertainty",),
    )
    rejected = _pair(
        "projection-rejected",
        status="rejected",
        delivery="not_applicable",
        scopes=("src/rejected.py",),
    )
    claims = [item[0] for item in (default, conditional, reference, review, rejected)]
    accepted_ids = {
        default[0].claim_id,
        conditional[0].claim_id,
        reference[0].claim_id,
    }
    delivered_ids = {default[0].claim_id, conditional[0].claim_id}

    accepted_only = project_claims(claims, "accepted_only")
    assert {item.claim_id for item in accepted_only.catalog_claims} == accepted_ids
    assert {item.claim_id for item in accepted_only.detail_claims} == accepted_ids
    assert {item.claim_id for item in accepted_only.path_claims} == delivered_ids

    plus_review = project_claims(claims, "accepted_plus_review")
    assert {item.claim_id for item in plus_review.catalog_claims} == {
        *accepted_ids,
        review[0].claim_id,
    }
    assert {item.claim_id for item in plus_review.path_claims} == {
        *delivered_ids,
        review[0].claim_id,
    }
    accepted_plan = _render(
        [default, conditional, reference, review, rejected],
        profile="accepted_only",
    )
    task_text = "\n".join(
        data.decode("utf-8")
        for relative, data in accepted_plan.files.items()
        if relative.endswith(".md")
    )
    assert review[0].claim_id not in task_text
    assert review[0].semantics.rule not in task_text
    assert rejected[0].claim_id not in task_text
    assert rejected[0].semantics.rule not in task_text
    assert reference[0].claim_id in accepted_plan.files["catalog.md"].decode()
    assert all(
        reference[0].claim_id not in data.decode("utf-8")
        for relative, data in accepted_plan.files.items()
        if relative.startswith("by-path/") and relative.endswith("constraints.md")
    )


def test_default_build_preflights_unrequested_review_projection() -> None:
    accepted = _pair("preflight-visible")
    unsafe_review = _pair(
        "preflight-hidden-review",
        status="review",
        delivery="not_applicable",
        rule="Unsafe review text contains a control: \x00",
        facets=("factual_uncertainty",),
    )
    bundle = _bundle([accepted, unsafe_review])
    canonical = make_canonical_claims_file(bundle)
    completeness = make_canonical_completeness_file(bundle)

    default_plan = render_consumer_view(
        bundle=bundle,
        repository_name="configured-repository-name",
        profile="accepted_only",
        canonical=canonical,
        completeness=completeness,
        git_artifacts=None,
    )
    assert unsafe_review[0].claim_id not in default_plan.files["catalog.md"].decode(
        "utf-8"
    )

    with pytest.raises(ValueError, match="forbidden control"):
        build_consumer_artifacts(
            bundle=bundle,
            repository_name="configured-repository-name",
            profiles=("accepted_only",),
            git_artifacts=None,
        )


def test_card_field_order_predicates_and_lossless_dynamic_fences() -> None:
    ordinary = _pair(
        "ordinary-card",
        condition=None,
        exception=None,
        risk="scoped_mutation",
    )[0]
    card = render_claim_card(ordinary, containing_relative="catalog.md")
    labels = [
        "**Rule:**",
        "**Applies to:**",
        "**Normative strength:**",
        "**Detail:**",
    ]
    positions = [card.index(item) for item in labels]
    assert positions == sorted(positions)
    assert "**Condition:**" not in card
    assert "**Status:**" not in card
    assert "Uncertainty" not in card
    assert "Exception" not in card
    assert "Use" not in card
    assert "Execution risk" not in card

    value = "Use **Unicode 规则**.\nKeep embedded ``` fences and `ticks`."
    rich = _pair(
        "rich-card",
        rule=value,
        condition="Use &copy; literally",
        exception="Never render ~~removed~~",
        delivery="reference_only",
        risk="external_side_effect",
    )[0]
    rich_card = render_claim_card(
        rich,
        containing_relative="by-path/docs/guide/constraints.md",
    )
    assert "````text" in rich_card
    assert "  Use **Unicode 规则**." in rich_card
    assert "  Keep embedded ``` fences and `ticks`." in rich_card
    assert "Use &copy; literally" in rich_card
    assert "Never render ~~removed~~" in rich_card
    assert "- **Use:** `reference_only`" in rich_card
    assert "- **Execution risk:** `external_side_effect`" in rich_card

    review_pair = _pair(
        "facet-order",
        status="review",
        delivery="not_applicable",
        facets=(
            "technical_or_coverage_gap",
            "factual_uncertainty",
            "negative_basis_insufficient",
            "conflict_or_currentness",
            "refinement_or_qualifier_needed",
            "channel_disagreement",
        ),
    )
    review = review_pair[0]
    review_card = render_claim_card(review, containing_relative="catalog.md")
    assert review.review_facets == sorted(review.review_facets)
    facet_positions = [
        review_card.index(consumer_module._REVIEW_EXPLANATIONS[facet]) for facet in review.review_facets
    ]
    assert facet_positions == sorted(facet_positions)
    assert all(facet not in review_card for facet in review.review_facets)
    assert "**Proposed normative strength:**" in review_card
    assert "not_applicable" not in review_card

    review_detail = _render(
        [review_pair],
        profile="accepted_plus_review",
    ).files[f"claims/{review.claim_id}.md"].decode("utf-8")
    explanations = [
        consumer_module._REVIEW_EXPLANATIONS[facet]
        for facet in review.review_facets
    ]
    explanation_positions = [review_detail.index(item) for item in explanations]
    assert explanation_positions == sorted(explanation_positions)


def test_review_directory_marker_is_navigation_only_and_accepted_rejects_it() -> None:
    review = _pair(
        "directory-review",
        status="review",
        delivery="not_applicable",
        scopes=("tests/",),
        facets=("refinement_or_qualifier_needed",),
        dependency_status="complete",
    )
    plan = _render([review], profile="accepted_plus_review")
    relative = "by-path/tests/constraints.md"
    assert relative in plan.files
    text = plan.files[relative].decode("utf-8")
    assert "`tests/`" in text
    assert "tests/**" not in text
    with pytest.raises(ValueError, match="directory-marker scope"):
        _pair("directory-accepted", scopes=("tests/",))


def test_multi_scope_root_and_per_view_dedup_with_zero_anchor_case() -> None:
    routed = _pair(
        "multi-anchor",
        scopes=(
            "**",
            "docs/*.md",
            "src/**/*.py",
            "src/*.py",
            "src/pkg/tool.py",
        ),
    )
    plan = _render([routed])
    expected = {
        "by-path/constraints.md",
        "by-path/docs/constraints.md",
        "by-path/src/constraints.md",
        "by-path/src/pkg/tool.py/constraints.md",
    }
    actual = {
        relative
        for relative in plan.files
        if relative.startswith("by-path/") and relative.endswith("constraints.md")
    }
    assert actual == expected
    assert plan.summary.path_card_occurrence_count == 4
    assert plan.summary.path_unique_claim_count == 1
    assert plan.files["by-path/src/constraints.md"].decode().count(
        routed[0].claim_id
    ) == 1

    reference = _pair(
        "zero-path-view",
        delivery="reference_only",
        scopes=("src/reference.py",),
    )
    zero = _render([reference])
    assert zero.summary.path_view_count == 0
    assert not any(
        relative.endswith("constraints.md") for relative in zero.files
    )
    assert zero.files["by-path/index.md"] == b"# Norm4Code Path View Index\n"


def test_hidden_conflicts_never_bypass_profile_isolation() -> None:
    accepted = _pair("visible-conflict")
    review = _pair(
        "hidden-conflict",
        status="review",
        delivery="not_applicable",
        facets=("conflict_or_currentness",),
    )
    accepted_claim = _replace_claim(
        accepted[0],
        conflicts_with=[review[0].claim_id],
    )
    accepted_plan = _render([(accepted_claim, accepted[1]), review])
    detail = accepted_plan.files[f"claims/{accepted_claim.claim_id}.md"].decode()
    assert "## Material conflicts" not in detail
    assert review[0].claim_id not in detail
    assert review[0].semantics.rule not in detail

    plus = _render(
        [(accepted_claim, accepted[1]), review],
        profile="accepted_plus_review",
    )
    plus_detail = plus.files[f"claims/{accepted_claim.claim_id}.md"].decode()
    assert "## Material conflicts" in plus_detail
    assert review[0].claim_id in plus_detail
    assert "Rule for hidden-conflict." in plus_detail


def test_detail_groups_every_evidence_without_machine_envelope_leakage() -> None:
    pair = _pair(
        "grouping",
        evidence=(
            ("support", "current_observation", "src/a.py", "Shared deterministic fact."),
            ("support", "current_observation", "src/b.py", "Shared deterministic fact."),
            ("counterevidence", "current_observation", "src/c.py", "Exact counter fact."),
        ),
        population=(2, 1, 0, 1),
    )
    plan = _render([pair])
    detail = plan.files[f"claims/{pair[0].claim_id}.md"].decode("utf-8")

    assert detail.count("### Evidence ") == 1
    assert "Evidence items" not in detail
    assert "`src/a.py:1`" in detail
    assert "`src/b.py:2`" in detail
    assert "Population support" not in detail
    assert detail.count("### Counterevidence ") == 1
    assert "Exact counter fact." in detail
    for forbidden in (
        "EVIDENCE-",
        "EVORIGIN-",
        "SOURCE-",
        "artifact_digest",
        "Content form",
        "Audit identity",
        "fixture_accepted",
    ):
        assert forbidden not in detail

    ambiguous = _pair(
        "ambiguous-denominator",
        evidence=(
            ("support", "current_observation", "src/one.py", "First fact."),
            ("support", "current_observation", "src/two.py", "Second fact."),
        ),
        population=(2, 1, 0, 0),
    )
    ambiguous_detail = _render([ambiguous]).files[
        f"claims/{ambiguous[0].claim_id}.md"
    ].decode("utf-8")
    assert ambiguous_detail.count("### Evidence ") == 2
    assert "Population support" not in ambiguous_detail


def test_case_local_evidence_resolution_does_not_shadow_duplicate_ids() -> None:
    first = _pair(
        "case-local-first",
        evidence=(("support", "current_observation", "src/shared.py", "First case fact."),),
    )
    second = _pair(
        "case-local-second",
        evidence=(("support", "current_observation", "src/second.py", "Second original fact."),),
    )
    shared_member = first[1].items[0].member_ids[0]
    second_member = second[1].items[0].member_ids[0]
    second_case_id = make_case_id(
        SNAPSHOT,
        sorted([shared_member, second_member]),
    )
    case_local_item = CurrentEvidenceItem.model_validate(
        {
            **first[1].items[0].model_dump(mode="python"),
            "description": "Fact resolved only inside the second case.",
        }
    )
    current_payload = first[1].model_dump(
        mode="python",
        exclude={"artifact_digest"},
    )
    current_payload.update(
        {
            "bundle_id": make_current_evidence_bundle_id(
                SNAPSHOT,
                PROFILE,
                second_case_id,
            ),
            "case_id": second_case_id,
            "items": [case_local_item],
        }
    )
    second_current = CurrentEvidenceBundle.model_validate(
        seal_payload(current_payload)
    )
    second_claim = _replace_claim(
        second[0],
        case_id=second_case_id,
        support=first[0].support,
        contributing_member_ids=[shared_member],
    )
    assert first[1].items[0].evidence_id == second_current.items[0].evidence_id

    plan = _render([first, (second_claim, second_current)])
    first_detail = plan.files[f"claims/{first[0].claim_id}.md"].decode()
    second_detail = plan.files[f"claims/{second_claim.claim_id}.md"].decode()
    assert "First case fact." in first_detail
    assert "Fact resolved only inside the second case." not in first_detail
    assert "Fact resolved only inside the second case." in second_detail
    assert "First case fact." not in second_detail


def test_missing_evidence_binding_and_explicit_digest_tamper_fail_closed() -> None:
    pair = _pair(
        "binding",
        evidence=(("support", "explicit", "docs/binding.md", "Exact excerpt."),),
    )
    missing_ref = CanonicalEvidenceRef.model_validate(
        {
            **pair[0].support[0].model_dump(mode="python"),
            "evidence_id": "EVIDENCE-" + "f" * 24,
        }
    )
    missing_claim = _replace_claim(pair[0], support=[missing_ref])
    with pytest.raises(ValueError, match="does not resolve case-locally"):
        _render([(missing_claim, pair[1])])

    current_payload = pair[1].model_dump(
        mode="python",
        exclude={"artifact_digest"},
    )
    current_payload["items"][0]["excerpt_digest"] = "f" * 64
    tampered_current = CurrentEvidenceBundle.model_validate(
        seal_payload(current_payload)
    )
    with pytest.raises(ValueError, match="explicit evidence excerpt digest differs"):
        _render([(pair[0], tampered_current)])


def test_explicit_long_excerpt_is_lossless_and_nonexplicit_limit_fails_closed() -> None:
    long_excerpt = "证据 `literal` " + "x" * 6_800
    explicit = _pair(
        "long-explicit",
        evidence=(("support", "explicit", "docs/long.md", long_excerpt),),
    )
    detail = _render([explicit]).files[
        f"claims/{explicit[0].claim_id}.md"
    ].decode("utf-8")
    assert long_excerpt in detail
    assert "..." not in detail

    with pytest.raises(ValueError, match="non-explicit current evidence description"):
        _pair(
            "long-observation",
            evidence=(
                (
                    "support",
                    "current_observation",
                    "src/long.py",
                    "y" * 601,
                ),
            ),
        )


def test_catalog_greedy_sharding_and_compact_card_preflight() -> None:
    pairs = [
        _pair(
            f"shard-{ordinal:03d}",
            rule=(f"Rule {ordinal:03d}: " + "x" * 470),
            scopes=(f"src/pkg/file_{ordinal:03d}.py",),
        )
        for ordinal in range(70)
    ]
    plan = _render(pairs)
    shards = sorted(
        relative
        for relative in plan.files
        if relative.startswith("catalog/") and relative.endswith(".md")
    )
    assert len(shards) >= 2
    assert plan.summary.catalog_shard_count == len(shards)
    assert len(plan.files["catalog.md"]) <= MAX_CATALOG_FILE_BYTES
    assert all(len(plan.files[item]) <= MAX_CATALOG_FILE_BYTES for item in shards)
    shard_text = "".join(plan.files[item].decode("utf-8") for item in shards)
    assert all(
        shard_text.count(pair[0].claim_id) == 1 for pair in pairs
    )
    assert "scope preview:" in plan.files["catalog.md"].decode()
    assert "CLAIM-" not in plan.files["catalog.md"].decode()
    assert all(pair[0].semantics.rule in shard_text for pair in pairs)

    claim = pairs[0][0]
    # This deliberately bypasses the acquisition-model length limits only to
    # exercise the renderer's independent 24 KiB publication preflight.
    huge_semantics = claim.semantics.model_copy(
        update={"applies_to": [f"src/{ordinal:04d}.py" for ordinal in range(2_000)]}
    )
    huge_claim = claim.model_copy(update={"semantics": huge_semantics})
    with pytest.raises(ValueError, match="compact card exceeds"):
        render_claim_card(huge_claim, containing_relative="catalog.md")
    assert MAX_RENDERED_COMPACT_CARD_BYTES == 24_576


def test_catalog_decision_below_equal_above_boundary_and_single_card_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claims = [_pair(f"boundary-{ordinal}")[0] for ordinal in range(2)]
    cards = [
        consumer_module.render_catalog_entry(item, containing_relative="catalog.md")
        for item in sorted(claims, key=lambda item: item.claim_id)
    ]
    complete = consumer_module._render_card_file(
        "# Norm4Code Claim Catalog",
        cards,
    )

    monkeypatch.setattr(
        consumer_module,
        "MAX_CATALOG_FILE_BYTES",
        len(complete) + 1,
    )
    below, below_shards = consumer_module._catalog_files(
        sorted(claims, key=lambda item: item.claim_id)
    )
    assert below_shards == 0
    assert below["catalog.md"] == complete

    monkeypatch.setattr(
        consumer_module,
        "MAX_CATALOG_FILE_BYTES",
        len(complete),
    )
    equal, equal_shards = consumer_module._catalog_files(
        sorted(claims, key=lambda item: item.claim_id)
    )
    assert equal_shards == 0
    assert len(equal["catalog.md"]) == len(complete)

    monkeypatch.setattr(
        consumer_module,
        "MAX_CATALOG_FILE_BYTES",
        len(complete) - 1,
    )
    above, above_shards = consumer_module._catalog_files(
        sorted(claims, key=lambda item: item.claim_id)
    )
    assert above_shards == 2
    assert set(above) == {
        "catalog.md",
        "catalog/0001.md",
        "catalog/0002.md",
    }

    one = claims[0]
    one_shard_card = consumer_module.render_catalog_entry(
        one,
        containing_relative="catalog/0001.md",
    )
    shard_size = len(
        consumer_module._render_card_file(
            "# Norm4Code Claim Catalog — 0001",
            [one_shard_card],
        )
    )
    unsharded_size = len(
        consumer_module._render_card_file(
            "# Norm4Code Claim Catalog",
            [consumer_module.render_catalog_entry(one, containing_relative="catalog.md")],
        )
    )
    impossible_size = min(shard_size, unsharded_size) - 1
    monkeypatch.setattr(
        consumer_module,
        "MAX_CATALOG_FILE_BYTES",
        impossible_size,
    )
    with pytest.raises(ValueError, match="cannot fit a bounded catalog shard"):
        consumer_module._catalog_files([one])


def test_hybrid_path_escaping_bounds_and_markdown_only_tree_digest() -> None:
    assert path_anchor("pyproject.toml") == "pyproject.toml"
    assert path_anchor("src/pkg/**/*.py") == "src/pkg"
    assert path_anchor("*.py") == ""
    assert path_view_relative("index.md") == "by-path/%69ndex.md/constraints.md"
    assert (
        path_view_relative("docs/constraints.md")
        == "by-path/docs/%63onstraints.md/constraints.md"
    )
    assert (
        path_view_relative("docs/100%/index.md")
        == "by-path/docs/100%25/%69ndex.md/constraints.md"
    )
    assert path_view_relative("%" * 85).startswith("by-path/" + "%25" * 85)
    with pytest.raises(ValueError, match="physical bound"):
        path_view_relative("%" * 86)
    within_total_bound = "/".join(["a" * 200] * 10)
    assert path_view_relative(within_total_bound).endswith("/constraints.md")
    with pytest.raises(ValueError, match="consumer bound"):
        path_view_relative("/".join(["a" * 200] * 11))
    for unsafe in ("../x.py", "src//x.py", "src/./x.py", "tests/"):
        with pytest.raises(ValueError, match="unsafe claim scope"):
            path_anchor(unsafe)

    plan = _render([_pair("digest")])
    original = consumer_tree_digest(plan.files)
    manifest_changed = dict(plan.files)
    manifest_changed["consumer-manifest.json"] += b" "
    assert consumer_tree_digest(manifest_changed) == original
    markdown_changed = dict(plan.files)
    markdown_changed["START.md"] += b"\n"
    assert consumer_tree_digest(markdown_changed) != original


def test_source_change_check_uses_second_snapshot_metadata_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3.unified import validation as validation_module

    source_run = tmp_path / "source-run"
    source_run.mkdir()
    artifact = source_run / "artifact.txt"
    artifact.write_bytes(b"alpha\n")
    initial_tree, initial_errors = validation_module._snapshot_tree(
        source_run,
        max_artifact_bytes=1_000_000_000,
    )
    assert initial_tree is not None
    assert initial_errors == []
    authenticated = {
        relative: (metadata.sha256, metadata.size)
        for relative, metadata in initial_tree.files.items()
    }

    real_snapshot_tree = validation_module._snapshot_tree
    snapshot_calls: list[Path] = []

    def counted_snapshot_tree(root: Path, *, max_artifact_bytes: int):
        snapshot_calls.append(root)
        return real_snapshot_tree(
            root,
            max_artifact_bytes=max_artifact_bytes,
        )

    def forbidden_snapshot_bytes(*args, **kwargs):
        raise AssertionError("source change check must not reopen snapshot files")

    monkeypatch.setattr(
        validation_module,
        "_snapshot_tree",
        counted_snapshot_tree,
    )
    monkeypatch.setattr(
        validation_module,
        "_snapshot_bytes",
        forbidden_snapshot_bytes,
    )
    # Preserve the byte length so the second snapshot's digest, rather than a
    # size shortcut or a third file read, must identify the mutation.
    artifact.write_bytes(b"omega\n")

    errors = consumer_module._source_files_unchanged(
        source_run,
        authenticated,
        initial_tree.directories,
    )
    assert snapshot_calls == [source_run]
    assert errors == ["source-run artifact changed: artifact.txt"]


def test_manifest_digest_and_physical_markdown_binding_fail_closed(
    tmp_path: Path,
) -> None:
    plan = _render([_pair("manifest-binding")])
    raw = plan.manifest.model_dump(mode="python")
    raw["artifact_digest"] = "f" * 64
    with pytest.raises(ValueError, match="manifest digest"):
        ConsumerManifest.model_validate(raw)

    _materialize(tmp_path, plan)
    start = tmp_path / "START.md"
    start.write_bytes(start.read_bytes() + b"\n")
    _, errors = consumer_module._view_manifest(tmp_path)
    assert any("task-tree digest" in item for item in errors)


def test_closed_view_rejects_extra_missing_and_symlink_but_ignores_modes(
    tmp_path: Path,
) -> None:
    plan = _render([_pair("closed")])
    _materialize(tmp_path, plan)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    assert _compare_consumer_directory(tmp_path, plan) == []

    (tmp_path / "foreign.md").write_text("foreign\n", encoding="utf-8")
    assert any("foreign.md" in item for item in _compare_consumer_directory(tmp_path, plan))
    (tmp_path / "foreign.md").unlink()
    (tmp_path / "catalog.md").unlink()
    (tmp_path / "catalog.md").symlink_to(tmp_path / "START.md")
    assert any(
        "symbolic link" in item or "symlink" in item
        for item in _compare_consumer_directory(tmp_path, plan)
    )


def test_raw_readable_fields_preserve_hostile_markdown_without_structural_injection() -> None:
    value = "format_input uses outlines.models."
    assert consumer_module._labeled_scalar("Rule", value) == [
        "- **Rule:**", "  ```text", f"  {value}", "  ```",
    ]
    hostile = "```\n# Forged heading\n[external](https://example.invalid)\n````"
    rendered = "\n".join(consumer_module._labeled_scalar("Rule", hostile))
    assert "  `````text" in rendered
    assert "\n  # Forged heading\n" in rendered
    assert rendered.endswith("  `````")
    with pytest.raises(ValueError, match="forbidden control"):
        consumer_module._labeled_scalar("Rule", "unsafe\x00")


def test_catalog_links_preserve_conditions_exceptions_and_review_semantics_in_details() -> None:
    pair = _pair(
        "conditional-index",
        rule="Use format_input from outlines.models.",
        condition="when changing adapters",
        exception="except generated adapters",
        delivery="condition_matched_only",
    )
    plan = _render([pair])
    catalog = plan.files["catalog.md"].decode()
    assert pair[0].semantics.rule in catalog
    assert "Normative strength" in catalog
    assert "Read applicability and details" in catalog
    detail = plan.files[f"claims/{pair[0].claim_id}.md"].decode()
    path = plan.files["by-path/src/pkg.py/constraints.md"].decode()
    for text in (catalog, detail, path):
        assert pair[0].semantics.condition in text
        assert pair[0].semantics.exception in text
        assert "**Status:**" not in text
