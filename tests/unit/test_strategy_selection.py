"""Unit tests for versioned registry parsing and fail-closed strategy selection."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from services.data_quality.gate import QualityReasonCode, QualityResult, QualityStatus
from services.strategy_selection.contracts import (
    EconomicAcceptancePolicy,
    FoldEvidence,
    LicenseStatus,
    ModelStrategyScore,
    PerformanceEvidence,
    RegimeEvidence,
    SelectionContext,
    SelectionStatus,
    StrategyDefinition,
    StrategyEvidence,
    StrategyLifecycle,
    StrategySelectionResult,
)
from services.strategy_selection.registry import (
    REGISTRY_SCHEMA_VERSION,
    StrategyRegistry,
    load_registry_document,
)
from services.strategy_selection.selector import select_strategy

D = Decimal
NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)
POLICY_REVISION = "alix-phase10-economic-thresholds-v1"
DATA_HASH = "b" * 64


def make_policy(**changes: Any) -> EconomicAcceptancePolicy:
    values: dict[str, Any] = {
        "revision": POLICY_REVISION,
        "owner_approved": True,
        "minimum_trade_count": 100,
        "minimum_win_rate": D("0.60"),
        "maximum_abs_drawdown": D("0.15"),
        "minimum_net_expectancy": D("0"),
        "max_evidence_age_seconds": 30 * 24 * 60 * 60,
    }
    values.update(changes)
    return EconomicAcceptancePolicy(**values)


def make_definition(**changes: Any) -> StrategyDefinition:
    values: dict[str, Any] = {
        "strategy_id": "trend_alpha",
        "version": "1.0.0",
        "code_sha256": "a" * 64,
        "lifecycle": StrategyLifecycle.SELECTABLE,
        "owner_approved": True,
        "eligible_regimes": ("trend", "range"),
        "supported_timeframes": ("1h",),
        "data_schema_version": "ohlcv.v1",
        "feature_schema_version": "feature-vector.v1",
        "required_features": ("atr", "trend_strength"),
        "required_test_ids": ("strategy-unit-suite", "strategy-contract-suite"),
        "prohibited_conditions": ("liquidity_shock",),
        "risk_profile_id": "spot-long-conservative",
        "risk_profile_revision": "risk-policy-v1",
        "max_data_age_seconds": 3600,
        "max_model_score_age_seconds": 300,
        "max_fee_bps": D("40"),
        "max_slippage_bps": D("10"),
        "max_spread_bps": D("15"),
        "minimum_quote_liquidity_usdt": D("50000"),
        "strategy_license_status": LicenseStatus.APPROVED,
        "data_license_status": LicenseStatus.APPROVED,
        "dependency_license_status": LicenseStatus.APPROVED,
        "market_type": "spot",
        "quote_currency": "USDT",
        "allow_short": False,
    }
    values.update(changes)
    return StrategyDefinition(**values)


def make_context(**changes: Any) -> SelectionContext:
    quality = QualityResult(
        passed=True,
        reasons=(),
        status=QualityStatus.VALID,
        reason_codes=(),
        fingerprint=DATA_HASH,
    )
    values: dict[str, Any] = {
        "decision_id": "decision-registry-test-001",
        "evaluated_at": NOW,
        "market_data_as_of": NOW - timedelta(minutes=1),
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "regime": "trend",
        "data_schema_version": "ohlcv.v1",
        "feature_schema_version": "feature-vector.v1",
        "available_features": ("atr", "trend_strength", "volume_zscore"),
        "market_conditions": (),
        "data_quality": quality,
        "data_fingerprint": DATA_HASH,
        "model_revision": "selector-model-v3",
        "estimated_fee_bps": D("25"),
        "estimated_slippage_bps": D("5"),
        "spread_bps": D("8"),
        "quote_liquidity_usdt": D("100000"),
    }
    values.update(changes)
    return SelectionContext(**values)


def make_evidence(definition: StrategyDefinition, **changes: Any) -> StrategyEvidence:
    metrics = PerformanceEvidence(
        trade_count=100,
        win_rate=D("0.70"),
        net_expectancy=D("0.012"),
        max_abs_drawdown=D("0.10"),
    )
    folds = (
        FoldEvidence(
            "fold-1",
            PerformanceEvidence(34, D("0.68"), D("0.011"), D("0.10")),
        ),
        FoldEvidence(
            "fold-2",
            PerformanceEvidence(33, D("0.70"), D("0.012"), D("0.12")),
        ),
        FoldEvidence(
            "fold-3",
            PerformanceEvidence(33, D("0.72"), D("0.013"), D("0.11")),
        ),
    )
    regimes = (
        RegimeEvidence(
            "trend",
            PerformanceEvidence(60, D("0.72"), D("0.014"), D("0.10")),
        ),
        RegimeEvidence(
            "range",
            PerformanceEvidence(40, D("0.67"), D("0.009"), D("0.12")),
        ),
    )
    values: dict[str, Any] = {
        "strategy_id": definition.strategy_id,
        "strategy_version": definition.version,
        "strategy_code_sha256": definition.code_sha256,
        "evaluation_id": "walk-forward-evaluation-v1",
        "evaluation_commit_sha": "c" * 40,
        "dataset_sha256": "d" * 64,
        "report_sha256": "e" * 64,
        "feature_schema_version": definition.feature_schema_version,
        "economic_policy_revision": POLICY_REVISION,
        "evaluated_at": NOW - timedelta(hours=2),
        "passed_test_ids": definition.required_test_ids,
        "metrics": metrics,
        "fee_bps": D("30"),
        "slippage_bps": D("8"),
        "folds": folds,
        "regimes": regimes,
    }
    values.update(changes)
    return StrategyEvidence(**values)


def make_score(definition: StrategyDefinition, **changes: Any) -> ModelStrategyScore:
    values: dict[str, Any] = {
        "strategy_id": definition.strategy_id,
        "strategy_version": definition.version,
        "model_revision": "selector-model-v3",
        "scored_at": NOW - timedelta(seconds=30),
        "score": D("0.80"),
    }
    values.update(changes)
    return ModelStrategyScore(**values)


def make_registry(*definitions: StrategyDefinition) -> StrategyRegistry:
    return StrategyRegistry(
        revision="strategy-registry-test-v1",
        economic_policy_revision=POLICY_REVISION,
        strategies=tuple(definitions),
    )


def run_selector(
    *,
    definitions: tuple[StrategyDefinition, ...] | None = None,
    context: SelectionContext | None = None,
    policy: EconomicAcceptancePolicy | None = None,
    scores: tuple[ModelStrategyScore, ...] | None = None,
    evidence: tuple[StrategyEvidence, ...] | None = None,
) -> StrategySelectionResult:
    selected_definitions = definitions or (make_definition(),)
    selected_context = context or make_context()
    selected_policy = policy or make_policy()
    selected_scores = scores
    if selected_scores is None:
        selected_scores = tuple(make_score(item) for item in selected_definitions)
    selected_evidence = evidence
    if selected_evidence is None:
        selected_evidence = tuple(make_evidence(item) for item in selected_definitions)
    return select_strategy(
        registry=make_registry(*selected_definitions),
        context=selected_context,
        economic_policy=selected_policy,
        model_scores=selected_scores,
        evidence_items=selected_evidence,
    )


def _selected_reasons(result: StrategySelectionResult) -> tuple[str, ...]:
    return result.candidate_rejections[0].reasons


def test_approved_oos_candidate_selects_deterministically_without_order_permission() -> (
    None
):
    definition = make_definition()
    context = make_context()
    policy = make_policy()
    scores = (make_score(definition),)
    evidence = (make_evidence(definition),)
    first = run_selector(
        definitions=(definition,),
        context=context,
        policy=policy,
        scores=scores,
        evidence=evidence,
    )
    second = run_selector(
        definitions=(definition,),
        context=context,
        policy=policy,
        scores=scores,
        evidence=evidence,
    )

    assert first.status is SelectionStatus.SELECTED
    assert first.selected_strategy_id == definition.strategy_id
    assert first.selected_strategy_version == definition.version
    assert first.registry_revision == "strategy-registry-test-v1"
    assert first.evidence_fingerprint == evidence[0].fingerprint
    assert first.model_revision == context.model_revision
    assert first.order_permission is False
    assert first.fingerprint == second.fingerprint


def test_model_score_is_only_a_ranker_after_deterministic_eligibility_gates() -> None:
    rejected = make_definition(
        strategy_id="strategy_alpha", lifecycle=StrategyLifecycle.RESEARCH_ONLY
    )
    accepted = make_definition(
        strategy_id="strategy_beta",
        code_sha256="f" * 64,
        lifecycle=StrategyLifecycle.SELECTABLE,
    )
    scores = (
        make_score(rejected, score=D("0.99")),
        make_score(accepted, score=D("0.50")),
    )
    evidence = (make_evidence(rejected), make_evidence(accepted))
    result = run_selector(
        definitions=(rejected, accepted), scores=scores, evidence=evidence
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.selected_strategy_id == "strategy_beta"
    assert result.candidate_rejections[0].strategy_id == "strategy_alpha"
    assert "strategy_not_selectable" in result.candidate_rejections[0].reasons


def test_score_ties_are_broken_by_stable_strategy_id_order() -> None:
    alpha = make_definition(strategy_id="strategy_alpha", code_sha256="a" * 64)
    beta = make_definition(strategy_id="strategy_beta", code_sha256="f" * 64)
    result = run_selector(
        definitions=(beta, alpha),
        scores=(make_score(beta, score=D("0.75")), make_score(alpha, score=D("0.75"))),
        evidence=(make_evidence(beta), make_evidence(alpha)),
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.selected_strategy_id == "strategy_alpha"


def test_empty_production_registry_abstains() -> None:
    config_path = Path("config/strategy_selection/strategy-registry.json")
    loaded = load_registry_document(config_path.read_text(encoding="utf-8"))
    assert loaded.registry is not None
    assert loaded.reason_codes == ()
    assert loaded.registry.strategies == ()

    result = select_strategy(
        registry=loaded.registry,
        context=make_context(),
        economic_policy=make_policy(),
        model_scores=(),
        evidence_items=(),
    )
    assert result.status is SelectionStatus.NO_TRADE
    assert result.reason_codes == ("strategy_registry_empty", "no_eligible_strategy")
    assert result.order_permission is False


def test_registry_loader_accepts_a_versioned_entry_with_string_decimals() -> None:
    definition = make_definition()
    entry = asdict(definition)
    document = json.dumps(
        {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "revision": "registry-parser-test-v1",
            "economic_policy_revision": POLICY_REVISION,
            "strategies": [entry],
        },
        default=str,
    )
    loaded = load_registry_document(document)

    assert loaded.registry is not None
    assert loaded.reason_codes == ()
    assert loaded.registry.strategies[0] == definition
    assert loaded.registry.fingerprint


@pytest.mark.parametrize(
    "document",
    (
        "{not-json",
        "[]",
        json.dumps(
            {
                "schema_version": "unsupported",
                "revision": "r1",
                "economic_policy_revision": POLICY_REVISION,
                "strategies": [],
            }
        ),
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "revision": "r1",
                "economic_policy_revision": POLICY_REVISION,
                "strategies": [],
                "unexpected": True,
            }
        ),
    ),
)
def test_invalid_registry_documents_fail_closed(document: str) -> None:
    loaded = load_registry_document(document)
    assert loaded.registry is None
    assert loaded.reason_codes == ("strategy_registry_document_invalid",)


def test_registry_rejects_duplicate_strategy_ids_and_unapproved_schema() -> None:
    definition = make_definition()
    with pytest.raises(ValueError, match="strategy_registry_ids_duplicated"):
        make_registry(definition, definition)
    with pytest.raises(ValueError, match="schema_version_unsupported"):
        StrategyRegistry(
            revision="r1",
            economic_policy_revision=POLICY_REVISION,
            strategies=(),
            schema_version="unknown",
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"lifecycle": StrategyLifecycle.RESEARCH_ONLY}, "strategy_not_selectable"),
        ({"lifecycle": StrategyLifecycle.RETIRED}, "strategy_not_selectable"),
        ({"owner_approved": False}, "strategy_owner_approval_missing"),
        (
            {"strategy_license_status": LicenseStatus.CONDITIONAL},
            "strategy_or_data_license_not_approved",
        ),
        (
            {"data_license_status": LicenseStatus.REJECTED},
            "strategy_or_data_license_not_approved",
        ),
        (
            {"dependency_license_status": LicenseStatus.CONDITIONAL},
            "strategy_or_data_license_not_approved",
        ),
        ({"eligible_regimes": ("range",)}, "strategy_regime_not_supported"),
        ({"supported_timeframes": ("15m",)}, "strategy_timeframe_not_supported"),
        ({"data_schema_version": "ohlcv.v2"}, "strategy_data_schema_mismatch"),
        ({"feature_schema_version": "features.v2"}, "strategy_feature_schema_mismatch"),
        (
            {"required_features": ("unavailable_feature",)},
            "strategy_required_features_missing",
        ),
        ({"max_data_age_seconds": 10}, "market_data_stale_for_strategy"),
        ({"max_fee_bps": D("10")}, "estimated_fee_above_strategy_limit"),
        ({"max_slippage_bps": D("2")}, "estimated_slippage_above_strategy_limit"),
        ({"max_spread_bps": D("2")}, "spread_above_strategy_limit"),
        (
            {"minimum_quote_liquidity_usdt": D("200000")},
            "quote_liquidity_below_strategy_minimum",
        ),
    ),
)
def test_candidate_constraints_reject_before_score_can_select(
    changes: dict[str, Any], expected: str
) -> None:
    definition = make_definition(**changes)
    result = run_selector(definitions=(definition,))
    assert result.status is SelectionStatus.NO_TRADE
    assert expected in _selected_reasons(result)
    assert result.order_permission is False


@pytest.mark.parametrize(
    ("context_changes", "expected"),
    (
        ({"symbol": "BTC/BTC"}, "spot_usdt_symbol_required"),
        ({"symbol": "btc/USDT"}, "spot_usdt_symbol_required"),
        ({"regime": "unknown"}, "strategy_regime_not_supported"),
        (
            {"market_conditions": ("liquidity_shock",)},
            "strategy_prohibited_market_condition_active",
        ),
        ({"available_features": ()}, "selection_features_missing"),
        (
            {"market_data_as_of": NOW + timedelta(seconds=1)},
            "market_data_timestamp_in_future",
        ),
        ({"estimated_fee_bps": None}, "estimated_fee_missing"),
        ({"estimated_slippage_bps": None}, "estimated_slippage_missing"),
        ({"spread_bps": None}, "spread_estimate_missing"),
        ({"quote_liquidity_usdt": None}, "quote_liquidity_missing"),
        ({"data_fingerprint": "f" * 64}, "data_quality_fingerprint_mismatch"),
    ),
)
def test_active_prohibited_market_condition_rejects_candidate() -> None:
    definition = make_definition(prohibited_conditions=("news_outage",))
    result = run_selector(
        definitions=(definition,),
        context=make_context(market_conditions=("news_outage",)),
    )

    assert result.status is SelectionStatus.NO_TRADE
    assert "strategy_prohibited_market_condition_active" in _selected_reasons(result)
    assert result.order_permission is False


def test_context_constraints_fail_closed(
    context_changes: dict[str, Any], expected: str
) -> None:
    result = run_selector(context=make_context(**context_changes))
    assert result.status is SelectionStatus.NO_TRADE
    reasons = result.reason_codes + tuple(
        reason
        for rejected in result.candidate_rejections
        for reason in rejected.reasons
    )
    assert expected in reasons


def test_missing_degraded_or_rejected_data_quality_never_selects() -> None:
    degraded = QualityResult(
        passed=True,
        reasons=("gap_detected",),
        status=QualityStatus.DEGRADED_ALLOWED,
        reason_codes=(QualityReasonCode.DATA_GAP,),
        fingerprint=DATA_HASH,
    )
    rejected = QualityResult(
        passed=False,
        reasons=("gap_detected",),
        status=QualityStatus.REJECTED,
        reason_codes=(QualityReasonCode.DATA_GAP,),
        fingerprint=DATA_HASH,
    )
    for quality, expected in (
        (None, "data_quality_result_missing"),
        (degraded, "data_quality_not_strictly_valid"),
        (rejected, "data_quality_not_strictly_valid"),
    ):
        result = run_selector(context=make_context(data_quality=quality))
        assert result.status is SelectionStatus.NO_TRADE
        assert expected in result.reason_codes


def test_missing_context_or_policy_and_unapproved_policy_abstain() -> None:
    definition = make_definition()
    score = make_score(definition)
    evidence = make_evidence(definition)
    registry = make_registry(definition)
    missing_context = select_strategy(
        registry=registry,
        context=None,
        economic_policy=make_policy(),
        model_scores=(score,),
        evidence_items=(evidence,),
    )
    missing_policy = select_strategy(
        registry=registry,
        context=make_context(),
        economic_policy=None,
        model_scores=(score,),
        evidence_items=(evidence,),
    )
    unapproved_policy = run_selector(policy=make_policy(owner_approved=False))

    assert "selection_context_missing" in missing_context.reason_codes
    assert "economic_acceptance_policy_missing" in missing_policy.reason_codes
    assert (
        "economic_acceptance_policy_not_owner_approved"
        in unapproved_policy.reason_codes
    )
    assert all(
        item.order_permission is False
        for item in (missing_context, missing_policy, unapproved_policy)
    )


def test_missing_registry_and_policy_revision_mismatch_abstain() -> None:
    context = make_context()
    policy = make_policy()
    definition = make_definition()
    score = make_score(definition)
    evidence = make_evidence(definition)
    missing_registry = select_strategy(
        registry=None,
        context=context,
        economic_policy=policy,
        model_scores=(score,),
        evidence_items=(evidence,),
        registry_failure_reasons=("strategy_registry_document_invalid",),
    )
    wrong_policy_registry = StrategyRegistry(
        revision="registry-v2",
        economic_policy_revision="different-policy-v2",
        strategies=(definition,),
    )
    mismatch = select_strategy(
        registry=wrong_policy_registry,
        context=context,
        economic_policy=policy,
        model_scores=(score,),
        evidence_items=(evidence,),
    )

    assert missing_registry.reason_codes == ("strategy_registry_document_invalid",)
    assert "economic_policy_revision_mismatch" in mismatch.reason_codes


@pytest.mark.parametrize(
    ("score_changes", "scores", "expected"),
    (
        ({}, (), "model_score_missing"),
        ({"strategy_version": "2.0.0"}, None, "model_score_version_mismatch"),
        ({"model_revision": "wrong-model-v1"}, None, "model_score_revision_mismatch"),
        (
            {"scored_at": NOW + timedelta(seconds=1)},
            None,
            "model_score_timestamp_in_future",
        ),
        ({"scored_at": NOW - timedelta(hours=1)}, None, "model_score_stale"),
        ({}, "duplicate", "model_score_ambiguous"),
    ),
)
def test_model_score_lineage_and_freshness_are_required(
    score_changes: dict[str, Any],
    scores: tuple[ModelStrategyScore, ...] | str | None,
    expected: str,
) -> None:
    definition = make_definition()
    score = make_score(definition, **score_changes)
    selected_scores: tuple[ModelStrategyScore, ...] = () if scores == () else (score,)
    if scores == "duplicate":
        selected_scores = (score, score)
    result = run_selector(definitions=(definition,), scores=selected_scores)
    assert result.status is SelectionStatus.NO_TRADE
    assert expected in _selected_reasons(result)


@pytest.mark.parametrize(
    ("evidence_changes", "expected"),
    (
        ({"strategy_code_sha256": "f" * 64}, "strategy_code_hash_mismatch"),
        ({"strategy_version": "2.0.0"}, "evidence_version_mismatch"),
        ({"feature_schema_version": "features.v2"}, "oos_feature_schema_mismatch"),
        (
            {"economic_policy_revision": "economic-policy-v2"},
            "oos_economic_policy_revision_mismatch",
        ),
        (
            {"evaluated_at": NOW + timedelta(seconds=1)},
            "oos_evidence_timestamp_in_future",
        ),
        ({"evaluated_at": NOW - timedelta(days=60)}, "oos_evidence_stale"),
        ({"passed_test_ids": ()}, "required_strategy_tests_missing"),
        ({"fee_bps": D("10")}, "oos_fee_assumption_below_current_estimate"),
        ({"slippage_bps": D("2")}, "oos_slippage_assumption_below_current_estimate"),
    ),
)
def test_evidence_lineage_freshness_and_cost_assumptions_are_enforced(
    evidence_changes: dict[str, Any], expected: str
) -> None:
    definition = make_definition()
    evidence = make_evidence(definition, **evidence_changes)
    result = run_selector(definitions=(definition,), evidence=(evidence,))
    assert result.status is SelectionStatus.NO_TRADE
    assert expected in _selected_reasons(result)


@pytest.mark.parametrize(
    ("metric_changes", "expected"),
    (
        ({"trade_count": 99}, "oos_insufficient_sample"),
        ({"win_rate": D("0.59")}, "oos_win_rate_below_threshold"),
        ({"max_abs_drawdown": D("0.16")}, "oos_drawdown_above_threshold"),
        ({"net_expectancy": D("0")}, "oos_net_expectancy_not_positive"),
    ),
)
def test_pre_registered_economic_thresholds_reject_failing_evidence(
    metric_changes: dict[str, Any], expected: str
) -> None:
    definition = make_definition()
    evidence = make_evidence(
        definition,
        metrics=PerformanceEvidence(
            **{
                "trade_count": 100,
                "win_rate": D("0.70"),
                "net_expectancy": D("0.012"),
                "max_abs_drawdown": D("0.10"),
                **metric_changes,
            }
        ),
    )
    result = run_selector(definitions=(definition,), evidence=(evidence,))
    assert result.status is SelectionStatus.NO_TRADE
    assert expected in _selected_reasons(result)


def test_fold_and_regime_stability_are_required() -> None:
    definition = make_definition()
    evidence = make_evidence(definition)
    negative_fold = replace(
        evidence.folds[1],
        metrics=PerformanceEvidence(33, D("0.6"), D("-0.01"), D("0.12")),
    )
    fold_failure = replace(
        evidence, folds=(evidence.folds[0], negative_fold, evidence.folds[2])
    )
    no_fold_evidence = replace(evidence, folds=())
    mismatch_fold_count = replace(
        evidence,
        folds=(
            evidence.folds[0],
            evidence.folds[1],
            replace(
                evidence.folds[2],
                metrics=PerformanceEvidence(20, D("0.72"), D("0.013"), D("0.11")),
            ),
        ),
    )
    negative_regime = replace(
        evidence.regimes[1],
        metrics=PerformanceEvidence(40, D("0.6"), D("-0.01"), D("0.12")),
    )
    regime_failure = replace(evidence, regimes=(evidence.regimes[0], negative_regime))
    missing_regime = replace(evidence, regimes=(evidence.regimes[0],))
    mismatch_regime_count = replace(
        evidence,
        regimes=(
            evidence.regimes[0],
            replace(
                evidence.regimes[1],
                metrics=PerformanceEvidence(30, D("0.67"), D("0.009"), D("0.12")),
            ),
        ),
    )
    for changed, expected in (
        (fold_failure, "oos_fold_expectancy_not_positive"),
        (no_fold_evidence, "oos_fold_count_below_minimum"),
        (mismatch_fold_count, "oos_fold_sample_count_mismatch"),
        (regime_failure, "oos_regime_expectancy_not_positive"),
        (missing_regime, "oos_regime_coverage_mismatch"),
        (mismatch_regime_count, "oos_regime_sample_count_mismatch"),
    ):
        result = run_selector(definitions=(definition,), evidence=(changed,))
        assert result.status is SelectionStatus.NO_TRADE
        assert expected in _selected_reasons(result)


def test_duplicate_evidence_and_mismatched_model_score_are_not_resolved_by_guessing() -> (
    None
):
    definition = make_definition()
    evidence = make_evidence(definition)
    duplicate_evidence = run_selector(
        definitions=(definition,), evidence=(evidence, evidence)
    )
    wrong_model_revision = run_selector(
        definitions=(definition,),
        scores=(make_score(definition, model_revision="other-model"),),
    )

    assert "oos_evidence_ambiguous" in _selected_reasons(duplicate_evidence)
    assert "model_score_revision_mismatch" in _selected_reasons(wrong_model_revision)


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: make_definition(version="one"), "semver"),
        (lambda: make_definition(market_type="futures"), "market_must_be_spot"),
        (lambda: make_definition(quote_currency="BTC"), "quote_must_be_usdt"),
        (lambda: make_definition(allow_short=True), "shorting_forbidden"),
        (lambda: make_definition(code_sha256="not-a-hash"), "invalid_hash"),
        (
            lambda: make_definition(eligible_regimes=("trend", "trend")),
            "contains_duplicates",
        ),
        (lambda: make_policy(minimum_trade_count=0), "minimum_trade_count_invalid"),
        (lambda: make_policy(max_evidence_age_seconds=0), "evidence_age_limit_invalid"),
        (
            lambda: PerformanceEvidence(1, D("1.1"), D("0.1"), D("0.1")),
            "win_rate_above_maximum",
        ),
    ),
)
def test_contracts_reject_invalid_strategy_policy_and_metrics(
    factory: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_result_contract_cannot_be_constructed_with_order_permission_true() -> None:
    result = run_selector()
    with pytest.raises(ValueError, match="cannot_grant_order_permission"):
        replace(result, order_permission=True)
