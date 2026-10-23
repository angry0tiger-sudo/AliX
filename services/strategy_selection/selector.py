"""Deterministic, fail-closed selector over versioned, evidence-backed strategies."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from decimal import Decimal

from services.data_quality.gate import QualityStatus
from services.strategy_selection.contracts import (
    CandidateRejection,
    EconomicAcceptancePolicy,
    LicenseStatus,
    ModelStrategyScore,
    SelectionContext,
    SelectionStatus,
    StrategyDefinition,
    StrategyEvidence,
    StrategyLifecycle,
    StrategySelectionResult,
    stable_fingerprint,
)
from services.strategy_selection.registry import StrategyRegistry


def select_strategy(
    *,
    registry: StrategyRegistry | None,
    context: SelectionContext | None,
    economic_policy: EconomicAcceptancePolicy | None,
    model_scores: tuple[ModelStrategyScore, ...],
    evidence_items: tuple[StrategyEvidence, ...],
    registry_failure_reasons: tuple[str, ...] = (),
) -> StrategySelectionResult:
    """Select a strategy candidate, or abstain with deterministic audit reasons.

    Passing this selector never grants order permission. Freqtrade's strategy and
    the independent Risk Gate remain separate, and the default production registry
    is intentionally empty while there is no economically accepted Champion.
    """
    decision_id = context.decision_id if context is not None else "unknown"
    evaluated_at = context.evaluated_at if context is not None else None
    global_reasons: list[str] = []
    if registry is None:
        global_reasons.extend(
            registry_failure_reasons or ("strategy_registry_unavailable",)
        )
    if context is None:
        global_reasons.append("selection_context_missing")
    else:
        global_reasons.extend(_context_reasons(context))
    if economic_policy is None:
        global_reasons.append("economic_acceptance_policy_missing")
    elif not economic_policy.owner_approved:
        global_reasons.append("economic_acceptance_policy_not_owner_approved")
    if (
        registry is not None
        and economic_policy is not None
        and registry.economic_policy_revision != economic_policy.revision
    ):
        global_reasons.append("economic_policy_revision_mismatch")
    if global_reasons:
        return _result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            status=SelectionStatus.NO_TRADE,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=tuple(dict.fromkeys(global_reasons)),
            candidate_rejections=(),
        )
    if registry is None or context is None or economic_policy is None:
        return _result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            status=SelectionStatus.NO_TRADE,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("selection_inputs_unavailable",),
            candidate_rejections=(),
        )
    if not registry.strategies:
        return _result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            status=SelectionStatus.NO_TRADE,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("strategy_registry_empty", "no_eligible_strategy"),
            candidate_rejections=(),
        )

    scored_candidates: list[tuple[Decimal, StrategyDefinition, StrategyEvidence]] = []
    rejections: list[CandidateRejection] = []
    for definition in registry.strategies:
        reasons, score, evidence = _candidate_eligibility(
            definition=definition,
            registry=registry,
            context=context,
            economic_policy=economic_policy,
            model_scores=model_scores,
            evidence_items=evidence_items,
        )
        if reasons or score is None or evidence is None:
            rendered = tuple(dict.fromkeys(reasons or ("candidate_not_eligible",)))
            rejections.append(
                CandidateRejection(strategy_id=definition.strategy_id, reasons=rendered)
            )
        else:
            scored_candidates.append((score.score, definition, evidence))

    if not scored_candidates:
        return _result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            status=SelectionStatus.NO_TRADE,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("no_eligible_strategy",),
            candidate_rejections=tuple(rejections),
        )

    scored_candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            candidate[1].strategy_id,
            candidate[1].version,
        )
    )
    selected_score, selected_definition, selected_evidence = scored_candidates[0]
    return _result(
        decision_id=decision_id,
        evaluated_at=evaluated_at,
        status=SelectionStatus.SELECTED,
        registry=registry,
        economic_policy=economic_policy,
        context=context,
        reason_codes=("highest_model_score_among_eligible_candidates",),
        candidate_rejections=tuple(rejections),
        selected_definition=selected_definition,
        selected_evidence=selected_evidence,
        model_score=selected_score,
        model_revision=context.model_revision,
    )


def _context_reasons(context: SelectionContext) -> tuple[str, ...]:
    reasons: list[str] = []
    if context.market_data_as_of > context.evaluated_at:
        reasons.append("market_data_timestamp_in_future")
    if (
        context.symbol != context.symbol.upper()
        or context.symbol.count("/") != 1
        or not context.symbol.endswith("/USDT")
        or not context.symbol.split("/", maxsplit=1)[0]
    ):
        reasons.append("spot_usdt_symbol_required")
    quality = context.data_quality
    if quality is None:
        reasons.append("data_quality_result_missing")
    elif not quality.passed or quality.status is not QualityStatus.VALID:
        reasons.append("data_quality_not_strictly_valid")
    elif not quality.fingerprint or quality.fingerprint != context.data_fingerprint:
        reasons.append("data_quality_fingerprint_mismatch")
    if not context.available_features:
        reasons.append("selection_features_missing")
    if context.estimated_fee_bps is None:
        reasons.append("estimated_fee_missing")
    if context.estimated_slippage_bps is None:
        reasons.append("estimated_slippage_missing")
    if context.spread_bps is None:
        reasons.append("spread_estimate_missing")
    if context.quote_liquidity_usdt is None:
        reasons.append("quote_liquidity_missing")
    return tuple(reasons)


def _candidate_eligibility(
    *,
    definition: StrategyDefinition,
    registry: StrategyRegistry,
    context: SelectionContext,
    economic_policy: EconomicAcceptancePolicy,
    model_scores: tuple[ModelStrategyScore, ...],
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[list[str], ModelStrategyScore | None, StrategyEvidence | None]:
    reasons: list[str] = []
    if definition.lifecycle is not StrategyLifecycle.SELECTABLE:
        reasons.append("strategy_not_selectable")
    if not definition.owner_approved:
        reasons.append("strategy_owner_approval_missing")
    if (
        definition.strategy_license_status is not LicenseStatus.APPROVED
        or definition.data_license_status is not LicenseStatus.APPROVED
        or definition.dependency_license_status is not LicenseStatus.APPROVED
    ):
        reasons.append("strategy_or_data_license_not_approved")
    if context.regime not in definition.eligible_regimes:
        reasons.append("strategy_regime_not_supported")
    if context.timeframe not in definition.supported_timeframes:
        reasons.append("strategy_timeframe_not_supported")
    if context.data_schema_version != definition.data_schema_version:
        reasons.append("strategy_data_schema_mismatch")
    if context.feature_schema_version != definition.feature_schema_version:
        reasons.append("strategy_feature_schema_mismatch")
    if not set(definition.required_features).issubset(context.available_features):
        reasons.append("strategy_required_features_missing")
    if set(definition.prohibited_conditions).intersection(context.market_conditions):
        reasons.append("strategy_prohibited_market_condition_active")
    if context.evaluated_at < context.market_data_as_of:
        reasons.append("market_data_timestamp_in_future")
    elif (
        context.evaluated_at - context.market_data_as_of
    ).total_seconds() > definition.max_data_age_seconds:
        reasons.append("market_data_stale_for_strategy")
    reasons.extend(_cost_and_liquidity_reasons(definition, context))

    score, score_reasons = _matching_model_score(
        definition=definition,
        context=context,
        model_scores=model_scores,
    )
    reasons.extend(score_reasons)
    item, evidence_reasons = _matching_evidence(
        definition=definition,
        context=context,
        registry=registry,
        economic_policy=economic_policy,
        evidence_items=evidence_items,
    )
    reasons.extend(evidence_reasons)
    return list(dict.fromkeys(reasons)), score, item


def _cost_and_liquidity_reasons(
    definition: StrategyDefinition, context: SelectionContext
) -> tuple[str, ...]:
    reasons: list[str] = []
    fee = context.estimated_fee_bps
    slippage = context.estimated_slippage_bps
    spread = context.spread_bps
    liquidity = context.quote_liquidity_usdt
    if fee is not None and fee > definition.max_fee_bps:
        reasons.append("estimated_fee_above_strategy_limit")
    if slippage is not None and slippage > definition.max_slippage_bps:
        reasons.append("estimated_slippage_above_strategy_limit")
    if spread is not None and spread > definition.max_spread_bps:
        reasons.append("spread_above_strategy_limit")
    if (
        liquidity is not None
        and liquidity < definition.minimum_quote_liquidity_usdt
    ):
        reasons.append("quote_liquidity_below_strategy_minimum")
    return tuple(reasons)


def _matching_model_score(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    model_scores: tuple[ModelStrategyScore, ...],
) -> tuple[ModelStrategyScore | None, tuple[str, ...]]:
    same_id = [score for score in model_scores if score.strategy_id == definition.strategy_id]
    matching = [
        score
        for score in same_id
        if score.strategy_version == definition.version
    ]
    if not matching:
        if same_id:
            return None, ("model_score_version_mismatch",)
        return None, ("model_score_missing",)
    if len(matching) != 1:
        return None, ("model_score_ambiguous",)
    score = matching[0]
    reasons: list[str] = []
    if score.model_revision != context.model_revision:
        reasons.append("model_score_revision_mismatch")
    if score.scored_at > context.evaluated_at:
        reasons.append("model_score_timestamp_in_future")
    elif (
        context.evaluated_at - score.scored_at
    ).total_seconds() > definition.max_model_score_age_seconds:
        reasons.append("model_score_stale")
    return score, tuple(reasons)


def _matching_evidence(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    registry: StrategyRegistry,
    economic_policy: EconomicAcceptancePolicy,
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[StrategyEvidence | None, tuple[str, ...]]:
    same_id = [
        item for item in evidence_items if item.strategy_id == definition.strategy_id
    ]
    matching = [
        item
        for item in same_id
        if item.strategy_version == definition.version
    ]
    if not matching:
        reason = "evidence_version_mismatch" if same_id else "oos_evidence_missing"
        return None, (reason,)
    if len(matching) != 1:
        return None, ("oos_evidence_ambiguous",)
    item = matching[0]
    reasons = _evidence_reasons(
        definition=definition,
        context=context,
        registry=registry,
        economic_policy=economic_policy,
        evidence=item,
    )
    return item, reasons


def _evidence_reasons(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    registry: StrategyRegistry,
    economic_policy: EconomicAcceptancePolicy,
    evidence: StrategyEvidence,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if evidence.strategy_code_sha256 != definition.code_sha256:
        reasons.append("strategy_code_hash_mismatch")
    if evidence.feature_schema_version != definition.feature_schema_version:
        reasons.append("oos_feature_schema_mismatch")
    if evidence.economic_policy_revision != economic_policy.revision:
        reasons.append("oos_economic_policy_revision_mismatch")
    if registry.economic_policy_revision != evidence.economic_policy_revision:
        reasons.append("registry_economic_policy_revision_mismatch")
    if evidence.evaluated_at > context.evaluated_at:
        reasons.append("oos_evidence_timestamp_in_future")
    elif (
        context.evaluated_at - evidence.evaluated_at
    ).total_seconds() > economic_policy.max_evidence_age_seconds:
        reasons.append("oos_evidence_stale")
    missing_tests = set(definition.required_test_ids) - set(evidence.passed_test_ids)
    if missing_tests:
        reasons.append("required_strategy_tests_missing")
    metrics = evidence.metrics
    if metrics.trade_count < economic_policy.minimum_trade_count:
        reasons.append("oos_insufficient_sample")
    if metrics.win_rate < economic_policy.minimum_win_rate:
        reasons.append("oos_win_rate_below_threshold")
    if metrics.max_abs_drawdown > economic_policy.maximum_abs_drawdown:
        reasons.append("oos_drawdown_above_threshold")
    if metrics.net_expectancy <= economic_policy.minimum_net_expectancy:
        reasons.append("oos_net_expectancy_not_positive")
    if len(evidence.folds) < 3:
        reasons.append("oos_fold_count_below_minimum")
    if sum(fold.metrics.trade_count for fold in evidence.folds) != metrics.trade_count:
        reasons.append("oos_fold_sample_count_mismatch")
    for fold in evidence.folds:
        if fold.metrics.trade_count < 1:
            reasons.append("oos_fold_insufficient_sample")
        if fold.metrics.net_expectancy <= economic_policy.minimum_net_expectancy:
            reasons.append("oos_fold_expectancy_not_positive")
        if fold.metrics.max_abs_drawdown > economic_policy.maximum_abs_drawdown:
            reasons.append("oos_fold_drawdown_above_threshold")
    known_regimes = {item.regime for item in evidence.regimes}
    if known_regimes != set(definition.eligible_regimes):
        reasons.append("oos_regime_coverage_mismatch")
    if context.regime not in known_regimes:
        reasons.append("oos_current_regime_evidence_missing")
    if sum(item.metrics.trade_count for item in evidence.regimes) != metrics.trade_count:
        reasons.append("oos_regime_sample_count_mismatch")
    for regime_evidence in evidence.regimes:
        if regime_evidence.metrics.trade_count < 1:
            reasons.append("oos_regime_insufficient_sample")
        if (
            regime_evidence.metrics.net_expectancy
            <= economic_policy.minimum_net_expectancy
        ):
            reasons.append("oos_regime_expectancy_not_positive")
        if (
            regime_evidence.metrics.max_abs_drawdown
            > economic_policy.maximum_abs_drawdown
        ):
            reasons.append("oos_regime_drawdown_above_threshold")
    if (
        context.estimated_fee_bps is not None
        and evidence.fee_bps < context.estimated_fee_bps
    ):
        reasons.append("oos_fee_assumption_below_current_estimate")
    if (
        context.estimated_slippage_bps is not None
        and evidence.slippage_bps < context.estimated_slippage_bps
    ):
        reasons.append("oos_slippage_assumption_below_current_estimate")
    return tuple(dict.fromkeys(reasons))


def _result(
    *,
    decision_id: str,
    evaluated_at: datetime | None,
    status: SelectionStatus,
    registry: StrategyRegistry | None,
    economic_policy: EconomicAcceptancePolicy | None,
    context: SelectionContext | None,
    reason_codes: tuple[str, ...],
    candidate_rejections: tuple[CandidateRejection, ...],
    selected_definition: StrategyDefinition | None = None,
    selected_evidence: StrategyEvidence | None = None,
    model_score: Decimal | None = None,
    model_revision: str | None = None,
) -> StrategySelectionResult:
    actual_time = context.evaluated_at if context is not None else evaluated_at
    actual_model_revision = model_revision or (
        context.model_revision if context is not None else None
    )
    payload = {
        "decision_id": decision_id,
        "evaluated_at": actual_time,
        "status": status.value,
        "selected_strategy_id": (
            selected_definition.strategy_id if selected_definition else None
        ),
        "selected_strategy_version": (
            selected_definition.version if selected_definition else None
        ),
        "registry_revision": registry.revision if registry else None,
        "registry_fingerprint": registry.fingerprint if registry else None,
        "economic_policy_revision": (
            economic_policy.revision if economic_policy else None
        ),
        "evidence_fingerprint": (
            selected_evidence.fingerprint if selected_evidence else None
        ),
        "model_revision": actual_model_revision,
        "model_score": model_score,
        "reason_codes": reason_codes,
        "candidate_rejections": [asdict(item) for item in candidate_rejections],
        "order_permission": False,
    }
    fingerprint = stable_fingerprint(payload)
    return StrategySelectionResult(
        decision_id=decision_id,
        evaluated_at=actual_time,
        status=status,
        selected_strategy_id=(
            selected_definition.strategy_id if selected_definition else None
        ),
        selected_strategy_version=(
            selected_definition.version if selected_definition else None
        ),
        registry_revision=registry.revision if registry else None,
        registry_fingerprint=registry.fingerprint if registry else None,
        economic_policy_revision=(
            economic_policy.revision if economic_policy else None
        ),
        evidence_fingerprint=(
            selected_evidence.fingerprint if selected_evidence else None
        ),
        model_revision=actual_model_revision,
        model_score=model_score,
        reason_codes=reason_codes,
        candidate_rejections=candidate_rejections,
        order_permission=False,
        fingerprint=fingerprint,
    )
