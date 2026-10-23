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
    """Select a candidate or abstain with an auditable reason and fingerprint.

    Scores rank candidates only after deterministic gates pass. This function
    never grants order permission; Freqtrade and the independent Risk Gate remain
    separate. The production registry is intentionally empty until evidence passes.
    """
    decision_id = context.decision_id if context is not None else "unknown"
    evaluated_at = context.evaluated_at if context is not None else None
    global_reasons = _global_input_reasons(
        registry=registry,
        context=context,
        economic_policy=economic_policy,
        registry_failure_reasons=registry_failure_reasons,
    )
    if global_reasons:
        return _no_trade_result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=global_reasons,
        )
    if registry is None or context is None or economic_policy is None:
        return _no_trade_result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("selection_inputs_unavailable",),
        )
    if not registry.strategies:
        return _no_trade_result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("strategy_registry_empty", "no_eligible_strategy"),
        )

    candidates, rejections = _eligible_candidates(
        registry=registry,
        context=context,
        economic_policy=economic_policy,
        model_scores=model_scores,
        evidence_items=evidence_items,
    )
    if not candidates:
        return _no_trade_result(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            registry=registry,
            economic_policy=economic_policy,
            context=context,
            reason_codes=("no_eligible_strategy",),
            candidate_rejections=tuple(rejections),
        )
    candidates.sort(key=_candidate_rank_key)
    score_value, definition, evidence = candidates[0]
    return _result(
        decision_id=decision_id,
        evaluated_at=evaluated_at,
        status=SelectionStatus.SELECTED,
        registry=registry,
        economic_policy=economic_policy,
        context=context,
        reason_codes=("highest_model_score_among_eligible_candidates",),
        candidate_rejections=tuple(rejections),
        selected_definition=definition,
        selected_evidence=evidence,
        model_score=score_value,
        model_revision=context.model_revision,
    )


def _global_input_reasons(
    *,
    registry: StrategyRegistry | None,
    context: SelectionContext | None,
    economic_policy: EconomicAcceptancePolicy | None,
    registry_failure_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if registry is None:
        reasons.extend(
            registry_failure_reasons or ("strategy_registry_unavailable",)
        )
    if context is None:
        reasons.append("selection_context_missing")
    else:
        reasons.extend(_context_reasons(context))
    if economic_policy is None:
        reasons.append("economic_acceptance_policy_missing")
    elif not economic_policy.owner_approved:
        reasons.append("economic_acceptance_policy_not_owner_approved")
    if (
        registry is not None
        and economic_policy is not None
        and registry.economic_policy_revision != economic_policy.revision
    ):
        reasons.append("economic_policy_revision_mismatch")
    return tuple(dict.fromkeys(reasons))


def _context_reasons(context: SelectionContext) -> tuple[str, ...]:
    return (
        _symbol_reasons(context)
        + _market_time_reasons(context)
        + _quality_reasons(context)
        + _market_measurement_reasons(context)
    )


def _symbol_reasons(context: SelectionContext) -> tuple[str, ...]:
    symbol = context.symbol
    if (
        symbol != symbol.upper()
        or symbol.count("/") != 1
        or not symbol.endswith("/USDT")
        or not symbol.split("/", maxsplit=1)[0]
    ):
        return ("spot_usdt_symbol_required",)
    return ()


def _market_time_reasons(context: SelectionContext) -> tuple[str, ...]:
    if context.market_data_as_of > context.evaluated_at:
        return ("market_data_timestamp_in_future",)
    return ()


def _quality_reasons(context: SelectionContext) -> tuple[str, ...]:
    quality = context.data_quality
    if quality is None:
        return ("data_quality_result_missing",)
    if not quality.passed or quality.status is not QualityStatus.VALID:
        return ("data_quality_not_strictly_valid",)
    if not quality.fingerprint or quality.fingerprint != context.data_fingerprint:
        return ("data_quality_fingerprint_mismatch",)
    if not context.available_features:
        return ("selection_features_missing",)
    return ()


def _market_measurement_reasons(context: SelectionContext) -> tuple[str, ...]:
    required_values = (
        ("estimated_fee_missing", context.estimated_fee_bps),
        ("estimated_slippage_missing", context.estimated_slippage_bps),
        ("spread_estimate_missing", context.spread_bps),
        ("quote_liquidity_missing", context.quote_liquidity_usdt),
    )
    return tuple(
        reason for reason, value in required_values if value is None
    )


def _eligible_candidates(
    *,
    registry: StrategyRegistry,
    context: SelectionContext,
    economic_policy: EconomicAcceptancePolicy,
    model_scores: tuple[ModelStrategyScore, ...],
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[list[tuple[Decimal, StrategyDefinition, StrategyEvidence]], list[CandidateRejection]]:
    candidates: list[tuple[Decimal, StrategyDefinition, StrategyEvidence]] = []
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
            rejections.append(
                CandidateRejection(
                    strategy_id=definition.strategy_id,
                    reasons=tuple(dict.fromkeys(reasons or ("candidate_not_eligible",))),
                )
            )
        else:
            candidates.append((score.score, definition, evidence))
    return candidates, rejections


def _candidate_rank_key(
    candidate: tuple[Decimal, StrategyDefinition, StrategyEvidence],
) -> tuple[Decimal, str, str]:
    return (-candidate[0], candidate[1].strategy_id, candidate[1].version)


def _candidate_eligibility(
    *,
    definition: StrategyDefinition,
    registry: StrategyRegistry,
    context: SelectionContext,
    economic_policy: EconomicAcceptancePolicy,
    model_scores: tuple[ModelStrategyScore, ...],
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[list[str], ModelStrategyScore | None, StrategyEvidence | None]:
    reasons = list(_definition_reasons(definition, context))
    score, score_reasons = _matching_model_score(
        definition=definition,
        context=context,
        model_scores=model_scores,
    )
    evidence, evidence_reasons = _matching_evidence(
        definition=definition,
        context=context,
        registry=registry,
        economic_policy=economic_policy,
        evidence_items=evidence_items,
    )
    reasons.extend(score_reasons)
    reasons.extend(evidence_reasons)
    return list(dict.fromkeys(reasons)), score, evidence


def _definition_reasons(
    definition: StrategyDefinition, context: SelectionContext
) -> tuple[str, ...]:
    return (
        _definition_approval_reasons(definition)
        + _definition_market_compatibility_reasons(definition, context)
        + _definition_feature_reasons(definition, context)
        + _strategy_freshness_reasons(definition, context)
        + _cost_and_liquidity_reasons(definition, context)
    )


def _definition_approval_reasons(
    definition: StrategyDefinition,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if definition.lifecycle is not StrategyLifecycle.SELECTABLE:
        reasons.append("strategy_not_selectable")
    if not definition.owner_approved:
        reasons.append("strategy_owner_approval_missing")
    licenses = (
        definition.strategy_license_status,
        definition.data_license_status,
        definition.dependency_license_status,
    )
    if any(status is not LicenseStatus.APPROVED for status in licenses):
        reasons.append("strategy_or_data_license_not_approved")
    return tuple(reasons)


def _definition_market_compatibility_reasons(
    definition: StrategyDefinition, context: SelectionContext
) -> tuple[str, ...]:
    reasons: list[str] = []
    if context.regime not in definition.eligible_regimes:
        reasons.append("strategy_regime_not_supported")
    if context.timeframe not in definition.supported_timeframes:
        reasons.append("strategy_timeframe_not_supported")
    if set(definition.prohibited_conditions).intersection(context.market_conditions):
        reasons.append("strategy_prohibited_market_condition_active")
    return tuple(reasons)


def _definition_feature_reasons(
    definition: StrategyDefinition, context: SelectionContext
) -> tuple[str, ...]:
    reasons: list[str] = []
    if context.data_schema_version != definition.data_schema_version:
        reasons.append("strategy_data_schema_mismatch")
    if context.feature_schema_version != definition.feature_schema_version:
        reasons.append("strategy_feature_schema_mismatch")
    if not set(definition.required_features).issubset(context.available_features):
        reasons.append("strategy_required_features_missing")
    return tuple(reasons)


def _strategy_freshness_reasons(
    definition: StrategyDefinition, context: SelectionContext
) -> tuple[str, ...]:
    if context.evaluated_at < context.market_data_as_of:
        return ("market_data_timestamp_in_future",)
    age = (context.evaluated_at - context.market_data_as_of).total_seconds()
    if age > definition.max_data_age_seconds:
        return ("market_data_stale_for_strategy",)
    return ()


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
    if liquidity is not None and liquidity < definition.minimum_quote_liquidity_usdt:
        reasons.append("quote_liquidity_below_strategy_minimum")
    return tuple(reasons)


def _matching_model_score(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    model_scores: tuple[ModelStrategyScore, ...],
) -> tuple[ModelStrategyScore | None, tuple[str, ...]]:
    matching = [
        score
        for score in model_scores
        if score.strategy_id == definition.strategy_id
        and score.strategy_version == definition.version
    ]
    if not matching:
        return None, _missing_score_reasons(definition, model_scores)
    if len(matching) != 1:
        return None, ("model_score_ambiguous",)
    score = matching[0]
    return score, _score_validity_reasons(score, definition, context)


def _missing_score_reasons(
    definition: StrategyDefinition, model_scores: tuple[ModelStrategyScore, ...]
) -> tuple[str, ...]:
    has_other_version = any(
        item.strategy_id == definition.strategy_id for item in model_scores
    )
    if has_other_version:
        return ("model_score_version_mismatch",)
    return ("model_score_missing",)


def _score_validity_reasons(
    score: ModelStrategyScore,
    definition: StrategyDefinition,
    context: SelectionContext,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if score.model_revision != context.model_revision:
        reasons.append("model_score_revision_mismatch")
    if score.scored_at > context.evaluated_at:
        reasons.append("model_score_timestamp_in_future")
    else:
        age = (context.evaluated_at - score.scored_at).total_seconds()
        if age > definition.max_model_score_age_seconds:
            reasons.append("model_score_stale")
    return tuple(reasons)


def _matching_evidence(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    registry: StrategyRegistry,
    economic_policy: EconomicAcceptancePolicy,
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[StrategyEvidence | None, tuple[str, ...]]:
    matching = [
        item
        for item in evidence_items
        if item.strategy_id == definition.strategy_id
        and item.strategy_version == definition.version
    ]
    if not matching:
        return None, _missing_evidence_reasons(definition, evidence_items)
    if len(matching) != 1:
        return None, ("oos_evidence_ambiguous",)
    evidence = matching[0]
    return evidence, _evidence_reasons(
        definition=definition,
        context=context,
        registry=registry,
        economic_policy=economic_policy,
        evidence=evidence,
    )


def _missing_evidence_reasons(
    definition: StrategyDefinition,
    evidence_items: tuple[StrategyEvidence, ...],
) -> tuple[str, ...]:
    has_other_version = any(
        item.strategy_id == definition.strategy_id for item in evidence_items
    )
    if has_other_version:
        return ("evidence_version_mismatch",)
    return ("oos_evidence_missing",)


def _evidence_reasons(
    *,
    definition: StrategyDefinition,
    context: SelectionContext,
    registry: StrategyRegistry,
    economic_policy: EconomicAcceptancePolicy,
    evidence: StrategyEvidence,
) -> tuple[str, ...]:
    return (
        _evidence_identity_reasons(
            definition, context, registry, economic_policy, evidence
        )
        + _evidence_metric_reasons(evidence, economic_policy)
        + _fold_evidence_reasons(evidence, economic_policy)
        + _regime_evidence_reasons(definition, context, evidence, economic_policy)
        + _evidence_cost_reasons(context, evidence)
    )


def _evidence_identity_reasons(
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
    else:
        age = (context.evaluated_at - evidence.evaluated_at).total_seconds()
        if age > economic_policy.max_evidence_age_seconds:
            reasons.append("oos_evidence_stale")
    missing_tests = set(definition.required_test_ids) - set(evidence.passed_test_ids)
    if missing_tests:
        reasons.append("required_strategy_tests_missing")
    return tuple(reasons)


def _evidence_metric_reasons(
    evidence: StrategyEvidence, economic_policy: EconomicAcceptancePolicy
) -> tuple[str, ...]:
    metrics = evidence.metrics
    reasons: list[str] = []
    if metrics.trade_count < economic_policy.minimum_trade_count:
        reasons.append("oos_insufficient_sample")
    if metrics.win_rate < economic_policy.minimum_win_rate:
        reasons.append("oos_win_rate_below_threshold")
    if metrics.max_abs_drawdown > economic_policy.maximum_abs_drawdown:
        reasons.append("oos_drawdown_above_threshold")
    if metrics.net_expectancy <= economic_policy.minimum_net_expectancy:
        reasons.append("oos_net_expectancy_not_positive")
    return tuple(reasons)


def _fold_evidence_reasons(
    evidence: StrategyEvidence, economic_policy: EconomicAcceptancePolicy
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(evidence.folds) < 3:
        reasons.append("oos_fold_count_below_minimum")
    if sum(fold.metrics.trade_count for fold in evidence.folds) != evidence.metrics.trade_count:
        reasons.append("oos_fold_sample_count_mismatch")
    for fold in evidence.folds:
        if fold.metrics.trade_count < 1:
            reasons.append("oos_fold_insufficient_sample")
        if fold.metrics.net_expectancy <= economic_policy.minimum_net_expectancy:
            reasons.append("oos_fold_expectancy_not_positive")
        if fold.metrics.max_abs_drawdown > economic_policy.maximum_abs_drawdown:
            reasons.append("oos_fold_drawdown_above_threshold")
    return tuple(dict.fromkeys(reasons))


def _regime_evidence_reasons(
    definition: StrategyDefinition,
    context: SelectionContext,
    evidence: StrategyEvidence,
    economic_policy: EconomicAcceptancePolicy,
) -> tuple[str, ...]:
    regimes = {item.regime for item in evidence.regimes}
    reasons: list[str] = []
    if regimes != set(definition.eligible_regimes):
        reasons.append("oos_regime_coverage_mismatch")
    if context.regime not in regimes:
        reasons.append("oos_current_regime_evidence_missing")
    if sum(item.metrics.trade_count for item in evidence.regimes) != evidence.metrics.trade_count:
        reasons.append("oos_regime_sample_count_mismatch")
    for item in evidence.regimes:
        if item.metrics.trade_count < 1:
            reasons.append("oos_regime_insufficient_sample")
        if item.metrics.net_expectancy <= economic_policy.minimum_net_expectancy:
            reasons.append("oos_regime_expectancy_not_positive")
        if item.metrics.max_abs_drawdown > economic_policy.maximum_abs_drawdown:
            reasons.append("oos_regime_drawdown_above_threshold")
    return tuple(dict.fromkeys(reasons))


def _evidence_cost_reasons(
    context: SelectionContext, evidence: StrategyEvidence
) -> tuple[str, ...]:
    reasons: list[str] = []
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
    return tuple(reasons)


def _no_trade_result(
    *,
    decision_id: str,
    evaluated_at: datetime | None,
    registry: StrategyRegistry | None,
    economic_policy: EconomicAcceptancePolicy | None,
    context: SelectionContext | None,
    reason_codes: tuple[str, ...],
    candidate_rejections: tuple[CandidateRejection, ...] = (),
) -> StrategySelectionResult:
    return _result(
        decision_id=decision_id,
        evaluated_at=evaluated_at,
        status=SelectionStatus.NO_TRADE,
        registry=registry,
        economic_policy=economic_policy,
        context=context,
        reason_codes=reason_codes,
        candidate_rejections=candidate_rejections,
    )


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
    actual_model_revision = model_revision
    if actual_model_revision is None and context is not None:
        actual_model_revision = context.model_revision
    selected_id, selected_version = _selected_identity(selected_definition)
    evidence_fingerprint = _selected_evidence_fingerprint(selected_evidence)
    registry_revision = registry.revision if registry is not None else None
    registry_fingerprint = registry.fingerprint if registry is not None else None
    policy_revision = economic_policy.revision if economic_policy is not None else None
    payload = {
        "decision_id": decision_id,
        "evaluated_at": actual_time,
        "status": status.value,
        "selected_strategy_id": selected_id,
        "selected_strategy_version": selected_version,
        "registry_revision": registry_revision,
        "registry_fingerprint": registry_fingerprint,
        "economic_policy_revision": policy_revision,
        "evidence_fingerprint": evidence_fingerprint,
        "model_revision": actual_model_revision,
        "model_score": model_score,
        "reason_codes": reason_codes,
        "candidate_rejections": [asdict(item) for item in candidate_rejections],
        "order_permission": False,
    }
    return StrategySelectionResult(
        decision_id=decision_id,
        evaluated_at=actual_time,
        status=status,
        selected_strategy_id=selected_id,
        selected_strategy_version=selected_version,
        registry_revision=registry_revision,
        registry_fingerprint=registry_fingerprint,
        economic_policy_revision=policy_revision,
        evidence_fingerprint=evidence_fingerprint,
        model_revision=actual_model_revision,
        model_score=model_score,
        reason_codes=reason_codes,
        candidate_rejections=candidate_rejections,
        order_permission=False,
        fingerprint=stable_fingerprint(payload),
    )


def _selected_identity(
    definition: StrategyDefinition | None,
) -> tuple[str | None, str | None]:
    if definition is None:
        return None, None
    return definition.strategy_id, definition.version


def _selected_evidence_fingerprint(
    evidence: StrategyEvidence | None,
) -> str | None:
    if evidence is None:
        return None
    return evidence.fingerprint
