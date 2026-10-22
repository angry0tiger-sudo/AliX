"""Deterministic dynamic sizing; no ML, I/O, exchange client, or policy writes."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from risk.contracts import (
    AccountSnapshot,
    AllocationResult,
    MarketRules,
    MarketSnapshot,
    ModelRiskProposal,
    RiskAllocation,
    RiskPolicy,
    estimated_loss_per_unit,
    floor_to_step,
    stable_fingerprint,
)


class DynamicRiskAllocator:
    """Convert a proposal to a bounded stop/size after fail-closed validation."""

    def allocate(
        self,
        *,
        policy: RiskPolicy | None,
        account: AccountSnapshot | None,
        market: MarketSnapshot | None,
        rules: MarketRules | None,
        proposal: ModelRiskProposal | None,
        now: datetime | None,
    ) -> AllocationResult:
        reasons = self._validate_inputs(policy, account, market, rules, proposal, now)
        if reasons:
            return self._rejected(reasons)

        assert policy is not None
        assert account is not None
        assert market is not None
        assert rules is not None
        assert proposal is not None
        assert now is not None

        daily_limit = min(
            account.equity * policy.daily_risk_budget_fraction,
            policy.absolute_daily_loss_cap,
        )
        reserved_loss = account.daily_loss + account.open_position_risk + account.open_order_risk
        daily_budget = max(Decimal("0"), daily_limit - reserved_loss)
        remaining_daily = daily_budget
        absolute_room = policy.absolute_daily_loss_cap - reserved_loss
        trade_base = min(
            account.equity * policy.per_trade_risk_fraction,
            remaining_daily,
            absolute_room,
        )
        if trade_base <= 0:
            return self._rejected(("risk_budget_exhausted",), daily_budget)

        gross_limit = min(
            policy.absolute_gross_exposure_cap,
            account.equity * policy.max_gross_exposure_fraction,
        )
        current_exposure = account.gross_exposure + account.open_order_notional
        remaining_exposure = gross_limit - current_exposure
        correlated_limit = account.equity * policy.max_correlated_exposure_fraction
        remaining_correlated = correlated_limit - account.correlated_exposure
        if remaining_exposure <= 0:
            return self._rejected(("gross_exposure_cap_exhausted",), daily_budget, trade_base)
        if remaining_correlated <= 0:
            return self._rejected(("correlated_exposure_cap_exhausted",), daily_budget, trade_base)

        confidence_factor = proposal.calibrated_confidence
        uncertainty_factor = Decimal("1") - proposal.uncertainty
        calibration_factor = Decimal("1") - proposal.calibration_error
        volatility = market.atr / market.price
        volatility_factor = min(Decimal("1"), policy.reference_atr_fraction / volatility)
        exposure_factor = min(Decimal("1"), remaining_exposure / gross_limit)
        correlation_factor = min(Decimal("1"), remaining_correlated / correlated_limit)
        regime_factor = policy.multiplier_for(proposal.regime)
        if regime_factor is None:
            return self._rejected(("regime_not_approved",), daily_budget, trade_base)

        trade_budget = (
            trade_base
            * confidence_factor
            * uncertainty_factor
            * calibration_factor
            * volatility_factor
            * exposure_factor
            * correlation_factor
            * regime_factor
        )
        if trade_budget <= 0:
            return self._rejected(("dynamic_risk_budget_zero",), daily_budget, trade_budget)

        stop_distance_raw = market.atr * policy.stop_atr_multiple
        raw_stop = (
            market.price - stop_distance_raw
            if proposal.side.value == "LONG"
            else market.price + stop_distance_raw
        )
        if raw_stop <= 0:
            return self._rejected(("stop_price_invalid",), daily_budget, trade_budget)
        stop_price = (
            floor_to_step(raw_stop, rules.price_tick)
            if proposal.side.value == "LONG"
            else ceil_to_step(raw_stop, rules.price_tick)
        )
        if stop_price <= 0:
            return self._rejected(("stop_price_invalid",), daily_budget, trade_budget)
        stop_distance = abs(market.price - stop_price)
        unit_loss = estimated_loss_per_unit(
            market.price, stop_price, policy.fee_rate, policy.slippage_rate
        )
        if unit_loss <= 0:
            return self._rejected(("unit_loss_invalid",), daily_budget, trade_budget)

        risk_quantity = floor_to_step(trade_budget / unit_loss, rules.quantity_step)
        cash_quantity = floor_to_step(
            account.free_balance * policy.max_stake_fraction_of_free_balance / market.price,
            rules.quantity_step,
        )
        exposure_quantity = floor_to_step(remaining_exposure / market.price, rules.quantity_step)
        correlation_quantity = floor_to_step(remaining_correlated / market.price, rules.quantity_step)
        market_quantity = min(
            risk_quantity,
            cash_quantity,
            exposure_quantity,
            correlation_quantity,
            rules.max_quantity,
        )
        quantity = floor_to_step(market_quantity, rules.quantity_step)
        stake = quantity * market.price
        estimated_loss = quantity * unit_loss

        if quantity < rules.min_quantity:
            return self._rejected(("quantity_below_market_minimum",), daily_budget, trade_budget)
        if stake < rules.min_notional:
            return self._rejected(("notional_below_market_minimum",), daily_budget, trade_budget)
        if stake > rules.max_notional:
            quantity = floor_to_step(rules.max_notional / market.price, rules.quantity_step)
            stake = quantity * market.price
            estimated_loss = quantity * unit_loss
        if quantity < rules.min_quantity or stake < rules.min_notional:
            return self._rejected(("market_limits_leave_no_valid_size",), daily_budget, trade_budget)
        if estimated_loss > trade_budget:
            return self._rejected(("calculated_loss_exceeds_budget",), daily_budget, trade_budget)

        fingerprint = stable_fingerprint(
            {
                "decision_id": proposal.decision_id,
                "model_revision": proposal.model_revision,
                "policy_revision": policy.revision,
                "account_hash": account.source_snapshot_hash,
                "market_revision": market.source_revision,
                "rules_revision": rules.revision,
                "daily_budget": daily_budget,
                "trade_budget": trade_budget,
                "quantity": quantity,
                "stake": stake,
                "stop": stop_price,
            }
        )
        allocation = RiskAllocation(
            decision_id=proposal.decision_id,
            model_revision=proposal.model_revision,
            policy_revision=policy.revision,
            account_snapshot_hash=account.source_snapshot_hash,
            market_revision=market.source_revision,
            rules_revision=rules.revision,
            symbol=market.symbol,
            side=proposal.side,
            created_at=now,
            daily_risk_budget=daily_budget,
            trade_risk_budget=trade_budget,
            quantity=quantity,
            stake=stake,
            entry_price=market.price,
            stop_price=stop_price,
            stop_distance=stop_distance,
            estimated_loss_including_costs=estimated_loss,
            regime=proposal.regime,
            atr_snapshot=market.atr,
            calibrated_confidence=proposal.calibrated_confidence,
            uncertainty=proposal.uncertainty,
            calibration_error=proposal.calibration_error,
            expected_edge_after_costs=proposal.expected_edge_after_costs,
            fingerprint=fingerprint,
        )
        return AllocationResult(True, (), daily_budget, trade_budget, allocation, fingerprint)

    def _validate_inputs(
        self,
        policy: RiskPolicy | None,
        account: AccountSnapshot | None,
        market: MarketSnapshot | None,
        rules: MarketRules | None,
        proposal: ModelRiskProposal | None,
        now: datetime | None,
    ) -> tuple[str, ...]:
        if any(value is None for value in (policy, account, market, rules, proposal, now)):
            return ("required_risk_input_missing",)
        assert policy is not None
        assert account is not None
        assert market is not None
        assert rules is not None
        assert proposal is not None
        assert now is not None
        if now.tzinfo is None or now.utcoffset() is None:
            return ("decision_time_timezone_required",)
        if not policy.owner_approved:
            return ("risk_policy_not_owner_approved",)
        if not account.reconciled or not account.orders_reconciled:
            return ("account_or_orders_not_reconciled",)
        if not rules.approved or rules.symbol != market.symbol:
            return ("market_rules_unapproved_or_mismatched",)
        if now < market.as_of or (now - market.as_of).total_seconds() > policy.max_market_age_seconds:
            return ("market_price_stale_or_future",)
        if now < account.as_of or (now - account.as_of).total_seconds() > policy.max_account_age_seconds:
            return ("account_snapshot_stale_or_future",)
        if market.symbol != rules.symbol or proposal.regime != market.regime:
            return ("market_or_regime_mismatch",)
        if proposal.atr_snapshot != market.atr:
            return ("atr_snapshot_mismatch",)
        if account.daily_loss >= policy.absolute_daily_loss_cap:
            return ("absolute_daily_loss_cap_hit",)
        if proposal.calibrated_confidence < policy.confidence_floor:
            return ("confidence_below_policy_floor",)
        if proposal.uncertainty > policy.max_uncertainty:
            return ("uncertainty_above_policy_limit",)
        if proposal.calibration_error > policy.max_calibration_error:
            return ("calibration_degraded",)
        if proposal.expected_edge_after_costs <= policy.minimum_edge_after_costs:
            return ("edge_after_costs_not_positive_enough",)
        if market.atr / market.price > policy.max_atr_fraction:
            return ("atr_spike_exceeds_policy_limit",)
        return ()

    def _rejected(
        self,
        reasons: tuple[str, ...],
        daily_budget: Decimal = Decimal("0"),
        trade_budget: Decimal = Decimal("0"),
    ) -> AllocationResult:
        fingerprint = stable_fingerprint(
            {
                "approved": False,
                "reasons": reasons,
                "daily_budget": daily_budget,
                "trade_budget": trade_budget,
            }
        )
        return AllocationResult(
            approved=False,
            reasons=reasons,
            daily_risk_budget=max(Decimal("0"), daily_budget),
            trade_risk_budget=max(Decimal("0"), trade_budget),
            allocation=None,
            fingerprint=fingerprint,
        )
