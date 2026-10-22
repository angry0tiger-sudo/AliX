"""Deterministic dynamic sizing; no ML, I/O, exchange client, or policy writes."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import cast

from risk.contracts import (
    AccountSnapshot,
    ceil_to_step,
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

        policy = cast(RiskPolicy, policy)
        account = cast(AccountSnapshot, account)
        market = cast(MarketSnapshot, market)
        rules = cast(MarketRules, rules)
        proposal = cast(ModelRiskProposal, proposal)
        now = cast(datetime, now)

        daily_budget, trade_base = self._base_budgets(policy, account)
        if trade_base <= 0:
            return self._rejected(("risk_budget_exhausted",), daily_budget)

        exposure = self._exposure_capacity(policy, account)
        (
            exposure_reason,
            gross_limit,
            remaining_exposure,
            correlated_limit,
            remaining_correlated,
        ) = exposure
        if exposure_reason is not None:
            return self._rejected((exposure_reason,), daily_budget, trade_base)

        dynamic_budget = self._dynamic_budget(
            policy,
            market,
            proposal,
            trade_base,
            gross_limit,
            remaining_exposure,
            correlated_limit,
            remaining_correlated,
        )
        if dynamic_budget <= 0:
            return self._rejected(
                ("dynamic_risk_budget_zero",), daily_budget, dynamic_budget
            )

        return self._build_allocation(
            policy=policy,
            account=account,
            market=market,
            rules=rules,
            proposal=proposal,
            now=now,
            daily_budget=daily_budget,
            trade_budget=dynamic_budget,
        )

    def _exposure_capacity(
        self, policy: RiskPolicy, account: AccountSnapshot
    ) -> tuple[str | None, Decimal, Decimal, Decimal, Decimal]:
        gross_limit = min(
            policy.absolute_gross_exposure_cap,
            account.equity * policy.max_gross_exposure_fraction,
        )
        remaining_exposure = (
            gross_limit - account.gross_exposure - account.open_order_notional
        )
        correlated_limit = account.equity * policy.max_correlated_exposure_fraction
        remaining_correlated = correlated_limit - account.correlated_exposure
        reason = None
        if remaining_exposure <= 0:
            reason = "gross_exposure_cap_exhausted"
        elif remaining_correlated <= 0:
            reason = "correlated_exposure_cap_exhausted"
        return (
            reason,
            gross_limit,
            remaining_exposure,
            correlated_limit,
            remaining_correlated,
        )

    def _dynamic_budget(
        self,
        policy: RiskPolicy,
        market: MarketSnapshot,
        proposal: ModelRiskProposal,
        trade_base: Decimal,
        gross_limit: Decimal,
        remaining_exposure: Decimal,
        correlated_limit: Decimal,
        remaining_correlated: Decimal,
    ) -> Decimal:
        regime_factor = policy.multiplier_for(proposal.regime)
        if regime_factor is None:
            return Decimal("0")
        volatility = market.atr / market.price
        return (
            trade_base
            * proposal.calibrated_confidence
            * (Decimal("1") - proposal.uncertainty)
            * (Decimal("1") - proposal.calibration_error)
            * min(Decimal("1"), policy.reference_atr_fraction / volatility)
            * min(Decimal("1"), remaining_exposure / gross_limit)
            * min(Decimal("1"), remaining_correlated / correlated_limit)
            * regime_factor
        )

    def _base_budgets(
        self, policy: RiskPolicy, account: AccountSnapshot
    ) -> tuple[Decimal, Decimal]:
        reserved_loss = (
            account.daily_loss + account.open_position_risk + account.open_order_risk
        )
        daily_limit = min(
            account.equity * policy.daily_risk_budget_fraction,
            policy.absolute_daily_loss_cap,
        )
        daily_budget = max(Decimal("0"), daily_limit - reserved_loss)
        absolute_room = policy.absolute_daily_loss_cap - reserved_loss
        trade_base = min(
            account.equity * policy.per_trade_risk_fraction,
            daily_budget,
            absolute_room,
        )
        return daily_budget, trade_base

    def _validate_inputs(
        self,
        policy: RiskPolicy | None,
        account: AccountSnapshot | None,
        market: MarketSnapshot | None,
        rules: MarketRules | None,
        proposal: ModelRiskProposal | None,
        now: datetime | None,
    ) -> tuple[str, ...]:
        if any(
            value is None for value in (policy, account, market, rules, proposal, now)
        ):
            return ("required_risk_input_missing",)
        return self._validate_present_inputs(
            cast(RiskPolicy, policy),
            cast(AccountSnapshot, account),
            cast(MarketSnapshot, market),
            cast(MarketRules, rules),
            cast(ModelRiskProposal, proposal),
            cast(datetime, now),
        )

    def _validate_present_inputs(
        self,
        policy: RiskPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        rules: MarketRules,
        proposal: ModelRiskProposal,
        now: datetime,
    ) -> tuple[str, ...]:
        if now.tzinfo is None or now.utcoffset() is None:
            return ("decision_time_timezone_required",)
        checks = (
            self._account_policy_reason(policy, account, rules, market),
            self._freshness_reason(policy, account, market, now),
            self._proposal_reason(policy, account, market, proposal),
        )
        return tuple(reason for reason in checks if reason is not None)

    def _account_policy_reason(
        self,
        policy: RiskPolicy,
        account: AccountSnapshot,
        rules: MarketRules,
        market: MarketSnapshot,
    ) -> str | None:
        if not policy.owner_approved:
            return "risk_policy_not_owner_approved"
        if not account.reconciled or not account.orders_reconciled:
            return "account_or_orders_not_reconciled"
        if not rules.approved or rules.symbol != market.symbol:
            return "market_rules_unapproved_or_mismatched"
        return None

    def _freshness_reason(
        self,
        policy: RiskPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        now: datetime,
    ) -> str | None:
        if (
            now < market.as_of
            or (now - market.as_of).total_seconds() > policy.max_market_age_seconds
        ):
            return "market_price_stale_or_future"
        if (
            now < account.as_of
            or (now - account.as_of).total_seconds() > policy.max_account_age_seconds
        ):
            return "account_snapshot_stale_or_future"
        return None

    def _proposal_reason(
        self,
        policy: RiskPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        proposal: ModelRiskProposal,
    ) -> str | None:
        if proposal.regime != market.regime:
            return "market_or_regime_mismatch"
        if proposal.atr_snapshot != market.atr:
            return "atr_snapshot_mismatch"
        if account.daily_loss >= policy.absolute_daily_loss_cap:
            return "absolute_daily_loss_cap_hit"
        if proposal.calibrated_confidence < policy.confidence_floor:
            return "confidence_below_policy_floor"
        if proposal.uncertainty > policy.max_uncertainty:
            return "uncertainty_above_policy_limit"
        if proposal.calibration_error > policy.max_calibration_error:
            return "calibration_degraded"
        if proposal.expected_edge_after_costs <= policy.minimum_edge_after_costs:
            return "edge_after_costs_not_positive_enough"
        if market.atr / market.price > policy.max_atr_fraction:
            return "atr_spike_exceeds_policy_limit"
        if policy.multiplier_for(proposal.regime) is None:
            return "regime_not_approved"
        return None

    def _build_allocation(
        self,
        *,
        policy: RiskPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        rules: MarketRules,
        proposal: ModelRiskProposal,
        now: datetime,
        daily_budget: Decimal,
        trade_budget: Decimal,
    ) -> AllocationResult:
        stop_terms = self._stop_terms(policy, market, rules, proposal)
        if isinstance(stop_terms, str):
            return self._rejected((stop_terms,), daily_budget, trade_budget)
        stop_price, stop_distance, unit_loss = stop_terms
        quantity = self._bounded_quantity(
            policy, account, market, rules, trade_budget, unit_loss
        )
        stake = quantity * market.price
        quantity, stake, estimated_loss = self._apply_max_notional(
            quantity, market, rules, unit_loss
        )
        size_reason = self._size_rejection(
            quantity, stake, estimated_loss, trade_budget, rules
        )
        if size_reason is not None:
            return self._rejected((size_reason,), daily_budget, trade_budget)

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
        return AllocationResult(
            True, (), daily_budget, trade_budget, allocation, fingerprint
        )

    def _stop_terms(
        self,
        policy: RiskPolicy,
        market: MarketSnapshot,
        rules: MarketRules,
        proposal: ModelRiskProposal,
    ) -> tuple[Decimal, Decimal, Decimal] | str:
        stop_distance_raw = market.atr * policy.stop_atr_multiple
        raw_stop = (
            market.price - stop_distance_raw
            if proposal.side.value == "LONG"
            else market.price + stop_distance_raw
        )
        if raw_stop <= 0:
            return "stop_price_invalid"
        stop_price = (
            floor_to_step(raw_stop, rules.price_tick)
            if proposal.side.value == "LONG"
            else ceil_to_step(raw_stop, rules.price_tick)
        )
        if stop_price <= 0:
            return "stop_price_invalid"
        stop_distance = abs(market.price - stop_price)
        unit_loss = estimated_loss_per_unit(
            market.price, stop_price, policy.fee_rate, policy.slippage_rate
        )
        if unit_loss <= 0:
            return "unit_loss_invalid"
        return stop_price, stop_distance, unit_loss

    def _apply_max_notional(
        self,
        quantity: Decimal,
        market: MarketSnapshot,
        rules: MarketRules,
        unit_loss: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal]:
        if quantity * market.price > rules.max_notional:
            quantity = floor_to_step(
                rules.max_notional / market.price, rules.quantity_step
            )
        stake = quantity * market.price
        estimated_loss = quantity * unit_loss
        return quantity, stake, estimated_loss

    def _size_rejection(
        self,
        quantity: Decimal,
        stake: Decimal,
        estimated_loss: Decimal,
        trade_budget: Decimal,
        rules: MarketRules,
    ) -> str | None:
        if quantity < rules.min_quantity:
            return "quantity_below_market_minimum"
        if stake < rules.min_notional:
            return "notional_below_market_minimum"
        if quantity < rules.min_quantity or stake < rules.min_notional:
            return "market_limits_leave_no_valid_size"
        if estimated_loss > trade_budget:
            return "calculated_loss_exceeds_budget"
        return None

    def _bounded_quantity(
        self,
        policy: RiskPolicy,
        account: AccountSnapshot,
        market: MarketSnapshot,
        rules: MarketRules,
        trade_budget: Decimal,
        unit_loss: Decimal,
    ) -> Decimal:
        candidates = (
            floor_to_step(trade_budget / unit_loss, rules.quantity_step),
            floor_to_step(
                account.free_balance
                * policy.max_stake_fraction_of_free_balance
                / market.price,
                rules.quantity_step,
            ),
            floor_to_step(
                (
                    min(
                        policy.absolute_gross_exposure_cap,
                        account.equity * policy.max_gross_exposure_fraction,
                    )
                    - account.gross_exposure
                    - account.open_order_notional
                )
                / market.price,
                rules.quantity_step,
            ),
            floor_to_step(
                (
                    account.equity * policy.max_correlated_exposure_fraction
                    - account.correlated_exposure
                )
                / market.price,
                rules.quantity_step,
            ),
            rules.max_quantity,
        )
        return min(candidates)

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
