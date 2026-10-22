"""Independent entry gate; a risk approval is not an order-submission command."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from risk.contracts import (
    AccountSnapshot,
    ActionType,
    MarketRules,
    MarketSnapshot,
    RiskAllocation,
    RiskPolicy,
    Side,
    estimated_loss_per_unit,
    stable_fingerprint,
)
from risk.kill_switch import KillSwitchState


@dataclass(frozen=True, slots=True)
class GateDecision:
    decision_id: str
    evaluated_at: datetime
    risk_approved: bool
    risk_reduction_allowed: bool
    order_permission: bool
    reasons: tuple[str, ...]
    estimated_loss_including_costs: Decimal
    fingerprint: str


class RiskGate:
    """Recompute deterministic risk constraints independently of the allocator."""

    def evaluate_entry(
        self,
        *,
        allocation: RiskAllocation | None,
        account: AccountSnapshot | None,
        market: MarketSnapshot | None,
        rules: MarketRules | None,
        policy: RiskPolicy | None,
        kill_switch: KillSwitchState | None,
        evaluated_at: datetime,
    ) -> GateDecision:
        identity = allocation.decision_id if allocation is not None else "unknown"
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            return self._decision(
                identity,
                evaluated_at,
                False,
                False,
                ("decision_time_timezone_required",),
                Decimal("0"),
            )
        if any(
            value is None
            for value in (allocation, account, market, rules, policy, kill_switch)
        ):
            return self._decision(
                identity,
                evaluated_at,
                False,
                False,
                ("required_gate_input_missing",),
                Decimal("0"),
            )

        assert allocation is not None
        assert account is not None
        assert market is not None
        assert rules is not None
        assert policy is not None
        assert kill_switch is not None

        estimate = (
            estimated_loss_per_unit(
                market.price,
                allocation.stop_price,
                policy.fee_rate,
                policy.slippage_rate,
            )
            * allocation.quantity
        )
        reasons = (
            self._identity_reasons(
                allocation, account, market, rules, policy, kill_switch
            )
            + self._freshness_and_cap_reasons(
                allocation, account, market, policy, evaluated_at
            )
            + self._risk_profile_reasons(allocation, account, market, policy)
            + self._market_structure_reasons(allocation, account, market, rules, policy)
            + self._loss_capacity_reasons(allocation, account, policy, estimate)
        )
        unique_reasons = tuple(dict.fromkeys(reasons))
        return self._decision(
            allocation.decision_id,
            evaluated_at,
            not unique_reasons,
            False,
            unique_reasons,
            estimate,
        )

    def _identity_reasons(
        self,
        allocation: RiskAllocation,
        account: AccountSnapshot,
        market: MarketSnapshot,
        rules: MarketRules,
        policy: RiskPolicy,
        kill_switch: KillSwitchState,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if kill_switch.latched:
            reasons.append("kill_switch_latched")
        if not policy.owner_approved:
            reasons.append("risk_policy_not_owner_approved")
        if not rules.approved:
            reasons.append("market_rules_unapproved")
        if not account.reconciled or not account.orders_reconciled:
            reasons.append("account_or_orders_not_reconciled")
        if allocation.policy_revision != policy.revision:
            reasons.append("policy_revision_mismatch")
        if allocation.account_snapshot_hash != account.source_snapshot_hash:
            reasons.append("account_snapshot_hash_mismatch")
        if allocation.market_revision != market.source_revision:
            reasons.append("market_revision_mismatch")
        if allocation.rules_revision != rules.revision:
            reasons.append("market_rules_revision_mismatch")
        if allocation.symbol != market.symbol or rules.symbol != market.symbol:
            reasons.append("market_symbol_mismatch")
        return tuple(reasons)

    def _freshness_and_cap_reasons(
        self,
        allocation: RiskAllocation,
        account: AccountSnapshot,
        market: MarketSnapshot,
        policy: RiskPolicy,
        evaluated_at: datetime,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if (
            evaluated_at < market.as_of
            or (evaluated_at - market.as_of).total_seconds()
            > policy.max_market_age_seconds
        ):
            reasons.append("market_price_stale_or_future")
        if (
            evaluated_at < account.as_of
            or (evaluated_at - account.as_of).total_seconds()
            > policy.max_account_age_seconds
        ):
            reasons.append("account_snapshot_stale_or_future")
        if (
            allocation.created_at > evaluated_at
            or (evaluated_at - allocation.created_at).total_seconds()
            > policy.max_market_age_seconds
        ):
            reasons.append("allocation_stale_or_future")
        if account.daily_loss >= policy.absolute_daily_loss_cap:
            reasons.append("absolute_daily_loss_cap_hit")
        reserved_loss = (
            account.daily_loss + account.open_position_risk + account.open_order_risk
        )
        if reserved_loss >= policy.absolute_daily_loss_cap:
            reasons.append("absolute_loss_capacity_exhausted")
        return tuple(reasons)

    def _risk_profile_reasons(
        self,
        allocation: RiskAllocation,
        account: AccountSnapshot,
        market: MarketSnapshot,
        policy: RiskPolicy,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        gross_limit = min(
            policy.absolute_gross_exposure_cap,
            account.equity * policy.max_gross_exposure_fraction,
        )
        if (
            account.gross_exposure + account.open_order_notional + allocation.stake
            > gross_limit
        ):
            reasons.append("gross_exposure_cap_exceeded")
        correlated_limit = account.equity * policy.max_correlated_exposure_fraction
        if account.correlated_exposure + allocation.stake > correlated_limit:
            reasons.append("correlated_exposure_cap_exceeded")
        if allocation.expected_edge_after_costs <= policy.minimum_edge_after_costs:
            reasons.append("edge_after_costs_not_positive_enough")
        if allocation.calibrated_confidence < policy.confidence_floor:
            reasons.append("confidence_below_policy_floor")
        if allocation.uncertainty > policy.max_uncertainty:
            reasons.append("uncertainty_above_policy_limit")
        if allocation.calibration_error > policy.max_calibration_error:
            reasons.append("calibration_degraded")
        if allocation.regime != market.regime:
            reasons.append("regime_snapshot_mismatch")
        if allocation.atr_snapshot != market.atr:
            reasons.append("atr_snapshot_mismatch")
        if market.atr / market.price > policy.max_atr_fraction:
            reasons.append("atr_spike_exceeds_policy_limit")
        return tuple(reasons)

    def _market_structure_reasons(
        self,
        allocation: RiskAllocation,
        account: AccountSnapshot,
        market: MarketSnapshot,
        rules: MarketRules,
        policy: RiskPolicy,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if allocation.side is Side.LONG and not allocation.stop_price < market.price:
            reasons.append("long_stop_not_below_entry")
        if allocation.side is Side.SHORT and not allocation.stop_price > market.price:
            reasons.append("short_stop_not_above_entry")
        if allocation.stop_price <= 0 or allocation.stop_distance != abs(
            market.price - allocation.stop_price
        ):
            reasons.append("stop_distance_invalid")
        if allocation.entry_price != market.price or allocation.stake != (
            allocation.quantity * market.price
        ):
            reasons.append("allocation_price_or_stake_mismatch")
        if (
            allocation.quantity < rules.min_quantity
            or allocation.quantity > rules.max_quantity
        ):
            reasons.append("quantity_outside_market_limits")
        if allocation.quantity % rules.quantity_step != 0:
            reasons.append("quantity_precision_invalid")
        if allocation.stop_price % rules.price_tick != 0:
            reasons.append("stop_price_precision_invalid")
        if (
            allocation.stake < rules.min_notional
            or allocation.stake > rules.max_notional
        ):
            reasons.append("notional_outside_market_limits")
        if (
            allocation.stake
            > account.free_balance * policy.max_stake_fraction_of_free_balance
        ):
            reasons.append("free_balance_limit_exceeded")
        return tuple(reasons)

    def _loss_capacity_reasons(
        self,
        allocation: RiskAllocation,
        account: AccountSnapshot,
        policy: RiskPolicy,
        estimate: Decimal,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if estimate > allocation.trade_risk_budget:
            reasons.append("recomputed_loss_exceeds_trade_budget")
        if estimate > allocation.estimated_loss_including_costs:
            reasons.append("reported_loss_understates_recomputed_loss")
        daily_limit = min(
            account.equity * policy.daily_risk_budget_fraction,
            policy.absolute_daily_loss_cap,
        )
        reserved_loss = (
            account.daily_loss + account.open_position_risk + account.open_order_risk
        )
        if estimate > daily_limit - reserved_loss:
            reasons.append("daily_risk_capacity_exceeded")
        if estimate > policy.absolute_daily_loss_cap - reserved_loss:
            reasons.append("absolute_risk_capacity_exceeded")
        return tuple(reasons)

    def evaluate_reduction(
        self, action: ActionType, decision_id: str, evaluated_at: datetime
    ) -> GateDecision:
        """Entry/news gates cannot block exits or risk-reducing intents."""
        if action not in (ActionType.EXIT, ActionType.REDUCE_RISK):
            return self._decision(
                decision_id,
                evaluated_at,
                False,
                False,
                ("not_a_risk_reducing_action",),
                Decimal("0"),
            )
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            return self._decision(
                decision_id,
                evaluated_at,
                False,
                True,
                ("decision_time_timezone_required_for_audit",),
                Decimal("0"),
            )
        return self._decision(decision_id, evaluated_at, False, True, (), Decimal("0"))

    def _decision(
        self,
        decision_id: str,
        evaluated_at: datetime,
        risk_approved: bool,
        risk_reduction_allowed: bool,
        reasons: tuple[str, ...],
        estimate: Decimal,
    ) -> GateDecision:
        fingerprint = stable_fingerprint(
            {
                "decision_id": decision_id,
                "evaluated_at": evaluated_at,
                "risk_approved": risk_approved,
                "risk_reduction_allowed": risk_reduction_allowed,
                "reasons": reasons,
                "estimated_loss": estimate,
            }
        )
        return GateDecision(
            decision_id=decision_id,
            evaluated_at=evaluated_at,
            risk_approved=risk_approved,
            risk_reduction_allowed=risk_reduction_allowed,
            order_permission=False,
            reasons=reasons,
            estimated_loss_including_costs=estimate,
            fingerprint=fingerprint,
        )
