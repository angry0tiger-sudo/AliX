"""Versioned contracts for a fail-closed risk engine.

All amounts are Decimal. The risk plane consumes reconciled snapshots and a
versioned, owner-approved policy; model outputs are proposals only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import StrEnum


def require_aware_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_timezone_required")


def require_decimal(
    name: str, value: Decimal, *, minimum: Decimal | None = None
) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name}_must_be_finite_decimal")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}_below_minimum")


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    require_decimal("value", value, minimum=Decimal("0"))
    require_decimal("step", step, minimum=Decimal("0.00000000000000000001"))
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    require_decimal("value", value, minimum=Decimal("0"))
    require_decimal("step", step, minimum=Decimal("0.00000000000000000001"))
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def estimated_loss_per_unit(
    entry_price: Decimal,
    stop_price: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> Decimal:
    """Conservatively include both-side fees and slippage in stop loss."""
    costs = fee_rate + slippage_rate
    return abs(entry_price - stop_price) + (entry_price + stop_price) * costs


def stable_fingerprint(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class Side(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class ActionType(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REDUCE_RISK = "REDUCE_RISK"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Reconciled account truth at one point in time; PnL fields are day-to-date."""

    as_of: datetime
    currency: str
    equity: Decimal
    free_balance: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    open_position_risk: Decimal
    open_order_risk: Decimal
    open_order_notional: Decimal
    gross_exposure: Decimal
    correlated_exposure: Decimal
    reconciled: bool
    orders_reconciled: bool
    source_snapshot_hash: str

    def __post_init__(self) -> None:
        require_aware_timestamp(self.as_of, "account_as_of")
        if not self.currency.strip() or not self.source_snapshot_hash.strip():
            raise ValueError("account_source_identity_missing")
        for name in (
            "equity",
            "free_balance",
            "fees_paid",
            "open_position_risk",
            "open_order_risk",
            "open_order_notional",
            "gross_exposure",
            "correlated_exposure",
        ):
            require_decimal(name, getattr(self, name), minimum=Decimal("0"))
        require_decimal("realized_pnl", self.realized_pnl)
        require_decimal("unrealized_pnl", self.unrealized_pnl)
        if self.equity <= 0:
            raise ValueError("equity_must_be_positive")
        if self.free_balance > self.equity:
            raise ValueError("free_balance_exceeds_equity")

    @property
    def net_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid

    @property
    def daily_loss(self) -> Decimal:
        return max(Decimal("0"), -self.net_pnl)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    symbol: str
    as_of: datetime
    price: Decimal
    atr: Decimal
    regime: str
    source_revision: str

    def __post_init__(self) -> None:
        require_aware_timestamp(self.as_of, "market_as_of")
        if (
            not self.symbol.strip()
            or self.symbol != self.symbol.upper()
            or not self.regime.strip()
            or not self.source_revision.strip()
        ):
            raise ValueError("market_identity_invalid")
        require_decimal(
            "market_price", self.price, minimum=Decimal("0.00000000000000000001")
        )
        require_decimal(
            "market_atr", self.atr, minimum=Decimal("0.00000000000000000001")
        )


@dataclass(frozen=True, slots=True)
class MarketRules:
    symbol: str
    revision: str
    approved: bool
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    min_notional: Decimal
    max_notional: Decimal

    def __post_init__(self) -> None:
        _validate_market_rules(self)


def _validate_market_rules(rules: MarketRules) -> None:
    if not rules.symbol.strip() or not rules.revision.strip():
        raise ValueError("market_rules_identity_missing")
    for name in (
        "price_tick",
        "quantity_step",
        "min_quantity",
        "max_quantity",
        "min_notional",
        "max_notional",
    ):
        require_decimal(name, getattr(rules, name), minimum=Decimal("0"))
    if rules.price_tick == 0 or rules.quantity_step == 0:
        raise ValueError("market_precision_step_must_be_positive")
    if rules.min_quantity <= 0 or rules.max_quantity < rules.min_quantity:
        raise ValueError("market_quantity_limits_invalid")
    if rules.min_notional <= 0 or rules.max_notional < rules.min_notional:
        raise ValueError("market_notional_limits_invalid")


@dataclass(frozen=True, slots=True)
class ModelRiskProposal:
    """Untrusted model metadata. It cannot approve an order or mutate policy."""

    decision_id: str
    model_revision: str
    side: Side
    regime: str
    atr_snapshot: Decimal
    calibrated_confidence: Decimal
    uncertainty: Decimal
    calibration_error: Decimal
    expected_edge_after_costs: Decimal

    def __post_init__(self) -> None:
        if not self.decision_id.strip() or not self.model_revision.strip():
            raise ValueError("model_revision_or_decision_id_missing")
        if not self.regime.strip():
            raise ValueError("proposal_regime_missing")
        require_decimal(
            "proposal_atr", self.atr_snapshot, minimum=Decimal("0.00000000000000000001")
        )
        for name in ("calibrated_confidence", "uncertainty", "calibration_error"):
            value = getattr(self, name)
            require_decimal(name, value, minimum=Decimal("0"))
            if value > 1:
                raise ValueError(f"{name}_out_of_range")
        require_decimal("expected_edge_after_costs", self.expected_edge_after_costs)


@dataclass(frozen=True, slots=True)
class InitialStakePolicy:
    """Versioned owner directive for the initial entry notional target."""

    revision: str = "initial-stake-10-usdt-v1"
    directive_id: str = "ALIX-OD-2026-10-23-INITIAL-STAKE-10-USDT"
    owner_approved: bool = True
    target_usdt: Decimal = Decimal("10")
    use_all_free_balance_below_target: bool = True

    def __post_init__(self) -> None:
        if not self.revision.strip() or not self.directive_id.strip():
            raise ValueError("initial_stake_policy_identity_missing")
        if not isinstance(self.owner_approved, bool):
            raise ValueError("initial_stake_policy_approval_invalid")
        if not isinstance(self.use_all_free_balance_below_target, bool):
            raise ValueError("initial_stake_policy_balance_rule_invalid")
        require_decimal(
            "initial_stake_target_usdt",
            self.target_usdt,
            minimum=Decimal("0.00000000000000000001"),
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "revision": self.revision,
                "directive_id": self.directive_id,
                "owner_approved": self.owner_approved,
                "target_usdt": self.target_usdt,
                "use_all_free_balance_below_target": (
                    self.use_all_free_balance_below_target
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Immutable policy loaded from an owner-controlled source of truth."""

    revision: str
    owner_approved: bool
    absolute_daily_loss_cap: Decimal
    absolute_gross_exposure_cap: Decimal
    daily_risk_budget_fraction: Decimal
    per_trade_risk_fraction: Decimal
    max_stake_fraction_of_free_balance: Decimal
    max_gross_exposure_fraction: Decimal
    max_correlated_exposure_fraction: Decimal
    confidence_floor: Decimal
    max_uncertainty: Decimal
    max_calibration_error: Decimal
    reference_atr_fraction: Decimal
    max_atr_fraction: Decimal
    stop_atr_multiple: Decimal
    fee_rate: Decimal
    slippage_rate: Decimal
    minimum_edge_after_costs: Decimal
    max_market_age_seconds: int
    max_account_age_seconds: int
    regime_multipliers: tuple[tuple[str, Decimal], ...]
    initial_stake_policy: InitialStakePolicy = field(default_factory=InitialStakePolicy)

    def __post_init__(self) -> None:
        if not self.revision.strip():
            raise ValueError("risk_policy_revision_missing")
        require_decimal(
            "absolute_daily_loss_cap",
            self.absolute_daily_loss_cap,
            minimum=Decimal("0.00000000000000000001"),
        )
        require_decimal(
            "absolute_gross_exposure_cap",
            self.absolute_gross_exposure_cap,
            minimum=Decimal("0.00000000000000000001"),
        )
        _validate_policy_budget_fractions(self)
        _validate_policy_quality_fractions(self)
        _validate_policy_costs_and_stops(self)
        if not isinstance(self.initial_stake_policy, InitialStakePolicy):
            raise ValueError("initial_stake_policy_missing_or_invalid")
        if self.max_market_age_seconds <= 0 or self.max_account_age_seconds <= 0:
            raise ValueError("freshness_limits_must_be_positive")
        _validate_policy_regimes(self.regime_multipliers)

    def multiplier_for(self, regime: str) -> Decimal | None:
        for known_regime, multiplier in self.regime_multipliers:
            if known_regime == regime:
                return multiplier
        return None


def _validate_policy_budget_fractions(policy: RiskPolicy) -> None:
    names = (
        "daily_risk_budget_fraction",
        "per_trade_risk_fraction",
        "max_stake_fraction_of_free_balance",
        "max_gross_exposure_fraction",
        "max_correlated_exposure_fraction",
    )
    for name in names:
        value = getattr(policy, name)
        require_decimal(name, value, minimum=Decimal("0.00000000000000000001"))
        if value > 1:
            raise ValueError(f"{name}_out_of_range")
    if policy.per_trade_risk_fraction > policy.daily_risk_budget_fraction:
        raise ValueError("per_trade_budget_exceeds_daily_budget")


def _validate_policy_quality_fractions(policy: RiskPolicy) -> None:
    for name in ("confidence_floor", "max_uncertainty", "max_calibration_error"):
        value = getattr(policy, name)
        require_decimal(name, value, minimum=Decimal("0"))
        if value > 1:
            raise ValueError(f"{name}_out_of_range")


def _validate_policy_costs_and_stops(policy: RiskPolicy) -> None:
    for name in ("reference_atr_fraction", "max_atr_fraction", "stop_atr_multiple"):
        require_decimal(
            name, getattr(policy, name), minimum=Decimal("0.00000000000000000001")
        )
    for name in ("fee_rate", "slippage_rate", "minimum_edge_after_costs"):
        require_decimal(name, getattr(policy, name), minimum=Decimal("0"))
    if policy.fee_rate >= 1 or policy.slippage_rate >= 1:
        raise ValueError("cost_rate_out_of_range")


def _validate_policy_regimes(
    regime_multipliers: tuple[tuple[str, Decimal], ...],
) -> None:
    names = [name for name, _ in regime_multipliers]
    if not names or len(names) != len(set(names)):
        raise ValueError("regime_multipliers_missing_or_duplicated")
    for name, multiplier in regime_multipliers:
        if not name.strip():
            raise ValueError("regime_name_missing")
        require_decimal(name + "_multiplier", multiplier, minimum=Decimal("0"))
        if multiplier > 1:
            raise ValueError("regime_multiplier_must_not_exceed_one")


@dataclass(frozen=True, slots=True)
class RiskAllocation:
    """Deterministic proposal after the allocator, not an order."""

    decision_id: str
    model_revision: str
    policy_revision: str
    account_snapshot_hash: str
    market_revision: str
    rules_revision: str
    symbol: str
    side: Side
    created_at: datetime
    daily_risk_budget: Decimal
    trade_risk_budget: Decimal
    quantity: Decimal
    stake: Decimal
    entry_price: Decimal
    stop_price: Decimal
    stop_distance: Decimal
    estimated_loss_including_costs: Decimal
    regime: str
    atr_snapshot: Decimal
    calibrated_confidence: Decimal
    uncertainty: Decimal
    calibration_error: Decimal
    expected_edge_after_costs: Decimal
    fingerprint: str
    initial_stake_policy_revision: str = "initial-stake-10-usdt-v1"
    initial_stake_policy_fingerprint: str = field(
        default_factory=lambda: InitialStakePolicy().fingerprint
    )

    def __post_init__(self) -> None:
        require_aware_timestamp(self.created_at, "allocation_created_at")
        _validate_allocation_identity(self)
        _validate_allocation_amounts(self)
        _validate_allocation_size(self)
        _validate_allocation_quality(self)
        require_decimal("expected_edge_after_costs", self.expected_edge_after_costs)


def _validate_allocation_identity(allocation: RiskAllocation) -> None:
    values = (
        allocation.decision_id,
        allocation.model_revision,
        allocation.policy_revision,
        allocation.account_snapshot_hash,
        allocation.market_revision,
        allocation.rules_revision,
        allocation.symbol,
        allocation.regime,
        allocation.fingerprint,
        allocation.initial_stake_policy_revision,
        allocation.initial_stake_policy_fingerprint,
    )
    if not all(value.strip() for value in values):
        raise ValueError("allocation_identity_missing")


def _validate_allocation_amounts(allocation: RiskAllocation) -> None:
    for name in (
        "daily_risk_budget",
        "trade_risk_budget",
        "quantity",
        "stake",
        "entry_price",
        "stop_price",
        "stop_distance",
        "estimated_loss_including_costs",
        "atr_snapshot",
    ):
        require_decimal(name, getattr(allocation, name), minimum=Decimal("0"))


def _validate_allocation_size(allocation: RiskAllocation) -> None:
    if (
        allocation.trade_risk_budget <= 0
        or allocation.quantity <= 0
        or allocation.stake <= 0
    ):
        raise ValueError("allocation_size_must_be_positive")
    if (
        allocation.entry_price <= 0
        or allocation.stop_price <= 0
        or allocation.stop_distance <= 0
    ):
        raise ValueError("allocation_stop_or_price_invalid")
    if allocation.estimated_loss_including_costs > allocation.trade_risk_budget:
        raise ValueError("allocation_loss_exceeds_budget")


def _validate_allocation_quality(allocation: RiskAllocation) -> None:
    for name in ("calibrated_confidence", "uncertainty", "calibration_error"):
        value = getattr(allocation, name)
        require_decimal(name, value, minimum=Decimal("0"))
        if value > 1:
            raise ValueError(f"{name}_out_of_range")


@dataclass(frozen=True, slots=True)
class AllocationResult:
    approved: bool
    reasons: tuple[str, ...]
    daily_risk_budget: Decimal
    trade_risk_budget: Decimal
    allocation: RiskAllocation | None
    fingerprint: str

    def __post_init__(self) -> None:
        require_decimal(
            "result_daily_budget", self.daily_risk_budget, minimum=Decimal("0")
        )
        require_decimal(
            "result_trade_budget", self.trade_risk_budget, minimum=Decimal("0")
        )
        if self.approved != (self.allocation is not None):
            raise ValueError("allocation_result_state_inconsistent")
        if self.approved and self.reasons:
            raise ValueError("approved_allocation_has_rejection_reasons")
        if not self.fingerprint.strip():
            raise ValueError("allocation_result_fingerprint_missing")
