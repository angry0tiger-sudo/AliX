"""Fractional-Kelly position sizing with volatility, stop-risk and exposure caps."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite


@dataclass(frozen=True, slots=True)
class KellySizingConfig:
    kelly_fraction: float = 0.25
    max_risk_fraction: float = 0.005
    max_position_fraction: float = 0.10
    target_atr_fraction: float = 0.02
    round_trip_cost_bps: int = 50
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 1.0

    def validate(self) -> None:
        numeric = (
            self.kelly_fraction,
            self.max_risk_fraction,
            self.max_position_fraction,
            self.target_atr_fraction,
            self.take_profit_atr,
            self.stop_loss_atr,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("kelly_configuration_must_be_finite")
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction_out_of_range")
        if not 0.0 < self.max_risk_fraction <= 0.02:
            raise ValueError("kelly_max_risk_fraction_out_of_range")
        if not 0.0 < self.max_position_fraction <= 0.25:
            raise ValueError("kelly_max_position_fraction_out_of_range")
        if not 0.0 < self.target_atr_fraction <= 0.20:
            raise ValueError("kelly_target_atr_fraction_out_of_range")
        if self.round_trip_cost_bps != 50:
            raise ValueError("nobitex_round_trip_cost_must_be_50_bps")
        if self.take_profit_atr <= 0 or self.stop_loss_atr <= 0:
            raise ValueError("kelly_barrier_multipliers_must_be_positive")


@dataclass(frozen=True, slots=True)
class KellySizingResult:
    approved: bool
    reason: str
    probability: float
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    volatility_scale: float
    position_fraction: float
    quote_notional: Decimal
    base_quantity: Decimal
    stop_loss_fraction_including_cost: float


def fractional_kelly_size(
    *,
    probability_of_net_profit: float,
    equity: Decimal,
    reference_price: Decimal,
    atr: Decimal,
    config: KellySizingConfig | None = None,
) -> KellySizingResult:
    """Size a long Spot position; all bps are round-trip and deducted once."""
    active = config or KellySizingConfig()
    active.validate()
    if (
        not isfinite(probability_of_net_profit)
        or not 0.0 <= probability_of_net_profit <= 1.0
    ):
        raise ValueError("kelly_probability_out_of_range")
    if not equity.is_finite() or equity <= 0:
        raise ValueError("kelly_equity_must_be_positive")
    if not reference_price.is_finite() or reference_price <= 0:
        raise ValueError("kelly_reference_price_must_be_positive")
    if not atr.is_finite() or atr <= 0:
        raise ValueError("kelly_atr_must_be_positive")
    atr_fraction = float(atr / reference_price)
    cost_fraction = active.round_trip_cost_bps / 10_000.0
    net_win = active.take_profit_atr * atr_fraction - cost_fraction
    net_loss = active.stop_loss_atr * atr_fraction + cost_fraction
    if net_win <= 0.0 or net_loss <= 0.0:
        return KellySizingResult(
            False,
            "NET_REWARD_NOT_POSITIVE_AFTER_COST",
            probability_of_net_profit,
            0.0,
            0.0,
            0.0,
            0.0,
            Decimal("0"),
            Decimal("0"),
            net_loss,
        )
    reward_to_risk = net_win / net_loss
    full_kelly = max(
        0.0,
        probability_of_net_profit - (1.0 - probability_of_net_profit) / reward_to_risk,
    )
    fractional = full_kelly * active.kelly_fraction
    volatility_scale = min(1.0, active.target_atr_fraction / atr_fraction)
    risk_cap_position = active.max_risk_fraction / net_loss
    position_fraction = min(
        fractional * volatility_scale,
        active.max_position_fraction,
        risk_cap_position,
    )
    if position_fraction <= 0.0:
        return KellySizingResult(
            False,
            "KELLY_EDGE_NOT_POSITIVE",
            probability_of_net_profit,
            full_kelly,
            fractional,
            volatility_scale,
            0.0,
            Decimal("0"),
            Decimal("0"),
            net_loss,
        )
    notional = equity * Decimal(str(position_fraction))
    quantity = notional / reference_price
    return KellySizingResult(
        True,
        "PASS",
        probability_of_net_profit,
        full_kelly,
        fractional,
        volatility_scale,
        position_fraction,
        notional,
        quantity,
        net_loss,
    )
