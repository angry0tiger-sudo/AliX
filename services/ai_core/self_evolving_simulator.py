"""Event-driven, research-only simulator with causal next-bar execution.

Signals are targets chosen after a candle closes. A target from candle i is
executed at candle i+1 open. Residual spot exposure is liquidated at the final
candle close. Fees and slippage are charged on every executed side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SimBar:
    """OHLCV candle with explicit open and close timestamps in milliseconds."""

    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Execution costs and accounting envelope; not a strategy signal."""

    fee_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 20.0
    initial_equity: float = 10_000.0
    max_abs_exposure: float = 0.95

    def __post_init__(self) -> None:
        for name, value in (
            ("fee_bps_per_side", self.fee_bps_per_side),
            ("slippage_bps_per_side", self.slippage_bps_per_side),
            ("initial_equity", self.initial_equity),
            ("max_abs_exposure", self.max_abs_exposure),
        ):
            _require_finite_number(name, value)
        if not 0 <= self.fee_bps_per_side < 10_000:
            raise ValueError("fee_bps_per_side_out_of_range")
        if not 0 <= self.slippage_bps_per_side < 10_000:
            raise ValueError("slippage_bps_per_side_out_of_range")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity_must_be_positive")
        if not 0 < self.max_abs_exposure <= 1:
            raise ValueError("max_abs_exposure_must_be_in_(0,1]")


@dataclass(frozen=True, slots=True)
class SimFill:
    """Simulated fill; never sent to an exchange."""

    timestamp_ms: int
    side: str
    quantity: float
    reference_price: float
    execution_price: float
    fee: float
    slippage_cost: float
    target_exposure: float


@dataclass(frozen=True, slots=True)
class EquityMark:
    """Close mark for a candle, or post-liquidation mark at the final close."""

    timestamp_ms: int
    cash: float
    quantity: float
    mark_price: float
    equity: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    initial_equity: float
    final_equity: float
    net_return: float
    max_drawdown: float
    total_fees: float
    total_slippage: float
    fills: tuple[SimFill, ...]
    equity_curve: tuple[EquityMark, ...]


def _require_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name}_must_be_finite")


def validate_bars(bars: Sequence[SimBar]) -> None:
    """Reject invalid, overlapping, unordered, or non-finite market bars."""

    if not bars:
        raise ValueError("bars_empty")
    previous: SimBar | None = None
    for bar in bars:
        if not isinstance(bar, SimBar):
            raise ValueError("bar_type_invalid")
        if (
            isinstance(bar.open_time_ms, bool)
            or not isinstance(bar.open_time_ms, int)
            or isinstance(bar.close_time_ms, bool)
            or not isinstance(bar.close_time_ms, int)
        ):
            raise ValueError("bar_timestamp_invalid")
        if bar.open_time_ms < 0 or bar.close_time_ms < bar.open_time_ms:
            raise ValueError("bar_timestamp_range_invalid")
        for name, value in (
            ("open", bar.open),
            ("high", bar.high),
            ("low", bar.low),
            ("close", bar.close),
            ("volume", bar.volume),
        ):
            _require_finite_number(name, value)
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            raise ValueError("bar_price_must_be_positive")
        if bar.volume < 0:
            raise ValueError("bar_volume_must_be_non_negative")
        if bar.high < max(bar.open, bar.close, bar.low):
            raise ValueError("bar_high_inconsistent")
        if bar.low > min(bar.open, bar.close, bar.high):
            raise ValueError("bar_low_inconsistent")
        if previous is not None:
            if bar.open_time_ms <= previous.open_time_ms:
                raise ValueError("bar_timestamps_not_strictly_increasing")
            if bar.open_time_ms <= previous.close_time_ms:
                raise ValueError("bars_overlap_or_close_time_invalid")
        previous = bar


def _validate_exposures(
    target_exposures: Sequence[float], bar_count: int, config: SimulationConfig
) -> None:
    if len(target_exposures) != bar_count:
        raise ValueError("target_exposure_count_mismatch")
    for target in target_exposures:
        _require_finite_number("target_exposure", target)
        if target < 0 or target > config.max_abs_exposure:
            raise ValueError("target_exposure_outside_spot_envelope")


def _rebalance(
    cash: float,
    quantity: float,
    reference_price: float,
    timestamp_ms: int,
    target_exposure: float,
    config: SimulationConfig,
) -> tuple[float, float, SimFill | None]:
    """Rebalance a spot long/flat position at the observed execution event."""

    equity_at_reference = cash + quantity * reference_price
    if not math.isfinite(equity_at_reference) or equity_at_reference <= 0:
        raise ValueError("equity_non_positive_at_execution")
    target_quantity = target_exposure * equity_at_reference / reference_price
    delta = target_quantity - quantity
    if delta == 0:
        return cash, quantity, None
    slippage_rate = config.slippage_bps_per_side / 10_000
    fee_rate = config.fee_bps_per_side / 10_000
    execution_price = reference_price * (
        1 + slippage_rate if delta > 0 else 1 - slippage_rate
    )
    fee = abs(delta * execution_price) * fee_rate
    slippage_cost = abs(delta) * abs(execution_price - reference_price)
    next_cash = cash - delta * execution_price - fee
    if not math.isfinite(next_cash):
        raise ValueError("cash_balance_non_finite")
    fill = SimFill(
        timestamp_ms=timestamp_ms,
        side="BUY" if delta > 0 else "SELL",
        quantity=abs(delta),
        reference_price=reference_price,
        execution_price=execution_price,
        fee=fee,
        slippage_cost=slippage_cost,
        target_exposure=target_exposure,
    )
    return next_cash, target_quantity, fill


def _max_drawdown(initial_equity: float, marks: Sequence[EquityMark]) -> float:
    peak = initial_equity
    largest_drawdown = 0.0
    for mark in marks:
        peak = max(peak, mark.equity)
        if peak > 0:
            largest_drawdown = max(
                largest_drawdown, (peak - mark.equity) / peak
            )
    return largest_drawdown


def simulate(
    bars: Sequence[SimBar],
    target_exposures: Sequence[float],
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Replay post-close target exposures at the next bar open, never same-bar."""

    validate_bars(bars)
    selected_config = config if config is not None else SimulationConfig()
    _validate_exposures(target_exposures, len(bars), selected_config)
    cash = selected_config.initial_equity
    quantity = 0.0
    fills: list[SimFill] = []
    marks: list[EquityMark] = []

    for index, bar in enumerate(bars):
        if index > 0:
            cash, quantity, fill = _rebalance(
                cash,
                quantity,
                bar.open,
                bar.open_time_ms,
                target_exposures[index - 1],
                selected_config,
            )
            if fill is not None:
                fills.append(fill)
        equity = cash + quantity * bar.close
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity_non_positive_at_bar_close")
        marks.append(
            EquityMark(
                timestamp_ms=bar.close_time_ms,
                cash=cash,
                quantity=quantity,
                mark_price=bar.close,
                equity=equity,
            )
        )

    last_bar = bars[-1]
    cash, quantity, terminal_fill = _rebalance(
        cash,
        quantity,
        last_bar.close,
        last_bar.close_time_ms,
        0.0,
        selected_config,
    )
    if terminal_fill is not None:
        fills.append(terminal_fill)
    final_equity = cash
    if not math.isfinite(final_equity) or final_equity <= 0:
        raise ValueError("final_equity_non_positive")
    marks.append(
        EquityMark(
            timestamp_ms=last_bar.close_time_ms,
            cash=cash,
            quantity=quantity,
            mark_price=last_bar.close,
            equity=final_equity,
        )
    )
    total_fees = sum(fill.fee for fill in fills)
    total_slippage = sum(fill.slippage_cost for fill in fills)
    return SimulationResult(
        initial_equity=selected_config.initial_equity,
        final_equity=final_equity,
        net_return=final_equity / selected_config.initial_equity - 1,
        max_drawdown=_max_drawdown(selected_config.initial_equity, marks),
        total_fees=total_fees,
        total_slippage=total_slippage,
        fills=tuple(fills),
        equity_curve=tuple(marks),
    )
