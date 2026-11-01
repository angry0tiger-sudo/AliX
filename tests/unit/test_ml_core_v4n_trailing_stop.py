from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    simulate_signals,
    validate_trade_management_config,
)


def _bars(rows: list[tuple[float, float, float, float]]) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=len(rows), freq="15min")
    return DataFrame(
        {
            "date": dates,
            "open": [item[0] for item in rows],
            "high": [item[1] for item in rows],
            "low": [item[2] for item in rows],
            "close": [item[3] for item in rows],
            "volume": [10.0] * len(rows),
        }
    )


def _run(
    rows: list[tuple[float, float, float, float]],
    execution: ExecutionConfig | None = None,
) -> dict[str, Any]:
    bars = _bars(rows)
    signals = [
        {
            "signal_timestamp": bars["date"].iloc[0],
            "execution_timestamp": bars["date"].iloc[1],
            "signal": "BUY",
            "raw_signal": "BUY",
            "buy_probability": 0.80,
        },
        {
            "signal_timestamp": bars["date"].iloc[2],
            "execution_timestamp": bars["date"].iloc[3],
            "signal": "HOLD",
            "raw_signal": "HOLD",
            "buy_probability": 0.20,
        },
    ]
    management = TradeManagementConfig(
        atr_period=1,
        take_profit_atr_multiplier=3.0,
        stop_loss_atr_multiplier=2.0,
        horizon_candles=12,
        confidence_threshold=0.60,
        trailing_stop_enabled=True,
        trailing_activation_atr=1.0,
        trailing_step_atr=1.0,
    )
    return simulate_signals(
        bars,
        signals,
        execution or ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        trade_management_config=management,
        atr_values=pd.Series([10.0] * len(bars), dtype=float),
    )


def test_trailing_stop_ratchets_up_in_full_atr_steps() -> None:
    result = _run(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 110.0, 99.0, 108.0),
            (108.0, 120.0, 106.0, 118.0),
            (109.0, 111.0, 108.0, 110.0),
        ]
    )
    trade = result["trades"][0]
    assert trade["initial_stop_loss_price"] == 80.0
    assert trade["stop_loss_price"] == 110.0
    assert trade["high_water_mark"] == 120.0
    assert trade["trailing_stop_active"] is True
    assert trade["trailing_stop_steps"] == 1
    assert trade["stop_exit_type"] == "TRAILING_STOP"
    assert trade["exit_reason"] == "TRAILING_STOP_HIT"
    assert result["trailing_stop_exit_count"] == 1


def test_trailing_stop_uses_completed_bar_and_takes_effect_next_bar() -> None:
    bars = _bars(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 110.0, 95.0, 108.0),
            (102.0, 104.0, 99.0, 100.0),
            (100.0, 101.0, 98.0, 99.0),
        ]
    )
    result = _run(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 110.0, 95.0, 108.0),
            (102.0, 104.0, 99.0, 100.0),
            (100.0, 101.0, 98.0, 99.0),
        ]
    )
    trade = result["trades"][0]
    assert trade["exit_timestamp"] == bars["date"].iloc[2].isoformat()
    assert trade["stop_exit_type"] == "BREAKEVEN_STOP"
    assert trade["exit_reason"] == "BREAKEVEN_STOP_HIT"


def test_breakeven_stop_accounts_for_both_sides_costs() -> None:
    execution = ExecutionConfig(fee_bps=25.0, slippage_bps=20.0)
    quantity = execution.initial_equity / 100.0
    fee_rate = execution.fee_bps / 10_000.0
    slippage_rate = execution.slippage_bps / 10_000.0
    entry_fill = 100.0 * (1.0 + slippage_rate)
    entry_fee = quantity * entry_fill * fee_rate
    entry_slippage = quantity * (entry_fill - 100.0)
    exit_cost_rate = fee_rate * (1.0 - slippage_rate) + slippage_rate
    breakeven = (100.0 + (entry_fee + entry_slippage) / quantity) / (
        1.0 - exit_cost_rate
    )
    result = _run(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 111.0, 90.0, 108.0),
            (breakeven, breakeven + 1.0, breakeven - 1.0, breakeven),
            (breakeven, breakeven + 0.5, breakeven - 0.5, breakeven),
        ],
        execution,
    )
    trade = result["trades"][0]
    assert trade["stop_exit_type"] == "BREAKEVEN_STOP"
    assert trade["breakeven_stop_price"] == pytest.approx(breakeven)
    assert trade["net_pnl_usd"] == pytest.approx(0.0, abs=1e-7)


def test_trailing_stop_configuration_fails_closed() -> None:
    validate_trade_management_config(TradeManagementConfig(trailing_stop_enabled=False))
    with pytest.raises(ValueError, match="trailing_stop_enabled"):
        validate_trade_management_config(
            TradeManagementConfig(trailing_stop_enabled=1)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="trailing_stop_settings"):
        validate_trade_management_config(
            TradeManagementConfig(trailing_activation_atr=0.0)
        )
    with pytest.raises(ValueError, match="trailing_stop_settings"):
        validate_trade_management_config(
            TradeManagementConfig(trailing_step_atr=float("nan"))
        )
