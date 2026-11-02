from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    V4O_TIME_EXIT_CANDLES,
    evaluate_candidate_time_exit_sweep,
    simulate_signals,
    validate_time_exit_candles,
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
    *,
    time_exit_candles: int | None = None,
) -> dict[str, Any]:
    bars = _bars(rows)
    signals = [
        {
            "signal_timestamp": bars["date"].iloc[index - 1],
            "execution_timestamp": bars["date"].iloc[index],
            "signal": "BUY" if index == 1 else "HOLD",
            "raw_signal": "BUY" if index == 1 else "HOLD",
            "buy_probability": 0.80 if index == 1 else 0.20,
        }
        for index in range(1, len(bars))
    ]
    management = TradeManagementConfig(
        atr_period=1,
        take_profit_atr_multiplier=3.0,
        stop_loss_atr_multiplier=2.0,
        horizon_candles=12,
        confidence_threshold=0.60,
        trailing_stop_enabled=False,
        time_exit_candles=time_exit_candles,
        time_exit_mfe_threshold_atr=0.5,
    )
    return simulate_signals(
        bars,
        signals,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        trade_management_config=management,
        atr_values=pd.Series([10.0] * len(bars), dtype=float),
    )


def _rows(
    *, third_bar_high: float = 104.0, next_open: float = 98.0
) -> list[tuple[float, float, float, float]]:
    exit_bar = (
        (next_open, next_open + 1.0, next_open - 1.0, next_open)
        if next_open < 80.0
        else (next_open, 100.0, 97.0, 99.0)
    )
    return [
        (100.0, 100.0, 100.0, 100.0),
        (100.0, 102.0, 99.0, 101.0),
        (101.0, 103.0, 98.0, 102.0),
        (102.0, third_bar_high, 99.0, 103.0),
        exit_bar,
        (99.0, 100.0, 98.0, 99.0),
    ]


def test_time_exit_uses_excursions_and_executes_at_next_open() -> None:
    result = _run(_rows(), time_exit_candles=3)
    trade = result["trades"][0]

    assert trade["exit_reason"] == "TIME_EXIT"
    assert trade["exit_trigger_type"] == "TIME_EXIT_OPEN"
    assert trade["exit_reference_price"] == 98.0
    assert trade["time_exit_candles"] == 3
    assert trade["time_exit_evaluated"] is True
    assert trade["time_exit_pending"] is True
    assert trade["time_exit_triggered"] is True
    assert trade["mfe_atr"] == pytest.approx(0.4)
    assert trade["mae_atr"] == pytest.approx(0.2)
    assert trade["net_pnl_usd"] < 0
    assert result["time_exit_count"] == 1
    assert result["metrics"]["mean_losing_trade_mfe_atr"] == pytest.approx(0.4)
    assert result["metrics"]["mean_losing_trade_mae_atr"] == pytest.approx(0.2)
    assert result["metrics"]["losing_trades_with_positive_mfe_count"] == 1


def test_baseline_records_excursions_without_time_exit() -> None:
    result = _run(_rows())
    trade = result["trades"][0]

    assert trade["exit_reason"] == "END_OF_FORWARD_WINDOW"
    assert trade["time_exit_triggered"] is False
    assert trade["time_exit_candles"] is None
    assert trade["mfe_atr"] == pytest.approx(0.4)
    assert trade["mae_atr"] == pytest.approx(0.3)
    assert result["metrics"]["mean_mfe_atr"] == pytest.approx(0.4)
    assert result["metrics"]["mean_mae_atr"] == pytest.approx(0.3)
    assert result["metrics"]["mean_losing_trade_mfe_atr"] == pytest.approx(0.4)


def test_mfe_equal_to_threshold_does_not_trigger_exit() -> None:
    result = _run(_rows(third_bar_high=105.0), time_exit_candles=3)
    trade = result["trades"][0]

    assert trade["mfe_atr"] == pytest.approx(0.5)
    assert trade["time_exit_evaluated"] is True
    assert trade["time_exit_pending"] is False
    assert trade["exit_reason"] == "END_OF_FORWARD_WINDOW"


def test_opening_stop_gap_has_priority_over_pending_time_exit() -> None:
    result = _run(_rows(next_open=79.0), time_exit_candles=3)
    trade = result["trades"][0]

    assert trade["exit_reason"] == "SL_HIT"
    assert trade["exit_trigger_type"] == "OPEN_GAP"
    assert trade["time_exit_pending"] is True
    assert trade["time_exit_triggered"] is False


def test_time_exit_sweep_includes_baseline_and_all_requested_horizons() -> None:
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 100.2, 99.8, 100.1),
            (100.1, 100.3, 99.7, 100.2),
            (100.2, 100.4, 99.8, 100.3),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.2, 99.5, 100.0),
        ]
    )
    records = [
        {
            "row_index": index,
            "timestamp": bars["date"].iloc[index].isoformat(),
            "probabilities": (
                {"BUY": 0.8, "SELL": 0.1, "HOLD": 0.1}
                if index == 0
                else {"BUY": 0.1, "SELL": 0.1, "HOLD": 0.8}
            ),
        }
        for index in range(len(bars) - 1)
    ]
    candidate = {
        "candidate_id": "synthetic-candidate",
        "scenario": "V4H_Tighter_SL",
        "pair": "BTC/USDT",
        "timeframe": "15m",
        "feature_profile": "Baseline",
        "weighting_policy": "weight_balanced",
        "selected_parameters": {},
        "hold_precision": 0.5,
        "hold_recall": 0.5,
        "selected_hold_threshold": 0.6,
        "forward_row_count": len(records),
        "probability_records": records,
    }
    base = TradeManagementConfig(
        atr_period=1,
        take_profit_atr_multiplier=3.0,
        stop_loss_atr_multiplier=2.0,
        horizon_candles=10,
        confidence_threshold=0.60,
        trailing_stop_enabled=False,
        time_exit_mfe_threshold_atr=0.5,
    )
    results = evaluate_candidate_time_exit_sweep(
        candidate,
        bars,
        ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
        base,
        time_exit_candles=V4O_TIME_EXIT_CANDLES,
    )

    assert [item["time_exit_candles"] for item in results] == [None, 3, 6, 9]
    assert all(len(item["candidate_result"]["trades"]) == 1 for item in results)


def test_time_exit_sweep_validation_fails_closed() -> None:
    assert validate_time_exit_candles([None, 3, 6, 9]) == (None, 3, 6, 9)
    with pytest.raises(ValueError, match="sweep_values"):
        validate_time_exit_candles([])
    with pytest.raises(ValueError, match="sweep_duplicate"):
        validate_time_exit_candles([None, 3, 3])
    with pytest.raises(ValueError, match="candles_invalid"):
        validate_time_exit_candles([None, True])
    with pytest.raises(ValueError, match="candles_invalid"):
        validate_time_exit_candles([None, 0])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"time_exit_candles": True}, "time_exit_candles_invalid"),
        ({"time_exit_candles": 0}, "time_exit_candles_invalid"),
        ({"time_exit_candles": 12}, "time_exit_must_precede"),
        ({"time_exit_mfe_threshold_atr": 0.0}, "time_exit_mfe_threshold_invalid"),
        (
            {"time_exit_mfe_threshold_atr": float("nan")},
            "time_exit_mfe_threshold_invalid",
        ),
        ({"time_exit_mfe_threshold_atr": True}, "time_exit_mfe_threshold_invalid"),
    ],
)
def test_invalid_time_exit_configuration_fails_closed(
    overrides: dict[str, Any], error: str
) -> None:
    config = TradeManagementConfig(**overrides)
    with pytest.raises(ValueError, match=error):
        validate_trade_management_config(config)
