"""Tests for native intraday data and causal Donchian/TTM sizing mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ai_ensemble.hf_donchian import (
    ATR_WINDOW,
    ROUND_TRIP_COST,
    add_donchian_indicators,
    backtest_donchian_split,
    ttm_position_multiplier,
    validate_intraday_ohlcv,
)


def _candles(rows: int = 80, interval_seconds: int = 900) -> pd.DataFrame:
    stamps = pd.date_range(
        "2022-03-21T00:00:00Z",
        periods=rows,
        freq=f"{interval_seconds}s",
    )
    close = np.full(rows, 100.0, dtype=np.float64)
    high = np.full(rows, 100.2, dtype=np.float64)
    low = np.full(rows, 99.8, dtype=np.float64)
    opened = np.full(rows, 100.0, dtype=np.float64)
    volume = np.ones(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "date": stamps,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.mark.parametrize(("interval", "seconds"), [("15m", 900), ("30m", 1800)])
def test_validate_native_intraday_series(interval: str, seconds: int) -> None:
    frame = _candles(interval_seconds=seconds)
    clean = validate_intraday_ohlcv(frame, interval)
    assert len(clean) == len(frame)
    assert (clean["date"].iloc[1] - clean["date"].iloc[0]).total_seconds() == seconds


def test_validate_intraday_rejects_missing_candle() -> None:
    with pytest.raises(ValueError, match="gap_or_irregular"):
        validate_intraday_ohlcv(_candles().drop(index=20), "15m")


def test_donchian_upper_excludes_the_signal_candle() -> None:
    frame = _candles(rows=40)
    frame.loc[39, ["high", "close"]] = [104.0, 103.0]
    result = add_donchian_indicators(validate_intraday_ohlcv(frame, "15m"))
    assert result["donchian_high_20"].iloc[39] == pytest.approx(100.2)
    assert bool(result["long_breakout"].iloc[39])


def test_ttm_multiplier_reduces_high_and_increases_low_volatility() -> None:
    assert ttm_position_multiplier(4.0, 1.0) == pytest.approx(0.5)
    assert ttm_position_multiplier(0.25, 1.0) == pytest.approx(1.5)
    assert ttm_position_multiplier(1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="reference_invalid"):
        ttm_position_multiplier(float("nan"), 1.0)


def test_entry_uses_next_open_and_round_trip_cost_once() -> None:
    frame = _candles(rows=65)
    frame.loc[39, ["high", "close"]] = [104.0, 103.0]
    frame.loc[40, ["open", "high", "low", "close"]] = [103.0, 105.0, 102.5, 104.0]
    frame.loc[41, ["open", "high", "low", "close"]] = [104.0, 106.0, 103.5, 105.0]
    risk = pd.DataFrame(
        {
            "available_at": [frame["date"].iloc[39]],
            "position_multiplier": [1.2],
        }
    )
    result = backtest_donchian_split(
        frame, risk, timeframe="15m", start_index=20, end_index=len(frame)
    )
    trades = result["trades"]
    assert trades
    trade = trades[0]
    assert trade["entry_time"] == frame["date"].iloc[40].isoformat()
    assert trade["exit_reason"] == "take_profit"
    assert trade["net_return"] == pytest.approx(trade["gross_return"] - ROUND_TRIP_COST)
    assert trade["risk_multiplier"] == pytest.approx(1.2)
    assert (
        result["fixed_size_baseline"]["trade_count"]
        == result["ttm_risk_managed"]["trade_count"]
    )


def test_stop_loss_wins_when_both_intrabar_barriers_are_touched() -> None:
    frame = _candles(rows=65)
    frame.loc[39, ["high", "close"]] = [104.0, 103.0]
    frame.loc[40, ["open", "high", "low", "close"]] = [103.0, 106.0, 100.0, 103.0]
    risk = pd.DataFrame(
        {
            "available_at": [frame["date"].iloc[39]],
            "position_multiplier": [1.0],
        }
    )
    result = backtest_donchian_split(
        frame, risk, timeframe="15m", start_index=20, end_index=len(frame)
    )
    assert result["trades"][0]["exit_reason"] == "stop_loss_both_touched"


def test_risk_forecast_must_be_available_before_entry() -> None:
    frame = _candles(rows=65)
    frame.loc[39, ["high", "close"]] = [104.0, 103.0]
    frame.loc[40, ["open", "high", "low", "close"]] = [103.0, 105.0, 102.5, 104.0]
    risk = pd.DataFrame(
        {
            "available_at": [frame["date"].iloc[41]],
            "position_multiplier": [1.0],
        }
    )
    with pytest.raises(ValueError, match="no_prior_ttm_forecast"):
        backtest_donchian_split(
            frame, risk, timeframe="15m", start_index=20, end_index=len(frame)
        )


def test_atr_uses_wilder_initial_simple_average_and_recursive_smoothing() -> None:
    frame = _candles(rows=40)
    frame.loc[0, ["high", "close"]] = [114.0, 100.0]
    result = add_donchian_indicators(validate_intraday_ohlcv(frame, "15m"))
    seed = (14.2 + (ATR_WINDOW - 1) * 0.4) / ATR_WINDOW
    assert result["atr_14"].iloc[ATR_WINDOW - 1] == pytest.approx(seed)
    assert result["atr_14"].iloc[ATR_WINDOW] == pytest.approx(
        (seed * (ATR_WINDOW - 1) + 0.4) / ATR_WINDOW
    )


def test_max_drawdown_includes_open_trade_mark_to_market() -> None:
    from research.ai_ensemble.hf_donchian import _portfolio_metrics

    frame = _candles(rows=5)
    frame.loc[2, ["open", "high", "low", "close"]] = [100.0, 100.2, 99.8, 100.0]
    frame.loc[3, ["open", "high", "low", "close"]] = [80.0, 90.0, 79.8, 80.0]
    frame.loc[4, ["open", "high", "low", "close"]] = [100.0, 106.0, 99.8, 105.0]
    trade = {
        "entry_index": 2,
        "exit_index": 4,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "net_return": 0.045,
        "risk_multiplier": 1.0,
    }
    metrics = _portfolio_metrics(
        [trade],
        data=frame,
        timestamps=frame["date"],
        start_index=1,
        end_index=5,
        risk_managed=True,
    )
    assert metrics["max_drawdown"] == pytest.approx(0.01)
