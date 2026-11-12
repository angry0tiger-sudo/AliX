"""Causal Donchian breakout research with an as-of TTM risk-size overlay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TIMEFRAME_SECONDS = {"15m": 900, "30m": 1800}
DONCHIAN_WINDOW = 20
ATR_WINDOW = 14
STOP_ATR_MULTIPLE = 2.0
TAKE_PROFIT_ATR_MULTIPLE = 4.0
ROUND_TRIP_COST = 0.005
FIXED_POSITION_FRACTION = 0.05
TTM_MULTIPLIER_MIN = 0.5
TTM_MULTIPLIER_MAX = 1.5


def validate_intraday_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Validate a native, continuous Nobitex candle series without filling gaps."""
    if interval not in TIMEFRAME_SECONDS:
        raise ValueError("hf_timeframe_out_of_policy")
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"hf_ohlcv_missing_columns:{sorted(missing)}")
    if len(frame) < DONCHIAN_WINDOW + ATR_WINDOW + 3:
        raise ValueError("hf_ohlcv_history_too_short")

    clean = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    timestamps = pd.to_datetime(clean["date"], utc=True, errors="raise")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("hf_timestamps_missing_or_duplicated")
    expected_delta = pd.Timedelta(seconds=TIMEFRAME_SECONDS[interval])
    if not bool((timestamps.diff().dropna() == expected_delta).all()):
        raise ValueError("hf_native_candle_gap_or_irregular_interval")
    clean["date"] = timestamps

    for column in ("open", "high", "low", "close", "volume"):
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    values = clean[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("hf_ohlcv_non_finite_values")
    if np.any(values[:, :4] <= 0.0) or np.any(values[:, 4] < 0.0):
        raise ValueError("hf_ohlcv_values_out_of_range")
    opened, high, low, close = (values[:, index] for index in range(4))
    if (
        np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
        or np.any(high < low)
    ):
        raise ValueError("hf_ohlc_invariant_failed")
    return clean.reset_index(drop=True)


def add_donchian_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Build prior-bar Donchian bands and Wilder ATR(14) without look-ahead."""
    result = frame.copy()
    prior_high = result["high"].shift(1)
    prior_low = result["low"].shift(1)
    result["donchian_high_20"] = prior_high.rolling(
        DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW
    ).max()
    result["donchian_low_20"] = prior_low.rolling(
        DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW
    ).min()

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_values = true_range.to_numpy(dtype=np.float64)
    atr_values = np.full(len(result), np.nan, dtype=np.float64)
    if len(result) >= ATR_WINDOW:
        atr_values[ATR_WINDOW - 1] = float(np.mean(true_range_values[:ATR_WINDOW]))
        for index in range(ATR_WINDOW, len(result)):
            atr_values[index] = (
                atr_values[index - 1] * (ATR_WINDOW - 1) + true_range_values[index]
            ) / ATR_WINDOW
    result["atr_14"] = atr_values
    result["long_breakout"] = result["close"] > result["donchian_high_20"]
    result["opposite_breakout"] = result["close"] < result["donchian_low_20"]
    return result


def ttm_position_multiplier(
    forecast_variance: float, training_reference: float
) -> float:
    """Map a 24h TTM variance forecast to bounded inverse-volatility sizing."""
    if (
        not np.isfinite(forecast_variance)
        or forecast_variance <= 0.0
        or not np.isfinite(training_reference)
        or training_reference <= 0.0
    ):
        raise ValueError("hf_ttm_variance_reference_invalid")
    multiplier = np.sqrt(training_reference / forecast_variance)
    return float(np.clip(multiplier, TTM_MULTIPLIER_MIN, TTM_MULTIPLIER_MAX))


def _risk_multiplier_at(
    entry_timestamp: pd.Timestamp,
    forecast_times_ns: np.ndarray,
    forecast_multipliers: np.ndarray,
) -> float:
    entry_ns = int(entry_timestamp.value)
    index = int(np.searchsorted(forecast_times_ns, entry_ns, side="right") - 1)
    if index < 0:
        raise ValueError("hf_entry_has_no_prior_ttm_forecast")
    return float(forecast_multipliers[index])


def _find_trade_exit(
    data: pd.DataFrame,
    *,
    entry_index: int,
    end_index: int,
    stop_price: float,
    target_price: float,
) -> tuple[int, float, str]:
    """Apply adverse-first intrabar barriers and next-open opposite-signal exits."""
    for index in range(entry_index, end_index):
        opened = float(data.iloc[index]["open"])
        high = float(data.iloc[index]["high"])
        low = float(data.iloc[index]["low"])

        if opened <= stop_price:
            return index, opened, "stop_loss_gap"
        if opened >= target_price:
            return index, target_price, "take_profit_gap"
        if low <= stop_price and high >= target_price:
            return index, stop_price, "stop_loss_both_touched"
        if low <= stop_price:
            return index, stop_price, "stop_loss"
        if high >= target_price:
            return index, target_price, "take_profit"
        if bool(data.iloc[index]["opposite_breakout"]) and index + 1 < end_index:
            return index + 1, float(data.iloc[index + 1]["open"]), "opposite_breakout"
    return end_index - 1, float(data.iloc[end_index - 1]["close"]), "split_end"


def _simulate_trade_stream(
    data: pd.DataFrame,
    risk_forecasts: pd.DataFrame,
    *,
    timeframe: str,
    start_index: int,
    end_index: int,
) -> list[dict[str, Any]]:
    forecast_times = (
        pd.to_datetime(risk_forecasts["available_at"], utc=True, errors="raise")
        .astype("int64")
        .to_numpy(dtype=np.int64)
    )
    forecast_multipliers = risk_forecasts["position_multiplier"].to_numpy(
        dtype=np.float64
    )
    if (
        forecast_times.size == 0
        or not np.all(np.diff(forecast_times) > 0)
        or not np.isfinite(forecast_multipliers).all()
        or np.any(forecast_multipliers < TTM_MULTIPLIER_MIN)
        or np.any(forecast_multipliers > TTM_MULTIPLIER_MAX)
    ):
        raise ValueError("hf_ttm_forecast_series_invalid")

    trades: list[dict[str, Any]] = []
    signal_index = start_index
    while signal_index + 1 < end_index:
        signal = data.iloc[signal_index]
        atr = float(signal["atr_14"])
        upper = float(signal["donchian_high_20"])
        if (
            not bool(signal["long_breakout"])
            or not np.isfinite(atr)
            or atr <= 0.0
            or not np.isfinite(upper)
        ):
            signal_index += 1
            continue

        entry_index = signal_index + 1
        entry = float(data.iloc[entry_index]["open"])
        stop_price = entry - STOP_ATR_MULTIPLE * atr
        target_price = entry + TAKE_PROFIT_ATR_MULTIPLE * atr
        if entry <= 0.0 or stop_price <= 0.0:
            signal_index += 1
            continue

        entry_timestamp = pd.Timestamp(data.iloc[entry_index]["date"])
        multiplier = _risk_multiplier_at(
            entry_timestamp, forecast_times, forecast_multipliers
        )
        exit_index, exit_price, exit_reason = _find_trade_exit(
            data,
            entry_index=entry_index,
            end_index=end_index,
            stop_price=stop_price,
            target_price=target_price,
        )
        gross_return = exit_price / entry - 1.0
        net_return = gross_return - ROUND_TRIP_COST
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("hf_trade_net_return_invalid")
        trades.append(
            {
                "timeframe": timeframe,
                "entry_time": entry_timestamp.isoformat(),
                "exit_time": pd.Timestamp(data.iloc[exit_index]["date"]).isoformat(),
                "entry_index": entry_index,
                "exit_index": exit_index,
                "entry_price": entry,
                "exit_price": exit_price,
                "atr_at_signal": atr,
                "stop_price": stop_price,
                "take_profit_price": target_price,
                "exit_reason": exit_reason,
                "gross_return": gross_return,
                "round_trip_cost": ROUND_TRIP_COST,
                "net_return": net_return,
                "risk_multiplier": multiplier,
                "ttm_position_fraction": FIXED_POSITION_FRACTION * multiplier,
            }
        )
        signal_index = max(exit_index, signal_index + 1)
    return trades


def _portfolio_metrics(
    trades: list[dict[str, Any]],
    *,
    data: pd.DataFrame,
    timestamps: pd.Series,
    start_index: int,
    end_index: int,
    risk_managed: bool,
) -> dict[str, float | int | None]:
    if not trades:
        return {
            "trade_count": 0,
            "trades_per_year": 0.0,
            "win_rate": None,
            "net_profit_factor": None,
            "max_drawdown": 0.0,
            "net_return": 0.0,
            "average_position_fraction": None,
            "average_risk_multiplier": None,
        }

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    gains = 0.0
    losses = 0.0
    wins = 0
    position_fractions: list[float] = []
    multipliers: list[float] = []
    for trade in trades:
        multiplier = float(trade["risk_multiplier"])
        position_fraction = (
            FIXED_POSITION_FRACTION * multiplier
            if risk_managed
            else FIXED_POSITION_FRACTION
        )
        entry_price = float(trade["entry_price"])
        entry_index = int(trade["entry_index"])
        exit_index = int(trade["exit_index"])
        if (
            entry_price <= 0.0
            or not 0 <= entry_index <= exit_index < len(data)
            or not np.isfinite(entry_price)
        ):
            raise ValueError("hf_portfolio_trade_mark_to_market_invalid")

        # Mark the open position at every candle close; use the actual fill on
        # the exit candle. Position sizing is fixed at entry for this trade.
        equity_before_trade = equity
        for candle_index in range(entry_index, exit_index + 1):
            mark_price = (
                float(trade["exit_price"])
                if candle_index == exit_index
                else float(data.iloc[candle_index]["close"])
            )
            marked_equity = equity_before_trade * (
                1.0 + position_fraction * (mark_price / entry_price - 1.0)
            )
            if not np.isfinite(marked_equity) or marked_equity <= 0.0:
                raise ValueError("hf_portfolio_equity_depleted")
            peak = max(peak, marked_equity)
            max_drawdown = max(max_drawdown, 1.0 - marked_equity / peak)

        pnl = equity_before_trade * position_fraction * float(trade["net_return"])
        if pnl > 0.0:
            gains += pnl
            wins += 1
        elif pnl < 0.0:
            losses -= pnl
        equity = equity_before_trade + pnl
        if equity <= 0.0:
            raise ValueError("hf_portfolio_equity_depleted")
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        position_fractions.append(position_fraction)
        multipliers.append(multiplier if risk_managed else 1.0)

    duration_years = max(
        (timestamps.iloc[end_index - 1] - timestamps.iloc[start_index]).total_seconds()
        / (365.25 * 24.0 * 3600.0),
        1.0 / 365.25,
    )
    profit_factor = gains / losses if losses > 0.0 else None
    return {
        "trade_count": len(trades),
        "trades_per_year": float(len(trades) / duration_years),
        "win_rate": float(wins / len(trades)),
        "net_profit_factor": float(profit_factor)
        if profit_factor is not None
        else None,
        "max_drawdown": float(max_drawdown),
        "net_return": float(equity - 1.0),
        "average_position_fraction": float(np.mean(position_fractions)),
        "average_risk_multiplier": float(np.mean(multipliers)),
    }


def backtest_donchian_split(
    ohlcv: pd.DataFrame,
    risk_forecasts: pd.DataFrame,
    *,
    timeframe: str,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    """Run a long-only, next-open Donchian breakout on one isolated time split."""
    data = add_donchian_indicators(validate_intraday_ohlcv(ohlcv, timeframe))
    if not 0 <= start_index < end_index <= len(data):
        raise ValueError("hf_split_indices_invalid")
    if end_index - start_index < DONCHIAN_WINDOW + ATR_WINDOW + 3:
        raise ValueError("hf_split_has_insufficient_candles")

    trades = _simulate_trade_stream(
        data,
        risk_forecasts,
        timeframe=timeframe,
        start_index=start_index,
        end_index=end_index,
    )
    timestamps = data["date"]
    fixed = _portfolio_metrics(
        trades,
        data=data,
        timestamps=timestamps,
        start_index=start_index,
        end_index=end_index,
        risk_managed=False,
    )
    managed = _portfolio_metrics(
        trades,
        data=data,
        timestamps=timestamps,
        start_index=start_index,
        end_index=end_index,
        risk_managed=True,
    )
    if fixed["trade_count"] != managed["trade_count"]:
        raise ValueError("hf_risk_overlay_changed_trade_count")
    return {
        "fixed_size_baseline": fixed,
        "ttm_risk_managed": managed,
        "same_trade_count": True,
        "trades": trades,
    }
