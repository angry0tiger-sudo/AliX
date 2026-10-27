"""Causal institutional-style teacher labels for offline imitation research.

All signals use completed candles only. This is a deterministic heuristic teacher,
not a claim that institutional order flow can be recovered from OHLCV alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    missing = set(_REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"missing_ohlcv_columns:{','.join(sorted(missing))}")
    frame = data.loc[:, _REQUIRED_COLUMNS].copy()
    numeric = frame.loc[:, ("open", "high", "low", "close", "volume")].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("ohlcv_contains_non_finite_values")
    timestamp = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp, unit="ms", utc=True, errors="coerce")
    else:
        timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("timestamps_must_be_valid")
    frame["timestamp"] = timestamp
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("ohlc_prices_must_be_positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume_must_be_non_negative")
    if (
        not frame["timestamp"].is_monotonic_increasing
        or frame["timestamp"].duplicated().any()
    ):
        raise ValueError("timestamps_must_be_strictly_increasing")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high_below_ohlc_component")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low_above_ohlc_component")
    return frame


def _higher_timeframe_bias(frame: pd.DataFrame, rule: str, ema_span: int) -> pd.Series:
    indexed = frame.set_index("timestamp")
    close = (
        indexed["close"].resample(rule, label="right", closed="right").last().dropna()
    )
    fast = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    slow = close.ewm(span=ema_span * 3, adjust=False, min_periods=ema_span * 3).mean()
    bias = pd.Series(
        np.where(fast > slow, 1, np.where(fast < slow, -1, 0)), index=close.index
    )
    # At each minute, use only the most recently completed higher-timeframe bar.
    return (
        bias.reindex(indexed.index, method="ffill")
        .fillna(0)
        .astype(np.int8)
        .reset_index(drop=True)
    )


def _smc_features(
    frame: pd.DataFrame, swing_window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    # Shift one candle so the current candle cannot define its own break level.
    prior_high = high.rolling(swing_window, min_periods=swing_window).max().shift(1)
    prior_low = low.rolling(swing_window, min_periods=swing_window).min().shift(1)
    bos_up = close > prior_high
    bos_down = close < prior_low

    # Three-candle fair-value gaps, confirmed only after the third candle closes.
    fvg_bull = low > high.shift(2)
    fvg_bear = high < low.shift(2)
    # A simple last-opposing-candle proxy for order-block zones; not exchange order-book data.
    previous_red = close.shift(1) < frame["open"].shift(1)
    previous_green = close.shift(1) > frame["open"].shift(1)
    bull_ob_low = frame["low"].shift(1).where(previous_red).ffill()
    bull_ob_high = frame["high"].shift(1).where(previous_red).ffill()
    bear_ob_low = frame["low"].shift(1).where(previous_green).ffill()
    bear_ob_high = frame["high"].shift(1).where(previous_green).ffill()
    pullback_bull = ((low <= prior_high) & (close >= prior_high)) | (
        fvg_bull & (close <= high.shift(2) + (low - high.shift(2)) * 0.5)
    )
    pullback_bear = ((high >= prior_low) & (close <= prior_low)) | (
        fvg_bear & (close >= low.shift(2) + (high - low.shift(2)) * 0.5)
    )
    pullback_bull |= close.between(bull_ob_low, bull_ob_high)
    pullback_bear |= close.between(bear_ob_low, bear_ob_high)
    return (
        bos_up.fillna(False) | pullback_bull.fillna(False),
        bos_down.fillna(False) | pullback_bear.fillna(False),
        prior_high,
    )


def _volume_confirmations(
    frame: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    # Rolling VWAP avoids using future data or requiring a session-boundary assumption.
    volume_sum = frame["volume"].rolling(window, min_periods=window).sum()
    vwap = (typical * frame["volume"]).rolling(
        window, min_periods=window
    ).sum() / volume_sum.replace(0, np.nan)
    # Rolling modal price-bin approximation to volume profile, using only trailing data.
    rolling_poc = frame["close"].rolling(window, min_periods=window).median()
    buy_confirm = (frame["close"] >= vwap) & (frame["close"] >= rolling_poc)
    sell_confirm = (frame["close"] <= vwap) & (frame["close"] <= rolling_poc)
    return buy_confirm.fillna(False), sell_confirm.fillna(False)


def generate_teacher_labels(
    data: pd.DataFrame,
    *,
    higher_timeframe: str = "4h",
    ema_span: int = 12,
    swing_window: int = 20,
    volume_window: int = 60,
    cooldown_candles: int = 5,
) -> pd.DataFrame:
    """Return features and 3-class teacher actions (HOLD=0, BUY=1, SELL=2).

    BUY/SELL denote desired target-position transitions, not executable orders.
    Labels are strictly historical and are intended only for offline research.
    """
    frame = _validate_ohlcv(data)
    for name, value, minimum in (
        ("ema_span", ema_span, 2),
        ("swing_window", swing_window, 2),
        ("volume_window", volume_window, 2),
        ("cooldown_candles", cooldown_candles, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name}_out_of_range")
    if higher_timeframe not in {"4h", "1D"}:
        raise ValueError("higher_timeframe_must_be_4h_or_1D")

    bias = _higher_timeframe_bias(frame, higher_timeframe, ema_span)
    bullish_structure, bearish_structure, prior_high = _smc_features(
        frame, swing_window
    )
    buy_volume, sell_volume = _volume_confirmations(frame, volume_window)
    buy_setup = (bias > 0) & bullish_structure & buy_volume
    sell_setup = (bias < 0) & bearish_structure & sell_volume

    actions = np.full(len(frame), ACTION_HOLD, dtype=np.int8)
    position = 0
    last_change = -cooldown_candles - 1
    for i in range(len(frame)):
        if i - last_change <= cooldown_candles:
            continue
        if position <= 0 and bool(buy_setup.iloc[i]):
            actions[i] = ACTION_BUY
            position = 1
            last_change = i
        elif position >= 0 and bool(sell_setup.iloc[i]):
            actions[i] = ACTION_SELL
            position = -1
            last_change = i

    result = frame.copy()
    result["higher_timeframe_bias"] = bias.to_numpy(dtype=np.int8)
    result["bos_reference_high"] = prior_high.to_numpy(dtype=np.float64)
    result["teacher_action"] = actions
    result["teacher_action_name"] = np.select(
        [actions == ACTION_BUY, actions == ACTION_SELL],
        ["BUY", "SELL"],
        default="HOLD",
    )
    result["teacher_position"] = np.where(
        actions == ACTION_BUY, 1, np.where(actions == ACTION_SELL, -1, 0)
    )
    return result
