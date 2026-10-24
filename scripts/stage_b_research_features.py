"""Shared research-only Stage B indicators.

These indicators are deliberately outside the production/global feature schema.
They use current closed-candle and historical values only; undefined warm-up
rows remain undefined rather than being imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.freqai_baseline.features import add_market_features

NEW_FEATURE_COLUMNS: tuple[str, ...] = (
    "%-rsi_14",
    "%-close_vs_sma_50",
    "%-close_vs_sma_200",
    "%-momentum_24",
    "%-volatility_24",
    "%-range_position_24",
)


def add_signal_features(candles: DataFrame) -> DataFrame:
    """Add past-only trend/momentum indicators; never fill undefined values."""
    result = add_market_features(candles)
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    returns = close.pct_change(fill_method=None)

    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    average_gain = gains.rolling(14, min_periods=14).mean()
    average_loss = losses.rolling(14, min_periods=14).mean()
    rsi_denominator = average_gain + average_loss
    rsi = 100.0 * average_gain / rsi_denominator.where(rsi_denominator > 0.0)
    # A fully flat 14-bar window has a defined neutral RSI by convention.
    result["%-rsi_14"] = rsi.mask(rsi_denominator.eq(0.0), 50.0)

    for window in (50, 200):
        rolling_mean = close.rolling(window, min_periods=window).mean()
        result[f"%-close_vs_sma_{window}"] = close / rolling_mean - 1.0

    result["%-momentum_24"] = close.pct_change(24, fill_method=None)
    result["%-volatility_24"] = returns.rolling(24, min_periods=24).std()

    rolling_high = high.rolling(24, min_periods=24).max()
    rolling_low = low.rolling(24, min_periods=24).min()
    range_width = rolling_high - rolling_low
    range_position = (close - rolling_low) / range_width.where(range_width > 0.0)
    flat_range = range_width.eq(0.0) & close.eq(rolling_low)
    # A flat 24-bar range places the close at its neutral midpoint.
    result["%-range_position_24"] = range_position.mask(flat_range, 0.5)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
