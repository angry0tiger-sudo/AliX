"""Versioned, point-in-time FreqAI features and target contracts."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

FEATURE_SCHEMA_VERSION: Final[str] = "alix.freqai.features.v2"
TARGET_SCHEMA_VERSION: Final[str] = "alix.freqai.forward-return.v1"
TARGET_COLUMN: Final[str] = "&-s_forward_return"
OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "%-return_1",
    "%-return_3",
    "%-return_6",
    "%-volatility_12",
    "%-range_pct",
    "%-body_pct",
    "%-volume_change_1",
    "%-close_vs_sma_12",
    "%-atr_proxy_14",
    "%-rsi_14",
    "%-macd_histogram",
    "%-volume_zscore_20",
    "%-atr_percentile_252",
)


def add_market_features(dataframe: DataFrame) -> DataFrame:
    """Add versioned point-in-time OHLCV features using closed candles only.

    RSI uses Wilder smoothing; MACD histogram is normalized by close; volume
    Z-score uses a trailing 20-bar population standard deviation; ATR percentile
    ranks current ATR/close against a trailing 252-bar window. All windows are
    right-aligned. Warm-up rows remain NaN and are never backfilled.
    """
    missing = sorted(set(OHLCV_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"OHLCV feature input is missing columns: {missing}")

    result = dataframe.copy()
    opened = pd.to_numeric(result["open"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    close = pd.to_numeric(result["close"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_proxy = true_range.rolling(14, min_periods=14).mean() / close

    delta = close.diff()
    average_gain = delta.clip(lower=0.0).ewm(
        alpha=1.0 / 14, adjust=False, min_periods=14
    ).mean()
    average_loss = (-delta.clip(upper=0.0)).ewm(
        alpha=1.0 / 14, adjust=False, min_periods=14
    ).mean()
    relative_strength = average_gain / average_loss.where(average_loss > 0.0)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask(average_loss.eq(0.0) & average_gain.gt(0.0), 100.0)
    rsi = rsi.mask(average_loss.eq(0.0) & average_gain.eq(0.0), 50.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_histogram = (macd - macd_signal) / close

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    volume_zscore = (volume - volume_mean) / volume_std.where(volume_std > 0.0)
    volume_zscore = volume_zscore.mask(
        volume_std.eq(0.0) & volume_mean.notna(), 0.0
    )

    atr_percentile = atr_proxy.rolling(252, min_periods=63).rank(pct=True)

    feature_values: dict[str, pd.Series] = {
        "%-return_1": returns,
        "%-return_3": close.pct_change(3, fill_method=None),
        "%-return_6": close.pct_change(6, fill_method=None),
        "%-volatility_12": returns.rolling(12, min_periods=12).std(),
        "%-range_pct": (high - low) / close,
        "%-body_pct": (close - opened) / opened,
        "%-volume_change_1": volume.pct_change(fill_method=None),
        "%-close_vs_sma_12": close / close.rolling(12, min_periods=12).mean() - 1.0,
        "%-atr_proxy_14": atr_proxy,
        "%-rsi_14": rsi,
        "%-macd_histogram": macd_histogram,
        "%-volume_zscore_20": volume_zscore,
        "%-atr_percentile_252": atr_percentile,
    }

    for name in FEATURE_COLUMNS:
        result[name] = feature_values[name].replace([np.inf, -np.inf], np.nan)
    return result


def forward_return_target(dataframe: DataFrame, horizon: int = 12) -> pd.Series:
    """Return the forward close-to-close return label for training only."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    return (close.shift(-horizon) / close - 1.0).rename(TARGET_COLUMN)
