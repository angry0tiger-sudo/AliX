"""Versioned, point-in-time FreqAI features and target contracts."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

FEATURE_SCHEMA_VERSION: Final[str] = "alix.freqai.features.v1"
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
)


def add_market_features(dataframe: DataFrame) -> DataFrame:
    """Add a compact feature set built only from current/past OHLCV candles.

    The dataframe must already be chronological and pass the Data Quality Gate.
    No centered windows, backfills, or future rows are used.
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
    returns = close.pct_change()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    feature_values: dict[str, pd.Series] = {
        "%-return_1": returns,
        "%-return_3": close.pct_change(3),
        "%-return_6": close.pct_change(6),
        "%-volatility_12": returns.rolling(12, min_periods=12).std(),
        "%-range_pct": (high - low) / close,
        "%-body_pct": (close - opened) / opened,
        "%-volume_change_1": volume.pct_change(),
        "%-close_vs_sma_12": close / close.rolling(12, min_periods=12).mean() - 1.0,
        "%-atr_proxy_14": (
            true_range.rolling(14, min_periods=14).mean() / close
        ),
    }

    for name in FEATURE_COLUMNS:
        result[name] = feature_values[name].replace([np.inf, -np.inf], np.nan)
    return result


def forward_return_target(dataframe: DataFrame, horizon: int = 12) -> pd.Series:
    """Return the forward close-to-close return label for training only."""
    if isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    return (close.shift(-horizon) / close - 1.0).rename(TARGET_COLUMN)
