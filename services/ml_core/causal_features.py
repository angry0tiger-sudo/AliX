"""Causal, window-local feature matrix builder for the new ML Core."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    OHLCV_COLUMNS,
    add_market_features,
)


def build_causal_feature_matrix(candles: DataFrame) -> DataFrame:
    """Build only the whitelisted features from the supplied window.

    Inputs must be UTC-aware, increasing and evenly spaced. Any target column,
    future sentinel or additional data column is discarded before engineering.
    No global state or previous-window data is reused.
    """
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    data = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and (
        not np.isfinite(deltas).all()
        or np.any(deltas <= 0)
        or not np.all(deltas == deltas[0])
    ):
        raise ValueError("timestamp_interval_inconsistent")
    data["date"] = dates.dt.tz_convert("UTC")

    engineered = add_market_features(data)
    features = engineered.loc[:, list(FEATURE_COLUMNS)].copy()
    if any(name.startswith("&-") for name in features.columns):
        raise ValueError("target_column_in_feature_matrix")
    return features
