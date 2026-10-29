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


def _validate_window_timestamps(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not _interval_deltas_are_consistent(deltas):
        raise ValueError("timestamp_interval_inconsistent")
    return dates.dt.tz_convert("UTC")


def _interval_deltas_are_consistent(deltas: np.ndarray) -> bool:
    return (
        bool(np.isfinite(deltas).all())
        and not bool(np.any(deltas <= 0))
        and bool(np.all(deltas == deltas[0]))
    )


def _validated_window(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")
    data = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    data["date"] = _validate_window_timestamps(data["date"])
    return data


def build_causal_feature_matrix(candles: DataFrame) -> DataFrame:
    """Build the whitelisted features only from the supplied window.

    Any target column, future sentinel or additional data column is discarded
    before feature engineering. No global state or prior-window data is reused.
    """
    data = _validated_window(candles)
    engineered = add_market_features(data)
    features = engineered.loc[:, list(FEATURE_COLUMNS)].copy()
    if any(name.startswith("&-") for name in features.columns):
        raise ValueError("target_column_in_feature_matrix")
    return features
