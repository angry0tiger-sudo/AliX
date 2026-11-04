"""V4-Y causal Bollinger breakout primary signals with OOF-only meta filtering."""

from __future__ import annotations

import math
from typing import Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.meta_labeling.v4x import (
    V4X_META_CALIBRATION_FRACTION as V4Y_META_CALIBRATION_FRACTION,
    V4X_MINIMUM_CLASS_ROWS as V4Y_MINIMUM_CLASS_ROWS,
    V4X_MINIMUM_FIT_ROWS as V4Y_MINIMUM_FIT_ROWS,
    V4X_OOF_FOLD_COUNT as V4Y_OOF_FOLD_COUNT,
    V4X_TARGET_ACCEPT_FRACTION as V4Y_TARGET_ACCEPT_FRACTION,
    build_costed_meta_labels,
    fit_lightgbm_meta_filter,
    generate_oof_primary_events as _generate_shared_oof_primary_events,
    predict_meta_probabilities,
    split_meta_fit_calibration,
)

V4Y_BOLLINGER_PERIOD: Final[int] = 20
V4Y_BOLLINGER_STDDEV: Final[float] = 2.0
V4Y_EMA_TREND_PERIOD: Final[int] = 200
V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS: Final[int] = 50
V4Y_PRIMARY_SIGNAL_GENERATION: Final[str] = (
    "CAUSAL_BOLLINGER_UPPER_BREAKOUT_EMA200_TREND"
)
V4Y_META_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "%-primary_bollinger_bandwidth_pct",
    "%-primary_close_vs_bollinger_upper_pct",
    "%-primary_close_vs_ema200_pct",
    "%-primary_ema200_slope_10_pct",
    "%-primary_breakout_distance_pct",
)


def _finite_feature_rows(features: DataFrame) -> np.ndarray:
    numeric = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return cast(np.ndarray, np.isfinite(numeric).all(axis=1))


def _validate_periods(
    bollinger_period: int,
    ema_period: int,
    standard_deviations: float,
) -> None:
    integer_values = (bollinger_period, ema_period)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise ValueError("v4y_primary_period_configuration_invalid")
    if bollinger_period < 2 or ema_period <= bollinger_period:
        raise ValueError("v4y_primary_period_configuration_invalid")
    if (
        isinstance(standard_deviations, bool)
        or not math.isfinite(float(standard_deviations))
        or float(standard_deviations) <= 0
    ):
        raise ValueError("v4y_bollinger_deviation_configuration_invalid")


def _validate_feature_inputs(bars: DataFrame, candle_features: DataFrame) -> None:
    if len(bars) != len(candle_features) or bars.empty:
        raise ValueError("v4y_primary_input_alignment_invalid")
    if not candle_features.columns.is_unique:
        raise ValueError("v4y_feature_columns_must_be_unique")
    if any(str(name).startswith("&-") for name in candle_features.columns):
        raise ValueError("v4y_target_column_in_primary_features")
    if set(V4Y_META_FEATURE_COLUMNS).intersection(candle_features.columns):
        raise ValueError("v4y_primary_feature_column_collision")
    if "close" not in bars.columns:
        raise ValueError("v4y_primary_close_column_missing")


def _validated_close(bars: DataFrame) -> Series:
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((close <= 0).any()):
        raise ValueError("v4y_primary_close_values_invalid")
    return close


def build_v4y_primary_features(
    bars: DataFrame,
    candle_features: DataFrame,
    *,
    bollinger_period: int = V4Y_BOLLINGER_PERIOD,
    standard_deviations: float = V4Y_BOLLINGER_STDDEV,
    ema_period: int = V4Y_EMA_TREND_PERIOD,
) -> tuple[DataFrame, Series]:
    """Build causal upper-band breakout BUY events confirmed above EMA200.

    A BUY is emitted only on the first closed candle above its Bollinger upper
    band after the previous closed candle was at or below the previous upper
    band, and only while close is above the causal EMA trend filter. Execution
    remains the following candle's open in the economic replay.
    """
    _validate_periods(bollinger_period, ema_period, standard_deviations)
    _validate_feature_inputs(bars, candle_features)
    close = _validated_close(bars)

    middle = close.rolling(window=bollinger_period, min_periods=bollinger_period).mean()
    deviation = close.rolling(
        window=bollinger_period, min_periods=bollinger_period
    ).std(ddof=0)
    upper = middle + float(standard_deviations) * deviation
    lower = middle - float(standard_deviations) * deviation
    ema = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()

    breakout = (close > upper) & (close.shift(1) <= upper.shift(1))
    trend_aligned = close > ema

    result = candle_features.reset_index(drop=True).copy()
    result["%-primary_bollinger_bandwidth_pct"] = (upper - lower) / middle
    result["%-primary_close_vs_bollinger_upper_pct"] = close / upper - 1.0
    result["%-primary_close_vs_ema200_pct"] = close / ema - 1.0
    result["%-primary_ema200_slope_10_pct"] = ema.pct_change(
        periods=10, fill_method=None
    )
    result["%-primary_breakout_distance_pct"] = close / upper.shift(1) - 1.0

    finite = _finite_feature_rows(result)
    signals = breakout.fillna(False).to_numpy(dtype=bool)
    signals &= trend_aligned.fillna(False).to_numpy(dtype=bool)
    signals &= finite
    return result, Series(signals, name="primary_buy_signal", dtype=bool)


def evaluate_primary_train_sample_gate(
    primary_buy_signal: Series,
    *,
    train_start_index: int,
    train_end_index: int,
    minimum_signal_count: int = V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
) -> dict[str, int | bool | str]:
    """Fail closed before meta-label training when Train has fewer than 50 BUYs."""
    if not pd.api.types.is_bool_dtype(primary_buy_signal.dtype):
        raise ValueError("v4y_primary_sample_gate_requires_boolean_signals")
    if primary_buy_signal.isna().any():
        raise ValueError("v4y_primary_sample_gate_signals_must_not_be_null")
    invalid_indices = (
        isinstance(train_start_index, bool)
        or isinstance(train_end_index, bool)
        or not isinstance(train_start_index, int)
        or not isinstance(train_end_index, int)
        or train_start_index < 0
        or train_end_index > len(primary_buy_signal)
        or train_end_index <= train_start_index
    )
    if invalid_indices:
        raise ValueError("v4y_primary_sample_gate_window_invalid")
    if (
        isinstance(minimum_signal_count, bool)
        or not isinstance(minimum_signal_count, int)
        or minimum_signal_count < 1
    ):
        raise ValueError("v4y_primary_sample_gate_minimum_invalid")

    observed = int(primary_buy_signal.iloc[train_start_index:train_end_index].sum())
    passed = observed >= minimum_signal_count
    return {
        "status": "PASS" if passed else "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS",
        "passed": passed,
        "observed_primary_buy_count": observed,
        "minimum_primary_buy_count": minimum_signal_count,
        "train_start_index": train_start_index,
        "train_end_index_exclusive": train_end_index,
    }


def generate_v4y_oof_primary_events(
    primary_buy_signal: Series,
    *,
    start_index: int,
    stop_index: int,
    fold_count: int = V4Y_OOF_FOLD_COUNT,
) -> DataFrame:
    """Reuse V4-X chronological OOF mechanics with explicit V4-Y provenance."""
    result = _generate_shared_oof_primary_events(
        primary_buy_signal,
        start_index=start_index,
        stop_index=stop_index,
        fold_count=fold_count,
    )
    result["primary_signal_generation"] = V4Y_PRIMARY_SIGNAL_GENERATION
    return result


__all__ = [
    "V4Y_BOLLINGER_PERIOD",
    "V4Y_BOLLINGER_STDDEV",
    "V4Y_EMA_TREND_PERIOD",
    "V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS",
    "V4Y_PRIMARY_SIGNAL_GENERATION",
    "V4Y_META_FEATURE_COLUMNS",
    "V4Y_META_CALIBRATION_FRACTION",
    "V4Y_MINIMUM_CLASS_ROWS",
    "V4Y_MINIMUM_FIT_ROWS",
    "V4Y_OOF_FOLD_COUNT",
    "V4Y_TARGET_ACCEPT_FRACTION",
    "build_costed_meta_labels",
    "build_v4y_primary_features",
    "evaluate_primary_train_sample_gate",
    "fit_lightgbm_meta_filter",
    "generate_v4y_oof_primary_events",
    "predict_meta_probabilities",
    "split_meta_fit_calibration",
]
