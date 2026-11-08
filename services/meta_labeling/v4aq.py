"""V4-AQ Funding-only Meta feature contract for both V4-AP primary strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.meta_labeling.v4x import (
    fit_lightgbm_meta_filter,
    predict_meta_probabilities,
)

V4AQ_SCHEMA_VERSION = "alix.ml-core.v4aq-meta-model.v2"
V4AQ_PRIMARY_LOOKBACK = 20
V4AQ_EMA_LOOKBACK = 200
V4AQ_META_FEATURE_COLUMNS = (
    "primary_context_1",
    "primary_context_2",
    "funding_rate_zscore",
    "funding_rate_ma",
)
MEAN_REVERSION_STRATEGY = "MEAN_REVERSION_RSI_LT_30_CLOSE_GT_EMA200"
DONCHIAN_STRATEGY = "DONCHIAN20_CLOSE_BREAKOUT"
SUPPORTED_STRATEGIES = (MEAN_REVERSION_STRATEGY, DONCHIAN_STRATEGY)


def _validate_inputs(
    bars: DataFrame,
    external_features: DataFrame,
    strategy: str,
) -> None:
    if bars.empty or len(bars) != len(external_features):
        raise ValueError("v4aq_feature_input_alignment_invalid")
    if not {"high", "low", "close"}.issubset(bars.columns):
        raise ValueError("v4aq_ohlc_columns_missing")
    external_columns = ("funding_rate_zscore", "funding_rate_ma")
    if tuple(external_features.columns) != external_columns:
        raise ValueError("v4aq_external_feature_schema_invalid")
    if not external_features.index.equals(bars.index):
        raise ValueError("v4aq_external_feature_index_mismatch")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError("v4aq_primary_strategy_not_supported")


def _validated_price(bars: DataFrame, column: str) -> Series:
    values = pd.to_numeric(bars[column], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or bool((values <= 0.0).any()):
        raise ValueError(f"v4aq_price_invalid:{column}")
    return values.reset_index(drop=True)


def _wilder_rsi(close: Series, period: int = 14) -> Series:
    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    average_gain = gains.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    movement = average_gain + average_loss
    result = 100.0 * average_gain / movement.where(movement > 0.0)
    return result.mask(movement.eq(0.0), 50.0)


def build_v4aq_meta_features(
    bars: DataFrame,
    external_features: DataFrame,
    *,
    strategy: str,
) -> DataFrame:
    """Build strategy-matched primary context plus two causal Funding features."""
    _validate_inputs(bars, external_features, strategy)
    original_index = bars.index
    normalized = bars.reset_index(drop=True)
    high = _validated_price(normalized, "high")
    low = _validated_price(normalized, "low")
    close = _validated_price(normalized, "close")
    if strategy == DONCHIAN_STRATEGY:
        prior_high = (
            high.shift(1)
            .rolling(V4AQ_PRIMARY_LOOKBACK, min_periods=V4AQ_PRIMARY_LOOKBACK)
            .max()
        )
        prior_low = (
            low.shift(1)
            .rolling(V4AQ_PRIMARY_LOOKBACK, min_periods=V4AQ_PRIMARY_LOOKBACK)
            .min()
        )
        context = DataFrame(
            {
                "primary_context_1": close / prior_high - 1.0,
                "primary_context_2": (prior_high - prior_low) / close,
            }
        )
    else:
        rsi = _wilder_rsi(close)
        ema = close.ewm(
            span=V4AQ_EMA_LOOKBACK,
            adjust=False,
            min_periods=V4AQ_EMA_LOOKBACK,
        ).mean()
        context = DataFrame(
            {
                "primary_context_1": rsi / 100.0,
                "primary_context_2": close / ema - 1.0,
            }
        )
    external = (
        external_features.reset_index(drop=True)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )
    if bool(np.isinf(external.to_numpy(dtype=float)).any()):
        raise ValueError("v4aq_external_feature_infinite")
    result = pd.concat([context, external], axis=1)
    if tuple(result.columns) != V4AQ_META_FEATURE_COLUMNS:
        raise ValueError("v4aq_meta_feature_schema_mismatch")
    result.index = original_index
    return result


def _validate_fit_matrix(features: DataFrame, reason: str) -> None:
    if tuple(features.columns) != V4AQ_META_FEATURE_COLUMNS:
        raise ValueError(f"v4aq_meta_feature_schema_mismatch:{reason}")
    values = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"v4aq_meta_features_not_complete:{reason}")


def fit_v4aq_lightgbm_meta_filter(
    fit_features: DataFrame,
    fit_labels: Series,
    calibration_features: DataFrame,
    *,
    target_accept_fraction: float = 0.50,
) -> tuple[Any, dict[str, Any]]:
    """Fit the existing purged OOF LightGBM model; calibrate without labels."""
    _validate_fit_matrix(fit_features, "fit")
    _validate_fit_matrix(calibration_features, "calibration")
    if len(fit_features) != len(fit_labels):
        raise ValueError("v4aq_meta_label_alignment_invalid")
    model, diagnostics = fit_lightgbm_meta_filter(
        fit_features,
        fit_labels,
        calibration_features,
        target_accept_fraction=target_accept_fraction,
    )
    diagnostics["feature_schema_version"] = V4AQ_SCHEMA_VERSION
    diagnostics["feature_columns"] = list(V4AQ_META_FEATURE_COLUMNS)
    diagnostics["external_features_used"] = [
        "funding_rate_zscore",
        "funding_rate_ma",
    ]
    return model, diagnostics


def predict_v4aq_meta_probabilities(
    model: Any,
    features: DataFrame,
) -> np.ndarray:
    """Predict positive-class probabilities for Primary BUY events."""
    _validate_fit_matrix(features, "prediction")
    return predict_meta_probabilities(model, features)
