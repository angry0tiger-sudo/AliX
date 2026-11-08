"""V4-AQ Donchian + point-in-time Funding/OI LightGBM meta feature contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Any

from services.meta_labeling.v4x import (
    fit_lightgbm_meta_filter,
    predict_meta_probabilities,
)

V4AQ_SCHEMA_VERSION = "alix.ml-core.v4aq-meta-model.v1"
V4AQ_PRIMARY_LOOKBACK = 20
V4AQ_META_FEATURE_COLUMNS = (
    "primary_close_vs_donchian_high_pct",
    "primary_donchian_channel_width_pct",
    "funding_rate_zscore",
    "oi_change",
)


def _validate_inputs(bars: DataFrame, external_features: DataFrame) -> None:
    if bars.empty or len(bars) != len(external_features):
        raise ValueError("v4aq_feature_input_alignment_invalid")
    if not {"high", "low", "close"}.issubset(bars.columns):
        raise ValueError("v4aq_ohlc_columns_missing")
    if tuple(external_features.columns) != ("funding_rate_zscore", "oi_change"):
        raise ValueError("v4aq_external_feature_schema_invalid")
    if not external_features.index.equals(bars.index):
        raise ValueError("v4aq_external_feature_index_mismatch")


def _validated_price(bars: DataFrame, column: str) -> Series:
    values = pd.to_numeric(bars[column], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or bool((values <= 0.0).any()):
        raise ValueError(f"v4aq_price_invalid:{column}")
    return values.reset_index(drop=True)


def build_v4aq_meta_features(
    bars: DataFrame,
    external_features: DataFrame,
) -> DataFrame:
    """Build causal Donchian context plus the two external numeric features."""
    _validate_inputs(bars, external_features)
    high = _validated_price(bars, "high")
    low = _validated_price(bars, "low")
    close = _validated_price(bars, "close")
    previous_high = high.shift(1).rolling(
        V4AQ_PRIMARY_LOOKBACK,
        min_periods=V4AQ_PRIMARY_LOOKBACK,
    ).max()
    previous_low = low.shift(1).rolling(
        V4AQ_PRIMARY_LOOKBACK,
        min_periods=V4AQ_PRIMARY_LOOKBACK,
    ).min()
    channel_width = (previous_high - previous_low) / close
    technical = DataFrame(
        {
            "primary_close_vs_donchian_high_pct": close / previous_high - 1.0,
            "primary_donchian_channel_width_pct": channel_width,
        },
        index=bars.index,
    )
    external = external_features.apply(pd.to_numeric, errors="coerce").astype(float)
    if bool(np.isinf(external.to_numpy(dtype=float)).any()):
        raise ValueError("v4aq_external_feature_infinite")
    result = pd.concat([technical, external], axis=1)
    if tuple(result.columns) != V4AQ_META_FEATURE_COLUMNS:
        raise ValueError("v4aq_meta_feature_schema_mismatch")
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
    """Fit the existing OOF LightGBM filter using AI-6's fixed four-column schema."""
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
        "oi_change",
    ]
    return model, diagnostics


def predict_v4aq_meta_probabilities(
    model: Any,
    features: DataFrame,
) -> np.ndarray:
    _validate_fit_matrix(features, "prediction")
    return predict_meta_probabilities(model, features)
