"""TTM-R2 risk-only forecasting contracts.

This module deliberately contains no exchange, order, or model-loading code.
The actual TTM dependency is loaded only inside the GitHub Actions research runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONTEXT_LENGTH = 512
PREDICTION_HORIZON = 24
TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
FORECAST_STRIDE = 24
TARGET_COLUMN = "log1p_return_variance"
DATASET_SHA256 = "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
EXPECTED_ROWS = 31_801
EXPECTED_START_UNIX = 1_635_712_200
EXPECTED_END_UNIX = 1_750_192_200


def prepare_risk_target(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create a causal hourly log(1 + squared-return) target without filling gaps."""
    required = {"date", "close"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV input is missing required columns: {sorted(missing)}")
    if len(ohlcv) < CONTEXT_LENGTH + PREDICTION_HORIZON + 30:
        raise ValueError(
            "OHLCV history is too short for the registered TTM-R2 protocol."
        )

    timestamps = pd.to_datetime(ohlcv["date"], utc=True, errors="raise")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Timestamps must be present and unique.")
    deltas = timestamps.diff().dropna()
    if not bool((deltas == pd.Timedelta(hours=1)).all()):
        raise ValueError(
            "Expected a continuous 1h series; missing or irregular candles are rejected."
        )

    close = pd.to_numeric(ohlcv["close"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("Close prices must be finite and strictly positive.")

    log_returns = np.empty_like(close)
    log_returns[0] = np.nan
    log_returns[1:] = np.diff(np.log(close))
    return_variance = np.square(log_returns)
    target = np.log1p(return_variance)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            TARGET_COLUMN: target,
            "return_variance": return_variance,
        }
    )
    # The first return is mathematically undefined; discard it rather than impute it.
    frame = frame.iloc[1:].reset_index(drop=True)
    if not np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=np.float64)).all():
        raise ValueError("Risk target contains non-finite values.")
    return frame


def make_evaluation_origins(row_count: int) -> dict[str, Any]:
    """Create chronological, non-overlapping 24h validation and holdout origins.

    Each origin is the final observed bar at decision time. The target window is
    exactly origin+1 through origin+24, wholly inside its assigned split.
    """
    if row_count <= CONTEXT_LENGTH + (2 * PREDICTION_HORIZON) + 30:
        raise ValueError(
            "Not enough observations for train/validation/holdout windows."
        )

    train_end = int(row_count * TRAIN_FRACTION)
    validation_end = int(row_count * VALIDATION_END_FRACTION)
    validation_origins = list(
        range(
            max(CONTEXT_LENGTH - 1, train_end - 1),
            validation_end - PREDICTION_HORIZON,
            FORECAST_STRIDE,
        )
    )
    holdout_origins = list(
        range(
            max(CONTEXT_LENGTH - 1, validation_end - 1),
            row_count - PREDICTION_HORIZON,
            FORECAST_STRIDE,
        )
    )
    if len(validation_origins) < 30 or len(holdout_origins) < 30:
        raise ValueError(
            "Validation and holdout must each contain at least 30 disjoint forecasts."
        )
    for origins, split_start, split_end in (
        (validation_origins, train_end, validation_end),
        (holdout_origins, validation_end, row_count),
    ):
        for origin in origins:
            if origin + 1 < split_start or origin + PREDICTION_HORIZON >= split_end:
                raise ValueError(
                    "Forecast target crosses a chronological split boundary."
                )
    return {
        "train_end": train_end,
        "validation_end": validation_end,
        "validation_origins": validation_origins,
        "holdout_origins": holdout_origins,
    }


def forecast_dataset_indices(origins: list[int]) -> list[int]:
    """Map a decision-time origin to ForecastDFDataset's first context-row index."""
    indices = [origin - CONTEXT_LENGTH + 1 for origin in origins]
    if any(index < 0 for index in indices):
        raise ValueError("A forecast origin does not have a complete causal context.")
    return indices


class ForecastContextOnlyDataset:
    """Expose only historical context tensors to the forecasting model."""

    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self._dataset[self._indices[item]]
        if not isinstance(sample, dict):
            raise TypeError("Forecast dataset samples must be mappings.")
        return {
            key: value
            for key, value in sample.items()
            if key not in {"future_values", "future_observed_mask"}
        }


def qlike_loss(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> float:
    """Scale-invariant QLIKE loss for strictly-positive variance forecasts."""
    actual = np.asarray(actual_variance, dtype=np.float64)
    forecast = np.asarray(forecast_variance, dtype=np.float64)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError(
            "Actual and forecast variance arrays must have equal non-zero shape."
        )
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("QLIKE inputs must be finite.")
    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError("QLIKE inputs must be strictly positive.")
    ratio = actual / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def score_variance_forecasts(
    actual_variance: np.ndarray,
    ttm_forecast_variance: np.ndarray,
    persistence_forecast_variance: np.ndarray,
) -> dict[str, float | int]:
    """Compare TTM risk forecasts with a trailing-24h realized-variance baseline."""
    actual = np.asarray(actual_variance, dtype=np.float64)
    forecast = np.asarray(ttm_forecast_variance, dtype=np.float64)
    baseline = np.asarray(persistence_forecast_variance, dtype=np.float64)
    if (
        actual.shape != forecast.shape
        or actual.shape != baseline.shape
        or actual.size == 0
    ):
        raise ValueError(
            "Actual, TTM, and persistence arrays must have equal non-zero shape."
        )
    if not all(np.isfinite(values).all() for values in (actual, forecast, baseline)):
        raise ValueError("Forecast evaluation contains non-finite values.")
    if any(np.any(values <= 0) for values in (actual, forecast, baseline)):
        raise ValueError("Variance values must be strictly positive.")

    actual_log = np.log(actual)
    forecast_log = np.log(forecast)
    baseline_log = np.log(baseline)
    ttm_qlike = qlike_loss(actual, forecast)
    baseline_qlike = qlike_loss(actual, baseline)
    improvement = (
        0.0 if baseline_qlike <= 0 else (baseline_qlike - ttm_qlike) / baseline_qlike
    )
    multipliers = np.minimum(1.0, np.sqrt(baseline / forecast))
    return {
        "sample_count": int(actual.size),
        "ttm_qlike": ttm_qlike,
        "persistence_qlike": baseline_qlike,
        "qlike_improvement_fraction": float(improvement),
        "ttm_log_variance_mae": float(np.mean(np.abs(actual_log - forecast_log))),
        "persistence_log_variance_mae": float(
            np.mean(np.abs(actual_log - baseline_log))
        ),
        "median_risk_only_position_multiplier": float(np.median(multipliers)),
        "scaled_down_fraction": float(np.mean(multipliers < 1.0)),
        "maximum_position_multiplier": float(np.max(multipliers)),
    }


def risk_forecast_gate(metrics: dict[str, float | int]) -> bool:
    """Pre-registered research-only gate; it never grants execution permission."""
    return bool(
        int(metrics["sample_count"]) >= 30
        and float(metrics["qlike_improvement_fraction"]) >= 0.05
        and float(metrics["ttm_log_variance_mae"])
        <= float(metrics["persistence_log_variance_mae"])
        and float(metrics["maximum_position_multiplier"]) <= 1.0
    )
