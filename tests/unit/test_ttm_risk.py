"""Contract tests for TTM-R2 risk-only evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ai_ensemble.ttm_risk import (
    CONTEXT_LENGTH,
    PREDICTION_HORIZON,
    forecast_dataset_indices,
    make_evaluation_origins,
    prepare_risk_target,
    qlike_loss,
    risk_forecast_gate,
    score_variance_forecasts,
)


def _ohlcv(rows: int = 2_000) -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-01", periods=rows, freq="h", tz="UTC")
    returns = np.sin(np.arange(rows, dtype=np.float64) / 17.0) * 0.001
    close = 40_000.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": timestamps, "close": close})


def test_risk_target_is_causal_and_does_not_fill_first_return() -> None:
    data = _ohlcv()
    result = prepare_risk_target(data)
    assert len(result) == len(data) - 1
    assert result["return_variance"].iloc[0] > 0.0
    assert np.isfinite(result["log1p_return_variance"]).all()


def test_risk_target_rejects_gaps_and_duplicate_timestamps() -> None:
    data = _ohlcv()
    with pytest.raises(ValueError, match="continuous 1h"):
        prepare_risk_target(data.drop(index=100).reset_index(drop=True))
    duplicated = data.copy()
    duplicated.loc[101, "date"] = duplicated.loc[100, "date"]
    with pytest.raises(ValueError, match="unique"):
        prepare_risk_target(duplicated)


def test_risk_target_rejects_invalid_prices() -> None:
    data = _ohlcv()
    data.loc[10, "close"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_risk_target(data)


def test_split_origins_are_causal_non_overlapping_and_inside_splits() -> None:
    row_count = 31_800
    split = make_evaluation_origins(row_count)
    for origins, lower, upper in (
        (split["validation_origins"], split["train_end"], split["validation_end"]),
        (split["holdout_origins"], split["validation_end"], row_count),
    ):
        assert len(origins) >= 30
        assert all(origin >= CONTEXT_LENGTH - 1 for origin in origins)
        assert all(origin + 1 >= lower for origin in origins)
        assert all(origin + PREDICTION_HORIZON < upper for origin in origins)
        assert all(b - a == PREDICTION_HORIZON for a, b in zip(origins, origins[1:]))


def test_dataset_indices_end_context_at_the_registered_origin() -> None:
    origins = [511, 535, 559]
    indices = forecast_dataset_indices(origins)
    assert indices == [0, 24, 48]
    assert all(
        index + CONTEXT_LENGTH - 1 == origin for index, origin in zip(indices, origins)
    )


def test_qlike_rejects_non_positive_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        qlike_loss(np.asarray([1.0, 2.0]), np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="equal non-zero shape"):
        qlike_loss(np.asarray([1.0]), np.asarray([1.0, 2.0]))


def test_score_and_gate_require_incremental_risk_forecast_value() -> None:
    actual = np.asarray([0.8, 1.2, 0.9, 1.1], dtype=np.float64)
    baseline = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    excellent = np.asarray([0.82, 1.18, 0.92, 1.08], dtype=np.float64)
    poor = np.asarray([1.4, 0.7, 1.5, 0.6], dtype=np.float64)
    good_metrics = score_variance_forecasts(actual, excellent, baseline)
    poor_metrics = score_variance_forecasts(actual, poor, baseline)
    assert good_metrics["maximum_position_multiplier"] <= 1.0
    assert risk_forecast_gate({**good_metrics, "sample_count": 30}) is True
    assert risk_forecast_gate({**poor_metrics, "sample_count": 30}) is False
    assert risk_forecast_gate({**good_metrics, "sample_count": 29}) is False
