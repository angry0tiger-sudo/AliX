from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_stage_b_signal_optimization import (
    BASELINE_CANDIDATE_ID,
    FEATURE_GROUPS,
    HORIZON_GRID,
    MODEL_PROFILES,
    add_signal_features,
    summarize_cost_hurdle,
)
from services.freqai_baseline.features import FEATURE_COLUMNS


def _ohlcv(rows: int = 280) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.0003 * index + 0.004 * np.sin(index / 7.0))
    opened = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(opened, close) * 1.002
    low = np.minimum(opened, close) * 0.998
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="h"),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.zeros(rows, dtype=float),
        }
    )


def test_extended_features_use_natural_warmup_without_imputation() -> None:
    candles = _ohlcv()
    featured = add_signal_features(candles)

    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert featured["%-close_vs_sma_200"].iloc[:199].isna().all()
    assert featured["%-close_vs_sma_200"].iloc[199:].notna().all()
    assert featured["%-rsi_14"].iloc[14:].notna().all()
    assert featured["%-volume_change_1"].isna().all()
    assert featured["%-close_vs_sma_200"].iloc[199] == pytest.approx(
        candles["close"].iloc[199] / candles["close"].iloc[:200].mean() - 1.0
    )


def test_extended_features_are_past_only() -> None:
    candles = _ohlcv()
    changed = candles.copy()
    split = 240
    changed.loc[split, "close"] = changed.loc[split, "close"] * 1.15
    changed.loc[split, "open"] = changed.loc[split - 1, "close"]
    changed.loc[split, "high"] = (
        max(changed.loc[split, "open"], changed.loc[split, "close"]) * 1.002
    )
    changed.loc[split, "low"] = (
        min(changed.loc[split, "open"], changed.loc[split, "close"]) * 0.998
    )

    original_features = add_signal_features(candles)
    changed_features = add_signal_features(changed)
    columns = [
        "%-rsi_14",
        "%-close_vs_sma_50",
        "%-close_vs_sma_200",
        "%-momentum_24",
        "%-volatility_24",
        "%-range_position_24",
    ]
    pd.testing.assert_frame_equal(
        original_features.loc[: split - 1, columns],
        changed_features.loc[: split - 1, columns],
    )


def test_cost_hurdle_counts_only_opportunities_strictly_above_cost() -> None:
    forecasts = [0.001, 0.006, 0.0061, 0.02]
    opportunities = [True, True, True, False]

    result = summarize_cost_hurdle(forecasts, opportunities, 0.006)

    assert result["prediction_count"] == 4
    assert result["entry_opportunity_count"] == 3
    assert result["forecasts_above_cost"] == 1
    assert result["take_rate_per_opportunity"] == pytest.approx(1 / 3)
    assert result["opportunity_forecast_p99"] is not None


def test_experiment_grid_preserves_baseline_and_closed_safety_defaults() -> None:
    assert BASELINE_CANDIDATE_ID == "stage_b_8|phase4_default|h12"
    assert HORIZON_GRID == (12, 6, 24)
    assert FEATURE_GROUPS == ("stage_b_8", "extended_trend_rsi")
    assert set(MODEL_PROFILES) == {"phase4_default", "shallow_regularized"}
    assert MODEL_PROFILES["phase4_default"]["n_estimators"] == 80
    assert (
        MODEL_PROFILES["shallow_regularized"]["num_leaves"]
        < (MODEL_PROFILES["phase4_default"]["num_leaves"])
    )


def test_cost_hurdle_rejects_mismatched_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="length_mismatch"):
        summarize_cost_hurdle([0.01], [True, False])
    with pytest.raises(ValueError, match="non_finite"):
        summarize_cost_hurdle([float("nan")], [True])


def test_flat_price_windows_have_explicit_neutral_technical_definitions() -> None:
    rows = 240
    candles = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="h"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.0),
            "low": np.full(rows, 100.0),
            "close": np.full(rows, 100.0),
            "volume": np.ones(rows, dtype=float),
        }
    )

    original = candles.copy(deep=True)
    featured = add_signal_features(candles)

    assert featured["%-rsi_14"].iloc[14:].eq(50.0).all()
    assert featured["%-range_position_24"].iloc[23:].eq(0.5).all()
    pd.testing.assert_frame_equal(candles, original)
