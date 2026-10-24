from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_stage_b_classification import (
    BASELINE_CANDIDATE_ID,
    HORIZON,
    MODEL_PROFILES,
    PROBABILITY_THRESHOLDS,
    VOLATILITY_CUTOFF_QUANTILE,
    event_labels,
    signal_mask,
    volatility_cutoff,
)
from scripts.stage_b_research_features import NEW_FEATURE_COLUMNS, add_signal_features


def _flat_candles(rows: int = 280) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="h"),
            "open": close.copy(),
            "high": close.copy(),
            "low": close.copy(),
            "close": close.copy(),
            "volume": np.zeros(rows),
        }
    )


def test_classification_target_is_upside_over_cost_not_absolute_magnitude() -> None:
    labels = event_labels([0.006, 0.00601, 0.02, -0.02, -0.0061, 0.0059])

    assert labels.tolist() == [0, 1, 1, 0, 0, 0]


def test_volatility_cutoff_uses_training_fold_values_only() -> None:
    training = np.arange(1.0, 101.0)
    cutoff = volatility_cutoff(training)

    assert cutoff == pytest.approx(75.25)
    assert VOLATILITY_CUTOFF_QUANTILE == 0.75


def test_signal_requires_probability_and_high_volatility_gates() -> None:
    probabilities = [0.9, 0.8, 0.4, 0.99]
    volatility = [0.8, 0.2, 0.9, 0.9]

    result = signal_mask(
        probabilities,
        volatility,
        probability_threshold=0.5,
        training_volatility_cutoff=0.7,
    )

    assert result.tolist() == [True, False, False, True]


def test_shared_technical_features_preserve_warmup_and_flat_window_definitions() -> None:
    candles = _flat_candles()
    original = candles.copy(deep=True)
    featured = add_signal_features(candles)

    assert set(NEW_FEATURE_COLUMNS).issubset(featured.columns)
    assert featured["%-close_vs_sma_200"].iloc[:199].isna().all()
    assert featured["%-close_vs_sma_200"].iloc[199:].notna().all()
    assert featured["%-rsi_14"].iloc[14:].eq(50.0).all()
    assert featured["%-range_position_24"].iloc[23:].eq(0.5).all()
    pd.testing.assert_frame_equal(candles, original)


def test_candidate_grid_is_fixed_and_cost_hurdle_is_not_a_candidate_parameter() -> None:
    assert HORIZON == 12
    assert PROBABILITY_THRESHOLDS == (0.35, 0.50, 0.65)
    assert MODEL_PROFILES == ("unweighted", "balanced")
    assert BASELINE_CANDIDATE_ID == "stage_b_8|unweighted|p0.50"


def test_non_finite_labels_bad_volatility_and_mismatched_arrays_fail_closed() -> None:
    with pytest.raises(ValueError, match="non_finite"):
        event_labels([0.01, float("nan")])
    with pytest.raises(ValueError, match="cutoff_input_invalid"):
        volatility_cutoff([0.1, float("inf")])
    with pytest.raises(ValueError, match="shape_invalid"):
        signal_mask([0.8], [0.1, 0.2], 0.5, 0.01)
