"""Causal CUSUM and Gaussian-HMM regime tests for V4-AM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.meta_labeling import v4am


def _market_frame(rows: int = 900) -> DataFrame:
    rng = np.random.default_rng(17)
    regimes = np.repeat(
        [0.0012, -0.0010, 0.0],
        [rows // 3, rows // 3, rows - 2 * (rows // 3)],
    )
    returns = regimes + rng.normal(0.0, 0.002, size=rows)
    close = 2_000.0 * np.exp(np.cumsum(returns))
    return DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})


def _base_features(bars: DataFrame) -> DataFrame:
    return DataFrame({"%-technical_feature": np.arange(len(bars), dtype=float)})


def _hmm_input(rows: int = 900) -> DataFrame:
    bars = _market_frame(rows)
    close_returns = np.log(bars["close"]).diff()
    volatility = close_returns.rolling(24, min_periods=24).std(ddof=0)
    return DataFrame(
        {
            v4am.LOG_RETURN_FEATURE: close_returns,
            v4am.REALIZED_VOLATILITY_FEATURE: volatility,
        }
    )


def test_cusum_is_causal_and_does_not_populate_final_holdout() -> None:
    bars = _market_frame(400)
    transformed, diagnostics = v4am.build_cusum_features(
        bars, _base_features(bars), apply_end=340
    )
    prefix_bars = bars.iloc[:300].reset_index(drop=True)
    prefix, _ = v4am.build_cusum_features(
        prefix_bars, _base_features(prefix_bars), apply_end=300
    )
    pd.testing.assert_series_equal(
        transformed.loc[:299, v4am.CUSUM_DRIFT_FEATURE].reset_index(drop=True),
        prefix[v4am.CUSUM_DRIFT_FEATURE].reset_index(drop=True),
    )
    assert diagnostics["lookahead_used"] is False
    assert diagnostics["total_event_count"] == (
        diagnostics["positive_event_count"] + diagnostics["negative_event_count"]
    )
    assert transformed.loc[340:, v4am.LOG_RETURN_FEATURE].isna().all()
    assert transformed.loc[340:, v4am.CUSUM_POSITIVE_EVENT_FEATURE].isna().all()


def test_cusum_rejects_bad_alignment_prices_and_apply_boundary() -> None:
    bars = _market_frame(100)
    with pytest.raises(ValueError, match="v4am_feature_alignment_invalid"):
        v4am.build_cusum_features(bars, _base_features(bars.iloc[:-1]), apply_end=90)
    invalid = bars.copy()
    invalid.loc[5, "close"] = 0.0
    with pytest.raises(ValueError, match="v4am_close_values_invalid"):
        v4am.build_cusum_features(invalid, _base_features(invalid), apply_end=90)
    with pytest.raises(ValueError, match="v4am_apply_window_invalid"):
        v4am.build_cusum_features(bars, _base_features(bars), apply_end=101)


def test_cusum_event_accumulator_resets_after_threshold_crossing() -> None:
    standardized = np.array([0.0] * 4 + [1.0] * 4 + [-1.0] * 4)
    positive, negative, drift = v4am._causal_cusum_events(
        standardized, threshold_sigma=2.0, drift_allowance_sigma=0.5
    )
    assert positive.sum() >= 1
    assert negative.sum() >= 1
    assert np.isfinite(drift).all()
    assert np.flatnonzero(positive)[0] < np.flatnonzero(negative)[0]


def test_hmm_is_train_only_and_emits_causal_normalized_posteriors() -> None:
    features = _hmm_input(900)
    transformed, diagnostics = v4am.build_hmm_regime_features(
        features, train_end=600, apply_end=765
    )
    assert diagnostics["algorithm"] == "GAUSSIAN_HMM_DIAGONAL"
    assert diagnostics["state_count"] == 3
    assert diagnostics["maximum_iterations"] == 15
    assert 1 <= diagnostics["iterations"] <= 15
    assert diagnostics["fit_end_index_exclusive"] == 600
    assert diagnostics["fit_scope"] == "OUTER_TRAIN_ONLY"
    assert diagnostics["posterior_method"] == "CAUSAL_FORWARD_FILTER"
    assert diagnostics["smoothing_used"] is False
    assert diagnostics["lookahead_used"] is False
    posterior_columns = [
        v4am.HMM_TRENDING_UP_FEATURE,
        v4am.HMM_TRENDING_DOWN_FEATURE,
        v4am.HMM_RANGING_FEATURE,
    ]
    posterior = transformed.loc[:764, posterior_columns].dropna()
    assert len(posterior) > 500
    np.testing.assert_allclose(
        np.asarray(posterior.sum(axis=1), dtype=float), 1.0, atol=1e-8
    )
    assert transformed.loc[765:, posterior_columns].isna().all().all()


def test_hmm_training_parameters_ignore_forward_validation_values() -> None:
    features = _hmm_input(900)
    _, first = v4am.build_hmm_regime_features(features, train_end=600, apply_end=765)
    altered = features.copy()
    altered.loc[600:, v4am.LOG_RETURN_FEATURE] = 0.15
    altered.loc[600:, v4am.REALIZED_VOLATILITY_FEATURE] = 0.08
    _, second = v4am.build_hmm_regime_features(altered, train_end=600, apply_end=765)
    assert first["state_means_original_scale"] == second["state_means_original_scale"]
    assert first["transition_matrix"] == second["transition_matrix"]


def test_hmm_forward_filter_is_prefix_causal() -> None:
    features = _hmm_input(900)
    first, _ = v4am.build_hmm_regime_features(features, train_end=600, apply_end=800)
    altered = features.copy()
    altered.loc[800:, v4am.LOG_RETURN_FEATURE] = -0.2
    altered.loc[800:, v4am.REALIZED_VOLATILITY_FEATURE] = 0.1
    second, _ = v4am.build_hmm_regime_features(altered, train_end=600, apply_end=900)
    for column in (
        v4am.HMM_TRENDING_UP_FEATURE,
        v4am.HMM_TRENDING_DOWN_FEATURE,
        v4am.HMM_RANGING_FEATURE,
    ):
        first_values = [float(value) for value in first.loc[:799, column].tolist()]
        second_values = [float(value) for value in second.loc[:799, column].tolist()]
        np.testing.assert_allclose(first_values, second_values, equal_nan=True)


def test_hmm_forward_filter_batched_emissions_match_stepwise_causal_reference() -> None:
    observations = np.asarray(
        [
            [np.nan, np.nan],
            [-0.2, 0.1],
            [0.1, 0.2],
            [np.nan, np.nan],
            [0.3, 0.4],
            [0.4, 0.2],
            [np.nan, np.nan],
            [-0.1, 0.3],
        ],
        dtype=float,
    )
    valid_mask = np.asarray([False, True, True, False, True, True, False, True])
    start = np.asarray([0.2, 0.5, 0.3], dtype=float)
    transition = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        dtype=float,
    )
    means = np.asarray([[-0.2, 0.1], [0.2, 0.2], [0.5, 0.4]], dtype=float)
    variances = np.asarray([[0.3, 0.2], [0.2, 0.4], [0.4, 0.3]], dtype=float)
    apply_end = 7
    actual = v4am._causal_forward_filter(
        observations, valid_mask, apply_end, start, transition, means, variances
    )
    expected = np.full((len(observations), len(start)), np.nan, dtype=float)
    log_start = np.log(np.clip(start, 1e-300, 1.0))
    log_transition = np.log(np.clip(transition, 1e-300, 1.0))
    previous: np.ndarray | None = None
    for index in range(apply_end):
        if not bool(valid_mask[index]):
            previous = None
            continue
        emission = v4am._emission_log_probabilities(
            observations[index : index + 1], means, variances
        )[0]
        if previous is None:
            current = log_start + emission
        else:
            current = (
                v4am._logsumexp(previous[:, np.newaxis] + log_transition, axis=0)
                + emission
            )
        normalizer = float(v4am._logsumexp(current, axis=0))
        previous = current - normalizer
        expected[index] = np.exp(previous)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14, equal_nan=True)
    assert np.isnan(actual[7]).all()


def test_hmm_rejects_invalid_train_window_and_missing_features() -> None:
    features = _hmm_input(300)
    with pytest.raises(ValueError, match="v4am_hmm_training_window_invalid"):
        v4am.build_hmm_regime_features(features, train_end=99, apply_end=250)
    with pytest.raises(ValueError, match="v4am_hmm_input_features_missing"):
        v4am.build_hmm_regime_features(
            DataFrame({"other": np.ones(300)}), train_end=200, apply_end=250
        )
