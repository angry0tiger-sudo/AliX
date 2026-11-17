"""Synthetic-only API, adapter and fail-closed decision tests for HGBQ-CQR."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from research.regime_r.ngboost_cqr.adapters import HGBQCQRAdapter
from research.regime_r.ngboost_cqr.contracts import (
    ConformalInterval,
    ContractViolation,
    DistributionForecast,
    validate_conformal_interval,
    validate_distribution_forecast,
)
from research.regime_r.ngboost_cqr.adapters import (
    GateAction,
    evaluate_cost_gate,
    evaluate_cost_scenarios,
)

FloatArray = NDArray[np.float64]


def _synthetic_data(sample_count: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Create a deterministic smooth signal with synthetic observation noise."""
    generator = np.random.default_rng(seed)
    features = np.linspace(-1.5, 1.5, sample_count, dtype=np.float64).reshape(-1, 1)
    noise = generator.normal(0.0, 0.001, size=sample_count).astype(np.float64)
    target = 0.004 * features[:, 0] + 0.002 * np.sin(2.0 * features[:, 0]) + noise
    return features, target


def _fitted_adapter() -> HGBQCQRAdapter:
    """Fit and conformalize only on independent deterministic synthetic arrays."""
    X_train, y_train = _synthetic_data(320, 42)
    X_calibration, y_calibration = _synthetic_data(160, 43)
    adapter = HGBQCQRAdapter(
        horizon="4h",
        confidence_level=0.90,
        l2_regularization=0.0,
        max_iter=40,
        min_samples_leaf=10,
        random_state=42,
    )
    adapter.fit(X_train, y_train, X_calibration, y_calibration)
    return adapter


def _interval(lower_bound: float, upper_bound: float) -> ConformalInterval:
    """Build a valid test interval without implying any empirical coverage."""
    return ConformalInterval(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        nominal_coverage=0.90,
        lower_correction=0.0,
        upper_correction=0.0,
        calibration_size=100,
    )


def test_adapter_fits_hgb_quantiles_and_returns_quantile_only_forecast() -> None:
    adapter = _fitted_adapter()
    forecast = adapter.predict_forecast(np.asarray([[0.15]], dtype=np.float64))

    assert adapter.is_fitted
    assert forecast.horizon == "4h"
    assert forecast.quantile_levels == pytest.approx((0.05, 0.5, 0.95))
    lower, median, upper = forecast.quantile_values
    assert lower <= median <= upper
    assert forecast.output_semantics == "quantile_summary"
    assert forecast.mean is None
    assert forecast.dispersion is None
    validate_distribution_forecast(forecast)


def test_mapie_cqr_returns_a_valid_conformal_interval() -> None:
    adapter = _fitted_adapter()
    interval = adapter.predict_interval(np.asarray([[0.15]], dtype=np.float64))

    validate_conformal_interval(interval)
    assert interval.nominal_coverage == pytest.approx(0.90)
    assert interval.calibration_size == 160
    assert interval.lower_bound <= interval.upper_bound


def test_adapter_fails_closed_before_fit_and_after_failed_fit() -> None:
    adapter = HGBQCQRAdapter(min_samples_leaf=10)
    row = np.asarray([[0.1]], dtype=np.float64)
    with pytest.raises(ContractViolation, match="adapter_not_fitted"):
        adapter.predict_forecast(row)
    with pytest.raises(ContractViolation, match="adapter_not_fitted"):
        adapter.predict_interval(row)

    X_train, y_train = _synthetic_data(20, 1)
    X_cal, y_cal = _synthetic_data(8, 2)
    with pytest.raises(
        ContractViolation,
        match="calibration_sample_size_below_finite_quantile_minimum",
    ):
        adapter.fit(X_train, y_train, X_cal, y_cal)
    assert not adapter.is_fitted


def test_fit_rejects_invalid_dimensions_and_non_finite_arrays() -> None:
    X_train, y_train = _synthetic_data(80, 10)
    X_cal, y_cal = _synthetic_data(40, 11)
    adapter = HGBQCQRAdapter(min_samples_leaf=8)

    with pytest.raises(
        ContractViolation, match="training_features_must_be_non_empty_2d"
    ):
        adapter.fit(X_train[:, 0], y_train, X_cal, y_cal)
    invalid_features = X_train.copy()
    invalid_features[0, 0] = np.nan
    with pytest.raises(ContractViolation, match="training_features_must_be_finite"):
        adapter.fit(invalid_features, y_train, X_cal, y_cal)


def test_fit_rejects_target_lengths_and_feature_width_mismatch() -> None:
    X_train, y_train = _synthetic_data(80, 20)
    X_cal, y_cal = _synthetic_data(40, 21)
    adapter = HGBQCQRAdapter(min_samples_leaf=8)

    with pytest.raises(
        ContractViolation, match="training_feature_target_length_mismatch"
    ):
        adapter.fit(X_train, y_train[:-1], X_cal, y_cal)
    X_cal_wide = np.column_stack((X_cal, X_cal[:, 0]))
    with pytest.raises(
        ContractViolation,
        match="training_calibration_feature_width_mismatch",
    ):
        adapter.fit(X_train, y_train, X_cal_wide, y_cal)


def test_fit_rejects_insufficient_training_rows_and_non_finite_targets() -> None:
    X_train, y_train = _synthetic_data(6, 30)
    X_cal, y_cal = _synthetic_data(40, 31)
    adapter = HGBQCQRAdapter(min_samples_leaf=8)

    with pytest.raises(
        ContractViolation,
        match="training_sample_size_below_min_samples_leaf",
    ):
        adapter.fit(X_train, y_train, X_cal, y_cal)
    valid_X, valid_y = _synthetic_data(40, 32)
    invalid_y = valid_y.copy()
    invalid_y[0] = np.inf
    with pytest.raises(
        ContractViolation, match="conformalization_target_must_be_finite"
    ):
        adapter.fit(valid_X, valid_y, X_cal, invalid_y)


def test_prediction_requires_one_row_with_the_fitted_feature_width() -> None:
    adapter = _fitted_adapter()
    with pytest.raises(ContractViolation, match="prediction_requires_exactly_one_row"):
        adapter.predict_forecast(np.zeros((2, 1), dtype=np.float64))
    with pytest.raises(ContractViolation, match="prediction_feature_width_mismatch"):
        adapter.predict_interval(np.zeros((1, 2), dtype=np.float64))
    with pytest.raises(ContractViolation, match="prediction_features_must_be_finite"):
        adapter.predict_forecast(np.asarray([[np.nan]], dtype=np.float64))


@pytest.mark.parametrize(
    (
        "horizon",
        "confidence_level",
        "l2_regularization",
        "max_iter",
        "min_samples_leaf",
        "random_state",
        "message",
    ),
    (
        ("2h", 0.90, 0.0, 80, 12, 42, "horizon_unsupported"),
        ("1h", 1.0, 0.0, 80, 12, 42, "confidence_level_invalid"),
        ("1h", 0.90, -0.1, 80, 12, 42, "l2_regularization_invalid"),
        ("1h", 0.90, 0.0, 0, 12, 42, "max_iter_invalid"),
        ("1h", 0.90, 0.0, 80, 0, 42, "min_samples_leaf_invalid"),
        ("1h", 0.90, 0.0, 80, 12, True, "random_state_invalid"),
    ),
)
def test_adapter_rejects_invalid_configuration(
    horizon: str,
    confidence_level: float,
    l2_regularization: float,
    max_iter: int,
    min_samples_leaf: int,
    random_state: int,
    message: str,
) -> None:
    with pytest.raises(ContractViolation, match=message):
        HGBQCQRAdapter(
            horizon=horizon,
            confidence_level=confidence_level,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )


def test_fit_rejects_non_numeric_and_empty_inputs() -> None:
    X_train, y_train = _synthetic_data(80, 40)
    X_cal, y_cal = _synthetic_data(40, 41)
    adapter = HGBQCQRAdapter(min_samples_leaf=8)

    with pytest.raises(ContractViolation, match="training_features_must_be_numeric"):
        adapter.fit(np.asarray([["not numeric"]]), y_train[:1], X_cal, y_cal)
    with pytest.raises(ContractViolation, match="training_target_must_be_numeric"):
        adapter.fit(X_train, np.asarray(["not numeric"]), X_cal, y_cal)
    with pytest.raises(
        ContractViolation, match="training_features_must_be_non_empty_2d"
    ):
        adapter.fit(np.empty((0, 1)), np.empty((0,)), X_cal, y_cal)


def test_fit_rejects_conformalization_target_length_mismatch() -> None:
    X_train, y_train = _synthetic_data(80, 42)
    X_cal, y_cal = _synthetic_data(40, 43)
    adapter = HGBQCQRAdapter(min_samples_leaf=8)

    with pytest.raises(
        ContractViolation,
        match="conformalization_feature_target_length_mismatch",
    ):
        adapter.fit(X_train, y_train, X_cal, y_cal[:-1])


def test_forecast_fails_closed_when_hgb_quantiles_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _fitted_adapter()
    assert adapter._models is not None
    lower_model, median_model, upper_model = adapter._models
    monkeypatch.setattr(
        lower_model,
        "predict",
        lambda _: np.asarray([0.02], dtype=np.float64),
    )
    monkeypatch.setattr(
        median_model,
        "predict",
        lambda _: np.asarray([0.01], dtype=np.float64),
    )
    monkeypatch.setattr(
        upper_model,
        "predict",
        lambda _: np.asarray([0.03], dtype=np.float64),
    )

    with pytest.raises(
        ContractViolation,
        match="forecast_quantile_values_not_monotonic",
    ):
        adapter.predict_forecast(np.asarray([[0.15]], dtype=np.float64))


@pytest.mark.parametrize(
    ("point_predictions", "intervals", "message"),
    (
        (
            np.asarray([0.0], dtype=np.float64),
            np.zeros((1, 2), dtype=np.float64),
            "mapie_cqr_prediction_shape_invalid",
        ),
        (
            np.asarray([0.0], dtype=np.float64),
            np.full((1, 2, 1), np.nan, dtype=np.float64),
            "mapie_cqr_prediction_must_be_finite",
        ),
        (
            np.asarray([99.0], dtype=np.float64),
            np.asarray([[[0.0], [1.0]]], dtype=np.float64),
            "mapie_cqr_median_prediction_mismatch",
        ),
    ),
)
def test_mapie_output_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    point_predictions: FloatArray,
    intervals: FloatArray,
    message: str,
) -> None:
    adapter = _fitted_adapter()
    assert adapter._conformalizer is not None
    fake_conformalizer = Mock(
        predict_interval=Mock(return_value=(point_predictions, intervals))
    )
    monkeypatch.setattr(adapter, "_conformalizer", fake_conformalizer)

    with pytest.raises(ContractViolation, match=message):
        adapter.predict_interval(np.asarray([[0.15]], dtype=np.float64))


def test_forecast_quantile_summary_does_not_fabricate_mean_or_dispersion() -> None:
    quantile_only = DistributionForecast(
        horizon="1h",
        mean=None,
        dispersion=None,
        quantile_levels=(0.05, 0.5, 0.95),
        quantile_values=(-0.01, 0.001, 0.02),
        output_semantics="quantile_summary",
    )
    validate_distribution_forecast(quantile_only)

    invalid = DistributionForecast(
        horizon="1h",
        mean=0.001,
        dispersion=0.01,
        quantile_levels=(0.05, 0.5, 0.95),
        quantile_values=(-0.01, 0.001, 0.02),
        output_semantics="quantile_summary",
    )
    with pytest.raises(
        ContractViolation,
        match="quantile_summary_must_not_claim_mean_or_dispersion",
    ):
        validate_distribution_forecast(invalid)


def test_conformal_interval_validator_rejects_non_finite_and_invalid_coverage() -> None:
    invalid_value = ConformalInterval(np.nan, 0.1, 0.9, 0.0, 0.0, 100)
    with pytest.raises(
        ContractViolation,
        match="conformal_interval_values_must_be_finite",
    ):
        validate_conformal_interval(invalid_value)
    invalid_coverage = ConformalInterval(0.0, 0.1, 1.0, 0.0, 0.0, 100)
    with pytest.raises(
        ContractViolation,
        match="conformal_interval_coverage_invalid",
    ):
        validate_conformal_interval(invalid_coverage)


def test_conformal_interval_validator_rejects_invalid_bounds_and_size() -> None:
    with pytest.raises(ContractViolation, match="conformal_interval_bounds_inverted"):
        validate_conformal_interval(_interval(0.02, 0.01))
    invalid_size = ConformalInterval(0.0, 0.1, 0.9, 0.0, 0.0, 0)
    with pytest.raises(
        ContractViolation,
        match="conformal_interval_calibration_size_invalid",
    ):
        validate_conformal_interval(invalid_size)


def test_low_uncertainty_lower_bound_enters_both_cost_scenarios() -> None:
    decisions = evaluate_cost_scenarios(_interval(0.012, 0.016))

    assert decisions[50].action is GateAction.ENTER
    assert decisions[90].action is GateAction.ENTER


def test_negative_lower_bound_abstains_for_both_cost_scenarios() -> None:
    decisions = evaluate_cost_scenarios(_interval(-0.004, 0.02))

    assert decisions[50].action is GateAction.ABSTAIN
    assert decisions[90].action is GateAction.ABSTAIN


def test_lower_bound_can_pass_50_bps_but_fail_90_bps() -> None:
    decisions = evaluate_cost_scenarios(_interval(0.007, 0.015))

    assert decisions[50].action is GateAction.ENTER
    assert decisions[90].action is GateAction.ABSTAIN


def test_cost_gate_is_strict_and_fails_closed_for_invalid_inputs() -> None:
    exact_50_bps = evaluate_cost_gate(_interval(0.005, 0.02), 50)
    invalid_cost = evaluate_cost_gate(_interval(0.02, 0.03), 75)
    missing_interval = evaluate_cost_gate(None, 50)
    inverted_interval = evaluate_cost_gate(_interval(0.03, 0.01), 50)

    assert exact_50_bps.action is GateAction.ABSTAIN
    assert invalid_cost.action is GateAction.ABSTAIN
    assert missing_interval.action is GateAction.ABSTAIN
    assert inverted_interval.action is GateAction.ABSTAIN
