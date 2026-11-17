"""HGB quantile + MAPIE CQR adapter for isolated Regime R research.

The adapter has no data-source, network, venue, order or backtest integration.
Callers supply separate in-memory training and conformalization arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from research.regime_r.ngboost_cqr.contracts import (
    COST_SCENARIOS_BPS,
    HORIZON_BARS,
    ConformalInterval,
    ContractViolation,
    DistributionForecast,
    make_cqr_interval,
    passes_cost_gate,
    validate_conformal_interval,
    validate_distribution_forecast,
)

ConformalizedQuantileRegressor: Any = getattr(
    import_module("mapie.regression"), "ConformalizedQuantileRegressor"
)
HistGradientBoostingRegressor: Any = getattr(
    import_module("sklearn.ensemble"), "HistGradientBoostingRegressor"
)

FloatArray = NDArray[np.float64]


def _validate_horizon(horizon: str) -> str:
    """Require one of the horizons defined by the existing Regime R contract."""
    if horizon not in HORIZON_BARS:
        raise ContractViolation("hgbq_horizon_unsupported")
    return horizon


def _finite_numeric_value(value: object, error_code: str) -> float:
    """Convert one numeric option while rejecting booleans and non-finite values."""
    if isinstance(value, bool):
        raise ContractViolation(error_code)
    if not isinstance(value, (int, float)):
        raise ContractViolation(error_code)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ContractViolation(error_code)
    return numeric_value


def _validate_confidence_level(value: float) -> float:
    """Require a finite confidence level strictly between zero and one."""
    confidence = _finite_numeric_value(value, "hgbq_confidence_level_invalid")
    if not 0.0 < confidence < 1.0:
        raise ContractViolation("hgbq_confidence_level_invalid")
    return confidence


def _validate_non_negative_float(value: float, error_code: str) -> float:
    """Require a finite non-negative floating-point configuration."""
    numeric_value = _finite_numeric_value(value, error_code)
    if numeric_value < 0.0:
        raise ContractViolation(error_code)
    return numeric_value


def _validate_positive_integer(value: int, error_code: str) -> int:
    """Require a positive integer, explicitly excluding bool values."""
    if isinstance(value, bool):
        raise ContractViolation(error_code)
    if not isinstance(value, int):
        raise ContractViolation(error_code)
    if value < 1:
        raise ContractViolation(error_code)
    return value


def _validate_random_state(value: int) -> int:
    """Require a deterministic integer seed, excluding bool values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolation("hgbq_random_state_invalid")
    return value


class HGBQCQRAdapter:
    """Fit three HGB quantile learners and conformalize them with MAPIE CQR.

    The forecast summary is quantile-only. The MAPIE CQR estimator list is
    deliberately ordered [lower, upper, median], as required by MAPIE 1.6.0
    prefit mode. The public DistributionForecast keeps the semantic order
    [lower, median, upper].
    """

    def __init__(
        self,
        *,
        horizon: str = "1h",
        confidence_level: float = 0.90,
        l2_regularization: float = 0.0,
        max_iter: int = 80,
        min_samples_leaf: int = 12,
        random_state: int = 42,
    ) -> None:
        self.horizon = _validate_horizon(horizon)
        self.confidence_level = _validate_confidence_level(confidence_level)
        self.l2_regularization = _validate_non_negative_float(
            l2_regularization,
            "hgbq_l2_regularization_invalid",
        )
        self.max_iter = _validate_positive_integer(max_iter, "hgbq_max_iter_invalid")
        self.min_samples_leaf = _validate_positive_integer(
            min_samples_leaf,
            "hgbq_min_samples_leaf_invalid",
        )
        self.random_state = _validate_random_state(random_state)
        self.quantile_levels = (
            (1.0 - self.confidence_level) / 2.0,
            0.5,
            (1.0 + self.confidence_level) / 2.0,
        )
        self._models: (
            tuple[
                HistGradientBoostingRegressor,
                HistGradientBoostingRegressor,
                HistGradientBoostingRegressor,
            ]
            | None
        ) = None
        self._conformalizer: ConformalizedQuantileRegressor | None = None
        self._calibration_size = 0

    @property
    def is_fitted(self) -> bool:
        """Return whether both quantile learners and CQR calibration are ready."""
        return self._models is not None and self._conformalizer is not None

    @staticmethod
    def _feature_matrix(values: ArrayLike, name: str) -> FloatArray:
        """Convert a feature input into a finite, non-empty float64 matrix."""
        try:
            matrix = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractViolation(f"{name}_must_be_numeric") from error
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ContractViolation(f"{name}_must_be_non_empty_2d")
        if not np.isfinite(matrix).all():
            raise ContractViolation(f"{name}_must_be_finite")
        return matrix

    @staticmethod
    def _target_vector(values: ArrayLike, name: str) -> FloatArray:
        """Convert a target input into a finite, non-empty float64 vector."""
        try:
            target = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractViolation(f"{name}_must_be_numeric") from error
        if target.ndim != 1 or target.shape[0] < 1:
            raise ContractViolation(f"{name}_must_be_non_empty_1d")
        if not np.isfinite(target).all():
            raise ContractViolation(f"{name}_must_be_finite")
        return target

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
    ) -> HGBQCQRAdapter:
        """Fit only the supplied training arrays and conformalize on a separate set.

        G2 invokes this method only with deterministic synthetic test arrays.
        Training and conformalization inputs are explicit and are never fetched
        from Nobitex, Binance, Bybit, a file path, or any external service.
        """
        self._models = None
        self._conformalizer = None
        self._calibration_size = 0

        train_features = self._feature_matrix(X_train, "training_features")
        train_target = self._target_vector(y_train, "training_target")
        calibration_features = self._feature_matrix(
            X_conformalize, "conformalization_features"
        )
        calibration_target = self._target_vector(
            y_conformalize, "conformalization_target"
        )

        if train_features.shape[0] != train_target.shape[0]:
            raise ContractViolation("training_feature_target_length_mismatch")
        if calibration_features.shape[0] != calibration_target.shape[0]:
            raise ContractViolation("conformalization_feature_target_length_mismatch")
        if train_features.shape[1] != calibration_features.shape[1]:
            raise ContractViolation("training_calibration_feature_width_mismatch")
        if train_features.shape[0] < self.min_samples_leaf:
            raise ContractViolation("training_sample_size_below_min_samples_leaf")

        minimum_calibration_size = math.ceil(
            self.confidence_level / (1.0 - self.confidence_level)
        )
        if calibration_target.shape[0] < minimum_calibration_size:
            raise ContractViolation(
                "calibration_sample_size_below_finite_quantile_minimum"
            )

        lower_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile_levels[0],
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False,
            random_state=self.random_state,
        )
        median_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile_levels[1],
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False,
            random_state=self.random_state,
        )
        upper_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile_levels[2],
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            early_stopping=False,
            random_state=self.random_state,
        )

        lower_model.fit(train_features, train_target)
        median_model.fit(train_features, train_target)
        upper_model.fit(train_features, train_target)

        conformalizer = ConformalizedQuantileRegressor(
            estimator=[lower_model, upper_model, median_model],
            confidence_level=self.confidence_level,
            prefit=True,
        )
        conformalizer.conformalize(calibration_features, calibration_target)

        self._models = (lower_model, median_model, upper_model)
        self._conformalizer = conformalizer
        self._calibration_size = int(calibration_target.shape[0])
        return self

    def _fitted_components(
        self,
    ) -> tuple[
        tuple[
            HistGradientBoostingRegressor,
            HistGradientBoostingRegressor,
            HistGradientBoostingRegressor,
        ],
        ConformalizedQuantileRegressor,
        int,
    ]:
        """Return fitted components or fail closed before any prediction."""
        if self._models is None or self._conformalizer is None:
            raise ContractViolation("hgbq_cqr_adapter_not_fitted_and_conformalized")
        return self._models, self._conformalizer, self._calibration_size

    def _single_feature_row(self, values: ArrayLike) -> FloatArray:
        """Require one finite prediction row with the fitted feature width."""
        models, _, _ = self._fitted_components()
        matrix = self._feature_matrix(values, "prediction_features")
        if matrix.shape[0] != 1:
            raise ContractViolation("prediction_requires_exactly_one_row")
        if matrix.shape[1] != models[0].n_features_in_:
            raise ContractViolation("prediction_feature_width_mismatch")
        return matrix

    def predict_forecast(self, X_synthetic: ArrayLike) -> DistributionForecast:
        """Return one ordered quantile-only forecast for a single feature row."""
        models, _, _ = self._fitted_components()
        features = self._single_feature_row(X_synthetic)
        lower_model, median_model, upper_model = models
        lower_value = float(lower_model.predict(features)[0])
        median_value = float(median_model.predict(features)[0])
        upper_value = float(upper_model.predict(features)[0])

        forecast = DistributionForecast(
            horizon=self.horizon,
            mean=None,
            dispersion=None,
            quantile_levels=self.quantile_levels,
            quantile_values=(lower_value, median_value, upper_value),
            output_semantics="quantile_summary",
        )
        validate_distribution_forecast(forecast)
        return forecast

    def predict_interval(self, X_synthetic: ArrayLike) -> ConformalInterval:
        """Return MAPIE's calibrated interval for one feature row."""
        _, conformalizer, calibration_size = self._fitted_components()
        features = self._single_feature_row(X_synthetic)
        forecast = self.predict_forecast(features)
        point_predictions, intervals = conformalizer.predict_interval(
            features,
            allow_infinite_bounds=False,
            symmetric_correction=False,
        )

        if point_predictions.shape != (1,) or intervals.shape != (1, 2, 1):
            raise ContractViolation("mapie_cqr_prediction_shape_invalid")
        if not np.isfinite(point_predictions).all() or not np.isfinite(intervals).all():
            raise ContractViolation("mapie_cqr_prediction_must_be_finite")
        if not np.isclose(
            float(point_predictions[0]),
            forecast.quantile_values[1],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ContractViolation("mapie_cqr_median_prediction_mismatch")

        calibrated_lower = float(intervals[0, 0, 0])
        calibrated_upper = float(intervals[0, 1, 0])
        raw_lower, _, raw_upper = forecast.quantile_values
        interval = make_cqr_interval(
            lower_quantile=raw_lower,
            upper_quantile=raw_upper,
            lower_correction=raw_lower - calibrated_lower,
            upper_correction=calibrated_upper - raw_upper,
            nominal_coverage=self.confidence_level,
            calibration_size=calibration_size,
        )
        validate_conformal_interval(interval)
        return interval


class GateAction(str, Enum):
    """Research-only decision labels; ENTER is not an order authorization."""

    ENTER = "ENTER"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class CostGateDecision:
    """One lower-bound decision for one preregistered round-trip cost ledger."""

    cost_bps: int
    action: GateAction
    reason: str


def evaluate_cost_gate(
    interval: ConformalInterval | None,
    cost_bps: int,
) -> CostGateDecision:
    """Require a valid conformal interval and strict lower-bound > cost.

    Any missing, malformed, non-finite or unsupported input returns ABSTAIN.
    The ENTER label only means that this research gate passed for one cost
    scenario; it never enables paper/live trading or grants order permission.
    """
    if not isinstance(interval, ConformalInterval):
        return CostGateDecision(
            cost_bps=cost_bps,
            action=GateAction.ABSTAIN,
            reason="conformal_interval_missing_or_invalid",
        )
    try:
        validate_conformal_interval(interval)
        passes = passes_cost_gate(interval.lower_bound, cost_bps)
    except ContractViolation as error:
        return CostGateDecision(
            cost_bps=cost_bps,
            action=GateAction.ABSTAIN,
            reason=str(error),
        )

    if not passes:
        return CostGateDecision(
            cost_bps=cost_bps,
            action=GateAction.ABSTAIN,
            reason="lower_bound_does_not_strictly_exceed_cost",
        )
    return CostGateDecision(
        cost_bps=cost_bps,
        action=GateAction.ENTER,
        reason="lower_bound_strictly_exceeds_cost",
    )


def evaluate_cost_scenarios(
    interval: ConformalInterval | None,
) -> dict[int, CostGateDecision]:
    """Evaluate the fixed 50/90 bps scenarios independently and fail-closed."""
    return {
        cost_bps: evaluate_cost_gate(interval, cost_bps)
        for cost_bps in COST_SCENARIOS_BPS
    }
