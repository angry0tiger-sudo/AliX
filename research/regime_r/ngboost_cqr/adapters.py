"""HGB quantile + MAPIE CQR adapter for isolated Regime R research.

The adapter has no data-source, network, venue, order or backtest integration.
Callers supply separate in-memory training and conformalization arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
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


OWNER_EXECUTION_ENV = "ALIX_REGIME_R_EXECUTION_AUTHORIZED"
PREREGISTRATION_SHA256 = (
    "92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa"
)
PREREGISTRATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "regime_r"
    / "hgbq_cqr_preregistration.json"
)


class ExecutionAuthorizationError(PermissionError):
    """Raised when an Owner authorization is absent for any model fit."""


def _require_owner_execution_authorization() -> None:
    """Require the exact Owner-set environment flag before every fit attempt."""
    if os.environ.get(OWNER_EXECUTION_ENV) != "true":
        raise ExecutionAuthorizationError("regime_r_fit_requires_owner_authorization")


def _config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one required object section from the frozen JSON configuration."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ContractViolation("hgbq_preregistration_config_incompatible")
    return value


def _validate_model_config(model: dict[str, Any]) -> dict[str, Any]:
    """Validate every HGB setting and quantile against the committed contract."""
    if model.get("estimator") != "sklearn.ensemble.HistGradientBoostingRegressor":
        raise ContractViolation("hgbq_preregistration_model_incompatible")
    common = model.get("common_parameters")
    if not isinstance(common, dict):
        raise ContractViolation("hgbq_preregistration_model_incompatible")
    expected: dict[str, Any] = {
        "loss": "quantile",
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "max_leaf_nodes": 8,
        "min_samples_leaf": 20,
        "l2_regularization": 1,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 42,
    }
    if any(
        type(common.get(name)) is not type(value) or common.get(name) != value
        for name, value in expected.items()
    ):
        raise ContractViolation("hgbq_preregistration_model_parameters_incompatible")
    quantiles = model.get("quantile_estimators")
    expected_quantiles = [
        {"name": "lower", "quantile": 0.05},
        {"name": "median", "quantile": 0.5},
        {"name": "upper", "quantile": 0.95},
    ]
    if quantiles != expected_quantiles:
        raise ContractViolation("hgbq_preregistration_quantiles_incompatible")
    return common


def _validate_conformalization_config(
    conformalization: dict[str, Any],
) -> float:
    """Validate the pinned MAPIE API, order, and nominal coverage."""
    if (
        conformalization.get("library") != "MAPIE"
        or conformalization.get("version") != "1.6.0"
        or conformalization.get("estimator")
        != "mapie.regression.ConformalizedQuantileRegressor"
        or conformalization.get("confidence_level") != 0.9
        or type(conformalization.get("confidence_level")) is not float
        or conformalization.get("prefit") is not True
        or conformalization.get("estimator_order") != ["lower", "upper", "median"]
        or conformalization.get(
            "calibration_data_must_be_disjoint_from_training_and_test"
        )
        is not True
        or conformalization.get("test_data_must_never_be_used_for_conformalization")
        is not True
    ):
        raise ContractViolation("hgbq_preregistration_conformalization_incompatible")
    return float(conformalization["confidence_level"])


def _validate_preregistration_config(config: dict[str, Any]) -> None:
    """Reject a correctly hashed but structurally incompatible configuration."""
    preregistration = _config_section(config, "preregistration")
    if (
        preregistration.get("status") != "PREREGISTERED_LOCKED"
        or preregistration.get("real_data_fit_authorized") is not False
        or preregistration.get("fit_authorized_in_g3") is not False
        or preregistration.get("holdout_access_authorized") is not False
        or preregistration.get("economic_backtest_authorized") is not False
    ):
        raise ContractViolation("hgbq_preregistration_authorization_contract_invalid")
    model = _config_section(config, "model")
    _validate_model_config(model)
    conformalization = _config_section(config, "conformalization")
    _validate_conformalization_config(conformalization)
    horizons = _config_section(config, "prediction_horizons")
    if horizons.get("unit") != "hours" or horizons.get("values") != [1, 4, 24]:
        raise ContractViolation("hgbq_preregistration_horizons_incompatible")


def _load_preregistered_config() -> tuple[dict[str, Any], str]:
    """Load only the byte-committed G3 configuration; there is no fallback."""
    try:
        raw_config = PREREGISTRATION_CONFIG_PATH.read_bytes()
    except OSError as error:
        raise ContractViolation("hgbq_preregistration_config_unavailable") from error
    config_sha256 = hashlib.sha256(raw_config).hexdigest()
    if config_sha256 != PREREGISTRATION_SHA256:
        raise ContractViolation("hgbq_preregistration_config_sha256_mismatch")
    try:
        config = json.loads(raw_config.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractViolation("hgbq_preregistration_config_invalid_json") from error
    if not isinstance(config, dict):
        raise ContractViolation("hgbq_preregistration_config_incompatible")
    _validate_preregistration_config(config)
    return config, config_sha256


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



class _OrderedQuantileView:
    """Expose one column of a row-wise monotone rearrangement to MAPIE."""

    def __init__(
        self,
        models: tuple[Any, Any, Any],
        quantile_index: int,
    ) -> None:
        if quantile_index not in (0, 1, 2):
            raise ContractViolation("ordered_quantile_view_index_invalid")
        self._models = models
        self._quantile_index = quantile_index
        self.n_features_in_ = int(models[0].n_features_in_)

    def fit(self, X: ArrayLike, y: ArrayLike) -> _OrderedQuantileView:
        """Reject accidental fitting; the view is always used in MAPIE prefit mode."""
        del X, y
        raise ContractViolation("ordered_quantile_view_is_prefit_only")

    def predict(self, X: ArrayLike) -> FloatArray:
        """Return the selected ordered quantile after deterministic rearrangement."""
        try:
            features = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractViolation("ordered_quantile_features_must_be_numeric") from error
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != self.n_features_in_
            or not np.isfinite(features).all()
        ):
            raise ContractViolation("ordered_quantile_features_invalid")
        return _ordered_quantile_predictions(self._models, features)[
            :, self._quantile_index
        ]


def _ordered_quantile_predictions(
    models: tuple[Any, Any, Any],
    features: FloatArray,
) -> FloatArray:
    """Rearrange independently fitted quantile predictions into monotone order."""
    predictions = [
        np.asarray(model.predict(features), dtype=np.float64)
        for model in models
    ]
    expected_shape = (features.shape[0],)
    if any(values.shape != expected_shape for values in predictions):
        raise ContractViolation("hgbq_prediction_shape_invalid")
    raw_values = np.column_stack(predictions)
    if not np.isfinite(raw_values).all():
        raise ContractViolation("hgbq_predictions_must_be_finite")
    return np.asarray(np.sort(raw_values, axis=1), dtype=np.float64)


class HGBQCQRAdapter:
    """Fit three HGB quantile learners and conformalize them with MAPIE CQR.

    The forecast summary is quantile-only. The MAPIE CQR estimator list is
    deliberately ordered [lower, upper, median], as required by MAPIE 1.6.0
    prefit mode. The public DistributionForecast keeps the semantic order
    [lower, median, upper].
    """

    def __init__(self, *, horizon: str = "1h") -> None:
        config, config_sha256 = _load_preregistered_config()
        model_config = _config_section(config, "model")
        common_parameters = _validate_model_config(model_config)
        conformalization = _config_section(config, "conformalization")

        self.horizon = _validate_horizon(horizon)
        self.config_sha256 = config_sha256
        self.confidence_level = _validate_conformalization_config(conformalization)
        self._common_model_parameters = dict(common_parameters)
        self.quantile_levels: tuple[float, float, float] = (
            float(model_config["quantile_estimators"][0]["quantile"]),
            float(model_config["quantile_estimators"][1]["quantile"]),
            float(model_config["quantile_estimators"][2]["quantile"]),
        )
        self.learning_rate = float(common_parameters["learning_rate"])
        self.max_depth = int(common_parameters["max_depth"])
        self.max_iter = int(common_parameters["max_iter"])
        self.l2_regularization = float(common_parameters["l2_regularization"])
        self.min_samples_leaf = int(common_parameters["min_samples_leaf"])
        self.random_state = int(common_parameters["random_state"])
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

    def _new_quantile_model(self, quantile: float) -> Any:
        """Build one learner using only the frozen common settings."""
        parameters = dict(self._common_model_parameters)
        parameters["quantile"] = quantile
        return HistGradientBoostingRegressor(**parameters)

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
        _require_owner_execution_authorization()
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

        lower_model = self._new_quantile_model(self.quantile_levels[0])
        median_model = self._new_quantile_model(self.quantile_levels[1])
        upper_model = self._new_quantile_model(self.quantile_levels[2])

        lower_model.fit(train_features, train_target)
        median_model.fit(train_features, train_target)
        upper_model.fit(train_features, train_target)

        fitted_models = (lower_model, median_model, upper_model)
        ordered_estimators = (
            _OrderedQuantileView(fitted_models, 0),
            _OrderedQuantileView(fitted_models, 2),
            _OrderedQuantileView(fitted_models, 1),
        )
        conformalizer = ConformalizedQuantileRegressor(
            estimator=ordered_estimators,
            confidence_level=self.confidence_level,
            prefit=True,
        )
        conformalizer.conformalize(calibration_features, calibration_target)

        self._models = fitted_models
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
        ordered_values = _ordered_quantile_predictions(models, features)
        lower_value, median_value, upper_value = (
            float(value) for value in ordered_values[0]
        )

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

    def predict_batch_intervals(
        self,
        X_synthetic: ArrayLike,
    ) -> tuple[tuple[DistributionForecast, ...], tuple[ConformalInterval, ...]]:
        """Predict quantile summaries and CQR intervals for a finite feature batch."""
        models, conformalizer, calibration_size = self._fitted_components()
        features = self._feature_matrix(X_synthetic, "prediction_features")
        if features.shape[1] != models[0].n_features_in_:
            raise ContractViolation("prediction_feature_width_mismatch")

        ordered_values = _ordered_quantile_predictions(models, features)
        row_count = features.shape[0]

        point_predictions, raw_intervals = conformalizer.predict_interval(
            features,
            allow_infinite_bounds=False,
            symmetric_correction=False,
        )
        point_values = np.asarray(point_predictions, dtype=np.float64)
        intervals = np.asarray(raw_intervals, dtype=np.float64)
        if point_values.shape != (row_count,) or intervals.shape != (row_count, 2, 1):
            raise ContractViolation("mapie_cqr_prediction_shape_invalid")
        if not np.isfinite(point_values).all() or not np.isfinite(intervals).all():
            raise ContractViolation("mapie_cqr_prediction_must_be_finite")

        forecasts: list[DistributionForecast] = []
        calibrated_intervals: list[ConformalInterval] = []
        for index in range(row_count):
            forecast = DistributionForecast(
                horizon=self.horizon,
                mean=None,
                dispersion=None,
                quantile_levels=self.quantile_levels,
                quantile_values=tuple(
                    float(value) for value in ordered_values[index]
                ),
                output_semantics="quantile_summary",
            )
            validate_distribution_forecast(forecast)
            if not np.isclose(
                float(point_values[index]),
                forecast.quantile_values[1],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ContractViolation("mapie_cqr_median_prediction_mismatch")

            raw_lower, _, raw_upper = forecast.quantile_values
            calibrated_lower = float(intervals[index, 0, 0])
            calibrated_upper = float(intervals[index, 1, 0])
            interval = make_cqr_interval(
                lower_quantile=raw_lower,
                upper_quantile=raw_upper,
                lower_correction=raw_lower - calibrated_lower,
                upper_correction=calibrated_upper - raw_upper,
                nominal_coverage=self.confidence_level,
                calibration_size=calibration_size,
            )
            validate_conformal_interval(interval)
            forecasts.append(forecast)
            calibrated_intervals.append(interval)

        return tuple(forecasts), tuple(calibrated_intervals)

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
