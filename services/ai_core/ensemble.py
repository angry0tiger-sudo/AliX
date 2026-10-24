"""Deterministic three-model ensemble; forecasts are not trading permissions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

MODEL_NAMES = ("xgboost", "random_forest", "lightgbm")


class Regressor(Protocol):
    def fit(self, features: DataFrame, target: np.ndarray) -> object: ...

    def predict(self, features: DataFrame) -> np.ndarray: ...


RegressorFactory = Callable[[], Regressor]


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    weights: tuple[tuple[str, float], ...] = (
        ("xgboost", 1.0 / 3.0),
        ("random_forest", 1.0 / 3.0),
        ("lightgbm", 1.0 / 3.0),
    )
    n_estimators: int = 48
    random_seed: int = 42
    max_depth: int = 4

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.weights)
        if names != MODEL_NAMES:
            raise ValueError("ai_ensemble_weights_must_name_all_models_in_order")
        values = tuple(weight for _, weight in self.weights)
        if any(not math.isfinite(weight) or weight < 0 for weight in values):
            raise ValueError("ai_ensemble_weights_must_be_finite_nonnegative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("ai_ensemble_weights_must_sum_to_one")
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 10
        ):
            raise ValueError("ai_ensemble_n_estimators_invalid")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth < 1
        ):
            raise ValueError("ai_ensemble_max_depth_invalid")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("ai_ensemble_random_seed_invalid")


@dataclass(frozen=True, slots=True)
class EnsemblePrediction:
    predictions: tuple[float, ...]
    per_model_predictions: tuple[tuple[str, tuple[float, ...]], ...]
    weights: tuple[tuple[str, float], ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("ai_ensemble_feature_names_missing")
        if any(not math.isfinite(value) for value in self.predictions):
            raise ValueError("ai_ensemble_prediction_non_finite")
        if any(
            not math.isfinite(value)
            for _, values in self.per_model_predictions
            for value in values
        ):
            raise ValueError("ai_ensemble_component_prediction_non_finite")


def default_model_factories(config: EnsembleConfig) -> dict[str, RegressorFactory]:
    """Build CPU-only estimators lazily so importing feature code stays lightweight."""
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    return {
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_seed,
            n_jobs=1,
            tree_method="hist",
            device="cpu",
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth + 2,
            min_samples_leaf=4,
            random_state=config.random_seed,
            n_jobs=1,
        ),
        "lightgbm": lambda: LGBMRegressor(
            objective="regression",
            n_estimators=config.n_estimators,
            learning_rate=0.03,
            num_leaves=min(15, 2 ** config.max_depth),
            max_depth=config.max_depth,
            random_state=config.random_seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        ),
    }


def _validate_training_inputs(
    training_features: DataFrame,
    target: Series | np.ndarray,
    prediction_features: DataFrame,
) -> np.ndarray:
    if training_features.empty or prediction_features.empty:
        raise ValueError("ai_ensemble_feature_data_empty")
    if tuple(training_features.columns) != tuple(prediction_features.columns):
        raise ValueError("ai_ensemble_feature_schema_mismatch")
    if training_features.columns.duplicated().any():
        raise ValueError("ai_ensemble_duplicate_feature_names")
    train = training_features.to_numpy(dtype=float)
    prediction = prediction_features.to_numpy(dtype=float)
    labels = np.asarray(target, dtype=float).reshape(-1)
    if train.shape[0] != len(labels) or len(labels) < 30:
        raise ValueError("ai_ensemble_training_sample_count_invalid")
    if train.shape[1] < 1 or prediction.shape[1] != train.shape[1]:
        raise ValueError("ai_ensemble_feature_shape_invalid")
    if not np.isfinite(train).all() or not np.isfinite(labels).all():
        raise ValueError("ai_ensemble_training_data_non_finite")
    if not np.isfinite(prediction).all():
        raise ValueError("ai_ensemble_prediction_data_non_finite")
    return labels


def _validate_factory_set(
    factories: Mapping[str, RegressorFactory],
) -> None:
    if set(factories) != set(MODEL_NAMES):
        raise ValueError("ai_ensemble_factory_set_must_contain_all_three_models")


def _predict_one_model(
    name: str,
    factory: RegressorFactory,
    training_features: DataFrame,
    target: np.ndarray,
    prediction_features: DataFrame,
) -> tuple[str, tuple[float, ...]]:
    estimator = factory()
    estimator.fit(training_features, target)
    predictions = np.asarray(estimator.predict(prediction_features), dtype=float).reshape(-1)
    if len(predictions) != len(prediction_features):
        raise ValueError(f"ai_ensemble_prediction_length_mismatch:{name}")
    if not np.isfinite(predictions).all():
        raise ValueError(f"ai_ensemble_prediction_non_finite:{name}")
    return name, tuple(float(value) for value in predictions)


def fit_weighted_ensemble(
    training_features: DataFrame,
    training_target: Series | np.ndarray,
    prediction_features: DataFrame,
    *,
    config: EnsembleConfig | None = None,
    factories: Mapping[str, RegressorFactory] | None = None,
) -> EnsemblePrediction:
    """Fit all three models on the supplied training window and average predictions."""
    selected_config = config or EnsembleConfig()
    target = _validate_training_inputs(
        training_features, training_target, prediction_features
    )
    selected_factories = dict(factories or default_model_factories(selected_config))
    _validate_factory_set(selected_factories)
    components = tuple(
        _predict_one_model(
            name,
            selected_factories[name],
            training_features,
            target,
            prediction_features,
        )
        for name in MODEL_NAMES
    )
    prediction_map = dict(components)
    weighted = np.zeros(len(prediction_features), dtype=float)
    for name, weight in selected_config.weights:
        weighted += np.asarray(prediction_map[name], dtype=float) * weight
    return EnsemblePrediction(
        predictions=tuple(float(value) for value in weighted),
        per_model_predictions=components,
        weights=selected_config.weights,
        feature_names=tuple(str(name) for name in training_features.columns),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if (
        actual_values.size == 0
        or actual_values.shape != predicted_values.shape
        or not np.isfinite(actual_values).all()
        or not np.isfinite(predicted_values).all()
    ):
        raise ValueError("ai_ensemble_metric_inputs_invalid")
    error = predicted_values - actual_values
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "directional_accuracy": float(
            np.mean(np.sign(predicted_values) == np.sign(actual_values))
        ),
    }


__all__ = [
    "EnsembleConfig",
    "EnsemblePrediction",
    "RegressorFactory",
    "default_model_factories",
    "fit_weighted_ensemble",
    "regression_metrics",
]
