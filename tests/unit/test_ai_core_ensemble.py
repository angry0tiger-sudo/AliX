from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ai_core.ensemble import (
    MODEL_NAMES,
    EnsembleConfig,
    EnsemblePrediction,
    RegressorFactory,
    default_model_factories,
    fit_weighted_ensemble,
    regression_metrics,
)


class ConstantRegressor:
    def __init__(self, prediction: float, *, length_delta: int = 0) -> None:
        self.prediction = prediction
        self.length_delta = length_delta
        self.fit_count = 0

    def fit(self, features: DataFrame, target: np.ndarray) -> object:
        self.fit_count += len(features)
        return self

    def predict(self, features: DataFrame) -> np.ndarray:
        count = len(features) + self.length_delta
        return np.full(count, self.prediction, dtype=float)


def dataset(rows: int = 50) -> tuple[DataFrame, np.ndarray, DataFrame]:
    train = pd.DataFrame(
        {
            "first": np.linspace(-1.0, 1.0, rows),
            "second": np.sin(np.arange(rows, dtype=float)),
        }
    )
    target = np.linspace(-0.02, 0.03, rows)
    prediction = pd.DataFrame(
        {"first": [0.1, 0.2], "second": [0.3, 0.4]}
    )
    return train, target, prediction


def factories(
    values: tuple[float, float, float], *, length_delta: int = 0
) -> Mapping[str, RegressorFactory]:
    return {
        name: (lambda value=value: ConstantRegressor(value, length_delta=length_delta))
        for name, value in zip(MODEL_NAMES, values, strict=True)
    }


def test_equal_weight_ensemble_fits_all_three_and_averages_outputs() -> None:
    train, target, prediction = dataset()
    result = fit_weighted_ensemble(
        train,
        target,
        prediction,
        factories=factories((0.1, 0.2, 0.4)),
    )
    assert len(result.predictions) == 2
    assert result.predictions == pytest.approx((0.7 / 3.0, 0.7 / 3.0))
    assert tuple(name for name, _ in result.per_model_predictions) == MODEL_NAMES
    assert result.weights == EnsembleConfig().weights
    assert result.feature_names == ("first", "second")
    assert all(len(values) == 2 for _, values in result.per_model_predictions)


def test_weighted_ensemble_uses_only_predeclared_normalized_weights() -> None:
    train, target, prediction = dataset()
    config = EnsembleConfig(
        weights=(
            ("xgboost", 0.5),
            ("random_forest", 0.3),
            ("lightgbm", 0.2),
        )
    )
    result = fit_weighted_ensemble(
        train,
        target,
        prediction,
        config=config,
        factories=factories((0.1, 0.2, 0.4)),
    )
    assert result.predictions == pytest.approx((0.19, 0.19))


@pytest.mark.parametrize(
    "weights",
    (
        (("xgboost", 0.5), ("random_forest", 0.5)),
        (("xgboost", 0.5), ("random_forest", 0.3), ("lightgbm", 0.3)),
        (("xgboost", -0.1), ("random_forest", 0.6), ("lightgbm", 0.5)),
        (("xgboost", float("nan")), ("random_forest", 0.5), ("lightgbm", 0.5)),
    ),
)
def test_ensemble_config_rejects_invalid_weight_contracts(
    weights: tuple[tuple[str, float], ...],
) -> None:
    with pytest.raises(ValueError):
        EnsembleConfig(weights=weights)


def test_ensemble_config_rejects_unbounded_model_configuration() -> None:
    with pytest.raises(ValueError, match="n_estimators_invalid"):
        EnsembleConfig(n_estimators=9)
    with pytest.raises(ValueError, match="max_depth_invalid"):
        EnsembleConfig(max_depth=0)
    with pytest.raises(ValueError, match="random_seed_invalid"):
        EnsembleConfig(random_seed=True)


def test_default_factories_are_cpu_pinned_and_cover_the_required_models() -> None:
    built = default_model_factories(EnsembleConfig(n_estimators=10))
    assert set(built) == set(MODEL_NAMES)
    assert all(callable(factory) for factory in built.values())


@pytest.mark.parametrize(
    ("train_change", "target_change", "prediction_change", "error"),
    (
        ("empty", "valid", "valid", "feature_data_empty"),
        ("wrong_columns", "valid", "valid", "feature_schema_mismatch"),
        ("duplicate_columns", "valid", "valid", "duplicate_feature_names"),
        ("bad_rows", "valid", "valid", "sample_count_invalid"),
        ("non_finite", "valid", "valid", "training_data_non_finite"),
        ("valid", "bad_target", "valid", "training_data_non_finite"),
        ("valid", "valid", "non_finite", "prediction_data_non_finite"),
    ),
)
def test_ensemble_rejects_invalid_training_or_prediction_inputs(
    train_change: str,
    target_change: str,
    prediction_change: str,
    error: str,
) -> None:
    train, target, prediction = dataset()
    if train_change == "empty":
        train = train.iloc[:0]
    elif train_change == "wrong_columns":
        train = train.rename(columns={"first": "other"})
    elif train_change == "duplicate_columns":
        train.columns = ["first", "first"]
        prediction.columns = ["first", "first"]
    elif train_change == "bad_rows":
        train = train.iloc[:20]
        target = target[:20]
    elif train_change == "non_finite":
        train.loc[0, "first"] = np.nan
    if target_change == "bad_target":
        target = target.copy()
        target[0] = np.inf
    if prediction_change == "non_finite":
        prediction.loc[0, "first"] = np.nan
    with pytest.raises(ValueError, match=error):
        fit_weighted_ensemble(
            train,
            target,
            prediction,
            factories=factories((0.1, 0.2, 0.3)),
        )


def test_ensemble_rejects_missing_factory_and_inconsistent_predictions() -> None:
    train, target, prediction = dataset()
    with pytest.raises(ValueError, match="factory_set"):
        fit_weighted_ensemble(
            train,
            target,
            prediction,
            factories={"xgboost": lambda: ConstantRegressor(0.1)},
        )
    with pytest.raises(ValueError, match="prediction_length_mismatch"):
        fit_weighted_ensemble(
            train,
            target,
            prediction,
            factories=factories((0.1, 0.2, 0.3), length_delta=1),
        )
    with pytest.raises(ValueError, match="prediction_non_finite"):
        fit_weighted_ensemble(
            train,
            target,
            prediction,
            factories=factories((0.1, float("inf"), 0.3)),
        )


def test_prediction_contract_and_regression_metrics_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="feature_names_missing"):
        EnsemblePrediction((), (), (), ())
    with pytest.raises(ValueError, match="prediction_non_finite"):
        EnsemblePrediction((float("nan"),), (), (), ("first",))
    metrics = regression_metrics(
        np.asarray([0.1, -0.1]), np.asarray([0.2, -0.2])
    )
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(0.1)
    assert metrics["directional_accuracy"] == 1.0
    with pytest.raises(ValueError, match="metric_inputs_invalid"):
        regression_metrics(np.asarray([1.0]), np.asarray([1.0, 2.0]))
