"""Unit coverage for V4-AH research-only primitives."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.meta_labeling import v4ah


def _event_frame(row_count: int = 250) -> DataFrame:
    indices = np.arange(row_count, dtype=int) * 3 + 10
    return DataFrame(
        {
            "event_index": indices,
            "label_end_idx": indices + 1,
            "meta_label": np.arange(row_count) % 2,
            "net_return": np.where(np.arange(row_count) % 2, 0.02, -0.01),
        }
    )


def _regime_frame(row_count: int = 260) -> DataFrame:
    rng = np.random.default_rng(42)
    return DataFrame(
        rng.normal(size=(row_count, len(v4ah.REGIME_INPUT_COLUMNS))),
        columns=list(v4ah.REGIME_INPUT_COLUMNS),
    )


def test_numeric_matrix_and_binary_label_guards() -> None:
    matrix = v4ah._numeric_matrix(DataFrame({"a": [1.0, 2.0]}), "bad")
    assert matrix.shape == (2, 1)
    labels = v4ah._validate_binary_labels(np.array([0, 1] * 8), 16)
    assert labels.sum() == 8
    with pytest.raises(ValueError, match="bad"):
        v4ah._numeric_matrix(DataFrame(), "bad")
    with pytest.raises(ValueError, match="bad"):
        v4ah._numeric_matrix(DataFrame({"a": [1.0, np.nan]}), "bad")
    duplicate = DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(ValueError, match="bad"):
        v4ah._numeric_matrix(duplicate, "bad")
    with pytest.raises(ValueError, match="v4ah_binary_labels_invalid"):
        v4ah._validate_binary_labels(np.ones(16), 16)
    with pytest.raises(ValueError, match="v4ah_binary_class_support_insufficient"):
        v4ah._validate_binary_labels(np.array([0] + [1] * 15), 16)


def test_regime_feature_pipeline_and_fail_closed_guards() -> None:
    frame = _regime_frame()
    result, diagnostics, scaler, model = v4ah.build_regime_features(
        frame,
        fit_end=200,
    )
    assert len(result) == len(frame)
    assert diagnostics["algorithm"] == "KMeans"
    assert diagnostics["fit_rows"] == 200
    assert diagnostics["hard_regime_filter_enabled"] is False
    assert scaler is not None and model is not None
    assert result["%-regime_state_valid"].eq(1.0).all()
    with pytest.raises(ValueError, match="v4ah_regime_training_window_invalid"):
        v4ah.build_regime_features(frame, fit_end=99)
    with pytest.raises(ValueError, match="v4ah_regime_cluster_count_invalid"):
        v4ah.build_regime_features(frame, fit_end=200, cluster_count=1)
    with pytest.raises(ValueError, match="v4ah_regime_input_schema_invalid"):
        v4ah.build_regime_features(
            frame.drop(columns=[v4ah.REGIME_INPUT_COLUMNS[0]]), fit_end=200
        )
    insufficient = frame.copy()
    insufficient.loc[:119, list(v4ah.REGIME_INPUT_COLUMNS)] = np.nan
    with pytest.raises(ValueError, match="v4ah_regime_training_support_insufficient"):
        v4ah.build_regime_features(insufficient, fit_end=200)
    collision = frame.copy()
    collision["%-regime_cluster_0"] = 0.0
    with pytest.raises(ValueError, match="v4ah_regime_feature_collision"):
        v4ah.build_regime_features(collision, fit_end=200)


def test_three_way_chronological_purged_split_and_guards() -> None:
    events = _event_frame()
    fit, temperature, threshold, diagnostics = v4ah.split_fit_temperature_threshold(
        events, outer_train_end=900
    )
    assert len(fit) == 175
    assert len(temperature) == 37
    assert len(threshold) == 38
    assert diagnostics["fit_labels_end_before_temperature"] is True
    assert diagnostics["temperature_labels_end_before_threshold"] is True
    assert diagnostics["all_slices_inside_outer_train"] is True
    with pytest.raises(ValueError, match="v4ah_fit_fraction_invalid"):
        v4ah.split_fit_temperature_threshold(
            events, outer_train_end=900, fit_fraction=0.1
        )
    duplicate = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="v4ah_training_events_duplicate"):
        v4ah.split_fit_temperature_threshold(duplicate, outer_train_end=900)
    outside = events.copy()
    outside.loc[outside.index[-1], "label_end_idx"] = 900
    with pytest.raises(ValueError, match="v4ah_training_label_crosses_outer_train"):
        v4ah.split_fit_temperature_threshold(outside, outer_train_end=900)
    purged = events.copy()
    purged.loc[:174, "label_end_idx"] = int(events["event_index"].iloc[175])
    with pytest.raises(ValueError, match="v4ah_fit_rows_insufficient_after_purge"):
        v4ah.split_fit_temperature_threshold(purged, outer_train_end=900)
    imbalanced = events.copy()
    imbalanced.loc[:174, "meta_label"] = 1
    with pytest.raises(ValueError, match="v4ah_binary_class_support_insufficient"):
        v4ah.split_fit_temperature_threshold(imbalanced, outer_train_end=900)


class _FakeLightGBMModel:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fit_rows = 0

    def fit(self, features: DataFrame, labels: np.ndarray) -> _FakeLightGBMModel:
        self.fit_rows = len(features)
        self.classes_ = np.unique(labels)
        return self


class _FakeShapExplainer:
    def __init__(self, model: Any) -> None:
        self.model = model

    def shap_values(self, features: DataFrame) -> list[np.ndarray]:
        importance = np.tile(
            np.arange(1, len(features.columns) + 1), (len(features), 1)
        )
        return [-importance, importance]


def test_shap_lightgbm_pipeline_with_stubbed_optional_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[_FakeLightGBMModel] = []

    def classifier(**kwargs: Any) -> _FakeLightGBMModel:
        model = _FakeLightGBMModel()
        models.append(model)
        return model

    lightgbm = ModuleType("lightgbm")
    setattr(lightgbm, "LGBMClassifier", classifier)
    shap = ModuleType("shap")
    setattr(shap, "TreeExplainer", _FakeShapExplainer)
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "shap", shap)
    features = DataFrame(
        np.arange(300, dtype=float).reshape(60, 5), columns=list("abcde")
    )
    labels = pd.Series(np.array([0, 1] * 30))
    model, selected, rankings, diagnostics = v4ah.fit_shap_selected_lightgbm(
        features,
        labels,
        top_k=3,
    )
    assert model is models[-1]
    assert len(models) == 2
    assert selected == ["e", "d", "c"]
    assert len(rankings) == 5
    assert diagnostics["importance_fit_scope"] == "PURGED_OOF_FIT_SLICE_ONLY"
    with pytest.raises(ValueError, match="v4ah_shap_top_k_invalid"):
        v4ah.fit_shap_selected_lightgbm(features, labels, top_k=0)


def test_probability_temperature_and_calibration_diagnostics() -> None:
    class ProbabilityModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [np.full(len(features), 0.4), np.full(len(features), 0.6)]
            )

    probabilities = v4ah._positive_probability(
        ProbabilityModel(), DataFrame({"x": [1.0, 2.0]})
    )
    assert np.allclose(probabilities, [0.6, 0.6])
    logits = v4ah._logit(np.array([0.25, 0.75]))
    assert logits[0] < 0.0 < logits[1]
    calibrated = v4ah.apply_temperature_scaling([0.2, 0.8], 1.5)
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    raw = np.linspace(0.1, 0.9, 40)
    labels = np.array([0, 1] * 20)
    temperature, diagnostics = v4ah.fit_temperature_scaler(raw, labels)
    assert 0.05 <= temperature <= 20.0
    assert diagnostics["optimizer_success"] is True
    report = v4ah.calibration_diagnostics(raw, labels, bin_count=8)
    assert report["bin_count"] == 8
    assert report["sample_count"] == 40
    assert report["brier_score"] >= 0.0
    with pytest.raises(ValueError, match="v4ah_model_class_order_invalid"):
        v4ah._positive_probability(
            type("BadModel", (), {"classes_": [1, 0]})(), DataFrame({"x": [1.0]})
        )
    with pytest.raises(ValueError, match="v4ah_temperature_inputs_invalid"):
        v4ah.apply_temperature_scaling([0.2], 0.0)
    with pytest.raises(ValueError, match="v4ah_temperature_calibration_inputs_invalid"):
        v4ah.fit_temperature_scaler(raw[:10], labels[:10])
    with pytest.raises(ValueError, match="v4ah_calibration_diagnostics_inputs_invalid"):
        v4ah.calibration_diagnostics([0.2], [2])


def test_net_expectancy_threshold_selection_and_tie_breaking() -> None:
    probabilities = np.linspace(0.01, 0.99, 40)
    returns = np.where(probabilities >= 0.5, 0.02, -0.01)
    threshold, diagnostics = v4ah.select_net_expectancy_threshold(
        probabilities, returns
    )
    assert threshold is not None and threshold >= 0.5
    assert diagnostics["status"] == "SELECTED_POSITIVE_EXPECTANCY_THRESHOLD"
    none, no_positive = v4ah.select_net_expectancy_threshold(
        probabilities, np.full(40, -0.01)
    )
    assert none is None
    assert no_positive["status"] == "NO_POSITIVE_EXPECTANCY_THRESHOLD"
    assert v4ah._threshold_candidate_is_better((0.1, 15, 0.5), (0.1, 12, 0.6))
    with pytest.raises(ValueError, match="v4ah_threshold_inputs_invalid"):
        v4ah.select_net_expectancy_threshold([0.2], [-1.0])


def test_adaptive_trailing_policy_training_and_prediction() -> None:
    rng = np.random.default_rng(7)
    features = DataFrame(rng.normal(size=(45, 4)), columns=list("abcd"))
    returns = np.tile(np.array([0.01, 0.02, 0.005]), (45, 1))
    ends = np.tile(np.array([50.0, 60.0, 70.0]), (45, 1))
    model, diagnostics = v4ah.fit_adaptive_trailing_policy(
        features,
        returns,
        ends,
        calibration_start_index=100,
    )
    predicted = v4ah.predict_trailing_steps(model, features)
    assert len(predicted) == 45
    assert set(predicted.tolist()).issubset(set(v4ah.V4AH_TRAILING_STEP_OPTIONS))
    assert diagnostics["all_counterfactual_exits_before_calibration"] is True
    with pytest.raises(ValueError, match="v4ah_trailing_counterfactual_schema_invalid"):
        v4ah.fit_adaptive_trailing_policy(
            features, np.ones((45, 2)), np.ones((45, 2)), calibration_start_index=100
        )
    with pytest.raises(
        ValueError, match="v4ah_trailing_policy_training_support_insufficient"
    ):
        v4ah.fit_adaptive_trailing_policy(
            features.iloc[:20],
            returns[:20],
            ends[:20],
            calibration_start_index=100,
        )


def test_fractional_kelly_and_sized_equity_curve() -> None:
    returns = np.array([0.02] * 16 + [-0.01] * 16)
    ratio, diagnostics = v4ah.estimate_training_payoff_ratio(returns)
    assert ratio == pytest.approx(2.0)
    assert diagnostics["training_rows"] == 32
    assert v4ah.fractional_kelly_fraction(0.6, ratio) == pytest.approx(0.1)
    assert v4ah.fractional_kelly_fraction(0.2, ratio) == 0.0
    trades = [
        {"net_return_fraction": 0.02},
        {"net_return_fraction": -0.01},
        {"net_return_fraction": 0.03},
    ]
    metrics, enriched = v4ah.summarize_sized_trades(trades, [0.1, 0.1, 0.1])
    assert metrics["trade_count"] == 3
    assert len(enriched) == 3
    assert metrics["ending_equity_usd"] > 10_000.0
    assert metrics["mean_position_fraction"] == pytest.approx(0.1)
    empty, _ = v4ah.summarize_sized_trades([], [])
    assert empty["trade_count"] == 0
    with pytest.raises(ValueError, match="v4ah_kelly_payoff_support_insufficient"):
        v4ah.estimate_training_payoff_ratio(np.array([0.01, -0.01] * 3))
    with pytest.raises(ValueError, match="v4ah_kelly_inputs_out_of_range"):
        v4ah.fractional_kelly_fraction(0.5, 0.0)
    with pytest.raises(ValueError, match="v4ah_sized_trade_alignment_invalid"):
        v4ah.summarize_sized_trades(trades, [0.1])
    with pytest.raises(ValueError, match="v4ah_initial_equity_invalid"):
        v4ah.summarize_sized_trades(trades, [0.1] * 3, initial_equity=0.0)
    with pytest.raises(ValueError, match="v4ah_sized_trade_values_invalid"):
        v4ah.summarize_sized_trades(trades, [1.1, 0.1, 0.1])
