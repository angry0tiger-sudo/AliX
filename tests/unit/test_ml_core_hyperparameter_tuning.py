from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

from services.ml_core.hyperparameter_tuning import (
    HYPERPARAMETER_DISTRIBUTIONS,
    build_purged_time_series_splits,
    tune_lightgbm_classifier,
)


class DeterministicClassifier(BaseEstimator):
    def __init__(
        self,
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        class_weight: str | None = None,
        max_depth: int = 6,
    ) -> None:
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.max_depth = max_depth

    def fit(self, features: DataFrame, labels: Series) -> DeterministicClassifier:
        assert len(features) == len(labels)
        self.is_fitted_ = True
        return self

    def predict(self, features: DataFrame) -> np.ndarray:
        codes = features["signal"].to_numpy(dtype=int) % 3
        return np.asarray(["BUY", "SELL", "HOLD"], dtype=object)[codes]


def _training_fixture(
    rows: int = 700,
) -> tuple[DataFrame, Series, list[int], list[int]]:
    indices = list(range(10, rows - 90))
    labels = Series(
        [["BUY", "SELL", "HOLD"][index % 3] for index in indices],
        index=indices,
        dtype=object,
    )
    features = DataFrame(
        {"signal": [index % 3 for index in indices]},
        index=indices,
    )
    label_end_indices = [
        min(index + 12, rows - 1) for index in range(rows)
    ]
    return features, labels, indices, label_end_indices


def test_hyperparameter_search_space_contains_exact_nine_grid_values() -> None:
    assert HYPERPARAMETER_DISTRIBUTIONS == {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
    }
    assert (
        len(HYPERPARAMETER_DISTRIBUTIONS["num_leaves"])
        * len(HYPERPARAMETER_DISTRIBUTIONS["learning_rate"])
        * len(HYPERPARAMETER_DISTRIBUTIONS["min_child_samples"])
        == 27
    )


def test_inner_time_series_folds_are_chronological_and_purged() -> None:
    _, labels, training_indices, label_end_indices = _training_fixture()

    folds, diagnostics = build_purged_time_series_splits(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles=12,
        n_splits=3,
    )

    assert len(folds) == len(diagnostics) == 3
    for (fit_positions, validation_positions), detail in zip(
        folds, diagnostics, strict=True
    ):
        fit_rows = [training_indices[index] for index in fit_positions]
        validation_rows = [
            training_indices[index] for index in validation_positions
        ]
        assert fit_rows
        assert validation_rows
        assert max(fit_rows) < min(validation_rows)
        assert max(label_end_indices[index] for index in fit_rows) < min(
            validation_rows
        )
        assert max(fit_rows) < min(validation_rows) - 12
        assert detail["maximum_training_label_end_index"] < detail[
            "validation_start_index"
        ]
        assert detail["pre_validation_embargo_candles"] == 12


def test_inner_cv_uses_only_supplied_outer_train_rows_and_is_deterministic() -> None:
    features, labels, training_indices, label_end_indices = _training_fixture()
    estimator = DeterministicClassifier(class_weight="balanced")

    first_model, first = tune_lightgbm_classifier(
        estimator,
        features,
        labels,
        training_indices,
        label_end_indices,
        horizon_candles=12,
        class_weight_policy="weight_balanced",
        minimum_hold_recall=0.20,
        n_iter=9,
        n_splits=3,
        random_state=42,
    )
    second_model, second = tune_lightgbm_classifier(
        estimator,
        features,
        labels,
        training_indices,
        label_end_indices,
        horizon_candles=12,
        class_weight_policy="weight_balanced",
        minimum_hold_recall=0.20,
        n_iter=9,
        n_splits=3,
        random_state=42,
    )

    assert first_model.is_fitted_ is True
    assert second_model.is_fitted_ is True
    assert first["candidate_count"] == second["candidate_count"] == 9
    assert first["candidate_fit_count"] == 27
    assert first["inner_cv_fold_count"] == 3
    assert first["inner_cv_scope"] == "OUTER_TRAIN_ONLY"
    assert first["validation_accessed_during_search"] is False
    assert first["final_refit_scope"] == "FULL_OUTER_TRAIN_ONLY"
    assert first["candidate_results"] == second["candidate_results"]
    assert first["selected_parameters"] == second["selected_parameters"]
    assert len(
        {
            tuple(sorted(item["parameters"].items()))
            for item in first["candidate_results"]
        }
    ) == 9
    assert first["selected_parameters"] in [
        item["parameters"] for item in first["candidate_results"]
    ]
    assert all(
        item["maximum_training_label_end_index"]
        < item["validation_start_index"]
        for item in first["inner_cv_folds"]
    )


def test_inner_cv_rejects_duplicate_or_unordered_outer_train_indices() -> None:
    _, labels, training_indices, label_end_indices = _training_fixture()
    invalid_indices = training_indices[:100] + training_indices[99:200]

    with pytest.raises(
        ValueError, match="tuning_training_indices_must_be_unique_and_increasing"
    ):
        build_purged_time_series_splits(
            invalid_indices,
            labels.iloc[: len(invalid_indices)],
            label_end_indices,
            horizon_candles=12,
            n_splits=3,
        )


def test_inner_cv_rejects_label_end_outside_dataset() -> None:
    _, labels, training_indices, label_end_indices = _training_fixture()
    label_end_indices[training_indices[0]] = len(label_end_indices)

    with pytest.raises(ValueError, match="tuning_label_end_index_invalid"):
        build_purged_time_series_splits(
            training_indices,
            labels,
            label_end_indices,
            horizon_candles=12,
            n_splits=3,
        )


def test_random_search_rejects_candidate_count_outside_search_space() -> None:
    features, labels, training_indices, label_end_indices = _training_fixture()

    with pytest.raises(
        ValueError, match="tuning_random_search_iterations_out_of_policy"
    ):
        tune_lightgbm_classifier(
            DeterministicClassifier(),
            features,
            labels,
            training_indices,
            label_end_indices,
            horizon_candles=12,
            class_weight_policy="weight_none",
            minimum_hold_recall=0.20,
            n_iter=28,
        )
