"""Leakage-safe, purged time-series hyperparameter search for multiclass LightGBM."""

from __future__ import annotations

from math import isfinite
from numbers import Integral
from typing import Any, Sequence, cast

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # type: ignore[import-untyped]
    RandomizedSearchCV,
    TimeSeriesSplit,
)

LABELS = ("BUY", "SELL", "HOLD")
HYPERPARAMETER_DISTRIBUTIONS: dict[str, list[int | float]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [20, 50, 100],
}
DEFAULT_RANDOM_SEARCH_ITERATIONS = 9
DEFAULT_INNER_CV_SPLITS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_MINIMUM_CLASS_SAMPLES = 20


def _positive_integer(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(reason)
    return int(value)


def _nonnegative_integer(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(reason)
    return int(value)


def _label_end_index(value: object, row_index: int, row_count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("tuning_label_end_index_invalid")
    try:
        numeric = float(cast(float | int | str, value))
    except (TypeError, ValueError):
        raise ValueError("tuning_label_end_index_invalid") from None
    if not isfinite(numeric) or not numeric.is_integer():
        raise ValueError("tuning_label_end_index_invalid")
    end_index = int(numeric)
    if end_index < row_index or end_index >= row_count:
        raise ValueError("tuning_label_end_index_invalid")
    return end_index


def _training_rows(
    training_indices: Sequence[object],
    label_end_indices: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    if len(training_indices) == 0:
        raise ValueError("tuning_training_indices_empty")
    rows: list[int] = []
    ends: list[int] = []
    for raw_index in training_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise ValueError("tuning_training_index_invalid")
        row_index = int(raw_index)
        if row_index < 0 or row_index >= len(label_end_indices):
            raise ValueError("tuning_training_index_invalid")
        rows.append(row_index)
        ends.append(
            _label_end_index(
                label_end_indices[row_index], row_index, len(label_end_indices)
            )
        )
    if rows != sorted(set(rows)):
        raise ValueError("tuning_training_indices_must_be_unique_and_increasing")
    return np.asarray(rows, dtype=int), np.asarray(ends, dtype=int)


def build_purged_time_series_splits(
    training_indices: Sequence[int],
    labels: Series,
    label_end_indices: Sequence[object],
    *,
    horizon_candles: int,
    n_splits: int = DEFAULT_INNER_CV_SPLITS,
    minimum_class_samples: int = DEFAULT_MINIMUM_CLASS_SAMPLES,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, int]]]:
    """Create expanding CV folds using only outer-Train rows and purged label windows.

    Returned indices are positional offsets into the supplied outer-Train matrix.
    Every inner validation block is strictly later than its fit block. Fit rows
    whose label horizon reaches the inner validation window, or falls inside the
    fixed pre-validation embargo, are excluded.
    """
    horizon = _positive_integer(horizon_candles, "tuning_horizon_invalid")
    split_count = _positive_integer(n_splits, "tuning_inner_cv_split_count_invalid")
    minimum = _positive_integer(
        minimum_class_samples, "tuning_minimum_class_samples_invalid"
    )
    train_rows, train_ends = _training_rows(training_indices, label_end_indices)
    if len(labels) != len(train_rows):
        raise ValueError("tuning_train_labels_length_mismatch")
    if len(train_rows) <= split_count:
        raise ValueError("tuning_inner_cv_rows_insufficient")

    splitter = TimeSeriesSplit(n_splits=split_count)
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    diagnostics: list[dict[str, int]] = []
    label_values = labels.to_numpy(dtype=object)

    for candidate_fit_positions, validation_positions in splitter.split(train_rows):
        if len(validation_positions) == 0:
            raise ValueError("tuning_inner_cv_validation_empty")
        validation_start = int(train_rows[int(validation_positions[0])])
        purged_fit_positions = np.asarray(
            [
                int(position)
                for position in candidate_fit_positions
                if int(train_rows[int(position)]) < validation_start - horizon
                and int(train_ends[int(position)]) < validation_start
            ],
            dtype=int,
        )
        if len(purged_fit_positions) == 0:
            raise ValueError("tuning_inner_cv_fit_empty_after_purge")
        counts = {
            label: int(np.count_nonzero(label_values[purged_fit_positions] == label))
            for label in LABELS
        }
        if any(counts[label] < minimum for label in LABELS):
            raise ValueError("tuning_inner_cv_class_support_insufficient")
        validation_positions = np.asarray(validation_positions, dtype=int)
        if int(train_rows[int(purged_fit_positions[-1])]) >= validation_start:
            raise ValueError("tuning_inner_cv_chronology_invalid")
        if int(np.max(train_ends[purged_fit_positions])) >= validation_start:
            raise ValueError("tuning_inner_cv_label_overlap")
        cv_splits.append((purged_fit_positions, validation_positions))
        diagnostics.append(
            {
                "train_rows": int(len(purged_fit_positions)),
                "validation_rows": int(len(validation_positions)),
                "train_start_index": int(train_rows[int(purged_fit_positions[0])]),
                "train_end_index": int(train_rows[int(purged_fit_positions[-1])]),
                "validation_start_index": validation_start,
                "validation_end_index": int(train_rows[int(validation_positions[-1])]),
                "maximum_training_label_end_index": int(
                    np.max(train_ends[purged_fit_positions])
                ),
                "pre_validation_embargo_candles": horizon,
            }
        )

    if len(cv_splits) != split_count:
        raise ValueError("tuning_inner_cv_fold_count_mismatch")
    return cv_splits, diagnostics


def _hold_precision_score(estimator: Any, features: DataFrame, actual: Series) -> float:
    predicted = estimator.predict(features)
    return float(
        precision_score(
            actual,
            predicted,
            labels=["HOLD"],
            average="macro",
            zero_division=0,
        )
    )


def _hold_recall_score(estimator: Any, features: DataFrame, actual: Series) -> float:
    predicted = estimator.predict(features)
    return float(
        recall_score(
            actual,
            predicted,
            labels=["HOLD"],
            average="macro",
            zero_division=0,
        )
    )


def _macro_f1_score(estimator: Any, features: DataFrame, actual: Series) -> float:
    predicted = estimator.predict(features)
    return float(
        f1_score(
            actual,
            predicted,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    )


def _select_candidate(
    cv_results: dict[str, Any], minimum_hold_recall: float
) -> tuple[int, str, int]:
    precisions = np.asarray(cv_results["mean_test_hold_precision"], dtype=float)
    recalls = np.asarray(cv_results["mean_test_hold_recall"], dtype=float)
    macro_f1 = np.asarray(cv_results["mean_test_macro_f1"], dtype=float)
    parameters = cv_results["params"]
    if (
        len(precisions) == 0
        or len(precisions) != len(recalls)
        or len(precisions) != len(macro_f1)
        or len(precisions) != len(parameters)
        or not np.isfinite(precisions).all()
        or not np.isfinite(recalls).all()
        or not np.isfinite(macro_f1).all()
    ):
        raise ValueError("tuning_cv_results_invalid")

    feasible = [
        index
        for index, recall in enumerate(recalls)
        if float(recall) >= minimum_hold_recall
    ]
    candidate_indices = feasible or list(range(len(parameters)))

    def ranking_key(index: int) -> tuple[float, float, float, int, int, int]:
        params = parameters[index]
        complexity_key = (
            -int(params["num_leaves"]),
            int(params["min_child_samples"]),
            -int(round(float(params["learning_rate"]) * 1000)),
        )
        if feasible:
            return (
                float(precisions[index]),
                float(macro_f1[index]),
                float(recalls[index]),
                *complexity_key,
            )
        return (
            float(macro_f1[index]),
            float(precisions[index]),
            float(recalls[index]),
            *complexity_key,
        )

    selected = max(candidate_indices, key=ranking_key)
    status = (
        "RECALL_CONSTRAINED_HOLD_PRECISION"
        if feasible
        else "MACRO_F1_FALLBACK_NO_RECALL_FEASIBLE"
    )
    return selected, status, len(feasible)


def tune_lightgbm_classifier(
    estimator: Any,
    features: DataFrame,
    labels: Series,
    training_indices: Sequence[int],
    label_end_indices: Sequence[object],
    *,
    horizon_candles: int,
    class_weight_policy: str,
    minimum_hold_recall: float,
    n_iter: int = DEFAULT_RANDOM_SEARCH_ITERATIONS,
    n_splits: int = DEFAULT_INNER_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, dict[str, Any]]:
    """Tune on outer Train only, then refit the chosen estimator on all outer Train."""
    iterations = _positive_integer(n_iter, "tuning_random_search_iterations_invalid")
    split_count = _positive_integer(n_splits, "tuning_inner_cv_split_count_invalid")
    seed = _nonnegative_integer(random_state, "tuning_random_state_invalid")
    if iterations > 27:
        raise ValueError("tuning_random_search_iterations_out_of_policy")
    if (
        not isinstance(minimum_hold_recall, (int, float))
        or isinstance(minimum_hold_recall, bool)
        or not isfinite(float(minimum_hold_recall))
        or not 0.0 <= float(minimum_hold_recall) <= 1.0
    ):
        raise ValueError("tuning_minimum_hold_recall_invalid")
    if not class_weight_policy.strip():
        raise ValueError("tuning_class_weight_policy_missing")
    train_rows, _ = _training_rows(training_indices, label_end_indices)
    if len(features) != len(labels) or len(labels) != len(train_rows):
        raise ValueError("tuning_train_matrix_alignment_invalid")

    cv_splits, fold_diagnostics = build_purged_time_series_splits(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles=horizon_candles,
        n_splits=split_count,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=HYPERPARAMETER_DISTRIBUTIONS,
        n_iter=iterations,
        scoring={
            "hold_precision": _hold_precision_score,
            "hold_recall": _hold_recall_score,
            "macro_f1": _macro_f1_score,
        },
        cv=cv_splits,
        refit=False,
        random_state=seed,
        n_jobs=1,
        pre_dispatch=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(features, labels)
    cv_results: dict[str, Any] = search.cv_results_
    best_index, selection_status, recall_feasible_count = _select_candidate(
        cv_results, float(minimum_hold_recall)
    )
    best_parameters = dict(cv_results["params"][best_index])
    fitted_estimator = clone(estimator)
    fitted_estimator.set_params(**best_parameters)
    fitted_estimator.fit(features, labels)

    mean_precision = float(cv_results["mean_test_hold_precision"][best_index])
    mean_recall = float(cv_results["mean_test_hold_recall"][best_index])
    mean_macro_f1 = float(cv_results["mean_test_macro_f1"][best_index])
    candidate_count = len(cv_results["params"])
    return fitted_estimator, {
        "search_method": "RANDOM_SEARCH",
        "search_space": {
            name: list(values) for name, values in HYPERPARAMETER_DISTRIBUTIONS.items()
        },
        "search_space_size": 27,
        "candidate_count": candidate_count,
        "inner_cv_method": "PURGED_EXPANDING_TIME_SERIES",
        "inner_cv_fold_count": len(cv_splits),
        "inner_cv_folds": fold_diagnostics,
        "inner_cv_scope": "OUTER_TRAIN_ONLY",
        "validation_accessed_during_search": False,
        "final_refit_scope": "FULL_OUTER_TRAIN_ONLY",
        "class_weight_policy": class_weight_policy,
        "minimum_hold_recall_for_candidate_selection": float(minimum_hold_recall),
        "recall_feasible_candidate_count": recall_feasible_count,
        "selection_status": selection_status,
        "selected_parameters": best_parameters,
        "selected_mean_cv_hold_precision": round(mean_precision, 6),
        "selected_mean_cv_hold_recall": round(mean_recall, 6),
        "selected_mean_cv_macro_f1": round(mean_macro_f1, 6),
        "candidate_results": [
            {
                "candidate_index": index,
                "parameters": dict(params),
                "mean_cv_hold_precision": round(
                    float(cv_results["mean_test_hold_precision"][index]), 6
                ),
                "mean_cv_hold_recall": round(
                    float(cv_results["mean_test_hold_recall"][index]), 6
                ),
                "mean_cv_macro_f1": round(
                    float(cv_results["mean_test_macro_f1"][index]), 6
                ),
            }
            for index, params in enumerate(cv_results["params"])
        ],
        "candidate_fit_count": candidate_count * len(cv_splits),
        "random_state": seed,
    }
