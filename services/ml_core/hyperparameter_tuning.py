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
TUNING_OBJECTIVES = ("HOLD_PRECISION", "BUY_F1")


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


def _validate_inner_cv_inputs(
    training_indices: Sequence[int],
    labels: Series,
    label_end_indices: Sequence[object],
    horizon_candles: int,
    n_splits: int,
    minimum_class_samples: int,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
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
    return horizon, split_count, minimum, train_rows, train_ends


def _purged_fit_positions(
    candidate_positions: np.ndarray,
    train_rows: np.ndarray,
    train_ends: np.ndarray,
    validation_start: int,
    horizon: int,
) -> np.ndarray:
    return np.asarray(
        [
            int(position)
            for position in candidate_positions
            if int(train_rows[int(position)]) < validation_start - horizon
            and int(train_ends[int(position)]) < validation_start
        ],
        dtype=int,
    )


def _validate_inner_fit_support(
    fit_positions: np.ndarray,
    label_values: np.ndarray,
    minimum_class_samples: int,
) -> dict[str, int]:
    if len(fit_positions) == 0:
        raise ValueError("tuning_inner_cv_fit_empty_after_purge")
    counts = {
        label: int(np.count_nonzero(label_values[fit_positions] == label))
        for label in LABELS
    }
    if any(count < minimum_class_samples for count in counts.values()):
        raise ValueError("tuning_inner_cv_class_support_insufficient")
    return counts


def _build_inner_cv_fold(
    candidate_positions: np.ndarray,
    validation_positions: np.ndarray,
    train_rows: np.ndarray,
    train_ends: np.ndarray,
    label_values: np.ndarray,
    *,
    horizon: int,
    minimum_class_samples: int,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
    if len(validation_positions) == 0:
        raise ValueError("tuning_inner_cv_validation_empty")
    validation_start = int(train_rows[int(validation_positions[0])])
    fit_positions = _purged_fit_positions(
        candidate_positions, train_rows, train_ends, validation_start, horizon
    )
    _validate_inner_fit_support(fit_positions, label_values, minimum_class_samples)
    if int(train_rows[int(fit_positions[-1])]) >= validation_start:
        raise ValueError("tuning_inner_cv_chronology_invalid")
    maximum_label_end = int(np.max(train_ends[fit_positions]))
    if maximum_label_end >= validation_start:
        raise ValueError("tuning_inner_cv_label_overlap")
    validation = np.asarray(validation_positions, dtype=int)
    diagnostics = {
        "train_rows": int(len(fit_positions)),
        "validation_rows": int(len(validation)),
        "train_start_index": int(train_rows[int(fit_positions[0])]),
        "train_end_index": int(train_rows[int(fit_positions[-1])]),
        "validation_start_index": validation_start,
        "validation_end_index": int(train_rows[int(validation[-1])]),
        "maximum_training_label_end_index": maximum_label_end,
        "pre_validation_embargo_candles": horizon,
    }
    return (fit_positions, validation), diagnostics


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
    horizon, split_count, minimum, train_rows, train_ends = _validate_inner_cv_inputs(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles,
        n_splits,
        minimum_class_samples,
    )
    splitter = TimeSeriesSplit(n_splits=split_count)
    label_values = labels.to_numpy(dtype=object)
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    diagnostics: list[dict[str, int]] = []
    for candidate_positions, validation_positions in splitter.split(train_rows):
        fold, fold_diagnostics = _build_inner_cv_fold(
            candidate_positions,
            validation_positions,
            train_rows,
            train_ends,
            label_values,
            horizon=horizon,
            minimum_class_samples=minimum,
        )
        cv_splits.append(fold)
        diagnostics.append(fold_diagnostics)
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


def _buy_f1_score(estimator: Any, features: DataFrame, actual: Series) -> float:
    """Score the BUY class only."""
    predicted = estimator.predict(features)
    return float(
        f1_score(
            actual,
            predicted,
            labels=["BUY"],
            average="macro",
            zero_division=0,
        )
    )


def _validate_tuning_objective(objective: str) -> str:
    if objective not in TUNING_OBJECTIVES:
        raise ValueError("tuning_objective_unsupported")
    return objective


def _validated_cv_score_arrays(
    cv_results: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    precisions = np.asarray(cv_results["mean_test_hold_precision"], dtype=float)
    recalls = np.asarray(cv_results["mean_test_hold_recall"], dtype=float)
    macro_f1 = np.asarray(cv_results["mean_test_macro_f1"], dtype=float)
    buy_f1 = np.asarray(cv_results["mean_test_buy_f1"], dtype=float)
    parameters = cv_results["params"]
    lengths_match = (
        len(precisions)
        == len(recalls)
        == len(macro_f1)
        == len(buy_f1)
        == len(parameters)
    )
    values_are_finite = (
        np.isfinite(precisions).all()
        and np.isfinite(recalls).all()
        and np.isfinite(macro_f1).all()
        and np.isfinite(buy_f1).all()
    )
    if len(precisions) == 0 or not lengths_match or not values_are_finite:
        raise ValueError("tuning_cv_results_invalid")
    return precisions, recalls, macro_f1, buy_f1, parameters


def _candidate_complexity_key(parameters: dict[str, Any]) -> tuple[int, int, int]:
    return (
        -int(parameters["num_leaves"]),
        int(parameters["min_child_samples"]),
        -int(round(float(parameters["learning_rate"]) * 1000)),
    )


def _candidate_ranking_key(
    index: int,
    precisions: np.ndarray,
    recalls: np.ndarray,
    macro_f1: np.ndarray,
    buy_f1: np.ndarray,
    parameters: list[dict[str, Any]],
    objective: str,
) -> tuple[float, float, float, float, int, int, int]:
    complexity = _candidate_complexity_key(parameters[index])
    precision = float(precisions[index])
    recall = float(recalls[index])
    macro = float(macro_f1[index])
    selected_buy_f1 = float(buy_f1[index])
    if objective == "BUY_F1":
        return (selected_buy_f1, macro, precision, recall, *complexity)
    return (precision, macro, recall, selected_buy_f1, *complexity)


def _select_candidate(
    cv_results: dict[str, Any],
    minimum_hold_recall: float,
    objective: str = "HOLD_PRECISION",
) -> tuple[int, str, int]:
    objective = _validate_tuning_objective(objective)
    precisions, recalls, macro_f1, buy_f1, parameters = _validated_cv_score_arrays(
        cv_results
    )
    feasible = [
        index
        for index, recall in enumerate(recalls)
        if float(recall) >= minimum_hold_recall
    ]
    candidate_indices = (
        list(range(len(parameters)))
        if objective == "BUY_F1"
        else feasible or list(range(len(parameters)))
    )
    selected = max(
        candidate_indices,
        key=lambda index: _candidate_ranking_key(
            index, precisions, recalls, macro_f1, buy_f1, parameters, objective
        ),
    )
    if objective == "BUY_F1":
        status = "BUY_F1_MAXIMIZED"
    else:
        status = (
            "RECALL_CONSTRAINED_HOLD_PRECISION"
            if feasible
            else "MACRO_F1_FALLBACK_NO_RECALL_FEASIBLE"
        )
    return selected, status, len(feasible)


def _validate_tuning_configuration(
    minimum_hold_recall: float,
    n_iter: int,
    n_splits: int,
    random_state: int,
    class_weight_policy: str,
) -> tuple[int, int, int]:
    iterations = _positive_integer(n_iter, "tuning_random_search_iterations_invalid")
    split_count = _positive_integer(n_splits, "tuning_inner_cv_split_count_invalid")
    seed = _nonnegative_integer(random_state, "tuning_random_state_invalid")
    if iterations > 27:
        raise ValueError("tuning_random_search_iterations_out_of_policy")
    recall_is_valid = (
        isinstance(minimum_hold_recall, (int, float))
        and not isinstance(minimum_hold_recall, bool)
        and isfinite(float(minimum_hold_recall))
        and 0.0 <= float(minimum_hold_recall) <= 1.0
    )
    if not recall_is_valid:
        raise ValueError("tuning_minimum_hold_recall_invalid")
    if not class_weight_policy.strip():
        raise ValueError("tuning_class_weight_policy_missing")
    return iterations, split_count, seed


def _build_randomized_search(
    estimator: Any,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    seed: int,
) -> Any:
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=HYPERPARAMETER_DISTRIBUTIONS,
        n_iter=iterations,
        scoring={
            "hold_precision": _hold_precision_score,
            "hold_recall": _hold_recall_score,
            "macro_f1": _macro_f1_score,
            "buy_f1": _buy_f1_score,
        },
        cv=cv_splits,
        refit=False,
        random_state=seed,
        n_jobs=1,
        pre_dispatch=1,
        error_score="raise",
        return_train_score=False,
    )


def _fit_selected_estimator(
    estimator: Any,
    features: DataFrame,
    labels: Series,
    parameters: dict[str, Any],
) -> Any:
    fitted_estimator = clone(estimator)
    fitted_estimator.set_params(**parameters)
    fitted_estimator.fit(features, labels)
    return fitted_estimator


def _candidate_result_records(cv_results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": index,
            "parameters": dict(parameters),
            "mean_cv_hold_precision": round(
                float(cv_results["mean_test_hold_precision"][index]), 6
            ),
            "mean_cv_hold_recall": round(
                float(cv_results["mean_test_hold_recall"][index]), 6
            ),
            "mean_cv_macro_f1": round(
                float(cv_results["mean_test_macro_f1"][index]), 6
            ),
            "mean_cv_buy_f1": round(float(cv_results["mean_test_buy_f1"][index]), 6),
        }
        for index, parameters in enumerate(cv_results["params"])
    ]


def _build_tuning_diagnostics(
    cv_results: dict[str, Any],
    fold_diagnostics: list[dict[str, int]],
    parameters: dict[str, Any],
    *,
    class_weight_policy: str,
    minimum_hold_recall: float,
    selection_status: str,
    recall_feasible_count: int,
    seed: int,
    tuning_objective: str,
) -> dict[str, Any]:
    best_index = next(
        index
        for index, candidate in enumerate(cv_results["params"])
        if candidate == parameters
    )
    candidate_count = len(cv_results["params"])
    return {
        "search_method": "RANDOM_SEARCH",
        "search_space": {
            name: list(values) for name, values in HYPERPARAMETER_DISTRIBUTIONS.items()
        },
        "search_space_size": 27,
        "candidate_count": candidate_count,
        "inner_cv_method": "PURGED_EXPANDING_TIME_SERIES",
        "inner_cv_fold_count": len(fold_diagnostics),
        "inner_cv_folds": fold_diagnostics,
        "inner_cv_scope": "OUTER_TRAIN_ONLY",
        "validation_accessed_during_search": False,
        "final_refit_scope": "FULL_OUTER_TRAIN_ONLY",
        "class_weight_policy": class_weight_policy,
        "selection_objective": tuning_objective,
        "minimum_hold_recall_for_candidate_selection": float(minimum_hold_recall),
        "minimum_hold_recall_applied_to_selection": tuning_objective
        == "HOLD_PRECISION",
        "recall_feasible_candidate_count": recall_feasible_count,
        "selection_status": selection_status,
        "selected_parameters": parameters,
        "selected_mean_cv_hold_precision": round(
            float(cv_results["mean_test_hold_precision"][best_index]), 6
        ),
        "selected_mean_cv_hold_recall": round(
            float(cv_results["mean_test_hold_recall"][best_index]), 6
        ),
        "selected_mean_cv_macro_f1": round(
            float(cv_results["mean_test_macro_f1"][best_index]), 6
        ),
        "selected_mean_cv_buy_f1": round(
            float(cv_results["mean_test_buy_f1"][best_index]), 6
        ),
        "candidate_results": _candidate_result_records(cv_results),
        "candidate_fit_count": candidate_count * len(fold_diagnostics),
        "random_state": seed,
    }


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
    tuning_objective: str = "HOLD_PRECISION",
) -> tuple[Any, dict[str, Any]]:
    """Tune on outer Train only, then refit the chosen estimator on all outer Train."""
    tuning_objective = _validate_tuning_objective(tuning_objective)
    iterations, split_count, seed = _validate_tuning_configuration(
        minimum_hold_recall,
        n_iter,
        n_splits,
        random_state,
        class_weight_policy,
    )
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
    search = _build_randomized_search(estimator, cv_splits, iterations, seed)
    search.fit(features, labels)
    cv_results: dict[str, Any] = search.cv_results_
    best_index, selection_status, recall_feasible_count = _select_candidate(
        cv_results, float(minimum_hold_recall), tuning_objective
    )
    best_parameters = dict(cv_results["params"][best_index])
    fitted_estimator = _fit_selected_estimator(
        estimator, features, labels, best_parameters
    )
    diagnostics = _build_tuning_diagnostics(
        cv_results,
        fold_diagnostics,
        best_parameters,
        class_weight_policy=class_weight_policy,
        minimum_hold_recall=minimum_hold_recall,
        selection_status=selection_status,
        recall_feasible_count=recall_feasible_count,
        seed=seed,
        tuning_objective=tuning_objective,
    )
    return fitted_estimator, diagnostics
