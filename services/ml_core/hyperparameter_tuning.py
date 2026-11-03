"""Leakage-safe, purged time-series hyperparameter search for multiclass LightGBM."""

from __future__ import annotations

from functools import partial
from math import isfinite
from numbers import Integral
from typing import Any, Sequence, cast

import numpy as np

from services.ml_core.binary_calibration import (
    DEFAULT_TARGET_BUY_FRACTION,
    select_dynamic_buy_threshold,
    validate_target_buy_fraction,
)
from pandas import DataFrame, Series
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
TUNING_OBJECTIVES = (
    "HOLD_PRECISION",
    "BUY_F1",
    "NET_EXPECTANCY_PROXY",
    "CUSTOM_ECONOMIC_SCORE",
)
DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS = 90.0
CUSTOM_SCORE_MINIMUM_TRADES = 30
CUSTOM_SCORE_TRADE_COUNT_PENALTY_BPS = 50.0
CUSTOM_SCORE_DRAWDOWN_PENALTY_WEIGHT = 0.5


def _validate_custom_score_economic_alignment(
    features: DataFrame,
    actual: Series,
    gross_returns_by_index: Series,
    estimated_cost_bps: float,
) -> float:
    if (
        isinstance(estimated_cost_bps, bool)
        or not isfinite(float(estimated_cost_bps))
        or estimated_cost_bps < 0
    ):
        raise ValueError("tuning_estimated_cost_bps_invalid")
    if (
        not isinstance(gross_returns_by_index, Series)
        or gross_returns_by_index.index.has_duplicates
    ):
        raise ValueError("tuning_economic_outcomes_invalid")
    if len(features) != len(actual):
        raise ValueError("tuning_economic_score_alignment_invalid")
    return float(estimated_cost_bps)


def _validate_custom_score_penalties(
    minimum_trade_count: int,
    trade_count_penalty_bps: float,
    drawdown_penalty_weight: float,
) -> tuple[int, float, float]:
    minimum = _positive_integer(
        minimum_trade_count, "tuning_custom_score_minimum_trade_count_invalid"
    )
    penalty_valid = (
        not isinstance(trade_count_penalty_bps, bool)
        and isfinite(float(trade_count_penalty_bps))
        and trade_count_penalty_bps >= 0
    )
    drawdown_valid = (
        not isinstance(drawdown_penalty_weight, bool)
        and isfinite(float(drawdown_penalty_weight))
        and drawdown_penalty_weight >= 0
    )
    if not penalty_valid or not drawdown_valid:
        raise ValueError("tuning_custom_score_penalty_invalid")
    return (
        minimum,
        float(trade_count_penalty_bps),
        float(drawdown_penalty_weight),
    )


def _validate_custom_economic_score_configuration(
    features: DataFrame,
    actual: Series,
    gross_returns_by_index: Series,
    estimated_cost_bps: float,
    minimum_trade_count: int,
    trade_count_penalty_bps: float,
    drawdown_penalty_weight: float,
) -> tuple[int, float, float, float]:
    estimated_cost = _validate_custom_score_economic_alignment(
        features, actual, gross_returns_by_index, estimated_cost_bps
    )
    minimum, count_penalty, drawdown_weight = _validate_custom_score_penalties(
        minimum_trade_count,
        trade_count_penalty_bps,
        drawdown_penalty_weight,
    )
    return minimum, estimated_cost, count_penalty, drawdown_weight


def _predicted_buy_gross_returns(
    estimator: Any,
    features: DataFrame,
    gross_returns_by_index: Series,
) -> np.ndarray:
    predicted = np.asarray(estimator.predict(features), dtype=object)
    if predicted.ndim != 1 or len(predicted) != len(features):
        raise ValueError("tuning_custom_score_predictions_invalid")
    gross = gross_returns_by_index.reindex(features.index).to_numpy(dtype=float)
    if len(gross) != len(features) or not bool(np.isfinite(gross).all()):
        raise ValueError("tuning_economic_outcomes_missing_or_non_finite")
    return cast(np.ndarray, gross[predicted == "BUY"])


def _custom_score_drawdown_and_expectancy(
    net_returns: np.ndarray,
) -> tuple[float, float]:
    equity = np.concatenate(([1.0], np.cumprod(1.0 + net_returns)))
    peaks = np.maximum.accumulate(equity)
    drawdown = float(np.max((peaks - equity) / peaks))
    expectancy_bps = float(np.mean(net_returns)) * 10_000.0
    return drawdown, expectancy_bps


def _custom_score_from_net_returns(
    net_returns: np.ndarray,
    *,
    trade_count: int,
    minimum_trade_count: int,
    trade_count_penalty_bps: float,
    drawdown_penalty_weight: float,
) -> float:
    shortfall_ratio = max(
        0.0, float(minimum_trade_count - trade_count) / float(minimum_trade_count)
    )
    count_penalty = trade_count_penalty_bps * shortfall_ratio
    if trade_count == 0:
        return -count_penalty
    if not bool(np.isfinite(net_returns).all()) or bool(np.any(net_returns <= -1.0)):
        raise ValueError("tuning_custom_score_net_returns_invalid")
    drawdown, expectancy_bps = _custom_score_drawdown_and_expectancy(net_returns)
    positive_expectancy_reward_bps = max(0.0, expectancy_bps)
    drawdown_bps = drawdown * 10_000.0
    return (
        expectancy_bps
        + positive_expectancy_reward_bps
        - drawdown_penalty_weight * drawdown_bps
        - count_penalty
    )


def Custom_Economic_Score(
    estimator: Any,
    features: DataFrame,
    actual: Series,
    *,
    gross_returns_by_index: Series,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
    minimum_trade_count: int = CUSTOM_SCORE_MINIMUM_TRADES,
    trade_count_penalty_bps: float = CUSTOM_SCORE_TRADE_COUNT_PENALTY_BPS,
    drawdown_penalty_weight: float = CUSTOM_SCORE_DRAWDOWN_PENALTY_WEIGHT,
) -> float:
    """Score predicted Spot BUY events using expectancy, drawdown and sample size.

    This inner-CV proxy rewards positive net expectancy, penalizes losses and
    drawdown, and applies a linear soft penalty below 30 trades. Costed execution
    replay remains the economic acceptance authority.
    """
    (
        minimum,
        estimated_cost,
        count_penalty_bps,
        drawdown_weight,
    ) = _validate_custom_economic_score_configuration(
        features,
        actual,
        gross_returns_by_index,
        estimated_cost_bps,
        minimum_trade_count,
        trade_count_penalty_bps,
        drawdown_penalty_weight,
    )
    selected = _predicted_buy_gross_returns(estimator, features, gross_returns_by_index)
    trade_count = len(selected)
    net_returns = selected - estimated_cost / 10_000.0
    return _custom_score_from_net_returns(
        net_returns,
        trade_count=trade_count,
        minimum_trade_count=minimum,
        trade_count_penalty_bps=count_penalty_bps,
        drawdown_penalty_weight=drawdown_weight,
    )


def custom_eval_metric(
    estimator: Any,
    features: DataFrame,
    actual: Series,
    *,
    gross_returns_by_index: Series,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
) -> float:
    """Aggregate predicted-BUY gross profit less gross loss and estimated costs.

    Event-level proxy used only on inner CV folds inside outer Train. The final
    economic execution replay remains the acceptance authority.
    """
    if (
        isinstance(estimated_cost_bps, bool)
        or not isfinite(float(estimated_cost_bps))
        or estimated_cost_bps < 0
    ):
        raise ValueError("tuning_estimated_cost_bps_invalid")
    if (
        not isinstance(gross_returns_by_index, Series)
        or gross_returns_by_index.index.has_duplicates
    ):
        raise ValueError("tuning_economic_outcomes_invalid")
    if len(features) != len(actual):
        raise ValueError("tuning_economic_score_alignment_invalid")
    buy_mask = np.asarray(estimator.predict(features) == "BUY", dtype=bool)
    if not bool(buy_mask.any()):
        return 0.0
    gross = gross_returns_by_index.reindex(features.index).to_numpy(dtype=float)
    if len(gross) != len(features) or not bool(np.isfinite(gross).all()):
        raise ValueError("tuning_economic_outcomes_missing_or_non_finite")
    selected_returns = gross[buy_mask]
    gross_profit = float(selected_returns[selected_returns > 0.0].sum())
    gross_loss = abs(float(selected_returns[selected_returns < 0.0].sum()))
    estimated_costs = len(selected_returns) * float(estimated_cost_bps) / 10_000.0
    return gross_profit - (gross_loss + estimated_costs)


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
    class_labels: Sequence[str] = LABELS,
) -> dict[str, int]:
    if len(fit_positions) == 0:
        raise ValueError("tuning_inner_cv_fit_empty_after_purge")
    counts = {
        label: int(np.count_nonzero(label_values[fit_positions] == label))
        for label in class_labels
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
    class_labels: Sequence[str] = LABELS,
    minimum_validation_class_samples: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
    if len(validation_positions) == 0:
        raise ValueError("tuning_inner_cv_validation_empty")
    validation_start = int(train_rows[int(validation_positions[0])])
    fit_positions = _purged_fit_positions(
        candidate_positions, train_rows, train_ends, validation_start, horizon
    )
    fit_counts = _validate_inner_fit_support(
        fit_positions, label_values, minimum_class_samples, class_labels
    )
    validation = np.asarray(validation_positions, dtype=int)
    validation_counts = {
        label: int(np.count_nonzero(label_values[validation] == label))
        for label in class_labels
    }
    if minimum_validation_class_samples is not None and any(
        count < minimum_validation_class_samples for count in validation_counts.values()
    ):
        raise ValueError("tuning_inner_cv_validation_class_support_insufficient")
    if int(train_rows[int(fit_positions[-1])]) >= validation_start:
        raise ValueError("tuning_inner_cv_chronology_invalid")
    maximum_label_end = int(np.max(train_ends[fit_positions]))
    if maximum_label_end >= validation_start:
        raise ValueError("tuning_inner_cv_label_overlap")
    diagnostics = {
        "train_rows": int(len(fit_positions)),
        "validation_rows": int(len(validation)),
        "train_start_index": int(train_rows[int(fit_positions[0])]),
        "train_end_index": int(train_rows[int(fit_positions[-1])]),
        "validation_start_index": validation_start,
        "validation_end_index": int(train_rows[int(validation[-1])]),
        "maximum_training_label_end_index": maximum_label_end,
        "pre_validation_embargo_candles": horizon,
        **{f"fit_{label.lower()}_count": count for label, count in fit_counts.items()},
        **{
            f"validation_{label.lower()}_count": count
            for label, count in validation_counts.items()
        },
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
    class_labels: Sequence[str] = LABELS,
    minimum_validation_class_samples: int | None = None,
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
            class_labels=class_labels,
            minimum_validation_class_samples=minimum_validation_class_samples,
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
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
]:
    precision = np.asarray(cv_results["mean_test_hold_precision"], dtype=float)
    recall = np.asarray(cv_results["mean_test_hold_recall"], dtype=float)
    macro = np.asarray(cv_results["mean_test_macro_f1"], dtype=float)
    buy = np.asarray(cv_results["mean_test_buy_f1"], dtype=float)
    expectancy_key = (
        "mean_test_custom_economic_score"
        if "mean_test_custom_economic_score" in cv_results
        else "mean_test_net_expectancy_proxy"
    )
    expectancy = np.asarray(
        cv_results.get(expectancy_key, np.zeros_like(buy)),
        dtype=float,
    )
    params = cv_results["params"]
    arrays = (precision, recall, macro, buy, expectancy)
    if not precision.size or len({len(item) for item in arrays} | {len(params)}) != 1:
        raise ValueError("tuning_cv_results_invalid")
    if not all(bool(np.isfinite(item).all()) for item in arrays):
        raise ValueError("tuning_cv_results_invalid")
    return precision, recall, macro, buy, expectancy, params


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
    net_expectancy: np.ndarray,
    parameters: list[dict[str, Any]],
    objective: str,
) -> tuple[float, float, float, float, float, int, int, int]:
    complexity = _candidate_complexity_key(parameters[index])
    precision, recall, macro = (
        float(precisions[index]),
        float(recalls[index]),
        float(macro_f1[index]),
    )
    selected_buy_f1 = float(buy_f1[index])
    if objective == "BUY_F1":
        return selected_buy_f1, macro, precision, recall, 0.0, *complexity
    if objective in {"NET_EXPECTANCY_PROXY", "CUSTOM_ECONOMIC_SCORE"}:
        return (
            float(net_expectancy[index]),
            selected_buy_f1,
            macro,
            precision,
            recall,
            *complexity,
        )
    return precision, macro, recall, selected_buy_f1, 0.0, *complexity


def _select_candidate(
    cv_results: dict[str, Any],
    minimum_hold_recall: float,
    objective: str = "HOLD_PRECISION",
) -> tuple[int, str, int]:
    objective = _validate_tuning_objective(objective)
    precisions, recalls, macro_f1, buy_f1, net, parameters = _validated_cv_score_arrays(
        cv_results
    )
    feasible = [
        i for i, recall in enumerate(recalls) if float(recall) >= minimum_hold_recall
    ]
    candidates = (
        list(range(len(parameters)))
        if objective in {"BUY_F1", "NET_EXPECTANCY_PROXY", "CUSTOM_ECONOMIC_SCORE"}
        else feasible or list(range(len(parameters)))
    )
    selected = max(
        candidates,
        key=lambda i: _candidate_ranking_key(
            i, precisions, recalls, macro_f1, buy_f1, net, parameters, objective
        ),
    )
    if objective == "BUY_F1":
        status = "BUY_F1_MAXIMIZED"
    elif objective == "NET_EXPECTANCY_PROXY":
        status = "NET_EXPECTANCY_PROXY_MAXIMIZED"
    elif objective == "CUSTOM_ECONOMIC_SCORE":
        status = "CUSTOM_ECONOMIC_SCORE_MAXIMIZED"
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
    *,
    economic_outcomes: Series | None = None,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
    tuning_objective: str = "HOLD_PRECISION",
) -> Any:
    scoring = {
        "hold_precision": _hold_precision_score,
        "hold_recall": _hold_recall_score,
        "macro_f1": _macro_f1_score,
        "buy_f1": _buy_f1_score,
    }
    if economic_outcomes is not None:
        scoring["net_expectancy_proxy"] = partial(
            custom_eval_metric,
            gross_returns_by_index=economic_outcomes,
            estimated_cost_bps=estimated_cost_bps,
        )
        if tuning_objective == "CUSTOM_ECONOMIC_SCORE":
            scoring["custom_economic_score"] = partial(
                Custom_Economic_Score,
                gross_returns_by_index=economic_outcomes,
                estimated_cost_bps=estimated_cost_bps,
            )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=HYPERPARAMETER_DISTRIBUTIONS,
        n_iter=iterations,
        scoring=scoring,
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
    net = np.asarray(
        cv_results.get(
            "mean_test_net_expectancy_proxy", np.zeros(len(cv_results["params"]))
        ),
        dtype=float,
    )
    custom = np.asarray(
        cv_results.get(
            "mean_test_custom_economic_score",
            np.full(len(cv_results["params"]), np.nan),
        ),
        dtype=float,
    )
    return [
        {
            "candidate_index": i,
            "parameters": dict(parameters),
            "mean_cv_hold_precision": round(
                float(cv_results["mean_test_hold_precision"][i]), 6
            ),
            "mean_cv_hold_recall": round(
                float(cv_results["mean_test_hold_recall"][i]), 6
            ),
            "mean_cv_macro_f1": round(float(cv_results["mean_test_macro_f1"][i]), 6),
            "mean_cv_buy_f1": round(float(cv_results["mean_test_buy_f1"][i]), 6),
            "mean_cv_net_expectancy_proxy": round(float(net[i]), 8),
            "mean_cv_custom_economic_score": (
                round(float(custom[i]), 8) if np.isfinite(custom[i]) else None
            ),
        }
        for i, parameters in enumerate(cv_results["params"])
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
        "selected_mean_cv_net_expectancy_proxy": round(
            float(
                cv_results.get(
                    "mean_test_net_expectancy_proxy", np.zeros(candidate_count)
                )[best_index]
            ),
            8,
        ),
        "selected_mean_cv_custom_economic_score": (
            round(
                float(cv_results["mean_test_custom_economic_score"][best_index]),
                8,
            )
            if "mean_test_custom_economic_score" in cv_results
            else None
        ),
        "economic_scorer": (
            "Custom_Economic_Score"
            if tuning_objective == "CUSTOM_ECONOMIC_SCORE"
            else "custom_eval_metric"
            if tuning_objective == "NET_EXPECTANCY_PROXY"
            else None
        ),
        "custom_economic_score_contract": {
            "minimum_trade_count_soft_target": CUSTOM_SCORE_MINIMUM_TRADES,
            "trade_count_penalty_max_bps": CUSTOM_SCORE_TRADE_COUNT_PENALTY_BPS,
            "drawdown_penalty_weight": CUSTOM_SCORE_DRAWDOWN_PENALTY_WEIGHT,
            "positive_net_expectancy_reward_multiplier": 2.0,
            "portfolio_overlap_modeled": False,
        }
        if tuning_objective == "CUSTOM_ECONOMIC_SCORE"
        else None,
        "estimated_round_trip_cost_bps": 90.0
        if tuning_objective in {"NET_EXPECTANCY_PROXY", "CUSTOM_ECONOMIC_SCORE"}
        else None,
        "economic_scorer_scope": "INNER_CV_FOLDS_WITHIN_OUTER_TRAIN"
        if tuning_objective in {"NET_EXPECTANCY_PROXY", "CUSTOM_ECONOMIC_SCORE"}
        else None,
        "portfolio_overlap_modeled": False,
        "candidate_results": _candidate_result_records(cv_results),
        "candidate_fit_count": candidate_count * len(fold_diagnostics),
        "random_state": seed,
    }


def _validate_tuning_training_alignment(
    features: DataFrame,
    labels: Series,
    training_indices: Sequence[int],
    label_end_indices: Sequence[object],
) -> None:
    train_rows, _ = _training_rows(training_indices, label_end_indices)
    if len(features) != len(labels) or len(labels) != len(train_rows):
        raise ValueError("tuning_train_matrix_alignment_invalid")


def _validate_tuning_economic_inputs(
    features: DataFrame,
    tuning_objective: str,
    economic_outcomes: Series | None,
) -> None:
    if (
        tuning_objective in {"NET_EXPECTANCY_PROXY", "CUSTOM_ECONOMIC_SCORE"}
        and economic_outcomes is None
    ):
        raise ValueError("tuning_economic_objective_requires_outcomes")
    if economic_outcomes is not None:
        if (
            not isinstance(economic_outcomes, Series)
            or economic_outcomes.index.has_duplicates
        ):
            raise ValueError("tuning_economic_outcomes_invalid")
        aligned = economic_outcomes.reindex(features.index)
        if aligned.isna().any() or not bool(
            np.isfinite(aligned.to_numpy(dtype=float)).all()
        ):
            raise ValueError("tuning_economic_outcomes_missing_or_non_finite")


def _validate_estimated_cost_bps(estimated_cost_bps: float) -> None:
    if (
        isinstance(estimated_cost_bps, bool)
        or not isfinite(float(estimated_cost_bps))
        or estimated_cost_bps < 0
    ):
        raise ValueError("tuning_estimated_cost_bps_invalid")


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
    economic_outcomes: Series | None = None,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
) -> tuple[Any, dict[str, Any]]:
    """Tune on outer Train only, then refit the selected model on all outer Train."""
    tuning_objective = _validate_tuning_objective(tuning_objective)
    iterations, split_count, seed = _validate_tuning_configuration(
        minimum_hold_recall, n_iter, n_splits, random_state, class_weight_policy
    )
    _validate_tuning_training_alignment(
        features, labels, training_indices, label_end_indices
    )
    _validate_tuning_economic_inputs(features, tuning_objective, economic_outcomes)
    _validate_estimated_cost_bps(estimated_cost_bps)
    cv_splits, fold_diagnostics = build_purged_time_series_splits(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles=horizon_candles,
        n_splits=split_count,
    )
    search = _build_randomized_search(
        estimator,
        cv_splits,
        iterations,
        seed,
        economic_outcomes=economic_outcomes,
        estimated_cost_bps=estimated_cost_bps,
        tuning_objective=tuning_objective,
    )
    search.fit(features, labels)
    results: dict[str, Any] = search.cv_results_
    best_index, selection_status, recall_feasible_count = _select_candidate(
        results, float(minimum_hold_recall), tuning_objective
    )
    best_parameters = dict(results["params"][best_index])
    fitted_estimator = _fit_selected_estimator(
        estimator, features, labels, best_parameters
    )
    diagnostics = _build_tuning_diagnostics(
        results,
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


BINARY_BUY_LABELS = ("BUY", "NO_BUY")
BINARY_BUY_OBJECTIVE = "BINARY_BUY_ECONOMIC_SCORE"
DEFAULT_BINARY_BUY_TARGET_FRACTION = DEFAULT_TARGET_BUY_FRACTION
DEFAULT_BINARY_MISSED_OPPORTUNITY_PENALTY_WEIGHT = 0.50


def _binary_buy_class_index(estimator: Any) -> int:
    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is None:
        raise ValueError("tuning_binary_probability_classes_missing")
    classes = [str(value) for value in raw_classes]
    if len(classes) != 2 or set(classes) != set(BINARY_BUY_LABELS):
        raise ValueError("tuning_binary_probability_classes_invalid")
    return classes.index("BUY")


def _validated_binary_probability_matrix(
    raw_probabilities: Any, row_count: int
) -> np.ndarray:
    matrix = np.asarray(raw_probabilities, dtype=float)
    if matrix.shape != (row_count, 2):
        raise ValueError("tuning_binary_probability_shape_invalid")
    valid_values = (
        bool(np.isfinite(matrix).all())
        and not bool(np.any(matrix < 0.0))
        and not bool(np.any(matrix > 1.0))
        and bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6))
    )
    if not valid_values:
        raise ValueError("tuning_binary_probabilities_invalid")
    return matrix


def _binary_buy_probabilities(estimator: Any, features: DataFrame) -> np.ndarray:
    """Return validated BUY probabilities independent of estimator class order."""
    class_index = _binary_buy_class_index(estimator)
    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError("tuning_binary_predict_proba_missing")
    matrix = _validated_binary_probability_matrix(
        predict_proba(features), len(features)
    )
    return matrix[:, class_index]


def _binary_buy_threshold(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError("tuning_binary_buy_threshold_invalid")
    return float(value)


def _validate_binary_actual_labels(actual: Series, row_count: int) -> None:
    if len(actual) != row_count or not bool(actual.isin(BINARY_BUY_LABELS).all()):
        raise ValueError("tuning_binary_actual_labels_invalid")


def _binary_buy_targets(actual: Series, row_count: int) -> np.ndarray:
    _validate_binary_actual_labels(actual, row_count)
    targets = np.asarray(actual.to_numpy(dtype=object) == "BUY", dtype=np.int64)
    if len(np.unique(targets)) != 2:
        raise ValueError("tuning_binary_actual_requires_both_classes")
    return targets


def _binary_buy_predictions(
    estimator: Any,
    features: DataFrame,
    actual: Series,
    threshold: float | None = None,
    target_buy_fraction: float = DEFAULT_BINARY_BUY_TARGET_FRACTION,
) -> np.ndarray:
    _binary_buy_targets(actual, len(features))
    probabilities = _binary_buy_probabilities(estimator, features)
    cutoff = (
        _binary_buy_threshold(threshold)
        if threshold is not None
        else float(
            select_dynamic_buy_threshold(
                probabilities, target_buy_fraction=target_buy_fraction
            )["threshold"]
        )
    )
    return probabilities >= cutoff


def _binary_buy_precision_score(
    estimator: Any, features: DataFrame, actual: Series
) -> float:
    predicted = _binary_buy_predictions(estimator, features, actual)
    targets = _binary_buy_targets(actual, len(features))
    return float(precision_score(targets, predicted, zero_division=0))


def _binary_buy_recall_score(
    estimator: Any, features: DataFrame, actual: Series
) -> float:
    predicted = _binary_buy_predictions(estimator, features, actual)
    targets = _binary_buy_targets(actual, len(features))
    return float(recall_score(targets, predicted, zero_division=0))


def _binary_buy_f1_score(estimator: Any, features: DataFrame, actual: Series) -> float:
    predicted = _binary_buy_predictions(estimator, features, actual)
    targets = _binary_buy_targets(actual, len(features))
    return float(f1_score(targets, predicted, zero_division=0))


def _binary_buy_roc_auc_score(
    estimator: Any, features: DataFrame, actual: Series
) -> float:
    targets = _binary_buy_targets(actual, len(features))
    probabilities = _binary_buy_probabilities(estimator, features)
    return float(roc_auc_score(targets, probabilities))


def _binary_buy_drawdown(net_returns: np.ndarray) -> float:
    if net_returns.size == 0:
        return 0.0
    equity = np.concatenate(([1.0], np.cumprod(1.0 + net_returns)))
    if not bool(np.isfinite(equity).all()) or bool(np.any(equity <= 0.0)):
        raise ValueError("tuning_binary_economic_equity_invalid")
    peaks = np.maximum.accumulate(equity)
    return float(np.max((peaks - equity) / peaks))


def _validate_binary_score_alignment(
    features: DataFrame, actual: Series, gross_returns_by_index: Series
) -> None:
    if (
        not isinstance(gross_returns_by_index, Series)
        or gross_returns_by_index.index.has_duplicates
        or len(features) != len(actual)
    ):
        raise ValueError("tuning_binary_economic_alignment_invalid")


def _validate_binary_score_costs(
    estimated_cost_bps: float,
    drawdown_penalty_weight: float,
    missed_opportunity_penalty_weight: float,
) -> None:
    for value, reason in (
        (estimated_cost_bps, "tuning_binary_estimated_cost_invalid"),
        (drawdown_penalty_weight, "tuning_binary_drawdown_penalty_invalid"),
        (
            missed_opportunity_penalty_weight,
            "tuning_binary_missed_opportunity_penalty_invalid",
        ),
    ):
        if isinstance(value, bool) or not isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(reason)


def _selected_binary_net_returns(
    features: DataFrame,
    actual: Series,
    gross_returns_by_index: Series,
    buy_probabilities: np.ndarray,
    cutoff: float,
    estimated_cost_bps: float,
) -> np.ndarray:
    selected = buy_probabilities >= cutoff
    gross = gross_returns_by_index.reindex(features.index).to_numpy(dtype=float)
    if len(gross) != len(features) or not bool(np.isfinite(gross).all()):
        raise ValueError("tuning_binary_economic_outcomes_missing_or_non_finite")
    net_returns = gross[selected] - float(estimated_cost_bps) / 10_000.0
    selected_actual = actual.to_numpy(dtype=object)[selected]
    false_positive_mask = selected_actual == "NO_BUY"
    net_returns[false_positive_mask] = -np.abs(net_returns[false_positive_mask])
    if not bool(np.isfinite(net_returns).all()) or bool(np.any(net_returns <= -1.0)):
        raise ValueError("tuning_binary_net_returns_invalid")
    return net_returns


def _binary_economic_score_value(
    net_returns: np.ndarray,
    drawdown_penalty_weight: float,
    missed_opportunity_penalty_bps: float,
) -> float:
    total_net_pnl_bps = float(np.sum(net_returns)) * 10_000.0
    positive_pnl_reward_bps = max(0.0, total_net_pnl_bps)
    drawdown_bps = _binary_buy_drawdown(net_returns) * 10_000.0
    return (
        total_net_pnl_bps
        + positive_pnl_reward_bps
        - float(drawdown_penalty_weight) * drawdown_bps
        - float(missed_opportunity_penalty_bps)
    )


def Binary_BUY_Economic_Score(
    estimator: Any,
    features: DataFrame,
    actual: Series,
    *,
    gross_returns_by_index: Series,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
    buy_probability_threshold: float | None = None,
    target_buy_fraction: float = DEFAULT_BINARY_BUY_TARGET_FRACTION,
    drawdown_penalty_weight: float = CUSTOM_SCORE_DRAWDOWN_PENALTY_WEIGHT,
    missed_opportunity_penalty_weight: float = (
        DEFAULT_BINARY_MISSED_OPPORTUNITY_PENALTY_WEIGHT
    ),
) -> float:
    """Score executed BUY outcomes and positive BUY opportunities missed.

    The threshold is derived from the fold's probability ranking when no
    explicit cutoff is supplied. Actual labels never participate in threshold
    selection. A missed opportunity is an actual BUY row with positive net
    outcome which did not pass the probability cutoff.
    """
    _validate_binary_actual_labels(actual, len(features))
    _validate_binary_score_alignment(features, actual, gross_returns_by_index)
    _validate_binary_score_costs(
        estimated_cost_bps,
        drawdown_penalty_weight,
        missed_opportunity_penalty_weight,
    )
    target_fraction = validate_target_buy_fraction(target_buy_fraction)
    probabilities = _binary_buy_probabilities(estimator, features)
    cutoff = (
        _binary_buy_threshold(buy_probability_threshold)
        if buy_probability_threshold is not None
        else float(
            select_dynamic_buy_threshold(
                probabilities, target_buy_fraction=target_fraction
            )["threshold"]
        )
    )
    net_returns = _selected_binary_net_returns(
        features,
        actual,
        gross_returns_by_index,
        probabilities,
        cutoff,
        estimated_cost_bps,
    )
    gross = gross_returns_by_index.reindex(features.index).to_numpy(dtype=float)
    selected = probabilities >= cutoff
    missed_buy_mask = (actual.to_numpy(dtype=object) == "BUY") & ~selected
    missed_net = np.maximum(
        gross[missed_buy_mask] - float(estimated_cost_bps) / 10_000.0, 0.0
    )
    missed_penalty_bps = (
        float(np.sum(missed_net)) * 10_000.0 * float(missed_opportunity_penalty_weight)
    )
    return _binary_economic_score_value(
        net_returns, drawdown_penalty_weight, missed_penalty_bps
    )


def _validate_binary_tuning_inputs(
    estimator: Any,
    features: DataFrame,
    labels: Series,
    training_indices: Sequence[int],
    label_end_indices: Sequence[object],
    *,
    class_weight_policy: str,
    horizon_candles: int,
    n_iter: int,
    n_splits: int,
    random_state: int,
    estimated_cost_bps: float,
    buy_probability_threshold: float | None,
    target_buy_fraction: float,
    missed_opportunity_penalty_weight: float,
) -> tuple[int, int, int, float, float | None, float, float]:
    del estimator
    iterations = _positive_integer(
        n_iter, "tuning_binary_random_search_iterations_invalid"
    )
    split_count = _positive_integer(
        n_splits, "tuning_binary_inner_cv_split_count_invalid"
    )
    seed = _nonnegative_integer(random_state, "tuning_binary_random_state_invalid")
    _positive_integer(horizon_candles, "tuning_binary_horizon_invalid")
    if iterations > 27:
        raise ValueError("tuning_random_search_iterations_out_of_policy")
    if not class_weight_policy.strip():
        raise ValueError("tuning_class_weight_policy_missing")
    _validate_tuning_training_alignment(
        features, labels, training_indices, label_end_indices
    )
    _binary_buy_targets(labels, len(features))
    if (
        isinstance(estimated_cost_bps, bool)
        or not isfinite(float(estimated_cost_bps))
        or estimated_cost_bps < 0.0
    ):
        raise ValueError("tuning_binary_estimated_cost_invalid")
    validate_target_buy_fraction(target_buy_fraction)
    threshold = (
        _binary_buy_threshold(buy_probability_threshold)
        if buy_probability_threshold is not None
        else None
    )
    if (
        isinstance(missed_opportunity_penalty_weight, bool)
        or not isfinite(float(missed_opportunity_penalty_weight))
        or missed_opportunity_penalty_weight < 0.0
    ):
        raise ValueError("tuning_binary_missed_opportunity_penalty_invalid")
    return (
        iterations,
        split_count,
        seed,
        float(estimated_cost_bps),
        threshold,
        float(target_buy_fraction),
        float(missed_opportunity_penalty_weight),
    )


def _validate_binary_tuning_economic_outcomes(
    features: DataFrame, economic_outcomes: Series
) -> None:
    if (
        not isinstance(economic_outcomes, Series)
        or economic_outcomes.index.has_duplicates
    ):
        raise ValueError("tuning_binary_economic_outcomes_invalid")
    aligned_outcomes = economic_outcomes.reindex(features.index)
    if aligned_outcomes.isna().any():
        raise ValueError("tuning_binary_economic_outcomes_missing_or_non_finite")
    if not bool(np.isfinite(aligned_outcomes.to_numpy(dtype=float)).all()):
        raise ValueError("tuning_binary_economic_outcomes_missing_or_non_finite")


def _build_binary_tuning_search(
    estimator: Any,
    *,
    iterations: int,
    seed: int,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    economic_outcomes: Series,
    estimated_cost: float,
    threshold: float | None,
    target_fraction: float,
    missed_penalty_weight: float,
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=HYPERPARAMETER_DISTRIBUTIONS,
        n_iter=iterations,
        scoring={
            "binary_buy_precision": _binary_buy_precision_score,
            "binary_buy_recall": _binary_buy_recall_score,
            "binary_buy_f1": _binary_buy_f1_score,
            "binary_roc_auc": _binary_buy_roc_auc_score,
            "binary_buy_economic_score": partial(
                Binary_BUY_Economic_Score,
                gross_returns_by_index=economic_outcomes,
                estimated_cost_bps=estimated_cost,
                buy_probability_threshold=threshold,
                target_buy_fraction=target_fraction,
                missed_opportunity_penalty_weight=missed_penalty_weight,
            ),
        },
        cv=cv_splits,
        refit=False,
        random_state=seed,
        n_jobs=1,
        pre_dispatch=1,
        error_score="raise",
        return_train_score=False,
    )


def _binary_tuning_cv_result_arrays(
    cv_results: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    score_values = np.asarray(
        cv_results["mean_test_binary_buy_economic_score"], dtype=float
    )
    precision_values = np.asarray(
        cv_results["mean_test_binary_buy_precision"], dtype=float
    )
    recall_values = np.asarray(cv_results["mean_test_binary_buy_recall"], dtype=float)
    f1_values = np.asarray(cv_results["mean_test_binary_buy_f1"], dtype=float)
    auc_values = np.asarray(cv_results["mean_test_binary_roc_auc"], dtype=float)
    parameters = cv_results["params"]
    arrays = (score_values, precision_values, recall_values, f1_values, auc_values)
    if not score_values.size:
        raise ValueError("tuning_binary_cv_results_invalid")
    if len({len(item) for item in arrays} | {len(parameters)}) != 1:
        raise ValueError("tuning_binary_cv_results_invalid")
    if not all(bool(np.isfinite(item).all()) for item in arrays):
        raise ValueError("tuning_binary_cv_results_invalid")
    return (
        score_values,
        precision_values,
        recall_values,
        f1_values,
        auc_values,
        parameters,
    )


def _select_binary_tuning_candidate(
    cv_results: dict[str, Any],
) -> tuple[dict[str, Any], int, dict[str, float]]:
    (
        score_values,
        precision_values,
        recall_values,
        f1_values,
        auc_values,
        parameters,
    ) = _binary_tuning_cv_result_arrays(cv_results)
    best_index = max(
        range(len(parameters)),
        key=lambda index: (
            float(score_values[index]),
            float(precision_values[index]),
            float(auc_values[index]),
            float(recall_values[index]),
            *_candidate_complexity_key(dict(parameters[index])),
        ),
    )
    selected_scores = {
        "economic_score": float(score_values[best_index]),
        "buy_precision": float(precision_values[best_index]),
        "buy_recall": float(recall_values[best_index]),
        "buy_f1": float(f1_values[best_index]),
        "roc_auc": float(auc_values[best_index]),
    }
    return dict(parameters[best_index]), len(parameters), selected_scores


def _binary_tuning_diagnostics(
    *,
    candidate_count: int,
    fold_diagnostics: list[dict[str, int]],
    minimum_fit: int,
    minimum_validation: int,
    class_weight_policy: str,
    threshold: float | None,
    target_fraction: float,
    best_parameters: dict[str, Any],
    selected_scores: dict[str, float],
    estimated_cost: float,
    missed_penalty_weight: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "search_method": "RANDOM_SEARCH",
        "candidate_count": candidate_count,
        "candidate_fit_count": candidate_count * len(fold_diagnostics),
        "inner_cv_method": "PURGED_EXPANDING_TIME_SERIES",
        "inner_cv_fold_count": len(fold_diagnostics),
        "inner_cv_scope": "OUTER_TRAIN_FIT_WINDOW_ONLY",
        "inner_cv_folds": fold_diagnostics,
        "class_labels": list(BINARY_BUY_LABELS),
        "minimum_fit_class_samples": minimum_fit,
        "minimum_validation_class_samples": minimum_validation,
        "validation_accessed_during_search": False,
        "final_refit_scope": "PRE_CALIBRATION_TRAIN_ONLY",
        "class_weight_policy": class_weight_policy,
        "selection_objective": BINARY_BUY_OBJECTIVE,
        "economic_scorer": "Binary_BUY_Economic_Score",
        "buy_probability_threshold": threshold,
        "threshold_selection": (
            "DYNAMIC_TOP_PERCENTILE_PER_INNER_CV_FOLD"
            if threshold is None
            else "EXPLICIT_FIXED_CUTOFF"
        ),
        "target_buy_fraction": target_fraction,
        "selected_parameters": best_parameters,
        "selected_mean_cv_binary_buy_economic_score": round(
            selected_scores["economic_score"], 8
        ),
        "selected_mean_cv_buy_precision": round(selected_scores["buy_precision"], 8),
        "selected_mean_cv_buy_recall": round(selected_scores["buy_recall"], 8),
        "selected_mean_cv_buy_f1": round(selected_scores["buy_f1"], 8),
        "selected_mean_cv_roc_auc": round(selected_scores["roc_auc"], 8),
        "custom_economic_score_contract": {
            "score_scope": "EXECUTED_BUY_PNL_MINUS_MISSED_POSITIVE_BUY_OPPORTUNITIES",
            "estimated_round_trip_cost_bps": estimated_cost,
            "threshold_selection": "DYNAMIC_TOP_PERCENTILE_PER_INNER_CV_FOLD",
            "target_buy_fraction": target_fraction,
            "missed_opportunity_penalty_weight": missed_penalty_weight,
            "missed_opportunity_definition": (
                "ACTUAL_BUY_WITH_POSITIVE_NET_OUTCOME_NOT_SELECTED"
            ),
            "missed_opportunity_penalty_bps": (
                "OBSERVED_FROM_COSTED_INNER_CV_OUTCOMES"
            ),
            "no_buy_predictions_are_penalized": True,
            "minimum_trade_count_penalty": False,
            "drawdown_penalty_weight": CUSTOM_SCORE_DRAWDOWN_PENALTY_WEIGHT,
            "positive_net_expectancy_reward_multiplier": 2.0,
        },
        "random_state": seed,
    }


def tune_lightgbm_binary_classifier(
    estimator: Any,
    features: DataFrame,
    labels: Series,
    training_indices: Sequence[int],
    label_end_indices: Sequence[object],
    *,
    horizon_candles: int,
    class_weight_policy: str,
    n_iter: int = DEFAULT_RANDOM_SEARCH_ITERATIONS,
    n_splits: int = DEFAULT_INNER_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    minimum_class_samples: int = DEFAULT_MINIMUM_CLASS_SAMPLES,
    minimum_validation_class_samples: int = DEFAULT_MINIMUM_CLASS_SAMPLES,
    economic_outcomes: Series,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_ROUND_TRIP_COST_BPS,
    buy_probability_threshold: float | None = None,
    target_buy_fraction: float = DEFAULT_BINARY_BUY_TARGET_FRACTION,
    missed_opportunity_penalty_weight: float = (
        DEFAULT_BINARY_MISSED_OPPORTUNITY_PENALTY_WEIGHT
    ),
) -> tuple[Any, dict[str, Any]]:
    """Tune LightGBM using dynamic-percentile, cost-aware binary BUY economics."""
    (
        iterations,
        split_count,
        seed,
        estimated_cost,
        threshold,
        target_fraction,
        missed_penalty_weight,
    ) = _validate_binary_tuning_inputs(
        estimator,
        features,
        labels,
        training_indices,
        label_end_indices,
        class_weight_policy=class_weight_policy,
        horizon_candles=horizon_candles,
        n_iter=n_iter,
        n_splits=n_splits,
        random_state=random_state,
        estimated_cost_bps=estimated_cost_bps,
        buy_probability_threshold=buy_probability_threshold,
        target_buy_fraction=target_buy_fraction,
        missed_opportunity_penalty_weight=missed_opportunity_penalty_weight,
    )
    minimum_fit = _positive_integer(
        minimum_class_samples, "tuning_binary_minimum_class_samples_invalid"
    )
    minimum_validation = _positive_integer(
        minimum_validation_class_samples,
        "tuning_binary_minimum_validation_class_samples_invalid",
    )
    _validate_binary_tuning_economic_outcomes(features, economic_outcomes)
    cv_splits, fold_diagnostics = build_purged_time_series_splits(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles=horizon_candles,
        n_splits=split_count,
        minimum_class_samples=minimum_fit,
        class_labels=BINARY_BUY_LABELS,
        minimum_validation_class_samples=minimum_validation,
    )
    search = _build_binary_tuning_search(
        estimator,
        iterations=iterations,
        seed=seed,
        cv_splits=cv_splits,
        economic_outcomes=economic_outcomes,
        estimated_cost=estimated_cost,
        threshold=threshold,
        target_fraction=target_fraction,
        missed_penalty_weight=missed_penalty_weight,
    )
    search.fit(features, labels)
    best_parameters, candidate_count, selected_scores = _select_binary_tuning_candidate(
        search.cv_results_
    )
    fitted_estimator = _fit_selected_estimator(
        estimator, features, labels, best_parameters
    )
    diagnostics = _binary_tuning_diagnostics(
        candidate_count=candidate_count,
        fold_diagnostics=fold_diagnostics,
        minimum_fit=minimum_fit,
        minimum_validation=minimum_validation,
        class_weight_policy=class_weight_policy,
        threshold=threshold,
        target_fraction=target_fraction,
        best_parameters=best_parameters,
        selected_scores=selected_scores,
        estimated_cost=estimated_cost,
        missed_penalty_weight=missed_penalty_weight,
        seed=seed,
    )
    return fitted_estimator, diagnostics
