"""A-only, research-only Stage B binary-classification experiment.

Positive class means a long-side 12-hour gross return strictly greater than
the unchanged 0.60% round-trip cost. A past-only train-fold volatility filter
blocks entries below that fold's training 75th-percentile volatility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts.run_stage_b_walk_forward import (
    A_HOLDOUT,
    A_TRAIN,
    HOUR,
    ROUND_TRIP_COST,
    SOURCE_RUN_ID,
    WINDOW,
    build_fixed_window_folds,
    feature_availability_diagnostics,
    select_segment_features,
    utc_text,
    verify_source_artifact,
)
from scripts.stage_b_research_features import NEW_FEATURE_COLUMNS, add_signal_features
from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    forward_return_target,
)
from services.freqai_baseline.walkforward import (
    BASELINE_MODEL_PARAMETERS,
    validate_ohlcv,
)

REPORT_SCHEMA = "alix.stage-b.classification-volatility.v1"
HORIZON = 12
FEATURE_GROUPS = ("stage_b_8", "extended_trend_rsi")
MODEL_PROFILES = ("unweighted", "balanced")
PROBABILITY_THRESHOLDS = (0.35, 0.50, 0.65)
VOLATILITY_CUTOFF_QUANTILE = 0.75
MIN_PRIOR_TRADES_FOR_SELECTION = 30
MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION = 100
FEE_BPS_SIDE = 25.0
SLIPPAGE_BPS_SIDE = 5.0
LABEL_COLUMN = "classification_event"
VOLATILITY_COLUMN = "past_volatility_filter"
BASELINE_CANDIDATE_ID = "stage_b_8|unweighted|p0.50"
CLOSED_FLAGS: dict[str, object] = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "economic_acceptance": "NO-GO",
    "model_promoted": False,
    "production_model_updated": False,
    "strategy_registry_updated": False,
    "orders_or_execution_api_used": False,
    "private_api_calls": 0,
    "order_endpoint_calls": 0,
    "real_order_attempts": 0,
}
CLASSIFIER_BASE_PARAMETERS: dict[str, object] = {
    **BASELINE_MODEL_PARAMETERS,
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 120,
    "learning_rate": 0.03,
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_commit_sha(commit_sha: str) -> None:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("classification_commit_sha_invalid")


def event_labels(forward_returns: Sequence[float]) -> np.ndarray:
    """Label the actionable long event: gross 12h return strictly greater than cost."""
    values = np.asarray(forward_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("classification_forward_returns_non_finite_or_not_1d")
    return (values > ROUND_TRIP_COST).astype(np.int8)


def volatility_cutoff(
    training_volatility: Sequence[float],
    quantile: float = VOLATILITY_CUTOFF_QUANTILE,
) -> float:
    """Derive a regime cutoff from training-fold values only."""
    values = np.asarray(training_volatility, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError("classification_volatility_cutoff_input_invalid")
    cutoff = float(np.quantile(values, quantile))
    if not math.isfinite(cutoff):
        raise ValueError("classification_volatility_cutoff_non_finite")
    return cutoff


def signal_mask(
    probabilities: Sequence[float],
    current_volatility: Sequence[float],
    probability_threshold: float,
    training_volatility_cutoff: float,
) -> np.ndarray:
    """Require classifier confidence and a high-volatility past-only regime."""
    probability = np.asarray(probabilities, dtype=float)
    volatility = np.asarray(current_volatility, dtype=float)
    if (
        probability.ndim != 1
        or volatility.ndim != 1
        or len(probability) != len(volatility)
    ):
        raise ValueError("classification_signal_array_shape_invalid")
    if not np.isfinite(probability).all() or not np.isfinite(volatility).all():
        raise ValueError("classification_signal_inputs_non_finite")
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("classification_probability_threshold_invalid")
    if not math.isfinite(training_volatility_cutoff):
        raise ValueError("classification_training_volatility_cutoff_invalid")
    return (probability >= probability_threshold) & (
        volatility >= training_volatility_cutoff
    )


def _candidate_features(
    engineered: DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, str]]]]:
    all_candidates = tuple(FEATURE_COLUMNS) + NEW_FEATURE_COLUMNS
    baseline, excluded_baseline = select_segment_features(engineered, FEATURE_COLUMNS)
    extended, excluded_extended = select_segment_features(engineered, all_candidates)
    selected = {"stage_b_8": baseline, "extended_trend_rsi": extended}
    excluded = {
        "stage_b_8": excluded_baseline,
        "extended_trend_rsi": excluded_extended,
    }
    for group, columns in selected.items():
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"classification_feature_contract_invalid:{group}")
    return selected, excluded


def _common_evaluation_bounds(
    engineered: DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Use one contiguous grid after feature warm-up and label maturity."""
    target = forward_return_target(engineered, horizon=HORIZON)
    numeric = engineered.loc[:, list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    valid = (
        numeric.notna().all(axis=1)
        & target.notna()
        & np.isfinite(target)
        & engineered["%-volatility_24"].notna()
    )
    positions = np.flatnonzero(valid.to_numpy(dtype=bool))
    if positions.size == 0:
        raise ValueError("classification_common_range_empty")
    first, last = int(positions[0]), int(positions[-1])
    if not bool(valid.iloc[first : last + 1].all()):
        raise ValueError("classification_common_range_has_internal_gap")
    dates = pd.to_datetime(engineered["date"], utc=True, errors="raise")
    common = dates.iloc[first : last + 1]
    if not (common.diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise ValueError("classification_common_grid_not_contiguous")
    return pd.Timestamp(common.iloc[0]), pd.Timestamp(common.iloc[-1])


def _build_dataset(
    engineered: DataFrame,
    feature_columns: Sequence[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> DataFrame:
    """Keep naturally complete rows in the shared range; never impute."""
    forward_returns = forward_return_target(engineered, horizon=HORIZON)
    features = engineered.loc[:, list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    volatility = pd.to_numeric(engineered["%-volatility_24"], errors="coerce")
    dates = pd.to_datetime(engineered["date"], utc=True, errors="raise")
    inside = dates.ge(start_utc) & dates.le(end_utc)
    valid = (
        features.notna().all(axis=1)
        & forward_returns.notna()
        & np.isfinite(forward_returns)
        & volatility.notna()
        & np.isfinite(volatility)
        & inside
    )
    selected = engineered.loc[valid, ["date", "close", *feature_columns]].copy()
    realized = forward_returns.loc[valid].to_numpy(dtype=float)
    selected[TARGET_COLUMN] = realized
    selected[LABEL_COLUMN] = event_labels(realized)
    selected[VOLATILITY_COLUMN] = volatility.loc[valid].to_numpy(dtype=float)
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError("classification_dataset_empty")
    timestamps = pd.to_datetime(selected["date"], utc=True, errors="raise")
    if timestamps.iloc[0] != start_utc or timestamps.iloc[-1] != end_utc:
        raise ValueError("classification_candidate_does_not_cover_common_bounds")
    if not (timestamps.diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise ValueError("classification_dataset_not_contiguous")
    values = selected.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(realized).all():
        raise ValueError("classification_dataset_non_finite")
    return selected


def _build_candidate_datasets(
    engineered: DataFrame,
    selected_features: dict[str, list[str]],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[dict[str, DataFrame], int]:
    datasets: dict[str, DataFrame] = {}
    reference_ns: np.ndarray | None = None
    reference_count: int | None = None
    for feature_group in FEATURE_GROUPS:
        dataset = _build_dataset(
            engineered, selected_features[feature_group], start_utc, end_utc
        )
        timestamps = (
            pd.to_datetime(dataset["date"], utc=True, errors="raise")
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )
        if reference_ns is None:
            reference_ns = timestamps
            reference_count = len(dataset)
        elif not np.array_equal(timestamps, reference_ns):
            raise ValueError("classification_candidate_timestamp_grid_mismatch")
        if len(dataset) != reference_count:
            raise ValueError("classification_candidate_row_count_mismatch")
        datasets[feature_group] = dataset
    if reference_count is None:
        raise ValueError("classification_candidate_grid_empty")
    return datasets, reference_count


def _drawdown_abs(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    equity, peak, worst = 1.0, 1.0, 0.0
    for value in returns:
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("classification_trade_return_invalid")
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return float(abs(worst))


def _economic_gate(
    trade_count: int,
    win_rate: float | None,
    expectancy: float | None,
    drawdown_abs: float | None,
) -> dict[str, Any]:
    count_pass = trade_count >= MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION
    win_pass = win_rate is not None and win_rate > 0.60
    expectancy_pass = expectancy is not None and expectancy > 0.0
    drawdown_pass = drawdown_abs is not None and drawdown_abs < 0.15
    passed = count_pass and win_pass and expectancy_pass and drawdown_pass
    status = (
        "INSUFFICIENT_SAMPLE"
        if not count_pass
        else "THRESHOLDS_MET_RESEARCH_ONLY"
        if passed
        else "NO_GO_THRESHOLDS_NOT_MET"
    )
    return {
        "required_trade_count": ">= 100",
        "required_win_rate": "> 0.60",
        "required_net_expectancy_per_trade": "> 0",
        "required_absolute_max_drawdown": "< 0.15",
        "trade_count_pass": count_pass,
        "win_rate_pass": win_pass,
        "expectancy_pass": expectancy_pass,
        "drawdown_pass": drawdown_pass,
        "status": status,
        "authorization_effect": "NONE; Champion remains null and trading stays NO-GO",
    }


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(trade["net_return"]) for trade in trades]
    count = len(returns)
    wins = sum(value > 0.0 for value in returns)
    rate = wins / count if count else None
    expectancy = float(np.mean(returns)) if count else None
    drawdown = _drawdown_abs(returns)
    return {
        "trade_count": count,
        "winning_trade_count": int(wins),
        "losing_or_flat_trade_count": int(count - wins),
        "win_rate": float(rate) if rate is not None else None,
        "net_expectancy_per_trade": expectancy,
        "net_return_compounded": (
            float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)
            if returns
            else 0.0
        ),
        "max_drawdown_abs": drawdown,
        "economic_threshold_evaluation": _economic_gate(
            count, rate, expectancy, drawdown
        ),
    }


def _fit_classifier(
    model_profile: str, train: DataFrame, feature_columns: list[str]
) -> Any:
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]

    labels = train[LABEL_COLUMN].to_numpy(dtype=np.int8)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        raise ValueError("classification_training_fold_has_single_class")
    if model_profile not in MODEL_PROFILES:
        raise ValueError(f"classification_model_profile_invalid:{model_profile}")
    parameters = dict(CLASSIFIER_BASE_PARAMETERS)
    if model_profile == "balanced":
        parameters["scale_pos_weight"] = negative / positive
    model = LGBMClassifier(**parameters)
    model.fit(train.loc[:, feature_columns], labels)
    return model


def _build_prediction_and_trade_rows(
    *,
    candidate_id: str,
    feature_group: str,
    model_profile: str,
    probability_threshold: float,
    fold_index: int,
    fold_id: str,
    scored: DataFrame,
    test: DataFrame,
    probabilities: np.ndarray,
    volatility_threshold: float,
    terminal_fold: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for offset, probability in enumerate(probabilities.tolist()):
        entry = pd.Timestamp(scored["date"].iloc[offset])
        current_volatility = float(scored[VOLATILITY_COLUMN].iloc[offset])
        opportunity = offset % HORIZON == 0 and (
            terminal_fold or offset + HORIZON < len(test)
        )
        high_volatility = current_volatility >= volatility_threshold
        take_trade = bool(
            opportunity and high_volatility and probability >= probability_threshold
        )
        gross_return = float(scored[TARGET_COLUMN].iloc[offset])
        predictions.append(
            {
                "candidate_id": candidate_id,
                "feature_group": feature_group,
                "model_profile": model_profile,
                "horizon_hours": HORIZON,
                "probability_threshold": probability_threshold,
                "fold_index": fold_index,
                "fold_id": fold_id,
                "timestamp_utc": utc_text(entry),
                "predicted_probability_up_over_cost": float(probability),
                "realized_forward_return": gross_return,
                "realized_positive_event": bool(scored[LABEL_COLUMN].iloc[offset]),
                "trade_opportunity": bool(opportunity),
                "past_volatility_24": current_volatility,
                "training_volatility_cutoff_q75": volatility_threshold,
                "high_volatility_regime": bool(high_volatility),
                "trade_taken": take_trade,
                "round_trip_cost_fraction": ROUND_TRIP_COST,
            }
        )
        if take_trade:
            net_return = gross_return - ROUND_TRIP_COST
            trades.append(
                {
                    "candidate_id": candidate_id,
                    "feature_group": feature_group,
                    "model_profile": model_profile,
                    "horizon_hours": HORIZON,
                    "probability_threshold": probability_threshold,
                    "fold_index": fold_index,
                    "fold_id": fold_id,
                    "entry_utc": utc_text(entry),
                    "exit_utc": utc_text(entry + pd.Timedelta(hours=HORIZON)),
                    "predicted_probability_up_over_cost": float(probability),
                    "past_volatility_24": current_volatility,
                    "training_volatility_cutoff_q75": volatility_threshold,
                    "gross_return": gross_return,
                    "net_return": float(net_return),
                    "won_after_costs": bool(net_return > 0.0),
                    "order_submitted": False,
                }
            )
    return predictions, trades


def _evaluate_candidate_fold(
    *,
    candidate_id: str,
    feature_group: str,
    feature_columns: list[str],
    model_profile: str,
    probability_threshold: float,
    fold_index: int,
    fold: Any,
    dataset: DataFrame,
) -> dict[str, Any]:
    train = dataset.iloc[: fold.train_stop]
    test = dataset.iloc[fold.test_start : fold.test_stop]
    if train.empty or test.empty:
        raise ValueError(f"classification_fold_empty:{fold.fold_id}")
    volatility_threshold = volatility_cutoff(
        train[VOLATILITY_COLUMN].to_numpy(dtype=float)
    )
    model = _fit_classifier(model_profile, train, feature_columns)
    terminal_fold = fold.test_stop == len(dataset)
    scored_count = len(test) if terminal_fold else len(test) - HORIZON
    scored = test.iloc[:scored_count]
    if scored.empty:
        raise ValueError(f"classification_fold_no_oos_rows:{fold.fold_id}")
    probabilities_matrix = model.predict_proba(scored.loc[:, feature_columns])
    class_index = list(model.classes_).index(1)
    probabilities = np.asarray(probabilities_matrix[:, class_index], dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"classification_oos_probabilities_invalid:{fold.fold_id}")
    predictions, trades = _build_prediction_and_trade_rows(
        candidate_id=candidate_id,
        feature_group=feature_group,
        model_profile=model_profile,
        probability_threshold=probability_threshold,
        fold_index=fold_index,
        fold_id=fold.fold_id,
        scored=scored,
        test=test,
        probabilities=probabilities,
        volatility_threshold=volatility_threshold,
        terminal_fold=terminal_fold,
    )
    opportunities = [row for row in predictions if row["trade_opportunity"]]
    high_volatility = [row for row in opportunities if row["high_volatility_regime"]]
    fold_report = {
        "fold_index": fold_index,
        "fold_id": fold.fold_id,
        "train_rows": len(train),
        "test_rows": len(test),
        "scored_oos_rows": len(scored),
        "purged_label_rows": HORIZON,
        "train_start_utc": utc_text(train["date"].iloc[0]),
        "train_end_utc": utc_text(train["date"].iloc[-1]),
        "test_start_utc": utc_text(test["date"].iloc[0]),
        "test_end_utc": utc_text(test["date"].iloc[-1]),
        "positive_event_rate_train": float(train[LABEL_COLUMN].mean()),
        "positive_event_rate_test": float(scored[LABEL_COLUMN].mean()),
        "probability_p50": float(np.quantile(probabilities, 0.50)),
        "probability_p90": float(np.quantile(probabilities, 0.90)),
        "training_volatility_cutoff_q75": volatility_threshold,
        "entry_opportunity_count": len(opportunities),
        "high_volatility_opportunity_count": len(high_volatility),
        "trade_count": len(trades),
        **_trade_summary(trades),
    }
    return {"fold_report": fold_report, "predictions": predictions, "trades": trades}


def _candidate_id(feature_group: str, model_profile: str, threshold: float) -> str:
    return f"{feature_group}|{model_profile}|p{threshold:.2f}"


def _evaluate_candidate(
    *,
    feature_group: str,
    feature_columns: list[str],
    model_profile: str,
    probability_threshold: float,
    dataset: DataFrame,
) -> dict[str, Any]:
    candidate_id = _candidate_id(feature_group, model_profile, probability_threshold)
    folds = build_fixed_window_folds(
        stop_index=len(dataset),
        initial_train_rows=A_TRAIN,
        horizon=HORIZON,
        window_rows=WINDOW,
        prefix=f"CLS-{feature_group}-{model_profile}-P{int(probability_threshold * 100):02d}",
    )
    fold_reports: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        result = _evaluate_candidate_fold(
            candidate_id=candidate_id,
            feature_group=feature_group,
            feature_columns=feature_columns,
            model_profile=model_profile,
            probability_threshold=probability_threshold,
            fold_index=index,
            fold=fold,
            dataset=dataset,
        )
        fold_reports.append(result["fold_report"])
        predictions.extend(result["predictions"])
        trades.extend(result["trades"])
    return {
        "candidate_id": candidate_id,
        "configuration": {
            "feature_group": feature_group,
            "feature_columns": feature_columns,
            "model_profile": model_profile,
            "model_parameters": dict(CLASSIFIER_BASE_PARAMETERS),
            "class_weight_rule": (
                "scale_pos_weight = negative / positive in current training fold"
                if model_profile == "balanced"
                else "unweighted binary log loss"
            ),
            "probability_threshold": probability_threshold,
            "volatility_filter_quantile": VOLATILITY_CUTOFF_QUANTILE,
            "horizon_hours": HORIZON,
            "positive_label": "12-hour gross long return strictly greater than 0.60%",
            "round_trip_cost_fraction": ROUND_TRIP_COST,
        },
        "folds": fold_reports,
        "predictions": predictions,
        "trades": trades,
    }


def _candidate_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["predictions"]
    opportunities = [row for row in rows if row["trade_opportunity"]]
    high_volatility = [row for row in opportunities if row["high_volatility_regime"]]
    probabilities = [float(row["predicted_probability_up_over_cost"]) for row in rows]
    return {
        "candidate_id": result["candidate_id"],
        "configuration": result["configuration"],
        "evaluated_fold_count": len(result["folds"]),
        "oos_prediction_count": len(rows),
        "entry_opportunity_count": len(opportunities),
        "high_volatility_opportunity_count": len(high_volatility),
        "high_volatility_opportunity_rate": (
            float(len(high_volatility) / len(opportunities)) if opportunities else None
        ),
        "probability_p50": float(np.quantile(probabilities, 0.50))
        if probabilities
        else None,
        "probability_p90": float(np.quantile(probabilities, 0.90))
        if probabilities
        else None,
        "positive_event_rate_oos": (
            float(np.mean([bool(row["realized_positive_event"]) for row in rows]))
            if rows
            else None
        ),
        **_trade_summary(result["trades"]),
    }


def _prior_candidate_score(
    trades: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, int], dict[str, Any]] | None:
    metrics = _trade_summary(trades)
    drawdown = metrics["max_drawdown_abs"]
    if (
        int(metrics["trade_count"]) < MIN_PRIOR_TRADES_FOR_SELECTION
        or metrics["win_rate"] is None
        or metrics["win_rate"] <= 0.60
        or metrics["net_expectancy_per_trade"] is None
        or metrics["net_expectancy_per_trade"] <= 0.0
        or drawdown is None
        or drawdown >= 0.15
    ):
        return None
    score = (
        float(metrics["net_expectancy_per_trade"]),
        float(metrics["win_rate"]),
        -float(drawdown),
        int(metrics["trade_count"]),
    )
    return score, metrics


def _select_policy_fold(
    candidate_results: dict[str, dict[str, Any]], fold_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible: list[tuple[tuple[float, float, float, int], str, int]] = []
    for candidate_id, result in sorted(candidate_results.items()):
        prior = [row for row in result["trades"] if int(row["fold_index"]) < fold_index]
        score = _prior_candidate_score(prior)
        if score is not None:
            ranking, metrics = score
            eligible.append((ranking, candidate_id, int(metrics["trade_count"])))
    if eligible:
        _, selected_id, prior_count = max(eligible, key=lambda item: item[0])
        reason = "selected_using_prior_oos_trades_only"
    else:
        selected_id = BASELINE_CANDIDATE_ID
        prior_count = 0
        reason = "fallback_no_prior_candidate_passed_selection_gates"
    selected = candidate_results[selected_id]
    current = [
        row for row in selected["trades"] if int(row["fold_index"]) == fold_index
    ]
    trace = {
        "fold_index": fold_index,
        "selected_candidate_id": selected_id,
        "selection_reason": reason,
        "prior_selected_trades_used": prior_count,
        "current_fold_results_used_for_selection": False,
        "current_fold_trade_count": len(current),
    }
    return trace, current


def _evaluate_adaptive_policy(
    candidate_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = candidate_results[BASELINE_CANDIDATE_ID]
    fold_count = len(baseline["folds"])
    if any(len(value["folds"]) != fold_count for value in candidate_results.values()):
        raise ValueError("classification_candidate_fold_count_mismatch")
    traces: list[dict[str, Any]] = []
    selected_trades: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        trace, trades = _select_policy_fold(candidate_results, fold_index)
        traces.append(trace)
        selected_trades.extend(trades)
    policy = {
        "selection_rule": (
            "Select from prior A-development OOS trades only; require >=30 prior "
            "trades, win rate >60%, positive net expectancy and drawdown <15%; "
            "otherwise fall back to the fixed unweighted p0.50 baseline."
        ),
        "prior_oos_only_selection": True,
        "minimum_prior_trades": MIN_PRIOR_TRADES_FOR_SELECTION,
        "fallback_candidate_id": BASELINE_CANDIDATE_ID,
        "selected_fold_count": fold_count,
        "fallback_fold_count": sum(
            "fallback_" in str(row["selection_reason"]) for row in traces
        ),
        **_trade_summary(selected_trades),
    }
    return policy, traces, selected_trades


def _flatten_candidate_results(
    candidate_results: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries = [_candidate_summary(value) for value in candidate_results.values()]
    predictions = [
        row for value in candidate_results.values() for row in value["predictions"]
    ]
    trades = [row for value in candidate_results.values() for row in value["trades"]]
    folds = [
        {
            "candidate_id": value["candidate_id"],
            "feature_group": value["configuration"]["feature_group"],
            "model_profile": value["configuration"]["model_profile"],
            "probability_threshold": value["configuration"]["probability_threshold"],
            **fold,
        }
        for value in candidate_results.values()
        for fold in value["folds"]
    ]
    return summaries, predictions, trades, folds


def _evaluate_grid(
    datasets: dict[str, DataFrame],
    selected_features: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for group, profile, threshold in product(
        FEATURE_GROUPS, MODEL_PROFILES, PROBABILITY_THRESHOLDS
    ):
        results[_candidate_id(group, profile, threshold)] = _evaluate_candidate(
            feature_group=group,
            feature_columns=selected_features[group],
            model_profile=profile,
            probability_threshold=threshold,
            dataset=datasets[group],
        )
    return results


def _prepare_context(input_dir: Path) -> dict[str, Any]:
    frame, source_report, segments, segment_hashes = verify_source_artifact(input_dir)
    segment_a = segments[0]
    if len(segment_a) <= A_HOLDOUT + A_TRAIN + WINDOW:
        raise ValueError("classification_segment_a_too_short")
    holdout_start = pd.Timestamp(segment_a["date"].iloc[-A_HOLDOUT])
    development = segment_a.iloc[:-A_HOLDOUT].copy().reset_index(drop=True)
    del frame, segments, segment_a
    validated, quality_fingerprint = validate_ohlcv(
        development.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy(),
        HOUR,
    )
    engineered = add_signal_features(validated)
    diagnostics = feature_availability_diagnostics(
        engineered, tuple(FEATURE_COLUMNS) + NEW_FEATURE_COLUMNS
    )
    selected, excluded = _candidate_features(engineered)
    start_utc, end_utc = _common_evaluation_bounds(
        engineered, selected["extended_trend_rsi"]
    )
    datasets, shared_rows = _build_candidate_datasets(
        engineered, selected, start_utc, end_utc
    )
    return {
        "source_report": source_report,
        "segment_hashes": segment_hashes,
        "holdout_start": holdout_start,
        "development": development,
        "quality_fingerprint": quality_fingerprint,
        "feature_diagnostics": diagnostics,
        "selected_features": selected,
        "excluded_features": excluded,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "datasets": datasets,
        "shared_row_count": shared_rows,
    }


def _build_report(
    commit_sha: str,
    context: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    summaries: list[dict[str, Any]],
    policy: dict[str, Any],
    policy_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    development = context["development"]
    source_report = context["source_report"]
    features = context["selected_features"]
    baseline = _candidate_summary(candidate_results[BASELINE_CANDIDATE_ID])
    return {
        "report_schema": REPORT_SCHEMA,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": context["segment_hashes"][0],
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "full_segment_a_input_rows": len(development) + A_HOLDOUT,
            "development_input_rows": len(development),
            "holdout_rows_reserved": A_HOLDOUT,
            "development_start_utc": utc_text(development["date"].iloc[0]),
            "development_end_utc": utc_text(development["date"].iloc[-1]),
            "reserved_holdout_start_utc": utc_text(context["holdout_start"]),
            "common_candidate_oos_start_utc": utc_text(context["start_utc"]),
            "common_candidate_oos_end_utc": utc_text(context["end_utc"]),
            "data_quality_gate_fingerprint": context["quality_fingerprint"],
        },
        "target": {
            "label_column": LABEL_COLUMN,
            "horizon_hours": HORIZON,
            "positive_class": (
                "forward close-to-close gross long return over 12 hours strictly "
                "greater than 0.006 (0.60%); downward large moves are not positive buy labels"
            ),
            "round_trip_cost_fraction": ROUND_TRIP_COST,
            "absolute_move_magnitude_used_as_buy_label": False,
        },
        "protocol": {
            "evaluation_split": "Stage A segment A development only",
            "holdout_evaluated": False,
            "holdout_used_for_feature_or_parameter_selection": False,
            "horizon_hours": HORIZON,
            "minimum_training_rows": A_TRAIN,
            "oos_window_hours": WINDOW,
            "walk_forward": "expanding, chronological, fresh classifier per fold, 12-hour label purge",
            "candidate_count": len(candidate_results),
            "feature_groups": {
                group: {
                    "selected_columns": features[group],
                    "excluded_features_without_imputation": context[
                        "excluded_features"
                    ][group],
                }
                for group in FEATURE_GROUPS
            },
            "feature_availability_diagnostics": context["feature_diagnostics"],
            "candidate_timestamp_alignment": True,
            "common_candidate_row_count": context["shared_row_count"],
            "probability_threshold_grid": list(PROBABILITY_THRESHOLDS),
            "probability_thresholds_selected_using": (
                "prior A-development OOS folds only; whole-grid aggregate ranking is descriptive"
            ),
            "probability_thresholds_optimized_on_holdout": False,
            "volatility_filter": {
                "feature": "%-volatility_24",
                "method": "current closed-candle volatility >= 75th percentile of the current training fold's volatility",
                "quantile": VOLATILITY_CUTOFF_QUANTILE,
                "cutoff_recomputed_per_fold_from_train_only": True,
                "uses_future_or_test_volatility_to_set_cutoff": False,
                "low_volatility_entry_allowed": False,
            },
            "feature_imputation": False,
            "forward_fill": False,
            "backfill": False,
            "interpolation": False,
            "resampling": False,
            "cost_hurdle": ROUND_TRIP_COST,
            "cost_hurdle_optimized": False,
            "fee_bps_per_side": FEE_BPS_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_SIDE,
            "class_profiles": {
                "unweighted": "fixed binary LightGBM loss with no class reweighting",
                "balanced": "scale_pos_weight = negative / positive in each training fold",
            },
            "adaptive_selection": (
                "Prior OOS trades only; >=30 prior trades, win rate >60%, positive net "
                "expectancy and drawdown <15%; otherwise fixed unweighted p0.50 baseline"
            ),
            "candidate_aggregate_ranking_used_for_adaptive_selection": False,
        },
        "fixed_baseline_candidate": baseline,
        "candidate_diagnostics": summaries,
        "adaptive_walkforward_policy": policy,
        "adaptive_policy_selection_trace": policy_trace,
        "strict_economic_acceptance": {
            "criteria": {
                "oos_trade_count": ">= 100",
                "win_rate": "> 60%",
                "net_expectancy_per_trade": "> 0 after modeled costs",
                "absolute_max_drawdown": "< 15%",
            },
            "threshold_relaxation": False,
            "aggregate_across_candidates_for_acceptance": False,
            "adaptive_policy": policy["economic_threshold_evaluation"],
        },
        "safety_invariants": dict(CLOSED_FLAGS),
        "limitations": [
            "This is research-only classification, not a production model or exchange execution backtest.",
            "The positive class is long-directional and is not an absolute-magnitude volatility event.",
            "A 100-trade / >60% win-rate result is still subject to positive expectancy and <15% drawdown gates.",
            "The final 90-day A holdout is removed before feature engineering and target creation.",
            "The data has no positive-volume rows in segment A; volume-derived features are not fabricated.",
            "No model promotion, order submission, or private API call is permitted by this experiment.",
        ],
    }


def _write_evidence(
    output_dir: Path,
    context: dict[str, Any],
    report: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    policy_trace: list[dict[str, Any]],
    policy_trades: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "classification-report.json", report)
    _write_json(
        output_dir / "input-verification.json",
        {
            "report_schema": REPORT_SCHEMA,
            "source_csv_sha256": context["source_report"]["csv_sha256"],
            "segment_a_rows_sha256": context["segment_hashes"][0],
            "development_rows": len(context["development"]),
            "holdout_rows_reserved": A_HOLDOUT,
            "holdout_evaluated": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    )
    prediction_columns = [
        "candidate_id",
        "feature_group",
        "model_profile",
        "horizon_hours",
        "probability_threshold",
        "fold_index",
        "fold_id",
        "timestamp_utc",
        "predicted_probability_up_over_cost",
        "realized_forward_return",
        "realized_positive_event",
        "trade_opportunity",
        "past_volatility_24",
        "training_volatility_cutoff_q75",
        "high_volatility_regime",
        "trade_taken",
        "round_trip_cost_fraction",
    ]
    trade_columns = [
        "candidate_id",
        "feature_group",
        "model_profile",
        "horizon_hours",
        "probability_threshold",
        "fold_index",
        "fold_id",
        "entry_utc",
        "exit_utc",
        "predicted_probability_up_over_cost",
        "past_volatility_24",
        "training_volatility_cutoff_q75",
        "gross_return",
        "net_return",
        "won_after_costs",
        "order_submitted",
    ]
    fold_columns = [
        "candidate_id",
        "feature_group",
        "model_profile",
        "probability_threshold",
        "fold_index",
        "fold_id",
        "train_rows",
        "test_rows",
        "scored_oos_rows",
        "purged_label_rows",
        "train_start_utc",
        "train_end_utc",
        "test_start_utc",
        "test_end_utc",
        "positive_event_rate_train",
        "positive_event_rate_test",
        "probability_p50",
        "probability_p90",
        "training_volatility_cutoff_q75",
        "entry_opportunity_count",
        "high_volatility_opportunity_count",
        "trade_count",
        "winning_trade_count",
        "losing_or_flat_trade_count",
        "win_rate",
        "net_expectancy_per_trade",
        "net_return_compounded",
        "max_drawdown_abs",
        "economic_threshold_evaluation",
    ]
    trace_columns = [
        "fold_index",
        "selected_candidate_id",
        "selection_reason",
        "prior_selected_trades_used",
        "current_fold_results_used_for_selection",
        "current_fold_trade_count",
    ]
    _write_csv(output_dir / "oos-predictions.csv", prediction_rows, prediction_columns)
    _write_csv(output_dir / "oos-trades.csv", trade_rows, trade_columns)
    _write_csv(output_dir / "candidate-fold-metrics.csv", fold_rows, fold_columns)
    _write_csv(output_dir / "adaptive-policy-trace.csv", policy_trace, trace_columns)
    _write_csv(output_dir / "adaptive-policy-trades.csv", policy_trades, trade_columns)


def run_classification_experiment(
    input_dir: Path, output_dir: Path, commit_sha: str
) -> dict[str, Any]:
    _validate_commit_sha(commit_sha)
    context = _prepare_context(input_dir)
    candidate_results = _evaluate_grid(
        context["datasets"], context["selected_features"]
    )
    policy, policy_trace, policy_trades = _evaluate_adaptive_policy(candidate_results)
    summaries, predictions, trades, folds = _flatten_candidate_results(
        candidate_results
    )
    report = _build_report(
        commit_sha, context, candidate_results, summaries, policy, policy_trace
    )
    if report["safety_invariants"] != CLOSED_FLAGS:
        raise ValueError("classification_closed_release_contract_failed")
    if report["target"]["round_trip_cost_fraction"] != 0.006:
        raise ValueError("classification_cost_hurdle_changed")
    if report["protocol"]["candidate_count"] != 12:
        raise ValueError("classification_candidate_count_contract_failed")
    _write_evidence(
        output_dir,
        context,
        report,
        predictions,
        trades,
        folds,
        policy_trace,
        policy_trades,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    report = run_classification_experiment(
        args.input_dir, args.output_dir, args.commit_sha
    )
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "created_by_commit_sha": report["created_by_commit_sha"],
                "candidate_count": report["protocol"]["candidate_count"],
                "baseline": report["fixed_baseline_candidate"],
                "adaptive_policy": report["adaptive_walkforward_policy"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
                "trading_authorization": report["safety_invariants"][
                    "trading_authorization"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
