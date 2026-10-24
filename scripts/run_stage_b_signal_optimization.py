"""Research-only walk-forward diagnostics for sparse Stage B signals.

This runner evaluates Stage A segment A development data only. It never reads
the reserved final 90-day holdout into feature engineering, training, selection,
or scoring. Candidate summaries are exploratory; only the adaptive policy uses
strictly prior OOS folds for candidate selection.
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
from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_market_features,
    forward_return_target,
)
from services.freqai_baseline.walkforward import (
    BASELINE_MODEL_PARAMETERS,
    validate_ohlcv,
)

REPORT_SCHEMA = "alix.stage-b.signal-optimization.v1"
HORIZON_GRID = (12, 6, 24)
FEATURE_GROUPS = ("stage_b_8", "extended_trend_rsi")
NEW_FEATURE_COLUMNS: tuple[str, ...] = (
    "%-rsi_14",
    "%-close_vs_sma_50",
    "%-close_vs_sma_200",
    "%-momentum_24",
    "%-volatility_24",
    "%-range_position_24",
)
MIN_PRIOR_TRADES_FOR_SELECTION = 30
MIN_OOS_TRADES_FOR_ECONOMIC_EVALUATION = 100
FEE_BPS_SIDE = 25.0
SLIPPAGE_BPS_SIDE = 5.0
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
MODEL_PROFILES: dict[str, dict[str, object]] = {
    "phase4_default": dict(BASELINE_MODEL_PARAMETERS),
    "shallow_regularized": {
        **BASELINE_MODEL_PARAMETERS,
        "n_estimators": 120,
        "learning_rate": 0.02,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 100,
        "reg_lambda": 2.0,
    },
}
BASELINE_CANDIDATE_ID = "stage_b_8|phase4_default|h12"


def _build_candidate_datasets(
    engineered: DataFrame,
    selected_features: dict[str, list[str]],
    common_start_utc: pd.Timestamp,
    common_end_utc: pd.Timestamp,
) -> tuple[dict[tuple[str, int], DataFrame], int]:
    """Build and verify the exact common timestamp grid for all configurations."""
    datasets: dict[tuple[str, int], DataFrame] = {}
    reference_timestamps: np.ndarray | None = None
    reference_row_count: int | None = None
    for feature_group in FEATURE_GROUPS:
        columns = selected_features[feature_group]
        for horizon in HORIZON_GRID:
            dataset = _build_dataset(
                engineered, columns, horizon, common_start_utc, common_end_utc
            )
            timestamps = (
                pd.to_datetime(dataset["date"], utc=True, errors="raise")
                .astype("int64")
                .to_numpy(dtype=np.int64)
            )
            if reference_timestamps is None:
                reference_timestamps = timestamps
                reference_row_count = len(dataset)
            elif not np.array_equal(timestamps, reference_timestamps):
                raise ValueError("optimization_candidate_timestamp_grid_mismatch")
            if len(dataset) != reference_row_count:
                raise ValueError("optimization_candidate_row_count_mismatch")
            datasets[(feature_group, horizon)] = dataset
    if reference_row_count is None:
        raise ValueError("optimization_candidate_grid_empty")
    return datasets, reference_row_count


def _evaluate_candidate_grid(
    datasets: dict[tuple[str, int], DataFrame],
    selected_features: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Evaluate the fixed grid of feature groups, horizons, and model profiles."""
    results: dict[str, dict[str, Any]] = {}
    for feature_group, horizon, model_profile in product(
        FEATURE_GROUPS, HORIZON_GRID, ("phase4_default", "shallow_regularized")
    ):
        candidate_id = _candidate_id(feature_group, model_profile, horizon)
        results[candidate_id] = _evaluate_candidate(
            candidate_id=candidate_id,
            feature_group=feature_group,
            feature_columns=selected_features[feature_group],
            model_profile=model_profile,
            horizon=horizon,
            dataset=datasets[(feature_group, horizon)],
        )
    return results


def _collect_candidate_rows(
    candidate_results: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Flatten candidate evidence into report summaries and reproducible CSV rows."""
    summaries = [_candidate_summary(result) for result in candidate_results.values()]
    predictions = [
        row for result in candidate_results.values() for row in result["predictions"]
    ]
    trades = [row for result in candidate_results.values() for row in result["trades"]]
    folds = [
        {
            "candidate_id": result["candidate_id"],
            "feature_group": result["configuration"]["feature_group"],
            "model_profile": result["configuration"]["model_profile"],
            "horizon_hours": result["configuration"]["horizon_hours"],
            **fold,
        }
        for result in candidate_results.values()
        for fold in result["folds"]
    ]
    return summaries, predictions, trades, folds



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
        raise ValueError("optimization_commit_sha_invalid")


def add_signal_features(candles: DataFrame) -> DataFrame:
    """Add past-only trend/momentum indicators; never fill undefined values."""
    result = add_market_features(candles)
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    returns = close.pct_change(fill_method=None)

    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    average_gain = gains.rolling(14, min_periods=14).mean()
    average_loss = losses.rolling(14, min_periods=14).mean()
    rsi_denominator = average_gain + average_loss
    rsi = 100.0 * average_gain / rsi_denominator.where(rsi_denominator > 0.0)
    # A fully flat 14-bar window has a defined neutral RSI by convention.
    result["%-rsi_14"] = rsi.mask(rsi_denominator.eq(0.0), 50.0)

    for window in (50, 200):
        rolling_mean = close.rolling(window, min_periods=window).mean()
        result[f"%-close_vs_sma_{window}"] = close / rolling_mean - 1.0

    result["%-momentum_24"] = close.pct_change(24, fill_method=None)
    result["%-volatility_24"] = returns.rolling(24, min_periods=24).std()

    rolling_high = high.rolling(24, min_periods=24).max()
    rolling_low = low.rolling(24, min_periods=24).min()
    range_width = rolling_high - rolling_low
    range_position = (close - rolling_low) / range_width.where(range_width > 0.0)
    flat_range = range_width.eq(0.0) & close.eq(rolling_low)
    # When the complete rolling range is flat, the close is at the neutral midpoint.
    result["%-range_position_24"] = range_position.mask(flat_range, 0.5)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def _candidate_features(
    engineered: DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, str]]]]:
    all_candidates = tuple(FEATURE_COLUMNS) + NEW_FEATURE_COLUMNS
    selected_baseline, excluded_baseline = select_segment_features(
        engineered, FEATURE_COLUMNS
    )
    selected_extended, excluded_extended = select_segment_features(
        engineered, all_candidates
    )
    selected = {
        "stage_b_8": selected_baseline,
        "extended_trend_rsi": selected_extended,
    }
    excluded = {
        "stage_b_8": excluded_baseline,
        "extended_trend_rsi": excluded_extended,
    }
    for group, columns in selected.items():
        if not columns:
            raise ValueError(f"optimization_feature_group_empty:{group}")
        if len(set(columns)) != len(columns):
            raise ValueError(f"optimization_feature_group_has_duplicates:{group}")
    return selected, excluded


def _common_evaluation_bounds(
    engineered: DataFrame,
    feature_columns: Sequence[str],
    max_horizon: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find one contiguous A-development time range shared by every candidate."""
    if max_horizon < 1:
        raise ValueError("optimization_max_horizon_invalid")
    target = forward_return_target(engineered, horizon=max_horizon)
    numeric = engineered.loc[:, list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    valid = numeric.notna().all(axis=1) & target.notna() & np.isfinite(target)
    positions = np.flatnonzero(valid.to_numpy(dtype=bool))
    if positions.size == 0:
        raise ValueError("optimization_common_evaluation_range_empty")
    first = int(positions[0])
    last = int(positions[-1])
    if not bool(valid.iloc[first : last + 1].all()):
        raise ValueError("optimization_common_evaluation_range_has_internal_gap")
    dates = pd.to_datetime(engineered["date"], utc=True, errors="raise")
    common_dates = dates.iloc[first : last + 1]
    if not (common_dates.diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise ValueError("optimization_common_evaluation_grid_not_contiguous")
    return pd.Timestamp(common_dates.iloc[0]), pd.Timestamp(common_dates.iloc[-1])


def _build_dataset(
    engineered: DataFrame,
    feature_columns: Sequence[str],
    horizon: int,
    common_start_utc: pd.Timestamp,
    common_end_utc: pd.Timestamp,
) -> DataFrame:
    """Build rows inside the common candidate grid without imputing values."""
    if horizon < 1 or common_start_utc > common_end_utc:
        raise ValueError("optimization_dataset_bounds_invalid")
    target = forward_return_target(engineered, horizon=horizon)
    numeric = engineered.loc[:, list(feature_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    dates = pd.to_datetime(engineered["date"], utc=True, errors="raise")
    in_common_range = dates.ge(common_start_utc) & dates.le(common_end_utc)
    valid = (
        numeric.notna().all(axis=1)
        & target.notna()
        & np.isfinite(target)
        & in_common_range
    )
    selected = engineered.loc[valid, ["date", "close", *feature_columns]].copy()
    selected[TARGET_COLUMN] = target.loc[valid].to_numpy(dtype=float)
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError("optimization_dataset_empty")
    selected_dates = pd.to_datetime(selected["date"], utc=True, errors="raise")
    if (
        selected_dates.iloc[0] != common_start_utc
        or selected_dates.iloc[-1] != common_end_utc
    ):
        raise ValueError("optimization_candidate_does_not_cover_common_bounds")
    if not (selected_dates.diff().dropna() == pd.Timedelta(seconds=HOUR)).all():
        raise ValueError("optimization_valid_rows_not_contiguous")
    feature_values = selected.loc[:, list(feature_columns)].to_numpy(dtype=float)
    labels = selected[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all() or not np.isfinite(labels).all():
        raise ValueError("optimization_dataset_has_non_finite_values")
    return selected


def summarize_cost_hurdle(
    forecasts: Sequence[float],
    opportunities: Sequence[bool],
    cost_fraction: float = ROUND_TRIP_COST,
) -> dict[str, int | float | None]:
    """Quantify signal sparsity at the unchanged round-trip-cost hurdle."""
    if len(forecasts) != len(opportunities):
        raise ValueError("optimization_forecast_opportunity_length_mismatch")
    values = np.asarray(forecasts, dtype=float)
    opportunity_mask = np.asarray(opportunities, dtype=bool)
    if not np.isfinite(values).all():
        raise ValueError("optimization_forecasts_non_finite")
    eligible = values[opportunity_mask]
    if eligible.size == 0:
        return {
            "prediction_count": int(values.size),
            "entry_opportunity_count": 0,
            "forecasts_above_cost": 0,
            "take_rate_per_opportunity": None,
            "opportunity_forecast_p50": None,
            "opportunity_forecast_p90": None,
            "opportunity_forecast_p95": None,
            "opportunity_forecast_p99": None,
            "opportunity_forecast_max": None,
        }
    selected_count = int(np.sum(eligible > cost_fraction))
    return {
        "prediction_count": int(values.size),
        "entry_opportunity_count": int(eligible.size),
        "forecasts_above_cost": selected_count,
        "take_rate_per_opportunity": float(selected_count / eligible.size),
        "opportunity_forecast_p50": float(np.quantile(eligible, 0.50)),
        "opportunity_forecast_p90": float(np.quantile(eligible, 0.90)),
        "opportunity_forecast_p95": float(np.quantile(eligible, 0.95)),
        "opportunity_forecast_p99": float(np.quantile(eligible, 0.99)),
        "opportunity_forecast_max": float(np.max(eligible)),
    }


def _drawdown_abs(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("optimization_trade_return_invalid")
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
        "authorization_effect": ("NONE; Champion remains null and trading stays NO-GO"),
    }


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(trade["net_return"]) for trade in trades]
    count = len(returns)
    wins = sum(value > 0.0 for value in returns)
    win_rate = wins / count if count else None
    expectancy = float(np.mean(returns)) if count else None
    drawdown = _drawdown_abs(returns)
    return {
        "trade_count": count,
        "winning_trade_count": int(wins),
        "losing_or_flat_trade_count": int(count - wins),
        "win_rate": float(win_rate) if win_rate is not None else None,
        "net_expectancy_per_trade": expectancy,
        "net_return_compounded": (
            float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)
            if returns
            else 0.0
        ),
        "max_drawdown_abs": drawdown,
        "economic_threshold_evaluation": _economic_gate(
            count, win_rate, expectancy, drawdown
        ),
    }


def _make_model(profile: str) -> Any:
    from lightgbm import LGBMRegressor  # type: ignore[import-not-found]

    if profile not in MODEL_PROFILES:
        raise ValueError(f"optimization_model_profile_invalid:{profile}")
    return LGBMRegressor(**dict(MODEL_PROFILES[profile]))


def _candidate_id(group: str, profile: str, horizon: int) -> str:
    return f"{group}|{profile}|h{horizon}"


def _build_fold_rows(
    *,
    candidate_id: str,
    feature_group: str,
    model_profile: str,
    horizon: int,
    fold_index: int,
    fold_id: str,
    scored: DataFrame,
    test: DataFrame,
    forecasts: np.ndarray,
    realized: np.ndarray,
    terminal_fold: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize forecast rows and cost-qualified paper trades for one fold."""
    predictions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for offset, (forecast, outcome) in enumerate(
        zip(forecasts.tolist(), realized.tolist(), strict=True)
    ):
        entry = pd.Timestamp(scored["date"].iloc[offset])
        opportunity = offset % horizon == 0 and (
            terminal_fold or offset + horizon < len(test)
        )
        take_trade = opportunity and forecast > ROUND_TRIP_COST
        prediction = {
            "candidate_id": candidate_id,
            "feature_group": feature_group,
            "model_profile": model_profile,
            "horizon_hours": horizon,
            "fold_index": fold_index,
            "fold_id": fold_id,
            "timestamp_utc": utc_text(entry),
            "forecast_return_horizon": float(forecast),
            "realized_forward_return": float(outcome),
            "round_trip_cost_fraction": ROUND_TRIP_COST,
            "trade_opportunity": bool(opportunity),
            "trade_taken": bool(take_trade),
        }
        predictions.append(prediction)
        if take_trade:
            trades.append(
                {
                    "candidate_id": candidate_id,
                    "feature_group": feature_group,
                    "model_profile": model_profile,
                    "horizon_hours": horizon,
                    "fold_index": fold_index,
                    "fold_id": fold_id,
                    "entry_utc": utc_text(entry),
                    "exit_utc": utc_text(entry + pd.Timedelta(hours=horizon)),
                    "forecast_return": float(forecast),
                    "gross_return": float(outcome),
                    "fee_bps_round_trip": 2.0 * FEE_BPS_SIDE,
                    "slippage_bps_round_trip": 2.0 * SLIPPAGE_BPS_SIDE,
                    "net_return": float(outcome - ROUND_TRIP_COST),
                    "won_after_costs": float(outcome - ROUND_TRIP_COST) > 0.0,
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
    horizon: int,
    fold_index: int,
    fold: Any,
    dataset: DataFrame,
) -> dict[str, Any]:
    """Fit and score one purged OOS fold, then return its audit evidence."""
    train = dataset.iloc[: fold.train_stop]
    test = dataset.iloc[fold.test_start : fold.test_stop]
    if train.empty or test.empty:
        raise ValueError(f"optimization_fold_empty:{fold.fold_id}")
    model = _make_model(model_profile)
    model.fit(
        train.loc[:, feature_columns],
        train[TARGET_COLUMN].to_numpy(dtype=float),
    )
    terminal_fold = fold.test_stop == len(dataset)
    scored_count = len(test) if terminal_fold else len(test) - horizon
    scored = test.iloc[:scored_count]
    if scored.empty:
        raise ValueError(f"optimization_fold_has_no_oos_rows:{fold.fold_id}")
    forecasts = np.asarray(model.predict(scored.loc[:, feature_columns]), dtype=float)
    realized = scored[TARGET_COLUMN].to_numpy(dtype=float)
    if (
        len(forecasts) != len(realized)
        or not np.isfinite(forecasts).all()
        or not np.isfinite(realized).all()
    ):
        raise ValueError(f"optimization_oos_values_invalid:{fold.fold_id}")
    predictions, trades = _build_fold_rows(
        candidate_id=candidate_id,
        feature_group=feature_group,
        model_profile=model_profile,
        horizon=horizon,
        fold_index=fold_index,
        fold_id=fold.fold_id,
        scored=scored,
        test=test,
        forecasts=forecasts,
        realized=realized,
        terminal_fold=terminal_fold,
    )
    hurdle = summarize_cost_hurdle(
        [float(row["forecast_return_horizon"]) for row in predictions],
        [bool(row["trade_opportunity"]) for row in predictions],
    )
    fold_report = {
        "fold_index": fold_index,
        "fold_id": fold.fold_id,
        "train_rows": len(train),
        "test_rows": len(test),
        "scored_oos_rows": len(scored),
        "purged_label_rows": horizon,
        "train_start_utc": utc_text(train["date"].iloc[0]),
        "train_end_utc": utc_text(train["date"].iloc[-1]),
        "test_start_utc": utc_text(test["date"].iloc[0]),
        "test_end_utc": utc_text(test["date"].iloc[-1]),
        **hurdle,
        **_trade_summary(trades),
    }
    return {"fold_report": fold_report, "predictions": predictions, "trades": trades}


def _evaluate_candidate(
    *,
    candidate_id: str,
    feature_group: str,
    feature_columns: list[str],
    model_profile: str,
    horizon: int,
    dataset: DataFrame,
) -> dict[str, Any]:
    """Evaluate one candidate using fresh estimators and expanding OOS folds."""
    folds = build_fixed_window_folds(
        stop_index=len(dataset),
        initial_train_rows=A_TRAIN,
        horizon=horizon,
        window_rows=WINDOW,
        prefix=f"SIG-{feature_group}-{model_profile}-H{horizon}",
    )
    fold_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        fold_result = _evaluate_candidate_fold(
            candidate_id=candidate_id,
            feature_group=feature_group,
            feature_columns=feature_columns,
            model_profile=model_profile,
            horizon=horizon,
            fold_index=fold_index,
            fold=fold,
            dataset=dataset,
        )
        fold_reports.append(fold_result["fold_report"])
        prediction_rows.extend(fold_result["predictions"])
        trades.extend(fold_result["trades"])
    return {
        "candidate_id": candidate_id,
        "configuration": {
            "feature_group": feature_group,
            "feature_columns": feature_columns,
            "model_profile": model_profile,
            "model_parameters": dict(MODEL_PROFILES[model_profile]),
            "horizon_hours": horizon,
            "round_trip_cost_fraction": ROUND_TRIP_COST,
            "cost_hurdle_optimized": False,
        },
        "folds": fold_reports,
        "predictions": prediction_rows,
        "trades": trades,
    }


def _candidate_summary(result: dict[str, Any]) -> dict[str, Any]:
    predictions = result["predictions"]
    trades = result["trades"]
    forecasts = [float(row["forecast_return_horizon"]) for row in predictions]
    opportunities = [bool(row["trade_opportunity"]) for row in predictions]
    return {
        "candidate_id": result["candidate_id"],
        "configuration": result["configuration"],
        "evaluated_fold_count": len(result["folds"]),
        "oos_prediction_count": len(predictions),
        **summarize_cost_hurdle(forecasts, opportunities),
        **_trade_summary(trades),
    }


def _prior_candidate_score(
    trades: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, int], dict[str, Any]] | None:
    metrics = _trade_summary(trades)
    if int(metrics["trade_count"]) < MIN_PRIOR_TRADES_FOR_SELECTION:
        return None
    drawdown = metrics["max_drawdown_abs"]
    if (
        metrics["win_rate"] is None
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


def _eligible_prior_candidates(
    candidate_results: dict[str, dict[str, Any]], fold_index: int
) -> list[tuple[tuple[float, float, float, int], str, int]]:
    """Rank only candidates that pass gates on strictly earlier OOS folds."""
    eligible: list[tuple[tuple[float, float, float, int], str, int]] = []
    for candidate_id, result in sorted(candidate_results.items()):
        prior_trades = [
            trade for trade in result["trades"] if int(trade["fold_index"]) < fold_index
        ]
        selection = _prior_candidate_score(prior_trades)
        if selection is not None:
            score, metrics = selection
            eligible.append((score, candidate_id, int(metrics["trade_count"])))
    return eligible


def _select_and_collect_policy_fold(
    candidate_results: dict[str, dict[str, Any]], fold_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select one configuration from prior folds and collect only current-fold trades."""
    eligible = _eligible_prior_candidates(candidate_results, fold_index)
    if eligible:
        _, chosen_id, prior_count = max(eligible, key=lambda item: item[0])
        reason = "selected_using_prior_oos_trades_only"
    else:
        chosen_id = BASELINE_CANDIDATE_ID
        prior_count = 0
        reason = "fallback_no_prior_candidate_passed_selection_gates"
    chosen = candidate_results[chosen_id]
    current_trades = [
        trade for trade in chosen["trades"] if int(trade["fold_index"]) == fold_index
    ]
    fold_report = chosen["folds"][fold_index]
    trace_row = {
        "fold_index": fold_index,
        "fold_id": fold_report["fold_id"],
        "selected_candidate_id": chosen_id,
        "selection_reason": reason,
        "prior_selected_trades_used": prior_count,
        "current_fold_results_used_for_selection": False,
        "current_fold_trade_count": len(current_trades),
    }
    return trace_row, current_trades


def _evaluate_adaptive_policy(
    candidate_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a fold-by-fold policy using prior OOS evidence only."""
    baseline = candidate_results[BASELINE_CANDIDATE_ID]
    fold_count = len(baseline["folds"])
    if any(len(result["folds"]) != fold_count for result in candidate_results.values()):
        raise ValueError("optimization_candidate_fold_count_mismatch")
    trace: list[dict[str, Any]] = []
    adaptive_trades: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        trace_row, current_trades = _select_and_collect_policy_fold(
            candidate_results, fold_index
        )
        trace.append(trace_row)
        adaptive_trades.extend(current_trades)
    policy = {
        "selection_rule": (
            "Use prior A-development OOS trades only; require >=30 prior trades, "
            "win rate >60%, net expectancy >0 and absolute max drawdown <15%; "
            "otherwise fall back to the fixed Stage B baseline."
        ),
        "threshold_was_optimized": False,
        "prior_oos_only_selection": True,
        "minimum_prior_trades": MIN_PRIOR_TRADES_FOR_SELECTION,
        "fallback_candidate_id": BASELINE_CANDIDATE_ID,
        "selected_fold_count": fold_count,
        "fallback_fold_count": sum(
            "fallback_" in str(row["selection_reason"]) for row in trace
        ),
        **_trade_summary(adaptive_trades),
    }
    return policy, trace, adaptive_trades


def _prepare_development_context(input_dir: Path) -> dict[str, Any]:
    """Authenticate Stage A and remove the reserved holdout before feature work."""
    _frame, source_report, segments, segment_hashes = verify_source_artifact(input_dir)
    segment_a = segments[0]
    if len(segment_a) <= A_HOLDOUT + A_TRAIN + WINDOW:
        raise ValueError("optimization_segment_a_too_short")
    holdout_start = pd.Timestamp(segment_a["date"].iloc[-A_HOLDOUT])
    development = segment_a.iloc[:-A_HOLDOUT].copy().reset_index(drop=True)
    del segment_a, segments, _frame
    candle_columns = ["date", "open", "high", "low", "close", "volume"]
    validated, quality_fingerprint = validate_ohlcv(
        development.loc[:, candle_columns].copy(), HOUR
    )
    engineered = add_signal_features(validated)
    feature_diagnostics = feature_availability_diagnostics(
        engineered, tuple(FEATURE_COLUMNS) + NEW_FEATURE_COLUMNS
    )
    selected_features, excluded_features = _candidate_features(engineered)
    common_start_utc, common_end_utc = _common_evaluation_bounds(
        engineered,
        selected_features["extended_trend_rsi"],
        max(HORIZON_GRID),
    )
    return {
        "source_report": source_report,
        "segment_hashes": segment_hashes,
        "holdout_start": holdout_start,
        "development": development,
        "quality_fingerprint": quality_fingerprint,
        "engineered": engineered,
        "feature_diagnostics": feature_diagnostics,
        "selected_features": selected_features,
        "excluded_features": excluded_features,
        "common_start_utc": common_start_utc,
        "common_end_utc": common_end_utc,
    }


def run_signal_optimization(
    input_dir: Path, output_dir: Path, commit_sha: str
) -> dict[str, Any]:
    _validate_commit_sha(commit_sha)
    context = _prepare_development_context(input_dir)
    source_report = context["source_report"]
    segment_hashes = context["segment_hashes"]
    holdout_start = context["holdout_start"]
    development = context["development"]
    quality_fingerprint = context["quality_fingerprint"]
    feature_diagnostics = context["feature_diagnostics"]
    selected_features = context["selected_features"]
    excluded_features = context["excluded_features"]
    common_start_utc = context["common_start_utc"]
    common_end_utc = context["common_end_utc"]
    candidate_datasets, shared_row_count = _build_candidate_datasets(
        context["engineered"],
        selected_features,
        common_start_utc,
        common_end_utc,
    )
    candidate_results = _evaluate_candidate_grid(candidate_datasets, selected_features)
    policy, policy_trace, policy_trades = _evaluate_adaptive_policy(candidate_results)
    candidate_summaries, prediction_rows, trade_rows, fold_rows = (
        _collect_candidate_rows(candidate_results)
    )
    baseline_summary = _candidate_summary(candidate_results[BASELINE_CANDIDATE_ID])

    report: dict[str, Any] = {
        "report_schema": REPORT_SCHEMA,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "created_by_commit_sha": commit_sha,
        "source": {
            "coverage_workflow_run_id": SOURCE_RUN_ID,
            "coverage_csv_sha256": source_report["csv_sha256"],
            "coverage_manifest_sha256": source_report["manifest_sha256"],
            "segment_a_rows_sha256": segment_hashes[0],
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "full_segment_a_input_rows": len(development) + A_HOLDOUT,
            "development_input_rows": len(development),
            "holdout_rows_reserved": A_HOLDOUT,
            "development_start_utc": utc_text(development["date"].iloc[0]),
            "development_end_utc": utc_text(development["date"].iloc[-1]),
            "reserved_holdout_start_utc": utc_text(holdout_start),
            "common_candidate_oos_start_utc": utc_text(common_start_utc),
            "common_candidate_oos_end_utc": utc_text(common_end_utc),
            "data_quality_gate_fingerprint": quality_fingerprint,
        },
        "root_cause_analysis": {
            "reference_stage_b_run_id": 40099106018,
            "reference_stage_b_commit_sha": "fa5d290762d5658dab2420f2d52231abdc436598",
            "reference_a_development": {
                "oos_prediction_rows": 30022,
                "non_overlapping_entry_opportunities": 2502,
                "round_trip_cost_hurdle": ROUND_TRIP_COST,
                "opportunity_forecast_p50": 0.0006794310897241,
                "opportunity_forecast_p90": 0.0015819533710991606,
                "opportunity_forecast_p95": 0.002200334075231763,
                "opportunity_forecast_p99": 0.004655422143839505,
                "opportunity_forecast_max": 0.0230800494054442,
                "trades_taken": 12,
                "take_rate_per_opportunity": 12 / 2502,
                "win_rate": 5 / 12,
                "net_expectancy_per_trade": -0.010708819671677161,
                "max_drawdown_abs": 0.204516803418077,
            },
            "primary_cause": (
                "The Stage B rule is a 12-hour forward-return regressor, not a "
                "Triple-Barrier probability filter. It accepts a trade only when "
                "the predicted 12-hour return is strictly greater than the fixed "
                "0.6% round-trip cost. On A development, the 99th percentile of "
                "entry-opportunity forecasts is about 0.466%, below the 0.6% hurdle; "
                "only 12 of 2,502 eligible entry times pass."
            ),
            "horizon_effect": (
                "The 12-hour non-overlap rule defines candidate entry times; it "
                "does not explain the very low take rate by itself. Horizons 6, 12 "
                "and 24 hours are tested as explicit research alternatives."
            ),
            "hmm_and_meta_label_scope": (
                "HMM and Triple-Barrier Meta-Labeling are not in the Stage B runner "
                "or its signal-generation path. Their separate Phase 5 experiment "
                "is not modified here; this experiment targets the directly observed "
                "Stage B LightGBM forecast-versus-cost bottleneck."
            ),
            "holdout_used_for_feature_or_parameter_selection": False,
        },
        "protocol": {
            "evaluation_split": "Stage A segment A development only",
            "holdout_evaluated_by_this_experiment": False,
            "holdout_used_for_feature_or_parameter_selection": False,
            "oos_window_hours": WINDOW,
            "minimum_training_rows": A_TRAIN,
            "purge_hours_by_candidate": list(HORIZON_GRID),
            "walk_forward": "expanding, chronological, no shuffle; fold-specific fresh model",
            "features_calculated_from_current_closed_candle_and_past_only": True,
            "feature_imputation": False,
            "forward_fill": False,
            "backfill": False,
            "interpolation": False,
            "resampling": False,
            "cost_hurdle": ROUND_TRIP_COST,
            "cost_hurdle_optimized": False,
            "fee_bps_per_side": FEE_BPS_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_SIDE,
            "horizons_hours": list(HORIZON_GRID),
            "common_candidate_oos_start_utc": utc_text(common_start_utc),
            "common_candidate_oos_end_utc": utc_text(common_end_utc),
            "common_candidate_row_count": shared_row_count,
            "candidate_timestamp_alignment": (
                "All 12 candidates use identical contiguous A-development timestamps; "
                "the shared end date allows maturity of the maximum 24-hour target."
            ),
            "candidate_timestamps_aligned": True,
            "feature_groups": {
                group: {
                    "selected_columns": selected_features[group],
                    "excluded_features_without_imputation": excluded_features[group],
                }
                for group in FEATURE_GROUPS
            },
            "feature_availability_diagnostics": feature_diagnostics,
            "model_profiles": MODEL_PROFILES,
            "candidate_count": len(candidate_results),
            "candidate_selection": (
                "Candidate summary aggregates are exploratory only. The adaptive "
                "policy selects using prior A-development OOS trades only; require "
                ">=30 prior trades, win rate >60%, positive net expectancy and "
                "drawdown <15%; otherwise fall back to the fixed Stage B baseline."
            ),
            "candidate_summary_ranking_used_for_selection": False,
        },
        "fixed_baseline_candidate": baseline_summary,
        "candidate_diagnostics": candidate_summaries,
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
            "aggregate_across_horizons_or_feature_groups": False,
            "adaptive_policy": policy["economic_threshold_evaluation"],
        },
        "safety_invariants": dict(CLOSED_FLAGS),
        "limitations": [
            "This is a research-only LightGBM regression experiment, not a strategy or exchange execution backtest.",
            "Candidate-grid summaries are exploratory and are not independent confirmatory evidence.",
            "The reserved A final holdout is excluded before feature engineering and is not scored.",
            "Volume-derived signals are not fabricated: segment A volume is zero for every source candle, so volume_change_1 remains unavailable.",
            "HMM and Meta-Labeling configuration are not changed because they are not in the Stage B signal path; they require a separate, explicitly scoped experiment.",
            "No candidate, even if its diagnostic metrics appear attractive, can promote a Champion or authorize orders.",
        ],
    }
    if report["run_status"] != "COMPLETED_RESEARCH_ONLY":
        raise ValueError("optimization_status_contract_failed")
    if report["safety_invariants"] != CLOSED_FLAGS:
        raise ValueError("optimization_closed_release_contract_failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "signal-optimization-report.json", report)
    _write_json(
        output_dir / "input-verification.json",
        {
            "report_schema": REPORT_SCHEMA,
            "source_csv_sha256": source_report["csv_sha256"],
            "segment_a_rows_sha256": segment_hashes[0],
            "development_rows": len(development),
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
        "fold_index",
        "fold_id",
        "timestamp_utc",
        "forecast_return_horizon",
        "realized_forward_return",
        "round_trip_cost_fraction",
        "trade_opportunity",
        "trade_taken",
    ]
    trade_columns = [
        "candidate_id",
        "feature_group",
        "model_profile",
        "horizon_hours",
        "fold_index",
        "fold_id",
        "entry_utc",
        "exit_utc",
        "forecast_return",
        "gross_return",
        "fee_bps_round_trip",
        "slippage_bps_round_trip",
        "net_return",
        "won_after_costs",
        "order_submitted",
    ]
    fold_columns = [
        "candidate_id",
        "feature_group",
        "model_profile",
        "horizon_hours",
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
        "prediction_count",
        "entry_opportunity_count",
        "forecasts_above_cost",
        "take_rate_per_opportunity",
        "opportunity_forecast_p50",
        "opportunity_forecast_p90",
        "opportunity_forecast_p95",
        "opportunity_forecast_p99",
        "opportunity_forecast_max",
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
        "fold_id",
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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    report = run_signal_optimization(args.input_dir, args.output_dir, args.commit_sha)
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "commit_sha": report["created_by_commit_sha"],
                "candidate_count": report["protocol"]["candidate_count"],
                "baseline": {
                    "trade_count": report["fixed_baseline_candidate"]["trade_count"],
                    "win_rate": report["fixed_baseline_candidate"]["win_rate"],
                    "net_expectancy_per_trade": report["fixed_baseline_candidate"][
                        "net_expectancy_per_trade"
                    ],
                    "max_drawdown_abs": report["fixed_baseline_candidate"][
                        "max_drawdown_abs"
                    ],
                },
                "adaptive_policy": {
                    "trade_count": report["adaptive_walkforward_policy"]["trade_count"],
                    "win_rate": report["adaptive_walkforward_policy"]["win_rate"],
                    "net_expectancy_per_trade": report["adaptive_walkforward_policy"][
                        "net_expectancy_per_trade"
                    ],
                    "max_drawdown_abs": report["adaptive_walkforward_policy"][
                        "max_drawdown_abs"
                    ],
                    "economic_status": report["adaptive_walkforward_policy"][
                        "economic_threshold_evaluation"
                    ]["status"],
                },
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
