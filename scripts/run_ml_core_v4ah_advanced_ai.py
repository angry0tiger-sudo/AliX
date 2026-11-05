"""V4-AH ETHUSDT 1h advanced AI meta-model economic validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

import scripts.run_ml_core_v4af_meta_final as v4af
from services.meta_labeling import v4ah as advanced
from services.ml_core.economic_backtest import simulate_signals

SCHEMA = "alix.ml-core-v4ah-advanced-ai.v1"
SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
DAYS = 730
REGIME_WARMUP_BARS = 200
INITIAL_EQUITY = 10_000.0
MINIMUM_TRADES = 30
MINIMUM_NET_PF = 1.5
MINIMUM_WIN_RATE = 0.45
MAXIMUM_DRAWDOWN = 15.0
CANDIDATE_NAMES = (
    "UNFILTERED_CONTROL",
    "META_NET_EXPECTANCY_FIXED",
    "META_SHAP_ADAPTIVE_TRAIL",
    "META_SHAP_ADAPTIVE_TRAIL_KELLY",
)
LABEL_COLUMNS = {
    "event_index",
    "fold_id",
    "label_end_idx",
    "meta_label",
    "net_pnl_usd",
    "net_return",
    "exit_reason",
    "primary_signal",
    "primary_oof",
}


def _validate_run_request(days: int, commit_sha: str) -> None:
    v4af._validate_run_request(days, commit_sha)
    if days != DAYS or v4af.SYMBOL != SYMBOL or v4af.TIMEFRAME != TIMEFRAME:
        raise ValueError("v4ah_market_scope_invalid")
    if advanced.V4AH_TOP_K_FEATURES != 30:
        raise ValueError("v4ah_top_k_contract_invalid")


def _candidate_id(name: str) -> str:
    if name not in CANDIDATE_NAMES:
        raise ValueError("v4ah_candidate_name_invalid")
    return f"{SYMBOL}|{TIMEFRAME}|SL2.0_ACT1.0_STEP1.5|{name}"


def _feature_columns(training: DataFrame) -> list[str]:
    columns = [str(name) for name in training.columns if name not in LABEL_COLUMNS]
    if not columns or not training.loc[:, columns].columns.is_unique:
        raise ValueError("v4ah_feature_columns_invalid")
    return columns


def _regime_features(features: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    transformed, diagnostics, _, _ = advanced.build_regime_features(
        features,
        fit_end=REGIME_WARMUP_BARS,
    )
    return transformed, diagnostics


def _oof_training_frame(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: pd.Series,
    train_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    return v4af._meta_oof_training_frame(
        bars,
        features,
        primary_signal,
        train_end,
    )


def _positive_finite_probability_map(
    features: DataFrame,
    event_indices: list[int],
    selected_features: list[str],
    model: Any,
    temperature: float,
) -> dict[int, float]:
    if not event_indices:
        return {}
    validation = (
        features.iloc[event_indices].loc[:, selected_features].reset_index(drop=True)
    )
    raw = advanced._positive_probability(model, validation)
    probabilities = advanced.apply_temperature_scaling(raw, temperature)
    return {
        index: float(probability)
        for index, probability in zip(event_indices, probabilities, strict=True)
    }


def _trade_management(step: float) -> Any:
    base = v4af._trade_management_config(None)
    return replace(base, trailing_step_atr=float(step))


def _single_event_replay(
    bars: DataFrame,
    event_index: int,
    window_end: int,
    trailing_step: float,
) -> tuple[dict[str, Any], int]:
    required_end = event_index + v4af.HORIZON_BARS_VALUE + 2
    if event_index < 0 or required_end > window_end or window_end > len(bars):
        raise ValueError("v4ah_counterfactual_event_window_invalid")
    context_start = max(0, event_index - REGIME_WARMUP_BARS)
    local = bars.iloc[context_start:required_end].reset_index(drop=True)
    local_event = event_index - context_start
    actions = v4af._build_forward_actions(local, [local_event], len(local))
    replay = simulate_signals(
        local,
        actions,
        v4af._execution_config(),
        trade_management_config=_trade_management(trailing_step),
    )
    trades = replay["trades"]
    if len(trades) != 1:
        raise ValueError("v4ah_isolated_event_trade_count_invalid")
    timestamps = pd.to_datetime(local["date"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("v4ah_isolated_event_timestamps_invalid")
    exit_timestamp = pd.Timestamp(str(trades[0]["exit_timestamp"]))
    if exit_timestamp.tzinfo is None:
        raise ValueError("v4ah_isolated_event_exit_timezone_invalid")
    lookup = {pd.Timestamp(value): index for index, value in enumerate(timestamps)}
    exit_local = lookup.get(exit_timestamp.tz_convert("UTC"))
    if exit_local is None or exit_local <= local_event:
        raise ValueError("v4ah_isolated_event_exit_index_invalid")
    return cast(dict[str, Any], trades[0]), context_start + exit_local


def _counterfactual_training_data(
    bars: DataFrame,
    fit_rows: DataFrame,
    selected_features: list[str],
    train_end: int,
) -> tuple[DataFrame, np.ndarray, np.ndarray]:
    feature_rows: list[dict[str, float]] = []
    return_rows: list[list[float]] = []
    end_rows: list[list[float]] = []
    for row in fit_rows.to_dict(orient="records"):
        event_index = int(row["event_index"])
        if event_index + v4af.HORIZON_BARS_VALUE + 2 > train_end:
            continue
        outcomes = [
            _single_event_replay(bars, event_index, train_end, step)
            for step in advanced.V4AH_TRAILING_STEP_OPTIONS
        ]
        feature_rows.append({name: float(row[name]) for name in selected_features})
        return_rows.append([float(item[0]["net_return_fraction"]) for item in outcomes])
        end_rows.append([float(item[1]) for item in outcomes])
    return (
        DataFrame.from_records(feature_rows, columns=selected_features),
        np.asarray(return_rows, dtype=float).reshape(-1, 3),
        np.asarray(end_rows, dtype=float).reshape(-1, 3),
    )


def _fit_trailing_policy(
    bars: DataFrame,
    fit_rows: DataFrame,
    temperature_rows: DataFrame,
    selected_features: list[str],
    train_end: int,
) -> tuple[Any, dict[str, Any]]:
    policy_features, candidate_returns, candidate_ends = _counterfactual_training_data(
        bars,
        fit_rows,
        selected_features,
        train_end,
    )
    calibration_start = int(temperature_rows["event_index"].min())
    return advanced.fit_adaptive_trailing_policy(
        policy_features,
        candidate_returns,
        candidate_ends,
        calibration_start_index=calibration_start,
    )


def _profit_factor(values: list[float]) -> tuple[float | None, bool]:
    winners = sum(value for value in values if value > 0.0)
    losers = abs(sum(value for value in values if value < 0.0))
    if losers > 0.0:
        return winners / losers, False
    return None, winners > 0.0


def _metrics_from_sized_trades(
    trades: list[dict[str, Any]],
    fractions: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, enriched = advanced.summarize_sized_trades(
        trades,
        fractions,
        initial_equity=INITIAL_EQUITY,
    )
    gross_pf, gross_inf = _profit_factor(
        [float(item["gross_market_pnl_usd"]) for item in trades]
    )
    metrics["gross_profit_factor"] = gross_pf
    metrics["gross_profit_factor_infinite_no_losses"] = gross_inf
    return metrics, enriched


def _unscaled_expectancy(trades: list[dict[str, Any]]) -> float:
    returns = [float(item["net_return_fraction"]) for item in trades]
    return float(np.mean(returns) * INITIAL_EQUITY) if returns else 0.0


def _pf_value(metrics: dict[str, Any]) -> float:
    if bool(metrics.get("net_profit_factor_infinite_no_losses")):
        return math.inf
    value = metrics.get("net_profit_factor")
    return -math.inf if value is None else float(value)


def _acceptance(
    metrics: dict[str, Any],
    unscaled_expectancy: float,
    primary_metrics: dict[str, Any],
    primary_expectancy: float,
) -> bool:
    trade_count = int(metrics.get("trade_count", 0))
    drawdown = metrics.get("max_drawdown_percent")
    win_rate = metrics.get("win_rate")
    return (
        trade_count >= MINIMUM_TRADES
        and _pf_value(metrics) > MINIMUM_NET_PF
        and drawdown is not None
        and math.isfinite(float(drawdown))
        and float(drawdown) < MAXIMUM_DRAWDOWN
        and win_rate is not None
        and float(win_rate) > MINIMUM_WIN_RATE
        and _pf_value(metrics) > _pf_value(primary_metrics)
        and unscaled_expectancy > primary_expectancy
    )


def _skipped_candidate(name: str, reason: str, event_count: int) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(name),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": name,
        "status": "SKIPPED_INSUFFICIENT_META_SUPPORT",
        "skip_reason": reason,
        "primary_buy_event_count": event_count,
        "meta_accepted_buy_count": 0,
        "meta_rejected_buy_count": event_count,
        "selected_meta_probability_threshold": None,
        "metrics": {
            "trade_count": 0,
            "gross_profit_factor": None,
            "gross_profit_factor_infinite_no_losses": False,
            "net_profit_factor": None,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": None,
            "win_rate": None,
            "net_pnl_total_usd": 0.0,
            "mean_unscaled_net_expectancy_usd_at_full_equity": 0.0,
        },
        "acceptance_target_met": False,
        "sequence_model_used": False,
    }


def _fixed_meta_candidate(
    bars: DataFrame,
    event_indices: list[int],
    forward_end: int,
    probabilities: dict[int, float],
    threshold: float,
    threshold_diagnostics: dict[str, Any],
    primary_metrics: dict[str, Any],
    primary_expectancy: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    name = "META_NET_EXPECTANCY_FIXED"
    candidate, trades, decisions = v4af._run_economic_candidate(
        bars=bars,
        event_indices=event_indices,
        candidate_id=_candidate_id(name),
        mode=name,
        threshold=threshold,
        target_accept_fraction=None,
        probability_by_index=probabilities,
        forward_end=forward_end,
        calibration_diagnostics=threshold_diagnostics,
    )
    expectancy = _unscaled_expectancy(trades)
    candidate["metrics"]["mean_unscaled_net_expectancy_usd_at_full_equity"] = expectancy
    candidate["acceptance_target_met"] = _acceptance(
        candidate["metrics"], expectancy, primary_metrics, primary_expectancy
    )
    candidate["sequence_model_used"] = False
    return candidate, trades, decisions


def _policy_candidate(
    bars: DataFrame,
    features: DataFrame,
    event_indices: list[int],
    forward_end: int,
    probabilities: dict[int, float],
    selected_features: list[str],
    threshold: float,
    trailing_model: Any,
    name: str,
    primary_metrics: dict[str, Any],
    primary_expectancy: float,
    payoff_ratio: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not event_indices:
        raise ValueError("v4ah_forward_primary_events_empty")
    validation = (
        features.iloc[event_indices].loc[:, selected_features].reset_index(drop=True)
    )
    steps = advanced.predict_trailing_steps(trailing_model, validation)
    step_by_index = dict(zip(event_indices, steps.tolist(), strict=True))
    accepted = [
        index for index in event_indices if probabilities.get(index, 0.0) >= threshold
    ]
    raw_trades: list[dict[str, Any]] = []
    fractions: list[float] = []
    decisions: list[dict[str, Any]] = []
    last_exit_index = -1
    for index in event_indices:
        probability = probabilities.get(index)
        is_accepted = probability is not None and probability >= threshold
        execution_status = "REJECTED_BY_META"
        if is_accepted:
            if index + 1 <= last_exit_index:
                execution_status = "SKIPPED_OVERLAPPING_POSITION"
            else:
                trade, exit_index = _single_event_replay(
                    bars,
                    index,
                    forward_end,
                    float(step_by_index[index]),
                )
                fraction = (
                    1.0
                    if payoff_ratio is None
                    else advanced.fractional_kelly_fraction(
                        float(probability), payoff_ratio
                    )
                )
                if fraction > 0.0:
                    raw_trades.append(
                        {
                            "candidate_id": _candidate_id(name),
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "mode": name,
                            "event_index": index,
                            "trailing_step_atr": float(step_by_index[index]),
                            "meta_probability": float(probability),
                            **trade,
                        }
                    )
                    fractions.append(fraction)
                    last_exit_index = exit_index
                    execution_status = "EXECUTED"
                else:
                    execution_status = "SKIPPED_ZERO_KELLY_FRACTION"
        decisions.append(
            {
                "candidate_id": _candidate_id(name),
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "mode": name,
                "event_index": index,
                "meta_probability": probability,
                "selected_threshold": threshold,
                "meta_filter_decision": "ACCEPT" if is_accepted else "REJECT",
                "trailing_step_atr": (
                    float(step_by_index[index]) if is_accepted else None
                ),
                "portfolio_execution": execution_status,
                "split_role": "FORWARD_VALIDATION",
            }
        )
    metrics, enriched = _metrics_from_sized_trades(raw_trades, fractions)
    expectancy = _unscaled_expectancy(raw_trades)
    candidate = {
        "candidate_id": _candidate_id(name),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": name,
        "status": "COMPLETED",
        "selected_meta_probability_threshold": threshold,
        "primary_buy_event_count": len(event_indices),
        "meta_accepted_buy_count": len(accepted),
        "meta_rejected_buy_count": len(event_indices) - len(accepted),
        "executed_trade_count": len(enriched),
        "achieved_validation_accept_fraction": len(accepted) / len(event_indices),
        "metrics": {
            **metrics,
            "mean_unscaled_net_expectancy_usd_at_full_equity": expectancy,
        },
        "acceptance_target_met": _acceptance(
            metrics, expectancy, primary_metrics, primary_expectancy
        ),
        "sequence_model_used": False,
    }
    return candidate, enriched, decisions


def _expected_support_shortfall(error: ValueError) -> bool:
    return str(error) in {
        "v4af_meta_oof_labels_empty",
        "v4ah_three_way_training_schema_or_support_invalid",
        "v4ah_fit_rows_insufficient_after_purge",
        "v4ah_temperature_rows_insufficient_after_purge",
        "v4ah_threshold_rows_insufficient",
        "v4ah_binary_class_support_insufficient",
        "v4ah_regime_training_support_insufficient",
        "v4ah_trailing_policy_training_support_insufficient",
        "v4ah_kelly_payoff_support_insufficient",
    }


def _fit_advanced_meta(
    bars: DataFrame,
    features: DataFrame,
    primary_signal: pd.Series,
    train_end: int,
    train_gate: dict[str, Any],
) -> dict[str, Any]:
    if not bool(train_gate.get("passed")):
        return {"skip_reason": "v4y_insufficient_primary_signals"}
    try:
        training, training_diagnostics = _oof_training_frame(
            bars, features, primary_signal, train_end
        )
        fit_rows, temperature_rows, threshold_rows, split_diagnostics = (
            advanced.split_fit_temperature_threshold(
                training,
                outer_train_end=train_end,
            )
        )
    except ValueError as error:
        if not _expected_support_shortfall(error):
            raise
        return {"skip_reason": str(error), "training_diagnostics": {}}
    feature_names = _feature_columns(training)
    model, selected, rankings, shap_diagnostics = advanced.fit_shap_selected_lightgbm(
        fit_rows.loc[:, feature_names],
        fit_rows["meta_label"],
    )
    raw_temperature = advanced._positive_probability(
        model, temperature_rows.loc[:, selected]
    )
    temperature, temperature_diagnostics = advanced.fit_temperature_scaler(
        raw_temperature,
        temperature_rows["meta_label"].to_numpy(dtype=int),
    )
    raw_threshold = advanced._positive_probability(
        model, threshold_rows.loc[:, selected]
    )
    calibrated_threshold = advanced.apply_temperature_scaling(
        raw_threshold,
        temperature,
    )
    threshold, threshold_diagnostics = advanced.select_net_expectancy_threshold(
        calibrated_threshold,
        threshold_rows["net_return"].to_numpy(dtype=float),
    )
    calibration = advanced.calibration_diagnostics(
        calibrated_threshold,
        threshold_rows["meta_label"].to_numpy(dtype=int),
    )
    if threshold is None:
        return {
            "skip_reason": "v4ah_no_positive_net_expectancy_threshold",
            "training_diagnostics": training_diagnostics,
            "split_diagnostics": split_diagnostics,
            "fit_diagnostics": shap_diagnostics,
            "shap_rankings": rankings,
            "temperature_diagnostics": temperature_diagnostics,
            "threshold_diagnostics": threshold_diagnostics,
            "calibration_diagnostics": calibration,
        }
    policy_model: Any = None
    policy_diagnostics: dict[str, Any] = {}
    policy_skip_reason: str | None = None
    try:
        policy_model, policy_diagnostics = _fit_trailing_policy(
            bars,
            fit_rows,
            temperature_rows,
            selected,
            train_end,
        )
    except ValueError as error:
        if not _expected_support_shortfall(error):
            raise
        policy_skip_reason = str(error)
    payoff_ratio: float | None = None
    payoff_diagnostics: dict[str, Any] = {}
    try:
        payoff_ratio, payoff_diagnostics = advanced.estimate_training_payoff_ratio(
            fit_rows["net_return"].to_numpy(dtype=float)
        )
    except ValueError as error:
        if not _expected_support_shortfall(error):
            raise
        payoff_diagnostics = {"status": "SKIPPED", "reason": str(error)}
    return {
        "skip_reason": None,
        "training_diagnostics": training_diagnostics,
        "split_diagnostics": split_diagnostics,
        "fit_diagnostics": shap_diagnostics,
        "shap_rankings": rankings,
        "temperature_diagnostics": temperature_diagnostics,
        "threshold_diagnostics": threshold_diagnostics,
        "calibration_diagnostics": calibration,
        "model": model,
        "selected_features": selected,
        "temperature": temperature,
        "threshold": threshold,
        "policy_model": policy_model,
        "trailing_policy_diagnostics": policy_diagnostics,
        "policy_skip_reason": policy_skip_reason,
        "payoff_ratio": payoff_ratio,
        "payoff_diagnostics": payoff_diagnostics,
    }


def _build_candidates(
    bars: DataFrame,
    features: DataFrame,
    event_indices: list[int],
    forward_end: int,
    primary: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    control, primary_trades, primary_decisions = primary["result"]
    candidates = [control]
    trades = list(primary_trades)
    decisions = list(primary_decisions)
    primary_metrics = dict(control["metrics"])
    primary_expectancy = _unscaled_expectancy(primary_trades)
    control["metrics"]["mean_unscaled_net_expectancy_usd_at_full_equity"] = (
        primary_expectancy
    )
    if not event_indices:
        candidates.extend(
            _skipped_candidate(name, "v4ah_forward_primary_events_empty", 0)
            for name in CANDIDATE_NAMES[1:]
        )
        return candidates, trades, decisions
    if meta.get("skip_reason") is not None:
        reason = str(meta["skip_reason"])
        candidates.extend(
            _skipped_candidate(name, reason, len(event_indices))
            for name in CANDIDATE_NAMES[1:]
        )
        return candidates, trades, decisions
    probabilities = _positive_finite_probability_map(
        features,
        event_indices,
        meta["selected_features"],
        meta["model"],
        float(meta["temperature"]),
    )
    fixed, fixed_trades, fixed_decisions = _fixed_meta_candidate(
        bars,
        event_indices,
        forward_end,
        probabilities,
        float(meta["threshold"]),
        meta["threshold_diagnostics"],
        primary_metrics,
        primary_expectancy,
    )
    candidates.append(fixed)
    trades.extend(fixed_trades)
    decisions.extend(fixed_decisions)
    if meta.get("policy_model") is None:
        reason = str(
            meta.get("policy_skip_reason") or "v4ah_trailing_policy_unavailable"
        )
        candidates.extend(
            _skipped_candidate(name, reason, len(event_indices))
            for name in CANDIDATE_NAMES[2:]
        )
        return candidates, trades, decisions
    adaptive, adaptive_trades, adaptive_decisions = _policy_candidate(
        bars,
        features,
        event_indices,
        forward_end,
        probabilities,
        meta["selected_features"],
        float(meta["threshold"]),
        meta["policy_model"],
        CANDIDATE_NAMES[2],
        primary_metrics,
        primary_expectancy,
    )
    candidates.append(adaptive)
    trades.extend(adaptive_trades)
    decisions.extend(adaptive_decisions)
    if meta.get("payoff_ratio") is None:
        candidates.append(
            _skipped_candidate(
                CANDIDATE_NAMES[3],
                str(meta["payoff_diagnostics"].get("reason", "v4ah_kelly_unavailable")),
                len(event_indices),
            )
        )
        return candidates, trades, decisions
    kelly, kelly_trades, kelly_decisions = _policy_candidate(
        bars,
        features,
        event_indices,
        forward_end,
        probabilities,
        meta["selected_features"],
        float(meta["threshold"]),
        meta["policy_model"],
        CANDIDATE_NAMES[3],
        primary_metrics,
        primary_expectancy,
        payoff_ratio=float(meta["payoff_ratio"]),
    )
    candidates.append(kelly)
    trades.extend(kelly_trades)
    decisions.extend(kelly_decisions)
    return candidates, trades, decisions


def _safety_report() -> dict[str, Any]:
    return {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "credentials_used": False,
        "sequence_model_used": False,
    }


def _validate_report(report: dict[str, Any], commit_sha: str) -> None:
    safety = report.get("safety", {})
    chronology = report.get("chronological_evaluation", {})
    candidates = report.get("candidates", [])
    if report.get("schema_version") != SCHEMA or report.get("commit_sha") != commit_sha:
        raise ValueError("v4ah_report_identity_invalid")
    if report.get("symbols") != [SYMBOL] or report.get("timeframes") != [TIMEFRAME]:
        raise ValueError("v4ah_report_scope_invalid")
    geometry = report.get("fixed_exit_geometry", {})
    costs = report.get("cost_contract", {})
    if geometry != {
        "initial_stop_atr": 2.0,
        "trailing_activation_atr": 1.0,
        "trailing_step_atr": 1.5,
        "take_profit_atr": 4.0,
        "horizon_bars": 24,
    }:
        raise ValueError("v4ah_report_exit_geometry_invalid")
    if costs != {
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
        "round_trip_cost_bps": 30.0,
    }:
        raise ValueError("v4ah_report_cost_contract_invalid")
    if [item.get("mode") for item in candidates] != list(CANDIDATE_NAMES):
        raise ValueError("v4ah_candidate_matrix_invalid")
    if (
        report.get("final_holdout_used") is not False
        or chronology.get("final_15_percent_holdout_untouched") is not True
    ):
        raise ValueError("v4ah_final_holdout_contamination")
    if not (
        safety.get("Champion") is None
        and safety.get("order_permission") is False
        and safety.get("trading_authorization") == "NO-GO"
        and safety.get("private_api_calls") == 0
        and safety.get("real_order_attempts") == 0
        and safety.get("sequence_model_used") is False
    ):
        raise ValueError("v4ah_fail_closed_safety_contract_invalid")
    if any(item.get("sequence_model_used") is not False for item in candidates):
        raise ValueError("v4ah_sequence_model_forbidden")


def _format_pf(metrics: dict[str, Any]) -> str:
    if metrics.get("net_profit_factor_infinite_no_losses"):
        return "INF"
    value = metrics.get("net_profit_factor")
    return "N/A" if value is None else f"{float(value):.4f}"


def _render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ah_step_summary_schema_invalid")
    lines = [
        "## AliX V4-AH Advanced AI Economic Validation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "Scope: ETHUSDT Spot, 1h, 730 days, public OHLCV only.",
        "Primary: causal Bollinger(20, 2.0) upper-band breakout with Close > EMA200.",
        "Execution: closed-candle signal, next-bar-open entry, no pyramiding.",
        "Geometry: SL=2.0 ATR, TP=4.0 ATR, trailing activation=1.0 ATR, reference step=1.5 ATR, horizon=24 bars.",
        "Costs: maker fee=10 bps/side, slippage=5 bps/side, round trip=30 bps.",
        "Training: outer 70%; purged OOF Fit/Temperature/Threshold slices=70/15/15.",
        "Evaluation: next 15% forward validation; final 15% holdout untouched.",
        "Model: LightGBM + TreeSHAP top-30 Fit-only selection; KMeans regimes as features.",
        "Calibration: Temperature Scaling (L-BFGS-B); threshold selected by positive net expectancy.",
        "",
        "| Candidate | Trades | Win Rate % | Net PF | Max DD % | Net PnL USD | Unscaled expectancy USD/trade | Acceptance | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["candidates"]:
        metrics = item["metrics"]
        expectancy = metrics.get("mean_unscaled_net_expectancy_usd_at_full_equity")
        win_rate = metrics.get("win_rate")
        win_text = "N/A" if win_rate is None else f"{float(win_rate) * 100.0:.2f}"
        drawdown = metrics.get("max_drawdown_percent")
        drawdown_text = "N/A" if drawdown is None else f"{float(drawdown):.2f}"
        lines.append(
            f"| {item['mode']} | {metrics.get('trade_count', 0)} "
            f"| {win_text} "
            f"| {_format_pf(metrics)} "
            f"| {drawdown_text} "
            f"| {float(metrics.get('net_pnl_total_usd', 0.0)):.2f} "
            f"| {'N/A' if expectancy is None else f'{float(expectancy):.4f}'} "
            f"| {'PASS' if item.get('acceptance_target_met') else 'NO'} "
            f"| {item['status']} |"
        )
    meta = report.get("meta_model", {})
    fit = meta.get("fit_diagnostics", {})
    threshold = meta.get("threshold_diagnostics", {})
    regime = report.get("regime_diagnostics", {})
    calibration = meta.get("calibration_diagnostics", {})
    lines.extend(
        [
            "",
            f"OOF labels: {report.get('training_diagnostics', {}).get('matured_costed_oof_label_count', 0)}.",
            f"SHAP top-k: {fit.get('top_k_selected', 0)}; selected features: {', '.join(fit.get('selected_features', []))}.",
            f"Regime: KMeans clusters={regime.get('cluster_count', 'N/A')}; fit rows={regime.get('fit_rows', 0)}; hard filter=false.",
            f"Temperature: {meta.get('temperature_diagnostics', {}).get('temperature', 'N/A')}.",
            f"Threshold: {threshold.get('threshold', 'N/A')}; selected rows={threshold.get('selected_rows', 0)}.",
            f"Calibration: Brier={calibration.get('brier_score', 'N/A')}; ECE={calibration.get('expected_calibration_error', 'N/A')}.",
            f"Trailing policy: {meta.get('trailing_policy_diagnostics', {}).get('model_family', 'SKIPPED')}.",
            f"Kelly: quarter-Kelly; max fraction={advanced.V4AH_MAXIMUM_KELLY_FRACTION:.2%}; fit={meta.get('payoff_diagnostics', {}).get('fit_scope', 'SKIPPED')}.",
            "",
            "Acceptance: Net PF >1.5, Max DD <15%, Win Rate >45%, Trades >=30, and outperformance vs Primary on Net PF and unscaled expectancy.",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only. No private API, credentials, live/paper orders, or production execution.",
            "Promotion: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    if not columns:
        columns = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    commit_sha: str,
    provenance: dict[str, Any],
    bars: DataFrame,
    train_end: int,
    forward_end: int,
    signal_counts: dict[str, Any],
    train_gate: dict[str, Any],
    training_diagnostics: dict[str, Any],
    regime_diagnostics: dict[str, Any],
    meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "symbols": [SYMBOL],
        "timeframes": [TIMEFRAME],
        "dataset_provenance": provenance,
        "dataset_days": DAYS,
        "bar_count": int(len(bars)),
        "fixed_exit_geometry": {
            "initial_stop_atr": v4af.INITIAL_STOP_ATR,
            "trailing_activation_atr": v4af.TRAILING_ACTIVATION_ATR,
            "trailing_step_atr": v4af.TRAILING_STEP_ATR,
            "take_profit_atr": v4af.TAKE_PROFIT_ATR,
            "horizon_bars": v4af.HORIZON_BARS_VALUE,
        },
        "cost_contract": {
            "fee_bps_per_side": v4af.FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": v4af.SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": v4af.ROUND_TRIP_COST_BPS,
        },
        "signal_counts": signal_counts,
        "primary_train_sample_gate": train_gate,
        "chronological_evaluation": {
            "train_end_index_exclusive": train_end,
            "forward_validation_end_index_exclusive": forward_end,
            "final_holdout_start_index": forward_end,
            "final_15_percent_holdout_untouched": True,
            "final_holdout_used": False,
        },
        "final_holdout_used": False,
        "training_diagnostics": training_diagnostics,
        "regime_diagnostics": regime_diagnostics,
        "meta_model": {
            "enabled": "model" in meta,
            **{
                key: value
                for key, value in meta.items()
                if key not in {"model", "policy_model"}
            },
        },
        "candidates": candidates,
        "trade_count": len(trades),
        "decision_count": len(decisions),
        "safety": _safety_report(),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
    }
    _validate_report(report, commit_sha)
    return report


def _empty_meta(
    reason: str, training_diagnostics: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "skip_reason": reason,
        "training_diagnostics": training_diagnostics or {},
        "fit_diagnostics": {},
        "shap_rankings": [],
        "temperature_diagnostics": {},
        "threshold_diagnostics": {},
        "calibration_diagnostics": {},
        "trailing_policy_diagnostics": {},
        "payoff_diagnostics": {},
    }


def run_v4ah(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_run_request(days, commit_sha)
    (
        provenance,
        bars,
        train_end,
        forward_end,
        research_bars,
        base_features,
        primary_signal,
    ) = v4af._load_research_context(input_dir, days, commit_sha)
    signal_counts, train_gate, event_indices = v4af._primary_signal_evaluation(
        primary_signal, train_end, forward_end
    )
    control = v4af._run_unfiltered_control(research_bars, event_indices, forward_end)
    try:
        features, regime_diagnostics = _regime_features(base_features)
    except ValueError as error:
        if not _expected_support_shortfall(error):
            raise
        regime_diagnostics = {"status": "SKIPPED", "reason": str(error)}
        meta = _empty_meta(str(error))
        prepared = {"result": control}
        candidates, trades, decisions = _build_candidates(
            research_bars, base_features, event_indices, forward_end, prepared, meta
        )
    else:
        meta = _fit_advanced_meta(
            research_bars, features, primary_signal, train_end, train_gate
        )
        prepared = {"result": control}
        candidates, trades, decisions = _build_candidates(
            research_bars, features, event_indices, forward_end, prepared, meta
        )
    training_diagnostics = cast(dict[str, Any], meta.get("training_diagnostics", {}))
    report = _build_report(
        commit_sha,
        provenance,
        bars,
        train_end,
        forward_end,
        signal_counts,
        train_gate,
        training_diagnostics,
        regime_diagnostics,
        meta,
        candidates,
        trades,
        decisions,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_trades_csv, trades)
    _write_csv(output_decisions_csv, decisions)
    summary = _render_step_summary(report)
    print(summary, end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(DAYS,), default=DAYS)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4ah(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["candidates"]) == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
