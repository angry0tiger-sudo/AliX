"""V4-AF ETHUSDT 1h Meta-Labeling on the owner-selected V4-AE exit geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    _load_manifested_data,
    resample_ohlcv,
)
from scripts.run_ml_core_v4ac_cost_optimization import (
    ATR_PERIOD,
    HORIZON_BARS,
    _build_forward_actions,
    _execution_config,
    _outer_boundaries,
    _summarize_replay,
)
from services.meta_labeling.v4y import (
    V4Y_EMA_TREND_PERIOD,
    V4Y_META_CALIBRATION_FRACTION,
    V4Y_META_FEATURE_COLUMNS,
    V4Y_MINIMUM_CLASS_ROWS,
    V4Y_MINIMUM_FIT_ROWS,
    V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
    V4Y_OOF_FOLD_COUNT,
    V4Y_PRIMARY_SIGNAL_GENERATION,
    build_v4y_primary_features,
    evaluate_primary_train_sample_gate,
    fit_lightgbm_meta_filter,
    generate_v4y_oof_primary_events,
    predict_meta_probabilities,
    split_meta_fit_calibration,
)
from services.ml_core.binary_calibration import select_dynamic_buy_threshold
from services.ml_core.economic_backtest import TradeManagementConfig, simulate_signals
from services.ml_core.extended_features import build_feature_profiles

SCHEMA = "alix.ml-core.v4af-meta-final.v1"
SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
FEATURE_PROFILE = "Combined"
MODE = "META_FILTERED"
CONTROL_MODE = "UNFILTERED_CONTROL"

INITIAL_STOP_ATR = 2.0
TRAILING_ACTIVATION_ATR = 1.5
TRAILING_STEP_ATR = 1.5
TAKE_PROFIT_ATR = 4.0
ATR_PERIOD_VALUE = ATR_PERIOD
HORIZON_BARS_VALUE = HORIZON_BARS

TARGET_ACCEPT_FRACTIONS = (0.30, 0.50, 0.70)
TRAIN_FRACTION = 0.70
FORWARD_END_FRACTION = 0.85
DAYS = 730
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
MINIMUM_TRADES_EXCLUSIVE = 30
MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE = 1.5
MAXIMUM_DRAWDOWN_INCLUSIVE = 15.0

V4AE_REFERENCE = {
    "run_id": 7,
    "commit_sha": "83abb739271718923222cdb26850771029d091b9",
    "artifact_id": 13403192059,
    "owner_selected_geometry": {
        "initial_stop_atr": 2.0,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 1.5,
        "trade_count": 42,
        "net_profit_factor": 1.1538,
        "max_drawdown_percent": 9.44,
    },
    "historical_best_cell": {
        "initial_stop_atr": 2.0,
        "trailing_activation_atr": 1.5,
        "trailing_step_atr": 2.0,
        "trade_count": 42,
        "net_profit_factor": 1.2477,
        "max_drawdown_percent": 8.97,
    },
}


def _validate_dataset_window(days: int) -> None:
    if days != DAYS:
        raise ValueError("v4af_dataset_window_must_be_730_days")


def _validate_commit_sha(commit_sha: str) -> None:
    valid = len(commit_sha) == 40 and all(
        character in "0123456789abcdef" for character in commit_sha
    )
    if not valid:
        raise ValueError("v4af_commit_sha_invalid")


def _validate_fixed_market_scope() -> None:
    if SYMBOL != "ETHUSDT" or TIMEFRAME != "1h":
        raise ValueError("v4af_scope_invalid")


def _validate_fixed_exit_and_cost_contract() -> None:
    geometry_is_valid = (
        INITIAL_STOP_ATR == 2.0
        and TRAILING_ACTIVATION_ATR == 1.5
        and TRAILING_STEP_ATR == 1.5
        and TAKE_PROFIT_ATR == 4.0
        and HORIZON_BARS_VALUE == 24
        and ROUND_TRIP_COST_BPS == 30.0
    )
    if not geometry_is_valid:
        raise ValueError("v4af_owner_geometry_or_cost_contract_invalid")


def _validate_meta_threshold_contract() -> None:
    if TARGET_ACCEPT_FRACTIONS != (0.30, 0.50, 0.70):
        raise ValueError("v4af_meta_threshold_matrix_invalid")


def _validate_run_request(days: int, commit_sha: str) -> None:
    _validate_dataset_window(days)
    _validate_commit_sha(commit_sha)
    _validate_fixed_market_scope()
    _validate_fixed_exit_and_cost_contract()
    _validate_meta_threshold_contract()


def _trade_management_config(
    confidence_threshold: float | None,
) -> TradeManagementConfig:
    threshold = None if confidence_threshold is None else float(confidence_threshold)
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("v4af_meta_threshold_invalid")
    return TradeManagementConfig(
        atr_period=ATR_PERIOD_VALUE,
        take_profit_atr_multiplier=TAKE_PROFIT_ATR,
        stop_loss_atr_multiplier=INITIAL_STOP_ATR,
        horizon_candles=HORIZON_BARS_VALUE,
        confidence_threshold=threshold,
        trailing_stop_enabled=True,
        trailing_activation_atr=TRAILING_ACTIVATION_ATR,
        trailing_step_atr=TRAILING_STEP_ATR,
    )


def _eligible_primary_event_indices(
    primary_buy_signal: Series,
    train_end: int,
    forward_end: int,
) -> list[int]:
    if len(primary_buy_signal) < forward_end or not 0 <= train_end < forward_end:
        raise ValueError("v4af_validation_signal_window_invalid")
    last_eligible = forward_end - HORIZON_BARS_VALUE - 2
    if last_eligible < train_end:
        raise ValueError("v4af_validation_window_has_no_matured_events")
    return [
        index
        for index in range(train_end, last_eligible + 1)
        if bool(primary_buy_signal.iloc[index])
    ]


def _label_one_oof_primary_event(
    event: dict[str, Any],
    train_bars: DataFrame,
    index_by_timestamp: dict[pd.Timestamp, int],
    train_end: int,
) -> dict[str, Any] | None:
    event_index = int(event["event_index"])
    if event.get("primary_signal") != "BUY" or event.get("primary_oof") is not True:
        raise ValueError("v4af_meta_labels_require_oof_primary_buy_events")
    if event_index + HORIZON_BARS_VALUE >= train_end:
        return None

    actions = _build_forward_actions(train_bars, [event_index], train_end)
    replay = simulate_signals(
        train_bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(None),
    )
    trades = replay["trades"]
    if len(trades) != 1:
        raise ValueError("v4af_isolated_oof_event_must_create_exactly_one_trade")
    trade = trades[0]
    exit_timestamp = pd.Timestamp(str(trade["exit_timestamp"]))
    if exit_timestamp.tzinfo is None:
        raise ValueError("v4af_training_exit_timestamp_timezone_required")
    exit_index = index_by_timestamp.get(exit_timestamp.tz_convert("UTC"))
    if exit_index is None or not event_index < exit_index < train_end:
        raise ValueError("v4af_training_label_not_matured_inside_outer_train")

    net_pnl = float(trade["net_pnl_usd"])
    net_return = float(trade["net_return_fraction"])
    if not math.isfinite(net_pnl) or not math.isfinite(net_return):
        raise ValueError("v4af_training_label_outcome_non_finite")
    return {
        "event_index": event_index,
        "fold_id": int(event["fold_id"]),
        "label_end_idx": exit_index,
        "meta_label": int(net_pnl > 0.0),
        "net_pnl_usd": net_pnl,
        "net_return": net_return,
        "exit_reason": str(trade["exit_reason"]),
        "primary_signal": "BUY",
        "primary_oof": True,
    }


def _validate_training_label_frame(
    labelled: DataFrame,
    train_end: int,
) -> None:
    if labelled["event_index"].duplicated().any():
        raise ValueError("v4af_meta_oof_labels_duplicate")
    if (labelled["label_end_idx"] >= train_end).any():
        raise ValueError("v4af_meta_training_outer_validation_contamination")


def _training_label_diagnostics(
    labelled: DataFrame,
    primary_event_count: int,
    immature_event_count: int,
    train_end: int,
) -> dict[str, Any]:
    class_counts = labelled["meta_label"].value_counts().to_dict()
    return {
        "oof_primary_event_count": primary_event_count,
        "matured_costed_oof_label_count": int(len(labelled)),
        "immature_oof_event_count": immature_event_count,
        "oof_fold_count": V4Y_OOF_FOLD_COUNT,
        "oof_fold_ids": sorted(
            int(value) for value in labelled["fold_id"].unique().tolist()
        ),
        "label_class_counts": {
            "meta_failure": int(class_counts.get(0, 0)),
            "meta_success": int(class_counts.get(1, 0)),
        },
        "all_primary_events_are_buy": bool(labelled["primary_signal"].eq("BUY").all()),
        "all_primary_events_are_oof": bool(labelled["primary_oof"].eq(True).all()),
        "maximum_training_label_end_index": int(labelled["label_end_idx"].max()),
        "outer_train_end_index": train_end,
        "training_labels_matured_before_validation": bool(
            int(labelled["label_end_idx"].max()) < train_end
        ),
        "label_target": (
            "NET_PNL_POSITIVE_AFTER_FEES_AND_SLIPPAGE_FROM_EXACT_V4AF_TRAILING_REPLAY"
        ),
        "label_geometry": {
            "initial_stop_atr": INITIAL_STOP_ATR,
            "trailing_activation_atr": TRAILING_ACTIVATION_ATR,
            "trailing_step_atr": TRAILING_STEP_ATR,
            "take_profit_atr": TAKE_PROFIT_ATR,
            "horizon_bars": HORIZON_BARS_VALUE,
        },
        "label_execution_costs": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
    }


def _training_oof_labels(
    bars: DataFrame,
    primary_events: DataFrame,
    train_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    """Label OOF BUY events with isolated replays of the exact V4-AF geometry."""
    train_bars = bars.iloc[:train_end].reset_index(drop=True)
    dates = pd.to_datetime(train_bars["date"], utc=True, errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("v4af_training_timestamps_invalid")
    index_by_timestamp = {
        pd.Timestamp(timestamp): index for index, timestamp in enumerate(dates.tolist())
    }
    records: list[dict[str, Any]] = []
    immature_event_count = 0
    for event in primary_events.to_dict(orient="records"):
        record = _label_one_oof_primary_event(
            cast(dict[str, Any], event),
            train_bars,
            index_by_timestamp,
            train_end,
        )
        if record is None:
            immature_event_count += 1
        else:
            records.append(record)
    if not records:
        raise ValueError("v4af_meta_oof_labels_empty")

    labelled = (
        DataFrame.from_records(records)
        .sort_values("event_index")
        .reset_index(drop=True)
    )
    _validate_training_label_frame(labelled, train_end)
    diagnostics = _training_label_diagnostics(
        labelled,
        len(primary_events),
        immature_event_count,
        train_end,
    )
    return labelled, diagnostics


def _join_event_features(
    features: DataFrame,
    labelled_events: DataFrame,
) -> DataFrame:
    indices = labelled_events["event_index"].to_numpy(dtype=int)
    if (indices < 0).any() or (indices >= len(features)).any():
        raise ValueError("v4af_meta_feature_event_index_out_of_bounds")
    selected = features.iloc[indices].reset_index(drop=True)
    if selected.empty or selected.isna().any(axis=None):
        raise ValueError("v4af_meta_training_features_invalid")
    selected.insert(0, "event_index", indices)
    training = labelled_events.merge(
        selected,
        on="event_index",
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    if len(training) != len(labelled_events):
        raise ValueError("v4af_meta_training_feature_alignment_invalid")
    if training.isna().any(axis=None):
        raise ValueError("v4af_meta_training_matrix_contains_null")
    return training.reset_index(drop=True)


def _calibrate_thresholds(
    calibration_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    values = np.asarray(calibration_probabilities, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("v4af_calibration_probabilities_invalid")
    results: list[dict[str, Any]] = []
    for fraction in TARGET_ACCEPT_FRACTIONS:
        diagnostics = select_dynamic_buy_threshold(
            values,
            target_buy_fraction=fraction,
        )
        results.append(
            {
                "target_accept_fraction": fraction,
                "selected_threshold": float(diagnostics["threshold"]),
                "method": str(diagnostics["method"]),
                "comparison": str(diagnostics["comparison"]),
                "target_percentile": float(diagnostics["target_percentile"]),
                "calibration_rows": int(diagnostics["calibration_rows"]),
                "target_selected_rows": int(diagnostics["target_selected_rows"]),
                "selected_calibration_rows": int(
                    diagnostics["selected_calibration_rows"]
                ),
                "selected_calibration_fraction": float(
                    diagnostics["selected_calibration_fraction"]
                ),
                "threshold_tie_rows": int(diagnostics["threshold_tie_rows"]),
                "tie_expansion_rows": int(diagnostics["tie_expansion_rows"]),
                "calibration_labels_used": False,
            }
        )
    return results


def _validation_probability_by_index(
    features: DataFrame,
    event_indices: list[int],
    model: Any,
) -> dict[int, float]:
    if not event_indices:
        return {}
    event_features = features.iloc[event_indices].reset_index(drop=True)
    probabilities = predict_meta_probabilities(model, event_features)
    return {
        index: float(probability)
        for index, probability in zip(event_indices, probabilities, strict=True)
    }


def _decision_rows(
    bars: DataFrame,
    event_indices: list[int],
    probability_by_index: dict[int, float] | None,
    candidate_id: str,
    mode: str,
    threshold: float | None,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for index in event_indices:
        probability = (
            1.0 if probability_by_index is None else probability_by_index.get(index)
        )
        accepted = (
            True
            if threshold is None
            else probability is not None and probability >= threshold
        )
        decisions.append(
            {
                "candidate_id": candidate_id,
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "mode": mode,
                "event_index": index,
                "signal_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
                "execution_timestamp": pd.Timestamp(
                    bars["date"].iloc[index + 1]
                ).isoformat(),
                "primary_signal": "BUY",
                "meta_probability": probability,
                "selected_threshold": threshold,
                "meta_filter_decision": "ACCEPT" if accepted else "REJECT",
                "executable_signal": "BUY" if accepted else "NO_ENTRY",
                "split_role": "FORWARD_VALIDATION",
            }
        )
    return decisions


def _candidate_acceptance_met(metrics: dict[str, Any]) -> bool:
    trade_count = int(metrics["trade_count"])
    net_pf = metrics["net_profit_factor"]
    net_pf_infinite = bool(metrics["net_profit_factor_infinite_no_losses"])
    drawdown = float(metrics["max_drawdown_percent"])
    pf_passed = net_pf_infinite or (
        net_pf is not None
        and math.isfinite(float(net_pf))
        and float(net_pf) > MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE
    )
    return (
        trade_count > MINIMUM_TRADES_EXCLUSIVE
        and pf_passed
        and math.isfinite(drawdown)
        and drawdown <= MAXIMUM_DRAWDOWN_INCLUSIVE
    )


def _run_economic_candidate(
    *,
    bars: DataFrame,
    event_indices: list[int],
    candidate_id: str,
    mode: str,
    threshold: float | None,
    target_accept_fraction: float | None,
    probability_by_index: dict[int, float] | None,
    forward_end: int,
    calibration_diagnostics: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_indices = (
        list(event_indices)
        if threshold is None
        else [
            index
            for index in event_indices
            if probability_by_index is not None
            and probability_by_index.get(index) is not None
            and float(probability_by_index[index]) >= threshold
        ]
    )
    actions = _build_forward_actions(bars, accepted_indices, forward_end)
    replay = simulate_signals(
        bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(threshold),
    )
    metrics = _summarize_replay(replay)
    decisions = _decision_rows(
        bars,
        event_indices,
        probability_by_index,
        candidate_id,
        mode,
        threshold,
    )
    trades = [
        {
            "candidate_id": candidate_id,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "mode": mode,
            "target_accept_fraction": target_accept_fraction,
            "selected_meta_probability_threshold": threshold,
            "initial_stop_atr": INITIAL_STOP_ATR,
            "trailing_activation_atr": TRAILING_ACTIVATION_ATR,
            "trailing_step_atr": TRAILING_STEP_ATR,
            "take_profit_atr": TAKE_PROFIT_ATR,
            **trade,
        }
        for trade in replay["trades"]
    ]
    candidate = {
        "candidate_id": candidate_id,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": mode,
        "status": "COMPLETED",
        "target_accept_fraction": target_accept_fraction,
        "selected_meta_probability_threshold": threshold,
        "threshold_calibration": calibration_diagnostics,
        "primary_buy_event_count": len(event_indices),
        "meta_accepted_buy_count": len(accepted_indices),
        "meta_rejected_buy_count": len(event_indices) - len(accepted_indices),
        "achieved_validation_accept_fraction": (
            len(accepted_indices) / float(len(event_indices)) if event_indices else 0.0
        ),
        "metrics": metrics,
        "acceptance_target_met": _candidate_acceptance_met(metrics),
        "sequence_model_used": False,
    }
    return candidate, trades, decisions


def _skipped_meta_candidate(
    target_accept_fraction: float,
    reason: str,
) -> dict[str, Any]:
    candidate_id = (
        f"{SYMBOL}|{TIMEFRAME}|SL{INITIAL_STOP_ATR:.1f}"
        f"|ACT{TRAILING_ACTIVATION_ATR:.1f}|STEP{TRAILING_STEP_ATR:.1f}"
        f"|META_TOP_{int(target_accept_fraction * 100)}"
    )
    return {
        "candidate_id": candidate_id,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "mode": MODE,
        "status": "SKIPPED_INSUFFICIENT_META_SUPPORT",
        "skip_reason": reason,
        "target_accept_fraction": target_accept_fraction,
        "selected_meta_probability_threshold": None,
        "threshold_calibration": None,
        "primary_buy_event_count": 0,
        "meta_accepted_buy_count": 0,
        "meta_rejected_buy_count": 0,
        "achieved_validation_accept_fraction": None,
        "metrics": {
            "trade_count": 0,
            "gross_profit_factor": None,
            "gross_profit_factor_infinite_no_losses": False,
            "net_profit_factor": None,
            "net_profit_factor_infinite_no_losses": False,
            "max_drawdown_percent": None,
            "net_pnl_total_usd": 0.0,
            "win_rate": None,
        },
        "acceptance_target_met": False,
        "sequence_model_used": False,
    }


def _candidate_id(mode: str, target_accept_fraction: float | None = None) -> str:
    geometry = (
        f"SL{INITIAL_STOP_ATR:.1f}_ACT{TRAILING_ACTIVATION_ATR:.1f}"
        f"_STEP{TRAILING_STEP_ATR:.1f}"
    )
    if mode == CONTROL_MODE:
        return f"{SYMBOL}|{TIMEFRAME}|{geometry}|{CONTROL_MODE}"
    if target_accept_fraction is None:
        raise ValueError("v4af_meta_candidate_fraction_required")
    return (
        f"{SYMBOL}|{TIMEFRAME}|{geometry}|META_TOP_{int(target_accept_fraction * 100)}"
    )


def _format_pf(value: Any, infinite: bool) -> str:
    if infinite:
        return "INF"
    return "N/A" if value is None else f"{float(value):.4f}"


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _step_summary_header(report: dict[str, Any]) -> list[str]:
    return [
        "## AliX V4-AF ETHUSDT Meta-Model on Fixed Trailing Geometry",
        "",
        f"Commit SHA: {report['commit_sha']}",
        f"Source V4-AE SHA: {report['v4ae_reference']['commit_sha']}",
        "",
        "Scope: ETHUSDT only; 1h only; 730-day public Binance Spot data.",
        (
            "Primary signal: unchanged causal Bollinger(20, 2.0) upper-band breakout "
            "with Close > EMA200; closed-candle signal, next-bar-open execution."
        ),
        (
            "Fixed exits: initial SL=2.0 ATR, activation=1.5 ATR, trailing step=1.5 ATR, "
            "take-profit=4.0 ATR, horizon=24 bars."
        ),
        (
            "The owner-selected Step=1.5 geometry is an explicit Owner Override. "
            "V4-AE's historical highest cell was Step=2.0 (Net PF=1.2477), not Step=1.5."
        ),
        (
            "Costs: 10 bps maker fee + 5 bps slippage per side; 30 bps round trip. "
            "Spot long-only, 1x equity, no pyramiding."
        ),
        (
            "Meta labels: isolated OOF BUY replays using the exact V4-AF trailing geometry "
            "and costs; positive label means net PnL > 0 after fees and slippage."
        ),
        (
            "Model: LightGBM binary classifier, trained on matured chronological OOF "
            "primary BUY events; purged fit/calibration split; calibration labels are unused."
        ),
        (
            "Thresholds: top 30%, 50%, and 70% of calibration probabilities, "
            "selected by order statistic with >= comparison and tie expansion reported."
        ),
        (
            "Chronological split: 70% train / 15% forward validation / final 15% untouched. "
            "No validation or final-holdout labels are used for fitting or threshold selection."
        ),
        "",
        (
            "| Mode | Target Top % | Threshold | Cal. Accepted % | Primary BUYs | "
            "Meta Accepted | Trades | Gross PF | Net PF | Max DD % | Net PnL USD | Acceptance | Status |"
        ),
        ("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"),
    ]


def _step_summary_candidate_row(item: dict[str, Any]) -> str:
    metrics = item["metrics"]
    calibration = item.get("threshold_calibration") or {}
    raw_threshold = item.get("selected_meta_probability_threshold")
    threshold_text = "N/A" if raw_threshold is None else f"{float(raw_threshold):.6f}"
    calibration_fraction = calibration.get("selected_calibration_fraction")
    target_fraction = item.get("target_accept_fraction")
    target_percent = None if target_fraction is None else float(target_fraction) * 100.0
    accepted_percent = (
        None if calibration_fraction is None else float(calibration_fraction) * 100.0
    )
    return (
        f"| {item['mode']} "
        f"| {_format_percent(target_percent)} "
        f"| {threshold_text} "
        f"| {_format_percent(accepted_percent)} "
        f"| {item.get('primary_buy_event_count', 0)} "
        f"| {item.get('meta_accepted_buy_count', 0)} "
        f"| {metrics.get('trade_count', 0)} "
        f"| {_format_pf(metrics.get('gross_profit_factor'), bool(metrics.get('gross_profit_factor_infinite_no_losses')))} "
        f"| {_format_pf(metrics.get('net_profit_factor'), bool(metrics.get('net_profit_factor_infinite_no_losses')))} "
        f"| {_format_percent(metrics.get('max_drawdown_percent'))} "
        f"| {_format_percent(metrics.get('net_pnl_total_usd'))} "
        f"| {'PASS' if item.get('acceptance_target_met') else 'NO'} "
        f"| {item['status']} |"
    )


def _step_summary_footer(report: dict[str, Any]) -> list[str]:
    meta = report["meta_model"]
    training = meta.get("training_diagnostics") or {}
    fit = meta.get("fit_diagnostics") or {}
    accepted_count = sum(
        bool(item.get("acceptance_target_met")) for item in report["candidates"]
    )
    return [
        "",
        f"Candidate rows: {len(report['candidates'])}; model fit performed: {meta['model_fit_performed']}.",
        (
            f"OOF labels: {training.get('matured_costed_oof_label_count', 0)}; "
            f"fit rows: {fit.get('fit_rows', 0)}; calibration rows: {fit.get('calibration_rows', 0)}."
        ),
        (f"Rows meeting Trades >30, Net PF >1.5, and Max DD <=15%: {accepted_count}."),
        "",
        (
            "V4-AE selected-geometry reference (Step=1.5): 42 trades, Net PF=1.1538, "
            "Max DD=9.44%. V4-AE historical best (Step=2.0): 42 trades, "
            "Net PF=1.2477, Max DD=8.97%."
        ),
        "",
        (
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0, sequence_model_used=false."
        ),
        "Research only; no private API, credentials, orders, or execution API are used.",
        (
            "Promotion status: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW. "
            "No automatic Champion selection."
        ),
    ]


def _render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4af_step_summary_report_schema_invalid")
    lines = _step_summary_header(report)
    lines.extend(_step_summary_candidate_row(item) for item in report["candidates"])
    lines.extend(_step_summary_footer(report))
    return "\n".join(lines) + "\n"


def _validate_report_identity(
    report: dict[str, Any],
    commit_sha: str,
) -> None:
    if report.get("schema_version") != SCHEMA or report.get("commit_sha") != commit_sha:
        raise ValueError("v4af_report_identity_invalid")


def _validate_report_scope(report: dict[str, Any]) -> None:
    if report.get("symbols") != [SYMBOL] or report.get("timeframes") != [TIMEFRAME]:
        raise ValueError("v4af_report_scope_invalid")
    if report.get("mode") != MODE or len(report.get("candidates", [])) != 4:
        raise ValueError("v4af_candidate_matrix_invalid")


def _validate_report_holdout(report: dict[str, Any]) -> None:
    chronological = report.get("chronological_evaluation")
    if (
        report.get("final_holdout_used") is not False
        or not isinstance(chronological, dict)
        or chronological.get("final_holdout_used") is not False
        or chronological.get("final_15_percent_holdout_untouched") is not True
    ):
        raise ValueError("v4af_final_holdout_must_remain_untouched")


def _validate_report_thresholds(report: dict[str, Any]) -> None:
    fractions = [
        item.get("target_accept_fraction")
        for item in report["candidates"]
        if item.get("mode") == MODE
    ]
    if fractions != list(TARGET_ACCEPT_FRACTIONS):
        raise ValueError("v4af_meta_threshold_matrix_invalid")


def _validate_report_safety(report: dict[str, Any]) -> None:
    safety = report["safety"]
    safety_is_closed = (
        safety.get("Champion") is None
        and safety.get("order_permission") is False
        and safety.get("trading_authorization") == "NO-GO"
        and safety.get("private_api_calls") == 0
        and safety.get("real_order_attempts") == 0
        and safety.get("sequence_model_used") is False
    )
    if not safety_is_closed:
        raise ValueError("v4af_fail_closed_safety_contract_invalid")


def _validate_report_candidates(report: dict[str, Any]) -> None:
    for item in report["candidates"]:
        candidate_scope_is_valid = (
            item.get("symbol") == SYMBOL and item.get("timeframe") == TIMEFRAME
        )
        if not candidate_scope_is_valid:
            raise ValueError("v4af_candidate_scope_invalid")
        if item.get("sequence_model_used") is not False:
            raise ValueError("v4af_sequence_model_forbidden")
        if item.get("status") not in {
            "COMPLETED",
            "SKIPPED_INSUFFICIENT_META_SUPPORT",
        }:
            raise ValueError("v4af_candidate_status_invalid")


def _validate_report(report: dict[str, Any], commit_sha: str) -> None:
    _validate_report_identity(report, commit_sha)
    _validate_report_scope(report)
    _validate_report_holdout(report)
    _validate_report_thresholds(report)
    _validate_report_safety(report)
    _validate_report_candidates(report)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    required_columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(required_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safety_report() -> dict[str, Any]:
    return {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "credentials_used": False,
        "short_positions_supported": False,
        "sequence_model_used": False,
    }


def _build_report(
    commit_sha: str,
    provenance: dict[str, Any],
    bars: DataFrame,
    train_end: int,
    forward_end: int,
    primary_signal_counts: dict[str, int],
    primary_sample_gate: dict[str, Any],
    training_diagnostics: dict[str, Any],
    fit_diagnostics: dict[str, Any],
    threshold_diagnostics: list[dict[str, Any]],
    model_fit_performed: bool,
    candidates: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": DAYS,
        "provider": "Binance Spot public klines API",
        "input_dataset": provenance,
        "symbols": [SYMBOL],
        "timeframes": [TIMEFRAME],
        "mode": MODE,
        "primary_model": {
            "generation": V4Y_PRIMARY_SIGNAL_GENERATION,
            "bollinger_period": 20,
            "bollinger_standard_deviations": 2.0,
            "ema_trend_period": V4Y_EMA_TREND_PERIOD,
            "meta_feature_columns": list(V4Y_META_FEATURE_COLUMNS),
        },
        "fixed_exit_geometry": {
            "initial_stop_atr": INITIAL_STOP_ATR,
            "trailing_activation_atr": TRAILING_ACTIVATION_ATR,
            "trailing_step_atr": TRAILING_STEP_ATR,
            "take_profit_atr": TAKE_PROFIT_ATR,
            "atr_period": ATR_PERIOD_VALUE,
            "horizon_bars": HORIZON_BARS_VALUE,
            "mechanics": (
                "trailing_step_atr is the stop ratchet increment after activation; "
                "the new stop becomes effective on the next bar."
            ),
        },
        "execution_cost_contract": {
            "fee_type_assumption": "MAKER",
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "initial_equity_usd": 10_000.0,
            "position_fraction": 1.0,
            "spot_long_only": True,
            "pyramiding": False,
        },
        "meta_model": {
            "enabled": True,
            "model_family": "LightGBM LGBMClassifier objective=binary",
            "model_fit_performed": model_fit_performed,
            "positive_class": 1,
            "negative_class": 0,
            "target": (
                "NET_PNL_POSITIVE_AFTER_FEES_AND_SLIPPAGE_FROM_EXACT_V4AF_TRAILING_REPLAY"
            ),
            "oof_fold_count": V4Y_OOF_FOLD_COUNT,
            "minimum_primary_train_signals": V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
            "minimum_fit_rows": V4Y_MINIMUM_FIT_ROWS,
            "minimum_rows_per_class": V4Y_MINIMUM_CLASS_ROWS,
            "calibration_fraction": V4Y_META_CALIBRATION_FRACTION,
            "target_accept_fractions": list(TARGET_ACCEPT_FRACTIONS),
            "threshold_selection": "TOP_PERCENTILE_ORDER_STATISTIC_ON_PURGED_CALIBRATION_PROBABILITIES",
            "calibration_labels_used": False,
            "validation_labels_used_for_threshold": False,
            "validation_probabilities_used_for_threshold": False,
            "training_diagnostics": training_diagnostics,
            "fit_diagnostics": fit_diagnostics,
            "threshold_diagnostics": threshold_diagnostics,
            "sequence_model_used": False,
        },
        "chronological_evaluation": {
            "train_fraction": TRAIN_FRACTION,
            "forward_validation_end_fraction": FORWARD_END_FRACTION,
            "outer_train_end_index": train_end,
            "forward_validation_end_exclusive": forward_end,
            "validation_start_index": train_end,
            "final_holdout_start_index": forward_end,
            "final_holdout_used": False,
            "final_15_percent_holdout_untouched": True,
            "validation_labels_used_for_fit": False,
            "validation_labels_used_for_threshold": False,
            "all_validation_trades_close_before_holdout": True,
        },
        "primary_signal_counts": primary_signal_counts,
        "primary_sample_gate": primary_sample_gate,
        "v4ae_reference": V4AE_REFERENCE,
        "acceptance_targets": {
            "trade_count_exclusive_minimum": MINIMUM_TRADES_EXCLUSIVE,
            "net_profit_factor_exclusive_minimum": MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE,
            "maximum_drawdown_inclusive": MAXIMUM_DRAWDOWN_INCLUSIVE,
            "purpose": "research_target_only_no_automatic_promotion",
        },
        "final_holdout_used": False,
        "candidates": candidates,
        "trades": trades,
        "decisions": decisions,
        "safety": _safety_report(),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
    }
    _validate_report(report, commit_sha)
    return report


def _load_research_context(
    input_dir: Path,
    days: int,
    commit_sha: str,
) -> tuple[dict[str, Any], DataFrame, int, int, DataFrame, DataFrame, Series]:
    source_bars, provenance = _load_manifested_data(
        input_dir,
        SYMBOL,
        days,
        commit_sha,
    )
    bars = resample_ohlcv(source_bars, TIMEFRAME)
    train_end, forward_end = _outer_boundaries(len(bars))
    research_bars = bars.iloc[:forward_end].reset_index(drop=True)
    profiles = build_feature_profiles(
        research_bars,
        profile_names=(FEATURE_PROFILE,),
    )
    if FEATURE_PROFILE not in profiles:
        raise ValueError("v4af_primary_feature_profile_missing")
    features, primary_buy_signal = build_v4y_primary_features(
        research_bars,
        profiles[FEATURE_PROFILE],
    )
    if not set(V4Y_META_FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("v4af_meta_feature_schema_invalid")
    return (
        provenance,
        bars,
        train_end,
        forward_end,
        research_bars,
        features,
        primary_buy_signal,
    )


def _primary_signal_evaluation(
    primary_buy_signal: Series,
    train_end: int,
    forward_end: int,
) -> tuple[dict[str, int], dict[str, Any], list[int]]:
    train_gate = evaluate_primary_train_sample_gate(
        primary_buy_signal,
        train_start_index=V4Y_EMA_TREND_PERIOD,
        train_end_index=train_end,
    )
    signal_counts = {
        "research_window_primary_buy_count": int(primary_buy_signal.sum()),
        "train_primary_buy_count": int(train_gate["observed_primary_buy_count"]),
        "forward_validation_primary_buy_count": int(
            primary_buy_signal.iloc[train_end:forward_end].sum()
        ),
    }
    event_indices = _eligible_primary_event_indices(
        primary_buy_signal,
        train_end,
        forward_end,
    )
    return signal_counts, train_gate, event_indices


def _run_unfiltered_control(
    research_bars: DataFrame,
    event_indices: list[int],
    forward_end: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return _run_economic_candidate(
        bars=research_bars,
        event_indices=event_indices,
        candidate_id=_candidate_id(CONTROL_MODE),
        mode=CONTROL_MODE,
        threshold=None,
        target_accept_fraction=None,
        probability_by_index=None,
        forward_end=forward_end,
        calibration_diagnostics=None,
    )


def _meta_oof_training_frame(
    research_bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: Series,
    train_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    oof_events = generate_v4y_oof_primary_events(
        primary_buy_signal,
        start_index=V4Y_EMA_TREND_PERIOD,
        stop_index=train_end,
        fold_count=V4Y_OOF_FOLD_COUNT,
    )
    labelled_events, training_diagnostics = _training_oof_labels(
        research_bars,
        oof_events,
        train_end,
    )
    return _join_event_features(features, labelled_events), training_diagnostics


def _meta_training_skip_reason(error: ValueError) -> str | None:
    code = str(error)
    if code == "v4x_oof_primary_signals_empty":
        return code
    if code.startswith("v4af_meta_oof_labels_"):
        return code
    if code.startswith("v4x_insufficient_meta_"):
        return code
    return None


def _prepare_oof_training(
    research_bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: Series,
    train_end: int,
    train_gate: dict[str, Any],
) -> dict[str, Any]:
    try:
        training, diagnostics = _meta_oof_training_frame(
            research_bars,
            features,
            primary_buy_signal,
            train_end,
        )
    except ValueError as error:
        skip_reason = _meta_training_skip_reason(error)
        if skip_reason is None:
            raise
        return {
            "skip_reason": skip_reason,
            "training_diagnostics": {"primary_sample_gate": train_gate},
        }
    diagnostics["primary_sample_gate"] = train_gate
    return {
        "skip_reason": None,
        "training": training,
        "training_diagnostics": diagnostics,
    }


def _prepare_purged_meta_split(
    training: DataFrame,
    train_end: int,
    training_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    try:
        fit_rows, calibration_rows = split_meta_fit_calibration(
            training,
            outer_train_end_index=train_end,
            calibration_fraction=V4Y_META_CALIBRATION_FRACTION,
            minimum_fit_rows=V4Y_MINIMUM_FIT_ROWS,
            minimum_class_rows=V4Y_MINIMUM_CLASS_ROWS,
        )
    except ValueError as error:
        skip_reason = _meta_training_skip_reason(error)
        if skip_reason is None:
            raise
        return {
            "skip_reason": skip_reason,
            "training_diagnostics": training_diagnostics,
        }
    return {
        "skip_reason": None,
        "fit_rows": fit_rows,
        "calibration_rows": calibration_rows,
        "training_diagnostics": training_diagnostics,
    }


def _prepare_meta_training(
    research_bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: Series,
    train_end: int,
    train_gate: dict[str, Any],
) -> dict[str, Any]:
    if not bool(train_gate["passed"]):
        return {
            "skip_reason": "v4y_insufficient_primary_signals",
            "training_diagnostics": {"primary_sample_gate": train_gate},
        }
    oof_preparation = _prepare_oof_training(
        research_bars,
        features,
        primary_buy_signal,
        train_end,
        train_gate,
    )
    if oof_preparation["skip_reason"] is not None:
        return oof_preparation
    return _prepare_purged_meta_split(
        oof_preparation["training"],
        train_end,
        oof_preparation["training_diagnostics"],
    )


def _fit_and_calibrate_meta_model(
    features: DataFrame,
    fit_rows: DataFrame,
    calibration_rows: DataFrame,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    feature_columns = list(features.columns)
    model, fit_diagnostics = fit_lightgbm_meta_filter(
        fit_rows.loc[:, feature_columns],
        fit_rows["meta_label"],
        calibration_rows.loc[:, feature_columns],
        target_accept_fraction=0.50,
    )
    calibration_probabilities = predict_meta_probabilities(
        model,
        calibration_rows.loc[:, feature_columns],
    )
    threshold_diagnostics = _calibrate_thresholds(calibration_probabilities)
    return model, fit_diagnostics, threshold_diagnostics


def _run_meta_threshold_candidates(
    research_bars: DataFrame,
    features: DataFrame,
    event_indices: list[int],
    forward_end: int,
    model: Any,
    threshold_diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation_probabilities = _validation_probability_by_index(
        features,
        event_indices,
        model,
    )
    candidates: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for calibration in threshold_diagnostics:
        fraction = float(calibration["target_accept_fraction"])
        threshold = float(calibration["selected_threshold"])
        candidate, candidate_trades, candidate_decisions = _run_economic_candidate(
            bars=research_bars,
            event_indices=event_indices,
            candidate_id=_candidate_id(MODE, fraction),
            mode=MODE,
            threshold=threshold,
            target_accept_fraction=fraction,
            probability_by_index=validation_probabilities,
            forward_end=forward_end,
            calibration_diagnostics=calibration,
        )
        candidates.append(candidate)
        trades.extend(candidate_trades)
        decisions.extend(candidate_decisions)
    return candidates, trades, decisions


def _skipped_meta_candidates(reason: str) -> list[dict[str, Any]]:
    return [
        _skipped_meta_candidate(fraction, reason)
        for fraction in TARGET_ACCEPT_FRACTIONS
    ]


def _run_meta_experiment(
    research_bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: Series,
    train_end: int,
    forward_end: int,
    event_indices: list[int],
    train_gate: dict[str, Any],
) -> dict[str, Any]:
    prepared = _prepare_meta_training(
        research_bars,
        features,
        primary_buy_signal,
        train_end,
        train_gate,
    )
    training_diagnostics = prepared["training_diagnostics"]
    if prepared["skip_reason"] is not None:
        skip_reason = str(prepared["skip_reason"])
        return {
            "training_diagnostics": training_diagnostics,
            "fit_diagnostics": {},
            "threshold_diagnostics": [],
            "model_fit_performed": False,
            "candidates": _skipped_meta_candidates(skip_reason),
            "trades": [],
            "decisions": [],
        }

    model, fit_diagnostics, threshold_diagnostics = _fit_and_calibrate_meta_model(
        features,
        prepared["fit_rows"],
        prepared["calibration_rows"],
    )
    candidates, trades, decisions = _run_meta_threshold_candidates(
        research_bars,
        features,
        event_indices,
        forward_end,
        model,
        threshold_diagnostics,
    )
    return {
        "training_diagnostics": training_diagnostics,
        "fit_diagnostics": fit_diagnostics,
        "threshold_diagnostics": threshold_diagnostics,
        "model_fit_performed": True,
        "candidates": candidates,
        "trades": trades,
        "decisions": decisions,
    }


def run_v4af(
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
        features,
        primary_buy_signal,
    ) = _load_research_context(input_dir, days, commit_sha)
    signal_counts, train_gate, event_indices = _primary_signal_evaluation(
        primary_buy_signal,
        train_end,
        forward_end,
    )
    control, control_trades, control_decisions = _run_unfiltered_control(
        research_bars,
        event_indices,
        forward_end,
    )
    meta = _run_meta_experiment(
        research_bars,
        features,
        primary_buy_signal,
        train_end,
        forward_end,
        event_indices,
        train_gate,
    )
    candidates = [control, *meta["candidates"]]
    trades = [*control_trades, *meta["trades"]]
    decisions = [*control_decisions, *meta["decisions"]]
    report = _build_report(
        commit_sha,
        provenance,
        bars,
        train_end,
        forward_end,
        signal_counts,
        train_gate,
        meta["training_diagnostics"],
        meta["fit_diagnostics"],
        meta["threshold_diagnostics"],
        bool(meta["model_fit_performed"]),
        candidates,
        trades,
        decisions,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    trade_columns = (
        "candidate_id",
        "symbol",
        "timeframe",
        "mode",
        "target_accept_fraction",
        "selected_meta_probability_threshold",
        "direction",
        "entry_timestamp",
        "exit_timestamp",
        "net_pnl_usd",
        "net_return_fraction",
        "exit_reason",
    )
    decision_columns = (
        "candidate_id",
        "symbol",
        "timeframe",
        "mode",
        "event_index",
        "signal_timestamp",
        "execution_timestamp",
        "meta_probability",
        "selected_threshold",
        "meta_filter_decision",
        "executable_signal",
        "split_role",
    )
    _write_csv(output_trades_csv, trades, trade_columns)
    _write_csv(output_decisions_csv, decisions, decision_columns)
    print(_render_step_summary(report), end="")
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
    report = run_v4af(
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
