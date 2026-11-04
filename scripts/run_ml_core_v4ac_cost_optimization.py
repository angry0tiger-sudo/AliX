"""V4-AC cost optimization with matched Primary-only and OOF Meta-filtered replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    SYMBOLS,
    TIMEFRAME_MINUTES,
    _load_diagnostic_datasets,
    resample_ohlcv,
)
from services.meta_labeling.v4y import (
    V4Y_EMA_TREND_PERIOD,
    V4Y_META_CALIBRATION_FRACTION,
    V4Y_OOF_FOLD_COUNT,
    V4Y_PRIMARY_SIGNAL_GENERATION,
    V4Y_TARGET_ACCEPT_FRACTION,
    build_v4y_primary_features,
    evaluate_primary_train_sample_gate,
    fit_lightgbm_meta_filter,
    generate_v4y_oof_primary_events,
    predict_meta_probabilities,
    split_meta_fit_calibration,
)
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    simulate_signals,
)
from services.ml_core.extended_features import build_feature_profiles

SCHEMA = "alix.ml-core.v4ac-cost-optimization.v1"
TIMEFRAMES = ("15m", "30m", "1h", "4h")
HORIZON_BARS = 24
EXIT_POLICIES = ("TB_1.5_1.0", "TRAIL_2.0")
MODES = ("PRIMARY_ONLY", "META_FILTERED")
FEATURE_PROFILE = "Combined"
ATR_PERIOD = 14
INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ENTRY_SLIPPAGE_BPS = 5.0
ROUND_TRIP_COST_BPS = 30.0
TRAIN_FRACTION = 0.70
FORWARD_END_FRACTION = 0.85
MAXIMUM_DRAWDOWN = 0.99
META_MINIMUM_TRAIN_SIGNALS = 50
META_TRAINING_CONTEXT_BARS = ATR_PERIOD * 2


def _validate_timeframe(timeframe: str) -> None:
    if timeframe not in TIMEFRAMES or timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("v4ac_timeframe_out_of_policy")


def _validate_run_request(days: int, commit_sha: str) -> None:
    if days != 730:
        raise ValueError("v4ac_dataset_window_must_be_730_days")
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("v4ac_commit_sha_invalid")
    if set(TIMEFRAMES) != {"15m", "30m", "1h", "4h"}:
        raise ValueError("v4ac_timeframe_matrix_invalid")
    if any(TIMEFRAME_MINUTES[item] > 240 for item in TIMEFRAMES):
        raise ValueError("v4ac_timeframe_above_four_hours")


def _outer_boundaries(row_count: int) -> tuple[int, int]:
    if row_count < 1_000:
        raise ValueError("v4ac_resampled_dataset_too_short")
    train_end = int(row_count * TRAIN_FRACTION)
    forward_end = int(row_count * FORWARD_END_FRACTION)
    if (
        train_end <= V4Y_EMA_TREND_PERIOD + V4Y_OOF_FOLD_COUNT
        or forward_end - train_end <= HORIZON_BARS + 2
        or forward_end >= row_count
    ):
        raise ValueError("v4ac_forward_window_configuration_invalid")
    return train_end, forward_end


def _primary_features_and_signals(bars: DataFrame) -> tuple[DataFrame, Series]:
    profiles = build_feature_profiles(bars, profile_names=(FEATURE_PROFILE,))
    if FEATURE_PROFILE not in profiles:
        raise ValueError("v4ac_primary_feature_profile_missing")
    features, signals = build_v4y_primary_features(bars, profiles[FEATURE_PROFILE])
    if len(features) != len(bars) or len(signals) != len(bars) or signals.dtype != bool:
        raise ValueError("v4ac_primary_signal_alignment_invalid")
    if not features.columns.is_unique:
        raise ValueError("v4ac_meta_feature_columns_not_unique")
    return features, signals


def _signal_record(bars: DataFrame, index: int, signal: str) -> dict[str, Any]:
    if index < 0 or index + 1 >= len(bars):
        raise ValueError("v4ac_signal_execution_index_invalid")
    signal_time = pd.Timestamp(bars["date"].iloc[index])
    execution_time = pd.Timestamp(bars["date"].iloc[index + 1])
    if signal_time.tzinfo is None or execution_time.tzinfo is None:
        raise ValueError("v4ac_signal_timestamp_timezone_required")
    return {
        "row_index": index,
        "signal_timestamp": signal_time.isoformat(),
        "execution_timestamp": execution_time.isoformat(),
        "signal": signal,
        "raw_signal": signal,
        "buy_probability": 1.0,
    }


def _build_forward_actions(
    bars: DataFrame, event_indices: list[int], forward_end: int
) -> list[dict[str, Any]]:
    if forward_end > len(bars) or forward_end < 2:
        raise ValueError("v4ac_forward_action_window_invalid")
    if event_indices != sorted(set(event_indices)):
        raise ValueError("v4ac_primary_events_must_be_unique_and_ordered")
    actions = [_signal_record(bars, index, "BUY") for index in event_indices]
    if any(index + 1 >= forward_end for index in event_indices):
        raise ValueError("v4ac_primary_event_outside_forward_window")
    sentinel = forward_end - 2
    if sentinel in event_indices:
        raise ValueError("v4ac_forward_sentinel_collides_with_primary_event")
    actions.append(_signal_record(bars, sentinel, "NO_ENTRY"))
    return actions


def _forward_primary_signal_indices(
    signals: Series, train_end: int, forward_end: int
) -> list[int]:
    if len(signals) < forward_end or not 0 <= train_end < forward_end:
        raise ValueError("v4ac_primary_signal_window_invalid")
    last_eligible = forward_end - HORIZON_BARS - 2
    if last_eligible < train_end:
        raise ValueError("v4ac_forward_window_has_no_matured_events")
    return [
        index
        for index in range(train_end, last_eligible + 1)
        if bool(signals.iloc[index])
    ]


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_equity=INITIAL_EQUITY,
        fee_bps=FEE_BPS_PER_SIDE,
        slippage_bps=SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=MAXIMUM_DRAWDOWN,
    )


def _trade_management_config(policy: str) -> TradeManagementConfig:
    if policy == "TB_1.5_1.0":
        return TradeManagementConfig(
            atr_period=ATR_PERIOD,
            take_profit_atr_multiplier=1.5,
            stop_loss_atr_multiplier=1.0,
            horizon_candles=HORIZON_BARS,
            trailing_stop_enabled=False,
        )
    if policy == "TRAIL_2.0":
        return TradeManagementConfig(
            atr_period=ATR_PERIOD,
            take_profit_atr_multiplier=4.0,
            stop_loss_atr_multiplier=2.0,
            horizon_candles=HORIZON_BARS,
            trailing_stop_enabled=True,
            trailing_activation_atr=2.0,
            trailing_step_atr=1.0,
        )
    raise ValueError("v4ac_exit_policy_out_of_policy")


def _profit_factor(pnls: list[float]) -> tuple[float | None, bool]:
    profit = sum(value for value in pnls if value > 0)
    loss = abs(sum(value for value in pnls if value < 0))
    if loss > 0:
        return round(profit / loss, 8), False
    return None, profit > 0


def _summarize_replay(replay: dict[str, Any]) -> dict[str, Any]:
    trades = replay["trades"]
    if any(item["direction"] != "LONG" for item in trades):
        raise ValueError("v4ac_non_spot_trade_detected")
    gross = [float(item["gross_market_pnl_usd"]) for item in trades]
    net = [float(item["net_pnl_usd"]) for item in trades]
    if not all(math.isfinite(value) for value in (*gross, *net)):
        raise ValueError("v4ac_trade_pnl_non_finite")
    gross_pf, gross_inf = _profit_factor(gross)
    net_pf, net_inf = _profit_factor(net)
    count = len(trades)
    drawdown = float(replay["metrics"]["max_drawdown_percent"])
    if not math.isfinite(drawdown) or drawdown < 0:
        raise ValueError("v4ac_candidate_drawdown_invalid")
    holding = [int(item["bars_held"]) for item in trades]
    return {
        "trade_count": count,
        "gross_profit_factor": gross_pf,
        "gross_profit_factor_infinite_no_losses": gross_inf,
        "net_profit_factor": net_pf,
        "net_profit_factor_infinite_no_losses": net_inf,
        "win_rate": round(sum(value > 0 for value in net) / count, 8) if count else 0.0,
        "median_bars_held": round(float(median(holding)), 4) if holding else None,
        "max_drawdown_percent": round(drawdown, 8),
        "gross_pnl_total_usd": round(sum(gross), 8),
        "net_pnl_total_usd": round(sum(net), 8),
        "fees_total_usd": round(sum(float(item["fees_usd"]) for item in trades), 8),
        "slippage_total_usd": round(
            sum(float(item["slippage_cost_usd"]) for item in trades), 8
        ),
    }


def _empty_metric_values() -> dict[str, Any]:
    return {
        "trade_count": None,
        "gross_profit_factor": None,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": None,
        "net_profit_factor_infinite_no_losses": False,
        "win_rate": None,
        "median_bars_held": None,
        "max_drawdown_percent": None,
        "gross_pnl_total_usd": None,
        "net_pnl_total_usd": None,
        "fees_total_usd": None,
        "slippage_total_usd": None,
    }


def _single_event_trade(
    bars: DataFrame, event_index: int, policy: str
) -> tuple[dict[str, Any], int]:
    start = max(0, event_index - META_TRAINING_CONTEXT_BARS)
    end = event_index + HORIZON_BARS + 2
    if end > len(bars):
        raise ValueError("v4ac_training_event_did_not_mature")
    local_bars = bars.iloc[start:end].reset_index(drop=True)
    local_event = event_index - start
    actions = _build_forward_actions(
        local_bars, [local_event], local_event + HORIZON_BARS + 2
    )
    replay = simulate_signals(
        local_bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(policy),
    )
    trades = replay["trades"]
    if len(trades) != 1:
        raise ValueError("v4ac_meta_training_event_trade_count_invalid")
    timestamp_to_index = {
        pd.Timestamp(value): index for index, value in enumerate(local_bars["date"])
    }
    exit_timestamp = pd.Timestamp(trades[0]["exit_timestamp"])
    exit_index = timestamp_to_index.get(exit_timestamp)
    if exit_index is None or exit_index <= local_event:
        raise ValueError("v4ac_meta_training_exit_index_invalid")
    return trades[0], start + exit_index


def _meta_training_rows(
    bars: DataFrame,
    features: DataFrame,
    primary_signals: Series,
    train_end: int,
    policy: str,
) -> tuple[DataFrame, dict[str, Any]]:
    oof = generate_v4y_oof_primary_events(
        primary_signals,
        start_index=V4Y_EMA_TREND_PERIOD,
        stop_index=train_end,
        fold_count=V4Y_OOF_FOLD_COUNT,
    )
    latest_mature_event = train_end - HORIZON_BARS - 2
    eligible = oof.loc[oof["event_index"] <= latest_mature_event].copy()
    records: list[dict[str, Any]] = []
    for item in eligible.to_dict(orient="records"):
        event_index = int(item["event_index"])
        trade, label_end_index = _single_event_trade(bars, event_index, policy)
        if label_end_index >= train_end:
            raise ValueError("v4ac_meta_training_label_crosses_outer_train")
        records.append(
            {
                "event_index": event_index,
                "fold_id": int(item["fold_id"]),
                "label_end_idx": label_end_index,
                "meta_label": int(float(trade["net_pnl_usd"]) > 0.0),
                "primary_signal": "BUY",
                "primary_oof": True,
                "exit_reason": str(trade["exit_reason"]),
                "net_return": float(trade["net_return_fraction"]),
            }
        )
    if not records:
        raise ValueError("v4ac_meta_training_events_empty")
    event_rows = DataFrame.from_records(records).sort_values("event_index")
    indices = event_rows["event_index"].to_numpy(dtype=int)
    selected_features = features.iloc[indices].reset_index(drop=True)
    if selected_features.isna().any(axis=None):
        raise ValueError("v4ac_meta_training_features_non_finite")
    training = pd.concat([event_rows.reset_index(drop=True), selected_features], axis=1)
    diagnostics = {
        "oof_event_count": int(len(oof)),
        "matured_oof_event_count": int(len(training)),
        "latest_mature_event_index": latest_mature_event,
        "all_primary_events_are_buy": bool(training["primary_signal"].eq("BUY").all()),
        "all_primary_events_are_oof": bool(training["primary_oof"].eq(True).all()),
        "maximum_label_end_index": int(training["label_end_idx"].max()),
        "labels_matured_before_outer_train_end": bool(
            int(training["label_end_idx"].max()) < train_end
        ),
        "label_exit_policy": policy,
        "label_cost_fee_bps_per_side": FEE_BPS_PER_SIDE,
        "label_cost_slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
    }
    return training, diagnostics


def _is_expected_meta_support_error(error: ValueError) -> bool:
    code = str(error)
    return code in {
        "v4x_oof_primary_signals_empty",
        "v4ac_meta_training_events_empty",
    } or code.startswith("v4x_insufficient_meta_")


def _train_meta_model(
    bars: DataFrame,
    features: DataFrame,
    primary_signals: Series,
    train_end: int,
    policy: str,
) -> tuple[Any | None, dict[str, Any], str]:
    gate = evaluate_primary_train_sample_gate(
        primary_signals,
        train_start_index=V4Y_EMA_TREND_PERIOD,
        train_end_index=train_end,
        minimum_signal_count=META_MINIMUM_TRAIN_SIGNALS,
    )
    if not bool(gate["passed"]):
        return None, {"sample_gate": gate}, "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"
    try:
        training, diagnostics = _meta_training_rows(
            bars, features, primary_signals, train_end, policy
        )
        fit_rows, calibration_rows = split_meta_fit_calibration(
            training, outer_train_end_index=train_end
        )
        feature_columns = list(features.columns)
        model, fit_diagnostics = fit_lightgbm_meta_filter(
            fit_rows.loc[:, feature_columns],
            fit_rows["meta_label"],
            calibration_rows.loc[:, feature_columns],
            target_accept_fraction=V4Y_TARGET_ACCEPT_FRACTION,
        )
    except ValueError as error:
        if not _is_expected_meta_support_error(error):
            raise
        return (
            None,
            {"sample_gate": gate, "skip_reason": str(error)},
            ("SKIPPED_INSUFFICIENT_META_SUPPORT"),
        )
    diagnostics.update(
        {
            "sample_gate": gate,
            "fit_rows": int(len(fit_rows)),
            "calibration_rows": int(len(calibration_rows)),
            "fit_max_label_end_index": int(fit_rows["label_end_idx"].max()),
            "calibration_start_event_index": int(calibration_rows["event_index"].min()),
            "fit_labels_purged_before_calibration": bool(
                int(fit_rows["label_end_idx"].max())
                < int(calibration_rows["event_index"].min())
            ),
            "validation_labels_used_for_fit": False,
            "validation_labels_used_for_threshold": False,
            "meta_model": fit_diagnostics,
        }
    )
    return model, diagnostics, "TRAINED"


def _forward_meta_probabilities(
    features: DataFrame, event_indices: list[int], model: Any
) -> dict[int, float]:
    if not event_indices:
        return {}
    event_features = features.iloc[event_indices].reset_index(drop=True)
    probabilities = predict_meta_probabilities(model, event_features)
    if len(probabilities) != len(event_indices):
        raise ValueError("v4ac_meta_probability_alignment_invalid")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("v4ac_meta_probability_values_invalid")
    return {
        index: float(probability)
        for index, probability in zip(event_indices, probabilities, strict=True)
    }


def _run_economic_cell(
    *,
    symbol: str,
    timeframe: str,
    policy: str,
    mode: str,
    bars: DataFrame,
    signal_indices: list[int],
    raw_signal_count: int,
    eligible_signal_count: int,
    forward_end: int,
    meta_status: str | None = None,
    meta_diagnostics: dict[str, Any] | None = None,
    meta_threshold: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_timeframe(timeframe)
    if mode not in MODES:
        raise ValueError("v4ac_mode_out_of_policy")
    candidate_id = f"{symbol}|{timeframe}|{HORIZON_BARS}|{policy}|{mode}"
    base = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "pair": f"{symbol.removesuffix('USDT')}/USDT",
        "timeframe": timeframe,
        "horizon_bars": HORIZON_BARS,
        "exit_policy": policy,
        "mode": mode,
        "primary_signal_count": raw_signal_count,
        "eligible_primary_signal_count": eligible_signal_count,
        "accepted_signal_count": len(signal_indices),
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "meta_status": meta_status,
        "meta_threshold": meta_threshold,
        "meta_diagnostics": meta_diagnostics or {},
    }
    if mode == "META_FILTERED" and meta_status != "TRAINED":
        return {
            **base,
            "status": meta_status or "SKIPPED_INSUFFICIENT_META_SUPPORT",
            "skip_reason": (meta_diagnostics or {}).get("skip_reason"),
            **_empty_metric_values(),
        }, []
    actions = _build_forward_actions(bars, signal_indices, forward_end)
    replay = simulate_signals(
        bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(policy),
    )
    metrics = _summarize_replay(replay)
    trade_rows = [
        {
            "candidate_id": candidate_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon_bars": HORIZON_BARS,
            "exit_policy": policy,
            "mode": mode,
            "trade_index": index,
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            **trade,
        }
        for index, trade in enumerate(replay["trades"], start=1)
    ]
    return {**base, "status": "COMPLETED", **metrics}, trade_rows


def _meta_skip_cell(
    *,
    symbol: str,
    timeframe: str,
    policy: str,
    raw_signal_count: int,
    eligible_signal_count: int,
    reason: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = f"{symbol}|{timeframe}|{HORIZON_BARS}|{policy}|META_FILTERED"
    return {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "pair": f"{symbol.removesuffix('USDT')}/USDT",
        "timeframe": timeframe,
        "horizon_bars": HORIZON_BARS,
        "exit_policy": policy,
        "mode": "META_FILTERED",
        "status": reason,
        "skip_reason": diagnostics.get("skip_reason", reason),
        "primary_signal_count": raw_signal_count,
        "eligible_primary_signal_count": eligible_signal_count,
        "accepted_signal_count": None,
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "meta_status": reason,
        "meta_threshold": None,
        "meta_diagnostics": diagnostics,
        **_empty_metric_values(),
    }


def _run_symbol_timeframe(
    symbol: str,
    source_bars: DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    symbol_windows: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        _validate_timeframe(timeframe)
        bars = resample_ohlcv(source_bars, timeframe)
        features, primary_signals = _primary_features_and_signals(bars)
        train_end, forward_end = _outer_boundaries(len(bars))
        primary_indices = _forward_primary_signal_indices(
            primary_signals, train_end, forward_end
        )
        raw_count = int(primary_signals.iloc[train_end:forward_end].sum())
        symbol_windows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "row_count": len(bars),
                "outer_train_end_index": train_end,
                "forward_end_exclusive": forward_end,
                "raw_forward_primary_signal_count": raw_count,
                "eligible_forward_primary_signal_count": len(primary_indices),
            }
        )
        for policy in EXIT_POLICIES:
            primary_cell, primary_trades = _run_economic_cell(
                symbol=symbol,
                timeframe=timeframe,
                policy=policy,
                mode="PRIMARY_ONLY",
                bars=bars,
                signal_indices=primary_indices,
                raw_signal_count=raw_count,
                eligible_signal_count=len(primary_indices),
                forward_end=forward_end,
            )
            cells.append(primary_cell)
            trades.extend(primary_trades)
            model, diagnostics, meta_status = _train_meta_model(
                bars, features, primary_signals, train_end, policy
            )
            if model is None:
                cells.append(
                    _meta_skip_cell(
                        symbol=symbol,
                        timeframe=timeframe,
                        policy=policy,
                        raw_signal_count=raw_count,
                        eligible_signal_count=len(primary_indices),
                        reason=meta_status,
                        diagnostics=diagnostics,
                    )
                )
                continue
            threshold = float(diagnostics["meta_model"]["selected_threshold"])
            probabilities = _forward_meta_probabilities(
                features, primary_indices, model
            )
            accepted = [
                index
                for index in primary_indices
                if probabilities.get(index, -1.0) >= threshold
            ]
            accepted_set = set(accepted)
            for index in primary_indices:
                probability = probabilities[index]
                decisions.append(
                    {
                        "candidate_id": (
                            f"{symbol}|{timeframe}|{HORIZON_BARS}|{policy}|META_FILTERED"
                        ),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "exit_policy": policy,
                        "split_role": "FORWARD_VALIDATION",
                        "row_index": index,
                        "timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
                        "primary_signal": "BUY",
                        "meta_probability": probability,
                        "selected_threshold": threshold,
                        "meta_filter_decision": "ACCEPT"
                        if index in accepted_set
                        else "REJECT",
                        "executable_signal": "BUY"
                        if index in accepted_set
                        else "NO_ENTRY",
                    }
                )
            meta_cell, meta_trades = _run_economic_cell(
                symbol=symbol,
                timeframe=timeframe,
                policy=policy,
                mode="META_FILTERED",
                bars=bars,
                signal_indices=accepted,
                raw_signal_count=raw_count,
                eligible_signal_count=len(primary_indices),
                forward_end=forward_end,
                meta_status=meta_status,
                meta_diagnostics=diagnostics,
                meta_threshold=threshold,
            )
            cells.append(meta_cell)
            trades.extend(meta_trades)
    return cells, trades, decisions, symbol_windows


def _format_pf(value: Any, infinite: bool) -> str:
    if infinite:
        return "INF"
    return "N/A" if value is None else f"{float(value):.4f}"


def render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ac_step_summary_report_schema_invalid")
    lines = [
        "## AliX V4-AC Cost Optimization & Meta Re-integration",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Primary: unchanged V4-Y causal Bollinger(20, 2.0) upper-band breakout with Close > EMA200.",
        "Decision: closed candle; execution: next-bar open. Spot long-only, 1x equity, no pyramiding.",
        "Timeframes: 15m / 30m / 1h / 4h only. Horizon: exactly 24 candles in every timeframe (6h / 12h / 24h / 96h respectively).",
        "Costs: 10 bps maker-fee assumption + 5 bps slippage assumption per side; 30 bps round trip.",
        "Important: fixed-bps next-open replay is a cost scenario, not a true limit-order queue, partial-fill, or non-fill simulator.",
        "Meta: separate OOF LightGBM model per symbol/timeframe/exit policy; policy-matched net-PnL labels; purged fit/calibration; validation labels never used.",
        "SKIPPED Meta rows have N/A metrics, not zero trades. Primary-only rows still run when Meta sample support is insufficient.",
        "",
        "| Symbol | TF | Horizon Bars | Exit | Mode | Fee bps/side | Slip bps/side | Primary BUYs | Eligible BUYs | Accepted BUYs | Trades | Gross PF | Net PF | Net PnL USD | Meta Threshold | Status |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["cells"]:
        if item["status"] != "COMPLETED":
            lines.append(
                f"| {item['symbol']} | {item['timeframe']} | {item['horizon_bars']} "
                f"| {item['exit_policy']} | {item['mode']} | {FEE_BPS_PER_SIDE:.1f} "
                f"| {SLIPPAGE_BPS_PER_SIDE:.1f} | {item['primary_signal_count']} "
                f"| {item['eligible_primary_signal_count']} | N/A | N/A | N/A | N/A | N/A "
                f"| N/A | {item['status']} |"
            )
            continue
        threshold = (
            "N/A"
            if item["meta_threshold"] is None
            else f"{float(item['meta_threshold']):.6f}"
        )
        lines.append(
            f"| {item['symbol']} | {item['timeframe']} | {item['horizon_bars']} "
            f"| {item['exit_policy']} | {item['mode']} | {FEE_BPS_PER_SIDE:.1f} "
            f"| {SLIPPAGE_BPS_PER_SIDE:.1f} | {item['primary_signal_count']} "
            f"| {item['eligible_primary_signal_count']} | {item['accepted_signal_count']} "
            f"| {item['trade_count']} "
            f"| {_format_pf(item['gross_profit_factor'], bool(item['gross_profit_factor_infinite_no_losses']))} "
            f"| {_format_pf(item['net_profit_factor'], bool(item['net_profit_factor_infinite_no_losses']))} "
            f"| {float(item['net_pnl_total_usd']):.2f} | {threshold} | COMPLETED |"
        )
    completed = sum(item["status"] == "COMPLETED" for item in report["cells"])
    skipped_meta = sum(
        item["mode"] == "META_FILTERED" and item["status"] != "COMPLETED"
        for item in report["cells"]
    )
    trained = sum(item["status"] == "TRAINED" for item in report["meta_scenarios"])
    lines.extend(
        [
            "",
            f"Candidate rows: {len(report['cells'])}/32; completed: {completed}; skipped Meta rows: {skipped_meta}.",
            f"Meta training scenarios: {trained}/{len(report['meta_scenarios'])} trained.",
            "This is a research comparison, not independent OOS promotion or profitability approval.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only; no private API, credentials, orders, or execution API are used.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_report(
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    meta_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        f"{symbol}|{timeframe}|{HORIZON_BARS}|{policy}|{mode}"
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
        for policy in EXIT_POLICIES
        for mode in MODES
    }
    observed = {str(item["candidate_id"]) for item in cells}
    if len(cells) != 32 or observed != expected:
        raise ValueError("v4ac_candidate_matrix_incomplete")
    if any(item["timeframe"] not in TIMEFRAMES for item in cells):
        raise ValueError("v4ac_report_timeframe_out_of_policy")
    if len(meta_scenarios) != 16:
        raise ValueError("v4ac_meta_scenario_matrix_incomplete")
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "symbols": list(SYMBOLS),
        "timeframes": list(TIMEFRAMES),
        "horizon_bars": HORIZON_BARS,
        "horizon_hours_by_timeframe": {
            timeframe: HORIZON_BARS * TIMEFRAME_MINUTES[timeframe] / 60
            for timeframe in TIMEFRAMES
        },
        "exit_policies": list(EXIT_POLICIES),
        "modes": list(MODES),
        "primary_model": {
            "definition": V4Y_PRIMARY_SIGNAL_GENERATION,
            "bollinger_period": 20,
            "bollinger_standard_deviations": 2.0,
            "trend_filter": "CLOSE_ABOVE_EMA200",
            "decision": "CLOSED_CANDLE",
            "execution": "NEXT_BAR_OPEN",
            "model_fit_performed": False,
            "signal_changed_from_v4ab": False,
            "sequence_model_used": False,
        },
        "meta_model": {
            "enabled": True,
            "family": "LightGBM LGBMClassifier objective=binary",
            "training_signal_requirement": "OOF_PRIMARY_BUY_ONLY",
            "training_label": "POLICY_MATCHED_NET_POSITIVE_AFTER_FEES_AND_SLIPPAGE",
            "separate_model_per_symbol_timeframe_exit_policy": True,
            "oof_fold_count": V4Y_OOF_FOLD_COUNT,
            "calibration_fraction": V4Y_META_CALIBRATION_FRACTION,
            "target_accept_fraction": V4Y_TARGET_ACCEPT_FRACTION,
            "minimum_primary_train_signals": META_MINIMUM_TRAIN_SIGNALS,
            "calibration_labels_used": False,
            "validation_labels_used_for_fit_or_threshold": False,
            "scenarios": meta_scenarios,
            "sequence_model_used": False,
        },
        "evaluation_window": {
            "method": "CHRONOLOGICAL_70_15_15",
            "common_event_window_per_symbol_timeframe": True,
            "candidate_windows": windows,
            "last_15_percent_final_holdout_used": False,
            "maximum_horizon_bars": HORIZON_BARS,
        },
        "execution_cost_contract": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "fee_type_assumption": "MAKER",
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "entry_slippage_bps": ENTRY_SLIPPAGE_BPS,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "assumption_only_not_limit_fill_simulation": True,
            "initial_equity_usd": INITIAL_EQUITY,
            "position_fraction": 1.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
        },
        "exit_contract": {
            "TB_1.5_1.0": {
                "take_profit_atr": 1.5,
                "stop_loss_atr": 1.0,
                "vertical_horizon_bars": HORIZON_BARS,
                "trailing_stop_enabled": False,
            },
            "TRAIL_2.0": {
                "initial_stop_atr": 2.0,
                "trailing_activation_atr": 2.0,
                "trailing_step_atr": 1.0,
                "terminal_take_profit_atr": 4.0,
                "vertical_horizon_bars": HORIZON_BARS,
            },
        },
        "cells": cells,
        "decision_rule": "OWNER_REVIEW_REQUIRED_NO_AUTOMATIC_PROMOTION",
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "credentials_used": False,
            "short_positions_supported": False,
            "sequence_model_used": False,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_v4ac(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_run_request(days, commit_sha)
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    meta_scenarios: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in datasets:
            raise ValueError("v4ac_required_symbol_dataset_missing")
        symbol_cells, symbol_trades, symbol_decisions, symbol_windows = (
            _run_symbol_timeframe(symbol, datasets[symbol])
        )
        cells.extend(symbol_cells)
        trades.extend(symbol_trades)
        decisions.extend(symbol_decisions)
        windows.extend(symbol_windows)
        for item in symbol_cells:
            if item["mode"] == "META_FILTERED":
                meta_scenarios.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "symbol": symbol,
                        "timeframe": item["timeframe"],
                        "exit_policy": item["exit_policy"],
                        "status": item["meta_status"] or item["status"],
                        "skip_reason": item.get("skip_reason"),
                        "diagnostics": item["meta_diagnostics"],
                    }
                )
    report = _build_report(commit_sha, days, provenance, cells, windows, meta_scenarios)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_trades_csv, trades)
    _write_csv(output_decisions_csv, decisions)
    print(render_step_summary(report), end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4ac(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["cells"]) == 32 else 1


if __name__ == "__main__":
    raise SystemExit(main())
