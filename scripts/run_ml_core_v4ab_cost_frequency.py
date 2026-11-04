"""Evaluate V4-Y Primary entries across fixed horizons and exit structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    SYMBOLS,
    _load_diagnostic_datasets,
    resample_ohlcv,
)
from services.meta_labeling.v4y import build_v4y_primary_features
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    simulate_signals,
)

SCHEMA = "alix.ml-core.v4ab-cost-frequency.v1"
TIMEFRAME = "1h"
HORIZONS = (24, 72, 120)
EXIT_POLICIES = ("TB_1.5_1.0", "TIME_N", "TRAIL_2.0")
MAX_HORIZON = max(HORIZONS)
ATR_PERIOD = 14
INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 25.0
SLIPPAGE_BPS_PER_SIDE = 20.0
TRAIN_FRACTION = 0.70
FORWARD_END_FRACTION = 0.85
TIME_EXIT_PROTECTIVE_ATR = 10.0
TIME_EXIT_MFE_ATR = 1_000_000.0


def _validate_run_request(days: int, commit_sha: str) -> None:
    if days != 730:
        raise ValueError("v4ab_dataset_window_must_be_730_days")
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("v4ab_commit_sha_invalid")


def _outer_boundaries(row_count: int) -> tuple[int, int]:
    if row_count < 1_000:
        raise ValueError("v4ab_resampled_dataset_too_short")
    train_end = int(row_count * TRAIN_FRACTION)
    forward_end = int(row_count * FORWARD_END_FRACTION)
    if (
        train_end <= 200
        or forward_end - train_end <= MAX_HORIZON + 2
        or forward_end >= row_count
    ):
        raise ValueError("v4ab_forward_window_configuration_invalid")
    return train_end, forward_end


def _primary_signal_series(bars: DataFrame) -> Series:
    """Delegate to the exact causal V4-Y closed-candle signal implementation."""
    _, signals = build_v4y_primary_features(bars, DataFrame(index=range(len(bars))))
    if len(signals) != len(bars) or signals.dtype != bool:
        raise ValueError("v4ab_primary_signal_alignment_invalid")
    return signals


def _forward_primary_signal_indices(
    signals: Series, train_end: int, forward_end: int
) -> list[int]:
    """Use one common 120-bar maturity cutoff for every cell of a symbol."""
    if len(signals) < forward_end or not 0 <= train_end < forward_end:
        raise ValueError("v4ab_primary_signal_window_invalid")
    last_eligible = forward_end - MAX_HORIZON - 2
    if last_eligible < train_end:
        raise ValueError("v4ab_forward_window_has_no_matured_events")
    return [
        index
        for index in range(train_end, last_eligible + 1)
        if bool(signals.iloc[index])
    ]


def _signal_record(bars: DataFrame, index: int, signal: str) -> dict[str, Any]:
    if index < 0 or index + 1 >= len(bars):
        raise ValueError("v4ab_signal_execution_index_invalid")
    signal_time = pd.Timestamp(bars["date"].iloc[index])
    execution_time = pd.Timestamp(bars["date"].iloc[index + 1])
    if signal_time.tzinfo is None or execution_time.tzinfo is None:
        raise ValueError("v4ab_signal_timestamp_timezone_required")
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
    """Build BUY actions plus a final NO_ENTRY action to close the full window."""
    if forward_end > len(bars) or forward_end < 2:
        raise ValueError("v4ab_forward_action_window_invalid")
    if event_indices != sorted(set(event_indices)):
        raise ValueError("v4ab_primary_events_must_be_unique_and_ordered")
    actions = [_signal_record(bars, i, "BUY") for i in event_indices]
    if any(i + 1 >= forward_end for i in event_indices):
        raise ValueError("v4ab_primary_event_outside_forward_window")
    sentinel = forward_end - 2
    if sentinel in event_indices:
        raise ValueError("v4ab_forward_sentinel_collides_with_primary_event")
    actions.append(_signal_record(bars, sentinel, "NO_ENTRY"))
    return actions


def _execution_config() -> ExecutionConfig:
    """Use the unchanged V4-Y 1x Spot fee/slippage contract."""
    return ExecutionConfig(
        initial_equity=INITIAL_EQUITY,
        fee_bps=FEE_BPS_PER_SIDE,
        slippage_bps=SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=0.99,
    )


def _trade_management_config(policy: str, horizon: int) -> TradeManagementConfig:
    if horizon not in HORIZONS:
        raise ValueError("v4ab_horizon_out_of_policy")
    if policy == "TB_1.5_1.0":
        return TradeManagementConfig(
            atr_period=ATR_PERIOD,
            take_profit_atr_multiplier=1.5,
            stop_loss_atr_multiplier=1.0,
            horizon_candles=horizon,
            trailing_stop_enabled=False,
        )
    if policy == "TIME_N":
        return TradeManagementConfig(
            atr_period=ATR_PERIOD,
            take_profit_atr_multiplier=TIME_EXIT_PROTECTIVE_ATR,
            stop_loss_atr_multiplier=TIME_EXIT_PROTECTIVE_ATR,
            horizon_candles=horizon + 1,
            trailing_stop_enabled=False,
            time_exit_candles=horizon,
            time_exit_mfe_threshold_atr=TIME_EXIT_MFE_ATR,
        )
    if policy == "TRAIL_2.0":
        return TradeManagementConfig(
            atr_period=ATR_PERIOD,
            take_profit_atr_multiplier=4.0,
            stop_loss_atr_multiplier=2.0,
            horizon_candles=horizon,
            trailing_stop_enabled=True,
            trailing_activation_atr=2.0,
            trailing_step_atr=1.0,
        )
    raise ValueError("v4ab_exit_policy_out_of_policy")


def _profit_factor(pnls: list[float]) -> tuple[float | None, bool]:
    profit = sum(value for value in pnls if value > 0)
    loss = abs(sum(value for value in pnls if value < 0))
    if loss > 0:
        return round(profit / loss, 8), False
    return None, profit > 0


def _pf_above(value: float | None, infinite: bool, threshold: float) -> bool:
    return infinite or (value is not None and value > threshold)


def _time_exit_trade_is_valid(trade: dict[str, Any], horizon: int) -> bool:
    return (
        trade["exit_reason"] == "TIME_EXIT"
        and trade["time_exit_triggered"] is True
        and int(trade["bars_held"]) - 1 == horizon
    )


def _time_exit_contract_violations(
    trades: list[dict[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    return [item for item in trades if not _time_exit_trade_is_valid(item, horizon)]


def _time_exit_violation_details(
    violations: list[dict[str, Any]], horizon: int
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for trade in violations:
        reason = str(trade.get("exit_reason", "UNKNOWN"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    samples = [
        {
            "entry_timestamp": item.get("entry_timestamp"),
            "exit_timestamp": item.get("exit_timestamp"),
            "exit_reason": item.get("exit_reason"),
            "exit_trigger_type": item.get("exit_trigger_type"),
            "time_exit_triggered": item.get("time_exit_triggered"),
            "bars_held": int(item["bars_held"]),
            "expected_bars_held": horizon + 1,
        }
        for item in violations[:5]
    ]
    return {
        "invalid_trade_count": len(violations),
        "exit_reason_counts": reason_counts,
        "sample_invalid_trades": samples,
    }


def _validate_replay_trades(
    trades: list[dict[str, Any]], policy: str, horizon: int
) -> None:
    if any(item["direction"] != "LONG" for item in trades):
        raise ValueError("v4ab_non_spot_trade_detected")
    if policy == "TIME_N" and _time_exit_contract_violations(trades, horizon):
        raise ValueError("v4ab_unconditional_time_exit_contract_failed")
    if policy != "TIME_N" and any(int(item["bars_held"]) > horizon for item in trades):
        raise ValueError("v4ab_trade_exceeded_registered_horizon")


def _validated_pnl_values(
    trades: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    gross = [float(item["gross_market_pnl_usd"]) for item in trades]
    net = [float(item["net_pnl_usd"]) for item in trades]
    if not all(math.isfinite(value) for value in (*gross, *net)):
        raise ValueError("v4ab_trade_pnl_non_finite")
    return gross, net


def _holding_periods(trades: list[dict[str, Any]], policy: str) -> list[int]:
    return [
        int(item["bars_held"]) - 1
        if policy == "TIME_N" and item["exit_reason"] == "TIME_EXIT"
        else int(item["bars_held"])
        for item in trades
    ]


def _validated_drawdown(replay: dict[str, Any]) -> float:
    drawdown = float(replay["metrics"]["max_drawdown_percent"])
    if not math.isfinite(drawdown) or drawdown < 0:
        raise ValueError("v4ab_candidate_drawdown_invalid")
    return round(drawdown, 8)


def _summarize_replay(
    replay: dict[str, Any], policy: str, horizon: int
) -> dict[str, Any]:
    trades = replay["trades"]
    _validate_replay_trades(trades, policy, horizon)
    gross, net = _validated_pnl_values(trades)
    gross_pf, gross_inf = _profit_factor(gross)
    net_pf, net_inf = _profit_factor(net)
    holding = _holding_periods(trades, policy)
    count = len(trades)
    return {
        "trade_count": count,
        "gross_profit_factor": gross_pf,
        "gross_profit_factor_infinite_no_losses": gross_inf,
        "net_profit_factor": net_pf,
        "net_profit_factor_infinite_no_losses": net_inf,
        "win_rate": round(sum(value > 0 for value in net) / count, 8) if count else 0.0,
        "gross_win_rate": round(sum(value > 0 for value in gross) / count, 8)
        if count
        else 0.0,
        "median_bars_held": round(float(median(holding)), 4) if holding else None,
        "max_drawdown_percent": _validated_drawdown(replay),
        "gross_pnl_total_usd": round(sum(gross), 8),
        "net_pnl_total_usd": round(sum(net), 8),
        "fees_total_usd": round(sum(float(item["fees_usd"]) for item in trades), 8),
        "slippage_total_usd": round(
            sum(float(item["slippage_cost_usd"]) for item in trades), 8
        ),
    }


def _skipped_time_exit_cell(
    symbol: str,
    raw_signal_count: int,
    eligible_signal_count: int,
    horizon: int,
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    pair = f"{symbol.removesuffix('USDT')}/USDT"
    candidate_id = f"{symbol}|{TIMEFRAME}|{horizon}|TIME_N"
    return {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "pair": pair,
        "timeframe": TIMEFRAME,
        "horizon_bars": horizon,
        "exit_policy": "TIME_N",
        "status": "SKIPPED_WARNING",
        "primary_signal_count": raw_signal_count,
        "eligible_primary_signal_count": eligible_signal_count,
        "trade_count": None,
        "gross_profit_factor": None,
        "gross_profit_factor_infinite_no_losses": False,
        "net_profit_factor": None,
        "net_profit_factor_infinite_no_losses": False,
        "win_rate": None,
        "gross_win_rate": None,
        "median_bars_held": None,
        "max_drawdown_percent": None,
        "gross_pnl_total_usd": None,
        "net_pnl_total_usd": None,
        "fees_total_usd": None,
        "slippage_total_usd": None,
        "warning_code": "v4ab_unconditional_time_exit_contract_failed",
        "warning_message": (
            "TIME_N produced at least one trade that closed before the "
            "registered next-open time exit; the cell is excluded from PF "
            "and PnL comparisons. Other cells continue."
        ),
        "warning_details": _time_exit_violation_details(violations, horizon),
    }


def _run_cell(
    symbol: str,
    bars: DataFrame,
    actions: list[dict[str, Any]],
    raw_signal_count: int,
    eligible_signal_count: int,
    horizon: int,
    policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay = simulate_signals(
        bars,
        actions,
        _execution_config(),
        trade_management_config=_trade_management_config(policy, horizon),
    )
    try:
        metrics = _summarize_replay(replay, policy, horizon)
    except ValueError as error:
        if (
            policy != "TIME_N"
            or str(error) != "v4ab_unconditional_time_exit_contract_failed"
        ):
            raise
        violations = _time_exit_contract_violations(replay["trades"], horizon)
        if not violations:
            raise
        skipped = _skipped_time_exit_cell(
            symbol,
            raw_signal_count,
            eligible_signal_count,
            horizon,
            violations,
        )
        print(
            "WARNING: "
            + json.dumps(
                {
                    "candidate_id": skipped["candidate_id"],
                    "status": skipped["status"],
                    "warning_code": skipped["warning_code"],
                    "warning_details": skipped["warning_details"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return skipped, []

    pair = f"{symbol.removesuffix('USDT')}/USDT"
    candidate_id = f"{symbol}|{TIMEFRAME}|{horizon}|{policy}"
    cell = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "pair": pair,
        "timeframe": TIMEFRAME,
        "horizon_bars": horizon,
        "exit_policy": policy,
        "status": "COMPLETED",
        "primary_signal_count": raw_signal_count,
        "eligible_primary_signal_count": eligible_signal_count,
        **metrics,
    }
    trades = [
        {
            "candidate_id": candidate_id,
            "symbol": symbol,
            "pair": pair,
            "timeframe": TIMEFRAME,
            "horizon_bars": horizon,
            "exit_policy": policy,
            "trade_index": index,
            **trade,
        }
        for index, trade in enumerate(replay["trades"], start=1)
    ]
    return cell, trades


def _cell_has_gross_edge(item: dict[str, Any]) -> bool:
    return _pf_above(
        item["gross_profit_factor"],
        bool(item["gross_profit_factor_infinite_no_losses"]),
        1.0,
    )


def _cell_meets_net_oos_rule(item: dict[str, Any]) -> bool:
    return int(item["trade_count"]) >= 30 and _pf_above(
        item["net_profit_factor"],
        bool(item["net_profit_factor_infinite_no_losses"]),
        1.3,
    )


def _cell_has_cost_drag(item: dict[str, Any]) -> bool:
    net_pf = item["net_profit_factor"]
    return _cell_has_gross_edge(item) and net_pf is not None and float(net_pf) < 1.0


def _decision_branch(
    gross_edge: list[dict[str, Any]],
    net_oos: list[dict[str, Any]],
    cost_drag: list[dict[str, Any]],
) -> tuple[str, str]:
    if net_oos:
        return (
            "NET_PF_GT_1_3_WITH_AT_LEAST_30_TRADES_FLAG_FOR_INDEPENDENT_OOS",
            "Flag qualifying cells for independent OOS confirmation; "
            "prioritize Meta-labeling only after confirmation.",
        )
    if not gross_edge:
        return (
            "NO_GROSS_PF_ABOVE_1_HALT_META_AND_PIVOT_PRIMARY_GENERATOR",
            "Halt Meta-model work; investigate a different Primary generator "
            "in a separate research phase.",
        )
    if cost_drag:
        return (
            "GROSS_EDGE_BUT_NET_PF_BELOW_1_COST_FREQUENCY_OPTIMIZATION",
            "Prioritize cost/frequency optimization; do not retrain Meta.",
        )
    return (
        "INTERMEDIATE_RESULT_NOT_EXPLICITLY_COVERED_BY_REGISTERED_RULE",
        "Preserve results and request an Owner decision; do not change "
        "thresholds or resume Meta-model training.",
    )


def _cell_is_completed(item: dict[str, Any]) -> bool:
    return item.get("status", "COMPLETED") == "COMPLETED"


def _cell_is_skipped(item: dict[str, Any]) -> bool:
    return item.get("status") == "SKIPPED_WARNING"


def _decision_rule(cells: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in cells if _cell_is_completed(item)]
    skipped = [item for item in cells if _cell_is_skipped(item)]
    gross_edge = [item for item in completed if _cell_has_gross_edge(item)]
    net_oos = [item for item in completed if _cell_meets_net_oos_rule(item)]
    cost_drag = [item for item in gross_edge if _cell_has_cost_drag(item)]

    if net_oos:
        branch, action = _decision_branch(gross_edge, net_oos, cost_drag)
        if skipped:
            action += (
                " The qualifying OOS cell is observed, but skipped cells remain "
                "explicitly unresolved."
            )
    elif skipped:
        branch = "INCOMPLETE_MATRIX_DUE_TO_SKIPPED_CELL_NO_FINAL_DECISION"
        action = (
            "No final Halt-Meta or Cost-Optimization decision is permitted: "
            "at least one TIME_N cell was skipped and may change the registered "
            "matrix-wide conclusion. Resolve the skipped cell before selecting "
            "a final decision branch."
        )
    else:
        branch, action = _decision_branch(gross_edge, net_oos, cost_drag)

    return {
        "decision_branch": branch,
        "action": action,
        "gross_edge_cell_ids": [item["candidate_id"] for item in gross_edge],
        "net_oos_confirmation_cell_ids": [item["candidate_id"] for item in net_oos],
        "cost_drag_cell_ids": [item["candidate_id"] for item in cost_drag],
        "skipped_cell_ids": [item["candidate_id"] for item in skipped],
        "completed_cell_count": len(completed),
        "skipped_cell_count": len(skipped),
        "matrix_complete": not skipped and len(completed) == 18,
    }


def _build_report(
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        f"{symbol}|{TIMEFRAME}|{horizon}|{policy}"
        for symbol in SYMBOLS
        for horizon in HORIZONS
        for policy in EXIT_POLICIES
    }
    if len(cells) != 18 or {str(item["candidate_id"]) for item in cells} != expected:
        raise ValueError("v4ab_candidate_matrix_incomplete")
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "symbols": list(SYMBOLS),
        "timeframes": [TIMEFRAME],
        "horizons_bars": list(HORIZONS),
        "exit_policies": list(EXIT_POLICIES),
        "primary_model": {
            "definition": "CAUSAL_BOLLINGER_UPPER_BREAKOUT_EMA200_TREND",
            "bollinger_period": 20,
            "bollinger_standard_deviations": 2.0,
            "trend_filter": "CLOSE_ABOVE_EMA200",
            "decision": "CLOSED_CANDLE",
            "execution": "NEXT_BAR_OPEN",
            "model_fit_performed": False,
            "meta_model_used": False,
            "sequence_model_used": False,
        },
        "evaluation_window": {
            "method": "CHRONOLOGICAL_70_15_15",
            "common_event_window": True,
            "candidate_windows": windows,
            "last_15_percent_final_holdout_used": False,
            "common_window_rule": (
                "Exclude the final 122 bars of the 70%-85% Forward interval "
                "so every 120-bar trade and next-open exit matures in-window."
            ),
        },
        "execution_cost_contract": {
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": 2 * (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE),
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
                "vertical_horizon": "CELL_HORIZON",
            },
            "TIME_N": {
                "exit": "UNCONDITIONAL_NEXT_BAR_OPEN_AFTER_N_COMPLETED_BARS",
                "protective_terminal_barriers_atr": TIME_EXIT_PROTECTIVE_ATR,
                "fail_closed_check": "EVERY_TRADE_EXIT_REASON_TIME_EXIT_AND_BARS_HELD_MINUS_ONE_EQUALS_N",
            },
            "TRAIL_2.0": {
                "initial_stop_atr": 2.0,
                "trailing_activation_atr": 2.0,
                "trailing_step_atr": 1.0,
                "terminal_take_profit_atr": 4.0,
                "vertical_horizon": "CELL_HORIZON",
            },
        },
        "cells": cells,
        "decision_rule": _decision_rule(cells),
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


def _format_pf(value: Any, infinite: bool) -> str:
    if infinite:
        return "INF"
    return "N/A" if value is None else f"{float(value):.4f}"


def render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ab_step_summary_report_schema_invalid")
    lines = [
        "## AliX V4-AB Cost-Frequency Diagnostic Sweep",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Primary: unchanged V4-Y Bollinger(20, 2.0) upper-band breakout with Close > EMA200.",
        "Decision timing: closed candle; execution: next-bar open.",
        "Scope: BTCUSDT / ETHUSDT; 1h; horizons 24 / 72 / 120 bars.",
        "Forward: common 70%-85% interval; final 122 bars excluded for maximum-horizon maturity; final 15% holdout untouched.",
        "Costs: 25 bps fee + 20 bps slippage per side; 90 bps round trip.",
        "Win Rate = executed trades with positive Net PnL after costs / executed trades.",
        "SKIPPED_WARNING means the cell has no valid PF/PnL result; N/A is not zero trades.",
        "",
        "| Symbol | Horizon | Exit | Primary BUYs (raw) | Eligible BUYs | Trades | Gross PF | Net PF | Win Rate | Median Bars Held | Max DD | Gross PnL USD | Net PnL USD | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["cells"]:
        if _cell_is_skipped(item):
            details = item["warning_details"]
            reason_text = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(details["exit_reason_counts"].items())
            )
            lines.append(
                f"| {item['symbol']} | {item['horizon_bars']} | TIME_N "
                f"| {item['primary_signal_count']} | {item['eligible_primary_signal_count']} "
                f"| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A "
                f"| SKIPPED_WARNING ({details['invalid_trade_count']} invalid; {reason_text}) |"
            )
            continue
        gross_pf = _format_pf(
            item["gross_profit_factor"],
            bool(item["gross_profit_factor_infinite_no_losses"]),
        )
        net_pf = _format_pf(
            item["net_profit_factor"],
            bool(item["net_profit_factor_infinite_no_losses"]),
        )
        median_bars = (
            "N/A"
            if item["median_bars_held"] is None
            else f"{float(item['median_bars_held']):.2f}"
        )
        lines.append(
            f"| {item['symbol']} | {item['horizon_bars']} | {item['exit_policy']} "
            f"| {item['primary_signal_count']} | {item['eligible_primary_signal_count']} "
            f"| {item['trade_count']} | {gross_pf} | {net_pf} "
            f"| {float(item['win_rate']):.2%} | {median_bars} "
            f"| {float(item['max_drawdown_percent']):.2f}% "
            f"| {float(item['gross_pnl_total_usd']):.2f} "
            f"| {float(item['net_pnl_total_usd']):.2f} | COMPLETED |"
        )
    decision = report["decision_rule"]
    lines.extend(
        [
            "",
            f"### Pre-registered decision: {decision['decision_branch']}",
            decision["action"],
            "",
            f"Completed cells: {decision['completed_cell_count']}/18.",
            f"Skipped warning cells: {decision['skipped_cell_count']}.",
            f"Gross-edge cells: {len(decision['gross_edge_cell_ids'])}.",
            f"Net OOS-priority cells (Net PF > 1.3 and trades >= 30): {len(decision['net_oos_confirmation_cell_ids'])}.",
            f"Cost-drag cells (Gross PF > 1.0 and Net PF < 1.0): {len(decision['cost_drag_cell_ids'])}.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only; no private API, credentials, order, or execution API is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_trade_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_v4ab(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_run_request(days, commit_sha)
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in datasets:
            raise ValueError("v4ab_required_symbol_dataset_missing")
        bars = resample_ohlcv(datasets[symbol], TIMEFRAME)
        primary = _primary_signal_series(bars)
        train_end, forward_end = _outer_boundaries(len(bars))
        event_indices = _forward_primary_signal_indices(primary, train_end, forward_end)
        raw_count = int(primary.iloc[train_end:forward_end].sum())
        actions = _build_forward_actions(bars, event_indices, forward_end)
        windows.append(
            {
                "symbol": symbol,
                "row_count": len(bars),
                "outer_train_end_index": train_end,
                "forward_end_exclusive": forward_end,
                "raw_forward_primary_signal_count": raw_count,
                "eligible_primary_signal_count": len(event_indices),
            }
        )
        for horizon in HORIZONS:
            for policy in EXIT_POLICIES:
                cell, rows = _run_cell(
                    symbol,
                    bars,
                    actions,
                    raw_count,
                    len(event_indices),
                    horizon,
                    policy,
                )
                cells.append(cell)
                trades.extend(rows)

    report = _build_report(commit_sha, days, provenance, cells, windows)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_trade_csv(output_trades_csv, trades)
    print(render_step_summary(report), end="")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4ab(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["cells"]) == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
