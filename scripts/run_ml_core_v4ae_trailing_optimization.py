"""V4-AE primary-only ATR trailing-stop grid on 1h public Spot data."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from scripts import run_ml_core_v4ac_cost_optimization as v4ac
from services.ml_core.economic_backtest import (
    TradeManagementConfig,
    simulate_signals,
)

SCHEMA = "alix.ml-core.v4ae-trailing-optimization.v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
TIMEFRAME = "1h"
HORIZON_BARS = 24
INITIAL_STOP_ATR_VALUES = (1.0, 1.5, 2.0)
TRAILING_ACTIVATION_ATR_VALUES = (0.5, 1.0, 1.5)
TRAILING_STEP_ATR_VALUES = (1.0, 1.5, 2.0)
TAKE_PROFIT_ATR = 4.0
MODE = "PRIMARY_ONLY"
MINIMUM_TRADES_EXCLUSIVE = 30
MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE = 1.5
MAXIMUM_DRAWDOWN_INCLUSIVE = 15.0
V4AD_BASELINE = {
    "commit_sha": "16f13e1a70898cf4790f3a990f653828aba0a0d4",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "exit_policy": "TRAIL_2.0_ACT1.5",
    "mode": "PRIMARY_ONLY",
    "accepted_signal_count": 52,
    "trade_count": 42,
    "net_profit_factor": 1.1351,
    "max_drawdown_percent": 9.85,
    "initial_stop_atr": 2.0,
    "trailing_activation_atr": 1.5,
    "trailing_step_atr": 1.0,
}


@dataclass(frozen=True)
class TrailingGridConfig:
    """One ATR-based trailing-stop configuration."""

    initial_stop_atr: float
    trailing_activation_atr: float
    trailing_step_atr: float


GRID = tuple(
    TrailingGridConfig(stop, activation, step)
    for stop in INITIAL_STOP_ATR_VALUES
    for activation in TRAILING_ACTIVATION_ATR_VALUES
    for step in TRAILING_STEP_ATR_VALUES
)


def _validate_grid_config(config: TrailingGridConfig) -> None:
    values = (
        config.initial_stop_atr,
        config.trailing_activation_atr,
        config.trailing_step_atr,
    )
    if any(
        isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0
        for value in values
    ):
        raise ValueError("v4ae_trailing_grid_parameter_invalid")
    if config.initial_stop_atr not in INITIAL_STOP_ATR_VALUES:
        raise ValueError("v4ae_initial_stop_out_of_policy")
    if config.trailing_activation_atr not in TRAILING_ACTIVATION_ATR_VALUES:
        raise ValueError("v4ae_trailing_activation_out_of_policy")
    if config.trailing_step_atr not in TRAILING_STEP_ATR_VALUES:
        raise ValueError("v4ae_trailing_step_out_of_policy")


def _parameter_token(value: float) -> str:
    return f"{value:.1f}"


def _policy_name(config: TrailingGridConfig) -> str:
    _validate_grid_config(config)
    return (
        f"TRAIL_SL{_parameter_token(config.initial_stop_atr)}"
        f"_ACT{_parameter_token(config.trailing_activation_atr)}"
        f"_STEP{_parameter_token(config.trailing_step_atr)}"
    )


def _candidate_id(symbol: str, config: TrailingGridConfig) -> str:
    if symbol not in SYMBOLS:
        raise ValueError("v4ae_symbol_out_of_policy")
    return f"{symbol}|{TIMEFRAME}|{HORIZON_BARS}|{_policy_name(config)}|{MODE}"


def _trade_management_config(config: TrailingGridConfig) -> TradeManagementConfig:
    _validate_grid_config(config)
    return TradeManagementConfig(
        atr_period=v4ac.ATR_PERIOD,
        take_profit_atr_multiplier=TAKE_PROFIT_ATR,
        stop_loss_atr_multiplier=config.initial_stop_atr,
        horizon_candles=HORIZON_BARS,
        trailing_stop_enabled=True,
        trailing_activation_atr=config.trailing_activation_atr,
        trailing_step_atr=config.trailing_step_atr,
    )


def _expected_candidate_ids() -> set[str]:
    return {
        _candidate_id(symbol, config) for symbol in SYMBOLS for config in GRID
    }


def _validate_candidate_matrix(cells: list[dict[str, Any]]) -> None:
    expected = _expected_candidate_ids()
    observed = {str(item.get("candidate_id")) for item in cells}
    if len(cells) != 54 or observed != expected:
        raise ValueError("v4ae_candidate_matrix_incomplete")
    for item in cells:
        if item.get("symbol") not in SYMBOLS or item.get("timeframe") != TIMEFRAME:
            raise ValueError("v4ae_candidate_scope_invalid")
        if item.get("mode") != MODE or item.get("status") != "COMPLETED":
            raise ValueError("v4ae_candidate_mode_or_status_invalid")
        config = TrailingGridConfig(
            item["initial_stop_atr"],
            item["trailing_activation_atr"],
            item["trailing_step_atr"],
        )
        _validate_grid_config(config)
        if item["exit_policy"] != _policy_name(config):
            raise ValueError("v4ae_candidate_exit_policy_mismatch")


def _candidate_acceptance_met(cell: dict[str, Any]) -> bool:
    trade_count = cell.get("trade_count")
    profit_factor = cell.get("net_profit_factor")
    drawdown = cell.get("max_drawdown_percent")
    pf_passed = bool(cell.get("net_profit_factor_infinite_no_losses")) or (
        profit_factor is not None
        and math.isfinite(float(profit_factor))
        and float(profit_factor) > MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE
    )
    return (
        trade_count is not None
        and int(trade_count) > MINIMUM_TRADES_EXCLUSIVE
        and pf_passed
        and drawdown is not None
        and math.isfinite(float(drawdown))
        and float(drawdown) <= MAXIMUM_DRAWDOWN_INCLUSIVE
    )


def _run_candidate(
    *,
    symbol: str,
    bars: DataFrame,
    signal_indices: list[int],
    raw_signal_count: int,
    forward_end: int,
    config: TrailingGridConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = _policy_name(config)
    candidate_id = _candidate_id(symbol, config)
    actions = v4ac._build_forward_actions(bars, signal_indices, forward_end)
    replay = simulate_signals(
        bars,
        actions,
        v4ac._execution_config(),
        trade_management_config=_trade_management_config(config),
    )
    metrics = v4ac._summarize_replay(replay)
    cell: dict[str, Any] = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "pair": f"{symbol.removesuffix('USDT')}/USDT",
        "timeframe": TIMEFRAME,
        "horizon_bars": HORIZON_BARS,
        "exit_policy": policy,
        "mode": MODE,
        "initial_stop_atr": config.initial_stop_atr,
        "trailing_activation_atr": config.trailing_activation_atr,
        "trailing_step_atr": config.trailing_step_atr,
        "take_profit_atr": TAKE_PROFIT_ATR,
        "primary_signal_count": raw_signal_count,
        "eligible_primary_signal_count": len(signal_indices),
        "accepted_signal_count": len(signal_indices),
        "fee_bps_per_side": v4ac.FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": v4ac.SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": v4ac.ROUND_TRIP_COST_BPS,
        "status": "COMPLETED",
        **metrics,
    }
    cell["acceptance_target_met"] = _candidate_acceptance_met(cell)
    trade_rows = [
        {
            "candidate_id": candidate_id,
            "symbol": symbol,
            "timeframe": TIMEFRAME,
            "horizon_bars": HORIZON_BARS,
            "exit_policy": policy,
            "mode": MODE,
            "initial_stop_atr": config.initial_stop_atr,
            "trailing_activation_atr": config.trailing_activation_atr,
            "trailing_step_atr": config.trailing_step_atr,
            "take_profit_atr": TAKE_PROFIT_ATR,
            "fee_bps_per_side": v4ac.FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": v4ac.SLIPPAGE_BPS_PER_SIDE,
            **trade,
        }
        for trade in replay["trades"]
    ]
    return cell, trade_rows


def _run_symbol(
    symbol: str, source_bars: DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bars = v4ac.resample_ohlcv(source_bars, TIMEFRAME)
    _, primary_signals = v4ac._primary_features_and_signals(bars)
    train_end, forward_end = v4ac._outer_boundaries(len(bars))
    signal_indices = v4ac._forward_primary_signal_indices(
        primary_signals, train_end, forward_end
    )
    raw_count = int(primary_signals.iloc[train_end:forward_end].sum())
    window = {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "row_count": len(bars),
        "outer_train_end_index": train_end,
        "forward_end_exclusive": forward_end,
        "raw_forward_primary_signal_count": raw_count,
        "eligible_forward_primary_signal_count": len(signal_indices),
        "forward_start_timestamp": pd.Timestamp(bars["date"].iloc[train_end]).isoformat(),
        "forward_end_timestamp": pd.Timestamp(
            bars["date"].iloc[forward_end - 1]
        ).isoformat(),
    }
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for config in GRID:
        cell, trade_rows = _run_candidate(
            symbol=symbol,
            bars=bars,
            signal_indices=signal_indices,
            raw_signal_count=raw_count,
            forward_end=forward_end,
            config=config,
        )
        cells.append(cell)
        trades.extend(trade_rows)
    return cells, trades, window


def _build_report(
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_candidate_matrix(cells)
    if len(windows) != len(SYMBOLS):
        raise ValueError("v4ae_evaluation_window_matrix_invalid")
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "symbols": list(SYMBOLS),
        "timeframes": [TIMEFRAME],
        "horizon_bars": HORIZON_BARS,
        "horizon_hours": 24.0,
        "mode": MODE,
        "grid": {
            "initial_stop_atr": list(INITIAL_STOP_ATR_VALUES),
            "trailing_activation_atr": list(TRAILING_ACTIVATION_ATR_VALUES),
            "trailing_step_atr": list(TRAILING_STEP_ATR_VALUES),
            "candidate_configurations_per_symbol": len(GRID),
            "total_candidate_configurations": len(cells),
            "take_profit_atr_fixed": TAKE_PROFIT_ATR,
        },
        "primary_model": v4ac._primary_model_report(),
        "meta_model": {
            "enabled": False,
            "model_fit_performed": False,
            "sequence_model_used": False,
            "reason": "V4-AE runs PRIMARY_ONLY and performs no Meta-Model training or filtering.",
        },
        "evaluation_window": v4ac._evaluation_window_report(windows),
        "execution_cost_contract": v4ac._execution_cost_report(),
        "exit_mechanics": {
            "initial_stop_atr": "Initial protective stop below entry.",
            "trailing_activation_atr": (
                "Favorable high-water mark required to activate the trailing ratchet."
            ),
            "trailing_step_atr": (
                "Increment used to ratchet the stop upward after activation; "
                "not a continuous offset below the high-water mark."
            ),
            "take_profit_atr": TAKE_PROFIT_ATR,
            "horizon_bars": HORIZON_BARS,
            "stop_first_intrabar_collision": True,
            "new_trailing_stop_applies_next_bar": True,
        },
        "acceptance_targets": {
            "trade_count_exclusive_minimum": MINIMUM_TRADES_EXCLUSIVE,
            "net_profit_factor_exclusive_minimum": (
                MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE
            ),
            "maximum_drawdown_inclusive": MAXIMUM_DRAWDOWN_INCLUSIVE,
            "purpose": "research_target_only_no_automatic_promotion",
        },
        "v4ad_reference": V4AD_BASELINE,
        "cells": cells,
        "decision_rule": "OWNER_REVIEW_REQUIRED_NO_AUTOMATIC_PROMOTION",
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": v4ac._safety_report(),
    }


def _format_pf(value: Any, infinite: bool) -> str:
    if infinite:
        return "INF"
    return "N/A" if value is None else f"{float(value):.4f}"


def render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ae_step_summary_report_schema_invalid")
    lines = [
        "## AliX V4-AE Primary-Only Trailing Stop Grid",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: BTCUSDT and ETHUSDT; 1h only; PRIMARY_ONLY; 24-bar horizon.",
        "Primary signal: unchanged V4-Y causal Bollinger(20, 2.0) upper-band breakout with Close > EMA200.",
        "Entry: next-bar open after a closed-candle BUY. Spot long-only, 1x equity, no pyramiding.",
        "Grid: 3 initial SL values x 3 activation values x 3 trailing-step values = 27 configurations per symbol; 54 total.",
        "Take-profit remains fixed at 4.0 ATR. Fees/slippage remain 10 bps maker + 5 bps slippage per side (30 bps round trip).",
        "Mechanics: trailing_step_atr is the stop ratchet increment after activation, not a continuous high-water-mark offset; the new stop applies on the next bar.",
        "Chronological split: 70% train / 15% forward evaluation / final 15% untouched holdout. Grid comparison is research-only and is not independent OOS promotion.",
        "Meta-Model: disabled; no fit, probability scoring, or Meta filtering occurs in V4-AE.",
        "",
        "| Symbol | Initial SL ATR | Activation ATR | Trailing Step ATR | Trades | Gross PF | Net PF | Max DD % | Net PnL USD | Acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["cells"]:
        lines.append(
            f"| {item['symbol']} | {float(item['initial_stop_atr']):.1f} "
            f"| {float(item['trailing_activation_atr']):.1f} "
            f"| {float(item['trailing_step_atr']):.1f} "
            f"| {item['trade_count']} "
            f"| {_format_pf(item['gross_profit_factor'], bool(item['gross_profit_factor_infinite_no_losses']))} "
            f"| {_format_pf(item['net_profit_factor'], bool(item['net_profit_factor_infinite_no_losses']))} "
            f"| {float(item['max_drawdown_percent']):.2f} "
            f"| {float(item['net_pnl_total_usd']):.2f} "
            f"| {'PASS' if item['acceptance_target_met'] else 'NO'} |"
        )
    completed = sum(item["status"] == "COMPLETED" for item in report["cells"])
    qualified = sum(bool(item["acceptance_target_met"]) for item in report["cells"])
    lines.extend(
        [
            "",
            f"Candidate rows: {len(report['cells'])}/54; completed: {completed}.",
            f"Rows meeting Trades >30, Net PF >1.5, and Max DD <=15%: {qualified}.",
            "V4-AD same-geometry reference: ETHUSDT 1h, SL=2.0 / Activation=1.5 / Step=1.0, 42 trades, Net PF=1.1351, Max DD=9.85%. V4-AD accepted BUYs were 52; they were not the executed trade count.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only; no private API, credentials, orders, or execution API are used.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4ae(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    v4ac._validate_run_request(days, commit_sha)
    if TIMEFRAME not in v4ac.TIMEFRAMES:
        raise ValueError("v4ae_timeframe_out_of_policy")
    datasets, provenance = v4ac._load_diagnostic_datasets(
        input_dir, days, commit_sha
    )
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in datasets:
            raise ValueError("v4ae_required_symbol_dataset_missing")
        symbol_cells, symbol_trades, window = _run_symbol(symbol, datasets[symbol])
        cells.extend(symbol_cells)
        trades.extend(symbol_trades)
        windows.append(window)
    report = _build_report(commit_sha, days, provenance, cells, windows)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    v4ac._write_csv(output_trades_csv, trades)
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
    report = run_v4ae(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["cells"]) == 54 else 1


if __name__ == "__main__":
    raise SystemExit(main())
