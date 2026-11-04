"""V4-AD focused Meta threshold and trailing-activation economic sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts import run_ml_core_v4ac_cost_optimization as v4ac
from services.ml_core.binary_calibration import select_dynamic_buy_threshold

SCHEMA = "alix.ml-core.v4ad-threshold-optimization.v1"
SYMBOLS = v4ac.SYMBOLS
TIMEFRAMES = ("1h", "4h")
BASE_EXIT_POLICIES = ("TB_1.5_1.0", "TRAIL_2.0")
REFINED_TRAIL_POLICY = "TRAIL_2.0_ACT1.5"
EXIT_POLICIES_BY_TIMEFRAME = {
    "1h": (*BASE_EXIT_POLICIES, REFINED_TRAIL_POLICY),
    "4h": BASE_EXIT_POLICIES,
}
TARGET_ACCEPT_FRACTIONS = (0.30, 0.50, 0.70)
MODES = ("PRIMARY_ONLY", "META_TOP_30", "META_TOP_50", "META_TOP_70")
MINIMUM_TRADES_EXCLUSIVE = 30
MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE = 1.5
MAXIMUM_DRAWDOWN_INCLUSIVE = 15.0


def _validate_timeframe(timeframe: str) -> None:
    if timeframe not in TIMEFRAMES:
        raise ValueError("v4ad_timeframe_out_of_scope")
    v4ac._validate_timeframe(timeframe)


def _target_key(target_accept_fraction: float) -> str:
    return str(int(round(target_accept_fraction * 100)))


def _mode_for_fraction(target_accept_fraction: float) -> str:
    if target_accept_fraction not in TARGET_ACCEPT_FRACTIONS:
        raise ValueError("v4ad_meta_target_accept_fraction_out_of_policy")
    return f"META_TOP_{_target_key(target_accept_fraction)}"


def _candidate_id(
    symbol: str, timeframe: str, policy: str, mode: str
) -> str:
    return f"{symbol}|{timeframe}|{v4ac.HORIZON_BARS}|{policy}|{mode}"


def _train_meta_model_threshold_sweep(
    bars: DataFrame,
    features: DataFrame,
    primary_signals: pd.Series,
    train_end: int,
    policy: str,
) -> tuple[Any | None, dict[str, Any], str]:
    """Fit one model and derive every cutoff from the same purged OOF calibration rows."""
    gate = v4ac.evaluate_primary_train_sample_gate(
        primary_signals,
        train_start_index=v4ac.V4Y_EMA_TREND_PERIOD,
        train_end_index=train_end,
        minimum_signal_count=v4ac.META_MINIMUM_TRAIN_SIGNALS,
    )
    if not bool(gate["passed"]):
        return None, {"sample_gate": gate}, "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"

    try:
        training, diagnostics = v4ac._meta_training_rows(
            bars, features, primary_signals, train_end, policy
        )
        fit_rows, calibration_rows = v4ac.split_meta_fit_calibration(
            training, outer_train_end_index=train_end
        )
        feature_columns = list(features.columns)
        model, fit_diagnostics = v4ac.fit_lightgbm_meta_filter(
            fit_rows.loc[:, feature_columns],
            fit_rows["meta_label"],
            calibration_rows.loc[:, feature_columns],
            target_accept_fraction=0.50,
        )
        calibration_features = calibration_rows.loc[:, feature_columns]
        calibration_probabilities = v4ac.predict_meta_probabilities(
            model, calibration_features
        )
        if (
            len(calibration_probabilities) != len(calibration_rows)
            or not np.isfinite(calibration_probabilities).all()
            or np.any(calibration_probabilities < 0)
            or np.any(calibration_probabilities > 1)
        ):
            raise ValueError("v4ad_calibration_probability_values_invalid")

        thresholds: dict[str, dict[str, Any]] = {}
        for fraction in TARGET_ACCEPT_FRACTIONS:
            selected = select_dynamic_buy_threshold(
                calibration_probabilities,
                target_buy_fraction=fraction,
            )
            threshold = float(selected["threshold"])
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("v4ad_selected_meta_threshold_invalid")
            thresholds[_target_key(fraction)] = {
                "target_accept_fraction": fraction,
                "selected_threshold": threshold,
                "threshold_diagnostics": selected,
            }
    except ValueError as error:
        if not v4ac._is_expected_meta_support_error(error):
            raise
        return (
            None,
            {"sample_gate": gate, "skip_reason": str(error)},
            "SKIPPED_INSUFFICIENT_META_SUPPORT",
        )

    diagnostics.update(
        {
            "sample_gate": gate,
            "fit_rows": int(len(fit_rows)),
            "calibration_rows": int(len(calibration_rows)),
            "fit_max_label_end_index": int(fit_rows["label_end_idx"].max()),
            "calibration_start_event_index": int(
                calibration_rows["event_index"].min()
            ),
            "fit_labels_purged_before_calibration": bool(
                int(fit_rows["label_end_idx"].max())
                < int(calibration_rows["event_index"].min())
            ),
            "validation_labels_used_for_fit": False,
            "validation_labels_used_for_threshold": False,
            "meta_model": {
                **fit_diagnostics,
                "calibration_labels_used": False,
                "threshold_selection_scope": (
                    "PURGED_OUTER_TRAIN_OOF_CALIBRATION_PROBABILITIES_ONLY"
                ),
                "target_accept_fractions": list(TARGET_ACCEPT_FRACTIONS),
                "thresholds": thresholds,
                "single_model_reused_across_thresholds": True,
            },
        }
    )
    return model, diagnostics, "TRAINED"


def _acceptance_target_met(cell: dict[str, Any]) -> bool:
    trade_count = cell.get("trade_count")
    profit_factor = cell.get("net_profit_factor")
    infinite_without_losses = bool(
        cell.get("net_profit_factor_infinite_no_losses", False)
    )
    drawdown = cell.get("max_drawdown_percent")
    pf_passed = infinite_without_losses or (
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


def _run_meta_variant(
    *,
    symbol: str,
    timeframe: str,
    policy: str,
    bars: DataFrame,
    primary_indices: list[int],
    raw_count: int,
    forward_end: int,
    model: Any | None,
    diagnostics: dict[str, Any],
    meta_status: str,
    probabilities: dict[int, float],
    target_accept_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mode = _mode_for_fraction(target_accept_fraction)
    candidate_id = _candidate_id(symbol, timeframe, policy, mode)
    threshold: float | None = None
    variant_diagnostics = {
        **diagnostics,
        "target_accept_fraction": target_accept_fraction,
    }

    if model is None:
        cell = v4ac._meta_skip_cell(
            symbol=symbol,
            timeframe=timeframe,
            policy=policy,
            raw_signal_count=raw_count,
            eligible_signal_count=len(primary_indices),
            reason=meta_status,
            diagnostics=diagnostics,
        )
        cell["candidate_id"] = candidate_id
        cell["mode"] = mode
        cell["meta_target_accept_fraction"] = target_accept_fraction
        cell["meta_threshold"] = None
        cell["acceptance_target_met"] = False
        scenario = {
            "candidate_id": candidate_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "exit_policy": policy,
            "mode": mode,
            "target_accept_fraction": target_accept_fraction,
            "selected_threshold": None,
            "status": meta_status,
            "skip_reason": diagnostics.get("skip_reason", meta_status),
            "diagnostics": variant_diagnostics,
        }
        return cell, [], [], scenario

    threshold_entry = diagnostics["meta_model"]["thresholds"][
        _target_key(target_accept_fraction)
    ]
    threshold = float(threshold_entry["selected_threshold"])
    accepted = [
        index
        for index in primary_indices
        if probabilities.get(index, -1.0) >= threshold
    ]
    accepted_set = set(accepted)
    decision_rows = [
        {
            "candidate_id": candidate_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "exit_policy": policy,
            "mode": mode,
            "split_role": "FORWARD_VALIDATION",
            "row_index": index,
            "timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
            "primary_signal": "BUY",
            "meta_target_accept_fraction": target_accept_fraction,
            "meta_probability": probabilities[index],
            "selected_threshold": threshold,
            "meta_filter_decision": "ACCEPT" if index in accepted_set else "REJECT",
            "executable_signal": "BUY" if index in accepted_set else "NO_ENTRY",
        }
        for index in primary_indices
    ]
    cell, trade_rows = v4ac._run_economic_cell(
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
        meta_diagnostics=variant_diagnostics,
        meta_threshold=threshold,
    )
    cell["candidate_id"] = candidate_id
    cell["mode"] = mode
    cell["meta_target_accept_fraction"] = target_accept_fraction
    cell["meta_threshold"] = threshold
    cell["realized_accept_fraction"] = (
        len(accepted) / len(primary_indices) if primary_indices else None
    )
    cell["acceptance_target_met"] = _acceptance_target_met(cell)
    for trade in trade_rows:
        trade["candidate_id"] = candidate_id
        trade["mode"] = mode
        trade["meta_target_accept_fraction"] = target_accept_fraction
        trade["meta_threshold"] = threshold

    scenario = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "exit_policy": policy,
        "mode": mode,
        "target_accept_fraction": target_accept_fraction,
        "selected_threshold": threshold,
        "status": meta_status,
        "skip_reason": None,
        "diagnostics": variant_diagnostics,
    }
    return cell, trade_rows, decision_rows, scenario


def _run_symbol_timeframe(
    symbol: str, source_bars: DataFrame
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []

    for timeframe in TIMEFRAMES:
        _validate_timeframe(timeframe)
        bars = v4ac.resample_ohlcv(source_bars, timeframe)
        features, primary_signals = v4ac._primary_features_and_signals(bars)
        train_end, forward_end = v4ac._outer_boundaries(len(bars))
        primary_indices = v4ac._forward_primary_signal_indices(
            primary_signals, train_end, forward_end
        )
        raw_count = int(primary_signals.iloc[train_end:forward_end].sum())
        windows.append(
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

        for policy in EXIT_POLICIES_BY_TIMEFRAME[timeframe]:
            primary_cell, primary_trades = v4ac._run_economic_cell(
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
            primary_cell["candidate_id"] = _candidate_id(
                symbol, timeframe, policy, "PRIMARY_ONLY"
            )
            primary_cell["meta_target_accept_fraction"] = None
            primary_cell["acceptance_target_met"] = False
            cells.append(primary_cell)
            trades.extend(primary_trades)

            model, diagnostics, status = _train_meta_model_threshold_sweep(
                bars, features, primary_signals, train_end, policy
            )
            probabilities = (
                v4ac._forward_meta_probabilities(features, primary_indices, model)
                if model is not None
                else {}
            )
            for fraction in TARGET_ACCEPT_FRACTIONS:
                cell, meta_trades, meta_decisions, scenario = _run_meta_variant(
                    symbol=symbol,
                    timeframe=timeframe,
                    policy=policy,
                    bars=bars,
                    primary_indices=primary_indices,
                    raw_count=raw_count,
                    forward_end=forward_end,
                    model=model,
                    diagnostics=diagnostics,
                    meta_status=status,
                    probabilities=probabilities,
                    target_accept_fraction=fraction,
                )
                cells.append(cell)
                trades.extend(meta_trades)
                decisions.extend(meta_decisions)
                scenarios.append(scenario)

    return cells, trades, decisions, windows, scenarios


def _validate_report_matrix(
    cells: list[dict[str, Any]], scenarios: list[dict[str, Any]]
) -> None:
    expected = {
        _candidate_id(symbol, timeframe, policy, mode)
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
        for policy in EXIT_POLICIES_BY_TIMEFRAME[timeframe]
        for mode in MODES
    }
    observed = {str(item["candidate_id"]) for item in cells}
    if len(cells) != 40 or observed != expected:
        raise ValueError("v4ad_candidate_matrix_incomplete")
    if any(item["timeframe"] not in TIMEFRAMES for item in cells):
        raise ValueError("v4ad_report_timeframe_out_of_policy")
    if any(
        item["exit_policy"] == REFINED_TRAIL_POLICY
        and item["timeframe"] != "1h"
        for item in cells
    ):
        raise ValueError("v4ad_refined_trailing_scope_invalid")
    expected_scenarios = {
        _candidate_id(symbol, timeframe, policy, mode)
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
        for policy in EXIT_POLICIES_BY_TIMEFRAME[timeframe]
        for mode in MODES
        if mode != "PRIMARY_ONLY"
    }
    observed_scenarios = {str(item["candidate_id"]) for item in scenarios}
    if len(scenarios) != 30 or observed_scenarios != expected_scenarios:
        raise ValueError("v4ad_meta_scenario_matrix_incomplete")
    if any(
        item["target_accept_fraction"] not in TARGET_ACCEPT_FRACTIONS
        for item in scenarios
    ):
        raise ValueError("v4ad_meta_target_accept_fraction_invalid")


def _exit_contract_report() -> dict[str, Any]:
    return {
        "TB_1.5_1.0": {
            "take_profit_atr": 1.5,
            "stop_loss_atr": 1.0,
            "vertical_horizon_bars": v4ac.HORIZON_BARS,
            "trailing_stop_enabled": False,
        },
        "TRAIL_2.0": {
            "initial_stop_atr": 2.0,
            "trailing_activation_atr": 2.0,
            "trailing_step_atr": 1.0,
            "terminal_take_profit_atr": 4.0,
            "vertical_horizon_bars": v4ac.HORIZON_BARS,
        },
        "TRAIL_2.0_ACT1.5": {
            "initial_stop_atr": 2.0,
            "trailing_activation_atr": 1.5,
            "trailing_step_atr": 1.0,
            "terminal_take_profit_atr": 4.0,
            "vertical_horizon_bars": v4ac.HORIZON_BARS,
            "scope": "1h_only_refined_trailing_comparison",
        },
    }


def _build_report(
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_report_matrix(cells, scenarios)
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "symbols": list(SYMBOLS),
        "timeframes": list(TIMEFRAMES),
        "horizon_bars": v4ac.HORIZON_BARS,
        "horizon_hours_by_timeframe": {
            timeframe: v4ac.HORIZON_BARS
            * v4ac.TIMEFRAME_MINUTES[timeframe]
            / 60
            for timeframe in TIMEFRAMES
        },
        "exit_policies_by_timeframe": {
            timeframe: list(policies)
            for timeframe, policies in EXIT_POLICIES_BY_TIMEFRAME.items()
        },
        "modes": list(MODES),
        "primary_model": v4ac._primary_model_report(),
        "meta_model": {
            "enabled": True,
            "family": "LightGBM LGBMClassifier objective=binary",
            "training_signal_requirement": "OOF_PRIMARY_BUY_ONLY",
            "training_label": "POLICY_MATCHED_NET_POSITIVE_AFTER_FEES_AND_SLIPPAGE",
            "one_model_per_symbol_timeframe_exit_policy": True,
            "same_model_reused_across_threshold_variants": True,
            "threshold_selection_scope": (
                "PURGED_OUTER_TRAIN_OOF_CALIBRATION_PROBABILITIES_ONLY"
            ),
            "target_accept_fractions": list(TARGET_ACCEPT_FRACTIONS),
            "calibration_labels_used": False,
            "validation_labels_used_for_fit_or_threshold": False,
            "scenarios": scenarios,
            "sequence_model_used": False,
        },
        "evaluation_window": v4ac._evaluation_window_report(windows),
        "execution_cost_contract": v4ac._execution_cost_report(),
        "exit_contract": _exit_contract_report(),
        "acceptance_targets": {
            "trade_count_exclusive_minimum": MINIMUM_TRADES_EXCLUSIVE,
            "net_profit_factor_exclusive_minimum": (
                MINIMUM_NET_PROFIT_FACTOR_EXCLUSIVE
            ),
            "maximum_drawdown_inclusive": MAXIMUM_DRAWDOWN_INCLUSIVE,
            "purpose": "research_target_only_no_automatic_promotion",
        },
        "cells": cells,
        "decision_rule": "OWNER_REVIEW_REQUIRED_NO_AUTOMATIC_PROMOTION",
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": v4ac._safety_report(),
    }


def _format_pf(value: Any, infinite: bool) -> str:
    return v4ac._format_pf(value, infinite)


def render_step_summary(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("v4ad_step_summary_report_schema_invalid")
    lines = [
        "## AliX V4-AD Meta Threshold and Trailing-Activation Sweep",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: BTCUSDT and ETHUSDT; 1h and 4h only; 24 bars per trade.",
        "Thresholds: Top 30%, Top 50%, and Top 70% calibration acceptance targets.",
        "Model control: one LightGBM model per symbol/timeframe/exit; the same model and calibration probabilities are reused across all three thresholds.",
        "Thresholds are selected from purged outer-training OOF calibration probabilities only. Validation labels/probabilities and the final 15% holdout do not select thresholds.",
        "Exit comparison: TB_1.5_1.0 and TRAIL_2.0 on both timeframes; TRAIL_2.0_ACT1.5 is an additional 1h-only comparison (2.0 ATR initial stop, 1.5 ATR activation, 1.0 ATR step).",
        "Costs: 10 bps maker-fee assumption + 5 bps slippage assumption per side; 30 bps round trip.",
        "",
        "| Symbol | TF | Exit | Mode | Target Accept | Selected Threshold | Accepted BUYs | Trades | Gross PF | Net PF | Max DD % | Net PnL USD | Acceptance Target | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["cells"]:
        target = (
            "BASELINE"
            if item.get("meta_target_accept_fraction") is None
            else f"{float(item['meta_target_accept_fraction']):.0%}"
        )
        threshold = (
            "N/A"
            if item.get("meta_threshold") is None
            else f"{float(item['meta_threshold']):.6f}"
        )
        if item["status"] != "COMPLETED":
            lines.append(
                f"| {item['symbol']} | {item['timeframe']} | {item['exit_policy']} "
                f"| {item['mode']} | {target} | {threshold} | N/A | N/A | N/A "
                f"| N/A | N/A | N/A | NO | {item['status']} |"
            )
            continue
        acceptance = "PASS" if item.get("acceptance_target_met") else "NO"
        lines.append(
            f"| {item['symbol']} | {item['timeframe']} | {item['exit_policy']} "
            f"| {item['mode']} | {target} | {threshold} "
            f"| {item['accepted_signal_count']} | {item['trade_count']} "
            f"| {_format_pf(item['gross_profit_factor'], bool(item['gross_profit_factor_infinite_no_losses']))} "
            f"| {_format_pf(item['net_profit_factor'], bool(item['net_profit_factor_infinite_no_losses']))} "
            f"| {float(item['max_drawdown_percent']):.2f} "
            f"| {float(item['net_pnl_total_usd']):.2f} | {acceptance} | COMPLETED |"
        )
    completed = sum(item["status"] == "COMPLETED" for item in report["cells"])
    qualified = sum(bool(item.get("acceptance_target_met")) for item in report["cells"])
    meta_scenarios = report["meta_model"]["scenarios"]
    trained = sum(item["status"] == "TRAINED" for item in meta_scenarios)
    lines.extend(
        [
            "",
            f"Candidate rows: {len(report['cells'])}/40; completed: {completed}.",
            f"Meta scenarios trained: {trained}/{len(meta_scenarios)}.",
            f"Rows meeting Trades >30, Net PF >1.5, and Max DD <=15%: {qualified}.",
            "A threshold acceptance fraction is a calibration target; realized accepted BUY and trade counts are reported separately.",
            "This is a research sweep, not independent OOS promotion or profitability approval.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false.",
            "Research only; no private API, credentials, orders, or execution API are used.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4ad(
    input_dir: Path,
    output_report: Path,
    output_trades_csv: Path,
    output_decisions_csv: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    v4ac._validate_run_request(days, commit_sha)
    datasets, provenance = v4ac._load_diagnostic_datasets(
        input_dir, days, commit_sha
    )
    cells: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        if symbol not in datasets:
            raise ValueError("v4ad_required_symbol_dataset_missing")
        symbol_cells, symbol_trades, symbol_decisions, symbol_windows, symbol_scenarios = (
            _run_symbol_timeframe(symbol, datasets[symbol])
        )
        cells.extend(symbol_cells)
        trades.extend(symbol_trades)
        decisions.extend(symbol_decisions)
        windows.extend(symbol_windows)
        scenarios.extend(symbol_scenarios)

    report = _build_report(
        commit_sha, days, provenance, cells, windows, scenarios
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    v4ac._write_csv(output_trades_csv, trades)
    v4ac._write_csv(output_decisions_csv, decisions)
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
    report = run_v4ad(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if len(report["cells"]) == 40 else 1


if __name__ == "__main__":
    raise SystemExit(main())
