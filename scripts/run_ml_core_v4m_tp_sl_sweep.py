"""Run V4-M TP/SL geometry sensitivity on persisted V4-H probabilities only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from pandas import DataFrame

from services.ml_core.economic_backtest import (
    V4H_SOURCE_RUN_ID,
    V4H_SOURCE_SHA,
    V4M_CONFIDENCE_THRESHOLD,
    V4M_TP_SL_GEOMETRIES,
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate_tp_sl_sweep,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_execution_config,
    validate_source_report,
    validate_tp_sl_geometries,
)

V4H_REPORT_SHA = "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
V4M_SCHEMA = "alix.ml-core.v4m-tp-sl-sweep.v1"
V4M_HORIZON_CANDLES = 12
V4M_ATR_PERIOD = 14
V4M_MAXIMUM_DRAWDOWN = 0.15
CSV_FIELDS = (
    "candidate_id",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "confidence_threshold",
    "stop_loss_atr_multiplier",
    "take_profit_atr_multiplier",
    "is_v4k_exit_geometry",
    "trade_count",
    "profit_factor",
    "max_drawdown_percent",
    "win_rate",
    "tp_exit_count",
    "sl_exit_count",
    "horizon_exit_count",
    "collision_exit_count",
    "sell_exit_count",
    "other_exit_count",
    "total_fees_usd",
    "total_slippage_cost_usd",
    "economic_candidate_gate_passed",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    if not path.is_file():
        raise ValueError(f"required_json_file_missing:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("economic_json_root_must_be_object")
    return payload


def _digest(path: Path) -> str:
    """Hash an immutable input artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _symbol_for_pair(pair: str) -> str:
    """Resolve the approved BTC and ETH Spot source universe."""
    symbol = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}.get(pair)
    if symbol is None:
        raise ValueError("economic_pair_out_of_policy")
    return symbol


def _source_metadata_by_symbol(
    source_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index the exact approved V4-H dataset provenance records."""
    datasets = source_report.get("input_datasets")
    if not isinstance(datasets, list) or any(
        not isinstance(item, dict) for item in datasets
    ):
        raise ValueError("v4h_source_dataset_manifest_invalid")
    indexed = {str(item.get("symbol")): item for item in datasets}
    if set(indexed) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("v4h_source_dataset_manifest_invalid")
    return indexed


def _execution_config() -> ExecutionConfig:
    """Keep account sizing, fees, and slippage fixed throughout V4-M."""
    config = ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=25.0,
        slippage_bps=20.0,
        position_fraction=1.0,
        maximum_drawdown=V4M_MAXIMUM_DRAWDOWN,
    )
    validate_execution_config(config)
    return config


def _base_management_config() -> TradeManagementConfig:
    """Fix V4-M confidence, ATR period, and vertical horizon."""
    return TradeManagementConfig(
        atr_period=V4M_ATR_PERIOD,
        take_profit_atr_multiplier=2.0,
        stop_loss_atr_multiplier=1.5,
        horizon_candles=V4M_HORIZON_CANDLES,
        confidence_threshold=V4M_CONFIDENCE_THRESHOLD,
    )


def _market_bars(
    candidates: list[dict[str, Any]],
    source_report: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], DataFrame], list[dict[str, Any]]]:
    """Load V4-H byte-identical minute candles and cache resampled bars."""
    metadata = _source_metadata_by_symbol(source_report)
    minute_data: dict[str, DataFrame] = {}
    cache: dict[tuple[str, str], DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = _symbol_for_pair(str(candidate["pair"]))
        if symbol not in minute_data:
            minute_data[symbol] = load_validated_minute_data(
                input_dir, metadata[symbol], current_commit_sha=commit_sha
            )
            manifest = _load_json(
                input_dir / f"binance-{symbol.lower()}-1m.manifest.json"
            )
            provenance.append(
                {
                    "symbol": symbol,
                    "row_count": int(manifest["row_count"]),
                    "start_utc": manifest["start_utc"],
                    "end_utc": manifest["end_utc"],
                    "csv_sha256": manifest["csv_sha256"],
                    "source_v4h_csv_sha256": metadata[symbol]["csv_sha256"],
                    "byte_identical_to_v4h": True,
                    "public_api_only": True,
                    "credentials_used": False,
                }
            )
        key = (symbol, str(candidate["timeframe"]))
        if key not in cache:
            cache[key] = resample_ohlcv(minute_data[symbol], key[1])
    return cache, provenance


def _validate_trade_ledger(
    trades: list[dict[str, Any]], stop_loss: float, take_profit: float
) -> dict[str, int]:
    """Fail closed on Spot, confidence, horizon, and exact ATR-barrier invariants."""
    directions = {"LONG": 0, "SHORT": 0}
    for trade in trades:
        direction = str(trade.get("direction"))
        if direction not in directions:
            raise ValueError("economic_trade_direction_invalid")
        directions[direction] += 1
        if direction != "LONG":
            raise RuntimeError("economic_spot_only_position_invariant_violation")
        confidence = float(trade.get("entry_confidence", 0.0))
        if not math.isfinite(confidence) or confidence < V4M_CONFIDENCE_THRESHOLD:
            raise RuntimeError("v4m_trade_confidence_invariant_violation")
        held = int(trade.get("bars_held", 0))
        if held < 1 or held > V4M_HORIZON_CANDLES:
            raise RuntimeError("v4m_trade_horizon_invariant_violation")
        reference = float(trade["barrier_reference_price"])
        atr_value = float(trade["atr_value"])
        expected_stop = reference - stop_loss * atr_value
        expected_target = reference + take_profit * atr_value
        if not math.isclose(
            float(trade["stop_loss_price"]),
            expected_stop,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise RuntimeError("v4m_stop_loss_geometry_invariant_violation")
        if not math.isclose(
            float(trade["take_profit_price"]),
            expected_target,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise RuntimeError("v4m_take_profit_geometry_invariant_violation")
    return directions


def _candidate_geometry_result(
    candidate: dict[str, Any],
    stop_loss: float,
    take_profit: float,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Attach selected geometry, requested metrics, and its trade ledger."""
    directions = _validate_trade_ledger(replay["trades"], stop_loss, take_profit)
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "confidence_threshold": V4M_CONFIDENCE_THRESHOLD,
        "stop_loss_atr_multiplier": stop_loss,
        "take_profit_atr_multiplier": take_profit,
        "is_v4k_exit_geometry": stop_loss == 1.5 and take_profit == 2.0,
        "raw_signal_counts": replay["raw_signal_counts"],
        "metrics": replay["metrics"],
        "total_fees_usd": replay["total_fees_usd"],
        "total_slippage_cost_usd": replay["total_slippage_cost_usd"],
        "exit_reason_counts": replay["exit_reason_counts"],
        "trade_direction_counts": directions,
        "trade_management": replay["trade_management"],
        "trades": replay["trades"],
    }


def _flatten_trades(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ledgers for descriptive aggregate outcome statistics."""
    return [trade for item in candidates for trade in item["trades"]]


def _exit_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Sum exit reasons across independent candidate simulations."""
    exits: dict[str, int] = {}
    for candidate in candidates:
        for reason, count in candidate["exit_reason_counts"].items():
            exits[reason] = exits.get(reason, 0) + int(count)
    return exits


def _trade_outcome_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate pooled-ledger PF and win rate without merging equity curves."""
    positive = [
        float(trade["net_pnl_usd"]) for trade in trades if trade["net_pnl_usd"] > 0
    ]
    negative = [
        float(trade["net_pnl_usd"]) for trade in trades if trade["net_pnl_usd"] < 0
    ]
    gross_profit = sum(positive)
    gross_loss = abs(sum(negative))
    count = len(trades)
    return {
        "total_trade_count": count,
        "winning_trade_count": len(positive),
        "losing_trade_count": len(negative),
        "win_rate": round(len(positive) / count, 8) if count else 0.0,
        "combined_ledger_profit_factor": (
            round(gross_profit / gross_loss, 8) if gross_loss > 0 else None
        ),
        "combined_ledger_gross_profit_usd": round(gross_profit, 8),
        "combined_ledger_gross_loss_usd": round(gross_loss, 8),
    }


def _candidate_risk_metrics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate PF and active-candidate drawdown without zero-trade bias."""
    defined_pf: list[float] = []
    active_dd: list[float] = []
    all_dd: list[float] = []
    active_count = 0
    passed = 0
    pf_passed = 0
    dd_passed = 0
    active_dd_passed = 0
    for candidate in candidates:
        metrics = candidate["metrics"]
        pf = metrics["profit_factor_net_of_costs"]
        dd = float(metrics["max_drawdown_percent"])
        all_dd.append(dd)
        if pf is not None:
            pf_value = float(pf)
            defined_pf.append(pf_value)
            if pf_value > 1.5:
                pf_passed += 1
        if dd < 15.0:
            dd_passed += 1
        if int(metrics["trade_count"]) > 0:
            active_count += 1
            active_dd.append(dd)
            if dd < 15.0:
                active_dd_passed += 1
        if bool(metrics["economic_candidate_gate_passed"]):
            passed += 1
    return {
        "active_candidate_count": active_count,
        "mean_defined_candidate_profit_factor": (
            round(sum(defined_pf) / len(defined_pf), 8) if defined_pf else None
        ),
        "defined_candidate_profit_factor_count": len(defined_pf),
        "mean_max_drawdown_percent_all_candidates": round(sum(all_dd) / len(all_dd), 8),
        "mean_max_drawdown_percent_active_candidates": (
            round(sum(active_dd) / len(active_dd), 8) if active_dd else None
        ),
        "economic_gate_passed_candidate_count": passed,
        "pf_above_1_5_candidate_count": pf_passed,
        "drawdown_below_15_candidate_count": dd_passed,
        "active_drawdown_below_15_candidate_count": active_dd_passed,
    }


def _best_defined_pf(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the best defined candidate PF for each geometry."""
    eligible = [
        item
        for item in candidates
        if item["metrics"]["profit_factor_net_of_costs"] is not None
    ]
    return max(
        eligible,
        key=lambda item: float(item["metrics"]["profit_factor_net_of_costs"]),
        default=None,
    )


def _aggregate_geometry(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build auditable aggregate statistics for one of the four geometries."""
    if len(candidates) != 13:
        raise ValueError("v4m_geometry_candidate_count_invalid")
    trades = _flatten_trades(candidates)
    outcomes = _trade_outcome_metrics(trades)
    risks = _candidate_risk_metrics(candidates)
    exits = _exit_counts(candidates)
    best = _best_defined_pf(candidates)
    return {
        "candidate_count": len(candidates),
        **outcomes,
        **risks,
        "exit_reason_counts": exits,
        "tp_exit_count": exits.get("TP_HIT", 0),
        "sl_exit_count": exits.get("SL_HIT", 0),
        "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
        "collision_exit_count": exits.get("COLLISION", 0),
        "sell_exit_count": exits.get("SELL", 0),
        "total_fees_usd": round(
            sum(float(item["total_fees_usd"]) for item in candidates), 8
        ),
        "total_slippage_cost_usd": round(
            sum(float(item["total_slippage_cost_usd"]) for item in candidates), 8
        ),
        "best_defined_profit_factor_candidate": (
            {
                "candidate_id": best["candidate_id"],
                "profit_factor": best["metrics"]["profit_factor_net_of_costs"],
                "trade_count": best["metrics"]["trade_count"],
                "max_drawdown_percent": best["metrics"]["max_drawdown_percent"],
            }
            if best is not None
            else None
        ),
    }


def _build_report(
    commit_sha: str,
    source_digest: str,
    candidates: list[dict[str, Any]],
    bars_cache: dict[tuple[str, str], DataFrame],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay all 13 candidate probability streams at each TP/SL geometry."""
    execution = _execution_config()
    base = _base_management_config()
    geometries = validate_tp_sl_geometries(V4M_TP_SL_GEOMETRIES)
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {
        geometry: [] for geometry in geometries
    }
    for candidate in candidates:
        key = (_symbol_for_pair(str(candidate["pair"])), str(candidate["timeframe"]))
        sweeps = evaluate_candidate_tp_sl_sweep(
            candidate,
            bars_cache[key],
            execution,
            base,
            tp_sl_geometries=geometries,
        )
        for sweep in sweeps:
            stop_loss = float(sweep["stop_loss_atr_multiplier"])
            take_profit = float(sweep["take_profit_atr_multiplier"])
            grouped[(stop_loss, take_profit)].append(
                _candidate_geometry_result(
                    candidate,
                    stop_loss,
                    take_profit,
                    sweep["candidate_result"],
                )
            )
    geometry_results = [
        {
            "stop_loss_atr_multiplier": stop_loss,
            "take_profit_atr_multiplier": take_profit,
            "is_v4k_exit_geometry": stop_loss == 1.5 and take_profit == 2.0,
            "aggregate": _aggregate_geometry(grouped[(stop_loss, take_profit)]),
            "candidates": grouped[(stop_loss, take_profit)],
        }
        for stop_loss, take_profit in geometries
    ]
    passed = sum(
        item["aggregate"]["economic_gate_passed_candidate_count"]
        for item in geometry_results
    )
    return {
        "schema_version": V4M_SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_report_sha256": source_digest,
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_TP_SL_GEOMETRY_SWEEP",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "candidate_count": len(candidates),
        "geometry_count": len(geometries),
        "candidate_geometry_pair_count": len(candidates) * len(geometries),
        "confidence_threshold": V4M_CONFIDENCE_THRESHOLD,
        "tp_sl_geometries": [
            {
                "stop_loss_atr_multiplier": stop_loss,
                "take_profit_atr_multiplier": take_profit,
                "is_v4k_exit_geometry": stop_loss == 1.5 and take_profit == 2.0,
            }
            for stop_loss, take_profit in geometries
        ],
        "geometry_baseline_note": (
            "SL=1.5/TP=2.0 is the V4-K exit geometry baseline only. V4-M fixes "
            "confidence at 0.60; V4-K used 0.55, so this is not a byte-for-byte "
            "reproduction of the complete V4-K configuration."
        ),
        "input_datasets": provenance,
        "execution_assumptions": {
            "initial_equity_usd": execution.initial_equity,
            "position_fraction_of_equity": execution.position_fraction,
            "fee_bps_per_side": execution.fee_bps,
            "slippage_bps_per_side": execution.slippage_bps,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
            "buy_semantics": "OPEN_LONG_IF_FLAT_AND_CONFIDENT",
            "sell_semantics": "CLOSE_LONG_AT_NEXT_BAR_OPEN",
            "hold_semantics": "KEEP_POSITION_UNTIL_BARRIER_OR_HORIZON",
        },
        "fixed_trade_management_config": {
            "confidence_threshold": V4M_CONFIDENCE_THRESHOLD,
            "atr_period": base.atr_period,
            "horizon_candles": base.horizon_candles,
            "atr_calculation": "SIMPLE_MEAN_TRUE_RANGE_FULL_WINDOW",
            "atr_reference": "SIGNAL_BAR_CLOSE_ONLY",
            "barrier_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "gap_exit_reference": "BAR_OPEN",
            "intrabar_exit_reference": "BARRIER_PRICE",
        },
        "geometry_results": geometry_results,
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_geometry_evaluation_count": len(candidates) * len(geometries),
            "passed_candidate_geometry_pair_count": passed,
            "status": "MET" if passed else "NOT_MET",
        },
        "validation_bias_caveat": (
            "V4-M replays the same V4-H Forward-Validation probabilities and candidate "
            "selection interval. This is exploratory TP/SL sensitivity analysis, not an "
            "independent out-of-sample promotion test."
        ),
        "model_artifact_limitation": (
            "V4-H contains persisted row-level probabilities and thresholds, not fitted "
            "estimators. V4-M performs no training or inference."
        ),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
        },
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the 4 x 13 geometry/model matrix into CSV records."""
    rows: list[dict[str, Any]] = []
    for geometry in report["geometry_results"]:
        stop_loss = geometry["stop_loss_atr_multiplier"]
        take_profit = geometry["take_profit_atr_multiplier"]
        for candidate in geometry["candidates"]:
            metrics = candidate["metrics"]
            exits = candidate["exit_reason_counts"]
            other_exits = sum(
                count
                for reason, count in exits.items()
                if reason
                not in {"TP_HIT", "SL_HIT", "HORIZON_EXPIRY", "COLLISION", "SELL"}
            )
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "confidence_threshold": V4M_CONFIDENCE_THRESHOLD,
                    "stop_loss_atr_multiplier": stop_loss,
                    "take_profit_atr_multiplier": take_profit,
                    "is_v4k_exit_geometry": candidate["is_v4k_exit_geometry"],
                    "trade_count": metrics["trade_count"],
                    "profit_factor": metrics["profit_factor_net_of_costs"],
                    "max_drawdown_percent": metrics["max_drawdown_percent"],
                    "win_rate": metrics["win_rate"],
                    "tp_exit_count": exits.get("TP_HIT", 0),
                    "sl_exit_count": exits.get("SL_HIT", 0),
                    "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                    "collision_exit_count": exits.get("COLLISION", 0),
                    "sell_exit_count": exits.get("SELL", 0),
                    "other_exit_count": other_exits,
                    "total_fees_usd": candidate["total_fees_usd"],
                    "total_slippage_cost_usd": candidate["total_slippage_cost_usd"],
                    "economic_candidate_gate_passed": metrics[
                        "economic_candidate_gate_passed"
                    ],
                }
            )
    return rows


def write_comparison_csv(report: dict[str, Any], output_path: Path) -> None:
    """Write exactly one row for every candidate/geometry pair."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(report))


def _aggregate_summary_row(item: dict[str, Any]) -> str:
    """Format one aggregate geometry row for GitHub Step Summary."""
    aggregate = item["aggregate"]
    ledger_pf = aggregate["combined_ledger_profit_factor"]
    mean_pf = aggregate["mean_defined_candidate_profit_factor"]
    active_dd = aggregate["mean_max_drawdown_percent_active_candidates"]
    return (
        f"| {item['stop_loss_atr_multiplier']:.1f} "
        f"| {item['take_profit_atr_multiplier']:.1f} "
        f"| {aggregate['total_trade_count']} "
        f"| {ledger_pf if ledger_pf is not None else 'N/A'} "
        f"| {mean_pf if mean_pf is not None else 'N/A'} "
        f"| {f'{active_dd:.2f}%' if active_dd is not None else 'N/A'} "
        f"| {aggregate['win_rate']:.2%} "
        f"| {aggregate['tp_exit_count']} "
        f"| {aggregate['sl_exit_count']} "
        f"| {aggregate['horizon_exit_count']} "
        f"| {aggregate['economic_gate_passed_candidate_count']}/13 |"
    )


def _candidate_summary_row(geometry: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Format one candidate/geometry metric row for GitHub Step Summary."""
    metrics = candidate["metrics"]
    pf = metrics["profit_factor_net_of_costs"]
    return (
        f"| {geometry['stop_loss_atr_multiplier']:.1f} "
        f"| {geometry['take_profit_atr_multiplier']:.1f} "
        f"| {candidate['pair']} | {candidate['timeframe']} "
        f"| {candidate['feature_profile']} | {candidate['weighting_policy']} "
        f"| {metrics['trade_count']} | {pf if pf is not None else 'N/A'} "
        f"| {metrics['max_drawdown_percent']:.2f}% "
        f"| {metrics['win_rate']:.2%} "
        f"| {candidate['exit_reason_counts'].get('TP_HIT', 0)} "
        f"| {candidate['exit_reason_counts'].get('SL_HIT', 0)} "
        f"| {candidate['exit_reason_counts'].get('HORIZON_EXPIRY', 0)} "
        f"| {'PASS' if metrics['economic_candidate_gate_passed'] else 'NO-GO'} |"
    )


def render_v4m_markdown(report: dict[str, Any]) -> str:
    """Render aggregate and per-model geometry comparisons in Step Summary."""
    lines = [
        "## AliX V4-M TP/SL Geometry Sweep",
        "",
        f"V4-M commit SHA: {report['commit_sha']}",
        f"V4-H source SHA: {report['source_v4h_commit_sha']}",
        "",
        "Replay only: the same 13 persisted V4-H probability streams.",
        "Confidence=0.60; ATR period=14; horizon=12; costs and Spot-only semantics remain fixed.",
        report["geometry_baseline_note"],
        "",
        "| SL ATR | TP ATR | Trades | Ledger PF | Mean model PF | Active mean DD | Win rate | TP exits | SL exits | Horizon exits | Gate passes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_aggregate_summary_row(item) for item in report["geometry_results"])
    lines.extend(
        [
            "",
            "### Per-model results",
            "",
            "| SL ATR | TP ATR | Pair | TF | Profile | Weight | Trades | PF | DD | Win rate | TP | SL | Horizon | Gate |",
            "|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for geometry in report["geometry_results"]:
        lines.extend(
            _candidate_summary_row(geometry, candidate)
            for candidate in geometry["candidates"]
        )
    lines.extend(
        [
            "",
            f"Candidate/geometry pairs passing PF > 1.5 and Max DD < 15%: "
            f"{report['economic_acceptance']['passed_candidate_geometry_pair_count']}/"
            f"{report['economic_acceptance']['candidate_geometry_evaluation_count']}.",
            "",
            "Combined-ledger PF is descriptive across independent model replays, not a single portfolio.",
            "V4-M uses the same V4-H Forward-Validation interval; independent OOS remains mandatory.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0, orders_or_execution_api_used=false.",
            "No model training or inference. Promotion remains blocked pending independent OOS and owner review.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4m_tp_sl_sweep(
    source_report_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    *,
    commit_sha: str,
) -> dict[str, Any]:
    """Validate V4-H provenance and replay all approved candidates at four geometries."""
    source_digest = _digest(source_report_path)
    if source_digest != V4H_REPORT_SHA:
        raise ValueError("v4h_source_report_digest_mismatch")
    source_report = _load_json(source_report_path)
    validate_source_report(source_report)
    candidates = extract_qualifying_candidates(source_report)
    if len(candidates) != 13:
        raise ValueError("v4m_expected_13_v4h_candidates")
    geometries = validate_tp_sl_geometries(V4M_TP_SL_GEOMETRIES)
    if len(geometries) != 4:
        raise ValueError("v4m_expected_four_tp_sl_geometries")
    bars_cache, provenance = _market_bars(
        candidates, source_report, input_dir, commit_sha
    )
    report = _build_report(
        commit_sha, source_digest, candidates, bars_cache, provenance
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_comparison_csv(report, output_csv_path)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(render_v4m_markdown(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": report["candidate_count"],
                "geometry_count": report["geometry_count"],
                "candidate_geometry_pair_count": report[
                    "candidate_geometry_pair_count"
                ],
                "confidence_threshold": report["confidence_threshold"],
                "aggregate_results": [
                    {
                        "stop_loss_atr_multiplier": item["stop_loss_atr_multiplier"],
                        "take_profit_atr_multiplier": item[
                            "take_profit_atr_multiplier"
                        ],
                        "aggregate": item["aggregate"],
                    }
                    for item in report["geometry_results"]
                ],
                "model_training_performed": False,
                "model_inference_performed": False,
                "Champion": None,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    """Parse the immutable-source V4-M GitHub Actions invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_v4m_tp_sl_sweep(
        args.source_report,
        args.input_dir,
        args.output,
        args.output_csv,
        commit_sha=args.commit_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
