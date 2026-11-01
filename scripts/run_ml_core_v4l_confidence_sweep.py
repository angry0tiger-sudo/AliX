"""Run the V4-L confidence sweep against persisted V4-H probabilities only."""

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
    V4L_CONFIDENCE_SWEEP_THRESHOLDS,
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate_confidence_sweep,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_confidence_thresholds,
    validate_execution_config,
    validate_source_report,
)

V4H_REPORT_SHA = "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
V4L_SCHEMA = "alix.ml-core.v4l-confidence-sweep.v1"
CSV_FIELDS = (
    "confidence_threshold",
    "candidate_id",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "trade_count",
    "profit_factor",
    "max_drawdown_percent",
    "win_rate",
    "total_fees_usd",
    "total_slippage_cost_usd",
    "sl_exit_count",
    "tp_exit_count",
    "collision_exit_count",
    "horizon_exit_count",
    "sell_exit_count",
    "confidence_rejected_buy_signal_count",
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
    """Resolve the approved BTC and ETH source universe."""
    symbol = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}.get(pair)
    if symbol is None:
        raise ValueError("economic_pair_out_of_policy")
    return symbol


def _source_metadata_by_symbol(
    source_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index exact V4-H dataset provenance records."""
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
    """Keep execution costs and account sizing fixed across the sweep."""
    config = ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=25.0,
        slippage_bps=20.0,
        position_fraction=1.0,
        maximum_drawdown=0.15,
    )
    validate_execution_config(config)
    return config


def _base_management_config() -> TradeManagementConfig:
    """Keep V4-K ATR barriers and horizon fixed while sweeping confidence."""
    return TradeManagementConfig(
        atr_period=14,
        take_profit_atr_multiplier=2.0,
        stop_loss_atr_multiplier=1.5,
        horizon_candles=12,
        confidence_threshold=0.55,
    )


def _market_bars(
    candidates: list[dict[str, Any]],
    source_report: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], DataFrame], list[dict[str, Any]]]:
    """Load byte-identical V4-H minute candles and cache timeframe bars."""
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
    trades: list[dict[str, Any]], threshold: float, horizon: int
) -> dict[str, int]:
    """Fail closed on non-Spot, low-confidence, or over-horizon trades."""
    counts = {"LONG": 0, "SHORT": 0}
    for trade in trades:
        direction = str(trade.get("direction"))
        if direction not in counts:
            raise ValueError("economic_trade_direction_invalid")
        counts[direction] += 1
        if direction != "LONG":
            raise RuntimeError("economic_spot_only_position_invariant_violation")
        confidence = float(trade.get("entry_confidence", 0.0))
        if not math.isfinite(confidence) or confidence < threshold:
            raise RuntimeError("v4l_trade_confidence_invariant_violation")
        held = int(trade.get("bars_held", 0))
        if held < 1 or held > horizon:
            raise RuntimeError("v4l_trade_horizon_invariant_violation")
        if not (
            0
            < float(trade["stop_loss_price"])
            < float(trade["barrier_reference_price"])
            < float(trade["take_profit_price"])
        ):
            raise RuntimeError("v4l_trade_barrier_order_invariant_violation")
    return counts


def _candidate_threshold_result(
    candidate: dict[str, Any],
    threshold: float,
    replay: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    """Keep requested metrics and a complete auditable trade ledger."""
    directions = _validate_trade_ledger(replay["trades"], threshold, horizon)
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "confidence_threshold": threshold,
        "raw_signal_counts": replay["raw_signal_counts"],
        "confidence_filter": replay["confidence_filter"],
        "metrics": replay["metrics"],
        "total_fees_usd": replay["total_fees_usd"],
        "total_slippage_cost_usd": replay["total_slippage_cost_usd"],
        "exit_reason_counts": replay["exit_reason_counts"],
        "trade_direction_counts": directions,
        "trade_management": replay["trade_management"],
        "trades": replay["trades"],
    }


def _flatten_trades(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten ledgers from the independent candidate replays."""
    return [trade for item in candidates for trade in item["trades"]]


def _aggregate_exit_counts(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Sum exit reasons without combining candidate equity curves."""
    exits: dict[str, int] = {}
    for item in candidates:
        for reason, count in item["exit_reason_counts"].items():
            exits[reason] = exits.get(reason, 0) + int(count)
    return exits


def _best_defined_pf(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the highest defined per-model PF for descriptive comparison."""
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


def _aggregate_trade_outcomes(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize net trade outcomes across independent candidate ledgers."""
    wins: list[float] = []
    losses: list[float] = []
    for trade in trades:
        pnl = float(trade["net_pnl_usd"])
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(pnl)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    trade_count = len(trades)
    return {
        "total_trade_count": trade_count,
        "winning_trade_count": len(wins),
        "losing_trade_count": len(losses),
        "win_rate": round(len(wins) / trade_count, 8) if trade_count else 0.0,
        "combined_ledger_profit_factor": (
            round(gross_profit / gross_loss, 8) if gross_loss > 0 else None
        ),
        "combined_ledger_gross_profit_usd": round(gross_profit, 8),
        "combined_ledger_gross_loss_usd": round(gross_loss, 8),
    }


def _aggregate_candidate_metrics(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize defined PF and active/all-candidate drawdown values."""
    defined_pf: list[float] = []
    active_dd: list[float] = []
    all_dd: list[float] = []
    active_count = 0
    for candidate in candidates:
        metrics = candidate["metrics"]
        profit_factor = metrics["profit_factor_net_of_costs"]
        if profit_factor is not None:
            defined_pf.append(float(profit_factor))
        drawdown = float(metrics["max_drawdown_percent"])
        all_dd.append(drawdown)
        if int(metrics["trade_count"]) > 0:
            active_count += 1
            active_dd.append(drawdown)
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
    }


def _aggregate_confidence_filter(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Sum raw, rejected, and accepted BUY entries for one threshold."""
    raw_buys = 0
    rejected_buys = 0
    for candidate in candidates:
        confidence_filter = candidate["confidence_filter"]
        raw_buys += int(confidence_filter["raw_buy_signal_count"])
        rejected_buys += int(confidence_filter["rejected_buy_signal_count"])
    return {
        "raw_buy_signal_count": raw_buys,
        "rejected_buy_signal_count": rejected_buys,
        "accepted_buy_signal_count": raw_buys - rejected_buys,
    }


def _aggregate_acceptance_counts(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Count candidate-level economic-gate and strict PF passes."""
    gate_passes = 0
    pf_passes = 0
    for candidate in candidates:
        metrics = candidate["metrics"]
        if bool(metrics["economic_candidate_gate_passed"]):
            gate_passes += 1
        profit_factor = metrics["profit_factor_net_of_costs"]
        if profit_factor is not None and float(profit_factor) > 1.5:
            pf_passes += 1
    return {
        "economic_gate_passed_candidate_count": gate_passes,
        "pf_above_1_5_candidate_count": pf_passes,
    }


def _aggregate_threshold(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate independent model ledgers for one confidence threshold."""
    if len(candidates) != 13:
        raise ValueError("v4l_threshold_candidate_results_invalid")
    trades = _flatten_trades(candidates)
    outcomes = _aggregate_trade_outcomes(trades)
    metrics = _aggregate_candidate_metrics(candidates)
    filters = _aggregate_confidence_filter(candidates)
    acceptance = _aggregate_acceptance_counts(candidates)
    exits = _aggregate_exit_counts(candidates)
    best = _best_defined_pf(candidates)
    return {
        "candidate_count": len(candidates),
        **outcomes,
        **metrics,
        **filters,
        **acceptance,
        "exit_reason_counts": exits,
        "sl_exit_count": exits.get("SL_HIT", 0),
        "tp_exit_count": exits.get("TP_HIT", 0),
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
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Replay all 13 candidates independently at each confidence threshold."""
    execution = _execution_config()
    base = _base_management_config()
    by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in thresholds
    }
    for candidate in candidates:
        key = (
            _symbol_for_pair(str(candidate["pair"])),
            str(candidate["timeframe"]),
        )
        sweeps = evaluate_candidate_confidence_sweep(
            candidate,
            bars_cache[key],
            execution,
            base,
            confidence_thresholds=thresholds,
        )
        for sweep in sweeps:
            threshold = float(sweep["confidence_threshold"])
            by_threshold[threshold].append(
                _candidate_threshold_result(
                    candidate,
                    threshold,
                    sweep["candidate_result"],
                    base.horizon_candles,
                )
            )
    threshold_results: list[dict[str, Any]] = [
        {
            "confidence_threshold": threshold,
            "aggregate": _aggregate_threshold(by_threshold[threshold]),
            "candidates": by_threshold[threshold],
        }
        for threshold in thresholds
    ]
    passed = sum(
        item["aggregate"]["economic_gate_passed_candidate_count"]
        for item in threshold_results
    )
    return {
        "schema_version": V4L_SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_report_sha256": source_digest,
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_CONFIDENCE_SWEEP",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "candidate_count": len(candidates),
        "confidence_thresholds": list(thresholds),
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
            "rejected_buy_semantics": "NO_ENTRY_NO_POSITION_CHANGE",
        },
        "fixed_trade_management_config": {
            "atr_period": base.atr_period,
            "take_profit_atr_multiplier": base.take_profit_atr_multiplier,
            "stop_loss_atr_multiplier": base.stop_loss_atr_multiplier,
            "horizon_candles": base.horizon_candles,
            "atr_calculation": "SIMPLE_MEAN_TRUE_RANGE_FULL_WINDOW",
            "atr_reference": "SIGNAL_BAR_CLOSE_ONLY",
            "barrier_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "gap_exit_reference": "BAR_OPEN",
            "intrabar_exit_reference": "BARRIER_PRICE",
            "confidence_filter_applies_to": "BUY_ENTRIES_ONLY",
            "sell_exit_filtering": "DISABLED",
        },
        "threshold_results": threshold_results,
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_threshold_evaluation_count": len(candidates) * len(thresholds),
            "passed_candidate_threshold_pair_count": passed,
            "status": "MET" if passed else "NOT_MET",
        },
        "validation_bias_caveat": (
            "V4-L replays the same V4-H Forward-Validation probabilities and candidate "
            "selection interval. This is exploratory sensitivity analysis, not an "
            "independent out-of-sample promotion test."
        ),
        "model_artifact_limitation": (
            "V4-H contains persisted row-level probabilities and thresholds, not fitted "
            "estimators. V4-L performs no training or inference."
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
    """Flatten all model/threshold comparisons into auditable CSV rows."""
    rows: list[dict[str, Any]] = []
    for threshold_result in report["threshold_results"]:
        threshold = threshold_result["confidence_threshold"]
        for item in threshold_result["candidates"]:
            metrics = item["metrics"]
            exits = item["exit_reason_counts"]
            rows.append(
                {
                    "confidence_threshold": threshold,
                    "candidate_id": item["candidate_id"],
                    "pair": item["pair"],
                    "timeframe": item["timeframe"],
                    "feature_profile": item["feature_profile"],
                    "weighting_policy": item["weighting_policy"],
                    "trade_count": metrics["trade_count"],
                    "profit_factor": metrics["profit_factor_net_of_costs"],
                    "max_drawdown_percent": metrics["max_drawdown_percent"],
                    "win_rate": metrics["win_rate"],
                    "total_fees_usd": item["total_fees_usd"],
                    "total_slippage_cost_usd": item["total_slippage_cost_usd"],
                    "sl_exit_count": exits.get("SL_HIT", 0),
                    "tp_exit_count": exits.get("TP_HIT", 0),
                    "collision_exit_count": exits.get("COLLISION", 0),
                    "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                    "sell_exit_count": exits.get("SELL", 0),
                    "confidence_rejected_buy_signal_count": item["confidence_filter"][
                        "rejected_buy_signal_count"
                    ],
                    "economic_candidate_gate_passed": metrics[
                        "economic_candidate_gate_passed"
                    ],
                }
            )
    return rows


def write_comparison_csv(report: dict[str, Any], output_path: Path) -> None:
    """Write one row for every candidate/threshold pair."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(report))


def render_v4l_markdown(report: dict[str, Any]) -> str:
    """Render aggregate and per-candidate comparisons in GitHub Step Summary."""
    lines = [
        "## AliX V4-L Confidence Filter Sweep",
        "",
        f"V4-L commit SHA: {report['commit_sha']}",
        f"V4-H source SHA: {report['source_v4h_commit_sha']}",
        "",
        "Replay only: the same 13 persisted V4-H probability streams.",
        "Only BUY confidence changes; TP=2.0 ATR, SL=1.5 ATR, horizon=12, costs remain fixed.",
        "",
        "| Threshold | Trades | Combined-ledger PF | Mean model PF | Active mean DD | Win rate | SL | Fees USD | Slippage USD | Gate passes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["threshold_results"]:
        aggregate = item["aggregate"]
        pooled_pf = aggregate["combined_ledger_profit_factor"]
        mean_pf = aggregate["mean_defined_candidate_profit_factor"]
        active_dd = aggregate["mean_max_drawdown_percent_active_candidates"]
        lines.append(
            f"| {item['confidence_threshold']:.2f} "
            f"| {aggregate['total_trade_count']} "
            f"| {pooled_pf if pooled_pf is not None else 'N/A'} "
            f"| {mean_pf if mean_pf is not None else 'N/A'} "
            f"| {f'{active_dd:.2f}%' if active_dd is not None else 'N/A'} "
            f"| {aggregate['win_rate']:.2%} "
            f"| {aggregate['sl_exit_count']} "
            f"| {aggregate['total_fees_usd']:.2f} "
            f"| {aggregate['total_slippage_cost_usd']:.2f} "
            f"| {aggregate['economic_gate_passed_candidate_count']}/13 |"
        )
    lines.extend(
        [
            "",
            "### Per-model results for every threshold",
            "",
            "| Threshold | Pair | TF | Profile | Weight | Trades | PF | DD | Win rate | Fees | Slippage | SL | Gate |",
            "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["threshold_results"]:
        threshold = item["confidence_threshold"]
        for candidate in item["candidates"]:
            metrics = candidate["metrics"]
            pf = metrics["profit_factor_net_of_costs"]
            lines.append(
                f"| {threshold:.2f} | {candidate['pair']} | {candidate['timeframe']} "
                f"| {candidate['feature_profile']} | {candidate['weighting_policy']} "
                f"| {metrics['trade_count']} | {pf if pf is not None else 'N/A'} "
                f"| {metrics['max_drawdown_percent']:.2f}% "
                f"| {metrics['win_rate']:.2%} "
                f"| {candidate['total_fees_usd']:.2f} "
                f"| {candidate['total_slippage_cost_usd']:.2f} "
                f"| {candidate['exit_reason_counts'].get('SL_HIT', 0)} "
                f"| {'PASS' if metrics['economic_candidate_gate_passed'] else 'NO-GO'} |"
            )
    lines.extend(
        [
            "",
            f"Candidate/threshold pairs passing PF > 1.5 and Max DD < 15%: "
            f"{report['economic_acceptance']['passed_candidate_threshold_pair_count']}/"
            f"{report['economic_acceptance']['candidate_threshold_evaluation_count']}.",
            "",
            "Combined-ledger PF is descriptive across independent model replays, not a single portfolio.",
            "The source interval is V4-H Forward-Validation, not independent OOS.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0, orders_or_execution_api_used=false.",
            "No model training or inference. Promotion remains blocked pending independent OOS and owner review.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4l_confidence_sweep(
    source_report_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    *,
    commit_sha: str,
    confidence_thresholds: list[float] | tuple[float, ...] = (
        V4L_CONFIDENCE_SWEEP_THRESHOLDS
    ),
) -> dict[str, Any]:
    """Validate V4-H source provenance and replay all approved candidates."""
    thresholds = validate_confidence_thresholds(confidence_thresholds)
    source_digest = _digest(source_report_path)
    if source_digest != V4H_REPORT_SHA:
        raise ValueError("v4h_source_report_digest_mismatch")
    source_report = _load_json(source_report_path)
    validate_source_report(source_report)
    candidates = extract_qualifying_candidates(source_report)
    if len(candidates) != 13:
        raise ValueError("v4l_expected_13_v4h_candidates")
    bars_cache, provenance = _market_bars(
        candidates, source_report, input_dir, commit_sha
    )
    report = _build_report(
        commit_sha, source_digest, candidates, bars_cache, provenance, thresholds
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
            summary.write(render_v4l_markdown(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": report["candidate_count"],
                "thresholds": report["confidence_thresholds"],
                "threshold_aggregates": [
                    {
                        "threshold": item["confidence_threshold"],
                        "aggregate": item["aggregate"],
                    }
                    for item in report["threshold_results"]
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
    """Parse the immutable-source V4-L GitHub Actions invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=list(V4L_CONFIDENCE_SWEEP_THRESHOLDS),
    )
    args = parser.parse_args()
    run_v4l_confidence_sweep(
        args.source_report,
        args.input_dir,
        args.output,
        args.output_csv,
        commit_sha=args.commit_sha,
        confidence_thresholds=args.confidence_thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
