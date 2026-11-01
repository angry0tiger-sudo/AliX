"""Run V4-N trailing-stop replay on persisted V4-H probabilities only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pandas import DataFrame

from services.ml_core.economic_backtest import (
    V4H_SOURCE_RUN_ID,
    V4H_SOURCE_SHA,
    V4N_CONFIDENCE_THRESHOLD,
    V4N_STOP_LOSS_ATR_MULTIPLIER,
    V4N_TAKE_PROFIT_ATR_MULTIPLIER,
    V4N_TRAILING_ACTIVATION_ATR,
    V4N_TRAILING_STEP_ATR,
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_execution_config,
    validate_source_report,
)

SOURCE_SHA256 = "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
SCHEMA = "alix.ml-core.v4n-trailing-stop.v1"
HORIZON = 12
ATR_PERIOD = 14
MAX_DRAWDOWN = 0.15
V4M_RUN_ID = 41345541356
V4M_BEST_LEDGER_PF = 0.04894419
V4M_SAME_GEOMETRY_LEDGER_PF = 0.03802472
CSV_FIELDS = (
    "candidate_id",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "confidence_threshold",
    "stop_loss_atr_multiplier",
    "take_profit_atr_multiplier",
    "trade_count",
    "profit_factor",
    "max_drawdown_percent",
    "win_rate",
    "initial_stop_exit_count",
    "breakeven_stop_exit_count",
    "trailing_stop_exit_count",
    "tp_exit_count",
    "horizon_exit_count",
    "collision_exit_count",
    "sell_exit_count",
    "total_fees_usd",
    "total_slippage_cost_usd",
    "economic_candidate_gate_passed",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    if not path.is_file():
        raise ValueError(f"required_json_file_missing:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("economic_json_root_must_be_object")
    return value


def _market_bars(
    candidates: list[dict[str, Any]],
    source: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], DataFrame], list[dict[str, Any]]]:
    """Load and verify the byte-identical V4-H minute data, then cache bars."""
    datasets = source.get("input_datasets")
    if not isinstance(datasets, list):
        raise ValueError("v4h_source_dataset_manifest_invalid")
    metadata = {str(item["symbol"]): item for item in datasets}
    if set(metadata) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("v4h_source_dataset_manifest_invalid")
    symbols = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    minute: dict[str, DataFrame] = {}
    bars: dict[tuple[str, str], DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = symbols.get(str(candidate["pair"]))
        if symbol is None:
            raise ValueError("economic_pair_out_of_policy")
        if symbol not in minute:
            minute[symbol] = load_validated_minute_data(
                input_dir, metadata[symbol], current_commit_sha=commit_sha
            )
            manifest = _load_json(
                input_dir / f"binance-{symbol.lower()}-1m.manifest.json"
            )
            provenance.append(
                {
                    "symbol": symbol,
                    "row_count": manifest["row_count"],
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
        if key not in bars:
            bars[key] = resample_ohlcv(minute[symbol], key[1])
    return bars, provenance


def _execution() -> ExecutionConfig:
    """Use the fixed V4-M cash, cost, and Spot-only assumptions."""
    config = ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=25.0,
        slippage_bps=20.0,
        position_fraction=1.0,
        maximum_drawdown=MAX_DRAWDOWN,
    )
    validate_execution_config(config)
    return config


def _management() -> TradeManagementConfig:
    """Return the fixed V4-N geometry and stepwise trailing-stop policy."""
    return TradeManagementConfig(
        atr_period=ATR_PERIOD,
        take_profit_atr_multiplier=V4N_TAKE_PROFIT_ATR_MULTIPLIER,
        stop_loss_atr_multiplier=V4N_STOP_LOSS_ATR_MULTIPLIER,
        horizon_candles=HORIZON,
        confidence_threshold=V4N_CONFIDENCE_THRESHOLD,
        trailing_stop_enabled=True,
        trailing_activation_atr=V4N_TRAILING_ACTIVATION_ATR,
        trailing_step_atr=V4N_TRAILING_STEP_ATR,
    )


def _validate_trade_direction_confidence_and_horizon(
    trade: dict[str, Any],
) -> None:
    """Enforce Spot-long, confidence, and vertical-barrier constraints."""
    if trade["direction"] != "LONG":
        raise RuntimeError("economic_spot_only_position_invariant_violation")
    if float(trade["entry_confidence"]) < V4N_CONFIDENCE_THRESHOLD:
        raise RuntimeError("v4n_trade_confidence_invariant_violation")
    if not 1 <= int(trade["bars_held"]) <= HORIZON:
        raise RuntimeError("v4n_trade_horizon_invariant_violation")


def _validate_trade_barrier_geometry(trade: dict[str, Any]) -> None:
    """Enforce the immutable initial two-ATR stop and three-ATR target."""
    reference = float(trade["barrier_reference_price"])
    atr = float(trade["atr_value"])
    initial_stop = reference - V4N_STOP_LOSS_ATR_MULTIPLIER * atr
    take_profit = reference + V4N_TAKE_PROFIT_ATR_MULTIPLIER * atr
    if abs(float(trade["initial_stop_loss_price"]) - initial_stop) > 1e-7:
        raise RuntimeError("v4n_initial_stop_geometry_invariant_violation")
    if abs(float(trade["take_profit_price"]) - take_profit) > 1e-7:
        raise RuntimeError("v4n_take_profit_geometry_invariant_violation")


def _validate_trade_trailing_stop(trade: dict[str, Any]) -> None:
    """Enforce a non-loosening stop and a non-negative step counter."""
    if float(trade["stop_loss_price"]) + 1e-9 < float(
        trade["initial_stop_loss_price"]
    ):
        raise RuntimeError("v4n_trailing_stop_monotonicity_invariant_violation")
    if int(trade["trailing_stop_steps"]) < 0:
        raise RuntimeError("v4n_trailing_stop_steps_invalid")


def _validate_candidate_trades(trades: list[dict[str, Any]]) -> None:
    """Apply every per-trade execution invariant before aggregating a candidate."""
    for trade in trades:
        _validate_trade_direction_confidence_and_horizon(trade)
        _validate_trade_barrier_geometry(trade)
        _validate_trade_trailing_stop(trade)


def _evaluate(
    candidate: dict[str, Any],
    market_bars: DataFrame,
    execution: ExecutionConfig,
    management: TradeManagementConfig,
) -> dict[str, Any]:
    """Replay persisted probabilities and validate every emitted trade."""
    replay = evaluate_candidate(candidate, market_bars, execution, management)
    trades = replay["trades"]
    _validate_candidate_trades(trades)
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "confidence_threshold": V4N_CONFIDENCE_THRESHOLD,
        "stop_loss_atr_multiplier": V4N_STOP_LOSS_ATR_MULTIPLIER,
        "take_profit_atr_multiplier": V4N_TAKE_PROFIT_ATR_MULTIPLIER,
        "metrics": replay["metrics"],
        "trades": trades,
        "exit_reason_counts": replay["exit_reason_counts"],
        "stop_exit_type_counts": replay["stop_exit_type_counts"],
        "initial_stop_exit_count": replay["initial_stop_exit_count"],
        "breakeven_stop_exit_count": replay["breakeven_stop_exit_count"],
        "trailing_stop_exit_count": replay["trailing_stop_exit_count"],
        "total_fees_usd": replay["total_fees_usd"],
        "total_slippage_cost_usd": replay["total_slippage_cost_usd"],
        "trade_direction_counts": {
            "LONG": sum(trade["direction"] == "LONG" for trade in trades),
            "SHORT": sum(trade["direction"] == "SHORT" for trade in trades),
        },
        "trade_management": replay["trade_management"],
    }


def _trade_outcomes(
    trades: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Separate winning and losing net trade outcomes across independent ledgers."""
    wins = [float(trade["net_pnl_usd"]) for trade in trades if trade["net_pnl_usd"] > 0]
    losses = [
        float(trade["net_pnl_usd"]) for trade in trades if trade["net_pnl_usd"] < 0
    ]
    return wins, losses


def _defined_candidate_profit_factors(
    results: list[dict[str, Any]],
) -> list[float]:
    """Collect defined per-candidate net profit factors."""
    return [
        float(result["metrics"]["profit_factor_net_of_costs"])
        for result in results
        if result["metrics"]["profit_factor_net_of_costs"] is not None
    ]


def _candidate_drawdowns(
    results: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Return active-candidate and all-candidate maximum drawdown samples."""
    active = [
        float(result["metrics"]["max_drawdown_percent"])
        for result in results
        if int(result["metrics"]["trade_count"]) > 0
    ]
    all_candidates = [
        float(result["metrics"]["max_drawdown_percent"]) for result in results
    ]
    return active, all_candidates


def _sum_count_maps(
    results: list[dict[str, Any]], field: str
) -> dict[str, int]:
    """Sum one named exit-count mapping across independent candidate replays."""
    total: dict[str, int] = {}
    for result in results:
        for key, value in result[field].items():
            total[key] = total.get(key, 0) + int(value)
    return total


def _best_defined_candidate(
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the candidate with the highest defined net profit factor."""
    return max(
        (
            result
            for result in results
            if result["metrics"]["profit_factor_net_of_costs"] is not None
        ),
        key=lambda result: float(result["metrics"]["profit_factor_net_of_costs"]),
        default=None,
    )


def _gate_pass_counts(
    results: list[dict[str, Any]], drawdowns: list[float]
) -> tuple[int, int, int]:
    """Count candidates meeting PF, drawdown, and combined economic gates."""
    passed = sum(
        bool(result["metrics"]["economic_candidate_gate_passed"])
        for result in results
    )
    pf_passed = sum(
        result["metrics"]["profit_factor_net_of_costs"] is not None
        and float(result["metrics"]["profit_factor_net_of_costs"]) > 1.5
        for result in results
    )
    drawdown_passed = sum(value < 15.0 for value in drawdowns)
    return passed, pf_passed, drawdown_passed


def _optional_mean(values: list[float]) -> float | None:
    """Return an eight-decimal mean for a non-empty sample."""
    if not values:
        return None
    return round(sum(values) / len(values), 8)


def _ledger_profit_factor(wins: list[float], losses: list[float]) -> float | None:
    """Return descriptive pooled PF for independent replay ledgers."""
    if not losses:
        return None
    return round(sum(wins) / abs(sum(losses)), 8)


def _win_rate(wins: list[float], losses: list[float]) -> float:
    """Return the pooled win rate, treating empty trade ledgers as zero."""
    total = len(wins) + len(losses)
    if total == 0:
        return 0.0
    return round(len(wins) / total, 8)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent ledgers descriptively without merging equity curves."""
    if len(results) != 13:
        raise ValueError("v4n_candidate_count_invalid")
    trades = [trade for result in results for trade in result["trades"]]
    wins, losses = _trade_outcomes(trades)
    profit_factors = _defined_candidate_profit_factors(results)
    active_drawdowns, drawdowns = _candidate_drawdowns(results)
    reasons = _sum_count_maps(results, "exit_reason_counts")
    stop_types = _sum_count_maps(results, "stop_exit_type_counts")
    passed, pf_passed, drawdown_passed = _gate_pass_counts(results, drawdowns)
    best = _best_defined_candidate(results)
    return {
        "candidate_count": len(results),
        "active_candidate_count": len(active_drawdowns),
        "total_trade_count": len(trades),
        "winning_trade_count": len(wins),
        "losing_trade_count": len(losses),
        "win_rate": _win_rate(wins, losses),
        "combined_ledger_profit_factor": _ledger_profit_factor(wins, losses),
        "mean_defined_candidate_profit_factor": _optional_mean(profit_factors),
        "defined_candidate_profit_factor_count": len(profit_factors),
        "mean_max_drawdown_percent_active_candidates": _optional_mean(active_drawdowns),
        "mean_max_drawdown_percent_all_candidates": _optional_mean(drawdowns),
        "worst_candidate_max_drawdown_percent": round(max(drawdowns), 8),
        "economic_gate_passed_candidate_count": passed,
        "pf_above_1_5_candidate_count": pf_passed,
        "drawdown_below_15_candidate_count": drawdown_passed,
        "exit_reason_counts": reasons,
        "stop_exit_type_counts": stop_types,
        "initial_stop_exit_count": stop_types.get("INITIAL_STOP", 0),
        "breakeven_stop_exit_count": stop_types.get("BREAKEVEN_STOP", 0),
        "trailing_stop_exit_count": stop_types.get("TRAILING_STOP", 0),
        "tp_exit_count": reasons.get("TP_HIT", 0),
        "horizon_exit_count": reasons.get("HORIZON_EXPIRY", 0),
        "collision_exit_count": reasons.get("COLLISION", 0),
        "sell_exit_count": reasons.get("SELL", 0),
        "total_fees_usd": round(
            sum(float(result["total_fees_usd"]) for result in results), 8
        ),
        "total_slippage_cost_usd": round(
            sum(float(result["total_slippage_cost_usd"]) for result in results), 8
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


def _report(
    commit_sha: str,
    source_digest: str,
    results: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the V4-N report with immutable controls and V4-M references."""
    aggregate = _aggregate(results)
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_report_sha256": source_digest,
        "v4m_reference_run_id": V4M_RUN_ID,
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_TRAILING_STOP",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "candidate_count": 13,
        "confidence_threshold": V4N_CONFIDENCE_THRESHOLD,
        "input_datasets": provenance,
        "fixed_trade_management_config": {
            "confidence_threshold": V4N_CONFIDENCE_THRESHOLD,
            "atr_period": ATR_PERIOD,
            "horizon_candles": HORIZON,
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_atr_multiplier": 3.0,
            "trailing_stop_enabled": True,
            "trailing_activation_atr": 1.0,
            "trailing_step_atr": 1.0,
            "trailing_stop_update_policy": "COMPLETED_BAR_HIGH_NEXT_BAR_EFFECTIVE",
            "trailing_breakeven_policy": "NET_OF_BOTH_SIDE_FEES_AND_SLIPPAGE",
        },
        "execution_assumptions": {
            "initial_equity_usd": 10000.0,
            "fee_bps_per_side": 25.0,
            "slippage_bps_per_side": 20.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
        },
        "v4m_reference": {
            "run_id": V4M_RUN_ID,
            "best_combined_ledger_profit_factor": V4M_BEST_LEDGER_PF,
            "best_combined_ledger_geometry": {"sl_atr": 2.5, "tp_atr": 2.5},
            "same_geometry_combined_ledger_profit_factor": V4M_SAME_GEOMETRY_LEDGER_PF,
            "same_geometry": {"sl_atr": 2.0, "tp_atr": 3.0},
        },
        "aggregate": aggregate,
        "candidates": results,
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_count": 13,
            "passed_candidate_count": aggregate["economic_gate_passed_candidate_count"],
            "status": "MET"
            if aggregate["economic_gate_passed_candidate_count"]
            else "NOT_MET",
        },
        "validation_bias_caveat": (
            "V4-N reuses the V4-H Forward-Validation interval for exploratory exit "
            "management research; independent OOS and owner review remain mandatory."
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
    """Create a per-candidate CSV comparison."""
    rows = []
    for item in report["candidates"]:
        metric, exits = item["metrics"], item["exit_reason_counts"]
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "pair": item["pair"],
                "timeframe": item["timeframe"],
                "feature_profile": item["feature_profile"],
                "weighting_policy": item["weighting_policy"],
                "confidence_threshold": 0.60,
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 3.0,
                "trade_count": metric["trade_count"],
                "profit_factor": metric["profit_factor_net_of_costs"],
                "max_drawdown_percent": metric["max_drawdown_percent"],
                "win_rate": metric["win_rate"],
                "initial_stop_exit_count": item["initial_stop_exit_count"],
                "breakeven_stop_exit_count": item["breakeven_stop_exit_count"],
                "trailing_stop_exit_count": item["trailing_stop_exit_count"],
                "tp_exit_count": exits.get("TP_HIT", 0),
                "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                "collision_exit_count": exits.get("COLLISION", 0),
                "sell_exit_count": exits.get("SELL", 0),
                "total_fees_usd": item["total_fees_usd"],
                "total_slippage_cost_usd": item["total_slippage_cost_usd"],
                "economic_candidate_gate_passed": metric[
                    "economic_candidate_gate_passed"
                ],
            }
        )
    return rows


def render_summary(report: dict[str, Any]) -> str:
    """Render the aggregate and per-model results in the Actions summary."""
    aggregate = report["aggregate"]
    lines = [
        "## AliX V4-N Trailing Stop Replay",
        f"Commit: {report['commit_sha']}",
        "Replay only: persisted V4-H probabilities; no training or inference.",
        "Confidence=0.60; SL=2 ATR; TP=3 ATR; ATR=14; horizon=12.",
        "Trail activation=+1 ATR; net-cost breakeven; +1 ATR per completed-bar step.",
        "",
        "| Trades | Ledger PF | Mean model PF | Active mean DD | Worst model DD | Initial SL | Breakeven | Trailing | TP | Horizon | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['total_trade_count']} | {aggregate['combined_ledger_profit_factor']} "
            f"| {aggregate['mean_defined_candidate_profit_factor']} "
            f"| {aggregate['mean_max_drawdown_percent_active_candidates']}% "
            f"| {aggregate['worst_candidate_max_drawdown_percent']}% "
            f"| {aggregate['initial_stop_exit_count']} | {aggregate['breakeven_stop_exit_count']} "
            f"| {aggregate['trailing_stop_exit_count']} | {aggregate['tp_exit_count']} "
            f"| {aggregate['horizon_exit_count']} | {aggregate['economic_gate_passed_candidate_count']}/13 |"
        ),
        "",
        "| Pair | TF | Profile | Weight | Trades | PF | Max DD | Trailing exits | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        metric = item["metrics"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {item['feature_profile']} "
            f"| {item['weighting_policy']} | {metric['trade_count']} "
            f"| {metric['profit_factor_net_of_costs']} | {metric['max_drawdown_percent']}% "
            f"| {item['trailing_stop_exit_count']} "
            f"| {'PASS' if metric['economic_candidate_gate_passed'] else 'NO-GO'} |"
        )
    lines.extend(
        [
            "",
            f"V4-M best grid PF={V4M_BEST_LEDGER_PF}; same SL=2/TP=3 PF={V4M_SAME_GEOMETRY_LEDGER_PF}.",
            "Combined-ledger PF is descriptive across independent candidate replays, not one portfolio.",
            "Active mean DD is the mean across active candidate replays; worst candidate DD is separate.",
            "Reused Forward-Validation interval: independent OOS remains required.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4n_trailing_stop(
    source_report_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    *,
    commit_sha: str,
) -> dict[str, Any]:
    """Validate immutable V4-H provenance and replay saved probabilities only."""
    source_digest = hashlib.sha256(source_report_path.read_bytes()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError("v4h_source_report_digest_mismatch")
    source = _load_json(source_report_path)
    validate_source_report(source)
    candidates = extract_qualifying_candidates(source)
    if len(candidates) != 13:
        raise ValueError("v4n_expected_13_v4h_candidates")
    bars, provenance = _market_bars(candidates, source, input_dir, commit_sha)
    execution, management = _execution(), _management()
    results = []
    symbols = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    for candidate in candidates:
        key = (symbols[candidate["pair"]], candidate["timeframe"])
        results.append(_evaluate(candidate, bars[key], execution, management))
    report = _report(commit_sha, source_digest, results, provenance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(report))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(render_summary(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": 13,
                "aggregate": report["aggregate"],
                "Champion": None,
                "trading_authorization": "NO-GO",
                "model_training_performed": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    """Parse the V4-N GitHub Actions invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_v4n_trailing_stop(
        args.source_report,
        args.input_dir,
        args.output,
        args.output_csv,
        commit_sha=args.commit_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
