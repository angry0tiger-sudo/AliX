"""Run V4-J Spot-only cost isolation using persisted V4-H probabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from services.ml_core.economic_backtest import (
    V4H_SOURCE_ARTIFACT_ID,
    V4H_SOURCE_RUN_ID,
    V4H_SOURCE_SHA,
    ExecutionConfig,
    build_forward_signals,
    evaluate_candidate,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_execution_config,
)

V4I_BASELINE_SCHEMA = "alix.ml-core.v4i-economic-evaluation.v1"
V4I_BASELINE_SHA = "6d30de1459ef113e4ef12bc3b07bb44d1001f2b5"
V4I_BASELINE_RUN_ID = 41311533034
V4I_BASELINE_ARTIFACT_ID = 13087625821
V4I_BASELINE_REPORT_SHA256 = (
    "980930ba0e088860d896b642f84eba0c5333a1de837f030b9ef272cfad120fc2"
)
V4J_SCHEMA = "alix.ml-core.v4j-cost-isolation.v1"
COST_PROFILE_SPECS = (
    ("zero_cost", 0.0, 0.0),
    ("fee_only", 25.0, 0.0),
    ("realistic", 25.0, 20.0),
)
CSV_FIELDS = (
    "candidate_id",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "mode",
    "spot_only",
    "profit_factor",
    "profit_factor_infinite_no_losses",
    "max_drawdown_percent",
    "net_return_percent",
    "trade_count",
    "win_rate",
    "average_win_usd",
    "average_loss_usd",
    "average_net_pnl_per_trade_usd",
    "total_fees_usd",
    "total_slippage_cost_usd",
    "long_trade_count",
    "short_trade_count",
    "economic_candidate_gate_passed",
    "max_drawdown_delta_vs_v4i_percentage_points",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    if not path.is_file():
        raise ValueError(f"required_json_file_missing:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("economic_json_root_must_be_object")
    return payload


def _symbol_for_pair(pair: str) -> str:
    """Resolve the approved V4-H symbol universe."""
    mapping = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    symbol = mapping.get(pair)
    if symbol is None:
        raise ValueError("economic_pair_out_of_policy")
    return symbol


def cost_profile_configs(
    *,
    initial_equity: float = 10_000.0,
    position_fraction: float = 1.0,
    maximum_drawdown: float = 0.15,
) -> dict[str, ExecutionConfig]:
    """Return the three fixed and auditable V4-J cost configurations."""
    configurations: dict[str, ExecutionConfig] = {}
    for name, fee_bps, slippage_bps in COST_PROFILE_SPECS:
        config = ExecutionConfig(
            initial_equity=initial_equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            position_fraction=position_fraction,
            maximum_drawdown=maximum_drawdown,
        )
        validate_execution_config(config)
        configurations[name] = config
    return configurations


def validate_v4i_baseline(
    report: dict[str, Any],
    expected_candidate_ids: list[str],
    source_report: dict[str, Any],
) -> None:
    """Require the immutable short-enabled V4-I report as the comparison base."""
    safety = report.get("safety")
    assumptions = report.get("execution_assumptions")
    candidates = report.get("candidates")
    if not isinstance(safety, dict) or not isinstance(assumptions, dict):
        raise ValueError("v4i_baseline_report_contract_invalid")
    if not isinstance(candidates, list):
        raise ValueError("v4i_baseline_report_contract_invalid")
    actual_ids = [str(item.get("candidate_id")) for item in candidates]
    source_datasets = {
        str(item.get("symbol")): item for item in source_report.get("input_datasets", [])
    }
    baseline_datasets = {
        str(item.get("symbol")): item for item in report.get("input_datasets", [])
    }
    dataset_match = set(source_datasets) == {"BTCUSDT", "ETHUSDT"}
    dataset_match = dataset_match and set(baseline_datasets) == set(source_datasets)
    if dataset_match:
        dataset_match = all(
            baseline_datasets[symbol].get("byte_identical_to_v4h") is True
            and baseline_datasets[symbol].get("csv_sha256")
            == source_datasets[symbol].get("csv_sha256")
            for symbol in source_datasets
        )
    valid = (
        report.get("schema_version") == V4I_BASELINE_SCHEMA
        and report.get("commit_sha") == V4I_BASELINE_SHA
        and report.get("source_v4h_commit_sha") == V4H_SOURCE_SHA
        and report.get("source_v4h_run_id") == V4H_SOURCE_RUN_ID
        and report.get("candidate_count") == 13
        and report.get("source_v4h_candidate_count") == 13
        and report.get("economic_evaluation_performed") is True
        and report.get("model_training_performed") is False
        and report.get("model_inference_performed") is False
        and report.get("persisted_probability_replay_only") is True
        and actual_ids == expected_candidate_ids
        and assumptions.get("sell_semantics") == "OPEN_SHORT"
        and assumptions.get("fee_bps_per_side") == 25.0
        and assumptions.get("slippage_bps_per_side") == 20.0
        and safety.get("Champion") is None
        and safety.get("order_permission") is False
        and safety.get("trading_authorization") == "NO-GO"
        and safety.get("private_api_calls") == 0
        and safety.get("real_order_attempts") == 0
        and dataset_match
    )
    if not valid:
        raise ValueError("v4i_baseline_report_contract_invalid")


def _source_metadata_by_symbol(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index the exact V4-H market-data provenance records."""
    datasets = report.get("input_datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError("v4h_source_dataset_manifest_count_invalid")
    indexed = {str(item.get("symbol")): item for item in datasets}
    if set(indexed) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("v4h_source_dataset_symbols_invalid")
    return indexed


def _market_bars_for_candidates(
    candidates: list[dict[str, Any]],
    report: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], Any], list[dict[str, Any]]]:
    """Load byte-identical V4-H minute data and cache resampled contexts."""
    source_metadata = _source_metadata_by_symbol(report)
    minute_data: dict[str, Any] = {}
    bars_cache: dict[tuple[str, str], Any] = {}
    provenance: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = _symbol_for_pair(str(candidate["pair"]))
        if symbol not in minute_data:
            minute_data[symbol] = load_validated_minute_data(
                input_dir,
                source_metadata[symbol],
                current_commit_sha=commit_sha,
            )
            manifest_path = input_dir / f"binance-{symbol.lower()}-1m.manifest.json"
            manifest = _load_json(manifest_path)
            provenance.append(
                {
                    "symbol": symbol,
                    "row_count": int(manifest["row_count"]),
                    "start_utc": manifest["start_utc"],
                    "end_utc": manifest["end_utc"],
                    "csv_sha256": manifest["csv_sha256"],
                    "source_v4h_csv_sha256": source_metadata[symbol]["csv_sha256"],
                    "byte_identical_to_v4h": True,
                    "public_api_only": True,
                    "credentials_used": False,
                }
            )
        key = (symbol, str(candidate["timeframe"]))
        if key not in bars_cache:
            bars_cache[key] = resample_ohlcv(
                minute_data[symbol], str(candidate["timeframe"])
            )
    return bars_cache, provenance


def _trade_direction_counts(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Count Long and Short trades, rejecting unexpected ledger directions."""
    counts = {"LONG": 0, "SHORT": 0}
    for trade in trades:
        direction = str(trade.get("direction"))
        if direction not in counts:
            raise ValueError("economic_trade_direction_invalid")
        counts[direction] += 1
    return counts


def _baseline_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract the V4-I baseline metrics and its actual Long/Short composition."""
    trades = candidate.get("trades")
    metrics = candidate.get("metrics")
    if not isinstance(trades, list) or not isinstance(metrics, dict):
        raise ValueError("v4i_baseline_candidate_payload_invalid")
    direction_counts = _trade_direction_counts(trades)
    short_pnl = sum(
        float(item["net_pnl_usd"])
        for item in trades
        if item.get("direction") == "SHORT"
    )
    return {
        "metrics": metrics,
        "total_fees_usd": float(candidate["total_fees_usd"]),
        "total_slippage_cost_usd": float(candidate["total_slippage_cost_usd"]),
        "trade_direction_counts": direction_counts,
        "short_trade_net_pnl_usd": round(short_pnl, 8),
    }


def _profile_result(
    candidate: dict[str, Any],
    bars: Any,
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Run one candidate under one fixed Spot-only cost profile."""
    result = evaluate_candidate(candidate, bars, config)
    direction_counts = _trade_direction_counts(result["trades"])
    if direction_counts["SHORT"] != 0:
        raise RuntimeError("economic_spot_only_position_invariant_violation")
    return {
        "metrics": result["metrics"],
        "total_fees_usd": result["total_fees_usd"],
        "total_slippage_cost_usd": result["total_slippage_cost_usd"],
        "trade_direction_counts": direction_counts,
        "trades": result["trades"],
        "equity_curve": result["equity_curve"],
    }


def _build_candidate_result(
    candidate: dict[str, Any],
    bars: Any,
    baseline_candidate: dict[str, Any],
    configs: dict[str, ExecutionConfig],
) -> dict[str, Any]:
    """Build the complete per-candidate baseline and Spot-only comparison."""
    baseline = _baseline_candidate_summary(baseline_candidate)
    profiles = {
        name: _profile_result(candidate, bars, config)
        for name, config in configs.items()
    }
    signals = build_forward_signals(candidate)
    realistic_metrics = profiles["realistic"]["metrics"]
    baseline_metrics = baseline["metrics"]
    dd_delta = round(
        realistic_metrics["max_drawdown_percent"]
        - baseline_metrics["max_drawdown_percent"],
        8,
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "scenario": candidate["scenario"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "selected_parameters": candidate["selected_parameters"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "selected_hold_threshold": candidate["selected_hold_threshold"],
        "forward_signal_count": len(signals),
        "signal_counts": {
            signal: sum(item["signal"] == signal for item in signals)
            for signal in ("BUY", "SELL", "HOLD")
        },
        "v4i_short_enabled_realistic_baseline": baseline,
        "spot_only_profiles": profiles,
        "spot_only_realistic_vs_v4i": {
            "max_drawdown_delta_percentage_points": dd_delta,
            "net_return_delta_percentage_points": round(
                realistic_metrics["net_return_percent"]
                - baseline_metrics["net_return_percent"],
                8,
            ),
            "trade_count_delta": (
                realistic_metrics["trade_count"] - baseline_metrics["trade_count"]
            ),
            "total_fees_delta_usd": round(
                profiles["realistic"]["total_fees_usd"]
                - baseline["total_fees_usd"],
                8,
            ),
            "total_slippage_delta_usd": round(
                profiles["realistic"]["total_slippage_cost_usd"]
                - baseline["total_slippage_cost_usd"],
                8,
            ),
        },
    }


def _profile_acceptance(
    candidates: list[dict[str, Any]], profile_name: str
) -> dict[str, Any]:
    """Apply strict numeric PF and drawdown gates to one cost profile."""
    pf_pass = []
    dd_pass = []
    passed = []
    for candidate in candidates:
        metrics = candidate["spot_only_profiles"][profile_name]["metrics"]
        pf = metrics["profit_factor_net_of_costs"]
        if pf is not None and pf > 1.5:
            pf_pass.append(candidate["candidate_id"])
        if metrics["max_drawdown_fraction"] < 0.15:
            dd_pass.append(candidate["candidate_id"])
        if metrics["economic_candidate_gate_passed"]:
            passed.append(candidate["candidate_id"])
    return {
        "candidate_count": len(candidates),
        "profit_factor_above_1_5_count": len(pf_pass),
        "drawdown_below_15_percent_count": len(dd_pass),
        "passed_candidate_count": len(passed),
        "status": "MET" if passed else "NOT_MET",
        "passing_candidate_ids": passed,
    }


def _build_report(
    *,
    commit_sha: str,
    source_report_digest: str,
    baseline_report_digest: str,
    candidates: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    bars_cache: dict[tuple[str, str], Any],
    provenance: list[dict[str, Any]],
    configs: dict[str, ExecutionConfig],
) -> dict[str, Any]:
    """Replay all candidates in all Spot-only cost profiles."""
    baseline_by_id = {
        item["candidate_id"]: item for item in baseline_report["candidates"]
    }
    results = []
    for candidate in candidates:
        symbol = _symbol_for_pair(str(candidate["pair"]))
        bars = bars_cache[(symbol, str(candidate["timeframe"]))]
        results.append(
            _build_candidate_result(
                candidate, bars, baseline_by_id[candidate["candidate_id"]], configs
            )
        )
    acceptance = {name: _profile_acceptance(results, name) for name in configs}
    drawdown_deltas = [
        item["spot_only_realistic_vs_v4i"]["max_drawdown_delta_percentage_points"]
        for item in results
    ]
    improved = sum(value < 0 for value in drawdown_deltas)
    worsened = sum(value > 0 for value in drawdown_deltas)
    unchanged = len(drawdown_deltas) - improved - worsened
    return {
        "schema_version": V4J_SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_artifact_id": V4H_SOURCE_ARTIFACT_ID,
        "source_v4h_report_sha256": source_report_digest,
        "source_v4i_baseline": {
            "schema_version": V4I_BASELINE_SCHEMA,
            "workflow_run_id": V4I_BASELINE_RUN_ID,
            "artifact_id": V4I_BASELINE_ARTIFACT_ID,
            "commit_sha": V4I_BASELINE_SHA,
            "report_sha256": baseline_report_digest,
            "sell_semantics": "OPEN_SHORT",
        },
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_COST_ISOLATION",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "source_v4h_candidate_count": len(candidates),
        "candidate_count": len(results),
        "input_datasets": provenance,
        "cost_profiles": {
            name: {
                "fee_bps_per_side": config.fee_bps,
                "slippage_bps_per_side": config.slippage_bps,
                "round_trip_nominal_cost_bps": 2.0
                * (config.fee_bps + config.slippage_bps),
            }
            for name, config in configs.items()
        },
        "spot_execution_assumptions": {
            "initial_equity_usd": next(iter(configs.values())).initial_equity,
            "position_fraction_of_equity": next(iter(configs.values())).position_fraction,
            "leverage": 1.0,
            "pyramiding": False,
            "concurrent_positions_per_candidate": 1,
            "entry_timing": "NEXT_BAR_OPEN_AFTER_SIGNAL_BAR_CLOSE",
            "buy_semantics": "OPEN_LONG_IF_FLAT",
            "sell_semantics": "CLOSE_LONG_OR_IGNORE_WHEN_FLAT",
            "hold_semantics": "CLOSE_LONG_OR_REMAIN_FLAT",
            "short_positions_supported": False,
            "exit_timing": "NEXT_BAR_OPEN_ON_SELL_OR_HOLD",
            "terminal_position": "FORCE_CLOSE_AT_FINAL_FORWARD_BAR_CLOSE",
            "drawdown_measurement": "BAR_CLOSE_MARK_TO_MARKET_PLUS_ENTRY_AND_EXIT_COST_EVENTS",
            "profit_factor_basis": "SUM_POSITIVE_NET_TRADE_PNL_DIVIDED_BY_ABSOLUTE_SUM_NEGATIVE_NET_TRADE_PNL",
        },
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_fraction": 0.15,
            "by_cost_profile": acceptance,
        },
        "zero_cost_signal_edge_assessment": {
            "profit_factor_above_1_5_count": acceptance["zero_cost"][
                "profit_factor_above_1_5_count"
            ],
            "status": (
                "NO_CANDIDATE_PF_ABOVE_1_5"
                if acceptance["zero_cost"]["profit_factor_above_1_5_count"] == 0
                else "AT_LEAST_ONE_CANDIDATE_PF_ABOVE_1_5"
            ),
            "interpretation": (
                "No-cost PF below the strict threshold means execution costs alone "
                "cannot explain the candidate's failure; the Spot-only signal and "
                "trade-management combination has insufficient measured edge."
            ),
        },
        "spot_only_drawdown_impact_vs_v4i": {
            "comparison": "V4-J Spot-only realistic Max DD minus V4-I short-enabled realistic Max DD",
            "negative_delta_means": "Spot-only reduced maximum drawdown",
            "candidate_count": len(drawdown_deltas),
            "drawdown_reduced_count": improved,
            "drawdown_increased_count": worsened,
            "drawdown_unchanged_count": unchanged,
            "mean_delta_percentage_points": round(
                sum(drawdown_deltas) / len(drawdown_deltas), 8
            ),
            "minimum_delta_percentage_points": min(drawdown_deltas),
            "maximum_delta_percentage_points": max(drawdown_deltas),
        },
        "candidates": results,
        "validation_bias_caveat": (
            "These 13 candidates were selected using HOLD Precision/Recall on this "
            "same post-calibration Forward-Validation interval. V4-J is a diagnostic "
            "cost and Spot-semantics isolation, not an independent promotion test."
        ),
        "model_artifact_limitation": (
            "The V4-H artifact contains persisted row-level probabilities and "
            "thresholds, not serialized fitted estimators. V4-J performs no training "
            "or inference."
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


def _format_pf(metrics: dict[str, Any]) -> str:
    """Format a numeric PF without converting undefined PF to infinity."""
    value = metrics["profit_factor_net_of_costs"]
    return "N/A" if value is None else f"{value:.4f}"


def render_v4j_markdown(report: dict[str, Any]) -> str:
    """Render the candidate-by-candidate Spot-only cost comparison."""
    lines = [
        "## AliX V4-J Spot-Only Cost Isolation",
        "",
        f"V4-J commit SHA: {report['commit_sha']}",
        f"V4-I baseline SHA: {report['source_v4i_baseline']['commit_sha']}",
        "",
        "| Pair | TF | V4-I PF / DD | Zero PF / DD | Fee PF / DD | Realistic PF / DD | Δ DD (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        baseline = candidate["v4i_short_enabled_realistic_baseline"]["metrics"]
        profiles = candidate["spot_only_profiles"]
        realistic = candidate["spot_only_realistic_vs_v4i"]
        lines.append(
            f"| {candidate['pair']} | {candidate['timeframe']} "
            f"| {_format_pf(baseline)} / {baseline['max_drawdown_percent']:.2f}% "
            f"| {_format_pf(profiles['zero_cost']['metrics'])} / "
            f"{profiles['zero_cost']['metrics']['max_drawdown_percent']:.2f}% "
            f"| {_format_pf(profiles['fee_only']['metrics'])} / "
            f"{profiles['fee_only']['metrics']['max_drawdown_percent']:.2f}% "
            f"| {_format_pf(profiles['realistic']['metrics'])} / "
            f"{profiles['realistic']['metrics']['max_drawdown_percent']:.2f}% "
            f"| {realistic['max_drawdown_delta_percentage_points']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "### Profile acceptance",
            "",
            "| Profile | PF > 1.5 | DD < 15% | Both gates |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in report["economic_acceptance"]["by_cost_profile"].items():
        lines.append(
            f"| {name} | {item['profit_factor_above_1_5_count']}/13 "
            f"| {item['drawdown_below_15_percent_count']}/13 "
            f"| {item['passed_candidate_count']}/13 |"
        )
    lines.extend(
        [
            "",
            f"Zero-cost candidates with numeric PF > 1.5: "
            f"{report['zero_cost_signal_edge_assessment']['profit_factor_above_1_5_count']}/13.",
            "",
            "Spot-only realistic DD change versus V4-I: "
            f"{report['spot_only_drawdown_impact_vs_v4i']['mean_delta_percentage_points']:+.2f} "
            "percentage points on average (negative means lower DD).",
            "",
            "Safety: Champion=null, order_permission=false, "
            "trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "No model training or inference; persisted V4-H probabilities only.",
            "V4-J uses the same Forward-Validation interval as V4-H candidate selection; "
            "independent OOS remains mandatory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Create four auditable comparison rows for every candidate."""
    rows: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        baseline = candidate["v4i_short_enabled_realistic_baseline"]
        baseline_dd = baseline["metrics"]["max_drawdown_percent"]
        modes = [("v4i_short_enabled_realistic_baseline", False, baseline, 0.0)]
        for name, result in candidate["spot_only_profiles"].items():
            delta = result["metrics"]["max_drawdown_percent"] - baseline_dd
            modes.append((name, True, result, round(delta, 8)))
        for mode, spot_only, result, dd_delta in modes:
            metrics = result["metrics"]
            directions = result["trade_direction_counts"]
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "mode": mode,
                    "spot_only": spot_only,
                    "profit_factor": metrics["profit_factor_net_of_costs"],
                    "profit_factor_infinite_no_losses": metrics[
                        "profit_factor_infinite_no_losses"
                    ],
                    "max_drawdown_percent": metrics["max_drawdown_percent"],
                    "net_return_percent": metrics["net_return_percent"],
                    "trade_count": metrics["trade_count"],
                    "win_rate": metrics["win_rate"],
                    "average_win_usd": metrics["average_win_usd"],
                    "average_loss_usd": metrics["average_loss_usd"],
                    "average_net_pnl_per_trade_usd": metrics[
                        "average_net_pnl_per_trade_usd"
                    ],
                    "total_fees_usd": result["total_fees_usd"],
                    "total_slippage_cost_usd": result["total_slippage_cost_usd"],
                    "long_trade_count": directions["LONG"],
                    "short_trade_count": directions["SHORT"],
                    "economic_candidate_gate_passed": metrics[
                        "economic_candidate_gate_passed"
                    ],
                    "max_drawdown_delta_vs_v4i_percentage_points": dd_delta,
                }
            )
    return rows


def write_comparison_csv(report: dict[str, Any], output_path: Path) -> None:
    """Write a four-mode-per-candidate CSV comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_comparison_csv_rows(report))


def run_v4j_cost_isolation(
    source_report_path: Path,
    baseline_report_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    *,
    commit_sha: str,
) -> dict[str, Any]:
    """Run fixed Spot-only cost profiles from persisted V4-H probability rows."""
    source_bytes = source_report_path.read_bytes()
    baseline_bytes = baseline_report_path.read_bytes()
    source_report = _load_json(source_report_path)
    baseline_report = _load_json(baseline_report_path)
    baseline_digest = hashlib.sha256(baseline_bytes).hexdigest()
    if baseline_digest != V4I_BASELINE_REPORT_SHA256:
        raise ValueError("v4i_baseline_report_digest_mismatch")
    candidates = extract_qualifying_candidates(source_report)
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    validate_v4i_baseline(baseline_report, candidate_ids, source_report)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    configs = cost_profile_configs()
    bars_cache, provenance = _market_bars_for_candidates(
        candidates, source_report, input_dir, commit_sha
    )
    report = _build_report(
        commit_sha=commit_sha,
        source_report_digest=source_digest,
        baseline_report_digest=baseline_digest,
        candidates=candidates,
        baseline_report=baseline_report,
        bars_cache=bars_cache,
        provenance=provenance,
        configs=configs,
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
            summary.write(render_v4j_markdown(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": report["candidate_count"],
                "cost_profiles": list(report["cost_profiles"]),
                "zero_cost_pf_above_1_5_count": report[
                    "zero_cost_signal_edge_assessment"
                ]["profit_factor_above_1_5_count"],
                "drawdown_impact": report["spot_only_drawdown_impact_vs_v4i"],
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
    """Parse the GitHub Actions V4-J invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_v4j_cost_isolation(
        args.source_report,
        args.baseline_report,
        args.input_dir,
        args.output,
        args.output_csv,
        commit_sha=args.commit_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
