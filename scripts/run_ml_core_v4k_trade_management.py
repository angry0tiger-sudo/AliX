"""Run V4-K Spot-only trade management from persisted V4-H probabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from services.ml_core.economic_backtest import (
    V4H_SOURCE_RUN_ID,
    V4H_SOURCE_SHA,
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_execution_config,
    validate_source_report,
    validate_trade_management_config,
)

V4J_SCHEMA = "alix.ml-core.v4j-cost-isolation.v1"
V4J_SHA = "fe5cf50b0084ea412345e82863569d084548faee"
V4J_RUN_ID = 41320586694
V4J_ARTIFACT_ID = 13091465117
V4J_REPORT_SHA = "74d76163d3c9acf1c36a99180d447d510e8bbb91fb12a70933b743d1e178d44b"
V4J_CSV_SHA = "ac9de8c14df17a6eb4cf7581135004737b15b94e662a87593d2e6ce5e3c92b0d"
V4H_REPORT_SHA = "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
V4K_SCHEMA = "alix.ml-core.v4k-trade-management.v1"
CSV_FIELDS = (
    "candidate_id",
    "pair",
    "timeframe",
    "v4j_profit_factor",
    "v4k_profit_factor",
    "profit_factor_delta",
    "v4j_max_drawdown_percent",
    "v4k_max_drawdown_percent",
    "max_drawdown_delta_percentage_points",
    "v4j_trade_count",
    "v4k_trade_count",
    "trade_count_reduction",
    "trade_count_reduction_percent",
    "v4j_net_return_percent",
    "v4k_net_return_percent",
    "v4k_total_fees_usd",
    "v4k_total_slippage_cost_usd",
    "tp_exit_count",
    "sl_exit_count",
    "collision_exit_count",
    "horizon_exit_count",
    "sell_exit_count",
    "confidence_rejected_buy_signals",
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


def trade_management_config(
    *,
    confidence_threshold: float = 0.55,
    atr_period: int = 14,
    take_profit_atr_multiplier: float = 2.0,
    stop_loss_atr_multiplier: float = 1.5,
    horizon_candles: int = 12,
) -> TradeManagementConfig:
    """Build and validate the V4-H-aligned V4-K configuration."""
    config = TradeManagementConfig(
        atr_period=atr_period,
        take_profit_atr_multiplier=take_profit_atr_multiplier,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        horizon_candles=horizon_candles,
        confidence_threshold=confidence_threshold,
    )
    validate_trade_management_config(config)
    return config


def _execution_config() -> ExecutionConfig:
    """Keep the realistic V4-J fee and slippage assumptions fixed."""
    config = ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=25.0,
        slippage_bps=20.0,
        position_fraction=1.0,
        maximum_drawdown=0.15,
    )
    validate_execution_config(config)
    return config


def _dataset_map(report: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """Index valid source market-data manifests by symbol."""
    datasets = report.get("input_datasets")
    if not isinstance(datasets, list) or any(
        not isinstance(item, dict) for item in datasets
    ):
        return None
    return {str(item.get("symbol")): item for item in datasets}


def _safety_is_closed(report: dict[str, Any]) -> bool:
    """Verify no Champion, order permission, or private execution is enabled."""
    safety = report.get("safety")
    return isinstance(safety, dict) and all(
        (
            safety.get("Champion") is None,
            safety.get("order_permission") is False,
            safety.get("trading_authorization") == "NO-GO",
            safety.get("private_api_calls") == 0,
            safety.get("real_order_attempts") == 0,
        )
    )


def _data_provenance_matches(
    source_report: dict[str, Any], baseline_report: dict[str, Any]
) -> bool:
    """Require both V4-J datasets to be byte-identical to V4-H inputs."""
    source = _dataset_map(source_report)
    baseline = _dataset_map(baseline_report)
    expected = {"BTCUSDT", "ETHUSDT"}
    if source is None or baseline is None:
        return False
    if set(source) != expected or set(baseline) != expected:
        return False
    return all(
        baseline[symbol].get("byte_identical_to_v4h") is True
        and baseline[symbol].get("csv_sha256") == source[symbol].get("csv_sha256")
        for symbol in expected
    )


def _candidate_ids_match(baseline: dict[str, Any], expected_ids: list[str]) -> bool:
    """Require exact candidate identity and ordering."""
    candidates = baseline.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(expected_ids):
        return False
    return all(
        isinstance(item, dict) and item.get("candidate_id") == expected
        for item, expected in zip(candidates, expected_ids, strict=True)
    )


def _v4j_identity_matches(baseline: dict[str, Any]) -> bool:
    """Verify immutable V4-J report identity and replay-only provenance."""
    checks = (
        baseline.get("schema_version") == V4J_SCHEMA,
        baseline.get("commit_sha") == V4J_SHA,
        baseline.get("source_v4h_commit_sha") == V4H_SOURCE_SHA,
        baseline.get("source_v4h_run_id") == V4H_SOURCE_RUN_ID,
        baseline.get("source_v4h_report_sha256") == V4H_REPORT_SHA,
        baseline.get("candidate_count") == 13,
        baseline.get("source_v4h_candidate_count") == 13,
        baseline.get("model_training_performed") is False,
        baseline.get("model_inference_performed") is False,
        baseline.get("persisted_probability_replay_only") is True,
        baseline.get("economic_evaluation_performed") is True,
        baseline.get("promotion_status")
        == "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
    )
    return all(checks)


def _v4j_spot_and_cost_contract_matches(baseline: dict[str, Any]) -> bool:
    """Verify V4-J was Spot-only and realistic costs match this replay."""
    semantics = baseline.get("spot_execution_assumptions")
    costs = (baseline.get("cost_profiles") or {}).get("realistic")
    return (
        isinstance(semantics, dict)
        and isinstance(costs, dict)
        and semantics.get("sell_semantics") == "CLOSE_LONG_OR_IGNORE_WHEN_FLAT"
        and semantics.get("short_positions_supported") is False
        and costs.get("fee_bps_per_side") == 25.0
        and costs.get("slippage_bps_per_side") == 20.0
    )


def validate_v4j_baseline(
    baseline: dict[str, Any],
    source_report: dict[str, Any],
    expected_candidate_ids: list[str],
) -> None:
    """Fail closed unless the exact V4-J report is a valid Spot-only baseline."""
    checks = (
        _v4j_identity_matches(baseline),
        _candidate_ids_match(baseline, expected_candidate_ids),
        _v4j_spot_and_cost_contract_matches(baseline),
        _safety_is_closed(baseline),
        _data_provenance_matches(source_report, baseline),
    )
    if not all(checks):
        raise ValueError("v4j_baseline_report_contract_invalid")


def _source_metadata_by_symbol(
    source_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index exact V4-H dataset provenance records."""
    datasets = _dataset_map(source_report)
    if datasets is None or set(datasets) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("v4h_source_dataset_manifest_invalid")
    return datasets


def _market_bars(
    candidates: list[dict[str, Any]],
    source_report: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], Any], list[dict[str, Any]]]:
    """Load byte-identical V4-H minute candles and cache timeframe bars."""
    metadata = _source_metadata_by_symbol(source_report)
    minute_data: dict[str, Any] = {}
    cache: dict[tuple[str, str], Any] = {}
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


def _candidate_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index V4-J candidate comparison records by stable candidate ID."""
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("v4j_baseline_candidate_payload_invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in indexed:
            raise ValueError("v4j_baseline_candidate_id_duplicate")
        indexed[candidate_id] = candidate
    return indexed


def _baseline_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    """Get the V4-J realistic Spot-only metrics for one candidate."""
    profiles = candidate.get("spot_only_profiles")
    realistic = profiles.get("realistic") if isinstance(profiles, dict) else None
    if not isinstance(realistic, dict) or not isinstance(
        realistic.get("metrics"), dict
    ):
        raise ValueError("v4j_baseline_realistic_metrics_missing")
    directions = realistic.get("trade_direction_counts")
    if not isinstance(directions, dict) or directions.get("SHORT") != 0:
        raise ValueError("v4j_baseline_not_spot_only")
    return realistic["metrics"]


def _validate_trade_ledger(
    trades: list[dict[str, Any]], management: TradeManagementConfig
) -> dict[str, int]:
    """Reject Short trades, low-confidence entries, or over-horizon positions."""
    counts = {"LONG": 0, "SHORT": 0}
    for trade in trades:
        direction = str(trade.get("direction"))
        if direction not in counts:
            raise ValueError("economic_trade_direction_invalid")
        counts[direction] += 1
        if direction != "LONG":
            raise RuntimeError("economic_spot_only_position_invariant_violation")
        if float(trade.get("entry_confidence", 0.0)) < management.confidence_threshold:
            raise RuntimeError("v4k_trade_confidence_invariant_violation")
        held = int(trade.get("bars_held", 0))
        if held < 1 or held > management.horizon_candles:
            raise RuntimeError("v4k_trade_horizon_invariant_violation")
        if not (
            0
            < float(trade["stop_loss_price"])
            < float(trade["barrier_reference_price"])
            < float(trade["take_profit_price"])
        ):
            raise RuntimeError("v4k_trade_barrier_order_invariant_violation")
    return counts


def _candidate_result(
    candidate: dict[str, Any],
    bars: Any,
    baseline_candidate: dict[str, Any],
    execution: ExecutionConfig,
    management: TradeManagementConfig,
) -> dict[str, Any]:
    """Replay one candidate and compare it with V4-J realistic Spot metrics."""
    result = evaluate_candidate(candidate, bars, execution, management)
    directions = _validate_trade_ledger(result["trades"], management)
    baseline = _baseline_metrics(baseline_candidate)
    current = result["metrics"]
    before_count, after_count = (
        int(baseline["trade_count"]),
        int(current["trade_count"]),
    )
    reduction = before_count - after_count
    reduction_pct = round(100.0 * reduction / before_count, 8) if before_count else None
    before_pf, after_pf = (
        baseline["profit_factor_net_of_costs"],
        current["profit_factor_net_of_costs"],
    )
    pf_delta = (
        round(float(after_pf) - float(before_pf), 8)
        if after_pf is not None and before_pf is not None
        else None
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "raw_signal_counts": result["raw_signal_counts"],
        "confidence_filter": result["confidence_filter"],
        "trade_management": result["trade_management"],
        "v4j_baseline": {
            "profit_factor": before_pf,
            "max_drawdown_percent": baseline["max_drawdown_percent"],
            "net_return_percent": baseline["net_return_percent"],
            "trade_count": before_count,
        },
        "v4k_result": {
            "metrics": current,
            "total_fees_usd": result["total_fees_usd"],
            "total_slippage_cost_usd": result["total_slippage_cost_usd"],
            "exit_reason_counts": result["exit_reason_counts"],
            "trade_direction_counts": directions,
            "trades": result["trades"],
        },
        "comparison": {
            "profit_factor_delta": pf_delta,
            "profit_factor_improved": pf_delta is not None and pf_delta > 0,
            "max_drawdown_delta_percentage_points": round(
                float(current["max_drawdown_percent"])
                - float(baseline["max_drawdown_percent"]),
                8,
            ),
            "max_drawdown_reduced": (
                float(current["max_drawdown_percent"])
                < float(baseline["max_drawdown_percent"])
            ),
            "trade_count_reduction": reduction,
            "trade_count_reduction_percent": reduction_pct,
        },
    }


def _aggregate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize PF, drawdown, and trade-count changes across candidates."""
    before = [int(item["v4j_baseline"]["trade_count"]) for item in candidates]
    after = [int(item["v4k_result"]["metrics"]["trade_count"]) for item in candidates]
    before_pfs = [
        float(item["v4j_baseline"]["profit_factor"])
        for item in candidates
        if item["v4j_baseline"]["profit_factor"] is not None
    ]
    after_pfs = [
        float(item["v4k_result"]["metrics"]["profit_factor_net_of_costs"])
        for item in candidates
        if item["v4k_result"]["metrics"]["profit_factor_net_of_costs"] is not None
    ]
    before_dd = [
        float(item["v4j_baseline"]["max_drawdown_percent"]) for item in candidates
    ]
    after_dd = [
        float(item["v4k_result"]["metrics"]["max_drawdown_percent"])
        for item in candidates
    ]
    before_total, after_total = sum(before), sum(after)
    return {
        "candidate_count": len(candidates),
        "v4j_total_trade_count": before_total,
        "v4k_total_trade_count": after_total,
        "total_trade_count_reduction": before_total - after_total,
        "total_trade_count_reduction_percent": (
            round(100.0 * (before_total - after_total) / before_total, 8)
            if before_total
            else None
        ),
        "v4j_mean_trade_count": round(sum(before) / len(before), 8),
        "v4k_mean_trade_count": round(sum(after) / len(after), 8),
        "v4j_mean_profit_factor": round(sum(before_pfs) / len(before_pfs), 8)
        if before_pfs
        else None,
        "v4k_mean_profit_factor": round(sum(after_pfs) / len(after_pfs), 8)
        if after_pfs
        else None,
        "profit_factor_improved_candidate_count": sum(
            item["comparison"]["profit_factor_improved"] for item in candidates
        ),
        "max_drawdown_reduced_candidate_count": sum(
            item["comparison"]["max_drawdown_reduced"] for item in candidates
        ),
        "trade_count_reduced_candidate_count": sum(
            item["comparison"]["trade_count_reduction"] > 0 for item in candidates
        ),
        "v4j_mean_max_drawdown_percent": round(sum(before_dd) / len(before_dd), 8),
        "v4k_mean_max_drawdown_percent": round(sum(after_dd) / len(after_dd), 8),
        "v4k_pf_above_1_5_candidate_count": sum(
            item["v4k_result"]["metrics"]["profit_factor_net_of_costs"] is not None
            and item["v4k_result"]["metrics"]["profit_factor_net_of_costs"] > 1.5
            for item in candidates
        ),
        "v4k_economic_gate_passed_candidate_count": sum(
            bool(item["v4k_result"]["metrics"]["economic_candidate_gate_passed"])
            for item in candidates
        ),
    }


def _build_report(
    commit_sha: str,
    source_digest: str,
    baseline_digest: str,
    csv_digest: str,
    candidates: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    bars_cache: dict[tuple[str, str], Any],
    provenance: list[dict[str, Any]],
    management: TradeManagementConfig,
) -> dict[str, Any]:
    """Build the immutable V4-J versus V4-K candidate comparison."""
    baseline_by_id = _candidate_map(baseline_report)
    execution = _execution_config()
    results = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in baseline_by_id:
            raise ValueError("v4j_baseline_candidate_missing")
        key = (_symbol_for_pair(str(candidate["pair"])), str(candidate["timeframe"]))
        results.append(
            _candidate_result(
                candidate,
                bars_cache[key],
                baseline_by_id[candidate_id],
                execution,
                management,
            )
        )
    aggregate = _aggregate(results)
    return {
        "schema_version": V4K_SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_report_sha256": source_digest,
        "source_v4j_baseline": {
            "schema_version": V4J_SCHEMA,
            "workflow_run_id": V4J_RUN_ID,
            "artifact_id": V4J_ARTIFACT_ID,
            "commit_sha": V4J_SHA,
            "report_sha256": baseline_digest,
            "comparison_csv_sha256": csv_digest,
        },
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_TRADE_MANAGEMENT",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "candidate_count": len(results),
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
        "trade_management_config": {
            "atr_period": management.atr_period,
            "take_profit_atr_multiplier": management.take_profit_atr_multiplier,
            "stop_loss_atr_multiplier": management.stop_loss_atr_multiplier,
            "horizon_candles": management.horizon_candles,
            "confidence_threshold": management.confidence_threshold,
            "atr_calculation": "SIMPLE_MEAN_TRUE_RANGE_FULL_WINDOW",
            "atr_reference": "SIGNAL_BAR_CLOSE_ONLY",
            "barrier_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "gap_exit_reference": "BAR_OPEN",
            "intrabar_exit_reference": "BARRIER_PRICE",
            "confidence_filter_applies_to": "BUY_ENTRIES_ONLY",
            "sell_exit_filtering": "DISABLED",
        },
        "aggregate": aggregate,
        "candidates": results,
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_count": len(results),
            "passed_candidate_count": aggregate[
                "v4k_economic_gate_passed_candidate_count"
            ],
            "status": "MET"
            if aggregate["v4k_economic_gate_passed_candidate_count"]
            else "NOT_MET",
        },
        "validation_bias_caveat": (
            "V4-K replays the same V4-H Forward-Validation probabilities and candidate "
            "selection interval as V4-I/V4-J. It is not an independent OOS promotion test."
        ),
        "model_artifact_limitation": (
            "V4-H contains persisted row-level probabilities and thresholds, not fitted "
            "estimators. V4-K performs no training or inference."
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


def render_v4k_markdown(report: dict[str, Any]) -> str:
    """Render the V4-J versus V4-K results in the Actions summary."""
    aggregate = report["aggregate"]
    lines = [
        "## AliX V4-K Trade Management and Exit Logic",
        "",
        f"V4-K commit SHA: {report['commit_sha']}",
        f"V4-J baseline SHA: {report['source_v4j_baseline']['commit_sha']}",
        "",
        "| Pair | TF | V4-J PF | V4-K PF | V4-J DD | V4-K DD | V4-J trades | V4-K trades | Δ DD (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["candidates"]:
        baseline, metrics = item["v4j_baseline"], item["v4k_result"]["metrics"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} "
            f"| {baseline['profit_factor'] if baseline['profit_factor'] is not None else 'N/A'} "
            f"| {metrics['profit_factor_net_of_costs'] if metrics['profit_factor_net_of_costs'] is not None else 'N/A'} "
            f"| {baseline['max_drawdown_percent']:.2f}% "
            f"| {metrics['max_drawdown_percent']:.2f}% "
            f"| {baseline['trade_count']} | {metrics['trade_count']} "
            f"| {item['comparison']['max_drawdown_delta_percentage_points']:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"- Trade count: {aggregate['v4j_total_trade_count']} → "
            f"{aggregate['v4k_total_trade_count']} "
            f"({aggregate['total_trade_count_reduction_percent'] or 0.0:.2f}% reduction).",
            f"- Mean PF: {aggregate['v4j_mean_profit_factor']} → "
            f"{aggregate['v4k_mean_profit_factor']}.",
            f"- Mean Max DD: {aggregate['v4j_mean_max_drawdown_percent']:.2f}% → "
            f"{aggregate['v4k_mean_max_drawdown_percent']:.2f}%.",
            f"- PF improved: {aggregate['profit_factor_improved_candidate_count']}/13; "
            f"DD reduced: {aggregate['max_drawdown_reduced_candidate_count']}/13; "
            f"trades reduced: {aggregate['trade_count_reduced_candidate_count']}/13.",
            f"- Economic gate passed: {aggregate['v4k_economic_gate_passed_candidate_count']}/13.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0.",
            "No model training or inference; independent OOS remains mandatory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one auditable V4-J to V4-K comparison row per candidate."""
    rows = []
    for item in report["candidates"]:
        baseline = item["v4j_baseline"]
        current = item["v4k_result"]["metrics"]
        exits = item["v4k_result"]["exit_reason_counts"]
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "pair": item["pair"],
                "timeframe": item["timeframe"],
                "v4j_profit_factor": baseline["profit_factor"],
                "v4k_profit_factor": current["profit_factor_net_of_costs"],
                "profit_factor_delta": item["comparison"]["profit_factor_delta"],
                "v4j_max_drawdown_percent": baseline["max_drawdown_percent"],
                "v4k_max_drawdown_percent": current["max_drawdown_percent"],
                "max_drawdown_delta_percentage_points": item["comparison"][
                    "max_drawdown_delta_percentage_points"
                ],
                "v4j_trade_count": baseline["trade_count"],
                "v4k_trade_count": current["trade_count"],
                "trade_count_reduction": item["comparison"]["trade_count_reduction"],
                "trade_count_reduction_percent": item["comparison"][
                    "trade_count_reduction_percent"
                ],
                "v4j_net_return_percent": baseline["net_return_percent"],
                "v4k_net_return_percent": current["net_return_percent"],
                "v4k_total_fees_usd": item["v4k_result"]["total_fees_usd"],
                "v4k_total_slippage_cost_usd": item["v4k_result"][
                    "total_slippage_cost_usd"
                ],
                "tp_exit_count": exits.get("TP_HIT", 0),
                "sl_exit_count": exits.get("SL_HIT", 0),
                "collision_exit_count": exits.get("COLLISION", 0),
                "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                "sell_exit_count": exits.get("SELL", 0),
                "confidence_rejected_buy_signals": item["confidence_filter"][
                    "rejected_buy_signal_count"
                ],
                "economic_candidate_gate_passed": current[
                    "economic_candidate_gate_passed"
                ],
            }
        )
    return rows


def write_comparison_csv(report: dict[str, Any], output_path: Path) -> None:
    """Write the candidate comparison CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(report))


def run_v4k_trade_management(
    source_report_path: Path,
    baseline_report_path: Path,
    baseline_csv_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    *,
    commit_sha: str,
    confidence_threshold: float = 0.55,
    atr_period: int = 14,
    take_profit_atr_multiplier: float = 2.0,
    stop_loss_atr_multiplier: float = 1.5,
    horizon_candles: int = 12,
) -> dict[str, Any]:
    """Replay persisted V4-H probabilities and compare against immutable V4-J."""
    source_report = _load_json(source_report_path)
    baseline_report = _load_json(baseline_report_path)
    source_digest, baseline_digest = (
        _digest(source_report_path),
        _digest(baseline_report_path),
    )
    csv_digest = _digest(baseline_csv_path)
    if source_digest != V4H_REPORT_SHA:
        raise ValueError("v4h_source_report_digest_mismatch")
    if baseline_digest != V4J_REPORT_SHA:
        raise ValueError("v4j_baseline_report_digest_mismatch")
    if csv_digest != V4J_CSV_SHA:
        raise ValueError("v4j_baseline_csv_digest_mismatch")
    validate_source_report(source_report)
    candidates = extract_qualifying_candidates(source_report)
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    validate_v4j_baseline(baseline_report, source_report, candidate_ids)
    management = trade_management_config(
        confidence_threshold=confidence_threshold,
        atr_period=atr_period,
        take_profit_atr_multiplier=take_profit_atr_multiplier,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        horizon_candles=horizon_candles,
    )
    bars_cache, provenance = _market_bars(
        candidates, source_report, input_dir, commit_sha
    )
    report = _build_report(
        commit_sha,
        source_digest,
        baseline_digest,
        csv_digest,
        candidates,
        baseline_report,
        bars_cache,
        provenance,
        management,
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
            summary.write(render_v4k_markdown(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": report["candidate_count"],
                "aggregate": report["aggregate"],
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
    """Parse the immutable-source V4-K GitHub Actions invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--take-profit-atr-multiplier", type=float, default=2.0)
    parser.add_argument("--stop-loss-atr-multiplier", type=float, default=1.5)
    parser.add_argument("--horizon-candles", type=int, default=12)
    args = parser.parse_args()
    run_v4k_trade_management(
        args.source_report,
        args.baseline_report,
        args.baseline_csv,
        args.input_dir,
        args.output,
        args.output_csv,
        commit_sha=args.commit_sha,
        confidence_threshold=args.confidence_threshold,
        atr_period=args.atr_period,
        take_profit_atr_multiplier=args.take_profit_atr_multiplier,
        stop_loss_atr_multiplier=args.stop_loss_atr_multiplier,
        horizon_candles=args.horizon_candles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
