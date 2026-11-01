"""Run V4-I economic replay from persisted V4-H predictions without training."""

from __future__ import annotations

import argparse
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
    evaluate_candidate,
    extract_qualifying_candidates,
    load_validated_minute_data,
    render_economic_markdown,
    resample_ohlcv,
    validate_execution_config,
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
    """Resolve the small approved V4-H symbol universe."""
    mapping = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    symbol = mapping.get(pair)
    if symbol is None:
        raise ValueError("economic_pair_out_of_policy")
    return symbol


def _source_metadata_by_symbol(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
        symbol = _symbol_for_pair(candidate["pair"])
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


def _build_report(
    *,
    commit_sha: str,
    source_report: dict[str, Any],
    source_report_digest: str,
    candidates: list[dict[str, Any]],
    bars_cache: dict[tuple[str, str], Any],
    provenance: list[dict[str, Any]],
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Replay every accepted V4-H candidate independently."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = _symbol_for_pair(candidate["pair"])
        bars = bars_cache[(symbol, str(candidate["timeframe"]))]
        results.append(evaluate_candidate(candidate, bars, config))
    passed = [
        item for item in results if item["metrics"]["economic_candidate_gate_passed"]
    ]
    return {
        "schema_version": "alix.ml-core.v4i-economic-evaluation.v1",
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_artifact_id": V4H_SOURCE_ARTIFACT_ID,
        "source_v4h_report_sha256": source_report_digest,
        "scope": "PERSISTED_FORWARD_VALIDATION_SIGNAL_REPLAY_ONLY",
        "economic_evaluation_performed": True,
        "economic_metrics_reported": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "source_v4h_candidate_count": len(candidates),
        "candidate_count": len(results),
        "input_datasets": provenance,
        "execution_assumptions": {
            "initial_equity_usd": config.initial_equity,
            "position_fraction_of_equity": config.position_fraction,
            "leverage": 1.0,
            "pyramiding": False,
            "concurrent_positions_per_candidate": 1,
            "entry_timing": "NEXT_BAR_OPEN_AFTER_SIGNAL_BAR_CLOSE",
            "exit_timing": "NEXT_BAR_OPEN_ON_HOLD_OR_DIRECTION_REVERSAL",
            "buy_semantics": "OPEN_LONG",
            "sell_semantics": "OPEN_SHORT",
            "hold_semantics": "CLOSE_POSITION_AND_REMAIN_FLAT",
            "fee_bps_per_side": config.fee_bps,
            "slippage_bps_per_side": config.slippage_bps,
            "round_trip_nominal_cost_bps": 2.0 * (config.fee_bps + config.slippage_bps),
            "short_borrow_or_funding_cost": "NOT_MODELED",
            "stop_loss": "NOT_MODELED",
            "take_profit": "NOT_MODELED",
            "trailing_stop": "NOT_MODELED",
            "terminal_position": "FORCE_CLOSED_AT_FINAL_FORWARD_BAR_CLOSE",
            "drawdown_measurement": "BAR_CLOSE_MARK_TO_MARKET_PLUS_ENTRY_AND_EXIT_COST_EVENTS",
            "profit_factor_basis": "SUM_POSITIVE_NET_TRADE_PNL_DIVIDED_BY_ABSOLUTE_SUM_NEGATIVE_NET_TRADE_PNL",
        },
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_fraction": config.maximum_drawdown,
            "candidate_count": len(results),
            "passed_candidate_count": len(passed),
            "status": "MET" if passed else "NOT_MET",
            "passing_candidate_ids": [item["candidate_id"] for item in passed],
        },
        "candidates": results,
        "validation_bias_caveat": (
            "The 13 candidates were selected using HOLD Precision/Recall on this same "
            "post-calibration Forward-Validation interval. Economic replay on that interval "
            "is selection-biased and is not an independent out-of-sample promotion test."
        ),
        "model_artifact_limitation": (
            "The V4-H artifact contains row-level probabilities and selected thresholds, "
            "not serialized fitted estimators. This run replays persisted predictions and "
            "does not train or infer any model."
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


def run_v4i_economic_replay(
    source_report_path: Path,
    input_dir: Path,
    output_path: Path,
    *,
    commit_sha: str,
    initial_equity: float = 10_000.0,
    fee_bps: float = 25.0,
    slippage_bps: float = 20.0,
    position_fraction: float = 1.0,
    maximum_drawdown: float = 0.15,
) -> dict[str, Any]:
    """Build an economic report solely from V4-H persisted predictions."""
    config = ExecutionConfig(
        initial_equity=initial_equity,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_fraction=position_fraction,
        maximum_drawdown=maximum_drawdown,
    )
    validate_execution_config(config)
    report_bytes = source_report_path.read_bytes()
    source_report = _load_json(source_report_path)
    candidates = extract_qualifying_candidates(source_report)
    digest = hashlib.sha256(report_bytes).hexdigest()
    bars_cache, provenance = _market_bars_for_candidates(
        candidates, source_report, input_dir, commit_sha
    )
    result = _build_report(
        commit_sha=commit_sha,
        source_report=source_report,
        source_report_digest=digest,
        candidates=candidates,
        bars_cache=bars_cache,
        provenance=provenance,
        config=config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(render_economic_markdown(result))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "source_v4h_commit_sha": V4H_SOURCE_SHA,
                "candidate_count": result["candidate_count"],
                "passed_candidate_count": result["economic_acceptance"][
                    "passed_candidate_count"
                ],
                "model_training_performed": False,
                "Champion": None,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=25.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--maximum-drawdown", type=float, default=0.15)
    args = parser.parse_args()
    run_v4i_economic_replay(
        args.source_report,
        args.input_dir,
        args.output,
        commit_sha=args.commit_sha,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        position_fraction=args.position_fraction,
        maximum_drawdown=args.maximum_drawdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
