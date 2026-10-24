"""Run the autonomous meta-strategy walk-forward without exchange access or order submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pandas import DataFrame

from scripts import run_stage_b_triple_barrier_orderbook as stage_b
from scripts.run_stage_b_walk_forward import A_HOLDOUT
from services.ai_core.autonomous_meta import (
    WalkForwardConfig,
    evaluate_historical_bars,
    load_strategy_registry,
)

REPORT_NAME = "autonomous-meta-strategy-report.json"
TRADES_NAME = "autonomous-meta-strategy-oos-trades.csv"
DECISIONS_NAME = "autonomous-meta-strategy-oos-decisions.csv"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = DataFrame(rows) if rows else DataFrame(columns=columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_verified_history(input_dir: Path) -> tuple[DataFrame, dict[str, Any]]:
    candles, source_report, segment_hash = stage_b._load_development(input_dir)
    positive_volume_rows = int((candles["volume"] > 0).sum())
    source = {
        "provider": "verified Stage A historical OHLCV artifact",
        "coverage_workflow_run_id": 40077609244,
        "coverage_csv_sha256": str(source_report["csv_sha256"]),
        "coverage_manifest_sha256": str(source_report["manifest_sha256"]),
        "segment_a_rows_sha256": segment_hash,
        "symbol": "BTCUSDT",
        "source_timeframe": "1h",
        "rows_used_for_research": len(candles),
        "source_volume_positive_rows": positive_volume_rows,
        "volume_observed": True,
        "reserved_final_holdout_rows": int(A_HOLDOUT),
        "final_holdout_used": False,
        "orderbook_or_spread_observed": False,
    }
    if source["final_holdout_used"] is not False:
        raise ValueError("autonomous_final_holdout_must_remain_reserved")
    return candles, source


def run_research(
    input_dir: Path,
    output_dir: Path,
    commit_sha: str,
    *,
    config: WalkForwardConfig | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    registry_path = (
        project_root / "configs" / "research" / "autonomous-meta-strategy-registry.json"
    )
    registry = load_strategy_registry(registry_path)
    candles, source = _load_verified_history(input_dir)
    report, trades, decisions = evaluate_historical_bars(
        candles,
        registry,
        commit_sha=commit_sha,
        source=source,
        config=config,
    )
    safety = report["safety_invariants"]
    if (
        safety["Champion"] is not None
        or safety["order_permission"] is not False
        or safety["trading_authorization"] != "NO-GO"
        or safety["real_order_attempts"] != 0
    ):
        raise ValueError("autonomous_meta_safety_contract_violated")
    report["artifact_paths"] = {
        "report": REPORT_NAME,
        "oos_trades": TRADES_NAME,
        "oos_decisions": DECISIONS_NAME,
    }
    _write_json(output_dir / REPORT_NAME, report)
    _write_rows(
        output_dir / TRADES_NAME,
        trades,
        [
            "policy",
            "fold",
            "strategy_id",
            "regime",
            "hmm_confidence",
            "entry_time",
            "exit_time",
            "net_return_bps",
            "net_pnl_quote",
            "order_permission",
            "real_order_attempted",
        ],
    )
    _write_rows(
        output_dir / DECISIONS_NAME,
        decisions,
        [
            "fold",
            "timestamp",
            "regime",
            "hmm_confidence",
            "strategy_id",
            "status",
            "reason",
            "entry_signal",
            "order_permission",
        ],
    )
    meta = report["economic_evaluation"]["meta_strategy_metrics"]
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "run_status": report["run_status"],
                "source_timeframe": report["source"]["source_timeframe"],
                "rows_used_for_research": report["source"]["rows_used_for_research"],
                "candidate_count": report["strategy_registry"]["candidate_count"],
                "scalping_status": report["strategy_registry"]["scalping_status"],
                "meta_trade_count": meta["trade_count"],
                "meta_win_rate_pct": meta["win_rate_pct"],
                "meta_expectancy_bps_per_trade": meta["expectancy_bps_per_trade"],
                "economic_acceptance": report["economic_evaluation"][
                    "economic_acceptance"
                ],
                "Champion": safety["Champion"],
                "order_permission": safety["order_permission"],
                "trading_authorization": safety["trading_authorization"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_research(args.input_dir, args.output_dir, args.commit_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
