"""Deterministically evaluate the ten saved recurrent-PPO checkpoints on Run #105 validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.rl_auxiliary_ppo import build_auxiliary_recurrent_ppo
from scripts.run_rl_trading_agent import (
    _evaluate_policy,
    _exclude_previous_holdout_window,
    _load_market_data,
    _sha256,
    _split_training_validation,
    _utc_from_ms,
)
from services.rl_research.rl_trading_environment import MarketObservationScaler

EXPECTED_STEPS = tuple(range(100_000, 1_000_001, 100_000))
CHECKPOINT_PREFIX = "rl_trading_agent_checkpoint_"
CHECKPOINT_SUFFIX = "_steps.zip"
EXPECTED_SOURCE_SHA = "62fc99cf92fd92cc92664cffff611024ff8fed4c"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    return parser.parse_args(argv)


def _checkpoint_inventory(
    report: dict[str, Any], checkpoint_dir: Path
) -> list[tuple[int, Path, str]]:
    checkpointing = report.get("checkpointing", {})
    entries = checkpointing.get("checkpoints", [])
    by_step: dict[int, dict[str, Any]] = {}
    for item in entries:
        step = int(item["timesteps"])
        if step in by_step:
            raise ValueError("duplicate_checkpoint_timestep_in_source_report")
        by_step[step] = item
    if set(by_step) != set(EXPECTED_STEPS):
        raise ValueError("source_report_does_not_list_exactly_ten_expected_checkpoints")

    inventory = []
    for step in EXPECTED_STEPS:
        item = by_step[step]
        path = checkpoint_dir / f"{CHECKPOINT_PREFIX}{step}{CHECKPOINT_SUFFIX}"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_missing:{path.name}")
        digest = _sha256(path)
        if digest != item.get("sha256"):
            raise ValueError(f"checkpoint_sha256_mismatch:{step}")
        inventory.append((step, path, digest))

    expected_names = {item[1].name for item in inventory}
    unexpected = [
        path.name
        for path in checkpoint_dir.glob("*.zip")
        if path.name.startswith(CHECKPOINT_PREFIX) and path.name not in expected_names
    ]
    if unexpected:
        raise ValueError("unexpected_checkpoint_files:" + ",".join(sorted(unexpected)))
    return inventory


def _action_percentages(metrics: dict[str, Any]) -> dict[str, Any]:
    counts = metrics.get("discrete_action_counts")
    if not isinstance(counts, dict):
        raise ValueError("evaluation_missing_discrete_action_counts")

    normalized = {
        "HOLD": int(counts.get("HOLD", 0)),
        "BUY": sum(
            int(value) for key, value in counts.items() if str(key).startswith("BUY")
        ),
        "SELL": sum(
            int(value) for key, value in counts.items() if str(key).startswith("SELL")
        ),
    }
    total = sum(normalized.values())
    if total != int(metrics["steps"]):
        raise ValueError("action_count_total_does_not_match_validation_steps")

    return {
        name: {
            "count": count,
            "percent": 100.0 * count / total if total else None,
        }
        for name, count in normalized.items()
    }


def _validate_source_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != "alix.rl-trading-agent-report.v11":
        raise ValueError("source_report_schema_invalid")
    if report.get("commit_sha") != EXPECTED_SOURCE_SHA:
        raise ValueError("source_report_commit_sha_mismatch")

    safety = report.get("safety_invariants", {})
    if (
        safety.get("Champion", "invalid") is not None
        or safety.get("order_permission") is not False
        or safety.get("trading_authorization") != "NO-GO"
        or safety.get("private_api_calls") != 0
        or safety.get("real_order_attempts") != 0
        or safety.get("orders_or_execution_api_used") is not False
    ):
        raise ValueError("source_report_safety_invariant_invalid")


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    json_path = out_dir / "rl-checkpoint-sweep.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = out_dir / "rl-checkpoint-sweep.csv"
    fields = [
        "checkpoint_timesteps",
        "hold_percent",
        "buy_percent",
        "sell_percent",
        "net_profit",
        "net_return",
        "max_drawdown",
        "closed_trades",
        "checkpoint_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["checkpoints"]:
            writer.writerow(
                {
                    "checkpoint_timesteps": row["checkpoint_timesteps"],
                    "hold_percent": row["action_distribution"]["HOLD"]["percent"],
                    "buy_percent": row["action_distribution"]["BUY"]["percent"],
                    "sell_percent": row["action_distribution"]["SELL"]["percent"],
                    "net_profit": row["net_profit"],
                    "net_return": row["net_return"],
                    "max_drawdown": row["max_drawdown"],
                    "closed_trades": row["closed_trades"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    _validate_source_report(source_report)

    data, manifest = _load_market_data(args.data, args.manifest)
    if manifest.get("csv_sha256") != source_report.get("dataset", {}).get("csv_sha256"):
        raise ValueError("source_dataset_hash_mismatch")

    analysis_data, excluded_rows = _exclude_previous_holdout_window(data)
    train_data, validation_data, _ = _split_training_validation(analysis_data)
    scaler = MarketObservationScaler.fit(train_data)

    source_split = source_report["split"]
    if len(validation_data) != int(source_split["validation_rows"]):
        raise ValueError("validation_rows_do_not_match_source_run")
    if (
        _utc_from_ms(float(validation_data[0, 0]))
        != source_split["validation_start_utc"]
    ):
        raise ValueError("validation_start_does_not_match_source_run")
    if (
        _utc_from_ms(float(validation_data[-1, 0]))
        != source_split["validation_end_utc"]
    ):
        raise ValueError("validation_end_does_not_match_source_run")
    if source_split.get("scaler_fit_on_training_only") is not True:
        raise ValueError("source_scaler_contract_invalid")

    inventory = _checkpoint_inventory(source_report, args.checkpoint_dir)
    execution = source_report["environment"]["execution"]
    fee_bps = float(execution["fee_bps_per_side"])
    slippage_bps = float(execution["slippage_bps_per_side"])
    training_seed = int(source_report["model"]["seed"])

    # Register the same dynamic custom policy class used during training before
    # Stable-Baselines deserializes its serialized policy_class reference.
    algorithm_class = build_auxiliary_recurrent_ppo()
    checkpoints = []
    for step, path, digest in inventory:
        model = algorithm_class.load(str(path.with_suffix("")), env=None, device="cpu")
        metrics = _evaluate_policy(
            model,
            validation_data,
            scaler,
            fee_bps,
            slippage_bps,
            seed=training_seed,
        )
        if metrics["steps"] != len(validation_data) - 1:
            raise RuntimeError(f"validation_pass_contract_failed:{step}")
        if metrics.get("evaluation_passes") != 1:
            raise RuntimeError(f"validation_pass_count_invalid:{step}")

        checkpoints.append(
            {
                "checkpoint_timesteps": step,
                "checkpoint_file": path.name,
                "checkpoint_sha256": digest,
                "loaded": True,
                "validation_steps": metrics["steps"],
                "action_distribution": _action_percentages(metrics),
                "net_profit": float(metrics["net_profit"]),
                "net_return": float(metrics["net_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "closed_trades": int(metrics["closed_trade_count"]),
                "total_fees": float(metrics["total_fees"]),
                "total_slippage": float(metrics["total_slippage"]),
                "evaluation_passes": 1,
            }
        )
        del model

    first_over_90 = next(
        (
            row["checkpoint_timesteps"]
            for row in checkpoints
            if row["action_distribution"]["HOLD"]["percent"] > 90.0
        ),
        None,
    )
    report = {
        "schema_version": "alix.rl-checkpoint-sweep.v1",
        "source_run_id": args.source_run_id,
        "source_training_commit_sha": source_report["commit_sha"],
        "diagnostic_commit_sha": args.commit_sha,
        "training_performed": False,
        "model_weights_modified": False,
        "new_checkpoints_created": False,
        "deterministic_inference": True,
        "validation_only": True,
        "final_holdout_scored_or_recomputed": False,
        "final_holdout_used_for_selection": False,
        "validation": {
            "dataset_sha256": manifest["csv_sha256"],
            "rows": len(validation_data),
            "start_utc": source_split["validation_start_utc"],
            "end_utc": source_split["validation_end_utc"],
            "scaler_fit_on_training_only": True,
            "source_validation_split_matched": True,
            "fee_bps_per_side": fee_bps,
            "slippage_bps_per_side": slippage_bps,
            "excluded_rows_after_analysis_window": excluded_rows,
        },
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "first_checkpoint_hold_over_90_percent": first_over_90,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "orders_or_execution_api_used": False,
            "real_order_attempts": 0,
            "production_promotion_allowed": False,
        },
    }
    _write_report(args.out_dir, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
