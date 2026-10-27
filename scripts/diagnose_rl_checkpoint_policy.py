"""Read-only policy-distribution audit of Run #133 recurrent PPO checkpoints.

Only the source run's Validation split is replayed. This script does not train,
select, or modify any model and never scores the final Holdout.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.rl_auxiliary_ppo import build_auxiliary_recurrent_ppo
from scripts.run_rl_trading_agent import (
    _exclude_previous_holdout_window,
    _load_market_data,
    _sha256,
    _split_training_validation,
    _utc_from_ms,
)
from services.rl_research.rl_trading_environment import (
    ACTION_NAMES,
    BUY_ACTION_INDICES,
    DEFAULT_INITIAL_EQUITY,
    SELL_ACTION_INDICES,
    BTCUSDTTradingEnv,
    MarketObservationScaler,
)

SOURCE_RUN_ID = 40477004834
SOURCE_COMMIT_SHA = "10f782b71fbf374f3812b5219fa89cc9c49de2fa"
REPORT_SCHEMA = "alix.rl-trading-agent-report.v15"
CHECKPOINT_STEPS = tuple(range(100_000, 1_000_001, 100_000))
HOLD_SHARE_THRESHOLD = 0.99
SAFETY_INVARIANTS = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "private_api_calls": 0,
    "orders_or_execution_api_used": False,
    "real_order_attempts": 0,
    "production_promotion_allowed": False,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--diagnostic-commit-sha", required=True)
    return parser.parse_args(argv)


def _validate_source(
    args: argparse.Namespace,
    source_report: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if args.source_run_id != SOURCE_RUN_ID:
        raise ValueError("source_run_id_not_run_133")
    if args.source_commit_sha != SOURCE_COMMIT_SHA:
        raise ValueError("unexpected_run_133_commit_sha")
    if source_report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("source_report_schema_invalid")
    if source_report.get("commit_sha") != SOURCE_COMMIT_SHA:
        raise ValueError("source_report_commit_sha_mismatch")
    if source_report.get("run_mode") != "historical_simulation_only":
        raise ValueError("source_run_mode_invalid")
    if source_report.get("dataset", {}).get("csv_sha256") != manifest.get("csv_sha256"):
        raise ValueError("source_dataset_sha256_mismatch")
    if source_report.get("dataset", {}).get("row_count") != manifest.get("row_count"):
        raise ValueError("source_dataset_row_count_mismatch")
    safety = source_report.get("safety_invariants", {})
    if (
        safety.get("Champion", "invalid") is not None
        or safety.get("order_permission") is not False
        or safety.get("trading_authorization") != "NO-GO"
        or safety.get("private_api_calls") != 0
        or safety.get("orders_or_execution_api_used") is not False
        or safety.get("real_order_attempts") != 0
        or source_report.get("production_promotion_allowed") is not False
    ):
        raise ValueError("source_safety_invariants_invalid")


def _checkpoint_inventory(
    checkpoint_dir: Path, source_report: dict[str, Any]
) -> list[tuple[int, Path]]:
    paths = sorted(
        checkpoint_dir.glob("rl_trading_agent_checkpoint_*_steps.zip"),
        key=lambda path: int(
            path.name.removeprefix("rl_trading_agent_checkpoint_").removesuffix(
                "_steps.zip"
            )
        ),
    )
    by_step: dict[int, dict[str, Any]] = {
        int(item["timesteps"]): item
        for item in source_report.get("checkpointing", {}).get("checkpoints", [])
    }
    observed_steps = [
        int(
            path.name.removeprefix("rl_trading_agent_checkpoint_").removesuffix(
                "_steps.zip"
            )
        )
        for path in paths
    ]
    if tuple(observed_steps) != CHECKPOINT_STEPS:
        raise ValueError("run_133_checkpoint_inventory_mismatch")
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise ValueError("source_report_checkpoint_inventory_mismatch")
    result: list[tuple[int, Path]] = []
    for step, path in zip(observed_steps, paths, strict=True):
        metadata = by_step[step]
        if Path(str(metadata.get("file", ""))).name != path.name:
            raise ValueError("source_checkpoint_filename_mismatch")
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"source_checkpoint_sha256_mismatch_at_{step}")
        if int(metadata.get("size_bytes", 0)) != path.stat().st_size:
            raise ValueError(f"source_checkpoint_size_mismatch_at_{step}")
        result.append((step, path))
    return result


def _initial_actor_state(model: Any, torch: Any) -> tuple[Any, Any]:
    shape = model.policy.lstm_hidden_state_shape
    hidden = torch.zeros(shape, dtype=torch.float32, device=model.policy.device)
    return hidden, hidden.clone()


def _probability_entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _evaluate_checkpoint_policy(
    checkpoint_step: int,
    checkpoint_path: Path,
    validation_data: np.ndarray,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
    seed: int,
    source_report: dict[str, Any],
    algorithm_class: Any,
    torch: Any,
) -> dict[str, Any]:
    model = algorithm_class.load(str(checkpoint_path), env=None, device="cpu")
    model.policy.set_training_mode(False)
    env = BTCUSDTTradingEnv(
        validation_data,
        scaler=scaler,
        initial_equity=DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
    )
    counts: Counter[str] = Counter({name: 0 for name in ACTION_NAMES})
    probability_sums = [0.0] * len(ACTION_NAMES)
    entropy_sum = 0.0
    hold_gt_buy_count = 0
    hold_99_count = 0
    first_hold_gt_buy_step: int | None = None
    first_hold_99_step: int | None = None
    step_count = 0
    recurrent_state = _initial_actor_state(model, torch)
    episode_start = True
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            observation_tensor, _ = model.policy.obs_to_tensor(observation)
            episode_starts = torch.as_tensor(
                [episode_start], dtype=torch.float32, device=model.policy.device
            )
            with torch.no_grad():
                distribution, next_state = model.policy.get_distribution(
                    observation_tensor, recurrent_state, episode_starts
                )
                probabilities = [
                    float(value)
                    for value in distribution.distribution.probs.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                ]
                action = int(
                    distribution.get_actions(deterministic=True)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)[0]
                )
            if len(probabilities) != len(ACTION_NAMES):
                raise ValueError("checkpoint_action_probability_dimension_invalid")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
                raise ValueError("checkpoint_action_probabilities_not_normalized")
            if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
                raise ValueError("checkpoint_action_probability_invalid")
            step_count += 1
            for index, probability in enumerate(probabilities):
                probability_sums[index] += probability
            entropy_sum += _probability_entropy(probabilities)
            buy_probability = sum(probabilities[index] for index in BUY_ACTION_INDICES)
            hold_probability = probabilities[0]
            if hold_probability > buy_probability:
                hold_gt_buy_count += 1
                if first_hold_gt_buy_step is None:
                    first_hold_gt_buy_step = step_count
            if hold_probability >= HOLD_SHARE_THRESHOLD:
                hold_99_count += 1
                if first_hold_99_step is None:
                    first_hold_99_step = step_count
            counts[ACTION_NAMES[action]] += 1
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.int64)
            )
            recurrent_state = next_state
            episode_start = bool(terminated or truncated)
            if episode_start:
                break
    finally:
        env.close()

    expected_steps = len(validation_data) - 1
    if step_count != expected_steps:
        raise ValueError("checkpoint_validation_pass_length_mismatch")
    mean_probabilities = {
        name: probability_sums[index] / step_count
        for index, name in enumerate(ACTION_NAMES)
    }
    side_probabilities = {
        "HOLD": mean_probabilities["HOLD"],
        "BUY": sum(
            mean_probabilities[ACTION_NAMES[index]]
            for index in BUY_ACTION_INDICES
        ),
        "SELL": sum(
            mean_probabilities[ACTION_NAMES[index]]
            for index in SELL_ACTION_INDICES
        ),
    }
    action_distribution = {
        name: {"count": counts[name], "percent": 100.0 * counts[name] / step_count}
        for name in ACTION_NAMES
    }
    report = {
        "checkpoint_timesteps": checkpoint_step,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_steps": step_count,
        "deterministic_action_counts": dict(counts),
        "deterministic_action_distribution": action_distribution,
        "mean_probability_by_action": mean_probabilities,
        "mean_probability_by_side": side_probabilities,
        "mean_policy_entropy_nats": entropy_sum / step_count,
        "normalized_mean_policy_entropy": (
            entropy_sum / step_count / math.log(len(ACTION_NAMES))
        ),
        "hold_deterministic_action_fraction": counts["HOLD"] / step_count,
        "hold_probability_ge_99pct_fraction": hold_99_count / step_count,
        "hold_probability_exceeds_aggregate_buy_probability_fraction": (
            hold_gt_buy_count / step_count
        ),
        "first_validation_step_where_hold_probability_exceeded_buy_probability": (
            first_hold_gt_buy_step
        ),
        "first_validation_step_where_hold_probability_reached_99pct": first_hold_99_step,
        "training_performed": False,
        "final_holdout_scored_or_recomputed": False,
        "used_for_checkpoint_selection": False,
    }
    if source_report.get("safety_invariants", {}).get("Champion") is not None:
        raise ValueError("source_champion_must_remain_null")
    del model
    return report


def _checkpoint_crossover_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("checkpoint_crossover_requires_rows")
    hold_over_buy = next(
        (
            row for row in rows
            if float(row["mean_probability_by_side"]["HOLD"])
            > float(row["mean_probability_by_side"]["BUY"])
        ),
        None,
    )
    first_99 = next(
        (
            row for row in rows
            if float(row["hold_deterministic_action_fraction"]) >= HOLD_SHARE_THRESHOLD
        ),
        None,
    )
    persistent_index = next(
        (
            index for index, row in enumerate(rows)
            if all(
                float(later["hold_deterministic_action_fraction"]) >= HOLD_SHARE_THRESHOLD
                for later in rows[index:]
            )
        ),
        None,
    )
    hold_over_buy_step = (
        int(hold_over_buy["checkpoint_timesteps"]) if hold_over_buy is not None else None
    )
    persistent_step = (
        int(rows[persistent_index]["checkpoint_timesteps"])
        if persistent_index is not None
        else None
    )
    previous_step = (
        int(rows[persistent_index - 1]["checkpoint_timesteps"])
        if persistent_index is not None and persistent_index > 0
        else None
    )
    return {
        "first_checkpoint_mean_hold_probability_exceeds_mean_buy_probability": (
            hold_over_buy_step
        ),
        "first_checkpoint_deterministic_hold_share_at_least_99pct": (
            int(first_99["checkpoint_timesteps"]) if first_99 is not None else None
        ),
        "first_persistent_checkpoint_with_deterministic_hold_share_at_least_99pct": (
            persistent_step
        ),
        "persistent_hold_share_threshold": HOLD_SHARE_THRESHOLD,
        "persistent_hold_bracket_previous_checkpoint_timestep": previous_step,
        "exact_training_timestep_identifiable": False,
        "resolution_steps": 100_000,
        "explanation": (
            "Run #133 checkpoints are saved every 100,000 training steps. This audit "
            "identifies the first checkpoint with the measured behavior and the preceding "
            "snapshot, but cannot reconstruct unsaved intermediate weights."
        ),
    }


def _write_checkpoint_csv(rows: list[dict[str, Any]], target: Path) -> None:
    fieldnames = [
        "checkpoint_timesteps",
        "checkpoint_sha256",
        "validation_steps",
        "hold_deterministic_action_fraction",
        "hold_probability_ge_99pct_fraction",
        "hold_probability_exceeds_aggregate_buy_probability_fraction",
        "mean_hold_probability",
        "mean_buy_probability",
        "mean_sell_probability",
        "mean_policy_entropy_nats",
        "normalized_mean_policy_entropy",
        *[f"count_{name}" for name in ACTION_NAMES],
    ]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            side = row["mean_probability_by_side"]
            record = {
                "checkpoint_timesteps": row["checkpoint_timesteps"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "validation_steps": row["validation_steps"],
                "hold_deterministic_action_fraction": row[
                    "hold_deterministic_action_fraction"
                ],
                "hold_probability_ge_99pct_fraction": row[
                    "hold_probability_ge_99pct_fraction"
                ],
                "hold_probability_exceeds_aggregate_buy_probability_fraction": row[
                    "hold_probability_exceeds_aggregate_buy_probability_fraction"
                ],
                "mean_hold_probability": side["HOLD"],
                "mean_buy_probability": side["BUY"],
                "mean_sell_probability": side["SELL"],
                "mean_policy_entropy_nats": row["mean_policy_entropy_nats"],
                "normalized_mean_policy_entropy": row[
                    "normalized_mean_policy_entropy"
                ],
            }
            record.update(
                {
                    f"count_{name}": row["deterministic_action_counts"][name]
                    for name in ACTION_NAMES
                }
            )
            writer.writerow(record)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    data, manifest = _load_market_data(args.data, args.manifest)
    _validate_source(args, source_report, manifest)

    analysis_data, _ = _exclude_previous_holdout_window(data)
    train_data, validation_data, _ = _split_training_validation(analysis_data)
    split = source_report.get("split", {})
    if len(train_data) != int(source_report["model"]["training_rows"]):
        raise ValueError("source_training_rows_mismatch")
    if len(validation_data) != int(split.get("validation_rows", -1)):
        raise ValueError("source_validation_rows_mismatch")
    if _utc_from_ms(float(validation_data[0, 0])) != split.get("validation_start_utc"):
        raise ValueError("source_validation_start_mismatch")
    if _utc_from_ms(float(validation_data[-1, 0])) != split.get("validation_end_utc"):
        raise ValueError("source_validation_end_mismatch")

    scaler = MarketObservationScaler.fit(train_data)
    fee_bps = float(source_report["environment"]["execution"]["fee_bps_per_side"])
    slippage_bps = float(source_report["environment"]["execution"]["slippage_bps_per_side"])
    seed = int(source_report["model"]["seed"])
    checkpoints = _checkpoint_inventory(args.checkpoint_dir, source_report)
    torch = importlib.import_module("torch")
    algorithm_class = build_auxiliary_recurrent_ppo()
    rows = [
        _evaluate_checkpoint_policy(
            step, path, validation_data, scaler, fee_bps, slippage_bps,
            seed, source_report, algorithm_class, torch
        )
        for step, path in checkpoints
    ]
    crossover = _checkpoint_crossover_summary(rows)
    report = {
        "schema_version": "alix.rl-trading-agent-checkpoint-policy-audit.v1",
        "audit_mode": "run_133_saved_checkpoints_validation_policy_distribution_only",
        "source": {
            "workflow_run_id": args.source_run_id,
            "training_commit_sha": args.source_commit_sha,
            "source_report_sha256": _sha256(args.source_report),
            "dataset_sha256": manifest["csv_sha256"],
            "validation_start_utc": split["validation_start_utc"],
            "validation_end_utc": split["validation_end_utc"],
            "validation_steps_per_checkpoint": len(validation_data) - 1,
        },
        "execution_scope": {
            "training_performed": False,
            "source_model_weights_modified": False,
            "new_checkpoints_created": False,
            "final_holdout_scored_or_recomputed": False,
            "checkpoint_used_for_selection": False,
            "production_promotion_allowed": False,
        },
        "checkpoint_count": len(rows),
        "checkpoints": rows,
        "behavior_crossover": crossover,
        "source_final_model_reference": {
            "hold_collapse_flag": source_report["evaluation"]["validation"]["flags"][
                "HOLD_COLLAPSE"
            ],
            "interpretation": (
                "This audit compares checkpoint policy distributions and deterministic "
                "actions; it does not attribute an economic loss to an exact training step."
            ),
        },
        "safety_invariants": dict(SAFETY_INVARIANTS),
        "diagnostic_commit_sha": args.diagnostic_commit_sha,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "rl-run133-checkpoint-policy-audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_checkpoint_csv(rows, args.out_dir / "rl-run133-checkpoint-policy-audit.csv")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "checkpoint_count": len(rows),
                "behavior_crossover": crossover,
                "training_performed": False,
                "final_holdout_scored_or_recomputed": False,
                "safety_invariants": report["safety_invariants"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
