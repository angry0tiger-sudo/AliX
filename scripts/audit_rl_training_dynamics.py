"""Audit archived PPO training dynamics without running or modifying training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

EXPECTED_SOURCE_RUN_ID = 40348192766
EXPECTED_SOURCE_COMMIT = "4571691ec6b11146ea6f3a93e0cc7c9e742ec578"
EXPECTED_SOURCE_SCHEMA = "alix.rl-trading-agent-report.v10"
REPORT_SCHEMA = "alix.rl-trading-agent-training-dynamics-audit.v1"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
TABLE_ROW = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|")
CHECKPOINT_SUFFIXES = {".zip", ".pt", ".pth", ".ckpt"}
TRACKED_METRICS = (
    "approx_kl",
    "auxiliary_loss",
    "auxiliary_loss_weight",
    "clip_fraction",
    "entropy_loss",
    "explained_variance",
    "fps",
    "learning_rate",
    "loss",
    "n_updates",
    "policy_gradient_loss",
    "time_elapsed",
    "value_loss",
)
SAFETY = {
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
    "private_api_calls": 0,
    "orders_or_execution_api_used": False,
    "real_order_attempts": 0,
    "production_promotion_allowed": False,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-run-metadata", type=Path, required=True)
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-commit-sha", required=True)
    return parser.parse_args(argv)


def _number(raw: str) -> float | None:
    try:
        value = float(raw.strip().replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_training_log_line(
    original_line: str,
    current_step: int | None,
    snapshots: dict[int, dict[str, float | int]],
) -> int | None:
    line = ANSI_ESCAPE.sub("", original_line)
    match = TABLE_ROW.search(line)
    if match is None:
        return current_step
    key = " ".join(match.group(1).strip().split())
    raw_value = match.group(2).strip()
    if key == "total_timesteps":
        parsed_step = _number(raw_value)
        if parsed_step is None or not parsed_step.is_integer():
            return None
        next_step = int(parsed_step)
        snapshots.setdefault(next_step, {"timesteps": next_step})
        return next_step
    if current_step is None or key not in TRACKED_METRICS or not raw_value:
        return current_step
    value = _number(raw_value)
    if value is not None:
        snapshots[current_step][key] = value
    return current_step


def _validated_training_snapshots(
    snapshots: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int]]:
    ordered = [snapshots[step] for step in sorted(snapshots)]
    result = [
        snapshot
        for snapshot in ordered
        if "entropy_loss" in snapshot and "auxiliary_loss" in snapshot
    ]
    if len(result) < 20:
        raise ValueError("insufficient_historical_training_metric_snapshots")
    if result[-1]["timesteps"] < 900_000:
        raise ValueError("historical_training_log_does_not_cover_late_training")
    return result


def parse_training_logs(logs_dir: Path) -> list[dict[str, float | int]]:
    """Parse the historical Stable-Baselines3 progress tables."""
    paths = sorted(
        path
        for path in logs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".log", ".out"}
    )
    if not paths:
        raise ValueError("source_training_job_logs_not_found")
    snapshots: dict[int, dict[str, float | int]] = {}
    for path in paths:
        current_step: int | None = None
        for original_line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            current_step = _parse_training_log_line(
                original_line, current_step, snapshots
            )
    return _validated_training_snapshots(snapshots)


def _validate_source_run_identity(source_run_id: int, source_commit_sha: str) -> None:
    if source_run_id != EXPECTED_SOURCE_RUN_ID:
        raise ValueError("unexpected_source_run_id")
    if source_commit_sha != EXPECTED_SOURCE_COMMIT:
        raise ValueError("unexpected_source_commit")


def _validate_run_metadata(source_run: dict[str, Any]) -> None:
    if source_run.get("id") != EXPECTED_SOURCE_RUN_ID:
        raise ValueError("source_run_metadata_id_mismatch")
    if source_run.get("head_sha") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("source_run_metadata_commit_mismatch")
    if (
        source_run.get("status") != "completed"
        or source_run.get("conclusion") != "success"
    ):
        raise ValueError("source_training_run_not_successful")


def _validate_source_report_identity(source_report: dict[str, Any]) -> None:
    if source_report.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError("source_report_schema_mismatch")
    if source_report.get("commit_sha") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("source_report_commit_mismatch")
    if source_report.get("run_mode") != "historical_simulation_only":
        raise ValueError("source_report_run_mode_mismatch")


def _validate_source_report_safety(source_report: dict[str, Any]) -> None:
    safety = source_report.get("safety_invariants", {})
    safety_is_invalid = (
        safety.get("Champion", "invalid") is not None
        or safety.get("order_permission") is not False
        or safety.get("trading_authorization") != "NO-GO"
        or safety.get("private_api_calls") != 0
        or safety.get("orders_or_execution_api_used") is not False
        or safety.get("real_order_attempts") != 0
        or source_report.get("production_promotion_allowed") is not False
    )
    if safety_is_invalid:
        raise ValueError("source_safety_invariants_invalid")


def _validate_source_model(source_report: dict[str, Any]) -> None:
    model = source_report.get("model", {})
    if (
        model.get("algorithm") != "RecurrentPPO"
        or model.get("total_timesteps_trained", 0) < 1_000_000
    ):
        raise ValueError("source_training_model_identity_invalid")
    digest = model.get("model_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source_model_digest_invalid")


def _validate_source(
    source_report: dict[str, Any],
    source_run: dict[str, Any],
    source_run_id: int,
    source_commit_sha: str,
) -> None:
    _validate_source_run_identity(source_run_id, source_commit_sha)
    _validate_run_metadata(source_run)
    _validate_source_report_identity(source_report)
    _validate_source_report_safety(source_report)
    _validate_source_model(source_report)


def _entropy_summary(
    snapshots: list[dict[str, float | int]],
) -> dict[str, Any]:
    points = [
        {
            "timesteps": int(item["timesteps"]),
            "entropy_nats": max(0.0, -float(item["entropy_loss"])),
            "entropy_loss": float(item["entropy_loss"]),
        }
        for item in snapshots
    ]
    minimum = min(points, key=lambda point: point["entropy_nats"])
    maximum = max(points, key=lambda point: point["entropy_nats"])
    threshold_steps: dict[str, int | None] = {}
    for threshold in (0.05, 0.01, 0.001):
        match = next(
            (point for point in points if point["entropy_nats"] <= threshold),
            None,
        )
        threshold_steps[f"entropy_at_or_below_{threshold:g}_nats"] = (
            int(match["timesteps"]) if match is not None else None
        )
    return {
        "metric_definition": "negative Stable-Baselines3 train/entropy_loss, in nats",
        "snapshot_count": len(points),
        "first_logged_timestep": points[0]["timesteps"],
        "last_logged_timestep": points[-1]["timesteps"],
        "maximum_observed_entropy_nats": maximum["entropy_nats"],
        "maximum_observed_entropy_timestep": maximum["timesteps"],
        "minimum_observed_entropy_nats": minimum["entropy_nats"],
        "minimum_observed_entropy_timestep": minimum["timesteps"],
        "first_snapshot_at_or_below_0_05_nats": threshold_steps[
            "entropy_at_or_below_0.05_nats"
        ],
        "threshold_crossings": threshold_steps,
        "exact_zero_observed": any(point["entropy_nats"] <= 1e-12 for point in points),
        "points": points,
        "interpretation": (
            "This is the PPO training logger's minibatch entropy at periodic updates, "
            "not a deterministic action-probability trace on a fixed Validation sample."
        ),
    }


def _nearest_snapshot(
    snapshots: list[dict[str, float | int]], target: int
) -> dict[str, Any]:
    nearest = min(snapshots, key=lambda item: abs(int(item["timesteps"]) - target))
    observed_step = int(nearest["timesteps"])
    entropy_loss = float(nearest["entropy_loss"])
    return {
        "requested_milestone_timestep": target,
        "nearest_logged_timestep": observed_step,
        "absolute_timestep_gap": abs(observed_step - target),
        "entropy_nats": max(0.0, -entropy_loss),
        "auxiliary_loss": float(nearest["auxiliary_loss"]),
        "auxiliary_loss_weight": float(nearest.get("auxiliary_loss_weight", 0.0)),
        "approx_kl": nearest.get("approx_kl"),
        "policy_gradient_loss": nearest.get("policy_gradient_loss"),
        "value_loss": nearest.get("value_loss"),
        "loss": nearest.get("loss"),
    }


def _checkpoint_inventory(artifact_dir: Path) -> dict[str, Any]:
    model_files = sorted(
        str(path.relative_to(artifact_dir))
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CHECKPOINT_SUFFIXES
        and re.search(r"checkpoint|step[_-]?\d+", path.name, re.IGNORECASE)
    )
    final_model = artifact_dir / "rl-agent" / "rl_trading_agent_recurrent_ppo.zip"
    return {
        "historical_checkpoint_files_found": model_files,
        "checkpoint_count": len(model_files),
        "final_model_exists": final_model.is_file(),
        "final_model_is_not_a_historical_checkpoint": True,
        "exact_hold_buy_crossover_recoverable": False,
        "reason_if_unavailable": (
            "This audit reads archived scalar logs and does not replay checkpoint models. "
            "The source Run #98 artifact contains no named historical step checkpoints."
        ),
    }


def _reward_evidence(source_report: dict[str, Any]) -> dict[str, Any]:
    cost = source_report.get("cost_curriculum", {})
    guidance = source_report.get("signal_guidance_curriculum", {})
    auxiliary = source_report.get("auxiliary_loss_curriculum", {})
    validation = source_report.get("evaluation", {}).get("validation", {})
    final_validation_keys = (
        "steps",
        "reward_sum",
        "discrete_action_counts",
        "low_volatility_hold_reward_count",
        "opportunity_cost_penalty_count",
        "transaction_cost_reward_penalty_total",
        "signal_guidance_reward_count",
        "holding_profit_reward_count",
        "holding_loss_penalty_count",
        "realized_profit_reward_count",
        "total_fees",
        "total_slippage",
    )
    return {
        "per_step_training_reward_component_timeseries_available": False,
        "reason": (
            "The original run did not persist per-step training env.info reward components. "
            "The environment emitted those values transiently, but the training callback "
            "only retained cost-curriculum and signal-guidance aggregates."
        ),
        "historical_training_aggregate_evidence": {
            "cost_curriculum_scheme": cost.get("scheme"),
            "ramp_start_timestep": cost.get("ramp_start_timestep"),
            "full_cost_start_timestep": cost.get("full_cost_start_timestep"),
            "steps_by_cost_stage": cost.get("steps_by_stage"),
            "target_fee_bps_per_side": cost.get("target_fee_bps_per_side"),
            "target_slippage_bps_per_side": cost.get("target_slippage_bps_per_side"),
            "signal_guidance_initial_scale": guidance.get("initial_reward_scale"),
            "signal_guidance_anneal_steps": guidance.get("anneal_steps"),
            "signal_guidance_final_scale": guidance.get("final_reward_scale"),
            "signal_guidance_reward_count": guidance.get("rewarded_action_count"),
            "signal_guidance_reward_total": guidance.get("reward_total"),
            "auxiliary_loss_initial_weight": auxiliary.get("initial_weight"),
            "auxiliary_loss_anneal_steps": auxiliary.get("anneal_steps"),
            "auxiliary_loss_final_weight": auxiliary.get(
                "effective_weight_at_training_end"
            ),
        },
        "final_validation_components_only_not_training": {
            key: validation[key] for key in final_validation_keys if key in validation
        },
    }


def _render_entropy_svg(entropy: dict[str, Any], target: Path) -> None:
    points = entropy["points"]
    width, height = 1000, 520
    left, right, top, bottom = 90, 960, 70, 430
    maximum = max(float(point["entropy_nats"]) for point in points)
    maximum = max(maximum, 0.05)
    first_step = int(points[0]["timesteps"])
    last_step = int(points[-1]["timesteps"])
    x_span = max(1, last_step - first_step)
    coords = []
    for point in points:
        x = left + (int(point["timesteps"]) - first_step) / x_span * (right - left)
        y = bottom - float(point["entropy_nats"]) / maximum * (bottom - top)
        coords.append(f"{x:.2f},{y:.2f}")
    threshold_y = bottom - 0.05 / maximum * (bottom - top)
    polyline = " ".join(coords)
    min_text = (
        f"Minimum logged entropy: {float(entropy['minimum_observed_entropy_nats']):.6f} "
        f"nats at step {int(entropy['minimum_observed_entropy_timestep']):,}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="32" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#202124">Run #98 PPO training entropy</text>
<text x="{left}" y="54" font-family="Arial,sans-serif" font-size="13" fill="#5f6368">Archived logger snapshots only; no training or inference performed</text>
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#5f6368" stroke-width="1"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#5f6368" stroke-width="1"/>
<line x1="{left}" y1="{threshold_y:.2f}" x2="{right}" y2="{threshold_y:.2f}" stroke="#b3261e" stroke-dasharray="6 5" stroke-width="1.5"/>
<text x="{right}" y="{threshold_y - 7:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#b3261e">0.05-nat low-entropy threshold</text>
<polyline points="{polyline}" fill="none" stroke="#1769aa" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
<text x="{left}" y="{bottom + 26}" font-family="Arial,sans-serif" font-size="12" fill="#5f6368">{first_step:,} steps</text>
<text x="{right}" y="{bottom + 26}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#5f6368">{last_step:,} steps</text>
<text x="{(left + right) / 2:.0f}" y="{height - 48}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#202124">Training timesteps</text>
<text x="26" y="{(top + bottom) / 2:.0f}" transform="rotate(-90 26 {(top + bottom) / 2:.0f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#202124">Entropy (nats)</text>
<text x="{left}" y="{height - 18}" font-family="Arial,sans-serif" font-size="12" fill="#5f6368">{escape(min_text)}</text>
</svg>
"""
    target.write_text(svg, encoding="utf-8")


def _write_metrics_csv(snapshots: list[dict[str, float | int]], target: Path) -> None:
    fields = ["timesteps", *TRACKED_METRICS, "entropy_nats"]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for snapshot in snapshots:
            row = dict(snapshot)
            row["entropy_nats"] = max(0.0, -float(snapshot["entropy_loss"]))
            writer.writerow(row)


def build_report(
    snapshots: list[dict[str, float | int]],
    source_report: dict[str, Any],
    source_run: dict[str, Any],
    source_artifact_dir: Path,
    source_run_id: int,
    source_commit_sha: str,
) -> dict[str, Any]:
    _validate_source(source_report, source_run, source_run_id, source_commit_sha)
    entropy = _entropy_summary(snapshots)
    final_validation = source_report.get("evaluation", {}).get("validation", {})
    action_counts = final_validation.get("discrete_action_counts", {})
    hold_count = int(action_counts.get("HOLD", 0))
    buy_count = sum(
        int(count) for name, count in action_counts.items() if name.startswith("BUY_")
    )
    milestones = [
        _nearest_snapshot(snapshots, step)
        for step in range(100_000, 1_000_001, 100_000)
    ]
    low_entropy_step = entropy["first_snapshot_at_or_below_0_05_nats"]
    low_entropy_snapshot = next(
        (item for item in snapshots if int(item["timesteps"]) == low_entropy_step),
        None,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "analysis_mode": "archived_training_logs_and_existing_reports_only",
        "source": {
            "workflow_run_id": source_run_id,
            "training_commit_sha": source_commit_sha,
            "training_run_conclusion": source_run.get("conclusion"),
            "training_steps_reported_by_source": source_report["model"][
                "total_timesteps_trained"
            ],
            "periodic_metric_interval_steps": Counter(
                int(b["timesteps"]) - int(a["timesteps"])
                for a, b in zip(snapshots, snapshots[1:])
            ).most_common(1)[0][0],
        },
        "execution_scope": {
            "training_performed": False,
            "model_loaded": False,
            "model_weights_modified": False,
            "new_checkpoints_created": False,
            "validation_replayed": False,
            "final_holdout_scored_or_recomputed": False,
            "reward_function_modified": False,
            "training_code_modified": False,
        },
        "training_entropy": entropy,
        "milestone_comparison": milestones,
        "checkpoint_analysis": {
            **_checkpoint_inventory(source_artifact_dir),
            "historical_checkpoint_policy_probabilities_available": False,
            "exact_timestep_when_hold_probability_exceeded_buy_probability": None,
            "explanation": (
                "The source artifact contains the final model only. Periodic PPO logger "
                "metrics cannot reconstruct the policy logits from earlier weights."
            ),
            "final_validation_reference": {
                "source_final_model_deterministic_action_counts": action_counts,
                "hold_actions": hold_count,
                "buy_actions": buy_count,
                "hold_exceeded_buy_in_final_model": hold_count > buy_count,
                "note": "This is the recorded final-model Validation result, not a checkpoint crossover time.",
            },
        },
        "reward_component_analysis": _reward_evidence(source_report),
        "causal_assessment": {
            "empirical_observation": (
                "Training entropy fell to near zero in the archived periodic logs while "
                "auxiliary guidance loss rose substantially."
            ),
            "first_logged_entropy_at_or_below_0_05_nats": entropy[
                "first_snapshot_at_or_below_0_05_nats"
            ],
            "minimum_logged_entropy_nats": entropy["minimum_observed_entropy_nats"],
            "minimum_logged_entropy_timestep": entropy[
                "minimum_observed_entropy_timestep"
            ],
            "auxiliary_loss_at_low_entropy_milestone": (
                float(low_entropy_snapshot["auxiliary_loss"])
                if low_entropy_snapshot is not None
                else None
            ),
            "auxiliary_loss_weight_at_low_entropy_milestone": (
                float(low_entropy_snapshot["auxiliary_loss_weight"])
                if low_entropy_snapshot is not None
                and "auxiliary_loss_weight" in low_entropy_snapshot
                else None
            ),
            "cause_proven": False,
            "interpretation": (
                "The timing is consistent with policy entropy collapse around the period "
                "when training guidance and auxiliary-loss weight were annealing away. "
                "This is correlation, not causal proof; saved logs cannot identify whether "
                "entropy coefficient, reward balance, or cost pressure dominated."
            ),
        },
        "safety_invariants": dict(SAFETY),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_run = json.loads(args.source_run_metadata.read_text(encoding="utf-8"))
    snapshots = parse_training_logs(args.logs_dir)
    report = build_report(
        snapshots,
        source_report,
        source_run,
        args.source_artifact_dir,
        args.source_run_id,
        args.source_commit_sha,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "rl-training-dynamics-audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(snapshots, args.out_dir / "training-metrics.csv")
    _render_entropy_svg(
        report["training_entropy"], args.out_dir / "training-entropy.svg"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "snapshot_count": report["training_entropy"]["snapshot_count"],
                "first_entropy_at_or_below_0_05_nats": report["training_entropy"][
                    "first_snapshot_at_or_below_0_05_nats"
                ],
                "minimum_entropy_nats": report["training_entropy"][
                    "minimum_observed_entropy_nats"
                ],
                "minimum_entropy_timestep": report["training_entropy"][
                    "minimum_observed_entropy_timestep"
                ],
                "historical_checkpoint_count": report["checkpoint_analysis"][
                    "checkpoint_count"
                ],
                "exact_hold_buy_crossover_timestep": report["checkpoint_analysis"][
                    "exact_timestep_when_hold_probability_exceeded_buy_probability"
                ],
                "training_reward_component_timeseries_available": report[
                    "reward_component_analysis"
                ]["per_step_training_reward_component_timeseries_available"],
                "safety_invariants": report["safety_invariants"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
