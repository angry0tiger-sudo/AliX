"""Diagnose Recurrent PPO deterministic versus sampled actions on Validation only."""

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

from scripts.run_rl_trading_agent import (
    _exclude_previous_holdout_window,
    _load_market_data,
    _sha256,
    _split_training_validation,
    _utc_from_ms,
)
from services.rl_research.rl_trading_environment import (
    ACTION_NAMES,
    BTCUSDTTradingEnv,
    DEFAULT_INITIAL_EQUITY,
    MarketObservationScaler,
    POSITION_SIZE_FRACTIONS,
)

SOURCE_RUN_ID = 40267297996
SOURCE_COMMIT_SHA = "ae57111a66afc56324d8c2170d5354261dc608fc"
PROBABILITY_FIELDS = (
    *(f"p_{name.lower()}" for name in ACTION_NAMES),
    *(f"p_size_{int(round(float(size) * 100))}" for size in POSITION_SIZE_FRACTIONS),
)
JOINT_PROBABILITY_FIELDS = tuple(
    f"joint_p_{name.lower()}_{int(round(float(size) * 100))}"
    for name in ACTION_NAMES
    for size in POSITION_SIZE_FRACTIONS
)
PROBABILITY_CSV_FIELDS = (
    "step_index",
    "timestamp_utc",
    *PROBABILITY_FIELDS,
    *JOINT_PROBABILITY_FIELDS,
    "direction_entropy",
    "size_entropy",
    "deterministic_action",
    "deterministic_size_fraction",
    "deterministic_selected_joint_probability",
    "stochastic_action",
    "stochastic_size_fraction",
    "stochastic_selected_joint_probability",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--expected-source-commit-sha", required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser.parse_args(argv)


def _validate_source_report_identity(
    args: argparse.Namespace, source_report: dict[str, Any]
) -> None:
    if source_report.get("schema_version") != "alix.rl-trading-agent-report.v6":
        raise ValueError("source_report_schema_invalid")
    if source_report.get("commit_sha") != args.expected_source_commit_sha:
        raise ValueError("source_report_commit_sha_mismatch")
    if args.expected_source_commit_sha != SOURCE_COMMIT_SHA:
        raise ValueError("unexpected_source_training_commit")


def _validate_source_model_metadata(source_report: dict[str, Any]) -> None:
    model_metadata = source_report.get("model", {})
    if model_metadata.get("algorithm") != "RecurrentPPO":
        raise ValueError("source_model_algorithm_invalid")
    action_space = source_report.get("environment", {}).get("action_space", {})
    if action_space.get("type") != "MultiDiscrete" or action_space.get("nvec") != [
        3,
        4,
    ]:
        raise ValueError("source_action_space_invalid")
    environment = source_report.get("environment", {})
    if environment.get("observation_contains_position_or_pnl") is not False:
        raise ValueError("source_observation_contract_invalid")
    if environment.get("observation_contains_technical_indicators") is not False:
        raise ValueError("source_observation_contract_invalid")
    if source_report.get("production_promotion_allowed") is not False:
        raise ValueError("source_promotion_invariant_invalid")


def _validate_source_model_artifact(
    args: argparse.Namespace, source_report: dict[str, Any]
) -> str:
    model_metadata = source_report.get("model", {})
    expected_model_hash = model_metadata.get("model_sha256")
    if (
        not isinstance(expected_model_hash, str)
        or _sha256(args.model) != expected_model_hash
    ):
        raise ValueError("source_model_sha256_mismatch")
    if args.model.name != model_metadata.get("model_artifact"):
        raise ValueError("source_model_filename_mismatch")
    return expected_model_hash


def _validate_source_dataset(
    source_report: dict[str, Any], manifest: dict[str, Any]
) -> None:
    expected_data_hash = source_report.get("dataset", {}).get("csv_sha256")
    if manifest.get("csv_sha256") != expected_data_hash:
        raise ValueError("source_dataset_sha256_mismatch")


def _validate_source_safety(source_report: dict[str, Any]) -> None:
    safety = source_report.get("safety_invariants", {})
    if safety.get("Champion", "invalid") is not None:
        raise ValueError("source_champion_invariant_invalid")
    if safety.get("order_permission") is not False:
        raise ValueError("source_order_permission_invariant_invalid")
    if safety.get("trading_authorization") != "NO-GO":
        raise ValueError("source_trading_authorization_invariant_invalid")
    if safety.get("private_api_calls") != 0:
        raise ValueError("source_private_api_calls_nonzero")
    if safety.get("orders_or_execution_api_used") is not False:
        raise ValueError("source_execution_api_invariant_invalid")
    if safety.get("real_order_attempts") != 0:
        raise ValueError("source_real_order_attempts_nonzero")


def _validate_source_artifacts(
    args: argparse.Namespace,
    source_report: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    _validate_source_report_identity(args, source_report)
    _validate_source_model_metadata(source_report)
    _validate_source_dataset(source_report, manifest)
    _validate_source_safety(source_report)
    return _validate_source_model_artifact(args, source_report)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _validate_probabilities(probabilities: Sequence[float], expected_size: int) -> None:
    if len(probabilities) != expected_size:
        raise ValueError("policy_probability_dimension_invalid")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ):
        raise ValueError("policy_probability_value_invalid")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("policy_probability_sum_invalid")


def _probability_summary(
    samples: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if not values:
            raise ValueError("policy_probability_samples_empty")
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "p05": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }
    return summary


def _count_distribution(
    labels: Sequence[str], counts: Counter[str], denominator: int
) -> dict[str, dict[str, int | float | None]]:
    return {
        label: {
            "count": int(counts.get(label, 0)),
            "fraction": (
                float(counts.get(label, 0) / denominator) if denominator else None
            ),
        }
        for label in labels
    }


def _classify_policy_modes(
    deterministic_counts: dict[str, int],
    stochastic_counts: dict[str, int],
) -> dict[str, str]:
    deterministic_trades = deterministic_counts.get(
        "BUY", 0
    ) + deterministic_counts.get("SELL", 0)
    stochastic_trades = stochastic_counts.get("BUY", 0) + stochastic_counts.get(
        "SELL", 0
    )
    if deterministic_trades == 0 and stochastic_trades > 0:
        return {
            "code": "STOCHASTIC_TRADES_ONLY",
            "interpretation": (
                "Sampling selected BUY/SELL while deterministic argmax selected no "
                "trades; the action distribution retains trade probability mass."
            ),
        }
    if deterministic_trades == 0 and stochastic_trades == 0:
        return {
            "code": "NO_BUY_SELL_SAMPLED",
            "interpretation": (
                "Neither mode selected BUY/SELL in this pass. Inspect the logged raw "
                "probabilities before attributing the result to deterministic argmax."
            ),
        }
    if deterministic_trades > 0 and stochastic_trades > 0:
        return {
            "code": "TRADES_IN_BOTH_MODES",
            "interpretation": (
                "Both deterministic and sampled action selection produced BUY/SELL requests."
            ),
        }
    return {
        "code": "DETERMINISTIC_TRADES_ONLY",
        "interpretation": (
            "Deterministic argmax selected BUY/SELL but this seeded stochastic sample did not."
        ),
    }


def _environment(
    data: Any,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
) -> BTCUSDTTradingEnv:
    return BTCUSDTTradingEnv(
        data,
        scaler=scaler,
        initial_equity=DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
        include_portfolio_state=False,
        signal_guidance_reward_scale=0.0,
    )


def _assert_observations_equal(left: Any, right: Any, error_code: str) -> None:
    if not np.allclose(left, right, rtol=0.0, atol=1e-7):
        raise RuntimeError(error_code)


def _initial_actor_state(model: Any, torch: Any) -> tuple[Any, Any]:
    state_shape = model.policy.lstm_hidden_state_shape
    zeros = torch.zeros(
        state_shape,
        dtype=torch.float32,
        device=model.policy.device,
    )
    return zeros, zeros.clone()


def _sample_policy_step(
    model: Any,
    observation: Any,
    actor_state: tuple[Any, Any],
    is_episode_start: bool,
    torch: Any,
) -> dict[str, Any]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    episode_start = torch.as_tensor(
        [is_episode_start],
        dtype=torch.float32,
        device=model.policy.device,
    )
    with torch.no_grad():
        distribution, next_actor_state = model.policy.get_distribution(
            observation_tensor,
            actor_state,
            episode_start,
        )
        components = getattr(distribution, "distribution", None)
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError("policy_multicategorical_distribution_invalid")
        direction_probabilities = [
            float(value)
            for value in components[0].probs.detach().cpu().numpy().reshape(-1)
        ]
        size_probabilities = [
            float(value)
            for value in components[1].probs.detach().cpu().numpy().reshape(-1)
        ]
        _validate_probabilities(direction_probabilities, len(ACTION_NAMES))
        _validate_probabilities(size_probabilities, len(POSITION_SIZE_FRACTIONS))
        deterministic_action = (
            distribution.get_actions(deterministic=True)
            .detach()
            .cpu()
            .numpy()
            .reshape(2)
            .astype(np.int64)
        )
        stochastic_action = (
            distribution.get_actions(deterministic=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(2)
            .astype(np.int64)
        )
    return {
        "direction_probabilities": direction_probabilities,
        "size_probabilities": size_probabilities,
        "deterministic_action": deterministic_action,
        "stochastic_action": stochastic_action,
        "next_actor_state": (
            next_actor_state[0].detach(),
            next_actor_state[1].detach(),
        ),
    }


def _append_probability_samples(
    samples: dict[str, list[float]],
    direction_probabilities: Sequence[float],
    size_probabilities: Sequence[float],
) -> tuple[float, float]:
    for index, name in enumerate(ACTION_NAMES):
        samples[f"p_{name.lower()}"].append(direction_probabilities[index])
    for index, fraction in enumerate(POSITION_SIZE_FRACTIONS):
        label = f"p_size_{int(round(float(fraction) * 100))}"
        samples[label].append(size_probabilities[index])
    direction_entropy = _entropy(direction_probabilities)
    size_entropy = _entropy(size_probabilities)
    samples["direction_entropy"].append(direction_entropy)
    samples["size_entropy"].append(size_entropy)
    return direction_entropy, size_entropy


def _build_probability_row(
    step_index: int,
    timestamp_ms: float,
    direction_probabilities: Sequence[float],
    size_probabilities: Sequence[float],
    deterministic_action: Any,
    stochastic_action: Any,
    direction_entropy: float,
    size_entropy: float,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "step_index": step_index,
        "timestamp_utc": _utc_from_ms(timestamp_ms),
        "direction_entropy": direction_entropy,
        "size_entropy": size_entropy,
    }
    actions = {
        "deterministic": deterministic_action,
        "stochastic": stochastic_action,
    }
    for mode, action in actions.items():
        direction_index = int(action[0])
        size_index = int(action[1])
        row[f"{mode}_action"] = str(ACTION_NAMES[direction_index])
        row[f"{mode}_size_fraction"] = float(POSITION_SIZE_FRACTIONS[size_index])
        row[f"{mode}_selected_joint_probability"] = (
            direction_probabilities[direction_index] * size_probabilities[size_index]
        )
    for index, name in enumerate(ACTION_NAMES):
        row[f"p_{name.lower()}"] = direction_probabilities[index]
    for index, fraction in enumerate(POSITION_SIZE_FRACTIONS):
        row[f"p_size_{int(round(float(fraction) * 100))}"] = size_probabilities[index]
    for direction_index, name in enumerate(ACTION_NAMES):
        for size_index, fraction in enumerate(POSITION_SIZE_FRACTIONS):
            label = f"joint_p_{name.lower()}_{int(round(float(fraction) * 100))}"
            row[label] = (
                direction_probabilities[direction_index]
                * size_probabilities[size_index]
            )
    return row


def _record_action_requests(
    actions: dict[str, Any],
    action_counts: dict[str, Counter[str]],
    size_counts: dict[str, Counter[str]],
) -> None:
    for mode, action in actions.items():
        direction = str(ACTION_NAMES[int(action[0])])
        fraction = float(POSITION_SIZE_FRACTIONS[int(action[1])])
        action_counts[mode][direction] += 1
        if direction in ("BUY", "SELL"):
            size_counts[mode][f"{int(round(fraction * 100))}%"] += 1


def _step_paired_environments(
    deterministic_env: BTCUSDTTradingEnv,
    stochastic_env: BTCUSDTTradingEnv,
    deterministic_action: Any,
    stochastic_action: Any,
    mask_reasons: dict[str, Counter[str]],
) -> dict[str, Any]:
    (
        next_deterministic,
        deterministic_reward,
        deterministic_done,
        deterministic_truncated,
        deterministic_info,
    ) = deterministic_env.step(deterministic_action)
    (
        next_stochastic,
        stochastic_reward,
        stochastic_done,
        stochastic_truncated,
        stochastic_info,
    ) = stochastic_env.step(stochastic_action)
    for mode, info in (
        ("deterministic", deterministic_info),
        ("stochastic", stochastic_info),
    ):
        reason = info.get("action_mask_reason")
        if isinstance(reason, str):
            mask_reasons[mode][reason] += 1
    return {
        "next_deterministic": next_deterministic,
        "next_stochastic": next_stochastic,
        "deterministic_reward": float(deterministic_reward),
        "stochastic_reward": float(stochastic_reward),
        "deterministic_done": bool(deterministic_done),
        "stochastic_done": bool(stochastic_done),
        "deterministic_truncated": bool(deterministic_truncated),
        "stochastic_truncated": bool(stochastic_truncated),
    }


def _validate_paired_step_boundary(step: dict[str, Any], is_last_step: bool) -> None:
    deterministic_boundary = (
        step["deterministic_done"],
        step["deterministic_truncated"],
    )
    stochastic_boundary = (
        step["stochastic_done"],
        step["stochastic_truncated"],
    )
    if deterministic_boundary != stochastic_boundary:
        raise RuntimeError("paired_policy_episode_boundaries_diverged")
    ended = bool(step["deterministic_done"] or step["deterministic_truncated"])
    if ended != is_last_step:
        raise RuntimeError("validation_episode_did_not_cover_exactly_one_pass")


def _build_mode_result(
    mode: str,
    env: BTCUSDTTradingEnv,
    action_counts: dict[str, Counter[str]],
    size_counts: dict[str, Counter[str]],
    mask_reasons: dict[str, Counter[str]],
    reward_totals: dict[str, float],
    expected_steps: int,
) -> dict[str, Any]:
    env_metrics = env.summary()
    if (
        env_metrics["signal_guidance_reward_count"] != 0
        or env_metrics["signal_guidance_reward_total"] != 0.0
    ):
        raise RuntimeError("guidance_reward_active_during_diagnostic")
    trade_actions = int(
        action_counts[mode].get("BUY", 0) + action_counts[mode].get("SELL", 0)
    )
    size_labels = [
        f"{int(round(float(value) * 100))}%" for value in POSITION_SIZE_FRACTIONS
    ]
    return {
        "policy_requested_action_distribution": _count_distribution(
            ACTION_NAMES, action_counts[mode], expected_steps
        ),
        "policy_requested_trade_size_distribution": _count_distribution(
            size_labels, size_counts[mode], trade_actions
        ),
        "policy_requested_trade_action_count": trade_actions,
        "action_mask_reason_counts": dict(sorted(mask_reasons[mode].items())),
        "reward_sum": float(reward_totals[mode]),
        "environment_summary": env_metrics,
    }


def _run_paired_validation_diagnostic(
    model: Any,
    validation_data: Any,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
    seed: int,
    output_csv: Path,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    model.policy.set_training_mode(False)
    deterministic_env = _environment(validation_data, scaler, fee_bps, slippage_bps)
    stochastic_env = _environment(validation_data, scaler, fee_bps, slippage_bps)
    deterministic_observation, _ = deterministic_env.reset(seed=seed)
    stochastic_observation, _ = stochastic_env.reset(seed=seed)
    actor_state = _initial_actor_state(model, torch)
    action_counts = {
        "deterministic": Counter[str](),
        "stochastic": Counter[str](),
    }
    size_counts = {
        "deterministic": Counter[str](),
        "stochastic": Counter[str](),
    }
    mask_reasons = {
        "deterministic": Counter[str](),
        "stochastic": Counter[str](),
    }
    reward_totals = {"deterministic": 0.0, "stochastic": 0.0}
    samples: dict[str, list[float]] = {name: [] for name in PROBABILITY_FIELDS}
    samples["direction_entropy"] = []
    samples["size_entropy"] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    expected_steps = len(validation_data) - 1
    try:
        with output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PROBABILITY_CSV_FIELDS)
            writer.writeheader()
            for step_index in range(expected_steps):
                _assert_observations_equal(
                    deterministic_observation,
                    stochastic_observation,
                    "paired_policy_observations_diverged",
                )
                policy_step = _sample_policy_step(
                    model,
                    deterministic_observation,
                    actor_state,
                    step_index == 0,
                    torch,
                )
                actor_state = policy_step["next_actor_state"]
                direction_probabilities = policy_step["direction_probabilities"]
                size_probabilities = policy_step["size_probabilities"]
                deterministic_action = policy_step["deterministic_action"]
                stochastic_action = policy_step["stochastic_action"]
                direction_entropy, size_entropy = _append_probability_samples(
                    samples, direction_probabilities, size_probabilities
                )
                writer.writerow(
                    _build_probability_row(
                        step_index,
                        float(validation_data[step_index, 0]),
                        direction_probabilities,
                        size_probabilities,
                        deterministic_action,
                        stochastic_action,
                        direction_entropy,
                        size_entropy,
                    )
                )
                _record_action_requests(
                    {
                        "deterministic": deterministic_action,
                        "stochastic": stochastic_action,
                    },
                    action_counts,
                    size_counts,
                )
                step = _step_paired_environments(
                    deterministic_env,
                    stochastic_env,
                    deterministic_action,
                    stochastic_action,
                    mask_reasons,
                )
                reward_totals["deterministic"] += step["deterministic_reward"]
                reward_totals["stochastic"] += step["stochastic_reward"]
                _validate_paired_step_boundary(step, step_index == expected_steps - 1)
                _assert_observations_equal(
                    step["next_deterministic"],
                    step["next_stochastic"],
                    "paired_policy_observations_diverged_after_step",
                )
                deterministic_observation = step["next_deterministic"]
                stochastic_observation = step["next_stochastic"]
        mode_results = {
            "deterministic": _build_mode_result(
                "deterministic",
                deterministic_env,
                action_counts,
                size_counts,
                mask_reasons,
                reward_totals,
                expected_steps,
            ),
            "stochastic": _build_mode_result(
                "stochastic",
                stochastic_env,
                action_counts,
                size_counts,
                mask_reasons,
                reward_totals,
                expected_steps,
            ),
        }
        return {
            "steps": expected_steps,
            "paired_observation_sequence_identical": True,
            "action_probability_summary": _probability_summary(samples),
            "mode_comparison": _classify_policy_modes(
                dict(action_counts["deterministic"]),
                dict(action_counts["stochastic"]),
            ),
            "deterministic": mode_results["deterministic"],
            "stochastic": mode_results["stochastic"],
        }
    finally:
        deterministic_env.close()
        stochastic_env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seed < 0:
        raise ValueError("diagnostic_seed_must_be_non_negative")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    data, manifest = _load_market_data(args.data, args.manifest)
    model_sha256 = _validate_source_artifacts(args, source_report, manifest)
    analysis_data, excluded_rows = _exclude_previous_holdout_window(data)
    train_data, validation_data, _ = _split_training_validation(analysis_data)
    scaler = MarketObservationScaler.fit(train_data)
    source_split = source_report["split"]
    validation_start = _utc_from_ms(float(validation_data[0, 0]))
    validation_end = _utc_from_ms(float(validation_data[-1, 0]))
    if source_split.get("validation_rows") != len(validation_data):
        raise ValueError("source_validation_row_count_mismatch")
    if source_split.get("validation_start_utc") != validation_start:
        raise ValueError("source_validation_start_mismatch")
    if source_split.get("validation_end_utc") != validation_end:
        raise ValueError("source_validation_end_mismatch")

    recurrent_ppo = getattr(importlib.import_module("sb3_contrib"), "RecurrentPPO")
    model = recurrent_ppo.load(str(args.model.with_suffix("")), env=None, device="cpu")
    model_before_hash = _sha256(args.model)
    execution = source_report["environment"]["execution"]
    output_csv = args.out_dir / "action-probabilities-validation.csv"
    diagnostic = _run_paired_validation_diagnostic(
        model=model,
        validation_data=validation_data,
        scaler=scaler,
        fee_bps=float(execution["fee_bps_per_side"]),
        slippage_bps=float(execution["slippage_bps_per_side"]),
        seed=args.seed,
        output_csv=output_csv,
    )
    model_after_hash = _sha256(args.model)
    if model_before_hash != model_after_hash or model_after_hash != model_sha256:
        raise RuntimeError("diagnostic_model_artifact_modified")
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "orders_or_execution_api_used": False,
        "real_order_attempts": 0,
        "model_output_is_order_permission": False,
        "production_promotion_allowed": False,
    }
    report: dict[str, Any] = {
        "schema_version": "alix.rl-trading-agent-policy-diagnostic.v1",
        "run_mode": "validation_only_paired_deterministic_stochastic_diagnostic",
        "source_workflow_run_id": SOURCE_RUN_ID,
        "source_training_commit_sha": args.expected_source_commit_sha,
        "diagnostic_commit_sha": args.commit_sha,
        "training_performed": False,
        "model_weights_modified": False,
        "validation_evaluations": 1,
        "final_holdout_scored": False,
        "final_holdout_metrics_computed": False,
        "source_model_sha256": model_sha256,
        "dataset": {
            "symbol": manifest["symbol"],
            "interval": manifest["interval"],
            "row_count": int(manifest["row_count"]),
            "csv_sha256": manifest["csv_sha256"],
        },
        "validation": {
            "rows": len(validation_data),
            "steps_per_mode": diagnostic["steps"],
            "start_utc": validation_start,
            "end_utc": validation_end,
            "scaler_fit_on_training_only": True,
            "source_validation_split_matched": True,
            "previous_window_rows_excluded": excluded_rows,
            "guidance_enabled": False,
            "fee_bps_per_side": float(execution["fee_bps_per_side"]),
            "slippage_bps_per_side": float(execution["slippage_bps_per_side"]),
            "stochastic_seed": args.seed,
        },
        "probability_semantics": {
            "action_space": "MultiDiscrete([3, 4])",
            "direction_probability_fields": list(
                PROBABILITY_FIELDS[: len(ACTION_NAMES)]
            ),
            "size_probability_fields": list(PROBABILITY_FIELDS[len(ACTION_NAMES) :]),
            "joint_probability_fields": list(JOINT_PROBABILITY_FIELDS),
            "joint_probability_calculation": "direction_marginal * size_marginal",
            "raw_policy_logits_masked": False,
            "csv_file": output_csv.name,
            "csv_sha256": _sha256(output_csv),
            "paired_observation_sequence_identical": diagnostic[
                "paired_observation_sequence_identical"
            ],
        },
        "diagnostic": {
            "steps": diagnostic["steps"],
            "action_probability_summary": diagnostic["action_probability_summary"],
            "mode_comparison": diagnostic["mode_comparison"],
            "deterministic": diagnostic["deterministic"],
            "stochastic": diagnostic["stochastic"],
            "interpretation_note": (
                "This is a policy-behavior diagnostic, not economic acceptance. "
                "Stochastic actions are sampled from the current policy distribution; "
                "no training, checkpoint selection, or holdout scoring is performed."
            ),
        },
        "safety_invariants": safety,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.out_dir / "rl-trading-agent-policy-diagnostic.json"
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + chr(10),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(output_json),
                "action_probabilities_csv": str(output_csv),
                "source_model_sha256": model_sha256,
                "validation": report["validation"],
                "mode_comparison": diagnostic["mode_comparison"],
                "deterministic_environment_summary": diagnostic["deterministic"][
                    "environment_summary"
                ],
                "stochastic_environment_summary": diagnostic["stochastic"][
                    "environment_summary"
                ],
                "safety_invariants": safety,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
