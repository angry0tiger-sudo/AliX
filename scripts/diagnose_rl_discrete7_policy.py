"""Diagnose the saved Discrete(7) Recurrent PPO policy on Validation only."""

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

from scripts.rl_auxiliary_ppo import (
    ACTION_MASK_POSITION_EPSILON,
    build_auxiliary_recurrent_ppo,
)
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
    CLOSE_COLUMN,
    BTCUSDTTradingEnv,
    DEFAULT_INITIAL_EQUITY,
    MarketObservationScaler,
    OBSERVATION_FIELDS,
    SELL_ACTION_INDICES,
)

SOURCE_RUN_ID = 40348192766
SOURCE_COMMIT_SHA = "4571691ec6b11146ea6f3a93e0cc7c9e742ec578"
REPORT_SCHEMA = "alix.rl-trading-agent-report.v10"
MASKED_LOGIT = -1e8
CSV_LOGIT_FIELDS = tuple(f"raw_logit_{i}" for i in range(len(ACTION_NAMES)))
CSV_MASKED_FIELDS = tuple(f"masked_logit_{i}" for i in range(len(ACTION_NAMES)))
CSV_PROBABILITY_FIELDS = tuple(f"probability_{name.lower()}" for name in ACTION_NAMES)
CSV_FIELDS = (
    "probe_kind",
    "step_index",
    "timestamp_utc",
    "current_position_fraction",
    "tensor_position_fraction",
    "target_action",
    "expected_masked_actions",
    *CSV_LOGIT_FIELDS,
    *CSV_MASKED_FIELDS,
    *CSV_PROBABILITY_FIELDS,
    "mask_check_passed",
    "unweighted_auxiliary_cross_entropy",
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
    parser.add_argument("--sample-count", type=int, default=6)
    return parser.parse_args(argv)


def _validate_source_identity(
    args: argparse.Namespace, report: dict[str, Any]
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("source_report_schema_invalid")
    if report.get("commit_sha") != args.expected_source_commit_sha:
        raise ValueError("source_report_commit_sha_mismatch")
    if args.expected_source_commit_sha != SOURCE_COMMIT_SHA:
        raise ValueError("unexpected_source_training_commit")


def _validate_source_action_contract(report: dict[str, Any]) -> None:
    action_space = report.get("environment", {}).get("action_space", {})
    if action_space.get("type") != "Discrete" or action_space.get("n") != 7:
        raise ValueError("source_action_space_must_be_discrete_seven")
    expected_actions = {str(i): name for i, name in enumerate(ACTION_NAMES)}
    if action_space.get("index_to_action") != expected_actions:
        raise ValueError("source_action_mapping_invalid")
    metadata = report.get("model", {})
    if metadata.get("algorithm") != "RecurrentPPO":
        raise ValueError("source_model_algorithm_invalid")
    if metadata.get("policy") != "ActionMaskedRecurrentActorCriticPolicy":
        raise ValueError("source_model_policy_invalid")


def _validate_source_model_hash(
    args: argparse.Namespace, report: dict[str, Any]
) -> str:
    metadata = report.get("model", {})
    if args.model.name != metadata.get("model_artifact"):
        raise ValueError("source_model_filename_mismatch")
    model_hash = metadata.get("model_sha256")
    if not isinstance(model_hash, str) or _sha256(args.model) != model_hash:
        raise ValueError("source_model_sha256_mismatch")
    return model_hash


def _validate_source_safety(report: dict[str, Any]) -> None:
    safety = report.get("safety_invariants", {})
    if (
        safety.get("Champion", "invalid") is not None
        or safety.get("order_permission") is not False
        or safety.get("trading_authorization") != "NO-GO"
        or safety.get("private_api_calls") != 0
        or safety.get("orders_or_execution_api_used") is not False
        or safety.get("real_order_attempts") != 0
    ):
        raise ValueError("source_safety_invariants_invalid")
    if report.get("production_promotion_allowed") is not False:
        raise ValueError("source_promotion_invariant_invalid")


def _validate_source_artifacts(
    args: argparse.Namespace,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    _validate_source_identity(args, report)
    _validate_source_action_contract(report)
    if manifest.get("csv_sha256") != report.get("dataset", {}).get("csv_sha256"):
        raise ValueError("source_dataset_sha256_mismatch")
    _validate_source_safety(report)
    return _validate_source_model_hash(args, report)

def _initial_actor_state(model: Any, torch: Any) -> tuple[Any, Any]:
    shape = model.policy.lstm_hidden_state_shape
    hidden = torch.zeros(shape, dtype=torch.float32, device=model.policy.device)
    return hidden, hidden.clone()


def _trace_policy_distribution(
    policy: Any,
    observation: np.ndarray,
    actor_state: tuple[Any, Any],
    episode_start: bool,
    capture_logits: bool,
    torch: Any,
) -> dict[str, Any]:
    observation_tensor, _ = policy.obs_to_tensor(observation)
    episode_starts = torch.as_tensor(
        [episode_start], dtype=torch.float32, device=policy.device
    )
    raw_rows: list[list[float]] = []
    masked_rows: list[list[float]] = []
    hook_handle: Any = None
    action_distribution: Any = None
    original_distribution_method: Any = None

    if capture_logits:

        def capture_raw(_module: Any, _inputs: Any, output: Any) -> None:
            raw_rows.append(output.detach().cpu().numpy().reshape(-1).tolist())

        def capture_masked(*args: Any, **kwargs: Any) -> Any:
            logits = kwargs.get("action_logits")
            if logits is None and args:
                logits = args[0]
            if logits is None:
                raise RuntimeError("masked_logits_not_observed")
            masked_rows.append(logits.detach().cpu().numpy().reshape(-1).tolist())
            return original_distribution_method(*args, **kwargs)

        hook_handle = policy.action_net.register_forward_hook(capture_raw)
        action_distribution = policy.action_dist
        original_distribution_method = action_distribution.proba_distribution
        action_distribution.proba_distribution = capture_masked

    try:
        distribution, next_state = policy.get_distribution(
            observation_tensor, actor_state, episode_starts
        )
    finally:
        if capture_logits:
            action_distribution.proba_distribution = original_distribution_method
            hook_handle.remove()

    action_probabilities = (
        distribution.distribution.probs.detach().cpu().numpy().reshape(-1).tolist()
    )
    action_index = int(
        distribution.get_actions(deterministic=True)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)[0]
    )
    position_index = OBSERVATION_FIELDS.index("current_position_fraction")
    observation_position = float(observation[position_index])
    tensor_position = float(observation_tensor[0, position_index].item())
    if not math.isclose(
        observation_position, tensor_position, rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("position_feature_changed_during_tensor_conversion")
    if not math.isclose(sum(action_probabilities), 1.0, abs_tol=1e-5):
        raise RuntimeError("masked_action_probabilities_do_not_sum_to_one")

    if capture_logits and (len(raw_rows) != 1 or len(masked_rows) != 1):
        raise RuntimeError("expected_one_raw_and_one_masked_logit_vector")
    return {
        "action_index": action_index,
        "action_name": ACTION_NAMES[action_index],
        "probabilities": action_probabilities,
        "raw_logits": raw_rows[0] if raw_rows else None,
        "masked_logits": masked_rows[0] if masked_rows else None,
        "observation_position_fraction": observation_position,
        "tensor_position_fraction": tensor_position,
        "next_actor_state": (next_state[0].detach(), next_state[1].detach()),
    }


def _expected_masked_indices(position_fraction: float) -> list[int]:
    if position_fraction >= 1.0 - ACTION_MASK_POSITION_EPSILON:
        return list(BUY_ACTION_INDICES)
    if position_fraction <= ACTION_MASK_POSITION_EPSILON:
        return list(SELL_ACTION_INDICES)
    return []


def _find_unexpected_masked_indices(
    masked_logits: Sequence[float], expected: Sequence[int]
) -> list[int]:
    return [
        index
        for index, value in enumerate(masked_logits)
        if value <= MASKED_LOGIT / 2 and index not in expected
    ]


def _unmasked_logits_unchanged(
    raw_logits: Sequence[float],
    masked_logits: Sequence[float],
    expected: Sequence[int],
) -> bool:
    for index in range(len(ACTION_NAMES)):
        if index not in expected:
            if not math.isclose(
                raw_logits[index], masked_logits[index], rel_tol=1e-6, abs_tol=1e-5
            ):
                return False
    return True


def _masked_logits_use_sentinel(
    masked_logits: Sequence[float], expected: Sequence[int]
) -> bool:
    return all(
        math.isclose(
            masked_logits[index], MASKED_LOGIT, rel_tol=0.0, abs_tol=1.0
        )
        for index in expected
    )


def _forbidden_probabilities_are_zero(
    probabilities: Sequence[float], expected: Sequence[int]
) -> bool:
    return all(probabilities[index] == 0.0 for index in expected)


def _check_mask_contract(probe: dict[str, Any]) -> dict[str, Any]:
    raw = probe["raw_logits"]
    masked = probe["masked_logits"]
    probabilities = probe.get("probabilities")
    if probabilities is None:
        probability_map = probe["post_mask_probabilities"]
        probabilities = [probability_map[name] for name in ACTION_NAMES]
    position = float(probe["observation_position_fraction"])
    expected = _expected_masked_indices(position)
    unexpected = _find_unexpected_masked_indices(masked, expected)
    logits_correct = _masked_logits_use_sentinel(masked, expected)
    probabilities_zero = _forbidden_probabilities_are_zero(probabilities, expected)
    legal_logits_unchanged = _unmasked_logits_unchanged(raw, masked, expected)
    probabilities_normalized = math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-5
    )
    passed = (
        logits_correct
        and probabilities_zero
        and legal_logits_unchanged
        and not unexpected
        and probabilities_normalized
    )
    return {
        "passed": passed,
        "position_fraction": position,
        "expected_masked_actions": [ACTION_NAMES[index] for index in expected],
        "masked_action_probabilities": {
            ACTION_NAMES[index]: probabilities[index] for index in expected
        },
        "masked_logits_use_sentinel": logits_correct,
        "forbidden_probabilities_exactly_zero": probabilities_zero,
        "unmasked_logits_unchanged": legal_logits_unchanged,
        "unexpected_masked_actions": [ACTION_NAMES[index] for index in unexpected],
    }

def _cross_entropy_from_logits(logits: Sequence[float], target: int) -> float:
    maximum = max(logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return float(log_normalizer - logits[target])


def _probe_record(
    kind: str,
    step_index: int,
    timestamp_ms: float,
    observation: np.ndarray,
    trace: dict[str, Any],
    target_index: int | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "probe_kind": kind,
        "step_index": step_index,
        "timestamp_utc": _utc_from_ms(timestamp_ms),
        "current_position_fraction": trace["observation_position_fraction"],
        "tensor_position_fraction": trace["tensor_position_fraction"],
        "deterministic_action": trace["action_name"],
        "raw_logits": trace["raw_logits"],
        "masked_logits": trace["masked_logits"],
        "post_mask_probabilities": {
            name: trace["probabilities"][index]
            for index, name in enumerate(ACTION_NAMES)
        },
        "observation_fields": {
            name: float(observation[index])
            for index, name in enumerate(OBSERVATION_FIELDS)
        },
        "is_synthetic_counterfactual": kind != "validation_actual",
    }
    if target_index is not None:
        record["auxiliary_guidance_target"] = ACTION_NAMES[target_index]
        record["unweighted_auxiliary_cross_entropy"] = _cross_entropy_from_logits(
            trace["masked_logits"], target_index
        )
        record["guidance_target_is_masked"] = target_index in _expected_masked_indices(
            float(trace["observation_position_fraction"])
        )
    record["mask_check"] = _check_mask_contract(record)
    return record


def _csv_rows(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        raw = probe["raw_logits"] or []
        masked = probe["masked_logits"] or []
        probabilities = probe["post_mask_probabilities"]
        check = probe["mask_check"]
        target = probe.get("auxiliary_guidance_target", "")
        row: dict[str, Any] = {
            "probe_kind": probe["probe_kind"],
            "step_index": probe["step_index"],
            "timestamp_utc": probe["timestamp_utc"],
            "current_position_fraction": probe["current_position_fraction"],
            "tensor_position_fraction": probe["tensor_position_fraction"],
            "target_action": target,
            "expected_masked_actions": "|".join(check["expected_masked_actions"]),
            "mask_check_passed": check["passed"],
            "unweighted_auxiliary_cross_entropy": probe.get(
                "unweighted_auxiliary_cross_entropy", ""
            ),
        }
        row.update({field: raw[i] for i, field in enumerate(CSV_LOGIT_FIELDS)})
        row.update({field: masked[i] for i, field in enumerate(CSV_MASKED_FIELDS)})
        row.update(
            {
                field: probabilities[ACTION_NAMES[i]]
                for i, field in enumerate(CSV_PROBABILITY_FIELDS)
            }
        )
        rows.append(row)
    return rows


def _append_counterfactual_probes(
    probes: list[dict[str, Any]],
    policy: Any,
    observation: np.ndarray,
    actor_state: tuple[Any, Any],
    step_index: int,
    timestamp_ms: float,
    episode_start: bool,
    torch: Any,
) -> None:
    position_index = OBSERVATION_FIELDS.index("current_position_fraction")
    for kind, hypothetical_position in (
        ("counterfactual_partial_position", 0.5),
        ("counterfactual_full_position", 1.0),
    ):
        synthetic_observation = observation.copy()
        synthetic_observation[position_index] = hypothetical_position
        trace = _trace_policy_distribution(
            policy, synthetic_observation, actor_state, episode_start, True, torch
        )
        probes.append(
            _probe_record(
                kind,
                step_index,
                timestamp_ms,
                synthetic_observation,
                trace,
                None,
            )
        )


def _validate_validation_scan(
    tensor_matches: bool,
    action_counts: Counter[str],
    guidance_counts: Counter[str],
    steps: int,
    probes: list[dict[str, Any]],
) -> None:
    if not tensor_matches:
        raise RuntimeError("position_observation_contract_failed")
    if sum(action_counts.values()) != steps or sum(guidance_counts.values()) != steps:
        raise RuntimeError("validation_diagnostic_step_count_mismatch")
    if any(not probe["mask_check"]["passed"] for probe in probes):
        raise RuntimeError("action_logit_mask_diagnostic_failed")
    if any(probe.get("guidance_target_is_masked", False) for probe in probes):
        raise RuntimeError("auxiliary_guidance_target_was_masked")


def _run_validation_diagnostic(
    model: Any,
    validation_data: np.ndarray,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
    seed: int,
    sample_count: int,
    output_csv: Path,
    source_auxiliary: dict[str, Any],
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    model.policy.set_training_mode(False)
    environment = BTCUSDTTradingEnv(
        validation_data,
        scaler=scaler,
        initial_equity=DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
        include_portfolio_state=True,
        signal_guidance_reward_scale=0.0,
    )
    observation, _ = environment.reset(seed=seed)
    actor_state = _initial_actor_state(model, torch)
    steps = len(validation_data) - 1
    if steps < 1:
        environment.close()
        raise ValueError("validation_split_too_small")
    selected_steps = set(
        np.linspace(0, steps - 1, num=min(sample_count, steps), dtype=np.int64).tolist()
    )
    first_selected_step = min(selected_steps)
    position_index = OBSERVATION_FIELDS.index("current_position_fraction")
    action_counts: Counter[str] = Counter()
    guidance_counts: Counter[str] = Counter()
    probability_sums = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    position_min = math.inf
    position_max = -math.inf
    position_nonzero_steps = 0
    tensor_matches = True
    probes: list[dict[str, Any]] = []
    try:
        for step_index in range(steps):
            position = float(observation[position_index])
            position_min = min(position_min, position)
            position_max = max(position_max, position)
            position_nonzero_steps += int(position > ACTION_MASK_POSITION_EPSILON)
            current_price = float(validation_data[step_index, CLOSE_COLUMN])
            target_index = int(environment._guidance_target_index(current_price))
            guidance_counts[ACTION_NAMES[target_index]] += 1
            selected = step_index in selected_steps
            trace = _trace_policy_distribution(
                model.policy,
                observation,
                actor_state,
                step_index == 0,
                selected,
                torch,
            )
            tensor_matches = tensor_matches and math.isclose(
                float(observation[position_index]),
                float(trace["tensor_position_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            probability_sums += np.asarray(trace["probabilities"], dtype=np.float64)
            action_counts[trace["action_name"]] += 1
            if selected:
                probes.append(
                    _probe_record(
                        "validation_actual",
                        step_index,
                        float(validation_data[step_index, 0]),
                        observation,
                        trace,
                        target_index,
                    )
                )
                if step_index == first_selected_step:
                    _append_counterfactual_probes(
                        probes,
                        model.policy,
                        observation,
                        actor_state,
                        step_index,
                        float(validation_data[step_index, 0]),
                        step_index == 0,
                        torch,
                    )
            next_observation, _reward, terminated, truncated, _info = environment.step(
                np.int64(trace["action_index"])
            )
            actor_state = trace["next_actor_state"]
            observation = next_observation
            if (terminated or truncated) != (step_index == steps - 1):
                raise RuntimeError(
                    "validation_diagnostic_did_not_cover_exactly_one_pass"
                )
    finally:
        environment.close()

    _validate_validation_scan(
        tensor_matches, action_counts, guidance_counts, steps, probes
    )
    rows = _csv_rows(probes)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    auxiliary_cross_entropy = [
        float(probe["unweighted_auxiliary_cross_entropy"])
        for probe in probes
        if "unweighted_auxiliary_cross_entropy" in probe
    ]
    final_weight = float(source_auxiliary["effective_weight_at_training_end"])
    mean_cross_entropy = (
        float(np.mean(auxiliary_cross_entropy)) if auxiliary_cross_entropy else None
    )
    return {
        "steps": steps,
        "selected_actual_probe_count": len(selected_steps),
        "recorded_probe_count": len(probes),
        "action_counts": {name: action_counts.get(name, 0) for name in ACTION_NAMES},
        "mean_post_mask_probabilities": {
            name: float(probability_sums[i] / steps)
            for i, name in enumerate(ACTION_NAMES)
        },
        "mean_hold_probability": float(probability_sums[0] / steps),
        "mean_buy_probability": float(
            sum(probability_sums[i] for i in BUY_ACTION_INDICES) / steps
        ),
        "mean_sell_probability": float(
            sum(probability_sums[i] for i in SELL_ACTION_INDICES) / steps
        ),
        "position_observation": {
            "field_name": OBSERVATION_FIELDS[position_index],
            "field_index": position_index,
            "input_matches_observation": tensor_matches,
            "minimum_fraction_seen": float(position_min),
            "maximum_fraction_seen": float(position_max),
            "nonzero_position_steps": position_nonzero_steps,
            "steps_checked": steps,
            "target_guidance_class_counts": {
                name: guidance_counts.get(name, 0) for name in ACTION_NAMES
            },
        },
        "masking": {
            "probe_checks_passed": True,
            "flat_position_forbidden_actions": [
                ACTION_NAMES[i] for i in SELL_ACTION_INDICES
            ],
            "full_position_forbidden_actions": [
                ACTION_NAMES[i] for i in BUY_ACTION_INDICES
            ],
            "sentinel_logit": MASKED_LOGIT,
            "probes": probes,
        },
        "auxiliary_loss": {
            "method": source_auxiliary["method"],
            "target_classes": source_auxiliary["target_classes"],
            "initial_weight": source_auxiliary["initial_weight"],
            "anneal_steps": source_auxiliary["anneal_steps"],
            "effective_weight_at_training_end": final_weight,
            "validation_loss_weight": 0.0,
            "validation_enabled_in_training_config": source_auxiliary[
                "validation_enabled"
            ],
            "final_holdout_enabled_in_training_config": source_auxiliary[
                "final_holdout_enabled"
            ],
            "validation_sampled_unweighted_cross_entropy": auxiliary_cross_entropy,
            "validation_sampled_mean_unweighted_cross_entropy": mean_cross_entropy,
            "validation_sampled_mean_weighted_loss_at_final_weight": (
                final_weight * mean_cross_entropy
                if mean_cross_entropy is not None
                else None
            ),
            "interpretation": (
                "Validation computes no gradients and applies no auxiliary-loss weight. "
                "Sampled cross-entropy measures current policy alignment with causal "
                "guidance labels; it cannot by itself prove that the training-only "
                "auxiliary loss caused the HOLD collapse."
            ),
        },
        "csv_sha256": _sha256(output_csv),
    }

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seed < 0 or args.sample_count < 3:
        raise ValueError("invalid_diagnostic_seed_or_sample_count")
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    data, manifest = _load_market_data(args.data, args.manifest)
    model_sha256 = _validate_source_artifacts(args, source_report, manifest)
    analysis_data, excluded_rows = _exclude_previous_holdout_window(data)
    train_data, validation_data, _ = _split_training_validation(analysis_data)
    source_split = source_report["split"]
    if (
        len(validation_data) != source_split["validation_rows"]
        or _utc_from_ms(float(validation_data[0, 0]))
        != source_split["validation_start_utc"]
        or _utc_from_ms(float(validation_data[-1, 0]))
        != source_split["validation_end_utc"]
    ):
        raise ValueError("source_validation_split_mismatch")
    scaler = MarketObservationScaler.fit(train_data)

    algorithm = build_auxiliary_recurrent_ppo()
    model = algorithm.load(str(args.model.with_suffix("")), env=None, device="cpu")
    if model.num_timesteps != source_report["model"]["total_timesteps_trained"]:
        raise ValueError("loaded_model_timestep_count_mismatch")
    model_hash_before = _sha256(args.model)
    execution = source_report["environment"]["execution"]
    output_csv = args.out_dir / "logits-validation-probes.csv"
    result = _run_validation_diagnostic(
        model,
        validation_data,
        scaler,
        float(execution["fee_bps_per_side"]),
        float(execution["slippage_bps_per_side"]),
        args.seed,
        args.sample_count,
        output_csv,
        source_report["auxiliary_loss_curriculum"],
    )
    if _sha256(args.model) != model_hash_before or model_hash_before != model_sha256:
        raise RuntimeError("diagnostic_model_artifact_modified")

    report = {
        "schema_version": "alix.rl-trading-agent-discrete7-diagnostic.v1",
        "run_mode": "validation_only_policy_logits_observation_auxiliary_diagnostic",
        "source_workflow_run_id": SOURCE_RUN_ID,
        "source_training_commit_sha": args.expected_source_commit_sha,
        "diagnostic_commit_sha": args.commit_sha,
        "source_model_sha256": model_sha256,
        "training_performed": False,
        "model_weights_modified": False,
        "final_holdout_scored": False,
        "final_holdout_metrics_computed": False,
        "validation": {
            "row_count": len(validation_data),
            "steps": result["steps"],
            "start_utc": source_split["validation_start_utc"],
            "end_utc": source_split["validation_end_utc"],
            "scaler_fit_on_training_only": True,
            "source_validation_split_matched": True,
            "rows_excluded_from_prior_holdout_window": excluded_rows,
            "fees_bps_per_side": float(execution["fee_bps_per_side"]),
            "slippage_bps_per_side": float(execution["slippage_bps_per_side"]),
            "stochastic_sampling_used": False,
        },
        "diagnostic": result,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "orders_or_execution_api_used": False,
            "real_order_attempts": 0,
            "model_output_is_order_permission": False,
            "production_promotion_allowed": False,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.out_dir / "rl-trading-agent-discrete7-diagnostic.json"
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(output_json),
                "probes_csv": str(output_csv),
                "validation": report["validation"],
                "action_counts": result["action_counts"],
                "mean_post_mask_probabilities": result["mean_post_mask_probabilities"],
                "position_observation": result["position_observation"],
                "auxiliary_loss": result["auxiliary_loss"],
                "masking_probe_results": [
                    {
                        "probe_kind": probe["probe_kind"],
                        "position_fraction": probe["current_position_fraction"],
                        "raw_logits": probe["raw_logits"],
                        "masked_logits": probe["masked_logits"],
                        "post_mask_probabilities": probe["post_mask_probabilities"],
                        "mask_check": probe["mask_check"],
                    }
                    for probe in result["masking"]["probes"]
                ],
                "safety_invariants": report["safety_invariants"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
