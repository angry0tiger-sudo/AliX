"""Train and evaluate the market-only recurrent PPO agent offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.rl_research.rl_trading_environment import (
    BTCUSDTTradingEnv,
    DEFAULT_INACTIVITY_PENALTY,
    DEFAULT_PROFITABLE_TRADE_BONUS,
    DEFAULT_INACTIVITY_GRACE_STEPS,
    MarketObservationScaler,
    OBSERVATION_FIELDS,
    validate_market_data,
)

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
MARKET_COLUMNS = ("open_time_ms", "open", "high", "low", "close", "volume")
DEFAULT_FEE_BPS = 25.0
DEFAULT_SLIPPAGE_BPS = 20.0
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
HOLDOUT_FRACTION = 0.20
MINIMUM_ROWS = 1_051_200
DRAWDOWN_ACCEPTANCE_LIMIT = 0.15
PREVIOUS_HOLDOUT_START_UTC = "2026-06-01T04:59:00+00:00"
PREVIOUS_HOLDOUT_START_MS = int(
    datetime.fromisoformat(PREVIOUS_HOLDOUT_START_UTC).timestamp() * 1000
)
ALGORITHM_NAME = "RecurrentPPO"
POLICY_NAME = "MlpLstmPolicy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest_identity(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "alix.self-evolving-data-manifest.v1":
        raise ValueError("market_manifest_schema_invalid")
    if payload.get("symbol") != "BTCUSDT" or payload.get("interval") != "1m":
        raise ValueError("market_manifest_symbol_or_interval_invalid")
    if payload.get("source_venue") != "binance_spot_public_market_data":
        raise ValueError("market_manifest_source_invalid")


def _validate_manifest_coverage(payload: dict[str, Any]) -> None:
    row_count = int(payload.get("row_count", 0))
    if row_count < MINIMUM_ROWS:
        raise ValueError("market_manifest_row_count_below_minimum")
    if int(payload.get("expected_minute_slots", 0)) != row_count:
        raise ValueError("market_manifest_expected_slots_mismatch")
    if int(payload.get("missing_minute_intervals_between_rows", -1)) != 0:
        raise ValueError("market_manifest_contains_minute_gaps")
    coverage = float(payload.get("coverage_fraction", 0.0))
    if not math.isclose(coverage, 1.0, abs_tol=1e-12):
        raise ValueError("market_manifest_coverage_incomplete")

    start_ms = int(datetime.fromisoformat(str(payload["start_utc"])).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(str(payload["end_utc"])).timestamp() * 1000)
    expected_range_rows = (end_ms - start_ms) // 60_000 + 1
    if end_ms < start_ms or expected_range_rows != row_count:
        raise ValueError("market_manifest_date_range_row_count_mismatch")


def _validate_manifest_authorization(safety: dict[str, Any]) -> None:
    if safety.get("Champion", "invalid") is not None:
        raise ValueError("market_manifest_champion_invariant_invalid")
    if safety.get("order_permission") is not False:
        raise ValueError("market_manifest_order_permission_invariant_invalid")
    if safety.get("trading_authorization") != "NO-GO":
        raise ValueError("market_manifest_trading_authorization_invariant_invalid")


def _validate_manifest_execution_safety(payload: dict[str, Any]) -> None:
    if payload.get("private_api_calls") != 0:
        raise ValueError("market_manifest_private_api_calls_nonzero")
    if payload.get("orders_or_execution_api_used") is not False:
        raise ValueError("market_manifest_execution_api_invariant_invalid")
    if payload.get("real_order_attempts") != 0:
        raise ValueError("market_manifest_real_order_attempts_nonzero")


def _validate_manifest_safety(payload: dict[str, Any]) -> None:
    safety = payload.get("safety_invariants")
    if not isinstance(safety, dict):
        raise ValueError("market_manifest_safety_invariants_missing")
    _validate_manifest_authorization(safety)
    _validate_manifest_execution_safety(payload)


def _validate_manifest_digest(payload: dict[str, Any]) -> None:
    digest = str(payload.get("csv_sha256", ""))
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest.lower()
    ):
        raise ValueError("market_manifest_sha256_invalid")


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("market_manifest_must_be_object")
    _validate_manifest_identity(payload)
    _validate_manifest_coverage(payload)
    _validate_manifest_safety(payload)
    _validate_manifest_digest(payload)
    return payload


def _load_market_data(
    csv_path: Path, manifest_path: Path
) -> tuple[Float64Array, dict[str, Any]]:
    manifest = _read_manifest(manifest_path)
    if manifest.get("csv_file") != csv_path.name:
        raise ValueError("market_manifest_filename_mismatch")
    observed_hash = _sha256(csv_path)
    if observed_hash != manifest["csv_sha256"]:
        raise ValueError("market_csv_sha256_mismatch")

    frame = pd.read_csv(
        csv_path,
        usecols=list(MARKET_COLUMNS),
        dtype={
            "open_time_ms": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    data = frame.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    if len(data) != int(manifest["row_count"]):
        raise ValueError("market_csv_row_count_mismatch")
    validate_market_data(data)
    start_ms = int(data[0, 0])
    end_ms = int(data[-1, 0])
    if start_ms != int(
        datetime.fromisoformat(str(manifest["start_utc"])).timestamp() * 1000
    ):
        raise ValueError("market_manifest_start_timestamp_mismatch")
    manifest_end_ms = int(
        datetime.fromisoformat(str(manifest["end_utc"])).timestamp() * 1000
    )
    if end_ms != manifest_end_ms:
        raise ValueError("market_manifest_end_timestamp_mismatch")
    return data, manifest


def _env_factory(
    data: Float64Array,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
) -> Callable[[], BTCUSDTTradingEnv]:
    def create() -> BTCUSDTTradingEnv:
        return BTCUSDTTradingEnv(
            data,
            scaler=scaler,
            initial_equity=10_000.0,
            fee_bps_per_side=fee_bps,
            slippage_bps_per_side=slippage_bps,
            random_start=False,
        )

    return create


def _evaluate_policy(
    model: Any,
    data: Float64Array,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
    seed: int,
) -> dict[str, Any]:
    env = BTCUSDTTradingEnv(
        data,
        scaler=scaler,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
    )
    observation, _ = env.reset(seed=seed)
    recurrent_state: Any = None
    episode_start = np.asarray([True], dtype=np.bool_)
    reward_sum = 0.0
    steps = 0
    while True:
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        action_vector = np.asarray(action, dtype=np.float32).reshape(2)
        observation, reward, terminated, truncated, _ = env.step(action_vector)
        reward_sum += reward
        steps += 1
        episode_start = np.asarray([terminated or truncated], dtype=np.bool_)
        if terminated or truncated:
            break
    result: dict[str, Any] = env.summary()
    result.update(
        {
            "steps": steps,
            "reward_sum": float(reward_sum),
            "evaluation_seed": seed,
            "evaluation_passes": 1,
        }
    )
    env.close()
    return result


def _evaluate_baseline(
    data: Float64Array,
    scaler: MarketObservationScaler,
    fee_bps: float,
    slippage_bps: float,
    baseline: str,
) -> dict[str, Any]:
    env = BTCUSDTTradingEnv(
        data,
        scaler=scaler,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
    )
    env.reset(seed=20261025)
    steps = 0
    while True:
        if baseline == "buy_and_hold" and steps == 0:
            action = np.asarray([1.0, 1.0], dtype=np.float32)
        else:
            action = np.asarray([0.0, 0.0], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    result: dict[str, Any] = env.summary()
    result.update({"steps": steps, "evaluation_passes": 1})
    env.close()
    return result


def _utc_from_ms(value: float) -> str:
    return datetime.fromtimestamp(value / 1000.0, UTC).isoformat()


def _economic_assessment(
    validation: dict[str, Any], holdout: dict[str, Any]
) -> dict[str, Any]:
    criteria = {
        "validation_net_return_positive": float(validation["net_return"]) > 0.0,
        "holdout_net_return_positive": float(holdout["net_return"]) > 0.0,
        "holdout_max_drawdown_at_or_below_15_percent": (
            float(holdout["max_drawdown"]) <= DRAWDOWN_ACCEPTANCE_LIMIT
        ),
    }
    reasons = [name for name, passed in criteria.items() if not passed]
    return {
        "status": "CANDIDATE_REQUIRES_OWNER_REVIEW" if not reasons else "NO-GO",
        "criteria": criteria,
        "reasons": reasons,
        "minimum_trade_count_gate": None,
        "profit_target_or_take_profit_gate": None,
        "timeframe_gate": None,
        "automatic_promotion_allowed": False,
        "note": (
            "This assessment is descriptive and does not authorize trading. "
            "No trade-count minimum is used to train or score the policy."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/rl-agent"))
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="0 trains over one chronological pass through the training split.",
    )
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--commit-sha", default="unknown")
    return parser.parse_args(argv)


def _validate_cli_arguments(args: argparse.Namespace) -> None:
    if args.timesteps < 0:
        raise ValueError("timesteps_must_be_non_negative")
    if not math.isfinite(args.fee_bps) or not 0 <= args.fee_bps < 10_000:
        raise ValueError("fee_bps_out_of_range")
    if not math.isfinite(args.slippage_bps) or not 0 <= args.slippage_bps < 10_000:
        raise ValueError("slippage_bps_out_of_range")


def _exclude_previous_holdout_window(
    data: Float64Array,
) -> tuple[Float64Array, int]:
    cutoff_index = int(
        np.searchsorted(data[:, 0], PREVIOUS_HOLDOUT_START_MS, side="left")
    )
    excluded_rows = len(data) - cutoff_index
    if cutoff_index <= 0 or excluded_rows <= 0:
        raise ValueError("previous_holdout_window_not_present_in_market_data")
    analysis_data = data[:cutoff_index]
    if int(analysis_data[-1, 0]) >= PREVIOUS_HOLDOUT_START_MS:
        raise ValueError("analysis_window_overlaps_previous_holdout")
    return analysis_data, excluded_rows


def _split_training_validation(
    data: Float64Array,
) -> tuple[Float64Array, Float64Array, int]:
    train_end = int(len(data) * TRAIN_FRACTION)
    validation_end = int(len(data) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train_data = data[:train_end]
    validation_data = data[train_end:validation_end]
    if len(train_data) < 100_000 or len(validation_data) < 10_000:
        raise ValueError("chronological_data_splits_too_small")
    return train_data, validation_data, validation_end


def _training_budget(
    training_rows: int, requested_timesteps: int, rollout_steps: int
) -> tuple[int, int]:
    requested_steps = requested_timesteps or (training_rows - 1)
    if requested_timesteps == 0:
        available_steps = min(requested_steps, training_rows - 1)
        training_steps = (available_steps // rollout_steps) * rollout_steps
    else:
        training_steps = (
            (requested_steps + rollout_steps - 1) // rollout_steps
        ) * rollout_steps
    if training_steps < rollout_steps:
        raise ValueError("effective_training_timesteps_below_rollout_size")
    return requested_steps, training_steps


@dataclass(frozen=True, slots=True)
class TrainingRun:
    model: Any
    torch_version: str
    model_path: Path
    requested_timesteps: int
    n_steps: int
    training_seconds: float


def _train_recurrent_ppo(
    train_data: Float64Array,
    scaler: MarketObservationScaler,
    args: argparse.Namespace,
    requested_steps: int,
    training_steps: int,
    n_steps: int,
) -> TrainingRun:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    recurrent_ppo = getattr(importlib.import_module("sb3_contrib"), "RecurrentPPO")
    dummy_vec_env = getattr(
        importlib.import_module("stable_baselines3.common.vec_env"), "DummyVecEnv"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "rl_trading_agent_recurrent_ppo.zip"
    envs = dummy_vec_env(
        [_env_factory(train_data, scaler, args.fee_bps, args.slippage_bps)]
    )
    envs.seed(args.seed)
    model = recurrent_ppo(
        POLICY_NAME,
        envs,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=0.2,
        max_grad_norm=0.5,
        seed=args.seed,
        device="cpu",
        verbose=1,
        policy_kwargs={
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
            "net_arch": [64, 64],
        },
    )
    started_at = time.monotonic()
    try:
        model.learn(
            total_timesteps=training_steps,
            progress_bar=False,
            reset_num_timesteps=True,
            log_interval=10,
        )
        model.save(str(model_path.with_suffix("")))
    finally:
        envs.close()
    training_seconds = time.monotonic() - started_at
    if not model_path.is_file():
        raise RuntimeError("trained_model_artifact_missing")
    loaded_model = recurrent_ppo.load(
        str(model_path.with_suffix("")), env=None, device="cpu"
    )
    return TrainingRun(
        model=loaded_model,
        torch_version=str(torch.__version__),
        model_path=model_path,
        requested_timesteps=requested_steps,
        n_steps=n_steps,
        training_seconds=float(training_seconds),
    )


def _evaluate_validation_and_holdout(
    data: Float64Array,
    validation_data: Float64Array,
    validation_end: int,
    scaler: MarketObservationScaler,
    args: argparse.Namespace,
    model: Any,
) -> dict[str, Any]:
    validation_metrics = _evaluate_policy(
        model, validation_data, scaler, args.fee_bps, args.slippage_bps, args.seed
    )
    validation_buy_and_hold = _evaluate_baseline(
        validation_data, scaler, args.fee_bps, args.slippage_bps, "buy_and_hold"
    )
    validation_cash = _evaluate_baseline(
        validation_data, scaler, args.fee_bps, args.slippage_bps, "cash"
    )

    # Slice the final holdout only after model fitting and validation evaluation.
    holdout_data = data[validation_end:]
    holdout_metrics = _evaluate_policy(
        model, holdout_data, scaler, args.fee_bps, args.slippage_bps, args.seed + 1
    )
    holdout_buy_and_hold = _evaluate_baseline(
        holdout_data, scaler, args.fee_bps, args.slippage_bps, "buy_and_hold"
    )
    holdout_cash = _evaluate_baseline(
        holdout_data, scaler, args.fee_bps, args.slippage_bps, "cash"
    )
    if holdout_metrics["steps"] != len(holdout_data) - 1:
        raise RuntimeError("holdout_evaluation_did_not_cover_exactly_one_pass")
    return {
        "holdout_data": holdout_data,
        "validation_metrics": validation_metrics,
        "validation_buy_and_hold": validation_buy_and_hold,
        "validation_cash": validation_cash,
        "holdout_metrics": holdout_metrics,
        "holdout_buy_and_hold": holdout_buy_and_hold,
        "holdout_cash": holdout_cash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_cli_arguments(args)
    data, manifest = _load_market_data(args.data, args.manifest)
    analysis_data, _ = _exclude_previous_holdout_window(data)
    train_data, validation_data, validation_end = _split_training_validation(analysis_data)
    scaler = MarketObservationScaler.fit(train_data)
    n_steps = 1_024
    requested_steps, training_steps = _training_budget(
        len(train_data), args.timesteps, n_steps
    )
    training = _train_recurrent_ppo(
        train_data, scaler, args, requested_steps, training_steps, n_steps
    )
    evaluation = _evaluate_validation_and_holdout(
        analysis_data, validation_data, validation_end, scaler, args, training.model
    )
    holdout_data = evaluation["holdout_data"]
    report = _build_report(
        args,
        manifest,
        train_data,
        validation_data,
        holdout_data,
        evaluation["validation_metrics"],
        evaluation["validation_buy_and_hold"],
        evaluation["validation_cash"],
        evaluation["holdout_metrics"],
        evaluation["holdout_buy_and_hold"],
        evaluation["holdout_cash"],
        training.requested_timesteps,
        training.n_steps,
        training.training_seconds,
        training.model_path,
        training.torch_version,
        training.model,
    )
    report_path = args.out_dir / "rl-trading-agent-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


def _build_report(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    train_data: Float64Array,
    validation_data: Float64Array,
    holdout_data: Float64Array,
    validation_metrics: dict[str, Any],
    validation_buy_and_hold: dict[str, Any],
    validation_cash: dict[str, Any],
    holdout_metrics: dict[str, Any],
    holdout_buy_and_hold: dict[str, Any],
    holdout_cash: dict[str, Any],
    requested_steps: int,
    n_steps: int,
    training_seconds: float,
    model_path: Path,
    torch_version: str,
    loaded_model: Any,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "alix.rl-trading-agent-report.v2",
        "run_mode": "historical_simulation_only",
        "commit_sha": args.commit_sha,
        "research_result": (
            "CANDIDATE_REQUIRES_OWNER_REVIEW"
            if _economic_assessment(validation_metrics, holdout_metrics)["status"]
            == "CANDIDATE_REQUIRES_OWNER_REVIEW"
            else "NO_ECONOMICALLY_ACCEPTABLE_STRATEGY_FOUND"
        ),
        "dataset": {
            "source_venue": manifest["source_venue"],
            "symbol": manifest["symbol"],
            "interval": manifest["interval"],
            "start_utc": manifest["start_utc"],
            "end_utc": manifest["end_utc"],
            "row_count": int(manifest["row_count"]),
            "csv_sha256": manifest["csv_sha256"],
            "coverage_fraction": float(manifest["coverage_fraction"]),
            "missing_minute_intervals_between_rows": int(
                manifest["missing_minute_intervals_between_rows"]
            ),
        },
        "split": {
            "method": "chronological_no_shuffle_excluding_previous_holdout",
            "previously_reported_holdout_excluded": True,
            "previously_reported_holdout_start_utc": PREVIOUS_HOLDOUT_START_UTC,
            "holdout_disjoint_from_previous_holdout": (
                float(holdout_data[-1, 0]) < PREVIOUS_HOLDOUT_START_MS
            ),
            "analysis_window_end_utc": _utc_from_ms(float(holdout_data[-1, 0])),
            "analysis_window_rows": len(train_data) + len(validation_data) + len(holdout_data),
            "rows_excluded_after_analysis_window": (
                int(manifest["row_count"])
                - len(train_data)
                - len(validation_data)
                - len(holdout_data)
            ),
            "holdout_novelty_note": (
                "This holdout is disjoint from the prior June-October 2026 holdout and "
                "is excluded from this run's train/validation splits. The underlying "
                "public history was used by earlier research runs, so it is not a "
                "never-before-downloaded market sample."
            ),
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "final_holdout_fraction": HOLDOUT_FRACTION,
            "training_rows": len(train_data),
            "validation_rows": len(validation_data),
            "final_holdout_rows": len(holdout_data),
            "training_start_utc": _utc_from_ms(float(train_data[0, 0])),
            "training_end_utc": _utc_from_ms(float(train_data[-1, 0])),
            "validation_start_utc": _utc_from_ms(float(validation_data[0, 0])),
            "validation_end_utc": _utc_from_ms(float(validation_data[-1, 0])),
            "final_holdout_start_utc": _utc_from_ms(float(holdout_data[0, 0])),
            "final_holdout_end_utc": _utc_from_ms(float(holdout_data[-1, 0])),
            "scaler_fit_on_training_only": True,
            "validation_used_for_model_selection": False,
            "final_holdout_used_for_training": False,
            "final_holdout_used_for_model_selection": False,
            "final_holdout_evaluation_performed": True,
            "final_holdout_evaluations": 1,
        },
        "environment": {
            "standard": "Gymnasium",
            "raw_market_fields": list(MARKET_COLUMNS),
            "observation_fields": list(OBSERVATION_FIELDS),
            "observation_contains_position_or_pnl": False,
            "observation_contains_technical_indicators": False,
            "action_space": {
                "type": "continuous Box",
                "shape": [2],
                "decision_signal_bounds": [-1.0, 1.0],
                "position_size_bounds": [0.0, 1.0],
                "sell_threshold": -1.0 / 3.0,
                "buy_threshold": 1.0 / 3.0,
                "buy_semantics": "target long allocation as a fraction of equity",
                "sell_semantics": "fraction of current holding to sell",
                "hold_semantics": "do not rebalance",
                "masking": {
                    "enforcement": "deterministic_environment_execution_boundary",
                    "policy_distribution_logits_masked": False,
                    "sell_requires_positive_inventory": True,
                    "buy_requires_unallocated_cash_and_must_increase_position": True,
                    "masked_action_is_non_executable": True,
                },
            },
            "execution": {
                "signal_observed_after_bar_close": True,
                "execution_event": "next_bar_open",
                "short_selling_enabled": False,
                "fee_bps_per_side": args.fee_bps,
                "slippage_bps_per_side": args.slippage_bps,
                "accounting_includes_fees": True,
                "accounting_includes_slippage": True,
                "drawdown_uses_intrabar_high_low": True,
                "fixed_timeframe_rule": None,
                "fixed_max_holding_period_candles": None,
                "fixed_take_profit_rule": None,
                "fixed_stop_loss_rule": None,
            },
            "reward": {
                "net_component": "log(current_equity / previous_equity)",
                "equity_includes_fees_and_slippage": True,
                "drawdown_increment_penalty": 5.0,
                "drawdown_level_penalty": 0.01,
                "inactivity_penalty_per_unchanged_position_step": DEFAULT_INACTIVITY_PENALTY,
                "inactivity_grace_steps": DEFAULT_INACTIVITY_GRACE_STEPS,
                "profitable_trade_bonus": DEFAULT_PROFITABLE_TRADE_BONUS,
                "profitable_trade_bonus_trigger": (
                    "completed round trip with cumulative realized PnL positive "
                    "after entry and exit fees and slippage"
                ),
                "formula": (
                    "log_net_return - 5.0 * max(0, risk_drawdown - prior_close_drawdown) "
                    "- 0.01 * risk_drawdown "
                    "- 0.001 * indicator(consecutive_unchanged_steps > 3) "
                    "+ 0.0001 * profitable_closed_trade_count"
                ),
                "hard_trade_count_minimum": None,
                "profit_target": None,
                "timeframe_constraint": None,
            },
        },
        "model": {
            "algorithm": ALGORITHM_NAME,
            "policy": POLICY_NAME,
            "stable_baselines3_version": version("stable-baselines3"),
            "sb3_contrib_version": version("sb3-contrib"),
            "gymnasium_version": version("gymnasium"),
            "torch_version": torch_version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": "cpu",
            "seed": args.seed,
            "requested_timesteps": requested_steps,
            "total_timesteps_trained": int(loaded_model.num_timesteps),
            "training_rows": len(train_data),
            "first_pass_training_coverage_fraction": min(
                1.0, float(loaded_model.num_timesteps) / (len(train_data) - 1)
            ),
            "estimated_full_pass_equivalents": float(
                loaded_model.num_timesteps / (len(train_data) - 1)
            ),
            "training_replays_training_split": loaded_model.num_timesteps
            > len(train_data) - 1,
            "n_envs": 1,
            "n_steps": n_steps,
            "batch_size": 256,
            "epochs_per_rollout": 4,
            "lstm_hidden_size": 64,
            "training_seconds": float(training_seconds),
            "model_artifact": model_path.name,
            "model_sha256": _sha256(model_path),
            "model_reload_roundtrip": "passed",
            "no_validation_checkpoint_selection": True,
        },
        "evaluation": {
            "validation": validation_metrics,
            "validation_buy_and_hold": validation_buy_and_hold,
            "validation_cash": validation_cash,
            "final_holdout": holdout_metrics,
            "final_holdout_buy_and_hold": holdout_buy_and_hold,
            "final_holdout_cash": holdout_cash,
            "holdout_net_return_vs_buy_and_hold": (
                float(holdout_metrics["net_return"])
                - float(holdout_buy_and_hold["net_return"])
            ),
            "holdout_pass_count": 1,
        },
        "economic_assessment": _economic_assessment(
            validation_metrics, holdout_metrics
        ),
        "production_promotion_allowed": False,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "orders_or_execution_api_used": False,
            "real_order_attempts": 0,
            "model_output_is_order_permission": False,
        },
    }
    return report


if __name__ == "__main__":
    raise SystemExit(main())
