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
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_NAMES,
    BTCUSDTTradingEnv,
    DEFAULT_INITIAL_EQUITY,
    DEFAULT_OPPORTUNITY_COST_PENALTY,
    MOVING_AVERAGE_WINDOW,
    POSITION_SIZE_FRACTIONS,
    DEFAULT_INACTIVITY_PENALTY,
    DEFAULT_INVALID_ACTION_PENALTY,
    DEFAULT_PROFITABLE_TRADE_BONUS,
    DEFAULT_INACTIVITY_GRACE_STEPS,
    MarketObservationScaler,
    OBSERVATION_FIELDS,
    validate_market_data,
)
from services.rl_research.rl_cost_curriculum import (
    CURRICULUM_MINIMUM_CLOSED_TRADES,
    CURRICULUM_RAMP_STEPS,
    CURRICULUM_RAMP_UPDATE_INTERVAL_STEPS,
    TrainingCostCurriculum,
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
            initial_equity=DEFAULT_INITIAL_EQUITY,
            fee_bps_per_side=fee_bps,
            slippage_bps_per_side=slippage_bps,
            random_start=False,
        )

    return create


def _decode_discrete_policy_action(
    action: Any,
) -> tuple[NDArray[np.int64], str, float]:
    action_vector = np.asarray(action, dtype=np.int64).reshape(2)
    direction_index, size_index = map(int, action_vector)
    if not 0 <= direction_index < len(ACTION_NAMES):
        raise ValueError("policy_direction_index_out_of_range")
    if not 0 <= size_index < len(POSITION_SIZE_FRACTIONS):
        raise ValueError("policy_size_index_out_of_range")
    return (
        action_vector,
        ACTION_NAMES[direction_index],
        POSITION_SIZE_FRACTIONS[size_index],
    )


def _new_policy_size_metrics() -> dict[str, Any]:
    return {
        "counts": {f"{fraction:.2f}": 0 for fraction in POSITION_SIZE_FRACTIONS},
        "sum": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "trade_count": 0,
    }


def _record_policy_size(
    metrics: dict[str, Any], requested_action: str, requested_size: float
) -> None:
    if requested_action not in ("BUY", "SELL"):
        return
    metrics["counts"][f"{requested_size:.2f}"] += 1
    metrics["sum"] += requested_size
    metrics["min"] = min(metrics["min"], requested_size)
    metrics["max"] = max(metrics["max"], requested_size)
    metrics["trade_count"] += 1


def _run_policy_evaluation_episode(
    model: Any,
    env: BTCUSDTTradingEnv,
    initial_observation: Float32Array,
) -> dict[str, Any]:
    observation = initial_observation
    recurrent_state: Any = None
    episode_start = np.asarray([True], dtype=np.bool_)
    trace: dict[str, Any] = {
        "reward_sum": 0.0,
        "steps": 0,
        "requested_size": _new_policy_size_metrics(),
        "action_mask_reason_counts": {},
    }
    while True:
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        action_vector, requested_action, requested_size = (
            _decode_discrete_policy_action(action)
        )
        _record_policy_size(trace["requested_size"], requested_action, requested_size)
        observation, reward, terminated, truncated, info = env.step(action_vector)
        reason = info.get("action_mask_reason")
        if isinstance(reason, str):
            reasons = trace["action_mask_reason_counts"]
            reasons[reason] = reasons.get(reason, 0) + 1
        trace["reward_sum"] += reward
        trace["steps"] += 1
        episode_start = np.asarray([terminated or truncated], dtype=np.bool_)
        if terminated or truncated:
            return trace


def _build_policy_evaluation_result(
    env: BTCUSDTTradingEnv, trace: dict[str, Any], seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = env.summary()
    sizes = trace["requested_size"]
    trade_count = int(sizes["trade_count"])
    result.update(
        {
            "steps": int(trace["steps"]),
            "reward_sum": float(trace["reward_sum"]),
            "evaluation_seed": seed,
            "evaluation_passes": 1,
            "deterministic_inference": True,
            "policy_action_encoding": "MultiDiscrete([direction, size_index])",
            "policy_requested_trade_size_fraction_counts": sizes["counts"],
            "policy_requested_trade_size_fraction_mean": (
                float(sizes["sum"]) / trade_count if trade_count else None
            ),
            "policy_requested_trade_size_fraction_min": (
                float(sizes["min"]) if trade_count else None
            ),
            "policy_requested_trade_size_fraction_max": (
                float(sizes["max"]) if trade_count else None
            ),
            "policy_requested_trade_size_action_count": trade_count,
            "policy_buy_zero_size_request_count": 0,
            "policy_action_mask_reason_counts": dict(
                sorted(trace["action_mask_reason_counts"].items())
            ),
        }
    )
    return result


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
        initial_equity=DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
    )
    try:
        observation, _ = env.reset(seed=seed)
        trace = _run_policy_evaluation_episode(model, env, observation)
        return _build_policy_evaluation_result(env, trace, seed)
    finally:
        env.close()


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
        initial_equity=DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        random_start=False,
    )
    env.reset(seed=20261025)
    steps = 0
    while True:
        if baseline == "buy_and_hold" and steps == 0:
            action = np.asarray(
                [ACTION_BUY, len(POSITION_SIZE_FRACTIONS) - 1], dtype=np.int64
            )
        else:
            action = np.asarray([ACTION_HOLD, 0], dtype=np.int64)
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
            "This assessment is descriptive and never authorizes trading. The "
            "curriculum's 100 completed-policy-trade threshold only triggers the "
            "cost ramp; it is not itself evidence of economic success."
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
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--commit-sha", default="unknown")
    return parser.parse_args(argv)


def _validate_cli_arguments(args: argparse.Namespace) -> None:
    if args.timesteps < 0:
        raise ValueError("timesteps_must_be_non_negative")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("ent_coef_must_be_finite_and_non_negative")
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
    entropy_coefficient: float
    cost_curriculum: dict[str, Any]


def _build_cost_curriculum_callback(curriculum: TrainingCostCurriculum) -> Any:
    """Create an SB3 callback lazily so evaluation-only imports stay lightweight."""
    base_callback = getattr(
        importlib.import_module("stable_baselines3.common.callbacks"), "BaseCallback"
    )

    def initialize_callback(self: Any) -> None:
        base_callback.__init__(self, verbose=0)
        self._last_costs = (0.0, 0.0)

    def on_training_step(self: Any) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, list) or len(infos) != 1:
            raise RuntimeError("cost_curriculum_requires_single_vectorized_environment")
        info = infos[0]
        if not isinstance(info, dict):
            raise RuntimeError("cost_curriculum_step_info_missing")
        closed_trades = info.get("policy_closed_trade_count")
        if (
            isinstance(closed_trades, bool)
            or not isinstance(closed_trades, int)
            or closed_trades < 0
        ):
            raise RuntimeError("policy_closed_trade_count_missing_or_invalid")
        next_fees, next_slippage = curriculum.observe_step(
            int(self.num_timesteps), closed_trades
        )
        next_costs = (next_fees, next_slippage)
        if next_costs != self._last_costs:
            updates = self.training_env.env_method(
                "set_transaction_costs", next_fees, next_slippage
            )
            if len(updates) != 1 or updates[0] is not None:
                raise RuntimeError("cost_curriculum_environment_update_failed")
            self._last_costs = next_costs
        return True

    callback_type = type(
        "CostCurriculumCallback",
        (base_callback,),
        {"__init__": initialize_callback, "_on_step": on_training_step},
    )
    return callback_type()


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
    curriculum = TrainingCostCurriculum(
        target_fee_bps=args.fee_bps,
        target_slippage_bps=args.slippage_bps,
        minimum_closed_trades=CURRICULUM_MINIMUM_CLOSED_TRADES,
        ramp_steps=CURRICULUM_RAMP_STEPS,
        ramp_update_interval_steps=CURRICULUM_RAMP_UPDATE_INTERVAL_STEPS,
    )
    envs = dummy_vec_env([_env_factory(train_data, scaler, 0.0, 0.0)])
    callback = _build_cost_curriculum_callback(curriculum)
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
        ent_coef=args.ent_coef,
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
            callback=callback,
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
        entropy_coefficient=float(args.ent_coef),
        cost_curriculum=curriculum.report(int(loaded_model.num_timesteps)),
    )


def _evaluate_validation_and_holdout(
    validation_data: Float64Array,
    holdout_data: Float64Array,
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

    # The latest tail is passed explicitly and evaluated exactly once.
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
    pre_cutoff_data, _ = _exclude_previous_holdout_window(data)
    train_data, validation_data, validation_end = _split_training_validation(
        pre_cutoff_data
    )
    # Keep the previously used final-holdout segment out of both fitting and
    # validation; use the chronologically latest tail as this model's final test.
    new_holdout_data = data[len(pre_cutoff_data) :]
    excluded_prior_candidate_holdout = pre_cutoff_data[validation_end:]
    if len(new_holdout_data) < 100_000 or len(excluded_prior_candidate_holdout) == 0:
        raise ValueError("new_holdout_or_prior_holdout_gap_too_small")
    scaler = MarketObservationScaler.fit(train_data)
    n_steps = 1_024
    requested_steps, training_steps = _training_budget(
        len(train_data), args.timesteps, n_steps
    )
    training = _train_recurrent_ppo(
        train_data, scaler, args, requested_steps, training_steps, n_steps
    )
    evaluation = _evaluate_validation_and_holdout(
        validation_data, new_holdout_data, scaler, args, training.model
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
        training,
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
    training: TrainingRun,
) -> dict[str, Any]:
    economic_assessment = _economic_assessment(validation_metrics, holdout_metrics)
    curriculum = training.cost_curriculum
    minimum_trade_threshold_met = bool(curriculum["minimum_trade_threshold_met"])
    ramp_completed = bool(curriculum["ramp_completed"])
    economic_assessment["criteria"]["minimum_trade_threshold_met"] = (
        minimum_trade_threshold_met
    )
    economic_assessment["criteria"]["training_cost_ramp_completed"] = ramp_completed
    economic_assessment["criteria"]["training_cost_curriculum_completed"] = (
        minimum_trade_threshold_met and ramp_completed
    )
    if not minimum_trade_threshold_met:
        economic_assessment["reasons"].append("minimum_trade_threshold_not_reached")
    if not ramp_completed:
        economic_assessment["reasons"].append("training_cost_ramp_incomplete")
    if not minimum_trade_threshold_met or not ramp_completed:
        economic_assessment["status"] = "NO-GO"

    report: dict[str, Any] = {
        "schema_version": "alix.rl-trading-agent-report.v4",
        "run_mode": "historical_simulation_only",
        "commit_sha": args.commit_sha,
        "research_result": (
            "CANDIDATE_REQUIRES_OWNER_REVIEW"
            if economic_assessment["status"] == "CANDIDATE_REQUIRES_OWNER_REVIEW"
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
            "method": "chronological_train_validation_with_prior-holdout-gap_and_latest-tail_test",
            "previous_run_candidate_holdout_start_utc": PREVIOUS_HOLDOUT_START_UTC,
            "previous_run_candidate_holdout_rows_excluded_from_current_fit": (
                int(manifest["row_count"])
                - len(train_data)
                - len(validation_data)
                - len(holdout_data)
            ),
            "latest_tail_used_as_current_final_holdout": True,
            "final_holdout_exposed_in_prior_research": True,
            "final_holdout_disjoint_from_current_train_validation": (
                float(holdout_data[0, 0]) > float(validation_data[-1, 0])
            ),
            "analysis_window_end_utc": _utc_from_ms(float(validation_data[-1, 0])),
            "analysis_window_rows": (
                len(train_data)
                + len(validation_data)
                + int(manifest["row_count"])
                - len(train_data)
                - len(validation_data)
                - len(holdout_data)
            ),
            "rows_excluded_between_validation_and_final_holdout": (
                int(manifest["row_count"])
                - len(train_data)
                - len(validation_data)
                - len(holdout_data)
            ),
            "holdout_novelty_note": (
                "The final holdout is the latest public-data tail beginning 2026-06-01 "
                "UTC. It is untouched by this model's training and validation, but this "
                "same calendar interval was inspected in prior research; it is not a "
                "new-to-research market sample."
            ),
            "fraction_denominators": (
                "training and validation fractions are measured against the pre-cutoff "
                "window; final_holdout_fraction is the target fraction of the full dataset. "
                "The previous candidate's held-out segment between validation and the "
                "latest tail is deliberately excluded from this model's fitting and scoring."
            ),
            "training_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "final_holdout_fraction": HOLDOUT_FRACTION,
            "final_holdout_fraction_actual": len(holdout_data)
            / int(manifest["row_count"]),
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
                "type": "MultiDiscrete",
                "nvec": [len(ACTION_NAMES), len(POSITION_SIZE_FRACTIONS)],
                "direction_index_to_action": {
                    str(index): name for index, name in enumerate(ACTION_NAMES)
                },
                "size_index_to_fraction": {
                    str(index): fraction
                    for index, fraction in enumerate(POSITION_SIZE_FRACTIONS)
                },
                "buy_semantics": (
                    "spend the selected fraction of currently available cash, "
                    "inclusive of fee and slippage costs"
                ),
                "sell_semantics": "sell the selected fraction of current inventory",
                "hold_semantics": "do not rebalance; size index is ignored",
                "zero_size_buy_allowed": False,
                "masking": {
                    "enforcement": "deterministic_environment_execution_boundary",
                    "policy_distribution_logits_masked": False,
                    "sell_requires_positive_inventory": True,
                    "buy_requires_unallocated_cash": True,
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
                "invalid_action_penalty_per_masked_request": DEFAULT_INVALID_ACTION_PENALTY,
                "inactivity_grace_steps": DEFAULT_INACTIVITY_GRACE_STEPS,
                "profitable_trade_bonus": DEFAULT_PROFITABLE_TRADE_BONUS,
                "opportunity_cost_penalty_when_flat_above_sma": (
                    DEFAULT_OPPORTUNITY_COST_PENALTY
                ),
                "opportunity_cost_moving_average_window": MOVING_AVERAGE_WINDOW,
                "opportunity_cost_indicator_visible_to_policy": False,
                "opportunity_cost_penalty_trigger": (
                    "HOLD while flat and current closed price is above the causal "
                    "moving average computed from the last completed candles"
                ),
                "profitable_trade_bonus_trigger": (
                    "completed round trip with cumulative realized PnL positive "
                    "after entry and exit fees and slippage"
                ),
                "invalid_action_penalty_trigger": (
                    "requested BUY or SELL disallowed by the pre-action legal mask"
                ),
                "formula": (
                    "log_net_return - 5.0 * max(0, risk_drawdown - prior_close_drawdown) "
                    "- 0.01 * risk_drawdown "
                    "- 0.001 * indicator(consecutive_unchanged_steps > 3) "
                    "- 0.001 * indicator(masked_BUY_or_SELL_request) "
                    "- 0.0005 * indicator(flat_HOLD_and_close_above_causal_SMA) "
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
            "torch_version": training.torch_version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "device": "cpu",
            "seed": args.seed,
            "requested_timesteps": training.requested_timesteps,
            "total_timesteps_trained": int(training.model.num_timesteps),
            "ent_coef": training.entropy_coefficient,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "training_rows": len(train_data),
            "first_pass_training_coverage_fraction": min(
                1.0, float(training.model.num_timesteps) / (len(train_data) - 1)
            ),
            "estimated_full_pass_equivalents": float(
                training.model.num_timesteps / (len(train_data) - 1)
            ),
            "training_replays_training_split": training.model.num_timesteps
            > len(train_data) - 1,
            "n_envs": 1,
            "n_steps": training.n_steps,
            "batch_size": 256,
            "n_epochs": 4,
            "epochs_per_rollout": 4,
            "lstm_hidden_size": 64,
            "training_seconds": float(training.training_seconds),
            "model_artifact": training.model_path.name,
            "model_sha256": _sha256(training.model_path),
            "model_reload_roundtrip": "passed",
            "no_validation_checkpoint_selection": True,
        },
        "cost_curriculum": training.cost_curriculum,
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
        "economic_assessment": economic_assessment,
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
