"""Train and evaluate the state-aware recurrent PPO agent offline."""

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
    ACTION_BUY_100,
    ACTION_HOLD,
    ACTION_NAMES,
    ACTION_SIDE_BY_INDEX,
    ACTION_SIZE_BY_INDEX,
    BTCUSDTTradingEnv,
    DEFAULT_INITIAL_EQUITY,
    DEFAULT_HOLDING_LOSS_PENALTY,
    DEFAULT_HOLDING_PROFIT_REWARD,
    DEFAULT_LOW_VOLATILITY_HOLD_REWARD,
    DEFAULT_OPPORTUNITY_COST_PENALTY,
    DEFAULT_REALIZED_PROFIT_REWARD_SCALE,
    MOVING_AVERAGE_WINDOW,
    POSITION_SIZE_FRACTIONS,
    DEFAULT_INACTIVITY_PENALTY,
    DEFAULT_INVALID_ACTION_PENALTY,
    DEFAULT_PROFITABLE_TRADE_BONUS,
    DEFAULT_INACTIVITY_GRACE_STEPS,
    DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER,
    LOW_VOLATILITY_LOOKBACK,
    LOW_VOLATILITY_RETURN_STD_THRESHOLD,
    MarketObservationScaler,
    OBSERVATION_FIELDS,
    validate_market_data,
)
from scripts.rl_auxiliary_ppo import (
    ACTION_MASK_POSITION_EPSILON,
    DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
    DEFAULT_AUXILIARY_LOSS_MINIMUM_WEIGHT,
    DEFAULT_AUXILIARY_LOSS_WEIGHT,
    auxiliary_loss_curriculum_report,
    build_auxiliary_recurrent_ppo,
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
DEFAULT_SIGNAL_GUIDANCE_REWARD_SCALE = 0.0002
DEFAULT_SIGNAL_GUIDANCE_REWARD_FLOOR = 0.00002
DEFAULT_SIGNAL_GUIDANCE_ANNEAL_STEPS = 300_000
DEFAULT_SIGNAL_GUIDANCE_UPDATE_INTERVAL_STEPS = 1_000
TRAINING_CHECKPOINT_INTERVAL_STEPS = 100_000
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
POLICY_NAME = "ActionMaskedRecurrentActorCriticPolicy"


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
    signal_guidance_reward_scale: float = 0.0,
) -> Callable[[], BTCUSDTTradingEnv]:
    def create() -> BTCUSDTTradingEnv:
        return BTCUSDTTradingEnv(
            data,
            scaler=scaler,
            initial_equity=DEFAULT_INITIAL_EQUITY,
            fee_bps_per_side=fee_bps,
            slippage_bps_per_side=slippage_bps,
            random_start=False,
            signal_guidance_reward_scale=signal_guidance_reward_scale,
        )

    return create


def _decode_discrete_policy_action(action: Any) -> tuple[np.int64, str, float]:
    values = np.asarray(action)
    if values.shape not in ((), (1,)) or values.dtype.kind not in ("i", "u"):
        raise ValueError("policy_action_must_be_one_integer_value")
    action_index = int(values.reshape(-1)[0]) if values.shape else int(values.item())
    if not 0 <= action_index < len(ACTION_NAMES):
        raise ValueError("policy_discrete_action_index_out_of_range")
    return (
        np.int64(action_index),
        ACTION_SIDE_BY_INDEX[action_index],
        ACTION_SIZE_BY_INDEX[action_index],
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
            "policy_action_encoding": "Discrete(7) combined direction-size action",
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
            action = np.asarray(ACTION_BUY_100, dtype=np.int64)
        else:
            action = np.asarray(ACTION_HOLD, dtype=np.int64)
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
    parser.add_argument("--ent-coef", type=float, default=0.1)
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
    signal_guidance_curriculum: dict[str, Any]
    auxiliary_loss_curriculum: dict[str, Any]
    reward_component_windows: list[dict[str, Any]]
    checkpoint_paths: tuple[Path, ...]


def _guidance_reward_scale_at_step(
    timestep: int,
    initial_scale: float = DEFAULT_SIGNAL_GUIDANCE_REWARD_SCALE,
    anneal_steps: int = DEFAULT_SIGNAL_GUIDANCE_ANNEAL_STEPS,
    minimum_scale: float = DEFAULT_SIGNAL_GUIDANCE_REWARD_FLOOR,
) -> float:
    """Linearly anneal the train-only SMA guidance reward to a nonzero floor."""
    if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
        raise ValueError("signal_guidance_timestep_invalid")
    if (
        isinstance(anneal_steps, bool)
        or not isinstance(anneal_steps, int)
        or anneal_steps < 1
    ):
        raise ValueError("signal_guidance_anneal_steps_invalid")
    if not math.isfinite(initial_scale) or initial_scale < 0.0:
        raise ValueError("signal_guidance_initial_scale_invalid")
    if not math.isfinite(minimum_scale) or minimum_scale < 0.0:
        raise ValueError("signal_guidance_floor_invalid")
    if minimum_scale > initial_scale:
        raise ValueError("signal_guidance_floor_exceeds_initial_scale")
    return float(
        max(minimum_scale, initial_scale * max(0.0, 1.0 - timestep / anneal_steps))
    )


def _build_cost_curriculum_callback(
    curriculum: TrainingCostCurriculum,
    initial_guidance_scale: float = DEFAULT_SIGNAL_GUIDANCE_REWARD_SCALE,
    guidance_anneal_steps: int = DEFAULT_SIGNAL_GUIDANCE_ANNEAL_STEPS,
    guidance_update_interval_steps: int = DEFAULT_SIGNAL_GUIDANCE_UPDATE_INTERVAL_STEPS,
    guidance_reward_floor: float = DEFAULT_SIGNAL_GUIDANCE_REWARD_FLOOR,
) -> Any:
    """Update costs and anneal a training-only causal trend hint to a floor."""
    if (
        not math.isfinite(initial_guidance_scale)
        or initial_guidance_scale < 0.0
        or not math.isfinite(guidance_reward_floor)
        or guidance_reward_floor < 0.0
        or guidance_reward_floor > initial_guidance_scale
        or isinstance(guidance_anneal_steps, bool)
        or guidance_anneal_steps < 1
        or isinstance(guidance_update_interval_steps, bool)
        or guidance_update_interval_steps < 1
        or guidance_anneal_steps % guidance_update_interval_steps != 0
    ):
        raise ValueError("signal_guidance_curriculum_configuration_invalid")
    base_callback = getattr(
        importlib.import_module("stable_baselines3.common.callbacks"), "BaseCallback"
    )

    def initialize_callback(self: Any) -> None:
        base_callback.__init__(self, verbose=0)
        self._last_costs = (0.0, 0.0)
        self._initial_guidance_scale = float(initial_guidance_scale)
        self._guidance_anneal_steps = guidance_anneal_steps
        self._guidance_update_interval_steps = guidance_update_interval_steps
        self._guidance_reward_floor = float(guidance_reward_floor)
        self._last_guidance_scale = float(initial_guidance_scale)
        self._guidance_reward_count = 0
        self._guidance_reward_total = 0.0
        self._guidance_buy_reward_count = 0
        self._guidance_sell_reward_count = 0
        self._reward_component_windows = {}

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
        timestep = int(self.num_timesteps)
        self._record_reward_component_step(timestep, info)
        next_fees, next_slippage = curriculum.observe_step(timestep, closed_trades)
        next_costs = (next_fees, next_slippage)
        if next_costs != self._last_costs:
            updates = self.training_env.env_method(
                "set_transaction_costs", next_fees, next_slippage
            )
            if len(updates) != 1 or updates[0] is not None:
                raise RuntimeError("cost_curriculum_environment_update_failed")
            self._last_costs = next_costs

        raw_guidance_reward = info.get("reward_signal_guidance", 0.0)
        if (
            isinstance(raw_guidance_reward, bool)
            or not isinstance(raw_guidance_reward, (int, float))
            or not math.isfinite(float(raw_guidance_reward))
            or raw_guidance_reward < 0.0
        ):
            raise RuntimeError("signal_guidance_reward_missing_or_invalid")
        if raw_guidance_reward > 0.0:
            guidance_action = info.get("signal_guidance_action")
            if guidance_action not in ("BUY", "SELL"):
                raise RuntimeError("signal_guidance_action_missing_or_invalid")
            self._guidance_reward_count += 1
            self._guidance_reward_total += float(raw_guidance_reward)
            if guidance_action == "BUY":
                self._guidance_buy_reward_count += 1
            else:
                self._guidance_sell_reward_count += 1

        if (
            timestep % self._guidance_update_interval_steps == 0
            or timestep >= self._guidance_anneal_steps
        ):
            next_guidance_scale = _guidance_reward_scale_at_step(
                timestep,
                self._initial_guidance_scale,
                self._guidance_anneal_steps,
                self._guidance_reward_floor,
            )
            if next_guidance_scale != self._last_guidance_scale:
                updates = self.training_env.env_method(
                    "set_signal_guidance_reward_scale", next_guidance_scale
                )
                if len(updates) != 1 or updates[0] is not None:
                    raise RuntimeError("signal_guidance_environment_update_failed")
                self._last_guidance_scale = next_guidance_scale
        return True

    def signal_guidance_report(self: Any) -> dict[str, Any]:
        timesteps = int(self.num_timesteps)
        return {
            "initial_reward_scale": self._initial_guidance_scale,
            "anneal_steps": self._guidance_anneal_steps,
            "update_interval_steps": self._guidance_update_interval_steps,
            "minimum_reward_scale": self._guidance_reward_floor,
            "final_reward_scale": self._last_guidance_scale,
            "floor_reached": math.isclose(
                self._last_guidance_scale, self._guidance_reward_floor, abs_tol=1e-12
            ),
            "anneal_completed": timesteps >= self._guidance_anneal_steps,
            "rewarded_action_count": self._guidance_reward_count,
            "reward_total": self._guidance_reward_total,
            "rewarded_buy_action_count": self._guidance_buy_reward_count,
            "rewarded_sell_action_count": self._guidance_sell_reward_count,
            "evaluation_enabled": False,
            "signal_definition": (
                "causal slope of the 20-completed-candle simple moving average; "
                "reward only a legal executed BUY when slope is positive or a legal "
                "executed SELL when slope is negative; no opposing-action penalty"
            ),
        }

    def record_reward_component_step(
        self: Any, timestep: int, info: dict[str, Any]
    ) -> None:
        window_index = (timestep - 1) // TRAINING_CHECKPOINT_INTERVAL_STEPS
        window = self._reward_component_windows.get(window_index)
        if window is None:
            reward_keys = (
                "reward_net_log_return",
                "reward_drawdown_increment_penalty",
                "reward_drawdown_level_penalty",
                "reward_inactivity_penalty",
                "reward_invalid_action_penalty",
                "reward_full_position_buy_penalty",
                "reward_opportunity_cost_penalty",
                "reward_holding_loss_penalty",
                "reward_holding_profit_bonus",
                "reward_low_volatility_hold_bonus",
                "reward_transaction_cost_penalty",
                "reward_signal_guidance",
                "reward_realized_profit",
                "reward_profitable_trade_bonus",
                "reward_total",
            )
            window = {
                "start_timestep": window_index * TRAINING_CHECKPOINT_INTERVAL_STEPS + 1,
                "end_timestep": timestep,
                "steps": 0,
                "reward_component_sums": {key: 0.0 for key in reward_keys},
                "transaction_cost_sums": {
                    "fees": 0.0,
                    "slippage": 0.0,
                    "combined_fees_and_slippage": 0.0,
                },
                "requested_action_counts": {},
                "executed_action_counts": {},
            }
            self._reward_component_windows[window_index] = window
        window["end_timestep"] = timestep
        window["steps"] += 1
        for key in window["reward_component_sums"]:
            raw_value = info.get(key)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise RuntimeError(
                    f"training_reward_component_missing_or_invalid:{key}"
                )
            window["reward_component_sums"][key] += float(raw_value)
        for info_key, summary_key in (
            ("fees_this_step", "fees"),
            ("slippage_this_step", "slippage"),
            ("transaction_costs_this_step", "combined_fees_and_slippage"),
        ):
            raw_value = info.get(info_key)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise RuntimeError(
                    f"training_cost_component_missing_or_invalid:{info_key}"
                )
            window["transaction_cost_sums"][summary_key] += float(raw_value)
        for info_key, summary_key in (
            ("requested_discrete_action_name", "requested_action_counts"),
            ("executed_action", "executed_action_counts"),
        ):
            action = info.get(info_key)
            if not isinstance(action, str) or not action:
                raise RuntimeError(f"training_action_telemetry_missing:{info_key}")
            counts = window[summary_key]
            counts[action] = counts.get(action, 0) + 1

    def reward_component_report(self: Any) -> list[dict[str, Any]]:
        return [
            self._reward_component_windows[index]
            for index in sorted(self._reward_component_windows)
        ]

    callback_type = type(
        "CostAndSignalGuidanceCurriculumCallback",
        (base_callback,),
        {
            "__init__": initialize_callback,
            "_on_step": on_training_step,
            "signal_guidance_report": signal_guidance_report,
            "_record_reward_component_step": record_reward_component_step,
            "reward_component_report": reward_component_report,
        },
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
    recurrent_ppo = build_auxiliary_recurrent_ppo()
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
    envs = dummy_vec_env(
        [
            _env_factory(
                train_data,
                scaler,
                0.0,
                0.0,
                signal_guidance_reward_scale=DEFAULT_SIGNAL_GUIDANCE_REWARD_SCALE,
            )
        ]
    )
    curriculum_callback = _build_cost_curriculum_callback(curriculum)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback_module = importlib.import_module("stable_baselines3.common.callbacks")
    checkpoint_callback = getattr(callback_module, "CheckpointCallback")(
        save_freq=TRAINING_CHECKPOINT_INTERVAL_STEPS,
        save_path=str(checkpoint_dir),
        name_prefix="rl_trading_agent_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )
    callback = getattr(callback_module, "CallbackList")(
        [curriculum_callback, checkpoint_callback]
    )
    envs.seed(args.seed)
    policy_class = getattr(recurrent_ppo, "policy_class", None)
    if not isinstance(policy_class, type):
        raise RuntimeError("action_masked_recurrent_policy_class_missing")
    model = recurrent_ppo(
        policy_class,
        envs,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        auxiliary_loss_weight=DEFAULT_AUXILIARY_LOSS_WEIGHT,
        auxiliary_loss_anneal_steps=DEFAULT_AUXILIARY_LOSS_ANNEAL_STEPS,
        auxiliary_loss_minimum_weight=DEFAULT_AUXILIARY_LOSS_MINIMUM_WEIGHT,
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
    checkpoint_paths = tuple(
        sorted(
            checkpoint_dir.glob("rl_trading_agent_checkpoint_*_steps.zip"),
            key=lambda path: int(
                path.name.removeprefix("rl_trading_agent_checkpoint_").removesuffix(
                    "_steps.zip"
                )
            ),
        )
    )
    expected_checkpoint_steps = list(
        range(
            TRAINING_CHECKPOINT_INTERVAL_STEPS,
            (training_steps // TRAINING_CHECKPOINT_INTERVAL_STEPS)
            * TRAINING_CHECKPOINT_INTERVAL_STEPS
            + 1,
            TRAINING_CHECKPOINT_INTERVAL_STEPS,
        )
    )
    checkpoint_steps = [
        int(
            path.name.removeprefix("rl_trading_agent_checkpoint_").removesuffix(
                "_steps.zip"
            )
        )
        for path in checkpoint_paths
    ]
    if checkpoint_steps != expected_checkpoint_steps:
        raise RuntimeError("training_checkpoint_inventory_mismatch")
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
        signal_guidance_curriculum=curriculum_callback.signal_guidance_report(),
        auxiliary_loss_curriculum=auxiliary_loss_curriculum_report(
            int(loaded_model.num_timesteps),
            float(loaded_model.auxiliary_loss_weight),
            int(loaded_model.auxiliary_loss_anneal_steps),
            float(loaded_model.auxiliary_loss_minimum_weight),
        ),
        reward_component_windows=curriculum_callback.reward_component_report(),
        checkpoint_paths=checkpoint_paths,
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
        "schema_version": "alix.rl-trading-agent-report.v11",
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
            "observation_contains_position_or_pnl": True,
            "observation_contains_technical_indicators": False,
            "observation_semantics": {
                "current_position_fraction": (
                    "marked BTC position value divided by total marked account equity, "
                    "clamped to [0, 1]"
                ),
                "unrealized_pnl_pct": (
                    "decimal return on remaining position cost basis, including entry costs; "
                    "0.05 means +5%, and flat positions report 0"
                ),
            },
            "action_space": {
                "type": "Discrete",
                "n": len(ACTION_NAMES),
                "index_to_action": {
                    str(index): name for index, name in enumerate(ACTION_NAMES)
                },
                "size_fraction_by_action": {
                    str(index): size
                    for index, size in enumerate(ACTION_SIZE_BY_INDEX)
                    if size > 0.0
                },
                "buy_semantics": (
                    "spend the action's fixed fraction of current available cash, "
                    "including simulated fee and slippage costs"
                ),
                "sell_semantics": "sell the action's fixed fraction of current inventory",
                "hold_semantics": "do not rebalance",
                "zero_size_trade_actions_allowed": False,
                "masking": {
                    "enforcement": "custom_recurrent_policy_raw_logits_before_categorical_distribution_and_environment_execution_boundary",
                    "policy_distribution_logits_masked": True,
                    "mask_applied_before_distribution_construction": True,
                    "masked_logit_value": -1e8,
                    "forbidden_action_probability_exactly_zero": True,
                    "buy_action_indices": [1, 2, 3],
                    "sell_action_indices": [4, 5, 6],
                    "hold_action_index": 0,
                    "position_fraction_full_threshold": 1.0
                    - ACTION_MASK_POSITION_EPSILON,
                    "position_fraction_flat_threshold": ACTION_MASK_POSITION_EPSILON,
                    "buy_actions_masked_when_position_fraction_at_or_above_full_threshold": True,
                    "sell_actions_masked_when_position_fraction_at_or_below_flat_threshold": True,
                    "masked_action_is_non_executable": True,
                    "invalid_buy_fallback_reward_penalty": 1.0,
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
                "invalid_action_penalty_default_per_masked_request": DEFAULT_INVALID_ACTION_PENALTY,
                "invalid_full_position_buy_penalty": 1.0,
                "invalid_full_position_buy_penalty_trigger": (
                    "fallback penalty when a BUY request is blocked because the "
                    "pre-action portfolio has no legal BUY capacity"
                ),
                "holding_profit_reward_per_profitable_hold": (
                    DEFAULT_HOLDING_PROFIT_REWARD
                ),
                "low_volatility_hold_reward": DEFAULT_LOW_VOLATILITY_HOLD_REWARD,
                "low_volatility_lookback_completed_candles": LOW_VOLATILITY_LOOKBACK,
                "low_volatility_return_std_threshold": LOW_VOLATILITY_RETURN_STD_THRESHOLD,
                "low_volatility_hold_reward_applies_only_to_HOLD": True,
                "transaction_cost_reward_multiplier": DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER,
                "transaction_cost_reward_penalty": (
                    "multiplier * (this_step_fee_cost + this_step_slippage_cost) "
                    "/ previous_equity; additional shaping alongside net log-equity return"
                ),
                "holding_loss_penalty_per_underwater_hold": (
                    DEFAULT_HOLDING_LOSS_PENALTY
                ),
                "holding_profit_reward_trigger": (
                    "HOLD while an open position's expected net liquidation value after "
                    "exit slippage and fees is above its position cost basis"
                ),
                "training_signal_guidance": {
                    "enabled_only_during_training": True,
                    "initial_reward_scale": DEFAULT_SIGNAL_GUIDANCE_REWARD_SCALE,
                    "minimum_reward_scale": DEFAULT_SIGNAL_GUIDANCE_REWARD_FLOOR,
                    "anneal_steps": DEFAULT_SIGNAL_GUIDANCE_ANNEAL_STEPS,
                    "update_interval_steps": DEFAULT_SIGNAL_GUIDANCE_UPDATE_INTERVAL_STEPS,
                    "final_reward_scale": DEFAULT_SIGNAL_GUIDANCE_REWARD_FLOOR,
                    "validation_enabled": False,
                    "final_holdout_enabled": False,
                    "signal_definition": (
                        "causal slope of the 20-completed-candle simple moving average"
                    ),
                },
                "realized_profit_reward_scale_of_initial_equity": (
                    DEFAULT_REALIZED_PROFIT_REWARD_SCALE
                ),
                "realized_profit_reward_trigger": (
                    "positive net profit from a policy-selected SELL that fully closes "
                    "the position; terminal forced liquidation earns no realized-profit reward"
                ),
                "holding_loss_penalty_trigger": (
                    "HOLD while expected net liquidation value after exit slippage and fees "
                    "is below the current position cost basis"
                ),
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
                    "- 1.0 * indicator(masked_BUY_request) "
                    "- 0.001 * indicator(masked_SELL_request) "
                    "- 0.0001 * indicator(HOLD_while_net_liquidation_is_below_cost_basis) "
                    "+ 0.0001 * indicator(HOLD_while_net_liquidation_is_above_cost_basis) "
                    "- 0.0005 * indicator(flat_HOLD_and_close_above_causal_SMA) "
                    "+ max(minimum_guidance_floor, annealed_training_only_scale) "
                    "* indicator(legal_BUY_when_causal_SMA_rises "
                    "or legal_SELL_when_causal_SMA_falls) "
                    "+ 0.5 * max(0, closed_trade_net_profit / initial_equity) "
                    " * indicator(policy_SELL_fully_closes_trade) "
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
            "training_algorithm": "RecurrentPPO with LSTM, Discrete(7) combined actions, pre-Categorical logit masking, floor-preserved seven-class guidance cross-entropy, minimum training-only SMA reward floor, elevated entropy coefficient, periodic checkpoints, low-volatility HOLD reward, and explicit transaction-cost reward penalty",
            "training_seconds": float(training.training_seconds),
            "model_artifact": training.model_path.name,
            "model_sha256": _sha256(training.model_path),
            "model_reload_roundtrip": "passed",
            "no_validation_checkpoint_selection": True,
        },
        "cost_curriculum": training.cost_curriculum,
        "signal_guidance_curriculum": training.signal_guidance_curriculum,
        "auxiliary_loss_curriculum": training.auxiliary_loss_curriculum,
        "training_reward_dynamics": {
            "window_size_steps": TRAINING_CHECKPOINT_INTERVAL_STEPS,
            "scope": "training_only; env.info telemetry; no validation or holdout rewards used",
            "component_windows": training.reward_component_windows,
        },
        "checkpointing": {
            "enabled": True,
            "interval_steps": TRAINING_CHECKPOINT_INTERVAL_STEPS,
            "directory": "checkpoints",
            "checkpoint_count": len(training.checkpoint_paths),
            "checkpoints": [
                {
                    "file": f"checkpoints/{path.name}",
                    "timesteps": int(
                        path.name.removeprefix(
                            "rl_trading_agent_checkpoint_"
                        ).removesuffix("_steps.zip")
                    ),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in training.checkpoint_paths
            ],
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
