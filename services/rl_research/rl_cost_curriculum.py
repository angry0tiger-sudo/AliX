"""Fixed transaction costs applied from the first offline RL training step."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _validate_cost(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value < 10_000.0:
        raise ValueError(f"{name}_out_of_range")
    return numeric_value


def _validate_observation_timestep(timestep: int, last_timestep: int) -> int:
    if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 1:
        raise ValueError("training_timesteps_must_be_contiguous_positive_integers")
    if last_timestep and timestep != last_timestep + 1:
        raise ValueError("training_timesteps_must_be_contiguous_positive_integers")
    return timestep


def _validate_episode_trade_count(episode_closed_trades: int) -> int:
    if (
        isinstance(episode_closed_trades, bool)
        or not isinstance(episode_closed_trades, int)
        or episode_closed_trades < 0
    ):
        raise ValueError("episode_closed_trade_count_must_be_non_negative_integer")
    return episode_closed_trades


def _episode_trade_delta(current_count: int, previous_count: int) -> int:
    if current_count < previous_count:
        return current_count
    return current_count - previous_count


@dataclass(slots=True)
class TrainingTransactionCosts:
    """Hold configured per-side costs constant throughout training."""

    target_fee_bps: float
    target_slippage_bps: float
    _total_policy_closed_trades: int = field(default=0, init=False)
    _last_episode_trade_count: int = field(default=0, init=False)
    _last_timestep: int = field(default=0, init=False)
    _steps_at_full_cost: int = field(default=0, init=False)
    _closed_trades_at_full_cost: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.target_fee_bps = _validate_cost("target_fee_bps", self.target_fee_bps)
        self.target_slippage_bps = _validate_cost(
            "target_slippage_bps", self.target_slippage_bps
        )

    def stage_for_timestep(self, timestep: int) -> str:
        """Return the only supported training stage: configured costs are active."""
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError("timestep_must_be_non_negative_integer")
        return "full_cost"

    def costs_for_timestep(self, timestep: int) -> tuple[float, float]:
        """Return configured costs for any zero-based or one-based training step."""
        self.stage_for_timestep(timestep)
        return self.target_fee_bps, self.target_slippage_bps

    def observe_step(
        self, timestep: int, episode_policy_closed_trades: int
    ) -> tuple[float, float]:
        """Record telemetry while keeping costs unchanged for the next step."""
        timestep = _validate_observation_timestep(timestep, self._last_timestep)
        episode_policy_closed_trades = _validate_episode_trade_count(
            episode_policy_closed_trades
        )
        trade_delta = _episode_trade_delta(
            episode_policy_closed_trades, self._last_episode_trade_count
        )
        self._steps_at_full_cost += 1
        self._closed_trades_at_full_cost += trade_delta
        self._total_policy_closed_trades += trade_delta
        self._last_episode_trade_count = episode_policy_closed_trades
        self._last_timestep = timestep
        return self.costs_for_timestep(timestep + 1)

    def report(self, training_timesteps: int) -> dict[str, object]:
        """Report evidence that no training steps used zero or ramped costs."""
        if (
            isinstance(training_timesteps, bool)
            or not isinstance(training_timesteps, int)
            or training_timesteps != self._last_timestep
        ):
            raise ValueError("training_cost_step_count_mismatch")
        full_cost_applied = (
            training_timesteps > 0 and self._steps_at_full_cost == training_timesteps
        )
        return {
            "scheme": "fixed_configured_costs_from_first_training_step",
            "initial_fee_bps_per_side": self.target_fee_bps,
            "initial_slippage_bps_per_side": self.target_slippage_bps,
            "target_fee_bps_per_side": self.target_fee_bps,
            "target_slippage_bps_per_side": self.target_slippage_bps,
            "zero_cost_steps": 0,
            "ramp_steps": 0,
            "training_timesteps": training_timesteps,
            "total_policy_closed_trade_count": self._total_policy_closed_trades,
            "initial_costs_applied_before_first_step": True,
            "all_training_steps_full_cost": full_cost_applied,
            "full_cost_steps": self._steps_at_full_cost,
            "steps_by_stage": {"full_cost": self._steps_at_full_cost},
            "closed_trades_by_stage": {"full_cost": self._closed_trades_at_full_cost},
            "last_training_step_cost_stage": (
                "full_cost" if training_timesteps else "not_started"
            ),
            "status": (
                "fixed_costs_verified"
                if full_cost_applied
                else "not_started_or_incomplete"
            ),
        }
