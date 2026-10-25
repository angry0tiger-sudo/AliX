"""Deterministic, accounting-based costs curriculum for offline RL training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CURRICULUM_MINIMUM_CLOSED_TRADES = 100
CURRICULUM_RAMP_STEPS = 100_000
CURRICULUM_RAMP_UPDATE_INTERVAL_STEPS = 1_000


def _validate_cost(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value < 10_000.0:
        raise ValueError(f"{name}_out_of_range")
    return numeric_value


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}_must_be_positive_integer")
    return value


def _validate_ramp_update_interval(ramp_steps: int, interval: int) -> int:
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("ramp_update_interval_must_divide_ramp_steps")
    if ramp_steps % interval != 0:
        raise ValueError("ramp_update_interval_must_divide_ramp_steps")
    return interval


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
class TrainingCostCurriculum:
    """Start cost-free, then ramp to configured costs after real round trips."""

    target_fee_bps: float
    target_slippage_bps: float
    minimum_closed_trades: int = CURRICULUM_MINIMUM_CLOSED_TRADES
    ramp_steps: int = CURRICULUM_RAMP_STEPS
    ramp_update_interval_steps: int = CURRICULUM_RAMP_UPDATE_INTERVAL_STEPS
    _ramp_start_timestep: int | None = field(default=None, init=False)
    _closed_trades_at_ramp_start: int | None = field(default=None, init=False)
    _total_policy_closed_trades: int = field(default=0, init=False)
    _last_episode_trade_count: int = field(default=0, init=False)
    _last_timestep: int = field(default=0, init=False)
    _steps_by_stage: dict[str, int] = field(
        default_factory=lambda: {"zero_cost": 0, "ramp": 0, "full_cost": 0},
        init=False,
    )
    _closed_trades_by_stage: dict[str, int] = field(
        default_factory=lambda: {"zero_cost": 0, "ramp": 0, "full_cost": 0},
        init=False,
    )

    def __post_init__(self) -> None:
        self.target_fee_bps = _validate_cost("target_fee_bps", self.target_fee_bps)
        self.target_slippage_bps = _validate_cost(
            "target_slippage_bps", self.target_slippage_bps
        )
        self.minimum_closed_trades = _validate_positive_integer(
            "minimum_closed_trades", self.minimum_closed_trades
        )
        self.ramp_steps = _validate_positive_integer("ramp_steps", self.ramp_steps)
        self.ramp_update_interval_steps = _validate_ramp_update_interval(
            self.ramp_steps, self.ramp_update_interval_steps
        )

    def stage_for_timestep(self, timestep: int) -> str:
        """Return the cost stage actually applied to a training decision."""
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError("timestep_must_be_non_negative_integer")
        if self._ramp_start_timestep is None or timestep < self._ramp_start_timestep:
            return "zero_cost"
        if timestep < self._ramp_start_timestep + self.ramp_steps:
            return "ramp"
        return "full_cost"

    def costs_for_timestep(self, timestep: int) -> tuple[float, float]:
        """Return costs for the specified one-based environment step."""
        stage = self.stage_for_timestep(timestep)
        if stage == "zero_cost":
            return 0.0, 0.0
        if stage == "full_cost":
            return self.target_fee_bps, self.target_slippage_bps
        assert self._ramp_start_timestep is not None
        elapsed_steps = timestep - self._ramp_start_timestep
        completed_ramp_steps = (
            elapsed_steps // self.ramp_update_interval_steps
        ) * self.ramp_update_interval_steps
        fraction = min(1.0, completed_ramp_steps / self.ramp_steps)
        return (
            self.target_fee_bps * fraction,
            self.target_slippage_bps * fraction,
        )

    def observe_step(
        self, timestep: int, episode_policy_closed_trades: int
    ) -> tuple[float, float]:
        """Record one training step and return costs for the next step.

        The episode counter can reset when the vectorized environment resets.
        The controller converts those episode-local counters into a monotonic
        aggregate and never counts forced terminal liquidations as policy trades.
        """
        timestep = _validate_observation_timestep(timestep, self._last_timestep)
        episode_policy_closed_trades = _validate_episode_trade_count(
            episode_policy_closed_trades
        )
        trade_delta = _episode_trade_delta(
            episode_policy_closed_trades, self._last_episode_trade_count
        )

        stage = self.stage_for_timestep(timestep)
        self._steps_by_stage[stage] += 1
        self._closed_trades_by_stage[stage] += trade_delta
        self._total_policy_closed_trades += trade_delta
        self._last_episode_trade_count = episode_policy_closed_trades
        self._last_timestep = timestep

        if (
            self._ramp_start_timestep is None
            and self._total_policy_closed_trades >= self.minimum_closed_trades
        ):
            self._ramp_start_timestep = timestep + 1
            self._closed_trades_at_ramp_start = self._total_policy_closed_trades

        return self.costs_for_timestep(timestep + 1)

    def report(self, training_timesteps: int) -> dict[str, object]:
        """Return sufficient evidence to audit an incomplete or completed ramp."""
        if (
            isinstance(training_timesteps, bool)
            or not isinstance(training_timesteps, int)
            or training_timesteps != self._last_timestep
        ):
            raise ValueError("curriculum_step_count_mismatch")
        full_cost_start = (
            self._ramp_start_timestep + self.ramp_steps
            if self._ramp_start_timestep is not None
            else None
        )
        ramp_completed = (
            full_cost_start is not None and training_timesteps >= full_cost_start
        )
        if self._ramp_start_timestep is None:
            status = "minimum_trade_threshold_not_reached"
        elif ramp_completed:
            status = "completed"
        else:
            status = "ramp_incomplete"
        return {
            "scheme": "zero_cost_until_policy_closed_trade_threshold_then_linear_ramp",
            "initial_fee_bps_per_side": 0.0,
            "initial_slippage_bps_per_side": 0.0,
            "target_fee_bps_per_side": self.target_fee_bps,
            "target_slippage_bps_per_side": self.target_slippage_bps,
            "minimum_completed_policy_round_trips": self.minimum_closed_trades,
            "ramp_steps": self.ramp_steps,
            "ramp_update_interval_steps": self.ramp_update_interval_steps,
            "ramp_start_timestep": self._ramp_start_timestep,
            "ramp_start_after_policy_closed_trade_count": (
                self._closed_trades_at_ramp_start
            ),
            "full_cost_start_timestep": full_cost_start,
            "training_timesteps": training_timesteps,
            "total_policy_closed_trade_count": self._total_policy_closed_trades,
            "minimum_trade_threshold_met": (
                self._total_policy_closed_trades >= self.minimum_closed_trades
            ),
            "ramp_started": self._ramp_start_timestep is not None,
            "ramp_completed": ramp_completed,
            "status": status,
            "last_training_step_cost_stage": (
                self.stage_for_timestep(training_timesteps)
                if training_timesteps
                else "not_started"
            ),
            "steps_by_stage": dict(self._steps_by_stage),
            "closed_trades_by_stage": dict(self._closed_trades_by_stage),
        }
