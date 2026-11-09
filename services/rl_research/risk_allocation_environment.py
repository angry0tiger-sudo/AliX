"""Offline Gymnasium environment for risk-only Spot position sizing.

The environment consumes an independent boolean Primary BUY signal. PPO can
choose only a target exposure fraction; it cannot create or reverse signals.
An action is observed after bar close and rebalanced at the next bar open.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

DEFAULT_INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 2.0 * (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE)
COST_RATE_PER_SIDE = (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000.0
SORTINO_DOWNSIDE_FLOOR = 0.001
SORTINO_REWARD_CLIP = 5.0
ACTION_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
OBSERVATION_FIELDS = (
    "atr_fraction",
    "drawdown_fraction",
    "current_position_fraction",
    "primary_buy_signal",
)
MARKET_COLUMN_COUNT = 6
TIMESTAMP_COLUMN = 0
OPEN_COLUMN = 1
HIGH_COLUMN = 2
LOW_COLUMN = 3
CLOSE_COLUMN = 4
VOLUME_COLUMN = 5

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name}_must_be_positive_and_finite")
    return number


def _validate_market_shape(data: Float64Array) -> None:
    if data.ndim != 2 or data.shape[1] != MARKET_COLUMN_COUNT:
        raise ValueError("market_data_must_have_six_columns")
    if data.shape[0] < 2:
        raise ValueError("market_data_requires_at_least_two_rows")


def _validate_market_finiteness(data: Float64Array) -> None:
    if not np.isfinite(data).all():
        raise ValueError("market_data_contains_non_finite_values")


def _validate_market_timestamps(timestamps: Float64Array, interval_ms: int) -> None:
    invalid = (
        np.any(timestamps <= 0.0)
        or np.any(timestamps != np.floor(timestamps))
        or np.any(np.diff(timestamps) != interval_ms)
    )
    if invalid:
        raise ValueError("market_timestamps_must_be_contiguous_integer_epochs")


def _validate_market_geometry(data: Float64Array) -> None:
    opened = data[:, OPEN_COLUMN]
    high = data[:, HIGH_COLUMN]
    low = data[:, LOW_COLUMN]
    close = data[:, CLOSE_COLUMN]
    volume = data[:, VOLUME_COLUMN]
    if np.any(np.minimum.reduce((opened, high, low, close)) <= 0.0):
        raise ValueError("market_prices_must_be_positive")
    if np.any(high < np.maximum(opened, close)):
        raise ValueError("market_candle_high_geometry_invalid")
    if np.any(low > np.minimum(opened, close)):
        raise ValueError("market_candle_low_geometry_invalid")
    if np.any(volume < 0.0):
        raise ValueError("market_volume_must_be_non_negative")


def _validate_market_data(data: Float64Array, interval_ms: int) -> None:
    _validate_market_shape(data)
    _validate_market_finiteness(data)
    _validate_market_timestamps(data[:, TIMESTAMP_COLUMN], interval_ms)
    _validate_market_geometry(data)


def _validate_bar_interval(interval_ms: int) -> None:
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
        raise ValueError("bar_interval_ms_must_be_positive_integer")
    if interval_ms <= 0:
        raise ValueError("bar_interval_ms_must_be_positive_integer")


def _validate_reward_penalty(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward_penalties_must_be_finite_and_non_negative")
    if not math.isfinite(float(value)):
        raise ValueError("reward_penalties_must_be_finite_and_non_negative")
    if value < 0.0:
        raise ValueError("reward_penalties_must_be_finite_and_non_negative")


def _validate_reward_penalties(penalties: tuple[float, ...]) -> None:
    for value in penalties:
        _validate_reward_penalty(value)


def _coerce_primary_signal(value: NDArray[Any]) -> NDArray[np.bool_]:
    if value.ndim != 1 or not np.isin(value, (0, 1, False, True)).all():
        raise ValueError("primary_signal_must_be_boolean")
    return value.astype(np.bool_, copy=True)


def _prepare_environment_inputs(
    market_data: Float64Array,
    atr_fraction: Float64Array,
    primary_signal: NDArray[np.bool_],
    bar_interval_ms: int,
) -> tuple[Float64Array, Float64Array, NDArray[np.bool_]]:
    data = np.asarray(market_data, dtype=np.float64)
    atr = np.asarray(atr_fraction, dtype=np.float64)
    signal = _coerce_primary_signal(np.asarray(primary_signal))
    _validate_market_data(data, bar_interval_ms)
    _validate_auxiliary_arrays(len(data), atr, signal)
    return (
        np.ascontiguousarray(data),
        np.ascontiguousarray(atr),
        np.ascontiguousarray(signal),
    )


def _validate_auxiliary_arrays(
    market_rows: int,
    atr_fraction: Float64Array,
    primary_signal: NDArray[np.bool_],
) -> None:
    if atr_fraction.ndim != 1 or len(atr_fraction) != market_rows:
        raise ValueError("atr_fraction_length_mismatch")
    if not np.isfinite(atr_fraction).all() or np.any(atr_fraction < 0.0):
        raise ValueError("atr_fraction_must_be_finite_and_non_negative")
    if primary_signal.ndim != 1 or len(primary_signal) != market_rows:
        raise ValueError("primary_signal_length_mismatch")


class RiskAllocationEnv(gym.Env[Float32Array, np.int64]):
    """Long-only Spot allocator with a five-level target-exposure action space."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        market_data: Float64Array,
        atr_fraction: Float64Array,
        primary_signal: NDArray[np.bool_],
        *,
        bar_interval_ms: int,
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
        drawdown_increment_penalty: float = 1.5,
        drawdown_level_penalty: float = 0.05,
        downside_return_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        _validate_bar_interval(bar_interval_ms)
        equity = _finite_positive(initial_equity, "initial_equity")
        penalties = (
            drawdown_increment_penalty,
            drawdown_level_penalty,
            downside_return_penalty,
        )
        _validate_reward_penalties(penalties)
        data, atr, signal = _prepare_environment_inputs(
            market_data, atr_fraction, primary_signal, bar_interval_ms
        )

        self._market = data
        self._atr_fraction = atr
        self._primary_signal = signal
        self.bar_interval_ms = bar_interval_ms
        self.initial_equity = equity
        self.drawdown_increment_penalty = float(drawdown_increment_penalty)
        self.drawdown_level_penalty = float(drawdown_level_penalty)
        self.downside_return_penalty = float(downside_return_penalty)

        self.action_space = spaces.Discrete(len(ACTION_FRACTIONS))
        self.observation_space = spaces.Box(
            low=np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32),
            high=np.full(len(OBSERVATION_FIELDS), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_portfolio()

    def _reset_portfolio(self) -> None:
        self._index = 0
        self._cash = self.initial_equity
        self._quantity = 0.0
        self._peak_equity = self.initial_equity
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._total_cost_quote = 0.0
        self._total_traded_quote = 0.0
        self._position_adjustments = 0
        self._downside_squared_sum = 0.0
        self._risk_observation_count = 0
        self._latest_downside_deviation = 0.0
        self._terminated = False

    def _equity(self, price: float) -> float:
        return self._cash + self._quantity * price

    def _position_fraction(self, price: float) -> float:
        equity = self._equity(price)
        if equity <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        return min(1.0, max(0.0, self._quantity * price / equity))

    def _observation(self) -> Float32Array:
        price = float(self._market[self._index, CLOSE_COLUMN])
        values = (
            float(self._atr_fraction[self._index]),
            self._current_drawdown,
            self._position_fraction(price),
            float(self._primary_signal[self._index]),
        )
        observation = np.asarray(values, dtype=np.float32)
        if not np.isfinite(observation).all():
            raise RuntimeError("risk_observation_non_finite")
        return observation

    def _rebalance(self, target_fraction: float, price: float) -> float:
        equity = self._equity(price)
        if equity <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        desired_quantity = target_fraction * equity / price
        delta_quantity = desired_quantity - self._quantity
        if abs(delta_quantity) <= 1e-12:
            return 0.0

        if delta_quantity > 0.0:
            affordable_quantity = self._cash / (price * (1.0 + COST_RATE_PER_SIDE))
            traded_quantity = min(delta_quantity, max(0.0, affordable_quantity))
            traded_notional = traded_quantity * price
            cost = traded_notional * COST_RATE_PER_SIDE
            self._cash -= traded_notional + cost
            self._quantity += traded_quantity
        else:
            traded_quantity = min(-delta_quantity, self._quantity)
            traded_notional = traded_quantity * price
            cost = traded_notional * COST_RATE_PER_SIDE
            self._cash += traded_notional - cost
            self._quantity -= traded_quantity

        self._total_cost_quote += cost
        self._total_traded_quote += traded_notional
        self._position_adjustments += int(traded_notional > 0.0)
        if self._cash < 0.0 and self._cash > -1e-8:
            self._cash = 0.0
        if self._cash < -1e-8 or self._quantity < -1e-12:
            raise RuntimeError("portfolio_accounting_invariant_failed")
        return cost

    def _update_drawdown(self, equity: float) -> float:
        self._peak_equity = max(self._peak_equity, equity)
        self._current_drawdown = max(0.0, 1.0 - equity / self._peak_equity)
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
        return self._current_drawdown

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Float32Array, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._reset_portfolio()
        return self._observation(), {
            "observation_fields": list(OBSERVATION_FIELDS),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "reward_method": "RUNNING_SORTINO_STYLE_LOG_RETURN_OVER_DOWNSIDE_RMS",
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        }

    def _validate_action(self, action: Any) -> int:
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("risk_action_must_be_integer")
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError("risk_action_out_of_range")
        return action_index

    def _resolve_target_exposure(
        self, action_index: int
    ) -> tuple[bool, float, float]:
        signal = bool(self._primary_signal[self._index])
        proposed_target = ACTION_FRACTIONS[action_index]
        current_price = float(self._market[self._index, CLOSE_COLUMN])
        current_fraction = self._position_fraction(current_price)
        applied_target = (
            proposed_target if signal else min(proposed_target, current_fraction)
        )
        return signal, proposed_target, applied_target

    def _sortino_style_component(self, log_return: float) -> float:
        downside_return = min(0.0, log_return)
        self._downside_squared_sum += downside_return * downside_return
        self._risk_observation_count += 1
        self._latest_downside_deviation = math.sqrt(
            self._downside_squared_sum / self._risk_observation_count
        )
        denominator = max(self._latest_downside_deviation, SORTINO_DOWNSIDE_FLOOR)
        ratio = log_return / denominator
        return max(-SORTINO_REWARD_CLIP, min(SORTINO_REWARD_CLIP, ratio))

    def _calculate_step_reward(
        self,
        log_return: float,
        prior_drawdown: float,
        drawdown: float,
    ) -> tuple[float, float]:
        risk_adjusted_return = self._sortino_style_component(log_return)
        downside_penalty = self.downside_return_penalty * max(0.0, -log_return) ** 2
        drawdown_increase = max(0.0, drawdown - prior_drawdown)
        reward = (
            risk_adjusted_return
            - self.drawdown_increment_penalty * drawdown_increase
            - self.drawdown_level_penalty * drawdown
            - downside_penalty
        )
        return reward, risk_adjusted_return

    def _liquidate_terminal_position(self, equity_before: float) -> tuple[float, float]:
        final_price = float(self._market[self._index, CLOSE_COLUMN])
        liquidation_cost = self._rebalance(0.0, final_price)
        final_equity = self._equity(final_price)
        if final_equity <= 0.0 or not math.isfinite(final_equity):
            raise RuntimeError("portfolio_equity_non_positive")
        self._update_drawdown(final_equity)
        terminal_log_return = math.log(final_equity / equity_before)
        terminal_reward = self._sortino_style_component(terminal_log_return)
        self._terminated = True
        return liquidation_cost, terminal_reward

    def _step_info(
        self,
        *,
        primary_signal: bool,
        proposed_target: float,
        applied_target: float,
        terminal_liquidation_cost: float,
        risk_adjusted_return: float,
    ) -> dict[str, Any]:
        final_price = float(self._market[self._index, CLOSE_COLUMN])
        return {
            "primary_signal": primary_signal,
            "proposed_target_fraction": proposed_target,
            "applied_target_fraction": applied_target,
            "current_position_fraction": self._position_fraction(final_price),
            "current_drawdown": self._current_drawdown,
            "max_drawdown": self._max_drawdown,
            "equity": self._equity(final_price),
            "total_cost_quote": self._total_cost_quote,
            "total_traded_quote": self._total_traded_quote,
            "terminal_liquidation_cost": terminal_liquidation_cost,
            "position_adjustments": self._position_adjustments,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "risk_only_action": True,
            "risk_adjusted_return_component": risk_adjusted_return,
            "downside_deviation_fraction": self._latest_downside_deviation,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        }

    def step(
        self, action: Any
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        if self._terminated or self._index >= len(self._market) - 1:
            raise RuntimeError("risk_environment_episode_already_terminated")
        action_index = self._validate_action(action)
        signal, proposed_target, applied_target = self._resolve_target_exposure(
            action_index
        )

        current_price = float(self._market[self._index, CLOSE_COLUMN])
        equity_before = self._equity(current_price)
        next_open = float(self._market[self._index + 1, OPEN_COLUMN])
        self._rebalance(applied_target, next_open)
        next_close = float(self._market[self._index + 1, CLOSE_COLUMN])
        equity_after = self._equity(next_close)
        if equity_after <= 0.0 or not math.isfinite(equity_after):
            raise RuntimeError("portfolio_equity_non_positive")

        prior_drawdown = self._current_drawdown
        drawdown = self._update_drawdown(equity_after)
        log_return = math.log(equity_after / equity_before)
        reward, risk_adjusted_return = self._calculate_step_reward(
            log_return, prior_drawdown, drawdown
        )

        self._index += 1
        terminated = self._index == len(self._market) - 1
        terminal_liquidation_cost = 0.0
        if terminated:
            terminal_liquidation_cost, terminal_reward = (
                self._liquidate_terminal_position(equity_after)
            )
            reward += terminal_reward

        info = self._step_info(
            primary_signal=signal,
            proposed_target=proposed_target,
            applied_target=applied_target,
            terminal_liquidation_cost=terminal_liquidation_cost,
            risk_adjusted_return=risk_adjusted_return,
        )
        return self._observation(), float(reward), terminated, False, info

    def close(self) -> None:
        """The offline environment owns no external resources."""
        return None
