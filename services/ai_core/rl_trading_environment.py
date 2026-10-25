"""Market-only Gymnasium environment for offline BTCUSDT RL research.

Actions are chosen after a completed one-minute candle and executed at the next
candle's open. Portfolio state is deliberately absent from observations. Cash,
inventory, costs and risk are maintained by the simulator only.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

INTERVAL_MS = 60_000
TIMESTAMP_COLUMN = 0
OPEN_COLUMN = 1
HIGH_COLUMN = 2
LOW_COLUMN = 3
CLOSE_COLUMN = 4
VOLUME_COLUMN = 5
MARKET_COLUMN_COUNT = 6

OBSERVATION_FIELDS = (
    "log_open_relative_to_training_anchor",
    "log_high_relative_to_training_anchor",
    "log_low_relative_to_training_anchor",
    "log_close_relative_to_training_anchor",
    "log_volume_relative_to_training_median",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "minute_of_week_sin",
    "minute_of_week_cos",
)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
BUY_THRESHOLD = 1.0 / 3.0
SELL_THRESHOLD = -1.0 / 3.0
DEFAULT_DRAWDOWN_INCREMENT_PENALTY = 5.0
DEFAULT_DRAWDOWN_LEVEL_PENALTY = 0.01

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]


def validate_market_data(market_data: Float64Array) -> None:
    """Fail closed on invalid candles, gaps, or non-minute timestamps."""

    data = np.asarray(market_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != MARKET_COLUMN_COUNT:
        raise ValueError("market_data_must_have_six_columns")
    if data.shape[0] < 2:
        raise ValueError("market_data_requires_at_least_two_rows")
    if not np.isfinite(data).all():
        raise ValueError("market_data_contains_non_finite_values")

    timestamps = data[:, TIMESTAMP_COLUMN]
    if np.any(timestamps < 0) or np.any(timestamps != np.floor(timestamps)):
        raise ValueError("market_timestamps_must_be_non_negative_integers")
    if np.any(np.diff(timestamps) != INTERVAL_MS):
        raise ValueError("market_timestamps_must_be_contiguous_one_minute_bars")

    opening = data[:, OPEN_COLUMN]
    high = data[:, HIGH_COLUMN]
    low = data[:, LOW_COLUMN]
    close = data[:, CLOSE_COLUMN]
    volume = data[:, VOLUME_COLUMN]
    if np.any(np.minimum.reduce((opening, high, low, close)) <= 0):
        raise ValueError("market_prices_must_be_positive")
    if np.any(high < np.maximum(np.maximum(opening, close), low)):
        raise ValueError("market_high_inconsistent")
    if np.any(low > np.minimum(np.minimum(opening, close), high)):
        raise ValueError("market_low_inconsistent")
    if np.any(volume < 0):
        raise ValueError("market_volume_must_be_non_negative")


class MarketObservationScaler:
    """Fit price and volume scales on training candles only."""

    __slots__ = ("price_anchor", "volume_scale")

    def __init__(self, price_anchor: float, volume_scale: float) -> None:
        if not math.isfinite(price_anchor) or price_anchor <= 0:
            raise ValueError("price_anchor_must_be_positive_and_finite")
        if not math.isfinite(volume_scale) or volume_scale <= 0:
            raise ValueError("volume_scale_must_be_positive_and_finite")
        self.price_anchor = float(price_anchor)
        self.volume_scale = float(volume_scale)

    @classmethod
    def fit(cls, training_data: Float64Array) -> MarketObservationScaler:
        """Fit scale constants without inspecting validation or holdout rows."""

        validate_market_data(training_data)
        price_anchor = float(training_data[0, CLOSE_COLUMN])
        median_volume = float(np.median(training_data[:, VOLUME_COLUMN]))
        volume_scale = median_volume if median_volume > 0 else 1.0
        return cls(price_anchor=price_anchor, volume_scale=volume_scale)


class BTCUSDTTradingEnv(gym.Env[Float32Array, Float32Array]):
    """Long-only, one-minute, cost-aware trading environment.

    Action component 0 maps to Sell / Hold / Buy. Component 1 is a continuous
    size in [0, 1]. Buy targets that fraction of equity, Sell reduces the
    existing holding by that fraction, and Hold makes no rebalance. The
    environment has no fixed timeframe, stop-loss, or take-profit rule.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: Float64Array,
        scaler: MarketObservationScaler | None = None,
        *,
        initial_equity: float = 10_000.0,
        fee_bps_per_side: float = 25.0,
        slippage_bps_per_side: float = 20.0,
        random_start: bool = False,
        min_remaining_steps: int = 32_768,
        drawdown_increment_penalty: float = DEFAULT_DRAWDOWN_INCREMENT_PENALTY,
        drawdown_level_penalty: float = DEFAULT_DRAWDOWN_LEVEL_PENALTY,
    ) -> None:
        super().__init__()
        validate_market_data(market_data)
        for name, value in (
            ("initial_equity", initial_equity),
            ("fee_bps_per_side", fee_bps_per_side),
            ("slippage_bps_per_side", slippage_bps_per_side),
            ("drawdown_increment_penalty", drawdown_increment_penalty),
            ("drawdown_level_penalty", drawdown_level_penalty),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}_must_be_numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}_must_be_finite")
        if initial_equity <= 0:
            raise ValueError("initial_equity_must_be_positive")
        if not 0 <= fee_bps_per_side < 10_000:
            raise ValueError("fee_bps_per_side_out_of_range")
        if not 0 <= slippage_bps_per_side < 10_000:
            raise ValueError("slippage_bps_per_side_out_of_range")
        if drawdown_increment_penalty < 0 or drawdown_level_penalty < 0:
            raise ValueError("drawdown_penalties_must_be_non_negative")
        if isinstance(min_remaining_steps, bool) or min_remaining_steps < 1:
            raise ValueError("min_remaining_steps_must_be_positive")

        self._market_data = np.ascontiguousarray(market_data, dtype=np.float64)
        self.scaler = scaler or MarketObservationScaler.fit(self._market_data)
        self.initial_equity = float(initial_equity)
        self.fee_bps_per_side = float(fee_bps_per_side)
        self.slippage_bps_per_side = float(slippage_bps_per_side)
        self.random_start = random_start
        self.min_remaining_steps = int(min_remaining_steps)
        self.drawdown_increment_penalty = float(drawdown_increment_penalty)
        self.drawdown_level_penalty = float(drawdown_level_penalty)

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(OBSERVATION_SIZE, -np.inf, dtype=np.float32),
            high=np.full(OBSERVATION_SIZE, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self._index = 0
        self._cash = self.initial_equity
        self._quantity = 0.0
        self._peak_equity = self.initial_equity
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_turnover = 0.0
        self._fill_count = 0
        self._buy_fill_count = 0
        self._sell_fill_count = 0
        self._final_equity: float | None = None

    def _observation(self, index: int) -> Float32Array:
        row = self._market_data[index]
        anchor = self.scaler.price_anchor
        log_prices = np.log(row[OPEN_COLUMN : CLOSE_COLUMN + 1] / anchor)
        scaled_volume = math.log1p(row[VOLUME_COLUMN] / self.scaler.volume_scale)
        minute_of_epoch = int(row[TIMESTAMP_COLUMN] // INTERVAL_MS)
        minute_of_day = minute_of_epoch % 1_440
        day_index = minute_of_epoch // 1_440
        day_of_week = (day_index + 3) % 7
        minute_of_week = day_of_week * 1_440 + minute_of_day
        daily_angle = 2.0 * math.pi * minute_of_day / 1_440
        weekly_angle = 2.0 * math.pi * minute_of_week / (7 * 1_440)
        values = (
            *log_prices.tolist(),
            scaled_volume,
            math.sin(daily_angle),
            math.cos(daily_angle),
            math.sin(weekly_angle),
            math.cos(weekly_angle),
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError("market_observation_invalid")
        return observation

    def _equity_at(self, price: float) -> float:
        equity = self._cash + self._quantity * price
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("simulated_equity_non_positive")
        return equity

    def _rebalance_to_quantity(self, target_quantity: float, price: float) -> None:
        delta = target_quantity - self._quantity
        if delta == 0.0:
            return
        if not math.isfinite(target_quantity) or target_quantity < 0:
            raise ValueError("target_quantity_invalid")

        slippage_rate = self.slippage_bps_per_side / 10_000.0
        fee_rate = self.fee_bps_per_side / 10_000.0
        is_buy = delta > 0
        execution_price = price * (1.0 + slippage_rate if is_buy else 1.0 - slippage_rate)
        fee = abs(delta * execution_price) * fee_rate
        slippage_cost = abs(delta) * abs(execution_price - price)
        next_cash = self._cash - delta * execution_price - fee
        if not math.isfinite(next_cash) or next_cash < -1e-7:
            raise ValueError("simulated_spot_cash_would_be_negative")

        self._cash = max(0.0, next_cash)
        self._quantity = target_quantity
        self._total_fees += fee
        self._total_slippage += slippage_cost
        self._total_turnover += abs(delta * execution_price)
        self._fill_count += 1
        if is_buy:
            self._buy_fill_count += 1
        else:
            self._sell_fill_count += 1

    def _target_for_buy(self, size: float, price: float) -> None:
        equity = self._equity_at(price)
        target_notional = size * equity
        target_quantity = target_notional / price
        if target_quantity > self._quantity:
            cost_adjusted_price = (
                price
                * (1.0 + self.slippage_bps_per_side / 10_000.0)
                * (1.0 + self.fee_bps_per_side / 10_000.0)
            )
            target_quantity = target_notional / cost_adjusted_price
        self._rebalance_to_quantity(target_quantity, price)

    def _apply_action(self, action: Float32Array, current_price: float, next_open: float) -> str:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("action_must_be_two_finite_values")
        signal = float(values[0])
        size = float(values[1])
        if not -1.0 <= signal <= 1.0 or not 0.0 <= size <= 1.0:
            raise ValueError("action_outside_declared_space")

        if signal > BUY_THRESHOLD:
            self._target_for_buy(size, next_open)
            return "BUY"
        if signal < SELL_THRESHOLD:
            target_quantity = self._quantity * (1.0 - size)
            self._rebalance_to_quantity(target_quantity, next_open)
            return "SELL"
        return "HOLD"

    def _update_drawdown(self, bar: Float64Array) -> float:
        high_equity = self._equity_at(float(bar[HIGH_COLUMN]))
        self._peak_equity = max(self._peak_equity, high_equity)
        low_equity = self._equity_at(float(bar[LOW_COLUMN]))
        intrabar_drawdown = max(0.0, (self._peak_equity - low_equity) / self._peak_equity)
        self._max_drawdown = max(self._max_drawdown, intrabar_drawdown)

        close_equity = self._equity_at(float(bar[CLOSE_COLUMN]))
        self._peak_equity = max(self._peak_equity, close_equity)
        close_drawdown = max(0.0, (self._peak_equity - close_equity) / self._peak_equity)
        self._current_drawdown = close_drawdown
        self._max_drawdown = max(self._max_drawdown, close_drawdown)
        return max(intrabar_drawdown, close_drawdown)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Float32Array, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "start_index" in options:
            raw_start = options["start_index"]
            if isinstance(raw_start, bool) or not isinstance(raw_start, int):
                raise ValueError("start_index_must_be_integer")
            if not 0 <= raw_start < len(self._market_data) - 1:
                raise ValueError("start_index_out_of_range")
            self._index = raw_start
        elif self.random_start:
            remaining = min(self.min_remaining_steps, len(self._market_data) - 1)
            max_start = len(self._market_data) - 1 - remaining
            self._index = int(self.np_random.integers(0, max_start + 1))
        else:
            self._index = 0

        self._cash = self.initial_equity
        self._quantity = 0.0
        self._peak_equity = self.initial_equity
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_turnover = 0.0
        self._fill_count = 0
        self._buy_fill_count = 0
        self._sell_fill_count = 0
        self._final_equity = None
        return self._observation(self._index), {"start_index": self._index}

    def step(
        self, action: Float32Array
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        if self._index >= len(self._market_data) - 1:
            raise RuntimeError("environment_episode_already_terminated")

        current_bar = self._market_data[self._index]
        next_index = self._index + 1
        next_bar = self._market_data[next_index]
        previous_equity = self._equity_at(float(current_bar[CLOSE_COLUMN]))
        previous_drawdown = self._current_drawdown

        action_name = self._apply_action(
            action,
            current_price=float(current_bar[CLOSE_COLUMN]),
            next_open=float(next_bar[OPEN_COLUMN]),
        )
        risk_drawdown = self._update_drawdown(next_bar)
        terminated = next_index == len(self._market_data) - 1
        if terminated:
            self._rebalance_to_quantity(0.0, float(next_bar[CLOSE_COLUMN]))
            final_equity = self._cash
            if not math.isfinite(final_equity) or final_equity <= 0:
                raise ValueError("final_simulated_equity_non_positive")
            self._final_equity = final_equity
            terminal_drawdown = max(
                0.0, (self._peak_equity - final_equity) / self._peak_equity
            )
            risk_drawdown = max(risk_drawdown, terminal_drawdown)
            self._current_drawdown = terminal_drawdown
            self._max_drawdown = max(self._max_drawdown, terminal_drawdown)
            current_equity = final_equity
        else:
            current_equity = self._equity_at(float(next_bar[CLOSE_COLUMN]))

        log_net_return = math.log(current_equity / previous_equity)
        worsening_drawdown = max(0.0, risk_drawdown - previous_drawdown)
        reward = (
            log_net_return
            - self.drawdown_increment_penalty * worsening_drawdown
            - self.drawdown_level_penalty * risk_drawdown
        )
        if not math.isfinite(reward):
            raise ValueError("reward_non_finite")

        self._index = next_index
        info: dict[str, Any] = {
            "timestamp_ms": int(next_bar[TIMESTAMP_COLUMN]),
            "action": action_name,
            "reward_net_log_return": log_net_return,
            "drawdown_for_reward": risk_drawdown,
        }
        if terminated:
            info.update(self.summary())
        return self._observation(self._index), float(reward), terminated, False, info

    def summary(self) -> dict[str, float | int]:
        """Return accounting evidence; never included in the market observation."""

        price = float(self._market_data[self._index, CLOSE_COLUMN])
        final_equity = self._final_equity
        if final_equity is None:
            final_equity = self._equity_at(price)
        return {
            "initial_equity": self.initial_equity,
            "final_equity": float(final_equity),
            "net_profit": float(final_equity - self.initial_equity),
            "net_return": float(final_equity / self.initial_equity - 1.0),
            "max_drawdown": float(self._max_drawdown),
            "total_fees": float(self._total_fees),
            "total_slippage": float(self._total_slippage),
            "total_turnover": float(self._total_turnover),
            "fill_count": self._fill_count,
            "buy_fill_count": self._buy_fill_count,
            "sell_fill_count": self._sell_fill_count,
        }
