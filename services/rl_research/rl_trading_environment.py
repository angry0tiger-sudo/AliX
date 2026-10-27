"""State-aware Gymnasium environment for offline BTCUSDT RL research.

Actions are chosen after a completed one-minute candle and executed at the next
candle's open. Market observations include marked portfolio exposure and the
unrealized return on the remaining position cost basis.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

DEFAULT_INITIAL_EQUITY = 10_000.0
DEFAULT_OPPORTUNITY_COST_PENALTY = 0.01
MOVING_AVERAGE_WINDOW = 20
ACTION_HOLD = 0
ACTION_BUY_25 = 1
ACTION_BUY_50 = 2
ACTION_BUY_100 = 3
ACTION_SELL_25 = 4
ACTION_SELL_50 = 5
ACTION_SELL_100 = 6
ACTION_BUY = ACTION_BUY_25
ACTION_SELL = ACTION_SELL_25
ACTION_NAMES = ("HOLD", "BUY_25", "BUY_50", "BUY_100", "SELL_25", "SELL_50", "SELL_100")
BUY_ACTION_INDICES = (ACTION_BUY_25, ACTION_BUY_50, ACTION_BUY_100)
SELL_ACTION_INDICES = (ACTION_SELL_25, ACTION_SELL_50, ACTION_SELL_100)
ACTION_SIDE_BY_INDEX = ("HOLD", "BUY", "BUY", "BUY", "SELL", "SELL", "SELL")
ACTION_SIZE_BY_INDEX = (0.0, 0.25, 0.50, 1.00, 0.25, 0.50, 1.00)
POSITION_SIZE_FRACTIONS = (0.25, 0.50, 1.00)
INTERVAL_MS = 60_000
TIMESTAMP_COLUMN = 0
OPEN_COLUMN = 1
HIGH_COLUMN = 2
LOW_COLUMN = 3
CLOSE_COLUMN = 4
VOLUME_COLUMN = 5
MARKET_COLUMN_COUNT = 6

MARKET_OBSERVATION_FIELDS = (
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
PORTFOLIO_OBSERVATION_FIELDS = (
    "current_position_fraction",
    "unrealized_pnl_pct",
    "position_holding_period_fraction",
)
OBSERVATION_FIELDS = (*MARKET_OBSERVATION_FIELDS, *PORTFOLIO_OBSERVATION_FIELDS)
MARKET_ONLY_OBSERVATION_SIZE = len(MARKET_OBSERVATION_FIELDS)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
BUY_THRESHOLD = 1.0 / 3.0
SELL_THRESHOLD = -1.0 / 3.0
DEFAULT_DRAWDOWN_INCREMENT_PENALTY = 5.0
DEFAULT_DRAWDOWN_LEVEL_PENALTY = 0.01
DEFAULT_INACTIVITY_PENALTY = 0.001
DEFAULT_INVALID_ACTION_PENALTY = 0.001
DEFAULT_INVALID_FULL_POSITION_BUY_PENALTY = 1.0
DEFAULT_MINIMUM_HOLDING_PERIOD_CANDLES = 15
DEFAULT_HOLDING_PROFIT_REWARD = 0.0001
DEFAULT_HOLDING_LOSS_PENALTY = 0.0001
DEFAULT_REALIZED_PROFIT_REWARD_SCALE = 0.05
DEFAULT_INACTIVITY_GRACE_STEPS = 3
DEFAULT_PROFITABLE_TRADE_BONUS = 0.0001
DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER = 2.0
DEFAULT_EARLY_EXIT_REWARD_PENALTY = 0.0001
DEFAULT_LARGE_PROFIT_REWARD_CAP = 0.01
DEFAULT_TRAILING_STOP_DISTANCE_PCT = 0.02
DEFAULT_TRAILING_STOP_ACTIVATION_PCT = 0.03
POSITION_FRACTION_EPSILON = 1e-9

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


def _validate_market_shape(data: Float64Array) -> None:
    if data.ndim != 2 or data.shape[1] != MARKET_COLUMN_COUNT:
        raise ValueError("market_data_must_have_six_columns")
    if data.shape[0] < 2:
        raise ValueError("market_data_requires_at_least_two_rows")
    if not np.isfinite(data).all():
        raise ValueError("market_data_contains_non_finite_values")


def _validate_market_timestamps(data: Float64Array) -> None:
    timestamps = data[:, TIMESTAMP_COLUMN]
    if np.any(timestamps < 0) or np.any(timestamps != np.floor(timestamps)):
        raise ValueError("market_timestamps_must_be_non_negative_integers")
    if np.any(np.diff(timestamps) != INTERVAL_MS):
        raise ValueError("market_timestamps_must_be_contiguous_one_minute_bars")


def _validate_candle_geometry(data: Float64Array) -> None:
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


def validate_market_data(market_data: Float64Array) -> None:
    """Fail closed on invalid candles, gaps, or non-minute timestamps."""

    data = np.asarray(market_data, dtype=np.float64)
    _validate_market_shape(data)
    _validate_market_timestamps(data)
    _validate_candle_geometry(data)


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


def _finite_numeric_parameter(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name}_must_be_finite")
    return numeric_value


def _validate_cost_basis_points(name: str, value: object) -> None:
    numeric_value = _finite_numeric_parameter(name, value)
    if not 0 <= numeric_value < 10_000:
        raise ValueError(f"{name}_out_of_range")


def _validate_drawdown_penalties(increment: object, level: object) -> None:
    increment_value = _finite_numeric_parameter("drawdown_increment_penalty", increment)
    level_value = _finite_numeric_parameter("drawdown_level_penalty", level)
    if increment_value < 0 or level_value < 0:
        raise ValueError("drawdown_penalties_must_be_non_negative")


def _validate_reward_shaping(
    inactivity_penalty: object,
    profitable_trade_bonus: object,
    invalid_action_penalty: object,
) -> None:
    inactivity_value = _finite_numeric_parameter(
        "inactivity_penalty", inactivity_penalty
    )
    bonus_value = _finite_numeric_parameter(
        "profitable_trade_bonus", profitable_trade_bonus
    )
    invalid_action_value = _finite_numeric_parameter(
        "invalid_action_penalty", invalid_action_penalty
    )
    if inactivity_value < 0 or bonus_value < 0 or invalid_action_value < 0:
        raise ValueError("reward_shaping_parameters_must_be_non_negative")


def _validate_environment_configuration(
    initial_equity: object,
    fee_bps_per_side: object,
    slippage_bps_per_side: object,
    min_remaining_steps: object,
    drawdown_increment_penalty: object,
    drawdown_level_penalty: object,
    inactivity_penalty: object,
    invalid_action_penalty: object,
    inactivity_grace_steps: object,
    profitable_trade_bonus: object,
) -> None:
    equity_value = _finite_numeric_parameter("initial_equity", initial_equity)
    if equity_value <= 0:
        raise ValueError("initial_equity_must_be_positive")
    _validate_cost_basis_points("fee_bps_per_side", fee_bps_per_side)
    _validate_cost_basis_points("slippage_bps_per_side", slippage_bps_per_side)
    _validate_drawdown_penalties(drawdown_increment_penalty, drawdown_level_penalty)
    _validate_reward_shaping(
        inactivity_penalty, profitable_trade_bonus, invalid_action_penalty
    )
    if (
        isinstance(inactivity_grace_steps, bool)
        or not isinstance(inactivity_grace_steps, int)
        or inactivity_grace_steps < 0
    ):
        raise ValueError("inactivity_grace_steps_must_be_non_negative_integer")
    if (
        isinstance(min_remaining_steps, bool)
        or not isinstance(min_remaining_steps, int)
        or min_remaining_steps < 1
    ):
        raise ValueError("min_remaining_steps_must_be_positive")


def _validate_minimum_holding_period_candles(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("minimum_holding_period_candles_must_be_positive_integer")


def _validate_portfolio_state_option(value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError("include_portfolio_state_must_be_boolean")


def _validate_trailing_stop_configuration(distance: object, activation: object) -> None:
    distance_value = _finite_numeric_parameter("trailing_stop_distance_pct", distance)
    activation_value = _finite_numeric_parameter(
        "trailing_stop_activation_pct", activation
    )
    if not 0.0 < distance_value < 1.0:
        raise ValueError("trailing_stop_distance_pct_out_of_range")
    if not 0.0 < activation_value < 1.0:
        raise ValueError("trailing_stop_activation_pct_out_of_range")
    if distance_value >= activation_value:
        raise ValueError("trailing_stop_distance_must_be_below_activation")


def _validate_signal_guidance_reward_scale(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError("signal_guidance_reward_scale_must_be_finite_and_non_negative")


class BTCUSDTTradingEnv(gym.Env[Float32Array, np.int64]):
    """Long-only, one-minute, cost-aware trading environment.

    Actions use Discrete(7): HOLD; BUY 25%, 50%, or 100% of spendable cash; and
    SELL 25%, 50%, or 100% of current inventory. Direction and size form one action.
    Execution-boundary legality remains in the environment, with policy-level masking
    in the recurrent PPO distribution. Flat HOLD above the causal moving average
    receives an opportunity-cost penalty; holding cash earns no bonus.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: Float64Array,
        scaler: MarketObservationScaler | None = None,
        *,
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
        fee_bps_per_side: float = 25.0,
        slippage_bps_per_side: float = 20.0,
        random_start: bool = False,
        include_portfolio_state: bool = True,
        min_remaining_steps: int = 32_768,
        minimum_holding_period_candles: int = DEFAULT_MINIMUM_HOLDING_PERIOD_CANDLES,
        drawdown_increment_penalty: float = DEFAULT_DRAWDOWN_INCREMENT_PENALTY,
        drawdown_level_penalty: float = DEFAULT_DRAWDOWN_LEVEL_PENALTY,
        inactivity_penalty: float = DEFAULT_INACTIVITY_PENALTY,
        invalid_action_penalty: float = DEFAULT_INVALID_ACTION_PENALTY,
        inactivity_grace_steps: int = DEFAULT_INACTIVITY_GRACE_STEPS,
        profitable_trade_bonus: float = DEFAULT_PROFITABLE_TRADE_BONUS,
        signal_guidance_reward_scale: float = 0.0,
        trailing_stop_distance_pct: float = DEFAULT_TRAILING_STOP_DISTANCE_PCT,
        trailing_stop_activation_pct: float = DEFAULT_TRAILING_STOP_ACTIVATION_PCT,
    ) -> None:
        super().__init__()
        _validate_minimum_holding_period_candles(minimum_holding_period_candles)
        _validate_portfolio_state_option(include_portfolio_state)
        validate_market_data(market_data)
        _validate_environment_configuration(
            initial_equity,
            fee_bps_per_side,
            slippage_bps_per_side,
            min_remaining_steps,
            drawdown_increment_penalty,
            drawdown_level_penalty,
            inactivity_penalty,
            invalid_action_penalty,
            inactivity_grace_steps,
            profitable_trade_bonus,
        )
        _validate_signal_guidance_reward_scale(signal_guidance_reward_scale)
        _validate_trailing_stop_configuration(
            trailing_stop_distance_pct, trailing_stop_activation_pct
        )

        self._market_data = np.ascontiguousarray(market_data, dtype=np.float64)
        self.scaler = scaler or MarketObservationScaler.fit(self._market_data)
        close_prices = self._market_data[:, CLOSE_COLUMN]
        cumulative_close = np.concatenate(([0.0], np.cumsum(close_prices)))
        self._opportunity_moving_average = np.full(len(close_prices), np.nan)
        self._opportunity_moving_average[MOVING_AVERAGE_WINDOW - 1 :] = (
            cumulative_close[MOVING_AVERAGE_WINDOW:]
            - cumulative_close[:-MOVING_AVERAGE_WINDOW]
        ) / MOVING_AVERAGE_WINDOW
        self.initial_equity = float(initial_equity)
        self.fee_bps_per_side = float(fee_bps_per_side)
        self.slippage_bps_per_side = float(slippage_bps_per_side)
        self.random_start = random_start
        self.include_portfolio_state = include_portfolio_state
        self.observation_fields = (
            OBSERVATION_FIELDS if include_portfolio_state else MARKET_OBSERVATION_FIELDS
        )
        self.min_remaining_steps = int(min_remaining_steps)
        self.minimum_holding_period_candles = minimum_holding_period_candles
        self.drawdown_increment_penalty = float(drawdown_increment_penalty)
        self.drawdown_level_penalty = float(drawdown_level_penalty)
        self.inactivity_penalty = float(inactivity_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.inactivity_grace_steps = inactivity_grace_steps
        self.profitable_trade_bonus = float(profitable_trade_bonus)
        self.signal_guidance_reward_scale = float(signal_guidance_reward_scale)
        self.trailing_stop_distance_pct = float(trailing_stop_distance_pct)
        self.trailing_stop_activation_pct = float(trailing_stop_activation_pct)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        observation_size = len(self.observation_fields)
        self.observation_space = spaces.Box(
            low=np.full(observation_size, -np.inf, dtype=np.float32),
            high=np.full(observation_size, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self._index = 0
        self._last_buy_index: int | None = None
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
        self._position_cost_basis = 0.0
        self._position_realized_profit = 0.0
        self._last_step_realized_profit = 0.0
        self._last_step_profitable_trade_count = 0
        self._last_closed_trade_net_profit: float | None = None
        self._last_closed_trade_return_pct: float | None = None
        self._last_closed_trade_exit_reason: str | None = None
        self._active_exit_reason: str | None = None
        self._trade_entry_index: int | None = None
        self._trade_entry_reference_price: float | None = None
        self._trade_entry_cost_total = 0.0
        self._highest_price_since_entry: float | None = None
        self._trailing_stop_price: float | None = None
        self._trailing_stop_activated = False
        self._trailing_stop_trigger_count = 0
        self._last_trailing_stop_triggered = False
        self._closed_trade_holding_periods: list[int] = []
        self._closed_trade_returns_percent: list[float] = []
        self._closed_trade_net_profits: list[float] = []
        self._closed_trade_exit_reasons: list[str] = []
        self._early_exit_penalty_count = 0
        self._early_exit_penalty_total = 0.0
        self._early_exit_count = 0
        self._closed_trade_count = 0
        self._policy_closed_trade_count = 0
        self._winning_trade_count = 0
        self._losing_trade_count = 0
        self._breakeven_trade_count = 0
        self._buy_action_count = 0
        self._sell_action_count = 0
        self._hold_action_count = 0
        self._discrete_action_counts = {name: 0 for name in ACTION_NAMES}
        self._masked_buy_action_count = 0
        self._buy_noop_action_count = 0
        self._zero_size_buy_action_count = 0
        self._buy_target_not_increasing_action_count = 0
        self._masked_sell_action_count = 0
        self._minimum_holding_period_blocked_action_count = 0
        self._invalid_action_penalty_count = 0
        self._full_position_buy_penalty_count = 0
        self._no_position_change_steps = 0
        self._consecutive_no_position_change_steps = 0
        self._max_consecutive_no_position_change_steps = 0
        self._position_change_count = 0
        self._final_equity: float | None = None
        self._opportunity_cost_penalty_count = 0
        self._holding_loss_penalty_count = 0
        self._holding_profit_reward_count = 0
        self._transaction_cost_reward_penalty_total = 0.0
        self._signal_guidance_reward_count = 0
        self._signal_guidance_reward_total = 0.0
        self._signal_guidance_buy_reward_count = 0
        self._signal_guidance_sell_reward_count = 0
        self._realized_profit_reward_count = 0
        self._realized_profit_reward_total = 0.0

    def set_transaction_costs(
        self, fee_bps_per_side: float, slippage_bps_per_side: float
    ) -> None:
        """Atomically update simulated costs for subsequent fills only.

        Fees already booked to cash and position cost basis are immutable.
        Invalid values leave both active settings unchanged.
        """
        _validate_cost_basis_points("fee_bps_per_side", fee_bps_per_side)
        _validate_cost_basis_points("slippage_bps_per_side", slippage_bps_per_side)
        self.fee_bps_per_side = float(fee_bps_per_side)
        self.slippage_bps_per_side = float(slippage_bps_per_side)

    def set_signal_guidance_reward_scale(self, reward_scale: float) -> None:
        """Change training-only shaping for subsequent actions."""
        if (
            isinstance(reward_scale, bool)
            or not isinstance(reward_scale, (int, float))
            or not math.isfinite(float(reward_scale))
            or reward_scale < 0.0
        ):
            raise ValueError(
                "signal_guidance_reward_scale_must_be_finite_and_non_negative"
            )
        self.signal_guidance_reward_scale = float(reward_scale)

    def _position_holding_candles(self, price: float) -> int:
        if (
            self._quantity <= self._quantity_tolerance(price)
            or self._last_buy_index is None
        ):
            return 0
        return max(0, self._index - self._last_buy_index + 1)

    def _minimum_holding_period_active(self, price: float) -> bool:
        return (
            self._quantity > self._quantity_tolerance(price)
            and self._position_holding_candles(price)
            < self.minimum_holding_period_candles
        )

    def _portfolio_observation_values(self, price: float) -> tuple[float, float, float]:
        equity = self._equity_at(price)
        position_value = self._quantity * price
        if (
            self._quantity <= self._quantity_tolerance(price)
            or self._position_cost_basis <= 0.0
        ):
            return 0.0, 0.0, 0.0
        position_fraction = min(1.0, max(0.0, position_value / equity))
        unrealized_pnl_pct = (
            position_value - self._position_cost_basis
        ) / self._position_cost_basis
        holding_period_fraction = min(
            1.0,
            self._position_holding_candles(price) / self.minimum_holding_period_candles,
        )
        if not all(
            math.isfinite(value)
            for value in (
                position_fraction,
                unrealized_pnl_pct,
                holding_period_fraction,
            )
        ):
            raise ValueError("portfolio_observation_non_finite")
        return (
            float(position_fraction),
            float(unrealized_pnl_pct),
            float(holding_period_fraction),
        )

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
        if self.include_portfolio_state:
            values = (
                *values,
                *self._portfolio_observation_values(float(row[CLOSE_COLUMN])),
            )
        observation = np.asarray(values, dtype=np.float32)
        if (
            observation.shape != (len(self.observation_fields),)
            or not np.isfinite(observation).all()
        ):
            raise ValueError("market_observation_invalid")
        return observation

    def _equity_at(self, price: float) -> float:
        equity = self._cash + self._quantity * price
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("simulated_equity_non_positive")
        return equity

    def _record_sale(
        self,
        sold_quantity: float,
        execution_price: float,
        fee: float,
        target_quantity: float,
    ) -> None:
        average_cost = self._position_cost_basis / self._quantity
        realized_profit = sold_quantity * (execution_price - average_cost) - fee
        self._position_cost_basis = max(
            0.0, self._position_cost_basis - sold_quantity * average_cost
        )
        self._position_realized_profit += realized_profit
        self._last_step_realized_profit += realized_profit
        if target_quantity != 0.0:
            return

        closed_profit = self._position_realized_profit
        self._last_closed_trade_net_profit = float(closed_profit)
        entry_cost = self._trade_entry_cost_total
        return_pct = 100.0 * closed_profit / entry_cost if entry_cost > 0.0 else None
        self._last_closed_trade_return_pct = return_pct
        exit_reason = self._active_exit_reason or "policy_sell"
        self._last_closed_trade_exit_reason = exit_reason
        if self._trade_entry_index is not None:
            self._closed_trade_holding_periods.append(
                max(1, self._index - self._trade_entry_index + 1)
            )
        self._closed_trade_net_profits.append(float(closed_profit))
        self._closed_trade_exit_reasons.append(exit_reason)
        if return_pct is not None:
            self._closed_trade_returns_percent.append(float(return_pct))
            if return_pct < 1.0:
                self._early_exit_count += 1
        self._closed_trade_count += 1
        tolerance = self.initial_equity * 1e-12
        if closed_profit > tolerance:
            self._winning_trade_count += 1
            self._last_step_profitable_trade_count += 1
        elif closed_profit < -tolerance:
            self._losing_trade_count += 1
        else:
            self._breakeven_trade_count += 1
        self._position_cost_basis = 0.0
        self._position_realized_profit = 0.0
        self._last_buy_index = None
        self._trade_entry_index = None
        self._trade_entry_reference_price = None
        self._trade_entry_cost_total = 0.0
        self._highest_price_since_entry = None
        self._trailing_stop_price = None
        self._trailing_stop_activated = False

    def _record_action(
        self,
        action_name: str,
        requested_action: str,
        executed_action: str,
        reason: str | None,
    ) -> None:
        if action_name not in self._discrete_action_counts:
            raise ValueError("action_name_invalid")
        self._discrete_action_counts[action_name] += 1
        if requested_action == "BUY":
            self._buy_action_count += 1
        elif requested_action == "SELL":
            self._sell_action_count += 1
        elif requested_action == "HOLD":
            self._hold_action_count += 1
        else:
            raise ValueError("action_side_invalid")
        if executed_action == "MASKED_BUY":
            self._masked_buy_action_count += 1
        elif executed_action == "NOOP_BUY":
            self._buy_noop_action_count += 1
            if reason == "buy_target_not_above_current_position":
                self._buy_target_not_increasing_action_count += 1
        elif executed_action == "MASKED_SELL":
            self._masked_sell_action_count += 1
        if reason == "minimum_holding_period_active":
            self._minimum_holding_period_blocked_action_count += 1

    def _quantity_tolerance(self, price: float) -> float:
        return max(1e-12, self.initial_equity / price * 1e-12)

    def _should_skip_rebalance(self, target_quantity: float, price: float) -> bool:
        delta = target_quantity - self._quantity
        return delta == 0.0 or (
            target_quantity > 0.0 and abs(delta) <= self._quantity_tolerance(price)
        )

    def _validate_target_quantity(self, target_quantity: float) -> None:
        if not math.isfinite(target_quantity) or target_quantity < 0.0:
            raise ValueError("target_quantity_invalid")

    def _fill_costs(
        self, delta: float, price: float, is_buy: bool
    ) -> tuple[float, float, float]:
        slippage_rate = self.slippage_bps_per_side / 10_000.0
        fee_rate = self.fee_bps_per_side / 10_000.0
        price_factor = 1.0 + slippage_rate if is_buy else 1.0 - slippage_rate
        execution_price = price * price_factor
        fee = abs(delta * execution_price) * fee_rate
        slippage_cost = abs(delta) * abs(execution_price - price)
        return execution_price, fee, slippage_cost

    def _validate_rebalanced_cash(self, next_cash: float) -> None:
        if not math.isfinite(next_cash) or next_cash < -1e-7:
            raise ValueError("simulated_spot_cash_would_be_negative")

    def _record_cost_basis(
        self,
        delta: float,
        execution_price: float,
        fee: float,
        target_quantity: float,
        is_buy: bool,
    ) -> None:
        if is_buy:
            entry_cost = delta * execution_price + fee
            self._position_cost_basis += entry_cost
            self._trade_entry_cost_total += entry_cost
        else:
            self._record_sale(-delta, execution_price, fee, target_quantity)

    def _record_fill(
        self, delta: float, execution_price: float, slippage_cost: float, is_buy: bool
    ) -> None:
        self._total_slippage += slippage_cost
        self._total_turnover += abs(delta * execution_price)
        self._fill_count += 1
        if is_buy:
            self._buy_fill_count += 1
        else:
            self._sell_fill_count += 1

    def _rebalance_to_quantity(self, target_quantity: float, price: float) -> None:
        if self._should_skip_rebalance(target_quantity, price):
            return
        self._validate_target_quantity(target_quantity)
        delta = target_quantity - self._quantity
        is_buy = delta > 0.0
        execution_price, fee, slippage_cost = self._fill_costs(delta, price, is_buy)
        next_cash = self._cash - delta * execution_price - fee
        self._validate_rebalanced_cash(next_cash)
        self._record_cost_basis(delta, execution_price, fee, target_quantity, is_buy)
        self._cash = max(0.0, next_cash)
        self._quantity = target_quantity
        self._total_fees += fee
        self._record_fill(delta, execution_price, slippage_cost, is_buy)

    def _legal_action_mask(self, price: float) -> dict[str, bool]:
        equity = self._equity_at(price)
        position_fraction = self._quantity * price / equity
        has_position = self._quantity > self._quantity_tolerance(price)
        has_spendable_cash = self._cash > equity * POSITION_FRACTION_EPSILON
        holding_period_active = self._minimum_holding_period_active(price)
        can_buy = (
            has_spendable_cash
            and position_fraction < 1.0 - POSITION_FRACTION_EPSILON
            and not holding_period_active
        )
        can_sell = has_position and not holding_period_active
        return {
            "HOLD": True,
            "BUY_25": can_buy,
            "BUY_50": can_buy,
            "BUY_100": can_buy,
            "SELL_25": can_sell,
            "SELL_50": can_sell,
            "SELL_100": can_sell,
        }

    def _target_for_buy(self, size: float, price: float) -> bool:
        if not 0.0 < size <= 1.0 or self._cash <= 0.0:
            return False
        target_notional = size * self._cash
        cost_adjusted_price = (
            price
            * (1.0 + self.slippage_bps_per_side / 10_000.0)
            * (1.0 + self.fee_bps_per_side / 10_000.0)
        )
        additional_quantity = target_notional / cost_adjusted_price
        target_quantity = self._quantity + additional_quantity
        if additional_quantity <= self._quantity_tolerance(price):
            return False
        previous_quantity = self._quantity
        self._rebalance_to_quantity(target_quantity, price)
        return self._quantity > previous_quantity

    def _validated_action_values(
        self, action: Int64Array | np.int64 | int
    ) -> tuple[int, str, str, float, int | None]:
        values = np.asarray(action)
        if values.shape not in ((), (1,)) or values.dtype.kind not in ("i", "u"):
            raise ValueError("action_must_be_one_integer_value")
        action_index = (
            int(values.reshape(-1)[0]) if values.shape else int(values.item())
        )
        if not 0 <= action_index < len(ACTION_NAMES):
            raise ValueError("discrete_action_index_out_of_range")
        action_name = ACTION_NAMES[action_index]
        requested_action = ACTION_SIDE_BY_INDEX[action_index]
        size = ACTION_SIZE_BY_INDEX[action_index]
        size_index = (
            None
            if action_index == ACTION_HOLD
            else (
                action_index - 1
                if action_index <= ACTION_BUY_100
                else action_index - ACTION_SELL_25
            )
        )
        return action_index, action_name, requested_action, size, size_index

    def _execute_buy_action(
        self,
        action_name: str,
        size: float,
        next_open: float,
        action_mask: dict[str, bool],
        current_price: float,
    ) -> tuple[str, str, dict[str, bool], str | None]:
        if not action_mask[action_name]:
            equity = self._equity_at(current_price)
            position_fraction = self._quantity * current_price / equity
            if self._minimum_holding_period_active(current_price):
                reason = "minimum_holding_period_active"
            else:
                reason = (
                    "position_fully_allocated"
                    if position_fraction >= 1.0 - POSITION_FRACTION_EPSILON
                    else "insufficient_spendable_cash"
                )
            return "BUY", "MASKED_BUY", action_mask, reason
        was_flat = self._quantity <= self._quantity_tolerance(current_price)
        if not self._target_for_buy(size, next_open):
            return (
                "BUY",
                "NOOP_BUY",
                action_mask,
                "buy_target_not_above_current_position",
            )
        self._last_buy_index = self._index + 1
        if was_flat:
            self._trade_entry_index = self._index + 1
            self._trade_entry_reference_price = float(next_open)
            self._highest_price_since_entry = float(next_open)
            self._trailing_stop_price = None
            self._trailing_stop_activated = False
        return "BUY", "BUY", action_mask, None

    def _sell_target_quantity(self, size: float, next_open: float) -> float | None:
        if not 0.0 < size <= 1.0:
            return None
        target_quantity = self._quantity * (1.0 - size)
        if target_quantity >= self._quantity - self._quantity_tolerance(next_open):
            return None
        return target_quantity

    def _execute_sell_action(
        self,
        action_name: str,
        size: float,
        next_open: float,
        action_mask: dict[str, bool],
    ) -> tuple[str, str, dict[str, bool], str | None]:
        if not action_mask[action_name]:
            reason = (
                "minimum_holding_period_active"
                if self._minimum_holding_period_active(next_open)
                else "no_position"
            )
            return "SELL", "MASKED_SELL", action_mask, reason
        target_quantity = self._sell_target_quantity(size, next_open)
        if target_quantity is None:
            return "SELL", "MASKED_SELL", action_mask, "sell_would_not_reduce_position"
        self._active_exit_reason = "policy_sell"
        try:
            self._rebalance_to_quantity(target_quantity, next_open)
        finally:
            self._active_exit_reason = None
        if self._quantity <= self._quantity_tolerance(next_open):
            self._last_buy_index = None
        return "SELL", "SELL", action_mask, None

    def _apply_action(
        self,
        action: Int64Array | np.int64 | int,
        current_price: float,
        next_open: float,
    ) -> tuple[str, str, dict[str, bool], str | None, int, str, float, int | None]:
        action_index, action_name, requested_action, size, size_index = (
            self._validated_action_values(action)
        )
        action_mask = self._legal_action_mask(current_price)
        if requested_action == "BUY":
            _, executed_action, mask, reason = self._execute_buy_action(
                action_name, size, next_open, action_mask, current_price
            )
        elif requested_action == "SELL":
            _, executed_action, mask, reason = self._execute_sell_action(
                action_name, size, next_open, action_mask
            )
        else:
            executed_action, mask, reason = "HOLD", action_mask, None
        return (
            requested_action,
            executed_action,
            mask,
            reason,
            action_index,
            action_name,
            size,
            size_index,
        )

    def _update_drawdown(self, bar: Float64Array) -> float:
        high_equity = self._equity_at(float(bar[HIGH_COLUMN]))
        self._peak_equity = max(self._peak_equity, high_equity)
        low_equity = self._equity_at(float(bar[LOW_COLUMN]))
        intrabar_drawdown = max(
            0.0, (self._peak_equity - low_equity) / self._peak_equity
        )
        self._max_drawdown = max(self._max_drawdown, intrabar_drawdown)

        close_equity = self._equity_at(float(bar[CLOSE_COLUMN]))
        self._peak_equity = max(self._peak_equity, close_equity)
        close_drawdown = max(
            0.0, (self._peak_equity - close_equity) / self._peak_equity
        )
        self._current_drawdown = close_drawdown
        self._max_drawdown = max(self._max_drawdown, close_drawdown)
        return max(intrabar_drawdown, close_drawdown)

    def _close_for_trailing_stop(self, fill_price: float) -> None:
        if self._quantity <= self._quantity_tolerance(fill_price):
            return
        self._active_exit_reason = "trailing_stop"
        try:
            self._rebalance_to_quantity(0.0, fill_price)
        finally:
            self._active_exit_reason = None
        self._last_buy_index = None
        self._trailing_stop_trigger_count += 1
        self._last_trailing_stop_triggered = True

    def _apply_gap_trailing_stop(self, next_open: float) -> bool:
        if (
            self._quantity <= self._quantity_tolerance(next_open)
            or self._trailing_stop_price is None
            or next_open > self._trailing_stop_price
        ):
            return False
        self._close_for_trailing_stop(next_open)
        return True

    def _apply_intrabar_trailing_stop(self, bar: Float64Array) -> bool:
        stop_price = self._trailing_stop_price
        if (
            self._quantity <= self._quantity_tolerance(float(bar[OPEN_COLUMN]))
            or stop_price is None
            or float(bar[LOW_COLUMN]) > stop_price
        ):
            return False
        fill_price = min(float(bar[OPEN_COLUMN]), stop_price)
        self._close_for_trailing_stop(fill_price)
        return True

    def _update_trailing_stop_from_completed_bar(self, bar: Float64Array) -> None:
        if (
            self._quantity <= self._quantity_tolerance(float(bar[CLOSE_COLUMN]))
            or self._trade_entry_reference_price is None
        ):
            return
        high = float(bar[HIGH_COLUMN])
        if self._highest_price_since_entry is None:
            self._highest_price_since_entry = self._trade_entry_reference_price
        self._highest_price_since_entry = max(self._highest_price_since_entry, high)
        activation_price = self._trade_entry_reference_price * (
            1.0 + self.trailing_stop_activation_pct
        )
        if self._highest_price_since_entry < activation_price:
            return
        self._trailing_stop_activated = True
        candidate = self._highest_price_since_entry * (
            1.0 - self.trailing_stop_distance_pct
        )
        if self._trailing_stop_price is None:
            self._trailing_stop_price = candidate
        else:
            self._trailing_stop_price = max(self._trailing_stop_price, candidate)

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
        self._last_buy_index = None
        self._peak_equity = self.initial_equity
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_turnover = 0.0
        self._fill_count = 0
        self._buy_fill_count = 0
        self._sell_fill_count = 0
        self._position_cost_basis = 0.0
        self._position_realized_profit = 0.0
        self._last_step_realized_profit = 0.0
        self._last_step_profitable_trade_count = 0
        self._last_closed_trade_net_profit = None
        self._last_closed_trade_return_pct = None
        self._last_closed_trade_exit_reason = None
        self._active_exit_reason = None
        self._trade_entry_index = None
        self._trade_entry_reference_price = None
        self._trade_entry_cost_total = 0.0
        self._highest_price_since_entry = None
        self._trailing_stop_price = None
        self._trailing_stop_activated = False
        self._trailing_stop_trigger_count = 0
        self._last_trailing_stop_triggered = False
        self._closed_trade_holding_periods = []
        self._closed_trade_returns_percent = []
        self._closed_trade_net_profits = []
        self._closed_trade_exit_reasons = []
        self._early_exit_penalty_count = 0
        self._early_exit_penalty_total = 0.0
        self._early_exit_count = 0
        self._closed_trade_count = 0
        self._policy_closed_trade_count = 0
        self._winning_trade_count = 0
        self._losing_trade_count = 0
        self._breakeven_trade_count = 0
        self._buy_action_count = 0
        self._sell_action_count = 0
        self._hold_action_count = 0
        self._discrete_action_counts = {name: 0 for name in ACTION_NAMES}
        self._masked_buy_action_count = 0
        self._buy_noop_action_count = 0
        self._zero_size_buy_action_count = 0
        self._buy_target_not_increasing_action_count = 0
        self._masked_sell_action_count = 0
        self._minimum_holding_period_blocked_action_count = 0
        self._invalid_action_penalty_count = 0
        self._full_position_buy_penalty_count = 0
        self._no_position_change_steps = 0
        self._consecutive_no_position_change_steps = 0
        self._max_consecutive_no_position_change_steps = 0
        self._position_change_count = 0
        self._final_equity = None
        self._opportunity_cost_penalty_count = 0
        self._holding_loss_penalty_count = 0
        self._holding_profit_reward_count = 0
        self._transaction_cost_reward_penalty_total = 0.0
        self._signal_guidance_reward_count = 0
        self._signal_guidance_reward_total = 0.0
        self._signal_guidance_buy_reward_count = 0
        self._signal_guidance_sell_reward_count = 0
        self._realized_profit_reward_count = 0
        self._realized_profit_reward_total = 0.0
        return self._observation(self._index), {"start_index": self._index}

    def _requested_action_blocked_by_legal_mask(
        self,
        requested_action: str,
        action_name: str,
        action_mask: dict[str, bool],
    ) -> bool:
        return requested_action in ("BUY", "SELL") and not action_mask[action_name]

    def _invalid_action_penalty_for_request(
        self,
        requested_action: str,
        action_name: str,
        action_mask: dict[str, bool],
    ) -> tuple[bool, float]:
        applied = self._requested_action_blocked_by_legal_mask(
            requested_action, action_name, action_mask
        )
        if applied:
            self._invalid_action_penalty_count += 1
        if applied and requested_action == "BUY":
            # Defense in depth: the policy-level mask should prevent this path,
            # but the environment remains a fail-closed execution boundary.
            self._full_position_buy_penalty_count += 1
            penalty = DEFAULT_INVALID_FULL_POSITION_BUY_PENALTY
        else:
            penalty = self.invalid_action_penalty if applied else 0.0
        return applied, penalty

    def _track_position_change(
        self, previous_quantity: float, next_open: float
    ) -> tuple[float, bool, float]:
        position_change_quantity = self._quantity - previous_quantity
        position_changed = abs(position_change_quantity) > self._quantity_tolerance(
            next_open
        )
        if position_changed:
            self._consecutive_no_position_change_steps = 0
            self._position_change_count += 1
        else:
            self._no_position_change_steps += 1
            self._consecutive_no_position_change_steps += 1
            self._max_consecutive_no_position_change_steps = max(
                self._max_consecutive_no_position_change_steps,
                self._consecutive_no_position_change_steps,
            )
        inactivity_penalty = (
            self.inactivity_penalty
            if self._consecutive_no_position_change_steps > self.inactivity_grace_steps
            else 0.0
        )
        return position_change_quantity, position_changed, inactivity_penalty

    def _settle_episode_step(
        self, next_index: int, next_bar: Float64Array, risk_drawdown: float
    ) -> tuple[bool, float, float]:
        terminated = next_index == len(self._market_data) - 1
        if terminated:
            self._active_exit_reason = "terminal_liquidation"
            try:
                self._rebalance_to_quantity(0.0, float(next_bar[CLOSE_COLUMN]))
            finally:
                self._active_exit_reason = None
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
        return terminated, risk_drawdown, current_equity

    def _opportunity_cost_penalty_for_action(
        self, requested_action: str, current_price: float
    ) -> float:
        if requested_action != "HOLD":
            return 0.0
        if self._quantity > self._quantity_tolerance(current_price):
            return 0.0
        moving_average = float(self._opportunity_moving_average[self._index])
        if not math.isfinite(moving_average) or current_price <= moving_average:
            return 0.0
        self._opportunity_cost_penalty_count += 1
        return DEFAULT_OPPORTUNITY_COST_PENALTY

    def _holding_loss_penalty_for_action(
        self, requested_action: str, current_price: float
    ) -> float:
        if requested_action != "HOLD":
            return 0.0
        if self._quantity <= self._quantity_tolerance(current_price):
            return 0.0
        liquidation_price = current_price * (
            1.0 - self.slippage_bps_per_side / 10_000.0
        )
        liquidation_fee = (
            self._quantity * liquidation_price * self.fee_bps_per_side / 10_000.0
        )
        expected_net_liquidation = self._quantity * liquidation_price - liquidation_fee
        if expected_net_liquidation >= self._position_cost_basis:
            return 0.0
        self._holding_loss_penalty_count += 1
        return DEFAULT_HOLDING_LOSS_PENALTY

    def _holding_profit_reward_for_action(
        self, requested_action: str, current_price: float
    ) -> float:
        if requested_action != "HOLD":
            return 0.0
        if self._quantity <= self._quantity_tolerance(current_price):
            return 0.0
        liquidation_price = current_price * (
            1.0 - self.slippage_bps_per_side / 10_000.0
        )
        liquidation_fee = (
            self._quantity * liquidation_price * self.fee_bps_per_side / 10_000.0
        )
        expected_net_liquidation = self._quantity * liquidation_price - liquidation_fee
        if expected_net_liquidation <= self._position_cost_basis:
            return 0.0
        self._holding_profit_reward_count += 1
        return DEFAULT_HOLDING_PROFIT_REWARD

    def _guidance_target_index(self, current_price: float) -> int:
        action_mask = self._legal_action_mask(current_price)
        if self._index < MOVING_AVERAGE_WINDOW:
            return ACTION_HOLD
        current_average = float(self._opportunity_moving_average[self._index])
        previous_average = float(self._opportunity_moving_average[self._index - 1])
        if not math.isfinite(current_average) or not math.isfinite(previous_average):
            return ACTION_HOLD
        if current_average > previous_average and action_mask["BUY_25"]:
            return ACTION_BUY_25
        if current_average < previous_average and action_mask["SELL_25"]:
            return ACTION_SELL_25
        return ACTION_HOLD

    def _signal_guidance_request_is_eligible(
        self, requested_action: str, executed_action: str
    ) -> bool:
        if self.signal_guidance_reward_scale <= 0.0:
            return False
        if (
            requested_action not in ("BUY", "SELL")
            or executed_action != requested_action
        ):
            return False
        return self._index >= MOVING_AVERAGE_WINDOW

    def _causal_sma_guidance_action(
        self, requested_action: str, current_average: float, previous_average: float
    ) -> str | None:
        if current_average > previous_average:
            return "BUY" if requested_action == "BUY" else None
        if current_average < previous_average:
            return "SELL" if requested_action == "SELL" else None
        return None

    def _record_signal_guidance_reward(self, signal_action: str) -> None:
        if signal_action == "BUY":
            self._signal_guidance_buy_reward_count += 1
        else:
            self._signal_guidance_sell_reward_count += 1
        self._signal_guidance_reward_count += 1
        self._signal_guidance_reward_total += self.signal_guidance_reward_scale

    def _signal_guidance_reward_for_action(
        self, requested_action: str, executed_action: str
    ) -> tuple[float, str | None]:
        if not self._signal_guidance_request_is_eligible(
            requested_action, executed_action
        ):
            return 0.0, None
        current_average = float(self._opportunity_moving_average[self._index])
        previous_average = float(self._opportunity_moving_average[self._index - 1])
        if not math.isfinite(current_average) or not math.isfinite(previous_average):
            return 0.0, None
        signal_action = self._causal_sma_guidance_action(
            requested_action, current_average, previous_average
        )
        if signal_action is None:
            return 0.0, None
        self._record_signal_guidance_reward(signal_action)
        return self.signal_guidance_reward_scale, signal_action

    def _realized_profit_reward_for_action(self, executed_action: str) -> float:
        del executed_action
        if self._last_closed_trade_net_profit is None:
            return 0.0
        if self._last_closed_trade_exit_reason not in ("policy_sell", "trailing_stop"):
            return 0.0
        closed_profit = self._last_closed_trade_net_profit
        if closed_profit <= 0.0:
            return 0.0
        return_pct = self._last_closed_trade_return_pct
        multiplier = 1.0
        if return_pct is not None and return_pct >= 5.0:
            multiplier = 3.0
        elif return_pct is not None and return_pct >= 3.0:
            multiplier = 2.0
        reward = min(
            DEFAULT_LARGE_PROFIT_REWARD_CAP,
            DEFAULT_REALIZED_PROFIT_REWARD_SCALE
            * closed_profit
            / self.initial_equity
            * multiplier,
        )
        self._realized_profit_reward_count += 1
        self._realized_profit_reward_total += reward
        return reward

    def _early_exit_penalty_for_closed_trade(self) -> float:
        """Penalize only small positive policy exits, never loss/stop exits."""
        return_pct = self._last_closed_trade_return_pct
        if (
            self._last_closed_trade_net_profit is None
            or self._last_closed_trade_exit_reason != "policy_sell"
            or return_pct is None
            or not 0.0 <= return_pct < 1.0
        ):
            return 0.0
        self._early_exit_penalty_count += 1
        self._early_exit_penalty_total += DEFAULT_EARLY_EXIT_REWARD_PENALTY
        return DEFAULT_EARLY_EXIT_REWARD_PENALTY

    def _calculate_step_reward(
        self,
        current_equity: float,
        previous_equity: float,
        risk_drawdown: float,
        previous_drawdown: float,
        inactivity_penalty: float,
        invalid_action_penalty: float,
        opportunity_cost_penalty: float,
        holding_loss_penalty: float,
        holding_profit_reward: float,
        transaction_cost_reward_penalty: float,
        signal_guidance_reward: float,
        realized_profit_reward: float,
        early_exit_penalty: float,
        profitable_trade_bonus_count: int,
    ) -> tuple[float, float, float]:
        log_net_return = math.log(current_equity / previous_equity)
        worsening_drawdown = max(0.0, risk_drawdown - previous_drawdown)
        profitable_trade_bonus = (
            self.profitable_trade_bonus * profitable_trade_bonus_count
        )
        reward = (
            log_net_return
            - self.drawdown_increment_penalty * worsening_drawdown
            - self.drawdown_level_penalty * risk_drawdown
            - inactivity_penalty
            - invalid_action_penalty
            - opportunity_cost_penalty
            - holding_loss_penalty
            - transaction_cost_reward_penalty
            - early_exit_penalty
            + holding_profit_reward
            + signal_guidance_reward
            + realized_profit_reward
            + profitable_trade_bonus
        )
        if not math.isfinite(reward):
            raise ValueError("reward_non_finite")
        return log_net_return, profitable_trade_bonus, reward

    def _resolve_step_action(
        self,
        action: Int64Array | np.int64 | int,
        current_price: float,
        next_bar: Float64Array,
    ) -> dict[str, Any]:
        gap_stop_triggered = self._apply_gap_trailing_stop(float(next_bar[OPEN_COLUMN]))
        closed_trades_before_action = self._closed_trade_count
        mask_reason: str | None
        if gap_stop_triggered:
            (
                action_index,
                action_name,
                requested_action,
                requested_size,
                requested_size_index,
            ) = self._validated_action_values(action)
            action_mask = self._legal_action_mask(current_price)
            executed_action = "SUPPRESSED_AFTER_TRAILING_STOP"
            mask_reason = "trailing_stop_triggered_at_open"
        else:
            (
                requested_action,
                executed_action,
                action_mask,
                mask_reason,
                action_index,
                action_name,
                requested_size,
                requested_size_index,
            ) = self._apply_action(
                action,
                current_price=current_price,
                next_open=float(next_bar[OPEN_COLUMN]),
            )
        self._record_action(action_name, requested_action, executed_action, mask_reason)
        policy_closes = self._closed_trade_count - closed_trades_before_action
        if policy_closes > 0 and self._last_closed_trade_exit_reason == "policy_sell":
            self._policy_closed_trade_count += policy_closes
        return {
            "gap_stop_triggered": gap_stop_triggered,
            "action_index": action_index,
            "action_name": action_name,
            "requested_action": requested_action,
            "requested_size": requested_size,
            "requested_size_index": requested_size_index,
            "executed_action": executed_action,
            "action_mask": action_mask,
            "mask_reason": mask_reason,
            "action_blocked": executed_action.startswith("MASKED_")
            or executed_action.startswith("SUPPRESSED_"),
            "full_position_buy_penalty_applied": (
                requested_action == "BUY" and not action_mask[action_name]
            ),
        }

    def _invalid_action_penalty_for_step(
        self,
        gap_stop_triggered: bool,
        requested_action: str,
        action_name: str,
        action_mask: dict[str, bool],
    ) -> tuple[bool, float]:
        if gap_stop_triggered:
            # Risk exits override a stale policy action and must not be penalized.
            return False, 0.0
        return self._invalid_action_penalty_for_request(
            requested_action, action_name, action_mask
        )

    def _profitable_trade_bonus_count_for_step(self) -> int:
        if self._last_closed_trade_exit_reason in ("policy_sell", "trailing_stop"):
            return self._last_step_profitable_trade_count
        return 0

    def step(
        self, action: Int64Array | np.int64 | int
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        if self._index >= len(self._market_data) - 1:
            raise RuntimeError("environment_episode_already_terminated")

        current_bar = self._market_data[self._index]
        next_index = self._index + 1
        next_bar = self._market_data[next_index]
        current_price = float(current_bar[CLOSE_COLUMN])
        guidance_target_index = self._guidance_target_index(current_price)
        previous_equity = self._equity_at(current_price)
        previous_drawdown = self._current_drawdown
        previous_quantity = self._quantity
        fees_before_action = self._total_fees
        slippage_before_action = self._total_slippage
        self._last_step_realized_profit = 0.0
        self._last_step_profitable_trade_count = 0
        self._last_closed_trade_net_profit = None
        self._last_closed_trade_return_pct = None
        self._last_closed_trade_exit_reason = None
        self._last_trailing_stop_triggered = False

        action_result = self._resolve_step_action(action, current_price, next_bar)
        gap_stop_triggered = bool(action_result["gap_stop_triggered"])
        action_index = int(action_result["action_index"])
        action_name = str(action_result["action_name"])
        requested_action = str(action_result["requested_action"])
        requested_size = float(action_result["requested_size"])
        requested_size_index = int(action_result["requested_size_index"])
        executed_action = str(action_result["executed_action"])
        action_mask = action_result["action_mask"]
        mask_reason: str | None = action_result["mask_reason"]
        action_blocked = bool(action_result["action_blocked"])
        full_position_buy_penalty_applied = bool(
            action_result["full_position_buy_penalty_applied"]
        )
        risk_drawdown = self._update_drawdown(next_bar)
        intrabar_stop_triggered = self._apply_intrabar_trailing_stop(next_bar)
        self._update_trailing_stop_from_completed_bar(next_bar)
        invalid_action_penalty_applied, invalid_action_penalty = (
            self._invalid_action_penalty_for_step(
                gap_stop_triggered, requested_action, action_name, action_mask
            )
        )
        fees_this_step = max(0.0, self._total_fees - fees_before_action)
        slippage_this_step = max(0.0, self._total_slippage - slippage_before_action)
        transaction_costs_this_step = fees_this_step + slippage_this_step
        transaction_cost_reward_penalty = (
            DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER
            * transaction_costs_this_step
            / previous_equity
        )
        self._transaction_cost_reward_penalty_total += transaction_cost_reward_penalty

        holding_loss_penalty = self._holding_loss_penalty_for_action(
            requested_action, current_price
        )
        holding_profit_reward = self._holding_profit_reward_for_action(
            requested_action, current_price
        )
        signal_guidance_reward, signal_guidance_action = (
            self._signal_guidance_reward_for_action(requested_action, executed_action)
        )
        realized_profit_reward = self._realized_profit_reward_for_action(
            executed_action
        )
        early_exit_penalty = self._early_exit_penalty_for_closed_trade()
        profitable_trade_bonus_count = self._profitable_trade_bonus_count_for_step()
        opportunity_cost_penalty = self._opportunity_cost_penalty_for_action(
            requested_action, current_price
        )
        position_change_quantity, position_changed, inactivity_penalty = (
            self._track_position_change(previous_quantity, float(next_bar[OPEN_COLUMN]))
        )

        terminated, risk_drawdown, current_equity = self._settle_episode_step(
            next_index, next_bar, risk_drawdown
        )
        log_net_return, profitable_trade_bonus, reward = self._calculate_step_reward(
            current_equity,
            previous_equity,
            risk_drawdown,
            previous_drawdown,
            inactivity_penalty,
            invalid_action_penalty,
            opportunity_cost_penalty,
            holding_loss_penalty,
            holding_profit_reward,
            transaction_cost_reward_penalty,
            signal_guidance_reward,
            realized_profit_reward,
            early_exit_penalty,
            profitable_trade_bonus_count,
        )

        self._index = next_index
        info: dict[str, Any] = {
            "timestamp_ms": int(next_bar[TIMESTAMP_COLUMN]),
            "action": requested_action,
            "requested_discrete_action_index": action_index,
            "requested_discrete_action_name": action_name,
            "guidance_target_index": guidance_target_index,
            "guidance_target_action": ACTION_SIDE_BY_INDEX[guidance_target_index],
            "requested_position_size_index": requested_size_index,
            "requested_position_size_fraction": requested_size,
            "executed_action": executed_action,
            "action_blocked": action_blocked,
            "action_mask_reason": mask_reason,
            "legal_action_mask": action_mask,
            "position_holding_period_candles": self._position_holding_candles(
                float(next_bar[CLOSE_COLUMN])
            ),
            "minimum_holding_period_candles": self.minimum_holding_period_candles,
            "minimum_holding_period_active": self._minimum_holding_period_active(
                float(next_bar[CLOSE_COLUMN])
            ),
            "position_changed": position_changed,
            "position_change_quantity": float(position_change_quantity),
            "consecutive_no_position_change_steps": self._consecutive_no_position_change_steps,
            "reward_net_log_return": log_net_return,
            "reward_drawdown_increment_penalty": self.drawdown_increment_penalty
            * max(0.0, risk_drawdown - previous_drawdown),
            "reward_drawdown_level_penalty": self.drawdown_level_penalty
            * risk_drawdown,
            "reward_total": reward,
            "drawdown_for_reward": risk_drawdown,
            "reward_inactivity_penalty": inactivity_penalty,
            "reward_invalid_action_penalty": float(invalid_action_penalty),
            "full_position_buy_penalty_applied": (
                requested_action == "BUY" and not action_mask[action_name]
            ),
            "reward_full_position_buy_penalty": float(
                DEFAULT_INVALID_FULL_POSITION_BUY_PENALTY
                * full_position_buy_penalty_applied
            ),
            "invalid_action_penalty_applied": invalid_action_penalty_applied,
            "reward_opportunity_cost_penalty": opportunity_cost_penalty,
            "reward_holding_loss_penalty": holding_loss_penalty,
            "reward_holding_profit_bonus": holding_profit_reward,
            "reward_transaction_cost_penalty": transaction_cost_reward_penalty,
            "fees_this_step": fees_this_step,
            "slippage_this_step": slippage_this_step,
            "transaction_costs_this_step": transaction_costs_this_step,
            "transaction_cost_reward_multiplier": DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER,
            "reward_signal_guidance": signal_guidance_reward,
            "signal_guidance_reward_applied": signal_guidance_reward > 0.0,
            "signal_guidance_action": signal_guidance_action,
            "signal_guidance_reward_scale": self.signal_guidance_reward_scale,
            "reward_realized_profit": realized_profit_reward,
            "realized_profit_reward_applied": realized_profit_reward > 0.0,
            "reward_early_exit_penalty": early_exit_penalty,
            "early_exit_penalty_applied": early_exit_penalty > 0.0,
            "reward_profitable_trade_bonus": profitable_trade_bonus,
            "policy_closed_trade_count": self._policy_closed_trade_count,
            "realized_profit_this_step": float(self._last_step_realized_profit),
            "closed_trade_net_profit_this_step": self._last_closed_trade_net_profit,
            "closed_trade_return_percent_this_step": self._last_closed_trade_return_pct,
            "closed_trade_exit_reason_this_step": self._last_closed_trade_exit_reason,
            "trailing_stop_triggered": self._last_trailing_stop_triggered,
            "trailing_stop_triggered_at_open": gap_stop_triggered,
            "trailing_stop_triggered_intrabar": intrabar_stop_triggered,
            "trailing_stop_active": self._trailing_stop_activated,
            "trailing_stop_price": self._trailing_stop_price,
        }
        if terminated:
            info.update(self.summary())
        return self._observation(self._index), float(reward), terminated, False, info

    def _summary_win_rate(self) -> float | None:
        decisive_trades = self._winning_trade_count + self._losing_trade_count
        if decisive_trades == 0:
            return None
        return self._winning_trade_count / decisive_trades

    def _summary_average_holding_period(self) -> float | None:
        if not self._closed_trade_holding_periods:
            return None
        return float(np.mean(self._closed_trade_holding_periods))

    def _summary_average_trade_return(self, profitable: bool) -> float | None:
        if profitable:
            values = [
                return_pct
                for profit, return_pct in zip(
                    self._closed_trade_net_profits,
                    self._closed_trade_returns_percent,
                    strict=False,
                )
                if profit > 0.0
            ]
        else:
            values = [
                return_pct
                for profit, return_pct in zip(
                    self._closed_trade_net_profits,
                    self._closed_trade_returns_percent,
                    strict=False,
                )
                if profit < 0.0
            ]
        if not values:
            return None
        return float(np.mean(values))

    def _summary_profit_factor(self) -> float | None:
        gross_profit = sum(
            value for value in self._closed_trade_net_profits if value > 0.0
        )
        gross_loss = sum(
            -value for value in self._closed_trade_net_profits if value < 0.0
        )
        if gross_loss <= 0.0:
            return None
        return float(gross_profit / gross_loss)

    def _summary_hold_collapse(self) -> bool:
        total_actions = sum(self._discrete_action_counts.values())
        return (
            total_actions > 0 and self._discrete_action_counts["HOLD"] == total_actions
        )

    def _summary_exit_reason_counts(self) -> dict[str, int]:
        return {
            reason: self._closed_trade_exit_reasons.count(reason)
            for reason in sorted(set(self._closed_trade_exit_reasons))
        }

    def summary(self) -> dict[str, Any]:
        """Return accounting and behavior evidence; never observed by the policy."""

        price = float(self._market_data[self._index, CLOSE_COLUMN])
        final_equity = self._final_equity
        if final_equity is None:
            final_equity = self._equity_at(price)
        win_rate = self._summary_win_rate()
        average_holding_period = self._summary_average_holding_period()
        average_win_percent = self._summary_average_trade_return(profitable=True)
        average_loss_percent = self._summary_average_trade_return(profitable=False)
        profit_factor = self._summary_profit_factor()
        hold_collapse = self._summary_hold_collapse()
        exit_reason_counts = self._summary_exit_reason_counts()
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
            "buy_action_count": self._buy_action_count,
            "sell_action_count": self._sell_action_count,
            "hold_action_count": self._hold_action_count,
            "discrete_action_counts": dict(self._discrete_action_counts),
            "closed_trade_count": self._closed_trade_count,
            "policy_closed_trade_count": self._policy_closed_trade_count,
            "winning_trade_count": self._winning_trade_count,
            "losing_trade_count": self._losing_trade_count,
            "breakeven_trade_count": self._breakeven_trade_count,
            "win_rate": float(win_rate) if win_rate is not None else None,
            "average_holding_period_candles": average_holding_period,
            "average_win_percent": average_win_percent,
            "average_loss_percent": average_loss_percent,
            "profit_factor": profit_factor,
            "early_exit_count": self._early_exit_count,
            "hold_collapse": hold_collapse,
            "flags": {"HOLD_COLLAPSE": hold_collapse},
            "closed_trade_exit_reason_counts": exit_reason_counts,
            "early_exit_penalty_count": self._early_exit_penalty_count,
            "early_exit_penalty_total": float(self._early_exit_penalty_total),
            "trailing_stop_trigger_count": self._trailing_stop_trigger_count,
            "trailing_stop_distance_pct": self.trailing_stop_distance_pct,
            "trailing_stop_activation_pct": self.trailing_stop_activation_pct,
            "masked_buy_action_count": self._masked_buy_action_count,
            "buy_noop_action_count": self._buy_noop_action_count,
            "zero_size_buy_action_count": self._zero_size_buy_action_count,
            "buy_target_not_increasing_action_count": (
                self._buy_target_not_increasing_action_count
            ),
            "masked_sell_action_count": self._masked_sell_action_count,
            "minimum_holding_period_blocked_action_count": (
                self._minimum_holding_period_blocked_action_count
            ),
            "minimum_holding_period_candles": self.minimum_holding_period_candles,
            "invalid_action_penalty_count": self._invalid_action_penalty_count,
            "full_position_buy_penalty_count": self._full_position_buy_penalty_count,
            "full_position_buy_penalty_total": float(
                self._full_position_buy_penalty_count
                * DEFAULT_INVALID_FULL_POSITION_BUY_PENALTY
            ),
            "opportunity_cost_penalty_count": self._opportunity_cost_penalty_count,
            "holding_loss_penalty_count": self._holding_loss_penalty_count,
            "holding_profit_reward_count": self._holding_profit_reward_count,
            "transaction_cost_reward_penalty_total": float(
                self._transaction_cost_reward_penalty_total
            ),
            "transaction_cost_reward_multiplier": DEFAULT_TRANSACTION_COST_REWARD_MULTIPLIER,
            "signal_guidance_reward_count": self._signal_guidance_reward_count,
            "signal_guidance_reward_total": self._signal_guidance_reward_total,
            "signal_guidance_buy_reward_count": self._signal_guidance_buy_reward_count,
            "signal_guidance_sell_reward_count": self._signal_guidance_sell_reward_count,
            "realized_profit_reward_count": self._realized_profit_reward_count,
            "realized_profit_reward_total": self._realized_profit_reward_total,
            "no_position_change_steps": self._no_position_change_steps,
            "max_consecutive_no_position_change_steps": (
                self._max_consecutive_no_position_change_steps
            ),
            "position_change_count": self._position_change_count,
            "inactivity_grace_steps": self.inactivity_grace_steps,
        }
