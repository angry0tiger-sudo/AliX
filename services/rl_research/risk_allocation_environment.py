"""Offline, signal-gated Gymnasium environment for binary signal filtering.

At the start of each contiguous primary-signal episode, PPO chooses SKIP or
TAKE. A TAKE is held for that signal episode and forcibly flattened when the
primary signal turns off. Fills occur at the next hourly open. The environment
never creates directional signals, shorts, leverages, or submits orders.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

DEFAULT_INITIAL_EQUITY = 10_000.0
FEE_BPS_PER_SIDE = 25.0
SLIPPAGE_BPS_PER_SIDE = 0.0
ROUND_TRIP_COST_BPS = 2.0 * (FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE)
FEE_RATE = FEE_BPS_PER_SIDE / 10_000.0
ACTION_FRACTIONS = (0.0, 1.0)
ACTION_NAMES = ("SKIP_SIGNAL", "TAKE_SIGNAL")
ACTION_SKIP = 0
ACTION_TAKE = 1
BAR_INTERVAL_MS = 3_600_000
MARKET_COLUMN_COUNT = 6
TIMESTAMP_COLUMN = 0
OPEN_COLUMN = 1
HIGH_COLUMN = 2
LOW_COLUMN = 3
CLOSE_COLUMN = 4
VOLUME_COLUMN = 5
RECENT_RETURN_FIELDS = ("return_1", "return_6", "return_24")
OBSERVATION_FIELDS = (
    "atr_fraction",
    "drawdown_fraction",
    "current_position_fraction",
    "return_1",
    "return_6",
    "return_24",
    "baseline_signal_probability",
    "primary_buy_signal",
    "signal_episode_decision_pending",
)
PRIMARY_SIGNAL_OBSERVATION_INDEX = OBSERVATION_FIELDS.index("primary_buy_signal")
SORTINO_DOWNSIDE_FLOOR = 0.001
SORTINO_REWARD_CLIP = 5.0
DRAWDOWN_LIMIT = 0.15
DRAWDOWN_EXCESS_PENALTY = 2.0
DRAWDOWN_INCREMENT_PENALTY = 0.5
DRAWDOWN_SEVERE_PENALTY = 4.0
POSITION_EPSILON = 1e-12

Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name}_must_be_positive_and_finite")
    return number


def _validate_market_shape_and_finiteness(data: Float64Array) -> None:
    if data.ndim != 2 or data.shape[1] != MARKET_COLUMN_COUNT:
        raise ValueError("market_data_must_have_six_columns")
    if data.shape[0] < 3:
        raise ValueError("market_data_requires_at_least_three_rows")
    if not np.isfinite(data).all():
        raise ValueError("market_data_contains_non_finite_values")


def _validate_market_timestamps(data: Float64Array, interval_ms: int) -> None:
    timestamps = data[:, TIMESTAMP_COLUMN]
    if (
        np.any(timestamps <= 0.0)
        or np.any(timestamps != np.floor(timestamps))
        or np.any(np.diff(timestamps) != interval_ms)
    ):
        raise ValueError("market_timestamps_must_be_contiguous_integer_epochs")


def _validate_market_price_geometry(data: Float64Array) -> None:
    opening = data[:, OPEN_COLUMN]
    high = data[:, HIGH_COLUMN]
    low = data[:, LOW_COLUMN]
    close = data[:, CLOSE_COLUMN]
    volume = data[:, VOLUME_COLUMN]
    if np.any(np.minimum.reduce((opening, high, low, close)) <= 0.0):
        raise ValueError("market_prices_must_be_positive")
    if np.any(high < np.maximum(opening, close)):
        raise ValueError("market_high_below_open_or_close")
    if np.any(low > np.minimum(opening, close)) or np.any(low > high):
        raise ValueError("market_low_above_open_close_or_high")
    if np.any(volume < 0.0):
        raise ValueError("market_volume_must_be_non_negative")


def _validate_market_data(data: Float64Array, interval_ms: int) -> None:
    _validate_market_shape_and_finiteness(data)
    _validate_market_timestamps(data, interval_ms)
    _validate_market_price_geometry(data)


def _validate_vector(
    name: str, values: Float64Array, row_count: int, *, non_negative: bool = False
) -> None:
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError(f"{name}_length_or_shape_invalid")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}_must_be_finite")
    if non_negative and np.any(values < 0.0):
        raise ValueError(f"{name}_must_be_non_negative")


def _coerce_signal(values: NDArray[Any], row_count: int) -> BoolArray:
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError("primary_signal_length_or_shape_invalid")
    if not np.isin(values, (0, 1, False, True)).all():
        raise ValueError("primary_signal_must_be_boolean")
    return values.astype(np.bool_, copy=True)


def _validate_bar_interval(bar_interval_ms: int) -> None:
    if isinstance(bar_interval_ms, bool) or not isinstance(bar_interval_ms, int):
        raise ValueError("bar_interval_ms_must_be_positive_integer")
    if bar_interval_ms <= 0:
        raise ValueError("bar_interval_ms_must_be_positive_integer")


def _validate_episode_steps(episode_steps: int | None, row_count: int) -> None:
    if episode_steps is None:
        return
    if isinstance(episode_steps, bool) or not isinstance(episode_steps, int):
        raise ValueError("episode_steps_must_be_at_least_two")
    if episode_steps < 2:
        raise ValueError("episode_steps_must_be_at_least_two")
    if episode_steps >= row_count:
        raise ValueError("episode_steps_must_be_shorter_than_market_data")


def _validate_environment_configuration(
    row_count: int,
    bar_interval_ms: int,
    random_start: bool,
    episode_steps: int | None,
) -> None:
    _validate_bar_interval(bar_interval_ms)
    if not isinstance(random_start, bool):
        raise ValueError("random_start_must_be_boolean")
    _validate_episode_steps(episode_steps, row_count)
    if ROUND_TRIP_COST_BPS != 50.0:
        raise RuntimeError("nobitex_round_trip_cost_contract_invalid")


def _validate_recent_returns(returns: Float64Array, row_count: int) -> None:
    if returns.ndim != 2 or returns.shape != (row_count, 3):
        raise ValueError("recent_returns_must_have_three_columns")
    if not np.isfinite(returns).all():
        raise ValueError("recent_returns_must_be_finite")
    if np.any(np.abs(returns) > 1.0):
        raise ValueError("recent_returns_out_of_range")


def _validate_signal_probabilities(probability: Float64Array, row_count: int) -> None:
    _validate_vector("baseline_signal_probability", probability, row_count)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("baseline_signal_probability_out_of_range")


def _prepare_environment_inputs(
    market_data: Float64Array,
    atr_fraction: Float64Array,
    recent_returns: Float64Array,
    primary_signal: NDArray[Any],
    baseline_signal_probability: Float64Array,
    bar_interval_ms: int,
) -> tuple[Float64Array, Float64Array, Float64Array, BoolArray, Float64Array]:
    data = np.asarray(market_data, dtype=np.float64)
    row_count = len(data)
    atr = np.asarray(atr_fraction, dtype=np.float64)
    returns = np.asarray(recent_returns, dtype=np.float64)
    probability = np.asarray(baseline_signal_probability, dtype=np.float64)
    signal = _coerce_signal(np.asarray(primary_signal), row_count)
    _validate_market_data(data, bar_interval_ms)
    _validate_vector("atr_fraction", atr, row_count, non_negative=True)
    _validate_recent_returns(returns, row_count)
    _validate_signal_probabilities(probability, row_count)
    return (
        np.ascontiguousarray(data),
        np.ascontiguousarray(atr),
        np.ascontiguousarray(returns),
        np.ascontiguousarray(signal),
        np.ascontiguousarray(probability),
    )


def _valid_random_signal_episode_starts(
    primary_signal: BoolArray, maximum_start: int
) -> NDArray[np.int64]:
    """Return window starts that never begin inside an existing signal episode."""
    valid = np.ones(maximum_start + 1, dtype=np.bool_)
    if maximum_start > 0:
        valid[1:] = ~(
            primary_signal[1 : maximum_start + 1] & primary_signal[:maximum_start]
        )
    return np.flatnonzero(valid).astype(np.int64)


def _new_observation_space() -> Any:
    return spaces.Box(
        low=np.asarray(
            [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        high=np.asarray(
            [np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
        ),
        dtype=np.float32,
    )


class RiskAllocationEnv(gym.Env[Float32Array, np.int64]):
    """Long-only binary filter over contiguous primary-signal episodes."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        market_data: Float64Array,
        atr_fraction: Float64Array,
        recent_returns: Float64Array,
        primary_signal: NDArray[Any],
        baseline_signal_probability: Float64Array,
        *,
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
        bar_interval_ms: int = BAR_INTERVAL_MS,
        random_start: bool = False,
        episode_steps: int | None = None,
    ) -> None:
        super().__init__()
        row_count = len(market_data)
        _validate_environment_configuration(
            row_count, bar_interval_ms, random_start, episode_steps
        )
        data, atr, returns, signal, probability = _prepare_environment_inputs(
            market_data,
            atr_fraction,
            recent_returns,
            primary_signal,
            baseline_signal_probability,
            bar_interval_ms,
        )
        equity = _finite_positive(initial_equity, "initial_equity")

        self._market = data
        self._atr_fraction = atr
        self._recent_returns = returns
        self._primary_signal = signal
        self._baseline_probability = probability
        self.initial_equity = equity
        self.bar_interval_ms = bar_interval_ms
        self.random_start = random_start
        self.episode_steps = episode_steps
        self.action_space = spaces.Discrete(len(ACTION_FRACTIONS))
        self.observation_space = _new_observation_space()
        self._reset_portfolio()

    def _reset_portfolio(self) -> None:
        self._index = 0
        self._episode_end = len(self._market) - 1
        self._cash = self.initial_equity
        self._quantity = 0.0
        self._position_basis = 0.0
        self._trade_invested_total = 0.0
        self._trade_realized_pnl = 0.0
        self._closed_trade_returns: list[float] = []
        self._peak_equity = self.initial_equity
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._downside_squared_sum = 0.0
        self._downside_observations = 0
        self._latest_downside_deviation = 0.0
        self._total_fees = 0.0
        self._total_turnover = 0.0
        self._position_adjustments = 0
        self._signal_episode_latched = False
        self._signal_episode_taken = False
        self._signal_episodes_seen = 0
        self._signal_episodes_taken = 0
        self._signal_episodes_skipped = 0
        self._ignored_actions = 0
        self._terminated = False
        self._last_info: dict[str, Any] = {}

    def _equity(self, price: float) -> float:
        return self._cash + self._quantity * price

    def _position_fraction(self, price: float) -> float:
        equity = self._equity(price)
        if equity <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        return min(1.0, max(0.0, self._quantity * price / equity))

    def _observation(self) -> Float32Array:
        row = self._market[self._index]
        price = float(row[CLOSE_COLUMN])
        values = np.asarray(
            (
                self._atr_fraction[self._index],
                self._current_drawdown,
                self._position_fraction(price),
                *self._recent_returns[self._index],
                self._baseline_probability[self._index],
                float(self._primary_signal[self._index]),
                float(
                    self._primary_signal[self._index]
                    and not self._signal_episode_latched
                ),
            ),
            dtype=np.float32,
        )
        if values.shape != (len(OBSERVATION_FIELDS),) or not np.isfinite(values).all():
            raise RuntimeError("risk_observation_invalid")
        return values

    def _close_trade_if_flat(self) -> None:
        if self._quantity > POSITION_EPSILON:
            return
        self._quantity = 0.0
        if self._trade_invested_total > 0.0:
            result = self._trade_realized_pnl / self._trade_invested_total
            if not math.isfinite(result) or result <= -1.0:
                raise RuntimeError("closed_trade_return_invalid")
            self._closed_trade_returns.append(result)
        self._position_basis = 0.0
        self._trade_invested_total = 0.0
        self._trade_realized_pnl = 0.0

    def _buy(self, quantity: float, price: float) -> tuple[float, float]:
        if quantity <= POSITION_EPSILON:
            return 0.0, 0.0
        notional = quantity * price
        fee = notional * FEE_RATE
        total_cost = notional + fee
        if total_cost > self._cash + 1e-8:
            raise RuntimeError("buy_exceeds_available_cash")
        if self._quantity <= POSITION_EPSILON:
            self._position_basis = 0.0
            self._trade_invested_total = 0.0
            self._trade_realized_pnl = 0.0
        self._cash -= total_cost
        self._quantity += quantity
        self._position_basis += total_cost
        self._trade_invested_total += total_cost
        self._total_fees += fee
        self._total_turnover += notional
        self._position_adjustments += 1
        return fee, notional

    def _sell(self, quantity: float, price: float) -> tuple[float, float]:
        sold = min(max(0.0, quantity), self._quantity)
        if sold <= POSITION_EPSILON:
            return 0.0, 0.0
        notional = sold * price
        fee = notional * FEE_RATE
        allocated_basis = self._position_basis * (sold / self._quantity)
        self._cash += notional - fee
        self._quantity -= sold
        self._position_basis -= allocated_basis
        self._trade_realized_pnl += notional - fee - allocated_basis
        self._total_fees += fee
        self._total_turnover += notional
        self._position_adjustments += 1
        self._close_trade_if_flat()
        return fee, notional

    def _rebalance(self, target_fraction: float, price: float) -> tuple[float, float]:
        equity = self._equity(price)
        if equity <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        desired_quantity = target_fraction * equity / price
        delta = desired_quantity - self._quantity
        if delta > POSITION_EPSILON:
            affordable = self._cash / (price * (1.0 + FEE_RATE))
            return self._buy(min(delta, max(0.0, affordable)), price)
        if delta < -POSITION_EPSILON:
            return self._sell(-delta, price)
        return 0.0, 0.0

    def _update_intrabar_drawdown(self, bar: Float64Array) -> float:
        high_equity = self._equity(float(bar[HIGH_COLUMN]))
        low_equity = self._equity(float(bar[LOW_COLUMN]))
        close_equity = self._equity(float(bar[CLOSE_COLUMN]))
        if min(high_equity, low_equity, close_equity) <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        self._peak_equity = max(self._peak_equity, high_equity)
        intrabar = max(0.0, 1.0 - low_equity / self._peak_equity)
        closing = max(0.0, 1.0 - close_equity / self._peak_equity)
        self._current_drawdown = max(intrabar, closing)
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
        return close_equity

    def _sortino_reward(self, log_return: float) -> float:
        downside = min(0.0, log_return)
        self._downside_squared_sum += downside * downside
        self._downside_observations += 1
        self._latest_downside_deviation = math.sqrt(
            self._downside_squared_sum / self._downside_observations
        )
        denominator = max(self._latest_downside_deviation, SORTINO_DOWNSIDE_FLOOR)
        score = log_return / denominator
        return max(-SORTINO_REWARD_CLIP, min(SORTINO_REWARD_CLIP, score))

    def _reward(self, log_return: float, prior_drawdown: float) -> tuple[float, float]:
        sortino_component = self._sortino_reward(log_return)
        drawdown_increase = max(0.0, self._current_drawdown - prior_drawdown)
        excess = max(0.0, self._current_drawdown - DRAWDOWN_LIMIT)
        penalty = (
            DRAWDOWN_INCREMENT_PENALTY * drawdown_increase
            + DRAWDOWN_EXCESS_PENALTY * excess
            + DRAWDOWN_SEVERE_PENALTY * excess * excess
        )
        return sortino_component - penalty, penalty

    def _metrics(self, equity: float) -> dict[str, Any]:
        returns = np.asarray(self._closed_trade_returns, dtype=np.float64)
        positive = float(returns[returns > 0.0].sum())
        negative = float(-returns[returns < 0.0].sum())
        pf = positive / negative if negative > 0.0 else None
        z_score: float | None = None
        if len(returns) >= 2:
            deviation = float(returns.std(ddof=1))
            if deviation > 0.0:
                z_score = float(returns.mean() / (deviation / math.sqrt(len(returns))))
        return {
            "final_equity": float(equity),
            "net_return_fraction": float(equity / self.initial_equity - 1.0),
            "net_profit_factor": pf,
            "net_profit_factor_infinite_no_losses": bool(
                len(returns) > 0 and negative == 0.0 and positive > 0.0
            ),
            "win_rate": float((returns > 0.0).mean()) if len(returns) else None,
            "max_drawdown": float(self._max_drawdown),
            "statistical_z_score": z_score,
            "trade_count": int(len(returns)),
            "total_fees_quote": float(self._total_fees),
            "total_turnover_quote": float(self._total_turnover),
            "position_adjustments": int(self._position_adjustments),
            "candidate_signal_episode_count": int(self._signal_episodes_seen),
            "accepted_signal_episode_count": int(self._signal_episodes_taken),
            "skipped_signal_episode_count": int(self._signal_episodes_skipped),
            "signal_acceptance_rate": (
                float(self._signal_episodes_taken / self._signal_episodes_seen)
                if self._signal_episodes_seen
                else None
            ),
            "ignored_action_count": int(self._ignored_actions),
            "mean_trade_return": float(returns.mean()) if len(returns) else None,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Float32Array, dict[str, Any]]:
        if options not in (None, {}):
            raise ValueError("reset_options_not_supported")
        super().reset(seed=seed)
        self._reset_portfolio()
        if self.random_start:
            maximum_start = len(self._market) - int(self.episode_steps or 2) - 1
            candidates = _valid_random_signal_episode_starts(
                self._primary_signal, maximum_start
            )
            self._index = int(self.np_random.choice(candidates))
        if self.episode_steps is not None:
            self._episode_end = min(
                self._index + self.episode_steps, len(self._market) - 1
            )
        else:
            self._episode_end = len(self._market) - 1
        if self._episode_end - self._index < 2:
            raise ValueError("episode_has_insufficient_market_rows")
        return self._observation(), {
            "observation_fields": list(OBSERVATION_FIELDS),
            "action_fractions": list(ACTION_FRACTIONS),
            "action_names": list(ACTION_NAMES),
            "signal_episode_policy": "ONE_DECISION_PER_SIGNAL_EPISODE; FORCE_FLAT_WHEN_SIGNAL_OFF",
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        }

    def _validate_action(self, action: Any) -> int:
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("risk_action_must_be_integer")
        value = int(action)
        if not self.action_space.contains(value):
            raise ValueError("risk_action_out_of_range")
        return value

    def _apply_signal_episode_decision(
        self, action_index: int, signal: bool
    ) -> tuple[bool, float]:
        signal_episode_started = signal and not self._signal_episode_latched
        if signal_episode_started:
            self._signal_episode_latched = True
            self._signal_episodes_seen += 1
            self._signal_episode_taken = action_index == ACTION_TAKE
            if self._signal_episode_taken:
                self._signal_episodes_taken += 1
            else:
                self._signal_episodes_skipped += 1
        elif signal:
            self._ignored_actions += 1
        else:
            self._signal_episode_latched = False
            self._signal_episode_taken = False
            self._ignored_actions += 1
        applied_target = (1.0 if self._signal_episode_taken else 0.0) if signal else 0.0
        return signal_episode_started, applied_target

    def _liquidate_terminal_position(
        self,
        next_bar: Float64Array,
        terminated: bool,
        fee: float,
        notional: float,
        close_equity: float,
    ) -> tuple[float, float, float]:
        if terminated and self._quantity > POSITION_EPSILON:
            terminal_price = float(next_bar[CLOSE_COLUMN])
            terminal_fee, terminal_notional = self._sell(self._quantity, terminal_price)
            fee += terminal_fee
            notional += terminal_notional
            close_equity = self._equity(terminal_price)
            self._peak_equity = max(self._peak_equity, close_equity)
            self._current_drawdown = max(
                self._current_drawdown, 1.0 - close_equity / self._peak_equity
            )
            self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
        return fee, notional, close_equity

    def _calculate_step_reward(
        self, previous_equity: float, close_equity: float, prior_drawdown: float
    ) -> tuple[float, float, float]:
        if previous_equity <= 0.0 or close_equity <= 0.0:
            raise RuntimeError("portfolio_equity_non_positive")
        log_return = math.log(close_equity / previous_equity)
        reward, drawdown_penalty = self._reward(log_return, prior_drawdown)
        return log_return, reward, drawdown_penalty

    def _build_step_info(
        self,
        next_bar: Float64Array,
        action_index: int,
        signal: bool,
        signal_episode_started: bool,
        proposed_target: float,
        applied_target: float,
        log_return: float,
        fee: float,
        notional: float,
        drawdown_penalty: float,
        close_equity: float,
    ) -> dict[str, Any]:
        return {
            **self._metrics(close_equity),
            "timestamp_ms": int(next_bar[TIMESTAMP_COLUMN]),
            "primary_buy_signal": signal,
            "signal_episode_started": bool(signal_episode_started),
            "signal_episode_taken": bool(
                signal_episode_started and self._signal_episode_taken
            ),
            "action_effective": bool(signal_episode_started),
            "action_name": ACTION_NAMES[action_index],
            "baseline_signal_probability": float(
                self._baseline_probability[self._index - 1]
            ),
            "proposed_target_fraction": float(proposed_target),
            "applied_target_fraction": float(applied_target),
            "current_position_fraction": self._position_fraction(
                float(next_bar[CLOSE_COLUMN])
            ),
            "current_drawdown": float(self._current_drawdown),
            "log_return_net_of_costs": float(log_return),
            "fill_fee_quote": float(fee),
            "fill_notional_quote": float(notional),
            "drawdown_penalty": float(drawdown_penalty),
            "risk_only_action": True,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
        }

    def step(
        self, action: Any
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("episode_already_terminated")
        action_index = self._validate_action(action)
        current_price = float(self._market[self._index, CLOSE_COLUMN])
        next_index = self._index + 1
        next_bar = self._market[next_index]
        execution_price = float(next_bar[OPEN_COLUMN])
        proposed_target = ACTION_FRACTIONS[action_index]
        signal = bool(self._primary_signal[self._index])
        signal_episode_started, applied_target = self._apply_signal_episode_decision(
            action_index, signal
        )
        previous_equity = self._equity(current_price)
        prior_drawdown = self._current_drawdown
        fee, notional = self._rebalance(applied_target, execution_price)
        close_equity = self._update_intrabar_drawdown(next_bar)
        terminated = next_index >= self._episode_end
        fee, notional, close_equity = self._liquidate_terminal_position(
            next_bar, terminated, fee, notional, close_equity
        )
        log_return, reward, drawdown_penalty = self._calculate_step_reward(
            previous_equity, close_equity, prior_drawdown
        )
        self._index = next_index
        self._terminated = terminated
        info = self._build_step_info(
            next_bar,
            action_index,
            signal,
            signal_episode_started,
            proposed_target,
            applied_target,
            log_return,
            fee,
            notional,
            drawdown_penalty,
            close_equity,
        )
        self._last_info = info
        return self._observation(), float(reward), terminated, False, info

    def get_metrics(self) -> dict[str, Any]:
        price = float(self._market[self._index, CLOSE_COLUMN])
        return self._metrics(self._equity(price))

    @property
    def closed_trade_returns(self) -> tuple[float, ...]:
        return tuple(self._closed_trade_returns)

    def close(self) -> None:
        self._terminated = True
