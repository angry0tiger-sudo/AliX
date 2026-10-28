"""Behavioral tests for the offline RL trading environment and accounting."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from services.rl_research.rl_trading_environment import (
    ACTION_BUY_100,
    ACTION_BUY_25,
    ACTION_HOLD,
    ACTION_SELL_100,
    ACTION_SELL_25,
    BTCUSDTTradingEnv,
    Int64Array,
    MarketObservationScaler,
    _finite_numeric_parameter,
    _validate_cost_basis_points,
    _validate_drawdown_penalties,
    _validate_environment_configuration,
    _validate_minimum_holding_period_candles,
    _validate_portfolio_state_option,
    _validate_reward_shaping,
    _validate_signal_guidance_reward_scale,
    _validate_trailing_stop_configuration,
    validate_market_data,
)

Float64Array = NDArray[np.float64]


def market_data(rows: int = 64, trend_per_candle: float = 0.5) -> Float64Array:
    """Create deterministic, contiguous synthetic OHLCV candles."""
    index = np.arange(rows, dtype=np.float64)
    close = 100.0 + trend_per_candle * index
    open_price = close.copy()
    high = close + 1.0
    low = close - 1.0
    volume = 10.0 + index
    timestamp = 1_700_000_000_000.0 + index * 60_000.0
    return np.column_stack((timestamp, open_price, high, low, close, volume)).astype(
        np.float64
    )


def make_env(
    *,
    rows: int = 64,
    trend_per_candle: float = 0.5,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    minimum_holding_period_candles: int = 1,
    include_portfolio_state: bool = True,
    random_start: bool = False,
    min_remaining_steps: int = 1,
    signal_guidance_reward_scale: float = 0.0,
    trailing_stop_distance_pct: float = 0.02,
    trailing_stop_activation_pct: float = 0.03,
) -> BTCUSDTTradingEnv:
    return BTCUSDTTradingEnv(
        market_data(rows, trend_per_candle),
        initial_equity=10_000.0,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        minimum_holding_period_candles=minimum_holding_period_candles,
        include_portfolio_state=include_portfolio_state,
        random_start=random_start,
        min_remaining_steps=min_remaining_steps,
        signal_guidance_reward_scale=signal_guidance_reward_scale,
        trailing_stop_distance_pct=trailing_stop_distance_pct,
        trailing_stop_activation_pct=trailing_stop_activation_pct,
    )


def buy_full_position(env: BTCUSDTTradingEnv) -> None:
    env.reset(seed=7)
    current_price = float(env._market_data[0, 4])
    next_open = float(env._market_data[1, 1])
    mask = env._legal_action_mask(current_price)
    result = env._execute_buy_action("BUY_100", 1.0, next_open, mask, current_price)
    assert result[1] == "BUY"
    assert env._quantity > 0.0
    assert env._trade_entry_cost_total > 0.0


def test_market_validation_rejects_bad_shape_timestamps_and_candles() -> None:
    with pytest.raises(ValueError, match="market_data_must_have_six_columns"):
        validate_market_data(np.zeros((3, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="market_data_must_have_six_columns"):
        validate_market_data(np.zeros(6, dtype=np.float64))
    with pytest.raises(ValueError, match="market_data_requires_at_least_two_rows"):
        validate_market_data(market_data(1))
    bad = market_data()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="market_data_contains_non_finite_values"):
        validate_market_data(bad)

    for timestamp, message in (
        (-1.0, "market_timestamps_must_be_non_negative_integers"),
        (1.5, "market_timestamps_must_be_non_negative_integers"),
    ):
        bad = market_data()
        bad[2, 0] = timestamp
        with pytest.raises(ValueError, match=message):
            validate_market_data(bad)
    bad = market_data()
    bad[5, 0] += 60_000.0
    with pytest.raises(ValueError, match="market_timestamps_must_be_contiguous"):
        validate_market_data(bad)

    bad = market_data()
    bad[2, 1] = 0.0
    with pytest.raises(ValueError, match="market_prices_must_be_positive"):
        validate_market_data(bad)
    bad = market_data()
    bad[2, 2] = bad[2, 3] - 1.0
    with pytest.raises(ValueError, match="market_high_inconsistent"):
        validate_market_data(bad)
    bad = market_data()
    bad[2, 3] = bad[2, 1] + 0.1
    with pytest.raises(ValueError, match="market_low_inconsistent"):
        validate_market_data(bad)
    bad = market_data()
    bad[2, 5] = -1.0
    with pytest.raises(ValueError, match="market_volume_must_be_non_negative"):
        validate_market_data(bad)


def test_scaler_is_fit_only_from_supplied_training_rows_and_handles_zero_volume() -> (
    None
):
    data = market_data()
    scaler = MarketObservationScaler.fit(data[:20])
    assert scaler.price_anchor == float(data[0, 4])
    assert scaler.volume_scale == pytest.approx(float(np.median(data[:20, 5])))

    zero_volume = market_data(8)
    zero_volume[:, 5] = 0.0
    fitted = MarketObservationScaler.fit(zero_volume)
    assert fitted.volume_scale == 1.0

    with pytest.raises(ValueError, match="price_anchor_must_be_positive_and_finite"):
        MarketObservationScaler(0.0, 1.0)
    with pytest.raises(ValueError, match="price_anchor_must_be_positive_and_finite"):
        MarketObservationScaler(float("inf"), 1.0)
    with pytest.raises(ValueError, match="volume_scale_must_be_positive_and_finite"):
        MarketObservationScaler(100.0, 0.0)
    with pytest.raises(ValueError, match="volume_scale_must_be_positive_and_finite"):
        MarketObservationScaler(100.0, float("nan"))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "sample_must_be_numeric"),
        ("12", "sample_must_be_numeric"),
        (float("nan"), "sample_must_be_finite"),
        (float("inf"), "sample_must_be_finite"),
    ],
)
def test_numeric_parameter_validation_is_strict(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _finite_numeric_parameter("sample", value)


@pytest.mark.parametrize("value", [-0.01, 10_000.0, float("nan"), True])
def test_cost_validation_rejects_out_of_range_values(value: object) -> None:
    with pytest.raises(ValueError):
        _validate_cost_basis_points("fee_bps", value)


def test_configuration_validators_reject_invalid_risk_and_reward_parameters() -> None:
    with pytest.raises(ValueError, match="drawdown_penalties_must_be_non_negative"):
        _validate_drawdown_penalties(-0.1, 0.01)
    with pytest.raises(ValueError, match="drawdown_penalties_must_be_non_negative"):
        _validate_drawdown_penalties(0.1, -0.01)
    for values in (
        (-1.0, 0.1, 0.1),
        (0.1, -1.0, 0.1),
        (0.1, 0.1, -1.0),
    ):
        with pytest.raises(
            ValueError, match="reward_shaping_parameters_must_be_non_negative"
        ):
            _validate_reward_shaping(*values)

    valid = (10_000.0, 25.0, 20.0, 1, 1.5, 0.01, 0.001, 0.001, 3, 0.0001)
    _validate_environment_configuration(*valid)
    invalid_configs = (
        (0.0, 25.0, 20.0, 1, 1.5, 0.01, 0.001, 0.001, 3, 0.0001),
        (10_000.0, 10_000.0, 20.0, 1, 1.5, 0.01, 0.001, 0.001, 3, 0.0001),
        (10_000.0, 25.0, 20.0, 0, 1.5, 0.01, 0.001, 0.001, 3, 0.0001),
        (10_000.0, 25.0, 20.0, 1, -1.0, 0.01, 0.001, 0.001, 3, 0.0001),
        (10_000.0, 25.0, 20.0, 1, 1.5, 0.01, -0.001, 0.001, 3, 0.0001),
        (10_000.0, 25.0, 20.0, 1, 1.5, 0.01, 0.001, 0.001, True, 0.0001),
    )
    for config in invalid_configs:
        with pytest.raises(ValueError):
            _validate_environment_configuration(*config)

    for value in (0, -1, True, 1.5):
        with pytest.raises(
            ValueError, match="minimum_holding_period_candles_must_be_positive_integer"
        ):
            _validate_minimum_holding_period_candles(value)
    with pytest.raises(ValueError, match="include_portfolio_state_must_be_boolean"):
        _validate_portfolio_state_option(1)
    for distance, activation in (
        (0.0, 0.03),
        (0.02, 1.0),
        (0.03, 0.03),
        (float("nan"), 0.1),
    ):
        with pytest.raises(ValueError):
            _validate_trailing_stop_configuration(distance, activation)
    for scale_value in (True, "0.1", -0.1, float("inf")):
        with pytest.raises(ValueError, match="signal_guidance_reward_scale"):
            _validate_signal_guidance_reward_scale(scale_value)


def test_constructor_enforces_configuration_and_observation_contracts() -> None:
    data = market_data()
    with pytest.raises(ValueError, match="initial_equity_must_be_positive"):
        BTCUSDTTradingEnv(data, initial_equity=0.0)
    with pytest.raises(ValueError, match="fee_bps_per_side_out_of_range"):
        BTCUSDTTradingEnv(data, fee_bps_per_side=10_000.0)
    with pytest.raises(ValueError, match="min_remaining_steps_must_be_positive"):
        BTCUSDTTradingEnv(data, min_remaining_steps=0)
    with pytest.raises(ValueError, match="signal_guidance_reward_scale"):
        BTCUSDTTradingEnv(data, signal_guidance_reward_scale=-1.0)

    env = make_env(include_portfolio_state=False)
    observation, info = env.reset(seed=3)
    assert observation.shape == (9,)
    assert env.observation_space.contains(observation)
    assert info == {"start_index": 0}
    env.close()


def test_cost_setters_are_atomic_and_guidance_scale_is_validated() -> None:
    env = make_env(fee_bps=10.0, slippage_bps=20.0)
    env.set_transaction_costs(30.0, 40.0)
    assert (env.fee_bps_per_side, env.slippage_bps_per_side) == (30.0, 40.0)
    with pytest.raises(ValueError):
        env.set_transaction_costs(50.0, -1.0)
    assert (env.fee_bps_per_side, env.slippage_bps_per_side) == (30.0, 40.0)

    env.set_signal_guidance_reward_scale(0.0005)
    assert env.signal_guidance_reward_scale == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="signal_guidance_reward_scale"):
        env.set_signal_guidance_reward_scale(float("nan"))
    assert env.signal_guidance_reward_scale == pytest.approx(0.0005)
    env.close()


def test_reset_supports_fixed_and_seeded_random_starts_and_rejects_bad_options() -> (
    None
):
    env = make_env(rows=40, random_start=True, min_remaining_steps=5)
    _, info = env.reset(seed=123)
    first_start = info["start_index"]
    _, second_info = env.reset(seed=123)
    assert second_info["start_index"] == first_start
    assert 0 <= first_start <= 34

    with pytest.raises(ValueError, match="start_index_must_be_integer"):
        env.reset(options={"start_index": True})
    with pytest.raises(ValueError, match="start_index_must_be_integer"):
        env.reset(options={"start_index": "2"})
    with pytest.raises(ValueError, match="start_index_out_of_range"):
        env.reset(options={"start_index": 39})

    env._fill_count = 9
    env._closed_trade_net_profits.append(100.0)
    _, fixed = env.reset(options={"start_index": 4})
    assert fixed["start_index"] == 4
    assert env._fill_count == 0
    assert env._closed_trade_net_profits == []
    env.close()


def test_observation_and_portfolio_state_reflect_realized_inventory_state() -> None:
    env = make_env()
    observation, _ = env.reset()
    assert observation.shape == (12,)
    assert env.observation_space.contains(observation)
    assert np.isfinite(observation).all()
    assert env._portfolio_observation_values(100.0) == (0.0, 0.0, 0.0)

    env._quantity = 2.0
    env._cash = 9_800.0
    env._position_cost_basis = 180.0
    env._last_buy_index = 0
    env._index = 1
    position_fraction, unrealized, holding_fraction = env._portfolio_observation_values(
        100.0
    )
    assert position_fraction == pytest.approx(200.0 / 10_000.0)
    assert unrealized == pytest.approx((200.0 - 180.0) / 180.0)
    assert holding_fraction == pytest.approx(1.0)
    assert env._position_holding_candles(100.0) == 2
    assert not env._minimum_holding_period_active(100.0)

    env._last_buy_index = None
    assert env._position_holding_candles(100.0) == 0
    env._quantity = 0.0
    assert not env._minimum_holding_period_active(100.0)
    env.close()


def test_equity_fill_costs_quantity_guards_and_rebalance_accounting() -> None:
    env = make_env(fee_bps=25.0, slippage_bps=20.0)
    buy_price, buy_fee, buy_slippage = env._fill_costs(2.0, 100.0, True)
    sell_price, sell_fee, sell_slippage = env._fill_costs(-2.0, 100.0, False)
    assert buy_price > 100.0 > sell_price
    assert buy_fee > 0.0 and sell_fee > 0.0
    assert buy_slippage > 0.0 and sell_slippage > 0.0
    assert env._equity_at(100.0) == pytest.approx(10_000.0)

    assert env._should_skip_rebalance(0.0, 100.0)
    env._rebalance_to_quantity(2.0, 100.0)
    assert env._quantity == pytest.approx(2.0)
    assert env._buy_fill_count == 1
    assert env._total_fees > 0.0
    assert env._total_slippage > 0.0
    assert env._should_skip_rebalance(2.0, 100.0)
    assert env._should_skip_rebalance(2.0 + 1e-14, 100.0)
    env._rebalance_to_quantity(1.0, 110.0)
    assert env._quantity == pytest.approx(1.0)
    assert env._sell_fill_count == 1
    env._rebalance_to_quantity(0.0, 105.0)
    assert env._quantity == pytest.approx(0.0)
    assert env._fill_count == 3

    with pytest.raises(ValueError, match="target_quantity_invalid"):
        env._validate_target_quantity(-1.0)
    with pytest.raises(ValueError, match="target_quantity_invalid"):
        env._validate_target_quantity(float("inf"))
    with pytest.raises(ValueError, match="simulated_spot_cash_would_be_negative"):
        env._validate_rebalanced_cash(-1e-4)
    env._validate_rebalanced_cash(-1e-8)
    env._cash = 0.0
    env._quantity = 0.0
    with pytest.raises(ValueError, match="simulated_equity_non_positive"):
        env._equity_at(100.0)
    env.close()


def test_action_decoding_rejects_invalid_actions_and_maps_all_valid_actions() -> None:
    env = make_env()
    assert env._validated_action_values(ACTION_HOLD) == (0, "HOLD", "HOLD", 0.0, None)
    assert env._validated_action_values(np.array([ACTION_BUY_100], dtype=np.int64)) == (
        3,
        "BUY_100",
        "BUY",
        1.0,
        2,
    )
    assert env._validated_action_values(ACTION_SELL_100) == (
        6,
        "SELL_100",
        "SELL",
        1.0,
        2,
    )
    for action in (
        np.array([1.0]),
        np.array([[1, 2]], dtype=np.int64),
        np.array([True]),
        -1,
        7,
    ):
        with pytest.raises(ValueError):
            env._validated_action_values(cast(Int64Array | np.int64 | int, action))
    env.close()


def test_legal_action_mask_and_buy_execution_fail_closed_at_boundaries() -> None:
    env = make_env(minimum_holding_period_candles=3)
    env.reset()
    flat = env._legal_action_mask(100.0)
    assert flat["HOLD"] and flat["BUY_25"] and not flat["SELL_25"]

    env._quantity = 1.0
    env._cash = 9_900.0
    env._position_cost_basis = 100.0
    env._last_buy_index = 0
    blocked = env._legal_action_mask(100.0)
    assert not blocked["BUY_25"] and not blocked["SELL_25"]

    masked = env._execute_buy_action("BUY_25", 0.25, 101.0, blocked, 100.0)
    assert masked[1] == "MASKED_BUY"
    assert masked[3] == "minimum_holding_period_active"

    env._index = 3
    assert env._legal_action_mask(100.0)["SELL_25"]
    assert env._target_for_buy(0.0, 100.0) is False
    assert env._target_for_buy(1.1, 100.0) is False
    env._cash = 0.0
    assert env._target_for_buy(0.5, 100.0) is False
    env._quantity = 100.0
    assert not env._legal_action_mask(100.0)["BUY_100"]
    assert env._target_for_buy(1e-15, 100.0) is False
    env.close()


def test_buy_and_sell_targets_handle_partial_and_full_closes() -> None:
    env = make_env()
    env.reset()
    mask = env._legal_action_mask(100.0)
    noop = env._execute_buy_action("BUY_25", 0.0, 100.0, mask, 100.0)
    assert noop[1] == "NOOP_BUY"

    bought = env._execute_buy_action("BUY_100", 1.0, 100.0, mask, 100.0)
    assert bought[1] == "BUY"
    assert env._trade_entry_reference_price == 100.0
    assert env._last_buy_index == 1

    assert env._sell_target_quantity(0.0, 100.0) is None
    assert env._sell_target_quantity(1.1, 100.0) is None
    assert env._sell_target_quantity(0.25, 100.0) == pytest.approx(env._quantity * 0.75)

    env._index = 1
    mask = env._legal_action_mask(101.0)
    partial = env._execute_sell_action("SELL_25", 0.25, 101.0, mask)
    assert partial[1] == "SELL"
    assert env._quantity > 0.0
    assert env._active_exit_reason is None
    full_mask = env._legal_action_mask(110.0)
    closed = env._execute_sell_action("SELL_100", 1.0, 110.0, full_mask)
    assert closed[1] == "SELL"
    assert env._quantity == pytest.approx(0.0)
    assert env._closed_trade_count == 1
    assert env._winning_trade_count == 1

    no_position = env._execute_sell_action(
        "SELL_25", 0.25, 110.0, env._legal_action_mask(110.0)
    )
    assert no_position[1] == "MASKED_SELL"
    assert no_position[3] == "no_position"
    env.close()


def test_action_ledger_and_invalid_action_penalties_are_explicit() -> None:
    env = make_env()
    env._record_action("HOLD", "HOLD", "HOLD", None)
    env._record_action("BUY_25", "BUY", "MASKED_BUY", "insufficient_spendable_cash")
    env._record_action(
        "BUY_25", "BUY", "NOOP_BUY", "buy_target_not_above_current_position"
    )
    env._record_action(
        "SELL_25", "SELL", "MASKED_SELL", "minimum_holding_period_active"
    )
    assert env._hold_action_count == 1
    assert env._buy_action_count == 2
    assert env._sell_action_count == 1
    assert env._masked_buy_action_count == 1
    assert env._buy_noop_action_count == 1
    assert env._buy_target_not_increasing_action_count == 1
    assert env._masked_sell_action_count == 1
    assert env._minimum_holding_period_blocked_action_count == 1
    with pytest.raises(ValueError, match="action_name_invalid"):
        env._record_action("NOPE", "HOLD", "HOLD", None)
    with pytest.raises(ValueError, match="action_side_invalid"):
        env._record_action("HOLD", "NOPE", "HOLD", None)

    assert not env._requested_action_blocked_by_legal_mask(
        "HOLD", "HOLD", {"HOLD": True}
    )
    assert env._requested_action_blocked_by_legal_mask(
        "SELL", "SELL_25", {"SELL_25": False}
    )
    applied, penalty = env._invalid_action_penalty_for_request(
        "BUY", "BUY_100", {"BUY_100": False}
    )
    assert applied and penalty == pytest.approx(1.0)
    applied, penalty = env._invalid_action_penalty_for_request(
        "SELL", "SELL_25", {"SELL_25": False}
    )
    assert applied and penalty == pytest.approx(env.invalid_action_penalty)
    assert env._invalid_action_penalty_for_step(
        True, "BUY", "BUY_100", {"BUY_100": False}
    ) == (False, 0.0)
    env.close()


def test_position_change_tracking_and_episode_settlement() -> None:
    env = make_env(rows=12)
    env.reset()
    change, changed, penalty = env._track_position_change(0.0, 100.0)
    assert (change, changed, penalty) == (0.0, False, 0.0)
    env._quantity = 2.0
    change, changed, penalty = env._track_position_change(0.0, 100.0)
    assert change == 2.0 and changed and penalty == 0.0
    env._quantity = 2.0
    env._consecutive_no_position_change_steps = env.inactivity_grace_steps
    _, changed, penalty = env._track_position_change(2.0, 100.0)
    assert not changed and penalty == pytest.approx(env.inactivity_penalty)

    settled, drawdown, equity = env._settle_episode_step(1, env._market_data[1], 0.01)
    assert not settled and equity > 0.0 and drawdown == 0.01

    env._quantity = 2.0
    env._cash = 9_800.0
    env._position_cost_basis = 200.0
    env._trade_entry_cost_total = 200.0
    env._trade_entry_index = 1
    env._trade_entry_reference_price = 100.0
    env._last_buy_index = 1
    final_index = len(env._market_data) - 1
    terminated, final_drawdown, final_equity = env._settle_episode_step(
        final_index, env._market_data[final_index], 0.0
    )
    assert terminated
    assert final_equity > 0.0
    assert final_drawdown >= 0.0
    assert env._final_equity == pytest.approx(final_equity)
    assert env._last_closed_trade_exit_reason == "terminal_liquidation"
    env.close()


def test_drawdown_and_trailing_stops_cover_open_gap_and_intrabar_paths() -> None:
    env = make_env()
    env.reset()
    drawdown = env._update_drawdown(env._market_data[1])
    assert drawdown >= 0.0
    assert not env._apply_gap_trailing_stop(100.0)
    assert not env._apply_intrabar_trailing_stop(env._market_data[1])
    env._close_for_trailing_stop(100.0)

    buy_full_position(env)
    env._trade_entry_reference_price = 100.0
    env._highest_price_since_entry = 100.0
    env._trailing_stop_price = 98.0
    assert env._apply_gap_trailing_stop(97.0)
    assert env._quantity == pytest.approx(0.0)
    assert env._last_closed_trade_exit_reason == "trailing_stop"
    assert env._last_trailing_stop_triggered
    assert env._trailing_stop_trigger_count == 1

    buy_full_position(env)
    env._trailing_stop_price = 98.0
    bar = env._market_data[1].copy()
    bar[1] = 100.0
    bar[2] = 102.0
    bar[3] = 97.0
    bar[4] = 101.0
    assert env._apply_intrabar_trailing_stop(bar)
    assert env._quantity == pytest.approx(0.0)
    assert env._last_closed_trade_exit_reason == "trailing_stop"

    buy_full_position(env)
    env._trade_entry_reference_price = 100.0
    env._highest_price_since_entry = None
    bar = env._market_data[1].copy()
    bar[2] = 105.0
    env._update_trailing_stop_from_completed_bar(bar)
    assert env._trailing_stop_activated
    first_stop = env._trailing_stop_price
    assert first_stop is not None
    bar[2] = 110.0
    env._update_trailing_stop_from_completed_bar(bar)
    assert env._trailing_stop_price is not None
    assert env._trailing_stop_price > first_stop
    bar[2] = 101.0
    env._update_trailing_stop_from_completed_bar(bar)
    assert env._trailing_stop_price == pytest.approx(107.8)
    env.close()


def test_opportunity_cost_holding_rewards_and_guidance_targets() -> None:
    env = make_env(signal_guidance_reward_scale=0.0002)
    env.reset()
    assert env._opportunity_cost_penalty_for_action("BUY", 120.0) == 0.0
    assert env._opportunity_cost_penalty_for_action("HOLD", 100.0) == 0.0
    env._index = 25
    current = float(env._market_data[25, 4])
    assert env._opportunity_cost_penalty_for_action("HOLD", current) > 0.0
    env._quantity = 1.0
    env._position_cost_basis = 100.0
    assert env._opportunity_cost_penalty_for_action("HOLD", current) == 0.0

    env._position_cost_basis = 200.0
    assert env._holding_loss_penalty_for_action("BUY", current) == 0.0
    assert env._holding_loss_penalty_for_action("HOLD", current) > 0.0
    env._position_cost_basis = 50.0
    assert env._holding_profit_reward_for_action("HOLD", current) > 0.0
    assert env._holding_profit_reward_for_action("SELL", current) == 0.0

    env._quantity = 0.0
    assert env._guidance_target_index(current) == ACTION_BUY_25
    assert env._signal_guidance_request_is_eligible("BUY", "BUY")
    assert not env._signal_guidance_request_is_eligible("HOLD", "HOLD")
    assert not env._signal_guidance_request_is_eligible("BUY", "MASKED_BUY")
    assert env._causal_sma_guidance_action("BUY", 2.0, 1.0) == "BUY"
    assert env._causal_sma_guidance_action("SELL", 2.0, 1.0) is None
    assert env._causal_sma_guidance_action("SELL", 1.0, 2.0) == "SELL"
    assert env._causal_sma_guidance_action("BUY", 1.0, 2.0) is None
    assert env._causal_sma_guidance_action("BUY", 1.0, 1.0) is None
    upward_reward = env._signal_guidance_reward_for_action("BUY", "BUY")
    assert upward_reward[0] == pytest.approx(0.0002)
    assert upward_reward[1] == "BUY"
    assert env._signal_guidance_reward_for_action("SELL", "SELL") == (0.0, None)
    env._record_signal_guidance_reward("SELL")
    assert env._signal_guidance_sell_reward_count == 1
    env.close()

    down = make_env(trend_per_candle=-0.2, signal_guidance_reward_scale=0.0002)
    down.reset()
    down._index = 25
    down._quantity = 1.0
    down._cash = 9_000.0
    down._position_cost_basis = 100.0
    down._last_buy_index = 0
    assert (
        down._guidance_target_index(float(down._market_data[25, 4])) == ACTION_SELL_25
    )
    downward_reward = down._signal_guidance_reward_for_action("SELL", "SELL")
    assert downward_reward[0] == pytest.approx(0.0002)
    assert downward_reward[1] == "SELL"
    down.close()


def test_realized_profit_early_exit_and_total_reward_validation() -> None:
    env = make_env()
    env.reset()
    assert env._realized_profit_reward_for_action("SELL") == 0.0
    env._last_closed_trade_net_profit = 100.0
    env._last_closed_trade_exit_reason = "terminal_liquidation"
    env._last_closed_trade_return_pct = 4.0
    assert env._realized_profit_reward_for_action("SELL") == 0.0
    env._last_closed_trade_exit_reason = "policy_sell"
    env._last_closed_trade_return_pct = 2.0
    standard = env._realized_profit_reward_for_action("SELL")
    env._last_closed_trade_return_pct = 3.5
    double = env._realized_profit_reward_for_action("SELL")
    env._last_closed_trade_return_pct = 6.0
    triple = env._realized_profit_reward_for_action("SELL")
    assert standard > 0.0 and double > standard and triple > double
    assert env._realized_profit_reward_count == 3

    assert env._early_exit_penalty_for_closed_trade() == 0.0
    env._last_closed_trade_return_pct = 0.5
    penalty = env._early_exit_penalty_for_closed_trade()
    assert penalty > 0.0
    env._last_closed_trade_exit_reason = "trailing_stop"
    assert env._early_exit_penalty_for_closed_trade() == 0.0
    env._last_closed_trade_exit_reason = "policy_sell"
    env._last_closed_trade_net_profit = None
    assert env._early_exit_penalty_for_closed_trade() == 0.0

    log_return, profitable_bonus, reward = env._calculate_step_reward(
        10_100.0,
        10_000.0,
        0.02,
        0.01,
        0.001,
        0.0,
        0.0,
        0.0,
        0.001,
        0.0002,
        0.0001,
        0.002,
        0.0,
        2,
    )
    assert log_return == pytest.approx(math.log(1.01))
    assert profitable_bonus == pytest.approx(2 * env.profitable_trade_bonus)
    assert math.isfinite(reward)
    with pytest.raises(ValueError, match="reward_non_finite"):
        env._calculate_step_reward(
            float("nan"),
            10_000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
    env.close()


def test_step_runs_buy_sell_and_terminal_liquidation_contracts() -> None:
    env = make_env(rows=12)
    first_observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(first_observation)
    observation, reward, terminated, truncated, info = env.step(ACTION_BUY_100)
    assert env.observation_space.contains(observation)
    assert math.isfinite(reward)
    assert not terminated and not truncated
    assert info["executed_action"] == "BUY"
    assert info["fees_this_step"] == 0.0
    assert info["transaction_costs_this_step"] == 0.0

    observation, reward, terminated, truncated, info = env.step(ACTION_SELL_100)
    assert env.observation_space.contains(observation)
    assert math.isfinite(reward)
    assert not terminated and not truncated
    assert info["executed_action"] == "SELL"
    assert info["closed_trade_exit_reason_this_step"] == "policy_sell"
    assert info["policy_closed_trade_count"] == 1
    assert info["closed_trade_net_profit_this_step"] is not None

    env.reset(options={"start_index": 10})
    current_price = float(env._market_data[10, 4])
    env._rebalance_to_quantity(1.0, current_price)
    env._position_cost_basis = current_price
    env._trade_entry_cost_total = current_price
    env._trade_entry_index = 10
    env._trade_entry_reference_price = current_price
    env._last_buy_index = 10
    observation, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert terminated and not truncated
    assert env.observation_space.contains(observation)
    assert math.isfinite(reward)
    assert info["closed_trade_exit_reason_this_step"] == "terminal_liquidation"
    assert "profit_factor" in info
    with pytest.raises(RuntimeError, match="environment_episode_already_terminated"):
        env.step(ACTION_HOLD)
    env.close()


def test_summary_reports_trade_statistics_and_hold_collapse() -> None:
    env = make_env()
    env.reset()
    empty = env.summary()
    assert empty["win_rate"] is None
    assert empty["profit_factor"] is None
    assert not empty["hold_collapse"]

    env._closed_trade_net_profits = [10.0, -5.0, 0.0]
    env._closed_trade_returns_percent = [2.0, -1.0, 0.0]
    env._closed_trade_holding_periods = [4, 6, 8]
    env._closed_trade_exit_reasons = ["policy_sell", "trailing_stop", "policy_sell"]
    env._winning_trade_count = 1
    env._losing_trade_count = 1
    env._breakeven_trade_count = 1
    env._closed_trade_count = 3
    env._discrete_action_counts["HOLD"] = 4
    env._discrete_action_counts["BUY_25"] = 1
    summary = env.summary()
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["profit_factor"] == pytest.approx(2.0)
    assert summary["average_holding_period_candles"] == pytest.approx(6.0)
    assert summary["average_win_percent"] == pytest.approx(2.0)
    assert summary["average_loss_percent"] == pytest.approx(-1.0)
    assert summary["closed_trade_exit_reason_counts"] == {
        "policy_sell": 2,
        "trailing_stop": 1,
    }
    assert not summary["hold_collapse"]

    env._discrete_action_counts = {name: 0 for name in ("HOLD", "BUY_25")}
    env._discrete_action_counts["HOLD"] = 4
    assert env._summary_hold_collapse()
    assert env._summary_average_trade_return(profitable=True) == pytest.approx(2.0)
    assert env._summary_average_trade_return(profitable=False) == pytest.approx(-1.0)
    env.close()
