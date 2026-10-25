from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from numpy.typing import NDArray

from services.rl_research.rl_trading_environment import (
    BTCUSDTTradingEnv,
    MarketObservationScaler,
    OBSERVATION_FIELDS,
    OBSERVATION_SIZE,
    validate_market_data,
)

Float64Array = NDArray[np.float64]


def _bars(
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> Float64Array:
    close_values = closes or [100.0, 100.0, 100.0, 100.0, 100.0]
    high_values = highs or [value + 1.0 for value in close_values]
    low_values = lows or [value - 1.0 for value in close_values]
    volume_values = volumes or [10.0] * len(close_values)
    timestamps = 1_700_000_000_000 + np.arange(len(close_values)) * 60_000
    return np.asarray(
        [
            timestamps,
            np.asarray(close_values, dtype=np.float64),
            np.asarray(high_values, dtype=np.float64),
            np.asarray(low_values, dtype=np.float64),
            np.asarray(close_values, dtype=np.float64),
            np.asarray(volume_values, dtype=np.float64),
        ],
        dtype=np.float64,
    ).T


def _env(data: Float64Array, **kwargs: Any) -> BTCUSDTTradingEnv:
    kwargs.setdefault("fee_bps_per_side", 0.0)
    kwargs.setdefault("slippage_bps_per_side", 0.0)
    kwargs.setdefault("min_remaining_steps", 1)
    return BTCUSDTTradingEnv(data, **kwargs)


def test_environment_satisfies_gymnasium_contract() -> None:
    env = _env(_bars())
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert info["start_index"] == 0
    env.close()


def test_observation_has_only_price_volume_and_time_fields() -> None:
    data = _bars()
    scaler = MarketObservationScaler.fit(data)
    env = _env(data, scaler=scaler)
    observation, _ = env.reset(seed=11)
    assert len(OBSERVATION_FIELDS) == 9
    assert observation.shape == (9,)
    assert np.isfinite(observation).all()
    assert all(
        any(
            token in name
            for token in ("open", "high", "low", "close", "volume", "minute_of")
        )
        for name in OBSERVATION_FIELDS
    )
    assert not any(
        token in name
        for token in ("position", "cash", "equity", "profit", "drawdown")
        for name in OBSERVATION_FIELDS
    )
    assert scaler.price_anchor == pytest.approx(100.0)
    assert scaler.volume_scale == pytest.approx(10.0)
    env.close()


def test_scaler_uses_training_only_and_handles_zero_median_volume() -> None:
    train = _bars(closes=[100.0, 101.0, 102.0], volumes=[0.0, 0.0, 0.0])
    scaler = MarketObservationScaler.fit(train)
    future = _bars(closes=[90_000.0, 91_000.0, 92_000.0], volumes=[1e12] * 3)
    future_scaler = MarketObservationScaler.fit(future)
    assert scaler.price_anchor == pytest.approx(100.0)
    assert scaler.volume_scale == pytest.approx(1.0)
    assert future_scaler.price_anchor == pytest.approx(90_000.0)
    assert future_scaler.volume_scale == pytest.approx(1e12)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data[:, 1] * 0.0, "market_prices_must_be_positive"),
        (lambda data: data[:, 2] * 0.0 + 1.0, "market_high_inconsistent"),
        (lambda data: data[:, 3] * 0.0 + 500.0, "market_low_inconsistent"),
        (lambda data: data[:, 5] * 0.0 - 1.0, "market_volume_must_be_non_negative"),
    ],
)
def test_invalid_market_geometry_fails_closed(mutate: object, message: str) -> None:
    data = _bars()
    column = {
        "market_prices_must_be_positive": 1,
        "market_high_inconsistent": 2,
        "market_low_inconsistent": 3,
        "market_volume_must_be_non_negative": 5,
    }[message]
    if message == "market_low_inconsistent":
        data[:, 2] = 600.0
    data[:, column] = mutate(data)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_market_data(data)


def test_market_data_shape_non_finite_timestamp_and_gap_fail_closed() -> None:
    with pytest.raises(ValueError, match="six_columns"):
        validate_market_data(np.ones((5, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="at_least_two_rows"):
        validate_market_data(_bars(closes=[100.0]))
    non_finite = _bars()
    non_finite[1, 4] = math.nan
    with pytest.raises(ValueError, match="non_finite"):
        validate_market_data(non_finite)
    fractional_timestamp = _bars()
    fractional_timestamp[1, 0] += 0.5
    with pytest.raises(ValueError, match="timestamps_must_be_non_negative_integers"):
        validate_market_data(fractional_timestamp)
    gap = _bars()
    gap[2:, 0] += 60_000
    with pytest.raises(ValueError, match="contiguous_one_minute"):
        validate_market_data(gap)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_equity": 0.0}, "initial_equity_must_be_positive"),
        ({"fee_bps_per_side": -1.0}, "fee_bps_per_side_out_of_range"),
        ({"slippage_bps_per_side": 10_000.0}, "slippage_bps_per_side_out_of_range"),
        (
            {"drawdown_increment_penalty": -1.0},
            "drawdown_penalties_must_be_non_negative",
        ),
        (
            {"inactivity_penalty": -0.1},
            "reward_shaping_parameters_must_be_non_negative",
        ),
        (
            {"profitable_trade_bonus": -0.1},
            "reward_shaping_parameters_must_be_non_negative",
        ),
        ({"min_remaining_steps": 0}, "min_remaining_steps_must_be_positive"),
    ],
)
def test_invalid_environment_configuration_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _env(_bars(), **kwargs)


def test_buy_hold_and_partial_sell_execute_on_the_next_bar() -> None:
    data = _bars()
    env = _env(data, initial_equity=1_000.0)
    observation, _ = env.reset(seed=2)
    assert observation.shape == (9,)

    observation, _, terminated, _, info = env.step(
        np.asarray([0.8, 0.5], dtype=np.float32)
    )
    assert terminated is False
    assert info["timestamp_ms"] == int(data[1, 0])
    assert info["action"] == "BUY"
    assert env._quantity == pytest.approx(5.0)

    observation, _, terminated, _, info = env.step(
        np.asarray([0.0, 0.9], dtype=np.float32)
    )
    assert terminated is False
    assert info["action"] == "HOLD"
    assert env._quantity == pytest.approx(5.0)

    observation, _, terminated, _, info = env.step(
        np.asarray([-0.8, 0.5], dtype=np.float32)
    )
    assert terminated is False
    assert info["action"] == "SELL"
    assert env._quantity == pytest.approx(2.5)
    assert observation.shape == (9,)
    env.close()


def test_buy_resizes_position_and_sell_can_close_all_holdings() -> None:
    env = _env(_bars(), initial_equity=1_000.0)
    env.reset(seed=3)
    env.step(np.asarray([0.8, 0.8], dtype=np.float32))
    assert env._quantity == pytest.approx(8.0)
    env.step(np.asarray([0.8, 0.25], dtype=np.float32))
    assert env._quantity == pytest.approx(2.5)
    env.step(np.asarray([-0.8, 1.0], dtype=np.float32))
    assert env._quantity == pytest.approx(0.0)
    env.close()


def test_terminal_liquidation_charges_fees_and_slippage() -> None:
    env = BTCUSDTTradingEnv(
        _bars(),
        initial_equity=1_000.0,
        fee_bps_per_side=25.0,
        slippage_bps_per_side=20.0,
        min_remaining_steps=1,
    )
    env.reset(seed=5)
    env.step(np.asarray([0.8, 1.0], dtype=np.float32))
    env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    _, reward, terminated, truncated, info = env.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )
    assert terminated is True
    assert truncated is False
    assert math.isfinite(reward)
    assert env._quantity == pytest.approx(0.0)
    assert info["fill_count"] == 2
    assert info["total_fees"] > 0.0
    assert info["total_slippage"] > 0.0
    assert info["final_equity"] < 1_000.0
    assert env.summary()["net_profit"] < 0.0
    env.close()


def test_drawdown_reward_uses_intrabar_peak_to_trough_risk() -> None:
    data = _bars(
        closes=[100.0, 100.0, 90.0],
        highs=[101.0, 101.0, 110.0],
        lows=[99.0, 99.0, 80.0],
    )
    env = _env(data, initial_equity=1_000.0)
    env.reset(seed=9)
    env.step(np.asarray([0.8, 1.0], dtype=np.float32))
    _, reward, terminated, _, info = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert info["max_drawdown"] > 0.20
    assert reward < math.log(0.9) - 1.0
    env.close()


def test_invalid_actions_and_post_terminal_step_are_rejected() -> None:
    env = _env(_bars(), initial_equity=1_000.0)
    env.reset(seed=12)
    with pytest.raises(ValueError, match="two_finite_values"):
        env.step(np.asarray([0.1], dtype=np.float32))
    with pytest.raises(ValueError, match="outside_declared_space"):
        env.step(np.asarray([1.1, 0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="outside_declared_space"):
        env.step(np.asarray([0.1, 1.1], dtype=np.float32))
    with pytest.raises(ValueError, match="two_finite_values"):
        env.step(np.asarray([math.nan, 0.5], dtype=np.float32))
    for _ in range(len(_bars()) - 1):
        _, _, terminated, _, _ = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    with pytest.raises(RuntimeError, match="already_terminated"):
        env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    env.close()


def test_random_start_is_seeded_and_explicit_start_is_validated() -> None:
    data = _bars()
    first = _env(data, random_start=True, min_remaining_steps=2)
    second = _env(data, random_start=True, min_remaining_steps=2)
    _, first_info = first.reset(seed=24)
    _, second_info = second.reset(seed=24)
    assert first_info["start_index"] == second_info["start_index"]
    _, info = first.reset(options={"start_index": 1})
    assert info["start_index"] == 1
    with pytest.raises(ValueError, match="start_index_must_be_integer"):
        first.reset(options={"start_index": True})
    with pytest.raises(ValueError, match="start_index_out_of_range"):
        first.reset(options={"start_index": len(data)})
    first.close()
    second.close()


def test_scaler_and_summary_reject_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="price_anchor_must_be_positive"):
        MarketObservationScaler(0.0, 1.0)
    with pytest.raises(ValueError, match="volume_scale_must_be_positive"):
        MarketObservationScaler(100.0, 0.0)
    env = _env(_bars())
    env.reset(seed=1)
    summary = env.summary()
    assert summary["final_equity"] == pytest.approx(10_000.0)
    assert summary["fill_count"] == 0
    env.close()



def test_hold_action_pays_configured_inactivity_penalty() -> None:
    env = _env(_bars())
    env.reset(seed=31)
    _, reward, terminated, _, info = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert terminated is False
    assert reward == pytest.approx(-0.001)
    assert info["action"] == "HOLD"
    assert info["reward_inactivity_penalty"] == pytest.approx(0.001)
    assert info["reward_profitable_trade_bonus"] == pytest.approx(0.0)
    assert env.summary()["hold_action_count"] == 1
    env.close()


def test_profitable_completed_trade_gets_intermediate_reward_bonus() -> None:
    env = _env(
        _bars(closes=[100.0, 100.0, 110.0, 110.0, 110.0]),
        initial_equity=1_000.0,
    )
    env.reset(seed=32)
    env.step(np.asarray([0.8, 1.0], dtype=np.float32))
    _, reward, terminated, _, info = env.step(
        np.asarray([-0.8, 1.0], dtype=np.float32)
    )
    assert terminated is False
    assert info["action"] == "SELL"
    assert info["closed_trade_net_profit_this_step"] == pytest.approx(100.0)
    assert info["reward_profitable_trade_bonus"] == pytest.approx(0.0001)
    assert reward > math.log(1.1)
    summary = env.summary()
    assert summary["closed_trade_count"] == 1
    assert summary["winning_trade_count"] == 1
    assert summary["losing_trade_count"] == 0
    assert summary["win_rate"] == pytest.approx(1.0)
    env.close()


def test_losing_completed_trade_receives_no_profit_bonus() -> None:
    env = _env(
        _bars(closes=[100.0, 100.0, 90.0, 90.0, 90.0]),
        initial_equity=1_000.0,
    )
    env.reset(seed=33)
    env.step(np.asarray([0.8, 1.0], dtype=np.float32))
    _, reward, terminated, _, info = env.step(
        np.asarray([-0.8, 1.0], dtype=np.float32)
    )
    assert terminated is False
    assert info["closed_trade_net_profit_this_step"] == pytest.approx(-100.0)
    assert info["reward_profitable_trade_bonus"] == pytest.approx(0.0)
    assert reward < math.log(0.9)
    summary = env.summary()
    assert summary["closed_trade_count"] == 1
    assert summary["winning_trade_count"] == 0
    assert summary["losing_trade_count"] == 1
    assert summary["win_rate"] == pytest.approx(0.0)
    env.close()


def test_partial_realizations_accumulate_before_profitable_trade_bonus() -> None:
    env = _env(
        _bars(closes=[100.0, 100.0, 110.0, 120.0, 120.0]),
        initial_equity=1_000.0,
    )
    env.reset(seed=34)
    env.step(np.asarray([0.8, 1.0], dtype=np.float32))
    _, _, _, _, partial_info = env.step(
        np.asarray([-0.8, 0.5], dtype=np.float32)
    )
    assert partial_info["closed_trade_net_profit_this_step"] is None
    assert partial_info["reward_profitable_trade_bonus"] == pytest.approx(0.0)
    _, _, terminated, _, final_info = env.step(
        np.asarray([-0.8, 1.0], dtype=np.float32)
    )
    assert terminated is False
    assert final_info["closed_trade_net_profit_this_step"] == pytest.approx(150.0)
    assert final_info["reward_profitable_trade_bonus"] == pytest.approx(0.0001)
    assert env.summary()["closed_trade_count"] == 1
    env.close()
