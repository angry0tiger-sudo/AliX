from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from services.rl_research.risk_allocation_environment import (
    ACTION_FRACTIONS,
    ACTION_NAMES,
    ACTION_SKIP,
    ACTION_TAKE,
    BAR_INTERVAL_MS,
    DRAWDOWN_EXCESS_PENALTY,
    DRAWDOWN_INCREMENT_PENALTY,
    DRAWDOWN_LIMIT,
    DRAWDOWN_SEVERE_PENALTY,
    FEE_BPS_PER_SIDE,
    OBSERVATION_FIELDS,
    ROUND_TRIP_COST_BPS,
    SLIPPAGE_BPS_PER_SIDE,
    RiskAllocationEnv,
)


def market_data(prices: tuple[float, ...] | None = None) -> NDArray[np.float64]:
    closes = np.asarray(
        prices or (100.0, 100.0, 100.0, 100.0, 100.0, 100.0), dtype=float
    )
    index = np.arange(len(closes), dtype=np.float64)
    timestamps = 1_700_000_000_000.0 + index * BAR_INTERVAL_MS
    opening = closes.copy()
    high = np.maximum(opening, closes) * 1.001
    low = np.minimum(opening, closes) * 0.999
    return np.column_stack(
        (timestamps, opening, high, low, closes, np.full(len(closes), 0.0))
    )


def make_env(
    prices: tuple[float, ...] | None = None,
    signals: tuple[bool, ...] | None = None,
    *,
    random_start: bool = False,
    episode_steps: int | None = None,
) -> RiskAllocationEnv:
    data = market_data(prices)
    rows = len(data)
    signal = np.asarray(signals or tuple(True for _ in range(rows)), dtype=np.bool_)
    atr = np.full(rows, 0.01, dtype=np.float64)
    recent = np.zeros((rows, 3), dtype=np.float64)
    probability = signal.astype(np.float64) * 0.8
    return RiskAllocationEnv(
        data,
        atr,
        recent,
        signal,
        probability,
        random_start=random_start,
        episode_steps=episode_steps,
    )


def test_binary_signal_filter_observation_and_fixed_cost_contract() -> None:
    env = make_env()
    observation, info = env.reset(seed=7)
    assert env.action_space.contains(ACTION_SKIP)
    assert env.action_space.contains(ACTION_TAKE)
    assert tuple(ACTION_FRACTIONS) == (0.0, 1.0)
    assert tuple(ACTION_NAMES) == ("SKIP_SIGNAL", "TAKE_SIGNAL")
    assert observation.shape == (len(OBSERVATION_FIELDS),)
    assert env.observation_space.contains(observation)
    assert observation[-1] == 1.0
    assert info["fee_bps_per_side"] == 25.0
    assert info["slippage_bps_per_side"] == 0.0
    assert FEE_BPS_PER_SIDE == 25.0
    assert SLIPPAGE_BPS_PER_SIDE == 0.0
    assert ROUND_TRIP_COST_BPS == 50.0
    assert info["action_names"] == ["SKIP_SIGNAL", "TAKE_SIGNAL"]
    assert info["Champion"] is None
    assert info["order_permission"] is False
    assert info["trading_authorization"] == "NO-GO"


def test_signal_episode_latches_take_and_forces_flat_when_signal_turns_off() -> None:
    env = make_env(signals=(True, True, False, False, True, True))
    observation, _ = env.reset(seed=1)
    assert observation[-1] == 1.0

    observation, _, _, _, entered = env.step(ACTION_TAKE)
    assert entered["signal_episode_started"] is True
    assert entered["action_effective"] is True
    assert entered["action_name"] == "TAKE_SIGNAL"
    assert entered["applied_target_fraction"] == 1.0
    assert entered["current_position_fraction"] > 0.99
    assert observation[-1] == 0.0

    _, _, _, _, latched = env.step(ACTION_SKIP)
    assert latched["signal_episode_started"] is False
    assert latched["action_effective"] is False
    assert latched["fill_notional_quote"] == pytest.approx(0.0)
    assert latched["applied_target_fraction"] == 1.0

    _, _, _, _, exited = env.step(ACTION_TAKE)
    assert exited["primary_buy_signal"] is False
    assert exited["applied_target_fraction"] == 0.0
    assert exited["fill_notional_quote"] > 0.0
    assert exited["current_position_fraction"] == pytest.approx(0.0)
    assert exited["trade_count"] == 1

    env.step(ACTION_TAKE)
    _, _, _, _, second_entry = env.step(ACTION_TAKE)
    assert second_entry["signal_episode_started"] is True
    assert second_entry["action_effective"] is True
    assert second_entry["candidate_signal_episode_count"] == 2
    assert second_entry["accepted_signal_episode_count"] == 2
    assert second_entry["trade_count"] == 2


def test_skip_decision_is_latched_for_the_whole_signal_episode() -> None:
    env = make_env(signals=(True, True, False, True, True, False))
    env.reset(seed=2)
    _, _, _, _, skipped = env.step(ACTION_SKIP)
    assert skipped["signal_episode_started"] is True
    assert skipped["applied_target_fraction"] == 0.0
    _, _, _, _, skipped_latched = env.step(ACTION_TAKE)
    assert skipped_latched["signal_episode_started"] is False
    assert skipped_latched["applied_target_fraction"] == 0.0
    assert skipped_latched["fill_notional_quote"] == pytest.approx(0.0)

    env.step(ACTION_SKIP)
    _, _, _, _, accepted = env.step(ACTION_TAKE)
    assert accepted["signal_episode_started"] is True
    assert accepted["signal_episode_taken"] is True
    assert accepted["applied_target_fraction"] == 1.0

    metrics = env.get_metrics()
    assert metrics["candidate_signal_episode_count"] == 2
    assert metrics["accepted_signal_episode_count"] == 1
    assert metrics["skipped_signal_episode_count"] == 1
    assert metrics["signal_acceptance_rate"] == pytest.approx(0.5)


def test_no_signal_forces_flat_after_a_downward_gap() -> None:
    env = make_env(
        prices=(100.0, 100.0, 50.0, 50.0, 50.0, 50.0),
        signals=(True, False, False, False, False, False),
    )
    env.reset()
    env.step(ACTION_TAKE)
    assert env._quantity > 0.0

    _, _, _, _, exited = env.step(ACTION_TAKE)
    assert env._quantity == pytest.approx(0.0)
    assert exited["fill_notional_quote"] > 0.0
    assert exited["applied_target_fraction"] == 0.0
    assert exited["current_position_fraction"] == pytest.approx(0.0)
    assert exited["trade_count"] == 1


def test_fee_is_charged_on_entry_and_forced_signal_exit() -> None:
    env = make_env(signals=(True, False, False, False, False, False))
    env.reset(seed=3)
    _, _, _, _, bought = env.step(ACTION_TAKE)
    assert bought["fill_fee_quote"] > 0.0
    buy_fee = bought["fill_fee_quote"]
    _, _, _, _, sold = env.step(ACTION_SKIP)
    sell_fee = sold["fill_fee_quote"]
    assert sell_fee > 0.0
    assert env.get_metrics()["total_fees_quote"] == pytest.approx(buy_fee + sell_fee)
    assert env.get_metrics()["total_turnover_quote"] > 0.0
    assert env.get_metrics()["trade_count"] == 1
    assert len(env.closed_trade_returns) == 1


def test_recent_returns_primary_probability_and_decision_pending_are_observable() -> (
    None
):
    data = market_data()
    recent = np.tile(np.asarray((0.01, -0.02, 0.03)), (len(data), 1))
    env = RiskAllocationEnv(
        data,
        np.full(len(data), 0.015),
        recent,
        np.ones(len(data), dtype=bool),
        np.full(len(data), 0.75),
    )
    observation, _ = env.reset()
    assert observation.tolist() == pytest.approx(
        [0.015, 0.0, 0.0, 0.01, -0.02, 0.03, 0.75, 1.0, 1.0]
    )
    observation, _, _, _, _ = env.step(ACTION_TAKE)
    assert observation[-1] == 0.0


def test_drawdown_penalty_is_scaled_but_hard_drawdown_observation_remains() -> None:
    data = market_data((100.0, 100.0, 80.0, 80.0, 80.0, 80.0))
    data[2, 3] = 79.0
    env = RiskAllocationEnv(
        data,
        np.full(len(data), 0.01),
        np.zeros((len(data), 3)),
        np.ones(len(data), dtype=bool),
        np.ones(len(data), dtype=float),
    )
    env.reset()
    env.step(ACTION_TAKE)
    _, reward, _, _, info = env.step(ACTION_SKIP)
    assert info["max_drawdown"] > DRAWDOWN_LIMIT
    assert info["drawdown_penalty"] > 0.0
    assert reward < 0.0
    assert DRAWDOWN_INCREMENT_PENALTY == 0.5
    assert DRAWDOWN_EXCESS_PENALTY == 2.0
    assert DRAWDOWN_SEVERE_PENALTY == 4.0


def test_terminal_liquidation_closes_taken_signal_episode() -> None:
    env = make_env(signals=(True, True, True, True, True, True), episode_steps=2)
    env.reset()
    env.step(ACTION_TAKE)
    _, _, terminated, truncated, info = env.step(ACTION_SKIP)
    assert terminated is True
    assert truncated is False
    assert info["current_position_fraction"] == pytest.approx(0.0)
    assert info["trade_count"] == 1
    assert info["accepted_signal_episode_count"] == 1
    with pytest.raises(RuntimeError, match="episode_already_terminated"):
        env.step(ACTION_SKIP)


def test_random_training_reset_is_seeded_and_stays_in_range() -> None:
    env = make_env(random_start=True, episode_steps=2)
    first, _ = env.reset(seed=17)
    first_index = env._index
    second, _ = env.reset(seed=17)
    assert env._index == first_index
    assert np.array_equal(first, second)
    assert env._episode_end - env._index == 2


def test_random_training_windows_never_start_inside_signal_episode() -> None:
    signals = (
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
    )
    env = make_env(
        prices=tuple(100.0 for _ in signals),
        signals=signals,
        random_start=True,
        episode_steps=4,
    )
    for seed in range(64):
        env.reset(seed=seed)
        start = env._index
        assert start == 0 or not (signals[start] and signals[start - 1])
        assert env._episode_end - start == 4


def test_invalid_market_shapes_gaps_and_candle_geometry_fail_closed() -> None:
    data = market_data()
    with pytest.raises(ValueError, match="six_columns"):
        RiskAllocationEnv(
            data[:, :5], np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6)
        )
    bad_gap = data.copy()
    bad_gap[2, 0] += 1.0
    with pytest.raises(ValueError, match="contiguous"):
        RiskAllocationEnv(bad_gap, np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6))
    bad_high = data.copy()
    bad_high[2, 2] = 1.0
    with pytest.raises(ValueError, match="high_below"):
        RiskAllocationEnv(
            bad_high, np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6)
        )
    bad_low = data.copy()
    bad_low[2, 3] = 1000.0
    with pytest.raises(ValueError, match="low_above"):
        RiskAllocationEnv(bad_low, np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6))
    bad_volume = data.copy()
    bad_volume[1, 5] = -1.0
    with pytest.raises(ValueError, match="volume_must"):
        RiskAllocationEnv(
            bad_volume, np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6)
        )
    nonfinite = data.copy()
    nonfinite[2, 4] = np.nan
    with pytest.raises(ValueError, match="non_finite"):
        RiskAllocationEnv(
            nonfinite, np.ones(6), np.zeros((6, 3)), np.ones(6), np.ones(6)
        )


def test_invalid_risk_inputs_and_configuration_fail_closed() -> None:
    data = market_data()
    signal = np.ones(6, dtype=bool)
    recent = np.zeros((6, 3))
    probability = np.ones(6)
    with pytest.raises(ValueError, match="atr_fraction_length"):
        RiskAllocationEnv(data, np.ones(5), recent, signal, probability)
    with pytest.raises(ValueError, match="atr_fraction_must_be_non_negative"):
        RiskAllocationEnv(data, -np.ones(6), recent, signal, probability)
    with pytest.raises(ValueError, match="recent_returns_must_have_three"):
        RiskAllocationEnv(data, np.ones(6), np.zeros((6, 2)), signal, probability)
    with pytest.raises(ValueError, match="recent_returns_must_be_finite"):
        invalid = recent.copy()
        invalid[1, 1] = np.inf
        RiskAllocationEnv(data, np.ones(6), invalid, signal, probability)
    with pytest.raises(ValueError, match="primary_signal_must_be_boolean"):
        RiskAllocationEnv(data, np.ones(6), recent, np.full(6, 2), probability)
    with pytest.raises(ValueError, match="baseline_signal_probability_out"):
        RiskAllocationEnv(data, np.ones(6), recent, signal, np.full(6, 2.0))
    with pytest.raises(ValueError, match="initial_equity"):
        RiskAllocationEnv(
            data, np.ones(6), recent, signal, probability, initial_equity=0.0
        )
    with pytest.raises(ValueError, match="episode_steps_must_be_shorter"):
        RiskAllocationEnv(
            data, np.ones(6), recent, signal, probability, episode_steps=6
        )


def test_action_validation_and_safety_metadata() -> None:
    env = make_env()
    env.reset()
    for invalid in (True, 1.5, "1", -1, 2):
        with pytest.raises(ValueError):
            env.step(invalid)
    _, _, _, _, info = env.step(ACTION_SKIP)
    assert info["risk_only_action"] is True
    assert info["private_api_calls"] == 0
    assert info["real_order_attempts"] == 0
    assert info["trading_authorization"] == "NO-GO"
    assert env.get_metrics()["max_drawdown"] >= 0.0
    env.close()


def test_reset_rejects_unknown_options_and_tiny_episodes() -> None:
    env = make_env()
    with pytest.raises(ValueError, match="reset_options_not_supported"):
        env.reset(options={"start": 2})
    with pytest.raises(ValueError, match="episode_steps_must_be_at_least_two"):
        make_env(episode_steps=1)


def test_metrics_handle_no_taken_signals_without_fabricating_profit_factor() -> None:
    env = make_env(signals=(False, False, False, False, False, False))
    env.reset()
    while True:
        _, _, done, _, _ = env.step(ACTION_SKIP)
        if done:
            break
    metrics = env.get_metrics()
    assert metrics["trade_count"] == 0
    assert metrics["candidate_signal_episode_count"] == 0
    assert metrics["net_profit_factor"] is None
    assert metrics["win_rate"] is None
    assert metrics["statistical_z_score"] is None
