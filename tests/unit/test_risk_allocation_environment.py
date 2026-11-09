from __future__ import annotations

import numpy as np
import pytest

from services.rl_research.risk_allocation_environment import (
    ACTION_FRACTIONS,
    ROUND_TRIP_COST_BPS,
    RiskAllocationEnv,
)


def market_data(
    rows: int = 6, *, prices: tuple[float, ...] | None = None
) -> np.ndarray:
    close = np.asarray(prices or tuple(100.0 for _ in range(rows)), dtype=np.float64)
    index = np.arange(rows, dtype=np.float64)
    timestamp = 1_700_000_000_000.0 + index * 3_600_000.0
    return np.column_stack(
        (timestamp, close, close * 1.01, close * 0.99, close, np.full(rows, 10.0))
    )


def make_env(
    *,
    rows: int = 6,
    prices: tuple[float, ...] | None = None,
    signals: tuple[bool, ...] | None = None,
) -> RiskAllocationEnv:
    data = market_data(rows, prices=prices)
    atr = np.full(rows, 0.01, dtype=np.float64)
    primary = np.asarray(
        signals if signals is not None else tuple(True for _ in range(rows)),
        dtype=np.bool_,
    )
    return RiskAllocationEnv(
        data,
        atr,
        primary,
        bar_interval_ms=3_600_000,
        initial_equity=10_000.0,
    )


def test_action_space_is_only_target_exposure_and_observation_is_risk_state() -> None:
    env = make_env()
    observation, info = env.reset(seed=7)
    assert env.action_space.n == 5
    assert ACTION_FRACTIONS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert observation.shape == (4,)
    assert observation.tolist() == pytest.approx([0.01, 0.0, 0.0, 1.0])
    assert info["round_trip_cost_bps"] == 30.0
    assert info["order_permission"] is False
    assert info["Champion"] is None


def test_primary_signal_blocks_new_exposure_but_allows_risk_reduction() -> None:
    env = make_env(signals=(False, False, True, False, False, False))
    env.reset()
    _, _, _, _, blocked = env.step(4)
    assert blocked["proposed_target_fraction"] == 1.0
    assert blocked["applied_target_fraction"] == 0.0
    assert blocked["current_position_fraction"] == pytest.approx(0.0)

    _, _, _, _, still_blocked = env.step(2)
    assert still_blocked["primary_signal"] is False
    assert still_blocked["current_position_fraction"] == pytest.approx(0.0)

    _, _, _, _, signalled = env.step(2)
    assert signalled["primary_signal"] is True
    assert signalled["current_position_fraction"] > 0.0

    _, _, _, _, reduced = env.step(0)
    assert reduced["primary_signal"] is False
    assert reduced["current_position_fraction"] == pytest.approx(0.0)


def test_cost_contract_applies_fee_and_slippage_on_each_fill() -> None:
    env = make_env(signals=(True, False, False, False, False, False))
    env.reset()
    _, _, _, _, bought = env.step(1)
    buy_cost = bought["total_cost_quote"]
    assert buy_cost > 0
    assert bought["total_traded_quote"] > 0
    _, _, _, _, sold = env.step(0)
    assert sold["current_position_fraction"] == pytest.approx(0.0)
    assert sold["total_cost_quote"] > buy_cost
    assert ROUND_TRIP_COST_BPS == 30.0
    assert sold["position_adjustments"] == 2


def test_terminal_step_liquidates_remaining_position_and_closes_episode() -> None:
    env = make_env(rows=3)
    env.reset()
    env.step(2)
    _, _, terminated, truncated, info = env.step(2)
    assert terminated is True
    assert truncated is False
    assert info["current_position_fraction"] == pytest.approx(0.0)
    assert info["terminal_liquidation_cost"] > 0
    with pytest.raises(RuntimeError, match="already_terminated"):
        env.step(0)


def test_drawdown_is_observed_and_penalized_after_price_decline() -> None:
    env = make_env(
        rows=5,
        prices=(100.0, 100.0, 80.0, 80.0, 80.0),
        signals=(True, True, False, False, False),
    )
    env.reset()
    _, _, _, _, entered = env.step(4)
    assert entered["current_position_fraction"] > 0.9
    observation, reward, _, _, info = env.step(4)
    assert info["current_drawdown"] > 0.0
    assert info["max_drawdown"] > 0.0
    assert observation[1] > 0.0
    assert reward < 0.0


def test_invalid_market_shapes_timestamps_prices_geometry_and_volume_fail_closed() -> None:
    data = market_data()
    with pytest.raises(ValueError, match="six_columns"):
        RiskAllocationEnv(data[:, :5], np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    with pytest.raises(ValueError, match="at_least_two"):
        RiskAllocationEnv(data[:1], np.ones(1), np.ones(1, dtype=bool), bar_interval_ms=3_600_000)

    bad = data.copy()
    bad[2, 0] += 1.0
    with pytest.raises(ValueError, match="contiguous"):
        RiskAllocationEnv(bad, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    bad = data.copy()
    bad[1, 1] = 0.0
    with pytest.raises(ValueError, match="prices_must_be_positive"):
        RiskAllocationEnv(bad, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    bad = data.copy()
    bad[1, 2] = bad[1, 3] - 1.0
    with pytest.raises(ValueError, match="geometry_invalid"):
        RiskAllocationEnv(bad, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    bad = data.copy()
    bad[1, 5] = -1.0
    with pytest.raises(ValueError, match="volume_must_be_non_negative"):
        RiskAllocationEnv(bad, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    bad = data.copy()
    bad[1, 4] = np.nan
    with pytest.raises(ValueError, match="contains_non_finite"):
        RiskAllocationEnv(bad, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)


def test_invalid_auxiliary_arrays_and_configuration_fail_closed() -> None:
    data = market_data()
    with pytest.raises(ValueError, match="atr_fraction_length"):
        RiskAllocationEnv(data, np.ones(5), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    with pytest.raises(ValueError, match="atr_fraction_must"):
        RiskAllocationEnv(data, np.asarray([0.01, 0.01, np.nan, 0.01, 0.01, 0.01]), np.ones(6, dtype=bool), bar_interval_ms=3_600_000)
    with pytest.raises(ValueError, match="primary_signal_must_be_boolean"):
        RiskAllocationEnv(data, np.ones(6), np.asarray([0, 0, 0, 0, 0, 2]), bar_interval_ms=3_600_000)
    with pytest.raises(ValueError, match="primary_signal_length"):
        RiskAllocationEnv(data, np.ones(6), np.ones(5, dtype=bool), bar_interval_ms=3_600_000)
    with pytest.raises(ValueError, match="bar_interval_ms"):
        RiskAllocationEnv(data, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=0)
    with pytest.raises(ValueError, match="initial_equity"):
        RiskAllocationEnv(data, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000, initial_equity=0.0)
    with pytest.raises(ValueError, match="reward_penalties"):
        RiskAllocationEnv(data, np.ones(6), np.ones(6, dtype=bool), bar_interval_ms=3_600_000, drawdown_level_penalty=-1.0)


def test_action_validation_and_non_live_safety_metadata() -> None:
    env = make_env()
    env.reset()
    for action in (True, 1.5, "1", -1, 5):
        with pytest.raises(ValueError):
            env.step(action)
    _, _, _, _, info = env.step(0)
    assert info["risk_only_action"] is True
    assert info["private_api_calls"] == 0
    assert info["real_order_attempts"] == 0
    assert info["trading_authorization"] == "NO-GO"
