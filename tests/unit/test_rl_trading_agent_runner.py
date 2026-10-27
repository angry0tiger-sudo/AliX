from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.run_rl_trading_agent import (
    PREVIOUS_HOLDOUT_START_MS,
    _checkpoint_validation_row,
    _entropy_coefficient_at_step,
    _verify_model_reload_roundtrip,
    _evaluate_policy,
    _exclude_previous_holdout_window,
    _guidance_reward_scale_at_step,
)
from services.rl_research.rl_trading_environment import (
    ACTION_BUY,
    ACTION_NAMES,
    MarketObservationScaler,
)


def test_pre_cutoff_window_separates_latest_holdout_tail() -> None:
    timestamps = np.asarray(
        [
            PREVIOUS_HOLDOUT_START_MS - 120_000,
            PREVIOUS_HOLDOUT_START_MS - 60_000,
            PREVIOUS_HOLDOUT_START_MS,
            PREVIOUS_HOLDOUT_START_MS + 60_000,
        ],
        dtype=np.float64,
    )
    data = np.column_stack((timestamps, np.ones((4, 5), dtype=np.float64) * 100.0))
    analysis, excluded_rows = _exclude_previous_holdout_window(data)
    assert len(analysis) == 2
    assert excluded_rows == 2
    assert analysis[-1, 0] < PREVIOUS_HOLDOUT_START_MS


def test_analysis_window_fails_closed_when_no_prior_holdout_tail_exists() -> None:
    timestamps = np.asarray(
        [PREVIOUS_HOLDOUT_START_MS - 120_000, PREVIOUS_HOLDOUT_START_MS - 60_000],
        dtype=np.float64,
    )
    data = np.column_stack((timestamps, np.ones((2, 5), dtype=np.float64) * 100.0))
    with pytest.raises(ValueError, match="previous_holdout_window_not_present"):
        _exclude_previous_holdout_window(data)


class _Fixed25PercentBuyPolicy:
    def predict(
        self,
        observation: NDArray[np.float32],
        *,
        state: Any,
        episode_start: NDArray[np.bool_],
        deterministic: bool,
    ) -> tuple[NDArray[np.int64], Any]:
        assert deterministic is True
        return np.asarray(ACTION_BUY, dtype=np.int64), state


def test_evaluation_reports_real_discrete_size_choices_without_zero_volume_buys() -> (
    None
):
    timestamps = 1_700_000_000_000 + np.arange(6) * 60_000
    prices = np.asarray([100.0, 101.0, 102.0, 101.0, 100.0, 99.0])
    data = np.column_stack(
        (
            timestamps,
            prices,
            prices + 1.0,
            prices - 1.0,
            prices,
            np.ones(6),
        )
    ).astype(np.float64)
    scaler = MarketObservationScaler.fit(data)

    metrics = _evaluate_policy(
        _Fixed25PercentBuyPolicy(),
        data,
        scaler,
        fee_bps=25.0,
        slippage_bps=20.0,
        seed=7,
    )

    assert metrics["initial_equity"] == pytest.approx(10_000.0)
    assert metrics["buy_action_count"] == metrics["steps"]
    assert metrics["masked_buy_action_count"] == 0
    assert metrics["buy_noop_action_count"] == 0
    assert metrics["zero_size_buy_action_count"] == 0
    assert metrics["policy_buy_zero_size_request_count"] == 0
    assert metrics["buy_fill_count"] == metrics["steps"]
    assert metrics["policy_requested_trade_size_fraction_counts"] == {
        "0.25": metrics["steps"],
        "0.50": 0,
        "1.00": 0,
    }
    assert metrics["policy_requested_trade_size_fraction_mean"] == pytest.approx(0.25)
    assert metrics["invalid_action_penalty_count"] == 0
    assert metrics["signal_guidance_reward_count"] == 0
    assert metrics["signal_guidance_reward_total"] == pytest.approx(0.0)
    assert metrics["total_fees"] > 0.0


def test_signal_guidance_reward_anneals_to_nonzero_floor() -> None:
    assert _guidance_reward_scale_at_step(0) == pytest.approx(0.0002)
    assert _guidance_reward_scale_at_step(150_000) == pytest.approx(0.0001)
    assert _guidance_reward_scale_at_step(270_000) == pytest.approx(0.00002)
    assert _guidance_reward_scale_at_step(300_000) == pytest.approx(0.00002)
    assert _guidance_reward_scale_at_step(1_000_000) == pytest.approx(0.00002)


def test_signal_guidance_floor_invalid_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="floor_exceeds_initial_scale"):
        _guidance_reward_scale_at_step(1, initial_scale=0.0001, minimum_scale=0.0002)
    with pytest.raises(ValueError, match="timestep_invalid"):
        _guidance_reward_scale_at_step(-1)


def test_entropy_coefficient_is_constant_at_0_1_from_step_one() -> None:
    assert _entropy_coefficient_at_step(0) == pytest.approx(0.1)
    assert _entropy_coefficient_at_step(49_999) == pytest.approx(0.1)
    assert _entropy_coefficient_at_step(50_000) == pytest.approx(0.1)
    assert _entropy_coefficient_at_step(1_000_000) == pytest.approx(0.1)


def test_entropy_schedule_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="timestep_invalid"):
        _entropy_coefficient_at_step(-1)
    with pytest.raises(ValueError, match="exploration_steps_invalid"):
        _entropy_coefficient_at_step(0, exploration_steps=0)


def test_checkpoint_validation_row_reports_action_sides_and_economics(
    tmp_path: Any,
) -> None:
    checkpoint_path = tmp_path / "rl_trading_agent_checkpoint_100000_steps.zip"
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    counts = {name: 0 for name in ACTION_NAMES}
    counts.update({"HOLD": 4, "BUY_25": 3, "SELL_25": 1})
    metrics = {
        "steps": 8,
        "discrete_action_counts": counts,
        "buy_action_count": 3,
        "sell_action_count": 1,
        "hold_action_count": 4,
        "net_profit": 15.0,
        "net_return": 0.0015,
        "max_drawdown": 0.02,
        "closed_trade_count": 2,
        "total_fees": 2.0,
        "total_slippage": 1.0,
        "evaluation_passes": 1,
        "deterministic_inference": True,
    }

    row = _checkpoint_validation_row(checkpoint_path, metrics)

    assert row["checkpoint_timesteps"] == 100_000
    assert row["action_distribution"]["HOLD"]["percent"] == pytest.approx(50.0)
    assert row["action_distribution"]["BUY"]["percent"] == pytest.approx(37.5)
    assert row["action_distribution"]["SELL"]["percent"] == pytest.approx(12.5)
    assert row["net_profit"] == pytest.approx(15.0)
    assert row["max_drawdown"] == pytest.approx(0.02)
    assert row["closed_trades"] == 2


class ActionMaskedRecurrentActorCriticPolicy:
    """Test-only stand-in named like the dynamically registered custom policy."""


class _SevenActionSpace:
    n = 7


class _RoundtripModel:
    def __init__(self, action: int = 1, timesteps: int = 100_000) -> None:
        self.action = action
        self.num_timesteps = timesteps
        self.policy = ActionMaskedRecurrentActorCriticPolicy()
        self.action_space = _SevenActionSpace()

    def predict(
        self,
        observation: NDArray[np.float32],
        *,
        state: Any,
        episode_start: NDArray[np.bool_],
        deterministic: bool,
    ) -> tuple[NDArray[np.int64], Any]:
        assert deterministic is True
        assert observation.dtype == np.float32
        assert episode_start.tolist() == [True]
        return np.asarray([self.action], dtype=np.int64), state


def test_model_reload_roundtrip_records_restored_policy_identity(tmp_path: Any) -> None:
    model_path = tmp_path / "rl_trading_agent_recurrent_ppo.zip"
    model_path.write_bytes(b"saved-model")
    original = _RoundtripModel(action=2)
    restored = _RoundtripModel(action=2)
    observation = np.zeros(11, dtype=np.float32)

    evidence = _verify_model_reload_roundtrip(
        original, restored, model_path, observation
    )

    assert evidence["status"] == "passed"
    assert evidence["model_sha256"]
    assert evidence["original_policy_class"] == "ActionMaskedRecurrentActorCriticPolicy"
    assert evidence["reloaded_policy_class"] == "ActionMaskedRecurrentActorCriticPolicy"
    assert evidence["policy_class_match"] is True
    assert evidence["deterministic_probe_action_match"] is True
    assert evidence["deterministic_probe_action"] == 2


def test_model_reload_roundtrip_fails_closed_when_prediction_changes(
    tmp_path: Any,
) -> None:
    model_path = tmp_path / "rl_trading_agent_recurrent_ppo.zip"
    model_path.write_bytes(b"saved-model")
    observation = np.zeros(11, dtype=np.float32)

    with pytest.raises(
        RuntimeError, match="reloaded_model_deterministic_prediction_mismatch"
    ):
        _verify_model_reload_roundtrip(
            _RoundtripModel(action=1),
            _RoundtripModel(action=3),
            model_path,
            observation,
        )
