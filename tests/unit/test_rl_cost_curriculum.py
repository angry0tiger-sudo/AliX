"""Behavioral tests for fixed-cost RL research accounting."""

from __future__ import annotations

import math
from typing import cast

import pytest

from services.rl_research.rl_cost_curriculum import TrainingTransactionCosts


@pytest.mark.parametrize(
    ("fee", "slippage"),
    [(0.0, 0.0), (25.0, 20.0), (9999.0, 9999.0)],
)
def test_configured_costs_are_fixed_from_the_first_step(
    fee: float, slippage: float
) -> None:
    costs = TrainingTransactionCosts(fee, slippage)
    assert costs.stage_for_timestep(0) == "full_cost"
    assert costs.stage_for_timestep(100) == "full_cost"
    assert costs.costs_for_timestep(0) == (fee, slippage)
    assert costs.costs_for_timestep(100) == (fee, slippage)
    assert costs.observe_step(1, 0) == (fee, slippage)
    assert costs.observe_step(2, 1) == (fee, slippage)
    assert costs.observe_step(3, 3) == (fee, slippage)
    report = costs.report(3)
    assert report["initial_costs_applied_before_first_step"] is True
    assert report["all_training_steps_full_cost"] is True
    assert report["zero_cost_steps"] == 0
    assert report["ramp_steps"] == 0
    assert report["full_cost_steps"] == 3
    assert report["total_policy_closed_trade_count"] == 3
    assert report["closed_trades_by_stage"] == {"full_cost": 3}
    assert report["status"] == "fixed_costs_verified"


@pytest.mark.parametrize(
    "value",
    [-1.0, 10_000.0, float("nan"), float("inf"), True, "25"],
)
def test_cost_constructor_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        TrainingTransactionCosts(cast(float, value), 20.0)
    with pytest.raises(ValueError):
        TrainingTransactionCosts(25.0, cast(float, value))


def test_curriculum_rejects_bad_timestamps_counts_and_report_lengths() -> None:
    costs = TrainingTransactionCosts(25.0, 20.0)
    with pytest.raises(ValueError, match="timestep_must_be_non_negative_integer"):
        costs.stage_for_timestep(-1)
    with pytest.raises(ValueError, match="timestep_must_be_non_negative_integer"):
        costs.stage_for_timestep(True)
    with pytest.raises(
        ValueError, match="training_timesteps_must_be_contiguous_positive_integers"
    ):
        costs.observe_step(0, 0)
    with pytest.raises(
        ValueError, match="episode_closed_trade_count_must_be_non_negative_integer"
    ):
        costs.observe_step(1, -1)
    with pytest.raises(
        ValueError, match="episode_closed_trade_count_must_be_non_negative_integer"
    ):
        costs.observe_step(1, True)

    assert costs.observe_step(1, 2) == (25.0, 20.0)
    with pytest.raises(
        ValueError, match="training_timesteps_must_be_contiguous_positive_integers"
    ):
        costs.observe_step(3, 2)
    with pytest.raises(ValueError, match="training_cost_step_count_mismatch"):
        costs.report(0)
    with pytest.raises(ValueError, match="training_cost_step_count_mismatch"):
        costs.report(2)
    report = costs.report(1)
    assert report["all_training_steps_full_cost"] is True
    assert report["last_training_step_cost_stage"] == "full_cost"


def test_episode_trade_counter_reset_does_not_create_negative_trade_delta() -> None:
    costs = TrainingTransactionCosts(10.0, 15.0)
    costs.observe_step(1, 3)
    costs.observe_step(2, 0)
    costs.observe_step(3, 2)
    report = costs.report(3)
    assert report["total_policy_closed_trade_count"] == 5
    assert report["closed_trades_by_stage"] == {"full_cost": 5}
    assert report["steps_by_stage"] == {"full_cost": 3}


def test_unstarted_report_is_explicit_and_never_claims_full_cost_steps() -> None:
    costs = TrainingTransactionCosts(1.0, 2.0)
    report = costs.report(0)
    assert report["all_training_steps_full_cost"] is False
    assert report["last_training_step_cost_stage"] == "not_started"
    assert report["status"] == "not_started_or_incomplete"
    assert report["steps_by_stage"] == {"full_cost": 0}
    assert math.isfinite(float(report["initial_fee_bps_per_side"]))
