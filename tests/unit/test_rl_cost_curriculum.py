from __future__ import annotations

import pytest

from services.rl_research.rl_cost_curriculum import TrainingTransactionCosts


def test_configured_costs_apply_from_first_step_and_never_ramp() -> None:
    costs = TrainingTransactionCosts(target_fee_bps=25.0, target_slippage_bps=20.0)

    assert costs.stage_for_timestep(0) == "full_cost"
    assert costs.stage_for_timestep(1) == "full_cost"
    assert costs.stage_for_timestep(1_000_000) == "full_cost"
    assert costs.costs_for_timestep(1) == (25.0, 20.0)
    assert costs.observe_step(1, 0) == (25.0, 20.0)
    assert costs.observe_step(2, 1) == (25.0, 20.0)
    assert costs.observe_step(3, 2) == (25.0, 20.0)
    assert costs.observe_step(4, 2) == (25.0, 20.0)

    report = costs.report(4)
    assert report["scheme"] == "fixed_configured_costs_from_first_training_step"
    assert report["initial_fee_bps_per_side"] == 25.0
    assert report["initial_slippage_bps_per_side"] == 20.0
    assert report["zero_cost_steps"] == 0
    assert report["ramp_steps"] == 0
    assert report["initial_costs_applied_before_first_step"] is True
    assert report["all_training_steps_full_cost"] is True
    assert report["full_cost_steps"] == 4
    assert report["steps_by_stage"] == {"full_cost": 4}
    assert report["total_policy_closed_trade_count"] == 2
    assert report["closed_trades_by_stage"] == {"full_cost": 2}
    assert report["status"] == "fixed_costs_verified"


def test_episode_counter_resets_do_not_erase_prior_completed_trades() -> None:
    costs = TrainingTransactionCosts(25.0, 20.0)
    costs.observe_step(1, 1)
    costs.observe_step(2, 2)
    costs.observe_step(3, 0)
    costs.observe_step(4, 1)

    report = costs.report(4)
    assert report["total_policy_closed_trade_count"] == 3
    assert report["closed_trades_by_stage"] == {"full_cost": 3}
    assert report["all_training_steps_full_cost"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_fee_bps": -1.0}, "target_fee_bps_out_of_range"),
        ({"target_slippage_bps": 10_000.0}, "target_slippage_bps_out_of_range"),
        ({"target_fee_bps": float("nan")}, "target_fee_bps_out_of_range"),
        ({"target_slippage_bps": True}, "target_slippage_bps_must_be_numeric"),
    ],
)
def test_invalid_fixed_cost_configuration_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {
        "target_fee_bps": 25.0,
        "target_slippage_bps": 20.0,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TrainingTransactionCosts(**defaults)  # type: ignore[arg-type]


def test_fixed_cost_policy_rejects_non_contiguous_observations() -> None:
    costs = TrainingTransactionCosts(25.0, 20.0)
    costs.observe_step(1, 0)
    with pytest.raises(ValueError, match="training_timesteps_must_be_contiguous"):
        costs.observe_step(3, 0)


def test_fixed_cost_policy_rejects_mismatched_report_step_count() -> None:
    costs = TrainingTransactionCosts(25.0, 20.0)
    costs.observe_step(1, 0)
    with pytest.raises(ValueError, match="training_cost_step_count_mismatch"):
        costs.report(2)
