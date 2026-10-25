from __future__ import annotations

import pytest

from services.rl_research.rl_cost_curriculum import TrainingCostCurriculum


def test_cost_ramp_waits_for_completed_policy_trades_and_reaches_target() -> None:
    curriculum = TrainingCostCurriculum(
        target_fee_bps=25.0,
        target_slippage_bps=20.0,
        minimum_closed_trades=2,
        ramp_steps=4,
        ramp_update_interval_steps=1,
    )

    assert curriculum.observe_step(1, 0) == (0.0, 0.0)
    assert curriculum.observe_step(2, 1) == (0.0, 0.0)
    assert curriculum.observe_step(3, 2) == (0.0, 0.0)
    assert curriculum.costs_for_timestep(4) == (0.0, 0.0)
    assert curriculum.costs_for_timestep(5) == pytest.approx((6.25, 5.0))
    assert curriculum.costs_for_timestep(6) == pytest.approx((12.5, 10.0))
    assert curriculum.costs_for_timestep(7) == pytest.approx((18.75, 15.0))
    assert curriculum.costs_for_timestep(8) == (25.0, 20.0)

    for timestep in range(4, 9):
        curriculum.observe_step(timestep, 2)
    report = curriculum.report(8)
    assert report["ramp_start_timestep"] == 4
    assert report["ramp_start_after_policy_closed_trade_count"] == 2
    assert report["minimum_trade_threshold_met"] is True
    assert report["ramp_completed"] is True
    assert report["steps_by_stage"] == {"zero_cost": 3, "ramp": 4, "full_cost": 1}


def test_episode_counter_resets_do_not_erase_prior_completed_trades() -> None:
    curriculum = TrainingCostCurriculum(
        target_fee_bps=25.0,
        target_slippage_bps=20.0,
        minimum_closed_trades=3,
        ramp_steps=4,
        ramp_update_interval_steps=1,
    )
    curriculum.observe_step(1, 1)
    curriculum.observe_step(2, 2)
    curriculum.observe_step(3, 0)
    curriculum.observe_step(4, 1)

    report = curriculum.report(4)
    assert report["total_policy_closed_trade_count"] == 3
    assert report["ramp_start_timestep"] == 5
    assert report["ramp_start_after_policy_closed_trade_count"] == 3
    assert report["closed_trades_by_stage"]["zero_cost"] == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_fee_bps": -1.0}, "target_fee_bps_out_of_range"),
        ({"target_slippage_bps": 10_000.0}, "target_slippage_bps_out_of_range"),
        ({"minimum_closed_trades": 0}, "minimum_closed_trades_must_be_positive_integer"),
        ({"ramp_steps": 0}, "ramp_steps_must_be_positive_integer"),
        (
            {"ramp_steps": 10, "ramp_update_interval_steps": 3},
            "ramp_update_interval_must_divide_ramp_steps",
        ),
    ],
)
def test_invalid_curriculum_configuration_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {
        "target_fee_bps": 25.0,
        "target_slippage_bps": 20.0,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TrainingCostCurriculum(**defaults)  # type: ignore[arg-type]


def test_curriculum_rejects_non_contiguous_observations() -> None:
    curriculum = TrainingCostCurriculum(25.0, 20.0)
    curriculum.observe_step(1, 0)
    with pytest.raises(ValueError, match="training_timesteps_must_be_contiguous"):
        curriculum.observe_step(3, 0)
