"""Unit tests for the checkpoint-sweep helper contracts."""

from __future__ import annotations

import pytest

from scripts.sweep_rl_checkpoints import EXPECTED_STEPS, _action_percentages


def test_checkpoint_sweep_targets_exact_ten_checkpoints() -> None:
    assert EXPECTED_STEPS == tuple(range(100_000, 1_000_001, 100_000))
    assert len(EXPECTED_STEPS) == 10


def test_action_percentages_aggregate_each_discrete_direction() -> None:
    result = _action_percentages(
        {
            "steps": 10,
            "discrete_action_counts": {
                "HOLD": 7,
                "BUY_25": 1,
                "BUY_50": 1,
                "SELL_25": 1,
            },
        }
    )
    assert result["HOLD"] == {"count": 7, "percent": 70.0}
    assert result["BUY"] == {"count": 2, "percent": 20.0}
    assert result["SELL"] == {"count": 1, "percent": 10.0}


def test_action_percentages_fail_closed_when_counts_do_not_match_steps() -> None:
    with pytest.raises(
        ValueError, match="action_count_total_does_not_match_validation_steps"
    ):
        _action_percentages(
            {"steps": 4, "discrete_action_counts": {"HOLD": 3, "BUY_25": 2}}
        )


def test_action_percentages_reject_missing_action_counts() -> None:
    with pytest.raises(ValueError, match="evaluation_missing_discrete_action_counts"):
        _action_percentages({"steps": 1})
