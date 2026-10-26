"""Unit tests for the checkpoint-sweep report helpers."""

from __future__ import annotations

from scripts.sweep_rl_checkpoints import EXPECTED_STEPS, _action_percentages


def test_checkpoint_sweep_targets_exact_ten_checkpoints() -> None:
    assert EXPECTED_STEPS == tuple(range(100_000, 1_000_001, 100_000))
    assert len(EXPECTED_STEPS) == 10


def test_action_percentages_use_total_validation_steps() -> None:
    result = _action_percentages({
        "steps": 10,
        "discrete_action_counts": {
            "HOLD": 7,
            "BUY_25": 1,
            "BUY_50": 1,
            "SELL_25": 1,
        },
    })
    assert result["HOLD"] == {"count": 7, "percent": 70.0}
    assert result["BUY"] == {"count": 2, "percent": 20.0}
    assert result["SELL"] == {"count": 1, "percent": 10.0}
