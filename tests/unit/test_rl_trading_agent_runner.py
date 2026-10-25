from __future__ import annotations

import numpy as np
import pytest

from scripts.run_rl_trading_agent import (
    PREVIOUS_HOLDOUT_START_MS,
    _evaluate_policy,
    _exclude_previous_holdout_window,
)
from services.rl_research.rl_trading_environment import MarketObservationScaler


def test_analysis_window_excludes_all_rows_at_or_after_previous_holdout() -> None:
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


class _DeterministicZeroSizeBuyPolicy:
    def predict(self, observation, *, state, episode_start, deterministic):
        assert deterministic is True
        return np.asarray([0.9, 0.0], dtype=np.float32), state


def test_evaluation_distinguishes_zero_size_policy_from_action_mask_block() -> None:
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
        _DeterministicZeroSizeBuyPolicy(),
        data,
        scaler,
        fee_bps=25.0,
        slippage_bps=20.0,
        seed=7,
    )

    assert metrics["initial_equity"] == pytest.approx(10_000.0)
    assert metrics["buy_action_count"] == metrics["steps"]
    assert metrics["masked_buy_action_count"] == 0
    assert metrics["buy_noop_action_count"] == metrics["steps"]
    assert metrics["zero_size_buy_action_count"] == metrics["steps"]
    assert metrics["policy_buy_zero_size_request_count"] == metrics["steps"]
    assert metrics["invalid_action_penalty_count"] == 0
    assert metrics["buy_fill_count"] == 0
    assert metrics["policy_position_size_max"] == pytest.approx(0.0)
    assert metrics["policy_action_mask_reason_counts"] == {
        "zero_buy_size": metrics["steps"]
    }
