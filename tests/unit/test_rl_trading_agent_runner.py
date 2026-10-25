from __future__ import annotations

import numpy as np
import pytest

from scripts.run_rl_trading_agent import (
    PREVIOUS_HOLDOUT_START_MS,
    _exclude_previous_holdout_window,
)


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
