"""Synthetic-only unit tests for Regime R's evaluation mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_regime_r_trials import (
    FEATURE_COLUMNS,
    build_features,
    build_target,
    profit_factor,
    simulate_costed_ledger,
)


def synthetic_frame(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-01T00:30:00Z", periods=rows, freq="h")
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.1
    opens = close - 0.02
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": close + 0.1,
            "low": opens - 0.1,
            "close": close,
            "volume": 10.0 + np.arange(rows, dtype=np.float64) % 7,
        }
    )


def test_feature_schema_and_causality() -> None:
    frame = synthetic_frame()
    features = build_features(frame)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.shape == (len(frame), 13)
    changed = frame.copy()
    changed.loc[70:, "close"] *= 1.5
    changed_features = build_features(changed)
    pd.testing.assert_series_equal(
        features.loc[60],
        changed_features.loc[60],
        check_names=False,
    )


def test_target_uses_next_open_and_horizon_close() -> None:
    frame = synthetic_frame(10)
    target = build_target(frame, 4)
    assert target[0] == frame.loc[4, "close"] / frame.loc[1, "open"] - 1.0
    assert np.isnan(target[-4:]).all()


def test_profit_factor_undefined_without_losses() -> None:
    assert profit_factor([{"net_return": 0.01}, {"net_return": 0.02}]) is None


def test_costed_ledger_strict_gate_and_no_overlap() -> None:
    frame = synthetic_frame(12)
    indices = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    lower = np.asarray([0.005, 0.02, 0.012, 0.004, 0.02, 0.02, 0.02])
    ledger = simulate_costed_ledger(
        frame, 0, 10, 2, indices, lower.astype(np.float64), 50
    )
    assert ledger["signal_count_above_lower_bound"] == 5
    assert ledger["skipped_signal_position_open"] >= 1
    assert ledger["trade_count"] >= 1
    assert ledger["unexecutable_signal_count"] == 0


def test_cost_equality_abstains() -> None:
    frame = synthetic_frame(8)
    ledger = simulate_costed_ledger(
        frame,
        0,
        8,
        1,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0.005, 0.004], dtype=np.float64),
        50,
    )
    assert ledger["signal_count_above_lower_bound"] == 0
    assert ledger["abstention_count"] == 2
    assert ledger["trade_count"] == 0
