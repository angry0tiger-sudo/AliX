"""Synthetic-only verification of the G4.6 volume-feature ablation contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.run_regime_r_g46_volume_ablation import (
    RETAINED_FEATURE_COLUMNS,
    VOLUME_FEATURE_COLUMNS,
    build_volume_ablation_features,
    evaluate_cost_acceptance,
    load_g46_preregistration,
)
from scripts.run_regime_r_trials import FEATURE_COLUMNS, build_features


def _frame(volume: np.ndarray) -> pd.DataFrame:
    rows = int(volume.size)
    timestamps = pd.date_range(
        "2022-01-01T00:30:00Z", periods=rows, freq="h"
    )
    close = 100.0 + np.arange(rows, dtype=np.float64) * 0.1
    opened = close - 0.02
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opened,
            "high": close + 0.1,
            "low": opened - 0.1,
            "close": close,
            "volume": volume.astype(np.float64),
        }
    )


def _passing_fold_records() -> list[dict[str, Any]]:
    return [
        {"metrics": {"signal_count_above_lower_bound": 1}},
        {"metrics": {"signal_count_above_lower_bound": 2}},
        {"metrics": {"signal_count_above_lower_bound": 1}},
    ]


def test_g46_preregistration_is_byte_locked_and_matches_feature_contract() -> None:
    prereg = load_g46_preregistration()

    assert prereg["preregistration"]["status"] == "PREREGISTERED_LOCKED"
    assert prereg["feature_ablation"]["removed_predictors"] == list(
        VOLUME_FEATURE_COLUMNS
    )
    assert prereg["feature_ablation"]["retained_predictors"] == list(
        RETAINED_FEATURE_COLUMNS
    )
    assert prereg["frozen_g3"]["must_remain_byte_identical"] is True
    assert prereg["safety"]["holdout_bytes_read"] is False


def test_ablation_removes_exactly_two_volume_predictors_and_preserves_order() -> None:
    frame = _frame(np.linspace(5.0, 25.0, 80, dtype=np.float64))
    baseline = build_features(frame)
    ablated = build_volume_ablation_features(frame)

    assert tuple(baseline.columns) == FEATURE_COLUMNS
    assert tuple(ablated.columns) == RETAINED_FEATURE_COLUMNS
    assert len(baseline.columns) == 13
    assert len(ablated.columns) == 11
    assert not set(VOLUME_FEATURE_COLUMNS).intersection(ablated.columns)
    pd.testing.assert_frame_equal(
        ablated,
        baseline.drop(columns=list(VOLUME_FEATURE_COLUMNS)),
    )


def test_retained_predictors_are_invariant_to_raw_volume_changes() -> None:
    volume_zero = np.zeros(96, dtype=np.float64)
    volume_variable = np.linspace(1.0, 10_000.0, 96, dtype=np.float64)
    ablated_zero = build_volume_ablation_features(_frame(volume_zero))
    ablated_variable = build_volume_ablation_features(_frame(volume_variable))

    pd.testing.assert_frame_equal(ablated_zero, ablated_variable)


@pytest.mark.parametrize("cost_bps", [50, 90])
def test_all_four_g46_gates_pass_at_each_cost_when_every_condition_is_met(
    cost_bps: int,
) -> None:
    result = evaluate_cost_acceptance(
        {
            "net_profit_factor": 1.10,
            "trade_count": 100,
            "max_drawdown": 0.15,
        },
        _passing_fold_records(),
        cost_bps,
    )

    assert result["all_gates_passed"] is True
    assert result["qualifying_lower_bound_folds"] == 3


@pytest.mark.parametrize(
    ("metrics", "fold_records", "failed_gate"),
    [
        (
            {"net_profit_factor": 1.099, "trade_count": 120, "max_drawdown": 0.10},
            _passing_fold_records(),
            "net_pf_gte_1_10",
        ),
        (
            {"net_profit_factor": 1.20, "trade_count": 99, "max_drawdown": 0.10},
            _passing_fold_records(),
            "trades_gte_100",
        ),
        (
            {"net_profit_factor": 1.20, "trade_count": 120, "max_drawdown": 0.1501},
            _passing_fold_records(),
            "max_dd_lte_15_percent",
        ),
        (
            {"net_profit_factor": 1.20, "trade_count": 120, "max_drawdown": 0.10},
            [
                {"metrics": {"signal_count_above_lower_bound": 1}},
                {"metrics": {"signal_count_above_lower_bound": 0}},
                {"metrics": {"signal_count_above_lower_bound": 1}},
            ],
            "lower_bound_cost_passed_in_all_three_folds",
        ),
    ],
)
def test_g46_acceptance_fails_closed_when_any_registered_gate_fails(
    metrics: dict[str, Any],
    fold_records: list[dict[str, Any]],
    failed_gate: str,
) -> None:
    result = evaluate_cost_acceptance(metrics, fold_records, 50)

    assert result["all_gates_passed"] is False
    assert result["gates"][failed_gate] is False
