from __future__ import annotations

from scripts.run_ml_core_real_data_baseline import (
    _build_walk_forward_summary,
    _rank_feature_profiles_by_forward_precision,
    _walk_forward_threshold_optimization_diagnostics,
)
from pandas import Series
import numpy as np


def test_forward_precision_is_primary_profile_ranking_key() -> None:
    ranking = _rank_feature_profiles_by_forward_precision(
        {
            "Baseline": {
                "mean_forward_hold_precision": 0.31,
                "forward_metric_experiment_count": 8,
            },
            "Baseline+Trend": {
                "mean_forward_hold_precision": 0.38,
                "forward_metric_experiment_count": 2,
            },
            "Baseline+Volatility": {
                "mean_forward_hold_precision": 0.34,
                "forward_metric_experiment_count": 3,
            },
        }
    )

    assert [item["feature_profile"] for item in ranking] == [
        "Baseline+Trend",
        "Baseline+Volatility",
        "Baseline",
    ]


def test_walk_forward_threshold_uses_calibration_and_scores_only_test() -> None:
    calibration_values = ["HOLD"] * 40 + ["BUY"] * 40 + ["SELL"] * 40
    calibration_y = Series(calibration_values, index=range(100, 220), dtype=object)
    calibration_probabilities = np.asarray(
        [
            [0.05, 0.05, 0.90]
            if label == "HOLD"
            else [0.90, 0.05, 0.05]
            if label == "BUY"
            else [0.05, 0.90, 0.05]
            for label in calibration_values
        ],
        dtype=float,
    )
    test_values = ["HOLD"] * 20 + ["BUY"] * 20 + ["SELL"] * 20
    test_y = Series(test_values, index=range(500, 560), dtype=object)
    test_probabilities = np.asarray(
        [
            [0.05, 0.05, 0.90]
            if label == "HOLD"
            else [0.90, 0.05, 0.05]
            if label == "BUY"
            else [0.05, 0.90, 0.05]
            for label in test_values
        ],
        dtype=float,
    )

    result = _walk_forward_threshold_optimization_diagnostics(
        calibration_y,
        test_y,
        calibration_probabilities,
        test_probabilities,
        minimum_precision=0.40,
        minimum_recall=0.20,
    )

    assert result["status"] == "SELECTED"
    assert result["calibration"]["row_count"] == 120
    assert result["calibration"]["start_row_index"] == 100
    assert result["calibration"]["end_row_index"] == 219
    assert result["post_calibration_validation"]["start_row_index"] == 500
    assert result["post_calibration_validation"]["end_row_index"] == 559
    assert result["post_calibration_validation"]["evaluation_scope"] == "WALK_FORWARD_TEST_ONLY"
    assert result["post_calibration_validation"]["hold_metrics"]["precision"] == 1.0
    assert result["post_calibration_validation"]["hold_metrics"]["recall"] == 1.0
    assert result["post_calibration_validation"]["calibration_overlap"] is False


def test_walk_forward_threshold_distinguishes_no_feasible_from_low_support() -> None:
    calibration_y = Series(
        ["HOLD"] * 40 + ["BUY"] * 40 + ["SELL"] * 40,
        index=range(120),
        dtype=object,
    )
    calibration_probabilities = np.tile(
        np.asarray([[0.45, 0.45, 0.10]], dtype=float), (120, 1)
    )
    test_y = Series(["HOLD", "BUY", "SELL"] * 20, index=range(300, 360))
    test_probabilities = np.tile(
        np.asarray([[0.45, 0.45, 0.10]], dtype=float), (60, 1)
    )

    result = _walk_forward_threshold_optimization_diagnostics(
        calibration_y,
        test_y,
        calibration_probabilities,
        test_probabilities,
        minimum_precision=0.40,
        minimum_recall=0.20,
    )

    assert result["calibration"]["hold_support"] == 40
    assert result["status"] == "NO_FEASIBLE_THRESHOLD"
    assert result["post_calibration_validation"] is None


def test_walk_forward_summary_compares_matched_profile_support() -> None:
    experiments = []
    for fold in (1, 2, 3):
        for pair, timeframe, baseline_precision, trend_precision in (
            ("BTC/USDT", "15m", 0.30, 0.35),
            ("ETH/USDT", "30m", 0.32, 0.37),
        ):
            for profile, precision in (
                ("Baseline", baseline_precision),
                ("Baseline+Trend", trend_precision),
            ):
                experiments.append(
                    {
                        "fold_index": fold,
                        "pair": pair,
                        "timeframe": timeframe,
                        "feature_profile": profile,
                        "status": "TRAINED",
                        "threshold_optimization": {
                            "post_calibration_validation": {
                                "hold_metrics": {
                                    "precision": precision,
                                    "recall": 0.25,
                                }
                            }
                        },
                    }
                )

    summary = _build_walk_forward_summary(
        experiments, ("Baseline", "Baseline+Trend"), 3
    )

    assert summary["trend_improved_fold_count"] == 3
    assert summary["forward_hold_precision_at_least_0_40_count_by_profile"] == {
        "Baseline": 0,
        "Baseline+Trend": 0,
    }
    for fold in summary["folds"]:
        assert fold["paired_comparison"]["matched_pair_timeframe_count"] == 2
        assert fold["paired_comparison"]["trend_minus_baseline_precision"] == 0.05
