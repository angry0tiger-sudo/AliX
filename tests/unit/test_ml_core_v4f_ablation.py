from __future__ import annotations

from typing import Any

from pandas import DataFrame

from scripts.run_ml_core_real_data_baseline import (
    _build_feature_ablation_summary,
    _run_diagnostic_experiments,
)
from services.ml_core.extended_features import FEATURE_PROFILE_ORDER


def test_ablation_orchestration_reuses_one_prepared_matrix_per_pair_timeframe(
    monkeypatch: Any,
) -> None:
    bars = DataFrame({"close": [100.0, 101.0]})
    labels = DataFrame(
        {
            "label": ["BUY", "HOLD"],
            "label_end_idx": [1, 1],
        }
    )
    matrices = {
        profile: DataFrame({f"feature_{index}": [1.0, 2.0]})
        for index, profile in enumerate(FEATURE_PROFILE_ORDER)
    }
    calls: list[dict[str, Any]] = []
    preparation_count = {"resample": 0, "features": 0}

    def fake_resample(candles: DataFrame, timeframe: str) -> DataFrame:
        preparation_count["resample"] += 1
        return bars

    def fake_labels(
        prepared_bars: DataFrame,
        *,
        horizon_candles: int,
        take_profit_multiplier: float,
        stop_loss_multiplier: float,
    ) -> DataFrame:
        assert prepared_bars is bars
        assert horizon_candles == 12
        return labels

    def fake_profiles(prepared_bars: DataFrame) -> dict[str, DataFrame]:
        assert prepared_bars is bars
        preparation_count["features"] += 1
        return matrices

    def fake_run(
        pair: str, timeframe: str, minute_candles: DataFrame, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append({"pair": pair, "timeframe": timeframe, **kwargs})
        return {"feature_profile": kwargs["feature_profile"]}

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.resample_ohlcv", fake_resample
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.apply_adaptive_triple_barrier",
        fake_labels,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.build_feature_profiles",
        fake_profiles,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.run_experiment", fake_run
    )

    datasets = {
        "BTCUSDT": DataFrame({"minute": [1]}),
        "ETHUSDT": DataFrame({"minute": [1]}),
    }
    result = _run_diagnostic_experiments(
        datasets,
        barrier_scenarios=[
            {
                "name": "V4F_Flat_12_Candle",
                "horizon_candles": 12,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
            }
        ],
        calibration_fraction=0.20,
        minimum_hold_precision=0.40,
        minimum_hold_recall=0.20,
        feature_profile_names=FEATURE_PROFILE_ORDER,
    )

    assert len(result) == 40
    assert preparation_count == {"resample": 8, "features": 8}
    assert [call["feature_profile"] for call in calls[:5]] == list(
        FEATURE_PROFILE_ORDER
    )
    for start in range(0, len(calls), len(FEATURE_PROFILE_ORDER)):
        group = calls[start : start + len(FEATURE_PROFILE_ORDER)]
        assert len({id(item["prepared_bars"]) for item in group}) == 1
        assert len({id(item["prepared_labels"]) for item in group}) == 1
        assert len({id(item["prepared_feature_profiles"]) for item in group}) == 1


def _trained_experiment(
    profile: str,
    *,
    macro_f1: float,
    raw_precision: float,
    forward_precision: float,
    forward_recall: float,
) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "scenario": "V4F_Flat_12_Candle",
        "status": "TRAINED",
        "classification_metrics": {"validation": {"macro": {"f1": macro_f1}}},
        "hold_metrics": {"precision": raw_precision},
        "threshold_optimization": {
            "post_calibration_validation": {
                "hold_metrics": {
                    "precision": forward_precision,
                    "recall": forward_recall,
                }
            }
        },
    }


def test_feature_ablation_summary_uses_forward_hold_metrics_and_baseline_deltas() -> (
    None
):
    experiments = [
        _trained_experiment(
            "Baseline",
            macro_f1=0.32,
            raw_precision=0.36,
            forward_precision=0.38,
            forward_recall=0.22,
        ),
        _trained_experiment(
            "Baseline+Trend",
            macro_f1=0.35,
            raw_precision=0.42,
            forward_precision=0.43,
            forward_recall=0.21,
        ),
        _trained_experiment(
            "Baseline+Volatility",
            macro_f1=0.33,
            raw_precision=0.39,
            forward_precision=0.40,
            forward_recall=0.20,
        ),
        _trained_experiment(
            "Baseline+Volume",
            macro_f1=0.31,
            raw_precision=0.40,
            forward_precision=0.39,
            forward_recall=0.24,
        ),
        _trained_experiment(
            "Combined",
            macro_f1=0.37,
            raw_precision=0.46,
            forward_precision=0.45,
            forward_recall=0.23,
        ),
    ]

    result = _build_feature_ablation_summary(experiments)

    assert result["highest_mean_forward_hold_precision_profile"] == "Combined"
    assert result["profiles"]["Baseline+Trend"]["forward_acceptance_status"] == "PASS"
    assert result["profiles"]["Combined"]["forward_acceptance_status"] == "PASS"
    assert (
        result["profiles"]["Combined"]["mean_validation_macro_f1_delta_vs_baseline"]
        == 0.05
    )
    assert (
        result["profiles"]["Combined"]["mean_forward_hold_precision_delta_vs_baseline"]
        == 0.07
    )
    assert result["combined_forward_acceptance"] == "PASS"


def test_feature_ablation_summary_fails_closed_without_forward_metrics() -> None:
    experiment = _trained_experiment(
        "Combined",
        macro_f1=0.37,
        raw_precision=0.46,
        forward_precision=0.45,
        forward_recall=0.23,
    )
    experiment["threshold_optimization"]["post_calibration_validation"] = None

    result = _build_feature_ablation_summary([experiment])

    assert result["profiles"]["Combined"]["forward_metric_experiment_count"] == 0
    assert (
        result["profiles"]["Combined"]["forward_acceptance_status"]
        == "NOT_MET_OR_INCOMPLETE"
    )
    assert result["combined_forward_acceptance"] == "NOT_MET_OR_INCOMPLETE"
