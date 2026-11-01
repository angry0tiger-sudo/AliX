from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

import scripts.run_ml_core_v4h_tuning_geometry as v4h
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier


def _market_frame(rows: int = 40) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    close = np.full(rows, 100.0)
    return DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )


def test_v4h_geometry_contract_preserves_three_approved_twelve_candle_scenarios() -> (
    None
):
    assert v4h._v4h_scenarios() == [
        {
            "name": "V4H_Current",
            "horizon_candles": 12,
            "take_profit_multiplier": 2.0,
            "stop_loss_multiplier": 1.0,
        },
        {
            "name": "V4H_Symmetric",
            "horizon_candles": 12,
            "take_profit_multiplier": 1.5,
            "stop_loss_multiplier": 1.5,
        },
        {
            "name": "V4H_Tighter_SL",
            "horizon_candles": 12,
            "take_profit_multiplier": 2.0,
            "stop_loss_multiplier": 1.5,
        },
    ]


@pytest.mark.parametrize(
    ("scenario", "expected_tp", "expected_sl"),
    [
        ("V4H_Current", 102.0, 99.0),
        ("V4H_Symmetric", 101.5, 98.5),
        ("V4H_Tighter_SL", 102.0, 98.5),
    ],
)
def test_geometry_changes_only_barrier_distances_and_preserves_label_contract(
    scenario: str, expected_tp: float, expected_sl: float
) -> None:
    candles = _market_frame()
    scenario_config = next(
        item for item in v4h._v4h_scenarios() if item["name"] == scenario
    )
    atr = pd.Series(np.ones(len(candles), dtype=float))

    labels = apply_adaptive_triple_barrier(
        candles,
        atr_values=atr,
        horizon_candles=scenario_config["horizon_candles"],
        take_profit_multiplier=scenario_config["take_profit_multiplier"],
        stop_loss_multiplier=scenario_config["stop_loss_multiplier"],
    )

    event = labels.iloc[14]
    assert event["take_profit_price"] == pytest.approx(expected_tp)
    assert event["stop_loss_price"] == pytest.approx(expected_sl)
    assert event["label"] == "HOLD"
    assert event["label_end_idx"] == 26
    assert set(labels["label"].dropna().unique()) <= {"BUY", "SELL", "HOLD"}


def test_v4h_orchestration_creates_exactly_forty_eight_experiments(
    monkeypatch: Any,
) -> None:
    bars = DataFrame({"date": range(1200), "close": range(1200)})
    labels = DataFrame(
        {
            "label": ["BUY", "SELL", "HOLD"] * 400,
            "label_end_idx": [min(index + 12, 1199) for index in range(1200)],
        }
    )
    features = {
        "Baseline": DataFrame({"base": range(1200)}),
        "Baseline+Trend": DataFrame({"base": range(1200), "trend": range(1200)}),
    }
    profile_calls: list[tuple[str, ...]] = []
    label_calls: list[dict[str, Any]] = []
    experiment_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(v4h, "resample_ohlcv", lambda candles, timeframe: bars)

    def fake_profiles(
        prepared: DataFrame, *, profile_names: tuple[str, ...] | None = None
    ) -> dict[str, DataFrame]:
        assert prepared is bars
        assert profile_names == v4h.V4H_FEATURE_PROFILES
        profile_calls.append(profile_names)
        return features

    def fake_labels(prepared: DataFrame, **kwargs: Any) -> DataFrame:
        assert prepared is bars
        label_calls.append(kwargs)
        return labels

    def fake_experiment(
        pair: str,
        timeframe: str,
        prepared_bars: DataFrame,
        prepared_labels: DataFrame,
        feature_matrices: dict[str, DataFrame],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert prepared_bars is bars
        assert prepared_labels is labels
        assert feature_matrices is features
        experiment_calls.append({"pair": pair, "timeframe": timeframe, **kwargs})
        return {
            "pair": pair,
            "timeframe": timeframe,
            "scenario": kwargs["scenario"]["name"],
            "feature_profile": kwargs["feature_profile"],
            "status": "TRAINED",
        }

    monkeypatch.setattr(v4h, "build_feature_profiles", fake_profiles)
    monkeypatch.setattr(v4h, "apply_adaptive_triple_barrier", fake_labels)
    monkeypatch.setattr(v4h, "run_v4h_experiment", fake_experiment)

    datasets = {
        "BTCUSDT": DataFrame({"minute": [1]}),
        "ETHUSDT": DataFrame({"minute": [1]}),
    }
    result = v4h.run_v4h_experiments(datasets)

    assert len(result) == 48
    assert len(experiment_calls) == 48
    assert len(profile_calls) == 8
    assert len(label_calls) == 24
    assert {
        (item["pair"], item["timeframe"], item["scenario"], item["feature_profile"])
        for item in result
    } == {
        (pair, timeframe, scenario, profile)
        for pair in ("BTC/USDT", "ETH/USDT")
        for timeframe in ("15m", "30m", "1h", "4h")
        for scenario in ("V4H_Current", "V4H_Symmetric", "V4H_Tighter_SL")
        for profile in v4h.V4H_FEATURE_PROFILES
    }


def test_v4h_report_aggregates_candidate_metrics_and_target_balance() -> None:
    experiments: list[dict[str, Any]] = []
    candidate_parameters = [
        {"num_leaves": leaves, "learning_rate": rate, "min_child_samples": child}
        for leaves in (15, 31, 63)
        for rate in (0.01, 0.05, 0.1)
        for child in (20, 50, 100)
    ][:9]
    for scenario in v4h._v4h_scenarios():
        for profile in v4h.V4H_FEATURE_PROFILES:
            for context in range(8):
                variants: dict[str, Any] = {}
                for policy in ("weight_none", "weight_balanced"):
                    variants[policy] = {
                        "classification_metrics": {
                            "validation": {"macro": {"f1": 0.50 + context / 100}}
                        },
                        "threshold_optimization": {
                            "post_calibration_validation": {
                                "hold_metrics": {"precision": 0.42, "recall": 0.25}
                            }
                        },
                        "hyperparameter_tuning": {
                            "selected_parameters": candidate_parameters[0],
                            "candidate_fit_count": 27,
                            "candidate_results": [
                                {
                                    "parameters": parameters,
                                    "mean_cv_hold_precision": 0.30 + index / 100,
                                    "mean_cv_hold_recall": 0.20 + index / 100,
                                    "mean_cv_macro_f1": 0.40 + index / 100,
                                }
                                for index, parameters in enumerate(candidate_parameters)
                            ],
                        },
                    }
                experiments.append(
                    {
                        "scenario": scenario["name"],
                        "pair": "BTC/USDT" if context < 4 else "ETH/USDT",
                        "timeframe": ("15m", "30m", "1h", "4h")[context % 4],
                        "feature_profile": profile,
                        "status": "TRAINED",
                        "label_distribution": {
                            "validation": {
                                "classes": {
                                    "BUY": {"count": 10},
                                    "SELL": {"count": 20},
                                    "HOLD": {"count": 30},
                                }
                            }
                        },
                        "model_variants": variants,
                    }
                )

    report = v4h.build_v4h_report(
        commit_sha="a" * 40,
        days=730,
        provenance=[],
        experiments=experiments,
        random_search_iterations=9,
        calibration_fraction=0.20,
    )

    assert report["experiment_count"] == 48
    assert report["trained_models"] == 96
    assert report["economic_evaluation_performed"] is False
    assert report["acceptance_policy"]["status"] == "MET"
    assert report["acceptance_policy"]["strict_acceptance_count"] == 96
    assert report["hyperparameter_tuning"]["candidate_fit_count"] == 96 * 27
    candidates = report["hyperparameter_tuning"]["candidate_summary"]
    assert candidates["candidate_combinations_evaluated"] == 9
    assert (
        candidates["best_by_mean_inner_cv_hold_precision"]["parameters"]
        == candidate_parameters[-1]
    )
    assert len(report["geometry_summary"]) == 3
    for geometry in report["geometry_summary"]:
        assert geometry["experiment_count"] == 16
        assert geometry["trained_model_count"] == 32
        assert geometry["label_distribution"]["total"] == 8 * 60
        assert geometry["strict_acceptance_count"] == 32
    assert report["safety"] == {
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
