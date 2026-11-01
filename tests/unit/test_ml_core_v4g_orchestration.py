from __future__ import annotations

from typing import Any

from pandas import DataFrame

from scripts.run_ml_core_real_data_baseline import (
    V4G_FEATURE_PROFILE_ORDER,
    _run_diagnostic_experiments,
)
from services.ml_core.walk_forward_validation import WalkForwardFold


def test_v4g_orchestration_runs_only_two_profiles_over_five_folds(
    monkeypatch: Any,
) -> None:
    bars = DataFrame({"date": range(1200), "close": range(1200)})
    labels = DataFrame(
        {
            "label": ["BUY", "SELL", "HOLD"] * 400,
            "label_end_idx": [min(index + 12, 1199) for index in range(1200)],
        }
    )
    matrices = {
        "Baseline": DataFrame({"baseline": range(1200)}),
        "Baseline+Trend": DataFrame({"baseline": range(1200), "trend": range(1200)}),
    }
    dummy_folds = tuple(
        WalkForwardFold(
            fold_index=index,
            train_indices=(0, 1, 2),
            calibration_indices=(20, 21, 22),
            test_indices=(40, 41, 42),
            diagnostics={"fold_index": index},
        )
        for index in (1, 2, 3, 4, 5)
    )
    calls: list[dict[str, Any]] = []
    fold_builder_calls: list[dict[str, Any]] = []
    profile_builder_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.resample_ohlcv",
        lambda candles, timeframe: bars,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.apply_adaptive_triple_barrier",
        lambda prepared, **kwargs: labels,
    )

    def fake_profiles(
        prepared: DataFrame, *, profile_names: tuple[str, ...] | None = None
    ) -> dict[str, DataFrame]:
        assert prepared is bars
        assert profile_names is not None
        profile_builder_calls.append(profile_names)
        return matrices

    def fake_folds(
        features: DataFrame,
        prepared_labels: DataFrame,
        *,
        fold_count: int,
        label_horizon_candles: int,
        calibration_fraction: float,
    ) -> tuple[WalkForwardFold, ...]:
        assert features is matrices["Baseline+Trend"]
        assert prepared_labels is labels
        fold_builder_calls.append(
            {
                "fold_count": fold_count,
                "label_horizon_candles": label_horizon_candles,
                "calibration_fraction": calibration_fraction,
            }
        )
        return dummy_folds

    def fake_run(
        pair: str,
        timeframe: str,
        minute_candles: DataFrame,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                **kwargs,
            }
        )
        return {
            "pair": pair,
            "timeframe": timeframe,
            "feature_profile": kwargs["feature_profile"],
            "fold_index": kwargs["fold"].fold_index,
        }

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.build_feature_profiles",
        fake_profiles,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.create_walk_forward_folds",
        fake_folds,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.run_walk_forward_experiment",
        fake_run,
    )

    datasets = {
        "BTCUSDT": DataFrame({"minute": [1]}),
        "ETHUSDT": DataFrame({"minute": [1]}),
    }
    result = _run_diagnostic_experiments(
        datasets,
        barrier_scenarios=[
            {
                "name": "V4G_Flat_12_Candle",
                "horizon_candles": 12,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
            }
        ],
        calibration_fraction=0.20,
        minimum_hold_precision=0.40,
        minimum_hold_recall=0.20,
        feature_profile_names=V4G_FEATURE_PROFILE_ORDER,
        walk_forward_folds=5,
    )

    assert len(result) == 80
    assert len(calls) == 80
    assert len(fold_builder_calls) == 8
    assert len(profile_builder_calls) == 8
    assert all(names == V4G_FEATURE_PROFILE_ORDER for names in profile_builder_calls)
    assert all(
        parameters
        == {
            "fold_count": 5,
            "label_horizon_candles": 12,
            "calibration_fraction": 0.20,
        }
        for parameters in fold_builder_calls
    )
    assert {(item["pair"], item["timeframe"]) for item in result} == {
        (pair, timeframe)
        for pair in ("BTC/USDT", "ETH/USDT")
        for timeframe in ("15m", "30m", "1h", "4h")
    }
    assert {(item["fold_index"], item["feature_profile"]) for item in result} == {
        (fold, profile)
        for fold in (1, 2, 3, 4, 5)
        for profile in V4G_FEATURE_PROFILE_ORDER
    }
