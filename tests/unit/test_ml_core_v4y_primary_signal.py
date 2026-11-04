"""V4-Y Primary Signal, sample gate, and OOF contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4y_primary_signal as v4y
from services.meta_labeling.v4y import (
    V4Y_BOLLINGER_PERIOD,
    V4Y_BOLLINGER_STDDEV,
    V4Y_EMA_TREND_PERIOD,
    V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
    V4Y_META_FEATURE_COLUMNS,
    V4Y_PRIMARY_SIGNAL_GENERATION,
    build_v4y_primary_features,
    evaluate_primary_train_sample_gate,
    generate_v4y_oof_primary_events,
)


def _bars(close: np.ndarray) -> DataFrame:
    values = np.asarray(close, dtype=float)
    return DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01T00:00:00Z", periods=len(values), freq="h"
            ),
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": np.full(len(values), 1000.0),
        }
    )


def test_bollinger_breakout_emits_one_causal_buy_above_ema200() -> None:
    close = np.concatenate((np.full(220, 100.0), np.full(40, 130.0)))
    bars = _bars(close)
    base_features = DataFrame({"base_feature": np.arange(len(close), dtype=float)})

    features, signal = build_v4y_primary_features(bars, base_features)

    event_indices = np.flatnonzero(signal.to_numpy(dtype=bool)).tolist()
    assert event_indices == [220]
    event_index = event_indices[0]
    assert event_index >= V4Y_EMA_TREND_PERIOD
    assert cast(float, features.loc[event_index, "%-primary_close_vs_ema200_pct"]) > 0
    breakout_distance = cast(
        float, features.loc[event_index, "%-primary_close_vs_bollinger_upper_pct"]
    )
    assert breakout_distance > 0
    assert set(V4Y_META_FEATURE_COLUMNS).issubset(features.columns)
    assert signal.dtype == bool


def test_primary_signal_prefix_is_invariant_to_future_price_changes() -> None:
    close = np.concatenate((np.full(220, 100.0), np.full(40, 130.0)))
    bars = _bars(close)
    features = DataFrame({"base_feature": np.arange(len(close), dtype=float)})
    _, original = build_v4y_primary_features(bars, features)

    changed = bars.copy()
    changed.loc[221:, ["open", "high", "low", "close"]] *= 0.5
    _, revised = build_v4y_primary_features(changed, features)

    assert original.iloc[:221].equals(revised.iloc[:221])


def test_bollinger_breakout_is_filtered_when_price_is_below_ema200() -> None:
    close = np.concatenate(
        (np.full(200, 100.0), np.full(20, 80.0), np.full(100, 90.0))
    )
    bars = _bars(close)
    features, signal = build_v4y_primary_features(
        bars, DataFrame({"base_feature": np.arange(len(close), dtype=float)})
    )

    upper_distance = cast(
        float, features.loc[220, "%-primary_close_vs_bollinger_upper_pct"]
    )
    trend_distance = cast(
        float, features.loc[220, "%-primary_close_vs_ema200_pct"]
    )
    assert upper_distance > 0
    assert trend_distance < 0
    assert not bool(signal.iloc[220])
    assert not bool(signal.any())


def test_bollinger_primary_rejects_invalid_periods_and_feature_schema() -> None:
    bars = _bars(np.linspace(100.0, 120.0, 260))
    base = DataFrame({"x": np.arange(260, dtype=float)})
    with pytest.raises(ValueError, match="period_configuration_invalid"):
        build_v4y_primary_features(bars, base, bollinger_period=30, ema_period=20)
    with pytest.raises(ValueError, match="deviation_configuration_invalid"):
        build_v4y_primary_features(bars, base, standard_deviations=0.0)
    with pytest.raises(ValueError, match="target_column_in_primary_features"):
        build_v4y_primary_features(
            bars, DataFrame({"&-future_return": np.zeros(260)})
        )
    with pytest.raises(ValueError, match="input_alignment_invalid"):
        build_v4y_primary_features(bars, DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="feature_columns_must_be_unique"):
        build_v4y_primary_features(
            bars, DataFrame(np.ones((260, 2)), columns=["x", "x"])
        )
    with pytest.raises(ValueError, match="primary_feature_column_collision"):
        build_v4y_primary_features(
            bars, DataFrame({V4Y_META_FEATURE_COLUMNS[0]: np.ones(260)})
        )
    with pytest.raises(ValueError, match="close_column_missing"):
        build_v4y_primary_features(
            bars.drop(columns=["close"]), base
        )


def test_bollinger_primary_rejects_boolean_parameters_and_empty_bars() -> None:
    with pytest.raises(ValueError, match="period_configuration_invalid"):
        build_v4y_primary_features(
            _bars(np.full(220, 100.0)),
            DataFrame({"x": np.ones(220)}),
            bollinger_period=True,
        )
    with pytest.raises(ValueError, match="deviation_configuration_invalid"):
        build_v4y_primary_features(
            _bars(np.full(220, 100.0)),
            DataFrame({"x": np.ones(220)}),
            standard_deviations=True,
        )
    with pytest.raises(ValueError, match="input_alignment_invalid"):
        build_v4y_primary_features(
            _bars(np.array([], dtype=float)), DataFrame({"x": np.array([])})
        )


def test_bollinger_primary_rejects_non_positive_or_non_finite_close() -> None:
    invalid = np.full(260, 100.0)
    invalid[220] = np.nan
    with pytest.raises(ValueError, match="close_values_invalid"):
        build_v4y_primary_features(
            _bars(invalid), DataFrame({"x": np.arange(260, dtype=float)})
        )
    invalid[220] = 0.0
    with pytest.raises(ValueError, match="close_values_invalid"):
        build_v4y_primary_features(
            _bars(invalid), DataFrame({"x": np.arange(260, dtype=float)})
        )


def test_primary_train_sample_gate_enforces_exact_50_signal_boundary() -> None:
    signals = Series(False, index=range(100), dtype=bool)
    signals.iloc[:49] = True
    blocked = evaluate_primary_train_sample_gate(
        signals, train_start_index=0, train_end_index=100
    )
    assert blocked["status"] == "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"
    assert blocked["passed"] is False
    assert blocked["observed_primary_buy_count"] == 49
    assert blocked["minimum_primary_buy_count"] == 50

    signals.iloc[49] = True
    accepted = evaluate_primary_train_sample_gate(
        signals, train_start_index=0, train_end_index=100
    )
    assert accepted["status"] == "PASS"
    assert accepted["passed"] is True
    assert accepted["observed_primary_buy_count"] == V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS


def test_candidate_sample_gate_skips_before_meta_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _bars(np.full(300, 100.0))
    features = DataFrame(
        {name: np.ones(300, dtype=float) for name in V4Y_META_FEATURE_COLUMNS}
    )
    signals = Series(False, index=range(300), dtype=bool)
    signals.iloc[200:249] = True

    monkeypatch.setattr(v4y, "resample_ohlcv", lambda source, timeframe: bars)
    monkeypatch.setattr(
        v4y, "_outer_boundaries", lambda rows, horizon: (250, 280)
    )
    monkeypatch.setattr(
        v4y,
        "build_feature_profiles",
        lambda frame, profile_names: {
            "Combined": DataFrame({"base": np.ones(300)})
        },
    )
    monkeypatch.setattr(
        v4y,
        "build_v4y_primary_features",
        lambda frame, base: (features, signals),
    )

    def unexpected_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("meta_training_must_not_run_before_sample_gate")

    monkeypatch.setattr(v4y, "_training_oof_events", unexpected_training)
    candidate = v4y._run_candidate(
        symbol="BTCUSDT", timeframe="15m", source_bars=bars
    )

    assert candidate["status"] == "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS"
    observed_train_signals = candidate["primary_signal_counts"][
        "train_primary_buy_event_count"
    ]
    assert observed_train_signals == 49
    assert candidate["primary_sample_gate"]["passed"] is False
    assert candidate.get("meta_training") is None


def test_primary_train_sample_gate_rejects_bad_signal_types_and_windows() -> None:
    with pytest.raises(ValueError, match="requires_boolean_signals"):
        evaluate_primary_train_sample_gate(
            Series([0, 1, 0]), train_start_index=0, train_end_index=3
        )
    with pytest.raises(ValueError, match="must_not_be_null"):
        evaluate_primary_train_sample_gate(
            Series([True, pd.NA], dtype="boolean"),
            train_start_index=0,
            train_end_index=2,
        )
    with pytest.raises(ValueError, match="window_invalid"):
        evaluate_primary_train_sample_gate(
            Series([True, False]), train_start_index=1, train_end_index=1
        )
    with pytest.raises(ValueError, match="minimum_invalid"):
        evaluate_primary_train_sample_gate(
            Series([True, False]),
            train_start_index=0,
            train_end_index=2,
            minimum_signal_count=0,
        )


def test_v4y_oof_events_reuse_chronological_folds_with_correct_provenance() -> None:
    signal = Series(False, index=range(100), dtype=bool)
    signal.iloc[[15, 20, 45, 55, 80]] = True

    result = generate_v4y_oof_primary_events(
        signal, start_index=10, stop_index=90, fold_count=4
    )

    assert result["event_index"].tolist() == [15, 20, 45, 55, 80]
    assert result["fold_id"].tolist() == [0, 0, 1, 2, 3]
    assert result["primary_signal"].eq("BUY").all()
    assert result["primary_oof"].all()
    assert result["primary_signal_generation"].eq(
        V4Y_PRIMARY_SIGNAL_GENERATION
    ).all()
    assert (
        result["primary_history_end_index"].to_numpy(dtype=int)
        <= result["event_index"].to_numpy(dtype=int)
    ).all()


def test_v4y_oof_events_reject_empty_and_invalid_windows() -> None:
    signal = Series(False, index=range(20), dtype=bool)
    with pytest.raises(ValueError, match="oof_window_invalid"):
        generate_v4y_oof_primary_events(
            signal, start_index=10, stop_index=10
        )
    with pytest.raises(ValueError, match="oof_fold_configuration_invalid"):
        generate_v4y_oof_primary_events(
            signal, start_index=0, stop_index=3, fold_count=4
        )
    with pytest.raises(ValueError, match="oof_primary_signals_empty"):
        generate_v4y_oof_primary_events(
            signal, start_index=0, stop_index=20
        )


def test_timeframe_horizons_preserve_24_hour_label_window() -> None:
    assert v4y._horizon_candles("15m") == 96
    assert v4y._horizon_candles("30m") == 48
    assert v4y._horizon_candles("1h") == 24
    assert all(
        candle_count * minutes == 24 * 60
        for candle_count, minutes in ((96, 15), (48, 30), (24, 60))
    )
    with pytest.raises(ValueError, match="timeframe_out_of_policy"):
        v4y._horizon_candles("4h")


def test_v4y_step_summary_reports_signal_counts_thresholds_and_economics() -> None:
    report = {
        "commit_sha": "a" * 40,
        "candidates": [
            {
                "candidate_id": "BTC/USDT|15m|Combined",
                "status": "TRAINED",
                "primary_signal_counts": {
                    "total_primary_buy_event_count": 145,
                    "train_primary_buy_event_count": 102,
                    "matured_oof_train_buy_event_count": 96,
                },
                "primary_sample_gate": {"passed": True},
                "training_diagnostics": {"fit_rows": 78, "calibration_rows": 15},
                "validation_signal_counts": {
                    "primary_buy_event_count": 24,
                    "meta_accepted_buy_count": 12,
                    "meta_rejected_buy_count": 12,
                },
                "selected_meta_probability_threshold": 0.61,
                "metrics": {
                    "trade_count": 12,
                    "win_rate": 0.5,
                    "net_profit_factor": 1.4,
                    "net_profit_factor_infinite_no_losses": False,
                    "net_pnl_usd": 125.0,
                    "max_drawdown_percent": 4.5,
                    "economic_candidate_gate_passed": False,
                },
            },
            {
                "candidate_id": "ETH/USDT|1h|Combined",
                "status": "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS",
                "skip_reason": "v4y_insufficient_primary_signals",
                "primary_signal_counts": {
                    "total_primary_buy_event_count": 72,
                    "train_primary_buy_event_count": 43,
                },
                "primary_sample_gate": {"passed": False},
                "training_diagnostics": {},
                "validation_signal_counts": {"primary_buy_event_count": 0},
                "metrics": {"trade_count": 0, "win_rate": None},
            },
        ],
    }

    summary = v4y._render_summary(report)

    assert "BTC/USDT|15m|Combined" in summary
    assert "0.610000" in summary
    assert "1.4000" in summary
    assert "125.00" in summary
    assert "4.50%" in summary
    assert "145" in summary and "102" in summary
    assert "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS" in summary
    assert "Primary signal target (>=100 across 730 days): 1/2" in summary
    assert "Train sample gate (>=50 Primary BUY): 1/2" in summary
    assert "trading_authorization=NO-GO" in summary
    assert "final holdout is untouched" in summary


def test_v4y_workflow_is_branch_scoped_and_keeps_ci_and_safety_gates() -> None:
    workflow = Path(
        ".github/workflows/ml-core-v4y-primary-signal.yml"
    ).read_text(encoding="utf-8")

    assert "research/ml-core-v4y-primary-signal" in workflow
    assert "Require standard CI success for this exact V4-Y commit" in workflow
    assert "TARGET_SHA" in workflow
    assert "actions: read" in workflow
    assert "Fetch complete public Binance Spot 730-day 1-minute datasets" in workflow
    assert (
        "Render V4-Y candidate metrics and thresholds to GitHub Step Summary"
        in workflow
    )
    assert "_render_summary" in workflow
    assert "alix-v4y-primary-signal-report.json" in workflow
    assert "Champion=null" in workflow
    assert "trading_authorization=NO-GO" in workflow
    assert "private_api_calls=0" in workflow
    assert "real_order_attempts=0" in workflow
    assert "LSTM" not in workflow


def test_v4y_report_keeps_non_promoting_safety_contract() -> None:
    report = v4y._build_v4y_report(
        "b" * 40,
        730,
        [],
        [],
    )

    assert report["primary_model"]["fit_performed"] is False
    minimum_signals = report["meta_model"]["sample_gate"][
        "minimum_primary_train_buy_signals"
    ]
    assert minimum_signals == 50
    assert report["label_contract"]["horizon_candles_by_timeframe"] == {
        "15m": 96,
        "30m": 48,
        "1h": 24,
    }
    assert report["safety"]["Champion"] is None
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert report["safety"]["private_api_calls"] == 0
    assert report["safety"]["real_order_attempts"] == 0
    assert report["safety"]["sequence_model_used"] is False


def test_v4y_primary_indicator_constants_are_explicit() -> None:
    assert V4Y_BOLLINGER_PERIOD == 20
    assert V4Y_BOLLINGER_STDDEV == 2.0
    assert V4Y_EMA_TREND_PERIOD == 200
