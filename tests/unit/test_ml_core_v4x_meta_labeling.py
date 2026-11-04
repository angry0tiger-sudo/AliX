"""V4-X causal SMA primary / OOF-only LightGBM Meta-Labeling tests."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4x_meta_labeling as v4x
from services.meta_labeling.v4x import (
    V4X_META_FEATURE_COLUMNS,
    build_costed_meta_labels,
    build_sma_primary_features,
    fit_lightgbm_meta_filter,
    generate_oof_primary_events,
    predict_meta_probabilities,
    split_meta_fit_calibration,
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


def _labelled_oof_events(rows: int = 100) -> DataFrame:
    event_indices = np.arange(50, 50 + rows, dtype=int)
    return DataFrame(
        {
            "event_index": event_indices,
            "fold_id": np.minimum(event_indices // 20, 4),
            "label_end_idx": event_indices + 1,
            "meta_label": np.arange(rows, dtype=int) % 2,
            "primary_signal": ["BUY"] * rows,
            "primary_oof": [True] * rows,
            "signal_feature": np.linspace(-2.0, 2.0, rows),
        }
    )


class FakeLightGBMClassifier:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.fitted_labels_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> FakeLightGBMClassifier:
        assert features.ndim == 2
        self.fitted_labels_ = np.asarray(labels, dtype=int).copy()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        signal = np.asarray(features[:, 0], dtype=float)
        positive = 1.0 / (1.0 + np.exp(-signal))
        return np.column_stack((1.0 - positive, positive))


def test_sma_primary_emits_only_bullish_crossovers_using_causal_features() -> None:
    close = np.concatenate(
        (np.full(60, 100.0), np.linspace(100.0, 120.0, 40), np.full(30, 120.0))
    )
    bars = _bars(close)
    base_features = DataFrame({"base_feature": np.arange(len(close), dtype=float)})

    features, signal = build_sma_primary_features(bars, base_features)

    event_indices = np.flatnonzero(signal.to_numpy(dtype=bool))
    assert len(event_indices) == 1
    event_index = int(event_indices[0])
    assert event_index >= 50
    assert cast(float, features.loc[event_index, "%-primary_sma_spread_pct"]) > 0
    assert set(V4X_META_FEATURE_COLUMNS).issubset(features.columns)
    assert cast(float, features.loc[event_index, "%-primary_close_vs_sma_slow_pct"]) > 0
    assert signal.dtype == bool


def test_sma_primary_rejects_invalid_periods_and_feature_schema() -> None:
    bars = _bars(np.linspace(100.0, 120.0, 80))
    with pytest.raises(ValueError, match="sma_period_configuration_invalid"):
        build_sma_primary_features(
            bars, DataFrame({"x": range(80)}), fast_period=50, slow_period=20
        )
    with pytest.raises(ValueError, match="target_column_in_primary_features"):
        build_sma_primary_features(
            bars,
            DataFrame({"&-future_return": np.zeros(80)}),
        )
    with pytest.raises(ValueError, match="input_alignment_invalid"):
        build_sma_primary_features(bars, DataFrame({"x": [1.0]}))


def test_oof_primary_events_are_disjoint_chronological_test_folds() -> None:
    signal = Series(False, index=range(100), dtype=bool)
    signal.iloc[[15, 20, 45, 55, 80]] = True

    result = generate_oof_primary_events(
        signal, start_index=10, stop_index=90, fold_count=4
    )

    assert result["event_index"].tolist() == [15, 20, 45, 55, 80]
    assert result["fold_id"].tolist() == [0, 0, 1, 2, 3]
    assert result["primary_signal"].eq("BUY").all()
    assert result["primary_oof"].all()
    assert (
        result["primary_history_end_index"].to_numpy(dtype=int)
        <= result["event_index"].to_numpy(dtype=int)
    ).all()
    for row in result.to_dict(orient="records"):
        assert (
            int(row["fold_start_index"])
            <= int(row["event_index"])
            < int(row["fold_stop_index"])
        )


def test_oof_generation_rejects_invalid_or_empty_windows() -> None:
    signal = Series(False, index=range(20), dtype=bool)
    with pytest.raises(ValueError, match="oof_window_invalid"):
        generate_oof_primary_events(signal, start_index=10, stop_index=10)
    with pytest.raises(ValueError, match="oof_fold_configuration_invalid"):
        generate_oof_primary_events(signal, start_index=0, stop_index=3, fold_count=4)
    with pytest.raises(ValueError, match="oof_primary_signals_empty"):
        generate_oof_primary_events(signal, start_index=0, stop_index=20)


def test_costed_meta_labels_use_net_result_and_only_primary_buy_oof_rows() -> None:
    bars = _bars(np.full(8, 100.0))
    events = DataFrame(
        {
            "event_index": [1, 2],
            "fold_id": [0, 0],
            "primary_signal": ["BUY", "BUY"],
            "primary_oof": [True, True],
        }
    )
    barriers = DataFrame(
        {
            "event_index": [1, 2],
            "label_end_idx": [3, 4],
            "entry_price": [100.2, 100.2],
            "take_profit_price": [103.0, 103.0],
            "stop_loss_price": [99.0, 99.0],
            "exit_reason": ["TP_HIT", "SL_HIT"],
        }
    )

    result = build_costed_meta_labels(
        bars,
        events,
        barriers,
        fee_bps_per_side=25.0,
        slippage_bps_per_side=20.0,
        entry_slippage_bps=20.0,
    )

    assert result["meta_label"].tolist() == [1, 0]
    assert cast(float, result.loc[0, "net_return"]) > 0
    assert cast(float, result.loc[1, "net_return"]) < 0
    assert result["primary_signal"].eq("BUY").all()
    assert result["primary_oof"].all()


def test_costed_meta_labels_reject_non_buy_or_non_oof_training_signals() -> None:
    bars = _bars(np.full(8, 100.0))
    events = DataFrame(
        {
            "event_index": [1],
            "fold_id": [0],
            "primary_signal": ["SELL"],
            "primary_oof": [True],
        }
    )
    barriers = DataFrame(
        {
            "event_index": [1],
            "label_end_idx": [3],
            "entry_price": [100.2],
            "take_profit_price": [103.0],
            "stop_loss_price": [99.0],
            "exit_reason": ["TP_HIT"],
        }
    )
    with pytest.raises(ValueError, match="primary_buy_only"):
        build_costed_meta_labels(
            bars,
            events,
            barriers,
            fee_bps_per_side=25.0,
            slippage_bps_per_side=20.0,
            entry_slippage_bps=20.0,
        )
    events["primary_signal"] = "BUY"
    events["primary_oof"] = False
    with pytest.raises(ValueError, match="oof_primary_signals"):
        build_costed_meta_labels(
            bars,
            events,
            barriers,
            fee_bps_per_side=25.0,
            slippage_bps_per_side=20.0,
            entry_slippage_bps=20.0,
        )


def test_meta_fit_calibration_split_purges_labels() -> None:
    events = _labelled_oof_events()
    fit, calibration = split_meta_fit_calibration(events, outer_train_end_index=300)

    calibration_start = int(calibration["event_index"].min())
    assert len(fit) >= 40
    assert fit["meta_label"].nunique() == 2
    assert int(fit["label_end_idx"].max()) < calibration_start
    assert calibration["event_index"].min() == calibration_start
    assert int(events["event_index"].max()) < 300
    assert fit["primary_signal"].eq("BUY").all()
    assert fit["primary_oof"].all()


def test_meta_split_rejects_leaks_and_bad_oof_provenance() -> None:
    events = _labelled_oof_events()
    events.loc[len(events) - 1, "label_end_idx"] = 300
    with pytest.raises(ValueError, match="outer_validation_contamination"):
        split_meta_fit_calibration(events, outer_train_end_index=300)

    events = _labelled_oof_events()
    events.loc[0, "primary_oof"] = False
    with pytest.raises(ValueError, match="requires_oof_primary_signals"):
        split_meta_fit_calibration(events, outer_train_end_index=300)


def test_lightgbm_meta_fit_uses_fit_labels_and_label_blind_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("lightgbm")
    module.LGBMClassifier = FakeLightGBMClassifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightgbm", module)

    fit_features = DataFrame(
        {
            "signal_feature": np.linspace(-2.0, 2.0, 80),
            "secondary_feature": np.sin(np.linspace(0.0, 5.0, 80)),
        }
    )
    fit_labels = Series((fit_features["signal_feature"] > 0).astype(int))
    calibration_features = DataFrame(
        {
            "signal_feature": np.linspace(-1.0, 1.0, 20),
            "secondary_feature": np.cos(np.linspace(0.0, 2.0, 20)),
        }
    )

    model, diagnostics = fit_lightgbm_meta_filter(
        fit_features,
        fit_labels,
        calibration_features,
        target_accept_fraction=0.5,
    )
    probabilities = predict_meta_probabilities(model, calibration_features)

    assert isinstance(model, FakeLightGBMClassifier)
    assert len(probabilities) == len(calibration_features)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert model.fitted_labels_ is not None
    np.testing.assert_array_equal(model.fitted_labels_, fit_labels.to_numpy())
    assert diagnostics["calibration_labels_used"] is False
    assert diagnostics["selected_using_validation_labels"] is False
    assert 0.0 <= diagnostics["selected_threshold"] <= 1.0
    assert model.kwargs["objective"] == "binary"
    assert model.kwargs["deterministic"] is True


def test_lightgbm_meta_fit_rejects_bad_labels_and_feature_misalignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("lightgbm")
    module.LGBMClassifier = FakeLightGBMClassifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    features = DataFrame({"x": np.linspace(-1.0, 1.0, 50)})
    calibration = DataFrame({"x": np.linspace(-0.5, 0.5, 10)})
    with pytest.raises(ValueError, match="both_binary_classes"):
        fit_lightgbm_meta_filter(
            features,
            Series(np.ones(50, dtype=int)),
            calibration,
        )
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        fit_lightgbm_meta_filter(
            features,
            Series(np.arange(50, dtype=int) % 2),
            DataFrame({"different": np.linspace(-0.5, 0.5, 10)}),
        )
    with pytest.raises(ValueError, match="meta_features_non_finite"):
        predict_meta_probabilities(
            FakeLightGBMClassifier(),
            DataFrame({"x": [np.nan]}),
        )


def test_v4x_step_summary_reports_candidate_thresholds_and_economics() -> None:
    report = {
        "commit_sha": "a" * 40,
        "candidates": [
            {
                "candidate_id": "BTC/USDT|1h|Combined",
                "status": "TRAINED",
                "training_diagnostics": {"fit_rows": 60, "calibration_rows": 12},
                "validation_signal_counts": {
                    "primary_buy_event_count": 12,
                    "meta_accepted_buy_count": 6,
                    "meta_rejected_buy_count": 6,
                },
                "selected_meta_probability_threshold": 0.61,
                "metrics": {
                    "trade_count": 6,
                    "win_rate": 0.5,
                    "net_profit_factor": 1.4,
                    "net_profit_factor_infinite_no_losses": False,
                    "net_pnl_usd": 125.0,
                    "max_drawdown_percent": 4.5,
                    "economic_candidate_gate_passed": False,
                },
            }
        ],
    }

    summary = v4x._render_summary(report)

    assert "BTC/USDT|1h|Combined" in summary
    assert "0.610000" in summary
    assert "1.4000" in summary
    assert "125.00" in summary
    assert "4.50%" in summary
    assert "trading_authorization=NO-GO" in summary
    assert "final holdout is untouched" in summary


def test_v4x_workflow_is_branch_scoped_and_keeps_ci_and_safety_gates() -> None:
    workflow = Path(".github/workflows/ml-core-v4x-meta-labeling.yml").read_text(
        encoding="utf-8"
    )

    assert "research/ml-core-v4x-meta-labeling" in workflow
    assert "Require standard CI success for this exact V4-X commit" in workflow
    assert "TARGET_SHA" in workflow
    assert "actions: read" in workflow
    assert "Fetch complete public Binance Spot 730-day 1-minute datasets" in workflow
    assert (
        "Render V4-X candidate metrics and thresholds to GitHub Step Summary"
        in workflow
    )
    assert "_render_summary" in workflow
    assert "alix-v4x-meta-labeling-report.json" in workflow
    assert "Champion=null" in workflow
    assert "trading_authorization=NO-GO" in workflow
    assert "private_api_calls=0" in workflow
    assert "real_order_attempts=0" in workflow
    assert "LSTM" not in workflow
