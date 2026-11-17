"""Synthetic-only unit tests for Regime R's evaluation mechanics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.regime_r.ngboost_cqr.contracts import (
    ConformalInterval,
    DistributionForecast,
)
from scripts.run_regime_r_trials import (
    FEATURE_COLUMNS,
    build_features,
    build_target,
    profit_factor,
    simulate_costed_ledger,
    summarize_forecast_diagnostics,
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


def test_forecast_diagnostics_summarize_quantiles_intervals_and_cost_distance() -> None:
    forecasts = (
        DistributionForecast(
            horizon="1h",
            mean=None,
            dispersion=None,
            quantile_levels=(0.05, 0.5, 0.95),
            quantile_values=(-0.02, 0.01, 0.04),
            output_semantics="quantile_summary",
        ),
        DistributionForecast(
            horizon="1h",
            mean=None,
            dispersion=None,
            quantile_levels=(0.05, 0.5, 0.95),
            quantile_values=(-0.01, -0.005, 0.005),
            output_semantics="quantile_summary",
        ),
    )
    intervals = (
        ConformalInterval(
            lower_bound=-0.03,
            upper_bound=0.06,
            nominal_coverage=0.9,
            lower_correction=0.01,
            upper_correction=0.02,
            calibration_size=20,
        ),
        ConformalInterval(
            lower_bound=-0.015,
            upper_bound=0.02,
            nominal_coverage=0.9,
            lower_correction=0.005,
            upper_correction=0.015,
            calibration_size=20,
        ),
    )

    result = summarize_forecast_diagnostics(forecasts, intervals)

    assert result["prediction_count"] == 2
    assert np.isclose(result["lower_quantile"]["mean"], -0.015)
    assert np.isclose(result["lower_quantile"]["std"], 0.005)
    assert np.isclose(result["median"]["mean"], 0.0025)
    assert np.isclose(result["upper_quantile"]["mean"], 0.0225)
    assert np.isclose(result["conformal_interval_width"]["mean"], 0.0625)
    assert result["median_positive"]["count"] == 1
    assert result["median_positive"]["percentage"] == 50.0
    assert np.isclose(result["lower_bound_to_cost"]["50"]["mean_distance_bps"], -275.0)
    assert np.isclose(result["lower_bound_to_cost"]["90"]["mean_distance_bps"], -315.0)


def test_feature_schema_and_causality() -> None:
    frame = synthetic_frame()
    features = build_features(frame)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.shape == (len(frame), 13)
    changed = frame.copy()
    changed.loc[70:, "close"] *= 1.5
    changed_features = build_features(changed)
    np.testing.assert_array_equal(
        features.loc[60].to_numpy(),
        changed_features.loc[60].to_numpy(),
    )


def test_target_uses_next_open_and_horizon_close() -> None:
    frame = synthetic_frame(10)
    target = build_target(frame, 4)
    expected = float(frame["close"].iloc[4]) / float(frame["open"].iloc[1]) - 1.0
    assert target[0] == expected
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


def test_constant_zero_volume_uses_neutral_zscore_without_backfilling_warmup() -> None:
    frame = synthetic_frame(64)
    frame["volume"] = 0.0
    features = build_features(frame)

    assert features["volume_zscore_24"].iloc[:23].isna().all()
    assert features["volume_zscore_24"].iloc[23:].eq(0.0).all()
    assert features["log_volume_change_1"].iloc[1:].eq(0.0).all()


def test_registered_fold_boundaries_allow_contiguous_shared_edges() -> None:
    import json

    from scripts.run_regime_r_trials import build_folds

    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "regime_r"
        / "hgbq_cqr_preregistration.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    anchor = pd.Timestamp("2021-10-31T20:30:00Z")
    end = anchor + pd.DateOffset(months=16)
    timestamps = pd.date_range(anchor, end=end, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps})

    folds = build_folds(frame, {"start_utc": anchor.isoformat()}, config)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_start < fold.train_end == fold.calibration_start
        assert fold.calibration_start < fold.calibration_end == fold.embargo_start
        assert fold.embargo_start < fold.embargo_end == fold.test_start
        assert fold.test_start < fold.test_end
