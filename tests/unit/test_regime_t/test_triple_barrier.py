"""Synthetic-only tests for the frozen Regime T triple-barrier contract."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_regime_t_logistic import FEATURE_COLUMNS, build_model
from scripts.label_regime_t_triple_barrier import (
    EXPECTED_PREREG_SHA256,
    PREREG_PATH,
    build_price_features,
    build_triple_barrier_labels,
)


def _synthetic_frame(rows: int = 90) -> pd.DataFrame:
    close = np.full(rows, 100.0, dtype=np.float64)
    opened = np.full(rows, 100.0, dtype=np.float64)
    high = np.full(rows, 100.5, dtype=np.float64)
    low = np.full(rows, 99.5, dtype=np.float64)
    timestamps = pd.date_range("2022-01-01T00:30:00Z", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.zeros(rows, dtype=np.float64),
        }
    )


def test_preregistration_is_locked_and_parameters_are_fixed() -> None:
    payload = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest() == EXPECTED_PREREG_SHA256
    assert payload["preregistration"]["status"] == "PREREGISTERED_LOCKED"
    assert payload["target"]["barriers"]["k"] == 2.0
    assert payload["target"]["time_barrier"]["max_holding_hours"] == 24
    assert payload["model"]["pipeline"] == [
        "sklearn.preprocessing.StandardScaler",
        "sklearn.linear_model.LogisticRegression",
    ]


def test_upper_and_lower_first_touch_labels_are_binary() -> None:
    frame = _synthetic_frame()
    vol = pd.Series(np.full(len(frame), 0.01, dtype=np.float64))
    frame.loc[6, "high"] = 103.0
    frame.loc[6, "close"] = 102.5
    frame.loc[16, "low"] = 97.0
    frame.loc[16, "close"] = 97.5

    labels = build_triple_barrier_labels(frame, volatility_fraction=vol)

    assert labels["label"].iloc[5] == 1
    assert labels.loc[5, "outcome"] == "UPPER_FIRST"
    assert labels["label"].iloc[15] == 0
    assert labels.loc[15, "outcome"] == "LOWER_FIRST"


def test_same_bar_double_touch_is_ambiguous_and_excluded() -> None:
    frame = _synthetic_frame()
    vol = pd.Series(np.full(len(frame), 0.01, dtype=np.float64))
    frame.loc[26, "high"] = 103.0
    frame.loc[26, "low"] = 97.0

    labels = build_triple_barrier_labels(frame, volatility_fraction=vol)

    assert labels.loc[25, "outcome"] == "AMBIGUOUS"
    assert pd.isna(labels.loc[25, "label"])


def test_no_touch_within_24_bars_is_censored_as_timeout() -> None:
    frame = _synthetic_frame()
    vol = pd.Series(np.full(len(frame), 0.01, dtype=np.float64))

    labels = build_triple_barrier_labels(frame, volatility_fraction=vol)

    assert labels.loc[40, "outcome"] == "TIMEOUT"
    assert pd.isna(labels.loc[40, "label"])


def test_tail_without_full_time_barrier_window_is_unavailable() -> None:
    frame = _synthetic_frame()
    vol = pd.Series(np.full(len(frame), 0.01, dtype=np.float64))

    labels = build_triple_barrier_labels(frame, volatility_fraction=vol)

    assert labels.loc[len(frame) - 24, "outcome"] == "UNAVAILABLE"


def test_price_features_have_exact_schema_and_ignore_volume() -> None:
    frame = _synthetic_frame(100)
    first = build_price_features(frame)
    changed_volume = frame.copy()
    changed_volume["volume"] = np.arange(len(frame), dtype=np.float64) * 100.0
    second = build_price_features(changed_volume)

    assert tuple(first.columns) == FEATURE_COLUMNS
    assert len(first.columns) == 11
    pd.testing.assert_frame_equal(first, second)


def test_price_features_do_not_depend_on_future_bars() -> None:
    frame = _synthetic_frame(100)
    baseline = build_price_features(frame)
    changed = frame.copy()
    changed.loc[70:, ["open", "high", "low", "close"]] *= 1.1
    updated = build_price_features(changed)

    pd.testing.assert_series_equal(baseline.iloc[60, :], updated.iloc[60, :])


def test_model_pipeline_is_scaled_logistic_regression() -> None:
    model = build_model()
    assert [name for name, _ in model.steps] == ["scale", "classifier"]
    assert model.named_steps["classifier"].get_params()["C"] == 1.0
    assert model.named_steps["classifier"].get_params()["max_iter"] == 2000


def test_non_registered_horizon_is_rejected() -> None:
    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="locked_target_parameters"):
        build_triple_barrier_labels(frame, horizon_bars=48)
