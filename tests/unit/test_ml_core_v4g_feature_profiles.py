from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.extended_features import (
    FEATURE_PROFILE_ORDER,
    build_feature_profiles,
)


def _candles(rows: int = 180) -> DataFrame:
    dates = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="15min")
    close = 100.0 + np.arange(rows, dtype=float) * 0.05
    return DataFrame(
        {
            "date": dates,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100.0 + np.arange(rows, dtype=float),
        }
    )


def test_selected_profile_builder_computes_only_baseline_and_trend_profiles() -> None:
    selected = ("Baseline", "Baseline+Trend")

    profiles = build_feature_profiles(_candles(), profile_names=selected)

    assert tuple(profiles) == selected
    assert profiles["Baseline"].shape[1] == 9
    assert profiles["Baseline+Trend"].shape[1] == 17
    assert all(
        column in profiles["Baseline+Trend"].columns
        for column in profiles["Baseline"].columns
    )


def test_default_profile_builder_preserves_v4f_profile_contract() -> None:
    profiles = build_feature_profiles(_candles())

    assert tuple(profiles) == FEATURE_PROFILE_ORDER
    assert [profiles[name].shape[1] for name in FEATURE_PROFILE_ORDER] == [
        9,
        17,
        12,
        13,
        24,
    ]
