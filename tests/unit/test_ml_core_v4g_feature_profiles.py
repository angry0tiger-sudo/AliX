from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.extended_features import (
    FEATURE_PROFILE_ORDER,
    build_extended_feature_groups,
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


def test_v4g_builds_only_trend_group_for_baseline_and_trend_profiles() -> None:
    groups = build_extended_feature_groups(_candles(), group_names=("Trend",))
    profiles = build_feature_profiles(
        _candles(), profile_names=("Baseline", "Baseline+Trend")
    )

    assert tuple(groups) == ("Trend",)
    assert tuple(profiles) == ("Baseline", "Baseline+Trend")
    assert profiles["Baseline"].shape[1] == 9
    assert profiles["Baseline+Trend"].shape[1] == 17


def test_selected_feature_group_builder_rejects_invalid_selection() -> None:
    with pytest.raises(
        ValueError, match="extended_feature_group_selection_invalid"
    ):
        build_extended_feature_groups(_candles(), group_names=())

    with pytest.raises(ValueError, match="extended_feature_group_unknown"):
        build_extended_feature_groups(_candles(), group_names=("Future",))


def test_selected_feature_profiles_reject_duplicate_or_unknown_profiles() -> None:
    with pytest.raises(ValueError, match="feature_profile_selection_invalid"):
        build_feature_profiles(_candles(), profile_names=("Baseline", "Baseline"))

    with pytest.raises(ValueError, match="feature_profile_unknown"):
        build_feature_profiles(_candles(), profile_names=("Baseline", "Unknown"))


def test_default_feature_profile_builder_preserves_v4f_profile_order() -> None:
    profiles = build_feature_profiles(_candles())

    assert tuple(profiles) == FEATURE_PROFILE_ORDER
