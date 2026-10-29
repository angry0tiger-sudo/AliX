from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.freqai_baseline.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    forward_return_target,
)
from services.ml_core.causal_features import build_causal_feature_matrix
from services.ml_core.purged_embargo import create_purged_embargo_split


def market_frame(rows: int = 80, *, start: str = "2026-01-01T00:00:00Z") -> DataFrame:
    dates = pd.date_range(start, periods=rows, freq="h")
    close = 100.0 + np.arange(rows, dtype=float) * 0.1
    opened = close - 0.02
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": close + 0.5,
            "low": opened - 0.5,
            "close": close,
            "volume": 10.0 + np.arange(rows, dtype=float),
        }
    )


def test_future_sentinel_and_future_prices_cannot_change_features_at_time_t() -> None:
    candles = market_frame()
    decision_index = 35
    candles["future_feature_sentinel"] = np.nan
    candles.loc[decision_index + 1 :, "future_feature_sentinel"] = 9.99e99
    original_features = build_causal_feature_matrix(candles)

    changed_future = candles.copy()
    changed_future.loc[decision_index + 1 :, "open"] = 50_000.0
    changed_future.loc[decision_index + 1 :, "high"] = 50_010.0
    changed_future.loc[decision_index + 1 :, "low"] = 49_990.0
    changed_future.loc[decision_index + 1 :, "close"] = 50_005.0
    changed_future.loc[decision_index + 1 :, "volume"] = 1_000_000.0
    changed_future["future_feature_sentinel"] = -1.0e100
    changed_features = build_causal_feature_matrix(changed_future)

    assert tuple(original_features.columns) == FEATURE_COLUMNS
    assert "future_feature_sentinel" not in original_features.columns
    pd.testing.assert_frame_equal(
        original_features.iloc[: decision_index + 1],
        changed_features.iloc[: decision_index + 1],
    )


def test_target_is_separate_and_never_enters_feature_matrix() -> None:
    candles = market_frame()
    candles[TARGET_COLUMN] = forward_return_target(candles, horizon=5)
    candles["future_prediction"] = np.arange(len(candles), dtype=float) + 1e9

    features = build_causal_feature_matrix(candles)
    target = forward_return_target(candles, horizon=5)

    assert TARGET_COLUMN not in features.columns
    assert "future_prediction" not in features.columns
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert target.iloc[10] == pytest.approx(
        (candles.iloc[15]["close"] / candles.iloc[10]["close"]) - 1.0
    )


def test_new_window_does_not_carry_rolling_state_from_previous_window() -> None:
    previous_window = market_frame(40)
    next_start = (previous_window["date"].iloc[-1] + pd.Timedelta(hours=1)).isoformat()
    current_window = market_frame(25, start=next_start)

    isolated_features = build_causal_feature_matrix(current_window)
    combined_features = build_causal_feature_matrix(
        pd.concat([previous_window, current_window], ignore_index=True)
    )
    combined_current = combined_features.iloc[len(previous_window) :].reset_index(
        drop=True
    )

    assert pd.isna(isolated_features.iloc[0]["%-return_1"])
    assert isolated_features.iloc[:13]["%-atr_proxy_14"].isna().all()
    assert pd.notna(combined_current.iloc[0]["%-return_1"])
    assert pd.isna(isolated_features.iloc[0]["%-return_1"])


def test_purge_uses_actual_label_window_overlap_not_fixed_horizon() -> None:
    row_count = 30
    label_end_indices = list(range(row_count))
    label_end_indices[5] = 12
    label_end_indices[10] = 14

    split = create_purged_embargo_split(
        label_end_indices,
        validation_start=10,
        validation_end=10,
        longest_feature_window=3,
    )

    assert split.validation_indices == (10,)
    assert split.purge_start == 11
    assert 5 in split.purged_indices
    assert 5 not in split.train_indices
    assert 6 in split.train_indices


def test_embargo_excludes_exact_longest_feature_window_after_test() -> None:
    row_count = 40
    label_end_indices = list(range(row_count))
    split = create_purged_embargo_split(
        label_end_indices,
        validation_start=10,
        validation_end=14,
        longest_feature_window=4,
    )

    expected_embargo = tuple(range(15, 19))
    assert split.embargo_size == 4
    assert split.embargo_indices == expected_embargo
    assert len(split.embargo_indices) == split.embargo_size
    assert not set(expected_embargo).intersection(split.train_indices)
    assert 19 in split.train_indices


def test_embargo_one_percent_floor_when_larger_than_feature_window() -> None:
    row_count = 250
    label_end_indices = list(range(row_count))
    split = create_purged_embargo_split(
        label_end_indices,
        validation_start=50,
        validation_end=60,
        longest_feature_window=1,
    )

    assert split.embargo_size == 2
    assert split.embargo_indices == (61, 62)
    assert 63 in split.train_indices


def test_purge_embargo_validation_rejects_invalid_configuration_and_indices() -> None:
    label_end_indices = list(range(40))
    with pytest.raises(ValueError, match="label_end_indices_empty"):
        create_purged_embargo_split(
            [], validation_start=5, validation_end=9, longest_feature_window=4
        )
    with pytest.raises(
        ValueError, match="longest_feature_window_must_be_positive_integer"
    ):
        create_purged_embargo_split(
            label_end_indices,
            validation_start=5,
            validation_end=9,
            longest_feature_window=0,
        )
    with pytest.raises(ValueError, match="validation_window_out_of_bounds"):
        create_purged_embargo_split(
            label_end_indices,
            validation_start=20,
            validation_end=50,
            longest_feature_window=4,
        )
    with pytest.raises(ValueError, match="label_end_index_out_of_bounds"):
        create_purged_embargo_split(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
            validation_start=5,
            validation_end=6,
            longest_feature_window=4,
        )
    with pytest.raises(ValueError, match="label_end_index_must_be_integer"):
        create_purged_embargo_split(
            [0, 1, 2.5, 3, 4, 5, 6, 7, 8, 9],
            validation_start=5,
            validation_end=6,
            longest_feature_window=4,
        )


def test_feature_builder_fails_closed_on_timestamps_and_ohlcv_schema() -> None:
    candles = market_frame(20)
    with pytest.raises(ValueError, match="ohlcv_missing_columns"):
        build_causal_feature_matrix(candles.drop(columns=["volume"]))
    with pytest.raises(ValueError, match="timezone_required"):
        naive = candles.copy()
        naive["date"] = naive["date"].dt.tz_localize(None)
        build_causal_feature_matrix(naive)
    with pytest.raises(ValueError, match="timestamp_interval_inconsistent"):
        gapped = candles.copy()
        gapped.loc[8, "date"] = pd.Timestamp("2026-01-01T08:30:00Z")
        build_causal_feature_matrix(gapped)
