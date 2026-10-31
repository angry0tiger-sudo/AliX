from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.freqai_baseline.features import FEATURE_COLUMNS, TARGET_COLUMN
from services.ml_core.extended_features import (
    EXTENDED_FEATURE_SCHEMA_VERSION,
    FEATURE_GROUP_COLUMNS,
    FEATURE_PROFILE_ORDER,
    build_extended_feature_groups,
    build_feature_profiles,
    feature_profile_columns,
    feature_profile_groups,
    feature_profile_manifest,
)


def market_frame(rows: int = 120) -> DataFrame:
    index = np.arange(rows, dtype=float)
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    close = 100.0 + 0.08 * index + 1.5 * np.sin(index / 5.0)
    opened = close - 0.2 * np.sin(index / 3.0)
    high = np.maximum(opened, close) + 0.3
    low = np.minimum(opened, close) - 0.3
    volume = 100.0 + (index % 7.0) * 2.0 + index * 0.1
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_profile_contract_preserves_nine_column_baseline() -> None:
    profiles = build_feature_profiles(market_frame())
    manifest = feature_profile_manifest()

    assert EXTENDED_FEATURE_SCHEMA_VERSION == "alix.ml-core.extended-features.v1"
    assert tuple(profiles) == FEATURE_PROFILE_ORDER
    assert tuple(profiles["Baseline"].columns) == FEATURE_COLUMNS
    assert len(profiles["Baseline"].columns) == 9
    assert len(profiles["Baseline+Trend"].columns) == 17
    assert len(profiles["Baseline+Volatility"].columns) == 12
    assert len(profiles["Baseline+Volume"].columns) == 13
    assert len(profiles["Combined"].columns) == 24
    assert [item["name"] for item in manifest] == list(FEATURE_PROFILE_ORDER)
    for item in manifest:
        assert isinstance(item["columns"], list)
        assert item["feature_count"] == len(item["columns"])


def test_extended_groups_are_finite_after_their_rolling_warmup() -> None:
    groups = build_extended_feature_groups(market_frame())

    assert set(groups) == {"Trend", "Volatility", "Volume"}
    for name, columns in FEATURE_GROUP_COLUMNS.items():
        frame = groups[name]
        assert tuple(frame.columns) == columns
        assert not np.isinf(frame.to_numpy(dtype=np.float64)).any()
        assert np.isfinite(frame.iloc[60:].to_numpy(dtype=np.float64)).all()


def test_all_profiles_have_no_infinity_or_post_warmup_nan() -> None:
    profiles = build_feature_profiles(market_frame())

    for name, frame in profiles.items():
        assert tuple(frame.columns) == feature_profile_columns(name)
        assert not np.isinf(frame.to_numpy(dtype=np.float64)).any()
        assert np.isfinite(frame.iloc[60:].to_numpy(dtype=np.float64)).all()


def test_future_rows_cannot_change_any_feature_at_the_decision_time() -> None:
    candles = market_frame()
    decision_index = 70
    original = build_feature_profiles(candles)
    changed = candles.copy()
    future = np.arange(len(changed) - decision_index - 1, dtype=float)
    changed.loc[decision_index + 1 :, "open"] = 300.0 + future
    changed.loc[decision_index + 1 :, "close"] = 300.5 + future
    changed.loc[decision_index + 1 :, "high"] = 301.0 + future
    changed.loc[decision_index + 1 :, "low"] = 299.5 + future
    changed.loc[decision_index + 1 :, "volume"] = 1_000_000.0 + future

    mutated = build_feature_profiles(changed)
    for profile in FEATURE_PROFILE_ORDER:
        pd.testing.assert_frame_equal(
            original[profile].iloc[: decision_index + 1],
            mutated[profile].iloc[: decision_index + 1],
        )


def test_prefix_calculation_matches_full_series_at_same_point_in_time() -> None:
    candles = market_frame()
    decision_index = 75
    full = build_feature_profiles(candles)
    prefix = build_feature_profiles(candles.iloc[: decision_index + 1])

    for profile in FEATURE_PROFILE_ORDER:
        np.testing.assert_allclose(
            full[profile].iloc[decision_index].to_numpy(dtype=np.float64),
            prefix[profile].iloc[-1].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def test_feature_windows_do_not_reuse_previous_window_state() -> None:
    candles = market_frame()
    isolated = build_feature_profiles(candles.iloc[40:].reset_index(drop=True))

    assert isolated["Baseline"].iloc[0]["%-return_1"] != pytest.approx(
        build_feature_profiles(candles)["Baseline"].iloc[40]["%-return_1"]
    )
    assert pd.isna(isolated["Combined"].iloc[0]["%-trend_ema_distance_12"])
    assert pd.isna(isolated["Combined"].iloc[0]["%-volume_relative_20"])


def test_target_and_extra_columns_are_never_feature_inputs() -> None:
    candles = market_frame()
    plain = build_feature_profiles(candles)
    augmented = candles.copy()
    augmented[TARGET_COLUMN] = np.arange(len(candles), dtype=float) * 1e9
    augmented["future_feature_sentinel"] = -1e100
    contaminated = build_feature_profiles(augmented)

    for profile in FEATURE_PROFILE_ORDER:
        assert TARGET_COLUMN not in contaminated[profile].columns
        assert "future_feature_sentinel" not in contaminated[profile].columns
        pd.testing.assert_frame_equal(plain[profile], contaminated[profile])


def test_constant_volume_uses_finite_zero_zscore_and_unit_relative_volume() -> None:
    candles = market_frame()
    candles["volume"] = 100.0
    volume = build_extended_feature_groups(candles)["Volume"]

    assert volume.loc[19, "%-volume_relative_20"] == pytest.approx(1.0)
    assert volume.loc[19, "%-volume_zscore_20"] == pytest.approx(0.0)
    assert np.isfinite(volume.iloc[60:].to_numpy(dtype=np.float64)).all()


def test_zero_volume_has_no_infinite_proxy_values() -> None:
    candles = market_frame()
    candles["volume"] = 0.0
    volume = build_extended_feature_groups(candles)["Volume"]

    assert not np.isinf(volume.to_numpy(dtype=np.float64)).any()
    assert volume["%-volume_relative_20"].isna().all()
    assert volume["%-volume_obv_imbalance_20"].isna().all()
    assert volume["%-volume_chaikin_money_flow_20"].isna().all()


def test_flat_zero_range_candles_produce_finite_adx_and_bollinger_width() -> None:
    candles = market_frame()
    candles["open"] = 100.0
    candles["high"] = 100.0
    candles["low"] = 100.0
    candles["close"] = 100.0
    groups = build_extended_feature_groups(candles)

    assert groups["Trend"].iloc[-1]["%-trend_adx_14"] == pytest.approx(0.0)
    assert groups["Trend"].iloc[-1]["%-trend_plus_di_14"] == pytest.approx(0.0)
    assert groups["Trend"].iloc[-1]["%-trend_minus_di_14"] == pytest.approx(0.0)
    assert groups["Volatility"].iloc[-1][
        "%-volatility_bollinger_bandwidth_20"
    ] == pytest.approx(0.0)
    assert not any(
        np.isinf(frame.to_numpy(dtype=np.float64)).any() for frame in groups.values()
    )


def test_invalid_ohlcv_schema_and_numeric_values_fail_closed() -> None:
    candles = market_frame()
    with pytest.raises(ValueError, match="ohlcv_missing_columns"):
        build_extended_feature_groups(candles.drop(columns=["volume"]))

    invalid_numeric = candles.copy()
    invalid_numeric.loc[20, "close"] = np.inf
    with pytest.raises(ValueError, match="ohlcv_value_not_finite"):
        build_extended_feature_groups(invalid_numeric)

    invalid_geometry = candles.copy()
    invalid_geometry.loc[20, "high"] = 0.5
    with pytest.raises(ValueError, match="ohlcv_geometry_invalid"):
        build_extended_feature_groups(invalid_geometry)


def test_invalid_timestamps_fail_closed() -> None:
    candles = market_frame()
    naive = candles.copy()
    naive["date"] = naive["date"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone_required"):
        build_extended_feature_groups(naive)

    duplicated = candles.copy()
    duplicated.loc[20, "date"] = duplicated.loc[19, "date"]
    with pytest.raises(ValueError, match="timestamps_must_be_unique_and_increasing"):
        build_extended_feature_groups(duplicated)

    irregular = candles.copy()
    irregular.loc[20, "date"] = pd.Timestamp("2026-01-01T20:10:00Z")
    with pytest.raises(ValueError, match="timestamp_interval_inconsistent"):
        build_extended_feature_groups(irregular)


def test_unknown_feature_profiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="feature_profile_unknown"):
        feature_profile_groups("TrendOnly")
    with pytest.raises(ValueError, match="feature_profile_unknown"):
        feature_profile_columns("Baseline+Unknown")
