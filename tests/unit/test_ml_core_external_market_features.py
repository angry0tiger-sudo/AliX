from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.external_market_features import (
    EXTERNAL_FEATURE_COLUMNS,
    build_external_market_features,
)

UTC = timezone.utc
START = pd.Timestamp("2026-01-01T00:00:00Z")


def _bars(count: int = 28) -> DataFrame:
    return DataFrame(
        {
            "date": pd.date_range(START, periods=count, freq="4h"),
            "open": np.full(count, 100.0),
            "high": np.full(count, 101.0),
            "low": np.full(count, 99.0),
            "close": np.full(count, 100.0),
            "volume": np.full(count, 10.0),
        }
    )


def _funding(count: int = 24) -> DataFrame:
    times = pd.date_range(START, periods=count, freq="4h")
    return DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": times,
            "funding_rate": np.linspace(-0.001, 0.001, count),
            "available_at": times + pd.Timedelta(minutes=5),
        }
    )


def _oi(count: int = 28) -> DataFrame:
    times = pd.date_range(START, periods=count, freq="4h")
    return DataFrame(
        {
            "symbol": "BTCUSDT",
            "period": "4h",
            "timestamp": times,
            "sum_open_interest": np.linspace(100.0, 200.0, count),
            "available_at": times + pd.Timedelta(hours=4, minutes=5),
        }
    )


def test_external_features_are_causal_and_have_exact_schema() -> None:
    bars = _bars()
    funding = _funding()
    oi = _oi()
    features, diagnostics = build_external_market_features(
        bars, funding, oi, symbol="BTCUSDT"
    )
    assert tuple(features.columns) == EXTERNAL_FEATURE_COLUMNS
    assert features.index.equals(bars.index)
    assert features["funding_rate_zscore"].iloc[:18].isna().all()
    assert np.isfinite(features["funding_rate_zscore"].iloc[20:]).all()
    assert features["oi_change"].iloc[:2].isna().all()
    assert np.isfinite(features["oi_change"].iloc[3:]).all()
    assert diagnostics["interpolation_used"] is False
    assert diagnostics["backfill_used"] is False
    assert diagnostics["future_observations_joined"] == 0


def test_external_features_preserve_custom_bar_index() -> None:
    bars = _bars().set_axis(range(100, 128))
    funding = _funding()
    oi = _oi()
    features, _ = build_external_market_features(bars, funding, oi, symbol="BTCUSDT")
    assert features.index.equals(bars.index)


def test_funding_and_oi_observations_are_not_visible_before_available_at() -> None:
    bars = _bars(24)
    funding = _funding(24)
    oi = _oi(24)
    bars.loc[0, "date"] = START - pd.Timedelta(hours=4)
    features, _ = build_external_market_features(bars, funding, oi, symbol="BTCUSDT")
    assert pd.isna(features["funding_rate_zscore"].iloc[0])
    assert pd.isna(features["oi_change"].iloc[0])


def test_stale_external_observations_become_missing_not_filled() -> None:
    bars = _bars(30)
    funding = _funding(22).iloc[:22].copy()
    oi = _oi(22).iloc[:22].copy()
    bars["date"] = pd.date_range(
        START + pd.Timedelta(days=10), periods=len(bars), freq="4h"
    )
    features, diagnostics = build_external_market_features(
        bars, funding, oi, symbol="BTCUSDT"
    )
    assert features.isna().all(axis=None)
    assert diagnostics["complete_external_feature_rows"] == 0


def test_missing_oi_interval_breaks_percentage_change_without_interpolation() -> None:
    bars = _bars(30)
    oi = _oi(24).drop(index=10).reset_index(drop=True)
    features, _ = build_external_market_features(
        bars, _funding(24), oi, symbol="BTCUSDT"
    )
    assert pd.isna(features["oi_change"].iloc[13])
    assert pd.isfinite(features["oi_change"].iloc[14])


def test_long_funding_gap_resets_rolling_zscore_warmup() -> None:
    funding = _funding(42)
    funding.loc[21:, "funding_time"] += pd.Timedelta(days=4)
    funding.loc[21:, "available_at"] += pd.Timedelta(days=4)
    bars = _bars(50)
    features, _ = build_external_market_features(
        bars, funding, _oi(50), symbol="BTCUSDT"
    )
    second_event_time = pd.Timestamp(funding["funding_time"].iloc[21])
    second_group_rows = bars["date"] >= second_event_time
    assert features.loc[second_group_rows, "funding_rate_zscore"].iloc[:18].isna().all()


def test_feature_builder_rejects_bad_bars_and_external_schema() -> None:
    with pytest.raises(ValueError, match="timeframe_not_supported"):
        build_external_market_features(_bars(), _funding(), _oi(), symbol="BTCUSDT", timeframe="1h")
    with pytest.raises(ValueError, match="bars_schema_invalid"):
        build_external_market_features(DataFrame(), _funding(), _oi(), symbol="BTCUSDT")
    duplicate = _bars()
    duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
    with pytest.raises(ValueError, match="not_unique_increasing"):
        build_external_market_features(duplicate, _funding(), _oi(), symbol="BTCUSDT")
    with pytest.raises(ValueError, match="funding_schema_or_data_invalid"):
        build_external_market_features(_bars(), DataFrame(), _oi(), symbol="BTCUSDT")


def test_feature_builder_rejects_symbol_period_timestamp_and_numeric_mismatch() -> None:
    funding = _funding()
    funding.loc[0, "symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="funding_symbol_mismatch"):
        build_external_market_features(_bars(), funding, _oi(), symbol="BTCUSDT")
    oi = _oi()
    oi.loc[1, "period"] = "1h"
    with pytest.raises(ValueError, match="oi_period_mismatch"):
        build_external_market_features(_bars(), _funding(), oi, symbol="BTCUSDT")
    funding = _funding()
    funding.loc[0, "available_at"] = funding.loc[0, "funding_time"]
    with pytest.raises(ValueError, match="available_before_conservative_lag"):
        build_external_market_features(_bars(), funding, _oi(), symbol="BTCUSDT")
    oi = _oi()
    oi.loc[0, "available_at"] = oi.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="available_before_period_close"):
        build_external_market_features(_bars(), _funding(), oi, symbol="BTCUSDT")
    oi = _oi()
    oi.loc[1, "sum_open_interest"] = 0.0
    with pytest.raises(ValueError, match="numeric_invalid"):
        build_external_market_features(_bars(), _funding(), oi, symbol="BTCUSDT")


def test_feature_builder_rejects_duplicate_or_unsorted_source_times() -> None:
    funding = _funding()
    funding.loc[4, "funding_time"] = funding.loc[3, "funding_time"]
    with pytest.raises(ValueError, match="not_unique_increasing:funding"):
        build_external_market_features(_bars(), funding, _oi(), symbol="BTCUSDT")
    oi = _oi()
    oi.loc[5, "timestamp"], oi.loc[6, "timestamp"] = (
        oi.loc[6, "timestamp"],
        oi.loc[5, "timestamp"],
    )
    oi["available_at"] = oi["timestamp"] + pd.Timedelta(hours=4, minutes=5)
    with pytest.raises(ValueError, match="not_unique_increasing:open_interest"):
        build_external_market_features(_bars(), _funding(), oi, symbol="BTCUSDT")


def test_funding_max_age_must_be_positive() -> None:
    with pytest.raises(ValueError, match="funding_max_age_invalid"):
        build_external_market_features(
            _bars(), _funding(), _oi(), symbol="BTCUSDT", funding_max_age=pd.Timedelta(0)
        )
