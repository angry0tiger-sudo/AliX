from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from services.ml_core.external_market_features import (
    EXTERNAL_FEATURE_COLUMNS,
    build_external_market_features,
)

START = pd.Timestamp("2026-01-01T00:00:00Z")


def _bars(count: int = 100) -> DataFrame:
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


def _funding(count: int = 80) -> DataFrame:
    times = pd.date_range(START, periods=count, freq="4h")
    rates = np.sin(np.arange(count) / 3.0) * 0.001
    return DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": times,
            "funding_rate": rates,
            "available_at": times + pd.Timedelta(minutes=5),
        }
    )


def test_funding_features_are_causal_and_have_exact_schema() -> None:
    bars = _bars()
    features, diagnostics = build_external_market_features(
        bars, _funding(100), symbol="BTCUSDT", timeframe="4h"
    )
    assert tuple(features.columns) == EXTERNAL_FEATURE_COLUMNS
    assert features.index.equals(bars.index)
    assert features["funding_rate_zscore"].iloc[:19].isna().all()
    assert features["funding_rate_ma"].iloc[:19].isna().all()
    assert np.isfinite(features.iloc[20:].to_numpy(dtype=float)).all()
    assert diagnostics["interpolation_used"] is False
    assert diagnostics["backfill_used"] is False
    assert diagnostics["future_observations_joined"] == 0
    assert int(diagnostics["complete_funding_feature_rows"]) > 0


def test_custom_indexes_and_future_publication_are_preserved() -> None:
    bars = _bars().set_axis(range(100, 200))
    features, _ = build_external_market_features(
        bars, _funding(), symbol="BTCUSDT", timeframe="4h"
    )
    assert features.index.equals(bars.index)
    before = bars.copy()
    before.loc[100, "date"] = START - pd.Timedelta(hours=4)
    delayed, _ = build_external_market_features(
        before, _funding(), symbol="BTCUSDT", timeframe="4h"
    )
    assert delayed.iloc[0].isna().all()


def test_stale_funding_observations_are_not_carried_forward() -> None:
    bars = _bars()
    bars["date"] += pd.Timedelta(days=30)
    features, diagnostics = build_external_market_features(
        bars, _funding(), symbol="BTCUSDT", timeframe="4h"
    )
    assert features.isna().all(axis=None)
    assert diagnostics["complete_funding_feature_rows"] == 0


def test_long_gap_resets_funding_rolling_warmup() -> None:
    funding = _funding(80)
    funding.loc[40:, "funding_time"] += pd.Timedelta(days=4)
    funding.loc[40:, "available_at"] += pd.Timedelta(days=4)
    bars = _bars(130)
    features, _ = build_external_market_features(
        bars, funding, symbol="BTCUSDT", timeframe="4h"
    )
    second_event = pd.Timestamp(funding["funding_time"].iloc[40])
    after_gap = bars["date"] >= second_event
    assert features.loc[after_gap, "funding_rate_zscore"].iloc[:19].isna().all()
    assert features.loc[after_gap, "funding_rate_ma"].iloc[:19].isna().all()


def test_builder_rejects_invalid_timeframes_bars_and_funding_schema() -> None:
    with pytest.raises(ValueError, match="timeframe_not_supported"):
        build_external_market_features(
            _bars(), _funding(), symbol="BTCUSDT", timeframe="1m"
        )
    with pytest.raises(ValueError, match="bars_schema_invalid"):
        build_external_market_features(
            DataFrame(), _funding(), symbol="BTCUSDT", timeframe="4h"
        )
    duplicate = _bars()
    duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
    with pytest.raises(ValueError, match="not_unique_increasing"):
        build_external_market_features(
            duplicate, _funding(), symbol="BTCUSDT", timeframe="4h"
        )
    with pytest.raises(ValueError, match="funding_schema_or_data_invalid"):
        build_external_market_features(
            _bars(), DataFrame(), symbol="BTCUSDT", timeframe="4h"
        )


def test_builder_rejects_wrong_symbol_early_availability_and_invalid_rates() -> None:
    funding = _funding()
    funding.loc[0, "symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="funding_symbol_mismatch"):
        build_external_market_features(
            _bars(), funding, symbol="BTCUSDT", timeframe="4h"
        )
    funding = _funding()
    funding.loc[0, "available_at"] = funding.loc[0, "funding_time"]
    with pytest.raises(ValueError, match="available_before_conservative_lag"):
        build_external_market_features(
            _bars(), funding, symbol="BTCUSDT", timeframe="4h"
        )
    funding = _funding()
    funding.loc[0, "funding_rate"] = np.inf
    with pytest.raises(ValueError, match="numeric_invalid"):
        build_external_market_features(
            _bars(), funding, symbol="BTCUSDT", timeframe="4h"
        )


def test_builder_rejects_duplicate_or_unsorted_funding_timestamps() -> None:
    funding = _funding()
    funding.loc[4, "funding_time"] = funding.loc[3, "funding_time"]
    with pytest.raises(ValueError, match="not_unique_increasing:funding"):
        build_external_market_features(
            _bars(), funding, symbol="BTCUSDT", timeframe="4h"
        )
    funding = _funding()
    funding.loc[10, "available_at"], funding.loc[11, "available_at"] = (
        funding.loc[11, "available_at"],
        funding.loc[10, "available_at"],
    )
    with pytest.raises(ValueError, match="not_unique_increasing:funding_available"):
        build_external_market_features(
            _bars(), funding, symbol="BTCUSDT", timeframe="4h"
        )


def test_funding_age_must_be_positive() -> None:
    with pytest.raises(ValueError, match="funding_max_age_invalid"):
        build_external_market_features(
            _bars(),
            _funding(),
            symbol="BTCUSDT",
            timeframe="4h",
            funding_max_age=pd.Timedelta(0),
        )
