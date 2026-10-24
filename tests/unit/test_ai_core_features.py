from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.ai_core.features import (
    AI_FEATURE_COLUMNS,
    BASE_AI_FEATURE_COLUMNS,
    CROSS_ASSET_FEATURE_COLUMNS,
    SENTIMENT_COLUMNS,
    _rsi,
    _volume_profile_proxy,
    aggregate_hourly_to_4h,
    build_ai_features,
)


def candles_4h(rows: int = 120, *, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 1_000.0 * np.exp(
        np.cumsum(0.0005 + 0.006 * np.sin(index / 7.0)) + offset
    )
    opened = np.r_[close[0] * 0.999, close[:-1]]
    high = np.maximum(opened, close) * (1.002 + 0.0002 * (index % 3))
    low = np.minimum(opened, close) * (0.998 - 0.0001 * (index % 2))
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01T00:30:00Z", periods=rows, freq="4h"
            ),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + (index % 17) * 7.0,
        }
    )


def candles_1h(rows: int = 12) -> pd.DataFrame:
    source = candles_4h(rows=rows)
    # Four hourly samples per synthetic 4H bar, with valid intrabar ranges.
    points: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        start = pd.Timestamp(row.date)
        for hour in range(4):
            opened = float(row.open) * (1.0 + hour * 0.0001)
            close = float(row.close) * (1.0 + hour * 0.0001)
            points.append(
                {
                    "date": start + pd.Timedelta(hours=hour),
                    "open": opened,
                    "high": max(opened, close) * 1.001,
                    "low": min(opened, close) * 0.999,
                    "close": close,
                    "volume": float(row.volume) / 4.0,
                }
            )
    return pd.DataFrame(points)


def test_feature_schema_has_58_base_features_and_six_real_eth_features() -> None:
    assert len(BASE_AI_FEATURE_COLUMNS) == 58
    assert len(CROSS_ASSET_FEATURE_COLUMNS) == 6
    assert len(AI_FEATURE_COLUMNS) == 64
    assert len(set(AI_FEATURE_COLUMNS)) == len(AI_FEATURE_COLUMNS)
    base = build_ai_features(candles_4h())
    assert all(name in base.columns for name in BASE_AI_FEATURE_COLUMNS)
    assert not set(CROSS_ASSET_FEATURE_COLUMNS).intersection(base.columns)


def test_feature_windows_are_causal_and_not_affected_by_future_rows() -> None:
    candles = candles_4h(100)
    shorter = build_ai_features(candles.iloc[:85].reset_index(drop=True))
    longer = build_ai_features(candles)
    left = shorter.loc[:, list(BASE_AI_FEATURE_COLUMNS)].to_numpy(dtype=float)
    right = longer.iloc[:85].loc[:, list(BASE_AI_FEATURE_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(left, right, equal_nan=True, rtol=0.0, atol=0.0)
    assert longer.loc[:, list(BASE_AI_FEATURE_COLUMNS)].iloc[:72].isna().any().any()


def test_cross_asset_features_require_exact_aligned_eth_data() -> None:
    btc = candles_4h(100)
    eth = candles_4h(100, offset=0.2)
    extended = build_ai_features(btc, cross_asset={"ETHUSDT": eth})
    assert all(name in extended.columns for name in AI_FEATURE_COLUMNS)
    assert extended.loc[:, list(CROSS_ASSET_FEATURE_COLUMNS)].iloc[40:].notna().all().all()

    shifted = eth.copy()
    shifted["date"] = shifted["date"] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="cross_asset_alignment_incomplete"):
        build_ai_features(btc, cross_asset={"ETHUSDT": shifted})
    with pytest.raises(ValueError, match="cross_asset_sources_missing"):
        build_ai_features(btc, cross_asset={"BTCUSDT": btc})


def test_hourly_to_four_hour_aggregation_is_exact_and_gap_aware() -> None:
    hourly = candles_1h()
    aggregated = aggregate_hourly_to_4h(hourly)
    assert len(aggregated) == 12
    assert aggregated["open"].iloc[0] == pytest.approx(hourly["open"].iloc[0])
    assert aggregated["close"].iloc[-1] == pytest.approx(hourly["close"].iloc[-1])
    assert aggregated["volume"].iloc[0] == pytest.approx(
        hourly["volume"].iloc[:4].sum()
    )
    with pytest.raises(ValueError, match="complete_4h_bars"):
        aggregate_hourly_to_4h(hourly.iloc[:-1])
    broken = hourly.copy()
    broken.loc[4, "date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="gap_or_mismatch"):
        aggregate_hourly_to_4h(broken)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("duplicate", "duplicate_timestamps"),
        ("descending", "not_chronological"),
        ("gap", "gap_or_mismatch"),
        ("naive", "timezone_aware"),
        ("infinite", "non_finite"),
        ("negative_volume", "out_of_range"),
        ("bad_ohlc", "ohlc_invariant"),
        ("missing", "columns_missing"),
    ),
)
def test_ohlcv_quality_validation_fails_closed(mutation: str, reason: str) -> None:
    frame = candles_4h(8)
    if mutation == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif mutation == "descending":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif mutation == "gap":
        frame.loc[2, "date"] += pd.Timedelta(minutes=1)
    elif mutation == "naive":
        frame["date"] = frame["date"].dt.tz_localize(None)
    elif mutation == "infinite":
        frame.loc[2, "volume"] = np.inf
    elif mutation == "negative_volume":
        frame.loc[2, "volume"] = -1.0
    elif mutation == "bad_ohlc":
        frame.loc[2, "high"] = frame.loc[2, "low"] - 1.0
    else:
        frame = frame.drop(columns=["low"])
    with pytest.raises(ValueError, match=reason):
        build_ai_features(frame)


def test_rsi_handles_no_losses_and_flat_prices_deterministically() -> None:
    increasing = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    flat = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    assert _rsi(increasing, 2).iloc[-1] == pytest.approx(100.0)
    assert _rsi(flat, 2).iloc[-1] == pytest.approx(50.0)


def test_volume_profile_proxy_reports_flat_price_and_zero_volume_windows() -> None:
    index = pd.RangeIndex(5)
    typical = pd.Series([10.0] * 5, index=index)
    close = typical.copy()
    point_of_control, area_position = _volume_profile_proxy(
        typical, pd.Series([0.0] * 5, index=index), close, 3
    )
    assert point_of_control.isna().all()
    assert area_position.isna().all()

    point_of_control, area_position = _volume_profile_proxy(
        typical, pd.Series([1.0] * 5, index=index), close, 3
    )
    assert point_of_control.iloc[-1] == pytest.approx(0.0)
    assert area_position.iloc[-1] == pytest.approx(0.5)


def test_point_in_time_sentiment_columns_are_joined_without_forward_fill() -> None:
    candles = candles_4h(10)
    sentiment = pd.DataFrame(
        {
            "date": candles["date"].iloc[[2, 5]],
            **{
                name: [0.1, 0.2] if name != "sentiment_available" else [True, True]
                for name in SENTIMENT_COLUMNS
            },
        }
    )
    joined = build_ai_features(candles, sentiment_features=sentiment)
    assert joined["sentiment_mean_24h"].isna().sum() == len(candles) - 2
    assert joined["sentiment_mean_24h"].iloc[2] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="sentiment_feature_schema_invalid"):
        build_ai_features(candles, sentiment_features=pd.DataFrame({"date": candles["date"]}))
