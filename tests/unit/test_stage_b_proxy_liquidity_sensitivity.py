from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_stage_b_proxy_liquidity_sensitivity import (
    BOOK_TIME,
    BOOK_MANIFEST_SCHEMA,
    MAX_BINANCE_VOLUME_SHARE,
    PROXY_DAILY_QUOTE_CAP,
    REFERENCE_ORDER_NOTIONAL,
    _align_proxy_volume,
    _scale_binance_quote_volume,
    _validate_bookdepth_manifest,
    add_banded_orderbook_features,
)


def _book_rows(
    timestamp: str,
    bid_scale: float = 3.0,
    ask_scale: float = 1.0,
    mid: float = 100_000.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for band in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        k = abs(band)
        scale = bid_scale if band < 0 else ask_scale
        depth = float(k) * scale
        implied_price = mid * (1.0 - 0.005 * k if band < 0 else 1.0 + 0.005 * k)
        rows.append(
            {
                BOOK_TIME: timestamp,
                "percentage": band,
                "depth": depth,
                "notional": depth * implied_price,
            }
        )
    return rows


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="h"),
            "open": [100_000.0, 100_000.0],
            "high": [100_500.0, 100_500.0],
            "low": [99_500.0, 99_500.0],
            "close": [100_000.0, 100_000.0],
            "volume": [0.0, 0.0],
        }
    )


def test_binance_proxy_never_exceeds_ten_percent_or_daily_estimate() -> None:
    klines = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "quote_volume": [20_000.0, 30_000.0, 20_000_000.0],
        }
    )
    scaled, stats = _scale_binance_quote_volume(klines)
    assert (scaled["proxy_scale_ratio"] <= MAX_BINANCE_VOLUME_SHARE).all()
    assert (scaled["proxy_quote_volume"] <= scaled["quote_volume"] * 0.10 + 1e-9).all()
    totals = scaled.groupby(scaled["date"].dt.floor("D"))["proxy_quote_volume"].sum()
    assert (totals <= PROXY_DAILY_QUOTE_CAP + 1e-6).all()
    assert totals.iloc[0] == pytest.approx(5_000.0)
    assert totals.iloc[1] == pytest.approx(PROXY_DAILY_QUOTE_CAP)
    assert stats["historical_nobitex_volume_replaced"] is False


def test_ten_quote_unit_order_requires_at_least_one_thousand_proxy_turnover() -> None:
    cap = np.asarray([1_000.0, 999.0]) * 0.01
    assert cap[0] >= REFERENCE_ORDER_NOTIONAL
    assert cap[1] < REFERENCE_ORDER_NOTIONAL


def test_banded_book_adapter_emits_only_relative_features_and_joins_prior_snapshot() -> (
    None
):
    first = "2024-01-01T00:59:30Z"
    second = "2024-01-01T01:59:30Z"
    book = pd.DataFrame(
        _book_rows(first) + _book_rows(second, bid_scale=1.0, ask_scale=3.0)
    )
    references = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([first, second], utc=True),
            "reference_price": [100_000.0, 100_000.0],
        }
    )
    features = add_banded_orderbook_features(_candles(), book, references)
    assert features["depth_imbalance_1pct"].iloc[0] == pytest.approx(0.5)
    assert features["depth_imbalance_1pct"].iloc[1] == pytest.approx(-0.5)
    assert features["notional_imbalance_5pct"].between(-1.0, 1.0).all()
    assert not any(
        column in features.columns for column in ("depth", "notional", "percentage")
    )
    assert len(features) == len(_candles())


def test_banded_book_adapter_fails_closed_on_missing_band() -> None:
    timestamp = "2024-01-01T00:59:30Z"
    book = pd.DataFrame(_book_rows(timestamp)[:-1])
    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([timestamp], utc=True),
            "reference_price": [100_000.0],
        }
    )
    with pytest.raises(ValueError, match="incomplete_percentage_bands"):
        add_banded_orderbook_features(_candles().iloc[:1], book, reference)


def test_banded_book_adapter_rejects_out_of_band_implied_price() -> None:
    timestamp = "2024-01-01T00:59:30Z"
    rows = _book_rows(timestamp)
    depth = rows[0]["depth"]
    assert isinstance(depth, (int, float))
    rows[0]["notional"] = float(depth) * 80_000.0
    book = pd.DataFrame(rows)
    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([timestamp], utc=True),
            "reference_price": [100_000.0],
        }
    )
    with pytest.raises(ValueError, match="implied_price_outside_declared_band"):
        add_banded_orderbook_features(_candles().iloc[:1], book, reference)


def test_bookdepth_manifest_checks_permission_and_both_checksums(
    tmp_path: Path,
) -> None:
    book_path = tmp_path / "book.csv"
    reference_path = tmp_path / "reference.csv"
    manifest_path = tmp_path / "manifest.json"
    book_path.write_text("timestamp,percentage,depth,notional\n", encoding="utf-8")
    reference_path.write_text("timestamp,reference_price\n", encoding="utf-8")
    manifest = {
        "schema": BOOK_MANIFEST_SCHEMA,
        "venue": "binance_usdm_futures",
        "symbol": "BTCUSDT",
        "source_url": "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/example.zip",
        "permission_status": "approved",
        "csv_sha256": hashlib.sha256(book_path.read_bytes()).hexdigest(),
        "reference_price_sha256": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        _validate_bookdepth_manifest(book_path, reference_path, manifest_path)[
            "permission_status"
        ]
        == "approved"
    )
    manifest["permission_status"] = "unverified"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="permission_not_approved"):
        _validate_bookdepth_manifest(book_path, reference_path, manifest_path)


def test_proxy_alignment_handles_nobitex_half_hour_phase_without_fabricating_volume() -> (
    None
):
    candles = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:30:00Z", "2024-01-01T01:30:00Z"], utc=True
            )
        }
    )
    # Source values remain actual Binance hourly quote-volume values. Only the
    # timestamp label is phase-aligned, and the report discloses that limitation.
    klines = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True
            ),
            "quote_volume": [200_000.0, 300_000.0],
        }
    )
    capacity, report = _align_proxy_volume(candles, klines)
    assert report["matched_candle_count"] == 2
    assert report["unmatched_candle_count"] == 0
    assert report["timestamp_phase_minutes"] == 30
    assert report["exact_same_window_turnover_claimed"] is False
    assert report["proxy_is_not_observed_nobitex_volume"] is True
    assert report["historical_nobitex_volume_replaced"] is False
    assert capacity.tolist() == pytest.approx([200.0, 300.0])


def test_proxy_alignment_rejects_mixed_nobitex_timestamp_phases() -> None:
    candles = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:30:00Z", "2024-01-01T01:00:00Z"], utc=True
            )
        }
    )
    klines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "quote_volume": [200_000.0],
        }
    )
    with pytest.raises(ValueError, match="timestamp_phase_not_uniform"):
        _align_proxy_volume(candles, klines)
