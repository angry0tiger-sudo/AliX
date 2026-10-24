from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from scripts.run_stage_b_triple_barrier_orderbook import (
    BOOK_TIME,
    _liquidity_capacity,
    _validate_orderbook_manifest,
    add_relative_orderbook_features,
)


def _snapshot(timestamp: str, bid_size: float, ask_size: float) -> dict[str, object]:
    row: dict[str, object] = {BOOK_TIME: timestamp}
    for level in range(1, 21):
        row[f"bid_price_{level}"] = 100.0 - (level - 1) * 0.1
        row[f"ask_price_{level}"] = 100.2 + (level - 1) * 0.1
        row[f"bid_qty_{level}"] = bid_size
        row[f"ask_qty_{level}"] = ask_size
    return row


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="h"),
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [0.0, 0.0],
        }
    )


def test_orderbook_features_are_relative_and_point_in_time() -> None:
    book = pd.DataFrame(
        [
            _snapshot("2024-01-01T00:59:30Z", 3.0, 1.0),
            _snapshot("2024-01-01T01:59:30Z", 1.0, 3.0),
        ]
    )
    features = add_relative_orderbook_features(_candles(), book)
    assert features["imbalance_l5"].iloc[0] == pytest.approx(0.5)
    assert features["imbalance_l20"].iloc[0] == pytest.approx(0.5)
    assert features["imbalance_l5"].iloc[1] == pytest.approx(-0.5)
    assert features["spread_bps"].gt(0.0).all()
    assert not any("qty" in name or "price_" in name for name in features.columns)


def test_stale_orderbook_snapshot_fails_closed() -> None:
    book = pd.DataFrame(
        [
            _snapshot("2024-01-01T00:58:00Z", 3.0, 1.0),
            _snapshot("2024-01-01T01:58:00Z", 1.0, 3.0),
        ]
    )
    with pytest.raises(ValueError, match="missing_or_stale"):
        add_relative_orderbook_features(_candles(), book)


def test_crossed_orderbook_fails_closed() -> None:
    first = _snapshot("2024-01-01T00:59:30Z", 3.0, 1.0)
    first["ask_price_1"] = 99.0
    book = pd.DataFrame([first, _snapshot("2024-01-01T01:59:30Z", 1.0, 3.0)])
    with pytest.raises(ValueError, match="crossed_or_locked"):
        add_relative_orderbook_features(_candles(), book)


def test_zero_source_volume_grants_zero_capacity() -> None:
    report, capacity = _liquidity_capacity(_candles(), "unknown")
    assert report["status"] == "BLOCKED_ZERO_SOURCE_VOLUME"
    assert report["positive_volume_rows"] == 0
    assert report["rows_with_positive_order_capacity"] == 0
    assert capacity.tolist() == [0.0, 0.0]


def test_verified_quote_volume_is_capped_at_one_percent() -> None:
    candles = _candles()
    candles["volume"] = [100.0, 50.0]
    report, capacity = _liquidity_capacity(candles, "quote_asset")
    assert report["status"] == "READY_FOR_CAP_CHECK"
    assert capacity.tolist() == pytest.approx([1.0, 0.5])



def test_orderbook_manifest_requires_approved_provenance_and_matching_checksum(tmp_path) -> None:
    book_path = tmp_path / "book.csv"
    book_path.write_text("snapshot_timestamp_utc,bid_price_1\n", encoding="utf-8")
    manifest = {
        "schema": "alix.binance-orderbook-manifest.v1",
        "venue": "binance_usdm_futures",
        "symbol": "BTCUSDT",
        "source_url": "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/example.zip",
        "permission_status": "approved",
        "csv_sha256": hashlib.sha256(book_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "book-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _validate_orderbook_manifest(book_path, manifest_path)["permission_status"] == "approved"

    manifest["permission_status"] = "unverified"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="permission_not_approved"):
        _validate_orderbook_manifest(book_path, manifest_path)
