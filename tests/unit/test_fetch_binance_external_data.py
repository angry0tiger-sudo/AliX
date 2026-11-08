from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from external_data.binance_futures import (
    BinanceFuturesSnapshot,
    FundingRatePoint,
    OpenInterestPoint,
)
from scripts import fetch_binance_external_data as fetcher

UTC = timezone.utc


def test_write_snapshot_emits_hashed_csvs_and_provenance_manifest(tmp_path) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    funding = tuple(
        FundingRatePoint(
            "BTCUSDT",
            start + timedelta(hours=8 * index),
            0.0001,
            50000.0,
            start + timedelta(hours=8 * index, minutes=5),
            "a" * 64,
        )
        for index in range(3)
    )
    oi = tuple(
        OpenInterestPoint(
            "BTCUSDT",
            "4h",
            start + timedelta(hours=4 * index),
            100.0 + index,
            5000000.0,
            start + timedelta(hours=4 * index, minutes=5),
            "b" * 64,
        )
        for index in range(3)
    )
    snapshot = BinanceFuturesSnapshot(
        "BTCUSDT",
        start,
        start + timedelta(days=1825),
        start + timedelta(days=1795),
        funding,
        oi,
        start + timedelta(days=1825),
        "4h",
    )
    manifest = fetcher._write_snapshot(
        tmp_path,
        snapshot,
        commit_sha="c" * 40,
        requested_days=1825,
    )
    funding_path = tmp_path / "binance-btcusdt-funding.csv"
    oi_path = tmp_path / "binance-btcusdt-open-interest.csv"
    manifest_path = tmp_path / "binance-btcusdt-manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["funding_csv_sha256"] == hashlib.sha256(funding_path.read_bytes()).hexdigest()
    assert persisted["open_interest_csv_sha256"] == hashlib.sha256(oi_path.read_bytes()).hexdigest()
    assert persisted["private_api_calls"] == 0
    assert persisted["credentials_used"] is False
    assert persisted["research_status"] == "BLOCKED_INSUFFICIENT_OI_HISTORY"
    assert pd.read_csv(funding_path).loc[0, "funding_rate"] == 0.0001
    assert manifest["open_interest_csv"] == oi_path.name
