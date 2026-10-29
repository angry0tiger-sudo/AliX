from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import pytest
from scripts import fetch_binance_public_ohlcv as fetcher


def klines(start: int, count: int) -> list[list[Any]]:
    return [
        [
            start + i * 60_000,
            "100",
            "101",
            "99",
            "100.5",
            "0" if i == 2 else "3.5",
            start + i * 60_000 + 59_999,
            "0",
            0,
            "0",
            "0",
            "0",
        ]
        for i in range(count)
    ]


def test_paginated_fetch_preserves_zero_volume_and_checks_continuity() -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - 60_000
    start = end - 1439 * 60_000
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> list[Any]:
        calls.append(params.copy())
        cursor = int(params["startTime"])
        count = min(1000, (end - cursor) // 60_000 + 1)
        return klines(cursor, count)
    rows = fetcher.fetch_historical_candles(
        days=1, now=now, request_json=provider, sleep_fn=lambda _seconds: None
    )
    assert len(rows) == 1440
    assert rows[0]["timestamp"] * 1000 == start
    assert rows[-1]["timestamp"] * 1000 == end
    assert rows[2]["volume"] == 0.0
    assert len(calls) == 2
    assert calls[1]["startTime"] == str(start + 1000 * 60_000)
    assert all(c["interval"] == "1m" and c["limit"] == "1000" for c in calls)


def test_incomplete_page_retries_then_fails_closed() -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    calls = 0

    def incomplete(params: dict[str, str]) -> list[Any]:
        nonlocal calls
        calls += 1
        return klines(int(params["startTime"]), 2)
    with pytest.raises(ValueError, match="binance_kline_page_count_mismatch"):
        fetcher.fetch_historical_candles(
            days=1, now=now, request_json=incomplete, sleep_fn=lambda _seconds: None
        )
    assert calls == 3


def test_manifest_hash_safety_fields_and_gap_rejection(tmp_path: Path) -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - 60_000
    start = end - 1439 * 60_000
    rows = [
        fetcher._parse_kline(row, start + i * 60_000)
        for i, row in enumerate(klines(start, 1440))
    ]
    manifest = fetcher.write_artifacts(
        rows,
        output_dir=tmp_path,
        days=1,
        retrieved_at=now,
        commit_sha="abc123",
        symbol="ETHUSDT",
    )
    csv_path = tmp_path / "binance-ethusdt-1m.csv"
    saved = json.loads((tmp_path / "binance-ethusdt-1m.manifest.json").read_text())
    assert manifest["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert saved["commit_sha"] == "abc123" and saved["missing_minute_intervals"] == 0
    assert saved["private_api_calls"] == 0 and saved["real_order_attempts"] == 0
    assert saved["Champion"] is None and saved["order_permission"] is False
    assert saved["trading_authorization"] == "NO-GO"
    with pytest.raises(ValueError, match="binance_historical_row_count_mismatch"):
        fetcher.write_artifacts(
            rows[:-1],
            output_dir=tmp_path,
            days=1,
            retrieved_at=now,
            commit_sha="abc123",
            symbol="ETHUSDT",
        )
