from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from external_data.bybit_futures import (
    BybitFundingSnapshot,
    FundingRatePoint,
    _content_hash,
)
from scripts import fetch_external_funding as fetcher

UTC = timezone.utc
START = datetime(2021, 1, 1, tzinfo=UTC)
END = START + timedelta(days=1825) - timedelta(minutes=1)
COMMIT = "a" * 40
RETRIEVED = datetime(2026, 11, 8, tzinfo=UTC)


def _snapshot(symbol: str = "BTCUSDT") -> BybitFundingSnapshot:
    rates = tuple(
        FundingRatePoint(
            symbol=symbol,
            funding_time=START + timedelta(hours=8 * index),
            funding_rate=(index - 1) * 0.0001,
            mark_price=50000.0,
            available_at=START + timedelta(hours=8 * index, minutes=5),
            content_hash=_content_hash({"funding": symbol, "index": index}),
        )
        for index in range(3)
    )
    return BybitFundingSnapshot(symbol, START, END, rates, RETRIEVED)


class FakeConnector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime, datetime]] = []

    def fetch(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> BybitFundingSnapshot:
        self.calls.append((symbol, start, end))
        return _snapshot(symbol)


def test_fetch_request_requires_exact_1825_day_timezone_aware_window() -> None:
    start, end = fetcher._validate_fetch_request(
        ("BTCUSDT", "ETHUSDT"),
        1825,
        COMMIT,
        START.isoformat(),
        END.isoformat(),
    )
    assert start == pd.Timestamp(START)
    assert end == pd.Timestamp(END)

    with pytest.raises(ValueError, match="symbol_selection_invalid"):
        fetcher._validate_fetch_request(
            (), 1825, COMMIT, START.isoformat(), END.isoformat()
        )
    with pytest.raises(ValueError, match="symbol_not_allowed"):
        fetcher._validate_fetch_request(
            ("DOGEUSDT",), 1825, COMMIT, START.isoformat(), END.isoformat()
        )
    with pytest.raises(ValueError, match="must_be_1825"):
        fetcher._validate_fetch_request(
            ("BTCUSDT",), 30, COMMIT, START.isoformat(), END.isoformat()
        )
    with pytest.raises(ValueError, match="commit_sha_invalid"):
        fetcher._validate_fetch_request(
            ("BTCUSDT",), 1825, "bad", START.isoformat(), END.isoformat()
        )
    with pytest.raises(ValueError, match="window_invalid"):
        fetcher._validate_fetch_request(
            ("BTCUSDT",), 1825, COMMIT, "2021-01-01T00:00:00", END.isoformat()
        )
    with pytest.raises(ValueError, match="window_must_match"):
        fetcher._validate_fetch_request(
            ("BTCUSDT",),
            1825,
            COMMIT,
            START.isoformat(),
            (END - timedelta(minutes=1)).isoformat(),
        )


def test_write_snapshot_persists_funding_csv_manifest_and_hash_only(
    tmp_path: Path,
) -> None:
    manifest = fetcher._write_snapshot(
        tmp_path,
        _snapshot(),
        commit_sha=COMMIT,
        requested_days=1825,
    )
    funding_path = tmp_path / "bybit-btcusdt-funding.csv"
    manifest_path = tmp_path / "bybit-btcusdt-manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert funding_path.is_file()
    assert manifest == persisted
    assert manifest["funding_csv"] == funding_path.name
    assert (
        manifest["funding_csv_sha256"]
        == hashlib.sha256(funding_path.read_bytes()).hexdigest()
    )
    assert len(pd.read_csv(funding_path)) == 3
    assert "open_interest" not in manifest
    assert "open_interest_csv" not in manifest
    assert manifest["public_api_only"] is True
    assert manifest["credentials_used"] is False
    assert manifest["private_api_calls"] == 0
    assert manifest["exchange_order_calls"] == 0
    assert manifest["Champion"] is None
    assert manifest["order_permission"] is False
    assert manifest["trading_authorization"] == "NO-GO"


def test_fetch_public_history_uses_same_exact_window_for_both_assets(
    tmp_path: Path,
) -> None:
    connector = FakeConnector()
    manifests = fetcher.fetch_public_history(
        symbols=("BTCUSDT", "ETHUSDT"),
        requested_days=1825,
        commit_sha=COMMIT,
        start_utc=START.isoformat(),
        end_utc=END.isoformat(),
        output_dir=tmp_path,
        connector=connector,  # type: ignore[arg-type]
    )

    assert [call[0] for call in connector.calls] == ["BTCUSDT", "ETHUSDT"]
    assert all(call[1] == START and call[2] == END for call in connector.calls)
    assert len(manifests) == 2
    for symbol in ("BTCUSDT", "ETHUSDT"):
        stem = symbol.lower()
        assert (tmp_path / f"bybit-{stem}-funding.csv").is_file()
        assert (tmp_path / f"bybit-{stem}-manifest.json").is_file()
    assert all(
        manifest["research_status"] == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
        for manifest in manifests
    )
