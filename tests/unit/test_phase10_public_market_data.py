from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.fetch_nobitex_public_ohlcv import (
    ALLOWED_SYMBOLS,
    CandleRecord,
    fetch_historical_candles,
    interval_seconds,
    parse_udf_payload,
    write_artifacts,
)
from scripts.run_phase10_public_price_observation import observe_public_price

UTC = timezone.utc


def make_payload(start: int, count: int = 2, *, seconds: int = 3600) -> dict[str, Any]:
    stamps = [start + i * seconds for i in range(count)]
    return {
        "s": "ok",
        "t": stamps,
        "o": [100.0] * count,
        "h": [102.0] * count,
        "l": [99.0] * count,
        "c": [101.0] * count,
        "v": [5.0] * count,
    }


def fake_getter(now: datetime, *, gap_symbol: str | None = None):
    def getter(params: dict[str, str]) -> dict[str, Any]:
        seconds = int(params["resolution"]) * 60
        start, end = int(params["from"]), int(params["to"])
        offset = seconds // 2
        first = ((start - offset + seconds - 1) // seconds) * seconds + offset
        stamps = list(range(first, end + 1, seconds))
        values = [stamp for stamp in stamps]
        if gap_symbol == params["symbol"] and len(values) > 3:
            values.pop(2)
        count = len(values)
        return {
            "s": "ok",
            "t": values,
            "o": [100.0] * count,
            "h": [102.0] * count,
            "l": [99.0] * count,
            "c": [101.0] * count,
            "v": [5.0] * count,
        }

    return getter


def test_interval_policy_rejects_unsupported_resolution() -> None:
    assert interval_seconds("1m") == 60
    assert interval_seconds("1h") == 3600
    with pytest.raises(ValueError, match="interval_out_of_policy"):
        interval_seconds("5m")


def test_parser_handles_offset_boundaries_and_rejects_bad_arrays() -> None:
    rows = parse_udf_payload(
        make_payload(1800001800),
        requested_from=1800000000,
        requested_to=1800005400,
    )
    assert len(rows) == 2
    assert str(rows[0]["date"]).endswith("Z")
    payload = make_payload(1800000000)
    payload["c"] = [100.0]
    with pytest.raises(ValueError, match="array_length"):
        parse_udf_payload(
            payload,
            requested_from=1800000000,
            requested_to=1800003600,
        )


def test_parser_rejects_invalid_ohlc_and_out_of_bounds_timestamps() -> None:
    payload = make_payload(1800000000)
    payload["h"] = [98.0, 98.0]
    with pytest.raises(ValueError, match="ohlc_invariant"):
        parse_udf_payload(
            payload,
            requested_from=1800000000,
            requested_to=1800003600,
        )
    with pytest.raises(ValueError, match="out_of_bounds"):
        parse_udf_payload(
            make_payload(1800000000),
            requested_from=1800003600,
            requested_to=1800007200,
        )


def test_hourly_fetch_is_bounded_and_rejects_missing_candles() -> None:
    now = datetime(2026, 10, 23, 4, 18, tzinfo=UTC)
    getter = fake_getter(now)
    rows = fetch_historical_candles(
        days=30,
        now=now,
        interval="1h",
        request_json=getter,
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 30 * 24
    assert rows[-1]["timestamp"] + 3600 <= int(now.timestamp())
    assert rows[0]["timestamp"] % 3600 == 1800
    with pytest.raises(ValueError, match="historical_days"):
        fetch_historical_candles(
            days=10,
            now=now,
            interval="1h",
            request_json=getter,
            sleep_fn=lambda _: None,
        )

    with pytest.raises(ValueError, match="missing_candle_intervals=1"):
        fetch_historical_candles(
            days=1,
            now=now,
            interval="1h",
            request_json=fake_getter(now, gap_symbol="BTCUSDT"),
            sleep_fn=lambda _: None,
        )


@pytest.mark.parametrize("symbol", ALLOWED_SYMBOLS)
def test_btc_and_eth_minute_ingestion_share_manifest_hash_guarantees(
    tmp_path: Path, symbol: str
) -> None:
    now = datetime(2026, 10, 23, 4, 18, tzinfo=UTC)
    rows = fetch_historical_candles(
        days=1,
        now=now,
        symbol=symbol,
        interval="1m",
        request_json=fake_getter(now),
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 1440
    assert all(row["date"].endswith("Z") for row in rows)
    assert all(
        later["timestamp"] - earlier["timestamp"] == 60
        for earlier, later in zip(rows, rows[1:])
    )

    manifest = write_artifacts(
        rows,
        output_dir=tmp_path / symbol,
        days=1,
        retrieved_at=now,
        commit_sha="a" * 40,
        symbol=symbol,
        interval="1m",
    )
    output_dir = tmp_path / symbol
    csv_path = output_dir / f"nobitex-{symbol.lower()}-1m.csv"
    manifest_path = output_dir / f"nobitex-{symbol.lower()}-1m.manifest.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["symbol"] == symbol
    assert manifest["interval"] == "1m"
    assert manifest["start_utc"] == rows[0]["date"]
    assert manifest["end_utc"] == rows[-1]["date"]
    assert manifest["row_count"] == 1440
    assert manifest["missing_minute_intervals"] == 0
    assert manifest["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert loaded == manifest
    assert len(str(manifest["csv_sha256"])) == 64
    assert manifest["public_api_only"] is True
    assert manifest["credentials_used"] is False
    assert manifest["private_api_calls"] == 0
    assert manifest["exchange_calls"] == 0
    assert manifest["real_order_attempts"] == 0
    assert manifest["order_permission"] is False
    assert manifest["Champion"] is None
    assert manifest["trading_authorization"] == "NO-GO"


def test_eth_minute_gap_fails_closed_with_counted_reason_code() -> None:
    now = datetime(2026, 10, 23, 4, 18, tzinfo=UTC)
    with pytest.raises(
        ValueError, match="nobitex_historical_gap_detected:missing_minute_intervals=1"
    ):
        fetch_historical_candles(
            days=1,
            now=now,
            symbol="ETHUSDT",
            interval="1m",
            request_json=fake_getter(now, gap_symbol="ETHUSDT"),
            sleep_fn=lambda _: None,
        )


def test_manifest_preserves_hourly_compatibility_and_closed_authorization(
    tmp_path: Path,
) -> None:
    rows: list[CandleRecord] = [
        {
            "timestamp": 1792629000,
            "date": "2026-10-22T00:30:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 3.0,
        },
        {
            "timestamp": 1792632600,
            "date": "2026-10-22T01:30:00Z",
            "open": 100.5,
            "high": 101.5,
            "low": 100.0,
            "close": 101.0,
            "volume": 4.0,
        },
    ]
    manifest = write_artifacts(
        rows,
        output_dir=tmp_path,
        days=30,
        retrieved_at=datetime.now(UTC),
        commit_sha="a" * 40,
        interval="1h",
    )
    assert manifest["csv_sha256"]
    assert manifest["interval"] == "1h"
    assert manifest["missing_intervals"] == 0
    assert manifest["missing_minute_intervals"] is None
    assert manifest["credentials_used"] is False
    assert manifest["order_permission"] is False
    assert manifest["Champion"] is None
    assert manifest["trading_authorization"] == "NO-GO"
    assert (tmp_path / "nobitex-btcusdt-1h.csv").is_file()


def test_price_observer_always_abstains_and_is_bounded() -> None:
    elapsed = [0.0]

    def sleep_and_advance(seconds: float) -> None:
        elapsed[0] += seconds

    records = observe_public_price(
        duration_seconds=300,
        fetch_price=lambda: 100.0,
        monotonic=lambda: elapsed[0],
        clock=lambda: datetime(2026, 10, 23, 4, 0, tzinfo=UTC),
        sleep_fn=sleep_and_advance,
        interval_seconds=60,
    )
    assert records
    assert all(row["decision"] == "ABSTAIN" for row in records)
    assert all(row["orders_submitted"] == 0 for row in records)
    with pytest.raises(ValueError, match="observation_duration"):
        observe_public_price(duration_seconds=3601)
