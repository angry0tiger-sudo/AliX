from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.fetch_public_hourly_bars import (
    HOUR_MS,
    HourlyCandle,
    parse_kline,
    validate_historical_rows,
)


def _row(index: int) -> HourlyCandle:
    timestamp_ms = index * HOUR_MS
    stamp = datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "timestamp_ms": timestamp_ms,
        "date": stamp,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10.0,
    }


def test_parse_kline_validates_hourly_ohlcv() -> None:
    row = parse_kline(
        [
            1_704_067_200_000,
            "100.0",
            "102.0",
            "99.0",
            "101.0",
            "10.0",
        ]
    )
    assert row["timestamp_ms"] == 1_704_067_200_000
    assert row["close"] == 101.0
    assert row["volume"] == 10.0


def test_parse_kline_rejects_invalid_ohlc() -> None:
    with pytest.raises(ValueError, match="binance_hourly_ohlc_invariant_failed"):
        parse_kline([0, "100", "99", "95", "101", "10"])


def test_historical_coverage_must_be_complete_and_contiguous() -> None:
    rows = [_row(index) for index in range(24)]
    validate_historical_rows(rows, days=1)

    rows_with_gap = [_row(index) for index in range(24)]
    rows_with_gap[10] = _row(11)
    with pytest.raises(ValueError, match="binance_hourly_gap_or_order_violation"):
        validate_historical_rows(rows_with_gap, days=1)


def test_historical_coverage_rejects_partial_day() -> None:
    with pytest.raises(ValueError, match="binance_hourly_row_coverage_mismatch"):
        validate_historical_rows([_row(index) for index in range(23)], days=1)
