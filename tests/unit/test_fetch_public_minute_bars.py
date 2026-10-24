from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.fetch_public_minute_bars import (
    KlineRow,
    _gap_count,
    parse_kline,
    subtract_calendar_months,
)


def test_calendar_month_subtraction_handles_short_months() -> None:
    current = datetime(2026, 3, 31, 12, 0, tzinfo=UTC)
    assert subtract_calendar_months(current, 1) == datetime(
        2026, 2, 28, 12, 0, tzinfo=UTC
    )


@pytest.mark.parametrize("months", [0, -1, True])
def test_invalid_calendar_month_count_is_rejected(months: int) -> None:
    with pytest.raises(ValueError):
        subtract_calendar_months(datetime(2026, 3, 1, tzinfo=UTC), months)


def test_binance_row_is_parsed_and_checked() -> None:
    row = parse_kline(
        [
            0,
            "100",
            "102",
            "99",
            "101",
            "4.5",
            59_999,
            "450",
            12,
            "2.1",
            "210",
            "0",
        ]
    )
    expected = KlineRow(
        0, 59_999, 100.0, 102.0, 99.0, 101.0, 4.5, 450.0, 12, 2.1, 210.0
    )
    assert row == expected


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [0, "nan", "102", "99", "101", "4", 59_999, "450", 12, "2", "210"],
        [0, "100", "98", "99", "101", "4", 59_999, "450", 12, "2", "210"],
        [0, "100", "102", "99", "101", "-4", 59_999, "450", 12, "2", "210"],
    ],
)
def test_invalid_binance_rows_are_rejected(payload: list[object]) -> None:
    with pytest.raises(ValueError):
        parse_kline(payload)


def test_gap_count_reports_missing_intervals_without_filling_them() -> None:
    rows = [
        KlineRow(0, 59_999, 100, 101, 99, 100, 1, 100, 2, 0.5, 50),
        KlineRow(120_000, 179_999, 100, 101, 99, 100, 1, 100, 2, 0.5, 50),
        KlineRow(240_000, 299_999, 100, 101, 99, 100, 1, 100, 2, 0.5, 50),
    ]
    assert _gap_count(rows) == 2
