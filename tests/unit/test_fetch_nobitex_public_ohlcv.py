from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import time

import pytest

from scripts import fetch_nobitex_public_ohlcv as fetcher
from scripts.fetch_nobitex_public_ohlcv import (
    _fetch_candle_pages,
    fetch_historical_candles,
)


def test_empty_oldest_page_is_classified_for_bounded_window_fallback() -> None:
    start = 1_800_000_000
    calls: list[dict[str, str]] = []

    def no_data(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params)
        return {"s": "no_data"}

    with pytest.raises(ValueError, match="nobitex_historical_empty_initial_page"):
        _fetch_candle_pages(
            start=start,
            end=start + 10 * 60,
            symbol="BTCUSDT",
            interval="1m",
            request_json=no_data,
            sleep_fn=lambda _seconds: None,
        )

    assert len(calls) == 3


def test_empty_page_inside_requested_window_remains_fatal() -> None:
    start = 1_800_000_000
    end = start + 450 * 60
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params)
        if len(calls) > 1:
            return {"s": "no_data"}
        timestamps = list(range(start, start + 400 * 60, 60))
        return {
            "s": "ok",
            "t": timestamps,
            "o": [100.0] * len(timestamps),
            "h": [101.0] * len(timestamps),
            "l": [99.0] * len(timestamps),
            "c": [100.5] * len(timestamps),
            "v": [1.0] * len(timestamps),
        }

    with pytest.raises(ValueError, match="nobitex_historical_empty_page"):
        _fetch_candle_pages(
            start=start,
            end=end,
            symbol="BTCUSDT",
            interval="1m",
            request_json=provider,
            sleep_fn=lambda _seconds: None,
        )

    assert len(calls) == 4


def test_partial_page_recovers_missing_prefix_without_synthetic_rows() -> None:
    start = 1_800_000_000
    end = start + 399 * 60
    calls: list[dict[str, str]] = []

    def payload(stamps: list[int]) -> dict[str, Any]:
        return {
            "s": "ok",
            "t": stamps,
            "o": [100.0] * len(stamps),
            "h": [101.0] * len(stamps),
            "l": [99.0] * len(stamps),
            "c": [100.5] * len(stamps),
            "v": [1.0] * len(stamps),
        }

    def provider(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params.copy())
        if int(params["to"]) == end + 60:
            # Simulate a non-empty, contiguous response that omits the first
            # 28 expected candles. A retry of the same page remains partial.
            return payload(list(range(start + 28 * 60, end + 1, 60)))
        # The targeted request returns only the actual missing candles.
        return payload(list(range(start, start + 28 * 60, 60)))

    rows = _fetch_candle_pages(
        start=start,
        end=end,
        symbol="BTCUSDT",
        interval="1m",
        request_json=provider,
        sleep_fn=lambda _seconds: None,
    )

    assert [row["timestamp"] for row in rows] == list(range(start, end + 1, 60))
    assert len(calls) == 4
    assert calls[-1]["from"] == str(start - 60)
    assert calls[-1]["to"] == str(start + 28 * 60)


def test_request_public_json_retries_rate_limit_and_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [429, 200]
    delays: list[float] = []

    class FakeResponse:
        def __init__(self, status: int) -> None:
            self.status = status

        def getheader(self, name: str) -> str | None:
            if name == "Retry-After" and self.status == 429:
                return "3"
            return None

        def read(self) -> bytes:
            return b'{"s":"ok"}'

    class FakeConnection:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> FakeResponse:
            return FakeResponse(statuses.pop(0))

        def close(self) -> None:
            return None

    monkeypatch.setattr(fetcher, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(time, "sleep", delays.append)

    assert fetcher.request_public_json({"symbol": "BTCUSDT"}) == {"s": "ok"}
    assert delays == [3.0]
    assert statuses == []


def test_fetch_history_chunks_by_day_and_retries_only_the_incomplete_day() -> None:
    now = datetime(2026, 10, 23, 4, 18, tzinfo=timezone.utc)
    seconds = 60
    offset = 30
    now_epoch = int(now.timestamp())
    expected_end = ((now_epoch - seconds - offset) // seconds) * seconds + offset
    expected_start = expected_end - (2 * 24 * 60 - 1) * seconds
    second_day_start = expected_start + 24 * 60 * seconds
    calls: list[dict[str, str]] = []
    second_day_first_page_attempts = 0

    def make_payload(stamps: list[int]) -> dict[str, Any]:
        count = len(stamps)
        return {
            "s": "ok",
            "t": stamps,
            "o": [100.0] * count,
            "h": [101.0] * count,
            "l": [99.0] * count,
            "c": [100.5] * count,
            "v": [1.0] * count,
        }

    def provider(params: dict[str, str]) -> dict[str, Any]:
        nonlocal second_day_first_page_attempts
        calls.append(params.copy())
        request_from = int(params["from"])
        request_to = int(params["to"])
        if request_from == second_day_start - seconds:
            second_day_first_page_attempts += 1
            if second_day_first_page_attempts <= 3:
                return {"s": "no_data"}
        first = ((request_from - offset + seconds - 1) // seconds) * seconds + offset
        stamps = list(range(first, request_to + 1, seconds))
        return make_payload(stamps)

    rows = fetch_historical_candles(
        days=2,
        now=now,
        symbol="BTCUSDT",
        interval="1m",
        request_json=provider,
        sleep_fn=lambda _seconds: None,
    )

    assert len(rows) == 2 * 24 * 60
    assert rows[0]["timestamp"] == expected_start
    assert rows[-1]["timestamp"] == expected_end
    assert all(
        later["timestamp"] - earlier["timestamp"] == seconds
        for earlier, later in zip(rows, rows[1:])
    )
    # The first day is fetched once; only the second day's first page is retried.
    assert sum(int(call["from"]) == expected_start - seconds for call in calls) == 1
    assert second_day_first_page_attempts == 4
    assert any(
        int(call["from"]) == second_day_start - seconds
        and int(call["to"]) == second_day_start + (400 - 1) * seconds + seconds
        for call in calls
    )
