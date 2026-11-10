from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import time

import pytest

from scripts import fetch_nobitex_public_ohlcv as fetcher
from scripts.binance_nobitex_gap_backfill import (
    BINANCE_BACKFILL_END_TIMESTAMP,
    BINANCE_BACKFILL_SOURCE,
    BINANCE_BACKFILL_START_TIMESTAMP,
)
from scripts.fetch_nobitex_public_ohlcv import (
    CandleRecord,
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


def test_empty_outage_page_uses_approved_binance_backfill_and_post_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gap = list(
        range(
            BINANCE_BACKFILL_START_TIMESTAMP,
            BINANCE_BACKFILL_END_TIMESTAMP + 1,
            3600,
        )
    )
    before = BINANCE_BACKFILL_START_TIMESTAMP - 3600
    after = BINANCE_BACKFILL_END_TIMESTAMP + 3600
    expected = [*gap, after]

    def candle(timestamp: int) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }

    anchors = {before: candle(before), after: candle(after)}
    backfill_rows = [candle(timestamp) for timestamp in gap]
    provenance = {"source": BINANCE_BACKFILL_SOURCE, "row_count": 23}
    monkeypatch.setattr(
        fetcher,
        "_fetch_page_with_retries",
        lambda **_: (expected, {}, expected),
    )
    monkeypatch.setattr(
        fetcher,
        "_recover_missing_timestamp_ranges",
        lambda **_: {},
    )
    monkeypatch.setattr(
        fetcher,
        "_request_filtered_candle_window",
        lambda *args, **kwargs: anchors,
    )
    monkeypatch.setattr(
        fetcher,
        "fetch_binance_gap_backfill",
        lambda **_: (backfill_rows, provenance),
    )

    source_overrides: list[dict[str, object]] = []
    rows = _fetch_candle_pages(
        start=BINANCE_BACKFILL_START_TIMESTAMP,
        end=after,
        symbol="BTCUSDT",
        interval="1h",
        request_json=lambda _params: {},
        sleep_fn=lambda _seconds: None,
        source_overrides=source_overrides,
    )

    assert [row["timestamp"] for row in rows] == expected
    assert len(source_overrides) == 1
    assert source_overrides[0]["source"] == BINANCE_BACKFILL_SOURCE
    assert source_overrides[0]["row_count"] == 23


def test_split_daily_chunks_backfill_only_missing_outage_rows_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gap = list(
        range(
            BINANCE_BACKFILL_START_TIMESTAMP,
            BINANCE_BACKFILL_END_TIMESTAMP + 1,
            3600,
        )
    )
    before = BINANCE_BACKFILL_START_TIMESTAMP - 3600
    after = BINANCE_BACKFILL_END_TIMESTAMP + 3600

    def candle(timestamp: int, close: float = 100.0) -> CandleRecord:
        return {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }

    anchors = {before: candle(before), after: candle(after)}
    backfill_rows = [candle(timestamp, 150.0) for timestamp in gap]
    provenance = {
        "source": BINANCE_BACKFILL_SOURCE,
        "start_timestamp": BINANCE_BACKFILL_START_TIMESTAMP,
        "end_timestamp": BINANCE_BACKFILL_END_TIMESTAMP,
        "row_count": 23,
    }
    backfill_calls: list[list[int]] = []
    monkeypatch.setattr(
        fetcher,
        "_request_filtered_candle_window",
        lambda *args, **kwargs: anchors,
    )

    def backfill(**kwargs: Any) -> tuple[list[CandleRecord], dict[str, object]]:
        backfill_calls.append(kwargs["missing_timestamps"])
        return backfill_rows, provenance

    monkeypatch.setattr(fetcher, "fetch_binance_gap_backfill", backfill)
    source_overrides: list[dict[str, object]] = []
    backfill_cache: dict[int, CandleRecord] = {}

    first_expected = list(
        range(
            BINANCE_BACKFILL_START_TIMESTAMP - 23 * 3600,
            BINANCE_BACKFILL_START_TIMESTAMP + 1,
            3600,
        )
    )
    first_missing = [BINANCE_BACKFILL_START_TIMESTAMP]
    first_page = {
        stamp: candle(stamp, 200.0)
        for stamp in first_expected
        if stamp not in first_missing
    }
    first_unresolved = fetcher._apply_approved_binance_backfill(
        expected=first_expected,
        unresolved=first_missing,
        page_by_stamp=first_page,
        seconds=3600,
        symbol="BTCUSDT",
        request_json=lambda _params: {},
        source_overrides=source_overrides,
        backfill_cache=backfill_cache,
        sleep_fn=lambda _seconds: None,
    )

    second_expected = list(range(gap[1], gap[1] + 24 * 3600, 3600))
    second_missing = gap[1:]
    second_page = {
        stamp: candle(stamp, 300.0)
        for stamp in second_expected
        if stamp not in second_missing
    }
    second_unresolved = fetcher._apply_approved_binance_backfill(
        expected=second_expected,
        unresolved=second_missing,
        page_by_stamp=second_page,
        seconds=3600,
        symbol="BTCUSDT",
        request_json=lambda _params: {},
        source_overrides=source_overrides,
        backfill_cache=backfill_cache,
        sleep_fn=lambda _seconds: None,
    )

    assert first_unresolved == []
    assert second_unresolved == []
    assert first_page[BINANCE_BACKFILL_START_TIMESTAMP]["close"] == 150.0
    assert second_page[gap[1]]["close"] == 150.0
    assert first_page[before]["close"] == 200.0
    assert second_page[after]["close"] == 300.0
    assert len(backfill_calls) == 1
    assert backfill_calls == [gap]
    assert set(backfill_cache) == set(gap) | {before, after}
    assert source_overrides == [provenance]

    unrelated_missing = [BINANCE_BACKFILL_START_TIMESTAMP, after + 3600]
    rejected = fetcher._apply_approved_binance_backfill(
        expected=first_expected,
        unresolved=unrelated_missing,
        page_by_stamp={},
        seconds=3600,
        symbol="BTCUSDT",
        request_json=lambda _params: {},
        source_overrides=source_overrides,
        backfill_cache=backfill_cache,
        sleep_fn=lambda _seconds: None,
    )
    assert rejected == unrelated_missing
    assert len(backfill_calls) == 1
    assert source_overrides == [provenance]


def test_backfill_retries_incomplete_nobitex_boundary_anchor_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gap = list(
        range(
            BINANCE_BACKFILL_START_TIMESTAMP,
            BINANCE_BACKFILL_END_TIMESTAMP + 1,
            3600,
        )
    )
    before = BINANCE_BACKFILL_START_TIMESTAMP - 3600
    after = BINANCE_BACKFILL_END_TIMESTAMP + 3600

    def candle(timestamp: int) -> CandleRecord:
        return {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }

    anchor_attempts: list[int] = []
    anchor_rows = {before: candle(before), after: candle(after)}

    def request_anchors(*_args: Any, **kwargs: Any) -> dict[int, CandleRecord]:
        timestamp = kwargs["start"]
        anchor_attempts.append(timestamp)
        if timestamp == after and anchor_attempts.count(after) == 1:
            return {}
        return {timestamp: anchor_rows[timestamp]}

    backfill_rows = [candle(timestamp) for timestamp in gap]
    provenance = {"source": BINANCE_BACKFILL_SOURCE, "row_count": 23}
    backfill_calls: list[list[int]] = []

    def backfill(**kwargs: Any) -> tuple[list[CandleRecord], dict[str, object]]:
        backfill_calls.append(kwargs["missing_timestamps"])
        return backfill_rows, provenance

    monkeypatch.setattr(fetcher, "_request_filtered_candle_window", request_anchors)
    monkeypatch.setattr(fetcher, "fetch_binance_gap_backfill", backfill)
    delays: list[float] = []
    _, recovered, actual_provenance = fetcher._fetch_approved_binance_backfill_rows(
        approved_gap=gap,
        anchor_start=before,
        anchor_end=after,
        seconds=3600,
        symbol="BTCUSDT",
        request_json=lambda _params: {},
        sleep_fn=delays.append,
    )

    assert anchor_attempts == [before, after, after]
    assert delays == [1]
    assert backfill_calls == [gap]
    assert len(recovered) == 23
    assert actual_provenance == provenance


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
    expected_start = expected_end - (7 * 24 * 60 - 1) * seconds
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
        days=7,
        now=now,
        symbol="BTCUSDT",
        interval="1m",
        request_json=provider,
        sleep_fn=lambda _seconds: None,
    )

    assert len(rows) == 7 * 24 * 60
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
