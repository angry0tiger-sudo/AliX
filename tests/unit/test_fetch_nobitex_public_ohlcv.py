from __future__ import annotations

from typing import Any
import time

import pytest

from scripts import fetch_nobitex_public_ohlcv as fetcher
from scripts.fetch_nobitex_public_ohlcv import _fetch_candle_pages


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
