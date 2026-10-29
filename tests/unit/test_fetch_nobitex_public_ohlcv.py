from __future__ import annotations

from typing import Any

import pytest

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
