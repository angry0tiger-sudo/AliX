"""Empirically inspect ETHUSDT 1m candle opens for 2023-03-24 UTC.

Uses only Binance's unauthenticated public market-data endpoint. It does not
create, amend, or submit orders and does not synthesize missing candles.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

HOST = "https://data-api.binance.vision"
HOSTNAME = "data-api.binance.vision"
ENDPOINT = "/api/v3/klines"
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
INTERVAL_MS = 60_000
PAGE_SIZE = 1000
PAGE_DELAY_SECONDS = 0.25
START = datetime(2023, 3, 24, 0, 0, tzinfo=timezone.utc)
END = datetime(2023, 3, 24, 23, 59, tzinfo=timezone.utc)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _fetch_page(cursor_ms: int, end_ms: int) -> list[list[Any]]:
    params = urlencode(
        {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": str(cursor_ms),
            "endTime": str(end_ms),
            "limit": str(PAGE_SIZE),
        }
    )
    request_path = f"{ENDPOINT}?{params}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with closing(
                http.client.HTTPSConnection(HOSTNAME, timeout=30)
            ) as connection:
                connection.request(
                    "GET",
                    request_path,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "AliX-gap-audit/1.0",
                    },
                )
                response = connection.getresponse()
                if response.status != http.client.OK:
                    raise ValueError(f"binance_http_status_invalid:{response.status}")
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError("binance_response_not_a_list")
            if not payload:
                raise ValueError(f"binance_empty_page_at_{cursor_ms}")
            if len(payload) > PAGE_SIZE:
                raise ValueError("binance_page_overflow")
            return payload
        except (
            http.client.HTTPException,
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"public_binance_fetch_failed:{last_error}") from None


def _validated_page_stamps(
    payload: list[list[Any]],
    cursor_ms: int,
    end_ms: int,
    previous_last_ms: int | None,
) -> list[int]:
    stamps: list[int] = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 6:
            raise ValueError("binance_kline_shape_invalid")
        stamp = item[0]
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise ValueError("binance_kline_timestamp_invalid")
        if stamp % INTERVAL_MS != 0 or not cursor_ms <= stamp <= end_ms:
            raise ValueError(f"binance_kline_timestamp_out_of_window:{stamp}")
        stamps.append(stamp)

    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("binance_page_timestamps_not_strictly_increasing")
    if stamps[0] < cursor_ms:
        raise ValueError("binance_page_regressed_before_cursor")
    if previous_last_ms is not None and stamps[0] <= previous_last_ms:
        raise ValueError("binance_duplicate_or_regressed_timestamp")
    return stamps


def _fetch_candle_stamps(start_ms: int, end_ms: int) -> tuple[list[int], int]:
    cursor_ms = start_ms
    stamps: list[int] = []
    page_count = 0
    while cursor_ms <= end_ms:
        payload = _fetch_page(cursor_ms, end_ms)
        page_stamps = _validated_page_stamps(
            payload,
            cursor_ms,
            end_ms,
            stamps[-1] if stamps else None,
        )
        stamps.extend(page_stamps)
        page_count += 1
        next_cursor = page_stamps[-1] + INTERVAL_MS
        if next_cursor <= cursor_ms:
            raise ValueError("binance_cursor_did_not_advance")
        cursor_ms = next_cursor
        if cursor_ms <= end_ms:
            time.sleep(PAGE_DELAY_SECONDS)
    return stamps, page_count


def _validate_day_boundaries(
    stamps: list[int], start_ms: int, end_ms: int
) -> None:
    if not stamps or stamps[0] != start_ms or stamps[-1] != end_ms:
        first = stamps[0] if stamps else None
        last = stamps[-1] if stamps else None
        raise ValueError(
            f"binance_day_boundary_mismatch:first={first}:last={last}"
        )


def _iso_utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _gap_ranges_from_stamps(stamps: list[int]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for previous, current in zip(stamps, stamps[1:]):
        delta = current - previous
        if delta % INTERVAL_MS:
            raise ValueError("binance_non_minute_gap_detected")
        missing = delta // INTERVAL_MS - 1
        if missing > 0:
            gaps.append(
                {
                    "start_utc": _iso_utc(previous + INTERVAL_MS),
                    "end_utc": _iso_utc(current - INTERVAL_MS),
                    "missing_candles": int(missing),
                }
            )
    return gaps


def _validate_gap_count(
    expected_count: int, observed_count: int, gaps: list[dict[str, Any]]
) -> int:
    missing_count = expected_count - observed_count
    observed_gap_count = sum(int(gap["missing_candles"]) for gap in gaps)
    if missing_count < 0 or missing_count != observed_gap_count:
        raise ValueError(
            f"binance_gap_count_mismatch:expected_missing={missing_count}:"
            f"observed_missing={observed_gap_count}"
        )
    return missing_count


def _build_report(
    stamps: list[int],
    page_count: int,
    expected_count: int,
    missing_count: int,
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "alix.v4ai.ethusdt-gap-discovery.v1",
        "provider": "Binance Spot public klines API",
        "host": HOST,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "window_start_utc": START.isoformat().replace("+00:00", "Z"),
        "window_end_utc": END.isoformat().replace("+00:00", "Z"),
        "expected_candle_opens": expected_count,
        "observed_candle_opens": len(stamps),
        "missing_candle_opens": missing_count,
        "gap_ranges": gaps,
        "first_observed_open_utc": _iso_utc(stamps[0]),
        "last_observed_open_utc": _iso_utc(stamps[-1]),
        "page_count": page_count,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def discover() -> dict[str, Any]:
    start_ms = _epoch_ms(START)
    end_ms = _epoch_ms(END)
    expected_count = 24 * 60
    stamps, page_count = _fetch_candle_stamps(start_ms, end_ms)
    _validate_day_boundaries(stamps, start_ms, end_ms)
    gaps = _gap_ranges_from_stamps(stamps)
    missing_count = _validate_gap_count(expected_count, len(stamps), gaps)
    return _build_report(stamps, page_count, expected_count, missing_count, gaps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("alix-v4ai-ethusdt-gap-discovery.json"),
    )
    args = parser.parse_args()
    report = discover()
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
