"""Fetch a gap-free 730-day Binance Spot BTCUSDT hourly OHLCV dataset.

Only the unauthenticated public klines endpoint is called. Missing candles are
rejected; the downloader never interpolates or fabricates OHLCV data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import ssl
import time
from datetime import UTC, datetime
from http.client import HTTPSConnection, HTTPException
from pathlib import Path
from typing import Any, Callable, TypedDict
from urllib.parse import urlencode

ENDPOINT_HOST = "data-api.binance.vision"
ENDPOINT_PATH = "/api/v3/klines"
PROVIDER = "Binance Spot public klines API"
INTERVAL = "1h"
HOUR_MS = 3_600_000
KLINES_LIMIT = 1_000
ALLOWED_DAYS = (30, 90, 180, 365, 730)
USER_AGENT = "AliX-Research/1.0 (public-market-data-only)"


class HourlyCandle(TypedDict):
    timestamp_ms: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("binance_hourly_integer_field_invalid")
    try:
        return int(value)
    except (ValueError, OverflowError):
        raise ValueError("binance_hourly_integer_field_invalid") from None


def _parse_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("binance_hourly_float_field_invalid")
    try:
        return float(value)
    except (ValueError, OverflowError):
        raise ValueError("binance_hourly_float_field_invalid") from None


def parse_kline(payload: object) -> HourlyCandle:
    """Parse one public Binance hourly candle with strict OHLCV validation."""
    if not isinstance(payload, list) or len(payload) < 6:
        raise ValueError("binance_hourly_kline_shape_invalid")

    timestamp_ms = _parse_integer(payload[0])
    opened, high, low, close, volume = (
        _parse_float(payload[index]) for index in range(1, 6)
    )
    prices = (opened, high, low, close)
    if timestamp_ms < 0 or timestamp_ms % HOUR_MS:
        raise ValueError("binance_hourly_timestamp_invalid")
    if any(not math.isfinite(value) or value <= 0 for value in prices):
        raise ValueError("binance_hourly_price_invalid")
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("binance_hourly_volume_invalid")
    if high < max(opened, low, close) or low > min(opened, high, close):
        raise ValueError("binance_hourly_ohlc_invariant_failed")

    date = (
        datetime.fromtimestamp(timestamp_ms / 1000, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return {
        "timestamp_ms": timestamp_ms,
        "date": date,
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _request_page(start_ms: int, end_ms: int) -> list[object]:
    params = urlencode(
        {
            "symbol": "BTCUSDT",
            "interval": INTERVAL,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": str(KLINES_LIMIT),
        }
    )
    connection = HTTPSConnection(
        ENDPOINT_HOST,
        timeout=20,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            f"{ENDPOINT_PATH}?{params}",
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        response = connection.getresponse()
        payload_bytes = response.read()
        if response.status != 200:
            raise ValueError("binance_hourly_http_status_invalid")
        payload: Any = json.loads(payload_bytes.decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("binance_hourly_response_not_list")
        return payload
    finally:
        connection.close()


def _request_page_with_retries(
    start_ms: int,
    end_ms: int,
    *,
    request_page: Callable[[int, int], list[object]],
    sleep_fn: Callable[[float], None],
) -> list[object]:
    for attempt in range(3):
        try:
            return request_page(start_ms, end_ms)
        except (HTTPException, TimeoutError, OSError, ValueError):
            if attempt == 2:
                raise RuntimeError("public_binance_hourly_request_failed") from None
            sleep_fn(2**attempt)
    raise RuntimeError("public_binance_hourly_request_failed")


def validate_historical_rows(rows: list[HourlyCandle], days: int) -> None:
    """Reject missing, duplicated, unordered, or incomplete hourly coverage."""
    expected_rows = days * 24
    if len(rows) != expected_rows or not rows:
        raise ValueError(
            f"binance_hourly_row_coverage_mismatch:expected={expected_rows}:"
            f"actual={len(rows)}"
        )
    prior: int | None = None
    for row in rows:
        stamp = row["timestamp_ms"]
        if prior is not None and stamp - prior != HOUR_MS:
            raise ValueError("binance_hourly_gap_or_order_violation")
        prior = stamp


def _validate_request_window(days: int, now: datetime) -> tuple[int, int]:
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("historical_days_out_of_policy")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("historical_cutoff_timezone_required")

    now_ms = int(now.astimezone(UTC).timestamp() * 1000)
    latest_closed_open_ms = (now_ms // HOUR_MS - 1) * HOUR_MS
    start_ms = latest_closed_open_ms - (days * 24 - 1) * HOUR_MS
    return start_ms, latest_closed_open_ms


def _append_candles(
    rows: list[HourlyCandle],
    candles: list[HourlyCandle],
    *,
    cursor_ms: int,
    end_ms: int,
) -> None:
    prior_count = len(rows)
    for candle in candles:
        stamp = candle["timestamp_ms"]
        if stamp < cursor_ms or stamp > end_ms:
            continue
        if rows and stamp <= rows[-1]["timestamp_ms"]:
            if stamp == rows[-1]["timestamp_ms"]:
                continue
            raise ValueError("binance_hourly_timestamp_order_invalid")
        rows.append(candle)

    if len(rows) == prior_count:
        raise ValueError("binance_hourly_page_did_not_advance")


def fetch_historical_candles(
    *,
    days: int,
    now: datetime,
    request_page: Callable[[int, int], list[object]] = _request_page,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[HourlyCandle], int]:
    start_ms, latest_closed_open_ms = _validate_request_window(days, now)
    cursor_ms = start_ms
    rows: list[HourlyCandle] = []
    page_count = 0

    while cursor_ms <= latest_closed_open_ms:
        payload = _request_page_with_retries(
            cursor_ms,
            latest_closed_open_ms,
            request_page=request_page,
            sleep_fn=sleep_fn,
        )
        page_count += 1
        candles = [parse_kline(item) for item in payload]
        _append_candles(
            rows, candles, cursor_ms=cursor_ms, end_ms=latest_closed_open_ms
        )
        cursor_ms = rows[-1]["timestamp_ms"] + HOUR_MS
        if cursor_ms <= latest_closed_open_ms:
            sleep_fn(0.1)

    validate_historical_rows(rows, days)
    if rows[0]["timestamp_ms"] != start_ms:
        raise ValueError("binance_hourly_start_timestamp_mismatch")
    if rows[-1]["timestamp_ms"] != latest_closed_open_ms:
        raise ValueError("binance_hourly_end_timestamp_mismatch")
    return rows, page_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts(
    rows: list[HourlyCandle],
    *,
    output_dir: Path,
    days: int,
    commit_sha: str,
    page_count: int,
) -> dict[str, Any]:
    validate_historical_rows(rows, days)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "binance-btcusdt-1h.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("date", "open", "high", "low", "close", "volume")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": "alix.public-hourly-market-data.v1",
        "provider": PROVIDER,
        "endpoint": f"https://{ENDPOINT_HOST}{ENDPOINT_PATH}",
        "venue": "Binance Spot",
        "symbol": "BTCUSDT",
        "interval": INTERVAL,
        "days": days,
        "start_utc": rows[0]["date"],
        "end_utc": rows[-1]["date"],
        "row_count": len(rows),
        "expected_rows": days * 24,
        "gap_count": 0,
        "public_market_data_pages": page_count,
        "csv_file": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "private_api_calls": 0,
        "orders_or_execution_api_used": False,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
        "commit_sha": commit_sha,
    }
    manifest_path = output_dir / "binance-btcusdt-1h.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, choices=ALLOWED_DAYS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()

    rows, page_count = fetch_historical_candles(days=args.days, now=datetime.now(UTC))
    manifest = write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=args.days,
        commit_sha=args.commit_sha,
        page_count=page_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
