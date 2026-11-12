"""Fetch Binance Spot 15m/30m proxy candles for Nobitex HF research only.

This legacy module path is retained for workflow compatibility. Intraday OHLCV
comes from Binance public market-data REST only; Nobitex is still the intended
execution venue. No orders, credentials, resampling, or interpolation exist.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.client import HTTPSConnection, HTTPException
import json
import math
from pathlib import Path
import ssl
import time
from typing import Any, Callable, TypedDict
from urllib.parse import urlencode

from scripts.fetch_nobitex_public_ohlcv import (
    PREHACK_END_TIMESTAMP,
    PREHACK_START_TIMESTAMP,
)

BINANCE_HOST = "data-api.binance.vision"
BINANCE_KLINES_ENDPOINT = f"https://{BINANCE_HOST}/api/v3/klines"
BINANCE_INTERVAL_SECONDS = {"15m": 900, "30m": 1800}
BINANCE_MAX_CANDLES_PER_REQUEST = 1000
BINANCE_PAGE_SLEEP_SECONDS = 0.2
BINANCE_MAX_HTTP_RETRIES = 5
BINANCE_MAX_RETRY_AFTER_SECONDS = 300.0

PROXY_START_TIMESTAMP = PREHACK_START_TIMESTAMP
PROXY_END_TIMESTAMP = PREHACK_END_TIMESTAMP
INTRADAY_INTERVALS = tuple(BINANCE_INTERVAL_SECONDS)
INTRADAY_END_TIMESTAMP = PROXY_END_TIMESTAMP


class ProxyCandle(TypedDict):
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def interval_seconds(interval: str) -> int:
    """Return an allowed native proxy interval length."""
    if interval not in BINANCE_INTERVAL_SECONDS:
        raise ValueError("binance_proxy_interval_out_of_policy")
    return BINANCE_INTERVAL_SECONDS[interval]


def _validate_range(interval: str, start_timestamp: int, end_timestamp: int) -> int:
    seconds = interval_seconds(interval)
    for name, value in (
        ("start", start_timestamp),
        ("end", end_timestamp),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"binance_proxy_{name}_timestamp_invalid")
    if end_timestamp < start_timestamp:
        raise ValueError("binance_proxy_timestamp_range_reversed")
    if start_timestamp % seconds or end_timestamp % seconds:
        raise ValueError("binance_proxy_timestamp_range_not_interval_aligned")
    return seconds


def _retry_delay_seconds(attempt: int, retry_after: str | None) -> float:
    """Respect Binance Retry-After; refuse to retry beyond a safe wait window."""
    try:
        delay = float(retry_after) if retry_after is not None else float(2**attempt)
    except (TypeError, ValueError):
        delay = float(2**attempt)
    if not math.isfinite(delay) or delay < 0.0:
        delay = float(2**attempt)
    if delay > BINANCE_MAX_RETRY_AFTER_SECONDS:
        raise RuntimeError("binance_retry_after_exceeds_safe_wait")
    return delay


TRANSIENT_HTTP_STATUSES = frozenset({408, 418, 429, 500, 502, 503, 504})


def request_binance_public_json(
    params: dict[str, str],
    *,
    timeout_seconds: float = 20.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[Any]:
    """Perform an unauthenticated, verified-TLS GET against Binance's data-only host."""
    path = "/api/v3/klines?" + urlencode(params)
    last_failure = "unknown"
    for attempt in range(BINANCE_MAX_HTTP_RETRIES):
        connection = HTTPSConnection(
            BINANCE_HOST,
            timeout=timeout_seconds,
            context=ssl.create_default_context(),
        )
        retry_after: str | None = None
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-Binance-Proxy-OHLCV/1",
                },
            )
            response = connection.getresponse()
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise ValueError("binance_klines_response_not_list")
                return payload
            if response.status not in TRANSIENT_HTTP_STATUSES:
                raise RuntimeError(
                    f"binance_public_api_http_status_invalid:{response.status}"
                )
            retry_after = response.getheader("Retry-After")
            last_failure = f"http_{response.status}"
        except (
            HTTPException,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_failure = type(exc).__name__
        finally:
            connection.close()
        if attempt == BINANCE_MAX_HTTP_RETRIES - 1:
            raise RuntimeError(
                f"binance_public_api_request_failed_after_bounded_retries:{last_failure}"
            ) from None
        sleep_fn(_retry_delay_seconds(attempt, retry_after))
    raise RuntimeError("binance_public_api_request_failed")


def _numeric_kline_values(raw: list[Any]) -> tuple[float, float, float, float, float]:
    values: list[float] = []
    for item in raw[1:6]:
        try:
            value = float(item)
        except (TypeError, ValueError):
            raise ValueError("binance_kline_value_invalid") from None
        if not math.isfinite(value):
            raise ValueError("binance_kline_value_not_finite")
        values.append(value)
    opened, high, low, close, volume = values
    if min(opened, high, low, close) <= 0.0 or volume < 0.0:
        raise ValueError("binance_kline_price_or_volume_out_of_range")
    if high < max(opened, close) or low > min(opened, close) or high < low:
        raise ValueError("binance_kline_ohlc_invariant_failed")
    return opened, high, low, close, volume


def _parse_kline(
    raw: Any,
    *,
    interval: str,
    requested_start: int,
    requested_end: int,
) -> ProxyCandle:
    seconds = interval_seconds(interval)
    if not isinstance(raw, list) or len(raw) < 6:
        raise ValueError("binance_kline_row_invalid")
    raw_open_time = raw[0]
    if (
        isinstance(raw_open_time, bool)
        or not isinstance(raw_open_time, int)
        or raw_open_time % 1000 != 0
    ):
        raise ValueError("binance_kline_open_time_invalid")
    timestamp = raw_open_time // 1000
    if not requested_start <= timestamp <= requested_end:
        raise ValueError("binance_kline_open_time_out_of_requested_range")
    if timestamp % seconds:
        raise ValueError("binance_kline_open_time_off_interval_grid")
    opened, high, low, close, volume = _numeric_kline_values(raw)
    candle_date = (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return {
        "timestamp": timestamp,
        "date": candle_date,
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _expected_page_window(
    start_timestamp: int, end_timestamp: int, seconds: int
) -> tuple[int, int]:
    expected_end = min(
        end_timestamp,
        start_timestamp + (BINANCE_MAX_CANDLES_PER_REQUEST - 1) * seconds,
    )
    expected_count = ((expected_end - start_timestamp) // seconds) + 1
    return expected_end, expected_count


def _page_request_params(
    interval: str, start_timestamp: int, end_timestamp: int
) -> dict[str, str]:
    return {
        "symbol": "BTCUSDT",
        "interval": interval,
        "startTime": str(start_timestamp * 1000),
        "endTime": str(end_timestamp * 1000),
        "limit": str(BINANCE_MAX_CANDLES_PER_REQUEST),
    }


def _validate_page_payload(payload: Any, interval: str, start_timestamp: int) -> None:
    if not isinstance(payload, list):
        raise ValueError("binance_klines_response_not_list")
    if not payload:
        raise ValueError(f"binance_proxy_empty_page:{interval}:{start_timestamp}")
    if len(payload) > BINANCE_MAX_CANDLES_PER_REQUEST:
        raise ValueError("binance_proxy_page_exceeds_limit")


def _parse_page_rows(
    payload: list[Any],
    *,
    interval: str,
    start_timestamp: int,
    end_timestamp: int,
) -> list[ProxyCandle]:
    return [
        _parse_kline(
            raw,
            interval=interval,
            requested_start=start_timestamp,
            requested_end=end_timestamp,
        )
        for raw in payload
    ]


def _validate_page_row_count(
    rows: list[ProxyCandle], expected_count: int, interval: str
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"binance_proxy_page_row_count_mismatch:{interval}:"
            f"expected={expected_count}:actual={len(rows)}"
        )


def _validate_page_boundaries(
    stamps: list[int],
    *,
    start_timestamp: int,
    expected_end: int,
    interval: str,
) -> None:
    if stamps[0] != start_timestamp or stamps[-1] != expected_end:
        raise ValueError(
            f"binance_proxy_page_boundary_mismatch:{interval}:"
            f"expected_first={start_timestamp}:actual_first={stamps[0]}:"
            f"expected_last={expected_end}:actual_last={stamps[-1]}:"
            f"rows={len(stamps)}"
        )


def _validate_page_grid(stamps: list[int], seconds: int, interval: str) -> None:
    for previous, current in zip(stamps, stamps[1:], strict=False):
        if current - previous != seconds:
            raise ValueError(f"binance_proxy_page_gap_detected:{interval}")


def _fetch_page(
    *,
    interval: str,
    start_timestamp: int,
    end_timestamp: int,
    request_json: Callable[[dict[str, str]], Any],
) -> list[ProxyCandle]:
    seconds = interval_seconds(interval)
    expected_end, expected_count = _expected_page_window(
        start_timestamp, end_timestamp, seconds
    )
    params = _page_request_params(interval, start_timestamp, expected_end)
    payload = request_json(params)
    _validate_page_payload(payload, interval, start_timestamp)
    rows = _parse_page_rows(
        payload,
        interval=interval,
        start_timestamp=start_timestamp,
        end_timestamp=expected_end,
    )
    stamps = [row["timestamp"] for row in rows]
    _validate_page_row_count(rows, expected_count, interval)
    _validate_page_boundaries(
        stamps,
        start_timestamp=start_timestamp,
        expected_end=expected_end,
        interval=interval,
    )
    _validate_page_grid(stamps, seconds, interval)
    return rows


def _validate_proxy_exact_range(
    stamps: list[int],
    *,
    interval: str,
    start_timestamp: int,
    end_timestamp: int,
    seconds: int,
) -> int:
    if not stamps:
        raise ValueError(f"binance_proxy_candles_empty:{interval}")
    expected_count = ((end_timestamp - start_timestamp) // seconds) + 1
    if stamps[0] != start_timestamp:
        raise ValueError(f"binance_proxy_exact_range_coverage_mismatch:{interval}")
    if stamps[-1] != end_timestamp:
        raise ValueError(f"binance_proxy_exact_range_coverage_mismatch:{interval}")
    if len(stamps) != expected_count:
        raise ValueError(f"binance_proxy_exact_range_coverage_mismatch:{interval}")
    return expected_count


def _validate_proxy_timestamp_grid(
    stamps: list[int], seconds: int, interval: str
) -> None:
    for previous, current in zip(stamps, stamps[1:], strict=False):
        if current - previous != seconds:
            raise ValueError(f"binance_proxy_gap_detected:{interval}")


def _proxy_price_values_valid(row: ProxyCandle) -> bool:
    prices = (row["open"], row["high"], row["low"], row["close"])
    return min(prices) > 0.0 and row["volume"] >= 0.0


def _proxy_ohlc_relations_valid(row: ProxyCandle) -> bool:
    return (
        row["high"] >= max(row["open"], row["close"])
        and row["low"] <= min(row["open"], row["close"])
        and row["high"] >= row["low"]
    )


def _validate_proxy_ohlc(row: ProxyCandle, interval: str) -> None:
    if not _proxy_price_values_valid(row) or not _proxy_ohlc_relations_valid(row):
        raise ValueError(f"binance_proxy_ohlc_invariant_failed:{interval}")


def _validate_proxy_candle(row: ProxyCandle, interval: str) -> None:
    expected_date = (
        datetime.fromtimestamp(row["timestamp"], timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    if row["date"] != expected_date:
        raise ValueError(f"binance_proxy_date_mismatch:{interval}")
    values = (
        row["open"],
        row["high"],
        row["low"],
        row["close"],
        row["volume"],
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"binance_proxy_non_finite_value:{interval}")
    _validate_proxy_ohlc(row, interval)


def _validate_proxy_rows(
    rows: list[ProxyCandle],
    *,
    interval: str,
    start_timestamp: int,
    end_timestamp: int,
) -> int:
    seconds = interval_seconds(interval)
    stamps = [int(row["timestamp"]) for row in rows]
    expected_count = _validate_proxy_exact_range(
        stamps,
        interval=interval,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        seconds=seconds,
    )
    _validate_proxy_timestamp_grid(stamps, seconds, interval)
    for row in rows:
        _validate_proxy_candle(row, interval)
    return expected_count


def fetch_proxy_intraday(
    interval: str,
    *,
    start_timestamp: int = PROXY_START_TIMESTAMP,
    end_timestamp: int = PROXY_END_TIMESTAMP,
    request_json: Callable[[dict[str, str]], Any] = request_binance_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[ProxyCandle], int]:
    """Fetch complete, native Binance BTCUSDT candles in chronological 1000-row pages."""
    seconds = _validate_range(interval, start_timestamp, end_timestamp)
    rows: list[ProxyCandle] = []
    cursor = start_timestamp
    request_count = 0
    while cursor <= end_timestamp:
        page = _fetch_page(
            interval=interval,
            start_timestamp=cursor,
            end_timestamp=end_timestamp,
            request_json=request_json,
        )
        if rows and page[0]["timestamp"] != rows[-1]["timestamp"] + seconds:
            raise ValueError(f"binance_proxy_cross_page_gap:{interval}")
        rows.extend(page)
        request_count += 1
        cursor = page[-1]["timestamp"] + seconds
        if cursor <= end_timestamp:
            sleep_fn(BINANCE_PAGE_SLEEP_SECONDS)
    _validate_proxy_rows(
        rows,
        interval=interval,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    return rows, request_count


def _artifact_stem(interval: str) -> str:
    interval_seconds(interval)
    return f"PROXY_BINANCE_{interval.upper()}"


def write_proxy_artifacts(
    rows: list[ProxyCandle],
    *,
    interval: str,
    output_dir: Path,
    start_timestamp: int,
    end_timestamp: int,
    retrieved_at: datetime,
    commit_sha: str,
    public_api_requests: int,
) -> dict[str, Any]:
    expected_count = _validate_proxy_rows(
        rows,
        interval=interval,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("binance_proxy_retrieved_at_timezone_required")
    if (
        isinstance(public_api_requests, bool)
        or not isinstance(public_api_requests, int)
        or public_api_requests < 1
    ):
        raise ValueError("binance_proxy_public_api_request_count_invalid")
    if not isinstance(commit_sha, str) or len(commit_sha) != 40:
        raise ValueError("binance_proxy_commit_sha_invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(interval)
    csv_path = output_dir / f"{stem}.csv"
    temp_path = output_dir / f"{stem}.csv.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ("date", "open", "high", "low", "close", "volume")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
    temp_path.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    seconds = interval_seconds(interval)
    manifest = {
        "schema_version": "alix.binance.proxy-ohlcv.v1",
        "dataset_id": stem,
        "dataset_role": "RESEARCH_PROXY_ONLY",
        "provider": "Binance Spot public market-data REST API",
        "endpoint": BINANCE_KLINES_ENDPOINT,
        "venue": "BINANCE_SPOT",
        "market_type": "SPOT",
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "interval": interval,
        "native_interval": True,
        "dataset_window": "2021_11_TO_2025_06_17_PROXY",
        "start_utc": rows[0]["date"],
        "end_utc": rows[-1]["date"],
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "expected_candles": expected_count,
        "row_count": len(rows),
        "requested_start_timestamp": start_timestamp,
        "requested_end_timestamp": end_timestamp,
        "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
        "missing_intervals": 0,
        "public_api_only": True,
        "market_data_only_endpoint": True,
        "public_api_requests": public_api_requests,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "trading_authorization": "NO-GO",
        "source_policy": "BINANCE_PROXY_RESEARCH_ONLY",
        "execution_venue": "NOBITEX",
        "proxy_not_execution_truth": True,
        "proxy_note": (
            "Binance Spot OHLCV is for research only and does not represent "
            "Nobitex quotes, spreads, fills, or execution prices."
        ),
        "no_resampling": True,
        "no_interpolation": True,
        "not_nobitex_backfill": True,
        "bar_interval_seconds": seconds,
        "csv_sha256": digest,
        "commit_sha": commit_sha,
        "data_authority": "THIRD_PARTY_PUBLIC_MARKET_DATA_NOT_NOBITEX_EXECUTION_TRUTH",
        "promotion_status": "BLOCKED_NO_NOBITEX_VALIDATION",
        "economic_acceptance": "NO-GO",
    }
    (output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", choices=INTRADAY_INTERVALS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    retrieved_at = datetime.now(timezone.utc)
    rows, request_count = fetch_proxy_intraday(args.interval)
    manifest = write_proxy_artifacts(
        rows,
        interval=args.interval,
        output_dir=args.output_dir,
        start_timestamp=PROXY_START_TIMESTAMP,
        end_timestamp=PROXY_END_TIMESTAMP,
        retrieved_at=retrieved_at,
        commit_sha=args.commit_sha,
        public_api_requests=request_count,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
