"""Fetch bounded public Nobitex OHLCV candles with strict gap and hash manifests.

Only GET /market/udf/history on the fixed public host is used. No credentials,
private endpoints, order calls, interpolation, resampling, or synthetic fill exist.
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
from typing import Any, Callable, TypedDict, cast
from urllib.parse import urlencode

ENDPOINT = "https://apiv2.nobitex.ir/market/udf/history"
SCHEMA_VERSION = "alix.nobitex.public-ohlcv.v3"
ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVAL_SECONDS = {"1m": 60, "1h": 3600}
ALLOWED_DAYS = (1, 7, 30, 90, 180, 365)
MAX_CANDLES_PER_REQUEST = 400


class CandleRecord(TypedDict):
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def interval_seconds(interval: str) -> int:
    """Return the fixed interval length, rejecting unsupported resolutions."""
    if interval not in INTERVAL_SECONDS:
        raise ValueError("nobitex_interval_out_of_policy")
    return INTERVAL_SECONDS[interval]


def _validate_symbol(symbol: str) -> None:
    if symbol not in ALLOWED_SYMBOLS:
        raise ValueError("nobitex_symbol_out_of_policy")


TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_HTTP_RETRIES = 5
MAX_RETRY_DELAY_SECONDS = 30.0


def _retry_delay_seconds(attempt: int, retry_after: str | None = None) -> float:
    """Use Retry-After when numeric, otherwise bounded exponential backoff."""
    try:
        delay = float(retry_after) if retry_after is not None else float(2**attempt)
    except (TypeError, ValueError):
        delay = float(2**attempt)
    return min(max(delay, 0.0), MAX_RETRY_DELAY_SECONDS)


def request_public_json(
    params: dict[str, str], *, timeout_seconds: float = 15.0
) -> dict[str, Any]:
    """Perform a verified-TLS public GET with bounded, status-aware retries."""
    path = "/market/udf/history?" + urlencode(params)
    last_failure = "unknown"
    for attempt in range(MAX_HTTP_RETRIES):
        connection = HTTPSConnection(
            "apiv2.nobitex.ir",
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
                    "User-Agent": "AliX-Public-OHLCV/2",
                },
            )
            response = connection.getresponse()
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("public_api_payload_not_object")
                return payload
            if response.status not in TRANSIENT_HTTP_STATUSES:
                raise ValueError("public_api_http_status_invalid")
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
        if attempt == MAX_HTTP_RETRIES - 1:
            raise RuntimeError(
                f"public_api_request_failed_after_bounded_retries:{last_failure}"
            ) from None
        time.sleep(_retry_delay_seconds(attempt, retry_after))
    raise RuntimeError("public_api_request_failed")


def _validated_udf_arrays(
    payload: dict[str, Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]]:
    if payload.get("s") == "no_data":
        return [], [], [], [], [], []
    if payload.get("s") != "ok":
        raise ValueError("nobitex_public_api_status_not_ok")
    names = ("t", "o", "h", "l", "c", "v")
    arrays = [payload.get(name) for name in names]
    if any(not isinstance(value, list) for value in arrays):
        raise ValueError("nobitex_ohlcv_arrays_missing")
    lengths = {len(value) for value in arrays if isinstance(value, list)}
    if len(lengths) != 1:
        raise ValueError("nobitex_ohlcv_array_length_mismatch")
    return tuple(cast(list[Any], value) for value in arrays)  # type: ignore[return-value]


def _parse_numeric_values(
    raw_values: tuple[Any, ...],
) -> tuple[float, float, float, float, float]:
    numbers: list[float] = []
    for raw in raw_values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError("nobitex_candle_value_invalid") from None
        if not math.isfinite(value):
            raise ValueError("nobitex_candle_value_not_finite")
        numbers.append(value)
    opened, high, low, close, volume = numbers
    if min(opened, high, low, close) <= 0 or volume < 0:
        raise ValueError("nobitex_candle_price_or_volume_out_of_range")
    if high < max(opened, close) or low > min(opened, close) or high < low:
        raise ValueError("nobitex_candle_ohlc_invariant_failed")
    return opened, high, low, close, volume


def _parse_candle(
    raw_timestamp: Any,
    raw_values: tuple[Any, ...],
    *,
    requested_from: int,
    requested_to: int,
    prior_timestamp: int | None,
) -> CandleRecord:
    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, int):
        raise ValueError("nobitex_candle_timestamp_invalid")
    stamp = raw_timestamp
    if stamp < requested_from or stamp > requested_to:
        raise ValueError("nobitex_candle_timestamp_out_of_bounds")
    if prior_timestamp is not None and stamp <= prior_timestamp:
        raise ValueError("nobitex_candle_timestamps_not_increasing")
    opened, high, low, close, volume = _parse_numeric_values(raw_values)
    candle_date = (
        datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z")
    )
    return {
        "timestamp": stamp,
        "date": candle_date,
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def parse_udf_payload(
    payload: dict[str, Any], *, requested_from: int, requested_to: int
) -> list[CandleRecord]:
    arrays = _validated_udf_arrays(payload)
    timestamps, opens, highs, lows, closes, volumes = arrays
    if not timestamps:
        return []
    rows: list[CandleRecord] = []
    prior: int | None = None
    for index, raw_timestamp in enumerate(timestamps):
        values = (
            opens[index],
            highs[index],
            lows[index],
            closes[index],
            volumes[index],
        )
        candle = _parse_candle(
            raw_timestamp,
            values,
            requested_from=requested_from,
            requested_to=requested_to,
            prior_timestamp=prior,
        )
        rows.append(candle)
        prior = candle["timestamp"]
    return rows


def _missing_intervals(stamps: list[int], seconds: int) -> int:
    """Count missing expected samples and reject non-monotonic/off-grid timestamps."""
    missing = 0
    previous: int | None = None
    for stamp in stamps:
        if previous is not None:
            delta = stamp - previous
            if delta <= 0:
                if delta == 0:
                    raise ValueError("nobitex_historical_duplicate_candle")
                raise ValueError("nobitex_historical_timestamps_not_monotonic")
            if delta % seconds != 0:
                raise ValueError("nobitex_historical_timestamp_off_interval_grid")
            missing += max(0, delta // seconds - 1)
        previous = stamp
    return missing


def _request_filtered_candle_window(
    params: dict[str, str],
    *,
    start: int,
    end: int,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
) -> dict[int, CandleRecord]:
    page = parse_udf_payload(
        request_json(params),
        requested_from=int(params["from"]),
        requested_to=int(params["to"]),
    )
    return {row["timestamp"]: row for row in page if start <= row["timestamp"] <= end}


def _candle_window_params(
    *, symbol: str, seconds: int, start: int, end: int
) -> dict[str, str]:
    # TradingView UDF treats `to` as an exclusive right boundary.
    # Keep the public API's logical window inclusive by advancing that bound
    # one second; the next hourly candle remains outside the requested window.
    return {
        "symbol": symbol,
        "resolution": str(seconds // 60),
        "from": str(start),
        "to": str(end + 1),
    }


def _fetch_page_with_retries(
    *,
    start: int,
    end: int,
    seconds: int,
    symbol: str,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> tuple[list[int], dict[int, CandleRecord], list[int]]:
    expected = list(range(start, end + 1, seconds))
    params = _candle_window_params(symbol=symbol, seconds=seconds, start=start, end=end)
    rows_by_stamp: dict[int, CandleRecord] = {}
    missing = expected
    for attempt in range(3):
        rows_by_stamp = _request_filtered_candle_window(
            params, start=start, end=end, request_json=request_json
        )
        missing = [stamp for stamp in expected if stamp not in rows_by_stamp]
        if not missing:
            break
        if attempt < 2:
            sleep_fn(2**attempt)
    return expected, rows_by_stamp, missing


def _validate_candle_page(
    *,
    expected: list[int],
    page_by_stamp: dict[int, CandleRecord],
    unresolved: list[int],
    is_initial_page: bool,
    page_start: int,
    stop: int,
    interval: str,
) -> list[CandleRecord]:
    if not page_by_stamp:
        error = (
            "nobitex_historical_empty_initial_page"
            if is_initial_page
            else "nobitex_historical_empty_page"
        )
        raise ValueError(f"{error}:{page_start}:{stop}")
    if unresolved:
        kind = (
            "missing_minute_intervals"
            if interval == "1m"
            else "missing_candle_intervals"
        )
        raise ValueError(
            f"nobitex_historical_gap_detected:{kind}={len(unresolved)}:"
            f"start={page_start}:stop={stop}:"
            f"first_missing={unresolved[0]}:last_missing={unresolved[-1]}"
        )
    return [page_by_stamp[stamp] for stamp in expected]


def _fetch_one_candle_page(
    *,
    start: int,
    end: int,
    symbol: str,
    interval: str,
    seconds: int,
    is_initial_page: bool,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> list[CandleRecord]:
    expected, page_by_stamp, missing = _fetch_page_with_retries(
        start=start,
        end=end,
        seconds=seconds,
        symbol=symbol,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    return _validate_candle_page(
        expected=expected,
        page_by_stamp=page_by_stamp,
        unresolved=missing,
        is_initial_page=is_initial_page,
        page_start=start,
        stop=end,
        interval=interval,
    )


def _fetch_candle_pages(
    *,
    start: int,
    end: int,
    symbol: str,
    interval: str,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> list[CandleRecord]:
    seconds = interval_seconds(interval)
    cursor = start
    rows: list[CandleRecord] = []
    while cursor <= end:
        stop = min(cursor + (MAX_CANDLES_PER_REQUEST - 1) * seconds, end)
        rows.extend(
            _fetch_one_candle_page(
                start=cursor,
                end=stop,
                symbol=symbol,
                interval=interval,
                seconds=seconds,
                is_initial_page=cursor == start,
                request_json=request_json,
                sleep_fn=sleep_fn,
            )
        )
        cursor = stop + seconds
        if cursor <= end:
            sleep_fn(1.1)
    return rows


def _validate_historical_rows(
    rows: list[CandleRecord],
    days: int,
    *,
    interval: str = "1h",
    expected_start: int | None = None,
    expected_end: int | None = None,
) -> None:
    if not rows:
        raise ValueError("nobitex_historical_candles_empty")
    seconds = interval_seconds(interval)
    stamps = [row["timestamp"] for row in rows]
    missing = _missing_intervals(stamps, seconds)
    if missing:
        if interval == "1m":
            raise ValueError(
                f"nobitex_historical_gap_detected:missing_minute_intervals={missing}"
            )
        raise ValueError(
            f"nobitex_historical_gap_detected:missing_candle_intervals={missing}"
        )
    if expected_start is not None and stamps[0] != expected_start:
        raise ValueError(
            "nobitex_historical_start_timestamp_mismatch:"
            f"expected={expected_start}:actual={stamps[0]}"
        )
    if expected_end is not None and stamps[-1] != expected_end:
        raise ValueError(
            "nobitex_historical_end_timestamp_mismatch:"
            f"expected={expected_end}:actual={stamps[-1]}"
        )
    expected_rows = days * 24 * 3600 // seconds
    if len(stamps) != expected_rows:
        raise ValueError(
            "nobitex_historical_row_coverage_mismatch:"
            f"expected={expected_rows}:actual={len(stamps)}:"
            f"first={stamps[0]}:last={stamps[-1]}"
        )


DAILY_CHUNK_RETRIES = 3
DAILY_CHUNK_RETRYABLE_PREFIXES = (
    "nobitex_historical_empty_initial_page:",
    "nobitex_historical_empty_page:",
    "nobitex_historical_gap_detected:",
    "nobitex_historical_row_coverage_mismatch:",
    "nobitex_historical_start_timestamp_mismatch:",
    "nobitex_historical_end_timestamp_mismatch:",
)


def _is_retryable_daily_chunk_error(error: ValueError) -> bool:
    return str(error).startswith(DAILY_CHUNK_RETRYABLE_PREFIXES)


def _fetch_one_daily_chunk(
    *,
    start: int,
    end: int,
    symbol: str,
    interval: str,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> list[CandleRecord]:
    last_error: ValueError | None = None
    for attempt in range(DAILY_CHUNK_RETRIES):
        try:
            rows = _fetch_candle_pages(
                start=start,
                end=end,
                symbol=symbol,
                interval=interval,
                request_json=request_json,
                sleep_fn=sleep_fn,
            )
            _validate_historical_rows(
                rows, 1, interval=interval, expected_start=start, expected_end=end
            )
            return rows
        except ValueError as exc:
            if not _is_retryable_daily_chunk_error(exc):
                raise
            last_error = exc
            if attempt < DAILY_CHUNK_RETRIES - 1:
                sleep_fn(2**attempt)
    assert last_error is not None
    raise ValueError(
        f"{last_error}:daily_chunk_start={start}:daily_chunk_end={end}:"
        f"attempts={DAILY_CHUNK_RETRIES}"
    ) from None


def _fetch_daily_candle_chunks(
    *,
    start: int,
    end: int,
    symbol: str,
    interval: str,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> list[CandleRecord]:
    seconds = interval_seconds(interval)
    per_day = 24 * 3600 // seconds
    cursor = start
    rows: list[CandleRecord] = []
    while cursor <= end:
        chunk_end = min(cursor + (per_day - 1) * seconds, end)
        rows.extend(
            _fetch_one_daily_chunk(
                start=cursor,
                end=chunk_end,
                symbol=symbol,
                interval=interval,
                request_json=request_json,
                sleep_fn=sleep_fn,
            )
        )
        cursor = chunk_end + seconds
        if cursor <= end:
            sleep_fn(1.1)
    return rows


def _latest_closed_candle_timestamp(
    *,
    now: datetime,
    symbol: str,
    interval: str,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Probe the provider candle grid and select the latest fully closed candle."""
    seconds = interval_seconds(interval)
    now_epoch = int(now.astimezone(timezone.utc).timestamp())
    requested_from = now_epoch - MAX_CANDLES_PER_REQUEST * seconds
    params = {
        "symbol": symbol,
        "resolution": str(seconds // 60),
        "from": str(requested_from),
        "to": str(now_epoch),
    }
    for attempt in range(3):
        payload = request_json(params)
        recent_rows = parse_udf_payload(
            payload, requested_from=requested_from, requested_to=now_epoch
        )
        closed_stamps = [
            row["timestamp"]
            for row in recent_rows
            if row["timestamp"] + seconds <= now_epoch
        ]
        if closed_stamps:
            return closed_stamps[-1]
        if attempt < 2:
            sleep_fn(2**attempt)
    raise ValueError("nobitex_historical_no_closed_candle")


PREHACK_START_TIMESTAMP = 1_635_712_200
PREHACK_END_TIMESTAMP = 1_750_192_200
PREHACK_INTERVAL_SECONDS = 3600
PREHACK_REQUESTED_DAYS = (PREHACK_END_TIMESTAMP - PREHACK_START_TIMESTAMP) // 86400
PREHACK_EXPECTED_CANDLES = (
    (PREHACK_END_TIMESTAMP - PREHACK_START_TIMESTAMP) // PREHACK_INTERVAL_SECONDS
) + 1


def _validate_timestamp_range(start_timestamp: int, end_timestamp: int) -> None:
    for name, value in (
        ("start_timestamp", start_timestamp),
        ("end_timestamp", end_timestamp),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"nobitex_{name}_invalid")
    if end_timestamp < start_timestamp:
        raise ValueError("nobitex_timestamp_range_reversed")


def _validate_exact_candle_coverage(
    rows: list[CandleRecord],
    *,
    start_timestamp: int,
    end_timestamp: int,
    seconds: int,
) -> None:
    stamps = [int(row["timestamp"]) for row in rows]
    expected_count = (end_timestamp - start_timestamp) // seconds + 1
    if not stamps:
        raise ValueError("nobitex_exact_range_coverage_mismatch")
    if stamps[0] != start_timestamp or stamps[-1] != end_timestamp:
        raise ValueError("nobitex_exact_range_coverage_mismatch")
    if len(stamps) != expected_count:
        raise ValueError("nobitex_exact_range_coverage_mismatch")
    if _missing_intervals(stamps, seconds) != 0:
        raise ValueError("nobitex_exact_range_coverage_mismatch")


def fetch_candles_between(
    *,
    start_timestamp: int,
    end_timestamp: int,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    request_json: Callable[[dict[str, str]], dict[str, Any]] = request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[CandleRecord]:
    _validate_timestamp_range(start_timestamp, end_timestamp)
    _validate_symbol(symbol)
    seconds = interval_seconds(interval)
    if (end_timestamp - start_timestamp) % seconds:
        raise ValueError("nobitex_timestamp_range_not_interval_aligned")
    rows = _fetch_candle_pages(
        start=start_timestamp,
        end=end_timestamp,
        symbol=symbol,
        interval=interval,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    _validate_exact_candle_coverage(
        rows,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        seconds=seconds,
    )
    return rows


def fetch_prehack_candles(
    *,
    request_json: Callable[[dict[str, str]], dict[str, Any]] = request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[CandleRecord]:
    rows = fetch_candles_between(
        start_timestamp=PREHACK_START_TIMESTAMP,
        end_timestamp=PREHACK_END_TIMESTAMP,
        symbol="BTCUSDT",
        interval="1h",
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    if len(rows) != PREHACK_EXPECTED_CANDLES:
        raise ValueError("pre_hack_nobitex_row_count_mismatch")
    return rows


def fetch_historical_candles(
    *,
    days: int,
    now: datetime,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    request_json: Callable[[dict[str, str]], dict[str, Any]] = request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[CandleRecord]:
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("historical_days_out_of_policy")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("historical_cutoff_timezone_required")
    _validate_symbol(symbol)
    seconds = interval_seconds(interval)
    end = _latest_closed_candle_timestamp(
        now=now,
        symbol=symbol,
        interval=interval,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    count = days * 24 * 3600 // seconds
    start = end - (count - 1) * seconds
    return _fetch_daily_candle_chunks(
        start=start,
        end=end,
        symbol=symbol,
        interval=interval,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )


def _artifact_stem(symbol: str, interval: str) -> str:
    _validate_symbol(symbol)
    interval_seconds(interval)
    return f"nobitex-{symbol.lower()}-{interval}"


def _source_manifest_fields() -> dict[str, object]:
    return {
        "gap_policy": "fail-closed on unresolved native Nobitex gaps",
        "source_policy": "NOBITEX_ONLY_NO_BACKFILL",
        "data_sources": ["Nobitex public UDF history API"],
    }


def write_artifacts(
    rows: list[CandleRecord],
    *,
    output_dir: Path,
    days: int,
    retrieved_at: datetime,
    commit_sha: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> dict[str, object]:
    if not rows:
        raise ValueError("nobitex_historical_candles_empty")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at_timezone_required")
    _validate_symbol(symbol)
    seconds = interval_seconds(interval)
    stamps = [int(row["timestamp"]) for row in rows]
    missing = _missing_intervals(stamps, seconds)
    if missing:
        raise ValueError(f"nobitex_historical_gap_detected:missing_intervals={missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(symbol, interval)
    csv_path = output_dir / f"{stem}.csv"
    temp_path = output_dir / f"{stem}.csv.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as stream:
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
    temp_path.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provider": "Nobitex public UDF history API",
        "endpoint": "/market/udf/history",
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "interval": interval,
        "start_utc": rows[0]["date"],
        "end_utc": rows[-1]["date"],
        "start_timestamp": stamps[0],
        "end_timestamp": stamps[-1],
        "expected_candles": len(rows),
        "row_count": len(rows),
        "requested_days": days,
        "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
        "missing_intervals": missing,
        "missing_minute_intervals": missing if interval == "1m" else None,
        **_source_manifest_fields(),
        "csv_sha256": digest,
        "commit_sha": commit_sha,
        "data_authority": "PUBLIC_MARKET_DATA_ONLY_NOT_ACCOUNT_ORDER_OR_FILL_TRUTH",
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }
    (output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, choices=ALLOWED_DAYS, default=30)
    parser.add_argument("--symbol", choices=ALLOWED_SYMBOLS, default="BTCUSDT")
    parser.add_argument("--interval", choices=tuple(INTERVAL_SECONDS), default="1m")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default="UNKNOWN")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    rows = fetch_historical_candles(
        days=args.days,
        now=now,
        symbol=args.symbol,
        interval=args.interval,
    )
    manifest = write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=args.days,
        retrieved_at=now,
        commit_sha=args.commit_sha,
        symbol=args.symbol,
        interval=args.interval,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
