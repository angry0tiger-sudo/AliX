"""Fetch real Binance Spot 1-minute OHLCV with strict public-data manifests."""

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
from typing import Any, Callable, cast
from urllib.parse import urlencode

HOST = "data-api.binance.vision"
ENDPOINT = "/api/v3/klines"
SCHEMA_VERSION = "alix.binance.public-ohlcv.v1"
ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALLOWED_DAYS = (1, 7, 30, 90, 730)
INTERVAL_MS = 60_000
PAGE_SIZE = 1000
PAGE_DELAY_SECONDS = 0.25
RETRY_STATUSES = frozenset({418, 429, 500, 502, 503, 504})
GAP_POLICY = "reject; no interpolation, resampling, or synthetic fill"


def _http_error_detail(body: bytes) -> str:
    """Extract a short server diagnostic without hiding HTTP failure context."""
    decoded = body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded[:200] or "empty_response"
    if isinstance(payload, dict):
        message = payload.get("msg")
        if isinstance(message, str) and message.strip():
            return message.strip()[:200]
    return decoded[:200] or "empty_response"


def request_public_json(
    params: dict[str, str], *, timeout_seconds: float = 15.0
) -> list[Any]:
    """GET from Binance's official unauthenticated market-data-only host."""
    last_failure = "unknown"
    for attempt in range(5):
        conn = HTTPSConnection(
            HOST, timeout=timeout_seconds, context=ssl.create_default_context()
        )
        retry_after: str | None = None
        try:
            conn.request(
                "GET",
                ENDPOINT + "?" + urlencode(params),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-Public-OHLCV/1",
                },
            )
            response = conn.getresponse()
            body = response.read()
            if response.status == 200:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, list):
                    raise ValueError("binance_klines_payload_not_list")
                return payload
            if response.status not in RETRY_STATUSES:
                detail = _http_error_detail(body)
                if response.status == 451:
                    raise RuntimeError(
                        "binance_public_api_restricted_location:"
                        f"status=451:host={HOST}:detail={detail}"
                    )
                raise RuntimeError(
                    f"binance_public_api_http_status_invalid:{response.status}:"
                    f"host={HOST}:detail={detail}"
                )
            retry_after = response.getheader("Retry-After")
            last_failure = f"http_{response.status}"
        except (HTTPException, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_failure = type(exc).__name__
        finally:
            conn.close()
        if attempt == 4:
            raise RuntimeError(
                f"binance_public_api_request_failed_after_bounded_retries:{last_failure}"
            ) from None
        try:
            delay = float(retry_after) if retry_after is not None else float(2**attempt)
        except (TypeError, ValueError):
            delay = float(2**attempt)
        time.sleep(min(max(delay, 0.0), 30.0))
    raise RuntimeError("binance_public_api_request_failed")


def _validate_symbol(symbol: str) -> None:
    if symbol not in ALLOWED_SYMBOLS:
        raise ValueError("binance_symbol_out_of_policy")


def _parse_utc_datetime(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name}_invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name}_timezone_required")
    epoch_ms = int(parsed.astimezone(timezone.utc).timestamp() * 1000)
    if epoch_ms % INTERVAL_MS:
        raise ValueError(f"{name}_must_align_to_minute")
    return datetime.fromtimestamp(epoch_ms / 1000, timezone.utc)


def _datetime_epoch_ms(value: datetime, name: str) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_timezone_required")
    epoch_ms = int(value.astimezone(timezone.utc).timestamp() * 1000)
    if epoch_ms % INTERVAL_MS:
        raise ValueError(f"{name}_must_align_to_minute")
    return epoch_ms


def _validate_kline_timestamp(raw: Any, expected_ms: int) -> int:
    if not isinstance(raw, list) or len(raw) < 6:
        raise ValueError("binance_kline_shape_invalid")
    stamp = raw[0]
    if isinstance(stamp, bool) or not isinstance(stamp, int):
        raise ValueError("binance_kline_timestamp_invalid")
    if stamp != expected_ms:
        raise ValueError(
            f"binance_kline_timestamp_mismatch:expected={expected_ms}:actual={stamp}"
        )
    return cast(int, stamp)


def _validated_kline_values(raw: list[Any]) -> tuple[float, float, float, float, float]:
    try:
        values = [float(value) for value in raw[1:6]]
    except (TypeError, ValueError):
        raise ValueError("binance_kline_value_invalid") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("binance_kline_value_not_finite")
    opened, high, low, close, volume = values
    if min(opened, high, low, close) <= 0 or volume < 0:
        raise ValueError("binance_kline_price_or_volume_out_of_range")
    if high < max(opened, close) or low > min(opened, close) or high < low:
        raise ValueError("binance_kline_ohlc_invariant_failed")
    return opened, high, low, close, volume


def _parse_kline(raw: Any, expected_ms: int) -> dict[str, Any]:
    stamp = _validate_kline_timestamp(raw, expected_ms)
    opened, high, low, close, volume = _validated_kline_values(raw)
    return {
        "timestamp": stamp // 1000,
        "date": datetime.fromtimestamp(stamp / 1000, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _fetch_page(
    cursor: int,
    end_ms: int,
    symbol: str,
    request_json: Callable[[dict[str, str]], list[Any]],
    sleep_fn: Callable[[float], None],
) -> tuple[list[dict[str, Any]], int]:
    expected_count = min(PAGE_SIZE, (end_ms - cursor) // INTERVAL_MS + 1)
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": str(cursor),
        "endTime": str(end_ms),
        "limit": str(PAGE_SIZE),
    }
    last_error: ValueError | None = None
    for attempt in range(3):
        try:
            payload = request_json(params)
            if len(payload) != expected_count:
                raise ValueError(
                    f"binance_kline_page_count_mismatch:expected={expected_count}:actual={len(payload)}:start={cursor}"
                )
            page = [
                _parse_kline(item, cursor + index * INTERVAL_MS)
                for index, item in enumerate(payload)
            ]
            return page, expected_count
        except ValueError as exc:
            last_error = exc
            if attempt < 2:
                sleep_fn(2**attempt)
    assert last_error is not None
    raise ValueError(f"{last_error}:attempts=3") from None


def _validate_complete_window(
    rows: list[dict[str, Any]], expected_count: int, start_ms: int, end_ms: int
) -> None:
    stamps = [int(row["timestamp"]) for row in rows]
    if len(rows) != expected_count:
        raise ValueError(
            f"binance_historical_row_count_mismatch:expected={expected_count}:actual={len(rows)}"
        )
    if any(current - previous != 60 for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("binance_historical_gap_detected")
    if stamps[0] * 1000 != start_ms or stamps[-1] * 1000 != end_ms:
        raise ValueError("binance_historical_window_boundary_mismatch")


def _validate_fetch_request(
    days: int,
    now: datetime,
    symbol: str,
    start_utc: datetime | None,
    end_utc: datetime | None,
) -> int:
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("historical_days_out_of_policy")
    if (start_utc is None) != (end_utc is None):
        raise ValueError("historical_window_requires_both_boundaries")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("historical_cutoff_timezone_required")
    _validate_symbol(symbol)
    return days * 1440


def _exact_historical_bounds(
    start_utc: datetime, end_utc: datetime, expected_count: int
) -> tuple[int, int]:
    start_ms = _datetime_epoch_ms(start_utc, "historical_start_utc")
    end_ms = _datetime_epoch_ms(end_utc, "historical_end_utc")
    if end_ms < start_ms:
        raise ValueError("historical_window_end_before_start")
    interval_ms = end_ms - start_ms
    expected_interval = (expected_count - 1) * INTERVAL_MS
    if interval_ms != expected_interval:
        raise ValueError("historical_window_does_not_match_requested_days")
    return start_ms, end_ms


def _rolling_historical_bounds(now: datetime, expected_count: int) -> tuple[int, int]:
    now_ms = int(now.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS - INTERVAL_MS
    start_ms = end_ms - (expected_count - 1) * INTERVAL_MS
    return start_ms, end_ms


def _resolve_historical_bounds(
    now: datetime,
    expected_count: int,
    start_utc: datetime | None,
    end_utc: datetime | None,
) -> tuple[int, int]:
    if start_utc is not None and end_utc is not None:
        return _exact_historical_bounds(start_utc, end_utc, expected_count)
    return _rolling_historical_bounds(now, expected_count)


def _fetch_historical_pages(
    start_ms: int,
    end_ms: int,
    symbol: str,
    request_json: Callable[[dict[str, str]], list[Any]],
    sleep_fn: Callable[[float], None],
) -> list[dict[str, Any]]:
    cursor = start_ms
    rows: list[dict[str, Any]] = []
    while cursor <= end_ms:
        page, page_count = _fetch_page(cursor, end_ms, symbol, request_json, sleep_fn)
        rows.extend(page)
        cursor += page_count * INTERVAL_MS
        if cursor <= end_ms:
            sleep_fn(PAGE_DELAY_SECONDS)
    return rows


def fetch_historical_candles(
    *,
    days: int,
    now: datetime,
    symbol: str = "BTCUSDT",
    request_json: Callable[[dict[str, str]], list[Any]] = request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    expected_count = _validate_fetch_request(days, now, symbol, start_utc, end_utc)
    start_ms, end_ms = _resolve_historical_bounds(
        now, expected_count, start_utc, end_utc
    )
    rows = _fetch_historical_pages(start_ms, end_ms, symbol, request_json, sleep_fn)
    _validate_complete_window(rows, expected_count, start_ms, end_ms)
    return rows


def write_artifacts(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    days: int,
    retrieved_at: datetime,
    commit_sha: str,
    symbol: str,
) -> dict[str, Any]:
    _validate_symbol(symbol)
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at_timezone_required")
    count = days * 1440
    stamps = [int(row["timestamp"]) for row in rows]
    if not rows or len(rows) != count:
        raise ValueError(
            f"binance_historical_row_count_mismatch:expected={count}:actual={len(rows)}"
        )
    if any(b - a != 60 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("binance_historical_gap_detected_before_write")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = output_dir / f"{stem}.csv"
    temporary = output_dir / f"{stem}.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("date", "open", "high", "low", "close", "volume")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    temporary.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provider": "Binance Spot public klines API",
        "host": HOST,
        "endpoint": ENDPOINT,
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "interval": "1m",
        "start_utc": rows[0]["date"],
        "end_utc": rows[-1]["date"],
        "row_count": len(rows),
        "requested_days": days,
        "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
        "missing_intervals": 0,
        "missing_minute_intervals": 0,
        "gap_policy": GAP_POLICY,
        "fetch_page_size_candles": PAGE_SIZE,
        "inter_page_delay_seconds": PAGE_DELAY_SECONDS,
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
    parser.add_argument("--interval", choices=("1m",), default="1m")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default="UNKNOWN")
    parser.add_argument(
        "--start-utc",
        help="Optional exact inclusive UTC candle-open timestamp from a source manifest.",
    )
    parser.add_argument(
        "--end-utc",
        help="Optional exact inclusive UTC candle-open timestamp from a source manifest.",
    )
    args = parser.parse_args()
    if (args.start_utc is None) != (args.end_utc is None):
        parser.error("--start-utc and --end-utc must be supplied together")
    now = datetime.now(timezone.utc)
    start_utc = (
        _parse_utc_datetime(args.start_utc, "historical_start_utc")
        if args.start_utc
        else None
    )
    end_utc = (
        _parse_utc_datetime(args.end_utc, "historical_end_utc")
        if args.end_utc
        else None
    )
    rows = fetch_historical_candles(
        days=args.days,
        now=now,
        symbol=args.symbol,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    manifest = write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=args.days,
        retrieved_at=now,
        commit_sha=args.commit_sha,
        symbol=args.symbol,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
