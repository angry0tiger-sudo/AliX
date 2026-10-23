"""Fetch bounded public Nobitex BTCUSDT hourly candles for manual Actions research.

Only GET /market/udf/history is used. No account/execution endpoints, credentials,
orders, notification calls, or external writes are implemented.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.client import HTTPSConnection, HTTPException
import ssl
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, TypedDict, cast
from urllib.parse import urlencode

ENDPOINT = "https://apiv2.nobitex.ir/market/udf/history"
SCHEMA_VERSION = "alix.nobitex.public-ohlcv.v1"
INTERVAL_SECONDS = 3600
ALLOWED_DAYS = (30, 90, 180, 365)
MAX_CANDLES_PER_REQUEST = 400


class CandleRecord(TypedDict):
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def request_public_json(
    params: dict[str, str], *, timeout_seconds: float = 15.0
) -> dict[str, Any]:
    """Perform a verified-TLS GET to a fixed public host; redirects are not followed."""
    path = "/market/udf/history?" + urlencode(params)
    for attempt in range(3):
        connection = HTTPSConnection(
            "apiv2.nobitex.ir",
            timeout=timeout_seconds,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "TraderBot/AliX-Phase10-Research",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("public_api_http_status_invalid")
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("public_api_payload_not_object")
            return payload
        except (HTTPException, TimeoutError, OSError, json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise RuntimeError(
                    "public_api_request_failed_after_bounded_retries"
                ) from None
            time.sleep(2**attempt)
        finally:
            connection.close()
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


def _fetch_candle_pages(
    *,
    start: int,
    end: int,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> list[CandleRecord]:
    cursor = start
    rows: list[CandleRecord] = []
    while cursor <= end:
        stop = min(cursor + (MAX_CANDLES_PER_REQUEST - 3) * INTERVAL_SECONDS, end)
        # Pad query bounds by one interval to tolerate providers that interpret
        # UDF range endpoints as exclusive, but retain only the exact target window.
        request_from = max(start, cursor - INTERVAL_SECONDS)
        request_to = min(end, stop + INTERVAL_SECONDS)
        params = {
            "symbol": "BTCUSDT",
            "resolution": "60",
            "from": str(request_from),
            "to": str(request_to),
        }
        batch: list[CandleRecord] = []
        for attempt in range(3):
            page = parse_udf_payload(
                request_json(params),
                requested_from=request_from,
                requested_to=request_to,
            )
            batch = [row for row in page if cursor <= row["timestamp"] <= stop]
            if batch:
                break
            if attempt < 2:
                sleep_fn(2**attempt)
        if not batch:
            raise ValueError(f"nobitex_historical_empty_page:{cursor}:{stop}")
        rows.extend(batch)
        cursor = batch[-1]["timestamp"] + INTERVAL_SECONDS
        if cursor <= end:
            sleep_fn(1.1)
    return rows


def _validate_historical_rows(rows: list[CandleRecord], days: int) -> None:
    rows.sort(key=lambda item: item["timestamp"])
    stamps = [row["timestamp"] for row in rows]
    if len(stamps) != len(set(stamps)):
        raise ValueError("nobitex_historical_duplicate_candle")
    for previous, current in zip(stamps, stamps[1:]):
        if current - previous != INTERVAL_SECONDS:
            raise ValueError("nobitex_historical_gap_detected_no_interpolation")
    if len(rows) < int(days * 24 * 0.98):
        raise ValueError("nobitex_historical_row_coverage_too_low")


def _latest_closed_candle_timestamp(
    *,
    now: datetime,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
) -> int:
    """Discover the provider's candle grid and select the latest fully closed bar."""
    now_epoch = int(now.astimezone(timezone.utc).timestamp())
    requested_from = now_epoch - MAX_CANDLES_PER_REQUEST * INTERVAL_SECONDS
    payload = request_json(
        {
            "symbol": "BTCUSDT",
            "resolution": "60",
            "from": str(requested_from),
            "to": str(now_epoch),
        }
    )
    recent_rows = parse_udf_payload(
        payload,
        requested_from=requested_from,
        requested_to=now_epoch,
    )
    closed_stamps = [
        row["timestamp"]
        for row in recent_rows
        if row["timestamp"] + INTERVAL_SECONDS <= now_epoch
    ]
    if not closed_stamps:
        raise ValueError("nobitex_historical_no_closed_candle")
    return closed_stamps[-1]


def fetch_historical_candles(
    *,
    days: int,
    now: datetime,
    request_json: Callable[[dict[str, str]], dict[str, Any]] = request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[CandleRecord]:
    if isinstance(days, bool) or days not in ALLOWED_DAYS:
        raise ValueError("historical_days_out_of_policy")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("historical_cutoff_timezone_required")
    end = _latest_closed_candle_timestamp(now=now, request_json=request_json)
    start = end - (days * 24 - 1) * INTERVAL_SECONDS
    rows = _fetch_candle_pages(
        start=start,
        end=end,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    if not rows:
        raise ValueError("nobitex_historical_candles_empty")
    _validate_historical_rows(rows, days)
    return rows


def write_artifacts(
    rows: list[CandleRecord],
    *,
    output_dir: Path,
    days: int,
    retrieved_at: datetime,
    commit_sha: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("nobitex_historical_candles_empty")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at_timezone_required")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nobitex-btcusdt-1h.csv"
    temporary = output_dir / "nobitex-btcusdt-1h.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
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
    temporary.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    stamps = [int(row["timestamp"]) for row in rows]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provider": "Nobitex public UDF history API",
        "endpoint": "/market/udf/history",
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "requested_days": days,
        "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
        "first_candle_epoch": stamps[0],
        "last_candle_epoch": stamps[-1],
        "row_count": len(rows),
        "gap_policy": "reject; no interpolation",
        "csv_sha256": digest,
        "commit_sha": commit_sha,
        "data_authority": "PUBLIC_MARKET_DATA_ONLY_NOT_ACCOUNT_ORDER_OR_FILL_TRUTH",
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }
    (output_dir / "nobitex-btcusdt-1h.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, choices=ALLOWED_DAYS, default=365)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default="UNKNOWN")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    rows = fetch_historical_candles(days=args.days, now=now)
    manifest = write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=args.days,
        retrieved_at=now,
        commit_sha=args.commit_sha,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
