"""Fetch bounded public Nobitex BTCUSDT hourly candles for manual Actions research.

Only GET /market/udf/history is used. No account/execution endpoints, credentials,
orders, notification calls, or external writes are implemented.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    request = Request(
        ENDPOINT + "?" + urlencode(params),
        headers={
            "Accept": "application/json",
            "User-Agent": "TraderBot/AliX-Phase10-Research",
        },
        method="GET",
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 200:
                    raise ValueError("public_api_http_status_invalid")
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("public_api_payload_not_object")
                return payload
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ):
            if attempt == 2:
                raise RuntimeError(
                    "public_api_request_failed_after_bounded_retries"
                ) from None
            time.sleep(2**attempt)
    raise RuntimeError("public_api_request_failed")


def parse_udf_payload(
    payload: dict[str, Any], *, requested_from: int, requested_to: int
) -> list[CandleRecord]:
    if payload.get("s") == "no_data":
        return []
    if payload.get("s") != "ok":
        raise ValueError("nobitex_public_api_status_not_ok")
    names = ("t", "o", "h", "l", "c", "v")
    arrays = {name: payload.get(name) for name in names}
    if any(not isinstance(value, list) for value in arrays.values()):
        raise ValueError("nobitex_ohlcv_arrays_missing")
    lengths = {len(value) for value in arrays.values() if isinstance(value, list)}
    if len(lengths) != 1:
        raise ValueError("nobitex_ohlcv_array_length_mismatch")
    if not all(isinstance(arrays[name], list) for name in names):
        raise ValueError("nobitex_ohlcv_arrays_invalid")
    timestamps = cast(list[Any], arrays["t"])
    opens = cast(list[Any], arrays["o"])
    highs = cast(list[Any], arrays["h"])
    lows = cast(list[Any], arrays["l"])
    closes = cast(list[Any], arrays["c"])
    volumes = cast(list[Any], arrays["v"])
    rows: list[CandleRecord] = []
    prior: int | None = None
    for index, raw_timestamp in enumerate(timestamps):
        if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, int):
            raise ValueError("nobitex_candle_timestamp_invalid")
        stamp = raw_timestamp
        # Do not require UTC epoch modulo 3600: venue candle boundaries can be
        # offset from UTC whole hours. Regular spacing is checked after retrieval.
        if stamp < requested_from or stamp > requested_to:
            raise ValueError("nobitex_candle_timestamp_out_of_bounds")
        if prior is not None and stamp <= prior:
            raise ValueError("nobitex_candle_timestamps_not_increasing")
        prior = stamp
        values: list[float] = []
        for raw in (
            opens[index],
            highs[index],
            lows[index],
            closes[index],
            volumes[index],
        ):
            try:
                item = float(raw)
            except (TypeError, ValueError):
                raise ValueError("nobitex_candle_value_invalid") from None
            if not math.isfinite(item):
                raise ValueError("nobitex_candle_value_not_finite")
            values.append(item)
        opened, high, low, close, volume = values
        if min(opened, high, low, close) <= 0 or volume < 0:
            raise ValueError("nobitex_candle_price_or_volume_out_of_range")
        if high < max(opened, close) or low > min(opened, close) or high < low:
            raise ValueError("nobitex_candle_ohlc_invariant_failed")
        rows.append(
            {
                "timestamp": stamp,
                "date": datetime.fromtimestamp(stamp, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "open": opened,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return rows


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
    end = (
        int(now.astimezone(timezone.utc).timestamp())
        // INTERVAL_SECONDS
        * INTERVAL_SECONDS
        - INTERVAL_SECONDS
    )
    cursor = end - days * 86400
    rows: list[CandleRecord] = []
    while cursor <= end:
        stop = min(cursor + (MAX_CANDLES_PER_REQUEST - 1) * INTERVAL_SECONDS, end)
        params = {
            "symbol": "BTCUSDT",
            "resolution": "60",
            "from": str(cursor),
            "to": str(stop),
        }
        batch = parse_udf_payload(
            request_json(params), requested_from=cursor, requested_to=stop
        )
        if not batch:
            raise ValueError("nobitex_historical_empty_page")
        rows.extend(batch)
        # Anchor the next page to the observed market-candle boundary, not the
        # UTC request window, to avoid dropping venue-offset candles at boundaries.
        cursor = int(batch[-1]["timestamp"]) + INTERVAL_SECONDS
        if cursor <= end:
            sleep_fn(1.1)
    rows.sort(key=lambda item: int(item["timestamp"]))
    seen: set[int] = set()
    for item in rows:
        stamp = int(item["timestamp"])
        if stamp in seen:
            raise ValueError("nobitex_historical_duplicate_candle")
        seen.add(stamp)
    for prior, current in zip(rows, rows[1:]):
        if int(current["timestamp"]) - int(prior["timestamp"]) != INTERVAL_SECONDS:
            raise ValueError("nobitex_historical_gap_detected_no_interpolation")
    if len(rows) < int(days * 24 * 0.98):
        raise ValueError("nobitex_historical_row_coverage_too_low")
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
