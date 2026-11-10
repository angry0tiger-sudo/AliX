"""Strict, timestamp-aligned Binance Spot fallback for one approved Nobitex outage."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.client import HTTPSConnection, HTTPException
import json
import math
import ssl
import time
from typing import Any, Callable, Mapping, TypedDict
from urllib.parse import urlencode

BINANCE_HOST = "data-api.binance.vision"
BINANCE_PATH = "/api/v3/klines"
BINANCE_BACKFILL_SOURCE = "BINANCE_BACKFILL_FOR_NOBITEX_OUTAGE"
BINANCE_BACKFILL_START_TIMESTAMP = 1750224600
BINANCE_BACKFILL_END_TIMESTAMP = 1750307400
BINANCE_BACKFILL_TIME_ZONE = "+03:30"
BINANCE_INTERVAL_SECONDS = 3600
MAX_BOUNDARY_SCALE_SPREAD_BPS = 100.0
TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_HTTP_RETRIES = 5
MAX_RETRY_DELAY_SECONDS = 30.0


class BackfillCandle(TypedDict):
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def _retry_delay_seconds(attempt: int, retry_after: str | None = None) -> float:
    try:
        delay = float(retry_after) if retry_after is not None else float(2**attempt)
    except (TypeError, ValueError):
        delay = float(2**attempt)
    return min(max(delay, 0.0), MAX_RETRY_DELAY_SECONDS)


def request_binance_public_klines(
    params: dict[str, str], *, timeout_seconds: float = 15.0
) -> Any:
    """Fetch public Binance Spot klines without credentials or private endpoints."""
    path = BINANCE_PATH + "?" + urlencode(params)
    last_failure = "unknown"
    for attempt in range(MAX_HTTP_RETRIES):
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
                    "User-Agent": "AliX-Public-OHLCV-Backfill/1",
                },
            )
            response = connection.getresponse()
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise ValueError("binance_klines_payload_not_array")
                return payload
            if response.status not in TRANSIENT_HTTP_STATUSES:
                raise ValueError("binance_public_api_http_status_invalid")
            retry_after = response.getheader("Retry-After")
            last_failure = f"http_{response.status}"
        except (HTTPException, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_failure = type(exc).__name__
        finally:
            connection.close()
        if attempt == MAX_HTTP_RETRIES - 1:
            raise RuntimeError(
                f"binance_public_api_request_failed_after_bounded_retries:{last_failure}"
            ) from None
        time.sleep(_retry_delay_seconds(attempt, retry_after))
    raise RuntimeError("binance_public_api_request_failed")


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"binance_backfill_invalid_{field}") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"binance_backfill_invalid_{field}")
    return number


def _parse_binance_timestamp(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("binance_backfill_timestamp_invalid")
    try:
        timestamp_ms = int(raw)
    except (TypeError, ValueError):
        raise ValueError("binance_backfill_timestamp_invalid") from None
    if timestamp_ms % 1000 != 0:
        raise ValueError("binance_backfill_timestamp_precision_invalid")
    return timestamp_ms // 1000


def _parse_binance_ohlcv(raw: list[Any]) -> tuple[float, float, float, float, float]:
    opened = _positive_number(raw[1], "open")
    high = _positive_number(raw[2], "high")
    low = _positive_number(raw[3], "low")
    close = _positive_number(raw[4], "close")
    try:
        volume = float(raw[5])
    except (TypeError, ValueError):
        raise ValueError("binance_backfill_invalid_volume") from None
    if not math.isfinite(volume) or volume < 0.0:
        raise ValueError("binance_backfill_invalid_volume")
    if high < max(opened, close) or low > min(opened, close) or low > high:
        raise ValueError("binance_backfill_ohlc_invariant_failed")
    return opened, high, low, close, volume


def _parse_one_binance_kline(
    raw: Any, expected_timestamps: set[int]
) -> tuple[int, BackfillCandle] | None:
    if not isinstance(raw, list) or len(raw) < 6:
        raise ValueError("binance_backfill_kline_shape_invalid")
    timestamp = _parse_binance_timestamp(raw[0])
    if timestamp not in expected_timestamps:
        return None
    opened, high, low, close, volume = _parse_binance_ohlcv(raw)
    row: BackfillCandle = {
        "timestamp": timestamp,
        "date": datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return timestamp, row


def _parse_binance_klines(
    payload: Any, expected_timestamps: list[int]
) -> tuple[dict[int, BackfillCandle], str]:
    if not isinstance(payload, list):
        raise ValueError("binance_klines_payload_not_array")
    expected = set(expected_timestamps)
    rows: dict[int, BackfillCandle] = {}
    for raw in payload:
        parsed = _parse_one_binance_kline(raw, expected)
        if parsed is None:
            continue
        timestamp, row = parsed
        if timestamp in rows:
            raise ValueError("binance_backfill_duplicate_timestamp")
        rows[timestamp] = row
    if sorted(rows) != expected_timestamps:
        raise ValueError("binance_backfill_timestamp_coverage_mismatch")
    raw_digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return rows, raw_digest


def _expected_binance_gap() -> list[int]:
    return list(
        range(
            BINANCE_BACKFILL_START_TIMESTAMP,
            BINANCE_BACKFILL_END_TIMESTAMP + BINANCE_INTERVAL_SECONDS,
            BINANCE_INTERVAL_SECONDS,
        )
    )


def _validate_backfill_request(
    *,
    missing_timestamps: list[int],
    nobitex_anchor_rows: Mapping[int, Mapping[str, Any]],
    symbol: str,
) -> tuple[list[int], int, int]:
    if symbol != "BTCUSDT":
        raise ValueError("binance_backfill_symbol_not_approved")
    expected_gap = _expected_binance_gap()
    if missing_timestamps != expected_gap:
        raise ValueError("binance_backfill_gap_not_approved")
    before_timestamp = BINANCE_BACKFILL_START_TIMESTAMP - BINANCE_INTERVAL_SECONDS
    after_timestamp = BINANCE_BACKFILL_END_TIMESTAMP + BINANCE_INTERVAL_SECONDS
    for anchor_timestamp in (before_timestamp, after_timestamp):
        if anchor_timestamp not in nobitex_anchor_rows:
            raise ValueError("binance_backfill_nobitex_boundary_anchor_missing")
    return expected_gap, before_timestamp, after_timestamp


def _fetch_binance_anchor_window(
    *,
    symbol: str,
    before_timestamp: int,
    after_timestamp: int,
    expected_timestamps: list[int],
    request_json: Callable[[dict[str, str]], Any],
) -> tuple[dict[int, BackfillCandle], str]:
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": str(before_timestamp * 1000),
        "endTime": str(after_timestamp * 1000),
        "timeZone": BINANCE_BACKFILL_TIME_ZONE,
        "limit": str(len(expected_timestamps) + 2),
    }
    raw_payload = request_json(params)
    return _parse_binance_klines(raw_payload, expected_timestamps)


def _calibrate_price_scale(
    *,
    nobitex_anchor_rows: Mapping[int, Mapping[str, Any]],
    binance_rows: Mapping[int, BackfillCandle],
    before_timestamp: int,
    after_timestamp: int,
) -> tuple[float, float, float, float]:
    pre_ratio = (
        _positive_number(
            nobitex_anchor_rows[before_timestamp].get("close"),
            "nobitex_pre_close",
        )
        / binance_rows[before_timestamp]["close"]
    )
    post_ratio = (
        _positive_number(
            nobitex_anchor_rows[after_timestamp].get("close"),
            "nobitex_post_close",
        )
        / binance_rows[after_timestamp]["close"]
    )
    ratio_midpoint = (pre_ratio + post_ratio) / 2.0
    spread_bps = abs(pre_ratio - post_ratio) / ratio_midpoint * 10_000.0
    if not math.isfinite(spread_bps) or spread_bps > MAX_BOUNDARY_SCALE_SPREAD_BPS:
        raise ValueError("binance_backfill_boundary_scale_drift")
    return math.sqrt(pre_ratio * post_ratio), pre_ratio, post_ratio, spread_bps


def _adjust_binance_gap_rows(
    *,
    expected_gap: list[int],
    binance_rows: Mapping[int, BackfillCandle],
    adjustment_factor: float,
) -> list[BackfillCandle]:
    adjusted_rows: list[BackfillCandle] = []
    for timestamp in expected_gap:
        raw = binance_rows[timestamp]
        adjusted: BackfillCandle = {
            "timestamp": timestamp,
            "date": raw["date"],
            "open": raw["open"] * adjustment_factor,
            "high": raw["high"] * adjustment_factor,
            "low": raw["low"] * adjustment_factor,
            "close": raw["close"] * adjustment_factor,
            "volume": raw["volume"],
        }
        _validate_adjusted_ohlc(adjusted)
        adjusted_rows.append(adjusted)
    return adjusted_rows


def _validate_adjusted_ohlc(row: BackfillCandle) -> None:
    if (
        row["high"] < max(row["open"], row["close"])
        or row["low"] > min(row["open"], row["close"])
        or row["low"] > row["high"]
    ):
        raise ValueError("binance_backfill_adjusted_ohlc_invariant_failed")


def _build_backfill_provenance(
    *,
    symbol: str,
    adjusted_rows: list[BackfillCandle],
    raw_digest: str,
    adjustment_factor: float,
    pre_ratio: float,
    post_ratio: float,
    spread_bps: float,
    before_timestamp: int,
    after_timestamp: int,
) -> dict[str, object]:
    return {
        "source": BINANCE_BACKFILL_SOURCE,
        "provider": "Binance Spot public market-data API",
        "endpoint": f"https://{BINANCE_HOST}{BINANCE_PATH}",
        "symbol": symbol,
        "interval": "1h",
        "time_zone": BINANCE_BACKFILL_TIME_ZONE,
        "start_timestamp": BINANCE_BACKFILL_START_TIMESTAMP,
        "end_timestamp": BINANCE_BACKFILL_END_TIMESTAMP,
        "start_utc": adjusted_rows[0]["date"],
        "end_utc": adjusted_rows[-1]["date"],
        "row_count": len(adjusted_rows),
        "raw_binance_klines_sha256": raw_digest,
        "price_adjustment_method": (
            "GEOMETRIC_MEAN_OF_SAME_TIMESTAMP_PRE_AND_POST_BOUNDARY_CLOSE_RATIOS"
        ),
        "price_adjustment_factor": adjustment_factor,
        "pre_gap_close_ratio": pre_ratio,
        "post_gap_close_ratio": post_ratio,
        "boundary_scale_spread_bps": spread_bps,
        "calibration_anchor_timestamps": {
            "pre_gap": before_timestamp,
            "post_gap": after_timestamp,
        },
        "adjusted_fields": ["open", "high", "low", "close"],
        "volume_adjusted": False,
        "volume_source": "BINANCE_SPOT_UNADJUSTED_BASE_ASSET_VOLUME",
    }


def fetch_binance_gap_backfill(
    *,
    missing_timestamps: list[int],
    nobitex_anchor_rows: Mapping[int, Mapping[str, Any]],
    symbol: str = "BTCUSDT",
    request_json: Callable[[dict[str, str]], Any] = request_binance_public_klines,
) -> tuple[list[BackfillCandle], dict[str, object]]:
    """Fetch the exact approved gap, scale prices, and validate both boundaries."""
    expected_gap, before_timestamp, after_timestamp = _validate_backfill_request(
        missing_timestamps=missing_timestamps,
        nobitex_anchor_rows=nobitex_anchor_rows,
        symbol=symbol,
    )
    expected_with_anchors = [before_timestamp, *expected_gap, after_timestamp]
    binance_rows, raw_digest = _fetch_binance_anchor_window(
        symbol=symbol,
        before_timestamp=before_timestamp,
        after_timestamp=after_timestamp,
        expected_timestamps=expected_with_anchors,
        request_json=request_json,
    )
    adjustment_factor, pre_ratio, post_ratio, spread_bps = _calibrate_price_scale(
        nobitex_anchor_rows=nobitex_anchor_rows,
        binance_rows=binance_rows,
        before_timestamp=before_timestamp,
        after_timestamp=after_timestamp,
    )
    adjusted_rows = _adjust_binance_gap_rows(
        expected_gap=expected_gap,
        binance_rows=binance_rows,
        adjustment_factor=adjustment_factor,
    )
    provenance = _build_backfill_provenance(
        symbol=symbol,
        adjusted_rows=adjusted_rows,
        raw_digest=raw_digest,
        adjustment_factor=adjustment_factor,
        pre_ratio=pre_ratio,
        post_ratio=post_ratio,
        spread_bps=spread_bps,
        before_timestamp=before_timestamp,
        after_timestamp=after_timestamp,
    )
    return adjusted_rows, provenance
