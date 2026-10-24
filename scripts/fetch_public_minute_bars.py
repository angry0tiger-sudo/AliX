"""Fetch public Binance Spot one-minute klines for research artifacts only."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence
import requests

BINANCE_PUBLIC_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL = "1m"
INTERVAL_MS = 60_000
KLINES_LIMIT = 1_000
MAX_CALENDAR_MONTHS = 24
USER_AGENT = "AliX-Research/1.0 (public-market-data-only)"


@dataclass(frozen=True, slots=True)
class KlineRow:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


def subtract_calendar_months(value: datetime, months: int) -> datetime:
    if isinstance(months, bool) or not isinstance(months, int) or months <= 0:
        raise ValueError("months_must_be_positive_integer")
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def parse_kline(payload: object) -> KlineRow:
    if not isinstance(payload, list) or len(payload) < 11:
        raise ValueError("binance_kline_shape_invalid")
    try:
        row = KlineRow(
            open_time_ms=int(payload[0]),
            open=float(payload[1]),
            high=float(payload[2]),
            low=float(payload[3]),
            close=float(payload[4]),
            volume=float(payload[5]),
            close_time_ms=int(payload[6]),
            quote_volume=float(payload[7]),
            trade_count=int(payload[8]),
            taker_buy_base_volume=float(payload[9]),
            taker_buy_quote_volume=float(payload[10]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("binance_kline_numeric_field_invalid") from exc
    if row.open_time_ms < 0 or row.close_time_ms < row.open_time_ms:
        raise ValueError("binance_kline_timestamps_invalid")
    prices = (row.open, row.high, row.low, row.close)
    volumes = (
        row.volume,
        row.quote_volume,
        row.taker_buy_base_volume,
        row.taker_buy_quote_volume,
    )
    if any(not math.isfinite(item) or item <= 0 for item in prices):
        raise ValueError("binance_kline_price_invalid")
    if row.high < max(row.open, row.close, row.low):
        raise ValueError("binance_kline_high_invalid")
    if row.low > min(row.open, row.close, row.high):
        raise ValueError("binance_kline_low_invalid")
    if any(not math.isfinite(item) or item < 0 for item in volumes):
        raise ValueError("binance_kline_volume_invalid")
    if row.trade_count < 0:
        raise ValueError("binance_kline_trade_count_invalid")
    return row


def _fetch_page(symbol: str, start_ms: int, end_ms: int) -> list[KlineRow]:
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": KLINES_LIMIT,
    }
    try:
        response = requests.get(
            BINANCE_PUBLIC_KLINES_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        payload: object = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("public_binance_market_data_request_failed") from exc
    if not isinstance(payload, list):
        raise ValueError("binance_kline_response_not_list")
    return [parse_kline(item) for item in payload]


def _append_page(
    rows: list[KlineRow],
    page: Sequence[KlineRow],
    start_ms: int,
    end_ms: int,
) -> None:
    for row in page:
        if not start_ms <= row.open_time_ms <= end_ms:
            continue
        if rows and row.open_time_ms <= rows[-1].open_time_ms:
            if row.open_time_ms < rows[-1].open_time_ms:
                raise ValueError("binance_kline_order_invalid")
            continue
        rows.append(row)


def _fetch_history(symbol: str, start_ms: int, end_ms: int) -> list[KlineRow]:
    cursor_ms = start_ms
    rows: list[KlineRow] = []
    while cursor_ms <= end_ms:
        page = _fetch_page(symbol, cursor_ms, end_ms)
        if not page:
            break
        _append_page(rows, page, start_ms, end_ms)
        next_cursor = page[-1].open_time_ms + INTERVAL_MS
        if next_cursor <= cursor_ms:
            raise ValueError("binance_kline_pagination_did_not_advance")
        cursor_ms = next_cursor
        if len(page) < KLINES_LIMIT:
            break
        time.sleep(0.1)
    return rows


def _gap_count(rows: Sequence[KlineRow]) -> int:
    return sum(
        max(0, (current.open_time_ms - previous.open_time_ms) // INTERVAL_MS - 1)
        for previous, current in zip(rows, rows[1:])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_history(symbol: str, months: int, out_dir: Path) -> dict[str, object]:
    if not symbol or not symbol.isascii() or not symbol.isalnum():
        raise ValueError("symbol_must_be_ascii_alphanumeric")
    if not 1 <= len(symbol) <= 20:
        raise ValueError("symbol_length_invalid")
    if isinstance(months, bool) or not 1 <= months <= MAX_CALENDAR_MONTHS:
        raise ValueError("months_out_of_supported_range")
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    latest_closed_open = int(now.timestamp() * 1000) - INTERVAL_MS
    start_datetime = subtract_calendar_months(now, months)
    start_ms = int(start_datetime.timestamp() * 1000)
    rows = _fetch_history(symbol.upper(), start_ms, latest_closed_open)
    if not rows:
        raise RuntimeError("no_public_minute_bars_returned")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"binance_{symbol.upper()}_1m.csv"
    fields = list(asdict(rows[0]).keys())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    expected_count = (latest_closed_open - start_ms) // INTERVAL_MS + 1
    missing_intervals = _gap_count(rows)
    manifest: dict[str, object] = {
        "schema_version": "alix.self-evolving-data-manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_venue": "binance_spot_public_market_data",
        "endpoint": BINANCE_PUBLIC_KLINES_URL,
        "symbol": symbol.upper(),
        "interval": INTERVAL,
        "start_utc": datetime.fromtimestamp(start_ms / 1000, UTC).isoformat(),
        "end_utc": datetime.fromtimestamp(latest_closed_open / 1000, UTC).isoformat(),
        "row_count": len(rows),
        "expected_minute_slots": expected_count,
        "missing_minute_intervals_between_rows": missing_intervals,
        "coverage_fraction": len(rows) / expected_count,
        "csv_file": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
        "private_api_calls": 0,
        "orders_or_execution_api_used": False,
        "real_order_attempts": 0,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/public-minute-bars")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = download_history(args.symbol, args.months, args.out_dir)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
