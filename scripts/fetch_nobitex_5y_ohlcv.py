"""Fetch exactly 1,825 contiguous days of hourly public Nobitex OHLCV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import time
from typing import Any, Callable

from scripts import fetch_nobitex_public_ohlcv as source

DAYS_PER_WINDOW = 365
WINDOW_COUNT = 5
DAYS_TOTAL = DAYS_PER_WINDOW * WINDOW_COUNT
CANDLES_PER_WINDOW = DAYS_PER_WINDOW * 24
CANDLES_TOTAL = CANDLES_PER_WINDOW * WINDOW_COUNT
INTERVAL_SECONDS = 3600


def fetch_five_year_candles(
    *,
    now: datetime,
    symbol: str = "BTCUSDT",
    request_json: Callable[[dict[str, str]], dict[str, Any]] = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[source.CandleRecord]:
    """Fetch five backward-adjacent 365-day windows; gaps fail the run."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("five_year_cutoff_timezone_required")
    cutoff = now.astimezone(timezone.utc)
    chunks: list[list[source.CandleRecord]] = []
    for _ in range(WINDOW_COUNT):
        chunk = source.fetch_historical_candles(
            days=DAYS_PER_WINDOW,
            now=cutoff,
            symbol=symbol,
            interval="1h",
            request_json=request_json,
            sleep_fn=sleep_fn,
        )
        if len(chunk) != CANDLES_PER_WINDOW:
            raise ValueError("five_year_window_row_count_mismatch")
        if chunks and int(chunk[-1]["timestamp"]) + INTERVAL_SECONDS != int(
            chunks[0][0]["timestamp"]
        ):
            raise ValueError("five_year_window_boundary_gap_or_overlap")
        chunks.insert(0, chunk)
        cutoff = datetime.fromtimestamp(int(chunk[0]["timestamp"]), timezone.utc)
    rows = [row for chunk in chunks for row in chunk]
    if len(rows) != CANDLES_TOTAL:
        raise ValueError("five_year_total_row_count_mismatch")
    timestamps = [int(row["timestamp"]) for row in rows]
    if any(
        current - previous != INTERVAL_SECONDS
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("five_year_ohlcv_has_duplicate_or_missing_candles")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=source.ALLOWED_SYMBOLS, default="BTCUSDT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    retrieved_at = datetime.now(timezone.utc)
    rows = fetch_five_year_candles(now=retrieved_at, symbol=args.symbol)
    manifest = source.write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=DAYS_TOTAL,
        retrieved_at=retrieved_at,
        commit_sha=args.commit_sha,
        symbol=args.symbol,
        interval="1h",
    )
    manifest["windowing_policy"] = "FIVE_ADJACENT_365_DAY_WINDOWS"
    manifest["total_candles_expected"] = CANDLES_TOTAL
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
