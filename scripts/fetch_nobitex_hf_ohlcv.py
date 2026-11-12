"""Fetch native 15m/30m Nobitex candles with backward UDF chunk scans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from scripts import fetch_nobitex_public_ohlcv as source

INTRADAY_SEARCH_START_TIMESTAMP = source.PREHACK_START_TIMESTAMP
INTRADAY_END_TIMESTAMP = source.PREHACK_END_TIMESTAMP
INTRADAY_INTERVALS = ("15m", "30m")
INTRADAY_CHUNK_CANDLES = 390
INTRADAY_CHUNK_SLEEP_SECONDS = 1.1


def _validate_intraday_search_bounds(
    interval: str, start_timestamp: int, end_timestamp: int
) -> int:
    if interval not in INTRADAY_INTERVALS:
        raise ValueError("hf_intraday_interval_out_of_policy")
    seconds = source.interval_seconds(interval)
    if end_timestamp <= start_timestamp:
        raise ValueError("hf_intraday_search_range_reversed")
    if start_timestamp % seconds or end_timestamp % seconds:
        raise ValueError("hf_intraday_search_range_not_interval_aligned")
    return seconds


def _fetch_intraday_chunk(
    interval: str,
    seconds: int,
    chunk_start: int,
    chunk_end: int,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
) -> list[source.CandleRecord]:
    params = {
        "symbol": "BTCUSDT",
        "resolution": str(seconds // 60),
        "from": str(chunk_start),
        "to": str(chunk_end + 1),
    }
    payload = request_json(params)
    return source.parse_udf_payload(
        payload,
        requested_from=chunk_start,
        requested_to=chunk_end + 1,
    )


def _merge_intraday_chunk(
    rows_by_timestamp: dict[int, source.CandleRecord],
    page_rows: list[source.CandleRecord],
    chunk_end: int,
    interval: str,
) -> None:
    for row in page_rows:
        stamp = int(row["timestamp"])
        if stamp > chunk_end:
            continue
        if stamp in rows_by_timestamp:
            raise ValueError(f"hf_intraday_duplicate_candle:{interval}:{stamp}")
        rows_by_timestamp[stamp] = row


def _validate_intraday_spacing(stamps: list[int], interval: str, seconds: int) -> None:
    if (stamps[-1] - stamps[0]) % seconds:
        raise ValueError(f"hf_intraday_gap_or_irregular_grid:{interval}")
    for previous, current in zip(stamps, stamps[1:], strict=False):
        if current - previous != seconds:
            raise ValueError(
                f"hf_intraday_gap_detected:{interval}:"
                f"previous={previous}:current={current}"
            )


def _validated_native_intraday_rows(
    rows_by_timestamp: dict[int, source.CandleRecord],
    interval: str,
    seconds: int,
    end_timestamp: int,
) -> tuple[int, list[source.CandleRecord]]:
    if not rows_by_timestamp:
        raise ValueError(f"hf_intraday_no_native_candles:{interval}")
    stamps = sorted(rows_by_timestamp)
    if stamps[-1] != end_timestamp:
        raise ValueError(f"hf_intraday_end_mismatch:{interval}")
    _validate_intraday_spacing(stamps, interval, seconds)
    expected_rows = ((end_timestamp - stamps[0]) // seconds) + 1
    if len(stamps) != expected_rows:
        raise ValueError(f"hf_intraday_row_count_mismatch:{interval}")
    return stamps[0], [rows_by_timestamp[stamp] for stamp in stamps]


def fetch_native_intraday(
    interval: str,
    *,
    start_timestamp: int = INTRADAY_SEARCH_START_TIMESTAMP,
    end_timestamp: int = INTRADAY_END_TIMESTAMP,
    request_json: Callable[
        [dict[str, str]], dict[str, Any]
    ] = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[int, list[source.CandleRecord]]:
    """Fetch a native interval backwards in explicit UDF from/to windows.

    Empty windows before the first available native candle are permitted. Any
    missing or irregular candle between returned observations is rejected after
    all chunks have been collected. The final cutoff must be present exactly.
    """
    seconds = _validate_intraday_search_bounds(interval, start_timestamp, end_timestamp)
    rows_by_timestamp: dict[int, source.CandleRecord] = {}
    cursor = end_timestamp
    while cursor >= start_timestamp:
        chunk_start = max(
            start_timestamp,
            cursor - (INTRADAY_CHUNK_CANDLES - 1) * seconds,
        )
        page_rows = _fetch_intraday_chunk(
            interval, seconds, chunk_start, cursor, request_json
        )
        _merge_intraday_chunk(rows_by_timestamp, page_rows, cursor, interval)
        cursor = chunk_start - seconds
        if cursor >= start_timestamp:
            sleep_fn(INTRADAY_CHUNK_SLEEP_SECONDS)

    return _validated_native_intraday_rows(
        rows_by_timestamp, interval, seconds, end_timestamp
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", choices=INTRADAY_INTERVALS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start_timestamp, rows = fetch_native_intraday(args.interval)
    interval_seconds = source.interval_seconds(args.interval)
    expected_rows = ((INTRADAY_END_TIMESTAMP - start_timestamp) // interval_seconds) + 1
    if len(rows) != expected_rows:
        raise ValueError(
            f"hf_intraday_expected_rows_mismatch:{args.interval}:"
            f"{expected_rows}:{len(rows)}"
        )
    manifest = source.write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=math.ceil((INTRADAY_END_TIMESTAMP - start_timestamp) / 86400),
        retrieved_at=now,
        commit_sha=args.commit_sha,
        symbol="BTCUSDT",
        interval=args.interval,
    )
    manifest.update(
        {
            "dataset_window": "PRE_HACK_NATIVE_INTRADAY_NO_BACKFILL",
            "availability_boundary": "BACKWARD_NATIVE_UDF_CHUNK_SCAN",
            "availability_search_lower_bound_timestamp": (
                INTRADAY_SEARCH_START_TIMESTAMP
            ),
            "request_direction": "BACKWARD_EXPLICIT_FROM_TO",
            "request_chunk_candles": INTRADAY_CHUNK_CANDLES,
            "start_timestamp": start_timestamp,
            "end_timestamp": INTRADAY_END_TIMESTAMP,
            "expected_candles": expected_rows,
            "row_count": len(rows),
            "requested_days_exact": (INTRADAY_END_TIMESTAMP - start_timestamp)
            / 86400.0,
            "no_resampling": True,
            "no_interpolation": True,
            "no_external_backfill": True,
        }
    )
    stem = f"nobitex-btcusdt-{args.interval}"
    (args.output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
