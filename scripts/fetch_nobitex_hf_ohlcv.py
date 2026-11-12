"""Fetch native 15m/30m Nobitex candles using the provider's actual start boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from scripts import fetch_nobitex_public_ohlcv as source

INTRADAY_SEARCH_START_TIMESTAMP = 1_647_820_800  # 2022-03-21T00:00:00Z
INTRADAY_END_TIMESTAMP = 1_750_192_200  # 2025-06-17T20:30:00Z
INTRADAY_INTERVALS = ("15m", "30m")
COUNTBACK_PROBE_SLEEP_SECONDS = 1.1


def _probe_last_native_candle(
    *,
    interval: str,
    to_timestamp: int,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    minimum_timestamp: int,
) -> int | None:
    """Return the last provider candle at or before a timestamp, if any."""
    seconds = source.interval_seconds(interval)
    params = {
        "symbol": "BTCUSDT",
        "resolution": str(seconds // 60),
        "to": str(to_timestamp + 1),
        "countback": "1",
    }
    payload = request_json(params)
    rows = source.parse_udf_payload(
        payload,
        requested_from=0,
        requested_to=to_timestamp + 1,
    )
    eligible = [
        int(row["timestamp"]) for row in rows if row["timestamp"] >= minimum_timestamp
    ]
    return eligible[-1] if eligible else None


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


def _probe_with_rate_limit(
    *,
    interval: str,
    timestamp: int,
    minimum_timestamp: int,
    probe_number: int,
    request_json: Callable[[dict[str, str]], dict[str, Any]],
    sleep_fn: Callable[[float], None],
) -> int | None:
    if probe_number:
        sleep_fn(COUNTBACK_PROBE_SLEEP_SECONDS)
    return _probe_last_native_candle(
        interval=interval,
        to_timestamp=timestamp,
        request_json=request_json,
        minimum_timestamp=minimum_timestamp,
    )


def find_first_native_candle_timestamp(
    interval: str,
    *,
    start_timestamp: int = INTRADAY_SEARCH_START_TIMESTAMP,
    end_timestamp: int = INTRADAY_END_TIMESTAMP,
    request_json: Callable[
        [dict[str, str]], dict[str, Any]
    ] = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Binary-search the earliest real candle using UDF countback=1 probes.

    A final exact-range fetch still validates every candle and fails closed on
    any gap. The probe only discovers the boundary; it never fills missing data.
    """
    seconds = _validate_intraday_search_bounds(interval, start_timestamp, end_timestamp)
    probe_number = 0
    lower_candle = _probe_with_rate_limit(
        interval=interval,
        timestamp=start_timestamp,
        minimum_timestamp=start_timestamp,
        probe_number=probe_number,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    probe_number += 1
    if lower_candle is not None:
        return lower_candle

    upper_candle = _probe_with_rate_limit(
        interval=interval,
        timestamp=end_timestamp,
        minimum_timestamp=start_timestamp,
        probe_number=probe_number,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    probe_number += 1
    if upper_candle is None:
        raise ValueError(f"hf_intraday_no_native_candles:{interval}")

    low_index = 0
    high_index = (end_timestamp - start_timestamp) // seconds
    while low_index < high_index:
        midpoint_index = (low_index + high_index) // 2
        midpoint = start_timestamp + midpoint_index * seconds
        midpoint_candle = _probe_with_rate_limit(
            interval=interval,
            timestamp=midpoint,
            minimum_timestamp=start_timestamp,
            probe_number=probe_number,
            request_json=request_json,
            sleep_fn=sleep_fn,
        )
        probe_number += 1
        if midpoint_candle is None:
            low_index = midpoint_index + 1
        else:
            high_index = midpoint_index

    first_timestamp = start_timestamp + low_index * seconds
    first_candle = _probe_with_rate_limit(
        interval=interval,
        timestamp=first_timestamp,
        minimum_timestamp=start_timestamp,
        probe_number=probe_number,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    if first_candle is None or first_candle != first_timestamp:
        raise ValueError(f"hf_intraday_first_native_candle_not_resolved:{interval}")
    return first_candle


def fetch_native_intraday(
    interval: str,
) -> tuple[int, list[source.CandleRecord]]:
    """Discover and then fetch the exact native interval without resampling."""
    start_timestamp = find_first_native_candle_timestamp(interval)
    rows = source.fetch_candles_between(
        start_timestamp=start_timestamp,
        end_timestamp=INTRADAY_END_TIMESTAMP,
        symbol="BTCUSDT",
        interval=interval,
    )
    return start_timestamp, rows


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
            "availability_boundary": "FIRST_NATIVE_CANDLE_DISCOVERED_BY_UDF_COUNTBACK",
            "availability_search_lower_bound_timestamp": (
                INTRADAY_SEARCH_START_TIMESTAMP
            ),
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
