"""Fetch the owner-committed post-hack Nobitex window with a strict gap report.

The public pre-hack fetcher is imported but never modified. This wrapper is acquisition
infrastructure only; it does not train, fit, backtest, evaluate, or authorize holdout use.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from scripts import fetch_nobitex_public_ohlcv as source

SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "holdout"
    / "posthack_window_spec.json"
)
JsonProvider = Callable[[dict[str, str]], dict[str, Any]]
SECONDS_PER_HOUR = 3600
CSV_NAME = "nobitex-btcusdt-posthack-1h.csv"
MANIFEST_NAME = "nobitex-btcusdt-posthack-1h.manifest.json"
GAP_NAME = "nobitex-btcusdt-posthack-1h.gaps.json"


def _timestamp(value: object, field: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"posthack_invalid_{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"posthack_invalid_{field}") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"posthack_timezone_must_be_utc_{field}")
    return int(parsed.timestamp())


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("posthack_spec_must_be_json_object")
    return parsed


def validate_window(start_timestamp: int, end_timestamp: int) -> int:
    if isinstance(start_timestamp, bool) or isinstance(end_timestamp, bool):
        raise ValueError("posthack_timestamp_invalid")
    if start_timestamp <= source.PREHACK_END_TIMESTAMP:
        raise ValueError("posthack_start_must_be_strictly_after_prehack_end")
    if end_timestamp < start_timestamp:
        raise ValueError("posthack_window_reversed")
    if (end_timestamp - start_timestamp) % SECONDS_PER_HOUR:
        raise ValueError("posthack_window_not_hourly_aligned")
    if start_timestamp <= source.PREHACK_END_TIMESTAMP <= end_timestamp:
        raise ValueError("posthack_window_overlaps_prehack")
    return (end_timestamp - start_timestamp) // SECONDS_PER_HOUR + 1


def window_from_spec(spec: dict[str, Any]) -> tuple[int, int, int]:
    window = spec.get("window")
    if not isinstance(window, dict):
        raise ValueError("posthack_spec_window_missing")
    start = _timestamp(window.get("start_utc"), "start_utc")
    end = _timestamp(window.get("end_utc"), "end_utc")
    expected = validate_window(start, end)
    if spec.get("expected_candle_count") != expected:
        raise ValueError("posthack_spec_expected_count_mismatch")
    symbol = spec.get("symbol")
    timeframe = spec.get("timeframe")
    if symbol not in source.ALLOWED_SYMBOLS or timeframe != "1h":
        raise ValueError("posthack_spec_symbol_or_timeframe_out_of_policy")
    if source.PREHACK_END_TIMESTAMP != 1750192200:
        raise ValueError("posthack_prehack_boundary_changed")
    return start, end, expected


def _utc_iso(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def fetch_window(
    *,
    start_timestamp: int,
    end_timestamp: int,
    symbol: str = "BTCUSDT",
    request_json: JsonProvider = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[source.CandleRecord], list[int], int]:
    validate_window(start_timestamp, end_timestamp)
    source._validate_symbol(symbol)
    seconds = source.interval_seconds("1h")
    records_by_timestamp: dict[int, source.CandleRecord] = {}
    duplicate_count = 0
    cursor = start_timestamp

    while cursor <= end_timestamp:
        page_end = min(
            cursor + (source.MAX_CANDLES_PER_REQUEST - 1) * seconds,
            end_timestamp,
        )
        page_expected, page_records, _page_missing = source._fetch_page_with_retries(
            start=cursor,
            end=page_end,
            seconds=seconds,
            symbol=symbol,
            request_json=request_json,
            sleep_fn=sleep_fn,
        )
        if not page_expected:
            raise ValueError("posthack_empty_expected_page")
        for timestamp, row in page_records.items():
            if timestamp in records_by_timestamp:
                duplicate_count += 1
            else:
                records_by_timestamp[timestamp] = row
        cursor = page_end + seconds
        if cursor <= end_timestamp:
            sleep_fn(1.1)

    expected_timestamps = list(
        range(start_timestamp, end_timestamp + 1, seconds)
    )
    unexpected = [
        timestamp
        for timestamp in records_by_timestamp
        if timestamp < start_timestamp
        or timestamp > end_timestamp
        or (timestamp - start_timestamp) % seconds != 0
    ]
    if unexpected:
        raise ValueError("posthack_unexpected_or_off_grid_timestamp")
    missing = [
        timestamp
        for timestamp in expected_timestamps
        if timestamp not in records_by_timestamp
    ]
    rows = [
        records_by_timestamp[timestamp]
        for timestamp in sorted(records_by_timestamp)
    ]
    return rows, missing, duplicate_count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def fetch_and_write_artifacts(
    *,
    output_dir: Path,
    start_timestamp: int,
    end_timestamp: int,
    expected_count: int,
    retrieved_at: datetime,
    commit_sha: str,
    run_id: str,
    symbol: str = "BTCUSDT",
    request_json: JsonProvider = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("posthack_retrieved_at_timezone_required")
    calculated_count = validate_window(start_timestamp, end_timestamp)
    if expected_count != calculated_count:
        raise ValueError("posthack_expected_count_argument_mismatch")
    rows, missing, duplicate_count = fetch_window(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        symbol=symbol,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_count = len(missing)
    manifest: dict[str, Any] = {
        "schema_version": "alix.nobitex.posthack-structural-manifest.v1",
        "status": "GAP_DETECTED" if gap_count else "COMPLETE",
        "venue": "Nobitex Spot",
        "symbol": symbol,
        "timeframe": "1h",
        "endpoint": source.ENDPOINT,
        "start_utc": _utc_iso(start_timestamp),
        "end_utc": _utc_iso(end_timestamp),
        "expected_row_count": expected_count,
        "actual_row_count": len(rows),
        "row_count": len(rows),
        "gap_count": gap_count,
        "duplicate_timestamp_count": duplicate_count,
        "dataset_sha256": None,
        "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
        "run_id": run_id,
        "commit_sha": commit_sha,
        "gap_policy": "ZERO_TOLERANCE_NO_BACKFILL_NO_INTERPOLATION_NO_MASK",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }

    if gap_count:
        _write_json(
            output_dir / GAP_NAME,
            {
                "schema_version": "alix.nobitex.posthack-gap-report.v1",
                "start_utc": _utc_iso(start_timestamp),
                "end_utc": _utc_iso(end_timestamp),
                "expected_row_count": expected_count,
                "actual_row_count": len(rows),
                "gap_count": gap_count,
                "missing_timestamps_epoch": missing,
                "missing_timestamps_utc": [_utc_iso(stamp) for stamp in missing],
                "endpoint": source.ENDPOINT,
                "run_id": run_id,
                "commit_sha": commit_sha,
                "backfill": False,
                "interpolation": False,
                "masking": False,
            },
        )
    else:
        source._validate_exact_candle_coverage(
            rows,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            seconds=SECONDS_PER_HOUR,
        )
        csv_path = output_dir / CSV_NAME
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("date", "open", "high", "low", "close", "volume"),
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
        manifest["dataset_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    _write_json(output_dir / MANIFEST_NAME, manifest)
    return {
        "status": manifest["status"],
        "expected_row_count": expected_count,
        "actual_row_count": len(rows),
        "gap_count": gap_count,
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "gap_report_path": str(output_dir / GAP_NAME) if gap_count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN")
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "UNKNOWN"))
    args = parser.parse_args()

    spec = load_spec()
    start, end, expected = window_from_spec(spec)
    summary = fetch_and_write_artifacts(
        output_dir=args.output_dir,
        start_timestamp=start,
        end_timestamp=end,
        expected_count=expected,
        retrieved_at=datetime.now(timezone.utc),
        commit_sha=args.commit_sha,
        run_id=args.run_id,
        symbol=str(spec["symbol"]),
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 2 if summary["gap_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
