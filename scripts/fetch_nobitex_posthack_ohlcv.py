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


def _merge_page_records(
    records_by_timestamp: dict[int, source.CandleRecord],
    page_records: dict[int, source.CandleRecord],
) -> int:
    duplicate_count = 0
    for timestamp, row in page_records.items():
        if timestamp in records_by_timestamp:
            duplicate_count += 1
        else:
            records_by_timestamp[timestamp] = row
    return duplicate_count


def _fetch_all_pages(
    *,
    start_timestamp: int,
    end_timestamp: int,
    symbol: str,
    request_json: JsonProvider,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[int, source.CandleRecord], int]:
    seconds = source.interval_seconds("1h")
    records_by_timestamp: dict[int, source.CandleRecord] = {}
    duplicate_count = 0
    cursor = start_timestamp

    while cursor <= end_timestamp:
        page_end = min(
            cursor + (source.MAX_CANDLES_PER_REQUEST - 1) * seconds,
            end_timestamp,
        )
        _page_expected, page_records, _page_missing = source._fetch_page_with_retries(
            start=cursor,
            end=page_end,
            seconds=seconds,
            symbol=symbol,
            request_json=request_json,
            sleep_fn=sleep_fn,
        )
        duplicate_count += _merge_page_records(records_by_timestamp, page_records)
        cursor = page_end + seconds
        if cursor <= end_timestamp:
            sleep_fn(1.1)

    return records_by_timestamp, duplicate_count


def _validate_record_timestamps(
    records_by_timestamp: dict[int, source.CandleRecord],
    *,
    start_timestamp: int,
    end_timestamp: int,
    seconds: int,
) -> list[dict[str, int]]:
    off_grid: list[dict[str, int]] = []
    for timestamp in records_by_timestamp:
        if timestamp < start_timestamp or timestamp > end_timestamp:
            raise ValueError("posthack_timestamp_out_of_bounds")
        offset = (timestamp - start_timestamp) % seconds
        if offset:
            off_grid.append({"timestamp_epoch": timestamp, "offset_seconds": offset})
    return sorted(off_grid, key=lambda item: item["timestamp_epoch"])


def fetch_window(
    *,
    start_timestamp: int,
    end_timestamp: int,
    symbol: str = "BTCUSDT",
    request_json: JsonProvider = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[source.CandleRecord], list[int], list[dict[str, int]], int]:
    validate_window(start_timestamp, end_timestamp)
    source._validate_symbol(symbol)
    seconds = source.interval_seconds("1h")
    records_by_timestamp, duplicate_count = _fetch_all_pages(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        symbol=symbol,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    off_grid = _validate_record_timestamps(
        records_by_timestamp,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        seconds=seconds,
    )
    expected_timestamps = range(start_timestamp, end_timestamp + 1, seconds)
    missing = [
        timestamp
        for timestamp in expected_timestamps
        if timestamp not in records_by_timestamp
    ]
    rows = [
        records_by_timestamp[timestamp] for timestamp in sorted(records_by_timestamp)
    ]
    return rows, missing, off_grid, duplicate_count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _grid_integrity_status(gap_count: int, off_grid_count: int) -> str:
    if gap_count and off_grid_count:
        return "GAP_AND_GRID_ANOMALY"
    if off_grid_count:
        return "GRID_ANOMALY"
    if gap_count:
        return "GAP_DETECTED"
    return "COMPLETE"


def _build_structural_manifest(
    *,
    status: str,
    symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    expected_count: int,
    actual_count: int,
    gap_count: int,
    off_grid_count: int,
    grid_anomaly: bool,
    distinct_offsets: list[int],
    duplicate_count: int,
    retrieved_at: datetime,
    run_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "alix.nobitex.posthack-structural-manifest.v2",
        "status": status,
        "venue": "Nobitex Spot",
        "symbol": symbol,
        "timeframe": "1h",
        "endpoint": source.ENDPOINT,
        "start_utc": _utc_iso(start_timestamp),
        "end_utc": _utc_iso(end_timestamp),
        "expected_row_count": expected_count,
        "actual_row_count": actual_count,
        "row_count": actual_count,
        "gap_count": gap_count,
        "off_grid_count": off_grid_count,
        "grid_anomaly": grid_anomaly,
        "distinct_off_grid_offsets_seconds": distinct_offsets,
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


def _write_grid_integrity_report(
    *,
    output_dir: Path,
    start_timestamp: int,
    end_timestamp: int,
    expected_count: int,
    actual_count: int,
    missing: list[int],
    off_grid: list[dict[str, int]],
    distinct_offsets: list[int],
    run_id: str,
    commit_sha: str,
) -> None:
    _write_json(
        output_dir / GAP_NAME,
        {
            "schema_version": "alix.nobitex.posthack-grid-integrity-report.v2",
            "start_utc": _utc_iso(start_timestamp),
            "end_utc": _utc_iso(end_timestamp),
            "expected_row_count": expected_count,
            "actual_row_count": actual_count,
            "gap_count": len(missing),
            "missing_timestamps_epoch": missing,
            "missing_timestamps_utc": [_utc_iso(stamp) for stamp in missing],
            "off_grid_count": len(off_grid),
            "grid_anomaly": bool(off_grid),
            "distinct_off_grid_offsets_seconds": distinct_offsets,
            "off_grid_timestamps": [
                {
                    "timestamp_epoch": item["timestamp_epoch"],
                    "timestamp_utc": _utc_iso(item["timestamp_epoch"]),
                    "offset_seconds": item["offset_seconds"],
                }
                for item in off_grid
            ],
            "endpoint": source.ENDPOINT,
            "run_id": run_id,
            "commit_sha": commit_sha,
            "backfill": False,
            "interpolation": False,
            "masking": False,
        },
    )


def _write_complete_dataset(
    *,
    output_dir: Path,
    rows: list[source.CandleRecord],
    start_timestamp: int,
    end_timestamp: int,
    expected_count: int,
) -> str:
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
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()


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

    rows, missing, off_grid, duplicate_count = fetch_window(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        symbol=symbol,
        request_json=request_json,
        sleep_fn=sleep_fn,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_count = len(missing)
    off_grid_count = len(off_grid)
    distinct_offsets = sorted({item["offset_seconds"] for item in off_grid})
    grid_anomaly = off_grid_count > 0
    manifest = _build_structural_manifest(
        status=_grid_integrity_status(gap_count, off_grid_count),
        symbol=symbol,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        expected_count=expected_count,
        actual_count=len(rows),
        gap_count=gap_count,
        off_grid_count=off_grid_count,
        grid_anomaly=grid_anomaly,
        distinct_offsets=distinct_offsets,
        duplicate_count=duplicate_count,
        retrieved_at=retrieved_at,
        run_id=run_id,
        commit_sha=commit_sha,
    )

    if gap_count or grid_anomaly:
        _write_grid_integrity_report(
            output_dir=output_dir,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            expected_count=expected_count,
            actual_count=len(rows),
            missing=missing,
            off_grid=off_grid,
            distinct_offsets=distinct_offsets,
            run_id=run_id,
            commit_sha=commit_sha,
        )
    else:
        manifest["dataset_sha256"] = _write_complete_dataset(
            output_dir=output_dir,
            rows=rows,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            expected_count=expected_count,
        )

    _write_json(output_dir / MANIFEST_NAME, manifest)
    return {
        "status": manifest["status"],
        "expected_row_count": expected_count,
        "actual_row_count": len(rows),
        "gap_count": gap_count,
        "off_grid_count": off_grid_count,
        "grid_anomaly": grid_anomaly,
        "distinct_off_grid_offsets_seconds": distinct_offsets,
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "gap_report_path": str(output_dir / GAP_NAME)
        if gap_count or grid_anomaly
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
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
    return 2 if summary["gap_count"] or summary["off_grid_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
