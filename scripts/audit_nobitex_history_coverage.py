"""Audit multi-year public Nobitex BTC/USDT hourly history coverage.

This data-availability audit performs no model training, backtest, private API call,
or order/execution request.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from http.client import HTTPException, HTTPSConnection
from pathlib import Path
import ssl
import time
from typing import Any, TypedDict
from urllib.parse import urlencode

HOST = "apiv2.nobitex.ir"
PATH = "/market/udf/history"
SCHEMA = "alix.nobitex.history-coverage.v1"
HOUR = 3600
MAX_RESPONSE_BARS = 400
WINDOW_BARS = 390
TIMEOUT_SECONDS = 15.0
RETRIES = 3
PAUSE_SECONDS = 1.1
LOOKBACK_YEARS = (3, 5, 10)


class AuditError(RuntimeError):
    """Raised when returned data cannot support a trustworthy coverage report."""


class Candle(TypedDict):
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def utc_iso(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def request_json(params: dict[str, str]) -> dict[str, Any]:
    request_path = f"{PATH}?{urlencode(params)}"
    for attempt in range(RETRIES):
        connection = HTTPSConnection(
            HOST,
            timeout=TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET",
                request_path,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-Public-History-Coverage-Audit",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                if attempt + 1 == RETRIES:
                    raise AuditError(f"public_api_http_status:{response.status}")
                time.sleep(2**attempt)
                continue
            try:
                payload = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise AuditError("public_api_json_invalid") from None
            if not isinstance(payload, dict):
                raise AuditError("public_api_payload_not_object")
            return payload
        except (HTTPException, OSError, TimeoutError) as error:
            if attempt + 1 == RETRIES:
                raise AuditError(
                    f"public_api_request_failed:{type(error).__name__}"
                ) from None
            time.sleep(2**attempt)
        finally:
            connection.close()
    raise AuditError("public_api_request_failed")


def finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AuditError(f"candle_value_invalid:{field}") from None
    if not math.isfinite(number):
        raise AuditError(f"candle_value_not_finite:{field}")
    return number



def _validated_ohlcv_arrays(payload: dict[str, Any]) -> list[list[Any]]:
    if payload.get("s") != "ok":
        raise AuditError("public_api_status_not_ok")
    names = ("t", "o", "h", "l", "c", "v")
    arrays = [payload.get(name) for name in names]
    if any(not isinstance(values, list) for values in arrays):
        raise AuditError("public_api_ohlcv_arrays_missing")
    columns = [values for values in arrays if isinstance(values, list)]
    if len({len(values) for values in columns}) != 1:
        raise AuditError("public_api_ohlcv_array_length_mismatch")
    return columns


def _validate_timestamp(
    value: Any, lower: int, upper: int, previous: int | None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError("candle_timestamp_invalid")
    if value < lower or value > upper:
        raise AuditError("candle_timestamp_out_of_requested_bounds")
    if previous is not None and value <= previous:
        raise AuditError("response_timestamps_not_strictly_increasing")
    return value


def _validate_candle_geometry(
    opened: float, high: float, low: float, close: float, volume: float
) -> None:
    if min(opened, high, low, close) <= 0:
        raise AuditError("candle_price_not_positive")
    if volume < 0:
        raise AuditError("candle_volume_negative")
    if high < max(opened, close):
        raise AuditError("candle_high_below_open_or_close")
    if low > min(opened, close):
        raise AuditError("candle_low_above_open_or_close")
    if high < low:
        raise AuditError("candle_high_below_low")


def _parse_candle_row(
    raw: tuple[Any, ...], lower: int, upper: int, previous: int | None
) -> Candle:
    timestamp = _validate_timestamp(raw[0], lower, upper, previous)
    opened = finite_number(raw[1], "open")
    high = finite_number(raw[2], "high")
    low = finite_number(raw[3], "low")
    close = finite_number(raw[4], "close")
    volume = finite_number(raw[5], "volume")
    _validate_candle_geometry(opened, high, low, close, volume)
    return {
        "timestamp": timestamp,
        "date": utc_iso(timestamp),
        "open": opened,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def parse_candles(
    payload: dict[str, Any], lower: int, upper: int
) -> list[Candle]:
    if payload.get("s") == "no_data":
        return []
    arrays = _validated_ohlcv_arrays(payload)
    rows = [
        _parse_candle_row(raw, lower, upper, previous)
        for raw, previous in _rows_with_previous(arrays)
    ]
    if len(rows) > MAX_RESPONSE_BARS:
        raise AuditError("public_api_response_exceeded_candle_limit")
    return rows


def _rows_with_previous(
    columns: list[list[Any]],
) -> list[tuple[tuple[Any, ...], int | None]]:
    previous: int | None = None
    result: list[tuple[tuple[Any, ...], int | None]] = []
    for raw in zip(*columns, strict=True):
        result.append((raw, previous))
        timestamp = raw[0]
        previous = timestamp if isinstance(timestamp, int) else None
    return result

def latest_closed_candle(now: int) -> int:
    lower = max(0, now - (MAX_RESPONSE_BARS - 10) * HOUR)
    params = {
        "symbol": "BTCUSDT",
        "resolution": "60",
        "from": str(lower),
        "to": str(now),
    }
    rows = parse_candles(request_json(params), lower, now)
    closed = [row["timestamp"] for row in rows if row["timestamp"] + HOUR <= now]
    if not closed:
        raise AuditError("latest_closed_candle_not_available")
    return max(closed)


def fetch_window(start: int, end: int) -> list[Candle]:
    lower = max(0, start - HOUR)
    upper = end + HOUR
    params = {
        "symbol": "BTCUSDT",
        "resolution": "60",
        "from": str(lower),
        "to": str(upper),
    }
    rows = parse_candles(request_json(params), lower, upper)
    return [row for row in rows if start <= row["timestamp"] <= end]


def segment_record(
    first: Candle, last: Candle, count: int
) -> dict[str, int | float | str]:
    hours = (last["timestamp"] - first["timestamp"]) / HOUR
    days = hours / 24.0
    return {
        "start_utc": first["date"],
        "end_utc": last["date"],
        "start_epoch": first["timestamp"],
        "end_epoch": last["timestamp"],
        "candle_count": count,
        "span_hours": hours,
        "span_days": days,
        "span_years_365_25_day": days / 365.25,
    }


def contiguous_segments(
    candles: list[Candle],
) -> list[dict[str, int | float | str]]:
    if not candles:
        return []
    result: list[dict[str, int | float | str]] = []
    first = previous = candles[0]
    count = 1
    for candle in candles[1:]:
        if candle["timestamp"] == previous["timestamp"] + HOUR:
            count += 1
        else:
            result.append(segment_record(first, previous, count))
            first = candle
            count = 1
        previous = candle
    result.append(segment_record(first, previous, count))
    return result


def find_gaps(candles: list[Candle]) -> list[dict[str, int | str]]:
    gaps: list[dict[str, int | str]] = []
    for previous, current in zip(candles, candles[1:]):
        difference = current["timestamp"] - previous["timestamp"]
        if difference <= HOUR:
            continue
        if difference % HOUR != 0:
            raise AuditError("candle_timestamps_not_on_hourly_grid")
        first_missing = previous["timestamp"] + HOUR
        last_missing = current["timestamp"] - HOUR
        gaps.append(
            {
                "first_missing_utc": utc_iso(first_missing),
                "last_missing_utc": utc_iso(last_missing),
                "missing_hour_count": difference // HOUR - 1,
            }
        )
    return gaps


def write_candles(path: Path, candles: list[Candle]) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("timestamp", "date", "open", "high", "low", "close", "volume"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(candles)
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _merge_window_batch(
    found: dict[int, Candle], batch: list[Candle], latest: int
) -> None:
    for candle in batch:
        stamp = candle["timestamp"]
        if (stamp - latest) % HOUR != 0:
            raise AuditError("candle_timestamp_not_aligned_to_hourly_grid")
        if stamp in found:
            raise AuditError("duplicate_candle_across_windows")
        found[stamp] = candle


def _collect_history(
    start: int, latest: int
) -> tuple[dict[int, Candle], list[dict[str, int | str]]]:
    found: dict[int, Candle] = {}
    windows: list[dict[str, int | str]] = []
    end = latest
    while end >= start:
        begin = max(start, end - (WINDOW_BARS - 1) * HOUR)
        batch = fetch_window(begin, end)
        _merge_window_batch(found, batch, latest)
        windows.append(
            {
                "start_utc": utc_iso(begin),
                "end_utc": utc_iso(end),
                "returned_candle_count": len(batch),
            }
        )
        if len(windows) % 10 == 0:
            print(
                f"scan_progress windows={len(windows)} candles={len(found)}",
                flush=True,
            )
        end = begin - HOUR
        if end >= start:
            time.sleep(PAUSE_SECONDS)
    return found, windows


def _coverage_summary(
    lookback_years: int,
    start: int,
    latest: int,
    candles: list[Candle],
    windows: list[dict[str, int | str]],
) -> dict[str, Any]:
    gaps = find_gaps(candles)
    segments = contiguous_segments(candles)
    longest = max(segments, key=lambda row: int(row["candle_count"]))
    recent = next(
        (row for row in segments if int(row["end_epoch"]) == latest),
        None,
    )
    first_stamp = candles[0]["timestamp"]
    last_stamp = candles[-1]["timestamp"]
    expected = lookback_years * 365 * 24
    leading = (first_stamp - start) // HOUR
    trailing = (latest - last_stamp) // HOUR
    missing_inside = (last_stamp - first_stamp) // HOUR + 1 - len(candles)
    full = (
        len(candles) == expected
        and first_stamp == start
        and last_stamp == latest
        and not gaps
    )
    return {
        "audit_status": "COMPLETED",
        "coverage_verdict": "GAP_FREE_FULL_LOOKBACK"
        if full
        else "PARTIAL_OR_GAPPED_HISTORY",
        "lookback_years_requested": lookback_years,
        "lookback_days_requested": lookback_years * 365,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "source": "Nobitex public UDF history API",
        "endpoint": f"https://{HOST}{PATH}",
        "scan_start_utc": utc_iso(start),
        "scan_end_utc": utc_iso(latest),
        "oldest_observed_candle_utc": candles[0]["date"],
        "newest_observed_candle_utc": candles[-1]["date"],
        "expected_hourly_slots": expected,
        "observed_candle_count": len(candles),
        "unobserved_slots_in_requested_window": expected - len(candles),
        "leading_unobserved_hours_before_oldest_observation": leading,
        "trailing_unobserved_hours_after_newest_observation": trailing,
        "missing_hours_between_observations": missing_inside,
        "inter_observation_gap_count": len(gaps),
        "empty_window_count": sum(
            int(w["returned_candle_count"]) == 0 for w in windows
        ),
        "windows_completed": len(windows),
        "public_market_data_request_count": len(windows) + 1,
        "full_requested_lookback_is_contiguous": full,
        "contiguous_segment_count": len(segments),
        "inter_observation_gaps": gaps,
        "contiguous_segments": segments,
        "longest_contiguous_segment": longest,
        "longest_contiguous_span_years": longest["span_years_365_25_day"],
        "contiguous_segment_ending_at_latest_candle": recent,
        "recent_contiguous_span_years": recent["span_years_365_25_day"]
        if recent
        else 0.0,
        "window_results": windows,
    }


def scan(lookback_years: int) -> tuple[dict[str, Any], list[Candle], int]:
    now = int(datetime.now(timezone.utc).timestamp())
    latest = latest_closed_candle(now)
    expected = lookback_years * 365 * 24
    start = latest - (expected - 1) * HOUR
    found, windows = _collect_history(start, latest)
    candles = [found[stamp] for stamp in sorted(found)]
    if not candles:
        raise AuditError("no_candles_found_in_requested_lookback")
    if latest not in found:
        raise AuditError("latest_probe_candle_not_recovered_by_full_scan")
    coverage = _coverage_summary(lookback_years, start, latest, candles, windows)
    return coverage, candles, latest

def safety_flags() -> dict[str, Any]:
    return {
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "order_endpoint_calls": 0,
        "real_order_attempts": 0,
        "model_training_performed": False,
        "backtest_performed": False,
        "Champion": None,
        "economic_acceptance": "NO-GO",
        "promotion_status": "BLOCKED_NO_STRATEGY_LEVEL_EVIDENCE",
        "production_model_updated": False,
        "strategy_registry_updated": False,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def append_step_summary(report: dict[str, Any]) -> None:
    summary_value = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_value:
        return
    lines = [
        "# Nobitex public history coverage audit",
        "",
        f"- Audit status: {report.get('audit_status', 'ERROR')}",
        f"- Coverage verdict: {report.get('coverage_verdict', 'UNCONFIRMED')}",
        f"- Requested lookback: {report.get('lookback_years_requested', 'unknown')} years",
        f"- Observed candles: {report.get('observed_candle_count', 0)}",
        f"- Oldest observed candle: {report.get('oldest_observed_candle_utc', 'unknown')}",
        f"- Newest observed candle: {report.get('newest_observed_candle_utc', 'unknown')}",
        f"- Gap-free span ending at latest candle: {report.get('recent_contiguous_span_years', 0)} years",
        f"- Longest contiguous segment in the scan: {report.get('longest_contiguous_span_years', 0)} years",
        f"- Inter-observation gap intervals: {report.get('inter_observation_gap_count', 0)}",
        f"- Missing hours between observations: {report.get('missing_hours_between_observations', 0)}",
        "",
        "Safety flags: Champion=null, order_permission=false, "
        "trading_authorization=NO-GO, economic_acceptance=NO-GO.",
        "",
        "Public OHLCV retrieval only. No model training or backtest is performed.",
        "",
    ]
    with Path(summary_value).open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_audit(lookback_years: int, commit_sha: str, output_dir: Path) -> dict[str, Any]:
    if lookback_years not in LOOKBACK_YEARS:
        raise AuditError("lookback_years_out_of_policy")
    if len(commit_sha) != 40 or any(c not in "0123456789abcdef" for c in commit_sha):
        raise AuditError("commit_sha_invalid")

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage, candles, latest = scan(lookback_years)
    csv_path = output_dir / "nobitex-btcusdt-1h-history-coverage.csv"
    csv_hash = write_candles(csv_path, candles)
    manifest = {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "csv_sha256": csv_hash,
        "safety_invariants": safety_flags(),
    }
    manifest_path = output_dir / "nobitex-btcusdt-1h-history-coverage.manifest.json"
    manifest_hash = write_json(manifest_path, manifest)
    report = {
        "schema_version": SCHEMA,
        **coverage,
        "latest_closed_candle_epoch": latest,
        "commit_sha": commit_sha,
        "csv_sha256": csv_hash,
        "manifest_sha256": manifest_hash,
        **safety_flags(),
    }
    write_json(output_dir / "nobitex-history-coverage-report.json", report)
    append_step_summary(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-years", type=int, choices=LOOKBACK_YEARS, default=10
    )
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_audit(args.lookback_years, args.commit_sha, args.output_dir)
    except (AuditError, OSError) as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": SCHEMA,
            "audit_status": "ERROR",
            "error_reason": str(error) or type(error).__name__,
            "lookback_years_requested": args.lookback_years,
            "commit_sha": args.commit_sha,
            **safety_flags(),
        }
        write_json(args.output_dir / "nobitex-history-coverage-report.json", report)
        append_step_summary(report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
