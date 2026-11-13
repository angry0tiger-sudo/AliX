"""Compute the canonical SHA-256 of the preregistered post-hack window spec.

This script is intended to run in GitHub Actions only. It performs no network I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from scripts import fetch_nobitex_public_ohlcv as public_fetcher

GRID_CONVENTION = (
    "hourly candles timestamped at :30 UTC (Iran UTC+3:30 local hour boundaries)"
)
EXPECTED_CANDLE_COUNT = 11_713
SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "holdout"
    / "posthack_window_spec.json"
)


def _timestamp(value: object, field: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"holdout_spec_invalid_{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"holdout_spec_invalid_{field}") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"holdout_spec_timezone_must_be_utc_{field}")
    return int(parsed.timestamp())


def canonical_spec_bytes(spec: dict[str, Any]) -> bytes:
    return json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_spec_hash(path: Path = SPEC_PATH) -> str:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("holdout_spec_must_be_json_object")
    spec = cast(dict[str, Any], parsed)
    window = spec.get("window")
    if not isinstance(window, dict):
        raise ValueError("holdout_spec_window_missing")
    if spec.get("spec_version") != 2:
        raise ValueError("holdout_spec_version_not_v2")
    if spec.get("grid_convention") != GRID_CONVENTION:
        raise ValueError("holdout_spec_grid_convention_invalid")
    start = _timestamp(window.get("start_utc"), "start_utc")
    end = _timestamp(window.get("end_utc"), "end_utc")
    timeframe = spec.get("timeframe")
    if timeframe != "1h":
        raise ValueError("holdout_spec_timeframe_not_1h")
    seconds = 3600
    if end < start or (end - start) % seconds:
        raise ValueError("holdout_spec_window_not_hourly_aligned")
    if start % seconds != 1800 or end % seconds != 1800:
        raise ValueError("holdout_spec_not_on_venue_grid")
    expected = (end - start) // seconds + 1
    if (
        expected != EXPECTED_CANDLE_COUNT
        or spec.get("expected_candle_count") != expected
    ):
        raise ValueError("holdout_spec_expected_candle_count_mismatch")
    prehack = spec.get("prehack_reference")
    if not isinstance(prehack, dict):
        raise ValueError("holdout_spec_prehack_reference_missing")
    if prehack.get("end_timestamp") != public_fetcher.PREHACK_END_TIMESTAMP:
        raise ValueError("holdout_spec_prehack_boundary_mismatch")
    minimum_separation = start - public_fetcher.PREHACK_END_TIMESTAMP
    if (
        minimum_separation != 1_137_600
        or prehack.get("minimum_separation_seconds") != minimum_separation
    ):
        raise ValueError("holdout_spec_minimum_separation_mismatch")
    return hashlib.sha256(canonical_spec_bytes(spec)).hexdigest()


def main() -> int:
    print(compute_spec_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
