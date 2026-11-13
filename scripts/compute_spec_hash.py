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


def _load_spec(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("holdout_spec_must_be_json_object")
    return cast(dict[str, Any], parsed)


def _validate_spec_identity(spec: dict[str, Any]) -> None:
    if spec.get("spec_version") != 2:
        raise ValueError("holdout_spec_version_not_v2")
    if spec.get("grid_convention") != GRID_CONVENTION:
        raise ValueError("holdout_spec_grid_convention_invalid")
    if spec.get("timeframe") != "1h":
        raise ValueError("holdout_spec_timeframe_not_1h")


def _parse_spec_window(spec: dict[str, Any]) -> tuple[int, int]:
    window = spec.get("window")
    if not isinstance(window, dict):
        raise ValueError("holdout_spec_window_missing")
    start = _timestamp(window.get("start_utc"), "start_utc")
    end = _timestamp(window.get("end_utc"), "end_utc")
    return start, end


def _validate_spec_grid(start: int, end: int) -> None:
    if end < start or (end - start) % 3600:
        raise ValueError("holdout_spec_window_not_hourly_aligned")
    if start % 3600 != 1800 or end % 3600 != 1800:
        raise ValueError("holdout_spec_not_on_venue_grid")


def _validate_expected_count(spec: dict[str, Any], start: int, end: int) -> None:
    expected = (end - start) // 3600 + 1
    if expected != EXPECTED_CANDLE_COUNT:
        raise ValueError("holdout_spec_expected_candle_count_mismatch")
    if spec.get("expected_candle_count") != expected:
        raise ValueError("holdout_spec_expected_candle_count_mismatch")


def _validate_prehack_separation(spec: dict[str, Any], start: int) -> None:
    prehack = spec.get("prehack_reference")
    if not isinstance(prehack, dict):
        raise ValueError("holdout_spec_prehack_reference_missing")
    if prehack.get("end_timestamp") != public_fetcher.PREHACK_END_TIMESTAMP:
        raise ValueError("holdout_spec_prehack_boundary_mismatch")
    minimum_separation = start - public_fetcher.PREHACK_END_TIMESTAMP
    if minimum_separation != 1_137_600:
        raise ValueError("holdout_spec_minimum_separation_mismatch")
    if prehack.get("minimum_separation_seconds") != minimum_separation:
        raise ValueError("holdout_spec_minimum_separation_mismatch")


def compute_spec_hash(path: Path = SPEC_PATH) -> str:
    spec = _load_spec(path)
    _validate_spec_identity(spec)
    start, end = _parse_spec_window(spec)
    _validate_spec_grid(start, end)
    _validate_expected_count(spec, start, end)
    _validate_prehack_separation(spec, start)
    return hashlib.sha256(canonical_spec_bytes(spec)).hexdigest()


def main() -> int:
    print(compute_spec_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
