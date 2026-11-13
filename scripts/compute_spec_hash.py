"""Compute the canonical SHA-256 of the preregistered post-hack window spec.

This script is intended to run in GitHub Actions only. It performs no network I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, cast

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
    start = _timestamp(window.get("start_utc"), "start_utc")
    end = _timestamp(window.get("end_utc"), "end_utc")
    timeframe = spec.get("timeframe")
    if timeframe != "1h":
        raise ValueError("holdout_spec_timeframe_not_1h")
    seconds = 3600
    if end < start or (end - start) % seconds:
        raise ValueError("holdout_spec_window_not_hourly_aligned")
    expected = (end - start) // seconds + 1
    if spec.get("expected_candle_count") != expected:
        raise ValueError("holdout_spec_expected_candle_count_mismatch")
    return hashlib.sha256(canonical_spec_bytes(spec)).hexdigest()


def main() -> int:
    print(compute_spec_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
