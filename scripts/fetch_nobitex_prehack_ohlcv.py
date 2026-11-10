"""Fetch the fixed pre-hack Nobitex BTCUSDT hourly dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from scripts import fetch_nobitex_public_ohlcv as source

PREHACK_START_TIMESTAMP = source.PREHACK_START_TIMESTAMP
PREHACK_END_TIMESTAMP = source.PREHACK_END_TIMESTAMP
REQUESTED_DAYS = source.PREHACK_REQUESTED_DAYS
EXPECTED_CANDLES = source.PREHACK_EXPECTED_CANDLES


def fetch_prehack_candles(
    *,
    request_json: Callable[
        [dict[str, str]], dict[str, Any]
    ] = source.request_public_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[source.CandleRecord]:
    return source.fetch_prehack_candles(
        request_json=request_json,
        sleep_fn=sleep_fn,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN"))
    args = parser.parse_args()
    retrieved_at = datetime.now(timezone.utc)
    rows = fetch_prehack_candles()
    manifest = source.write_artifacts(
        rows,
        output_dir=args.output_dir,
        days=REQUESTED_DAYS,
        retrieved_at=retrieved_at,
        commit_sha=args.commit_sha,
        symbol="BTCUSDT",
        interval="1h",
    )
    manifest.update(
        {
            "dataset_window": "PRE_HACK_NOBITEX_ONLY",
            "start_timestamp": PREHACK_START_TIMESTAMP,
            "end_timestamp": PREHACK_END_TIMESTAMP,
            "expected_candles": EXPECTED_CANDLES,
            "requested_days": REQUESTED_DAYS,
        }
    )
    manifest_path = args.output_dir / "nobitex-btcusdt-1h.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
