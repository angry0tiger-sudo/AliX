"""Diagnose the public Nobitex hourly timestamp grid outside the sealed holdout.

Only structural metadata is written. Candle values are parsed in memory by the shared
post-hack fetch wrapper and are never printed or persisted by this diagnostic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts import fetch_nobitex_posthack_ohlcv as wrapper
from scripts import fetch_nobitex_public_ohlcv as source

DIAGNOSE_START_TIMESTAMP = 1_793_577_600
DIAGNOSE_END_TIMESTAMP = 1_794_441_600
HOLDOUT_END_TIMESTAMP = 1_793_491_200
ARTIFACT_NAME = "nobitex-grid-diagnose.json"
REQUEST_TIMEOUT_SECONDS = 15.0
JsonProvider = Callable[[dict[str, str]], dict[str, Any]]


def _utc_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def validate_diagnose_window(start_timestamp: int, end_timestamp: int) -> int:
    if (
        start_timestamp != DIAGNOSE_START_TIMESTAMP
        or end_timestamp != DIAGNOSE_END_TIMESTAMP
    ):
        raise ValueError("diagnose_window_not_owner_approved")
    if start_timestamp <= HOLDOUT_END_TIMESTAMP:
        raise ValueError("diagnose_window_overlaps_holdout")
    return wrapper.validate_window(start_timestamp, end_timestamp)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_public_json(
    params: dict[str, str], status: dict[str, int]
) -> dict[str, Any]:
    url = source.ENDPOINT + "?" + urlencode(params)
    opener = build_opener(_NoRedirect)
    try:
        response = opener.open(
            Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-Grid-Diagnose/1",
                },
                method="GET",
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        with response:
            status["http_status"] = int(response.status)
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        status["http_status"] = int(exc.code)
        exc.close()
        raise RuntimeError("diagnose_http_error") from None
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("diagnose_transport_or_payload_error") from None
    if not isinstance(payload, dict):
        raise ValueError("diagnose_payload_not_object")
    return payload


def _metadata(
    *,
    http_status: int,
    rows: list[source.CandleRecord],
    missing: list[int],
    off_grid: list[dict[str, int]],
) -> dict[str, Any]:
    timestamps = [row["timestamp"] for row in rows]
    return {
        "http_status": http_status,
        "candle_count": len(rows),
        "off_grid_count": len(off_grid),
        "distinct_offsets": sorted(
            {item["offset_seconds"] for item in off_grid}
        ),
        "first_ts": min(timestamps) if timestamps else None,
        "last_ts": max(timestamps) if timestamps else None,
        "gap_count": len(missing),
    }


def _write_artifact(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ARTIFACT_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_diagnosis(
    *,
    start_timestamp: int,
    end_timestamp: int,
    output_dir: Path,
    commit_sha: str,
    run_id: str,
    request_json: JsonProvider | None = None,
) -> tuple[dict[str, Any], bool]:
    validate_diagnose_window(start_timestamp, end_timestamp)
    status = {"http_status": 0}

    def tracked_provider(params: dict[str, str]) -> dict[str, Any]:
        if request_json is None:
            return _request_public_json(params, status)
        payload = request_json(params)
        status["http_status"] = 200
        return payload

    try:
        rows, missing, off_grid, _duplicate_count = wrapper.fetch_window(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            symbol="BTCUSDT",
            request_json=tracked_provider,
        )
        metrics = _metadata(
            http_status=status["http_status"],
            rows=rows,
            missing=missing,
            off_grid=off_grid,
        )
        error_code = None
        succeeded = True
    except Exception as exc:
        metrics = {
            "http_status": status["http_status"],
            "candle_count": 0,
            "off_grid_count": 0,
            "distinct_offsets": [],
            "first_ts": None,
            "last_ts": None,
            "gap_count": None,
        }
        error_code = type(exc).__name__
        succeeded = False

    record = {
        "schema_version": "alix.nobitex.grid-diagnose.v1",
        "mode": "DIAGNOSE",
        **metrics,
        "start_utc": _utc_iso(start_timestamp),
        "end_utc": _utc_iso(end_timestamp),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "timeframe": "1h",
        "endpoint": source.ENDPOINT,
        "run_id": run_id,
        "commit_sha": commit_sha,
        "error_code": error_code,
        "prices_persisted": False,
        "ohlcv_persisted": False,
    }
    _write_artifact(output_dir, record)
    return metrics, succeeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--commit-sha", default=os.environ.get("GITHUB_SHA", "UNKNOWN")
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "UNKNOWN"))
    args = parser.parse_args()

    try:
        metrics, succeeded = run_diagnosis(
            start_timestamp=args.start,
            end_timestamp=args.end,
            output_dir=args.output_dir,
            commit_sha=args.commit_sha,
            run_id=args.run_id,
        )
    except ValueError:
        metrics = {
            "http_status": 0,
            "candle_count": 0,
            "off_grid_count": 0,
            "distinct_offsets": [],
            "first_ts": None,
            "last_ts": None,
            "gap_count": None,
        }
        print(json.dumps(metrics, sort_keys=True, allow_nan=False))
        return 2

    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
