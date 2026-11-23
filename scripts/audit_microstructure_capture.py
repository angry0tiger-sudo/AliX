"""Independent in-run audit for Regime M microstructure capture artifacts.

Only reads the public-feed CSVs and manifest in the supplied artifact directory.
It performs no network access and does not access credentials, models, Holdout or Post-Hack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "orderbook.csv": (
        "captured_at_utc",
        "source",
        "exchange_timestamp_ms",
        "offset",
        "best_bid_price",
        "best_bid_size",
        "best_ask_price",
        "best_ask_size",
        "spread",
        "mid_price",
        "bids_top_json",
        "asks_top_json",
    ),
    "trades.csv": (
        "captured_at_utc",
        "source",
        "exchange_timestamp_ms",
        "side",
        "price",
        "volume",
        "offset",
    ),
    "gaps.csv": (
        "stream",
        "gap_start_utc",
        "gap_end_utc",
        "gap_start_ms",
        "gap_end_ms",
        "reason",
        "previous_offset",
        "next_offset",
        "replay_attempted",
        "replay_available",
    ),
}
SAFETY_INVARIANTS: dict[str, object] = {
    "authorization_header_sent": False,
    "websocket_connection_token_sent": False,
    "private_api_calls": 0,
    "holdout_bytes_read": False,
    "posthack_data_read": False,
    "real_order_attempts": 0,
    "paper_orders": 0,
    "Champion": None,
    "order_permission": False,
    "trading_authorization": "NO-GO",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"AUTO_AUDIT_FAILED: {message}")


def _manifest_timestamp(value: object, field: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{field}_MISSING")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"AUTO_AUDIT_FAILED: {field}_INVALID") from exc
    _require(parsed.tzinfo is not None, f"{field}_MUST_BE_TIMEZONE_AWARE")
    return parsed


def _load_manifest(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = root / "manifest.json"
    checksum_path = root / "manifest.sha256"
    _require(manifest_path.is_file(), "manifest.json_MISSING")
    _require(checksum_path.is_file(), "manifest.sha256_MISSING")

    manifest_bytes = manifest_path.read_bytes()
    manifest_value: Any = json.loads(manifest_bytes.decode("utf-8"))
    _require(isinstance(manifest_value, dict), "MANIFEST_NOT_OBJECT")
    manifest = cast(dict[str, Any], manifest_value)

    digest = _sha256(manifest_bytes)
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split()
    _require(
        len(checksum_parts) == 2
        and checksum_parts[0] == digest
        and checksum_parts[1] == "manifest.json",
        "MANIFEST_SHA256_MISMATCH",
    )

    file_metadata_value = manifest.get("files")
    _require(isinstance(file_metadata_value, dict), "MANIFEST_FILES_MISSING")
    file_metadata = cast(dict[str, Any], file_metadata_value)
    _require(
        set(file_metadata) == set(CSV_FIELDS),
        "MANIFEST_CSV_FILE_SET_MISMATCH",
    )
    return manifest, file_metadata, digest


def _audit_csv_files(
    root: Path, file_metadata: dict[str, Any]
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    actual_counts: dict[str, int] = {}
    file_audit: dict[str, dict[str, Any]] = {}
    for filename, expected_fields in CSV_FIELDS.items():
        path = root / filename
        _require(path.is_file(), f"{filename}_MISSING")
        raw = path.read_bytes()
        digest = _sha256(raw)
        metadata_value = file_metadata[filename]
        _require(isinstance(metadata_value, dict), f"{filename}_METADATA_INVALID")
        metadata = cast(dict[str, Any], metadata_value)
        label = filename.removesuffix(".csv").upper()
        _require(metadata.get("sha256") == digest, f"{label}_CSV_SHA256_MISMATCH")
        _require(
            metadata.get("bytes") == len(raw),
            f"{label}_CSV_BYTE_COUNT_MISMATCH",
        )

        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                tuple(reader.fieldnames or ()) == expected_fields,
                f"{label}_CSV_HEADER_MISMATCH",
            )
            count = sum(1 for _ in reader)

        _require(
            metadata.get("records") == count,
            f"{label}_CSV_MANIFEST_RECORD_COUNT_MISMATCH",
        )
        actual_counts[filename] = count
        file_audit[filename] = {
            "records": count,
            "bytes": len(raw),
            "sha256": digest,
        }
    return actual_counts, file_audit


def _validate_record_counts(
    manifest: dict[str, Any], actual_counts: dict[str, int]
) -> int:
    record_counts_value = manifest.get("record_counts")
    _require(isinstance(record_counts_value, dict), "RECORD_COUNTS_MISSING")
    record_counts = cast(dict[str, Any], record_counts_value)
    count_keys = {
        "orderbook.csv": "orderbook",
        "trades.csv": "trades",
        "gaps.csv": "gaps",
    }
    for filename, key in count_keys.items():
        _require(
            record_counts.get(key) == actual_counts[filename],
            f"{key.upper()}_RECORD_COUNT_MISMATCH",
        )

    gap_count = actual_counts["gaps.csv"]
    _require(manifest.get("gap_count") == gap_count, "GAP_COUNT_MISMATCH")
    _require(actual_counts["orderbook.csv"] > 0, "ORDERBOOK_EMPTY")
    _require(actual_counts["trades.csv"] > 0, "TRADES_EMPTY")
    return gap_count


def _validate_reconnects(manifest: dict[str, Any]) -> dict[str, int]:
    fields = (
        "reconnect_attempts",
        "reconnect_successes",
        "reconnect_failures",
    )
    reconnects: dict[str, int] = {}
    for field in fields:
        value = manifest.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{field.upper()}_INVALID",
        )
        reconnects[field] = value
    attempts = reconnects["reconnect_attempts"]
    outcomes = reconnects["reconnect_successes"] + reconnects["reconnect_failures"]
    _require(attempts == outcomes, "RECONNECT_COUNTERS_INCONSISTENT")
    return reconnects


def _validate_safety(manifest: dict[str, Any]) -> None:
    for key, expected in SAFETY_INVARIANTS.items():
        _require(manifest.get(key) == expected, f"SAFETY_INVARIANT_{key}_VIOLATION")


def _capture_timestamps(
    manifest: dict[str, Any],
) -> tuple[datetime, datetime, float]:
    started = _manifest_timestamp(manifest.get("started_at_utc"), "started_at_utc")
    finished = _manifest_timestamp(manifest.get("finished_at_utc"), "finished_at_utc")
    actual_seconds = (finished - started).total_seconds()
    _require(actual_seconds > 0, "CAPTURE_DURATION_NOT_POSITIVE")
    return started, finished, actual_seconds


def audit_capture(root: Path) -> dict[str, Any]:
    manifest, file_metadata, manifest_digest = _load_manifest(root)
    actual_counts, file_audit = _audit_csv_files(root, file_metadata)
    gap_count = _validate_record_counts(manifest, actual_counts)
    reconnects = _validate_reconnects(manifest)
    _validate_safety(manifest)
    started, finished, actual_seconds = _capture_timestamps(manifest)

    return {
        "run_id": manifest.get("workflow_run_id"),
        "market": manifest.get("market"),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "actual_capture_seconds": round(actual_seconds, 3),
        "actual_capture_minutes": round(actual_seconds / 60, 3),
        "capture_status": manifest.get("capture_status"),
        "files": file_audit,
        "gaps": gap_count,
        "reconnects": reconnects,
        "manifest_sha256": manifest_digest,
        "safety_invariants": "PASS",
    }


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "## Regime M automatic artifact audit",
        f"- Run: {result['run_id']}",
        f"- Market: {result['market']}",
        f"- Manifest start: {result['started_at_utc']}",
        f"- Manifest end: {result['finished_at_utc']}",
        f"- Actual manifest capture: {result['actual_capture_minutes']} minutes ({result['actual_capture_seconds']} seconds)",
        f"- Capture status: {result['capture_status']}",
        "",
        "| CSV | Records | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for filename, item in result["files"].items():
        lines.append(
            f"| {filename} | {item['records']} | {item['bytes']} | {item['sha256']} |"
        )
    reconnects = result["reconnects"]
    lines.extend(
        [
            "",
            f"- Gap records: **{result['gaps']}**",
            f"- Reconnect attempts / successes / failures: **{reconnects['reconnect_attempts']} / {reconnects['reconnect_successes']} / {reconnects['reconnect_failures']}**",
            f"- Manifest SHA-256: {result['manifest_sha256']}",
            "- Safety invariants: **PASS**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Regime M capture CSVs and manifest."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    result = audit_capture(args.artifact_dir)
    summary = _summary(result)
    print(json.dumps(result, sort_keys=True))
    print(summary, end="")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
