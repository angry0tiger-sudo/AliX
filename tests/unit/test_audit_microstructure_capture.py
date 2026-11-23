from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_microstructure_capture import audit_capture


HEADERS = {
    "orderbook.csv": [
        "captured_at_utc", "source", "exchange_timestamp_ms", "offset",
        "best_bid_price", "best_bid_size", "best_ask_price", "best_ask_size",
        "spread", "mid_price", "bids_top_json", "asks_top_json",
    ],
    "trades.csv": [
        "captured_at_utc", "source", "exchange_timestamp_ms", "side",
        "price", "volume", "offset",
    ],
    "gaps.csv": [
        "stream", "gap_start_utc", "gap_end_utc", "gap_start_ms", "gap_end_ms",
        "reason", "previous_offset", "next_offset", "replay_attempted",
        "replay_available",
    ],
}


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(root: Path) -> Path:
    root.mkdir()
    rows = {
        "orderbook.csv": [[
            "2026-11-23T00:00:00Z", "websocket", "1795392000000", "10",
            "100", "1", "101", "2", "1", "100.5", "[[\\"100\\",\\"1\\"]]",
            "[[\\"101\\",\\"2\\"]]",
        ]],
        "trades.csv": [[
            "2026-11-23T00:00:01Z", "websocket", "1795392001000", "buy",
            "100", "0.1", "20",
        ]],
        "gaps.csv": [],
    }
    files = {}
    for name, header in HEADERS.items():
        path = root / name
        _write_csv(path, header, rows[name])
        data = path.read_bytes()
        files[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "records": len(rows[name]),
        }
    manifest = {
        "workflow_run_id": "test-run",
        "market": "BTCIRT",
        "started_at_utc": "2026-11-23T00:00:00Z",
        "finished_at_utc": "2026-11-23T00:01:00Z",
        "record_counts": {"orderbook": 1, "trades": 1, "gaps": 0},
        "gap_count": 0,
        "files": files,
        "reconnect_attempts": 1,
        "reconnect_successes": 1,
        "reconnect_failures": 0,
        "capture_status": "NO_GAPS_OBSERVED",
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
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    (root / "manifest.json").write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8"
    )
    return root


def test_auto_audit_counts_records_hashes_and_reconnects(tmp_path: Path) -> None:
    result = audit_capture(_fixture(tmp_path / "capture"))

    assert result["files"]["orderbook.csv"]["records"] == 1
    assert result["files"]["trades.csv"]["records"] == 1
    assert result["files"]["gaps.csv"]["records"] == 0
    assert result["gaps"] == 0
    assert result["reconnects"] == {
        "reconnect_attempts": 1,
        "reconnect_successes": 1,
        "reconnect_failures": 0,
    }
    assert result["actual_capture_seconds"] == 60.0
    assert result["safety_invariants"] == "PASS"


def test_auto_audit_rejects_csv_hash_mismatch(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "capture")
    with (root / "trades.csv").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")

    with pytest.raises(ValueError, match="TRADES_CSV_SHA256_MISMATCH"):
        audit_capture(root)


def test_auto_audit_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "capture")
    (root / "manifest.sha256").write_text(
        "0" * 64 + "  manifest.json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="MANIFEST_SHA256_MISMATCH"):
        audit_capture(root)


def test_auto_audit_rejects_inconsistent_reconnect_counters(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "capture")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reconnect_attempts"] = 3
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="RECONNECT_COUNTERS_INCONSISTENT"):
        audit_capture(root)
