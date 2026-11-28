from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.analyze_m8_microstructure import (
    InputError,
    analyze_archives,
    compute_features,
    correlation_summary,
)


def _book_row() -> dict[str, str]:
    bids = [["100", "2"], ["99", "2"], ["98", "3"], ["97", "4"], ["96", "5"]]
    asks = [["102", "1"], ["103", "1"], ["104", "1"], ["105", "1"], ["106", "1"]]
    return {
        "captured_at_utc": "2026-11-25T05:35:38.982+00:00",
        "source": "websocket",
        "exchange_timestamp_ms": "1795584938982",
        "offset": "1",
        "best_bid_price": "100",
        "best_bid_size": "2",
        "best_ask_price": "102",
        "best_ask_size": "1",
        "spread": "2",
        "mid_price": "101",
        "bids_top_json": json.dumps(bids),
        "asks_top_json": json.dumps(asks),
    }


def _write_csv(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _make_archive(tmp_path: Path, market: str) -> tuple[Path, Path]:
    row = _book_row()
    book_fields = list(row)
    trade_fields = [
        "captured_at_utc",
        "source",
        "exchange_timestamp_ms",
        "side",
        "price",
        "volume",
        "offset",
    ]
    gap_fields = [
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
    ]
    files = {
        "orderbook.csv": _write_csv([row], book_fields),
        "trades.csv": _write_csv([], trade_fields),
        "gaps.csv": _write_csv([], gap_fields),
    }
    metadata = {
        name: {
            "sha256": hashlib.sha256(contents).hexdigest(),
            "bytes": len(contents),
            "records": 1 if name == "orderbook.csv" else 0,
        }
        for name, contents in files.items()
    }
    manifest = {
        "market": market,
        "workflow_run_id": "123",
        "capture_status": "NO_GAPS_OBSERVED",
        "record_counts": {"orderbook": 1, "trades": 0, "gaps": 0},
        "gap_count": 0,
        "files": metadata,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    archive_path = tmp_path / "m8-test.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr(
            "manifest.sha256",
            f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json\n",
        )
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    inventory = {
        "expected_archive_count": 1,
        "artifacts_by_sha256": {
            digest: {
                "run_id": 123,
                "artifact_id": 456,
                "market": market,
                "orderbook_rows": 1,
            }
        },
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return archive_path, inventory_path


def test_compute_features_uses_top_five_depth_and_top_level_microprice() -> None:
    mid, features = compute_features(_book_row())

    assert mid == 101.0
    assert features["obi_top5"] == pytest.approx(11 / 21)
    assert features["microprice_edge_bps"] == pytest.approx(
        ((102 * 2 + 100 * 1) / 3 / 101 - 1) * 10_000
    )
    assert features["spread_bps"] == pytest.approx(2 / 101 * 10_000)
    assert features["depth_top5"] == pytest.approx(21.0)


def test_correlation_summary_reports_perfect_rank_and_linear_relationship() -> None:
    result = correlation_summary([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])

    assert result["n"] == 4
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pearson_p_two_sided"] == 0.0


def test_market_mismatch_fails_closed_without_computing_correlations(
    tmp_path: Path,
) -> None:
    archive, inventory = _make_archive(tmp_path, "BTCIRT")

    report = analyze_archives([archive], inventory, expected_market="BTCUSDT")

    assert report["status"] == "BLOCKED_MARKET_MISMATCH"
    assert report["correlation_status"] == "NOT_COMPUTED"
    assert report["correlations"] == {}
    assert report["decision"] == "NEW_BTCUSDT_CAPTURE_REQUIRED"


def test_market_mismatch_does_not_compute_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, inventory = _make_archive(tmp_path, "BTCIRT")

    def fail_if_called(_row: dict[str, str]) -> tuple[float, dict[str, float]]:
        raise AssertionError("feature computation must not run for a market mismatch")

    monkeypatch.setattr(
        "scripts.analyze_m8_microstructure.compute_features", fail_if_called
    )
    report = analyze_archives([archive], inventory, expected_market="BTCUSDT")

    assert report["status"] == "BLOCKED_MARKET_MISMATCH"
    assert report["correlation_status"] == "NOT_COMPUTED"


def test_btcusdt_archive_passes_integrity_and_run_id_type_normalization(
    tmp_path: Path,
) -> None:
    archive, inventory = _make_archive(tmp_path, "BTCUSDT")

    report = analyze_archives([archive], inventory, expected_market="BTCUSDT")

    assert report["status"] == "ANALYZED"
    assert report["correlation_status"] == "COMPUTED"
    assert report["correlations"]["5m"]["obi_top5"]["n"] == 0


def test_unregistered_or_changed_archive_is_rejected(tmp_path: Path) -> None:
    archive, inventory = _make_archive(tmp_path, "BTCUSDT")
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(InputError, match="UNREGISTERED_ARCHIVE_SHA256"):
        analyze_archives([archive], inventory, expected_market="BTCUSDT")


def test_attach_trade_flow_imbalance_uses_websocket_rows_and_trailing_window() -> None:
    from scripts.analyze_m8_microstructure import Snapshot, _attach_trade_flow_imbalance

    snapshots = [Snapshot(100_000, 100.0, {})]
    trade_rows = [
        {"source": "websocket", "exchange_timestamp_ms": "41000", "side": "buy", "volume": "3"},
        {"source": "websocket", "exchange_timestamp_ms": "50000", "side": "sell", "volume": "1"},
        {"source": "websocket", "exchange_timestamp_ms": "39000", "side": "buy", "volume": "100"},
        {"source": "rest_recent_seed", "exchange_timestamp_ms": "99000", "side": "sell", "volume": "500"},
    ]

    _attach_trade_flow_imbalance(snapshots, trade_rows)

    assert snapshots[0].features["trade_flow_imbalance_60s"] == pytest.approx(0.5)


def test_attach_trade_flow_imbalance_no_volume_is_neutral_zero() -> None:
    from scripts.analyze_m8_microstructure import Snapshot, _attach_trade_flow_imbalance

    snapshots = [Snapshot(100_000, 100.0, {})]
    _attach_trade_flow_imbalance(snapshots, [])
    assert snapshots[0].features["trade_flow_imbalance_60s"] == 0.0
