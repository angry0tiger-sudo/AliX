from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import fetch_nobitex_public_ohlcv as public_fetcher
from scripts import fetch_nobitex_posthack_ohlcv as posthack


def payload(timestamps: list[int]) -> dict[str, Any]:
    size = len(timestamps)
    return {
        "s": "ok",
        "t": timestamps,
        "o": [100.0] * size,
        "h": [101.0] * size,
        "l": [99.0] * size,
        "c": [100.5] * size,
        "v": [1.0] * size,
    }


def test_posthack_start_is_strictly_after_prehack_end() -> None:
    spec = posthack.load_spec()
    start, end, expected = posthack.window_from_spec(spec)
    assert start == 1_751_328_000
    assert end == 1_793_491_200
    assert expected == 11_713
    assert start > public_fetcher.PREHACK_END_TIMESTAMP


@pytest.mark.parametrize(
    "start",
    [
        public_fetcher.PREHACK_END_TIMESTAMP,
        public_fetcher.PREHACK_END_TIMESTAMP - 3600,
    ],
)
def test_wrapper_rejects_prehack_overlap(start: int) -> None:
    with pytest.raises(ValueError, match="strictly_after_prehack_end"):
        posthack.validate_window(start, start + 3600)


def test_fixed_window_has_zero_timestamp_overlap() -> None:
    spec = posthack.load_spec()
    start, end, _ = posthack.window_from_spec(spec)
    prehack_grid = set(
        range(
            public_fetcher.PREHACK_START_TIMESTAMP,
            public_fetcher.PREHACK_END_TIMESTAMP + 1,
            public_fetcher.PREHACK_INTERVAL_SECONDS,
        )
    )
    posthack_grid = set(range(start, end + 1, 3600))
    assert prehack_grid.isdisjoint(posthack_grid)


def test_wrapper_collects_full_gap_list_across_all_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(public_fetcher, "MAX_CANDLES_PER_REQUEST", 3)
    start = 1_751_328_000
    end = start + 6 * 3600
    missing = {start + 3600, start + 5 * 3600}
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params.copy())
        stamps = [
            stamp
            for stamp in range(int(params["from"]), int(params["to"]), 3600)
            if stamp not in missing
        ]
        return payload(stamps)

    rows, gaps, off_grid, duplicates = posthack.fetch_window(
        start_timestamp=start,
        end_timestamp=end,
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 5
    assert gaps == sorted(missing)
    assert off_grid == []
    assert duplicates == 0
    assert len(calls) == 7


def test_gap_artifact_is_structural_and_contains_every_missing_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(public_fetcher, "MAX_CANDLES_PER_REQUEST", 3)
    start = 1_751_328_000
    end = start + 6 * 3600
    missing = {start + 3600, start + 5 * 3600}

    def provider(params: dict[str, str]) -> dict[str, Any]:
        stamps = [
            stamp
            for stamp in range(int(params["from"]), int(params["to"]), 3600)
            if stamp not in missing
        ]
        return payload(stamps)

    summary = posthack.fetch_and_write_artifacts(
        output_dir=tmp_path,
        start_timestamp=start,
        end_timestamp=end,
        expected_count=7,
        retrieved_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        commit_sha="mock-sha",
        run_id="mock-run",
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    manifest = json.loads((tmp_path / posthack.MANIFEST_NAME).read_text())
    gap_report = json.loads((tmp_path / posthack.GAP_NAME).read_text())

    assert summary["status"] == "GAP_DETECTED"
    assert manifest["gap_count"] == 2
    assert manifest["off_grid_count"] == 0
    assert manifest["grid_anomaly"] is False
    assert manifest["actual_row_count"] == 5
    assert gap_report["missing_timestamps_epoch"] == sorted(missing)
    assert len(gap_report["missing_timestamps_utc"]) == 2
    assert gap_report["off_grid_count"] == 0
    assert gap_report["off_grid_timestamps"] == []
    assert "close" not in manifest
    assert not (tmp_path / posthack.CSV_NAME).exists()


def test_complete_mock_window_writes_hash_and_price_data_only_to_csv(
    tmp_path: Path,
) -> None:
    start = 1_751_328_000
    end = start + 2 * 3600

    def provider(params: dict[str, str]) -> dict[str, Any]:
        return payload(list(range(int(params["from"]), int(params["to"]), 3600)))

    summary = posthack.fetch_and_write_artifacts(
        output_dir=tmp_path,
        start_timestamp=start,
        end_timestamp=end,
        expected_count=3,
        retrieved_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        commit_sha="mock-sha",
        run_id="mock-run",
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    manifest = json.loads((tmp_path / posthack.MANIFEST_NAME).read_text())
    csv_text = (tmp_path / posthack.CSV_NAME).read_text()

    assert summary["status"] == "COMPLETE"
    assert manifest["gap_count"] == 0
    assert manifest["off_grid_count"] == 0
    assert manifest["grid_anomaly"] is False
    assert manifest["expected_row_count"] == manifest["actual_row_count"] == 3
    assert manifest["dataset_sha256"]
    assert "100.0" in csv_text
    assert "100.0" not in (tmp_path / posthack.MANIFEST_NAME).read_text()


def test_prehack_public_fetcher_contract_remains_importable_and_fixed() -> None:
    assert public_fetcher.ENDPOINT == ("https://apiv2.nobitex.ir/market/udf/history")
    assert public_fetcher.PREHACK_START_TIMESTAMP == 1_635_712_200
    assert public_fetcher.PREHACK_END_TIMESTAMP == 1_750_192_200
    assert public_fetcher.PREHACK_EXPECTED_CANDLES == 31_801


def test_wrapper_package_entrypoint_is_importable_without_network() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "nobitex-posthack.yml").read_text(
        encoding="utf-8"
    )
    assert "python -m scripts.fetch_nobitex_posthack_ohlcv" in workflow

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fetch_nobitex_posthack_ohlcv",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_off_grid_and_gap_are_reported_together_without_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(public_fetcher, "MAX_CANDLES_PER_REQUEST", 3)
    start = 1_751_328_000
    end = start + 6 * 3600
    missing = {start + 3600, start + 5 * 3600}
    off_grid_timestamp = start + 1800

    def provider(params: dict[str, str]) -> dict[str, Any]:
        stamps = [
            stamp
            for stamp in range(int(params["from"]), int(params["to"]), 3600)
            if stamp not in missing
        ]
        if int(params["from"]) == start:
            stamps.append(off_grid_timestamp)
        return payload(sorted(stamps))

    summary = posthack.fetch_and_write_artifacts(
        output_dir=tmp_path,
        start_timestamp=start,
        end_timestamp=end,
        expected_count=7,
        retrieved_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        commit_sha="mock-sha",
        run_id="mock-run",
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    manifest_text = (tmp_path / posthack.MANIFEST_NAME).read_text()
    manifest = json.loads(manifest_text)
    report_text = (tmp_path / posthack.GAP_NAME).read_text()
    report = json.loads(report_text)

    assert summary["status"] == "GAP_AND_GRID_ANOMALY"
    assert summary["gap_count"] == 2
    assert summary["off_grid_count"] == 1
    assert summary["distinct_off_grid_offsets_seconds"] == [1800]
    assert manifest["grid_anomaly"] is True
    assert report["missing_timestamps_epoch"] == sorted(missing)
    assert report["missing_timestamps_utc"] == [
        posthack._utc_iso(stamp) for stamp in sorted(missing)
    ]
    assert report["off_grid_timestamps"] == [
        {
            "timestamp_epoch": off_grid_timestamp,
            "timestamp_utc": posthack._utc_iso(off_grid_timestamp),
            "offset_seconds": 1800,
        }
    ]
    assert report["distinct_off_grid_offsets_seconds"] == [1800]
    assert not (tmp_path / posthack.CSV_NAME).exists()
    for forbidden in ("open", "high", "low", "close", "volume"):
        assert f'"{forbidden}"' not in manifest_text
        assert f'"{forbidden}"' not in report_text


def test_prehack_hourly_grid_distance_validation_contract_is_unchanged() -> None:
    first = public_fetcher.PREHACK_START_TIMESTAMP
    last = public_fetcher.PREHACK_END_TIMESTAMP
    stamps = list(range(first, last + 1, public_fetcher.PREHACK_INTERVAL_SECONDS))
    assert len(stamps) == public_fetcher.PREHACK_EXPECTED_CANDLES
    assert (
        public_fetcher._missing_intervals(
            stamps, public_fetcher.PREHACK_INTERVAL_SECONDS
        )
        == 0
    )
