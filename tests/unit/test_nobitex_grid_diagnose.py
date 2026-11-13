from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import diagnose_nobitex_grid as diagnose
from scripts import fetch_nobitex_posthack_ohlcv as wrapper


def _row(timestamp: int) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "date": "2026-11-02T00:00:00Z",
        "open": 123.0,
        "high": 124.0,
        "low": 122.0,
        "close": 123.5,
        "volume": 9.0,
    }


def test_diagnose_window_is_fixed_and_strictly_after_holdout() -> None:
    assert diagnose.validate_diagnose_window(
        diagnose.DIAGNOSE_START_TIMESTAMP,
        diagnose.DIAGNOSE_END_TIMESTAMP,
    ) == 241
    assert diagnose.DIAGNOSE_START_TIMESTAMP > diagnose.HOLDOUT_END_TIMESTAMP
    with pytest.raises(ValueError, match="not_owner_approved"):
        diagnose.validate_diagnose_window(1_762_041_600, 1_762_905_600)


def test_diagnose_artifact_contains_only_structural_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = diagnose.DIAGNOSE_START_TIMESTAMP
    end = diagnose.DIAGNOSE_END_TIMESTAMP
    rows = [_row(start), _row(start + 1801), _row(end)]
    missing = [start + 3600]
    off_grid = [{"timestamp_epoch": start + 1801, "offset_seconds": 1801}]

    def fake_fetch_window(**kwargs: Any):
        assert kwargs["start_timestamp"] == start
        assert kwargs["end_timestamp"] == end
        return rows, missing, off_grid, 0

    monkeypatch.setattr(wrapper, "fetch_window", fake_fetch_window)
    metrics, succeeded = diagnose.run_diagnosis(
        start_timestamp=start,
        end_timestamp=end,
        output_dir=tmp_path,
        commit_sha="mock-sha",
        run_id="mock-run",
        request_json=lambda _: {"s": "ok"},
    )

    artifact_text = (tmp_path / diagnose.ARTIFACT_NAME).read_text(encoding="utf-8")
    artifact = __import__("json").loads(artifact_text)
    assert succeeded is True
    assert metrics == {
        "http_status": 200,
        "candle_count": 3,
        "off_grid_count": 1,
        "distinct_offsets": [1801],
        "first_ts": start,
        "last_ts": end,
        "gap_count": 1,
    }
    assert artifact["prices_persisted"] is False
    assert artifact["ohlcv_persisted"] is False
    for forbidden in ('"open"', '"high"', '"low"', '"close"', '"volume"'):
        assert forbidden not in artifact_text
    for raw_value in ("123.0", "124.0", "122.0", "123.5", "9.0"):
        assert raw_value not in artifact_text


def test_diagnose_gate_bypasses_nonce_parser_but_not_acquisition_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "nobitex-posthack.yml").read_text(
        encoding="utf-8"
    )
    gate_start = workflow.index("      - name: Validate trigger nonce")
    gate_end = workflow.index("      - name: Probe Nobitex", gate_start)
    gate = workflow[gate_start:gate_end]
    diagnose_branch = gate.index('if [[ "$trigger_value" == "DIAGNOSE" ]]')
    nonce_read = gate.index('nonce="$(python')
    assert diagnose_branch < nonce_read
    assert 'echo "mode=DIAGNOSE"' in gate
    assert 'echo "mode=ACQUISITION"' in gate
    assert "if: steps.gate.outputs.mode == 'ACQUISITION'" in workflow
    assert "if: steps.gate.outputs.mode == 'DIAGNOSE'" in workflow
    assert "--start 1793577600" in workflow
    assert "--end 1794441600" in workflow


def test_diagnose_cli_help_does_not_make_network_requests() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.diagnose_nobitex_grid", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--start" in result.stdout
    assert "--end" in result.stdout
