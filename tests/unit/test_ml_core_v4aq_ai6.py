from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_ml_core_v4aq_ai6 as ai6


def _write_symbol_fixture(
    directory: Path, symbol: str, commit_sha: str
) -> None:
    stem = f"binance-{symbol.lower()}"
    funding_path = directory / f"{stem}-funding.csv"
    oi_path = directory / f"{stem}-open-interest.csv"
    funding_path.write_text(
        "symbol,funding_time,funding_rate,available_at\n", encoding="utf-8"
    )
    oi_path.write_text(
        "symbol,period,timestamp,sum_open_interest,available_at\n", encoding="utf-8"
    )
    funding_hash = hashlib.sha256(funding_path.read_bytes()).hexdigest()
    oi_hash = hashlib.sha256(oi_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "alix.binance.usdm-external-market.v1",
        "commit_sha": commit_sha,
        "symbol": symbol,
        "requested_days": 1825,
        "public_api_only": True,
        "credentials_used": False,
        "api_key_header_sent": False,
        "private_api_calls": 0,
        "exchange_order_calls": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "funding": {
            "observed_span_days": 1824.0,
            "row_count": 5000,
            "coverage_status": "FULL_REQUESTED_HISTORY",
            "full_requested_history_available": True,
        },
        "open_interest": {
            "observed_span_days": 30.0,
            "row_count": 180,
            "coverage_status": "INSUFFICIENT_HISTORICAL_WINDOW",
            "full_requested_history_available": False,
        },
        "funding_csv": funding_path.name,
        "funding_csv_sha256": funding_hash,
        "open_interest_csv": oi_path.name,
        "open_interest_csv_sha256": oi_hash,
    }
    (directory / f"{stem}-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_ai6_runner_blocks_fit_and_writes_explicit_coverage_verdict(
    tmp_path: Path,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    commit_sha = "d" * 40
    _write_symbol_fixture(external_dir, "BTCUSDT", commit_sha)
    _write_symbol_fixture(external_dir, "ETHUSDT", commit_sha)
    output_report = tmp_path / "out" / "report.json"
    output_trades = tmp_path / "out" / "trades.csv"
    output_decisions = tmp_path / "out" / "decisions.csv"

    report = ai6.run_v4aq_ai6(
        tmp_path / "spot-not-required-when-blocked",
        external_dir,
        output_report,
        output_trades,
        output_decisions,
        commit_sha=commit_sha,
    )

    assert report["status"] == "BLOCKED_INSUFFICIENT_EXTERNAL_HISTORY"
    assert report["model"]["fit_performed"] is False
    assert report["economic_results"] is None
    assert report["Champion"] is None
    assert report["order_permission"] is False
    assert report["trading_authorization"] == "NO-GO"
    assert report["private_api_calls"] == 0
    assert output_trades.is_file() and output_decisions.is_file()
    summary = ai6._render_summary(report)
    assert "LightGBM fit: NOT RUN" in summary
    assert "INSUFFICIENT_HISTORICAL_WINDOW" in summary


def test_ai6_runner_rejects_manifest_hash_or_safety_mismatch(
    tmp_path: Path,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    commit_sha = "e" * 40
    _write_symbol_fixture(external_dir, "BTCUSDT", commit_sha)
    _write_symbol_fixture(external_dir, "ETHUSDT", commit_sha)
    manifest_path = external_dir / "binance-btcusdt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["private_api_calls"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_contract_invalid"):
        ai6._read_symbol_manifest(external_dir, "BTCUSDT", commit_sha)


def test_ai6_acceptance_helper_does_not_promote_net_pf_without_dsr() -> None:
    result = ai6._candidate_acceptance(
        {
            "trade_count": 100,
            "win_rate": 0.55,
            "net_profit_factor": 1.25,
        }
    )
    assert result["trade_count_passed"] is True
    assert result["win_rate_passed"] is True
    assert result["net_profit_factor_passed"] is True
    assert result["dsr_passed"] is False
    assert result["statistical_edge_accepted"] is False
