from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from services.reporting.contracts import SourceSnapshot
from services.reporting.png import render_equity_chart
from services.reporting.snapshot import build_reporting_snapshot
from services.reporting.engine import DailyReport, ReportingEngine
from services.reporting.sources import (
    FreqtradeSQLiteTradeSource,
    ReportingSourceError,
    SQLiteSimulationLedgerSource,
)
from services.simulation.ledger import SQLiteDecisionLedger, SimulationDecisionRecord, SimulationStatus

UTC = timezone.utc
NOW = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)


def test_freqtrade_adapter_checks_schema_and_never_writes(tmp_path: Path) -> None:
    db = tmp_path / "trades.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, pair TEXT, is_open INTEGER, open_date TEXT, close_date TEXT, close_profit_abs REAL)")
        connection.execute("INSERT INTO trades VALUES (1, 'BTC/USDT', 0, '2026-10-22 01:00:00', '2026-10-22 02:00:00', 1.25)")
        connection.execute("INSERT INTO trades VALUES (2, 'BTC/USDT', 1, '2026-10-23 06:00:00', NULL, NULL)")
    before = db.stat().st_size
    snapshot = FreqtradeSQLiteTradeSource(db).read(cutoff=NOW)
    assert snapshot.summary["closed_trade_count"] == 1
    assert snapshot.summary["open_trade_count"] == 1
    assert snapshot.summary["freqtrade_reported_profit_abs_sum"] == 1.25
    assert snapshot.reconciled is False
    assert snapshot.summary["account_snapshot_available"] is False
    with sqlite3.connect(db) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"trades"}
    assert db.stat().st_size == before


def test_freqtrade_adapter_rejects_missing_or_incompatible_database(tmp_path: Path) -> None:
    with pytest.raises(ReportingSourceError, match="source_database_missing"):
        FreqtradeSQLiteTradeSource(tmp_path / "missing.sqlite3").read(cutoff=NOW)
    db = tmp_path / "bad.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, pair TEXT)")
    with pytest.raises(ReportingSourceError, match="freqtrade_trade_schema_mismatch"):
        FreqtradeSQLiteTradeSource(db).read(cutoff=NOW)


def test_simulation_ledger_source_preserves_synthetic_only_provenance(tmp_path: Path) -> None:
    from decimal import Decimal

    db = tmp_path / "ledger.sqlite3"
    ledger = SQLiteDecisionLedger(db)
    record = SimulationDecisionRecord(
        decision_id="phase10-synthetic-1", decided_at=NOW,
        input_fingerprint="a" * 64, model_revision="fixture-model-v1",
        risk_approved=True, reasons=(), simulation_status=SimulationStatus.SIMULATED,
        simulated_quantity=Decimal("1"), simulated_entry_price=Decimal("100"),
        simulated_stop_price=Decimal("95"), simulated_exit_price=Decimal("95"),
        simulated_net_pnl=Decimal("-5"),
        trace_json=json.dumps({"source": "synthetic"}),
    )
    ledger.append(record)
    source = SQLiteSimulationLedgerSource(db).read(cutoff=NOW)
    assert source.status == "SYNTHETIC_ONLY"
    assert source.reconciled is False
    assert source.summary["synthetic_net_pnl"] == -5.0


def test_missing_sources_degrade_and_suppress_financial_truth() -> None:
    snapshot = build_reporting_snapshot(
        [], generated_at=NOW, data_cutoff=NOW, timezone_name="UTC",
        scope="daily", code_revision="candidate-sha",
    )
    assert snapshot.status == "DEGRADED"
    assert snapshot.reconciliation_status == "NOT_RECONCILED"
    assert snapshot.metrics["net_pnl"] is None
    assert snapshot.metrics["equity"] is None
    assert snapshot.order_permission is False
    assert snapshot.champion is None
    assert snapshot.to_dict()["snapshot_hash"] == snapshot.snapshot_hash


def test_snapshot_hash_changes_when_source_payload_changes() -> None:
    class FakeSource:
        source_type = "freqtrade_trade_db"

        def __init__(self, value: float) -> None:
            self.value = value

        def read(self, *, cutoff: datetime) -> SourceSnapshot:
            return SourceSnapshot(
                source_id="fake-trade-db", source_type=self.source_type, status="OK",
                observed_at=NOW, data_cutoff=cutoff, source_revision="freqtrade-fixture-v1",
                authoritative=True, reconciled=False, summary={"profit": self.value},
            )

    first = build_reporting_snapshot([FakeSource(1.0)], generated_at=NOW, data_cutoff=NOW,
                                     timezone_name="UTC", scope="daily", code_revision="same-sha")
    second = build_reporting_snapshot([FakeSource(2.0)], generated_at=NOW, data_cutoff=NOW,
                                      timezone_name="UTC", scope="daily", code_revision="same-sha")
    assert first.report_id == second.report_id
    assert first.snapshot_hash != second.snapshot_hash


def test_source_exception_becomes_degraded_without_exposing_message() -> None:
    class BrokenSource:
        source_type = "freqtrade_trade_db"

        def read(self, *, cutoff: datetime) -> SourceSnapshot:
            raise RuntimeError("password=never-export-this")

    snapshot = build_reporting_snapshot([BrokenSource()], generated_at=NOW, data_cutoff=NOW,
                                        timezone_name="UTC", scope="daily", code_revision="sha")
    assert snapshot.status == "DEGRADED"
    assert all(item.error_code is None or "never-export-this" not in item.error_code for item in snapshot.sources)


def test_chart_is_valid_png_and_suppresses_unreconciled_series() -> None:
    from struct import unpack

    blank = render_equity_chart([100.0, 120.0, 90.0], reconciliation_status="NOT_RECONCILED")
    drawn = render_equity_chart([100.0, 120.0, 90.0], reconciliation_status="RECONCILED")
    assert blank.startswith(b"\x89PNG\r\n\x1a\n")
    assert drawn.startswith(b"\x89PNG\r\n\x1a\n")
    assert blank != drawn
    assert unpack(">II", drawn[16:24]) == (720, 360)
    try:
        render_equity_chart([1.0, 2.0], reconciliation_status="RECONCILED", width=800)
    except ValueError as exc:
        assert "versioned" in str(exc)
    else:
        raise AssertionError("non-versioned chart dimensions should fail")


def test_reporting_engine_renders_snapshot_text_and_png_fallback() -> None:
    snapshot = build_reporting_snapshot(
        [], generated_at=NOW, data_cutoff=NOW, timezone_name="UTC",
        scope="daily", code_revision="phase10-test-sha",
    )
    engine = ReportingEngine(default_chat_id="")
    rendered = engine.render_snapshot_text(snapshot)
    image = engine.render_snapshot_png(snapshot)
    assert "NOT RECONCILED" in rendered
    assert "order_permission=false" in rendered
    assert image.startswith(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
    assert engine.render_text(DailyReport(date="2026-10-23", summary="synthetic", metrics={})).startswith("AliX paper report")
