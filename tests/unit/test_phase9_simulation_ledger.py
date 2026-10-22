from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import cast, Literal

import pytest

from services.simulation.ledger import (
    LedgerConflictError,
    SQLiteDecisionLedger,
    SimulationDecisionRecord,
    SimulationStatus,
)

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


def record(
    decision_id: str = "phase9-decision-1",
    *,
    input_seed: str = "input-a",
    status: SimulationStatus = SimulationStatus.SIMULATED,
    risk_approved: bool = True,
) -> SimulationDecisionRecord:
    simulated = status is SimulationStatus.SIMULATED
    return SimulationDecisionRecord(
        decision_id=decision_id,
        decided_at=NOW,
        input_fingerprint=hashlib.sha256(input_seed.encode()).hexdigest(),
        model_revision="synthetic-model-v1",
        risk_approved=risk_approved,
        reasons=(),
        simulation_status=status,
        simulated_quantity=Decimal("1") if simulated else Decimal("0"),
        simulated_entry_price=Decimal("100") if simulated else Decimal("0"),
        simulated_stop_price=Decimal("96") if simulated else Decimal("0"),
        simulated_exit_price=Decimal("96") if simulated else Decimal("0"),
        simulated_net_pnl=Decimal("-4.392") if simulated else Decimal("0"),
        trace_json='{"inputs":{"sample":"synthetic-fixture"},"model":{"revision":"model-v1"},"risk":{"decision":"NO_GO"}}',
    )


def test_ledger_is_idempotent_and_snapshot_is_explicitly_synthetic(tmp_path: Path) -> None:
    ledger = SQLiteDecisionLedger(tmp_path / "ledger.sqlite")
    assert ledger.append(record())
    assert not ledger.append(record())
    stored = ledger.get_record("phase9-decision-1")
    assert stored is not None
    assert stored["order_permission"] is False
    assert stored["source_status"] == "SYNTHETIC_ONLY"
    trace = stored["trace"]
    assert isinstance(trace, dict)
    assert trace["inputs"] == {"sample": "synthetic-fixture"}
    assert trace["model"] == {"revision": "model-v1"}
    assert trace["risk"] == {"decision": "NO_GO"}

    snapshot = ledger.snapshot(data_cutoff=NOW, generated_at=NOW)
    assert snapshot.row_count == 1
    assert snapshot.simulated_count == 1
    assert snapshot.simulated_net_pnl == Decimal("-4.392")
    assert snapshot.order_permission is False
    assert snapshot.reconciliation_status == "NOT_RECONCILED"
    assert len(snapshot.snapshot_hash) == 64


def test_ledger_rejects_reused_decision_id_with_different_payload(tmp_path: Path) -> None:
    ledger = SQLiteDecisionLedger(tmp_path / "ledger.sqlite")
    assert ledger.append(record())
    with pytest.raises(LedgerConflictError, match="payload_conflict"):
        ledger.append(record(input_seed="different-input"))


def test_sqlite_triggers_keep_decision_records_append_only(tmp_path: Path) -> None:
    ledger = SQLiteDecisionLedger(tmp_path / "ledger.sqlite")
    ledger.append(record())
    with sqlite3.connect(ledger.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute(
                "UPDATE simulation_decision_ledger SET payload_json = '{}' "
                "WHERE decision_id = ?",
                ("phase9-decision-1",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute(
                "DELETE FROM simulation_decision_ledger WHERE decision_id = ?",
                ("phase9-decision-1",),
            )


def test_ledger_fails_closed_for_order_permission_and_unsafe_status(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="order_permission"):
        SimulationDecisionRecord(
            decision_id="unsafe",
            decided_at=NOW,
            input_fingerprint=hashlib.sha256(b"unsafe").hexdigest(),
            model_revision=None,
            risk_approved=False,
            reasons=("synthetic_only",),
            simulation_status=SimulationStatus.REJECTED,
            order_permission=cast(Literal[False], True),
        )
    with pytest.raises(ValueError, match="requires_approved_risk"):
        record(
            status=SimulationStatus.SIMULATED,
            risk_approved=False,
        )


def test_empty_snapshot_is_not_mistaken_for_reconciled_truth(tmp_path: Path) -> None:
    snapshot = SQLiteDecisionLedger(tmp_path / "ledger.sqlite").snapshot(
        data_cutoff=NOW, generated_at=NOW
    )
    assert snapshot.row_count == 0
    assert snapshot.simulated_count == 0
    assert snapshot.simulated_net_pnl == Decimal("0")
    assert snapshot.source_status == "SYNTHETIC_ONLY"
    assert snapshot.reconciliation_status == "NOT_RECONCILED"
