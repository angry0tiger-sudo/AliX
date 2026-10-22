"""Append-only ledger for synthetic Phase 9 decisions; never an order ledger."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LedgerConflictError(RuntimeError):
    """A replay reused a decision ID for different immutable content."""


class SimulationStatus(StrEnum):
    SIMULATED = "SIMULATED"
    REJECTED = "REJECTED"
    NOT_SIMULATED = "NOT_SIMULATED"
    SYNTHETIC_UNKNOWN_NO_RETRY = "SYNTHETIC_UNKNOWN_NO_RETRY"


@dataclass(frozen=True, slots=True)
class SimulationDecisionRecord:
    """Synthetic decision evidence, explicitly incapable of authorizing orders."""

    decision_id: str
    decided_at: datetime
    input_fingerprint: str
    model_revision: str | None
    risk_approved: bool
    reasons: tuple[str, ...]
    simulation_status: SimulationStatus
    simulated_quantity: Decimal = Decimal("0")
    simulated_entry_price: Decimal = Decimal("0")
    simulated_stop_price: Decimal = Decimal("0")
    simulated_exit_price: Decimal = Decimal("0")
    simulated_net_pnl: Decimal = Decimal("0")
    order_permission: Literal[False] = False
    trace_json: str = "{}"

    def __post_init__(self) -> None:
        timestamp = self.decided_at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("simulation_decision_timestamp_must_be_aware")
        object.__setattr__(self, "decided_at", timestamp.astimezone(timezone.utc))
        if not self.decision_id.strip():
            raise ValueError("simulation_decision_id_required")
        if not _SHA256_RE.fullmatch(self.input_fingerprint):
            raise ValueError("simulation_input_fingerprint_invalid")
        if self.model_revision is not None and not self.model_revision.strip():
            raise ValueError("simulation_model_revision_invalid")
        if self.order_permission is not False:
            raise ValueError("simulation_order_permission_must_remain_false")
        try:
            decoded_trace = json.loads(self.trace_json)
        except json.JSONDecodeError:
            raise ValueError("simulation_trace_json_invalid") from None
        if not isinstance(decoded_trace, dict):
            raise ValueError("simulation_trace_must_be_json_object")
        object.__setattr__(
            self,
            "trace_json",
            json.dumps(
                decoded_trace,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
        for name in (
            "simulated_quantity",
            "simulated_entry_price",
            "simulated_stop_price",
            "simulated_exit_price",
            "simulated_net_pnl",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name}_must_be_finite_decimal")
        for name in (
            "simulated_quantity",
            "simulated_entry_price",
            "simulated_stop_price",
            "simulated_exit_price",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}_must_not_be_negative")
        if self.simulation_status is SimulationStatus.SIMULATED:
            if (
                not self.risk_approved
                or self.simulated_quantity <= 0
                or self.simulated_entry_price <= 0
                or self.simulated_stop_price <= 0
                or self.simulated_exit_price <= 0
            ):
                raise ValueError("simulated_fill_requires_approved_risk_and_valid_prices")
        elif self.simulated_quantity != 0:
            raise ValueError("non_fill_status_must_not_contain_simulated_quantity")
        if (
            self.simulation_status
            in (SimulationStatus.REJECTED, SimulationStatus.NOT_SIMULATED)
            and self.risk_approved
        ):
            raise ValueError("non_simulated_decision_must_not_be_risk_approved")
        if (
            self.simulation_status is SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY
            and not self.risk_approved
        ):
            raise ValueError("synthetic_unknown_requires_a_risk_approved_candidate")


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    report_id: str
    data_cutoff: datetime
    generated_at: datetime
    snapshot_hash: str
    row_count: int
    simulated_count: int
    rejected_count: int
    not_simulated_count: int
    unknown_no_retry_count: int
    simulated_net_pnl: Decimal
    source_status: str = "SYNTHETIC_ONLY"
    reconciliation_status: str = "NOT_RECONCILED"
    order_permission: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.data_cutoff.tzinfo is None
            or self.data_cutoff.utcoffset() is None
            or self.generated_at.tzinfo is None
            or self.generated_at.utcoffset() is None
        ):
            raise ValueError("ledger_snapshot_timestamps_must_be_aware")
        if not _SHA256_RE.fullmatch(self.snapshot_hash):
            raise ValueError("ledger_snapshot_hash_invalid")
        if self.order_permission is not False:
            raise ValueError("ledger_snapshot_order_permission_must_remain_false")
        if self.source_status != "SYNTHETIC_ONLY":
            raise ValueError("ledger_snapshot_source_must_remain_synthetic")
        if self.reconciliation_status != "NOT_RECONCILED":
            raise ValueError("synthetic_ledger_must_not_claim_reconciliation")


def _record_payload(record: SimulationDecisionRecord) -> dict[str, object]:
    return {
        "decision_id": record.decision_id,
        "decided_at": record.decided_at.isoformat(),
        "input_fingerprint": record.input_fingerprint,
        "model_revision": record.model_revision,
        "risk_approved": record.risk_approved,
        "reasons": list(record.reasons),
        "simulation_status": record.simulation_status.value,
        "simulated_quantity": str(record.simulated_quantity),
        "simulated_entry_price": str(record.simulated_entry_price),
        "simulated_stop_price": str(record.simulated_stop_price),
        "simulated_exit_price": str(record.simulated_exit_price),
        "simulated_net_pnl": str(record.simulated_net_pnl),
        "order_permission": False,
        "source_status": "SYNTHETIC_ONLY",
        "trace": json.loads(record.trace_json),
    }


class SQLiteDecisionLedger:
    """Local append-only simulation ledger; not connected to Freqtrade or exchanges."""

    def __init__(self, database_path: str | Path) -> None:
        rendered_path = str(database_path).strip()
        if not rendered_path or rendered_path == ":memory:":
            raise ValueError("durable_simulation_ledger_path_required")
        self.database_path = rendered_path
        Path(rendered_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=5,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_decision_ledger (
                    decision_id TEXT PRIMARY KEY,
                    decided_at TEXT NOT NULL,
                    record_hash TEXT NOT NULL CHECK(length(record_hash) = 64),
                    payload_json TEXT NOT NULL,
                    order_permission INTEGER NOT NULL DEFAULT 0
                        CHECK(order_permission = 0)
                );
                CREATE TRIGGER IF NOT EXISTS simulation_ledger_no_update
                BEFORE UPDATE ON simulation_decision_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'simulation_ledger_append_only');
                END;
                CREATE TRIGGER IF NOT EXISTS simulation_ledger_no_delete
                BEFORE DELETE ON simulation_decision_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'simulation_ledger_append_only');
                END;
                """
            )

    def append(self, record: SimulationDecisionRecord) -> bool:
        """Append once; return False for identical replay and reject conflicting reuse."""
        if record.order_permission is not False:
            raise ValueError("simulation_order_permission_must_remain_false")
        payload_json = json.dumps(
            _record_payload(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        record_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT record_hash FROM simulation_decision_ledger WHERE decision_id = ?",
                (record.decision_id,),
            ).fetchone()
            if prior is not None:
                if prior["record_hash"] != record_hash:
                    raise LedgerConflictError("simulation_decision_id_payload_conflict")
                connection.commit()
                return False
            connection.execute(
                """
                INSERT INTO simulation_decision_ledger
                    (decision_id, decided_at, record_hash, payload_json, order_permission)
                VALUES (?, ?, ?, ?, 0)
                """,
                (
                    record.decision_id,
                    record.decided_at.isoformat(),
                    record_hash,
                    payload_json,
                ),
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_record(self, decision_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM simulation_decision_ledger
                WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise RuntimeError("simulation_ledger_payload_invalid")
        return payload

    def snapshot(
        self,
        *,
        data_cutoff: datetime,
        generated_at: datetime,
    ) -> LedgerSnapshot:
        if (
            data_cutoff.tzinfo is None
            or data_cutoff.utcoffset() is None
            or generated_at.tzinfo is None
            or generated_at.utcoffset() is None
        ):
            raise ValueError("ledger_snapshot_timestamps_must_be_aware")
        cutoff = data_cutoff.astimezone(timezone.utc)
        generated = generated_at.astimezone(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_hash, payload_json FROM simulation_decision_ledger
                WHERE decided_at <= ?
                ORDER BY decided_at, decision_id
                """,
                (cutoff.isoformat(),),
            ).fetchall()
        record_hashes: list[str] = []
        statuses: list[str] = []
        simulated_pnl = Decimal("0")
        for row in rows:
            record_hashes.append(str(row["record_hash"]))
            payload = json.loads(str(row["payload_json"]))
            statuses.append(str(payload["simulation_status"]))
            if payload["simulation_status"] == SimulationStatus.SIMULATED.value:
                simulated_pnl += Decimal(str(payload["simulated_net_pnl"]))
        snapshot_body = json.dumps(
            {
                "data_cutoff": cutoff.isoformat(),
                "record_hashes": record_hashes,
                "source_status": "SYNTHETIC_ONLY",
                "reconciliation_status": "NOT_RECONCILED",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        snapshot_hash = hashlib.sha256(snapshot_body.encode("utf-8")).hexdigest()
        return LedgerSnapshot(
            report_id=f"alix-sim-report-{snapshot_hash[:16]}",
            data_cutoff=cutoff,
            generated_at=generated,
            snapshot_hash=snapshot_hash,
            row_count=len(rows),
            simulated_count=statuses.count(SimulationStatus.SIMULATED.value),
            rejected_count=statuses.count(SimulationStatus.REJECTED.value),
            not_simulated_count=statuses.count(SimulationStatus.NOT_SIMULATED.value),
            unknown_no_retry_count=statuses.count(
                SimulationStatus.SYNTHETIC_UNKNOWN_NO_RETRY.value
            ),
            simulated_net_pnl=simulated_pnl,
        )
