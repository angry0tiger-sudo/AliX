"""Read-only adapters for the pinned Freqtrade trade table and synthetic ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from services.reporting.contracts import SourceSnapshot


class ReportingSourceError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field}_timezone_required")
    return value.astimezone(timezone.utc)


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ReportingSourceError("source_database_missing")
    location = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    try:
        connection = sqlite3.connect(location, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.Error:
        raise ReportingSourceError("source_database_unavailable") from None


def _read_freqtrade_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "trades" not in tables:
        raise ReportingSourceError("freqtrade_trades_table_missing")
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(trades)").fetchall()
    }
    required = {"id", "pair", "is_open", "open_date", "close_date", "close_profit_abs"}
    if not required.issubset(columns):
        raise ReportingSourceError("freqtrade_trade_schema_mismatch")
    return connection.execute(
        "SELECT id, pair, is_open, open_date, close_date, close_profit_abs FROM trades"
    ).fetchall()


def _classify_trade_row(
    row: sqlite3.Row, cutoff: datetime
) -> tuple[str, str | None, float | None]:
    opened = _timestamp(row["open_date"])
    if opened is None or opened > cutoff:
        return "ignored", None, None
    if bool(row["is_open"]):
        return "open", str(row["id"]), None
    closed = _timestamp(row["close_date"])
    if closed is None or closed > cutoff:
        return "ignored", None, None
    try:
        profit = float(row["close_profit_abs"])
    except (TypeError, ValueError):
        return "closed_unknown_profit", str(row["id"]), None
    if not (profit == profit and abs(profit) != float("inf")):
        return "closed_unknown_profit", str(row["id"]), None
    return "closed", str(row["id"]), profit


def _summarize_trade_rows(rows: list[sqlite3.Row], cutoff: datetime) -> dict[str, Any]:
    closed_count = open_count = unknown_profit = 0
    profit_total = 0.0
    closed_ids: list[str] = []
    for row in rows:
        status, trade_id, profit = _classify_trade_row(row, cutoff)
        if status == "ignored":
            continue
        if status == "open":
            open_count += 1
            continue
        closed_count += 1
        if trade_id is not None:
            closed_ids.append(trade_id)
        if profit is None:
            unknown_profit += 1
        else:
            profit_total += profit
    return {
        "observed_trade_row_count": len(rows),
        "closed_trade_count": closed_count,
        "open_trade_count": open_count,
        "freqtrade_reported_profit_abs_sum": profit_total
        if unknown_profit == 0
        else None,
        "profit_rows_unknown": unknown_profit,
        "profit_field": "trades.close_profit_abs; reported, not account-reconciled",
        "fee_totals_available": False,
        "account_snapshot_available": False,
        "closed_ids_hash": hashlib.sha256(",".join(closed_ids).encode()).hexdigest(),
        "reconciliation_status": "NOT_RECONCILED",
    }


class FreqtradeSQLiteTradeSource:
    """Schema-reflected reader for Freqtrade 2026.7 SQLite trades; read-only."""

    source_type = "freqtrade_trade_db"
    SOURCE_REVISION = "freqtrade-2026.7-trades-contract"

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)

    def read(self, *, cutoff: datetime) -> SourceSnapshot:
        cutoff_utc = _utc(cutoff, "report_cutoff")
        connection = _read_only(self._path)
        try:
            rows = _read_freqtrade_rows(connection)
        except ReportingSourceError:
            raise
        except sqlite3.Error:
            raise ReportingSourceError("freqtrade_trade_read_failed") from None
        finally:
            connection.close()
        summary = _summarize_trade_rows(rows, cutoff_utc)
        return SourceSnapshot(
            source_id="freqtrade-sqlite-readonly",
            source_type=self.source_type,
            status="OK",
            observed_at=datetime.now(timezone.utc),
            data_cutoff=cutoff_utc,
            source_revision=self.SOURCE_REVISION,
            authoritative=True,
            reconciled=False,
            summary=summary,
        )


def _read_simulation_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "simulation_decision_ledger" not in tables:
        raise ReportingSourceError("simulation_ledger_table_missing")
    return connection.execute(
        "SELECT decided_at, payload_json FROM simulation_decision_ledger"
    ).fetchall()


def _ledger_payload_at_cutoff(
    row: sqlite3.Row, cutoff: datetime
) -> dict[str, Any] | None:
    decided = _timestamp(row["decided_at"])
    if decided is None or decided > cutoff:
        return None
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        raise ReportingSourceError("simulation_ledger_payload_invalid") from None
    if (
        not isinstance(payload, dict)
        or payload.get("source_status") != "SYNTHETIC_ONLY"
    ):
        raise ReportingSourceError("simulation_ledger_provenance_mismatch")
    return payload


def _simulated_pnl(payload: dict[str, Any]) -> tuple[bool, float]:
    if payload.get("simulation_status") != "SIMULATED":
        return False, 0.0
    try:
        return True, float(payload.get("simulated_net_pnl", "0"))
    except (TypeError, ValueError):
        raise ReportingSourceError("simulation_ledger_amount_invalid") from None


def _summarize_simulation_rows(
    rows: list[sqlite3.Row], cutoff: datetime
) -> dict[str, Any]:
    count = simulated = 0
    pnl = 0.0
    for row in rows:
        payload = _ledger_payload_at_cutoff(row, cutoff)
        if payload is None:
            continue
        count += 1
        is_simulated, net_pnl = _simulated_pnl(payload)
        if is_simulated:
            simulated += 1
            pnl += net_pnl
    return {
        "decision_count": count,
        "simulated_count": simulated,
        "synthetic_net_pnl": pnl,
        "financial_truth": False,
        "reconciliation_status": "NOT_RECONCILED",
    }


class SQLiteSimulationLedgerSource:
    """Read Phase 9's append-only simulated ledger without claiming financial truth."""

    source_type = "decision_ledger"
    SOURCE_REVISION = "alix-phase9-simulation-ledger-v1"

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)

    def read(self, *, cutoff: datetime) -> SourceSnapshot:
        cutoff_utc = _utc(cutoff, "report_cutoff")
        connection = _read_only(self._path)
        try:
            rows = _read_simulation_rows(connection)
        except ReportingSourceError:
            raise
        except sqlite3.Error:
            raise ReportingSourceError("simulation_ledger_read_failed") from None
        finally:
            connection.close()
        return SourceSnapshot(
            source_id="phase9-simulation-ledger-readonly",
            source_type=self.source_type,
            status="SYNTHETIC_ONLY",
            observed_at=datetime.now(timezone.utc),
            data_cutoff=cutoff_utc,
            source_revision=self.SOURCE_REVISION,
            authoritative=False,
            reconciled=False,
            summary=_summarize_simulation_rows(rows, cutoff_utc),
        )
