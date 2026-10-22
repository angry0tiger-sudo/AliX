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


class FreqtradeSQLiteTradeSource:
    """Schema-reflected reader for Freqtrade 2026.7 SQLite trades; read-only."""

    source_type = "freqtrade_trade_db"
    SOURCE_REVISION = "freqtrade-2026.7-trades-contract"
    REQUIRED_COLUMNS = frozenset(
        {"id", "pair", "is_open", "open_date", "close_date", "close_profit_abs"}
    )
    OPTIONAL_COLUMNS = frozenset(
        {
            "stake_amount",
            "amount",
            "open_rate",
            "close_rate",
            "fee_open",
            "fee_close",
            "exit_reason",
            "strategy",
        }
    )

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)

    def read(self, *, cutoff: datetime) -> SourceSnapshot:
        cutoff_utc = _utc(cutoff, "report_cutoff")
        connection = _read_only(self._path)
        try:
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
            if not self.REQUIRED_COLUMNS.issubset(columns):
                raise ReportingSourceError("freqtrade_trade_schema_mismatch")
            selected = sorted(self.REQUIRED_COLUMNS | (self.OPTIONAL_COLUMNS & columns))
            rows = connection.execute(
                "SELECT "
                + ", ".join('"' + name + '"' for name in selected)
                + ' FROM "trades"'
            ).fetchall()
        except sqlite3.Error:
            raise ReportingSourceError("freqtrade_trade_read_failed") from None
        finally:
            connection.close()

        closed_count = open_count = unknown_profit = 0
        profit_total = 0.0
        closed_ids: list[str] = []
        for row in rows:
            opened = _timestamp(row["open_date"])
            if opened is None or opened > cutoff_utc:
                continue
            if bool(row["is_open"]):
                open_count += 1
                continue
            closed = _timestamp(row["close_date"])
            if closed is None or closed > cutoff_utc:
                continue
            closed_count += 1
            closed_ids.append(str(row["id"]))
            try:
                profit = float(row["close_profit_abs"])
            except (TypeError, ValueError):
                unknown_profit += 1
            else:
                if not (profit == profit and abs(profit) != float("inf")):
                    unknown_profit += 1
                else:
                    profit_total += profit
        summary: dict[str, Any] = {
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
            "closed_ids_hash": hashlib.sha256(
                ",".join(closed_ids).encode()
            ).hexdigest(),
            "reconciliation_status": "NOT_RECONCILED",
        }
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
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "simulation_decision_ledger" not in tables:
                raise ReportingSourceError("simulation_ledger_table_missing")
            rows = connection.execute(
                "SELECT decided_at, payload_json FROM simulation_decision_ledger"
            ).fetchall()
        except sqlite3.Error:
            raise ReportingSourceError("simulation_ledger_read_failed") from None
        finally:
            connection.close()
        count = simulated = 0
        pnl = 0.0
        for row in rows:
            decided = _timestamp(row["decided_at"])
            if decided is None or decided > cutoff_utc:
                continue
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                raise ReportingSourceError(
                    "simulation_ledger_payload_invalid"
                ) from None
            if (
                not isinstance(payload, dict)
                or payload.get("source_status") != "SYNTHETIC_ONLY"
            ):
                raise ReportingSourceError("simulation_ledger_provenance_mismatch")
            count += 1
            if payload.get("simulation_status") == "SIMULATED":
                simulated += 1
                try:
                    pnl += float(payload.get("simulated_net_pnl", "0"))
                except (TypeError, ValueError):
                    raise ReportingSourceError(
                        "simulation_ledger_amount_invalid"
                    ) from None
        return SourceSnapshot(
            source_id="phase9-simulation-ledger-readonly",
            source_type=self.source_type,
            status="SYNTHETIC_ONLY",
            observed_at=datetime.now(timezone.utc),
            data_cutoff=cutoff_utc,
            source_revision=self.SOURCE_REVISION,
            authoritative=False,
            reconciled=False,
            summary={
                "decision_count": count,
                "simulated_count": simulated,
                "synthetic_net_pnl": pnl,
                "financial_truth": False,
                "reconciliation_status": "NOT_RECONCILED",
            },
        )
