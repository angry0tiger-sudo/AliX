"""Latched, manually re-armed kill switch with a durable SQLite implementation.

The model/NLP process must not receive a store/controller capability. Unknown or
unreadable persisted state is a halt condition, never permission to continue.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Protocol

from risk.contracts import require_aware_timestamp


class KillSwitchStateUnavailable(RuntimeError):
    """Persisted halt state could not be safely established."""


class RearmRejected(RuntimeError):
    """A manual re-arm request did not satisfy all authorization checks."""


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    latched: bool
    reason: str
    latched_at: datetime | None
    version: int
    last_transition: datetime
    rearm_approval_id: str | None = None

    def __post_init__(self) -> None:
        require_aware_timestamp(self.last_transition, "kill_switch_transition")
        if self.latched_at is not None:
            require_aware_timestamp(self.latched_at, "kill_switch_latched_at")
        if self.version < 1:
            raise ValueError("kill_switch_version_invalid")
        if self.latched and not self.reason.strip():
            raise ValueError("latched_state_requires_reason")
        if not self.latched and self.latched_at is not None:
            raise ValueError("unlatched_state_must_not_keep_latched_at")


@dataclass(frozen=True, slots=True)
class ManualRearmEvidence:
    operator_id: str
    owner_approval_id: str
    reason: str
    observed_version: int
    account_reconciled: bool
    orders_reconciled: bool
    supervisor_healthy: bool
    risk_policy_unchanged: bool


class KillSwitchStore(Protocol):
    """Capability boundary: only supervisor/controller receives this interface."""

    def read(self) -> KillSwitchState: ...

    def latch(self, reason: str, now: datetime, actor: str) -> KillSwitchState: ...

    def rearm(
        self,
        expected_version: int,
        approval_id: str,
        operator_id: str,
        reason: str,
        now: datetime,
    ) -> bool: ...


class SQLiteKillSwitchStore:
    """Atomic durable latch for one local risk domain, not an exchange adapter."""

    def __init__(self, database_path: str) -> None:
        if not database_path.strip() or database_path == ":memory:":
            raise ValueError("durable_kill_switch_database_path_required")
        self._database_path = database_path
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path, timeout=5, isolation_level=None
            )
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    latched INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    latched_at TEXT,
                    version INTEGER NOT NULL,
                    last_transition TEXT NOT NULL,
                    rearm_approval_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kill_switch_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    approval_id TEXT
                )
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                INSERT OR IGNORE INTO kill_switch_state
                    (singleton, latched, reason, latched_at, version,
                     last_transition, rearm_approval_id)
                VALUES (1, 1, 'initial_state_unknown', ?, 1, ?, NULL)
                """,
                (now, now),
            )
            connection.execute(
                """
                INSERT INTO kill_switch_events
                    (occurred_at, action, reason, actor, state_version, approval_id)
                SELECT ?, 'INITIAL_LATCH', 'initial_state_unknown',
                       'system_startup', 1, NULL
                WHERE changes() = 1
                """,
                (now,),
            )
        except (OSError, sqlite3.Error) as exc:
            raise KillSwitchStateUnavailable(
                "kill_switch_store_initialization_failed"
            ) from exc
        finally:
            if connection is not None:
                connection.close()
        if os.name != "nt":
            try:
                os.chmod(self._database_path, 0o600)
            except OSError as exc:
                raise KillSwitchStateUnavailable(
                    "kill_switch_store_permissions_failed"
                ) from exc

    def read(self) -> KillSwitchState:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self._database_path, timeout=5)
            row = connection.execute(
                """
                SELECT latched, reason, latched_at, version, last_transition,
                       rearm_approval_id
                FROM kill_switch_state WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                raise KillSwitchStateUnavailable("kill_switch_state_row_missing")
            return self._state_from_row(row)
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            raise KillSwitchStateUnavailable("kill_switch_state_unreadable") from exc
        finally:
            if connection is not None:
                connection.close()

    def latch(self, reason: str, now: datetime, actor: str) -> KillSwitchState:
        require_aware_timestamp(now, "kill_switch_latch")
        if not reason.strip() or not actor.strip():
            raise ValueError("kill_switch_latch_reason_and_actor_required")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path, timeout=5, isolation_level=None
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT latched, reason, latched_at, version, last_transition,
                       rearm_approval_id
                FROM kill_switch_state WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                raise KillSwitchStateUnavailable("kill_switch_state_row_missing")
            current = self._state_from_row(row)
            if current.latched:
                next_state = current
                action = "LATCH_ALREADY_SET"
                event_reason = reason
            else:
                next_version = current.version + 1
                connection.execute(
                    """
                    UPDATE kill_switch_state
                    SET latched = 1, reason = ?, latched_at = ?, version = ?,
                        last_transition = ?, rearm_approval_id = NULL
                    WHERE singleton = 1 AND version = ?
                    """,
                    (
                        reason,
                        now.isoformat(),
                        next_version,
                        now.isoformat(),
                        current.version,
                    ),
                )
                if connection.total_changes < 1:
                    raise KillSwitchStateUnavailable("kill_switch_latch_compare_failed")
                next_state = KillSwitchState(True, reason, now, next_version, now, None)
                action = "LATCH"
                event_reason = reason
            connection.execute(
                """
                INSERT INTO kill_switch_events
                    (occurred_at, action, reason, actor, state_version, approval_id)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (now.isoformat(), action, event_reason, actor, next_state.version),
            )
            connection.commit()
            return next_state
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            if connection is not None:
                connection.rollback()
            if isinstance(exc, ValueError):
                raise
            raise KillSwitchStateUnavailable("kill_switch_latch_write_failed") from exc
        finally:
            if connection is not None:
                connection.close()

    def rearm(
        self,
        expected_version: int,
        approval_id: str,
        operator_id: str,
        reason: str,
        now: datetime,
    ) -> bool:
        require_aware_timestamp(now, "kill_switch_rearm")
        if not approval_id.strip() or not operator_id.strip() or not reason.strip():
            raise ValueError("kill_switch_rearm_audit_fields_required")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path, timeout=5, isolation_level=None
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT latched, reason, latched_at, version, last_transition,
                       rearm_approval_id
                FROM kill_switch_state WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                raise KillSwitchStateUnavailable("kill_switch_state_row_missing")
            current = self._state_from_row(row)
            if not current.latched or current.version != expected_version:
                connection.rollback()
                return False
            next_version = current.version + 1
            connection.execute(
                """
                UPDATE kill_switch_state
                SET latched = 0, reason = '', latched_at = NULL, version = ?,
                    last_transition = ?, rearm_approval_id = ?
                WHERE singleton = 1 AND version = ? AND latched = 1
                """,
                (next_version, now.isoformat(), approval_id, expected_version),
            )
            if connection.total_changes < 1:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO kill_switch_events
                    (occurred_at, action, reason, actor, state_version, approval_id)
                VALUES (?, 'MANUAL_REARM', ?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    reason,
                    operator_id,
                    next_version,
                    approval_id,
                ),
            )
            connection.commit()
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            if connection is not None:
                connection.rollback()
            if isinstance(exc, ValueError):
                raise
            raise KillSwitchStateUnavailable("kill_switch_rearm_write_failed") from exc
        finally:
            if connection is not None:
                connection.close()

    def audit_events(self) -> tuple[tuple[object, ...], ...]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self._database_path, timeout=5)
            rows = connection.execute(
                """
                SELECT occurred_at, action, reason, actor, state_version, approval_id
                FROM kill_switch_events ORDER BY event_id
                """
            ).fetchall()
            return tuple(tuple(row) for row in rows)
        except (OSError, sqlite3.Error) as exc:
            raise KillSwitchStateUnavailable("kill_switch_audit_unreadable") from exc
        finally:
            if connection is not None:
                connection.close()

    def _state_from_row(self, row: tuple[object, ...]) -> KillSwitchState:
        latched_at = datetime.fromisoformat(str(row[2])) if row[2] is not None else None
        return KillSwitchState(
            latched=bool(row[0]),
            reason=str(row[1]),
            latched_at=latched_at,
            version=int(row[3]),
            last_transition=datetime.fromisoformat(str(row[4])),
            rearm_approval_id=(str(row[5]) if row[5] is not None else None),
        )


class InMemoryKillSwitchStore:
    """Test adapter only. It does not survive process restart."""

    def __init__(self, initial_state: KillSwitchState | None = None) -> None:
        now = datetime.now(timezone.utc)
        self._state = initial_state or KillSwitchState(
            True, "initial_state_unknown", now, 1, now, None
        )
        self._lock = RLock()

    def read(self) -> KillSwitchState:
        with self._lock:
            return self._state

    def latch(self, reason: str, now: datetime, actor: str) -> KillSwitchState:
        require_aware_timestamp(now, "kill_switch_latch")
        if not reason.strip() or not actor.strip():
            raise ValueError("kill_switch_latch_reason_and_actor_required")
        with self._lock:
            if self._state.latched:
                return self._state
            self._state = KillSwitchState(
                True, reason, now, self._state.version + 1, now, None
            )
            return self._state

    def rearm(
        self,
        expected_version: int,
        approval_id: str,
        operator_id: str,
        reason: str,
        now: datetime,
    ) -> bool:
        require_aware_timestamp(now, "kill_switch_rearm")
        with self._lock:
            if not self._state.latched or self._state.version != expected_version:
                return False
            self._state = KillSwitchState(
                False, "", None, self._state.version + 1, now, approval_id
            )
            return True


class KillSwitchController:
    """Manual re-arm capability; never handed to model/NLP workers."""

    def __init__(self, store: KillSwitchStore) -> None:
        self._store = store

    def manual_rearm(
        self, evidence: ManualRearmEvidence, now: datetime
    ) -> KillSwitchState:
        require_aware_timestamp(now, "manual_rearm")
        if not all(
            value.strip()
            for value in (
                evidence.operator_id,
                evidence.owner_approval_id,
                evidence.reason,
            )
        ):
            raise RearmRejected("manual_rearm_audit_fields_missing")
        if not all(
            (
                evidence.account_reconciled,
                evidence.orders_reconciled,
                evidence.supervisor_healthy,
                evidence.risk_policy_unchanged,
            )
        ):
            raise RearmRejected("manual_rearm_preconditions_failed")
        current = self._store.read()
        if not current.latched:
            raise RearmRejected("kill_switch_not_latched")
        if current.version != evidence.observed_version:
            raise RearmRejected("kill_switch_version_changed")
        rearmed = self._store.rearm(
            evidence.observed_version,
            evidence.owner_approval_id,
            evidence.operator_id,
            evidence.reason,
            now,
        )
        if not rearmed:
            raise RearmRejected("kill_switch_rearm_compare_failed")
        return self._store.read()
