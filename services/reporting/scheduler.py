"""Durable daily scheduling with explicit timezone/DST and recovery policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import re
from pathlib import Path
from threading import Event
from time import sleep
from typing import Callable
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, make_url

UTC = timezone.utc
_SAFE_ERROR = re.compile(r"^[a-z0-9_]{1,100}$")


class LeaseLostError(RuntimeError):
    """The caller no longer owns the persisted report-run lease."""


@dataclass(frozen=True, slots=True)
class ReportRunLease:
    job_key: str
    scheduled_for: datetime
    run_id: str
    lease_owner: str
    attempt: int


@dataclass(frozen=True, slots=True)
class ReportRunOutcome:
    run_id: str
    status: str
    attempt: int
    error_code: str | None = None


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field}_timezone_required")
    return value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat(timespec="microseconds")


def _run_id(job_key: str, scheduled_for: datetime) -> str:
    material = "alix.report.run.v1:" + job_key + ":" + _stamp(scheduled_for)
    return sha256(material.encode("utf-8")).hexdigest()


def scheduled_time_for_date(
    run_date: date,
    timezone_name: str,
    *,
    hour: int = 7,
    minute: int = 0,
    fold_policy: str = "first",
) -> datetime:
    """Resolve a wall time; DST gaps shift to first valid minute, folds choose one side."""
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("scheduled_wall_time_out_of_range")
    if fold_policy not in {"first", "second"}:
        raise ValueError("scheduled_fold_policy_invalid")
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("reporting_timezone_invalid") from None
    intended = datetime.combine(run_date, time(hour=hour, minute=minute))
    for offset in range(181):
        wall = intended + timedelta(minutes=offset)
        candidates: list[datetime] = []
        for fold in (0, 1):
            local = wall.replace(tzinfo=zone, fold=fold)
            round_trip = local.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
            if round_trip == wall and all(
                local.astimezone(UTC) != item.astimezone(UTC) for item in candidates
            ):
                candidates.append(local)
        if candidates:
            candidates.sort(key=lambda item: item.astimezone(UTC))
            return candidates[0] if fold_policy == "first" else candidates[-1]
    raise ValueError("reporting_timezone_gap_exceeds_three_hours")


def next_daily_run(
    after: datetime,
    timezone_name: str,
    *,
    hour: int = 7,
    minute: int = 0,
    fold_policy: str = "first",
) -> datetime:
    """Return the next daily local wall time strictly after the supplied instant."""
    current = _utc(after, "after")
    try:
        local_date = current.astimezone(ZoneInfo(timezone_name)).date()
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("reporting_timezone_invalid") from None
    for offset in range(370):
        result = scheduled_time_for_date(
            local_date + timedelta(days=offset),
            timezone_name,
            hour=hour,
            minute=minute,
            fold_policy=fold_policy,
        )
        if result.astimezone(UTC) > current:
            return result
    raise ValueError("next_daily_run_not_found")


class ReportSchedulerStore:
    """Durable single-flight schedule state. SQLite is one-host; PostgreSQL is shared."""

    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("scheduler_database_url_required")
        parsed = make_url(database_url)
        if (
            parsed.get_backend_name() == "sqlite"
            and parsed.database is not None
            and parsed.database not in ("", ":memory:")
        ):
            Path(parsed.database).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        options: dict[str, object] = {"pool_pre_ping": True}
        if parsed.get_backend_name() == "sqlite":
            options["connect_args"] = {"timeout": 10}
        self._engine: Engine = create_engine(database_url, **options)
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text("""
                    CREATE TABLE IF NOT EXISTS alix_report_runs (
                        job_key VARCHAR(100) NOT NULL,
                        scheduled_for VARCHAR(40) NOT NULL,
                        run_id VARCHAR(64) NOT NULL UNIQUE,
                        status VARCHAR(24) NOT NULL,
                        lease_owner VARCHAR(64),
                        lease_until VARCHAR(40),
                        attempt INTEGER NOT NULL DEFAULT 0,
                        started_at VARCHAR(40) NOT NULL,
                        finished_at VARCHAR(40),
                        error_code VARCHAR(100),
                        PRIMARY KEY (job_key, scheduled_for)
                    )
                """)
                )
                connection.execute(
                    text("""
                    CREATE TABLE IF NOT EXISTS alix_report_run_events (
                        event_id VARCHAR(64) PRIMARY KEY,
                        run_id VARCHAR(64) NOT NULL,
                        occurred_at VARCHAR(40) NOT NULL,
                        event_type VARCHAR(32) NOT NULL,
                        status VARCHAR(24) NOT NULL,
                        error_code VARCHAR(100)
                    )
                """)
                )
        except Exception:
            self._engine.dispose()
            raise

    def close(self) -> None:
        self._engine.dispose()

    @staticmethod
    def _event(
        connection: Connection,
        run_id: str,
        now: datetime,
        event_type: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        connection.execute(
            text("""
            INSERT INTO alix_report_run_events
                (event_id, run_id, occurred_at, event_type, status, error_code)
            VALUES (:event_id, :run_id, :occurred_at, :event_type, :status, :error_code)
        """),
            {
                "event_id": uuid4().hex,
                "run_id": run_id,
                "occurred_at": _stamp(now),
                "event_type": event_type,
                "status": status,
                "error_code": error_code,
            },
        )

    def claim(
        self,
        job_key: str,
        scheduled_for: datetime,
        *,
        now: datetime,
        lease_seconds: int = 900,
        max_attempts: int = 3,
    ) -> ReportRunLease | None:
        if not job_key.strip() or len(job_key) > 100:
            raise ValueError("scheduler_job_key_invalid")
        if lease_seconds < 1 or max_attempts < 1:
            raise ValueError("scheduler_retry_or_lease_invalid")
        scheduled, now_text = _stamp(scheduled_for), _stamp(now)
        until = _stamp(_utc(now, "now") + timedelta(seconds=lease_seconds))
        run_id, token = _run_id(job_key, scheduled_for), uuid4().hex
        with self._engine.begin() as connection:
            connection.execute(
                text("""
                INSERT INTO alix_report_runs
                    (job_key, scheduled_for, run_id, status, lease_owner, lease_until,
                     attempt, started_at, finished_at, error_code)
                VALUES (:job, :scheduled, :run_id, 'CLAIMED', :owner, :until,
                        1, :now, NULL, NULL)
                ON CONFLICT (job_key, scheduled_for) DO NOTHING
            """),
                {
                    "job": job_key,
                    "scheduled": scheduled,
                    "run_id": run_id,
                    "owner": token,
                    "until": until,
                    "now": now_text,
                },
            )
            row = (
                connection.execute(
                    text("""
                SELECT run_id, status, lease_owner, lease_until, attempt
                FROM alix_report_runs WHERE job_key=:job AND scheduled_for=:scheduled
            """),
                    {"job": job_key, "scheduled": scheduled},
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            if row["lease_owner"] == token and row["status"] == "CLAIMED":
                self._event(connection, run_id, now, "CLAIMED", "CLAIMED")
                return ReportRunLease(
                    job_key, _utc(scheduled_for, "scheduled_for"), run_id, token, 1
                )
            result = connection.execute(
                text("""
                UPDATE alix_report_runs
                SET status='CLAIMED', lease_owner=:owner, lease_until=:until,
                    attempt=attempt+1, started_at=:now, finished_at=NULL, error_code=NULL
                WHERE job_key=:job AND scheduled_for=:scheduled AND attempt < :max_attempts
                  AND (status='FAILED' AND lease_until <= :now
                       OR status='CLAIMED' AND lease_until <= :now)
            """),
                {
                    "owner": token,
                    "until": until,
                    "now": now_text,
                    "job": job_key,
                    "scheduled": scheduled,
                    "max_attempts": max_attempts,
                },
            )
            if result.rowcount != 1:
                return None
            current = (
                connection.execute(
                    text("""
                SELECT attempt FROM alix_report_runs
                WHERE job_key=:job AND scheduled_for=:scheduled
            """),
                    {"job": job_key, "scheduled": scheduled},
                )
                .mappings()
                .first()
            )
            attempt = int(current["attempt"]) if current is not None else 2
            self._event(connection, run_id, now, "RECLAIMED", "CLAIMED")
            return ReportRunLease(
                job_key, _utc(scheduled_for, "scheduled_for"), run_id, token, attempt
            )

    def finish(
        self,
        lease: ReportRunLease,
        *,
        success: bool,
        now: datetime,
        error_code: str | None = None,
        retry_delay_seconds: int = 60,
    ) -> None:
        if retry_delay_seconds < 1:
            raise ValueError("scheduler_retry_delay_invalid")
        status = "SUCCEEDED" if success else "FAILED"
        safe_error = None
        if not success:
            candidate = (error_code or "report_job_failed").lower()
            safe_error = (
                candidate if _SAFE_ERROR.fullmatch(candidate) else "unclassified_error"
            )
        next_retry = (
            _stamp(_utc(now, "now") + timedelta(seconds=retry_delay_seconds))
            if not success
            else None
        )
        with self._engine.begin() as connection:
            result = connection.execute(
                text("""
                UPDATE alix_report_runs
                SET status=:status, finished_at=:finished, lease_owner=NULL,
                    lease_until=:retry_after, error_code=:error
                WHERE job_key=:job AND scheduled_for=:scheduled AND run_id=:run_id
                  AND status='CLAIMED' AND lease_owner=:owner
            """),
                {
                    "status": status,
                    "finished": _stamp(now),
                    "retry_after": next_retry,
                    "error": safe_error,
                    "job": lease.job_key,
                    "scheduled": _stamp(lease.scheduled_for),
                    "run_id": lease.run_id,
                    "owner": lease.lease_owner,
                },
            )
            if result.rowcount != 1:
                raise LeaseLostError("report_scheduler_lease_lost")
            self._event(connection, lease.run_id, now, "FINISHED", status, safe_error)

    def skip_late(
        self, job_key: str, scheduled_for: datetime, *, now: datetime
    ) -> bool:
        scheduled, run_id = _stamp(scheduled_for), _run_id(job_key, scheduled_for)
        with self._engine.begin() as connection:
            result = connection.execute(
                text("""
                INSERT INTO alix_report_runs
                    (job_key, scheduled_for, run_id, status, lease_owner, lease_until,
                     attempt, started_at, finished_at, error_code)
                VALUES (:job, :scheduled, :run_id, 'SKIPPED_LATE', NULL, NULL,
                        0, :now, :now, 'catch_up_window_expired')
                ON CONFLICT (job_key, scheduled_for) DO NOTHING
            """),
                {
                    "job": job_key,
                    "scheduled": scheduled,
                    "run_id": run_id,
                    "now": _stamp(now),
                },
            )
            if result.rowcount != 1:
                return False
            self._event(
                connection,
                run_id,
                now,
                "SKIPPED_LATE",
                "SKIPPED_LATE",
                "catch_up_window_expired",
            )
            return True

    def status_for(self, job_key: str, scheduled_for: datetime) -> str | None:
        with self._engine.connect() as connection:
            result = connection.execute(
                text("""
                SELECT status FROM alix_report_runs
                WHERE job_key=:job AND scheduled_for=:scheduled
            """),
                {"job": job_key, "scheduled": _stamp(scheduled_for)},
            ).first()
        return str(result[0]) if result else None


def execute_once(
    store: ReportSchedulerStore,
    *,
    job_key: str,
    scheduled_for: datetime,
    now: datetime,
    job: Callable[[str], object],
    max_attempts: int = 3,
) -> ReportRunOutcome | None:
    lease = store.claim(job_key, scheduled_for, now=now, max_attempts=max_attempts)
    if lease is None:
        return None
    try:
        job(lease.run_id)
    except Exception as exc:
        code = type(exc).__name__.lower()
        safe_code = code if _SAFE_ERROR.fullmatch(code) else "report_job_failed"
        delay = min(60 * (2 ** (lease.attempt - 1)), 600)
        store.finish(
            lease,
            success=False,
            now=now,
            error_code=safe_code,
            retry_delay_seconds=delay,
        )
        return ReportRunOutcome(lease.run_id, "FAILED", lease.attempt, safe_code)
    store.finish(lease, success=True, now=now)
    return ReportRunOutcome(lease.run_id, "SUCCEEDED", lease.attempt)


def run_daily_scheduler(
    store: ReportSchedulerStore,
    *,
    job_key: str,
    timezone_name: str,
    job: Callable[[str], object],
    stop_event: Event,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    interval_seconds: int = 30,
    catch_up_window: timedelta = timedelta(hours=6),
    sleep_fn: Callable[[float], None] = sleep,
) -> None:
    """Explicit opt-in daemon loop; missed runs beyond six hours are recorded as skipped."""
    if interval_seconds < 1 or catch_up_window.total_seconds() < 0:
        raise ValueError("scheduler_poll_or_catch_up_policy_invalid")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("reporting_timezone_invalid") from None
    while not stop_event.is_set():
        now = _utc(clock(), "scheduler_clock")
        scheduled = scheduled_time_for_date(
            now.astimezone(ZoneInfo(timezone_name)).date(), timezone_name
        )
        occurrence = scheduled.astimezone(UTC)
        if now >= occurrence:
            if now <= occurrence + catch_up_window:
                execute_once(
                    store, job_key=job_key, scheduled_for=scheduled, now=now, job=job
                )
            else:
                store.skip_late(job_key, scheduled, now=now)
        sleep_fn(float(interval_seconds))
