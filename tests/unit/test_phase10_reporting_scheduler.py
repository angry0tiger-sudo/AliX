from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.reporting.scheduler import (
    LeaseLostError,
    ReportSchedulerStore,
    execute_once,
    next_daily_run,
    scheduled_time_for_date,
)

UTC = timezone.utc


def test_dst_gap_is_shifted_to_first_valid_minute() -> None:
    result = scheduled_time_for_date(
        date(2026, 3, 29), "Europe/Berlin", hour=2, minute=30
    )
    assert result.isoformat().startswith("2026-03-29T03:00:00+02:00")


def test_dst_fold_policy_selects_exactly_one_of_two_instants() -> None:
    first = scheduled_time_for_date(
        date(2026, 10, 25), "Europe/Berlin", hour=2, minute=30
    )
    second = scheduled_time_for_date(
        date(2026, 10, 25), "Europe/Berlin", hour=2, minute=30, fold_policy="second"
    )
    assert first.fold == 0 and second.fold == 1
    assert first.astimezone(UTC) < second.astimezone(UTC)


def test_invalid_timezone_and_wall_time_fail_closed() -> None:
    with pytest.raises(ValueError, match="reporting_timezone_invalid"):
        scheduled_time_for_date(date(2026, 1, 1), "Not/AZone")
    with pytest.raises(ValueError, match="scheduled_wall_time_out_of_range"):
        scheduled_time_for_date(date(2026, 1, 1), "UTC", hour=25)
    with pytest.raises(ValueError, match="scheduled_fold_policy_invalid"):
        scheduled_time_for_date(date(2026, 1, 1), "UTC", fold_policy="both")


def test_next_daily_run_requires_aware_timestamp_and_is_strictly_future() -> None:
    with pytest.raises(ValueError, match="after_timezone_required"):
        next_daily_run(datetime(2026, 10, 23, 7, 0), "UTC")
    now = datetime(2026, 10, 23, 3, 30, tzinfo=UTC)
    result = next_daily_run(now, "Asia/Tehran")
    assert result.astimezone(UTC) > now
    assert result.hour == 7


def test_scheduler_claim_is_idempotent_and_completed_run_cannot_replay(
    tmp_path: Path,
) -> None:
    db = tmp_path / "scheduler.sqlite3"
    first_store = ReportSchedulerStore("sqlite:///" + str(db))
    second_store = ReportSchedulerStore("sqlite:///" + str(db))
    scheduled = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)
    lease = first_store.claim("daily", scheduled, now=scheduled)
    assert lease is not None
    assert second_store.claim("daily", scheduled, now=scheduled) is None
    first_store.finish(lease, success=True, now=scheduled + timedelta(seconds=1))
    assert (
        second_store.claim("daily", scheduled, now=scheduled + timedelta(days=1))
        is None
    )
    assert second_store.status_for("daily", scheduled) == "SUCCEEDED"
    first_store.close()
    second_store.close()


def test_expired_lease_is_reclaimed_and_old_lease_cannot_finish(tmp_path: Path) -> None:
    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    scheduled = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)
    first = store.claim(
        "daily", scheduled, now=scheduled, lease_seconds=10, max_attempts=2
    )
    assert first is not None
    second = store.claim(
        "daily",
        scheduled,
        now=scheduled + timedelta(seconds=11),
        lease_seconds=10,
        max_attempts=2,
    )
    assert second is not None and second.attempt == 2
    with pytest.raises(LeaseLostError):
        store.finish(first, success=True, now=scheduled + timedelta(seconds=12))
    store.finish(second, success=True, now=scheduled + timedelta(seconds=13))
    store.close()


def test_failed_run_records_sanitized_error_and_enforces_backoff(
    tmp_path: Path,
) -> None:
    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    scheduled = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)
    result = execute_once(
        store,
        job_key="daily",
        scheduled_for=scheduled,
        now=scheduled,
        job=lambda _: (_ for _ in ()).throw(
            RuntimeError("secret-token-must-not-be-logged")
        ),
    )
    assert result is not None and result.status == "FAILED"
    assert result.error_code == "runtimeerror"
    assert (
        store.claim("daily", scheduled, now=scheduled + timedelta(seconds=10)) is None
    )
    retry = store.claim(
        "daily", scheduled, now=scheduled + timedelta(seconds=61), max_attempts=2
    )
    assert retry is not None and retry.attempt == 2
    store.finish(
        retry,
        success=False,
        now=scheduled + timedelta(seconds=62),
        error_code="bad token",
    )
    assert (
        store.claim(
            "daily", scheduled, now=scheduled + timedelta(seconds=180), max_attempts=2
        )
        is None
    )
    assert store.status_for("daily", scheduled) == "FAILED"
    store.close()


def test_repeat_execute_once_does_not_run_job_twice(tmp_path: Path) -> None:
    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    scheduled = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)
    calls: list[str] = []
    first = execute_once(
        store, job_key="daily", scheduled_for=scheduled, now=scheduled, job=calls.append
    )
    second = execute_once(
        store, job_key="daily", scheduled_for=scheduled, now=scheduled, job=calls.append
    )
    assert first is not None and first.status == "SUCCEEDED"
    assert second is None
    assert len(calls) == 1
    store.close()


def test_explicit_skip_late_is_a_terminal_state(tmp_path: Path) -> None:
    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    scheduled = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)
    assert store.skip_late("daily", scheduled, now=scheduled + timedelta(hours=7))
    assert not store.skip_late("daily", scheduled, now=scheduled + timedelta(hours=8))
    assert store.status_for("daily", scheduled) == "SKIPPED_LATE"
    assert store.claim("daily", scheduled, now=scheduled + timedelta(hours=8)) is None
    store.close()


def test_bad_database_url_does_not_create_a_successful_run(tmp_path: Path) -> None:
    path = tmp_path / "plain-file"
    path.write_text("not a sqlite database", encoding="utf-8")
    with pytest.raises(Exception):
        store = ReportSchedulerStore("sqlite:///" + str(path))
        store.claim("daily", datetime.now(UTC), now=datetime.now(UTC))


def test_daily_scheduler_runs_once_with_injected_clock(tmp_path: Path) -> None:
    from threading import Event
    from services.reporting.scheduler import run_daily_scheduler

    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    stop = Event()
    now = datetime(2026, 10, 23, 7, 1, tzinfo=UTC)
    calls: list[str] = []

    def stop_after_tick(seconds: float) -> None:
        assert seconds == 30.0
        stop.set()

    run_daily_scheduler(
        store,
        job_key="daily",
        timezone_name="UTC",
        job=calls.append,
        stop_event=stop,
        clock=lambda: now,
        sleep_fn=stop_after_tick,
    )
    assert len(calls) == 1
    assert (
        store.status_for("daily", datetime(2026, 10, 23, 7, 0, tzinfo=UTC))
        == "SUCCEEDED"
    )
    store.close()


def test_daily_scheduler_skips_stale_catch_up(tmp_path: Path) -> None:
    from threading import Event
    from services.reporting.scheduler import run_daily_scheduler

    store = ReportSchedulerStore("sqlite:///" + str(tmp_path / "scheduler.sqlite3"))
    stop = Event()
    now = datetime(2026, 10, 23, 14, 0, tzinfo=UTC)

    def stop_after_tick(_: float) -> None:
        stop.set()

    run_daily_scheduler(
        store,
        job_key="daily",
        timezone_name="UTC",
        job=lambda _: pytest.fail("out-of-window catch-up must not run"),
        stop_event=stop,
        clock=lambda: now,
        sleep_fn=stop_after_tick,
    )
    assert (
        store.status_for("daily", datetime(2026, 10, 23, 7, 0, tzinfo=UTC))
        == "SKIPPED_LATE"
    )
    store.close()
