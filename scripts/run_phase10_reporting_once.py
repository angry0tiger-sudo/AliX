"""Explicit reporting scheduler entry point; no notifications or trading transport."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event
from typing import Any

from services.reporting.png import render_equity_chart
from services.reporting.scheduler import ReportSchedulerStore, execute_once, run_daily_scheduler
from services.reporting.snapshot import build_reporting_snapshot
from services.reporting.sources import FreqtradeSQLiteTradeSource, SQLiteSimulationLedgerSource


def _aware_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError("timestamp_must_be_iso8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp_requires_timezone")
    return parsed


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(content, bytes):
        temporary.write_bytes(content)
    else:
        temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _report_text(snapshot: Any) -> str:
    lines = [
        "AliX Daily Report | " + snapshot.report_date,
        "report_id=" + snapshot.report_id,
        "revision=" + snapshot.revision,
        "snapshot_hash=" + snapshot.snapshot_hash,
        "generated_at=" + snapshot.generated_at.isoformat(),
        "data_cutoff=" + snapshot.data_cutoff.isoformat(),
        "timezone=" + snapshot.timezone_name,
        "status=" + snapshot.status,
        "reconciliation_status=" + snapshot.reconciliation_status,
    ]
    if snapshot.reconciliation_status != "RECONCILED":
        lines.append("FINANCIAL RESULTS NOT RECONCILED: PnL/equity/drawdown are uncertified.")
    for source in snapshot.sources:
        lines.append("source=" + source.source_type + " status=" + source.status
                     + " revision=" + source.source_revision + " error=" + str(source.error_code))
    for key, value in sorted(snapshot.metrics.items()):
        lines.append(str(key) + "=" + str(value))
    lines.extend(["order_permission=false", "Champion=null",
                  "promotion_status=BLOCKED_NO_ECONOMIC_GATE", "trading_authorization=NO-GO"])
    return "\n".join(lines) + "\n"


def _build_report(*, cutoff: datetime, timezone_name: str, code_revision: str,
                  output_dir: Path, freqtrade_db: Path | None,
                  simulation_ledger_db: Path | None) -> dict[str, object]:
    sources = []
    if freqtrade_db is not None:
        sources.append(FreqtradeSQLiteTradeSource(freqtrade_db))
    if simulation_ledger_db is not None:
        sources.append(SQLiteSimulationLedgerSource(simulation_ledger_db))
    generated = datetime.now(timezone.utc)
    snapshot = build_reporting_snapshot(
        sources, generated_at=generated, data_cutoff=cutoff, timezone_name=timezone_name,
        scope="daily", code_revision=code_revision)
    payload = snapshot.to_dict()
    prefix = snapshot.report_id + "-r" + snapshot.revision
    _atomic_write(output_dir / (prefix + ".json"),
                  json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(output_dir / (prefix + ".txt"), _report_text(snapshot))
    series: list[float] = []
    maybe_series = snapshot.metrics.get("reconciled_equity_curve")
    if snapshot.reconciliation_status == "RECONCILED" and isinstance(maybe_series, list):
        series = [float(value) for value in maybe_series]
    _atomic_write(output_dir / (prefix + ".png"),
                  render_equity_chart(series, reconciliation_status=snapshot.reconciliation_status))
    return {"report_id": snapshot.report_id, "revision": snapshot.revision,
            "snapshot_hash": snapshot.snapshot_hash, "status": snapshot.status,
            "reconciliation_status": snapshot.reconciliation_status,
            "order_permission": False, "Champion": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--daemon", action="store_true", help="explicit opt-in scheduler loop")
    parser.add_argument("--scheduled-for", type=_aware_timestamp)
    parser.add_argument("--timezone", default=os.getenv("ALIx_REPORTING_TIMEZONE", "Asia/Tehran"))
    parser.add_argument("--database-url", default=os.getenv("ALIx_REPORT_SCHEDULER_DATABASE_URL",
                                                             "sqlite:///runtime/phase10-scheduler.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/reports"))
    parser.add_argument("--freqtrade-db", type=Path)
    parser.add_argument("--simulation-ledger-db", type=Path)
    parser.add_argument("--commit-sha", default=os.getenv("GITHUB_SHA", "UNPINNED"))
    args = parser.parse_args()
    if args.once and args.scheduled_for is None:
        parser.error("--scheduled-for is required with --once")
    if not args.commit_sha.strip() or args.commit_sha == "UNPINNED":
        parser.error("--commit-sha must be supplied by CI or a pinned release")
    store = ReportSchedulerStore(args.database_url)
    def job(run_id: str) -> object:
        del run_id
        cutoff = args.scheduled_for or datetime.now(timezone.utc)
        return _build_report(cutoff=cutoff, timezone_name=args.timezone,
                             code_revision=args.commit_sha, output_dir=args.output_dir,
                             freqtrade_db=args.freqtrade_db,
                             simulation_ledger_db=args.simulation_ledger_db)
    try:
        if args.once:
            outcome = execute_once(store, job_key="daily-report",
                                   scheduled_for=args.scheduled_for, now=datetime.now(timezone.utc), job=job)
            result = {"outcome": outcome.status if outcome else "SKIPPED_DUPLICATE_OR_ACTIVE",
                      "run_id": outcome.run_id if outcome else None,
                      "order_permission": False, "Champion": None}
            print(json.dumps(result, sort_keys=True))
            return 0 if outcome is None or outcome.status == "SUCCEEDED" else 1
        stop = Event()
        run_daily_scheduler(store, job_key="daily-report", timezone_name=args.timezone, job=job, stop_event=stop)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
