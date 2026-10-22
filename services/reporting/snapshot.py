"""Compose read-only source snapshots; reconciliation is a hard gate for financial truth."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from services.reporting.contracts import (
    REPORT_SCHEMA_VERSION,
    ReportingSnapshot,
    ReportingSource,
    SourceSnapshot,
)
from services.reporting.sources import ReportingSourceError

_REQUIRED = frozenset({"freqtrade_trade_db", "decision_ledger", "account_snapshot"})


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("report_timestamp_timezone_required")
    return value.astimezone(timezone.utc)


def _unavailable(source_type: str, cutoff: datetime, code: str) -> SourceSnapshot:
    safe = code if code.replace("_", "").isalnum() else "source_unavailable"
    return SourceSnapshot(
        source_id="unavailable-" + source_type,
        source_type=source_type,
        status="UNAVAILABLE",
        observed_at=datetime.now(timezone.utc),
        data_cutoff=cutoff,
        source_revision="unavailable-v1",
        authoritative=False,
        reconciled=False,
        summary={},
        error_code=safe,
    )


def build_reporting_snapshot(
    sources: Iterable[ReportingSource],
    *,
    generated_at: datetime,
    data_cutoff: datetime,
    timezone_name: str,
    scope: str,
    code_revision: str,
) -> ReportingSnapshot:
    generated, cutoff = _utc(generated_at), _utc(data_cutoff)
    if not scope.strip() or not code_revision.strip():
        raise ValueError("report_scope_or_code_revision_missing")
    try:
        report_date = generated.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("reporting_timezone_invalid") from None
    results: list[SourceSnapshot] = []
    for source in sources:
        try:
            result = source.read(cutoff=cutoff)
            if _utc(result.data_cutoff) != cutoff:
                raise ReportingSourceError("source_cutoff_mismatch")
            results.append(result)
        except ReportingSourceError as exc:
            results.append(_unavailable(source.source_type, cutoff, exc.code))
        except Exception:
            results.append(
                _unavailable(source.source_type, cutoff, "unexpected_source_error")
            )
    present = {row.source_type for row in results}
    for missing in sorted(_REQUIRED - present):
        results.append(_unavailable(missing, cutoff, "required_source_not_configured"))
    by_type = {row.source_type: row for row in results}
    trade, ledger, account = (
        by_type["freqtrade_trade_db"],
        by_type["decision_ledger"],
        by_type["account_snapshot"],
    )
    linked = bool(
        trade.status == "OK"
        and ledger.status == "OK"
        and account.status == "OK"
        and trade.reconciled
        and ledger.reconciled
        and account.reconciled
        and ledger.summary.get("trade_linkage_ok") is True
        and account.summary.get("order_reconciliation_status") == "RECONCILED"
    )
    mismatch = any(
        row.status == "DEGRADED" or row.error_code == "source_cutoff_mismatch"
        for row in results
    )
    reconciliation = (
        "RECONCILED" if linked else ("MISMATCH" if mismatch else "NOT_RECONCILED")
    )
    metrics: dict[str, object] = {
        "net_pnl": None,
        "net_pnl_status": "NOT_RECONCILED",
        "gross_pnl": None,
        "total_fees": None,
        "slippage_estimate": None,
        "equity": None,
        "drawdown": None,
        "exposure": None,
        "closed_trade_count": None,
        "open_trade_count": None,
        "win_rate": None,
        "win_rate_sample_size": 0,
        "win_rate_time_period": None,
        "synthetic_results_are_financial_truth": False,
    }
    if trade.status == "OK":
        metrics["closed_trade_count"] = trade.summary.get("closed_trade_count")
        metrics["open_trade_count"] = trade.summary.get("open_trade_count")
        metrics["freqtrade_reported_profit_abs_sum"] = trade.summary.get(
            "freqtrade_reported_profit_abs_sum"
        )
        metrics["freqtrade_profit_status"] = "REPORTED_NOT_RECONCILED"
    if ledger.status == "SYNTHETIC_ONLY":
        metrics["synthetic_decision_count"] = ledger.summary.get("decision_count")
        metrics["synthetic_simulated_count"] = ledger.summary.get("simulated_count")
        metrics["synthetic_net_pnl"] = ledger.summary.get("synthetic_net_pnl")
        metrics["synthetic_net_pnl_status"] = "SYNTHETIC_ONLY_NOT_RECONCILED"
    required_ok = all(
        by_type[key].status == "OK" and by_type[key].reconciled for key in _REQUIRED
    )
    status = "READY" if required_ok and reconciliation == "RECONCILED" else "DEGRADED"
    report_id = (
        "alix-"
        + sha256(
            (scope + ":" + report_date + ":" + REPORT_SCHEMA_VERSION).encode()
        ).hexdigest()[:24]
    )
    return ReportingSnapshot(
        report_id=report_id,
        report_date=report_date,
        generated_at=generated,
        data_cutoff=cutoff,
        timezone_name=timezone_name,
        code_revision=code_revision,
        schema_version=REPORT_SCHEMA_VERSION,
        status=status,
        reconciliation_status=reconciliation,
        sources=tuple(results),
        metrics=metrics,
    )
