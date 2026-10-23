"""Compose attributed read-only sources; unreconciled data never certifies financial truth."""

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


def _report_date(generated: datetime, timezone_name: str) -> str:
    try:
        return generated.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("reporting_timezone_invalid") from None


def _read_source(source: ReportingSource, cutoff: datetime) -> SourceSnapshot:
    try:
        result = source.read(cutoff=cutoff)
        if _utc(result.data_cutoff) != cutoff:
            raise ReportingSourceError("source_cutoff_mismatch")
        return result
    except ReportingSourceError as exc:
        return _unavailable(source.source_type, cutoff, exc.code)
    except Exception:
        return _unavailable(source.source_type, cutoff, "unexpected_source_error")


def _read_sources(
    sources: Iterable[ReportingSource], cutoff: datetime
) -> list[SourceSnapshot]:
    results = [_read_source(source, cutoff) for source in sources]
    present = {source.source_type for source in results}
    for missing in sorted(_REQUIRED - present):
        results.append(_unavailable(missing, cutoff, "required_source_not_configured"))
    return results


def _linked_sources(
    trade: SourceSnapshot, ledger: SourceSnapshot, account: SourceSnapshot
) -> bool:
    return bool(
        trade.status == "OK"
        and ledger.status == "OK"
        and account.status == "OK"
        and trade.reconciled
        and ledger.reconciled
        and account.reconciled
        and ledger.summary.get("trade_linkage_ok") is True
        and account.summary.get("order_reconciliation_status") == "RECONCILED"
    )


def _reconciliation_status(
    results: list[SourceSnapshot],
    trade: SourceSnapshot,
    ledger: SourceSnapshot,
    account: SourceSnapshot,
) -> str:
    if _linked_sources(trade, ledger, account):
        return "RECONCILED"
    mismatch = any(
        source.status == "DEGRADED" or source.error_code == "source_cutoff_mismatch"
        for source in results
    )
    return "MISMATCH" if mismatch else "NOT_RECONCILED"


def _base_metrics() -> dict[str, object]:
    return {
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


def _add_trade_metrics(metrics: dict[str, object], trade: SourceSnapshot) -> None:
    if trade.status != "OK":
        return
    metrics["closed_trade_count"] = trade.summary.get("closed_trade_count")
    metrics["open_trade_count"] = trade.summary.get("open_trade_count")
    metrics["freqtrade_reported_profit_abs_sum"] = trade.summary.get(
        "freqtrade_reported_profit_abs_sum"
    )
    metrics["freqtrade_profit_status"] = "REPORTED_NOT_RECONCILED"


def _add_ledger_metrics(metrics: dict[str, object], ledger: SourceSnapshot) -> None:
    if ledger.status != "SYNTHETIC_ONLY":
        return
    metrics["synthetic_decision_count"] = ledger.summary.get("decision_count")
    metrics["synthetic_simulated_count"] = ledger.summary.get("simulated_count")
    metrics["synthetic_net_pnl"] = ledger.summary.get("synthetic_net_pnl")
    metrics["synthetic_net_pnl_status"] = "SYNTHETIC_ONLY_NOT_RECONCILED"


def _build_metrics(trade: SourceSnapshot, ledger: SourceSnapshot) -> dict[str, object]:
    metrics = _base_metrics()
    _add_trade_metrics(metrics, trade)
    _add_ledger_metrics(metrics, ledger)
    return metrics


def _all_required_sources_reconciled(
    by_type: dict[str, SourceSnapshot]
) -> bool:
    return all(
        by_type[source_type].status == "OK" and by_type[source_type].reconciled
        for source_type in _REQUIRED
    )


def _report_id(scope: str, report_date: str) -> str:
    material = scope + ":" + report_date + ":" + REPORT_SCHEMA_VERSION
    return "alix-" + sha256(material.encode()).hexdigest()[:24]


def build_reporting_snapshot(
    sources: Iterable[ReportingSource],
    *,
    generated_at: datetime,
    data_cutoff: datetime,
    timezone_name: str,
    scope: str,
    code_revision: str,
) -> ReportingSnapshot:
    """Read each configured source at a common cutoff and default to DEGRADED."""
    generated, cutoff = _utc(generated_at), _utc(data_cutoff)
    if not scope.strip() or not code_revision.strip():
        raise ValueError("report_scope_or_code_revision_missing")
    report_date = _report_date(generated, timezone_name)
    results = _read_sources(sources, cutoff)
    by_type = {source.source_type: source for source in results}
    trade, ledger, account = (
        by_type["freqtrade_trade_db"],
        by_type["decision_ledger"],
        by_type["account_snapshot"],
    )
    reconciliation = _reconciliation_status(results, trade, ledger, account)
    required_ok = _all_required_sources_reconciled(by_type)
    status = "READY" if required_ok and reconciliation == "RECONCILED" else "DEGRADED"
    return ReportingSnapshot(
        report_id=_report_id(scope, report_date),
        report_date=report_date,
        generated_at=generated,
        data_cutoff=cutoff,
        timezone_name=timezone_name,
        code_revision=code_revision,
        schema_version=REPORT_SCHEMA_VERSION,
        status=status,
        reconciliation_status=reconciliation,
        sources=tuple(results),
        metrics=_build_metrics(trade, ledger),
    )
