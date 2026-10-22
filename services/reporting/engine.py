"""Deterministic reporting boundary; outbound delivery is injected and disabled by default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from services.reporting.contracts import ReportingSnapshot
from services.reporting.png import render_equity_chart
from services.notifications.bale import (
    DEFAULT_BALE_CHAT_ID,
    BaleNotificationService,
    NotificationResult,
)


@dataclass(frozen=True)
class DailyReport:
    date: str
    summary: str
    metrics: dict[str, float]


class ReportingEngine:
    def __init__(self, default_chat_id: str | None = None) -> None:
        resolved_chat_id = (
            default_chat_id
            if default_chat_id is not None
            else os.getenv("BALE_CHAT_ID", DEFAULT_BALE_CHAT_ID)
        )
        self.default_chat_id = resolved_chat_id.strip()

    def render_snapshot_text(self, report: ReportingSnapshot) -> str:
        """Render reconciled or degraded source reports without overstating PnL."""
        lines = [
            f"AliX Daily Report | {report.report_date}",
            f"report_id={report.report_id}",
            f"report_revision={report.revision}",
            f"snapshot_hash={report.snapshot_hash}",
            f"data_cutoff={report.data_cutoff.isoformat()}",
            f"timezone={report.timezone_name}",
            f"status={report.status}",
            f"reconciliation_status={report.reconciliation_status}",
        ]
        if report.reconciliation_status != "RECONCILED":
            lines.append("FINANCIAL RESULTS NOT RECONCILED: PnL/equity/drawdown are uncertified.")
        for source in report.sources:
            lines.append(
                f"source={source.source_type} status={source.status} "
                f"reconciled={str(source.reconciled).lower()}"
            )
        for key, value in sorted(report.metrics.items()):
            lines.append(f"{key}={value}")
        lines.extend([
            "order_permission=false", "Champion=null",
            "promotion_status=BLOCKED_NO_ECONOMIC_GATE", "trading_authorization=NO-GO",
        ])
        return "\n".join(lines) + "\n"

    def render_snapshot_png(self, report: ReportingSnapshot) -> bytes:
        """Return a versioned PNG; unreconciled account metrics are never plotted."""
        values: list[float] = []
        raw = report.metrics.get("reconciled_equity_curve")
        if report.reconciliation_status == "RECONCILED" and isinstance(raw, list):
            values = [float(value) for value in raw]
        return render_equity_chart(values, reconciliation_status=report.reconciliation_status)

    def render_text(self, report: DailyReport) -> str:
        metrics = ", ".join(
            f"{key}={value:.4f}" for key, value in sorted(report.metrics.items())
        )
        return f"AliX paper report | {report.date}\n{report.summary}\n{metrics}"

    async def deliver_bale(
        self,
        report: DailyReport,
        *,
        service: BaleNotificationService | None = None,
    ) -> NotificationResult:
        notifier = service or BaleNotificationService.from_environment(
            default_chat_id=self.default_chat_id
        )
        return await notifier.send_text(self.render_text(report))

    def deliver(
        self,
        report: DailyReport,
        *,
        enabled: bool = False,
        chat_id: str | None = None,
        allowlist: set[str] | None = None,
        sender: Callable[[str, str], bool] | None = None,
    ) -> bool:
        recipient = (chat_id or self.default_chat_id).strip()
        if (
            not enabled
            or not recipient
            or allowlist is None
            or recipient not in allowlist
            or sender is None
        ):
            return False
        try:
            return bool(sender(recipient, self.render_text(report)))
        except Exception:
            return False
