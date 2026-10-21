"""Deterministic reporting boundary; outbound delivery is injected and disabled by default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

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
        self.default_chat_id = (
            default_chat_id
            or os.getenv("BALE_CHAT_ID", DEFAULT_BALE_CHAT_ID)
        ).strip()

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
