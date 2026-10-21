"""Deterministic reporting boundary; outbound delivery is injected and disabled by default."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class DailyReport:
    date: str
    summary: str
    metrics: dict[str, float]

class ReportingEngine:
    def render_text(self, report: DailyReport) -> str:
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in sorted(report.metrics.items()))
        return f"AliX paper report | {report.date}\n{report.summary}\n{metrics}"

    def deliver(self, report: DailyReport, *, enabled: bool = False,
                chat_id: str | None = None, allowlist: set[str] | None = None,
                sender: Callable[[str, str], bool] | None = None) -> bool:
        if not enabled or not chat_id or allowlist is None or chat_id not in allowlist or sender is None:
            return False
        try:
            return bool(sender(chat_id, self.render_text(report)))
        except Exception:
            return False
