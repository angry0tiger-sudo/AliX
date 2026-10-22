"""Versioned, provenance-aware reporting contracts; never grant trading authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Protocol

REPORT_SCHEMA_VERSION = "alix.daily-report.v1"
PNG_CHART_SCHEMA_VERSION = "alix.report-chart.v1"


def stable_hash(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field}_timezone_required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    source_id: str
    source_type: str
    status: str
    observed_at: datetime
    data_cutoff: datetime
    source_revision: str
    authoritative: bool
    reconciled: bool
    summary: dict[str, Any]
    error_code: str | None = None

    def __post_init__(self) -> None:
        _utc(self.observed_at, "source_observed_at")
        _utc(self.data_cutoff, "source_data_cutoff")
        if (
            not self.source_id.strip()
            or not self.source_type.strip()
            or not self.source_revision.strip()
        ):
            raise ValueError("report_source_identity_missing")
        if self.status not in {"OK", "DEGRADED", "UNAVAILABLE", "SYNTHETIC_ONLY"}:
            raise ValueError("report_source_status_invalid")
        if self.reconciled and self.status != "OK":
            raise ValueError("non_ok_source_cannot_claim_reconciliation")

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "source_id": self.source_id,
                "source_type": self.source_type,
                "status": self.status,
                "observed_at": _utc(self.observed_at, "observed_at").isoformat(),
                "data_cutoff": _utc(self.data_cutoff, "data_cutoff").isoformat(),
                "source_revision": self.source_revision,
                "authoritative": self.authoritative,
                "reconciled": self.reconciled,
                "summary": self.summary,
                "error_code": self.error_code,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["observed_at"] = _utc(self.observed_at, "observed_at").isoformat()
        result["data_cutoff"] = _utc(self.data_cutoff, "data_cutoff").isoformat()
        result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True, slots=True)
class ReportingSnapshot:
    report_id: str
    report_date: str
    generated_at: datetime
    data_cutoff: datetime
    timezone_name: str
    code_revision: str
    schema_version: str
    status: str
    reconciliation_status: str
    sources: tuple[SourceSnapshot, ...]
    metrics: dict[str, Any]
    order_permission: bool = False
    champion: None = None
    promotion_status: str = "BLOCKED_NO_ECONOMIC_GATE"
    trading_authorization: str = "NO-GO"

    def __post_init__(self) -> None:
        _utc(self.generated_at, "report_generated_at")
        _utc(self.data_cutoff, "report_data_cutoff")
        if self.order_permission is not False or self.champion is not None:
            raise ValueError("report_safety_state_must_remain_closed")
        if (
            self.promotion_status != "BLOCKED_NO_ECONOMIC_GATE"
            or self.trading_authorization != "NO-GO"
        ):
            raise ValueError("report_promotion_or_trading_state_must_remain_no_go")
        if self.status not in {"READY", "DEGRADED"}:
            raise ValueError("report_status_invalid")
        if self.reconciliation_status not in {
            "RECONCILED",
            "NOT_RECONCILED",
            "MISMATCH",
        }:
            raise ValueError("report_reconciliation_status_invalid")

    def _payload_without_hash(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "report_schema_version": self.schema_version,
            "report_id": self.report_id,
            "report_date": self.report_date,
            "generated_at": _utc(self.generated_at, "generated_at").isoformat(),
            "data_cutoff": _utc(self.data_cutoff, "data_cutoff").isoformat(),
            "timezone": self.timezone_name,
            "code_revision": self.code_revision,
            "status": self.status,
            "reconciliation_status": self.reconciliation_status,
            "sources": [source.to_dict() for source in self.sources],
            "metrics": self.metrics,
            "order_permission": False,
            "Champion": None,
            "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
            "trading_authorization": "NO-GO",
        }
        payload["report_revision"] = stable_hash(payload)[:12]
        return payload

    @property
    def snapshot_hash(self) -> str:
        return stable_hash(self._payload_without_hash())

    @property
    def revision(self) -> str:
        return str(self._payload_without_hash()["report_revision"])

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["snapshot_hash"] = stable_hash(payload)
        return payload


class ReportingSource(Protocol):
    source_type: str

    def read(self, *, cutoff: datetime) -> SourceSnapshot: ...
