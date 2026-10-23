"""Mock-only round-trip execution with mandatory Risk Gate approval and a ledger."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

from risk.gate import GateDecision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
META_CONFIDENCE_FLOOR = Decimal("0.70")
SHADOW_ENGINE_REVISION = "shadow-exchange-v1"


class ShadowLedgerConflictError(RuntimeError):
    """A replay attempted to reuse a decision ID for different synthetic content."""


@dataclass(frozen=True, slots=True)
class ShadowFill:
    """Persisted synthetic fill. It can never represent an exchange order."""

    decision_id: str
    status: Literal["SIMULATED", "REJECTED"]
    input_fingerprint: str
    gate_fingerprint: str
    meta_confidence: Decimal
    quantity: Decimal
    entry_fill_price: Decimal
    exit_fill_price: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    rejection_reasons: tuple[str, ...] = ()
    order_permission: Literal[False] = False
    source_status: Literal["SYNTHETIC_ONLY"] = "SYNTHETIC_ONLY"

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("shadow_decision_id_required")
        if _SHA256.fullmatch(self.input_fingerprint) is None:
            raise ValueError("shadow_input_fingerprint_invalid")
        if _SHA256.fullmatch(self.gate_fingerprint) is None:
            raise ValueError("shadow_gate_fingerprint_invalid")
        if self.order_permission is not False or self.source_status != "SYNTHETIC_ONLY":
            raise ValueError("shadow_permission_or_source_invariant_failed")
        if not self.meta_confidence.is_finite() or not Decimal(
            "0"
        ) <= self.meta_confidence <= Decimal("1"):
            raise ValueError("shadow_meta_confidence_invalid")
        for name in (
            "quantity",
            "entry_fill_price",
            "exit_fill_price",
            "entry_fee",
            "exit_fee",
            "gross_pnl",
            "net_pnl",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"shadow_{name}_must_be_finite_decimal")
        if (
            min(
                self.quantity,
                self.entry_fill_price,
                self.exit_fill_price,
                self.entry_fee,
                self.exit_fee,
            )
            < 0
        ):
            raise ValueError("shadow_amount_must_not_be_negative")
        if self.status not in {"SIMULATED", "REJECTED"}:
            raise ValueError("shadow_status_invalid")
        if (
            self.status == "SIMULATED"
            and min(self.quantity, self.entry_fill_price, self.exit_fill_price) <= 0
        ):
            raise ValueError("shadow_fill_prices_and_quantity_must_be_positive")
        if self.status == "SIMULATED" and self.rejection_reasons:
            raise ValueError("shadow_filled_record_cannot_have_rejection_reasons")
        if self.status == "REJECTED" and not self.rejection_reasons:
            raise ValueError("shadow_rejected_record_requires_reason")
        if self.status == "REJECTED" and any(
            value != 0
            for value in (
                self.quantity,
                self.entry_fill_price,
                self.exit_fill_price,
                self.entry_fee,
                self.exit_fee,
                self.gross_pnl,
                self.net_pnl,
            )
        ):
            raise ValueError("shadow_rejection_must_not_claim_fill")


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _record_payload(fill: ShadowFill) -> dict[str, object]:
    data = asdict(fill)
    for key, value in tuple(data.items()):
        if isinstance(value, Decimal):
            data[key] = str(value)
    return data


def _fill_from_payload(payload: dict[str, object]) -> ShadowFill:
    decimal_names = (
        "meta_confidence",
        "quantity",
        "entry_fill_price",
        "exit_fill_price",
        "entry_fee",
        "exit_fee",
        "gross_pnl",
        "net_pnl",
    )
    values: dict[str, object] = dict(payload)
    for name in decimal_names:
        values[name] = Decimal(str(payload[name]))
    reasons = payload.get("rejection_reasons", [])
    if not isinstance(reasons, list) or not all(
        isinstance(item, str) for item in reasons
    ):
        raise RuntimeError("shadow_rejection_reasons_invalid")
    values["rejection_reasons"] = tuple(reasons)
    return ShadowFill(**values)  # type: ignore[arg-type]


def _validate_inputs(
    *,
    decision: GateDecision,
    confidence: Decimal,
    quantity: Decimal,
    entry_price: Decimal,
    exit_price: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> None:
    if not isinstance(decision, GateDecision):
        raise ValueError("shadow_risk_gate_decision_required")
    if not _SHA256.fullmatch(decision.fingerprint):
        raise ValueError("shadow_risk_gate_fingerprint_invalid")
    values = {
        "confidence": confidence,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
    }
    for name, value in values.items():
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"shadow_{name}_must_be_finite_decimal")
    if not Decimal("0") <= confidence <= Decimal("1"):
        raise ValueError("shadow_confidence_out_of_range")
    if min(quantity, entry_price, exit_price) <= 0:
        raise ValueError("shadow_quantity_and_reference_prices_must_be_positive")
    if not Decimal("0") <= fee_rate <= Decimal("0.02"):
        raise ValueError("shadow_fee_rate_out_of_range")
    if not Decimal("0") <= slippage_rate <= Decimal("0.05"):
        raise ValueError("shadow_slippage_rate_out_of_range")


class ShadowExchange:
    """SQLite-backed, idempotent, long-only paper execution with explicit costs.

    A mock fill is possible only after a positive Risk Gate decision. The shadow
    exchange does not bypass, substitute for, or grant permission through that gate.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path), timeout=5, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_round_trips (
                    decision_id TEXT PRIMARY KEY,
                    input_fingerprint TEXT NOT NULL CHECK(length(input_fingerprint) = 64),
                    record_hash TEXT NOT NULL CHECK(length(record_hash) = 64),
                    payload_json TEXT NOT NULL,
                    order_permission INTEGER NOT NULL DEFAULT 0 CHECK(order_permission = 0),
                    source_status TEXT NOT NULL DEFAULT 'SYNTHETIC_ONLY'
                        CHECK(source_status = 'SYNTHETIC_ONLY')
                );
                CREATE TRIGGER IF NOT EXISTS shadow_ledger_no_update
                BEFORE UPDATE ON shadow_round_trips
                BEGIN SELECT RAISE(ABORT, 'shadow_ledger_append_only'); END;
                CREATE TRIGGER IF NOT EXISTS shadow_ledger_no_delete
                BEFORE DELETE ON shadow_round_trips
                BEGIN SELECT RAISE(ABORT, 'shadow_ledger_append_only'); END;
                """
            )

    def simulate_round_trip(
        self,
        *,
        decision: GateDecision,
        meta_confidence: Decimal,
        quantity: Decimal,
        entry_reference_price: Decimal,
        exit_reference_price: Decimal,
        fee_rate: Decimal,
        slippage_rate: Decimal,
    ) -> ShadowFill:
        """Apply modeled buy/sell slippage and per-side fees; persist once by decision ID."""
        _validate_inputs(
            decision=decision,
            confidence=meta_confidence,
            quantity=quantity,
            entry_price=entry_reference_price,
            exit_price=exit_reference_price,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        request = {
            "engine_revision": SHADOW_ENGINE_REVISION,
            "decision_id": decision.decision_id,
            "gate_fingerprint": decision.fingerprint,
            "risk_approved": decision.risk_approved,
            "order_permission": decision.order_permission,
            "meta_confidence": str(meta_confidence),
            "quantity": str(quantity),
            "entry_reference_price": str(entry_reference_price),
            "exit_reference_price": str(exit_reference_price),
            "fee_rate": str(fee_rate),
            "slippage_rate": str(slippage_rate),
        }
        input_fingerprint = hashlib.sha256(
            _canonical(request).encode("utf-8")
        ).hexdigest()
        accepted = (
            decision.risk_approved is True
            and decision.order_permission is False
            and meta_confidence >= META_CONFIDENCE_FLOOR
        )
        rejection_reasons = tuple(
            reason
            for condition, reason in (
                (decision.risk_approved is not True, "risk_gate_rejected"),
                (
                    decision.order_permission is not False,
                    "order_permission_must_remain_false",
                ),
                (meta_confidence < META_CONFIDENCE_FLOOR, "meta_confidence_below_0.70"),
            )
            if condition
        )
        if accepted:
            entry_fill = entry_reference_price * (Decimal("1") + slippage_rate)
            exit_fill = exit_reference_price * (Decimal("1") - slippage_rate)
            entry_fee = entry_fill * quantity * fee_rate
            exit_fee = exit_fill * quantity * fee_rate
            gross_pnl = (exit_fill - entry_fill) * quantity
            net_pnl = gross_pnl - entry_fee - exit_fee
            status: Literal["SIMULATED", "REJECTED"] = "SIMULATED"
        else:
            entry_fill = exit_fill = entry_fee = exit_fee = Decimal("0")
            gross_pnl = net_pnl = Decimal("0")
            quantity = Decimal("0")
            status = "REJECTED"
        fill = ShadowFill(
            decision_id=decision.decision_id,
            status=status,
            input_fingerprint=input_fingerprint,
            gate_fingerprint=decision.fingerprint,
            meta_confidence=meta_confidence,
            quantity=quantity,
            entry_fill_price=entry_fill,
            exit_fill_price=exit_fill,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            rejection_reasons=rejection_reasons,
        )
        return self._append_once(fill)

    def _append_once(self, fill: ShadowFill) -> ShadowFill:
        payload_json = _canonical(_record_payload(fill))
        record_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT input_fingerprint, record_hash, payload_json FROM shadow_round_trips WHERE decision_id = ?",
                (fill.decision_id,),
            ).fetchone()
            if prior is not None:
                if prior["input_fingerprint"] != fill.input_fingerprint:
                    raise ShadowLedgerConflictError(
                        "shadow_decision_id_payload_conflict"
                    )
                existing = json.loads(str(prior["payload_json"]))
                if not isinstance(existing, dict):
                    raise RuntimeError("shadow_ledger_payload_invalid")
                existing_json = _canonical(existing)
                existing_hash = hashlib.sha256(
                    existing_json.encode("utf-8")
                ).hexdigest()
                if existing_hash != prior["record_hash"]:
                    raise RuntimeError("shadow_ledger_record_hash_mismatch")
                connection.commit()
                return _fill_from_payload(existing)
            connection.execute(
                """
                INSERT INTO shadow_round_trips
                    (decision_id, input_fingerprint, record_hash, payload_json, order_permission, source_status)
                VALUES (?, ?, ?, ?, 0, 'SYNTHETIC_ONLY')
                """,
                (fill.decision_id, fill.input_fingerprint, record_hash, payload_json),
            )
            connection.commit()
            return fill
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_fill(self, decision_id: str) -> ShadowFill | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM shadow_round_trips WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise RuntimeError("shadow_ledger_payload_invalid")
        return _fill_from_payload(payload)

    def row_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM shadow_round_trips"
            ).fetchone()
        if row is None:
            raise RuntimeError("shadow_ledger_count_unavailable")
        return int(row["count"])
