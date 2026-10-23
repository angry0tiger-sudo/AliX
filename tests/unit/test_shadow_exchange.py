"""Tests for Risk-Gate-bound mock execution and durable idempotency."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import sqlite3

import pytest

from risk.gate import GateDecision
from services.simulation.shadow_exchange import (
    ShadowExchange,
    ShadowLedgerConflictError,
)


def _decision(*, approved: bool = True, permission: bool = False, decision_id: str = "shadow-1") -> GateDecision:
    return GateDecision(
        decision_id=decision_id,
        evaluated_at=datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc),
        risk_approved=approved,
        risk_reduction_allowed=False,
        order_permission=permission,
        reasons=() if approved else ("daily_loss_cap",),
        estimated_loss_including_costs=Decimal("1"),
        fingerprint="a" * 64,
    )


def _execute(exchange: ShadowExchange, decision: GateDecision, confidence: str = "0.80", quantity: str = "2"):
    return exchange.simulate_round_trip(
        decision=decision,
        meta_confidence=Decimal(confidence),
        quantity=Decimal(quantity),
        entry_reference_price=Decimal("100"),
        exit_reference_price=Decimal("110"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.001"),
    )


def test_shadow_fill_applies_round_trip_slippage_fees_and_stays_synthetic(tmp_path) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    result = _execute(exchange, _decision())
    assert result.status == "SIMULATED"
    assert result.entry_fill_price == Decimal("100.100")
    assert result.exit_fill_price == Decimal("109.890")
    assert result.net_pnl == result.gross_pnl - result.entry_fee - result.exit_fee
    assert result.order_permission is False
    assert result.source_status == "SYNTHETIC_ONLY"
    stored = exchange.get_fill("shadow-1")
    assert stored == result
    assert exchange.row_count() == 1


@pytest.mark.parametrize(
    ("decision", "confidence"),
    [
        (_decision(approved=False), "0.99"),
        (_decision(permission=True), "0.99"),
        (_decision(), "0.69"),
    ],
)
def test_shadow_never_fills_if_gate_or_meta_filter_rejects(tmp_path, decision, confidence) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    result = _execute(exchange, decision, confidence)
    assert result.status == "REJECTED"
    assert result.quantity == 0
    assert result.entry_fill_price == 0
    assert result.order_permission is False
    assert exchange.row_count() == 1


def test_shadow_replay_is_idempotent_and_conflicting_reuse_is_rejected(tmp_path) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    first = _execute(exchange, _decision())
    replay = _execute(exchange, _decision())
    assert replay == first
    assert exchange.row_count() == 1
    with pytest.raises(ShadowLedgerConflictError, match="payload_conflict"):
        _execute(exchange, _decision(), quantity="3")
    assert exchange.row_count() == 1


def test_shadow_validation_and_append_only_invariants(tmp_path) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    with pytest.raises(ValueError, match="positive"):
        exchange.simulate_round_trip(
            decision=_decision(),
            meta_confidence=Decimal("0.8"),
            quantity=Decimal("0"),
            entry_reference_price=Decimal("100"),
            exit_reference_price=Decimal("101"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.001"),
        )
    assert exchange.get_fill("missing") is None
    with sqlite3.connect(tmp_path / "shadow.sqlite3") as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute("UPDATE shadow_round_trips SET source_status='LIVE' WHERE decision_id='shadow-1'")
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute("DELETE FROM shadow_round_trips WHERE decision_id='shadow-1'")
