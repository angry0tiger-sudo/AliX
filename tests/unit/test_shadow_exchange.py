"""Tests for Risk-Gate-bound mock execution and durable idempotency."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from risk.allocator import DynamicRiskAllocator
from risk.contracts import (
    AccountSnapshot,
    MarketRules,
    MarketSnapshot,
    ModelRiskProposal,
    RiskPolicy,
    Side,
)
from risk.gate import GateDecision, RiskGate
from risk.kill_switch import KillSwitchState
from services.simulation.shadow_exchange import (
    ShadowExchange,
    ShadowFill,
    ShadowLedgerConflictError,
)


def _decision(
    *, approved: bool = True, permission: bool = False, decision_id: str = "shadow-1"
) -> GateDecision:
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


def _execute(
    exchange: ShadowExchange,
    decision: GateDecision,
    confidence: str = "0.80",
    quantity: str = "2",
) -> ShadowFill:
    return exchange.simulate_round_trip(
        decision=decision,
        meta_confidence=Decimal(confidence),
        quantity=Decimal(quantity),
        entry_reference_price=Decimal("100"),
        exit_reference_price=Decimal("110"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.001"),
    )


def test_shadow_fill_applies_round_trip_slippage_fees_and_stays_synthetic(
    tmp_path: Path,
) -> None:
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
def test_shadow_never_fills_if_gate_or_meta_filter_rejects(
    tmp_path: Path, decision: GateDecision, confidence: str
) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    result = _execute(exchange, decision, confidence)
    assert result.status == "REJECTED"
    assert result.quantity == 0
    assert result.entry_fill_price == 0
    assert result.order_permission is False
    assert result.rejection_reasons
    assert exchange.row_count() == 1


def test_shadow_replay_is_idempotent_and_conflicting_reuse_is_rejected(
    tmp_path: Path,
) -> None:
    exchange = ShadowExchange(tmp_path / "shadow.sqlite3")
    first = _execute(exchange, _decision())
    replay = _execute(exchange, _decision())
    assert replay == first
    assert exchange.row_count() == 1
    with pytest.raises(ShadowLedgerConflictError, match="payload_conflict"):
        _execute(exchange, _decision(), quantity="3")
    assert exchange.row_count() == 1


def test_shadow_validation_and_append_only_invariants(tmp_path: Path) -> None:
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
            connection.execute(
                "UPDATE shadow_round_trips SET source_status='LIVE' WHERE decision_id='shadow-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute(
                "DELETE FROM shadow_round_trips WHERE decision_id='shadow-1'"
            )


def test_shadow_fill_consumes_an_actual_positive_risk_gate_decision(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    policy = RiskPolicy(
        revision="shadow-integration-policy-v1",
        owner_approved=True,
        absolute_daily_loss_cap=Decimal("100"),
        absolute_gross_exposure_cap=Decimal("10000"),
        daily_risk_budget_fraction=Decimal("0.05"),
        per_trade_risk_fraction=Decimal("0.01"),
        max_stake_fraction_of_free_balance=Decimal("0.5"),
        max_gross_exposure_fraction=Decimal("1"),
        max_correlated_exposure_fraction=Decimal("0.8"),
        confidence_floor=Decimal("0.3"),
        max_uncertainty=Decimal("0.8"),
        max_calibration_error=Decimal("0.4"),
        reference_atr_fraction=Decimal("0.02"),
        max_atr_fraction=Decimal("0.1"),
        stop_atr_multiple=Decimal("2"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.001"),
        minimum_edge_after_costs=Decimal("0.0001"),
        max_market_age_seconds=60,
        max_account_age_seconds=60,
        regime_multipliers=(("low", Decimal("1")), ("high", Decimal("0.6"))),
    )
    account = AccountSnapshot(
        as_of=now,
        currency="USDT",
        equity=Decimal("10000"),
        free_balance=Decimal("9000"),
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        fees_paid=Decimal("0"),
        open_position_risk=Decimal("0"),
        open_order_risk=Decimal("0"),
        open_order_notional=Decimal("0"),
        gross_exposure=Decimal("0"),
        correlated_exposure=Decimal("0"),
        reconciled=True,
        orders_reconciled=True,
        source_snapshot_hash="a" * 64,
    )
    market = MarketSnapshot(
        symbol="BTC/USDT",
        as_of=now,
        price=Decimal("100"),
        atr=Decimal("2"),
        regime="low",
        source_revision="shadow-market-v1",
    )
    rules = MarketRules(
        symbol="BTC/USDT",
        revision="shadow-market-rules-v1",
        approved=True,
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("1000"),
        min_notional=Decimal("10"),
        max_notional=Decimal("100000"),
    )
    proposal = ModelRiskProposal(
        decision_id="shadow-risk-gate-decision",
        model_revision="test-meta-model-v1",
        side=Side.LONG,
        regime="low",
        atr_snapshot=Decimal("2"),
        calibrated_confidence=Decimal("0.8"),
        uncertainty=Decimal("0.1"),
        calibration_error=Decimal("0.1"),
        expected_edge_after_costs=Decimal("0.01"),
    )
    allocation_result = DynamicRiskAllocator().allocate(
        policy=policy,
        account=account,
        market=market,
        rules=rules,
        proposal=proposal,
        now=now,
    )
    assert allocation_result.approved
    assert allocation_result.allocation is not None

    gate_decision = RiskGate().evaluate_entry(
        allocation=allocation_result.allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, now),
        evaluated_at=now,
    )
    assert gate_decision.risk_approved is True
    assert gate_decision.order_permission is False

    exchange = ShadowExchange(tmp_path / "actual-risk-gate.sqlite3")
    simulated = exchange.simulate_round_trip(
        decision=gate_decision,
        meta_confidence=proposal.calibrated_confidence,
        quantity=allocation_result.allocation.quantity,
        entry_reference_price=market.price,
        exit_reference_price=Decimal("101"),
        fee_rate=policy.fee_rate,
        slippage_rate=policy.slippage_rate,
    )
    assert simulated.status == "SIMULATED"
    assert simulated.source_status == "SYNTHETIC_ONLY"
    assert simulated.order_permission is False
    assert simulated.gate_fingerprint == gate_decision.fingerprint
    assert exchange.get_fill(proposal.decision_id) == simulated
