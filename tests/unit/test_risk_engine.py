"""Deterministic tests for sizing, independent gates and latched recovery."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from risk.allocator import DynamicRiskAllocator
from risk.contracts import (
    AccountSnapshot,
    ActionType,
    MarketRules,
    MarketSnapshot,
    ModelRiskProposal,
    RiskPolicy,
    Side,
)
from risk.evaluation import OosObservation, compare_walk_forward_oos
from risk.gate import RiskGate
from risk.kill_switch import (
    KillSwitchController,
    KillSwitchState,
    ManualRearmEvidence,
    RearmRejected,
    SQLiteKillSwitchStore,
)
from risk.supervisor import (
    RiskSupervisor,
    SupervisorHealth,
    SupervisorStatus,
)

D = Decimal
NOW = datetime.now(timezone.utc) + timedelta(minutes=5)


def make_policy(**changes: Any) -> RiskPolicy:
    values: dict[str, Any] = {
        "revision": "owner-policy-v1",
        "owner_approved": True,
        "absolute_daily_loss_cap": D("100"),
        "absolute_gross_exposure_cap": D("10000"),
        "daily_risk_budget_fraction": D("0.05"),
        "per_trade_risk_fraction": D("0.01"),
        "max_stake_fraction_of_free_balance": D("0.5"),
        "max_gross_exposure_fraction": D("1"),
        "max_correlated_exposure_fraction": D("0.8"),
        "confidence_floor": D("0.3"),
        "max_uncertainty": D("0.8"),
        "max_calibration_error": D("0.4"),
        "reference_atr_fraction": D("0.02"),
        "max_atr_fraction": D("0.1"),
        "stop_atr_multiple": D("2"),
        "fee_rate": D("0.001"),
        "slippage_rate": D("0.001"),
        "minimum_edge_after_costs": D("0.0001"),
        "max_market_age_seconds": 60,
        "max_account_age_seconds": 60,
        "regime_multipliers": (("low", D("1")), ("high", D("0.6"))),
    }
    values.update(changes)
    return RiskPolicy(**values)


def make_account(**changes: Any) -> AccountSnapshot:
    values: dict[str, Any] = {
        "as_of": NOW,
        "currency": "USDT",
        "equity": D("10000"),
        "free_balance": D("9000"),
        "realized_pnl": D("0"),
        "unrealized_pnl": D("0"),
        "fees_paid": D("0"),
        "open_position_risk": D("0"),
        "open_order_risk": D("0"),
        "open_order_notional": D("0"),
        "gross_exposure": D("0"),
        "correlated_exposure": D("0"),
        "reconciled": True,
        "orders_reconciled": True,
        "source_snapshot_hash": "account-snapshot-sha256",
    }
    values.update(changes)
    return AccountSnapshot(**values)


def make_market(**changes: Any) -> MarketSnapshot:
    values: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "as_of": NOW,
        "price": D("100"),
        "atr": D("2"),
        "regime": "low",
        "source_revision": "market-snapshot-v1",
    }
    values.update(changes)
    return MarketSnapshot(**values)


def make_rules(**changes: Any) -> MarketRules:
    values: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "revision": "market-rules-v1",
        "approved": True,
        "price_tick": D("0.01"),
        "quantity_step": D("0.001"),
        "min_quantity": D("0.001"),
        "max_quantity": D("1000"),
        "min_notional": D("10"),
        "max_notional": D("100000"),
    }
    values.update(changes)
    return MarketRules(**values)


def make_proposal(**changes: Any) -> ModelRiskProposal:
    values: dict[str, Any] = {
        "decision_id": "decision-001",
        "model_revision": "model-sha256-v1",
        "side": Side.LONG,
        "regime": "low",
        "atr_snapshot": D("2"),
        "calibrated_confidence": D("0.8"),
        "uncertainty": D("0.1"),
        "calibration_error": D("0.1"),
        "expected_edge_after_costs": D("0.01"),
    }
    values.update(changes)
    return ModelRiskProposal(**values)


def make_health(**changes: Any) -> SupervisorHealth:
    values: dict[str, Any] = {
        "risk_service_healthy": True,
        "market_price_fresh": True,
        "account_reconciled": True,
        "orders_reconciled": True,
        "ledger_writable": True,
        "heartbeat_fresh": True,
        "policy_loaded": True,
    }
    values.update(changes)
    return SupervisorHealth(**values)


def make_allocation(
    *,
    policy: RiskPolicy | None = None,
    account: AccountSnapshot | None = None,
    market: MarketSnapshot | None = None,
    rules: MarketRules | None = None,
    proposal: ModelRiskProposal | None = None,
):
    selected_policy = policy or make_policy()
    result = DynamicRiskAllocator().allocate(
        policy=selected_policy,
        account=account or make_account(),
        market=market or make_market(),
        rules=rules or make_rules(),
        proposal=proposal or make_proposal(),
        now=NOW,
    )
    assert result.approved and result.allocation is not None, result.reasons
    return result.allocation


def make_armed_store(path: Path) -> SQLiteKillSwitchStore:
    store = SQLiteKillSwitchStore(str(path))
    state = store.read()
    KillSwitchController(store).manual_rearm(
        ManualRearmEvidence(
            operator_id="owner",
            owner_approval_id="owner-approval-fixture",
            reason="test-only initial setup after synthetic reconciliation",
            observed_version=state.version,
            account_reconciled=True,
            orders_reconciled=True,
            supervisor_healthy=True,
            risk_policy_unchanged=True,
        ),
        NOW,
    )
    return store


def test_allocator_reduces_size_for_volatility_uncertainty_calibration_and_regime() -> (
    None
):
    allocator = DynamicRiskAllocator()
    policy = make_policy()
    baseline = allocator.allocate(
        policy=policy,
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=make_proposal(),
        now=NOW,
    )
    assert baseline.allocation is not None

    volatile_market = make_market(atr=D("5"))
    volatile = allocator.allocate(
        policy=policy,
        account=make_account(),
        market=volatile_market,
        rules=make_rules(),
        proposal=make_proposal(atr_snapshot=D("5")),
        now=NOW,
    )
    assert volatile.allocation is not None
    assert volatile.trade_risk_budget < baseline.trade_risk_budget
    assert volatile.allocation.quantity < baseline.allocation.quantity

    uncertain = allocator.allocate(
        policy=policy,
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=make_proposal(uncertainty=D("0.4")),
        now=NOW,
    )
    assert uncertain.allocation is not None
    assert uncertain.trade_risk_budget < baseline.trade_risk_budget

    degraded_calibration = allocator.allocate(
        policy=policy,
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=make_proposal(calibration_error=D("0.3")),
        now=NOW,
    )
    assert degraded_calibration.allocation is not None
    assert degraded_calibration.trade_risk_budget < baseline.trade_risk_budget

    high_regime = make_market(regime="high")
    regime_result = allocator.allocate(
        policy=policy,
        account=make_account(),
        market=high_regime,
        rules=make_rules(),
        proposal=make_proposal(regime="high"),
        now=NOW,
    )
    assert regime_result.allocation is not None
    assert regime_result.trade_risk_budget < baseline.trade_risk_budget


def test_confidence_degradation_lowers_size_and_low_confidence_fails_closed() -> None:
    base = make_allocation()
    reduced = make_allocation(proposal=make_proposal(calibrated_confidence=D("0.5")))
    assert reduced.trade_risk_budget < base.trade_risk_budget
    rejected = DynamicRiskAllocator().allocate(
        policy=make_policy(),
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=make_proposal(calibrated_confidence=D("0.2")),
        now=NOW,
    )
    assert not rejected.approved
    assert "confidence_below_policy_floor" in rejected.reasons


def test_atr_spike_regime_shift_missing_input_and_stale_price_fail_closed() -> None:
    allocator = DynamicRiskAllocator()
    spike = make_market(atr=D("11"))
    result = allocator.allocate(
        policy=make_policy(),
        account=make_account(),
        market=spike,
        rules=make_rules(),
        proposal=make_proposal(atr_snapshot=D("11")),
        now=NOW,
    )
    assert not result.approved
    assert "atr_spike_exceeds_policy_limit" in result.reasons

    unknown_regime = make_market(regime="unknown")
    result = allocator.allocate(
        policy=make_policy(),
        account=make_account(),
        market=unknown_regime,
        rules=make_rules(),
        proposal=make_proposal(regime="unknown"),
        now=NOW,
    )
    assert not result.approved
    assert "regime_not_approved" in result.reasons

    missing = allocator.allocate(
        policy=make_policy(),
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=None,
        now=NOW,
    )
    assert not missing.approved
    assert "required_risk_input_missing" in missing.reasons

    stale_market = make_market(as_of=NOW - timedelta(seconds=120))
    stale = allocator.allocate(
        policy=make_policy(),
        account=make_account(),
        market=stale_market,
        rules=make_rules(),
        proposal=make_proposal(),
        now=NOW,
    )
    assert not stale.approved
    assert "market_price_stale_or_future" in stale.reasons


def test_allocator_fails_closed_for_policy_account_and_edge_contract_violations() -> (
    None
):
    allocator = DynamicRiskAllocator()
    args = {
        "account": make_account(),
        "market": make_market(),
        "rules": make_rules(),
        "proposal": make_proposal(),
        "now": NOW,
    }
    unapproved = allocator.allocate(policy=make_policy(owner_approved=False), **args)
    assert not unapproved.approved
    assert "risk_policy_not_owner_approved" in unapproved.reasons
    unreconciled = allocator.allocate(
        policy=make_policy(), **{**args, "account": make_account(reconciled=False)}
    )
    assert not unreconciled.approved
    assert "account_or_orders_not_reconciled" in unreconciled.reasons
    edge = allocator.allocate(
        policy=make_policy(),
        **{**args, "proposal": make_proposal(expected_edge_after_costs=D("0"))},
    )
    assert not edge.approved
    assert "edge_after_costs_not_positive_enough" in edge.reasons


def test_stake_stop_fees_slippage_and_market_precision_are_consistent() -> None:
    policy = make_policy()
    account = make_account()
    market = make_market()
    rules = make_rules()
    allocation = make_allocation(
        policy=policy, account=account, market=market, rules=rules
    )
    assert allocation.stop_price < market.price
    assert allocation.stop_distance == market.price - allocation.stop_price
    assert allocation.quantity % rules.quantity_step == 0
    assert allocation.stop_price % rules.price_tick == 0
    assert allocation.stake == allocation.quantity * market.price
    assert (
        allocation.stake
        <= account.free_balance * policy.max_stake_fraction_of_free_balance
    )
    assert (
        allocation.estimated_loss_including_costs
        > allocation.quantity * allocation.stop_distance
    )
    assert allocation.estimated_loss_including_costs <= allocation.trade_risk_budget
    assert allocation.rules_revision == rules.revision


def test_independent_gate_approves_risk_not_order_permission() -> None:
    policy = make_policy()
    account = make_account()
    market = make_market()
    rules = make_rules()
    allocation = make_allocation(
        policy=policy, account=account, market=market, rules=rules
    )
    state = KillSwitchState(False, "", None, 2, NOW)
    decision = RiskGate().evaluate_entry(
        allocation=allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=state,
        evaluated_at=NOW,
    )
    assert decision.risk_approved
    assert not decision.order_permission
    assert not decision.risk_reduction_allowed
    assert decision.estimated_loss_including_costs <= allocation.trade_risk_budget


def test_gate_rejects_invalid_stop_policy_revision_missing_switch_and_stale_price() -> (
    None
):
    policy = make_policy()
    account = make_account()
    market = make_market()
    rules = make_rules()
    allocation = make_allocation(
        policy=policy, account=account, market=market, rules=rules
    )
    state = KillSwitchState(False, "", None, 2, NOW)
    gate = RiskGate()

    invalid_stop = replace(allocation, stop_price=market.price + D("1"))
    decision = gate.evaluate_entry(
        allocation=invalid_stop,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=state,
        evaluated_at=NOW,
    )
    assert not decision.risk_approved
    assert "long_stop_not_below_entry" in decision.reasons

    decision = gate.evaluate_entry(
        allocation=allocation,
        account=account,
        market=market,
        rules=make_rules(revision="market-rules-v2"),
        policy=policy,
        kill_switch=state,
        evaluated_at=NOW,
    )
    assert not decision.risk_approved
    assert "market_rules_revision_mismatch" in decision.reasons

    decision = gate.evaluate_entry(
        allocation=allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=None,
        evaluated_at=NOW,
    )
    assert not decision.risk_approved
    assert "required_gate_input_missing" in decision.reasons

    decision = gate.evaluate_entry(
        allocation=allocation,
        account=make_account(orders_reconciled=False),
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=state,
        evaluated_at=NOW,
    )
    assert not decision.risk_approved
    assert "account_or_orders_not_reconciled" in decision.reasons


def test_absolute_cap_is_independent_of_high_model_confidence(tmp_path: Path) -> None:
    policy = make_policy()
    account = make_account(realized_pnl=D("-100"))
    market = make_market()
    proposal = make_proposal(calibrated_confidence=D("1"), uncertainty=D("0"))
    allocation_result = DynamicRiskAllocator().allocate(
        policy=policy,
        account=account,
        market=market,
        rules=make_rules(),
        proposal=proposal,
        now=NOW,
    )
    assert not allocation_result.approved
    assert "absolute_daily_loss_cap_hit" in allocation_result.reasons

    store = make_armed_store(tmp_path / "switch.db")
    decision = RiskSupervisor(store).check(
        account=account, policy=policy, health=make_health(), now=NOW
    )
    assert decision.status is SupervisorStatus.HALTED
    assert decision.reason == "absolute_daily_loss_cap_hit"
    assert store.read().latched


def test_supervisor_never_automatically_rearms_and_halts_on_exposure_cap(
    tmp_path: Path,
) -> None:
    store = SQLiteKillSwitchStore(str(tmp_path / "switch.db"))
    state = store.read()
    supervisor = RiskSupervisor(store)
    safe = supervisor.check(
        account=make_account(), policy=make_policy(), health=make_health(), now=NOW
    )
    assert safe.status is SupervisorStatus.HALTED
    assert store.read().latched
    assert store.read().version == state.version

    # Explicit manual recovery is required before this separate cap test.
    store = make_armed_store(tmp_path / "exposure.db")
    decision = RiskSupervisor(store).check(
        account=make_account(gross_exposure=D("10000")),
        policy=make_policy(),
        health=make_health(),
        now=NOW,
    )
    assert decision.status is SupervisorStatus.HALTED
    assert decision.reason == "absolute_gross_exposure_cap_hit"


def test_supervisor_api_timeout_crash_and_unknown_order_state_latch(
    tmp_path: Path,
) -> None:
    cases = (
        (
            make_health(risk_service_healthy=False),
            make_account(),
            "supervisor_health_degraded",
        ),
        (
            make_health(heartbeat_fresh=False),
            make_account(),
            "supervisor_health_degraded",
        ),
        (
            make_health(),
            make_account(orders_reconciled=False),
            "account_or_orders_not_reconciled",
        ),
    )
    for index, (health, account, expected_reason) in enumerate(cases):
        store = make_armed_store(tmp_path / f"fault-{index}.db")
        result = RiskSupervisor(store).check(
            account=account, policy=make_policy(), health=health, now=NOW
        )
        assert result.status is SupervisorStatus.HALTED
        assert result.reason == expected_reason
        assert store.read().latched


def test_risk_service_down_and_stale_account_or_market_never_approve_entry() -> None:
    allocator = DynamicRiskAllocator()
    missing = allocator.allocate(
        policy=make_policy(),
        account=make_account(),
        market=make_market(),
        rules=make_rules(),
        proposal=None,
        now=NOW,
    )
    assert not missing.approved
    assert missing.allocation is None
    assert (
        not RiskGate()
        .evaluate_entry(
            allocation=missing.allocation,
            account=make_account(),
            market=make_market(),
            rules=make_rules(),
            policy=make_policy(),
            kill_switch=KillSwitchState(False, "", None, 2, NOW),
            evaluated_at=NOW,
        )
        .risk_approved
    )

    stale_account = make_account(as_of=NOW - timedelta(seconds=120))
    stale = allocator.allocate(
        policy=make_policy(),
        account=stale_account,
        market=make_market(),
        rules=make_rules(),
        proposal=make_proposal(),
        now=NOW,
    )
    assert not stale.approved
    assert "account_snapshot_stale_or_future" in stale.reasons


def test_kill_switch_survives_store_restart_and_unknown_status_stays_halted(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "durable-switch.db")
    store = make_armed_store(Path(path))
    halted = RiskSupervisor(store).check(
        account=make_account(orders_reconciled=False),
        policy=make_policy(),
        health=make_health(orders_reconciled=False),
        now=NOW,
    )
    assert halted.status is SupervisorStatus.HALTED
    restarted_store = SQLiteKillSwitchStore(path)
    assert restarted_store.read().latched
    healthy_after_restart = RiskSupervisor(restarted_store).check(
        account=make_account(), policy=make_policy(), health=make_health(), now=NOW
    )
    assert healthy_after_restart.status is SupervisorStatus.HALTED
    assert restarted_store.read().latched


def test_kill_switch_rearm_requires_current_manual_evidence_and_is_audited(
    tmp_path: Path,
) -> None:
    store = SQLiteKillSwitchStore(str(tmp_path / "manual-rearm.db"))
    controller = KillSwitchController(store)
    state = store.read()
    bad = ManualRearmEvidence(
        operator_id="owner",
        owner_approval_id="approval-1",
        reason="reconciliation incomplete",
        observed_version=state.version,
        account_reconciled=False,
        orders_reconciled=True,
        supervisor_healthy=True,
        risk_policy_unchanged=True,
    )
    with pytest.raises(RearmRejected, match="preconditions_failed"):
        controller.manual_rearm(bad, NOW)

    stale = replace(
        bad,
        reason="stale version",
        observed_version=state.version - 1,
        account_reconciled=True,
    )
    with pytest.raises(RearmRejected, match="version_changed"):
        controller.manual_rearm(stale, NOW)

    valid = replace(
        bad,
        reason="manual re-arm after reconciled synthetic state",
        account_reconciled=True,
    )
    result = controller.manual_rearm(valid, NOW)
    assert not result.latched
    assert result.rearm_approval_id == "approval-1"
    assert any(row[1] == "MANUAL_REARM" for row in store.audit_events())


def test_risk_reductions_are_not_blocked_by_entry_halt() -> None:
    gate = RiskGate()
    result = gate.evaluate_reduction(ActionType.EXIT, "exit-1", NOW)
    assert result.risk_reduction_allowed
    assert not result.risk_approved
    assert not result.order_permission
    result = gate.evaluate_reduction(ActionType.REDUCE_RISK, "reduce-1", NOW)
    assert result.risk_reduction_allowed
    invalid = gate.evaluate_reduction(ActionType.ENTRY, "entry-1", NOW)
    assert not invalid.risk_reduction_allowed
    assert "not_a_risk_reducing_action" in invalid.reasons


def test_oos_metrics_report_drawdown_cvar_cost_turnover_and_regime_stability() -> None:
    observations = (
        OosObservation(
            "fold-1",
            NOW - timedelta(days=3),
            NOW - timedelta(days=2),
            "low",
            D("0.02"),
            D("0.01"),
            D("0.001"),
            D("0.002"),
            D("1"),
            D("1.2"),
        ),
        OosObservation(
            "fold-1",
            NOW - timedelta(days=3),
            NOW - timedelta(days=1),
            "high",
            D("-0.04"),
            D("-0.02"),
            D("0.003"),
            D("0.002"),
            D("1.5"),
            D("1.1"),
        ),
        OosObservation(
            "fold-2",
            NOW - timedelta(hours=12),
            NOW,
            "low",
            D("0.01"),
            D("0.005"),
            D("0.001"),
            D("0.001"),
            D("0.7"),
            D("0.8"),
        ),
    )
    report = compare_walk_forward_oos(observations, tail_fraction=D("0.2"))
    assert report.fold_count == 2
    assert report.allocator.observation_count == 3
    assert report.allocator.max_drawdown > 0
    assert report.allocator.cvar_tail_loss == D("0.04")
    assert report.allocator.total_cost == D("0.005")
    assert report.allocator.total_turnover == D("3.2")
    assert len(report.regime_mean_returns) == 2
    assert report.economic_decision.startswith("NOT_EVALUATED")


def test_oos_metrics_reject_temporal_leakage() -> None:
    with pytest.raises(ValueError, match="oos_temporal_leakage"):
        OosObservation(
            "fold-1",
            NOW,
            NOW,
            "low",
            D("0.01"),
            D("0.01"),
            D("0"),
            D("0"),
            D("0"),
            D("0"),
        )


def test_contracts_reject_invalid_policy_ranges_and_regime_registry() -> None:
    with pytest.raises(ValueError, match="risk_policy_revision_missing"):
        make_policy(revision=" ")
    with pytest.raises(ValueError, match="absolute_daily_loss_cap"):
        make_policy(absolute_daily_loss_cap=D("0"))
    with pytest.raises(ValueError, match="daily_risk_budget_fraction_out_of_range"):
        make_policy(daily_risk_budget_fraction=D("1.1"))
    with pytest.raises(ValueError, match="per_trade_budget_exceeds_daily_budget"):
        make_policy(
            daily_risk_budget_fraction=D("0.005"), per_trade_risk_fraction=D("0.01")
        )
    with pytest.raises(ValueError, match="confidence_floor_out_of_range"):
        make_policy(confidence_floor=D("1.1"))
    with pytest.raises(ValueError, match="cost_rate_out_of_range"):
        make_policy(fee_rate=D("1"))
    with pytest.raises(ValueError, match="freshness_limits_must_be_positive"):
        make_policy(max_market_age_seconds=0)
    with pytest.raises(ValueError, match="regime_multipliers_missing_or_duplicated"):
        make_policy(regime_multipliers=())
    with pytest.raises(ValueError, match="regime_multipliers_missing_or_duplicated"):
        make_policy(regime_multipliers=(("low", D("1")), ("low", D("0.5"))))
    with pytest.raises(ValueError, match="regime_multiplier_must_not_exceed_one"):
        make_policy(regime_multipliers=(("low", D("1.1")),))


def test_contracts_reject_invalid_snapshots_rules_and_proposals() -> None:
    with pytest.raises(ValueError, match="market_as_of_timezone_required"):
        make_market(as_of=datetime(2026, 10, 22, 10, 0))
    with pytest.raises(ValueError, match="market_price_must_be_finite_decimal"):
        make_market(price=D("NaN"))
    with pytest.raises(ValueError, match="market_identity_invalid"):
        make_market(symbol="btc/usdt")
    with pytest.raises(ValueError, match="market_precision_step_must_be_positive"):
        make_rules(price_tick=D("0"))
    with pytest.raises(ValueError, match="free_balance"):
        make_account(free_balance=D("-1"))
    with pytest.raises(ValueError, match="uncertainty_out_of_range"):
        make_proposal(uncertainty=D("1.1"))
    with pytest.raises(ValueError, match="model_revision_or_decision_id_missing"):
        make_proposal(model_revision="")


def test_allocator_rejects_exhausted_exposure_budget_and_invalid_stop() -> None:
    allocator = DynamicRiskAllocator()
    args = {
        "policy": make_policy(),
        "market": make_market(),
        "rules": make_rules(),
        "proposal": make_proposal(),
        "now": NOW,
    }
    gross = allocator.allocate(
        **{**args, "account": make_account(gross_exposure=D("10000"))}
    )
    assert not gross.approved
    assert "gross_exposure_cap_exhausted" in gross.reasons

    correlated = allocator.allocate(
        **{**args, "account": make_account(correlated_exposure=D("8000"))}
    )
    assert not correlated.approved
    assert "correlated_exposure_cap_exhausted" in correlated.reasons

    bad_stop_market = make_market(price=D("1"), atr=D("0.8"))
    bad_stop = allocator.allocate(
        policy=make_policy(max_atr_fraction=D("1")),
        account=make_account(),
        market=bad_stop_market,
        rules=make_rules(),
        proposal=make_proposal(atr_snapshot=D("0.8")),
        now=NOW,
    )
    assert not bad_stop.approved
    assert "stop_price_invalid" in bad_stop.reasons


def test_short_position_stop_rounding_and_gate_contract() -> None:
    policy = make_policy()
    account = make_account()
    market = make_market()
    rules = make_rules()
    proposal = make_proposal(side=Side.SHORT)
    allocation = make_allocation(
        policy=policy, account=account, market=market, rules=rules, proposal=proposal
    )
    assert allocation.stop_price > market.price
    decision = RiskGate().evaluate_entry(
        allocation=allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, NOW),
        evaluated_at=NOW,
    )
    assert decision.risk_approved
    assert not decision.order_permission


def test_gate_rejects_latched_switch_price_tampering_and_underreported_loss() -> None:
    policy = make_policy()
    account = make_account()
    market = make_market()
    rules = make_rules()
    allocation = make_allocation(
        policy=policy, account=account, market=market, rules=rules
    )
    gate = RiskGate()
    latched = gate.evaluate_entry(
        allocation=allocation,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(True, "test-latch", NOW, 3, NOW),
        evaluated_at=NOW,
    )
    assert not latched.risk_approved
    assert "kill_switch_latched" in latched.reasons

    tampered = replace(allocation, quantity=allocation.quantity + rules.quantity_step)
    result = gate.evaluate_entry(
        allocation=tampered,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, NOW),
        evaluated_at=NOW,
    )
    assert not result.risk_approved
    assert "allocation_price_or_stake_mismatch" in result.reasons

    understated = replace(allocation, estimated_loss_including_costs=D("0.01"))
    result = gate.evaluate_entry(
        allocation=understated,
        account=account,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, NOW),
        evaluated_at=NOW,
    )
    assert not result.risk_approved
    assert "reported_loss_understates_recomputed_loss" in result.reasons


def test_gate_rejects_insufficient_free_balance_and_stale_allocation() -> None:
    policy = make_policy()
    market = make_market()
    rules = make_rules()
    allocation = make_allocation(
        policy=policy, account=make_account(), market=market, rules=rules
    )
    low_balance = make_account(free_balance=D("100"))
    result = RiskGate().evaluate_entry(
        allocation=allocation,
        account=low_balance,
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, NOW),
        evaluated_at=NOW,
    )
    assert not result.risk_approved
    assert "free_balance_limit_exceeded" in result.reasons
    old_allocation = replace(allocation, created_at=NOW - timedelta(seconds=120))
    result = RiskGate().evaluate_entry(
        allocation=old_allocation,
        account=make_account(),
        market=market,
        rules=rules,
        policy=policy,
        kill_switch=KillSwitchState(False, "", None, 2, NOW),
        evaluated_at=NOW,
    )
    assert not result.risk_approved
    assert "allocation_stale_or_future" in result.reasons


def test_kill_switch_idempotent_latch_and_in_memory_test_adapter() -> None:
    from risk.kill_switch import InMemoryKillSwitchStore

    memory = InMemoryKillSwitchStore()
    first = memory.read()
    assert first.latched
    assert memory.latch("repeat", NOW, "test").latched
    state = memory.read()
    controller = KillSwitchController(memory)
    rearmed = controller.manual_rearm(
        ManualRearmEvidence(
            operator_id="owner",
            owner_approval_id="owner-approval-fixture",
            reason="manual in-memory test re-arm",
            observed_version=state.version,
            account_reconciled=True,
            orders_reconciled=True,
            supervisor_healthy=True,
            risk_policy_unchanged=True,
        ),
        NOW,
    )
    assert not rearmed.latched
    state = memory.read()
    with pytest.raises(RearmRejected, match="kill_switch_not_latched"):
        controller.manual_rearm(
            ManualRearmEvidence(
                operator_id="owner",
                owner_approval_id="owner-approval",
                reason="already rearmed",
                observed_version=state.version,
                account_reconciled=True,
                orders_reconciled=True,
                supervisor_healthy=True,
                risk_policy_unchanged=True,
            ),
            NOW,
        )


def test_kill_switch_rejects_bad_state_and_unreadable_persistence(
    tmp_path: Path,
) -> None:
    from risk.kill_switch import KillSwitchStateUnavailable

    with pytest.raises(ValueError, match="durable_kill_switch_database_path_required"):
        SQLiteKillSwitchStore(":memory:")
    with pytest.raises(ValueError, match="latched_state_requires_reason"):
        KillSwitchState(True, "", NOW, 1, NOW)
    with pytest.raises(ValueError, match="unlatched_state_must_not_keep_latched_at"):
        KillSwitchState(False, "", NOW, 1, NOW)
    with pytest.raises(ValueError, match="kill_switch_version_invalid"):
        KillSwitchState(True, "latched", NOW, 0, NOW)

    store = SQLiteKillSwitchStore(str(tmp_path / "read-failure.db"))
    store._database_path = str(tmp_path / "nonexistent-directory" / "missing.db")
    with pytest.raises(
        KillSwitchStateUnavailable, match="kill_switch_state_unreadable"
    ):
        store.read()
    decision = RiskSupervisor(store).check(
        account=make_account(), policy=make_policy(), health=make_health(), now=NOW
    )
    assert decision.status is SupervisorStatus.HALT_UNCONFIRMED
    assert decision.reason == "kill_switch_state_unavailable"


def test_kill_switch_unknown_row_and_manual_rearm_version_compare(
    tmp_path: Path,
) -> None:
    import sqlite3

    from risk.kill_switch import KillSwitchStateUnavailable

    path = tmp_path / "empty-state.db"
    store = SQLiteKillSwitchStore(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM kill_switch_state")
    with pytest.raises(
        KillSwitchStateUnavailable, match="kill_switch_state_row_missing"
    ):
        store.read()

    live_store = SQLiteKillSwitchStore(str(tmp_path / "compare.db"))
    state = live_store.read()
    assert not live_store.rearm(
        state.version + 1, "approval", "operator", "stale CAS", NOW
    )
    assert live_store.rearm(
        state.version, "approval", "operator", "manual test recovery", NOW
    )
    rearmed = live_store.read()
    assert not live_store.rearm(
        rearmed.version, "approval", "operator", "not latched", NOW
    )


def test_supervisor_missing_inputs_degraded_policy_stale_and_bad_time(
    tmp_path: Path,
) -> None:
    store = make_armed_store(tmp_path / "supervisor-cases.db")
    supervisor = RiskSupervisor(store)
    decision = supervisor.check(
        account=None, policy=make_policy(), health=make_health(), now=NOW
    )
    assert decision.status is SupervisorStatus.HALTED
    assert decision.reason == "critical_supervisor_input_missing"

    store = make_armed_store(tmp_path / "unapproved-policy.db")
    decision = RiskSupervisor(store).check(
        account=make_account(),
        policy=make_policy(owner_approved=False),
        health=make_health(),
        now=NOW,
    )
    assert decision.status is SupervisorStatus.HALTED
    assert decision.reason == "risk_policy_not_owner_approved"

    store = make_armed_store(tmp_path / "stale-supervisor.db")
    decision = RiskSupervisor(store).check(
        account=make_account(as_of=NOW - timedelta(seconds=120)),
        policy=make_policy(),
        health=make_health(),
        now=NOW,
    )
    assert decision.status is SupervisorStatus.HALTED
    assert decision.reason == "account_snapshot_stale_or_future"

    decision = RiskSupervisor(store).check(
        account=make_account(),
        policy=make_policy(),
        health=make_health(),
        now=datetime(2026, 10, 22, 10, 0),
    )
    assert decision.status is SupervisorStatus.HALT_UNCONFIRMED
    assert decision.reason == "supervisor_time_invalid"


def test_oos_comparison_rejects_missing_data_bad_tail_and_conflicting_fold() -> None:
    from risk.evaluation import compare_walk_forward_oos

    with pytest.raises(ValueError, match="oos_observations_missing"):
        compare_walk_forward_oos((), tail_fraction=D("0.1"))
    observation = OosObservation(
        "fold",
        NOW - timedelta(days=2),
        NOW - timedelta(days=1),
        "low",
        D("0.01"),
        D("0.02"),
        D("0"),
        D("0"),
        D("0"),
        D("0"),
    )
    with pytest.raises(ValueError, match="tail_fraction_out_of_range"):
        compare_walk_forward_oos((observation,), tail_fraction=D("1.1"))
    other = OosObservation(
        "fold",
        NOW - timedelta(days=3),
        NOW,
        "high",
        D("-0.01"),
        D("-0.02"),
        D("0"),
        D("0"),
        D("0"),
        D("0"),
    )
    with pytest.raises(ValueError, match="fold_train_cutoff_inconsistent"):
        compare_walk_forward_oos((observation, other), tail_fraction=D("0.1"))


def test_oos_contract_rejects_invalid_returns_costs_and_missing_identity() -> None:
    with pytest.raises(ValueError, match="oos_fold_or_regime_missing"):
        OosObservation(
            "",
            NOW - timedelta(days=2),
            NOW - timedelta(days=1),
            "low",
            D("0"),
            D("0"),
            D("0"),
            D("0"),
            D("0"),
            D("0"),
        )
    with pytest.raises(ValueError, match="return_below_total_loss"):
        OosObservation(
            "fold",
            NOW - timedelta(days=2),
            NOW - timedelta(days=1),
            "low",
            D("-1.1"),
            D("0"),
            D("0"),
            D("0"),
            D("0"),
            D("0"),
        )
    with pytest.raises(ValueError, match="allocator_cost_below_minimum"):
        OosObservation(
            "fold",
            NOW - timedelta(days=2),
            NOW - timedelta(days=1),
            "low",
            D("0"),
            D("0"),
            D("-1"),
            D("0"),
            D("0"),
            D("0"),
        )
