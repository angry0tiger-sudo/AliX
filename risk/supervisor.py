"""Independent supervisor for absolute caps and persistent halt state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from risk.contracts import (
    AccountSnapshot,
    MarketRules,
    MarketSnapshot,
    RiskAllocation,
    RiskPolicy,
    Side,
    estimated_loss_per_unit,
    floor_to_step,
    require_aware_timestamp,
    stable_fingerprint,
)
from risk.kill_switch import KillSwitchState, KillSwitchStore


class SupervisorStatus(StrEnum):
    ACTIVE = "ACTIVE"
    HALTED = "HALTED"
    HALT_UNCONFIRMED = "HALT_UNCONFIRMED"


@dataclass(frozen=True, slots=True)
class SupervisorHealth:
    risk_service_healthy: bool
    market_price_fresh: bool
    account_reconciled: bool
    orders_reconciled: bool
    ledger_writable: bool
    heartbeat_fresh: bool
    policy_loaded: bool

    @property
    def healthy(self) -> bool:
        return all(
            (
                self.risk_service_healthy,
                self.market_price_fresh,
                self.account_reconciled,
                self.orders_reconciled,
                self.ledger_writable,
                self.heartbeat_fresh,
                self.policy_loaded,
            )
        )


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    status: SupervisorStatus
    reason: str
    state_version: int | None
    latched: bool


class SupervisorProposalAction(StrEnum):
    PASS_TO_RISK_GATE = "PASS_TO_RISK_GATE"
    RESIZE_REQUIRED = "RESIZE_REQUIRED"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class SupervisorProposalDecision:
    action: SupervisorProposalAction
    decision_id: str
    evaluated_at: datetime | None
    state_version: int | None
    reason_codes: tuple[str, ...]
    risk_policy_revision: str | None
    allocation_fingerprint: str | None
    requested_quantity: Decimal | None
    requested_stake: Decimal | None
    maximum_allowed_quantity: Decimal | None
    suggested_quantity: Decimal | None
    suggested_stake: Decimal | None
    order_permission: bool
    fingerprint: str

    def __post_init__(self) -> None:
        if self.evaluated_at is not None:
            require_aware_timestamp(self.evaluated_at, "supervisor_review_time")
        if not self.decision_id.strip() or not self.reason_codes:
            raise ValueError("supervisor_proposal_decision_identity_missing")
        if self.order_permission is not False:
            raise ValueError("supervisor_cannot_grant_order_permission")
        if len(self.fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.fingerprint
        ):
            raise ValueError("supervisor_proposal_fingerprint_invalid")
        if self.action is SupervisorProposalAction.RESIZE_REQUIRED:
            if self.suggested_quantity is None or self.suggested_stake is None:
                raise ValueError("supervisor_resize_target_missing")
        elif self.suggested_quantity is not None or self.suggested_stake is not None:
            raise ValueError("supervisor_non_resize_must_not_propose_new_size")


class RiskSupervisor:
    """Observe; latch on faults; never re-arm or mutate policy automatically."""

    def __init__(self, store: KillSwitchStore) -> None:
        self._store = store

    def check(
        self,
        *,
        account: AccountSnapshot | None,
        policy: RiskPolicy | None,
        health: SupervisorHealth | None,
        now: datetime,
    ) -> SupervisorDecision:
        if now.tzinfo is None or now.utcoffset() is None:
            return SupervisorDecision(
                SupervisorStatus.HALT_UNCONFIRMED,
                "supervisor_time_invalid",
                None,
                True,
            )
        try:
            state = self._store.read()
        except Exception:
            return SupervisorDecision(
                SupervisorStatus.HALT_UNCONFIRMED,
                "kill_switch_state_unavailable",
                None,
                True,
            )
        if state.latched:
            return self._halted(state, "kill_switch_latched")
        reason = self._fault_reason(account, policy, health, now)
        if not reason:
            return SupervisorDecision(
                SupervisorStatus.ACTIVE,
                "all_supervisor_checks_passed",
                state.version,
                False,
            )
        try:
            latched_state = self._store.latch(reason, now, "risk_supervisor")
        except Exception:
            return SupervisorDecision(
                SupervisorStatus.HALT_UNCONFIRMED,
                "halt_requested_but_persistence_unconfirmed",
                state.version,
                True,
            )
        return self._halted(latched_state, reason)

    def review_entry_proposal(
        self,
        *,
        allocation: RiskAllocation | None,
        account: AccountSnapshot | None,
        market: MarketSnapshot | None,
        rules: MarketRules | None,
        policy: RiskPolicy | None,
        health: SupervisorHealth | None,
        now: datetime,
    ) -> SupervisorProposalDecision:
        decision_id = allocation.decision_id if allocation is not None else "unknown"
        if now.tzinfo is None or now.utcoffset() is None:
            return _build_proposal_decision(
                SupervisorProposalAction.REJECT,
                decision_id,
                None,
                None,
                ("supervisor_time_invalid",),
                allocation,
                policy,
            )
        if (
            allocation is None
            or account is None
            or market is None
            or rules is None
            or policy is None
            or health is None
        ):
            return _build_proposal_decision(
                SupervisorProposalAction.REJECT,
                decision_id,
                now,
                None,
                ("supervisor_proposal_input_missing",),
                allocation,
                policy,
            )
        supervisor_state = self.check(
            account=account, policy=policy, health=health, now=now
        )
        if supervisor_state.status is not SupervisorStatus.ACTIVE:
            return _build_proposal_decision(
                SupervisorProposalAction.REJECT,
                decision_id,
                now,
                supervisor_state.state_version,
                (supervisor_state.reason,),
                allocation,
                policy,
            )
        return _evaluate_proposal(
            allocation=allocation,
            account=account,
            market=market,
            rules=rules,
            policy=policy,
            now=now,
            state_version=supervisor_state.state_version,
        )

    def _fault_reason(
        self,
        account: AccountSnapshot | None,
        policy: RiskPolicy | None,
        health: SupervisorHealth | None,
        now: datetime,
    ) -> str:
        if account is None or policy is None or health is None:
            return "critical_supervisor_input_missing"
        if not health.healthy:
            return "supervisor_health_degraded"
        if not policy.owner_approved:
            return "risk_policy_not_owner_approved"
        account_fault = self._account_fault(account, policy, now)
        if account_fault:
            return account_fault
        return self._absolute_cap_fault(account, policy)

    def _account_fault(
        self, account: AccountSnapshot, policy: RiskPolicy, now: datetime
    ) -> str:
        if not account.reconciled or not account.orders_reconciled:
            return "account_or_orders_not_reconciled"
        if (
            now < account.as_of
            or (now - account.as_of).total_seconds() > policy.max_account_age_seconds
        ):
            return "account_snapshot_stale_or_future"
        return ""

    def _absolute_cap_fault(self, account: AccountSnapshot, policy: RiskPolicy) -> str:
        if (
            account.daily_loss + account.open_position_risk + account.open_order_risk
            >= policy.absolute_daily_loss_cap
        ):
            return "absolute_daily_loss_cap_hit"
        if (
            account.gross_exposure + account.open_order_notional
            >= policy.absolute_gross_exposure_cap
        ):
            return "absolute_gross_exposure_cap_hit"
        return ""

    def _halted(self, state: KillSwitchState, reason: str) -> SupervisorDecision:
        return SupervisorDecision(SupervisorStatus.HALTED, reason, state.version, True)


def _evaluate_proposal(
    *,
    allocation: RiskAllocation,
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
    policy: RiskPolicy,
    now: datetime,
    state_version: int | None,
) -> SupervisorProposalDecision:
    decision_id = allocation.decision_id
    identity_reasons = _proposal_identity_reasons(
        allocation, account, market, rules, policy, now
    )
    if identity_reasons:
        return _build_proposal_decision(
            SupervisorProposalAction.REJECT,
            decision_id,
            now,
            state_version,
            identity_reasons,
            allocation,
            policy,
        )
    target_quantity, target_stake, target_reason = _initial_stake_target(
        account, market, rules, policy
    )
    if target_reason is not None or target_quantity is None or target_stake is None:
        return _build_proposal_decision(
            SupervisorProposalAction.REJECT,
            decision_id,
            now,
            state_version,
            (target_reason or "initial_stake_target_invalid",),
            allocation,
            policy,
        )
    maximum_quantity = _maximum_entry_quantity(
        allocation, account, market, rules, policy
    )
    if maximum_quantity < target_quantity:
        return _build_proposal_decision(
            SupervisorProposalAction.REJECT,
            decision_id,
            now,
            state_version,
            ("supervisor_cap_conflicts_with_initial_stake",),
            allocation,
            policy,
            maximum_allowed_quantity=maximum_quantity,
        )
    if allocation.stake != allocation.quantity * market.price:
        return _build_proposal_decision(
            SupervisorProposalAction.REJECT,
            decision_id,
            now,
            state_version,
            ("supervisor_allocation_stake_quantity_mismatch",),
            allocation,
            policy,
            maximum_allowed_quantity=maximum_quantity,
        )
    if allocation.quantity < target_quantity:
        return _build_proposal_decision(
            SupervisorProposalAction.REJECT,
            decision_id,
            now,
            state_version,
            ("supervisor_proposal_below_initial_stake_target",),
            allocation,
            policy,
            maximum_allowed_quantity=maximum_quantity,
        )
    if allocation.quantity > target_quantity:
        return _build_proposal_decision(
            SupervisorProposalAction.RESIZE_REQUIRED,
            decision_id,
            now,
            state_version,
            ("supervisor_requires_owner_target_size",),
            allocation,
            policy,
            maximum_allowed_quantity=maximum_quantity,
            suggested_quantity=target_quantity,
            suggested_stake=target_stake,
        )
    return _build_proposal_decision(
        SupervisorProposalAction.PASS_TO_RISK_GATE,
        decision_id,
        now,
        state_version,
        ("supervisor_review_passed_risk_gate_still_required",),
        allocation,
        policy,
        maximum_allowed_quantity=maximum_quantity,
    )


def _proposal_identity_reasons(
    allocation: RiskAllocation,
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
    policy: RiskPolicy,
    now: datetime,
) -> tuple[str, ...]:
    return (
        _proposal_policy_reasons(allocation, rules, policy)
        + _proposal_source_reasons(allocation, account, market, rules)
        + _proposal_freshness_reasons(allocation, market, policy, now)
        + _proposal_scope_reasons(allocation, market, account)
    )


def _proposal_policy_reasons(
    allocation: RiskAllocation, rules: MarketRules, policy: RiskPolicy
) -> tuple[str, ...]:
    reasons: list[str] = []
    stake_policy = policy.initial_stake_policy
    if not rules.approved:
        reasons.append("market_rules_unapproved")
    if not policy.owner_approved:
        reasons.append("risk_policy_not_owner_approved")
    if allocation.policy_revision != policy.revision:
        reasons.append("policy_revision_mismatch")
    if allocation.initial_stake_policy_revision != stake_policy.revision:
        reasons.append("initial_stake_policy_revision_mismatch")
    if allocation.initial_stake_policy_fingerprint != stake_policy.fingerprint:
        reasons.append("initial_stake_policy_fingerprint_mismatch")
    return tuple(reasons)


def _proposal_source_reasons(
    allocation: RiskAllocation,
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if allocation.account_snapshot_hash != account.source_snapshot_hash:
        reasons.append("account_snapshot_hash_mismatch")
    if allocation.market_revision != market.source_revision:
        reasons.append("market_revision_mismatch")
    if allocation.rules_revision != rules.revision:
        reasons.append("market_rules_revision_mismatch")
    return tuple(reasons)


def _proposal_freshness_reasons(
    allocation: RiskAllocation,
    market: MarketSnapshot,
    policy: RiskPolicy,
    now: datetime,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        market.as_of > now
        or (now - market.as_of).total_seconds() > policy.max_market_age_seconds
    ):
        reasons.append("market_price_stale_or_future")
    if (
        allocation.created_at > now
        or (now - allocation.created_at).total_seconds() > policy.max_market_age_seconds
    ):
        reasons.append("allocation_stale_or_future")
    return tuple(reasons)


def _proposal_scope_reasons(
    allocation: RiskAllocation, market: MarketSnapshot, account: AccountSnapshot
) -> tuple[str, ...]:
    reasons: list[str] = []
    if allocation.side is not Side.LONG:
        reasons.append("short_entry_forbidden_in_v1")
    if allocation.symbol != market.symbol or market.symbol.count("/") != 1:
        reasons.append("market_symbol_mismatch")
    if account.currency != "USDT" or not market.symbol.endswith("/USDT"):
        reasons.append("initial_stake_usdt_market_required")
    return tuple(reasons)


def _initial_stake_target(
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
    policy: RiskPolicy,
) -> tuple[Decimal | None, Decimal | None, str | None]:
    stake_policy = policy.initial_stake_policy
    if not stake_policy.owner_approved:
        return None, None, "initial_stake_policy_not_owner_approved"
    if (
        account.currency != "USDT"
        or market.symbol.count("/") != 1
        or not market.symbol.endswith("/USDT")
    ):
        return None, None, "initial_stake_usdt_market_required"
    below_target = account.free_balance < stake_policy.target_usdt
    if below_target and not stake_policy.use_all_free_balance_below_target:
        return None, None, "initial_stake_below_target_policy_disabled"
    reserve = Decimal("1") + policy.fee_rate + policy.slippage_rate
    target_notional = (
        account.free_balance / reserve if below_target else stake_policy.target_usdt
    )
    quantity = floor_to_step(target_notional / market.price, rules.quantity_step)
    stake = quantity * market.price
    reason = _initial_stake_market_reason(
        quantity, stake, below_target, account, market, rules, policy
    )
    if reason is not None:
        return None, None, reason
    return quantity, stake, None


def _initial_stake_market_reason(
    quantity: Decimal,
    stake: Decimal,
    below_target: bool,
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
    policy: RiskPolicy,
) -> str | None:
    target = policy.initial_stake_policy.target_usdt
    if not below_target and stake < target:
        return "initial_stake_target_not_representable_at_market_precision"
    if quantity < rules.min_quantity:
        return "quantity_below_market_minimum"
    if quantity > rules.max_quantity or stake > rules.max_notional:
        return "initial_stake_target_exceeds_market_maximum"
    if stake < rules.min_notional:
        return "notional_below_market_minimum"
    reserve = Decimal("1") + policy.fee_rate + policy.slippage_rate
    if stake * reserve > account.free_balance:
        return "initial_stake_fee_reserve_insufficient"
    return None


def _maximum_entry_quantity(
    allocation: RiskAllocation,
    account: AccountSnapshot,
    market: MarketSnapshot,
    rules: MarketRules,
    policy: RiskPolicy,
) -> Decimal:
    unit_loss = estimated_loss_per_unit(
        market.price, allocation.stop_price, policy.fee_rate, policy.slippage_rate
    )
    if unit_loss <= 0:
        return Decimal("0")
    reserved_loss = (
        account.daily_loss + account.open_position_risk + account.open_order_risk
    )
    daily_limit = min(
        account.equity * policy.daily_risk_budget_fraction,
        policy.absolute_daily_loss_cap,
    )
    risk_room = max(
        Decimal("0"),
        min(
            account.equity * policy.per_trade_risk_fraction,
            daily_limit - reserved_loss,
            policy.absolute_daily_loss_cap - reserved_loss,
        ),
    )
    gross_limit = min(
        policy.absolute_gross_exposure_cap,
        account.equity * policy.max_gross_exposure_fraction,
    )
    gross_room = max(
        Decimal("0"),
        gross_limit - account.gross_exposure - account.open_order_notional,
    )
    correlated_room = max(
        Decimal("0"),
        account.equity * policy.max_correlated_exposure_fraction
        - account.correlated_exposure,
    )
    reserve = Decimal("1") + policy.fee_rate + policy.slippage_rate
    notional_caps = (
        risk_room / unit_loss * market.price,
        account.free_balance / reserve,
        account.free_balance * policy.max_stake_fraction_of_free_balance,
        gross_room,
        correlated_room,
        rules.max_notional,
    )
    quantities = [
        floor_to_step(cap / market.price, rules.quantity_step) for cap in notional_caps
    ]
    quantities.append(rules.max_quantity)
    return min(quantities)


def _build_proposal_decision(
    action: SupervisorProposalAction,
    decision_id: str,
    now: datetime | None,
    state_version: int | None,
    reason_codes: tuple[str, ...],
    allocation: RiskAllocation | None,
    policy: RiskPolicy | None,
    *,
    maximum_allowed_quantity: Decimal | None = None,
    suggested_quantity: Decimal | None = None,
    suggested_stake: Decimal | None = None,
) -> SupervisorProposalDecision:
    payload = {
        "action": action.value,
        "decision_id": decision_id,
        "evaluated_at": now,
        "state_version": state_version,
        "reason_codes": reason_codes,
        "risk_policy_revision": policy.revision if policy else None,
        "initial_stake_policy_fingerprint": (
            policy.initial_stake_policy.fingerprint if policy else None
        ),
        "allocation_fingerprint": allocation.fingerprint if allocation else None,
        "requested_quantity": allocation.quantity if allocation else None,
        "requested_stake": allocation.stake if allocation else None,
        "maximum_allowed_quantity": maximum_allowed_quantity,
        "suggested_quantity": suggested_quantity,
        "suggested_stake": suggested_stake,
        "order_permission": False,
    }
    return SupervisorProposalDecision(
        action=action,
        decision_id=decision_id,
        evaluated_at=now,
        state_version=state_version,
        reason_codes=reason_codes,
        risk_policy_revision=policy.revision if policy else None,
        allocation_fingerprint=allocation.fingerprint if allocation else None,
        requested_quantity=allocation.quantity if allocation else None,
        requested_stake=allocation.stake if allocation else None,
        maximum_allowed_quantity=maximum_allowed_quantity,
        suggested_quantity=suggested_quantity,
        suggested_stake=suggested_stake,
        order_permission=False,
        fingerprint=stable_fingerprint(payload),
    )
