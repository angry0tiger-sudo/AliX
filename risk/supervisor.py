"""Independent supervisor for absolute caps and persistent halt state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from risk.contracts import AccountSnapshot, RiskPolicy
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
        if now < account.as_of or (
            now - account.as_of
        ).total_seconds() > policy.max_account_age_seconds:
            return "account_snapshot_stale_or_future"
        return ""

    def _absolute_cap_fault(
        self, account: AccountSnapshot, policy: RiskPolicy
    ) -> str:
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

    def _halted(
        self, state: KillSwitchState, reason: str
    ) -> SupervisorDecision:
        return SupervisorDecision(
            SupervisorStatus.HALTED, reason, state.version, True
        )
