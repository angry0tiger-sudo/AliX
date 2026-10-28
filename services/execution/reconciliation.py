"""Idempotent protection placement and fail-closed exchange reconciliation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from services.execution.contracts import (
    CLIENT_ORDER_PREFIX,
    DEFAULT_STOP_LOSS_FRACTION,
    DEFAULT_TAKE_PROFIT_FRACTION,
    ORDER_ID_TOLERANCE,
    ExchangeGateway,
    LocalPosition,
    OrderMapping,
    ProtectionPlan,
    ReconciliationReport,
    RecoveryStore,
    build_protection_plan,
    plan_from_record,
    record_for_plan,
)


@dataclass(frozen=True, slots=True)
class _PositionOutcome:
    protected: bool = False
    closed: bool = False
    action: str | None = None
    issue: str | None = None


def _as_mapping_sequence(value: object) -> tuple[OrderMapping, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ValueError, ArithmeticError):
        return default
    return result if result.is_finite() else default


def _client_id(order: OrderMapping) -> str:
    value = order.get("clientOrderId", order.get("clientOrderID", ""))
    return str(value) if value is not None else ""


def _order_id(order: OrderMapping) -> str:
    value = order.get("id", order.get("orderId", ""))
    return str(value) if value is not None else ""


def _oco_is_active(oco: OrderMapping) -> bool:
    list_status = str(
        oco.get("listOrderStatus", oco.get("listStatusType", ""))
    ).upper()
    if list_status in {"EXECUTING", "EXEC_STARTED"}:
        return True
    reports = _as_mapping_sequence(oco.get("orderReports", oco.get("orders")))
    return any(
        str(report.get("status", "")).upper() in {"NEW", "PARTIALLY_FILLED"}
        for report in reports
    )


def _oco_order_reports(oco: OrderMapping) -> tuple[OrderMapping, ...]:
    return _as_mapping_sequence(oco.get("orderReports", oco.get("orders")))


def _active_owned_order_ids(
    open_orders: Sequence[OrderMapping],
) -> set[str]:
    return {
        _client_id(order)
        for order in open_orders
        if _client_id(order).startswith(CLIENT_ORDER_PREFIX)
    }


def _filled_exit_order(
    position: LocalPosition,
    oco: OrderMapping,
    trades: Sequence[OrderMapping],
) -> tuple[OrderMapping | None, str | None]:
    reports = _oco_order_reports(oco)
    filled_ids = {
        _order_id(report)
        for report in reports
        if str(report.get("status", "")).upper() == "FILLED"
        and str(report.get("side", "SELL")).upper() == "SELL"
        and _order_id(report)
    }
    matching = tuple(
        trade
        for trade in trades
        if str(trade.get("order", trade.get("orderId", ""))) in filled_ids
        and str(trade.get("side", "")).lower() == "sell"
    )
    if not filled_ids:
        return None, None
    if not matching:
        return None, "filled_oco_leg_missing_from_trade_history"
    filled_quantity = sum(
        (_decimal(trade.get("amount", trade.get("qty"))) for trade in matching),
        Decimal("0"),
    )
    total_cost = sum(
        (
            _decimal(
                trade.get("cost"),
                _decimal(trade.get("price"))
                * _decimal(trade.get("amount", trade.get("qty"))),
            )
            for trade in matching
        ),
        Decimal("0"),
    )
    tolerance = max(position.quantity * Decimal("0.001"), ORDER_ID_TOLERANCE)
    if filled_quantity + tolerance < position.quantity:
        return None, "partial_oco_exit_requires_manual_reconciliation"
    if filled_quantity - position.quantity > tolerance or filled_quantity <= 0:
        return None, "oco_exit_quantity_mismatch"
    average = total_cost / filled_quantity
    latest = max(int(trade.get("timestamp") or 0) for trade in matching)
    order_id = str(matching[-1].get("order", matching[-1].get("orderId", "")))
    report = next((item for item in reports if _order_id(item) == order_id), {})
    normalized: dict[str, object] = {
        "id": order_id,
        "symbol": position.pair,
        "type": str(report.get("type", "STOP_LOSS_LIMIT")),
        "side": "sell",
        "status": "closed",
        "amount": float(position.quantity),
        "filled": float(filled_quantity),
        "remaining": 0.0,
        "price": float(average),
        "average": float(average),
        "cost": float(total_cost),
        "timestamp": latest,
        "lastTradeTimestamp": latest,
        "fee": None,
        "info": dict(report),
    }
    return normalized, None


class CrashRecoveryCoordinator:
    """Own the lifecycle of exchange-side OCO protection and local recovery."""

    def __init__(
        self,
        store: RecoveryStore,
        gateway: ExchangeGateway,
        *,
        take_profit_fraction: Decimal = DEFAULT_TAKE_PROFIT_FRACTION,
        stop_loss_fraction: Decimal = DEFAULT_STOP_LOSS_FRACTION,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.take_profit_fraction = take_profit_fraction
        self.stop_loss_fraction = stop_loss_fraction

    def protect_after_fill(
        self,
        position: LocalPosition,
        *,
        stop_loss_fraction: Decimal | None = None,
    ) -> ReconciliationReport:
        """Persist the deterministic plan before attempting the exchange request."""
        try:
            record = self.store.get_protection(position.trade_id)
            if record is None:
                plan = build_protection_plan(
                    position,
                    stop_loss_fraction=stop_loss_fraction or self.stop_loss_fraction,
                    take_profit_fraction=self.take_profit_fraction,
                )
                self.store.save_protection(
                    position.trade_id,
                    record_for_plan(
                        plan,
                        status="PROTECTION_PENDING",
                        simulated=self.gateway.is_simulated,
                    ),
                )
            else:
                plan = plan_from_record(position, record)
            outcome = self._ensure_protected(position, plan)
        except Exception as exc:
            return ReconciliationReport(
                halted=True,
                simulated=self.gateway.is_simulated,
                positions_checked=1,
                positions_protected=0,
                positions_closed=0,
                issues=(f"protect_after_fill_failed:{type(exc).__name__}",),
            )
        issues = (outcome.issue,) if outcome.issue else ()
        actions = (outcome.action,) if outcome.action else ()
        return ReconciliationReport(
            halted=bool(issues),
            simulated=self.gateway.is_simulated,
            positions_checked=1,
            positions_protected=int(outcome.protected),
            positions_closed=int(outcome.closed),
            actions=actions,
            issues=issues,
        )

    def reconcile_state(self) -> ReconciliationReport:
        """Compare local open trades with balances, open orders, OCO lists and fills."""
        try:
            positions = tuple(self.store.open_positions())
            balances = self.gateway.fetch_balances()
            open_orders = tuple(self.gateway.fetch_open_orders())
        except Exception as exc:
            return ReconciliationReport(
                halted=True,
                simulated=self.gateway.is_simulated,
                positions_checked=0,
                positions_protected=0,
                positions_closed=0,
                issues=(f"exchange_snapshot_failed:{type(exc).__name__}",),
            )

        plans: dict[str, ProtectionPlan] = {}
        issues: list[str] = []
        actions: list[str] = []
        for position in positions:
            try:
                plans[position.trade_id] = self._plan_for(position)
            except (ValueError, ArithmeticError):
                issues.append(f"invalid_local_protection_record:{position.trade_id}")

        expected_ids = {
            client_id
            for plan in plans.values()
            for client_id in (
                plan.list_client_order_id,
                plan.take_profit_client_order_id,
                plan.stop_loss_client_order_id,
            )
        }
        orphan_ids = sorted(
            client_id
            for client_id in _active_owned_order_ids(open_orders)
            if client_id not in expected_ids
        )
        if orphan_ids:
            issues.append("orphan_alix_orders:" + ",".join(orphan_ids))

        checked = 0
        protected = 0
        closed = 0
        still_open: list[LocalPosition] = []
        for position in positions:
            checked += 1
            plan = plans.get(position.trade_id)
            if plan is None:
                continue
            try:
                outcome = self._reconcile_position(position, plan, open_orders)
            except Exception as exc:
                outcome = _PositionOutcome(
                    issue=f"position_reconciliation_failed:{position.trade_id}:{type(exc).__name__}"
                )
            if outcome.protected:
                protected += 1
            if outcome.closed:
                closed += 1
            else:
                still_open.append(position)
            if outcome.action:
                actions.append(outcome.action)
            if outcome.issue:
                issues.append(outcome.issue)

        issues.extend(self._balance_issues(still_open, balances))
        return ReconciliationReport(
            halted=bool(issues),
            simulated=self.gateway.is_simulated,
            positions_checked=checked,
            positions_protected=protected,
            positions_closed=closed,
            actions=tuple(actions),
            issues=tuple(issues),
        )

    def cancel_protection(self, trade_id: str) -> bool:
        """Cancel the existing OCO before Freqtrade submits a competing exit."""
        try:
            record = self.store.get_protection(trade_id)
            if record is None:
                return True
            position = next(
                (
                    item
                    for item in self.store.open_positions()
                    if item.trade_id == trade_id
                ),
                None,
            )
            if position is None:
                return True
            plan = plan_from_record(position, record)
            status = str(record.get("status", ""))
            if status == "EXIT_PENDING":
                return True
            oco = self.gateway.fetch_oco(position.pair, plan.list_client_order_id)
            if oco is not None and _oco_is_active(oco):
                self.gateway.cancel_oco(position.pair, plan.list_client_order_id)
            remaining = self.gateway.fetch_open_orders()
            matching = {
                plan.take_profit_client_order_id,
                plan.stop_loss_client_order_id,
            }
            if any(_client_id(order) in matching for order in remaining):
                raise RuntimeError("protective_leg_remains_open_after_cancel")
            self.store.save_protection(
                trade_id,
                record_for_plan(
                    plan,
                    status="EXIT_PENDING",
                    simulated=self.gateway.is_simulated,
                    exchange_order_list_id=str(record.get("exchange_order_list_id") or "") or None,
                ),
            )
            return True
        except Exception:
            return False

    def _plan_for(self, position: LocalPosition) -> ProtectionPlan:
        record = self.store.get_protection(position.trade_id)
        if record is not None:
            return plan_from_record(position, record)
        return build_protection_plan(
            position,
            stop_loss_fraction=self.stop_loss_fraction,
            take_profit_fraction=self.take_profit_fraction,
        )

    def _ensure_protected(
        self, position: LocalPosition, plan: ProtectionPlan
    ) -> _PositionOutcome:
        open_orders = tuple(self.gateway.fetch_open_orders())
        oco = self.gateway.fetch_oco(position.pair, plan.list_client_order_id)
        if oco is not None and _oco_is_active(oco):
            self._save_status(position, plan, "PROTECTED")
            return _PositionOutcome(protected=True, action=f"protection_confirmed:{position.trade_id}")
        matching_ids = {
            plan.take_profit_client_order_id,
            plan.stop_loss_client_order_id,
        }
        if any(_client_id(order) in matching_ids for order in open_orders):
            return _PositionOutcome(issue=f"oco_list_missing_but_leg_open:{position.trade_id}")
        result = self.gateway.place_oco(plan)
        exchange_id = result.get("orderListId")
        status = "SIMULATED_PROTECTED" if self.gateway.is_simulated else "PROTECTED"
        self.store.save_protection(
            position.trade_id,
            record_for_plan(
                plan,
                status=status,
                simulated=self.gateway.is_simulated,
                exchange_order_list_id=str(exchange_id) if exchange_id is not None else None,
            ),
        )
        return _PositionOutcome(protected=True, action=f"protection_placed:{position.trade_id}")

    def _reconcile_position(
        self,
        position: LocalPosition,
        plan: ProtectionPlan,
        open_orders: Sequence[OrderMapping],
    ) -> _PositionOutcome:
        record = self.store.get_protection(position.trade_id)
        if record is not None and str(record.get("status", "")) == "EXIT_PENDING":
            if position.pending_exit:
                return _PositionOutcome(action=f"exit_pending:{position.trade_id}")
            if any(
                _client_id(order)
                in {plan.take_profit_client_order_id, plan.stop_loss_client_order_id}
                for order in open_orders
            ):
                return _PositionOutcome(issue=f"exit_pending_has_open_oco_leg:{position.trade_id}")
            self.store.save_protection(
                position.trade_id,
                record_for_plan(
                    plan, status="PROTECTION_PENDING", simulated=self.gateway.is_simulated
                ),
            )

        oco = self.gateway.fetch_oco(position.pair, plan.list_client_order_id)
        if oco is not None and _oco_is_active(oco):
            self._save_status(position, plan, "PROTECTED")
            return _PositionOutcome(
                protected=True, action=f"protection_reconciled:{position.trade_id}"
            )

        if oco is not None:
            trades = self.gateway.fetch_my_trades(
                position.pair, position.entry_timestamp_ms
            )
            external_order, issue = _filled_exit_order(position, oco, trades)
            if issue:
                return _PositionOutcome(issue=f"{issue}:{position.trade_id}")
            if external_order is not None:
                closed_record = record_for_plan(
                    plan,
                    status="EXTERNALLY_CLOSED",
                    simulated=self.gateway.is_simulated,
                    exchange_order_list_id=str(oco.get("orderListId") or "") or None,
                )
                self.store.mark_externally_closed(
                    position.trade_id, external_order, closed_record
                )
                return _PositionOutcome(
                    closed=True, action=f"external_oco_exit_reconciled:{position.trade_id}"
                )

        matching_ids = {
            plan.take_profit_client_order_id,
            plan.stop_loss_client_order_id,
        }
        if any(_client_id(order) in matching_ids for order in open_orders):
            return _PositionOutcome(
                issue=f"oco_list_missing_but_leg_open:{position.trade_id}"
            )
        if position.pending_exit:
            return _PositionOutcome(action=f"local_exit_pending:{position.trade_id}")
        return self._ensure_protected(position, plan)

    def _save_status(
        self, position: LocalPosition, plan: ProtectionPlan, status: str
    ) -> None:
        record = self.store.get_protection(position.trade_id) or {}
        self.store.save_protection(
            position.trade_id,
            record_for_plan(
                plan,
                status="SIMULATED_PROTECTED" if self.gateway.is_simulated else status,
                simulated=self.gateway.is_simulated,
                exchange_order_list_id=str(record.get("exchange_order_list_id") or "") or None,
            ),
        )

    @staticmethod
    def _balance_issues(
        positions: Sequence[LocalPosition], balances: Mapping[str, Decimal]
    ) -> list[str]:
        required: dict[str, Decimal] = defaultdict(Decimal)
        for position in positions:
            required[position.base_asset] += position.quantity
        issues: list[str] = []
        for asset, quantity in required.items():
            available = balances.get(asset, Decimal("0"))
            tolerance = max(quantity * Decimal("0.001"), ORDER_ID_TOLERANCE)
            if available + tolerance < quantity:
                issues.append(f"exchange_balance_below_local_position:{asset}")
        return issues
