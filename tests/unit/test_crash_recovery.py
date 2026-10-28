"""Fault-injection tests for restart recovery and OCO reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

import pytest

from services.execution.contracts import (
    LocalPosition,
    OrderMapping,
    ProtectionPlan,
    build_protection_plan,
    record_for_plan,
)
from services.execution.reconciliation import CrashRecoveryCoordinator
from services.execution.simulated_oco import SimulatedOcoGateway


class MemoryStore:
    """Small durable-store substitute shared across simulated process restarts."""

    def __init__(self, positions: Sequence[LocalPosition]) -> None:
        self.positions = {position.trade_id: position for position in positions}
        self.protections: dict[str, dict[str, object]] = {}
        self.closed: dict[str, OrderMapping] = {}

    def open_positions(self) -> Sequence[LocalPosition]:
        return tuple(self.positions.values())

    def get_protection(self, trade_id: str) -> Mapping[str, object] | None:
        return self.protections.get(trade_id)

    def save_protection(
        self, trade_id: str, record: Mapping[str, object]
    ) -> None:
        self.protections[trade_id] = dict(record)

    def mark_externally_closed(
        self,
        trade_id: str,
        order: OrderMapping,
        protection_record: Mapping[str, object],
    ) -> None:
        self.closed[trade_id] = dict(order)
        self.protections[trade_id] = dict(protection_record)
        self.positions.pop(trade_id, None)


class FaultInjectingGateway:
    """Mock exchange with exchange-side persistence and network fault injection."""

    is_simulated = True

    def __init__(self, balance: Decimal = Decimal("2")) -> None:
        self.balance = balance
        self.lists: dict[str, dict[str, object]] = {}
        self.place_calls = 0
        self.fail_after_accept_once = False
        self.fail_snapshot = False
        self.trade_history: list[dict[str, object]] = []

    def fetch_balances(self) -> Mapping[str, Decimal]:
        if self.fail_snapshot:
            raise TimeoutError("simulated_network_disconnect")
        return {"BTC": self.balance}

    def fetch_open_orders(self) -> Sequence[OrderMapping]:
        if self.fail_snapshot:
            raise TimeoutError("simulated_network_disconnect")
        result: list[OrderMapping] = []
        for oco in self.lists.values():
            if str(oco["listOrderStatus"]) != "EXECUTING":
                continue
            reports = oco["orderReports"]
            assert isinstance(reports, list)
            result.extend(
                {
                    "id": report.get("id", ""),
                    "clientOrderId": report.get("clientOrderId", ""),
                    "side": report.get("side", "SELL"),
                    "status": report.get("status", "NEW"),
                }
                for report in reports
                if isinstance(report, Mapping)
                and report.get("status") in {"NEW", "PARTIALLY_FILLED"}
            )
        return tuple(result)

    def fetch_my_trades(
        self, pair: str, since_ms: int
    ) -> Sequence[OrderMapping]:
        del pair, since_ms
        if self.fail_snapshot:
            raise TimeoutError("simulated_network_disconnect")
        return tuple(self.trade_history)

    def fetch_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping | None:
        del pair
        return self.lists.get(list_client_order_id)

    def place_oco(self, plan: ProtectionPlan) -> OrderMapping:
        self.place_calls += 1
        existing = self.lists.get(plan.list_client_order_id)
        if existing is not None:
            return existing
        result: dict[str, object] = {
            "orderListId": f"SIM-LIST-{self.place_calls}",
            "listClientOrderId": plan.list_client_order_id,
            "listOrderStatus": "EXECUTING",
            "orderReports": [
                {
                    "id": f"SIM-TP-{self.place_calls}",
                    "clientOrderId": plan.take_profit_client_order_id,
                    "status": "NEW",
                    "side": "SELL",
                    "type": "LIMIT_MAKER",
                },
                {
                    "id": f"SIM-SL-{self.place_calls}",
                    "clientOrderId": plan.stop_loss_client_order_id,
                    "status": "NEW",
                    "side": "SELL",
                    "type": "STOP_LOSS_LIMIT",
                },
            ],
        }
        self.lists[plan.list_client_order_id] = result
        if self.fail_after_accept_once:
            self.fail_after_accept_once = False
            raise TimeoutError("response_lost_after_exchange_accepted_order")
        return result

    def cancel_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping:
        del pair
        oco = self.lists[list_client_order_id]
        oco["listOrderStatus"] = "ALL_DONE"
        reports = oco["orderReports"]
        assert isinstance(reports, list)
        for report in reports:
            if isinstance(report, dict) and report.get("status") in {"NEW", "PARTIALLY_FILLED"}:
                report["status"] = "CANCELED"
        return oco


class CountingSimulatedGateway(SimulatedOcoGateway):
    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)
        self.place_calls = 0

    def place_oco(self, plan: ProtectionPlan) -> OrderMapping:
        self.place_calls += 1
        return super().place_oco(plan)


def _position(*, pending_exit: bool = False) -> LocalPosition:
    return LocalPosition(
        trade_id="trade-17",
        pair="BTC/USDT",
        base_asset="BTC",
        quantity=Decimal("2"),
        entry_price=Decimal("100"),
        entry_timestamp_ms=1_790_000_000_000,
        entry_order_id="entry-exchange-17",
        pending_exit=pending_exit,
    )


def test_protection_plan_has_stable_client_order_ids_and_sane_prices() -> None:
    position = _position()
    first = build_protection_plan(position)
    second = build_protection_plan(position)
    assert first == second
    assert first.stop_price == Decimal("99.00")
    assert first.take_profit_price == Decimal("101.500")
    assert first.list_client_order_id.startswith("alixr-")
    assert len(first.list_client_order_id) <= 36
    assert len(first.take_profit_client_order_id) <= 36
    assert len(first.stop_loss_client_order_id) <= 36


def test_entry_fill_creates_exchange_side_simulated_oco_only_once() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = CountingSimulatedGateway(store)
    coordinator = CrashRecoveryCoordinator(store, gateway)

    first = coordinator.protect_after_fill(position)
    second = coordinator.protect_after_fill(position)

    assert first.halted is False
    assert first.positions_protected == 1
    assert second.halted is False
    assert second.positions_protected == 1
    assert gateway.place_calls == 1
    record = store.get_protection(position.trade_id)
    assert record is not None
    assert record["status"] == "SIMULATED_PROTECTED"
    orders = gateway.fetch_open_orders()
    assert len(orders) == 2
    assert {str(order["clientOrderId"]) for order in orders} == {
        str(record["take_profit_client_order_id"]),
        str(record["stop_loss_client_order_id"]),
    }


def test_restart_reconciliation_recovers_existing_protection_without_duplicate() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = CountingSimulatedGateway(store)
    first_process = CrashRecoveryCoordinator(store, gateway)
    assert first_process.protect_after_fill(position).halted is False

    restarted_process = CrashRecoveryCoordinator(store, gateway)
    report = restarted_process.reconcile_state()

    assert report.halted is False
    assert report.positions_checked == 1
    assert report.positions_protected == 1
    assert report.positions_closed == 0
    assert gateway.place_calls == 1
    assert "protection_reconciled:trade-17" in report.actions


def test_network_response_loss_reconciles_accepted_oco_instead_of_reposting() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    gateway.fail_after_accept_once = True
    coordinator = CrashRecoveryCoordinator(store, gateway)

    attempted = coordinator.protect_after_fill(position)
    recovered = coordinator.reconcile_state()

    assert attempted.halted is True
    assert recovered.halted is False
    assert recovered.positions_protected == 1
    assert gateway.place_calls == 1
    record = store.get_protection(position.trade_id)
    assert record is not None
    assert record["status"] == "PROTECTED"


def test_missing_exchange_protection_is_recreated_deterministically() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    coordinator = CrashRecoveryCoordinator(store, gateway)
    assert coordinator.protect_after_fill(position).halted is False
    old_plan = store.get_protection(position.trade_id)
    assert old_plan is not None
    old_client_id = old_plan["list_client_order_id"]

    gateway.lists.clear()
    report = coordinator.reconcile_state()

    assert report.halted is False
    assert report.positions_protected == 1
    assert gateway.place_calls == 2
    new_record = store.get_protection(position.trade_id)
    assert new_record is not None
    assert new_record["list_client_order_id"] == old_client_id


def test_exchange_snapshot_failure_halts_without_claiming_reconciliation() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    gateway.fail_snapshot = True

    report = CrashRecoveryCoordinator(store, gateway).reconcile_state()

    assert report.halted is True
    assert report.positions_protected == 0
    assert report.issues == ("exchange_snapshot_failed:TimeoutError",)


def test_balance_mismatch_halts_even_when_protection_orders_exist() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway(balance=Decimal("0"))
    coordinator = CrashRecoveryCoordinator(store, gateway)
    assert coordinator.protect_after_fill(position).halted is False

    report = coordinator.reconcile_state()

    assert report.halted is True
    assert report.positions_protected == 1
    assert "exchange_balance_below_local_position:BTC" in report.issues


def test_filled_exchange_oco_is_reconciled_to_closed_local_position() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    coordinator = CrashRecoveryCoordinator(store, gateway)
    assert coordinator.protect_after_fill(position).halted is False
    record = store.get_protection(position.trade_id)
    assert record is not None
    list_id = str(record["list_client_order_id"])
    oco = gateway.lists[list_id]
    oco["listOrderStatus"] = "ALL_DONE"
    reports = oco["orderReports"]
    assert isinstance(reports, list)
    tp = reports[0]
    assert isinstance(tp, dict)
    tp["id"] = "SIM-TP-1"
    tp["status"] = "FILLED"
    gateway.trade_history.append(
        {
            "order": "SIM-TP-1",
            "side": "sell",
            "amount": "2",
            "price": "101.5",
            "cost": "203",
            "timestamp": position.entry_timestamp_ms + 1000,
        }
    )

    report = coordinator.reconcile_state()

    assert report.halted is False
    assert report.positions_closed == 1
    assert report.positions_protected == 0
    assert store.open_positions() == ()
    assert store.closed[position.trade_id]["status"] == "closed"
    closed_record = store.get_protection(position.trade_id)
    assert closed_record is not None
    assert closed_record["status"] == "EXTERNALLY_CLOSED"


def test_exit_pending_does_not_recreate_orders_or_duplicate_exit() -> None:
    position = _position(pending_exit=True)
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    coordinator = CrashRecoveryCoordinator(store, gateway)
    plan = build_protection_plan(position)
    store.save_protection(
        position.trade_id,
        record_for_plan(plan, status="EXIT_PENDING", simulated=True),
    )

    report = coordinator.reconcile_state()

    assert report.halted is False
    assert gateway.place_calls == 0
    assert report.actions == ("exit_pending:trade-17",)


def test_orphan_owned_order_halts_reconciliation_fail_closed() -> None:
    position = _position()
    store = MemoryStore([position])
    gateway = FaultInjectingGateway()
    gateway.lists["alixr-unknown-list"] = {
        "orderListId": "SIM-ORPHAN",
        "listClientOrderId": "alixr-unknown-list",
        "listOrderStatus": "EXECUTING",
        "orderReports": [
            {
                "id": "orphan-order-1",
                "clientOrderId": "alixr-unknown-tp",
                "status": "NEW",
                "side": "SELL",
            }
        ],
    }

    report = CrashRecoveryCoordinator(store, gateway).reconcile_state()

    assert report.halted is True
    assert any(issue.startswith("orphan_alix_orders:") for issue in report.issues)
