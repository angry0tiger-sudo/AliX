"""Durable, network-free OCO simulation backed by Freqtrade custom trade data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from services.execution.contracts import (
    CLIENT_ORDER_PREFIX,
    LocalPosition,
    OrderMapping,
    ProtectionPlan,
    RecoveryStore,
)


class SimulatedOcoGateway:
    """Simulate exchange-protection orders without creating an exchange connection."""

    is_simulated = True

    def __init__(self, store: RecoveryStore) -> None:
        self._store = store

    def fetch_balances(self) -> Mapping[str, Decimal]:
        balances: dict[str, Decimal] = {}
        for position in self._store.open_positions():
            balances[position.base_asset] = (
                balances.get(position.base_asset, Decimal("0")) + position.quantity
            )
        return balances

    def fetch_open_orders(self) -> Sequence[OrderMapping]:
        result: list[OrderMapping] = []
        for position in self._store.open_positions():
            record = self._store.get_protection(position.trade_id)
            if record is None or record.get("status") != "SIMULATED_PROTECTED":
                continue
            result.extend(
                (
                    {
                        "id": f"sim-{position.trade_id}-tp",
                        "clientOrderId": record.get("take_profit_client_order_id", ""),
                        "symbol": position.pair,
                        "side": "sell",
                        "status": "open",
                    },
                    {
                        "id": f"sim-{position.trade_id}-sl",
                        "clientOrderId": record.get("stop_loss_client_order_id", ""),
                        "symbol": position.pair,
                        "side": "sell",
                        "status": "open",
                    },
                )
            )
        return tuple(result)

    def fetch_my_trades(
        self, pair: str, since_ms: int
    ) -> Sequence[OrderMapping]:
        del pair, since_ms
        return ()

    def fetch_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping | None:
        for position in self._store.open_positions():
            if position.pair != pair:
                continue
            record = self._store.get_protection(position.trade_id)
            if (
                record is None
                or record.get("list_client_order_id") != list_client_order_id
                or record.get("status") != "SIMULATED_PROTECTED"
            ):
                continue
            return {
                "orderListId": f"SIM-{position.trade_id}",
                "listClientOrderId": list_client_order_id,
                "listOrderStatus": "EXECUTING",
                "orderReports": [
                    {
                        "clientOrderId": record.get("take_profit_client_order_id", ""),
                        "status": "NEW",
                        "side": "SELL",
                    },
                    {
                        "clientOrderId": record.get("stop_loss_client_order_id", ""),
                        "status": "NEW",
                        "side": "SELL",
                    },
                ],
            }
        return None

    def place_oco(self, plan: ProtectionPlan) -> OrderMapping:
        return {
            "orderListId": f"SIM-{plan.trade_id}",
            "listClientOrderId": plan.list_client_order_id,
            "listOrderStatus": "EXECUTING",
            "orderReports": [
                {
                    "clientOrderId": plan.take_profit_client_order_id,
                    "status": "NEW",
                    "side": "SELL",
                },
                {
                    "clientOrderId": plan.stop_loss_client_order_id,
                    "status": "NEW",
                    "side": "SELL",
                },
            ],
        }

    def cancel_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping:
        return {
            "symbol": pair,
            "listClientOrderId": list_client_order_id,
            "listOrderStatus": "ALL_DONE",
            "simulated": True,
        }
