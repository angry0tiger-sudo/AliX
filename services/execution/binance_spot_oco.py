"""Binance Spot OCO adapter; construction is protected by the explicit live gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, cast

from services.execution.contracts import OrderMapping, ProtectionPlan


def _mapping(value: object) -> OrderMapping:
    return value if isinstance(value, Mapping) else {}


def _method(api: object, *names: str) -> Any:
    for name in names:
        candidate = getattr(api, name, None)
        if callable(candidate):
            return candidate
    raise RuntimeError("binance_oco_api_method_unavailable")


def _symbol(pair: str) -> str:
    return pair.split(":", maxsplit=1)[0].replace("/", "").replace("-", "").upper()


def _precise(api: object, method_name: str, pair: str, value: Decimal) -> Decimal:
    method = getattr(api, method_name, None)
    if callable(method):
        return Decimal(str(method(pair, float(value))))
    return value


class BinanceSpotOcoGateway:
    """Places one exchange-native OCO list and uses stable client IDs on retries."""

    is_simulated = False

    def __init__(self, api: object) -> None:
        self._api = api

    def fetch_balances(self) -> Mapping[str, Decimal]:
        result = _mapping(_method(self._api, "fetch_balance")())
        total = _mapping(result.get("total"))
        free = _mapping(result.get("free"))
        used = _mapping(result.get("used"))
        assets = set(total) | set(free) | set(used)
        return {
            str(asset): Decimal(str(total.get(asset, Decimal("0")) or 0))
            if total.get(asset) is not None
            else Decimal(str(free.get(asset, 0) or 0))
            + Decimal(str(used.get(asset, 0) or 0))
            for asset in assets
        }

    def fetch_open_orders(self) -> Sequence[OrderMapping]:
        result = _method(self._api, "fetch_open_orders")()
        if not isinstance(result, list):
            raise RuntimeError("binance_open_orders_response_invalid")
        return tuple(item for item in result if isinstance(item, Mapping))

    def fetch_my_trades(
        self, pair: str, since_ms: int
    ) -> Sequence[OrderMapping]:
        result = _method(self._api, "fetch_my_trades")(pair, since_ms)
        if not isinstance(result, list):
            raise RuntimeError("binance_trades_response_invalid")
        return tuple(item for item in result if isinstance(item, Mapping))

    def fetch_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping | None:
        method = _method(
            self._api,
            "private_get_order_list",
            "privateGetOrderList",
        )
        try:
            result = method(
                {
                    "symbol": _symbol(pair),
                    "origClientOrderId": list_client_order_id,
                }
            )
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "unknown order" in message or "-2011" in message:
                return None
            raise
        mapped = _mapping(result)
        return mapped or None

    def place_oco(self, plan: ProtectionPlan) -> OrderMapping:
        quantity = _precise(self._api, "amount_to_precision", plan.pair, plan.quantity)
        target = _precise(
            self._api, "price_to_precision", plan.pair, plan.take_profit_price
        )
        stop = _precise(self._api, "price_to_precision", plan.pair, plan.stop_price)
        limit = _precise(
            self._api,
            "price_to_precision",
            plan.pair,
            stop * Decimal("0.995"),
        )
        if not (limit < stop < plan.entry_price < target):
            raise ValueError("binance_oco_prices_invalid_after_precision")
        result = _method(
            self._api,
            "private_post_order_list_oco",
            "privatePostOrderListOco",
        )(
            {
                "symbol": _symbol(plan.pair),
                "side": "SELL",
                "quantity": str(quantity),
                "listClientOrderId": plan.list_client_order_id,
                "aboveType": "LIMIT_MAKER",
                "abovePrice": str(target),
                "aboveClientOrderId": plan.take_profit_client_order_id,
                "belowType": "STOP_LOSS_LIMIT",
                "belowStopPrice": str(stop),
                "belowPrice": str(limit),
                "belowTimeInForce": "GTC",
                "belowClientOrderId": plan.stop_loss_client_order_id,
            }
        )
        mapped = _mapping(result)
        if not mapped:
            raise RuntimeError("binance_oco_placement_response_invalid")
        return mapped

    def cancel_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping:
        result = _method(
            self._api,
            "private_delete_order_list",
            "privateDeleteOrderList",
        )(
            {
                "symbol": _symbol(pair),
                "listClientOrderId": list_client_order_id,
            }
        )
        mapped = _mapping(result)
        if not mapped:
            raise RuntimeError("binance_oco_cancel_response_invalid")
        return mapped
