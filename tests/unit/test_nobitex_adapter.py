from __future__ import annotations

from decimal import Decimal
from collections.abc import Mapping
from typing import Any

import pytest

from integrations.nobitex.adapter import (
    LiveExecutionDisabled,
    MarketMappingError,
    NobitexAdapter,
    OrderStatus,
    ReadOnlyTransportUnavailable,
    UnknownOrderStatus,
    map_order_status,
    normalize_spot_market,
    normalize_spot_pair,
)


class FakeReadOnlyTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get_json(
        self, path: str, params: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        self.calls.append((path, dict(params) if params is not None else None))
        return self.response


def market_payload() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "base": "btc",
        "quote": "usdt",
        "marketType": "spot",
        "pricePrecision": 2,
        "amountPrecision": 6,
        "minAmount": "0.0001",
        "maxAmount": "10",
        "supportedOrderTypes": ["limit", "market"],
        "supportedSides": ["buy", "sell"],
    }


def test_nobitex_adapter_does_not_connect_without_injected_read_transport() -> None:
    adapter = NobitexAdapter()

    with pytest.raises(ReadOnlyTransportUnavailable):
        adapter.market_stats("BTC/USDT")


def test_read_only_transport_receives_normalized_market_stats_request() -> None:
    transport = FakeReadOnlyTransport({"stats": {"lastPrice": "100"}})
    result = NobitexAdapter(transport).market_stats("btc/usdt")

    assert result == {"stats": {"lastPrice": "100"}}
    assert transport.calls == [
        ("/market/stats", {"srcCurrency": "btc", "dstCurrency": "usdt"})
    ]


def test_spot_market_metadata_is_normalized() -> None:
    market = normalize_spot_market(market_payload())

    assert market.normalized_symbol == "BTC/USDT"
    assert market.settlement == "USDT"
    assert market.market_type == "spot"
    assert market.min_amount == Decimal("0.0001")
    assert market.supported_order_types == ("limit", "market")
    assert market.supported_sides == ("buy", "sell")


def test_market_mapper_rejects_derivative_or_incomplete_market() -> None:
    derivative = market_payload()
    derivative["marketType"] = "future"
    with pytest.raises(MarketMappingError):
        normalize_spot_market(derivative)

    incomplete = market_payload()
    incomplete.pop("supportedSides")
    with pytest.raises(MarketMappingError):
        normalize_spot_market(incomplete)


def test_pair_contract_is_spot_usdt_only() -> None:
    assert normalize_spot_pair("btc/usdt") == "BTC/USDT"
    with pytest.raises(ValueError, match="BASE/USDT"):
        normalize_spot_pair("BTC/IRT")
    with pytest.raises(ValueError):
        normalize_spot_pair("BTC-USDT")


def test_order_payload_is_validated_but_never_submitted() -> None:
    payload = NobitexAdapter.build_order_payload(
        pair="BTC/USDT",
        side="buy",
        order_type="limit",
        amount="0.01",
        price="100000",
        client_order_id="alix-1",
    )

    assert payload == {
        "type": "buy",
        "execution": "limit",
        "srcCurrency": "btc",
        "dstCurrency": "usdt",
        "amount": "0.01",
        "clientOrderId": "alix-1",
        "tradeType": "spot",
        "price": "100000",
    }
    with pytest.raises(LiveExecutionDisabled):
        NobitexAdapter().submit_order(**payload)


def test_order_payload_rejects_invalid_amount_order_and_client_id() -> None:
    base = {
        "pair": "BTC/USDT",
        "side": "buy",
        "order_type": "limit",
        "amount": "0.1",
        "client_order_id": "valid-id",
        "price": "5",
    }
    for updates in (
        {"amount": "NaN"},
        {"order_type": "stop_limit"},
        {"client_order_id": "x" * 33},
        {"price": None},
    ):
        with pytest.raises(ValueError):
            NobitexAdapter.build_order_payload(**(base | updates))  # type: ignore[arg-type]


def test_known_order_statuses_map_and_unknown_state_blocks() -> None:
    assert (
        map_order_status("ACTIVE", matched_amount="0", amount="1") == OrderStatus.OPEN
    )
    assert (
        map_order_status("ACTIVE", matched_amount="1.0", amount="1")
        == OrderStatus.FILLED
    )
    assert map_order_status("canceled") == OrderStatus.CANCELLED
    assert map_order_status("inactive") == OrderStatus.REJECTED
    with pytest.raises(UnknownOrderStatus):
        map_order_status("new-state")
