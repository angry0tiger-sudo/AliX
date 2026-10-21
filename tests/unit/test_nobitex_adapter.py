from __future__ import annotations

from decimal import Decimal
from collections.abc import Mapping
from typing import Any

import pytest

from integrations.nobitex.adapter import (
    LiveExecutionDisabled,
    MarketMappingError,
    NobitexAdapter,
    NobitexAdapterError,
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
            NobitexAdapter.build_order_payload(**(base | updates))


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


def test_exchange_market_symbol_resolution_matches_novaryn_spot_contract() -> None:
    transport = FakeReadOnlyTransport(
        {
            "stats": {
                "BTCUSDT": {},
                "ETH-USDT": {},
                "BTCIRT": {},
                "1M_PEPEUSDT": {},
            }
        }
    )

    resolved = NobitexAdapter(transport).get_exchange_market_symbols()

    assert resolved == {
        "1MPEPEUSDT": "1M_PEPEUSDT",
        "BTCUSDT": "BTCUSDT",
        "ETHUSDT": "ETHUSDT",
    }
    assert transport.calls == [("/market/stats", None)]


def test_exchange_market_symbol_resolution_rejects_ambiguous_or_invalid_payload() -> (
    None
):
    ambiguous = FakeReadOnlyTransport({"stats": {"1M_PEPEUSDT": {}, "1MPEPEUSDT": {}}})
    with pytest.raises(MarketMappingError, match="ambiguous"):
        NobitexAdapter(ambiguous).get_exchange_market_symbols()

    invalid_payload = FakeReadOnlyTransport({"status": "ok"})
    with pytest.raises(NobitexAdapterError, match="invalid market stats"):
        NobitexAdapter(invalid_payload).get_exchange_market_symbols()

    no_usdt = FakeReadOnlyTransport({"stats": {"BTCIRT": {}}})
    with pytest.raises(NobitexAdapterError, match="no USDT"):
        NobitexAdapter(no_usdt).get_exchange_market_symbols()


def test_market_mapper_rejects_invalid_precision_limits_and_equal_currencies() -> None:
    cases: list[dict[str, Any]] = [
        {"pricePrecision": -1},
        {"amountPrecision": True},
        {"minAmount": "-0.1"},
        {"maxPrice": "NaN"},
        {"base": "BTC", "quote": "BTC"},
        {"symbol": ""},
    ]
    for changes in cases:
        with pytest.raises(MarketMappingError):
            normalize_spot_market(market_payload() | changes)


def test_market_mapper_accepts_missing_optional_limits_and_precision() -> None:
    payload = market_payload()
    for key in (
        "pricePrecision",
        "amountPrecision",
        "minAmount",
        "maxAmount",
        "minPrice",
        "maxPrice",
    ):
        payload.pop(key, None)
    market = normalize_spot_market(payload)
    assert market.price_precision is None
    assert market.amount_precision is None
    assert market.min_amount is None
    assert market.max_price is None


def test_market_stats_rejects_non_mapping_response() -> None:
    class InvalidResponseTransport:
        def get_json(self, path: str, params: Mapping[str, str] | None = None) -> Any:
            return []

    with pytest.raises(NobitexAdapterError, match="must be an object"):
        NobitexAdapter(InvalidResponseTransport()).market_stats("BTC/USDT")


def test_symbol_normalizers_reject_empty_values_and_pair_rejects_unsafe_chars() -> None:
    from integrations.nobitex.adapter import (
        _compact_market_symbol,
        _exchange_market_symbol,
    )

    with pytest.raises(MarketMappingError, match="empty Nobitex"):
        _exchange_market_symbol(" ")
    with pytest.raises(MarketMappingError, match="empty market symbol"):
        _compact_market_symbol("")
    with pytest.raises(ValueError, match="unsupported characters"):
        normalize_spot_pair("BTC_$ /USDT")


def test_order_payload_rejects_bad_side_and_nonpositive_price() -> None:
    with pytest.raises(ValueError, match="side"):
        NobitexAdapter.build_order_payload(
            pair="BTC/USDT",
            side="hold",
            order_type="market",
            amount="1",
            client_order_id="test",
        )
    with pytest.raises(ValueError, match="positive and finite"):
        NobitexAdapter.build_order_payload(
            pair="BTC/USDT",
            side="buy",
            order_type="limit",
            amount="1",
            client_order_id="test",
            price="0",
        )
