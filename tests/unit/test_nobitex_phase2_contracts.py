from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from integrations.nobitex.adapter import (
    MarketMappingError,
    OrderStatus,
    SpotMarketMetadata,
    UnknownOrderStatus,
    map_order_status,
)
from integrations.nobitex.contracts import (
    PaginationRequest,
    ReadRetryPolicy,
    normalize_spot_balance,
    normalize_spot_order,
    validate_order_against_market,
)


def spot_market() -> SpotMarketMetadata:
    return SpotMarketMetadata(
        exchange_symbol="BTCUSDT",
        normalized_symbol="BTC/USDT",
        base="BTC",
        quote="USDT",
        settlement="USDT",
        market_type="spot",
        price_precision=2,
        amount_precision=4,
        min_amount=Decimal("0.001"),
        max_amount=Decimal("10"),
        min_price=Decimal("1"),
        max_price=Decimal("1000000"),
        supported_order_types=("limit", "market"),
        supported_sides=("buy", "sell"),
    )


def test_balance_snapshot_normalizes_currency_and_available_amount() -> None:
    snapshot = normalize_spot_balance(
        {"wallets": {"Tether": {"balance": "100.50", "blocked": "12.25"}}},
        "usdt",
    )

    assert snapshot.currency == "USDT"
    assert snapshot.total == Decimal("100.50")
    assert snapshot.blocked == Decimal("12.25")
    assert snapshot.available == Decimal("88.25")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"wallets": {"USDT": []}},
        {"wallets": {"USDT": {"balance": "NaN", "blocked": "0"}}},
        {"wallets": {"USDT": {"balance": "1", "blocked": "-1"}}},
        {"wallets": {"USDT": {"balance": "1", "blocked": "2"}}},
    ],
)
def test_balance_snapshot_rejects_invalid_wallet_data(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(MarketMappingError):
        normalize_spot_balance(payload, "USDT")


def order_payload(**changes: Any) -> dict[str, Any]:
    order: dict[str, Any] = {
        "id": 173546223,
        "type": "sell",
        "execution": "Limit",
        "status": "Active",
        "srcCurrency": "Bitcoin",
        "dstCurrency": "Tether",
        "price": "9750.01",
        "amount": "1.2500",
        "matchedAmount": "0.25",
        "averagePrice": "9749.50",
        "totalPrice": "2437.375",
        "fee": "0.05",
        "feeCurrency": "USDT",
        "clientOrderId": "alix-phase2-1",
    }
    order.update(changes)
    return {"order": order}


def test_order_snapshot_classifies_partial_fill_and_preserves_fee_fields() -> None:
    snapshot = normalize_spot_order(order_payload())

    assert snapshot.exchange_order_id == "173546223"
    assert snapshot.status == OrderStatus.PARTIALLY_FILLED
    assert snapshot.pair == "BTC/USDT"
    assert snapshot.amount == Decimal("1.2500")
    assert snapshot.matched_amount == Decimal("0.25")
    assert snapshot.price == Decimal("9750.01")
    assert snapshot.average_price == Decimal("9749.50")
    assert snapshot.total_price == Decimal("2437.375")
    assert snapshot.fee_cost == Decimal("0.05")
    assert snapshot.fee_currency == "USDT"
    assert snapshot.client_order_id == "alix-phase2-1"


@pytest.mark.parametrize(
    ("status", "matched", "expected"),
    [
        ("DONE", "1", OrderStatus.FILLED),
        ("CANCELED", "0", OrderStatus.CANCELLED),
        ("INACTIVE", "0", OrderStatus.REJECTED),
        ("ACTIVE", "0", OrderStatus.OPEN),
        ("NEW", "0", OrderStatus.SUBMITTED),
    ],
)
def test_order_snapshot_maps_known_states(
    status: str, matched: str, expected: str
) -> None:
    snapshot = normalize_spot_order(
        order_payload(status=status, amount="1", matchedAmount=matched)
    )
    assert snapshot.status == expected


def test_order_snapshot_fails_closed_on_unknown_state_and_bad_amounts() -> None:
    with pytest.raises(UnknownOrderStatus):
        map_order_status("unrecognized-state", matched_amount="0", amount="1")
    with pytest.raises(MarketMappingError):
        normalize_spot_order(order_payload(matchedAmount="2", amount="1"))
    with pytest.raises(UnknownOrderStatus):
        map_order_status("ACTIVE", matched_amount="NaN", amount="1")
    with pytest.raises(MarketMappingError):
        normalize_spot_order(order_payload(clientOrderId="x" * 33))


def test_order_snapshot_rejects_non_spot_quote_and_ambiguous_identity() -> None:
    with pytest.raises(ValueError):
        normalize_spot_order(order_payload(srcCurrency="Bitcoin", dstCurrency="IRT"))
    with pytest.raises(MarketMappingError):
        normalize_spot_order(order_payload(id=None, market=None))


def test_order_validation_accepts_values_within_market_rules() -> None:
    validate_order_against_market(
        spot_market(),
        pair="btc/usdt",
        side="BUY",
        order_type="limit",
        amount="0.0123",
        price="123.45",
    )


@pytest.mark.parametrize(
    ("pair", "side", "order_type", "amount", "price"),
    [
        ("BTC/IRT", "buy", "limit", "1", "100"),
        ("BTC/USDT", "hold", "limit", "1", "100"),
        ("BTC/USDT", "buy", "unsupported", "1", "100"),
        ("BTC/USDT", "buy", "limit", "0.0001", "100"),
        ("BTC/USDT", "buy", "limit", "1.00001", "100"),
        ("BTC/USDT", "buy", "limit", "1", "100.001"),
        ("BTC/USDT", "buy", "limit", "11", "100"),
        ("BTC/USDT", "buy", "limit", "1", None),
    ],
)
def test_order_validation_rejects_market_rule_violations(
    pair: str, side: str, order_type: str, amount: str, price: str | None
) -> None:
    with pytest.raises(ValueError):
        validate_order_against_market(
            spot_market(),
            pair=pair,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
        )


def test_read_retry_policy_is_bounded_and_scoped_to_transient_read_errors() -> None:
    policy = ReadRetryPolicy(max_retries=2)

    assert policy.allows_retry(408, 0)
    assert policy.allows_retry(429, 1)
    assert policy.allows_retry(503, 0)
    assert not policy.allows_retry(401, 0)
    assert not policy.allows_retry(400, 0)
    assert not policy.allows_retry(200, 0)
    assert not policy.allows_retry(503, 2)


def test_pagination_request_validates_bounds() -> None:
    assert PaginationRequest() == PaginationRequest(page=1, page_size=100)
    with pytest.raises(ValueError):
        PaginationRequest(page=0)
    with pytest.raises(ValueError):
        PaginationRequest(page_size=101)
