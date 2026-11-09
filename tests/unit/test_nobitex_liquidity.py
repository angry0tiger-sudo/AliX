from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, cast

import pytest

from integrations.nobitex.liquidity import (
    DEFAULT_MAX_PARTICIPATION,
    InsufficientMakerDepth,
    LiquidityGateRejected,
    MockOrderBookTransport,
    NobitexMockOrderBookAdapter,
    OrderBookError,
)


NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)


def book_response(
    *,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
    last_update: object = NOW_MS,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "status": status,
        "lastUpdate": last_update,
        "bids": bids if bids is not None else [["999", "5"], ["998", "5"], ["997", "5"]],
        "asks": asks if asks is not None else [["1001", "5"], ["1002", "5"], ["1003", "5"]],
        "lastTradePrice": "1000",
    }


def make_adapter(response: dict[str, Any]) -> tuple[NobitexMockOrderBookAdapter, MockOrderBookTransport]:
    transport = MockOrderBookTransport({"BTCUSDT": cast(dict[str, Any], response)})
    return NobitexMockOrderBookAdapter(transport), transport


def test_mock_transport_uses_documented_get_path_and_never_networks() -> None:
    adapter, transport = make_adapter(book_response())
    snapshot = adapter.fetch_order_book("btc/usdt", now=NOW)
    assert snapshot.symbol == "BTCUSDT"
    assert transport.calls == ["/v3/orderbook/BTCUSDT"]


def test_book_parses_unix_millisecond_timestamp_and_calculates_mid() -> None:
    adapter, _ = make_adapter(book_response(last_update=str(NOW_MS)))
    snapshot = adapter.fetch_order_book("BTC/USDT", now=NOW)
    assert snapshot.last_update == NOW
    assert snapshot.mid_price == Decimal("1000")
    assert snapshot.age_seconds == 0.0


def test_freshness_and_book_structure_fail_closed() -> None:
    stale = NOW_MS - 6_000
    adapter, _ = make_adapter(book_response(last_update=stale))
    with pytest.raises(OrderBookError, match="stale_or_from_future"):
        adapter.fetch_order_book("BTC/USDT", now=NOW)

    future, _ = make_adapter(book_response(last_update=NOW_MS + 1_000))
    with pytest.raises(OrderBookError, match="stale_or_from_future"):
        future.fetch_order_book("BTC/USDT", now=NOW)

    bad_status, _ = make_adapter(book_response(status="failed"))
    with pytest.raises(OrderBookError, match="status_not_ok"):
        bad_status.fetch_order_book("BTC/USDT", now=NOW)


def test_invalid_timestamp_and_missing_or_unsorted_levels_are_rejected() -> None:
    for value in (None, True, "not-a-time", NOW_MS // 1000):
        adapter, _ = make_adapter(book_response(last_update=value))
        with pytest.raises(OrderBookError):
            adapter.fetch_order_book("BTC/USDT", now=NOW)

    missing, _ = make_adapter(book_response(bids=[]))
    with pytest.raises(OrderBookError, match="bids_levels_missing"):
        missing.fetch_order_book("BTC/USDT", now=NOW)

    unsorted, _ = make_adapter(book_response(bids=[["998", "1"], ["999", "2"]]))
    with pytest.raises(OrderBookError, match="descending"):
        unsorted.fetch_order_book("BTC/USDT", now=NOW)


def test_invalid_prices_amounts_and_crossed_book_are_rejected() -> None:
    invalid_amount, _ = make_adapter(book_response(bids=[["999", "0"]]))
    with pytest.raises(OrderBookError, match="positive"):
        invalid_amount.fetch_order_book("BTC/USDT", now=NOW)

    crossed, _ = make_adapter(book_response(bids=[["1001", "1"]], asks=[["1001", "1"]]))
    with pytest.raises(OrderBookError, match="crossed"):
        crossed.fetch_order_book("BTC/USDT", now=NOW)


def test_mock_transport_rejects_unsupported_paths_and_missing_symbols() -> None:
    transport = MockOrderBookTransport({"BTCUSDT": book_response()})
    with pytest.raises(OrderBookError, match="path_not_allowed"):
        transport.get_json("/market/stats")
    with pytest.raises(OrderBookError, match="symbol_invalid"):
        transport.get_json("/v3/orderbook/BTC/USDT")
    with pytest.raises(OrderBookError, match="fixture_missing"):
        transport.get_json("/v3/orderbook/ETHUSDT")


def test_liquidity_gate_checks_buy_asks_and_sell_bids_within_half_percent() -> None:
    adapter, _ = make_adapter(
        book_response(
            bids=[["999", "1"], ["996", "10"], ["994", "20"]],
            asks=[["1001", "1"], ["1004", "10"], ["1006", "20"]],
        )
    )
    snapshot = adapter.fetch_order_book("BTC/USDT", now=NOW)
    buy = adapter.liquidity_gate(
        snapshot, side="buy", intended_quote_notional=Decimal("12000")
    )
    sell = adapter.liquidity_gate(
        snapshot, side="sell", intended_quote_notional=Decimal("12000")
    )
    assert buy.approved is True
    assert sell.approved is False
    assert buy.available_quote_depth == Decimal("11041")
    assert sell.available_quote_depth == Decimal("10959")


def test_liquidity_gate_rejects_insufficient_depth_and_invalid_inputs() -> None:
    adapter, _ = make_adapter(book_response())
    snapshot = adapter.fetch_order_book("BTC/USDT", now=NOW)
    result = adapter.liquidity_gate(
        snapshot, side="buy", intended_quote_notional=Decimal("999999")
    )
    assert result.approved is False
    with pytest.raises(LiquidityGateRejected):
        adapter.plan_maker_slices(
            snapshot, side="buy", intended_quote_notional=Decimal("999999")
        )
    with pytest.raises(ValueError, match="buy_or_sell"):
        adapter.liquidity_gate(
            snapshot, side="hold", intended_quote_notional=Decimal("1")
        )
    with pytest.raises(OrderBookError, match="positive"):
        adapter.liquidity_gate(
            snapshot, side="buy", intended_quote_notional=Decimal("0")
        )
    with pytest.raises(ValueError, match="price_band"):
        adapter.liquidity_gate(
            snapshot,
            side="buy",
            intended_quote_notional=Decimal("1"),
            price_band_fraction=Decimal("1"),
        )


def test_maker_slices_are_passive_bounded_and_explicitly_mock_only() -> None:
    adapter, _ = make_adapter(book_response())
    snapshot = adapter.fetch_order_book("BTC/USDT", now=NOW)
    plan = adapter.plan_maker_slices(
        snapshot,
        side="buy",
        intended_quote_notional=Decimal("500"),
        max_participation=Decimal("0.10"),
    )
    assert plan.execution_permission is False
    assert plan.liquidity.approved is True
    assert sum(item.quote_notional for item in plan.child_slices) == Decimal("500")
    assert all(item.limit_price < snapshot.asks[0].price for item in plan.child_slices)
    assert all(item.post_only_required for item in plan.child_slices)
    assert all(not item.exchange_enforced_post_only_confirmed for item in plan.child_slices)
    assert all(item.execution_mode == "MOCK_ONLY" for item in plan.child_slices)


def test_maker_slicing_rejects_insufficient_passive_depth_and_invalid_participation() -> None:
    adapter, _ = make_adapter(book_response())
    snapshot = adapter.fetch_order_book("BTC/USDT", now=NOW)
    with pytest.raises(InsufficientMakerDepth):
        adapter.plan_maker_slices(
            snapshot,
            side="buy",
            intended_quote_notional=Decimal("5000"),
            max_participation=Decimal("0.01"),
        )
    with pytest.raises(ValueError, match="max_participation"):
        adapter.plan_maker_slices(
            snapshot,
            side="sell",
            intended_quote_notional=Decimal("100"),
            max_participation=Decimal("0"),
        )
    sell = adapter.plan_maker_slices(
        snapshot,
        side="sell",
        intended_quote_notional=Decimal("500"),
        max_participation=DEFAULT_MAX_PARTICIPATION,
    )
    assert all(item.limit_price > snapshot.bids[0].price for item in sell.child_slices)
