from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from integrations.nobitex.mvp import (
    BookLevel,
    Candle,
    DonchianAtrStrategy,
    LiquidityRejected,
    LiveExecutionBlocked,
    MvpConfig,
    NobitexPublicOrderBookClient,
    OrderBookError,
    OrderBookSnapshot,
    TokenBucket,
    liquidity_gate,
    parse_order_book,
    plan_maker_slices,
    plan_maker_slices_for_base_quantity,
    run_paper_simulation,
    submit_live_maker_order,
    synthetic_order_book,
)

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def payload() -> dict[str, object]:
    return {
        "status": "ok",
        "lastUpdate": int(NOW.timestamp() * 1000),
        "bids": [["999", "100"], ["998", "100"], ["997", "100"]],
        "asks": [["1001", "100"], ["1002", "100"], ["1003", "100"]],
    }


def test_orderbook_parser_accepts_official_shape_and_checks_freshness() -> None:
    parsed = parse_order_book(payload(), "BTCUSDT", received_at=NOW)
    assert parsed.mid_price == Decimal("1000")
    assert parsed.age_seconds == 0.0
    stale_payload = payload()
    stale_payload["lastUpdate"] = int((NOW - timedelta(seconds=6)).timestamp() * 1000)
    with pytest.raises(OrderBookError, match="stale_or_from_future"):
        parse_order_book(stale_payload, "BTCUSDT", received_at=NOW)


def test_orderbook_parser_rejects_crossed_and_unsorted_books() -> None:
    crossed = payload()
    crossed["asks"] = [["998", "100"], ["1002", "100"]]
    with pytest.raises(OrderBookError, match="crossed"):
        parse_order_book(crossed, "BTCUSDT", received_at=NOW)
    unsorted = payload()
    unsorted["bids"] = [["998", "100"], ["999", "100"]]
    with pytest.raises(OrderBookError, match="descending"):
        parse_order_book(unsorted, "BTCUSDT", received_at=NOW)


def test_liquidity_gate_rejects_depth_below_full_intended_order() -> None:
    book = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update=NOW,
        received_at=NOW,
        bids=(BookLevel(Decimal("999"), Decimal("0.1")),),
        asks=(BookLevel(Decimal("1001"), Decimal("0.1")),),
    )
    result = liquidity_gate(book, side="buy", intended_quote_notional=Decimal("500"))
    assert not result.approved
    assert result.available_quote_depth < result.intended_quote_notional
    with pytest.raises(LiquidityRejected):
        plan_maker_slices(book, side="buy", intended_quote_notional=Decimal("500"))


def test_order_slicing_respects_depth_participation_and_non_crossing_prices() -> None:
    book = synthetic_order_book(1000.0, "BTCUSDT")
    plan = plan_maker_slices(book, side="buy", intended_quote_notional=Decimal("500"))
    assert plan.liquidity.approved
    assert plan.execution_permission is False
    assert plan.child_slices
    assert sum(
        (part.quote_notional for part in plan.child_slices), Decimal("0")
    ) == Decimal("500")
    assert all(part.post_only_required for part in plan.child_slices)
    assert all(
        not part.exchange_enforced_post_only_confirmed for part in plan.child_slices
    )
    assert all(part.limit_price < book.asks[0].price for part in plan.child_slices)
    for part in plan.child_slices:
        matching = next(level for level in book.bids if level.price == part.limit_price)
        assert part.quote_notional <= matching.quote_notional * Decimal("0.10")


def test_sell_slices_are_passive_and_bounded_by_ask_depth() -> None:
    book = synthetic_order_book(1000.0, "BTCUSDT")
    plan = plan_maker_slices(book, side="sell", intended_quote_notional=Decimal("500"))
    assert plan.liquidity.approved
    assert all(part.limit_price > book.bids[0].price for part in plan.child_slices)
    assert sum(
        (part.quote_notional for part in plan.child_slices), Decimal("0")
    ) == Decimal("500")


def test_sell_slices_match_requested_base_quantity() -> None:
    book = synthetic_order_book(1000.0, "BTCUSDT")
    intended = Decimal("1.234567")
    plan = plan_maker_slices_for_base_quantity(
        book, side="sell", intended_base_quantity=intended
    )
    planned = sum((part.base_quantity for part in plan.child_slices), Decimal("0"))
    assert abs(planned - intended) <= Decimal("1e-12")
    assert plan.liquidity.approved


def test_token_bucket_uses_refill_without_bursting_past_capacity() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    bucket = TokenBucket(1, 1.0, clock=clock, sleeper=sleeper)
    bucket.acquire()
    bucket.acquire()
    assert sleeps == [1.0]


def test_config_defaults_fail_closed_and_live_is_blocked() -> None:
    config = MvpConfig.from_environment({})
    assert config.dry_run is True
    assert config.allow_live_trading is False
    with pytest.raises(LiveExecutionBlocked):
        MvpConfig.from_environment(
            {"ALIX_DRY_RUN": "false", "ALIX_ALLOW_LIVE_TRADING": "true"}
        )


def test_public_orderbook_network_is_blocked_by_default() -> None:
    client = NobitexPublicOrderBookClient()
    with pytest.raises(LiveExecutionBlocked, match="explicit_owner_approval"):
        client.fetch_order_book("BTCUSDT", now=NOW)


def test_live_order_submission_is_hard_blocked_without_network_path() -> None:
    book = synthetic_order_book(1000.0, "BTCUSDT")
    plan = plan_maker_slices(book, side="buy", intended_quote_notional=Decimal("100"))
    with pytest.raises(LiveExecutionBlocked, match="postOnly"):
        submit_live_maker_order(plan)


def test_donchian_strategy_uses_prior_closed_bars_and_two_atr_trailing() -> None:
    strategy = DonchianAtrStrategy()
    start = NOW
    for index in range(20):
        candle = Candle(
            start + timedelta(minutes=30 * index),
            100.0,
            100.1,
            99.9,
            100.0,
            10.0,
        )
        assert strategy.on_bar(candle) is None
    breakout = Candle(start + timedelta(minutes=600), 100.0, 103.0, 99.9, 102.0, 10.0)
    decision = strategy.on_bar(breakout)
    assert decision is not None
    assert decision.action == "BUY"
    assert decision.reason == "DONCHIAN_20_CLOSE_BREAKOUT"
    strategy.register_entry_fill(
        decision.reference_price, decision.atr, decision.bar_high
    )
    next_bar = Candle(start + timedelta(minutes=630), 102.0, 105.0, 101.9, 104.5, 10.0)
    assert strategy.on_bar(next_bar) is None
    assert strategy.trailing_stop is not None
    old_stop = strategy.trailing_stop
    pullback = Candle(
        start + timedelta(minutes=660),
        104.0,
        104.2,
        old_stop - 0.1,
        old_stop,
        10.0,
    )
    exit_decision = strategy.on_bar(pullback)
    assert exit_decision is not None
    assert exit_decision.action == "SELL"
    assert exit_decision.reason == "TRAILING_STOP_2_ATR"


def test_24_simulated_hours_produce_an_auditable_paper_only_report() -> None:
    report = run_paper_simulation(MvpConfig(), hours=24, seed=17)
    assert report["simulated_hours"] == 24
    assert report["counts"]["bars_processed"] == 48
    assert report["counts"]["buy_signals"] >= 1
    assert report["execution"]["api_calls"] == 0
    assert report["execution"]["actual_exchange_orders"] == 0
    assert len(report["events"]) == (
        report["counts"]["simulated_buy_fills"]
        + report["counts"]["simulated_sell_fills"]
        + report["counts"]["liquidity_rejections"]
    )
    buy_fill = next(
        event["simulated_fill"]
        for event in report["events"]
        if event["action"] == "BUY"
    )
    sell_event = next(event for event in report["events"] if event["action"] == "SELL")
    assert sell_event["intended_base_quantity"] == buy_fill["base_quantity"]
    assert abs(
        Decimal(sell_event["simulated_fill"]["base_quantity"])
        - Decimal(buy_fill["base_quantity"])
    ) <= Decimal("1e-12")
    assert report["safety"]["order_permission"] is False
    assert report["safety"]["trading_authorization"] == "NO-GO"
    assert report["profitability_claim"] is False
