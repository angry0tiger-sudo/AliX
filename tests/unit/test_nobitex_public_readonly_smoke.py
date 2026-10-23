"""Synthetic tests for safe parsing of read-only public Nobitex responses."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.run_nobitex_public_readonly_smoke import build_report

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def make_stats(**market_changes: object) -> dict[str, object]:
    market: dict[str, object] = {
        "isClosed": False,
        "bestBuy": "99900",
        "bestSell": "100100",
        "latest": "100000",
    }
    market.update(market_changes)
    return {"status": "ok", "stats": {"btc-usdt": market}}


def make_orderbook(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ok",
        "lastUpdate": 1792742400000,
        "lastTradePrice": "100000",
        "bids": [["99900", "0.5"], ["99800", "0.4"]],
        "asks": [["100100", "0.5"], ["100200", "0.4"]],
    }
    payload.update(changes)
    return payload


def test_public_smoke_reports_read_only_market_facts_and_no_execution_permission() -> (
    None
):
    report = build_report(
        stats_payload=make_stats(),
        orderbook_payload=make_orderbook(),
        observed_at=NOW,
    )

    assert report["mode"] == "PUBLIC_READ_ONLY_MARKET_SMOKE"
    assert report["pair"] == "BTC/USDT"
    assert report["credentials_used"] is False
    assert report["http_methods_used"] == ["GET"]
    assert report["orders_submitted"] == 0
    assert report["order_permission"] is False
    assert report["Champion"] is None
    assert report["trading_authorization"] == "NO-GO"
    limits = report["limits"]
    assert limits["account_fee_schedule"] == "NOT_VERIFIED_WITHOUT_ACCOUNT_READ_AUTH"
    assert limits["individual_order_status"] == "NOT_QUERIED_WITHOUT_READ_TOKEN"
    assert report["public_orderbook"]["observed_spread_bps"] == "20"


@pytest.mark.parametrize(
    ("stats", "orderbook", "reason"),
    (
        ({"status": "failed"}, make_orderbook(), "public_smoke_stats_status_not_ok"),
        (make_stats(latest="0"), make_orderbook(), "public_smoke_latest_price_invalid"),
        (make_stats(), {"status": "failed"}, "public_smoke_orderbook_status_not_ok"),
        (
            make_stats(),
            make_orderbook(bids=[["0", "0.5"]]),
            "public_smoke_bids_price_invalid",
        ),
        (make_stats(), make_orderbook(asks=[]), "public_smoke_asks_missing"),
        (
            make_stats(),
            make_orderbook(bids=[["100200", "1"]], asks=[["100100", "1"]]),
            "public_smoke_orderbook_crossed",
        ),
    ),
)
def test_bad_public_market_payloads_fail_closed(
    stats: dict[str, object],
    orderbook: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        build_report(stats_payload=stats, orderbook_payload=orderbook, observed_at=NOW)


def test_public_smoke_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="public_smoke_timestamp_timezone_required"):
        build_report(
            stats_payload=make_stats(),
            orderbook_payload=make_orderbook(),
            observed_at=datetime(2026, 10, 23),
        )
