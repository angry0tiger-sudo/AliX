"""Manual, read-only public Nobitex market smoke; never sends orders or uses secrets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http.client import HTTPSConnection
import json
from pathlib import Path
import ssl
from typing import Any

_HOST = "apiv2.nobitex.ir"
_STATS_PATH = "/market/stats?srcCurrency=btc&dstCurrency=usdt"
_ORDERBOOK_PATH = "/v3/orderbook/BTCUSDT"
_ALLOWED_PATHS = (_STATS_PATH, _ORDERBOOK_PATH)
_USER_AGENT = "AliX-Nobitex-ReadOnly-Smoke/1"
_MAX_RESPONSE_BYTES = 1_000_000


def _fetch_json(path: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    if path not in _ALLOWED_PATHS:
        raise ValueError("public_smoke_path_not_allowlisted")
    connection = HTTPSConnection(
        _HOST,
        timeout=timeout_seconds,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("public_smoke_http_status_invalid")
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ValueError("public_smoke_response_too_large")
        payload = json.loads(body.decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(payload, dict):
        raise ValueError("public_smoke_payload_not_object")
    return payload


def _decimal(value: object, reason: str) -> Decimal:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(reason)
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(reason) from error
    if not number.is_finite() or number <= 0:
        raise ValueError(reason)
    return number


def _levels(value: object, name: str) -> tuple[tuple[Decimal, Decimal], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"public_smoke_{name}_missing")
    result: list[tuple[Decimal, Decimal]] = []
    for row in value[:20]:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"public_smoke_{name}_level_invalid")
        price = _decimal(row[0], f"public_smoke_{name}_price_invalid")
        amount = _decimal(row[1], f"public_smoke_{name}_amount_invalid")
        result.append((price, amount))
    return tuple(result)


def _extract_market_stats(
    payload: dict[str, Any],
) -> tuple[dict[str, object], Decimal, Decimal, Decimal]:
    if payload.get("status") != "ok":
        raise ValueError("public_smoke_stats_status_not_ok")
    raw_stats = payload.get("stats")
    market = raw_stats.get("btc-usdt") if isinstance(raw_stats, dict) else None
    if not isinstance(market, dict):
        raise ValueError("public_smoke_btc_usdt_market_missing")
    latest = _decimal(market.get("latest"), "public_smoke_latest_price_invalid")
    best_buy = _decimal(market.get("bestBuy"), "public_smoke_best_buy_invalid")
    best_sell = _decimal(market.get("bestSell"), "public_smoke_best_sell_invalid")
    if best_buy > best_sell:
        raise ValueError("public_smoke_market_stats_crossed")
    report = {
        "status": payload["status"],
        "is_closed": market.get("isClosed"),
        "latest": str(latest),
        "best_buy": str(best_buy),
        "best_sell": str(best_sell),
    }
    return report, latest, best_buy, best_sell


def _extract_orderbook(
    payload: dict[str, Any],
) -> dict[str, object]:
    if payload.get("status") != "ok":
        raise ValueError("public_smoke_orderbook_status_not_ok")
    bids = _levels(payload.get("bids"), "bids")
    asks = _levels(payload.get("asks"), "asks")
    top_bid = max(price for price, _ in bids)
    top_ask = min(price for price, _ in asks)
    if top_bid > top_ask:
        raise ValueError("public_smoke_orderbook_crossed")
    mid = (top_bid + top_ask) / Decimal("2")
    spread_bps = (top_ask - top_bid) / mid * Decimal("10000")
    bid_depth = sum((price * amount for price, amount in bids), Decimal("0"))
    ask_depth = sum((price * amount for price, amount in asks), Decimal("0"))
    last_trade = payload.get("lastTradePrice")
    return {
        "status": payload["status"],
        "last_update": payload.get("lastUpdate"),
        "last_trade_price": (
            str(_decimal(last_trade, "public_smoke_last_trade_invalid"))
            if last_trade is not None
            else None
        ),
        "best_bid": str(top_bid),
        "best_ask": str(top_ask),
        "observed_spread_bps": str(spread_bps),
        "top20_bid_quote_depth_usdt": str(bid_depth),
        "top20_ask_quote_depth_usdt": str(ask_depth),
        "bid_level_count_checked": len(bids),
        "ask_level_count_checked": len(asks),
    }


def build_report(
    *,
    stats_payload: dict[str, Any],
    orderbook_payload: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("public_smoke_timestamp_timezone_required")
    market_report, _, _, _ = _extract_market_stats(stats_payload)
    orderbook_report = _extract_orderbook(orderbook_payload)
    return {
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "mode": "PUBLIC_READ_ONLY_MARKET_SMOKE",
        "pair": "BTC/USDT",
        "public_market_stats": market_report,
        "public_orderbook": orderbook_report,
        "limits": {
            "market_minimum_and_precision": "NOT_EXPOSED_BY_THESE_PUBLIC_RESPONSES",
            "account_fee_schedule": "NOT_VERIFIED_WITHOUT_ACCOUNT_READ_AUTH",
            "individual_order_status": "NOT_QUERIED_WITHOUT_READ_TOKEN",
        },
        "data_authority": "NOBITEX_PUBLIC_MARKET_ENDPOINTS_ONLY",
        "credentials_used": False,
        "http_methods_used": ["GET"],
        "orders_submitted": 0,
        "order_permission": False,
        "Champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
    }


def run_smoke() -> dict[str, Any]:
    stats = _fetch_json(_STATS_PATH)
    orderbook = _fetch_json(_ORDERBOOK_PATH)
    return build_report(
        stats_payload=stats,
        orderbook_payload=orderbook,
        observed_at=datetime.now(timezone.utc),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary_keys = (
        "mode",
        "pair",
        "credentials_used",
        "orders_submitted",
        "order_permission",
        "trading_authorization",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
