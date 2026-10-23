"""Bounded public-price observation that always abstains; it never places orders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http.client import HTTPSConnection
import ssl
from pathlib import Path
import time
from typing import Callable


def fetch_public_btc_usdt_price(*, timeout_seconds: float = 10.0) -> float:
    """Read a public ticker from a fixed host over verified TLS, without redirects."""
    path = "/market/stats?srcCurrency=btc&dstCurrency=usdt"
    connection = HTTPSConnection(
        "apiv2.nobitex.ir",
        timeout=timeout_seconds,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "application/json",
                "User-Agent": "AliX-Phase10-Observer/1",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("public_price_http_status_invalid")
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("public_price_payload_not_ok")
    stats = payload.get("stats")
    market = stats.get("btc-usdt") if isinstance(stats, dict) else None
    if not isinstance(market, dict):
        raise ValueError("public_price_market_missing")
    try:
        price = float(market.get("latest", "nan"))
    except (ValueError, TypeError):
        raise ValueError("public_price_value_invalid") from None
    if price <= 0 or price != price or price in (float("inf"), float("-inf")):
        raise ValueError("public_price_value_invalid")
    return price

def observe_public_price(
    *,
    duration_seconds: int,
    fetch_price: Callable[[], float] = fetch_public_btc_usdt_price,
    monotonic: Callable[[], float] = time.monotonic,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], None] = time.sleep,
    interval_seconds: int = 60,
) -> list[dict[str, object]]:
    if isinstance(duration_seconds, bool) or duration_seconds not in (300, 900, 3600):
        raise ValueError("observation_duration_must_be_bounded")
    if interval_seconds < 1 or interval_seconds > 60:
        raise ValueError("observation_poll_interval_invalid")
    start = monotonic()
    deadline = start + duration_seconds
    records: list[dict[str, object]] = []
    while monotonic() < deadline:
        at = clock()
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("observation_clock_timezone_required")
        try:
            value = fetch_price()
            status, reason = "PUBLIC_PRICE_OBSERVED", "SIGNAL_EXECUTION_DISABLED"
        except Exception:
            value = None
            status, reason = "DEGRADED", "PUBLIC_PRICE_UNAVAILABLE_FAIL_CLOSED"
        records.append(
            {
                "observed_at": at.astimezone(timezone.utc).isoformat(),
                "pair": "BTC/USDT",
                "price": value,
                "decision": "ABSTAIN",
                "reason": reason,
                "status": status,
                "data_authority": "PUBLIC_PRICE_ONLY",
                "order_permission": False,
                "Champion": None,
                "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
                "trading_authorization": "NO-GO",
                "orders_submitted": 0,
                "credentials_used": False,
            }
        )
        remaining = deadline - monotonic()
        if remaining > 0:
            sleep_fn(min(float(interval_seconds), remaining))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-seconds", type=int, choices=(300, 900, 3600), default=3600
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = observe_public_price(duration_seconds=args.duration_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "record_count": len(records),
                "mode": "PUBLIC_PRICE_OBSERVATION_ONLY",
                "order_permission": False,
                "Champion": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
