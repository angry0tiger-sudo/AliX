"""Read-only HTTP reachability probe for alternative-data public endpoints.

Output order:
1. Binance Futures funding-rate HTTP status.
2. Bybit funding-history HTTP status.
3. OKX current funding-rate HTTP status.
4. OKX historical funding-rate HTTP status and a bounded response snippet.

"000" means that no HTTP response was received.
"""

from __future__ import annotations

import json

import httpx


HEADERS = {
    "Accept": "application/json",
    "User-Agent": "AliX-Alternative-Data-Access-Probe/1.0",
}
TIMEOUT = httpx.Timeout(15.0, connect=5.0)

STATUS_ENDPOINTS = (
    "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1",
    "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1",
    "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
)
OKX_HISTORICAL_ENDPOINT = (
    "https://www.okx.com/api/v5/public/funding-rate-history"
    "?instId=BTC-USDT-SWAP&limit=1"
)
RESPONSE_SNIPPET_LIMIT = 500


def get_response(url: str) -> httpx.Response | None:
    """Return the HTTP response, or None when transport fails."""
    try:
        return httpx.get(url, headers=HEADERS, timeout=TIMEOUT)
    except httpx.HTTPError:
        return None


def response_snippet(response: httpx.Response) -> str:
    """Render a bounded JSON response snippet without interpreting its fields."""
    try:
        payload = response.json()
        rendered = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError:
        rendered = response.text.replace("\n", " ").replace("\r", " ")
    return rendered[:RESPONSE_SNIPPET_LIMIT]


def main() -> None:
    for endpoint in STATUS_ENDPOINTS:
        response = get_response(endpoint)
        status = response.status_code if response is not None else 0
        print(f"{status:03d}")

    historical_response = get_response(OKX_HISTORICAL_ENDPOINT)
    if historical_response is None:
        print("000 <no-http-response>")
        return

    print(
        f"{historical_response.status_code:03d} {response_snippet(historical_response)}"
    )


if __name__ == "__main__":
    main()
