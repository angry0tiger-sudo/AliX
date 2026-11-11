"""Read-only HTTP reachability probe for public alternative-data endpoints.

Output order is fixed: Binance Futures, Bybit, OKX.
Each line is only the final HTTP status code. "000" means no HTTP response
was received (for example, DNS, TLS, connection, or timeout failure).
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINTS = (
    "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1",
    "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1",
    "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
)


def http_status(url: str) -> int:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "AliX-Alternative-Data-Access-Probe/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return int(response.status)
    except HTTPError as error:
        return int(error.code)
    except (URLError, TimeoutError, OSError):
        return 0


def main() -> None:
    for endpoint in ENDPOINTS:
        status = http_status(endpoint)
        print(f"{status:03d}")


if __name__ == "__main__":
    main()
