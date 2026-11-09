# AliX Nobitex MVP

This release branch contains a **paper-only MVP** for a long-only Spot strategy:

- Entry: close strictly above the highest high of the previous 20 completed bars.
- ATR: Wilder ATR(14).
- Exit: trailing stop ratcheted to highest completed-bar high minus 2 ATR.
- Execution intent: passive limit-maker slices, 0.5% liquidity band, maximum 10% participation per visible level.
- Risk sizing: 0.5% of configured paper equity at the 2-ATR stop, capped at 10% of equity.
- No leverage, no shorting, no market orders, no real orders.

**No profitability claim is made.** This release is an execution-preparation and paper-simulation deliverable, not a live-trading authorization.

## Important deployment blocker: maker-only is not verified

The current published Nobitex Spot order documentation describes `POST /market/orders/add` with `execution=limit`, but does not document an exchange-enforced `postOnly` parameter. A limit price that does not cross the visible book can still become marketable between the book read and order acceptance. Therefore this release has **no live order-submission method**. The adapter raises `LiveExecutionBlocked` before any order request.

The official published API host is `https://apiv2.nobitex.ir` (not the older `api.nobitex.ir` host). The public order-book route is `GET /v3/orderbook/{SYMBOL}`; it requires no token. This code includes a GET-only client, but the 24-hour paper run does not call the exchange.

## 1. Setup

Python 3.12 is required. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the example environment file:

```bash
# Windows PowerShell:
Copy-Item config/nobitex-mvp.env.example .env
# Linux/macOS:
cp config/nobitex-mvp.env.example .env
```

The paper MVP needs **no API key**. The credential fields in the example are intentionally empty.

The public order-book client is network-disabled by default. A public GET
requires both `dry_run=False` and an explicit `owner_approved=True` constructor
argument. The 24-hour simulation does not instantiate or call that client.

### API key handling

The API key string previously supplied by the Owner is the public half of a key pair, not a bearer token. Nobitex API-Key signing requires the matching private signing key as well. The private half must never be pasted into chat, committed, logged, or placed in a checked-in file. This MVP does not implement authenticated signing because there is no live order route until the maker-only contract is verified.

## 2. Run the 24-hour simulated paper session

```bash
python -m scripts.run_nobitex_mvp --simulate-hours 24 --seed 17
```

This replays 48 synthetic 30-minute candles (24 simulated market-hours) in accelerated time. It uses a deterministic synthetic order book and simulated maker-intent fills. It does not contact Nobitex.

The JSON report records:
- every generated BUY/SELL signal, reference price, ATR and stop;
- intended quote notional, liquidity-gate evidence, and every planned child slice;
- simulated fills, exact base-quantity position closure, child-slice count, and liquidity rejections;
- zero exchange API calls, zero real orders, and zero real fills;
- `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO`.

A simulated fill is not evidence that an order would fill on the exchange.

## 3. Tests

```bash
pytest -q tests/unit/test_nobitex_mvp.py
```

The tests cover order-book parsing/freshness, the 0.5% depth gate, child-order slicing, participation caps, non-crossing limit prices, token-bucket refill, Donchian/ATR/trailing-stop behavior, the hard live-order block, and a 24-hour simulated run.

GitHub Actions also runs the repository Standard CI and a dedicated 24-hour simulated-paper smoke job on this release branch.

## 4. Live mode: currently unavailable

Do **not** set `ALIX_DRY_RUN=false` or `ALIX_ALLOW_LIVE_TRADING=true`. The current code intentionally refuses that configuration.

Before a future live release, all of the following must be completed:

1. Nobitex must confirm an exchange-enforced post-only parameter or equivalent order type for Spot order creation. A normal `limit` order is not a maker-only guarantee.
2. The API-Key authentication/signature contract must be verified against current official documentation. The public key alone is insufficient; its paired private key is required.
3. A separate Owner approval must authorize authenticated API calls and live order submission.
4. A live adapter must be implemented with idempotent client order IDs, order-state reconciliation, cancel/replace controls, balance checks, fee/precision metadata, and emergency halt behavior.
5. The 24-hour paper soak and all release CI gates must pass on the exact release SHA.

Until then, no live execution switch is provided. This is deliberate fail-closed behavior, not an unfinished hidden toggle.

## Nobitex API references

- Official API documentation: https://apidocs.nobitex.ir/
- Official API documentation source: https://github.com/nobitex/docs-api
- Public order-book endpoint: `GET /v3/orderbook/{SYMBOL}`
- Spot order endpoint: `POST /market/orders/add` (limit parameters are documented; post-only is not documented)
- Official shared order-placement limit: 300 requests per 10 minutes. The MVP's local token-bucket policy is more conservative.

## Safety state

```text
ALIX_DRY_RUN=true
ALIX_ALLOW_LIVE_TRADING=false
Champion=null
order_permission=false
trading_authorization=NO-GO
private_api_calls=0
real_order_attempts=0
nobitex_api_calls=0
```
