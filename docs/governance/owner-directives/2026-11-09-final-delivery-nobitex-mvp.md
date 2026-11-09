# Owner Directive: Final Delivery — Nobitex MVP

Date: 2026-11-09

## Scope

The release branch is based on the last known green V4-AP commit:
- Base SHA: 798ef2d49bcbc715d35b16862a2cf83aa0865486
- V4-AP diagnostic workflow run #26: 42493418590, success
- Standard CI run #2034: 42493418617, success

The release MVP is a simple long-only Spot strategy:
- Donchian-20 close breakout using the previous 20 completed bars;
- Wilder ATR(14);
- trailing stop ratcheted to highest completed-bar high minus 2 ATR;
- passive maker-intent child slices;
- 0.5% order-book liquidity band;
- 10% maximum participation per visible price level;
- risk sizing at 0.5% equity risk, capped at 10% position notional.

## Research freeze

No further work is authorized on research/ml-core-ai6-rl-nobitex under this directive. Phase AI-6 RL is DEFERRED, not failed and not evaluated. The release branch does not modify the research branch.

## Nobitex execution contract

The official documentation presently documents public GET /v3/orderbook/{SYMBOL} and Spot POST /market/orders/add with execution=limit. It does not document an exchange-enforced post-only flag. Maker-only execution therefore cannot be guaranteed from the published contract. No live order endpoint is implemented in this release. The read-only order-book transport is GET-only.

## Safety

- ALIX_DRY_RUN=true by default.
- ALIX_ALLOW_LIVE_TRADING=false by default.
- No authenticated calls or orders are made by the 24-hour paper simulation.
- Champion=null.
- order_permission=false.
- trading_authorization=NO-GO.
- 30 bps is not used as a live fee assumption.
- main remains untouched.

A future live-release directive is required after Nobitex confirms post-only support and the full API-Key signing pair is stored securely.
