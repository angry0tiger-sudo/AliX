# External Data Pivot — ABORTED

**Owner decision:** ABORTED on 2026-11-11. The final execution venue is Nobitex Spot. External derivatives data will not be used in this research phase.

## Evidence and reason for closure

The GitHub Actions reachability probe on the research branch returned:

| Provider endpoint | HTTP status | Decision |
|---|---:|---|
| Binance Futures funding-rate endpoint | 451 | Geo/access blocked from the GitHub Actions runner |
| Bybit linear funding-history endpoint | 403 | Access blocked from the GitHub Actions runner |
| OKX current funding-rate endpoint | 200 | Endpoint reachable; this does not establish historical coverage |
| OKX funding-rate-history endpoint | 200 | Endpoint reachable, but the tested `limit=1` response returned a single recent record with `method=current_period`, not historical coverage for the pre-hack research window |

The OKX probe response contained `fundingRate=0.0001`, `instId=BTC-USDT-SWAP`, and `fundingTime=1794355200000`. HTTP 200 proved endpoint reachability only. It did not demonstrate that a complete, gap-free series exists for the Nobitex dataset window from November 2021 through June 2025. No missing values will be interpolated, fabricated, or backfilled from future observations.

The alternative-data experiment is therefore closed. No Funding Rate, Open Interest, derivatives, sentiment, news, or other external features are accepted in this phase.

## Replacement research scope

- Use only the immutable Nobitex BTC/USDT Spot 1-hour pre-hack OHLCV dataset.
- Dataset contract: 31,801 candles, Nobitex public UDF provenance, exact hourly timestamp grid, no gaps, and SHA-256 verified manifest.
- The frozen primary Ensemble and AI-6 PPO risk filter receive OHLCV-derived causal features only.
- Validation and final holdout stay chronological and untouched; no future data or external backfill.
- Cost assumption remains Nobitex Spot maker fee 25 bps per side, 50 bps round trip, no shorting, and no leverage.

## Safety and execution boundary

OKX is not an execution venue. The execution venue remains Nobitex Spot. This research workflow is offline and does not use the Nobitex key or make account, private, balance, or order API calls.

`Champion = null`, `order_permission = false`, and `trading_authorization = NO-GO` remain mandatory. The external-data pivot closure does not authorize Paper Trading or live orders.
