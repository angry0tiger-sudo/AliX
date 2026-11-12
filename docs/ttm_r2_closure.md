# TTM-R2 Phase Closure

**Status:** CLOSED FOR THE 1h OHLCV ENTRY-STRATEGY TRACK  
**Decision:** NO-GO for strategy promotion; Champion=null  
**Reference commit:** ad75f44800d3024bc50e7654b1762f49298ef0d9

## Executive decision

TTM-R2 passed the pre-registered volatility-forecast research gate on the chronological 1h Nobitex holdout. The same evaluation did **not** establish an economically profitable entry strategy. The 1h OHLCV entry teacher remained materially unprofitable after a 50 bps round-trip cost. TTM improved risk sizing and reduced drawdown, but it did not change the teacher's entries, win rate, or trade count.

The TTM-R2 phase is therefore closed as a **successful risk-forecasting component**, not as a trading strategy and not as a promotion candidate.

## Reproducible experiment identity

| Field | Value |
|---|---|
| Model | ibm-granite/granite-timeseries-ttm-r2 |
| Immutable model revision | d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f |
| Weight SHA-256 | a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710 |
| Dataset | Nobitex BTCUSDT, native 1h, public UDF history only |
| Rows | 31,801 |
| Dataset SHA-256 | 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e |
| Data window | 2021-10-31 20:30 UTC to 2025-06-17 20:30 UTC |
| Commit | ad75f44800d3024bc50e7654b1762f49298ef0d9 |
| Workflow | [TTM-R2 Run #25, attempt 2](https://github.com/angry0tiger-sudo/AliX/actions/runs/43094730462) |

## Final holdout: variance forecasting

There were 198 non-overlapping holdout forecasts. QLIKE and log-variance MAE are lower-is-better.

| Metric | TTM-R2 | 24h persistence | Result |
|---|---:|---:|---|
| QLIKE | 0.27785937 | 0.39347957 | TTM improved 29.38% |
| Log-variance MAE | 0.63270646 | 0.69222535 | TTM improved |
| Forecast samples | 198 | 198 | Meets 30-sample minimum |

The risk-forecast verdict was PASS_RISK_FORECAST_RESEARCH_ONLY. This is evidence for forecasting value only. The target floor was applied to 0.11% of hourly outputs, and 1.01% of holdout forecast origins touched the floor.

## Final holdout: offline position-sizing replay

Both variants used the same frozen OHLCV-only entry teacher, identical trade events, 50 bps round-trip costs, and the same trade count. TTM was a sizing overlay only.

| Metric | Fixed 10% position | TTM risk-sized |
|---|---:|---:|
| Net Profit Factor | 0.3731 | 0.3748 |
| Maximum drawdown | 21.63% | 18.26% |
| Win rate | 31.67% | 31.67% |
| Trade count | 480 | 480 |
| Net return | -21.56% | -18.17% |

The TTM overlay reduced maximum drawdown by 3.37 percentage points and improved Net PF by approximately 0.0018. However:
- Net PF remained far below the 1.1 promotion threshold.
- Maximum drawdown remained above the 15% target.
- Win rate and trade count were unchanged.
- The entry logic, not the risk forecaster, remains the dominant unresolved economic problem.

## Architecture decision

1. Retain TTM-R2 as an offline, risk-only volatility forecast component.
2. Do not allow TTM output to generate, suppress, or reverse entries.
3. Do not wire the multiplier into exchange/runtime execution.
4. Continue the research pivot using native 15m and 30m Nobitex OHLCV, with the validated hourly TTM forecast used only as a causal sizing overlay.
5. Require holdout Net PF >= 1.1, max drawdown < 15%, and more than 200 trades per year before considering a strategy for a separate paper-trading authorization review.

## Safety and non-promotion invariants

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- Zero private API calls, real-order attempts, and paper orders
- No TTM multiplier applied to runtime position sizing

This closure does not authorize paper or live trading. A successful forecast gate is not an economic acceptance gate.
