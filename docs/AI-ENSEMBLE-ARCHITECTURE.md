# AliX Nobitex AI Ensemble Research Architecture

## Fixed data contract

Native Nobitex BTCUSDT Spot 1h candles only. The inclusive window is
`1635712200` (2021-11-01 00:00 Asia/Tehran) through
`1750192200` (2025-06-18 00:00 Asia/Tehran), exactly 31,801 contiguous
candles. The logical window is inclusive; because TradingView UDF uses an exclusive right boundary, the request sends `from=start` and `to=end+1 second`, then retains only returned candles through the requested end.
No Binance backfill, external-source substitution, interpolation, forward
fill, synthetic replacement or silent gap repair is permitted. The same
Nobitex request can be retried; a remaining gap fails the run.

The dataset manifest pins the data source, timestamps, actual and expected
row counts, missing intervals, content SHA-256 and safety state. The trainer
validates the exact expected timestamp grid.

## Chronos-2 feature

The frozen zero-shot checkpoint is `amazon/chronos-2`, pinned to revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`. Runtime:
`chronos-forecasting==2.3.2`, CPU-only inference.

Each forecast uses 512 completed hourly candles to predict 24 future log
closes. Past-only covariates are log return, open gap, range fraction and log
volume. Forecasts refresh every 24 candles. Differences in predicted log
prices produce the hourly-return feature `chronos_pred`. Step one is aligned
as-of to the forecast-origin decision row; the following 23 steps are aligned
to the next 23 rows. The first 511 rows remain warm-up and are excluded from
both Baseline and Candidate. Each inference is independent
(`cross_learning=False`); future realized candles, labels, account state and
execution information are not inputs.

## Controlled evaluation

Baseline uses the existing OHLCV features. Candidate adds only
`chronos_pred`. Both use the same rows, targets, nested chronological Purged
Walk-Forward folds, embargo, parameters, seeds, thresholds and trade
simulation. LightGBM and XGBoost remain the base models, with the forward-
filtered Gaussian HMM and out-of-fold Logistic Regression meta-learner.

Triple Barrier remains 24 candles, 2 ATR take profit, 1 ATR stop loss,
next-bar-open entry, stop-first collision, and 50 bps round-trip costs. The
same decisions are also evaluated at 90 bps, with the additional 40 bps per
trade deducted. Metrics include Net PF, win rate, maximum drawdown,
compounded return, trade count and the one-sample normal-approximation
Statistical Z-Score of non-overlapping net trade returns. This is not a
Deflated Sharpe Ratio or multiple-testing correction.

## Safety invariants

```text
Champion = null
order_permission = false
trading_authorization = NO-GO
private_api_calls = 0
real_order_attempts = 0
ALIX_DRY_RUN = true
ALIX_ALLOW_LIVE_TRADING = false
```

Passing CI or achieving a metric threshold does not authorize trading.
