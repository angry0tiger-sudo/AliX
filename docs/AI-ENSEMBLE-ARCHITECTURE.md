# AliX Nobitex AI Ensemble Research Architecture

## Scope and data contract

This branch is isolated from \`main\` and \`release/nobitex-mvp\`. The AI
ensemble is research-only. Training uses one five-year hourly BTCUSDT Spot
OHLCV dataset fetched from the public Nobitex UDF history endpoint. Model
features use OHLCV and technical indicators only. Order-book, taker-flow,
funding-rate, and other external microstructure archives are not required.

The 1h timeframe is the selected model timeframe. The fetch workflow combines
five backward-adjacent 365-day windows and requires exactly 43,800 contiguous
hourly candles. The CSV manifest records provider, symbol, date range, row
count, missing intervals, content SHA-256, and source commit. Training verifies
the manifest, CSV hash, first/last timestamps, 1h interval, and 24 rows per day.

No interpolation, forward-fill, synthetic replacement, or silent gap repair is
permitted. Synthetic rows exist only in unit tests and are not economic evidence.

## Causal OHLCV feature pipeline

The model uses completed OHLCV candles only:
- RSI(14);
- MACD(12/26/9), signal, and histogram;
- ATR(14) and ATR fraction;
- Bollinger %B and width (20, 2);
- 24-candle realized volatility;
- 24-candle Volume Z-Score;
- one- and three-candle returns, log return, and high-low range fraction.

Indicator warm-up NaNs remain visible and are excluded explicitly.

## Ensemble and regime model

LightGBM and XGBoost are the base models. A three-state diagonal Gaussian HMM
models DOWN/RANGE/UP regimes. HMM parameters are fit on each training segment;
validation probabilities use forward filtering only.

The Logistic Regression meta-learner is trained from out-of-fold base-model
probabilities and forward-filtered HMM probabilities. Outer evaluation uses
expanding chronological Purged Walk-Forward folds, purges overlapping labels,
and applies a 24-candle embargo. Trade-return evaluation permits one open
position and skips entries until the active label window ends.

## Labels, costs, and metrics

The existing Triple-Barrier contract is retained:
- horizon: 24 hourly candles;
- take profit: 2 ATR;
- stop loss: 1 ATR;
- entry: next candle open;
- collision: stop loss first;
- round-trip cost: 50 bps (25 bps per side).

Each fold reports trade count, win rate, net Profit Factor, compounded net
return, maximum drawdown, trade Sharpe, and Statistical Z-Score. The Statistical
Z-Score is the one-sample normal-approximation statistic for mean non-overlapping
net trade return against a zero-mean null. It is not a Deflated Sharpe Ratio or
a multiple-testing correction. It is null when the sample or its variance is
insufficient.

## Risk and paper-plan boundary

Fractional Kelly sizing retains the existing risk, volatility, and position
caps. The Nobitex adapter can produce only a bounded paper maker-slice plan
after liquidity checks. It has no order-write method. Mock order-book adapter
tests prove interface mechanics only; they do not create historical execution
evidence. Model training does not consume historical microstructure.

## Fixed safety state

\`\`\`text
Champion = null
order_permission = false
trading_authorization = NO-GO
private_api_calls = 0
real_order_attempts = 0
orders_or_execution_api_used = false
ALIX_DRY_RUN = true
ALIX_ALLOW_LIVE_TRADING = false
\`\`\`

Training artifacts and passing CI do not grant promotion or trading permission.
