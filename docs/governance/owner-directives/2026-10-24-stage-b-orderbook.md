# Owner Directive: Stage B Triple-Barrier and relative Order Book research
Date: 2026-10-24

The owner selected Binance USD-M Futures BTCUSDT for direction-only Order Book research. Binance absolute depth must never be extrapolated to Nobitex liquidity.

Requirements:
- Evaluate 24-hour and 48-hour Triple-Barrier labels with fixed, disclosed barrier multipliers and past-only volatility.
- Keep fee assumptions fixed at 25 bps per side; stress slippage at 20, 35 and 50 bps per side, producing 90, 120 and 150 bps round trip.
- Use Binance Level-2 only for normalized imbalance and relative spread; never use absolute depth for Nobitex sizing.
- Cap order notional at 1% of contemporaneous Nobitex quote notional. Zero or unverified volume units grant zero capacity.
- No private API, order endpoint, imputation, forward-fill, backfill, interpolation, resampling to repair gaps, or fabricated book/volume data.
- Preserve the reserved 90-day holdout, chronological OOS folds, matured-label purge, all quality gates and all safety invariants.

The pinned Stage A Segment A has zero positive-volume rows and the immutable artifact contains no Binance Order Book snapshots. Only the OHLCV control may run now; the book-enhanced model and economic acceptance remain blocked until valid permitted data is supplied. This directive does not modify Champion, order permission, economic acceptance or trading authorization.
