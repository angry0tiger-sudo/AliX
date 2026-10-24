# Stage B Proxy Liquidity Sensitivity

## Status and meaning

This is a research-only sensitivity run. The original Stage A candle volume field is not changed. The script obtains actual public Binance Spot BTCUSDT hourly klines from the official Binance Vision archive, then creates a separately named proxy turnover series solely for applying a hypothetical 1% participation cap.

The BTC proxy rule is fixed before evaluation:

- The daily owner-supplied estimate of 1,500,000 quote units is an upper bound, not an observed value for each historical date.
- For each UTC day, proxy volume is Binance hourly quote_asset_volume multiplied by min(10%, 1,500,000 / Binance daily quote_asset_volume).
- Consequently, the proxy uses at least a 90% discount from Binance and does not exceed the 1,500,000 daily cap.
- Missing or unmatched source hours receive zero proxy capacity. No interpolation, equal-hour allocation, or fabricated trade volume is permitted.
- A 10 quote-unit order needs proxy quote turnover of at least 1,000 quote units in its individual candle to meet the 1% cap.
- The ETH and ZEC estimates are not used because the current Stage A artifact is BTCUSDT-only.

This makes a reproducible bounded proxy scenario, not historical Nobitex volume. Any resulting gate pass is a proxy sensitivity result and cannot change economic acceptance from NO-GO.

## Revisited Triple-Barrier grid

The grid keeps 24- and 48-hour maximum holding periods and uses eight combinations:

- Profit-taking multipliers: 2.0 and 3.0.
- Stop-loss multipliers: 0.75 and 1.25.
- Target volatility is trailing 24-hour volatility with a 0.51% floor. Thus the smallest take-profit hurdle is 2.0 × 0.51% = 1.02%; the actual barrier can be larger when historical volatility is higher.
- The probability threshold remains fixed at 0.50.
- Costs remain 25 bps per side fee plus 20, 35, or 50 bps per side slippage, for 90, 120, or 150 bps round trip.

A barrier event is not a model-selected trade. Predictions, proxy capacity and costs must all align before a row counts as a proxy-eligible trade.

## Banded Binance bookDepth adapter

The adapter accepts the official percentage-band schema: timestamp, percentage, depth, and notional. It requires the complete set of -5, -4, -3, -2, -1, +1, +2, +3, +4, +5 at each timestamp, monotonic cumulative depth/notional, positive finite fields, and a separate timestamped reference-price CSV. Every notional/depth implied average price must fall inside its declared percentage band, with a 0.5% reference-price tolerance. The reference stream must be no older than 60 seconds.

Only normalized relative imbalance features at 1%, 2%, and 5% bands are produced. The adapter does not invent a 20-level L2 book, does not emit absolute depth as a model feature, and does not infer Nobitex execution liquidity. This banded format does not provide a reliable best-bid/ask spread feature.

The model-enhanced group remains BLOCKED unless a complete, timely, price-aligned dataset is provided with a provenance manifest. The manifest schema is alix.binance-bookdepth-manifest.v1 and requires the official Binance Vision source, venue/symbol identity, explicit approved permission status, and matching SHA-256 checksums for both the band CSV and reference-price CSV.

## Acceptance and safety

The experiment computes proxy-only diagnostics for the same hard thresholds: at least 100 proxy-eligible out-of-sample trades, win rate greater than 60%, positive net expectancy after costs, maximum drawdown below 15%, and all three slippage scenarios passing. All criteria must pass together for a proxy sensitivity candidate to be labelled as passing. Regardless of these results:

- economic_evaluation.status remains NOT_EVALUATED_PROXY_ONLY;
- economic_acceptance remains NO-GO;
- Champion remains null;
- order_permission remains false;
- trading_authorization remains NO-GO;
- no private API or order endpoint is used;
- the reserved final 90-day holdout remains untouched;
- PR #36 is not modified by this branch or workflow.

A future real economic evaluation still requires historically valid Nobitex candle quote turnover and a separately verified, temporally aligned market-data set. Proxy results cannot satisfy that requirement.


## Timestamp-phase alignment correction

The pinned Nobitex candles are timestamped on the half-hour UTC phase (for example HH:30Z), while Binance one-hour kline labels are aligned to HH:00Z. An exact timestamp join therefore produces zero matches even when both histories cover the same dates.

The sensitivity runner now validates that each source has a uniform timestamp phase and shifts only the Binance kline label by the observed fixed Nobitex phase before matching. It does not shift or modify the measured volume value. The resulting value still describes Binance's original UTC-hour interval, which only partially overlaps the Nobitex candle interval; the output explicitly records exact_same_window_turnover_claimed=false. This supports a disclosed proxy-window sensitivity only, not historical Nobitex turnover, an exact per-candle liquidity measurement, or economic acceptance. Mixed timestamp phases fail closed.
