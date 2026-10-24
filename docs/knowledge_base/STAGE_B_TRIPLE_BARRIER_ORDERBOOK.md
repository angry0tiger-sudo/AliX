# Stage B Triple-Barrier + Relative Order-Book Research

## Status and input gates

This is a separate research-only experiment branch. It runs Triple-Barrier outcome diagnostics and an OHLCV-only chronological OOS control from the immutable Stage A artifact. The book-enhanced model and executable economic evaluation require permitted historical Binance USD-M Futures Level-2 snapshots and positive, unit-verified Nobitex candle quote volume. A blocked gate is not an economic result.

## Cross-venue policy

- Binance BTCUSDT Order Book is used only for normalized directional features: top-5 imbalance, top-20 imbalance and best-level spread in basis points. Absolute depth and quantity columns are never model features or a proxy for Nobitex liquidity.
- Input snapshots must have UTC timestamps and twenty sorted bid/ask price levels with finite positive prices, non-negative quantities, unique timestamps and a non-crossed best quote.
- Candle timestamps represent candle-open times. Decision time is candle close. Only the latest earlier/equal snapshot at most 60 seconds old may be joined. Missing or stale snapshots fail closed; no fill.
- The pinned Stage A artifact has no historical Binance Level-2 snapshots. No private API credentials or order endpoints are used to fetch them.

## Nobitex liquidity cap

Maximum order notional is 1% of contemporaneous Nobitex candle quote notional. Base-asset volume is multiplied by close; quote-asset volume is already notional. The unit must be explicitly verified whenever volume is positive. Zero volume or unknown units grants zero order capacity. No volume is fabricated.

All 58,655 source Segment A candles have zero volume. Its reserved final 90-day holdout is removed before features, targets and model fitting. The pinned data therefore cannot support executable economic evaluation.

## Fixed experiment design

- Horizons: 24 and 48 hourly bars.
- Profit-taking barriers: 1.5 and 2.0 times trailing 24-hour volatility.
- Stop-loss barriers: 0.75 and 1.0 times trailing 24-hour volatility.
- Target volatility is historical and known at event time. Existing Triple-Barrier implementation is reused; same-candle PT/SL collisions resolve conservatively to stop-loss.
- Chronological expanding OOS folds use a three-year initial train window and 90-day test windows. Training label end must precede test start, and OOS labels must mature before test end.
- Costs: 25 bps per-side fee plus 20, 35 and 50 bps per-side slippage, giving 90, 120 and 150 bps round trip.
- Probability cutoff 0.50 is fixed and not optimized. Sixteen configurations are planned: two feature groups times eight horizon/barrier configurations. Without real book snapshots, only eight OHLCV controls execute; eight book candidates remain blocked.
- Counterfactual predictions are reported separately from executable trades and cannot satisfy acceptance.

Economic acceptance requires at least 100 executable OOS trades, win rate above 60%, positive net expectancy and absolute maximum drawdown below 15%, all at once. Until data gates pass, status is NOT_EVALUATED_BLOCKED_INPUT_DATA and economic acceptance stays NO-GO.

## Safety

The workflow downloads only the pinned public Stage A artifact, produces JSON/CSV research artifacts, and verifies provenance and immutable safety states. Tests cover normalized relative features, stale/crossed book rejection and the 1% local-volume cap. It makes no private API calls and never submits orders. Production schema and model/strategy registries remain unchanged.


## Provenance gate update

If Order Book data is later supplied, the input must be accompanied by a JSON manifest with schema `alix.binance-orderbook-manifest.v1`, venue `binance_usdm_futures`, symbol `BTCUSDT`, an HTTPS Binance source URL, `permission_status: approved`, and a matching CSV SHA-256. The approved status is an explicit owner/provider authorization assertion, not something inferred from file availability. Economic acceptance requires every candidate to pass all three slippage scenarios (90/120/150 bps round trip), not just the most optimistic one. The 100 quote-unit position is a fixed research sizing unit used only to test the 1% local-liquidity cap; it does not modify production stake settings.
