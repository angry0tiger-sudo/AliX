# V4-AC Owner Directive Record

- Directive ID: V4AC-COST-OPTIMIZATION-2026-11-04
- Authority: explicit project-owner directive
- Status: authorized for research implementation; no promotion or trading authorization
- Base commit: a79389dd4a0bdf065b2a4012e613c5c09e137745
- Branch: research/ml-core-v4ac-cost-optimization

## Approved research matrix

- Symbols: BTCUSDT, ETHUSDT
- Timeframes: 15m, 30m, 1h, 4h only; every timeframe above 4h is prohibited
- Holding horizon: exactly 24 bars per timeframe (6h, 12h, 24h, 96h respectively)
- Exit policies: TB_1.5_1.0 and TRAIL_2.0
- Comparison modes: PRIMARY_ONLY and META_FILTERED
- Costs: 10 bps maker-fee assumption and 5 bps slippage assumption per side; 30 bps round trip
- Primary signal: unchanged V4-Y Bollinger(20, 2.0) upper-band breakout with Close > EMA200; closed-candle decision and next-bar-open execution
- Meta: deterministic OOF LightGBM, separately trained per symbol/timeframe/exit policy using outcomes produced by the same exit policy and the same cost contract

## Evaluation and risk controls

The outer split remains chronological 70/15/15. Only matured training events are eligible for model fitting. Fit/calibration remains purged and calibration is label-blind. The forward validation interval is used for comparison only; the final 15% holdout remains untouched.

The fixed-bps next-open simulator is an economic cost scenario, not a queue-aware limit-order simulator. It does not model maker fill probability, queue priority, partial fills, cancellations, or non-fills. Binance public Spot candles remain the historical price proxy; the fee/slippage settings are hypothetical Nobitex-style assumptions and are not verified Nobitex fee schedules or execution behavior.

Research-only restrictions remain: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, sequence_model_used=false. V4-AA, V4-AB, and main are not modified.

## Acceptance criteria

- Exact 32-cell matrix and 16 Meta-training scenarios are present.
- Only the four approved timeframes are accepted; no timeframe above 4h can enter the matrix.
- Every candidate uses exactly 24 bars and both exit policies are tested.
- Both Primary-only and Meta-filtered outputs are reported for each candidate.
- Standard CI succeeds on the exact commit without lowering any gate or coverage threshold.
- Economic workflow requires Standard CI success for that exact commit before fetching data or running the sweep.
- Step Summary and downloadable artifacts preserve all metrics, skipped reasons, provenance, and safety invariants.
- Promotion remains blocked pending independent OOS and owner review.

The implementation commit SHA is bound by the GitHub commit metadata for this record and the accompanying source changes.
