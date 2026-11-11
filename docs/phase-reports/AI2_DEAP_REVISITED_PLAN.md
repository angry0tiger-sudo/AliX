# Phase AI-2 Revisited — DEAP Genetic Programming

Status: Owner-approved research phase; offline implementation only.
Execution branch: research/ml-core-deap-evolution
Dataset: immutable Nobitex BTC/USDT Spot 1h, 31,801 pre-hack candles.
Cost: 25 bps maker fee per side, 0 bps slippage, 50 bps round trip.

## Search contract

- Population 500; 30 generations; tournament selection (size 5).
- One-point crossover at 70%; uniform subtree mutation at 25%.
- Strongly typed boolean expression trees. Numeric terminals are bounded, normalized causal OHLCV features; the root returns BUY or NO_BUY.
- Primitives: add, sub, mul, div, gt, lt, logical AND and logical OR.
- Numeric operations are finite, bounded and safely divided. Tree height is at most 5 and node count at most 31.
- Long-only / flat Spot simulation. No shorting, leverage, private API, exchange adapter, or order API.

## Features and timing

The 1h tree consumes RSI_14, MACD_hist, ATR_14, Close (log distance from causal EMA100), Volume_ZScore, ROC_6 and Range_14. All indicators use information available at or before the current candle close. Features are normalized into finite bounded ranges. An entry decision made at bar close is filled at the next bar open.

The boolean signal is latched as a position while true. The position exits at the next open after the rule becomes false, or after 48 completed bars if the rule remains true. A forced terminal liquidation closes any remaining position. Drawdown uses a conservative intrabar high-water mark versus the candle low.

## Fitness and selection isolation

Evolutionary fitness uses only the first chronological 70% of rows, excluding 48 bars before the boundary. The search objective:
- applies a severe penalty below 200 completed trades per year and an absolute 150-trade support floor;
- heavily penalizes Net PF below 1.0;
- applies a massive penalty for maximum drawdown above 15%;
- rewards after-cost PF and net return when gates are met;
- applies a small tree-size and height complexity penalty;
- checks three chronological training-only folds to discourage single-regime solutions.

The final 30% is not used to score fitness, select the Hall-of-Fame genome, tune a parameter, or select a checkpoint. It begins after a 48-bar embargo and is evaluated once after the winning genome is frozen. Owner-review eligibility requires Net PF >= 1.1, Max Drawdown < 15%, more than 200 completed trades per year on the untouched OOS segment, and positive net return. The OOS trade-count gates are evaluated strictly, not rounded for pass/fail.

## Cost, provenance and safety

The input is accepted only if the CSV SHA-256 matches the approved AI-6 Nobitex artifact and its manifest pins the source, 31,801 rows, exact hourly grid, producer commit and no-backfill policy. No API call is made by the evolution workflow.

CI success is engineering evidence, not economic acceptance. A failing economic result is reported as NO-GO; it must not trigger another search against the OOS segment.

Safety invariants:
- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- private_api_calls = 0
- real_order_attempts = 0
- orders_or_execution_api_used = false
