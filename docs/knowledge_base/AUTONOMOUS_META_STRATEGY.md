# Autonomous Meta-Strategy: Technical Roadmap and Research Contract

## Goal

Build an autonomous *research selector* that can choose among multiple explicitly registered strategies based on a forward-filtered Hidden Markov Model (HMM) market regime. The implementation is a research decision system, not a production trading agent: a selection is never an order permission.

## Architecture

1. **Verified market input**: consume the immutable Stage A historical artifact through the existing provenance and holdout checks. Use hourly OHLCV bars. No network fetching or order-book proxy is substituted for missing venue data.
2. **Causal feature layer**: EMA(8/21), MACD(12,26,9), RSI(14), Bollinger bands(20), ATR(14), previous 20-bar high/low, and volume relative to rolling mean. Previous breakout levels are shifted by one bar so the current candle cannot define its own breakout threshold.
3. **Regime layer**: fit the existing three-state Gaussian HMM on the training segment of each expanding fold, then forward-filter only the subsequent test segment. States are \`trend\`, \`range\`, and \`high_volatility\`. The maximum posterior is not calibrated probability; the selector applies a minimum-confidence abstention threshold.
4. **Strategy registry and selector**:
   - \`trend_following\`: EMA alignment plus MACD confirmation in \`trend\`.
   - \`mean_reversion\`: oversold RSI plus price at/below the lower Bollinger band in \`range\`.
   - \`volatility_breakout\`: close above the *previous* 20-bar high in \`high_volatility\`. This prototype is long-only, so downside breakouts abstain rather than short.
   - \`scalping\`: short holding horizon, positive observed volume and relative volume at least 1.5x, plus short-term positive momentum. With zero positive source-volume observations, it remains blocked.
5. **Dynamic risk plan**: ATR determines stop/target distances; regime-specific multipliers widen the plan in high volatility and tighten it in range conditions. The risk budget equals equity × configured risk fraction × HMM confidence; notional is separately capped. These quantities are simulation-only.
6. **Evaluation**: expanding chronological folds compare the meta-selector against each of the four fixed-strategy baselines. Entry is no earlier than the next bar open. If stop and target both appear to hit in one OHLC bar, the stop is assumed first. Assumed fees and slippage are deducted from simulated P&L.
7. **Audit artifacts**: persist a JSON report, trade ledger, and selector-decision ledger. Registry code hash, input hashes, fold boundaries, strategy selection, sizing inputs, costs and safety fields are recorded.

## Current limits

The source workflow reserves the final 90-day holdout; this evaluator loads development rows only. The Stage A input can contain price history while still having zero positive source-volume observations. That is enough to test price-only signals, but not to validate a high-volume scalping strategy. Observed spread/order-book liquidity is also unavailable; fees and slippage in this simulation are explicit assumptions rather than exchange-validated fills.

The report therefore uses \`ASSUMPTION_BASED_HISTORICAL_BAR_SIMULATION\` and permanently leaves \`economic_acceptance = NO-GO\`. A better win rate or expectancy in one development comparison is exploratory, not proof of future profitability.

## Acceptance gates

- Four unique strategies, each registered \`RESEARCH_ONLY\`, owner-unapproved, long-only, and scoped to 1h bars.
- HMM trained on past rows only; feature calculations causal; no reserved holdout access.
- Unknown regime, missing features, invalid ATR, low HMM confidence, or unavailable required volume yields no entry.
- Dynamic stop/target distances and confidence-scaled, exposure-capped sizing are recorded for simulated trades.
- Report includes win rate, per-trade net expectancy, total simulated return, drawdown, and per-baseline comparisons. Empty samples remain \`null\`, never zero-filled.
- CI, coverage, static typing, security and complexity gates remain unchanged.
- \`Champion = null\`, \`order_permission = false\`, \`trading_authorization = NO-GO\`; no exchange endpoint or private API is called, and no real order is attempted.

## Later phases (not authorized by this change)

1. Obtain provenance-pinned history with actual positive volume and separately approved spread/order-book data for execution sensitivity.
2. Calibrate HMM confidence on training-only data and tune thresholds inside nested development folds.
3. Add an untouched economic validation period, fee/slippage sensitivity, and minimum sample floors per regime.
4. Audit selector churn, risk concentration, regime instability and strategy-level sample counts.
5. Require explicit owner review before any model, strategy registry, or execution permission can change.
