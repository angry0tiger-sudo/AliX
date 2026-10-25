# Owner Directive: RL Reward Shaping and Minimum Training Budget

**Date:** 2026-10-25  
**Scope:** `research/rl-trading-agent` only  
**Authority:** Explicit user instruction. This directive overrides earlier roadmap text that limited the run to one pass through the training split.

## Required changes

1. Subtract `0.001` from reward on each policy `HOLD` action. This is a learning signal only; it is not a requirement to produce a minimum number of fills.
2. Add `0.0001` only when a complete position round trip closes with cumulative realized PnL positive after both entry and exit fees and slippage. Partial-sale PnL is accumulated until the position is fully closed. Unrealized PnL and losing/breakeven completed trades receive no bonus.
3. Train Recurrent PPO for at least 1,000,000 environment steps. The verified training split is used exclusively; after reaching its end, replay that same chronological training split from its beginning. No validation or final-holdout candle may be used during training or reward tuning.
4. Report action counts, completed trades, win/loss/breakeven counts, win rate, net profit/return, total fees/slippage, and max drawdown for validation and final holdout. Evaluate the untouched holdout once.

## Known risk and interpretation

The fixed Hold penalty can cause churn, and the profit bonus can distort policy learning if made too large. Keep the exact requested values for this experiment, report costs and economic performance separately from shaped reward, and do not infer strategy quality merely from increased activity. If the agent trades more but net economics deteriorate, report that result without retuning against holdout.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- No private exchange APIs, order endpoints, live orders, automatic promotion, or weakened CI/complexity gates.
