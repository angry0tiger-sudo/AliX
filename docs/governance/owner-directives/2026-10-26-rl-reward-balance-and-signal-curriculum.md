# Owner Directive: Balanced Hold Reward and Annealed Trend Guidance

**Date:** 2026-10-26  
**Branch:** `research/rl-trading-agent`  
**Authority:** Explicit owner instruction; supersedes the earlier reward/entropy coefficients for this experiment. Safety gates and evidence requirements are unchanged.

## Required experiment

1. Reduce the execution-boundary penalty for a masked BUY/SELL request to `0.001`. The legal mask remains active and a masked action must never execute.
2. Balance HOLD shaping with `+0.0001` when an open position's expected net liquidation value after exit slippage and fees is above position cost basis, and `-0.0001` when it is below cost basis. No HOLD position-PnL shaping while flat or at break-even.
3. Set RecurrentPPO `ent_coef=0.05`.
4. Add a training-only causal trend hint: reward `0.0002` for a legal executed BUY when the 20-completed-candle SMA slope is positive, or a legal executed SELL when the slope is negative. Do not penalize the opposing action. Anneal the coefficient linearly to exactly zero over the first 300,000 environment steps, updating every 1,000 steps.
5. Set guidance scale to zero for Validation and final Holdout. Do not expose the moving average, portfolio state, PnL, or reward signals through observations. Do not retune against Holdout.
6. Train at least 1,000,000 steps after all CI gates pass on the exact commit. Report actual selected actions, fills, costs, PnL, drawdown, and realized results; increased trading activity alone is not success.

## Immutable safety and quality requirements

`Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`. Zero private exchange API calls, zero real orders, no execution endpoint, no automatic promotion. No CI, coverage, complexity, architecture, or security gate may be weakened.
