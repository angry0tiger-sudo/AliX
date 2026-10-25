# Owner Directive: Discrete RL Actions and Causal Opportunity Cost

**Date:** 2026-10-25  
**Branch:** `research/rl-trading-agent`  
**Authority:** Explicit owner instruction; supersedes prior continuous-action clauses in the RL Trading Agent and Action Masking directives. Immutable safety and CI requirements remain unchanged.

## Required architecture

1. Use RecurrentPPO with Gymnasium `MultiDiscrete([3, 4])`: direction index maps to `HOLD`, `BUY`, `SELL`; size index maps only to 25%, 50%, 75%, or 100%. There is no zero-size bucket.
2. For BUY, spend the selected fraction of currently available cash, inclusive of fees and slippage. For SELL, liquidate the selected fraction of current inventory. The size index is ignored on HOLD. If BUY has legal spendable cash, the selected amount must create a positive fill; exhausted cash remains fail-closed at the execution boundary.
3. Compute a 20-completed-candle moving average causally from market history. Apply a small `0.0005` opportunity-cost penalty only when the agent is flat, selects HOLD, and the current closed price is above that average. Do not add the moving average, portfolio state, or reward components to the observation. Do not penalize HOLD while invested.
4. Retrain RecurrentPPO for at least 1,000,000 steps with the discrete action contract. Report requested size-bucket counts, fills, no-op/masked actions, opportunity-cost penalties, net PnL/return, drawdown, fees and slippage.
5. Use data strictly before 2026-06-01 04:59 UTC for training/validation, preserving 70% training and 10% validation of that pre-cutoff window. Exclude the intervening prior candidate holdout segment from this model's fitting and scoring. Use the latest tail beginning at 2026-06-01 04:59 UTC as this model's final holdout and evaluate it exactly once.
6. Disclose that the latest tail was inspected in earlier research. It is out-of-sample for this discrete-action model's fit, but not novel to the full research history. Do not describe it as unseen across all prior work.
7. The training workflow must fail closed unless CI completes successfully for the exact training commit before any model training starts.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- Zero private exchange API calls, zero real order attempts, and no execution API use.
- No automatic promotion, no weakened CI/quality/complexity/coverage gates.
