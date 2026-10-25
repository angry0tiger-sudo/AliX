# Owner Directive: RL Action Masking, True Inactivity, and New Holdout

**Date:** 2026-10-25  
**Scope:** `research/rl-trading-agent` only  
**Authority:** Explicit user instruction. This directive supersedes earlier reward-shaping semantics for the new experiment.

## Required behavior

1. Block SELL before simulation execution whenever inventory is zero; block BUY whenever the position is fully allocated and when the requested buy target would reduce the position. Count masked requests separately from executed fills.
2. Keep RecurrentPPO and its continuous action space. Masking is enforced deterministically at the environment execution boundary; do not claim distribution-logit masking. Expose the pre-action legal mask, requested action, executed outcome, and block reason in the simulator's `info`.
3. Apply the `0.001` inactivity penalty when actual position quantity does not change for more than three consecutive steps, regardless of whether the request was HOLD or an illegal/masked action. Any actual position change resets the streak. Keep the `0.0001` bonus only for a completed round trip that realizes positive PnL after fees and slippage.
4. Exclude the previously evaluated holdout interval beginning 2026-06-01 04:59 UTC. Choose the new chronological analysis window only from rows strictly before that timestamp, split 70/10/20, train for at least 1,000,000 timesteps, and evaluate the new holdout exactly once.
5. Report requested actions, masked actions, actual position changes, unchanged-position streaks, completed trades, win/loss counts, net returns, drawdown, fees and slippage.

## Data provenance qualification

The new holdout window must be disjoint from the prior June-October 2026 holdout and must not be part of the new model's training or validation splits. It is sourced from the same previously downloaded public history; it is therefore not an entirely new market sample at the research-history level. State this qualification in the report.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- No private exchange APIs, order endpoints, live orders, automatic promotion, or weakened CI/complexity gates.
