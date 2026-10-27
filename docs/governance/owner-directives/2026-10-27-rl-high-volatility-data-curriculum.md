# Owner Directive: Entropy Fix and High-Volatility Data Curriculum

**Date:** 2026-10-27  
**Branch:** `research/rl-trading-agent`  
**Scope:** Offline historical research only. This directive does not grant deployment or trading authority.

## Required experiment

1. Set both `--ent-coef` and `--exploration-ent-coef` to exactly `0.2` in the GitHub Actions research workflow. The generated report and workflow verifier must prove that the effective value is `0.2`.
2. Fit all observation scaling on the chronological training split only.
3. For the first requested 500,000 PPO environment steps, train on a deterministic 120,000-candle window selected only from the training split. Select candidates from the top quartile of realized one-minute volatility, then choose the candidate with the strongest absolute endpoint log return. Candidate windows are sampled at a one-day stride plus the final possible window.
4. Align the stage boundary upward to a complete PPO rollout. With 1,024-step rollouts, the configured 500,000-step boundary is expected to be 500,736 actual environment steps. Report the requested and actual boundary explicitly.
5. Continue the same model and optimizer state on the full chronological training split for the remaining steps. Keep validation and final holdout fully excluded from regime selection, fitting, and training. Preserve checkpoint diagnostics, economic evaluation, and all existing tests and quality gates.
6. The curriculum may increase market-regime exposure, but must never force an action or treat more trading as evidence of success. Judge the result by net performance, drawdown, costs, action distribution, and checkpoint stability.

## Immutable safety and quality requirements

`Champion = null`, `order_permission = false`, and `trading_authorization = NO-GO`. No live/paper execution API calls, no real orders, no automatic promotion, and no weakening of CI, coverage, architecture, complexity, or security gates.
