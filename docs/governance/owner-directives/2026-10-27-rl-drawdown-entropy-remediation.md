# Owner Directive: Drawdown, Transaction-Cost, and Entropy Remediation

**Date:** 2026-10-27  
**Branch:** `research/rl-trading-agent`  
**Scope:** New historical-simulation-only research run following Run #133.

## Required experiment

1. Reduce `drawdown_increment_penalty` from 5.0 to 1.5. Keep the separate drawdown-level penalty and all risk/accounting mechanics intact.
2. Reduce `transaction_cost_reward_multiplier` from 2.0 to 1.0. Do not reduce or omit actual simulated fees or slippage: configured 25 bps fee and 20 bps slippage per side remain applied from the first training step.
3. Set both entropy-coefficient schedule endpoints to 0.2 so effective PPO `ent_coef` remains constant at 0.2 from the first rollout onward. Report the effective coefficient and schedule explicitly.
4. Diagnose Run #133 checkpoints at 100k-step intervals, especially 300k and 400k, on its exact Validation split. Record deterministic action counts, mean action probabilities, policy entropy, HOLD share, and the first checkpoint crossover. Do not claim an exact unsaved training timestep and do not recompute any final-Holdout score during this audit.
5. Train one new 1,000,000-step recurrent PPO experiment through GitHub Actions only. Evaluate every new saved checkpoint on Validation only and evaluate the current final Holdout once using the existing research workflow. Do not use Validation or Holdout to select a checkpoint.

## Hard invariants

- `Champion = null`, `order_permission = false`, and `trading_authorization = NO-GO`.
- `private_api_calls = 0`, `orders_or_execution_api_used = false`, and `real_order_attempts = 0`.
- Model output remains distinct from order permission; production promotion remains prohibited.
- No CI, lint, typing, security, dependency, architecture, complexity, coverage, or test gate may be weakened.
- A technically successful run is not evidence of economic acceptability. Negative validation return or drawdown above the existing 15% acceptance limit remains NO-GO.
