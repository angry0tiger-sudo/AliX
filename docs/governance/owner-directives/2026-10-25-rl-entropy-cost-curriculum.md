# Owner Directive: RL Entropy and Accounting-Based Cost Curriculum

**Date:** 2026-10-25  
**Scope:** `research/rl-trading-agent` only  
**Authority:** Explicit project-owner instruction. Preserve the existing CI, static-analysis, security, typing, architecture, and Xenon gates.

## Required experiment

1. Set RecurrentPPO `ent_coef=0.01` for the first reproducible iteration. Record this and the full relevant model hyperparameters in the report. A future `0.05` experiment must be a separate run with a separate commit/seed record.
2. Apply an additional `0.001` reward penalty only when a selected BUY/SELL request is rejected by the pre-action legal mask. Keep this separate in code, per-step info, aggregate counts, and report from the inactivity penalty and profitable completed-trade bonus. An invalid request must never create a fill.
3. Begin training with fee and slippage at zero. Start a deterministic ramp only after at least 100 completed policy round trips, counted from real simulated fills and position accounting, not the number of BUY/SELL requests. Forced terminal liquidation does not count toward the threshold.
4. Ramp to the configured per-side fee/slippage over 100,000 environment steps, with deterministic updates at 1,000-step intervals. Report actual target costs, ramp start timestep, completed-trade count at transition, full-cost start timestep, steps/trades per stage, and whether target costs were experienced during training.
5. Validation, the final holdout, and all baselines must always use configured real costs. Training-only zero costs must never leak into evaluation.
6. Keep the one-million-step minimum, chronological split, one final holdout evaluation, data provenance disclosure, and strict CI gates. Report incomplete trade threshold/ramp explicitly; an incomplete curriculum must force economic `NO-GO`.

## Immutable safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- No private exchange APIs, order endpoints, live orders, automatic promotion, or weakening of quality gates.
