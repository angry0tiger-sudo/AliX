# Owner Directive: Adaptive Autonomous AI

Date: 2026-10-25

This directive supersedes fixed-parameter assumptions for the research policy on \`research/adaptive-autonomous-ai\`. It does not amend or activate the prior branch, production strategy registry, or execution path.

## Superseding research requirements

- The policy must choose among 1h, 4h and 1d contexts from current market conditions; no timeframe is hard-wired as the sole decision horizon.
- Per-trade risk budget and position quantity must vary with model score, trend strength and volatility rather than reuse one fixed percentage.
- A protective stop must incorporate both ATR and prior market structure; invalid or excessively wide stop geometry means no entry.
- The adaptive policy must not create a fixed take-profit. Trailing exits should allow stronger trends more room and tighten when momentum deteriorates.
- Every closed simulated trade must create a post-trade learning-ledger event. Bounded parameter changes may affect subsequent entries only.
- The policy must compare against fixed 1h, 4h and 1d research baselines on verified development data.

## Clarifications to prevent unsafe interpretation

“Autonomous” does not mean unlimited authority or unbounded self-modification. For v1, feedback remains deterministic, explainable, capped and versioned. The current stop/failure labels are hypotheses from observed trade fields, not proof of causation. Confidence is not a calibrated probability. The 0.10%-0.75% adaptive risk envelope and 95% notional cap are safety guardrails, not a constant trade risk. If the support-based stop would exceed the configured loss envelope, abstain instead of tightening it artificially.

The current prototype is an adaptive research policy, not a trained/promotion-ready ML model. Economic acceptance cannot be inferred from passing tests or from a relative improvement against a negative baseline.

## Immutable governance contract

- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- \`economic_acceptance = NO-GO\` until separately approved economic evidence exists
- no production promotion, production registry mutation, exchange client, private API, real order or live credentials
- no fabricated volume, spread, liquidity or order-book measurements
- final holdout reserved and unused
- no weakening quality gates to pass CI

Any conflicting legacy documentation remains in place for its original experiment; this directive governs only the new research branch until the owner explicitly authorizes a broader migration.
