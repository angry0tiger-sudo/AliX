# ADR-032: Four-Arm Kronos / Chronos Experiment Preregistration

- **Status:** Accepted for preregistration only; experiment execution is not authorized.
- **Date:** 2026-11-13.
- **Active branch:** research/nobitex-hf-strategy.
- **Registration:** docs/research/preregistration/FOUR_ARM_KRONOS_CHRONOS.md.
- **Trial family:** NOBITEX-FOUR-ARM-KC-20261113.
- **Frozen main:** 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb (unchanged).
- **Governance reference:** Owner-cited MRD v0.9 §5, Column 2 (Trial Registry / committed holdout) and §13, Action Order, item 6.
- **Related decision:** ADR-0012, Pre-registration, Historical Trial Accounting and Holdout Integrity.

## Context

The Owner directed a four-arm comparison of the Nobitex Donchian-20 / ATR(14) / 2-ATR trailing-stop baseline against Kronos-Small, Chronos-2, and a combined model arm. The research must be preregistered before any training, fit, model comparison, or economic evaluation.

The historical trial registry records materially different trial units across V4-T, V4-AP, DEAP, AI-6 and HF research. It explicitly prohibits inventing a single global N by summing unlike quantities. GLOBAL_N therefore remains UNKNOWN pending complete candidate-level reconciliation.

ADR-0012 also records that the final 12.5% of the 31,801-row pre-hack dataset has prior AI-6 / DEAP exposure. That dataset can support the registered development and walk-forward comparison, but not a claim of an untouched final holdout.

## Decision

Accept and record exactly four planned arms:

1. A — Baseline: Donchian-20 + ATR(14) + 2-ATR trailing stop.
2. B — Baseline + Kronos-Small.
3. C — Baseline + Chronos-2.
4. D — Baseline + Kronos-Small + Chronos-2.

The registered dataset is only the fixed pre-hack Nobitex BTC/USDT 1h dataset (31,801 candles; SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e). No post-hack holdout access is permitted during this experiment.

The target, 24h horizon, triple-barrier labels, 24h purge, 24h embargo, train-only scaler, past-only as-of alignment, 512-candle model warm-up, 24h model refresh, minimum three chronological walk-forward folds, gates and Cost-50 / Cost-90 scenarios are fixed in the preregistration file.

The four planned trial IDs are registered before evaluation. Historical trials must be included in DSR accounting; GLOBAL_N remains UNKNOWN and must not be replaced by a guessed sum.

## Holdout rule — final holdout remains sealed

The post-hack holdout is a final governance gate, not a development or tuning dataset. It may be accessed only after:

1. at least one of the four arms has passed every preregistered gate on the permitted pre-hack evaluation; and
2. a separate, explicit Owner Release Authorization has been recorded.

A passing arm is eligible for a paper-only phase; it does not itself authorize accessing the final holdout, private API calls, or live orders. No arm pass means documented NO-GO and the holdout remains closed.

## Execution boundary

This ADR authorizes documentation and preregistration only. It does not authorize model loading, inference, training, fit, backtest, tuning, or economic evaluation. Exact model/weight hashes, model decision mapping, fold cut points, deterministic seed, label collision/price-anchor semantics, execution-cost decomposition, and DSR implementation must be locked in an immutable execution record before a later Owner run directive.

The existing final holdout, spec v3, V2 commitment, DISPATCH_TRIGGER, public fetcher, and frozen main are outside the scope of this ADR and remain unchanged.

## Safety invariants

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false

## Numbering note

The repository's existing ADR files at the base commit reached ADR-0017. The Owner explicitly assigned identifier ADR-032 for this preregistration. This record uses that requested identifier; no intermediate ADR files are inferred, generated, or renumbered.
