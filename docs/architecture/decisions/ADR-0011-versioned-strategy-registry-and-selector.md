# ADR-0011: Versioned Strategy Registry and Regime-Aware Selection

- Status: Proposed; research-only implementation candidate.
- Date: 2026-10-23.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/17.
- Related governance blocker: https://github.com/angry0tiger-sudo/AliX/issues/16.
- Owner Directive: docs/governance/owner-directives/2026-10-23-strategy-registry-selection.md.
- Base main SHA: 57103cf4cf7af720ec1166794df9dd62954cd38a.

## Context

AliXStrategy and AliXFreqAIBaselineStrategy intentionally emit no entry/exit signals. The repository has independent Data Quality and Risk Gate components, but it does not yet have a versioned, auditable strategy-selection boundary. The latest Phase 10 market-data research produced only 16 trades, below the pre-registered minimum of 100, and did not pass the economic gate. No production strategy is therefore selectable by default.

## Decisions

1. Keep strategy selection and regime inference in services/strategy_selection. The versioned HMM forward filter consumes point-in-time features only after strict Data Quality validation. It requires a pinned model hash, model freshness, compatible feature schema, valid transition probabilities and minimum posterior confidence/margin. It outputs a regime label or NO_TRADE, never an order. The selector then consumes immutable market/data-quality context, a registry snapshot, version-matched model rankings and OOS evidence. Neither module has exchange, order, secret or risk-policy mutation dependencies.
2. A registry entry pins strategy ID/version/code SHA-256, lifecycle, approved regime/timeframes, OHLCV/feature schema, required features/tests, prohibited conditions, risk-profile reference, liquidity/cost limits and separate strategy/data/dependency license states.
3. The runtime registry starts empty. Research-only, retired, unapproved or license-conditional entries cannot be selected. An entry is not activated by a model score or by adding code to the repository.
4. The pre-registered economic acceptance thresholds remain those documented by Phase 10: at least 100 OOS trades, win rate at least 60%, absolute maximum drawdown no more than 15%, and positive net expectancy after the evidence's fees/slippage assumptions. This ADR does not lower or replace that gate.
5. Beyond those global thresholds, selector eligibility requires at least three populated OOS folds with positive after-cost expectancy and drawdown within the approved maximum, plus complete regime evidence for every regime declared by the strategy. Each declared regime must have nonzero sample count, positive after-cost expectancy and drawdown within the approved maximum. These are conservative selection-robustness checks, not a claim that the economic acceptance gate has passed.
6. Evidence binds the exact strategy code hash, dataset hash, report hash, evaluation commit, feature schema, test IDs and economic-policy revision. Evidence age is checked against an explicit owner-approved policy; no implicit freshness TTL is invented by the selector.
7. Model scores only rank candidates after every deterministic eligibility gate passes. Ties are broken by strategy ID and version so repeated inputs produce the same result. Missing, stale, duplicated, mismatched or future-dated inputs result in NO_TRADE with reason codes and a fingerprint.
8. The selection result is a strategy proposal only. It always emits order_permission=false. Risk allocation, independent Risk Gate, Kill Switch, supervisor, Freqtrade order lifecycle and release authorization remain separate and unchanged.
9. The module does not enable AliXStrategy signals, load private credentials, call an exchange or promote a Champion. There is no production registry entry until authorized economic evidence and later release gates exist.

## Trade-offs

- Strict fold/regime checks can result in abstention even when aggregate metrics pass; this favors evidence stability over forcing a trade.
- Requiring lineage and test evidence adds registry maintenance, but removes implicit coupling between code, data, model score and stale performance summaries.
- The empty initial registry means the selector is mechanically testable but cannot select a real strategy. Connecting it to runtime is a separate, reviewable step after this contract is accepted.

## Validation and closure

- Unit tests use synthetic fixtures only and validate HMM transitions/posteriors, schema/data quality/freshness rejection, confidence/margin abstention, deterministic ordering, candidate rejection reasons, lineage, economic thresholds, licensing, regime compatibility and empty-registry abstention.
- The HMM implementation is an inference contract only. No fitted model parameters have been approved for production and no runtime wiring activates it. The production selector registry remains empty until separately approved OOS evidence and release authorization exist.
- CI must pass unchanged lint, type, security, dependency, architecture, complexity, coverage and Freqtrade-runtime gates on the exact candidate SHA.
- Successful mechanics CI is not economic acceptance, end-to-end order execution, or live authorization.
