# Strategy Registry and Selector Acceptance

Status: IMPLEMENTATION CANDIDATE / RESEARCH ONLY / NO-GO for economic acceptance and trading.

Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/17.
ADR: docs/architecture/decisions/ADR-0011-versioned-strategy-registry-and-selector.md.
Owner Directive: docs/governance/owner-directives/2026-10-23-strategy-registry-selection.md.
Governance blocker: https://github.com/angry0tiger-sudo/AliX/issues/16.

## Scope

Implement a pure, deterministic selector that returns a strategy proposal or NO_TRADE. No exchange calls, order submission, risk policy updates, Freqtrade signal activation, live credentials or promotion are in scope.

## Acceptance checklist

- [ ] Exact-head CI passes with unchanged quality, coverage, type, security, dependency, architecture, complexity and runtime gates.
- [x] Registry schema and immutable revision are versioned; malformed documents fail closed.
- [x] Registry entry pins code hash, lifecycle/owner approval, licenses, regime/timeframe, data/feature schema, requirements/tests, risk-profile reference, cost/liquidity limits and prohibited conditions.
- [x] Candidate scores rank only after data quality, market constraints, evidence lineage and economic eligibility pass.
- [x] OOS evidence is pinned to dataset/report/code/evaluation hashes and economic-policy revision; stale or mismatched evidence is rejected.
- [x] Minimum 100 trades, win rate >= 60%, absolute maximum drawdown <= 15%, and positive net expectancy are enforced from the supplied approved policy, not values self-declared by the evidence.
- [x] Fold/regime stability and complete sample accounting are required for selector eligibility.
- [x] Missing, invalid, degraded, future, duplicate or stale inputs return NO_TRADE with stable reasons and a reproducible fingerprint.
- [x] Deterministic tie-break by strategy ID/version.
- [x] order_permission always remains false.
- [x] Production registry is empty; no current strategy is implicitly promoted.
- [ ] Runtime integration with Trade Proposal and append-only Decision Ledger is implemented and tested.
- [x] Architecture test rejects direct imports of exchange/order/secret-capable and risk packages; runtime ACL isolation remains unproven.
- [ ] Reviewed merge through effective main branch protections.

## Current evidence and limits

The Phase 10 public research result had 16 trades and was marked INSUFFICIENT_SAMPLE. It does not satisfy the 100-trade minimum. The production registry is therefore deliberately empty. Unit fixtures prove mechanics only and are not market-performance evidence.

The selector output is not an order. Risk allocation, independent Risk Gate, Kill Switch, supervisor, order reconciliation and Freqtrade remain separate. Current release flags stay unchanged:

- Champion = null
- order_permission = false
- promotion_status = BLOCKED_NO_ECONOMIC_GATE
- economic acceptance = NO-GO
- trading authorization = NO-GO
