# ADR-0013: Risk Supervisor Entry Review Contract

- Status: implementation candidate; proposal-layer only.
- Date: 2026-10-23.
- Related MRD: ADR-007 (Risk Supervisor separate from model and release gate).
- Owner Directive: docs/governance/owner-directives/2026-10-23-initial-stake-10-usdt.md.
- Related issue: https://github.com/angry0tiger-sudo/AliX/issues/19.
- Base main SHA: 57103cf4cf7af720ec1166794df9dd62954cd38a.

## Contract

The existing persistent supervisor health check and Kill Switch latching remain intact. review_entry_proposal rechecks account/market/policy identity and freshness, the initial-stake target, market min/max, fee reserve, risk-budget room, max-stake fraction, gross/correlated exposure and V1 Long-only scope.

- PASS_TO_RISK_GATE: the existing proposal matches the owner target and current caps. This is not an order approval.
- RESIZE_REQUIRED: the requested allocation exceeds the target and the target itself remains inside all supervisor-computed caps. A revised proposal must pass RiskGate again.
- REJECT: any missing, stale, unreconciled, mismatched, unapproved, out-of-scope or cap-conflicting state.

## Boundaries

Every result sets order_permission=false. Supervisor never mutates RiskPolicy or InitialStakePolicy, accesses credentials, submits orders or automatically re-arms Kill Switch. Freqtrade remains the only order-lifecycle owner. Independent RiskGate remains mandatory and must recompute size, costs, market rules, account/order revisions and Kill Switch state.

V1 entry is Spot/USDT/Long-only. A Short entry is rejected by allocator and RiskGate; exit and REDUCE_RISK behavior must remain available.

## Validation

Synthetic-only tests must cover pass-to-Gate, target correction, cap conflict, degraded health, stale/mismatched revision, invalid policy, order-permission invariant and Short-entry rejection. Runtime orchestration into Freqtrade and the append-only Decision Ledger is not yet accepted.
