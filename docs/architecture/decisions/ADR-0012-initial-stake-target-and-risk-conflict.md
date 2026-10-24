# ADR-0012: Initial Stake Target and Protective-Risk Conflict

- Status: Implementation candidate; owner-directed policy, not live authorization.
- Date: 2026-10-23.
- Owner Directive: docs/governance/owner-directives/2026-10-23-initial-stake-10-usdt.md.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/19.
- Base commit: 57103cf4cf7af720ec1166794df9dd62954cd38a.

## Context

The owner requires an initial-entry target of 10 USDT, or the full available tradable balance after a conservative cost reserve when free balance is below 10 USDT. This conflicts with the generic allocator's former behavior of deriving stake directly from risk budget and silently reducing quantity to fit maximum notional.

## Decision

- Represent the directive as frozen, versioned InitialStakePolicy nested in immutable RiskPolicy. Include a stable fingerprint and a directive ID.
- For balance >= 10 USDT, target 10 USDT. For balance < 10 USDT, target floor-to-quantity-step of free balance divided by (1 + fee_rate + slippage_rate).
- Reject rather than silently shrink a target if it violates market min/max, isn't representable at the current quantity step, cannot afford conservative costs, or exceeds the quantity bounded by independent loss/exposure/stake-fraction limits.
- Do not split orders or retry around min/max restrictions.
- Bind each allocation to the stake-policy revision/fingerprint. RiskGate independently recomputes expected target and rejects mismatch or unapproved/mismatched policy.
- Existing risk budgets, max-stake fraction, exposure limits, freshness and reconciliation checks remain unchanged and retain veto priority.
- No order endpoint or exchange client is added. Only Freqtrade may later execute an approved Trade Proposal under separately approved release controls.

## Trade-offs

- A 10 USDT target with only 10 USDT free may be rejected because fees/slippage reserve cannot be afforded. This is intentional: costs cannot be wished away by a spreadsheet.
- Below-target balance may still produce NO-GO if the existing max-stake fraction or risk caps conflict with using the full balance. This ADR deliberately does not weaken protective caps.
- Quantity precision may make an exact 10 USDT notional impossible; the allocator rejects instead of rounding down to an undersized trade.
- The rule applies to initial ENTRY sizing only; existing exits and risk-reduction paths remain independently available.

## Validation

CI must cover balances above/equal/below 10 USDT, fee affordability, precision/min/max market rules, conflict with risk budget/exposure, stale/unapproved/mismatched policy and independent Gate recalculation. No real credentials or live order calls are permitted in CI.
