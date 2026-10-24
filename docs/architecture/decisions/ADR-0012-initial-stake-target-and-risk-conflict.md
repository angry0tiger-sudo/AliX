# ADR-0012: Initial Stake Target and Protective-Risk Conflict

- Status: implementation candidate; not live authorization.
- Date: 2026-10-23.
- Owner Directive: docs/governance/owner-directives/2026-10-23-initial-stake-10-usdt.md.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/19.
- V1 scope issue: https://github.com/angry0tiger-sudo/AliX/issues/20.
- Base main SHA: 57103cf4cf7af720ec1166794df9dd62954cd38a.

## Decision

- Store a strict, versioned InitialStakePolicy document at config/risk/initial-stake-policy.json and parse it through risk/initial_stake_policy.py.
- If policy configuration is absent or invalid, the loader returns no policy; there is no fallback. The value object defaults to owner_approved=false. Any synthetic fixture must explicitly opt into an approved policy.
- For balance >= 10 USDT, target 10 USDT. For balance < 10 USDT, target floor-to-quantity-step of free balance divided by (1 + fee_rate + slippage_rate).
- Reject rather than silently shrink the target if it violates market min/max, is not representable at the current quantity step, cannot cover costs, or exceeds independently calculated loss/exposure/stake-fraction capacity.
- Bind each allocation to policy revision/fingerprint. RiskGate independently recomputes target and rejects policy or size mismatch.
- RiskSupervisor can pass a proposal to RiskGate, request correction to the owner target where all limits permit it, or reject. Its result always has order_permission=false.
- Allocator and RiskGate reject Short ENTRY under V1 Spot/USDT/Long-only. Exit and REDUCE_RISK paths must remain available.

## Trade-offs

A 10 USDT target with 10 USDT free can be rejected when reserved fees/slippage are not affordable. Below-target balance can also be rejected if existing caps or exchange minimums conflict. These rejections are intentional; implementation must not weaken protection to force an entry.

## Validation

CI must cover balance above/equal/below target; cost reserve; market minimum/maximum and precision; risk-budget/exposure conflict; policy approval/revision/fingerprint mismatch; independent RiskGate recalculation; Supervisor pass/resize/reject; Long-only rejection; and invalid policy failure. CI remains synthetic only; no live order or secret use is permitted.
