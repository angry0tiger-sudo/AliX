# Owner Directive: Initial Stake Target of 10 USDT

- directive_id: ALIX-OD-2026-10-23-INITIAL-STAKE-10-USDT
- owner: angry0tiger-sudo
- timestamp: 2026-10-23
- status: owner-directed policy contract and tests in an implementation candidate branch
- implementation branch: feat/owner-initial-stake-10-usdt
- tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/19
- affected docs: docs/architecture/decisions/ADR-0012-initial-stake-target-and-risk-conflict.md
- affected components: risk/contracts.py; risk/allocator.py; risk/gate.py
- test requirements: target above/equal/below 10 USDT; fee/slippage reserve; amount precision; market min/max quantity/notional; insufficient risk budget and exposure; policy approval/revision/fingerprint mismatch; independent Risk Gate recomputation; no splitting or silent downsizing
- security impact: no order submission path or credential access is added
- financial impact: target size changes from dynamically computed size to an owner-mandated notional target; risk gates retain veto authority
- GO/NO-GO: this directive authorizes implementation and synthetic testing only. Economic acceptance and trading authorization remain NO-GO.

## Required semantics

1. Initial entry target is 10 USDT when free tradable balance is at least 10 USDT.
2. When free balance is below 10 USDT, the target is the whole available tradable amount after a conservative fee/slippage reserve. Quantity is rounded down to the exchange's actual amount step.
3. The target never overrides independent daily/per-trade loss budgets, max stake fraction, gross or correlated exposure, Kill Switch, reconciliation/freshness rules, market minimums/maximums or fee affordability.
4. If the target is unrepresentable at market precision, is below the real exchange minimum, exceeds the real market maximum, cannot cover costs, or conflicts with protective caps, reject with a stable reason. Do not silently reduce the 10 USDT target, split orders, or retry around exchange limits.
5. Both DynamicRiskAllocator and the independent RiskGate enforce the same immutable policy revision. The allocation carries policy revision/fingerprint; RiskGate recomputes expected target amount from current account, market, fees and market rules.
6. The model-facing proposal has no policy write capability. No LLM, strategy or training path can change this policy. Exits and REDUCE_RISK remain independent.
7. This is not an authorization for live orders. Tests use synthetic snapshots only and never claim real exchange order coverage.
