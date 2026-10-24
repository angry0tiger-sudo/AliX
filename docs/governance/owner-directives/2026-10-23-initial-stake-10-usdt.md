# Owner Directive: Initial Stake Target of 10 USDT

- directive_id: ALIX-OD-2026-10-23-INITIAL-STAKE-10-USDT
- owner: angry0tiger-sudo
- timestamp: 2026-10-23
- status: implementation candidate in PR #21; not live authorization
- branch: feat/owner-initial-stake-10-usdt
- tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/19
- V1 scope issue: https://github.com/angry0tiger-sudo/AliX/issues/20
- governance blocker: https://github.com/angry0tiger-sudo/AliX/issues/16
- affected components: config/risk/initial-stake-policy.json; risk/contracts.py; risk/initial_stake_policy.py; risk/allocator.py; risk/gate.py; risk/supervisor.py
- approval status: policy is owner-directed, but CI/review and runtime integration remain pending
- release status: economic acceptance and trading authorization remain NO-GO

## Required semantics

1. Initial ENTRY target is 10 USDT when free tradable balance is at least 10 USDT.
2. When free balance is below 10 USDT, target the available tradable balance after conservative fee/slippage reserve, then round down to the actual market quantity step.
3. This target does not override per-trade/daily loss budgets, max-stake fraction, gross/correlated exposure, Kill Switch, freshness/reconciliation, market minimum/maximum, amount precision or cost affordability.
4. If the target cannot be represented, violates market limits, cannot cover costs, or conflicts with protective caps, reject. Do not silently downsize, split orders or retry around exchange limits.
5. DynamicRiskAllocator and independent RiskGate enforce the same immutable policy revision/fingerprint. The allocation carries policy identity; the Risk Gate independently recomputes expected target.
6. InitialStakePolicy defaults to unapproved. The strict versioned config loader returns no policy on missing/invalid documents; it never substitutes an approved default.
7. RiskSupervisor.review_entry_proposal may pass an allocation to RiskGate, request correction to the owner target if caps permit, or reject. It never grants order_permission or auto-rearms Kill Switch.
8. V1 ENTRY is Spot/USDT/Long-only. Allocator and Risk Gate reject Short entry, without disabling exits/REDUCE_RISK for existing state.
9. No real exchange write, secret use or live-order testing is authorized by this directive.
