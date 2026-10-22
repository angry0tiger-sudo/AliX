# Phase 7 Acceptance: Dynamic Risk Allocation, Risk Gate and Kill Switch

Status: IMPLEMENTATION CANDIDATE CREATED; EXACT-HEAD CI NOT YET VERIFIED; RUNTIME INTEGRATION NO-GO; ECONOMIC ACCEPTANCE NO-GO; NO TRADING.

Canonical MRD: docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md.
Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 7.
Owner Directive: docs/governance/owner-directives/2026-10-22-phase-7.md.
ADR: docs/architecture/decisions/ADR-0007-phase-7-risk-engine.md.
Runbook: docs/runbook/PHASE_7_RISK_HALT_AND_REARM.md.
Base commit: 5873ba52ab827109a0be450670e971796295fa8e.
Work branch: feat/phase-7-risk-engine.

## Scope

- Versioned immutable policy and reconciled account/market/market-rule contracts.
- Deterministic allocator using ATR, regime multipliers, calibrated confidence, uncertainty, calibration error, after-cost edge, remaining daily/absolute risk capacity, free balance, exposure and correlation.
- Independent Risk Gate re-computing loss at stop with fees and slippage, market precision, stake, stop validity, freshness, account reconciliation and cap room.
- Durable SQLite Kill Switch with latched-on-unknown initialization, fail-closed read/write errors, versioned audit events and explicit manual re-arm only.
- Independent supervisor for absolute loss/exposure caps, data/reconciliation health, ledger health, risk-service health and heartbeat.
- Entry-only rejection behavior; EXIT and REDUCE_RISK intents are not denied by the entry gate.
- Deterministic OOS metric calculator and synthetic fault-injection report. No execution or exchange integration is added.

## Implementation and evidence checklist

- [ ] Exact final PR-head Push and Pull Request workflows succeed; logs are inspected.
- [ ] Risk package is included in compile, Ruff lint/format, strict Mypy, Bandit, Xenon and the existing coverage gate without lowering thresholds.
- [ ] Confidence/calibration, higher ATR, changed regime, uncertainty, exposure and correlation change risk budget or reject entry deterministically.
- [ ] Stake, stop, fee/slippage estimate, quantity step, price tick, min/max notional and free balance are independently validated by the Risk Gate.
- [ ] Missing or stale snapshot, unapproved policy/market rules, unknown regime, missing allocation, unknown Kill Switch state and service fault fail closed.
- [ ] Absolute caps remain independent of confidence; durable latch survives store re-instantiation and never auto-resets.
- [ ] Manual re-arm fails when approval, account/order reconciliation, supervisor health, policy consistency or version matching is absent.
- [ ] EXIT/REDUCE_RISK bypass entry denial, while order_permission remains false.
- [ ] Fault-injection JSON records each scenario's pass/fail on the exact commit SHA and is attached to CI.
- [ ] No real provider/API/exchange request, order submission/cancellation, notification, model/data download, paper/live/dry-run session or host-network change occurs.
- [ ] Paired walk-forward/OOS report covers drawdown, tail loss/CVaR, costs, turnover and regime stability against a simple baseline using owner-approved history and pre-registered economic thresholds.

## Explicitly open gates

- [ ] Reconcile the contract with the actual authoritative account, trade/fill, fee and pending-order sources. This phase uses injected synthetic snapshots, not exchange/account truth.
- [ ] Deploy and test process isolation and operating-system ACLs so the model/NLP identity has no writable access to policy or Kill Switch storage.
- [ ] Connect supervisor heartbeat/ledger and kill-switch reads to a separately managed runtime service; this candidate contains contracts and local store mechanics only.
- [ ] Run paired OOS evaluation on an authorized historical snapshot with documented hash, chronology, benchmark and owner-approved acceptance thresholds. No permitted source/threshold is currently evidenced.
- [ ] Review position-management semantics for every supported market/instrument before any future integration.

## Governance

Champion = null.
promotion_status = BLOCKED_NO_ECONOMIC_GATE.
Economic acceptance = NO-GO.
Trading authorization = NO-GO.

A green synthetic CI proves implementation mechanics only. It does not prove profitability, a safe live deployment, a reconciled account source, legal/data approvals or permission to trade.
