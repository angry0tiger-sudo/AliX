# Phase 7 Acceptance: Dynamic Risk Allocation, Risk Gate and Kill Switch

Status: IMPLEMENTATION CANDIDATE; IMPLEMENTATION SHA CI VERIFIED; RUNTIME INTEGRATION NO-GO; ECONOMIC ACCEPTANCE NO-GO; NO TRADING.

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

- [x] Implementation SHA `489ad189e92b90fd5b2f548d486d703c1a1e7a29` has successful Push CI run 426 and Pull Request CI run 427; job results/logs inspected ([Push](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830919965), [PR](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830927121)). A documentation-only commit changes the branch HEAD and must be checked separately before review readiness.
- [x] Risk package is included in compile, Ruff lint/format, strict Mypy, Bandit, Xenon/complexity and the existing coverage gate without lowering thresholds; all relevant steps passed and coverage is 90.68% against the unchanged 90% floor.
- [ ] Confidence/calibration, higher ATR, changed regime, uncertainty, exposure and correlation change risk budget or reject entry deterministically.
- [ ] Stake, stop, fee/slippage estimate, quantity step, price tick, min/max notional and free balance are independently validated by the Risk Gate.
- [ ] Missing or stale snapshot, unapproved policy/market rules, unknown regime, missing allocation, unknown Kill Switch state and service fault fail closed.
- [ ] Absolute caps remain independent of confidence; durable latch survives store re-instantiation and never auto-resets.
- [ ] Manual re-arm fails when approval, account/order reconciliation, supervisor health, policy consistency or version matching is absent.
- [ ] EXIT/REDUCE_RISK bypass entry denial, while order_permission remains false.
- [x] Fault-injection JSON records synthetic scenario results for implementation SHA `489ad189e92b90fd5b2f548d486d703c1a1e7a29`; all 28 scenarios passed and the artifact is attached to the PR run ([artifact](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830927121/artifacts/12449171737)).
- [ ] No real provider/API/exchange request, order submission/cancellation, notification, model/data download, paper/live/dry-run session or host-network change occurs.
- [ ] Paired walk-forward/OOS report covers drawdown, tail loss/CVaR, costs, turnover and regime stability against a simple baseline using owner-approved history and pre-registered economic thresholds.


## Verified implementation CI evidence

The following evidence is specific to implementation SHA `489ad189e92b90fd5b2f548d486d703c1a1e7a29` ([commit](https://github.com/angry0tiger-sudo/AliX/commit/489ad189e92b90fd5b2f548d486d703c1a1e7a29)).

- Push workflow run 426: [completed successfully](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830919965).
- Pull Request workflow run 427: [completed successfully](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830927121).
- Both runs' `quality` and `freqtrade-runtime-smoke` jobs succeeded.
- Ruff lint/format, strict Mypy, Bandit, dependency checks, architecture contracts and complexity gate all passed.
- The mock-only suite passed **159 tests** with **90.68% total coverage** (minimum gate remains 90%); 4 Phase 1 safety/dependency contract tests also passed.
- The synthetic Phase 7 fault-injection report records **28/28 passed** and is available as the [CI artifact](https://github.com/angry0tiger-sudo/AliX/actions/runs/39830927121/artifacts/12449171737).
- Runtime smoke kept the configuration closed; Phase 4/5 mechanics checks were synthetic. No order was submitted and no live trading was authorized.

A green implementation CI is evidence of tested mechanics, not production deployment, economic efficacy or permission to trade. The commit made to record this evidence creates a new branch HEAD; do not infer that new HEAD's CI result from these runs. The new HEAD requires its own successful Push and Pull Request workflows before the PR is marked ready for review.

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
