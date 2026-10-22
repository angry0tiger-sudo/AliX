# Phase 9 Acceptance: End-to-End Testing and Final Preparation

Status: IN PROGRESS; no phase-completion claim is made until exact-head CI and all required evidence are reviewed.

- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/11.
- Owner Directive: docs/governance/owner-directives/2026-10-23-phase-9.md.
- ADR: docs/architecture/decisions/ADR-0009-phase-9-synthetic-e2e-and-postgres-validation.md.
- Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 9.
- Phase 8 merge base: b2ed5734523d560e30224ef7e1b93a4bfa308a6a.

## Roadmap acceptance checklist

- [x] Implemented synthetic E2E and contract suites pass in CI on exact source SHA `4c9168186d9da69235174a8dd15e5fa1ecdf6f7b`; the complete roadmap matrix remains open where unimplemented real-execution/reporting requirements are explicitly listed below.
- [ ] Entry/no-entry, exit/risk reduction, dynamic stop, stale/conflicting news, model unavailable, gap, market halt and order-timeout semantics have explicit tests. Current synthetic runner covers baseline entry, hold/no-entry, stale candles, candle gaps, stale/conflicting news, future external availability, unavailable model, ATR spike, latched Kill Switch and synthetic unknown outcome; real order/cancel/partial-fill paths remain unimplemented and blocked.
- [ ] Dynamic sizing and stop behaviour are evaluated across low/high volatility, ATR spikes, uncertainty/calibration degradation, and correlated exposure without changing the absolute cap. Existing risk unit tests and the new synthetic runner are implementation mechanics only; historical economics remain unverified.
- [ ] Final backtest inputs, dependency/model revisions, configuration and data hashes are recorded; owner-approved economic criteria are preregistered before evaluating performance.
- [ ] No timeout/restart path can blindly re-submit an unknown real order. Current tests exercise no-retry ledger semantics synthetically only; no exchange submit or reconciliation endpoint exists in this phase.
- [ ] Prediction latency, news queue lag, PostgreSQL write throughput, worker restart time and provider rate limits have measured, reproducible capacity results.
- [ ] Security audit covers secrets, permissions, dependency vulnerabilities, log redaction and retention; no unresolved high severity finding exists without documented mitigation/owner risk acceptance.
- [ ] Reporting consumes the actual Freqtrade database and a durable Decision Ledger, reconciles source-of-truth positions/fills/fees, calculates realized/unrealized/net PnL, equity/drawdown, exposure, open orders and Risk Gate/Kill Switch/Data Quality/model/connectivity state.
- [ ] Report identifiers, data cutoff, snapshot hash, generation timestamp, delivery status, versioned plots, timezone, bounded retry, idempotency, text fallback and redaction are tested with source evidence.
- [ ] Mismatch between report and the authoritative source of truth marks the report `DEGRADED`; unverified data is never presented as confirmed PnL.
- [x] Real PostgreSQL Control Plane integration and concurrent duplicate Worker-slot contention pass in the disposable GitHub Actions PostgreSQL service (2 tests passed in each Push/PR run).
- [ ] The owner-controlled Windows/PostgreSQL test runbook is reviewed. This runbook is prepared as documentation only; the assistant does not execute it on the owner's machine.
- [x] No real order, exchange write, trading connection, live/paper session, or real API credential is used; CI report confirms zero exchange calls and zero real-order attempts.
- [x] `order_permission = false`, `Champion = null`, `promotion_status = BLOCKED_NO_ECONOMIC_GATE`, economic acceptance `NO-GO`, and trading authorization `NO-GO` are preserved in the synthetic E2E report.

## Verified implementation-source CI evidence

Source commit under test: `4c9168186d9da69235174a8dd15e5fa1ecdf6f7b`.

- Push CI: [run 39859073512](https://github.com/angry0tiger-sudo/AliX/actions/runs/39859073512) — all three required jobs succeeded on this SHA.
- PR CI: [run 39859084239](https://github.com/angry0tiger-sudo/AliX/actions/runs/39859084239) — all three required jobs succeeded on this SHA.
- Quality: Ruff lint/format, strict Mypy (75 source files), Bandit, dependency lock/audit, architecture contracts and Xenon all passed.
- Unit tests and coverage: **200 passed, 2 skipped, 13 warnings; 90.65% coverage** against the unchanged 90% minimum.
- PostgreSQL integration: **2 passed**, including a real disposable PostgreSQL service and concurrent duplicate Worker-slot test.
- Synthetic E2E: 11 scenarios; expected status matrix passed; report artifact `phase9-synthetic-e2e-report` uploaded by the Push run. The ledger records 11 synthetic rows with 1 simulation, 8 rejections, 1 not-simulated decision and 1 unknown/no-retry outcome. It is explicitly `SYNTHETIC_ONLY` and `NOT_RECONCILED`.
- Freqtrade runtime smoke, Phase 1 safety contracts (4 passed), and Starter Kit/documentation validation passed.

These links prove the implementation source SHA above only. The docs commit has its own SHA and must independently pass exact-head Push and PR CI. Full Phase 9 acceptance remains IN PROGRESS/NO-GO because actual Freqtrade database/Decision Ledger reconciliation, source-of-truth PnL, historical economic acceptance, capacity benchmarks, report-delivery retry/plots, and owner-host OS/filesystem isolation remain unverified.

## Evidence boundary

A synthetic result must be tagged `SYNTHETIC_ONLY`; a synthetic ledger must state `NOT_RECONCILED`. Green CI proves only the tested code/configuration and ephemeral services. It does not prove real-market profitability, exchange compatibility, actual Freqtrade DB integration, owner-host filesystem/OS isolation, or production readiness.

## Current explicit blockers

- The new Phase 9 scenario runner/ledger and PostgreSQL integration job have not yet completed exact-head CI at this revision.
- Existing `services/reporting/engine.py` renders generic reports; it is not yet connected to the actual Freqtrade DB and source-of-truth Decision Ledger.
- No authoritative exchange reconciliation is performed. No historical OOS evidence has been accepted under preregistered owner-approved economic gates.
- Real order timeout, partial-fill, cancel and restart recovery are not implemented or exercised because the owner forbids real orders and no exchange write/execute path is authorized.
- Capacity benchmarks, actual owner-host Docker/OS isolation, local Windows filesystem permissions and production PostgreSQL role/backup/migration drills remain external release gates.
