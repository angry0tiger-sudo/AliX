# Phase 9 Acceptance: End-to-End Testing and Final Preparation

Status: IN PROGRESS; no phase-completion claim is made until exact-head CI and all required evidence are reviewed.

- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/11.
- Owner Directive: docs/governance/owner-directives/2026-10-23-phase-9.md.
- ADR: docs/architecture/decisions/ADR-0009-phase-9-synthetic-e2e-and-postgres-validation.md.
- Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 9.
- Phase 8 merge base: b2ed5734523d560e30224ef7e1b93a4bfa308a6a.

## Roadmap acceptance checklist

- [ ] All required end-to-end tests and contract tests pass in CI on the exact candidate SHA.
- [ ] Entry/no-entry, exit/risk reduction, dynamic stop, stale/conflicting news, model unavailable, gap, market halt and order-timeout semantics have explicit tests. Current synthetic runner covers baseline entry, hold/no-entry, stale candles, candle gaps, stale/conflicting news, future external availability, unavailable model, ATR spike, latched Kill Switch and synthetic unknown outcome; real order/cancel/partial-fill paths remain unimplemented and blocked.
- [ ] Dynamic sizing and stop behaviour are evaluated across low/high volatility, ATR spikes, uncertainty/calibration degradation, and correlated exposure without changing the absolute cap. Existing risk unit tests and the new synthetic runner are implementation mechanics only; historical economics remain unverified.
- [ ] Final backtest inputs, dependency/model revisions, configuration and data hashes are recorded; owner-approved economic criteria are preregistered before evaluating performance.
- [ ] No timeout/restart path can blindly re-submit an unknown real order. Current tests exercise no-retry ledger semantics synthetically only; no exchange submit or reconciliation endpoint exists in this phase.
- [ ] Prediction latency, news queue lag, PostgreSQL write throughput, worker restart time and provider rate limits have measured, reproducible capacity results.
- [ ] Security audit covers secrets, permissions, dependency vulnerabilities, log redaction and retention; no unresolved high severity finding exists without documented mitigation/owner risk acceptance.
- [ ] Reporting consumes the actual Freqtrade database and a durable Decision Ledger, reconciles source-of-truth positions/fills/fees, calculates realized/unrealized/net PnL, equity/drawdown, exposure, open orders and Risk Gate/Kill Switch/Data Quality/model/connectivity state.
- [ ] Report identifiers, data cutoff, snapshot hash, generation timestamp, delivery status, versioned plots, timezone, bounded retry, idempotency, text fallback and redaction are tested with source evidence.
- [ ] Mismatch between report and the authoritative source of truth marks the report `DEGRADED`; unverified data is never presented as confirmed PnL.
- [ ] Real PostgreSQL Control Plane integration and concurrent duplicate Worker-slot contention pass in the disposable GitHub Actions PostgreSQL service.
- [ ] The owner-controlled Windows/PostgreSQL test runbook is reviewed. This runbook is prepared as documentation only; the assistant does not execute it on the owner's machine.
- [ ] No real order, exchange write, trading connection, live/paper session, or real API credential is used; no actual trade order is attempted.
- [ ] `order_permission = false`, `Champion = null`, `promotion_status = BLOCKED_NO_ECONOMIC_GATE`, economic acceptance `NO-GO`, and trading authorization `NO-GO` are preserved.

## Evidence boundary

A synthetic result must be tagged `SYNTHETIC_ONLY`; a synthetic ledger must state `NOT_RECONCILED`. Green CI proves only the tested code/configuration and ephemeral services. It does not prove real-market profitability, exchange compatibility, actual Freqtrade DB integration, owner-host filesystem/OS isolation, or production readiness.

## Current explicit blockers

- The new Phase 9 scenario runner/ledger and PostgreSQL integration job have not yet completed exact-head CI at this revision.
- Existing `services/reporting/engine.py` renders generic reports; it is not yet connected to the actual Freqtrade DB and source-of-truth Decision Ledger.
- No authoritative exchange reconciliation is performed. No historical OOS evidence has been accepted under preregistered owner-approved economic gates.
- Real order timeout, partial-fill, cancel and restart recovery are not implemented or exercised because the owner forbids real orders and no exchange write/execute path is authorized.
- Capacity benchmarks, actual owner-host Docker/OS isolation, local Windows filesystem permissions and production PostgreSQL role/backup/migration drills remain external release gates.
