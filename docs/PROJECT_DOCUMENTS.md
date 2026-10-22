# AliX canonical project documents

This index is the repository entry point for architecture, phase order, owner directives, and acceptance evidence. The stable canonical paths explicitly requested by the project owner are under `docs/knowledge_base/`.

## Canonical references

- [Master Requirements Document (MRD)](knowledge_base/AliX_MRD_Dynamic_Risk_FA.md): canonical stable path for architecture, boundaries, safety policy, model/data contracts, risk and governance requirements.
- [Project Roadmap](knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md): canonical stable path for phase order, phase deliverables, acceptance checklists, and phase-specific GO/NO-GO conditions.
- [Project Rules](knowledge_base/PROJECT_RULES.md): repository workflow and mandatory safety conventions.
- [Open tasks and deployment gates](knowledge_base/OPEN_TASKS.md): outstanding runtime and external evidence.
- [Phase 4 acceptance](knowledge_base/PHASE_4_ACCEPTANCE.md): versioned checklist for the current phase.
- [Phase 4 status report](phase-reports/PHASE_4_STATUS.md): current implementation state and evidence.
- [Phase 5 acceptance](knowledge_base/PHASE_5_ACCEPTANCE.md): Triple-Barrier and OOF Meta-Labeling contracts, tests, and release gates.
- [Phase 5 status report](phase-reports/PHASE_5_STATUS.md): implementation evidence, synthetic-only evaluation, and open gates.
- [ADR-0005: Phase 5 Plan B](architecture/decisions/ADR-0005-phase-5-plan-b.md): decision to use isolated research services pending exact DeepAlpha source/runtime proof.
- [Owner Directive: Phase 5](governance/owner-directives/2026-10-22-phase-5.md): explicit owner authorization and non-trading scope.
- [Owner Directive: Phase 4 baseline](governance/owner-directives/2026-10-22-phase-4.md): baseline scope and safety boundary.
- [Owner Directive: canonical knowledge-base paths](governance/owner-directives/2026-10-22-phase-4-knowledge-base-canonical-paths.md): explicit instruction establishing the stable document paths above.

## Phase 6: External Data and local NLP (offline only)

- [Phase 6 acceptance](knowledge_base/PHASE_6_ACCEPTANCE.md): provider contracts, point-in-time tests and external approval gates.
- [Phase 6 status report](phase-reports/PHASE_6_STATUS.md): exact-head CI evidence and unresolved blockers.
- [Data provider registry](data-provider-registry.md): source rights, rate limits, retention, timestamp semantics and activation status.
- [ADR-0006](architecture/decisions/ADR-0006-phase-6-external-data-nlp.md): contract-first external data and local NLP.
- [Owner Directive](governance/owner-directives/2026-10-22-phase-6.md): authorization and scope boundaries.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/7.

## Phase 7: Dynamic Risk Allocation, Risk Gate and Kill Switch

- [Phase 7 acceptance](knowledge_base/PHASE_7_ACCEPTANCE.md): risk-plane, failure and remaining release/economic gates.
- [Phase 7 status report](phase-reports/PHASE_7_STATUS.md): exact-head CI evidence and explicit blockers.
- [ADR-0007](architecture/decisions/ADR-0007-phase-7-risk-engine.md): independent risk-plane and latched halt design.
- [Owner Directive](governance/owner-directives/2026-10-22-phase-7.md): scope, fail-closed rules and non-trading authorization.
- [Risk halt/re-arm runbook](runbook/PHASE_7_RISK_HALT_AND_REARM.md): operator process for incident, reconciliation and manual re-arm.
- [Risk package](../risk/README.md): isolated contracts and implementation boundaries.

## Phase 8: Local-first Control Plane and isolated Worker runtime

- [Phase 8 acceptance](knowledge_base/PHASE_8_ACCEPTANCE.md): tenant authorization, local token mode, Worker isolation and exact-head CI gates.
- [Phase 8 status report](phase-reports/PHASE_8_STATUS.md): branch-specific implementation evidence and remaining runtime blockers.
- [ADR-0008](architecture/decisions/ADR-0008-phase-8-local-single-user-control-plane.md): PostgreSQL Control Plane, local owner token, per-Worker SQLite and isolated Containers.
- [Owner Directive](governance/owner-directives/2026-10-22-phase-8.md): explicit local-first choices, runtime limits and no-trading boundary.
- [Local Compose runbook](runbook/PHASE_8_LOCAL_COMPOSE.md): owner-controlled local services and host dispatcher.
- [Tenant isolation runbook](runbook/PHASE_8_TENANT_ISOLATION.md): authorization invariants and fail-closed recovery.

## Phase 9: End-to-end testing and final preparation

- [Phase 9 acceptance](knowledge_base/PHASE_9_ACCEPTANCE.md): roadmap checklist, evidence boundaries and remaining operational release gates.
- [Phase 9 status report](phase-reports/PHASE_9_STATUS.md): exact-SHA CI evidence, synthetic-only status, and blockers.
- [ADR-0009](architecture/decisions/ADR-0009-phase-9-synthetic-e2e-and-postgres-validation.md): synthetic E2E ledger and disposable PostgreSQL verification strategy.
- [Owner Directive](governance/owner-directives/2026-10-23-phase-9.md): GitHub-only workflow, no real orders/live execution connections, and preserved `order_permission=false`.
- [Windows/PostgreSQL runbook](runbook/PHASE_9_WINDOWS_POSTGRES_E2E.md): instructions prepared for future owner-run validation; not run by the assistant.
- [Synthetic E2E runner](../scripts/run_phase9_e2e.py): deterministic data-quality/model/risk/simulation-ledger/report mechanics; no exchange I/O.
- [Phase 9 tracking issue](https://github.com/angry0tiger-sudo/AliX/issues/11).

## Preserved compatibility copies

These earlier paths are retained so existing links and history are not broken. The contents are preserved; the canonical stable paths are the two copies under `knowledge_base/`.

- [MRD compatibility copy](architecture/AliX_MRD_Dynamic_Risk_FA.md)
- [Roadmap compatibility copy](roadmap/AliX_Roadmap_Dynamic_Risk_FA.md)

## Authority and change control

1. Explicit owner instructions establish the intended internal project direction. When an instruction conflicts with an existing repository document, record the instruction as a versioned Owner Directive and update the affected documents through a reviewed pull request.
2. This authority rule does not make a failed safety, security, legal, data-quality, model-evaluation, or release gate pass. Record each blocked gate and its evidence; do not silently skip it.
3. A phase is complete only when the roadmap acceptance criteria are evidenced and the phase report records an explicit GO, CONDITIONAL GO, or NO-GO decision.
4. CI with synthetic fixtures validates implementation mechanics only. It does not establish live-provider compatibility, historic-market performance, profitability, or authority to trade.
5. The Freqtrade order lifecycle remains the only owner of order execution. Phase 4 uses offline research and a signal-free strategy scaffold; model output is never order permission.
6. A Champion remains unset until owner-approved economic thresholds have been pre-registered and the permitted, versioned out-of-sample evidence passes every required gate.

## How these documents are kept discoverable

CI executes `scripts/validate_project_docs.py`, which requires both canonical MRD/roadmap copies under `docs/knowledge_base/` as well as the preserved compatibility paths, acceptance/status reports, and current Owner Directives. Every new phase must update this index/acceptance record as needed. Do not replace or rewrite the preserved Starter Kit archive as a shortcut for updating live repository documents.


## Phase 10: Scheduler, reporting and public-market research

- [Phase 10 acceptance](knowledge_base/PHASE_10_ACCEPTANCE.md): scheduler/reconciliation gates, public data research and safety invariants.
- [Phase 10 status](phase-reports/PHASE_10_STATUS.md): implementation/CI evidence and explicit blockers.
- [ADR-0010](architecture/decisions/ADR-0010-phase-10-reporting-scheduler-public-research.md): durable schedule state, read-only reporting and public-market research boundaries.
- [Owner Directive: Phase 10](governance/owner-directives/2026-10-23-phase-10.md): GitHub-only work, manual public data fetch, no order execution and no trading authorization.
- [GitHub Actions / Windows handover runbook](runbook/PHASE_10_GITHUB_ACTIONS_AND_WINDOWS_HANDOVER.md): future owner-run steps, prepared but not executed.
- [Phase 10 issue](https://github.com/angry0tiger-sudo/AliX/issues/13).
