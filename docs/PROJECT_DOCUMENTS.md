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
