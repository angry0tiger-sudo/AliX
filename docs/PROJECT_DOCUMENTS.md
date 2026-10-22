# AliX canonical project documents

This index is the repository entry point for architecture, phase order, owner directives, and acceptance evidence. Review these documents before changing the phase scope or promoting models.

## Canonical references

- [Master Requirements Document (MRD)](architecture/AliX_MRD_Dynamic_Risk_FA.md): architecture, boundaries, safety policy, model/data contracts, risk and governance requirements.
- [Project Roadmap](roadmap/AliX_Roadmap_Dynamic_Risk_FA.md): phase order, phase deliverables, acceptance checklists, and phase-specific GO/NO-GO conditions.
- [Project Rules](knowledge_base/PROJECT_RULES.md): repository workflow and mandatory safety conventions.
- [Open tasks and deployment gates](knowledge_base/OPEN_TASKS.md): outstanding runtime and external evidence.
- [Phase 4 acceptance](knowledge_base/PHASE_4_ACCEPTANCE.md): versioned checklist for the current phase.
- [Phase 4 status report](phase-reports/PHASE_4_STATUS.md): current implementation state and evidence.
- [Owner Directive: Phase 4](governance/owner-directives/2026-10-22-phase-4.md): current owner instruction and required boundaries.

## Authority and change control

1. Explicit owner instructions establish the intended internal project direction. When an instruction conflicts with an existing repository document, record the instruction as a versioned Owner Directive and update the affected documents through a reviewed pull request.
2. This authority rule does not make a failed safety, security, legal, data-quality, model-evaluation, or release gate pass. Record each blocked gate and its evidence; do not silently skip it.
3. A phase is complete only when the roadmap acceptance criteria are evidenced and the phase report records an explicit GO, CONDITIONAL GO, or NO-GO decision.
4. CI with synthetic fixtures validates implementation mechanics only. It does not establish live-provider compatibility, historic-market performance, profitability, or authority to trade.
5. The Freqtrade order lifecycle remains the only owner of order execution. Phase 4 uses offline research and a signal-free strategy scaffold; model output is never order permission.
6. A Champion remains unset until owner-approved economic thresholds have been pre-registered and the permitted, versioned out-of-sample evidence passes every required gate.

## How these documents are kept discoverable

CI validates the presence of the MRD, roadmap, phase acceptance/status reports, and current Owner Directive. Every new phase must update this index/acceptance record as needed. Do not replace or rewrite the preserved Starter Kit archive as a shortcut for updating the live repository documents.
