# AliX Knowledge Base

This folder is the entry point for authoritative project rules and implementation notes.

- [Project rules](PROJECT_RULES.md): owner authority, end-to-end phase gates, paper-only testing, and security boundaries.
- [Starter Kit archive](../../AliX_GitHub_Starter_Kit.zip): original archive retained unchanged at repository root. The archive contains its own MRD, roadmap, ADRs, provider registry, connectivity matrix, validator, and docs-quality workflow.
- [Phase 0 checklist](PHASE_0_ACCEPTANCE.md): evidence required before that phase was accepted.
- [Phase 1 checklist](PHASE_1_ACCEPTANCE.md): current quality and safety gates.
- [Novaryn source contract](../architecture/NOVARYN_REFERENCE.md): pinned source commit and selected contract mapping.
- [Phase 1 status report](../phase-reports/PHASE_1_STATUS.md): current candidate state and remaining gates.
- [Pinned direct dependencies](../../requirements.txt) and [transitive lock](../../requirements.lock): reproducible Python 3.12 dependency set.
- [Open tasks](OPEN_TASKS.md): explicit external and local-only blockers.
- [Phase 7 acceptance](PHASE_7_ACCEPTANCE.md): independent risk plane, dynamic sizing, kill switch and economic gates.
- [Phase 7 status report](../phase-reports/PHASE_7_STATUS.md): implementation evidence and NO-GO blockers.
- [Phase 7 risk halt/re-arm runbook](../runbook/PHASE_7_RISK_HALT_AND_REARM.md): incident and manual recovery procedure.

The original archive is retained unchanged. Extracted source documents should be promoted into the canonical `docs/` tree only after checksum and content review; do not silently overwrite existing material.
