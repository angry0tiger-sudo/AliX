# Phase 6 Acceptance: External Data, Point-in-Time Features and Local NLP

Status: IMPLEMENTATION CI PASSED ON SHA 52bc49012a945bf6c3bb411b6a4d3933a9a69aa4 AND REPORT-SYNC CI PASSED ON SHA 416b46ebdbb59eb0f095fc7636368ca6b36de462; PR #8 MERGED BY OWNER; LIVE PROVIDERS AND MODEL LICENSES NOT APPROVED; ECONOMIC ACCEPTANCE NO-GO; NO TRADING.

Canonical MRD: docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md, sections 2.7 and 4.2-4.7.
Canonical Roadmap: docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md, Phase 6.
Owner Directive: docs/governance/owner-directives/2026-10-22-phase-6.md.
ADR: docs/architecture/decisions/ADR-0006-phase-6-external-data-nlp.md.
Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/7.
Tested implementation SHA: `52bc49012a945bf6c3bb411b6a4d3933a9a69aa4`.
Push CI: [run #402 / ID 39809700265](https://github.com/angry0tiger-sudo/AliX/actions/runs/39809700265) — success (Quality and Runtime).
PR CI: [run #403 / ID 39809712528](https://github.com/angry0tiger-sudo/AliX/actions/runs/39809712528) — success (Quality and Runtime).
Quality evidence: 131 passed, 91.18% coverage (90% minimum retained), 4 Phase 1 safety/dependency contracts passed, 13 warnings; the dependency audit found no known vulnerabilities.
Both runs target the same tested implementation SHA above. Report-sync CI also passed on SHA 416b46ebdbb59eb0f095fc7636368ca6b36de462; PR #8 was subsequently merged by the owner at 5873ba52ab827109a0be450670e971796295fa8e.

## Implementation scope

- Versioned provider and record contracts; injected RSS/Atom and JSON market-analytics adapters with no built-in network transport.
- Separate in-memory raw News Store, numeric External Feature Store and NLP Feature Store interfaces. Durable persistence is not included.
- Pre-transport source/license checks, stable reason-coded errors, timestamp/provenance validation and the shared Phase 3 external-record Data Quality Gate for numeric analytics.
- Revision-aware point-in-time join with explicit decision timestamp and time window; only the latest eligible revision can be selected for an event.
- Local Hugging Face adapter gated by local model files, SHA-256 hashes, immutable revision, model and dataset approvals and commercial-use approval. CI uses fake classifiers and temporary synthetic model/tokenizer bytes only.
- Structured NLP output remains uncalibrated by default and is never an order signal, risk approval or model promotion.

## Implementation acceptance checklist

- [x] CI passes on the exact final PR head and its logs are reviewed.
- [x] Compile, Ruff lint/format, strict Mypy, Bandit, dependency-lock verification, pip-audit, import architecture, Xenon, tests/coverage, Phase 1 safety contracts and docs validation pass without weakening existing gates.
- [x] RSS fixture path checks XML parse, unsafe/malformed XML, publication timestamp, duplicates, unmapped assets, unapproved sources, timeout/status/rate-limit and oversized responses.
- [x] Numeric analytics checks schema, finite value, timestamp provenance, source approval, rate-limit controls and shared Data Quality Gate.
- [x] Stores reject invalid/unapproved records, detect duplicates and reject conflicting revisions.
- [x] Point-in-time tests cover data before/after decision, late arrival, revisions, duplicate content, asset mapping, missing/naive timestamps, window bounds and future-addition invariance.
- [x] NLP tests cover manifest revision/hash/licensing, invalid input/output/confidence, timeout/unavailable behavior, local-only loading, artifact verification and lack of trading side effects.
- [x] Main trading configuration remains stopped, dry-run, zero trades/stake, blank credentials and empty whitelist; strategy, Risk Gate and Kill Switch are unchanged.
- [x] CI uses injected fake transports and fixture classifiers only. No API call, paid service, real model download, exchange request, notification or host networking change is made.

## External approval and research gates remain open

- [ ] At least one real numeric analytics source and one real news feed have provider-specific documented approval for the intended processing, retention and commercial use.
- [ ] Current provider schemas, tiers, quotas, retention and timestamp semantics are confirmed in an authorized review.
- [ ] The owner selects a specific NLP model/tokenizer revision after weight, training-dataset, news-source and dependency licenses are reviewed; hashes and attribution are approved.
- [ ] A permitted historical news/feature snapshot exists with its actual historical availability times. Current ingestion cannot substitute for historical availability.
- [ ] Owner-authorized labeled crypto-news data supports per-class F1, calibration, language/event slices and ablation against a no-news baseline.
- [ ] Durable stores, production monitoring, provider deployment and connectivity remain separate review gates.

## Required governance state

- Champion = null.
- promotion_status = BLOCKED_NO_ECONOMIC_GATE.
- Economic acceptance = NO-GO.
- Trading authorization = NO-GO.
- No order submit/cancel, exchange API call, live data download, paid API call, external notification, live/dry-run/paper session or host-network change is authorized or performed by Phase 6.

Implementation and report-sync CI gates passed before the owner merged PR #8 at 5873ba52ab827109a0be450670e971796295fa8e. External provider/model/data approval and economic acceptance remain NO-GO. This phase cannot activate providers or models, promote a Champion, authorize live connectivity or permit trading.
