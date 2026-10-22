# Phase 5 Acceptance: Triple-Barrier and Meta-Labeling

Status: **IMPLEMENTATION CANDIDATE; EXACT-HEAD CI AND ARTIFACT REVIEW PENDING; ECONOMIC ACCEPTANCE NO-GO; NO TRADING**.

Canonical MRD: [docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md](AliX_MRD_Dynamic_Risk_FA.md), section 4.6 Plan B.
Canonical Roadmap: [docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md](AliX_Roadmap_Dynamic_Risk_FA.md), Phase 5.
Owner directive: [Phase 5 owner directive](../governance/owner-directives/2026-10-22-phase-5.md).
Architecture decision: [ADR-0005 Plan B](../architecture/decisions/ADR-0005-phase-5-plan-b.md).
Tracking issue: [#5](https://github.com/angry0tiger-sudo/AliX/issues/5).

## Implementation and acceptance checklist

### Method selection and contracts
- [x] Plan B selected for this research implementation in ADR-0005 because the exact upstream package/source/runtime contract is not yet established.
- [x] Barrier settings, timeframe, cost assumptions, label schema, meta model schema and feature names are explicit/versioned.
- [ ] Owner-reviewed final acceptance of ADR-0005 and model-risk scope.
- [ ] Any future production FreqAI lifecycle integration is separately tested; this research pipeline is not a registered production model.

### Triple-Barrier labels
- [x] Barriers use a historical volatility target and first-touch PT/SL or vertical expiry.
- [x] Same-candle PT/SL collision resolves conservatively to stop-loss.
- [x] Events are checked for timezone-aware, unique/increasing, gap-consistent candles and valid OHLCV.
- [x] Gross return, fees, slippage, net return, barrier label, and label-end index are emitted as outcome columns, not feature columns.
- [ ] Exact-head CI passes the full Phase 5 tests and synthetic mechanics run.

### OOF primary and Meta-Label model
- [x] Primary predictions are generated only on held-out chronological folds.
- [x] Training labels are purged when their label-end index is not strictly before the next test start.
- [x] Meta training validates OOF provenance and requires mature barrier outcomes before the evaluation cutoff.
- [x] Meta classifier uses versioned features and JSON-native save/reload, without pickle/joblib.
- [x] Invalid or unavailable model/input returns a false research-filter flag; output cannot create entry/exit or risk-approval fields.
- [ ] Exact-head CI proves integrated multi-fold evaluation and reload behavior.

### Evaluation and governance
- [x] Synthetic report implementation records dataset/config hashes, folds, OOS metrics, barrier outcomes, costs, runtime versions, and reload checks.
- [x] Evaluation compares Meta-Label scores with a historical-rate baseline on later folds.
- [ ] Inspect the published exact-head synthetic artifact and confirm its report commit matches the tested PR head.
- [ ] Evaluate approved historical data only after the owner supplies it and authorizes its use; no historical economic run is performed in this phase handoff.
- [ ] Independent economic/risk review and any pre-registered owner-approved acceptance thresholds remain open.

## Safety gates

- Synthetic/mock/offline only: no exchange or provider requests, external writes, notifications, or network changes.
- No real, dry-run, or paper trading session and no order submission/cancellation.
- Meta probabilities and research_filter_pass are not strategy signals, Risk Gate approval, position sizing authorization, or order permission.
- champion must remain null; promotion_status must remain BLOCKED_NO_ECONOMIC_GATE.
- Economic acceptance and trading authorization remain NO-GO regardless of CI results.

## Decision rules

- **GO for implementation review:** exact-head quality/runtime CI, focused tests, artifact verification, and documentation checks pass.
- **NO-GO for economic acceptance or Champion:** no approved historical dataset/hash and no pre-registered owner-approved economic thresholds.
- **NO-GO for all trading:** Phase 5 is research-only and does not change any trading authorization.
