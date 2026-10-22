# ADR-0005: Use the isolated Phase 5 Plan B research pipeline

- **Status:** Accepted for Phase 5 research implementation; economic/production acceptance remains NO-GO
- **Date:** 2026-10-22
- **Owner authorization:** docs/governance/owner-directives/2026-10-22-phase-5.md
- **Scope:** Triple-Barrier labels, OOF primary signal provenance, Meta-Label filtering and offline walk-forward evaluation

## Context

The canonical MRD section 4.5 requires a source/license/runtime compatibility audit before accepting DeepAlpha. It calls out a mismatch between package and repository documentation and requires install, train/predict, reload, no-look-ahead and out-of-sample evidence. Section 4.6 describes Plan B: a custom FreqAI-compatible workflow or separated research packages, triple-barrier outcomes, OOF primary signals, a meta target that includes costs, purging/embargo, versioned artifacts, and fail-closed behavior.

The PyPI project page for deepalpha-freqai 1.1.4 describes an installable plugin and declares MIT licensing: https://pypi.org/project/deepalpha-freqai/. The linked upstream GitHub project describes a separate plugin installation/copy path: https://github.com/stefanoviana/deepalpha. Those descriptions alone do not establish that the exact 1.1.4 wheel, repository revision, dependency graph and Freqtrade 2026.7 integration path are equivalent or compatible. No claim of an upstream source-level audit or production compatibility is made by this ADR.

## Decision

Use the MRD Plan B as isolated services for this phase. Keep future-derived outcomes in the Triple-Barrier target table; never add them to the input feature matrix. Generate primary predictions only on held-out chronological folds. Remove training rows whose primary-label horizon reaches the next test interval. Fit the meta model only on earlier OOF records whose own barrier labels have matured before the test interval. Apply costs when determining binary meta outcomes and include simple baselines in each meta evaluation fold.

Use scikit-learn and NumPy/Pandas already pinned by the project for the research-only meta classifier, and serialize its standardized logistic parameters to versioned JSON rather than pickle/joblib. The model output is a probability and a research filter flag only.

No Freqtrade internal files are patched. This phase does not register the research model as a production FreqAI plugin and does not connect the filter output to a strategy signal or Risk Gate. Runtime/lifecycle integration remains an explicit future gate requiring its own contract and evidence.

## Consequences and unresolved dependency issue

- The Plan B code has no import-time or runtime dependency on DeepAlpha.
- The existing direct pin deepalpha-freqai==1.1.4 in the inherited Phase 4 requirements/lock is not removed by this ADR; it predates this phase. Removing it and regenerating/auditing the transitive lock is a separate open dependency task. Do not describe the package as removed from the environment or lock.
- Synthetic mechanics verify label ordering, OOF provenance, embargo/purge, costs, model save/reload and fail-closed output only.
- No trading, Champion promotion, real-market claim or economic acceptance follows from this decision.
