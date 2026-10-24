# Owner Directive - AI Core Rebuild Research Branch

- directive_id: ALIX-OD-2026-10-24-AI-CORE-REBUILD
- owner: angry0tiger-sudo
- timestamp: 2026-10-24
- branch: research/ai-core-rebuild
- base_sha: 88375db74d1de00e47813f51805ea25015f8346e
- approval_status: research implementation authorized; production release not authorized.

## Scope
Implement a causal 4H feature schema, CPU-only three-model ensemble, causal HMM fold inference, three unapproved research strategy candidates, and optional point-in-time Persian RSS sentiment support. All work stays on the named research branch. Existing PRs and the production strategy registry remain unchanged.

## Explicit non-authorizations
- No Champion assignment, model promotion, production model update or strategy-registry update.
- No order/execute endpoints, private exchange APIs, live keys, live trading or generated paper fills.
- No weakening of existing Ruff, strict Mypy, Bandit, dependency audit/lock, import-linter, Xenon, coverage or runtime checks.
- No provider/data/model licence or technical approval is implied by code existence.
- GitHub Actions must not fetch market/provider data over the network. It may retrieve the immutable existing Stage A artifact via GitHub's artifact service. Cross-asset input is optional and must be supplied as an approved/hash-pinned dataset.
- NLP output is unavailable to model training until provider rights, retention, model hashes and calibration evidence are approved.

## Required tests and evidence
- 4H bars are exact groups of four contiguous hourly bars; gaps, partial bars and naive timestamps fail closed.
- Feature outputs are causal; cross-asset labels must align exactly after the explicitly recorded fixed timestamp-phase mapping.
- All three ensemble component models train per fold; weights sum to one; non-finite/mismatched feature and prediction data fail closed.
- HMM inference is forward-only and uses only training-fit parameters; every route remains research-only or NO_TRADE.
- RSS records enforce source approval, normalization, content hash and published/ingested/available/processed time contracts; no neutral imputation for unavailable sentiment.
- Exact-head CI and research workflow must pass before declaring the branch implementation validated.

## Impact / rollback
- **Risk:** medium/high research risk from model complexity and multiple comparisons; execution risk constrained by the absence of order capability in the new code.
- **Security:** CPU-only pinned estimator, no credentials, no external provider transport in the workflow; Bandit and dependency audit remain required.
- **Rollback:** revert the branch commits; do not merge this candidate into a release branch without a separate review.
- **Resulting GO/NO-GO:** implementation validation may become technically green, but economic acceptance and trading authorization remain **NO-GO**. Required immutable values: Champion = null, order_permission = false.
