# ADR-0006: Contract-first external data and local NLP for Phase 6

- Status: Accepted for offline contract implementation only; production providers/model and economic acceptance remain NO-GO.
- Date: 2026-10-22.
- Owner authorization: docs/governance/owner-directives/2026-10-22-phase-6.md.
- Base commit: ce360a770586e76dc5a0f7b2421a371322e6d738.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/7.

## Context

The canonical MRD sections 2.7 and 4.2-4.7 and Roadmap Phase 6 require modular providers, rights/provenance records, separate numeric and text stores, a local NLP boundary, point-in-time semantics and abstention on invalid data. The repository had a Phase 3 external-record gate and an external_data README, but no provider adapters, stores, temporal feature join or NLP service.

The task prohibits real external API calls, paid provider use, downloads, exchange connectivity, notifications and all trading sessions. Actual provider terms and model/dataset licenses are not verified by this implementation.

## Decision

1. Create versioned provider and record contracts tested only with injected fixture transports. No concrete HTTP client or active provider is configured.
2. Keep raw text, numeric analytics and derived NLP features in separate store interfaces.
3. Reuse the Phase 3 external-record Data Quality Gate for numeric observations. News and NLP outputs also use strict schema, provenance, license, finite-value and timestamp checks.
4. Implement revision-aware as-of joining against available_at with an explicit window. Future, out-of-window, unmapped or unapproved records cannot affect a historical feature set.
5. Require pre-provisioned local model files, SHA-256 checks, immutable commit-style revision and documented approvals for model weights, training dataset and commercial use. The local loader uses local_files_only and trust_remote_code=False; no network fallback exists.
6. Keep sentiment uncalibrated unless a separately evidenced calibration process is reviewed. event_type remains unknown because no event classifier is selected. NLP produces research features only.
7. Keep inherited Transformers and DeepAlpha pins unchanged. Dependency cleanup requires a separate audited lock change.
8. Provider/model/history onboarding, durable stores, production monitoring and actual NLP ablation remain open gates.

## Consequences

- Contract mechanics can be verified without external credentials or networks.
- Passing mock tests does not approve any live provider or NLP model and does not prove commercial rights, model quality or market performance.
- Champion is null, promotion is blocked, and no trading authorization is granted.
- TradingView remains blocked for automated/non-display trading until explicit permission is documented.
