# Owner Directive: Technical-Only Approval of Pinned AI Artifacts

- directive_id: ALIX-OD-2026-10-23-TECHNICAL-APPROVAL-RUN-39976295482
- owner: angry0tiger-sudo
- timestamp: 2026-10-23T17:26:00Z
- source_run: https://github.com/angry0tiger-sudo/AliX/actions/runs/39976295482
- source_commit_sha: 5a74ad4585313a2490416a0c3f83947e603c4738
- dataset_sha256: 177e041034fcd91a57bcee59edf6762fcdec85e7a2f4e8b27951f664d9a88856
- approval_status: OWNER_APPROVED_TECHNICAL_ONLY; economic acceptance remains NO-GO
- affected_components: approval records, strategy Meta-Model scoring, technical artifact promotion, shadow-only validation
- related_PR: recorded in the accepted PR and merge commit

## Scope

The Owner explicitly authorized review and technical-only approval of the exact artifacts from Run #39976295482 after the public-data, hash, loader, HMM holdout, Meta-Label validation, and fail-closed checks were examined.

The approved HMM artifact is a deterministic technical-only promotion of the source artifact: its nested model owner_approved flag is set to true, the model digest is recalculated, and the approved payload is loaded only if that digest matches the separate source-controlled approval record. The source training artifact and training report remain unchanged and unapproved.

- Source HMM model digest: e907a1adcad8639cf0fd77bef36adea1999feff574a812eedcaf4ad44e7926fd
- Technical-only HMM model digest: c3735b5ce711ecc490c85d75db78d6347900116117f5ad148db7a58c6e7e5329
- Meta-Model digest: e9dfc1006d312ca6361232599af0f7d47d9dbe217b473cd308a4b6689e7cb9a0

## Explicit exclusions and safety

This directive authorizes technical inference and shadow-only validation for these exact, hash-pinned artifacts. It does not accept economic performance, promote either model to Champion, make a strategy selectable, authorize an order, or authorize live trading.

The Meta-Model holdout recorded 112 test events, zero events passing the 0.70 research threshold, and negative mean net return across all proposals. Accordingly, the strategy registry remains empty until separate strategy-level OOS evidence passes the repository's pre-registered economic policy.

- Champion: null
- order_permission: false
- economic_acceptance: NO-GO
- trading_authorization: NO-GO
- live trading: prohibited

## Required verification

- Verify the source bundle, exact dataset and artifact hashes, original unapproved HMM load, promoted technical-only HMM load, and Meta-Model approval identity.
- Run point-in-time inference using fresh public-only OHLCV after the model's training timestamp.
- Require missing, invalid, or stale model/market data to fail closed.
- Verify an empty strategy registry produces no entry candidate.
- Verify the Shadow Exchange denied-risk path produces a REJECTED synthetic record with zero quantity and zero PnL.
- Preserve all current quality gates, including minimum 90% coverage.
