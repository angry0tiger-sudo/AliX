# ADR-039: Nobitex Holdout V2 Dual-Status Seal and Pre-Hack Research Authorization

- Status: Accepted
- Date: 2026-11-14 UTC
- Scope: Interpret the existing V2 commitment without modifying it; record separate owner authorization for ADR-038 pre-hack four-arm research only.
- Frozen main: 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb.
- Research branch: research/nobitex-hf-strategy.

## Decision 1 — preserve the dual-status commitment

The immutable HOLDOUT_COMMITMENT_V2.md contains two standalone status records:
- Line 3: status: PENDING_DISPATCH — the original commitment record.
- Line 112: status: FETCHED_LOCKED — the appended terminal acquisition record.

The posthack workflow collects every line beginning exactly with status: and requires the complete list to equal ["PENDING_DISPATCH"]. The actual list is ["PENDING_DISPATCH", "FETCHED_LOCKED"]; therefore the parser verdict is WINDOW_CONSUMED. The final recorded lifecycle state is FETCHED_LOCKED.

This is a deliberate fail-closed dual-status seal, not an invitation to rewrite history. V2 is not to be normalized, truncated, reordered, or otherwise modified. The original pending record and terminal fetched-locked record remain immutable evidence.

The acquisition manifest for Run 43443886254 records Spec v3, dataset SHA-256 435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee, 11,700 rows, zero missing timestamps, and dataset_persisted=false. This metadata does not release the dataset or authorize its consumption.

## Decision 2 — separate pre-hack owner authorization

Owner authorization ID: OWNER-2026-11-14-FOURARM.

Authorized scope:
- Acquire the exact pinned public model weights through GitHub Actions and verify each digest.
- Execute the twelve ADR-038 candidate cells on the first 27,825 rows of the immutable pre-hack CSV only.
- Preserve ALIX_DRY_RUN=true, ALIX_ALLOW_LIVE_TRADING=false, and every non-trading safety invariant.

This authorization does NOT authorize:
- Any post-hack holdout access, download, parsing, inference, or evaluation.
- Any change to V2, Spec v3, the public fetcher, the persisted pre-hack CSV, or frozen main.
- Paper trading, private API calls, real order attempts, Champion selection, or order permission.
- Any extra model, threshold, feature, or directional trial beyond the twelve registered cells.

The commit containing this ADR is the immutable authorization-recording commit. Its full Git SHA is copied into owner_authorization_commit_sha in the execution lock. The lock validator remains mandatory and must reject missing, malformed, mismatched, or tampered fields.

## Safety state

- Champion=null
- order_permission=false
- trading_authorization=NO-GO
- private_api_calls=0
- real_order_attempts=0
- posthack_holdout_access=FORBIDDEN
- main=FROZEN

A pre-hack result cannot open the holdout. Any future holdout release requires a distinct Owner Release Authorization.
