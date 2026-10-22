# Phase 8 Tenant Isolation and Runtime Recovery

## Access invariants

- The authenticated principal establishes identity; tenant IDs in path/body are only resource selectors.
- Every read/write resolves the tenant and active membership in the same repository path that performs the operation.
- Platform-admin authority is available only to the local owner token in the single-user mode.
- Workers receive a random per-generation token. The database stores only its SHA-256 digest.
- A Worker token is valid only for its exact worker ID, current generation, active Tenant and START_REQUESTED lifecycle state.
- Worker identity is not a tenant credential. The internal heartbeat endpoint returns no account, secret, policy or other-worker data.

## Failure response

| Fault | Required action |
|---|---|
| Missing/invalid owner token | Reject request; do not expose resource existence. |
| Forged Tenant/account/Worker association | Reject request; no data returned or changed. |
| Membership revoked | Reject subsequent scoped requests immediately; no membership cache is used. |
| Duplicate Worker slot | Return conflict; retain existing slot and audit history. |
| Unexpired runtime-command lease owned by another dispatcher | Reject acknowledgement and do not mutate Worker state. |
| Expired command lease | Allow one dispatcher to claim it by conditional update. |
| Invalid or stale Worker token/generation | Reject heartbeat; no heartbeat timestamp update. |
| API/DB unavailable | Control-plane operation fails; never default to permissive identity or active Worker state. |
| Runtime start/stop failure | Quarantine Worker and revoke credential; manual inspection required. |
| SQLite identity mismatch | Worker exits without overwriting the existing identity record. |

## Operational boundary

This phase's Worker process is inert and does not place orders. These controls are implementation contracts and test evidence, not proof of OS security against a compromised host owner, Docker daemon or root process. Production Freqtrade wiring requires its own architecture and release gate.
