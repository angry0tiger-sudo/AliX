# ADR-0009: Synthetic End-to-End Verification and PostgreSQL Test Service

- Status: Accepted for Phase 9 implementation; full roadmap acceptance remains pending.
- Date: 2026-10-23.
- Owner Directive: docs/governance/owner-directives/2026-10-23-phase-9.md.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/11.
- Base after Phase 8 merge: b2ed5734523d560e30224ef7e1b93a4bfa308a6a.

## Context

Phase 8 established the local-first Control Plane and a heartbeat-only Worker. It did not connect the full research/risk path to a real Freqtrade trade database, did not implement a verified source-of-truth execution ledger, and did not exercise PostgreSQL or host-level isolation in the Phase 8 GitHub Actions workflow. Phase 9 must increase integration evidence without inferring that synthetic mechanics prove real market behaviour.

## Decisions

1. Add an offline Phase 9 runner that calls the existing candle/external-record/model-output quality gates, point-in-time news join, dynamic allocator, independent Risk Gate, synthetic simulation ledger and report renderer.
2. Add a dedicated append-only SQLite simulation ledger for deterministic Phase 9 tests. Its rows are explicitly simulation records, use a mandatory `order_permission=false`, have immutable record hashes, reject conflicting decision-ID replays and distinguish an unknown synthetic outcome that must not be retried.
3. The Phase 9 runner has no exchange client, Freqtrade order lifecycle, external network transport, secret loader or real order attempt. Reporting tests inject a fake sender and never contact Bale.
4. Add a dedicated GitHub Actions PostgreSQL service using pinned PostgreSQL 17.6 and synthetic credentials. Integration tests verify the PostgreSQL dialect, schema version, tenant/account/Worker repository lifecycle, audit records, duplicate-slot denial and concurrent slot contention. This ephemeral database is destroyed with the runner and is not production infrastructure.
5. Add `compose.phase9-test.yaml` as an explicitly test-only, synthetic PostgreSQL profile bound to loopback for a future owner-run Windows validation. Do not publish a port or otherwise weaken the production `compose.yaml`.
6. Keep all established quality gates and coverage thresholds unchanged. Phase 9 artifacts include the tested commit SHA and label their source/status clearly.
7. Keep reporting in the existing Reporting Engine boundary for now. A synthetic ledger snapshot is not the actual Freqtrade database and does not support an exchange-reconciled PnL claim.

## Consequences and explicit non-goals

- `order_permission` remains false, `Champion` remains null, and promotion/trading authorization remains `NO-GO`.
- A successful PostgreSQL CI job proves Control Plane behaviour against a real PostgreSQL server in a disposable test environment only. It does not prove Windows filesystem ACLs, Docker daemon security, OS isolation, remote provider compatibility or production operations.
- The simulation ledger does not implement the real order lifecycle, fills, cancels or recovery from an actual exchange timeout. Such scenarios can only be represented synthetically until a separately authorized, isolated non-trading contract environment and architecture are approved.
- Full Roadmap Phase 9 acceptance remains blocked until actual Freqtrade DB/Decision Ledger integration, source-of-truth reconciliation, historical out-of-sample report, performance/capacity tests, report delivery/retry evidence and security review are implemented and demonstrated without changing the trading policy.
