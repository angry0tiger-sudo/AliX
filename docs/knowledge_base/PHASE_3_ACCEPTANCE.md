# Phase 3 Acceptance: Data Quality Gate and Connectivity Manager

Status: **IMPLEMENTATION CANDIDATE; final status depends on CI for the candidate SHA and owner review**.

Owner directive: enter Phase 3 after Phase 2 PR #1 was merged into `main`.

## Data Quality Gate

- [ ] OHLCV has a versioned contract, stable reason codes, numeric/finite checks,
  OHLC invariants, strict chronology, duplicate detection, configurable gap
  detection, stale/future timestamp detection, and a stable input fingerprint.
- [ ] Gaps are rejected by default. `DEGRADED_ALLOWED` is possible only under an
  explicit gap policy; no automatic interpolation/repair is performed.
- [ ] Feature vectors validate schema/version, exact feature membership,
  numeric finite values and `available_at <= decision_timestamp`.
- [ ] External records validate provider/source identity, version, event/ingest/
  availability timestamps, unit, quote currency, revision, licence status and
  rate-limit state. Unapproved sources and blocked rate limits are rejected.
- [ ] Model outputs validate version, model revision, proposal vocabulary,
  confidence/uncertainty bounds and as-of timing. Model output is never a Risk
  Gate approval or order instruction.
- [ ] Synthetic regression tests cover all known fault classes and reason codes.
- [x] OHLCV validation is called from `AliXStrategy.populate_indicators`
  before future feature computation, so Freqtrade backtest and runtime use the
  same strategy hook. Feature-vector/model inference integration remains
  gated until the FreqAI baseline exists.

## Connectivity Manager

- [ ] Only allowlisted routes are probed or activated.
- [ ] Routes are attempted by deterministic priority and route ID; disabled
  routes are skipped and duplicate IDs fail closed before any probe.
- [ ] Probe and activation failures have safe reason codes; raw exceptions and
  credential-bearing details are never retained.
- [ ] Per-route snapshots expose health, reason code, probe latency and whether
  activation was attempted.
- [ ] No route activation, VPN/proxy/tunnel process launch or host-network
  mutation is performed from GitHub Actions.
- [ ] Unknown order outcomes require reconciliation before route switching or
  retry; no new executor or parallel order lifecycle is introduced.

## Local system test preparation

- [ ] Runbook documents a clean Python 3.12 environment, pinned lockfile install,
  compile/lint/type/security/architecture/complexity checks, unit coverage and
  Freqtrade safe-config smoke.
- [ ] Real external connectivity tests are separated as owner-authorized,
  read-only first, and carry rollback/evidence instructions.
- [ ] Expected outcomes, evidence format and NO-GO/stop conditions are explicit.

## External deployment gates (not covered by mock-only CI)

- [ ] Verify approved live data providers, official endpoint contracts, licence/
  retention/rate-limit policy and current schema in an authorized environment.
- [ ] Validate exchange/network route reachability, DNS/TLS, authentication,
  egress allowlist and reconnect behavior without capital or exchange writes.
- [ ] Wire the Gate into the actual feature/inference boundary after interface
  review; prove identical semantics for backtest and runtime paths.
- [ ] Prove route switch and timeout behavior against reconciliation policy.
- [ ] Local owner-run system tests have been executed and evidence recorded.

## Decision rule

Phase 3 remains **NO-GO for live trading** until the quality gate is wired into
the runtime boundary and external/local deployment evidence is reviewed. CI
success validates the candidate code and fixtures only, not live-provider or
network compatibility.
