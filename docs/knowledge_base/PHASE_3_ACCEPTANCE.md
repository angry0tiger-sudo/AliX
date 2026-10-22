# Phase 3 Acceptance: Data Quality Gate and Connectivity Manager

Status: **CONTRACT/CI FOUNDATION VERIFIED ON IMPLEMENTATION SHA; FINAL DOC-UPDATED SHA AND OWNER REVIEW PENDING**.

Implementation SHA: `a3c5a7e2748e47021b3f8eb52f5f694ef9d75517`.

- Push CI: [39717114585](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717114585) — success.
- Pull-request CI: [39717119556](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717119556) — success.
- Evidence: 91 passed, 92.62% coverage (90% minimum), 4 Phase 1 safety/dependency tests passed, Strict Mypy and all configured quality/security/dependency/complexity gates passed; Freqtrade runtime smoke passed with safe config and no exchange connection.


Owner directive: enter Phase 3 after Phase 2 PR #1 was merged into `main`.

## Data Quality Gate

- [x] OHLCV has a versioned contract, stable reason codes, numeric/finite checks,
  OHLC invariants, strict chronology, duplicate detection, configurable gap
  detection, stale/future timestamp detection, and a stable input fingerprint.
- [x] Gaps are rejected by default. `DEGRADED_ALLOWED` is possible only under an
  explicit gap policy; no automatic interpolation/repair is performed.
- [x] Feature vectors validate schema/version, exact feature membership,
  numeric finite values and `available_at <= decision_timestamp`.
- [x] External records validate provider/source identity, version, event/ingest/
  availability timestamps, unit, quote currency, revision, licence status and
  rate-limit state. Unapproved sources and blocked rate limits are rejected.
- [x] Model outputs validate version, model revision, proposal vocabulary,
  confidence/uncertainty bounds and as-of timing. Model output is never a Risk
  Gate approval or order instruction.
- [x] Synthetic regression tests cover the implemented fault classes and reason codes (91 tests passed; residual provider/network cases remain deployment gates).
- [x] OHLCV validation is called from `AliXStrategy.populate_indicators`
  before future feature computation, so Freqtrade backtest and runtime use the
  same strategy hook. Feature-vector/model inference integration remains
  gated until the FreqAI baseline exists.

## Connectivity Manager

- [x] Only allowlisted routes are probed or activated.
- [x] Routes are attempted by deterministic priority and route ID; disabled
  routes are skipped and duplicate IDs fail closed before any probe.
- [x] Probe and activation failures have safe reason codes; raw exceptions and
  credential-bearing details are never retained.
- [x] Per-route snapshots expose health, reason code, probe latency and whether
  activation was attempted.
- [x] No route activation, VPN/proxy/tunnel process launch or host-network
  mutation is performed from GitHub Actions.
- [ ] Unknown order outcomes require reconciliation before route switching or
  retry; no new executor or parallel order lifecycle is introduced.

## Local system test preparation

- [x] Runbook documents a clean Python 3.12 environment, pinned lockfile install,
  compile/lint/type/security/architecture/complexity checks, unit coverage and
  Freqtrade safe-config smoke.
- [x] Real external connectivity tests are separated as owner-authorized,
  read-only first, and carry rollback/evidence instructions.
- [x] Expected outcomes, evidence format and NO-GO/stop conditions are explicit.

## External deployment gates (not covered by mock-only CI)

- [ ] Verify approved live data providers, official endpoint contracts, licence/
  retention/rate-limit policy and current schema in an authorized environment.
- [ ] Validate exchange/network route reachability, DNS/TLS, authentication,
  egress allowlist and reconnect behavior without capital or exchange writes.
- [ ] Wire feature-vector/model-output validation into the future FreqAI inference boundary after interface review; prove point-in-time semantics there.
- [ ] Prove route switch and timeout behavior against reconciliation policy.
- [ ] Local owner-run system tests have been executed and evidence recorded.

## Decision rule

Phase 3 contract/CI foundation is verified for the implementation SHA. It remains **NO-GO for live trading** until external/local deployment evidence, FreqAI inference integration, and route-switch/reconciliation controls are reviewed. CI success validates code and fixtures only, not live-provider or network compatibility.
