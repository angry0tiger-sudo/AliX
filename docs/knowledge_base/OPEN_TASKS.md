# Phase 1 Open Tasks and Deployment Verification

# Current repository checkpoint — 2026-10-22

- Verified main HEAD before Phase 7 work: 5873ba52ab827109a0be450670e971796295fa8e (Phase 6 merge commit).
- Phase 5 PR #6 was subsequently merged at that SHA. Main push CI run #380 / ID 39733266520 passed on the exact main SHA; this evidence is for Phase 5 only.
- Phase 6 tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/7. Work branch: feat/phase-6-external-data-nlp.
- Phase 6 tested implementation SHA: `52bc49012a945bf6c3bb411b6a4d3933a9a69aa4`. Push CI [#402 / ID 39809700265](https://github.com/angry0tiger-sudo/AliX/actions/runs/39809700265) and PR CI [#403 / ID 39809712528](https://github.com/angry0tiger-sudo/AliX/actions/runs/39809712528) both passed with Quality and Runtime green. Quality evidence: 131 unit tests passed, 91.18% coverage, 4 Phase 1 safety contracts passed, pip-audit found no known vulnerabilities. Report-sync SHA `416b46ebdbb59eb0f095fc7636368ca6b36de462` also passed Push CI [#404 / ID 39810335179](https://github.com/angry0tiger-sudo/AliX/actions/runs/39810335179) and PR CI [#405 / ID 39810343496](https://github.com/angry0tiger-sudo/AliX/actions/runs/39810343496); both Quality and Runtime jobs green. PR #8 was subsequently merged by the owner at 5873ba52ab827109a0be450670e971796295fa8e; merge confirmed in live GitHub PR metadata.
- Phase 6 remains mock-only. Live provider rights, approved model/dataset revisions, owner-authorized historical news snapshots, calibration/ablation and durable stores remain open. Champion null; promotion blocked; economic acceptance and trading authorization NO-GO.


# Phase 8 checkpoint — local-first Control Plane (implementation candidate)

- Owner-approved branch: feat/phase-8-multitenant-control-plane.
- Base main SHA: 900eeef90b4e7c1d0e5652fad22f932a2aad9978.
- Owner Directive: docs/governance/owner-directives/2026-10-22-phase-8.md.
- ADR: docs/architecture/decisions/ADR-0008-phase-8-local-single-user-control-plane.md.
- Scope: local token mode, PostgreSQL Control Plane schema, per-Worker SQLite, scoped Worker heartbeat, transactional runtime outbox, and isolated non-trading Compose Worker.
- Status: in progress; do not claim completion until Push and PR CI pass on exact candidate SHAs.
- No Freqtrade Worker, real exchange key, exchange/provider request, live/dry-run/paper session, order operation or host-network change is authorized.
- Trading authorization remains NO-GO; order_permission remains false.

# Phase 4 checkpoint — merged; economics remain blocked

- Phase 4 PR #4 was merged by the owner on 2026-10-22; merge commit and current main at Phase 5 start: `f553daa742e09c66c054f4566a22491fcfaa80a7`.
- Final pre-merge exact-head CI: [run #333 / ID 39725145516](https://github.com/angry0tiger-sudo/AliX/actions/runs/39725145516), on `3987dbc3084c456dd8cf8eb5709bee6f480dc18a`; both quality and runtime smoke passed. The Phase 4 synthetic artifact was explicitly mechanics-only.
- Canonical MRD/Roadmap remain under `docs/knowledge_base/`; earlier architecture/roadmap copies are retained.
- Phase 4 economic acceptance: **NO-GO**. Champion remains null and promotion is blocked; no trading is authorized.

# Phase 5 tracking checkpoint — owner-authorized, offline only

- Owner directive: `docs/governance/owner-directives/2026-10-22-phase-5.md`.
- Architecture decision: `docs/architecture/decisions/ADR-0005-phase-5-plan-b.md`; isolated MRD section 4.6 Plan B services; no DeepAlpha imports or Freqtrade core patches.
- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/5.
- Branch: `feat/phase-5-triple-barrier-meta-labeling`; base at Phase 5 start: `f553daa742e09c66c054f4566a22491fcfaa80a7`.
- Tested code SHA: `59ab355a04c9b9e6452bb4767b81d4c0fab7f3a2`; [CI run #369 / ID 39729707614](https://github.com/angry0tiger-sudo/AliX/actions/runs/39729707614) passed both quality and runtime smoke.
- Evidence: 111 unit tests passed, 91.38% coverage against 90% required, 4 Phase 1 safety contracts passed; compile/Ruff/strict Mypy/Bandit/dependency lock/pip-audit/import-linter/Xenon/docs validation all passed. Runtime smoke used pinned Freqtrade 2026.7 with closed configuration and no exchange connection.
- Latest exact-code synthetic artifact: [ID 12404705949](https://github.com/angry0tiger-sudo/AliX/actions/runs/39729707614/artifacts/12404705949). It records 1,400 synthetic candles, a VALID shared Data Quality Gate fingerprint, 800 OOF primary predictions, 800 Triple-Barrier events, and 3 evaluated meta folds; JSON model reload max score difference was 0.0.
- Synthetic mechanics only: the historical-rate score baseline beat Meta-Label Brier/log-loss in all evaluated folds, and the filtered proposal mean net return was more negative than the unfiltered mean in all evaluated folds. These values are not market evidence.
- Historical dataset/source/license/hash is not approved; economic acceptance criteria are not pre-registered. Economic acceptance remains **NO-GO**. Champion null; `promotion_status=BLOCKED_NO_ECONOMIC_GATE`; trading authorization NO-GO.
- Closure update (2026-10-22): PR #6 was subsequently merged into main at ce360a770586e76dc5a0f7b2421a371322e6d738. Historical synthetic metrics remain mechanics-only and are not market-performance evidence.
- No exchange/provider requests, downloads, order writes, notifications, paper/live/dry-run sessions, or host networking changes are authorized or performed.

# Phase 3 contract/CI checkpoint

- Implementation candidate SHA: `a3c5a7e2748e47021b3f8eb52f5f694ef9d75517`.
- [Push CI 39717114585](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717114585) and
  [PR CI 39717119556](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717119556): success.
- Evidence: 91 unit tests passed; 92.62% coverage; 4 Phase 1 safety/dependency
  contracts passed; Mypy, Ruff, Bandit, pip-audit, import-linter, Xenon and
  Freqtrade safe-config smoke passed.
- Scope includes versioned quality contracts, fail-closed connectivity status
  and failover diagnostics, strategy-level OHLCV gating, and the local system
  test runbook. No live data/provider or host networking was exercised.
- Remaining gates: owner-run local system tests; current provider schemas and
  licence/rate-limit review; live egress/DNS/TLS authentication review; future
  FreqAI inference wiring; reconciliation before route switch/retry; owner review.
- Report/acceptance follow-up SHA `101c67dbcba5b773616e6664425e96c05c0b1c43`:
  push CI [39717394080](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717394080)
  and PR CI [39717398121](https://github.com/angry0tiger-sudo/AliX/actions/runs/39717398121)
  both completed successfully.
- These checks establish the contract/CI foundation only; no live data provider,
  order write, Bale send or host-network activation was exercised.



## Phase 2 contract/CI checkpoint

- Contract implementation commit `8743469f5f072ea482e5d016859ebd7e8d32a92c`
  passed push CI run [39669097682](https://github.com/angry0tiger-sudo/AliX/actions/runs/39669097682)
  and pull-request CI run [39669102977](https://github.com/angry0tiger-sudo/AliX/actions/runs/39669102977).
- Evidence: 70 unit tests passed, 92.30% coverage, 4 separate Phase 1 safety/
  dependency contract tests passed; strict Mypy, security, dependency, architecture,
  complexity, and Freqtrade safe-config gates passed.
- The PR's docs/template-updated SHA must pass its own workflow before this
  checkpoint is considered current for that exact SHA.
- The gates for live schema/market verification, authorized read-only account
  reads, Freqtrade boundary compatibility, reconciliation/recovery, and
  Novaryn license/provenance remain open.



## Phase 2 — Spot/USDT contract integration (in progress)

- AliX Phase 2 has started from the owner-approved Phase 1 HEAD.
- The current implementation remains mock-only: typed order/balance snapshots,
  partial-fill classification, precision/minimum/maximum validation, read retry
  policy, and pagination checks are contract helpers, not a live Nobitex client.
- Do not enable any order-write path or claim market metadata is verified from
  fixtures. No order submission/cancellation or live API call is permitted in CI.
- Open Phase 2 gates: validate current official API payloads and USDT markets in
  an authorized read-only environment; exercise fills/fees/status/limits and rate
  limits; test restart reconciliation and unknown submission outcomes; prove
  the single Freqtrade -> adapter -> Nobitex execution boundary without
  modifying Freqtrade core; resolve source license/provenance.



## Novaryn source and adapter provenance
- Reference repository: https://github.com/angry0tiger-sudo/Novaryn
- Inspected commit: `54e66ad40b5a7e422002619f7dad9378ecdf09f9`.
- Status: source contract and selected deterministic behaviors are mapped into `integrations/nobitex/adapter.py`; no full source-module copy was made.
- Remaining blocker: obtain and record authoritative source license/provenance and compatibility approval before any further literal code reuse.
- The adapter has no default HTTP transport and no order-write path. `submit_order` always raises `LiveExecutionDisabled`.

## Local/authorized Nobitex market verification
- Status: NEEDS AUTHORIZED RUNTIME REVIEW.
- Sample data: `config/environments/market-whitelist.sample.json`.
- Structural contract: `config/environments/market-whitelist.schema.json`.
- The sample contains no markets and is not an approved trading whitelist.
- The read-only market-stats and USDT symbol resolution behavior are unit-tested with injected fakes only.
- Before any later phase enables an actual market, verify current official market metadata, symbol/precision/limits, quote/base currency, trading availability, and credentials from an authorized environment. Record the evidence and review timestamp.
- GitHub CI does not call the live Nobitex API.

## Bale reporting integration
- Bale client: BaleIO pinned to immutable commit `66be3000af6e62332e1764f227aed64847a0d9ba`.
- Default chat ID: `832112407`; override using `BALE_CHAT_ID`.
- Bot token remains a runtime secret in `BALE_BOT_TOKEN`; `ALIx_REPORTING_ENABLED=false` remains the default.
- Status: mock transport, allowlist, error handling and cleanup are covered by unit tests. No real Bale message has been sent.
- Remaining deployment gate: authorize a real-token send to the intended allowlisted chat and record the result without logging credentials.

## VPN / proxy / tunnel operations
- Status: route selection/failover is unit-tested using injected health and activation callbacks.
- No OS networking changes, VPN process starts, proxy activation or tunnel creation has happened in CI.
- Remaining deployment gate: verify the approved clients and recovery behavior in an authorized target environment before enabling any activation callback that mutates host/network state.

## Freqtrade runtime
- Version baseline: `freqtrade==2026.7`, aligned with Novaryn's documented integration contract.
- Runtime smoke test installs Freqtrade, runs its version command and loads the strategy while checking dry-run/stopped/zero-trade/empty-credential defaults. It does not contact an exchange.
- Remaining deployment gate: verify the complete Novaryn-compatible exchange boundary, candles/balances/open-order reads and order reconciliation/recovery before considering any production runtime.

## Provider/model/data licensing
- Status: OPEN pending authoritative source evidence.
- Record provider/model name, immutable version, license, permitted commercial use, redistribution constraints and source URL before adding external model/data artifacts.
- Package availability does not by itself imply license approval.

## Dependency reproducibility
- Direct dependencies are pinned in `requirements.txt`; the transitive lock is committed in `requirements.lock`.
- CI performs a clean resolver dry run, regenerates the lock constrained by the committed pins, diffs it, and audits the locked dependency set.
- Continue upgrading only through passing CI. Do not bypass failed audits with blanket exclusions.

## Starter Kit integrity
- The original archive remains unchanged at repository root.
- `artifacts-manifest/starter-kit.json` pins both Git blob SHA-1 and SHA-256.
- The validator recomputes both checksums, verifies ZIP integrity and required embedded documents, and validates the market whitelist sample against its JSON Schema.


# Phase 10 checkpoint: scheduler, reporting contracts and public market research

- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/13.
- Base main SHA: `7893c87a8eea7fc10cf91b7cac3d891797b1f9e5`.
- Owner Directive: `docs/governance/owner-directives/2026-10-23-phase-10.md`.
- ADR: `docs/architecture/decisions/ADR-0010-phase-10-reporting-scheduler-public-research.md`.
- Status: IN PROGRESS / NO-GO pending exact-head CI and review of market-data artifacts.
- Public Nobitex OHLCV retrieval is a manual GitHub Actions workflow. CSV/manifest/research report remain workflow artifacts, not source-controlled market history.
- Scheduler state is persisted by schedule slot with a lease, bounded retry/backoff and run history; SQLite is one-host only and PostgreSQL is the shared-store option.
- Freqtrade and synthetic-ledger adapters are read-only. The Phase 9 ledger remains SYNTHETIC_ONLY / NOT_RECONCILED. An account snapshot and reconciled decision/order linkage are not yet provided.
- The current AliX strategy scaffolds remain signal-free. Walk-forward output is a fixed-horizon research rule, not a full Freqtrade/RiskGate execution test.
- `Champion=null`, `order_permission=false`, promotion blocked and trading authorization NO-GO remain mandatory. Phase 10 PR must not be merged without separate explicit owner authorization.


# Strategy Registry and regime-aware selector — 2026-10-23

- Tracking issue: https://github.com/angry0tiger-sudo/AliX/issues/17.
- Work branch: feat/strategy-registry-selector; base main SHA: 57103cf4cf7af720ec1166794df9dd62954cd38a.
- ADR: docs/architecture/decisions/ADR-0011-versioned-strategy-registry-and-selector.md.
- Owner Directive: docs/governance/owner-directives/2026-10-23-strategy-registry-selection.md.
- Status: implementation candidate / research only; the production registry is empty. The selector cannot approve risk or grant order permission.
- The selector requires code/data/report/evaluation lineage, valid data quality, exact schema and regime compatibility, license approval, current cost/liquidity context, complete walk-forward folds/regime evidence and pre-registered economic thresholds.
- Economic gate remains at least 100 OOS trades, win rate >= 60%, absolute maximum drawdown <= 15% and positive net expectancy after costs. The latest Phase 10 result has 16 trades and remains INSUFFICIENT_SAMPLE.
- No strategy is selected by default; no Freqtrade entry/exit signals or order paths are enabled.
- Related merge blocker: issue #16. Repository ruleset for main was observed with enforcement disabled; classic branch-protection status was not readable by the current GitHub integration. Verify effective branch protection/review requirements before merging exchange/data-path changes.
- Current safety state remains Champion=null, order_permission=false, promotion blocked, economic acceptance NO-GO and trading authorization NO-GO.
