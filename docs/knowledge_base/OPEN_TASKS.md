# Phase 1 Open Tasks and Deployment Verification

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
- The documentation-only follow-up must pass CI on its own SHA. These results
  establish only the implementation candidate baseline, not live readiness.



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
