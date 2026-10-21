# Phase 1 Open Tasks and Deployment Verification

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
