# Phase 0 Open Tasks and Local Verification

## Open task: Novaryn adapter
- Status: OPEN / BLOCKED.
- Blocker: official repository URL, pinned commit, license and API contract are unavailable.
- No adapter compatibility is claimed. See [adapter task](../../integrations/novaryn/README.md).

## Local-only market whitelist verification
- Status: NEEDS LOCAL SYSTEM REVIEW.
- Sample data: `config/environments/market-whitelist.sample.json`.
- Structural contract: `config/environments/market-whitelist.schema.json`.
- The sample contains no markets and is not an approved trading whitelist.
- Before any later phase can enable a market, verify official exchange market metadata, symbol precision/limits, quote/base currencies, and trading availability from an authorized environment; record evidence and review date.
- GitHub CI must remain mock-only and must not contact the live Nobitex API.

## Provider/model/data licensing
- Status: OPEN / BLOCKED pending source evidence.
- Record provider/model names, version or immutable revision, license, permitted commercial use, redistribution restrictions, and source URL before integrating external artifacts.
- Do not assume that package availability implies license approval.

## Bale reporting integration
- Status: OPEN / BLOCKED pending contract verification.
- Confirm endpoint behavior, authentication method, timezone/schedule, recipient/chat allowlist and delivery failure behavior in an authorized environment.
- CI uses injected senders and must not send real messages.

## Runtime dependencies and reproducibility
- Direct dependencies are pinned in `requirements.txt`; the transitive lock is committed in `requirements.lock`.
- CI performs a clean resolver dry run, regenerates the lock with pinned pip/pip-tools, diffs it against the committed lock, and runs pip-audit against the locked set.
- A full runtime integration test with the Freqtrade stack remains a separate gate before any deployment.

## Starter Kit integrity
- The original archive remains unchanged at repository root.
- `artifacts-manifest/starter-kit.json` pins both the Git blob SHA-1 and SHA-256.
- The validator recomputes both checksums, verifies ZIP integrity and required embedded documents, and validates the market whitelist sample against its JSON Schema.
