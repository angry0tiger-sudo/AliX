# Phase 0 Open Tasks and Local Verification

## Open task: Novaryn adapter
- Status: OPEN / BLOCKED.
- Blocker: official repository URL, pinned commit, license and API contract are unavailable.
- No adapter compatibility is claimed. See [adapter task](../../integrations/novaryn/README.md).

## Local-only market whitelist verification
- Status: NEEDS LOCAL SYSTEM REVIEW.
- The sample schema is in `config/environments/market-whitelist.sample.json`.
- The sample has no enabled markets and is not an approved trading whitelist.
- Before any later phase can enable a market, verify official exchange market metadata, symbol precision/limits, quote/base currencies, and trading availability from an authorized environment; record evidence and review date.
- GitHub CI must remain mock-only and must not contact the live Nobitex API.

## Starter Kit checksum
The original archive's Git blob SHA is not a file SHA-256. The exact SHA-256 must be established by running the validator in GitHub Actions before it can be pinned in the manifest.
