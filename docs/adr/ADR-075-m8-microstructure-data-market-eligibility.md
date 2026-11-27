# ADR-075 — M8 Microstructure: M1.6 Data Market Mismatch

- **Date:** 2026-11-27
- **Status:** ACCEPTED — BLOCKED_DATA_MARKET_MISMATCH / CORRELATIONS_NOT_COMPUTED
- **Scope:** M1.6 raw L2 Order Book artifacts and M8 BTCUSDT 5m/15m microstructure study
- **Branch:** `research/nobitex-hf-strategy`

## 1. Decision

The five recovered M1.6 capture artifacts are **Nobitex BTCIRT**, not BTCUSDT. The requested M8 study is intended to assess BTCUSDT. Therefore this corpus is ineligible for the requested study.

**No BTCUSDT feature/target correlation was calculated.** The correct result is `NOT_COMPUTED`, not zero, not statistically insignificant, and not a NO-GO finding about the BTCUSDT microstructure hypothesis. A fresh, independently captured and hash-pinned BTCUSDT L2 dataset is required before the 5-minute / 15-minute hypothesis can be tested.

The M8 analyzer is explicitly fail-closed: after checking the archive SHA, ZIP file set, manifest checksum, market and run identity, it returns `BLOCKED_MARKET_MISMATCH` before parsing order-book rows into features when the market does not equal the expected market.

## 2. M1.6 source-artifact recovery and integrity

All five GitHub Actions ZIP artifacts were downloaded. The ZIP SHA-256 values below were recomputed from the recovered bytes and matched the GitHub artifact digests. The raw ZIPs were copied to the persistent Library folder `/AliX_M1.6_Immutable_Archive/`; each Library copy was materialized again and re-hashed, with all five hashes matching the original. The original bytes were not rewritten, unpacked/repacked, normalized, or overwritten.

| Segment | Run ID | Artifact ID | Order-book rows | Trades | Gap rows | ZIP SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 45553679888 | 15016992581 | 32,122 | 5,048 | 2 | `aafe15ed9da1d6f6e8badf905627665695acfd605e926b242856f43888980ce8` |
| 2 | 45620061415 | 15054886094 | 37,562 | 8,201 | 0 | `c25c202e5d556912bc40e6a9fdcf78adb5a23db32f3f760ad3a1024d79cf8961` |
| 3 | 45683066882 | 15077454919 | 34,076 | 2,664 | 2 | `c7b7ec592cda5fec117976a422d2257a908c2b260d89b4b9b89ef87ab4f20a62` |
| 4 | 45735110603 | 15095790023 | 35,381 | 3,808 | 4 | `ec4befa0ad6fc96cf1874cec4ba15b2a2dc5ca27e58adf82c33eb62f18393332` |
| 5 | 45775145557 | 15121226078 | 32,463 | 5,902 | 0 | `c52cb93bfc345e1ba66104f7f0e4f8ea3664687e87d3dd89e44ea9236279db88` |
| **Total** | | | **171,604** | **25,623** | **8** | |

Every manifest states `market=BTCIRT`. The capture records also identify the five artifacts as separately dispatched capture segments and explicitly mark that segment boundaries must not be joined as continuous data. Three segments contain explicit gap records. Even for a future BTCIRT-specific study, segments and gap intervals must remain hard discontinuities.

### Persistent archive locations

- Raw ZIPs and README: `/AliX_M1.6_Immutable_Archive/` in the user's persistent Library.
- Machine-readable provenance inventory: [`docs/research/m1_6_artifact_inventory.json`](../research/m1_6_artifact_inventory.json).
- The five original Actions artifacts currently have their original GitHub retention windows; the Library copies are the durable copies verified for this handoff. The repository inventory pins the source hashes but does not embed the binary ZIP files.

## 3. OffsetTracker provenance

The M1.6 collection followed the OffsetTracker reconnect-baseline fix recorded in ADR-058. Initial connection and reconnect paths obtain a fresh REST offset baseline before accepting WebSocket deltas. The original capture evidence, reconnect handling, and tests remain unchanged. This validates the capture sequence mechanics; it does not change the instrument identity from BTCIRT to BTCUSDT.

## 4. M8 analysis contract prepared

The following research-only implementation has been added:

- `scripts/analyze_m8_microstructure.py`
- `.github/workflows/m8-microstructure-analysis.yml`
- `tests/unit/test_m8_microstructure.py`
- `docs/research/m1_6_artifact_inventory.json`

The lightweight, no-ML analyzer is designed to compute:
- `OBI_top5 = (bid_depth_top5 - ask_depth_top5) / (bid_depth_top5 + ask_depth_top5)`
- Microprice and its mid-relative edge in basis points, using best bid/ask prices and displayed best-level sizes.
- Bid-ask spread relative to mid in basis points.
- Total displayed top-five depth.

Targets are forward mid-price log returns at 5 minutes and 15 minutes, expressed in basis points. Pearson and Spearman statistics, two-sided p-values, and Bonferroni-adjusted p-values are reported for four features at both horizons. Sampling uses a non-overlapping UTC grid for each horizon, a maximum 5-second feature-snapshot staleness, a maximum 30-second future-snapshot lag, and rejects windows that intersect explicit order-book gaps. Capture segments are processed independently and never stitched.

The workflow contains **only** `workflow_dispatch`, downloads the five registered artifacts, checks their exact archive SHA-256 values, and uploads the report and verified downloaded ZIPs as an Actions artifact. The expected instrument is fixed to BTCUSDT for this study.

## 5. Dispatch constraint and execution status

The workflow file currently exists only on `research/nobitex-hf-strategy`. The repository default branch is `main`, and the M8 workflow file is absent from `main`. GitHub documents that a `workflow_dispatch` event is accepted only when the workflow file exists on the default branch ([GitHub Actions event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)).

Because the owner explicitly froze `main`, the new workflow cannot be manually dispatched while that freeze remains in force. No alternate `push` trigger was added. Consequently:
- **M8 Actions execution:** NOT RUN.
- **BTCUSDT correlation:** NOT COMPUTED.
- **Evidence of zero/positive correlation:** NONE.
- **Current decision:** `NEW_BTCUSDT_CAPTURE_REQUIRED`.

This is an infrastructure/data-eligibility block, not a statistical result. The code and manual-only workflow are staged on the research branch, but a workflow-dispatch execution requires the workflow definition to be present on the default branch first.

## 6. Safety and scope

Only public capture artifacts and their manifests were inspected. No Holdout or Post-Hack bytes were accessed. No private API calls, live/paper orders, model fitting, or trading authorization occurred. No paid service or TinyFish was used. `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain unchanged. `main` remains frozen at `a1c1a658906cd79be62796885e3ed782b66a267d` (tree `5761c6c34143f8e6924f53a4e745c48932a5f791`).

## 7. Next admissible step

Acquire a fresh BTCUSDT L2 capture through the approved public exchange feed, preserving raw snapshots, timestamps, offsets, reconnect baselines, explicit gap records, capture manifests, and cryptographic hashes. The capture must identify the instrument as BTCUSDT in every manifest. Only after this source gate is satisfied should M8 correlations be run. No DeepLOB or complex ML model is authorized by this ADR.
