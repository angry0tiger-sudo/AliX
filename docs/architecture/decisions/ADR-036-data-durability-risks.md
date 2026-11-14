# ADR-036: Data Durability Risks for the Pre-Hack Dataset and Post-Hack Holdout

- **Status:** Open; Owner decision required for pre-hack preservation.
- **Date:** 2026-11-14.
- **Governance:** MRD v0.9, QWEN.md, ADR-0014, ADR-0017, ADR-032, ADR-034, ADR-035.
- **Scope:** Data availability and integrity only. This decision does not authorize dataset access, experiment execution, or holdout release.

## 1. Pre-hack artifact durability risk

The only verified retained copy of the pre-hack CSV currently identified for the four-arm experiment is a GitHub Actions artifact:

- Workflow: Nobitex HF research run #41.
- Workflow Run ID: `43274590426`.
- Artifact ID: `13971011148`.
- Artifact name: `alix-binance-hf-proxy-research`.
- File: `nobitex-btcusdt-1h.csv`.
- Expected CSV SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Expected rows: `31,801`.
- Artifact created: `2026-11-12T18:05:52Z`.
- **Actual API-reported `expires_at`: `2026-12-12T18:05:52Z`.**

This file is not present as a regular tracked file in the current repository tree. Expiration of this artifact without another verified copy would remove the currently identified retrievable CSV source.

### Options requiring an Owner decision

1. **Owner download and preserve:** The Owner downloads artifact `13971011148` before its exact expiry, preserves the original file and its source metadata, and verifies the CSV SHA-256 before treating it as the retained pre-hack input.
2. **GitHub Actions reproduction:** The Owner authorizes a dedicated GitHub Actions workflow to reproduce only the fixed pre-hack window using the already-approved public-data fetcher, then compare the newly produced CSV SHA-256 against the registered value `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`. An exact mismatch, missing interval, altered serialization, or row-count mismatch must fail closed; it must not silently replace the registered dataset.

No option is selected by this ADR. No reproduction was performed as part of this documentation change. Any reproduction must leave the public fetcher code and its protected blob unchanged.

## 2. Post-hack holdout durability and non-persistence risk

The successful v3 acquisition did **not** persist the raw post-hack CSV or OHLCV bytes. The append-only V2 commitment records `dataset_persisted=false`; its artifact contains a structural manifest and spec fingerprint only. Consequently:

- V2's `FETCHED_LOCKED` status represents a verified structural acquisition commitment and canonical dataset digest, not a retained raw-data file.
- The committed evidence identifies Spec v3, `11,700` rows, acquisition Run ID `43443886254`, and the dataset SHA-256 below.
- No raw holdout rows, prices, or OHLCV values are available from the retained V2 artifact.

Current immutable holdout commitment:

- Spec: `config/holdout/posthack_window_spec.json`, version `3`.
- Inclusive window: `2025-07-01T13:30:00Z` through `2026-11-01T00:30:00Z`.
- Expected rows: `11,700`.
- Dataset SHA-256: `435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee`.
- Acquisition Run ID: `43443886254`.
- Current `DISPATCH_TRIGGER`: `PROBE`.

### Future holdout opening protocol

Only after the preregistered conditions and separate Owner Release Authorization are satisfied:

1. Re-fetch the exact Spec v3 window from the same approved public Nobitex source. Do not change the inclusive start/end timestamps, :30 UTC grid, or expected count.
2. Enforce zero tolerance for missing, off-grid, duplicate, reordered, or malformed candles. Do not interpolate, backfill, mask, shift, or synthesize data.
3. Compute the canonical dataset SHA-256 using the registered serialization and compare it byte-for-byte to `435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee`.
4. Require exactly `11,700` rows and zero missing/off-grid/duplicate timestamps, in addition to the exact digest match.
5. Any mismatch invalidates the prior holdout commitment for use: **STOP**. Do not inspect outcomes, select a replacement window, or proceed with a partial dataset. A new data commitment requires a new, separately reviewed Owner decision and preregistration.

The current documentation operation did not fetch, parse, or expose any post-hack OHLCV value and does not open the holdout.

## 3. Current safety and governance state

- `main=44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` remains FROZEN.
- Public fetcher remains unchanged at blob `01638aeeda04ee7d19584009780e88dc68a395bc`.
- Spec v3 remains unchanged.
- V2 remains `FETCHED_LOCKED`; its raw bytes remain non-persisted.
- `DISPATCH_TRIGGER=PROBE`.
- Four-arm execution lock remains `NOT_AUTHORIZED`.
- `Champion=null`.
- `order_permission=false`.
- `trading_authorization=NO-GO`.
- `private_api_calls=0`.
- `real_order_attempts=0`.

This ADR records durability risks and the future integrity protocol only. It is not a release, execution, or data-access authorization.
