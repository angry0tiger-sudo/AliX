# ADR-037: Persist the Exact Public Pre-Hack Nobitex Dataset

- Status: Accepted under the explicit Owner directive dated 2026-11-14.
- Date: 2026-11-14.
- Governance: MRD v0.9, QWEN.md, ADR-0010, ADR-0012, ADR-032, ADR-036.
- Scope: One exact, already-acquired public Nobitex BTC/USDT Spot 1h pre-hack dataset. This ADR does not authorize new market-data acquisition, holdout access, model evaluation, or trading.

## Context and expiring source

The exact 31,801-row pre-hack CSV was retained only in GitHub Actions artifact 13971011148, from workflow Run ID 43274590426. GitHub's artifact metadata reported the actual expiry as 2026-12-12T18:05:52Z. Expiration would remove the only currently identified retrievable copy and undermine reproducibility of the registered four-arm research dataset.

The artifact was downloaded and its ZIP digest verified as ae7a4d00fe2ec753b28868e4849fea4d45c88f1fd25b366b64298d7fec809b21. The CSV bytes were independently hashed before persistence; the exact registered digest matched.

## Dataset identity and durable path

- Venue/source: Nobitex public UDF history API; public market OHLCV only.
- Instrument/timeframe: BTCUSDT Spot, 1h.
- Inclusive candle-open range: 2021-10-31T20:30:00Z through 2025-06-17T20:30:00Z.
- Rows: 31,801.
- CSV bytes: 1,897,551.
- CSV SHA-256: 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e.
- Source artifact ID: 13971011148; source run ID: 43274590426.
- Artifact expiry: 2026-12-12T18:05:52Z.
- Artifact ZIP SHA-256: ae7a4d00fe2ec753b28868e4849fea4d45c88f1fd25b366b64298d7fec809b21.
- Durable CSV path: data/nobitex/prehack/nobitex-btcusdt-1h.csv.
- Adjacent manifest: data/nobitex/prehack/nobitex-btcusdt-1h.manifest.json.

The repository copy is the byte-for-byte CSV from the source artifact, not a re-fetch or regenerated serialization. The manifest preserves the source acquisition metadata and adds the GitHub artifact provenance, original expiry and pre-persistence verification evidence.

## Narrow supersession of Phase-10 data-storage convention

ADR-0010, Decision 7, established the Phase-10 convention to upload market-data CSVs as Actions artifacts and not commit market data. That convention is expressly superseded only for the exact dataset identified in this ADR.

Reasons for this narrow exception:

1. The data is public exchange OHLCV, not personal data, account data, credentials, private API output, order/fill truth, or secrets.
2. The current source artifact has a fixed, near-term expiry (2026-12-12T18:05:52Z); losing it would destroy the only identified retrievable copy of the exact byte sequence.
3. Re-fetching later cannot guarantee byte-identical serialization or an identical historical source response. The registered SHA-256 must remain reproducible.
4. The dataset is the fixed, already-registered input to research; retaining exact bytes is necessary for auditability, controlled reruns and future evidence comparison.

This exception does not change the default Phase-10 rule for other market data, other exchanges, new datasets, or future acquisitions. Those require their own Owner authorization and governance record.

## Integrity and usage limits

- The CSV SHA-256 must remain exactly 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e; any mismatch is a hard failure.
- Row count, exact hourly grid, first/last timestamps, and OHLC invariants are validated before commit.
- No cross-venue substitution, backfill, interpolation, masking, synthetic fill or timestamp adjustment is permitted.
- The known contaminated final 12.5% segment remains excluded from confirmatory scoring under the forecast-driven amendment; persistence does not make it pristine.
- This ADR does not resolve GLOBAL_N=UNKNOWN or prove a positive economic edge.
- No post-hack holdout bytes are stored here. V2 remains FETCHED_LOCKED, dataset_persisted=false; future access remains subject to exact Spec v3 re-fetch and SHA-256 match and a separate Owner Release Authorization.

## Safety boundary

- Champion=null
- order_permission=false
- trading_authorization=NO-GO
- private_api_calls=0
- real_order_attempts=0
- ALIX_DRY_RUN=true
- ALIX_ALLOW_LIVE_TRADING=false
- main=FROZEN

This is a data-retention decision only. It does not execute the four-arm experiment, authorize Paper Trading, or release the post-hack holdout.
