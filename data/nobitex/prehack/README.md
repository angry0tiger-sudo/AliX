# Pre-Hack Nobitex Dataset

This directory is reserved for the one fixed public Nobitex BTC/USDT Spot 1h pre-hack dataset registered for AliX research.

- Canonical CSV: nobitex-btcusdt-1h.csv
- Expected rows: 31,801
- Required CSV SHA-256: 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e
- Source: public Nobitex UDF history endpoint; no credentials or private API.
- Source Actions artifact: 13971011148 from run 43274590426, expiry 2026-12-12T18:05:52Z.

The persistence workflow verifies the artifact's exact CSV digest, manifest, row count, hourly timestamp grid and OHLC invariants before writing the dataset and adjacent provenance manifest.

This directory is strictly pre-hack. No post-hack holdout bytes may be stored here. The holdout remains sealed, non-persisted and governed by Spec v3 and a separate Owner Release Authorization.
