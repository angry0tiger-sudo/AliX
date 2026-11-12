# HOLDOUT COMMITMENT — AliX Native Nobitex 1h

- Commitment date: **2026-11-12 UTC**
- Branch/HEAD at commitment: `research/nobitex-hf-strategy` / `de1055b05331c98e4ee8f83034d65bdcc82b4748`
- Dataset: Nobitex Public UDF BTC/USDT Spot OHLCV, 1h.
- Rows: **31,801**.
- Full dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Full data range: **2021-10-31T20:30:00Z — 2025-06-17T20:30:00Z**.
- Integrity: exact hourly grid; zero missing intervals in the verified manifest.
- Provenance: public Nobitex history only; not account/order/fill truth.

## Holdout integrity decision

**Status: BLOCKED_CONTAMINATED — the referenced historical dataset cannot supply a fresh untouched holdout.**

The dataset SHA and range above identify the immutable source dataset, but they do not make its terminal slice unopened. Historical trials have already consumed overlapping observations:

1. AI-6 final PPO evaluation used rows `27825..31800` as its final 12.5% holdout:
   - Run ID `43000569350`, commit `7464e308d4e1efdb573a8225fb3a392ae7614199`.
   - Reported holdout interval: 2025-01-03T05:30:00Z through 2025-06-17T20:30:00Z.
   - Baseline trades=225; PPO trades=0.
2. DEAP AI-2 Revisited used OOS beginning at row `22308` through the last row, which includes the complete AI-6 terminal slice:
   - Run ID `43022191014`, commit `9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92`.
   - OOS trades=4.

For contamination tracking only, the already-exposed terminal candidate slice [27825,31801) was fingerprinted as SHA-256:
`979a7c559c8eaca3d6e55c84b225fb27aa9dcdd482a425c6e7a822be6278cbd3`.
This fingerprint is **not** a valid new holdout commitment.

## Explicit non-consumption commitment

As of 2026-11-12, the HF/Binance workstream is stopped. No new four-arm Kronos/Chronos trial is running or authorized to consume the historical terminal slice as a fresh holdout.

I commit that:
- No current or future trial may use rows [22308,31801) of this historical dataset as an untouched final holdout.
- No alternate split, new hash of a subrange, or different model family may relabel already-exposed rows as untouched.
- The four-arm experiment cannot use this dataset to claim a one-time unseen final holdout.
- A future valid holdout requires a genuinely new native Nobitex window (new CSV + manifest), immutable SHA-256, exact boundaries and zero gaps, committed before any model/researcher sees it.

This is a commitment **not to consume the contaminated slice again**, not a claim that the slice was historically unseen. The full dataset hash remains recorded for reproducibility and data provenance only.

## Safety invariants

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `private_api_calls=0`
- `real_order_attempts=0`
