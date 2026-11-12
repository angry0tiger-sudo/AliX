# HOLDOUT COMMITMENT — AliX Native Nobitex 1h

Commitment date: 2026-11-12 UTC.
Reference active-branch HEAD at audit time: `bc493e24b9e3f849b6d9fbd0b776d15b39b8061f`.
Status: **BLOCKED — no valid untouched final holdout remains in the supplied 31,801-row dataset.**

## 1. Immutable source dataset

- Venue: Nobitex public UDF history API.
- Pair: BTC/USDT Spot.
- Timeframe: 1h.
- CSV: `nobitex-btcusdt-1h.csv`.
- Rows: 31,801.
- Start: 2021-10-31T20:30:00Z.
- End: 2025-06-17T20:30:00Z.
- CSV SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
- Exact run #41 artifact: [Run 43274590426](https://github.com/angry0tiger-sudo/AliX/actions/runs/43274590426),
  Artifact ID `13971011148`, ZIP SHA-256
  `ae7a4d00fe2ec753b28868e4849fea4d45c88f1fd25b366b64298d7fec809b21`.
- Manifest SHA-256 for the downloaded run #41 artifact:
  `985d9ad56733f6d59655f4565b1f5bbf57357d047b8eaa2994b6254dc8d5ba37`.
- Manifest records `missing_intervals=0`, `private_api_calls=0`, `real_order_attempts=0`,
  `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`.

The CSV SHA was recomputed from the downloaded artifact and matches the supplied owner hash.

## 2. Candidate final 12.5% slice fingerprint (NOT a valid sealed holdout)

For an auditable contamination record, the terminal AI-6 holdout slice was fingerprinted using the
exact row boundaries already recorded by the AI-6 report:

- Source CSV row indexing: data rows are zero-based after the CSV header.
- `holdout_start_row = 27825`.
- `holdout_end_row_exclusive = 31801`.
- Data rows in the candidate slice: 3,976.
- Candidate slice starts at: `2025-01-03T05:30:00Z`.
- Candidate slice ends at: `2025-06-17T20:30:00Z`.
- Canonical byte serialization: original CSV header bytes followed by the original unmodified CSV
  row bytes for data-row indices [27825, 31801); no reformatting or newline normalization.
- Candidate slice SHA-256:
  `979a7c559c8eaca3d6e55c84b225fb27aa9dcdd482a425c6e7a822be6278cbd3`.

This is a fingerprint for audit and accidental-reuse prevention only. It is **not** a valid new
holdout commitment.

## 3. Why the candidate is already opened / contaminated

The candidate slice is not untouched:

1. The final AI-6 PPO report explicitly declares the same exact holdout boundaries
   (`holdout_start_row=27825`, `holdout_end_row_exclusive=31801`, `final_holdout_fraction=0.125`).
   It evaluates the baseline on 225 trades and PPO on 0 trades.
   - Research protocol: `docs/research/AI6-PPO-RISK-RESEARCH.md`,
     blob SHA `c5c6491835af99aaadafc1205154ce45dcb33b9c`.
   - Final verdict: `docs/phase-reports/AI6_RL_FINAL_VERDICT_2026-11-11.md`,
     blob SHA `bd535da99af29c53a4ec81366f8a3b20db2fc5c9`.
   - Workflow run #22: `43000569350`.
   - Model/training commit: `7464e308d4e1efdb573a8225fb3a392ae7614199`.
2. DEAP's final OOS starts at `oos_start_row=22308` and runs through the last source row.
   Therefore the entire candidate slice [27825,31801) is inside the DEAP OOS segment.
   - DEAP report artifact: Artifact ID `13855652715`, final run `43022191014`.
   - DEAP report commit: `9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92`.

A segment evaluated for historical model selection/reporting cannot be relabeled as a new untouched
holdout merely because the next experiment uses different models. It has already been opened.

## 4. Commitment decision

- `CANDIDATE_SLICE_SHA256` is recorded above for reproducibility.
- `VALID_UNTOUCHED_HOLDOUT_SHA256 = NONE`.
- `HOLDOUT_COMMITMENT_STATUS = BLOCKED_CONTAMINATED`.
- Do not use the candidate slice for a new final economic verdict.
- Do not hash a different subset of this same dataset and call it untouched; the DEAP OOS covered
  rows 22308 through 31800, so every later row is already exposed.
- The requested four-arm Kronos/Chronos experiment must not report an untouched final holdout result
  from this dataset. It remains blocked until a genuinely new, native Nobitex dataset window is
  obtained, validated, frozen, and committed by hash before any model sees it.

A future eligible commitment must include a new CSV and manifest SHA-256, exact row/time boundaries,
zero unresolved gaps, a recorded commit date, and evidence that no model, researcher, tuning process
or earlier evaluation has observed the final segment.

## 5. Safety

No new API calls were made to create this commitment; the dataset was recovered from the public-data
artifact of run #41. No private API or order endpoint was used.

- Champion=null
- order_permission=false
- trading_authorization=NO-GO
- ALIX_DRY_RUN=true
- ALIX_ALLOW_LIVE_TRADING=false
- private_api_calls=0
- real_order_attempts=0
