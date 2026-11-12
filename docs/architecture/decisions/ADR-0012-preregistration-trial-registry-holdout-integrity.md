# ADR-0012: Pre-registration, Historical Trial Accounting and Holdout Integrity

- Status: Accepted.
- Date: 2026-11-12.
- Active branch: `research/nobitex-hf-strategy`.
- Decision commit parent: `bc493e24b9e3f849b6d9fbd0b776d15b39b8061f`.
- Frozen main: `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb` (untouched).
- Related owner instruction: AliX post-governance execution directive, 2026-11-12.
- Artifacts:
  - `docs/research/preregistration/TEMPLATE.md`
  - `docs/research/TRIAL_REGISTRY.md`
  - `docs/research/HOLDOUT_COMMITMENT.md`

## Context

Prior experiments span HF proxy attempts, V4-T candidate variants, V4-AP cells, AI-6 PPO
policies, and DEAP genetic-programming genomes. Workflow run counts are not identical to
candidate-variant counts, and a failed/cancelled workflow may have performed partial evaluation.
A global multiple-testing count must therefore not be inferred by adding unlike quantities.

The 31,801-row native Nobitex 1h CSV is verified at SHA-256
`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.
The final 12.5% segment of this file is not fresh: AI-6 evaluated it as its final holdout, and
DEAP's later OOS segment also includes it. Hashing that segment cannot make it untouched again.

## Decisions

1. Every future training, tuning, model comparison, sweep or economic evaluation must have an
   immutable preregistration before its first evaluation. The template defines the hypothesis,
   target, horizon, timeframe, source data + SHA, chronological split, purge/embargo, cost-50 and
   cost-90, absolute gates, seed, planned variant count, model weight revision/hash, warm-up,
   as-of alignment, and pretraining-corpus overlap disclosure.
2. The historical trial registry separates execution attempts from actual candidate evaluations.
   It records exact known counts and IDs, preserves all workflow run IDs/commit SHAs, and writes
   UNKNOWN plus a lower bound wherever candidate-level history cannot be reconstructed.
3. No guessed aggregate N is permitted. A global exact N remains UNKNOWN until failed/cancelled
   workflow runs and candidate-level ledgers are reconciled. Candidate IDs must be unique within
   each future preregistered family.
4. The previously used AI-6/DEAP final slice is fingerprinted for contamination tracking but is
   explicitly rejected as a new holdout. The current native dataset has no demonstrably untouched
   final holdout.
5. The four-arm Kronos/Chronos economic experiment may not claim an untouched final-holdout result
   from the existing 31,801-row dataset. It is blocked pending a genuinely new native Nobitex window
   with a new immutable manifest/hash and pre-exposure commitment. No alternate split of the same
   already-exposed rows may be relabeled as fresh holdout.
6. The Bybit work in this commit is limited to a diagnostic public Spot Kline probe. A complete
   fetcher is prohibited until the runner probe returns HTTP 200 / API success and confirms every
   candle in the exact 2023-03-24 12:30–14:00 UTC 15m grid. No Binance/Bybit data splice, gap
   masking, interpolation, backfill or window shifting is allowed.

## Consequences

- DSR/PBO calculations for new cohorts must include all preregistered variants and any related
  historical trial IDs in the family; UNKNOWN is a blocking evidence status, not zero.
- The current holdout commitment status is `BLOCKED_CONTAMINATED`.
- No new four-arm training is launched in this stage.
- A Bybit probe result will be recorded separately in a follow-up ADR, including HTTP status,
  returned record counts, timestamp bounds and exact-window completeness.
- No change is made to the frozen `main` branch or to any safety permission.

## Safety invariants

- Champion=null
- order_permission=false
- trading_authorization=NO-GO
- ALIX_DRY_RUN=true
- ALIX_ALLOW_LIVE_TRADING=false
- private_api_calls=0
- real_order_attempts=0
