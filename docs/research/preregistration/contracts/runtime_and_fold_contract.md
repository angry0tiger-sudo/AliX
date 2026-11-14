# Four-Arm Runtime, Fold, Signal and Execution Contract — ADR-038

**Registration:** NOBITEX-FOUR-ARM-KC-20261113 / ADR-038 amendment  
**Scope:** Nobitex BTCUSDT Spot 1h pre-hack research only; no post-hack inputs.

## 1. Model inputs and fixed adapters

- Kronos-Small: NeoQuasar/Kronos-small with separately pinned NeoQuasar/Kronos-Tokenizer-base nested at models.kronos_small.tokenizer.
- Chronos-2: autogluon/chronos-2-small; Chronos-T5 is not substituted.
- Zero-shot CPU inference only; no fine-tuning or parameter updates.
- Common context: latest 512 completed hourly candles through decision close (t).
- Forecast horizon: next 24 hourly bars; refresh cadence: every 24 bars.
- Kronos output: predicted 24th-step close divided by decision close minus one.
- Chronos output: median (0.5 quantile) at step 24 divided by decision close minus one.
- Arm B: Kronos return strictly greater than the candidate threshold.
- Arm C: Chronos-2 return strictly greater than the candidate threshold.
- Arm D: arithmetic mean of the two percentage returns strictly greater than the candidate threshold.
- Arm A: Donchian-20 close breakout control, independent of forecast threshold.
- B/C/D are independent forecast-entry rules. No baseline-and-filter combination is permitted.

## 2. Registered candidate cells

The only thresholds are 120, 180 and 240 bps. The twelve IDs are:
- NOBITEX-KC26-FD-120-A/B/C/D
- NOBITEX-KC26-FD-180-A/B/C/D
- NOBITEX-KC26-FD-240-A/B/C/D

Each cell is counted in the registered trial cohort. Arm A's repeated signal is computed once and referenced in its three cells; this repetition is explicitly disclosed. Earlier NOBITEX-KC26-A/B/C/D IDs remain preserved as superseded before evaluation.

## 3. Confirmatory data and forecast cache

- Source CSV: 31,801 rows, exact SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e.
- Confirmatory prefix: first 27,825 rows only.
- Exclude the final 3,976 rows from every confirmatory purpose, including ATR, features, model context, inference, folds, thresholds, trades, and metrics.
- Build one immutable forecast cache across the entire eligible prefix after 512 context bars are available.
- Each model is invoked once per forecast anchor on the 24-bar refresh cadence. All folds and all threshold cells reuse the same cache.
- Cache identity binds source SHA, confirmatory row count, model revisions, weight digests, forecast horizon, context, refresh cadence, adapter/contract fingerprints and deterministic runtime.
- Missing/duplicate anchors, non-finite forecasts, cache digest mismatch or incomplete cache fail closed. Partial cache reuse is forbidden.

## 4. Deterministic chronological folds

For N=27,825, prefix endpoints are 19,477; 23,651; 27,825. Each prefix uses expanding 70/15/15 chronological train/validation/test segments with 24-bar purge/embargo on both sides of each boundary. The exact integer ranges are emitted in every manifest. No shuffle is used.

## 5. Common risk and execution mechanics

- Shared risk for all arms and thresholds: Wilder ATR(14), 2×ATR initial stop and trailing distance, 24-bar maximum holding, MaxDD gate <=15%.
- Entries occur at the next candle open after the signal close.
- Trailing stop ratchets only after a completed bar and is active from the next bar.
- Gap-through stops/targets fill at the open; same-candle dual touches use conservative stop-first handling.
- One open long position per arm, no pyramiding, leverage or short selling.
- Round-trip costs are exactly 50 and 90 bps, reported separately and applied once.

## 6. Statistical and economic decision

- Aggregate and every one of three OOS folds must meet NetPF >=1.10 at both costs and MaxDD <=15%; aggregate trades >=100.
- GLOBAL_N=UNKNOWN; DSR remains null and no PASS/winner is possible without defensible DSR >0.
- Report the audited historical lower bound, unresolved remainder, and 1×/2×/5×/10× sensitivity without inventing a numeric DSR.
- Fewer than 100 trades is INSUFFICIENT_SAMPLE.

## 7. Safety

- The runner validates the owner execution lock and safety environment before reading the source CSV, models or weight artifacts.
- Dataset source is pre-hack Git path only. Holdout files, V2 and Spec v3 are not inputs.
- Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0, ALIX_DRY_RUN=true, ALIX_ALLOW_LIVE_TRADING=false.
