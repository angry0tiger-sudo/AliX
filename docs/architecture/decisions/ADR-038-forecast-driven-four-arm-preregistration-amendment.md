# ADR-038: Pre-execution amendment — forecast-driven four-arm experiment

- **Status:** Accepted for preregistration amendment only; execution remains NOT_AUTHORIZED
- **Date:** 2026-11-14 UTC
- **Supersedes:** the signal-entry and trial-family portions of ADR-032 before evaluation; ADR-032 remains preserved
- **Scope:** AliX / Nobitex BTCUSDT Spot 1h / pre-hack research only
- **Frozen main:** \`44bcaeae7eefd358cf063394b0e7f4f0b83af8eb\` (untouched)
- **Research branch:** \`research/nobitex-hf-strategy\`
- **Base SHA:** \`0427d9943232ae82d218909859f4709db96bd57c\`

## Context and decision

Historical owner-provided evidence motivating this prospective amendment:

- Baseline Profit Factor observations: **0.3743 / 0.4370**.
- AI-6 produced **zero trades**.
- TTM PF changed from **0.3731 to 0.3748**, remaining below break-even.
- Meta-labeling performed worse than its teacher.
- Under MRD v0.9 §13.8, filtering a negative-expectancy teacher cannot create positive expectancy from the teacher's existing signal.

No result from the amended arms or amended thresholds has been viewed. This is a pre-evaluation redesign, not a post-result selection or p-hacking exercise.

**Decision:** replace the filter-based model treatment with a prospective forecast-driven directional family. Preserve the old trial IDs and mark them superseded; never delete their records.

## Revised arms and common risk contract

| Arm | Entry definition |
|---|---|
| A | Donchian-20 close breakout control |
| B | Entry from pinned Kronos-Small 24-hour forecast |
| C | Entry from pinned Chronos-2 24-hour forecast |
| D | Entry from arithmetic mean of the two model forecasts |

Common constraints for every arm and threshold:

- ATR(14) is the common volatility/risk measure.
- Trailing stop distance is 2 × ATR.
- Maximum holding period is 24 hourly bars.
- Maximum drawdown gate is **≤15%**.
- Preserve the registered 50 bps and 90 bps round-trip cost scenarios.
- No arm-specific risk, sizing, fill, or safety bypass is permitted.
- Model repositories, immutable revisions and weight SHA-256 values remain pinned in \`config/four_arm/model_pins.json\`.

The earlier threshold expression \`max(2 × 90 bps, ATR(14)/close)\` is superseded. The owner-provided BTC 1h ATR/close expectation of approximately 30–80 bps implies that expression is effectively fixed near 180 bps in ordinary conditions and risks an under-supported sample. It is not retained as the sole threshold.

## Independent threshold sensitivity and trial accounting

Pre-register exactly three fixed forecast thresholds: **120 bps, 180 bps, and 240 bps**. Each threshold is a separate candidate cell and a separate trial. Register every arm × threshold cell, including the repeated Donchian control, so the threshold choice is not hidden from DSR accounting.

| Threshold | Arm A | Arm B | Arm C | Arm D |
|---:|---|---|---|---|
| 120 bps | NOBITEX-KC26-FD-120-A | NOBITEX-KC26-FD-120-B | NOBITEX-KC26-FD-120-C | NOBITEX-KC26-FD-120-D |
| 180 bps | NOBITEX-KC26-FD-180-A | NOBITEX-KC26-FD-180-B | NOBITEX-KC26-FD-180-C | NOBITEX-KC26-FD-180-D |
| 240 bps | NOBITEX-KC26-FD-240-A | NOBITEX-KC26-FD-240-B | NOBITEX-KC26-FD-240-C | NOBITEX-KC26-FD-240-D |

Arm A is the same Donchian control repeated as the matched control in each threshold cell. That repetition is disclosed, not represented as a distinct signal rule. The earlier IDs \`NOBITEX-KC26-A/B/C/D\` remain in the registry with status **SUPERSEDED_BEFORE_EVALUATION**.

No additional threshold, model revision, ensemble weight, feature rule, seed variant, or entry filter may be tried without a new append-only registration before evaluation.

## Forecast-cache architecture requirement

Forecast computation MUST be separated from fold evaluation and threshold evaluation:

1. Build one immutable forecast cache over the eligible confirmatory prefix, after the common 512-bar context is available.
2. Invoke each pinned model only once per forecast anchor using the 24-bar refresh cadence.
3. All folds, thresholds and arms B/C/D read the same cached forecasts. They MUST NOT call the models again.
4. Cache identity must include dataset SHA-256, confirmatory row count, both model revisions and weight digests, forecast horizon, context length, refresh cadence, code/adapter fingerprints, and deterministic runtime configuration.
5. A mismatch, missing or duplicate anchor, non-finite value, or unverifiable cache file invalidates the run and fails closed. Partial caches must not be silently reused.
6. Expected workload is approximately **1,140 forecast anchors** (one inference per model per anchor), not repeated fold/threshold recomputation approaching 10,000 calls.

This cache is a mandatory architecture gate, not an optional optimization. The repair refactor does not claim that the future cache or revised forecast-to-entry rules have already been implemented. The execution lock must remain NOT_AUTHORIZED until those implementation contracts, fingerprints and mock-only tests are complete.

## Confirmatory dataset boundary

The immutable source remains the 31,801-row pre-hack dataset with SHA-256:

\`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e\`

For confirmatory evaluation:

- Use only the first **27,825 rows**.
- Exclude the final **3,976 rows** from every confirmatory purpose: signal/feature construction, scaling, model context, forecast inference, fold assignment, threshold selection, trade generation, cache generation, and metrics.
- Keep the source CSV immutable. This amendment does not authorize rewriting or relabelling source data.
- The post-hack holdout remains sealed and inaccessible.

## Statistical accounting

- \`GLOBAL_N = UNKNOWN\` remains.
- DSR is mandatory, not waived.
- Report the auditable historical lower bound, unresolved remainder, and sensitivity at the defensible lower bound and at **2×, 5×, and 10×** that bound.
- Include historical trials and all registered threshold cells that reach candidate evaluation.
- If the unresolved remainder prevents a defensible \`DSR > 0\` conclusion, the result is **INSUFFICIENT_EVIDENCE / NO-GO**. Do not guess global N.

## Predeclared alternative and stopping rule

The only documented model advantage currently recorded is volatility forecasting: TTM-R2 QLIKE **0.27786** versus **0.39348**, a **29.38% lower** QLIKE value.

- If all four directional arms fail their economic/statistical gates at all three thresholds, declare **NO-GO for the directional method family**. Do not continue directional threshold tuning.
- One separately preregistered volatility-oriented strategy pivot is permitted, based on the stated volatility-forecasting evidence.
- If that single volatility-oriented pivot also fails its own preregistered gates, close the project with a documented **NO-GO**. No further path is authorized by this ADR.

## Authorization and safety

This ADR authorizes only the prospective preregistration amendment. It does not authorize model acquisition, model loading, inference, backtesting, economic evaluation, paper orders, or live orders in this turn.

- \`authorization_status = NOT_AUTHORIZED\`
- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- \`private_api_calls = 0\`
- \`real_order_attempts = 0\`
- \`posthack_holdout_access = FORBIDDEN\`

ADR-032, its trial identifiers, prior results, and all previous registry entries remain preserved.
