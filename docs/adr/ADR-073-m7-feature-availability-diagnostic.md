# ADR-073 — M7 Feature Availability Diagnostic and G3.0 Feature-Only Supersession

- **Status:** DIAGNOSTIC COMPLETE — G2.0 CLOSED NO-GO; G3.0 FEATURE-ONLY PREREGISTRATION AND EVALUATION IN PROGRESS
- **Date:** 2026-11-27
- **Branch:** \`research/nobitex-hf-strategy\`
- **Data scope:** Nobitex public Spot PRE-HACK only
- **Supersedes:** No previous ADR is erased. This record refines the G2.0 root cause and authorizes the single feature-only G3.0 correction.

## 1. Executive finding

The M7 G2.0 empty-panel cause is conclusively isolated to \`relative_volume_shock_24h\`. Across all nine exact-history-qualified symbols, beta was fully available, while the relative-volume feature was rejected on every decision row because its prior-24h median base-asset volume was nonpositive.

- \`beta_24h_to_btc\`: **0 / 7,943 rejected (0.0000%) per symbol**.
- \`relative_volume_shock_24h\`: **7,943 / 7,943 rejected (100.0000%) per symbol**.
- Rejection reason: \`prior_24h_median_volume_nonpositive\`, 7,943 occurrences per symbol.
- Both-suspect-features rejected together: 0 rows.
- Beta-only rejected: 0 rows.
- Volume-only rejected: 7,943 rows per symbol.
- Other registered features or target invalid: 0 rows.

This is a source-data/feature compatibility failure, not a beta calculation failure and not an economic-performance measurement.

## 2. Immutable diagnostic execution evidence

- **Workflow:** Temporary M7 Feature Diagnostic Runner #10
- **Run ID:** \`46057363991\`
- **Run URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46057363991
- **Commit SHA:** \`84de77c643cd4ee0c6034914b72303e83fb72f39\`
- **Exact Standard CI parent:** CI #2685, Run ID \`46057363918\`, successful
- **Diagnostic artifact:** \`alix-m7-feature-availability-diagnostic\`
- **Artifact ID:** \`15227458008\`
- **Artifact ZIP SHA-256:** \`4d2769bf26e5832779c5ba87de34f23d58ee1b57d8f322deea048a8930548aa5\`
- **Artifact URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46057363991/artifacts/15227458008
- **G2.0 preregistration SHA-256:** \`e5a1dfdfa92912635c6949934f353f2aff89c9a26cbc5e384de89a5d0d488817\`
- **Qualification manifest SHA-256:** \`9e445e649ea0706a47677f6d65607b9f7e50588b51b63c4e9140e350dab2abe4\`
- **BTCUSDT source SHA-256:** \`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e\`

The runner checked out the exact diagnostic commit after its Standard CI passed, loaded only the immutable nine-pair PRE-HACK dataset and BTCUSDT context series, and ran no model fitting, economic backtest, or new data acquisition.

## 3. Per-symbol feature rejection results

There were 7,943 four-hour decision timestamps per symbol on the 1h input grid. Every qualified symbol returned exactly the same feature-availability counts:

| Symbol | Beta rejected | Beta rejection | Volume-shock rejected | Volume rejection |
|---|---:|---:|---:|---:|
| ETHUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| BNBUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| ADAUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| XRPUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| DOGEUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| TRXUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| LTCUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| LINKUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |
| DOTUSDT | 0 / 7,943 | 0.0000% | 7,943 / 7,943 | 100.0000% |

### 3.1 Exact volume rejection cause

For each of the nine assets, the reason counters over all 7,943 decisions were:

- \`prior_24h_median_volume_nonpositive\`: **7,943**
- \`prior_24h_median_volume_nonfinite\`: 0
- \`current_completed_hour_volume_invalid\`: 0
- \`volume_helper_returned_none\`: 0
- \`volume_shock_output_nonfinite\`: 0

The input-data integrity check requires OHLCV volume to be finite and nonnegative. Given that invariant, a nonpositive rolling median means the prior-24h median volume was zero at every registered decision timestamp. The diagnostic does not claim that every individual hourly volume observation across the complete CSV was zero; it establishes that every sampled trailing 24h median was zero.

The beta feature had no invalid rolling return windows, no invalid BTC variance, and no nonfinite beta result on any decision row.

## 4. Candidate-panel impact

| Candidate | Rows before qualification | Beta rejections | Volume rejections | Other feature/target invalid | Accepted rows | Total rejection |
|---|---:|---:|---:|---:|---:|---:|
| 5PAIR G2.0 | 39,715 | 0 | 39,715 | 0 | 0 | 100.0000% |
| 9PAIR G2.0 | 71,487 | 0 | 71,487 | 0 | 0 | 100.0000% |

Thus \`relative_volume_shock_24h\` alone accounts for the complete panel collapse. No other registered field or target caused a row rejection.

## 5. Decision and feature recommendation

**Decision:** retain \`beta_24h_to_btc\` unchanged. Replace only \`relative_volume_shock_24h\` with a price-derived 24-hour log return:

\`log_return_24h = ln(close[t] / close[t-24])\`

This uses the target asset's last 24 completed hourly closes ending at the decision hour. It is point-in-time, requires positive validated close prices, and does not depend on the zero-valued base-asset volume series. It is preferred over removing the fourth feature because the preregistration maintains four fixed features and this alternative preserves a simple, interpretable 24h asset-momentum signal. No volatility sweep, alternative feature sweep, or model enlargement is authorized.

## 6. G3.0 superseding preregistration boundary

The superseding revision is \`M7-CS-G3.0\`. The **only change to the predictive feature contract** is:

- Remove \`relative_volume_shock_24h\`.
- Add \`log_return_24h\` with the formula above.

Everything else remains fixed: the same two candidate universes and trial identities, same nine qualified symbols, same BTCUSDT context source, same 1h input cadence, 4h forward target and holding horizon, StandardScaler + Ridge(alpha=1.0), long-only top-two selection, weights, folds, purge/embargo, Cost-50/Cost-90 assumptions, economic gates, and safety rules.

G2.0 remains an archived diagnostic NO-GO. It is not relabelled as a pass and no G2.0 model was fitted. G3.0 must pass its own exact-parent Standard CI before the M7 evaluation workflow is triggered.

## 7. Safety and access invariants

The diagnostic report explicitly records:

- \`new_data_acquisition = false\`
- \`model_fitting = false\`
- \`economic_backtest = false\`
- \`holdout_bytes_read = false\`
- \`posthack_data_read = false\`
- \`private_api_calls = 0\`
- \`paper_orders = 0\`
- \`real_order_attempts = 0\`
- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- \`local_execution = false\`

No paid service or wallet was used. No TinyFish workflow was run. No changes were made to \`main\`. No Holdout/Post-Hack records were read.
