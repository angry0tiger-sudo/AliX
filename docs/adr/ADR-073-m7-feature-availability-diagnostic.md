# ADR-073 — M7 Feature Availability Diagnostic and G3.0 Feature-Only Supersession

- **Status:** FINAL — G2.0 FEATURE_QUALITY NO-GO; G3.0 ECONOMIC NO-GO
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Data scope:** Nobitex public Spot PRE-HACK only
- **Supersedes:** No previous ADR is erased. This record refines the G2.0 root cause and authorizes the single feature-only G3.0 correction.

## 1. Executive finding

The M7 G2.0 empty-panel cause is conclusively isolated to `relative_volume_shock_24h`. Across all nine exact-history-qualified symbols, beta was fully available, while the relative-volume feature was rejected on every decision row because its prior-24h median base-asset volume was nonpositive.

- `beta_24h_to_btc`: **0 / 7,943 rejected (0.0000%) per symbol**.
- `relative_volume_shock_24h`: **7,943 / 7,943 rejected (100.0000%) per symbol**.
- Rejection reason: `prior_24h_median_volume_nonpositive`, 7,943 occurrences per symbol.
- Both-suspect-features rejected together: 0 rows.
- Beta-only rejected: 0 rows.
- Volume-only rejected: 7,943 rows per symbol.
- Other registered features or target invalid: 0 rows.

This is a source-data/feature compatibility failure, not a beta calculation failure and not an economic-performance measurement.

## 2. Immutable diagnostic execution evidence

- **Workflow:** Temporary M7 Feature Diagnostic Runner #10
- **Run ID:** `46057363991`
- **Run URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46057363991
- **Commit SHA:** `84de77c643cd4ee0c6034914b72303e83fb72f39`
- **Exact Standard CI parent:** CI #2685, Run ID `46057363918`, successful
- **Diagnostic artifact:** `alix-m7-feature-availability-diagnostic`
- **Artifact ID:** `15227458008`
- **Artifact ZIP SHA-256:** `4d2769bf26e5832779c5ba87de34f23d58ee1b57d8f322deea048a8930548aa5`
- **Artifact URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46057363991/artifacts/15227458008
- **G2.0 preregistration SHA-256:** `e5a1dfdfa92912635c6949934f353f2aff89c9a26cbc5e384de89a5d0d488817`
- **Qualification manifest SHA-256:** `9e445e649ea0706a47677f6d65607b9f7e50588b51b63c4e9140e350dab2abe4`
- **BTCUSDT source SHA-256:** `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`

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

- `prior_24h_median_volume_nonpositive`: **7,943**
- `prior_24h_median_volume_nonfinite`: 0
- `current_completed_hour_volume_invalid`: 0
- `volume_helper_returned_none`: 0
- `volume_shock_output_nonfinite`: 0

The input-data integrity check requires OHLCV volume to be finite and nonnegative. Given that invariant, a nonpositive rolling median means the prior-24h median volume was zero at every registered decision timestamp. The diagnostic does not claim that every individual hourly volume observation across the complete CSV was zero; it establishes that every sampled trailing 24h median was zero.

The beta feature had no invalid rolling return windows, no invalid BTC variance, and no nonfinite beta result on any decision row.

## 4. Candidate-panel impact

| Candidate | Rows before qualification | Beta rejections | Volume rejections | Other feature/target invalid | Accepted rows | Total rejection |
|---|---:|---:|---:|---:|---:|---:|
| 5PAIR G2.0 | 39,715 | 0 | 39,715 | 0 | 0 | 100.0000% |
| 9PAIR G2.0 | 71,487 | 0 | 71,487 | 0 | 0 | 100.0000% |

Thus `relative_volume_shock_24h` alone accounts for the complete panel collapse. No other registered field or target caused a row rejection.

## 5. Decision and feature recommendation

**Decision:** retain `beta_24h_to_btc` unchanged. Replace only `relative_volume_shock_24h` with a price-derived 24-hour log return:

`log_return_24h = ln(close[t] / close[t-24])`

This uses the target asset's last 24 completed hourly closes ending at the decision hour. It is point-in-time, requires positive validated close prices, and does not depend on the zero-valued base-asset volume series. It is preferred over removing the fourth feature because the preregistration maintains four fixed features and this alternative preserves a simple, interpretable 24h asset-momentum signal. No volatility sweep, alternative feature sweep, or model enlargement is authorized.

## 6. G3.0 superseding preregistration boundary

The superseding revision is `M7-CS-G3.0`. The **only change to the predictive feature contract** is:

- Remove `relative_volume_shock_24h`.
- Add `log_return_24h` with the formula above.

Everything else remains fixed: the same two candidate universes and trial identities, same nine qualified symbols, same BTCUSDT context source, same 1h input cadence, 4h forward target and holding horizon, StandardScaler + Ridge(alpha=1.0), long-only top-two selection, weights, folds, purge/embargo, Cost-50/Cost-90 assumptions, economic gates, and safety rules.

G2.0 remains an archived diagnostic NO-GO. It is not relabelled as a pass and no G2.0 model was fitted. G3.0 must pass its own exact-parent Standard CI before the M7 evaluation workflow is triggered.

## 7. Safety and access invariants

The diagnostic report explicitly records:

- `new_data_acquisition = false`
- `model_fitting = false`
- `economic_backtest = false`
- `holdout_bytes_read = false`
- `posthack_data_read = false`
- `private_api_calls = 0`
- `paper_orders = 0`
- `real_order_attempts = 0`
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `local_execution = false`

No paid service or wallet was used. No TinyFish workflow was run. No changes were made to `main`. No Holdout/Post-Hack records were read.


## 8. G3.0 economic evaluation — final result

The feature-only G3.0 preregistration was evaluated after Standard CI passed on the exact implementation/config parent commit.

- **Evaluation workflow:** M7 Cross-Sectional Ridge Economic Evaluation #4
- **Run ID:** 46060086225
- **Run URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225
- **Evaluation trigger commit:** d1fa26fae926fb8d2ac5e4424fe6e6af532fda3f
- **Exact implementation/config parent:** ba8977e86b2414674fc30e67bdeb6a97b357fa65
- **Exact-parent Standard CI:** #2689, Run ID 46059360239, **SUCCESS**
- **Evaluation result:** **SUCCESSFUL EXECUTION — ECONOMIC NO-GO**
- **Artifact:** alix-m7-cross-sectional-results
- **Artifact ID:** 15229065411
- **Artifact ZIP SHA-256:** a1e5a737a7015a58cd6a2c77ed69bf6e3125c3ed4ec04b779d33764148ee23f0
- **Artifact URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225/artifacts/15229065411

### 8.1 Final OOS metrics

All metrics below come from the completed G3.0 PRE-HACK report. **Trades** is the completed OOS trade count; **MaxDD** is the compounded chronological OOS NAV drawdown. Both cost scenarios use the same completed OOS trade set per candidate.

| Candidate | Cost | Completed OOS trades | NetPF | MaxDD | Valid folds | Stable folds | Economic gates |
|---|---:|---:|---:|---:|---:|---:|---|
| 5PAIR | 50 bps | 4,565 | 0.503741 | 99.997453% | 4/4 | 0/4 | FAIL |
| 5PAIR | 90 bps | 4,565 | 0.262960 | 99.9999997% | 4/4 | 0/4 | FAIL |
| 9PAIR | 50 bps | 5,959 | 0.424442 | 99.999872% | 4/4 | 0/4 | FAIL |
| 9PAIR | 90 bps | 5,959 | 0.193591 | 99.9999999992% | 4/4 | 0/4 | FAIL |

Exact report values:

- 5PAIR Cost-50: NetPF 0.5037411959543473, MaxDD 0.9999745341863353, net PnL fraction -0.9999744099771347.
- 5PAIR Cost-90: NetPF 0.26296047707462983, MaxDD 0.9999999973844268, net PnL fraction -0.9999999973844296.
- 9PAIR Cost-50: NetPF 0.424441886855759, MaxDD 0.9999987218407761, net PnL fraction -0.9999987218407831.
- 9PAIR Cost-90: NetPF 0.19359069717494629, MaxDD 0.9999999999920927, net PnL fraction -0.9999999999920951.

### 8.2 Gate analysis

The panel is no longer empty in G3.0:

- 5PAIR panel: **39,715 rows**.
- 9PAIR panel: **71,487 rows**.
- The locked model completed four chronological walk-forward fits for each candidate.
- All four folds met the minimum trade-count validity condition for each candidate/cost scenario.
- **Zero of four folds were stable** for every candidate/cost scenario.
- Every NetPF is below 1.00, and therefore below the preregistered minimum of 1.10.
- Every MaxDD is far above the maximum permitted 15%.
- Although OOS trade counts exceed the 100-trade minimum, that alone cannot satisfy the family acceptance rule.

The economic acceptance field is false for both candidates at both cost levels. The registered overall family verdict is **NO-GO**.

### 8.3 Final decision

**M7 G3.0: NO-GO.** The feature correction repaired panel availability and allowed the locked model to be evaluated, but the Cross-Sectional Ridge strategy did not demonstrate a viable economic edge. Both candidate universes lost essentially all initial NAV over the pooled OOS path, even at the lower 50 bps round-trip cost. The 90 bps scenario was worse.

This is now a measured negative economic result, not an empty-panel or insufficient-data result. No additional feature/model/alpha/universe/threshold sweep is authorized by this ADR. G2.0 remains archived as a feature-availability NO-GO; G3.0 is recorded separately as an economic NO-GO.

### 8.4 Final safety and repository state

The evaluation report confirms PRE-HACK scope only, with no Holdout or Post-Hack bytes read, zero private API calls, zero paper or real order attempts, Champion = null, order_permission = false, and trading_authorization = NO-GO.

No paid service, wallet, or TinyFish was used. The main branch remained unchanged; its protected tree SHA is still 5761c6c34143f8e6924f53a4e745c48932a5f791. The temporary push-based diagnostic runner is no longer required after the successful diagnostic artifact and is removed from the research branch; the dedicated workflow-dispatch-only diagnostic definition remains retained.


## 8. G3.0 feature-only economic evaluation

The feature-only G3.0 preregistration replaced the unavailable volume-shock feature with the natural-log 24-hour close return: log_return_24h = ln(close[t] / close[t-24]). No other model, policy, universe, target, fold, cost, or gate parameter changed.

- **Preregistration revision:** M7-CS-G3.0
- **Preregistration SHA-256:** 2fe9ae965565c8ee7fc17f6f92bc6d5774566900fc22ef2f09b8d73f00308527
- **Implementation/config parent:** ba8977e86b2414674fc30e67bdeb6a97b357fa65
- **Exact-parent Standard CI:** [Run #2689 / ID 46059360239](https://github.com/angry0tiger-sudo/AliX/actions/runs/46059360239), SUCCESS
- **Evaluation workflow:** [M7 Cross-Sectional Ridge Economic Evaluation Run #4 / ID 46060086225](https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225), SUCCESS for execution and report validation
- **Evaluation trigger commit:** d1fa26fae926fb8d2ac5e4424fe6e6af532fda3f
- **Artifact:** alix-m7-cross-sectional-results, ID 15229065411
- **Artifact ZIP SHA-256:** a1e5a737a7015a58cd6a2c77ed69bf6e3125c3ed4ec04b779d33764148ee23f0
- **Artifact URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225/artifacts/15229065411

Both candidate panels were fully populated and four fold models were fitted for each candidate. The panel row counts were 39,715 for 5PAIR and 71,487 for 9PAIR; each candidate had fit_count=4. All four folds were valid, but zero folds were stable for either candidate.

### 8.1 Economic metrics

| Candidate | Cost | NetPF | MaxDD | Completed OOS trades | Valid / stable folds | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 5PAIR G3.0 | 50 bps | 0.503741 | 99.997453% | 4,565 | 4/4 / 0/4 | NO-GO |
| 5PAIR G3.0 | 90 bps | 0.262960 | 99.9999997% | 4,565 | 4/4 / 0/4 | NO-GO |
| 9PAIR G3.0 | 50 bps | 0.424442 | 99.999872% | 5,959 | 4/4 / 0/4 | NO-GO |
| 9PAIR G3.0 | 90 bps | 0.193591 | 99.999999999% | 5,959 | 4/4 / 0/4 | NO-GO |

### 8.2 Economic gate interpretation

- The minimum trade-count gate (100) passed for both candidates.
- The minimum valid-fold gate (3/4) passed with 4/4 valid folds.
- The NetPF gate (>=1.10) failed under both cost scenarios for both candidates.
- The MaxDD gate (<=15%) failed severely under both cost scenarios.
- The stable-fold gate (>=3/4) failed with 0/4 stable folds.

The family-level G3.0 verdict is NO-GO. Unlike G2.0, this is a measured economic failure after the panel was populated and the four walk-forward models were fitted. The near-total drawdowns are observed backtest metrics under the preregistered accounting rules; they are not live or paper-trading results.

## 9. Final M7 decision and safety state

**M7 G2.0:** BLOCKED_DATA_OR_FEATURE_QUALITY / NO-GO; retained as the failed feature-availability revision, with no model fit.

**M7 G3.0:** ECONOMIC NO-GO; both candidates were fully evaluated and failed the NetPF, MaxDD and stable-fold gates. The 24h log-return feature resolved the empty-panel problem but did not produce an economic edge.

No further feature or threshold tuning is authorized by this ADR. Any new strategy or feature hypothesis requires a new prospective preregistration before outcomes are inspected.

The evaluation report confirms new_data_acquisition_during_evaluation=false, holdout_bytes_read=false, posthack_data_read=false, private_api_calls=0, paper_orders=0, and real_order_attempts=0. Champion=null, order_permission=false, and trading_authorization=NO-GO remain unchanged. The 1h cadence is the input sampling interval; the only registered prediction target remains the 4h forward return.


## 8. G3.0 feature-only economic evaluation

The feature-only G3.0 preregistration replaced the unavailable volume-shock feature with the natural-log 24-hour close return: log_return_24h = ln(close[t] / close[t-24]). No other model, policy, universe, target, fold, cost, or gate parameter changed.

- **Preregistration revision:** M7-CS-G3.0
- **Preregistration SHA-256:** 2fe9ae965565c8ee7fc17f6f92bc6d5774566900fc22ef2f09b8d73f00308527
- **Implementation/config parent:** ba8977e86b2414674fc30e67bdeb6a97b357fa65
- **Exact-parent Standard CI:** [Run #2689 / ID 46059360239](https://github.com/angry0tiger-sudo/AliX/actions/runs/46059360239), SUCCESS
- **Evaluation workflow:** [M7 Cross-Sectional Ridge Economic Evaluation Run #4 / ID 46060086225](https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225), SUCCESS for execution and report validation
- **Evaluation trigger commit:** d1fa26fae926fb8d2ac5e4424fe6e6af532fda3f
- **Artifact:** alix-m7-cross-sectional-results, ID 15229065411
- **Artifact ZIP SHA-256:** a1e5a737a7015a58cd6a2c77ed69bf6e3125c3ed4ec04b779d33764148ee23f0
- **Artifact URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/46060086225/artifacts/15229065411

Both candidate panels were fully populated and four fold models were fitted for each candidate. The panel row counts were 39,715 for 5PAIR and 71,487 for 9PAIR; each candidate had fit_count=4. All four folds were valid, but zero folds were stable for either candidate.

### 8.1 Economic metrics

| Candidate | Cost | NetPF | MaxDD | Completed OOS trades | Valid / stable folds | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 5PAIR G3.0 | 50 bps | 0.503741 | 99.997453% | 4,565 | 4/4 / 0/4 | NO-GO |
| 5PAIR G3.0 | 90 bps | 0.262960 | 99.9999997% | 4,565 | 4/4 / 0/4 | NO-GO |
| 9PAIR G3.0 | 50 bps | 0.424442 | 99.999872% | 5,959 | 4/4 / 0/4 | NO-GO |
| 9PAIR G3.0 | 90 bps | 0.193591 | 99.999999999% | 5,959 | 4/4 / 0/4 | NO-GO |

### 8.2 Economic gate interpretation

- The minimum trade-count gate (100) passed for both candidates.
- The minimum valid-fold gate (3/4) passed with 4/4 valid folds.
- The NetPF gate (>=1.10) failed under both cost scenarios for both candidates.
- The MaxDD gate (<=15%) failed severely under both cost scenarios.
- The stable-fold gate (>=3/4) failed with 0/4 stable folds.

The family-level G3.0 verdict is NO-GO. Unlike G2.0, this is a measured economic failure after the panel was populated and the four walk-forward models were fitted. The near-total drawdowns are observed backtest metrics under the preregistered accounting rules; they are not live or paper-trading results.

## 9. Final M7 decision and safety state

**M7 G2.0:** BLOCKED_DATA_OR_FEATURE_QUALITY / NO-GO; retained as the failed feature-availability revision, with no model fit.

**M7 G3.0:** ECONOMIC NO-GO; both candidates were fully evaluated and failed the NetPF, MaxDD and stable-fold gates. The 24h log-return feature resolved the empty-panel problem but did not produce an economic edge.

No further feature or threshold tuning is authorized by this ADR. Any new strategy or feature hypothesis requires a new prospective preregistration before outcomes are inspected.

The evaluation report confirms new_data_acquisition_during_evaluation=false, holdout_bytes_read=false, posthack_data_read=false, private_api_calls=0, paper_orders=0, and real_order_attempts=0. Champion=null, order_permission=false, and trading_authorization=NO-GO remain unchanged. The 1h cadence is the input sampling interval; the only registered prediction target remains the 4h forward return.
