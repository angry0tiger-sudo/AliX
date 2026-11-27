# ADR-064 — M5.2 BTCUSDT Fear & Greed Evaluation: Economic NO-GO

- **Status:** **EXECUTED — STANDARD CI GREEN — ECONOMIC NO-GO**
- **Date:** 2026-11-27
- **Research branch:** `research/nobitex-hf-strategy`
- **Evaluation commit:** `d48b2c11b98c01a3bf9a09cc9d71114080db2a94`
- **Target:** Nobitex Spot / USDT — `BTCUSDT` only
- **Preregistration:** `ALIX-M5-2-SENTIMENT-G1`
- **Preregistration SHA-256:** `851d6c4b39c41cacec8ccd712cf17bb40dd42dd45f2bb8813920a7f2ebe481ba`
- **Evaluation:** [M5.2 workflow run #2 — 46000460512](https://github.com/angry0tiger-sudo/AliX/actions/runs/46000460512)
- **Standard CI:** [#45998903837](https://github.com/angry0tiger-sudo/AliX/actions/runs/45998903837)
- **Four-Arm Quality Preflight:** [#45998903704](https://github.com/angry0tiger-sudo/AliX/actions/runs/45998903704)

## 1. Decision

The frozen M5.2 experiment completed on the registered pre-hack Nobitex BTCUSDT dataset. All four registered candidates were fitted and evaluated with the preregistered Logistic Regression model and the four fixed expanding walk-forward folds.

**Economic decision: NO-GO.**

The OHLCV-only control failed its absolute economic gates. Adding the Alternative.me Fear & Greed feature did not improve NetPF or net PnL in either horizon under either cost scenario. The sentiment treatment produced strictly lower pooled NetPF and strictly lower pooled net PnL than its paired control in all four comparisons. No candidate met the complete registered acceptance criteria.

This is a negative result for the exact registered M5.2 candidate family and data window. It is not a universal proof that all sentiment information is useless. However, the registered Fear & Greed feature did not create incremental economic value here and made the observed results worse.

`GLOBAL_DSR=NOT_COMPUTABLE` and `GLOBAL_N=UNKNOWN` remain unchanged. Accordingly, **the first-economic-edge claim is not permitted**. Model/portfolio research status does not grant trading authority: `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`.

## 2. Execution and CI evidence

### 2.1 Standard CI gate

Before dispatch, Standard CI was independently verified as completed successfully on the exact evaluation SHA `d48b2c11b98c01a3bf9a09cc9d71114080db2a94`.

- Standard CI run: `45998903837` — **SUCCESS**.
- Quality job: **SUCCESS**.
- Tests: **1,127 passed, 5 skipped**.
- Total coverage: **90.36%**, meeting the configured 90% floor.
- Ruff lint and format, strict Mypy, Bandit, dependency resolution, transitive lock verification, pip-audit, architecture contracts and Xenon: **PASS**.
- PostgreSQL integration, Freqtrade runtime smoke and four-arm bare-environment lock checks: **PASS**.
- Four-Arm Quality Preflight `45998903704`: **SUCCESS**, 1,127 passed, 5 skipped, zero reported violations.

### 2.2 Dispatch and execution

GitHub required the manual `workflow_dispatch` workflow to be present on the default branch. With the Owner's explicit temporary exception, the workflow was added to `main`, dispatched against the research branch, and removed immediately after GitHub confirmed creation of the run.

The authoritative GitHub Actions API confirms:

| Field | Value |
|---|---|
| Workflow | `M5.2 BTCUSDT Sentiment Model Evaluation` |
| Run ID / number | `46000460512` / `#2` |
| Event | `workflow_dispatch` |
| Branch | `research/nobitex-hf-strategy` |
| Head SHA | `d48b2c11b98c01a3bf9a09cc9d71114080db2a94` |
| Conclusion | **SUCCESS** |
| Evaluation job | **SUCCESS** |
| Result artifact ID | `15198335893` |

All eight substantive job steps completed successfully, including the exact-SHA Standard CI gate, checkout, pinned model runtime installation, locked evaluation, report/safety validation and artifact upload.

An earlier dispatch attempt, run `45998406261` on the pre-fix SHA `82cc499a31816987707b574832bafcd1e176db29`, failed before any fitting or backtest. The cause was a duplicated path prefix: preregistration paths were already repository-root-relative, while the loader prepended the sentiment directory a second time. The loader and report-binding code now resolve and validate repository-relative paths from the repository root. The successful run #2 used the corrected code; the preregistration and data snapshots were not changed.

### 2.3 Main restoration record

Temporary workflow add commit: `e6dfe548c98d32ed9765d52f4522c7115fc4a9fe`.

Temporary workflow deletion commit: `a1c1a658906cd79be62796885e3ed782b66a267d`.

After cleanup, GitHub reports:

- Current `main` HEAD: `a1c1a658906cd79be62796885e3ed782b66a267d`.
- Current `main` tree SHA: **`5761c6c34143f8e6924f53a4e745c48932a5f791`** — exact match to the required baseline tree.
- `.github/workflows/m5-2-sentiment-evaluation.yml`: absent from `main`.

**Important history note:** the file contents/tree were restored exactly, but the temporary add and delete commits remain in the `main` commit history. The branch HEAD therefore differs from the original pre-exception commit `cd9b13d42ac0cf0edd2df066ff81580e028146d6`; the verified invariant is the exact original tree SHA, not an unchanged commit graph.

## 3. Immutable data and scope

### 3.1 Market data

| Binding | Value |
|---|---|
| Venue / market | Nobitex Spot / `BTCUSDT` |
| Timeframe | 1h |
| Scope | `PRE_HACK_ONLY` |
| Rows | 31,801 |
| Start | `2021-10-31T20:30:00Z` |
| End | `2025-06-17T20:30:00Z` |
| Missing hourly intervals | 0 |
| OHLCV CSV SHA-256 | `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e` |
| Source manifest SHA-256 | `b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26` |
| M5.2 data-lock SHA-256 | `e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf` |

No new OHLCV or sentiment acquisition was performed for this evaluation.

### 3.2 Fear & Greed binding

- Provider: Alternative.me Fear & Greed Index; 3,218 records in the frozen snapshot.
- Raw SHA-256: `4adba36f7dd95792520e078f7578c316c23e80a7b8a0ace81df490bf06c740df`.
- Normalized SHA-256: `e0578509cf3f8a6eeac39ddbd2edeab876e47027d759558e120c6f8b3a9f7f61`.
- Alignment: latest record with `available_at <= decision_time`, forward-fill only after availability.
- Availability convention: observation timestamp plus a conservative 24-hour proxy; the provider does not supply historical measured publication times.
- Common paired decisions: **31,777**.
- OHLCV warm-up excluded: 24 rows.
- Rows excluded for lack of prior sentiment availability: 0.
- Unique daily sentiment observations used: 1,325 (hourly repetitions are dependent observations).

This is a global BTC sentiment composite, not a Nobitex pair-specific signal.

## 4. Registered trial results

The four logical candidates, both cost scenarios and all folds are reported below. MaxDD is converted from the report's fractional value to percent of the running NAV peak.

| Trial | NetPF 50bps | NetPF 90bps | Trades | MaxDD 50bps | MaxDD 90bps | Valid folds | Stable folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control 1h | 0.199045 | 0.059637 | 2,243 | 1,010.40% | 1,907.20% | 4/4 | 0/4 |
| Sentiment 1h | 0.134926 | 0.037924 | 4,615 | 2,213.11% | 4,058.71% | 4/4 | 0/4 |
| Control 4h | 0.315225 | 0.127434 | 6,737 | 749.66% | 1,423.36% | 4/4 | 0/4 |
| Sentiment 4h | 0.297999 | 0.118593 | 8,349 | 939.69% | 1,774.59% | 4/4 | 0/4 |

All eight pooled NetPF observations are below 1.10 and below 1.00. All eight pooled MaxDD observations exceed the 15% maximum. Completed OOS trade counts meet the 100-trade minimum, and all four folds meet the valid-fold count requirement; no fold qualifies as stable because no fold has positive net PnL and NetPF above 1.00.

### 4.1 Paired incremental comparison

Net PnL below is expressed as a fraction of initial equity. The treatment must strictly improve both NetPF and net PnL to count as incremental value.

| Horizon | Cost | Control NetPF | Sentiment NetPF | Δ NetPF | Control net PnL | Sentiment net PnL | Paired value |
|---|---:|---:|---:|---:|---:|---:|---|
| 1h | 50bps | 0.199045 | 0.134926 | -0.064119 | -10.096431 | -22.123514 | FAIL |
| 1h | 90bps | 0.059637 | 0.037924 | -0.021713 | -19.068431 | -40.583514 | FAIL |
| 4h | 50bps | 0.315225 | 0.297999 | -0.017225 | -7.489932 | -9.390146 | FAIL |
| 4h | 90bps | 0.127434 | 0.118593 | -0.008840 | -14.226932 | -17.739146 | FAIL |

Sentiment has strictly lower NetPF and strictly lower net PnL than its paired OHLCV control in every horizon/cost comparison. It generates more trades (4,615 vs 2,243 at 1h; 8,349 vs 6,737 at 4h), but the increased frequency does not translate to economic value.

### 4.2 Fold stability

All 16 candidate-fold evaluations were completed. Every fold's net PnL was negative and every fold NetPF was below 1.00 under both costs. The fold-level NetPF ranges were:

| Trial | Cost-50 fold NetPF range | Cost-90 fold NetPF range | Stable folds |
|---|---:|---:|---:|
| Control 1h | 0.105573–0.269144 | 0.017999–0.102457 | 0/4 |
| Sentiment 1h | 0.065785–0.263855 | 0.012398–0.095288 | 0/4 |
| Control 4h | 0.225417–0.347187 | 0.079703–0.143497 | 0/4 |
| Sentiment 4h | 0.211498–0.346809 | 0.075216–0.143016 | 0/4 |

## 5. Interpretation and metric caveat

The report's `max_drawdown` is calculated as peak-to-trough NAV drawdown. The reported values exceeding 100% mean the simulated NAV went below zero. This follows from the preregistered fixed cohort sizing relative to initial equity, while the simulator continues later cohorts without resizing to remaining NAV or imposing an insolvency stop.

These figures are therefore research-simulation metrics, not a claim of executable, self-financing spot-account performance. This caveat does not change the NO-GO decision: pooled NetPF is far below its gate, pooled net PnL is negative, every fold is unstable, and sentiment is strictly worse than control. Do not reinterpret or normalize the MaxDD values after seeing outcomes; any capital-constrained portfolio redesign requires a superseding preregistration.

## 6. Model, report and prediction artifacts

- Estimator: `StandardScaler + LogisticRegression`.
- scikit-learn: `1.9.0`.
- Frozen classifier parameters: `penalty=l2`, `C=1`, `solver=lbfgs`, `fit_intercept=true`, `class_weight=null`, `max_iter=1000`, `tol=0.0001`, `random_state=42`.
- Probability threshold: 0.5.
- Fits: **16** (four candidates × four folds).
- OOS prediction rows: **101,648** across the four candidates and four folds.
- Report file: `m5_2_evaluation_report.json`, 75,423 bytes; SHA-256 `5bbeb32ca88cecca1997c6dbecb98066618e340bf6be26d6088d43203e48a1b7`.
- Prediction file: `m5_2_oos_predictions.csv`, 22,378,010 bytes; SHA-256 `2c654c324cd9b84547beefb28637efe56d32dc5ed5834a0f79fc7c01262bbfcc`.
- GitHub artifact: `alix-m5-2-btcusdt-sentiment-results`, artifact ID `15198335893`; archive SHA-256 `2fba87fc1509de1447bb3331b9ef7dc4296ea371ba0d5522c238478f9bee45b8`.

The artifact contains the JSON evaluation report and full OOS predictions. The M5.2 workflow summary and artifact are attached to [run #46000460512](https://github.com/angry0tiger-sudo/AliX/actions/runs/46000460512).

## 7. Final gates and safety record

| Acceptance criterion | Result |
|---|---|
| NetPF >= 1.10 for sentiment candidates at both costs/horizons | **FAIL** |
| At least 100 completed OOS trades | PASS |
| MaxDD <= 15% | **FAIL** |
| At least 3 valid folds | PASS (4/4) |
| At least 3 stable folds | **FAIL** (0/4) |
| Strict paired improvement in NetPF and net PnL | **FAIL** (0/4 comparisons) |
| Sentiment absolute gates | **FAIL** |
| Overall economic verdict | **NO-GO** |
| First economic edge claim permitted | **false** |
| DSR / global trial accounting | `NOT_COMPUTABLE` / `UNKNOWN` |

Execution safety checks recorded by the report:

- Market: BTCUSDT only; data scope: PRE-HACK only.
- Holdout bytes read: `false`.
- Post-Hack data read: `false`.
- External market data requests / new acquisition: 0 / 0.
- Exchange network access: `false`.
- Private API calls: 0.
- Paper orders / real order attempts: 0 / 0.
- Local execution: `false`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.

## 8. Final disposition

The temporary dispatch obstacle was resolved using the Owner-authorized add/dispatch/delete sequence. The workflow ran on the exact research SHA after Standard CI passed, and the original `main` tree SHA was restored and verified. The M5.2 economic experiment is complete and **NO-GO**.

Fear & Greed did not improve the frozen OHLCV Logistic Regression baseline. It reduced NetPF and net PnL in all four paired comparisons, while all candidates failed the registered economic and fold-stability gates. No Champion, live/paper authority, feature promotion, tuning, alternate model or expanded market scope is authorized by this ADR.

Any future strategy change, capital-constrained execution accounting, alternate sentiment source, feature/parameter change or new candidate family requires a new prospective owner authorization and superseding preregistration. No Holdout or Post-Hack access is permitted.
