# ADR-056 — Regime T Triple-Barrier Target and Logistic Baseline

- **Status:** EXECUTED — NO-GO (55% owner screening threshold not met)
- **Date:** 2026-11-18
- **Branch:** research/nobitex-hf-strategy
- **Experiment:** REGIME-T-TB-LOGISTIC-01
- **Preregistration:** config/regime_t/triple_barrier_preregistration.json
- **Preregistration SHA-256:** b6118f4e7420b26c8caf471ef7281b7b452c031ed7a68b6d8eb5a7d210d103ad
- **Dataset SHA-256:** 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e
- **Dataset manifest SHA-256:** b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26
- **Frozen G3 SHA-256:** 92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa

## 1. Context and decision

G4.6 closed the HGBQ-CQR economic path under Regime P after complete abstention and exact equality of all 90 G4.5/G4.6 diagnostic statistics objects. This ADR establishes a separate experiment, not a continuation, mutation, or revival of HGBQ-CQR. The hypothesis is that a volatility-scaled first-touch event target may be more learnable than fixed-horizon return regression.

The experiment is restricted to the historically exposed Nobitex PRE-HACK BTCUSDT one-hour OHLCV file (31,801 rows). Evidence is exploratory, not confirmatory. No Holdout or post-hack bytes, market-data fetch, exchange API, paper trading, or order execution is permitted.

## 2. Locked target definition

The task is long-only binary first-touch classification, conditional on one of the horizontal barriers being touched before the 24-hour time barrier.

1. Decision after close of bar t; features use closed OHLC data through t only.
2. Entry at open of bar t+1, fixed before inspecting that bar's high/low.
3. Volatility is Wilder ATR(14) divided by close at t. True Range is max(high-low, abs(high-previous_close), abs(low-previous_close)); first row uses high-low. ATR seed is the arithmetic mean of the first 14 TR values, followed by (13 * prior_ATR + current_TR) / 14.
4. Upper barrier = entry_open(t+1) * (1 + 2 * ATR14_fraction(t)).
5. Lower barrier = entry_open(t+1) * (1 - 2 * ATR14_fraction(t)).
6. Time barrier = 24 future hourly bars, t+1 through t+24 inclusive.
7. Upper-first is class 1; lower-first is class 0.

The earliest future bar whose high/low touches a barrier determines the first-touch class. If one OHLC bar touches both barriers, intrabar ordering is unknowable and the event is AMBIGUOUS, not guessed. If neither barrier is touched within 24 bars, the event is TIMEOUT. Rows lacking a full future window or valid volatility are UNAVAILABLE. All non-binary outcomes remain separately counted and are excluded from binary fitting/scoring. Fees and slippage are not incorporated into this statistical label.

## 3. Feature and model contract

Exactly 11 price/time features are used; the feature builder does not read Volume and Volume is never passed to the estimator:

- log_return_1, log_return_4, log_return_24
- realized_volatility_6, realized_volatility_24
- range_fraction_1, atr_fraction_14
- hour_sin, hour_cos, weekday_sin, weekday_cos

One fixed pipeline is registered: StandardScaler fit on each fold's training rows only, followed by LogisticRegression with solver lbfgs, C=1.0, L2 penalty, no class weights, max_iter=2000, random_state=42. The probability threshold is 0.5. No tuning, feature selection, threshold sweep, or candidate comparison is authorized.

## 4. Temporal protocol and leakage controls

The original three G3 calendar folds, UTC anchor and relative offsets are reused without changing the G3 file:

- 6-month rolling training window
- 1-month calibration interval reserved and unused
- 1-month embargo interval reserved and unused
- 1-month test window
- 3-calendar-month fold shift

A training decision is eligible only if its full 24-hour label observation window ends strictly before that fold's training boundary. The same full-horizon rule applies at the test boundary. Labels must be fully resolved before their relevant boundary. Scalers are fitted on training rows only. Test rows do not enter fitting, normalization, calibration, or threshold selection.

## 5. Evaluation and decision rule

The primary statistic is pooled out-of-fold accuracy across all three non-overlapping test windows. Report per-fold accuracy, balanced accuracy, class counts, confusion matrices with rows=true [0,1] and columns=predicted [0,1], 50% uniform-random expected accuracy, and the training-majority baseline evaluated on each fold's test set. Timeout, ambiguous, and unavailable counts are mandatory.

The owner's initial screening criterion is pooled accuracy strictly above 55%. Passing only justifies a separately authorized next-stage model study; it does not prove trading edge, statistical significance, profitability, or execution permission. A result near 50%, or one that does not beat the fold-majority baseline, is not meaningful positive evidence. This registration authorizes no rerun or post-outcome tuning.

## 6. Safety and change control

The workflow is workflow_dispatch only and must run only when the selected ref is research/nobitex-hf-strategy. It verifies exact SHA-256 commitments for PRE-HACK data, manifest, frozen G3 JSON and this preregistration. Scripts require explicit Regime T authorization, ALIX_DRY_RUN=true, and ALIX_ALLOW_LIVE_TRADING=false.

Safety invariants:
- real_data_scope = PRE_HACK_ONLY
- holdout_bytes_read = false
- posthack_data_read = false
- private_api_calls = 0
- real_order_attempts = 0
- paper_orders = 0
- exchange_network_access = false
- Champion = null
- order_permission = false
- trading_authorization = NO-GO

No G3 file/parameter, G4.6 file, HGBQ-CQR code, or Regime P result is modified. main remains frozen. The temporary dispatch workflow was exposed on main only during completed dispatch windows and was removed after each run. The main branch tree was verified restored to the original SHA 5761c6c34143f8e6924f53a4e745c48932a5f791.

## 7. Execution history, results, and decision

### 7.1 Infrastructure attempts (no model outcome)

- **Run #1 / `44340086418`**, Research SHA `e1d478556386404dde4b15c94af4ce21d8a353ba`: GitHub cancelled during the `setup-python` pip-cache restore. Dependency installation, hash verification, synthetic tests, label generation, and model fitting did not run. No experiment output was produced.
- **Run #2 / `44348023980`**, Research SHA `5df5cd0be6364b186e6d7242a24f61294170ce32`: PRE-HACK data, manifest, G3, and preregistration commitments verified successfully. The synthetic test command `pytest --no-cov -q tests/unit/test_regime_t` then failed because the isolated workflow environment had not installed `pytest-cov`, which provides the `--no-cov` option. The label-generation and model-fit steps were skipped. No labels, predictions, accuracy, or fold outcomes were produced.
- Infrastructure-only correction: pin `pytest-cov==7.0.0` in the workflow dependency installation. The target, features, fold boundaries, learner, model parameters, threshold, dataset, G3 file, and preregistration JSON remained unchanged. The preregistration SHA-256 remained `b6118f4e7420b26c8caf471ef7281b7b452c031ed7a68b6d8eb5a7d210d103ad`. As neither prior run reached labeling or fitting, no model outcome had been observed before the successful experiment below.

### 7.2 Completed experiment provenance

- **Experiment:** `REGIME-T-TB-LOGISTIC-01`
- **Successful workflow:** [Run #3 / `44349379156`](https://github.com/angry0tiger-sudo/AliX/actions/runs/44349379156)
- **Conclusion:** `success`
- **Research ref / commit:** `research/nobitex-hf-strategy` / `5767c0c7e473f7932e4f6392b9684526d205040c`
- **Standard CI:** [Run #2512 / `44348338969`](https://github.com/angry0tiger-sudo/AliX/actions/runs/44348338969), `success` on the same commit
- **Dataset:** PRE-HACK only, 31,801 rows, SHA-256 `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- **Manifest SHA-256:** `b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26`
- **Frozen G3 SHA-256:** `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- **Preregistration SHA-256:** `b6118f4e7420b26c8caf471ef7281b7b452c031ed7a68b6d8eb5a7d210d103ad`
- **Artifact:** `regime-t-triple-barrier-prehack-5767c0c7e473f7932e4f6392b9684526d205040c`, artifact ID `14448360682`, archive digest `sha256:cd5fb3d91b75922df19d41d62e1821145b281519caa6c9fc6d95f95878b6d10b`
- Artifact files: `triple-barrier-labels.csv`, `labeling-summary.json`, `logistic-evaluation.json`, and `out-of-fold-predictions.csv`.

### 7.3 Target outcome counts

The 31,801 input rows resolved into:

| Outcome | Count | Treatment |
|---|---:|---|
| `LOWER_FIRST` (class 0) | 12,405 | Included in binary fitting/scoring |
| `UPPER_FIRST` (class 1) | 12,335 | Included in binary fitting/scoring |
| `TIMEOUT` | 6,970 | Censored; excluded |
| `AMBIGUOUS` | 54 | Censored; excluded |
| `UNAVAILABLE` | 37 | Censored; excluded |

Binary events totalled 24,740. The 7,061 timeout, ambiguous, and unavailable rows were not assigned a binary class for model scoring.

### 7.4 Walk-forward model results

One fixed `StandardScaler + LogisticRegression` pipeline was fit once per registered fold. No tuning or threshold sweep was performed. Calibration and embargo partitions remained unused.

| Fold | Train rows | Test rows | Accuracy | Balanced accuracy | Training-majority baseline |
|---|---:|---:|---:|---:|---:|
| WF-01 | 3,555 | 598 | 49.33% | 48.79% | 45.82% |
| WF-02 | 3,557 | 525 | 60.57% | 59.43% | 55.24% |
| WF-03 | 3,543 | 529 | 43.67% | 46.58% | 34.22% |
| **Pooled OOF** | — | **1,652** | **51.09%** | **51.57%** | **45.10%** |

Pooled confusion matrix, with rows=true `[0,1]` and columns=predicted `[0,1]`: `[[421, 324], [484, 423]]`.

Relative to the 50% uniform-random expected accuracy, pooled accuracy lifted by only **1.09 percentage points**. It exceeded the pooled fold-specific training-majority baseline by **5.99 percentage points**. However, two of the three folds were below 50%, and fold accuracy ranged from 43.67% to 60.57%, showing material temporal instability. Pooled balanced accuracy was only 51.57%.

### 7.5 Decision

- **Owner screening threshold:** pooled accuracy strictly greater than 55%.
- **Observed pooled accuracy:** 51.09%.
- **Threshold result:** `owner_threshold_passed = false`.
- **Research decision:** `NO-GO`; the registered screening criterion was not met.

The baseline is numerically above 50% pooled accuracy, but the lift is small (+1.09 pp), balanced accuracy is near chance, and performance is not stable across the three temporal folds. The observed advantage over the majority comparator does not override the preregistered 55% screen. No statistical-significance claim is made, and this classification result is not an economic backtest: barrier labels do not incorporate fees/slippage and do not establish positive net trading returns. No next-stage model study, additional candidate, tuning, threshold change, or rerun is authorized by this ADR after observing this outcome.

### 7.6 Safety and restoration attestation

The successful report and labeling summary independently recorded and the workflow's final validation asserted:

- `real_data_scope = PRE_HACK_ONLY`
- `holdout_bytes_read = false` and `posthack_data_read = false`
- `private_api_calls = 0`, `real_order_attempts = 0`, `paper_orders = 0`
- `exchange_network_access = false`
- `Champion = null`, `order_permission = false`, `trading_authorization = NO-GO`

The POST-HACK Holdout was not accessed. No market/exchange network or private API was used. The temporary workflow was deleted from `main` after Run #3 completed; the current `main` tree SHA was verified as exactly `5761c6c34143f8e6924f53a4e745c48932a5f791`.

Architecture First. Evidence Always. Fail Closed.
