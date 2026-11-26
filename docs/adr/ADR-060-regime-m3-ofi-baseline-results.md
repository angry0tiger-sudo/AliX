# ADR-060 — Regime M3.5 OFI Baseline Results and NO-GO

- **Status:** **NO-GO — registered Logistic Regression baseline did not establish an economic edge**
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Predecessor:** ADR-059 — Regime M3 OFI Preregistration and Baseline Lock
- **Prere-registration ID:** `ALIX-M3-OFI-G1`
- **Prere-registration SHA-256:** `c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4`
- **Implementation/evaluation commit:** `10c01b7988a87ca3e9ac0510a4f03e0d1260ee6b`
- **Workflow:** [AliX Continuous Integration, Run #2586](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105)
- **Evaluation job:** `regime-m3-ofi-evaluation / evaluate-ofi` — completed successfully
- **Result artifact:** [regime-m3-ofi-evaluation-45898505105.zip](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105/artifacts/15155002532)
- **Artifact ID:** `15155002532`
- **Artifact ZIP SHA-256:** `78c53dbb6f060b4d22117fb726b49750ace3b035d6bc5079357517a37e63edf7`

## 1. Decision summary

The M3.5 implementation and empirical evaluation completed successfully as an engineering workflow. The economic experiment did **not** pass.

The registered baseline was evaluated against the five frozen M1.6 public BTCIRT capture artifacts, with the exact eight preregistered features, one-minute decisions, five-minute feature windows, four chronological expanding walk-forward folds, 1h/4h horizons, StandardScaler fitted on training data only, and the registered Logistic Regression estimator. The fixed 90 bps all-in round-trip cost was applied to each completed simulated cohort.

The pooled out-of-sample results are:

| Horizon | Economic verdict | NetPF | Completed OOS trades | MaxDD | Net PnL / initial equity | Stable folds |
|---|---|---:|---:|---:|---:|---:|
| 1h | **NO-GO** | **0.0734699215** | 665 | 8.2759% | -7.9225% | 0/4 |
| 4h | **INSUFFICIENT_SAMPLE / NO-GO** | **0.2326940033** | 82 | 0.7494% | -0.1214% | 0/4 |

Neither horizon satisfies the registered acceptance gates. The 1h horizon has adequate trade count and acceptable drawdown, but its NetPF is far below 1.10 and no fold is stable. The 4h horizon fails the NetPF and trade-count gates and has insufficient training support in three of its four folds. Its drawdown being below 15% does not compensate for these failures.

**Overall decision: NO-GO.** The simple OFI Logistic Regression baseline did not demonstrate an economically meaningful, stable edge under the frozen protocol and 90 bps cost assumption. This is a decision about this registered baseline and this dataset, not a proof that every possible microstructure signal or model is ineffective.

## 2. Evidence and protocol integrity

### 2.1 Execution evidence

- GitHub Actions Run #2586 was triggered by `workflow_dispatch` on `research/nobitex-hf-strategy`.
- Run conclusion: `success`.
- The four prerequisite CI jobs completed successfully: `quality`, `four-arm-lock-bare-env`, `freqtrade-runtime-smoke`, and `postgres-integration`.
- The dedicated `evaluate-ofi` job completed successfully, including preregistration verification, five artifact downloads, the locked evaluation, summary publication, and result upload.
- Other research jobs in the same workflow were skipped; they were not part of this OFI evaluation.
- Standard CI and the independent Four-Arm Quality Preflight were successful on the evaluated implementation lineage. The Standard CI run immediately preceding this dispatch recorded 90.36% coverage, 1,099 passed tests, and 5 skipped tests.

A successful workflow means the evaluation ran to completion and produced a valid report. It does **not** mean the economic hypothesis passed.

### 2.2 Immutable preregistration and dataset

The evaluator verified the exact locked configuration SHA-256:

`c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4`

All five registered M1.6 artifact identities and ZIP digests matched the preregistration. All five artifacts were downloaded successfully. The dataset contained:

- 105,020.393 seconds of captured data (29 hours, 10 minutes, 20.393 seconds);
- five chronological capture segments;
- eight recorded within-segment gap intervals;
- zero data-quality exclusion rows in the segment manifests;
- 100 REST recent-trade seed rows excluded (20 per segment);
- no stitching across segment boundaries or recorded gaps.

The registered M1.6 artifacts are those listed in ADR-059 §3 and `config/regime_m3/ofi_preregistration.json`; this evaluation did not add, replace, or extend the dataset.

### 2.3 Feature and model contract

Exactly these eight columns were used:

1. `best_level_ofi_5m`
2. `mlofi_level_2_5m`
3. `mlofi_level_3_5m`
4. `mlofi_level_4_5m`
5. `mlofi_level_5_5m`
6. `microprice_deviation_bps`
7. `signed_trade_volume_ratio_5m`
8. `order_book_imbalance_top5`

Decisions were evaluated on the one-minute UTC grid using the trailing five-minute feature window. The model pipeline was `StandardScaler` followed by `sklearn.linear_model.LogisticRegression`; the scaler was fitted on training rows only. Locked estimator parameters were L2, `C=1.0`, `solver=lbfgs`, `class_weight=null`, `max_iter=1000`, `tol=0.0001`, and `random_state=42`. The fixed long-entry threshold was `P(up) >= 0.5`.

The trade-type preflight returned `PASS_PUBLIC_API_TYPE_PROXY_ONLY`, with 10,502 `buy` and 15,021 `sell` public WebSocket events. The mapping remained `buy=+volume; sell=-volume`; no aggressor-side interpretation was claimed.

### 2.4 Fixed acceptance gates

Both horizons had to satisfy every registered condition independently:

| Gate | Required |
|---|---:|
| Pooled OOS Net Profit Factor | >= 1.10 |
| Completed OOS trades | >= 100 |
| Maximum drawdown | <= 15% |
| Stable temporal folds | >= 3 of 4 |

A fold is stable only when it has at least 20 completed OOS trades, positive net PnL, and fold NetPF greater than 1.00. Insufficient folds remain insufficient; no fold, threshold, horizon, feature, estimator parameter, or minimum was changed after the outcomes were observed.

## 3. Pooled economic results and gate decisions

### 3.1 One-hour horizon

| Metric | Observed | Gate | Result |
|---|---:|---:|---|
| NetPF | 0.0734699215 | >= 1.10 | **FAIL** |
| Completed OOS trades | 665 | >= 100 | PASS |
| MaxDD | 8.2759% | <= 15% | PASS |
| Stable folds | 0/4 | >= 3/4 | **FAIL** |
| Net PnL / initial equity | -7.9225% | positive fold requirement | **FAIL** |

All four 1h folds were trainable and each produced at least 20 completed trades, but every fold had negative net PnL and NetPF below 1.00. None qualified as stable.

### 3.2 Four-hour horizon

| Metric | Observed | Gate | Result |
|---|---:|---:|---|
| NetPF | 0.2326940033 | >= 1.10 | **FAIL** |
| Completed OOS trades | 82 | >= 100 | **FAIL** |
| MaxDD | 0.7494% | <= 15% | PASS |
| Stable folds | 0/4 | >= 3/4 | **FAIL** |
| Net PnL / initial equity | -0.1214% | positive fold requirement | **FAIL** |

Only WF-04 passed the training preflight and fitted a model. WF-01 had zero eligible training rows; WF-02 and WF-03 had no class-0 training observations. They therefore remained `INSUFFICIENT_SAMPLE` and were not fitted. The 82 completed trades all came from the one evaluated test fold, which is not enough to satisfy the pooled trade-count or fold-stability requirements.

## 4. Per-fold results

### 4.1 One-hour folds

| Fold | Training rows | Test rows | Completed trades | NetPF | MaxDD | Net PnL / initial equity | Stable |
|---|---:|---:|---:|---:|---:|---:|---|
| WF-01 | 220 | 285 | 90 | 0.6418592726 | 0.7790% | -0.2143% | No |
| WF-02 | 505 | 226 | 171 | 0.0144889973 | 2.2710% | -1.8913% | No |
| WF-03 | 731 | 263 | 209 | 0.0024871425 | 3.7348% | -3.5877% | No |
| WF-04 | 994 | 285 | 195 | 0.0850952660 | 2.8698% | -2.2292% | No |
| **Pooled** | — | — | **665** | **0.0734699215** | **8.2759%** | **-7.9225%** | **0/4** |

Training class counts by fold were WF-01 `131/89`, WF-02 `208/297`, WF-03 `277/454`, and WF-04 `447/547` (class 0 / class 1). Each fold passed the registered training preflight, but none passed the fold-stability rule.

### 4.2 Four-hour folds

| Fold | Training rows | Class 0 / Class 1 | Test rows | Fold outcome | Trades | NetPF | MaxDD | Net PnL |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| WF-01 | 0 | 0 / 0 | 105 | INSUFFICIENT_SAMPLE; no fit | — | — | — | — |
| WF-02 | 105 | 0 / 105 | 46 | INSUFFICIENT_SAMPLE; no fit | — | — | — | — |
| WF-03 | 151 | 0 / 151 | 83 | INSUFFICIENT_SAMPLE; no fit | — | — | — | — |
| WF-04 | 234 | 83 / 151 | 105 | EVALUATED; unstable | 82 | 0.2326940033 | 0.7494% | -0.1214% |
| **Pooled available OOS** | — | — | — | **INSUFFICIENT_SAMPLE / NO-GO** | **82** | **0.2326940033** | **0.7494%** | **-0.1214%** |

The 4h stability report recorded one eligible fold with at least 20 trades and zero stable folds, against the requirement of three stable folds.

## 5. Diagnostic baselines

These registered comparators are reported for context and do not replace the OFI model or its acceptance gates.

| Horizon / comparator | Trades | NetPF | MaxDD | Net PnL / initial equity |
|---|---:|---:|---:|---:|
| 1h always-long | 1,059 | 0.0856608474 | 12.1405% | -11.8044% |
| 1h training-fold majority direction | 774 | 0.0316228120 | 10.4296% | -10.0935% |
| 4h always-long | 339 | 1.2189793701 | 1.1142% | +0.1509% |
| 4h training-fold majority direction | 234 | 0.1250290185 | 1.1226% | -0.6030% |
| No-trade cash baseline | 0 | Undefined | 0.0000% | 0.0000% |

The 4h always-long comparator had a positive but small pooled return and a NetPF above 1.10. It is a diagnostic control, not the registered OFI estimator; it has no authority to convert the OFI result to PASS. The registered Logistic Regression model materially underperformed this comparator on the available 4h sample. This observation must not be used to select an alternative trading rule post hoc.

## 6. Final decision and consequences

**M3.5 Baseline OFI: NO-GO.**

1. The 1h baseline generated sufficient trade count but failed NetPF by a wide margin and produced zero stable folds.
2. The 4h baseline failed NetPF and minimum trade count; three folds could not be fitted under the locked training-class requirements, and the only evaluated fold was unprofitable and unstable.
3. Therefore the OFI baseline has not demonstrated an economic edge after the registered 90 bps round-trip cost.
4. The result does not authorize Holdout evaluation. Holdout and Post-Hack remain sealed.
5. No DeepLOB, Transformer, alternative estimator, feature change, horizon change, threshold sweep, retraining, or post-hoc adjustment is authorized by this ADR.
6. `Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain unchanged.

The appropriate conclusion is limited and explicit: **the registered simple OFI microstructure baseline did not show a significant, stable economic edge in this experiment.** This is not a universal claim that all microstructure information is useless.

## 7. Safety and data boundary confirmation

The result report records these invariants:

- `holdout_bytes_read=false`
- `posthack_data_read=false`
- `private_api_calls=0`
- `real_order_attempts=0`
- `paper_orders=0`
- `external_data_requests=0`
- `local_execution=false`
- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`

Execution was on the GitHub-hosted runner with `ALIX_DRY_RUN=true` and `ALIX_ALLOW_LIVE_TRADING=false`. The simulation used mid-price marks and must not be represented as an executable-fill or live execution test.

## 8. References and reproducibility

1. [ADR-059 — Regime M3 OFI Preregistration and Baseline Lock](ADR-059-regime-m3-ofi-preregistration.md).
2. [Locked preregistration JSON](../../config/regime_m3/ofi_preregistration.json).
3. [M3.5 workflow run #2586](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105).
4. [M3.5 result artifact #15155002532](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105/artifacts/15155002532).
5. [Evaluated implementation commit](https://github.com/angry0tiger-sudo/AliX/commit/10c01b7988a87ca3e9ac0510a4f03e0d1260ee6b).

The result JSON is the detailed machine-readable record. The GitHub Actions summary and evaluation logs provide the run-level corroboration.

## 9. Repository integrity

At the time of this decision, `main` remained at commit `cd9b13d42ac0cf0edd2df066ff81580e028146d6`, whose root Tree SHA is `5761c6c34143f8e6924f53a4e745c48932a5f791`. This matches the previously frozen main Tree SHA. ADR-060 is recorded only on `research/nobitex-hf-strategy`; no file or reference on `main` was changed.

**M3.5 outcome: NO-GO. Holdout remains sealed.**

Architecture First. Evidence Always. Fail Closed.
