
# ADR-060 — Regime M3.5 OFI Baseline Walk-Forward Evaluation

- **Status:** **CLOSED — CLOSED_NO_GO (Owner-approved 2026-11-27)** — evaluation verdict **INSUFFICIENT_SAMPLE / NO-GO**
- **Date:** 2026-11-27
- **Branch:** research/nobitex-hf-strategy
- **Evaluation commit:** 10c01b7988a87ca3e9ac0510a4f03e0d1260ee6b
- **Prere-registration ID:** ALIX-M3-OFI-G1
- **Prere-registration:** config/regime_m3/ofi_preregistration.json
- **Prere-registration SHA-256:** c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4
- **Implementation:** scripts/run_regime_m3_ofi.py
- **Mock tests:** tests/unit/test_regime_m3_ofi.py
- **Evaluation workflow:** .github/workflows/regime-m3-ofi-evaluation.yml
- **M3.5 GitHub Actions run:** [Run #2586](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105) — workflow_dispatch, completed successfully
- **Standard CI:** [Run #2585](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855203) — completed successfully
- **Quality preflight:** [Run #125](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855197) — completed successfully
- **Result artifact:** regime-m3-ofi-evaluation-45898505105 — artifact ID 15155002532, SHA-256 78c53dbb6f060b4d22117fb726b49750ace3b035d6bc5079357517a37e63edf7

## 1. Decision

The locked M3.5 OFI baseline was implemented, mock-tested, and evaluated on the exact five M1.6 public BTCIRT capture artifacts registered by ADR-059. All required Standard CI and preflight gates passed. The manual GitHub Actions evaluation completed successfully and produced a machine-readable result artifact.

**The economic hypothesis is not accepted. Neither the 1h nor the 4h Logistic Regression trial passed its preregistered acceptance gates. The overall decision is INSUFFICIENT_SAMPLE / NO-GO.**

This is an empirical research rejection/insufficiency decision, not a software-delivery failure. No strategy approval or trading permission is granted. The current safety state remains Champion=null, order_permission=false, and trading_authorization=NO-GO.

## 2. Execution and quality evidence

The final evaluation used commit 10c01b7988a87ca3e9ac0510a4f03e0d1260ee6b on research/nobitex-hf-strategy.

| Gate | GitHub Actions run | Result |
|---|---:|---|
| Standard CI | 45897855203 | **PASS** |
| Quality preflight | 45897855197 | **PASS** |
| Manual M3.5 evaluation | 45898505105 | **PASS — evaluation completed** |

The Standard CI quality job passed Ruff lint and format, strict Mypy, Bandit, dependency resolution and lock verification, pip-audit, architecture contracts, Xenon, the mock-only unit suite, and the existing synthetic/safety workflows. Coverage was **90.36%**, with **1,099 passed and 5 skipped** in the coverage-enabled unit run. The preflight completed its independent static, security, dependency, complexity, and pytest checks successfully.

The evaluation Job verified the locked preregistration digest and all five registered artifact IDs/digests before downloading them. It then completed the feature/label preflight and four-fold evaluation, published the Actions summary, and uploaded the result JSON. No empirical result was produced by a local execution.

## 3. Dataset boundary and provenance

Only the five M1.6 accepted public Nobitex BTCIRT segments in the ADR-059 manifest were used:

| Segment | Capture run ID | Artifact ID | ZIP SHA-256 |
|---:|---:|---:|---|
| 1 | 45553679888 | 15016992581 | aafe15ed9da1d6f6e8badf905627665695acfd605e926b242856f43888980ce8 |
| 2 | 45620061415 | 15054886094 | c25c202e5d556912bc40e6a9fdcf78adb5a23db32f3f760ad3a1024d79cf8961 |
| 3 | 45683066882 | 15077454919 | c7b7ec592cda5fec117976a422d2257a908c2b260d89b4b9b89ef87ab4f20a62 |
| 4 | 45735110603 | 15095790023 | ec4befa0ad6fc96cf1874cec4ba15b2a2dc5ca27e58adf82c33eb62f18393332 |
| 5 | 45775145557 | 15121226078 | c52cb93bfc345e1ba66104f7f0e4f8ea3664687e87d3dd89e44ea9236279db88 |

The evaluation report records **105,020.393 seconds** of capture time (29 hours, 10 minutes, 20.393 seconds). The five segments are not a continuous stream; all WebSocket disconnects and uncovered inter-segment intervals remained hard boundaries. No stitching, gap interpolation, replay, or cross-session state carry was used.

The five manifests contained **171,604 order-book records** and **25,523 WebSocket trade events** (buy=10,502, sell=15,021). A further **100 REST recent-trade seed rows** were excluded from the signed-volume feature. The trade-type preflight returned PASS_PUBLIC_API_TYPE_PROXY_ONLY: the registered mapping buy=+volume, sell=-volume was used only as an API-type proxy. No aggressor-initiated interpretation was claimed.

## 4. Locked method executed

The run used exactly the eight registered features and no additional indicators, feature selection, PCA, interactions, alternative estimators, or threshold search:

1. best_level_ofi_5m
2. mlofi_level_2_5m
3. mlofi_level_3_5m
4. mlofi_level_4_5m
5. mlofi_level_5_5m
6. microprice_deviation_bps
7. signed_trade_volume_ratio_5m
8. order_book_imbalance_top5

Decision cadence was one minute with a trailing five-minute feature window. The two fixed binary targets were the sign of the forward mid-price log return at 1h and 4h. Any feature/target sample intersecting a gap, reconnect/session boundary, segment boundary, malformed book, missing required level, or immature target was excluded.

The sole estimator was the preregistered StandardScaler + LogisticRegression pipeline (scikit-learn 1.9.0; L2, C=1, lbfgs, intercept enabled, no class weights, max_iter=1000, tol=0.0001, random_state=42). The scaler was fitted on each fold's training rows only. A long-only cohort was opened at P(up) >= 0.5; otherwise the model abstained.

The fixed economic simulation used midprice-to-midprice returns, one **90 bps** round-trip friction charge per completed cohort, no leverage, and fixed notional of 1/60 of initial equity for 1h or 1/240 for 4h. Cohorts were held for exactly their registered horizon. This is not an executable-fill simulation. Overlapping one-minute forecasts create dependent cohorts; completed cohort count must not be interpreted as independent observations.

The four chronological folds and all training, class-balance, purge, and stability rules remained exactly as preregistered.

## 5. Economic acceptance gates

The same gates applied independently to each horizon:

| Gate | Required |
|---|---:|
| Pooled OOS Net Profit Factor | >= 1.10 |
| Completed OOS trades | >= 100 |
| Absolute maximum drawdown | <= 15% |
| Stable folds | At least 3 of 4 |

A stable fold requires at least 20 completed trades, positive net PnL, and fold NetPF > 1.00. Both horizons must pass every gate; one horizon cannot compensate for the other.

## 6. Results — 1h horizon

**Verdict: NO-GO**

| Metric | Observed | Gate | Result |
|---|---:|---:|---|
| Pooled NetPF | **0.07347** | >= 1.10 | **FAIL** |
| Completed OOS trades | **665** | >= 100 | PASS |
| MaxDD | **8.276%** | <= 15% | PASS |
| Stable folds | **0 / 4** | >= 3 / 4 | **FAIL** |
| Net PnL / initial equity | **-7.923%** | Reporting only | Negative |

All four fixed folds had enough completed trades to be evaluated, but none was stable. Every fold had negative net PnL and NetPF below 1.00.

| Fold | Train rows | Class 0 / 1 | Test rows | Completed trades | Net PnL / initial equity | NetPF | MaxDD | Stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WF-01 | 220 | 131 / 89 | 285 | 90 | -0.214% | 0.64186 | 0.779% | No |
| WF-02 | 505 | 208 / 297 | 226 | 171 | -1.891% | 0.01449 | 2.271% | No |
| WF-03 | 731 | 277 / 454 | 263 | 209 | -3.588% | 0.00249 | 3.735% | No |
| WF-04 | 994 | 447 / 547 | 285 | 195 | -2.229% | 0.08510 | 2.870% | No |

The 1h model did not establish an economic edge after the preregistered friction assumption. The trade-count and drawdown gates passing do not offset the severe NetPF and fold-stability failures.

### 1h reporting baselines

| Baseline | Trades | NetPF | Net PnL / initial equity | MaxDD |
|---|---:|---:|---:|---:|
| Always long, same horizon and cost | 1,059 | 0.08566 | -11.804% | 12.140% |
| No-trade cash | 0 | Undefined | 0.000% | 0.000% |
| Training-fold majority direction | 774 | 0.03162 | -10.094% | 10.430% |

The fixed model's aggregate loss was less severe than the reporting baselines, but it remained far below the registered PF requirement and had no stable fold. This relative comparison does not change the verdict.

## 7. Results — 4h horizon

**Verdict: INSUFFICIENT_SAMPLE / NO-GO**

| Metric | Observed | Gate | Result |
|---|---:|---:|---|
| Pooled NetPF | **0.23269** | >= 1.10 | **FAIL** |
| Completed OOS trades | **82** | >= 100 | **FAIL** |
| MaxDD | **0.749%** | <= 15% | PASS |
| Stable folds | **0 / 4** | >= 3 / 4 | **FAIL** |
| Net PnL / initial equity | **-0.121%** | Reporting only | Negative |

The 4h protocol could not produce an eligible training set in three of the four fixed folds. Those folds were not fitted, retried, or altered.

| Fold | Train rows | Class 0 / 1 | Test rows | Training preflight | Evaluation |
|---|---:|---:|---:|---|---|
| WF-01 | 0 | 0 / 0 | 105 | INSUFFICIENT_SAMPLE | No fit |
| WF-02 | 105 | 0 / 105 | 46 | INSUFFICIENT_SAMPLE | No fit: one training class |
| WF-03 | 151 | 0 / 151 | 83 | INSUFFICIENT_SAMPLE | No fit: one training class |
| WF-04 | 234 | 83 / 151 | 105 | PASS | Evaluated |

WF-04, the only fitted 4h fold, produced **82 completed trades**, **NetPF 0.23269**, **net PnL -0.1214%**, and **MaxDD 0.7494%**. It was not stable. The pooled 4h trial therefore failed both sample sufficiency and the economic gates.

### 4h reporting baselines

| Baseline | Trades | NetPF | Net PnL / initial equity | MaxDD |
|---|---:|---:|---:|---:|
| Always long, same horizon and cost | 339 | 1.21898 | +0.151% | 1.114% |
| No-trade cash | 0 | Undefined | 0.000% | 0.000% |
| Training-fold majority direction | 234 | 0.12503 | -0.603% | 1.123% |

The always-long result is a preregistered reporting baseline, not an alternative model candidate and not a basis for changing the trial. It outperformed the fitted OFI model in this observed 4h sample. It does not convert the OFI result into a PASS, nor authorize a new strategy.

## 8. Interpretation

The registered OFI Logistic baseline did not demonstrate the required economic edge:

- **1h:** enough completed cohorts and acceptable MaxDD, but pooled NetPF was only 0.07347 and zero folds were stable. All four OOS folds lost money after costs.
- **4h:** only one fold could be fitted; the first three folds failed the preregistered minimum training/class requirements. The model produced only 82 completed OOS cohorts, NetPF 0.23269, and zero stable folds.
- **Overall:** both horizons failed independently. The exact registered verdict is INSUFFICIENT_SAMPLE / NO-GO.

The 4h limitation is a property of the fixed data boundary and the fixed horizon/fold protocol. It must not be “fixed” by shortening the horizon, relaxing class counts, changing fold boundaries, treating a segment as continuous, or introducing additional capture data after observing outcomes. Any such study would require a superseding preregistration and explicit Owner authorization. No post-outcome tuning, threshold sweep, candidate substitution, or selective rerun is approved by this ADR.

This evidence supports **not promoting this OFI baseline**. It does not establish that every possible order-flow model is ineffective; it establishes that these two preregistered G1 trials did not pass their registered tests.

## 9. Safety invariants and scope closure

The report records:

| Invariant | Recorded result |
|---|---|
| Champion | null |
| order_permission | false |
| trading_authorization | NO-GO |
| private_api_calls | 0 |
| real_order_attempts | 0 |
| paper_orders | 0 |
| Holdout bytes read | false |
| Post-Hack data read | false |
| External data requests during evaluation | 0 |
| Local execution | false |

Only the five registered M1.6 public artifacts were used. Holdout, Post-Hack, PRE-HACK legacy, private/account data, and unaudited external data remained excluded. No order was attempted or submitted. The evaluation was a research-only midprice simulation; there is no claim of executable fills and no claim of statistical independence for overlapping cohorts.

The collector scripts/capture_nobitex_microstructure.py and preregistration config/regime_m3/ofi_preregistration.json were not modified. The repository main branch was not modified. The result artifact is retained by GitHub Actions under ID 15155002532.

**Final M3.5 outcome: INSUFFICIENT_SAMPLE / NO-GO. No Champion, order permission, or trading authorization is created.**

## 10. References

1. ADR-057 — public Nobitex market-data and buy/sell field semantics.
2. ADR-058 — M1.6 capture acceptance, five-segment dataset boundary and coverage evidence.
3. ADR-059 — canonical locked OFI preregistration and acceptance gates.
4. GitHub Actions [Standard CI run #2585](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855203).
5. GitHub Actions [quality preflight run #125](https://github.com/angry0tiger-sudo/AliX/actions/runs/45897855197).
6. GitHub Actions [M3.5 evaluation run #2586](https://github.com/angry0tiger-sudo/AliX/actions/runs/45898505105).
7. M3.5 result artifact regime-m3-ofi-evaluation-45898505105 (artifact ID 15155002532).
## 11. Frozen main integrity

At the time ADR-060 was finalized, `refs/heads/main` remained at commit `cd9b13d42ac0cf0edd2df066ff81580e028146d6`. The root Tree SHA of that commit was independently read from the Git commit object as `5761c6c34143f8e6924f53a4e745c48932a5f791`, matching the previously frozen main Tree SHA. This ADR and its supporting documentation remain confined to `research/nobitex-hf-strategy`; no commit or file on `main` was changed.


## 12. Owner-approved closure — 2026-11-27

**Lifecycle decision: CLOSED. Canonical status: `CLOSED_NO_GO`.** The Owner explicitly approved Option A and closed the M3 OFI baseline path. This is a consensual project decision, not merely an automated evaluator label.

The closure rationale is horizon-specific and must not be collapsed into a generic “17 trades” claim:

- **1h:** 665 completed out-of-sample one-minute cohorts cleared the registered 100-cohort minimum, but NetPF was only 0.0734699215, net PnL was -7.9225%, and fold stability was 0/4. The evidence demonstrates no economic edge for this registered baseline; the 1h failure was not caused by low signal count.
- **4h:** only WF-04 could be fitted; WF-01 had no training rows and WF-02/WF-03 each had a single training class. The trial completed 82 OOS cohorts, NetPF 0.2326940033, net PnL -0.1214%, and stability 0/4. This horizon is both economically unsuccessful and sample/training-data insufficient under the locked protocol.

Accordingly, both registered M3 logical trials are now **`CLOSED_NO_GO`**. The M3 OFI path is closed on the existing 29h 10m 20.393s five-segment M1.6 dataset. No DeepLOB, Transformer, larger model, alternative estimator, feature expansion, threshold search, or other model complexity may be trained or tested on this dataset. No further M3 data collection is authorized by this closure. A materially new future research proposal would require a separate Owner decision and prospective preregistration; this ADR grants no such permission.

The Owner-approved closure does **not** claim that all conceivable microstructure strategies are impossible. It records that these two registered OFI trials did not meet their economic gates and that this dataset cannot support the proposed higher-capacity follow-on work without unacceptable overfitting risk.

### Final CI and repository integrity

- Standard CI [Run #2591](https://github.com/angry0tiger-sudo/AliX/actions/runs/45942283970), head `a2eb76bbc0e083994df5d185eea348cb5eeb0231`: **SUCCESS**. All four jobs passed: `quality`, `four-arm-lock-bare-env`, `postgres-integration`, and `freqtrade-runtime-smoke`. The Quality job completed strict Mypy, Bandit, dependency/lock/audit checks, architecture and complexity gates, unit/coverage tests, synthetic E2E and fault-injection, and safety/documentation checks successfully.
- Frozen `main` remains commit `cd9b13d42ac0cf0edd2df066ff81580e028146d6`, Tree SHA `5761c6c34143f8e6924f53a4e745c48932a5f791`.
- Only this research branch's documentation is changed by the closure record. No source code, workflow, model, capture data, or `main` content is changed.

### Safety closeout

`Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`; `private_api_calls=0`; `real_order_attempts=0`; `paper_orders=0`; `holdout_bytes_read=false`; `posthack_data_read=false`; `local_execution=false`. Holdout remains sealed.
