# Four-Arm Experiment: Baseline vs Kronos-Small vs Chronos-2 vs Combined

**Registration ID:** NOBITEX-FOUR-ARM-KC-20261113  
**Revision:** 1 (design-only preregistration)  
**Registration date:** 2026-11-13 UTC  
**Status:** PREREGISTERED — execution has NOT been authorized by this document  
**Owner directive:** Four-Arm Kronos/Chronos preregistration (2026-11-13)  
**Governance:** MRD v0.9; ADR-0012; ADR-032  
**Registration branch / base SHA:** research/nobitex-hf-strategy / 20b26d43f4948b34d50019577a56a8c164f41bab  
**Frozen main:** 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb (untouched)

## 1. Hypothesis and objective

**Hypothesis:** Adding a time-series foundation-model forecast to the existing Nobitex Donchian-20 / ATR(14) / 2-ATR trailing-stop baseline can improve risk-adjusted, after-cost out-of-sample results on the fixed pre-hack Nobitex 1h research dataset. The combined Kronos-Small + Chronos-2 arm is tested as a fourth, separately preregistered configuration; no superiority is assumed.

**Null:** None of the model-augmented arms satisfies every preregistered economic and statistical gate at both required cost scenarios across the required chronological folds.

**Objective:** Compare exactly four planned arms under identical data rows, chronological folds, label definitions, execution assumptions, risk limits, and cost scenarios. The four arms are a single preregistered family. They are not four independent opportunities to tune thresholds after seeing results.

No profitability claim is made by this registration. A PASS means eligibility to enter a separately authorized paper phase only; it is not a live-trading authorization.

## 2. Four arms

| Arm | Trial ID | Configuration | Permitted difference |
|---|---|---|---|
| A | NOBITEX-KC26-A | Baseline: Donchian-20 + ATR(14) + 2-ATR trailing stop | No foundation-model input |
| B | NOBITEX-KC26-B | Baseline + Kronos-Small | Kronos-Small forecast input only |
| C | NOBITEX-KC26-C | Baseline + Chronos-2 | Chronos-2 forecast input only |
| D | NOBITEX-KC26-D | Baseline + Kronos-Small + Chronos-2 | Both forecast inputs; combination semantics must be locked before execution |

### 2.1 Baseline definition and source

The baseline is the existing paper-only Nobitex MVP strategy on branch release/nobitex-mvp, commit 9f5e5f38d2145dc428173329a74c558b1f053ab4:

- Entry: candle close strictly above the highest high of the previous 20 completed candles (Donchian-20 close breakout).
- Volatility measure: ATR(14), Wilder convention as documented by the release.
- Exit: trailing stop ratcheted to the highest completed-bar high minus 2 × ATR.
- Source implementation: integrations/nobitex/mvp.py, blob 39deec55bdb8c8422efcbf84d34b1dd548cf8c6b.
- Release description: README-NOBITEX-MVP.md, blob 6e963524e60ab551849e84b683efe6d02a78001c.

The baseline's existing long-only / paper-only risk and execution constraints remain common to every arm. Foundation-model outputs must not independently bypass baseline risk, stop, position-size, data-quality, or safety controls.

### 2.2 Model-treatment boundary

B, C, and D may differ from A only by their preregistered forecast inputs and the one frozen forecast-to-decision adapter. No arm may change the baseline indicator periods, stop multiplier, risk sizing, eligible row set, fold boundaries, execution simulator, or fee assumptions.

The owner directive does not yet define:
- immutable Kronos-Small and Chronos-2 model repository revisions and weight SHA-256 values;
- runtime/library versions and model-license/provenance evidence;
- the exact forecast-to-entry/decision mapping and thresholds;
- the D-arm combination rule (for example, consensus vs. a fixed aggregation);
- pretraining-corpus overlap disclosure against the evaluation dates.

These are **hard pre-execution locks**, not tuning opportunities. They must be recorded in an immutable execution-lock append before any model load, inference, fit, or evaluation. No “latest” model alias, mutable tag, guessed weight hash, or post-result threshold selection is permitted. Until these fields are locked, this document registers the four-arm family but does not authorize its execution.

## 3. Data contract

**Only allowed dataset:** the existing Nobitex native BTC/USDT Spot 1h pre-hack dataset.

| Field | Preregistered value |
|---|---|
| Dataset | Nobitex BTC/USDT Spot OHLCV, 1h |
| Window | November 2021 through 2025-06-17 |
| Row count | 31,801 candles |
| CSV SHA-256 | 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e |
| Source reference | ADR-0012 and the pre-hack dataset ledger |
| Use in this experiment | Development / chronological walk-forward evaluation only |

The SHA and row count above are the fixed pre-hack source identity. This documentation-only turn did not fetch, open, copy, or inspect the pre-hack dataset bytes.

### 3.1 Contamination disclosure

ADR-0012 records that the final 12.5% segment of this 31,801-row pre-hack dataset is not a fresh, untouched holdout: AI-6 used it as its final holdout and DEAP's later OOS segment also includes it. Therefore:

- No result from this pre-hack dataset may be described as an untouched final-holdout result.
- The pre-hack dataset is restricted to this registered development / walk-forward comparison.
- The separately committed post-hack holdout remains sealed. No access is permitted in this experiment or its preparation.
- The final holdout may be considered only after at least one arm passes the complete preregistered gates **and** the Owner issues separate Release Authorization.

No post-hack row, price, timestamp series, OHLCV value, artifact, or manifest is an input to this experiment.

## 4. Target, horizon, and labels

- **Target:** forward return[t+1 : t+24h], using the 1h timeline.
- **Forecast horizon:** 24 hours = 24 hourly candles.
- **Label family:** triple-barrier.
- **Profit-taking barrier:** +2 × ATR(14).
- **Stop-loss barrier:** −2 × ATR(14).
- **Vertical barrier / maximum holding period:** 24 hours (24 hourly candles).
- ATR used to set barriers must be calculated from information available at decision time t only.

The exact executable-price anchor for return[t+1 : t+24h], barrier-touch fill convention, and the treatment of a candle that touches both horizontal barriers are not specified by the owner directive. They must be frozen in the execution-lock append before any label generation or model evaluation. No favorable tie-break may be selected after observing outcomes.

## 5. Point-in-time and leakage controls

1. **As-of rule:** every feature and model input at decision time t may use only observations available at or before t. No future candle, revised future label, or future-derived normalization is allowed.
2. **Purge:** 24 hours (24 hourly bars) around every train/evaluation boundary.
3. **Embargo:** 24 hours (24 hourly bars) after the relevant boundary.
4. **Scaler:** fitted on the training segment of each fold only. Validation and test segments must never fit, update, or influence a scaler.
5. **Model warm-up:** 512 historical 1h candles for Kronos-Small / Chronos-2. Warm-up rows are context only and are excluded from scoring/trade counts.
6. **Model refresh:** forecast refresh cadence is 24 hours; the schedule and exact as-of timestamp must be identical wherever a model is used.
7. **Common rows:** all arms use the same eligible OOS timestamps after common warm-up, data-quality checks, and purge/embargo removal. No arm-specific row deletion to improve its result.
8. **No random shuffle:** chronological ordering is mandatory; leakage checks fail closed.

## 6. Fold design

- Minimum: **three** chronological walk-forward folds.
- Split type: walk-forward, chronological; no random K-fold and no shuffled rows.
- The same fold boundaries, eligible timestamps, purge, embargo, and scoring windows apply to A/B/C/D.
- Every fold must have explicit train, validation, and OOS/test intervals. Training data may precede validation/test data only.
- Fold-level metrics and pooled out-of-fold metrics must both be retained; no fold may be omitted after results are observed.

The owner directive specifies the minimum fold count but not exact cut points or train/validation/test ratios. The exact UTC and row-index boundaries must be computed from the immutable pre-hack manifest and committed in the execution lock before the first fit or evaluation. A fold definition may not be changed after any result is observed.

## 7. Cost scenarios

Every arm must be reported separately under both all-in round-trip scenarios:

| Scenario | Round-trip cost | Reporting |
|---|---:|---|
| Cost-50 | 50 bps | Mandatory, separate result |
| Cost-90 | 90 bps | Mandatory, separate result |

No averaging, substitution, silent fee change, or presentation of gross metrics as net metrics is allowed. Cost decomposition and fill/slippage mechanics must be identical across arms and frozen in the execution lock before evaluation.

## 8. Absolute acceptance gates

An arm can be declared PASS only against the preregistered gates:

- Net Profit Factor (after costs) ≥ 1.10.
- Completed trades ≥ 100.
- Maximum drawdown ≤ 15%.
- Deflated Sharpe Ratio: DSR > 0, using the historical and current trial cohort described in Section 9.
- Stability across at least three chronological walk-forward folds.

All five gates are mandatory. Both Cost-50 and Cost-90 results must be reported separately for every arm. Fold-level values must be shown alongside pooled OOS values; a pooled result must not conceal a failed or unstable fold. The exact fold-stability aggregation rule and the exact DSR implementation/configuration must be frozen before execution. Missing fold evidence, insufficient trade support, undefined metrics, unpinned models, or unreconciled DSR evidence produce NO-GO / INSUFFICIENT_EVIDENCE, not a pass.

## 9. Multiple testing and DSR trial accounting

The DSR cohort must include the historical trials recorded in docs/research/TRIAL_REGISTRY.md and all four preregistered arms. Failed, cancelled, partial, rejected, and tuning variants must be included whenever they reached candidate evaluation. Execution attempts that demonstrably failed before candidate evaluation are recorded as attempts, not automatically counted as economic trials.

Known historical records include:

| Family | Registered / evidenced quantity | Qualification |
|---|---:|---|
| V4-T | 32 candidate variants; 4 trained | 24 sample-gate rejections and 4 inner-CV failures are retained in the ledger |
| V4-AP | 36 registered candidate cells | 36 Sharpe estimates in the completed cohort |
| DEAP | 7,842 verified unique genome evaluations | Six earlier failed runs may contain additional partial evaluations |
| DEAP additional owner-reported ledger | 828 genomes | Owner-reported; overlap / independent identity not fully reconciled |
| AI-6 | At least 2 distinct evaluated policy experiments; 3 workflow execution records | Exact historical configuration count remains unknown |
| HF research | 41 execution records / 42 attempts in the recorded #1–#41 ledger | Unique economic configurations and evaluations remain UNKNOWN; later HF ledger entries #44 and #45 are also recorded |

These units are not homogeneous and must not be blindly summed into one number. The 828 owner-reported DEAP genomes must remain separately identified until overlap is reconciled. The HF workflow execution count is not a proven economic-trial count.

**GLOBAL_N remains UNKNOWN** pending final candidate-level consolidation. No guessed exact global N may be written. Before any arm can claim DSR PASS, the DSR calculation must disclose the exact reconciled count if available, the auditable conservative lower bound, the unresolved remainder, and a sensitivity result. If the unknown remainder prevents a defensible DSR>0 determination, the arm is INSUFFICIENT_EVIDENCE / NO-GO.

New planned trial IDs (four configurations; zero evaluated in this registration):
- NOBITEX-KC26-A
- NOBITEX-KC26-B
- NOBITEX-KC26-C
- NOBITEX-KC26-D

Every additional model revision, threshold, feature set, ensemble weight, seed-driven candidate, or post-result tuning variant is a new trial and must be registered before evaluation.

## 10. Fixed decision rule

1. **No arm passes:** document NO-GO; do not open the final holdout.
2. **Exactly one arm passes:** that arm is eligible for the separately authorized paper phase. This is not live authorization.
3. **More than one arm passes:** select the passing arm with the highest DSR, using the same reconciled trial cohort and calculation method.
4. A result that cannot satisfy the gates because of unknown DSR N, insufficient support, incomplete folds, or missing provenance is not a winner.
5. No post-result retuning, threshold search, arm redesign, or model replacement is allowed under these trial IDs. A new immutable preregistration and new trial IDs are required first.

The final post-hack holdout remains closed until one arm has passed every gate and the Owner has issued separate Release Authorization. Passing this pre-hack comparison alone does not open the holdout.

## 11. Execution-lock items required before the next run

The following values are intentionally not guessed. They must be committed before any model load, inference, training/fit, label generation, backtest, or economic evaluation:

- Exact Kronos-Small and Chronos-2 repositories, immutable revisions, weight SHA-256 values, runtime/library versions, licenses, and pretraining-corpus overlap disclosures.
- Forecast-to-decision mapping for B and C, and the exact combined decision rule for D.
- Exact fold UTC timestamps and row indices, validation/test allocation, and deterministic seed(s).
- Exact target price anchor, barrier fill/collision/tie convention, and treatment of vertical-barrier expiry.
- Exact cost decomposition, slippage/fill simulator revision, and execution assumptions.
- Exact DSR method/configuration and auditable trial-count sensitivity using the reconciled registry.

If any item is missing or cannot be verified, execution remains blocked. Completing this list requires a separate immutable execution-lock record and an explicit Owner run directive. This preregistration itself is not an instruction to start the experiment.

## 12. Safety and status

This commit registers documentation only. No dataset was read, no model was loaded, no model fit/training was performed, and no backtest or economic evaluation was run.

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- post-hack holdout access = FORBIDDEN in this experiment
- execution status = NOT_STARTED

**Owner approval:** this owner directive authorizes preregistration documentation only. Experiment execution requires a later explicit Owner directive.
