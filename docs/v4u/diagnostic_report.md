# V4-U Diagnostic Root Cause Report — V4-T Run #20

**Status:** Diagnostic complete; no model training or parameter tuning performed for V4-U.  
**Date:** 2026-11-03  
**Repository:** `angry0tiger-sudo/AliX`  
**Branch:** `research/ml-core-v4t-economic-score`  
**Analyzed V4-T source SHA:** `d5fe33c5cf32bb41cc083acc11a9935d30324173`  
**Run:** [GitHub Actions Run #20 / ID 41625662854](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625662854)  
**Artifact:** [alix-v4t-economic-score / ID 13229820663](https://github.com/angry0tiger-sudo/AliX/actions/runs/41625662854/artifacts/13229820663)  
**Artifact ZIP SHA-256:** `7fddda74100eb554506c7a8716688eb528a7aa88bac5f1c549169c63f85a8997`

---

## 1. Executive finding

Run #20 completed successfully and passed its report, probability-row, economic and safety validation. The artifact records 16 experiments, 32 candidate variants, only 4 trained variants, and 10,512 persisted row-level probability records.

The failures are not one generic “insufficient data” problem:

1. **The symmetric 1R control trained zero candidates because its HOLD class failed the configured support gates in all four pair/timeframe contexts.** The contexts had adequate total row counts, but too few HOLD labels, particularly in outer validation. All 16 control candidate variants were skipped as `SKIPPED_INSUFFICIENT_SAMPLES`.
2. **The 1.5R primary arm had 8 candidate variants skipped by the outer sample gate and 4 skipped by the purged Inner-CV class-support preflight.** The remaining 4 candidates all came from BTC/USDT 1h.
3. **The 1.5R target does make the TP event harder to reach than 1R**, and its label distribution reflects that: HOLD support rises, while SELL/stop outcomes dominate. But the evidence does not establish that the 1.5 ATR target alone explains every skip. The ETH 1h context passed the outer sample gate but failed the complete Inner-CV support requirement; its per-fold class counts were not persisted.
4. **The economic failure is not explained solely by transaction costs.** The two candidates that actually produced trades had pooled gross profit factor below 1 before fees/slippage (0.790). Applying the configured costs reduced pooled net PF to 0.0388.
5. **The headline 43.06% drawdown is not the drawdown of a combined portfolio.** It is the mean candidate drawdown over four trained variants, two of which placed no trades and contributed zero drawdown. The two active candidates individually reached 88.41% and 83.81% maximum drawdown. Cross-candidate overlap was not modeled.

The current experiment therefore provides evidence of a severe class-support problem, weak BUY discrimination, and a negative gross as well as net trading result. It does **not** prove that the market was merely choppy, that microstructure data will solve the problem, or that 4h/1d is a better timeframe.

## 2. Scope, provenance and hard constraints

### 2.1 What was analyzed

The GitHub artifact contained:

- `alix-v4t-economic-report.json`
- `alix-v4t-economic-comparison.csv`
- `alix-v4t-trades.csv`
- `alix-v4t-row-level-probabilities.csv`
- `alix-v4t-feature-importance.csv`

The workflow log independently confirms the report SHA, the candidate/experiment counts, the trained-candidate count, validation PASS and the closed safety fields. The report schema is `alix.ml-core.v4t-economic-score.v1`. The run requested 730 days of public Binance Spot one-minute input for BTCUSDT and ETHUSDT (1,051,200 rows each; zero missing one-minute intervals), then resampled to 1h and 4h.

V4-U is a read-only analysis of those persisted outputs. No V4-U model, estimator, search, parameter change, target change, objective change, economic replay or new training run was performed. No production source was changed.

### 2.2 Non-negotiable constraints

The report and any subsequent work remain subject to the following state:

| Invariant | Required state |
|---|---|
| Champion | `null` |
| Order permission | `false` |
| Trading authorization | `NO-GO` |
| Private API calls | `0` |
| Real order attempts | `0` |
| Execution scope | Spot only; no short positions |
| Sequence/LSTM model | Not permitted |
| CI coverage and quality gates | Unchanged; coverage requirement remains at least 90% |
| Main | Must remain unchanged |
| V4-S | Must remain unchanged |

Run #20's report confirms `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, `private_api_calls=0` and `real_order_attempts=0`. It also records `sequence_model_used=false` and promotion blocked pending independent OOS and owner review.

## 3. Sample Gate root cause

### 3.1 Experiment structure and thresholds

The design is 2 symbols × 2 timeframes × 2 arms × 2 feature profiles = 16 experiments. Two weighting-policy variants are produced for each experiment, giving 32 candidates total. Each arm therefore contains 8 experiments and 16 candidate variants.

V4-T uses:

- Train minimum: 20 samples per class.
- Outer validation minimum: 30 samples per class.
- Purged Inner-CV minimum: 20 samples per class in each of 3 expected folds.
- Timeframes in this run: 1h and 4h.
- Arms: symmetric 1R (TP 1.0 ATR / SL 1.0 ATR) and asymmetric 1.5R (TP 1.5 ATR / SL 1.0 ATR), both with 24-candle label horizon.

The support failure is specifically dominated by the **HOLD** class. Total training/validation row counts are not generally absent; the required per-class support is.

### 3.2 HOLD support by pair and timeframe

The following counts are from the persisted experiment-level label distributions. Percentages are calculated against the corresponding train or validation row count.

| Arm | Pair | TF | Train rows | Train HOLD | Train HOLD % | Validation rows | Validation HOLD | Validation HOLD % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Control 1R | BTC/USDT | 1h | 12,164 | 186 | 1.529% | 2,628 | 6 | 0.228% |
| Control 1R | BTC/USDT | 4h | 2,966 | 9 | 0.303% | 657 | 0 | 0.000% |
| Control 1R | ETH/USDT | 1h | 12,164 | 89 | 0.732% | 2,628 | 11 | 0.419% |
| Control 1R | ETH/USDT | 4h | 2,966 | 16 | 0.539% | 657 | 0 | 0.000% |
| Primary 1.5R | BTC/USDT | 1h | 12,164 | 299 | 2.458% | 2,628 | 38 | 1.446% |
| Primary 1.5R | BTC/USDT | 4h | 2,966 | 45 | 1.517% | 657 | 2 | 0.304% |
| Primary 1.5R | ETH/USDT | 1h | 12,164 | 236 | 1.940% | 2,628 | 37 | 1.408% |
| Primary 1.5R | ETH/USDT | 4h | 2,966 | 59 | 1.989% | 657 | 8 | 1.218% |

### 3.3 Exact skip accounting

| Arm | Candidate variants | Trained | `SKIPPED_INSUFFICIENT_SAMPLES` | `SKIPPED_INSUFFICIENT_INNER_CV_CLASS_SUPPORT` |
|---|---:|---:|---:|---:|
| `CONTROL_SYMMETRIC_1R` | 16 | 0 | 16 | 0 |
| `PRIMARY_1_5R` | 16 | 4 | 8 | 4 |
| **Total** | **32** | **4** | **24** | **4** |

**Control 1R: all 16 variants skipped at the outer sample gate.**

- BTC/USDT 1h had 186 HOLD train rows, so train support passed, but only 6 HOLD validation rows, below the required 30.
- ETH/USDT 1h had 89 HOLD train rows but only 11 validation HOLD rows, again below 30.
- BTC/USDT 4h had only 9 train HOLD rows and zero validation HOLD rows.
- ETH/USDT 4h had 16 train HOLD rows and zero validation HOLD rows.

The control therefore did not fail because the workflow lacked market rows. It failed because the three-class target produced too few HOLD examples to meet both train and validation support contracts. The Inner-CV stage was not reached.

**Primary 1.5R: four distinct outcomes across the four contexts.**

- BTC/USDT 1h: outer train/validation support passed and all three Inner-CV folds passed preflight; 4 candidate variants trained.
- ETH/USDT 1h: outer sample gate passed (236 train HOLD and 37 validation HOLD), but all 4 variants were skipped by `SKIPPED_INSUFFICIENT_INNER_CV_CLASS_SUPPORT`. Diagnostics record `tuning_inner_cv_class_support_insufficient`, expected fold count 3 and minimum class support 20. The artifact does not preserve per-fold class counts, so the exact failing fold and class cannot be identified from Run #20.
- BTC/USDT 4h: 4 variants skipped because validation HOLD support was 2, below 30.
- ETH/USDT 4h: 4 variants skipped because validation HOLD support was 8, below 30.

Thus the owner-reported “12 of 16 did not train” is correct, but its root cause is **8 outer sample-gate skips plus 4 Inner-CV support skips**, not 12 identical sample failures.

### 3.4 Is 1.5 ATR within 24 candles too difficult?

The label contract defines BUY when the TP barrier is touched first, SELL when the SL barrier is touched first (same-candle collisions are stop-first), and HOLD when neither barrier is touched during the 24-candle horizon. Therefore, HOLD means **no barrier touched**, not “the market was flat” and not “the model abstained”.

Relative to symmetric 1R, the 1.5R primary arm has higher HOLD counts in each context, consistent with a farther TP creating more events that do not reach either barrier before expiry. But the HOLD share remains small, and SELL remains the dominant class. Only BTC 1h passed the full support pipeline. ETH 1h passed outer validation but could not supply adequate support in all three inner folds.

The data therefore support this precise conclusion: **the chosen triple-barrier geometry creates a highly imbalanced three-class label, and the 1.5 ATR TP materially reduces the number of completed TP events; however, the artifact does not prove that the 1.5 ATR distance alone caused all training failures.** The per-fold Inner-CV distributions are missing, and the 4h validation sets are especially sparse.

## 4. Trade-log analysis — PRIMARY_1_5R

### 4.1 Scope of the 423 trade rows

All 423 trade rows belong to BTC/USDT 1h and only two trained candidate variants:

| Candidate | Trades | TP | SL | Collision | Model SELL exit | Net PF | Candidate max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline / `weight_balanced` | 230 | 26 | 118 | 2 | 84 | 0.03751 | 88.41% |
| Baseline+Trend / `weight_balanced` | 193 | 24 | 101 | 1 | 67 | 0.04016 | 83.81% |
| **Combined rows** | **423** | **50** | **219** | **3** | **151** | **0.03880 pooled** | **Not a portfolio DD** |

Two other candidates did train but produced no trades. Their calibration window had no raw BUY signals, so the dynamic BUY threshold was set to 1.0 and the forward BUY count remained zero.

### 4.2 TP/SL hit ratio and exit semantics

The raw exit-reason distribution across 423 trade rows is:

| Exit reason | Count | Share of all trade rows | Interpretation |
|---|---:|---:|---|
| `TP_HIT` | 50 | 11.82% | 1.5 ATR take-profit barrier reached |
| `SL_HIT` | 219 | 51.77% | 1.0 ATR stop-loss barrier reached |
| `COLLISION` | 3 | 0.71% | Both barriers touched in one candle; stop wins |
| `SELL` | 151 | 35.70% | Model SELL signal closed an existing long at next-bar open |

Among barrier-coded exits only (TP, SL and collision), 50/272 (18.38%) were TP and 222/272 (81.62%) were stop-loss-like. Across all trade rows, TP was 50/423 (11.82%) and stop-loss-like exits were 222/423 (52.48%).

**Do not classify the 151 `SELL` exits as stop-loss hits.** The execution source defines Spot-only long positions: a model SELL closes an open long; the replay does not open a short position. Likewise, the model's SELL label represents the target's stop-loss-first outcome, but an execution-time `SELL` reason is a model-triggered position close. Those are different fields and semantics.

The trade-log outcome counts are not a direct 50-versus-219 model confusion matrix. They describe how open positions exited after model signals and execution logic.

### 4.3 Holding period

Holding period was calculated from each trade's entry timestamp to its exit timestamp:

- Mean: **83.26 minutes** (1.39 hours).
- Median: **60 minutes**.
- Minimum: **0 minutes**.
- Maximum: **900 minutes** (15 hours).

By exit reason, mean holding time was approximately 106.8 minutes for TP, 43.6 minutes for SL, 0 minutes for collision, and 134.7 minutes for model SELL exits. No trade in this log ended with `HORIZON_EXPIRY`; positions closed earlier through TP, SL, collision or a model SELL. The 24-candle label horizon is not the same as observed position holding time.

### 4.4 Gross versus net economics

| Measure | Combined active trade rows |
|---|---:|
| Trade rows | 423 |
| Gross profit factor (before modeled costs) | 0.78991 |
| Net profit factor (after modeled costs) | 0.03880 |
| Gross market P&L | -$979.24 |
| Net P&L | -$17,214.13 |
| Fees | $9,019.39 |
| Slippage cost | $7,215.51 |
| Total modeled fees + slippage | $16,234.89 |
| Net winners / losers | 54 / 369 |

The gross PF below 1 means the active candidates were already losing before the configured fees and slippage. Costs then amplified the loss substantially. The economic failure is therefore not accurately described as “only a cost problem”; both signal/trade selection and transaction costs are material.

### 4.5 Drawdown interpretation and overlap limitation

The arm-level report's `mean_candidate_max_drawdown_percent=43.0562` is an arithmetic mean across four trained candidates. Two trained variants generated zero trades and therefore had zero drawdown. The two active variants instead had maximum drawdowns of 88.41% and 83.81%.

The artifact also records `portfolio_overlap_modeled=false`. The two active candidate logs share 119 entry timestamps; the union of their entry timestamps is 304, not 423. Consequently:

- 423 is the number of candidate trade rows, not 423 unique independent market opportunities.
- The pooled net PF is a diagnostic aggregation across two candidate replays, not a portfolio-aware combined strategy result.
- The 43.06% mean candidate DD must not be quoted as portfolio max drawdown.
- Candidate-specific drawdowns are the relevant evidence available from this artifact.

## 5. Direction prediction quality

### 5.1 Evaluation method and actual label prevalence

The row-level probability artifact contains four trained BTC/USDT 1h candidate streams, each with 2,628 validation rows: 525 calibration rows and 2,103 forward-validation rows. The forward-validation target distribution is identical for these candidates:

| Actual target | Forward rows | Prevalence |
|---|---:|---:|
| BUY | 442 | 21.02% |
| SELL | 1,625 | 77.27% |
| HOLD | 36 | 1.71% |
| **Total** | **2,103** | **100.00%** |

The prediction action below uses the repository's persisted V4-T rule: HOLD only when `prob_hold >= 0.60` and greater than both directional probabilities; otherwise BUY when `prob_buy >= prob_sell`, else SELL. This is the raw three-way action, before the additional BUY-only confidence filter used for execution.

### 5.2 Forward action confusion matrices

**Baseline / weight-balanced**

| Predicted \ Actual | BUY | SELL | HOLD |
|---|---:|---:|---:|
| BUY | 198 | 244 | 0 |
| SELL | 648 | 970 | 7 |
| HOLD | 32 | 4 | 0 |

- BUY precision: **22.55%**; BUY recall: **44.80%**.
- SELL precision: **79.64%**; SELL recall: **59.69%**.
- HOLD precision and recall: **0%**.
- Three-way action accuracy: **55.54%**.

**Baseline+Trend / weight-balanced**

| Predicted \ Actual | BUY | SELL | HOLD |
|---|---:|---:|---:|
| BUY | 213 | 229 | 0 |
| SELL | 658 | 965 | 6 |
| HOLD | 30 | 6 | 0 |

- BUY precision: **23.64%**; BUY recall: **48.19%**.
- SELL precision: **80.42%**; SELL recall: **59.38%**.
- HOLD precision and recall: **0%**.
- Three-way action accuracy: **56.02%**.

BUY prevalence is already 21.02%. The two balanced variants' raw BUY precision of 22.55% and 23.64% is only modestly above that prevalence baseline. The HOLD class is never correctly identified in these forward action outputs.

### 5.3 Confidence-filtered BUY entries

The first 525 validation rows were used for calibration; the later 2,103 rows were used for forward replay. The selected BUY thresholds and accepted BUY outcomes were:

| Candidate | Selected BUY threshold | Accepted forward BUY signals | Actual BUY among accepted | Accepted BUY precision |
|---|---:|---:|---:|---:|
| Baseline / weight-balanced | 0.39951518 | 333 | 65 | 19.52% |
| Baseline+Trend / weight-balanced | 0.40725685 | 308 | 68 | 22.08% |

The confidence filter therefore did not produce a high-quality BUY subset. The other two trained variants had `NO_BUY_SIGNALS_IN_CALIBRATION`, selected threshold 1.0 and generated no BUY trades in forward validation. Their no-trade result is not evidence of a successful risk filter; it is evidence that the calibration segment could not produce a BUY threshold from their raw predictions.

**Diagnosis:** the dominant failure is not merely “a choppy market”. The artifact shows strong SELL-class prevalence, zero HOLD recall, weak BUY precision close to the BUY base rate, and no profitable accepted BUY subset. Those are measurable prediction/decision-quality failures. A market-regime explanation remains a hypothesis because this artifact does not contain a separate regime attribution analysis.

## 6. Existing feature and timeframe contracts

### 6.1 This is not only an RSI/MACD/ATR model

The baseline feature contract currently contains 13 point-in-time OHLCV features:

- Returns at 1, 3 and 6 bars.
- Volatility, range and candle-body percentage.
- Volume change and volume Z-score.
- Close versus SMA.
- ATR proxy and ATR percentile.
- RSI(14) and normalized MACD histogram.

The extended Trend profile adds EMA distance/spread/slope, momentum, ADX, +DI and -DI. V4-T uses the `Baseline` and `Baseline+Trend` profiles. The extended code also defines separate Volatility and Volume groups, but those profiles were not selected in V4-T Run #20.

Accordingly, it is inaccurate to describe the run as testing only RSI, MACD and ATR. It used a broader set of 13 baseline features and 8 added trend features in the second profile. It is equally inaccurate to claim that all extended feature groups have been evaluated: the separate Volatility and Volume profiles were not part of this run.

No order-book depth, bid/ask queue, trade-tick flow or other true microstructure feed is present in the artifact. The run used OHLCV candles. Microstructure feature value cannot be established without an authorized source, data rights, timestamps and a causal alignment contract.

### 6.2 Timeframe evidence

- V4-T Run #20 tested **1h and 4h only**.
- 1h BTC was the only pair/timeframe context to pass all support gates.
- Both 4h contexts failed outer validation HOLD support in the 1.5R arm.
- 15m and 30m were not tested in this run. The evidence cannot justify removing them.
- The current execution-timeframe mapping in `economic_backtest.py` includes 15m, 30m, 1h and 4h, but not 1d. The current V4-T contract also declares only 1h and 4h.
- 1d would therefore require an explicit implementation/contract review before it could be evaluated. It also has fewer bars over a fixed calendar history, so it cannot be assumed to solve sample-support scarcity.

The failed 4h support does not prove that every higher timeframe will fail, but it is a warning against making “go higher timeframe” the default fix. The only successfully trained context is 1h, yet its economics are unacceptable.

## 7. Architecture alternatives for owner review

No option below is authorized or implemented by this report. They are alternatives for a separately approved next phase, not an invitation to start another tuning sweep.

### Option A — Timeframe redesign

**Proposal:** first investigate the failed label support by timeframe and data horizon; do not remove 15m/30m or adopt 4h/1d solely from Run #20.

**Evidence and trade-offs**

- 1h BTC is the only context with sufficient support through the three-fold Inner-CV preflight, but its economic result is severely negative.
- 4h validation HOLD support is inadequate in both arms and both assets.
- 15m/30m were not tested here, so there is no comparative result.
- 1d is absent from the V4-T timeframe contract and current replay mapping. Supporting it requires a separate timeframe/ATR/horizon/execution audit.
- Fewer high-timeframe bars can make per-class support worse, not better. A longer calendar window, if considered, would need separate data-availability, regime and leakage review rather than an assumed benefit.

**Assessment:** do not pivot to 4h/1d as the presumed fix; the current 4h evidence is negative and 1d is unsupported. Do not delete 15m/30m based on a run that never evaluated them. Timeframe selection should follow a target-support and economic design review, not a parameter sweep.

### Option B — Primary model plus Meta-Label filter

**Proposal:** preserve the separation between a primary directional/event proposal and a secondary model that estimates whether the proposal is likely to have positive net outcome after costs. This is a filter, not an order signal.

**Existing project architecture**

AliX already contains a research-only Meta-Label implementation under `services/meta_labeling/`:

- Primary predictions are generated on chronological out-of-fold (OOF) folds.
- Primary-label horizons are purged before test folds.
- Meta training is limited to prior OOF events whose barrier labels have matured.
- The binary meta target is positive net return after modeled fees and slippage.
- The LogisticRegression meta model is serialized as JSON and evaluated walk-forward.
- Its output is a research filter flag only; it does not create entry/exit orders, Risk Gate approval or position sizing.

**Important distinction:** the existing architecture does **not** make Model 2 a position-sizing model. A size controller would be a separate risk/sizing component with an independently reviewed exposure, drawdown, liquidity and authorization contract. Meta confidence must never become order permission.

**Evidence and trade-offs**

- This may separate “is the primary proposal directionally correct?” from “is this proposal worth taking after costs?” and can help identify whether filtering adds value.
- It also reduces sample size because the secondary model only trains on eligible OOF events with matured outcomes; the current class-support problem could become worse.
- Primary signal errors propagate to the meta stage.
- The project's recorded Phase 5 evaluation is synthetic mechanics only. In all three evaluated folds, the historical-rate baseline beat Meta-Label on Brier score and log-loss, and the filtered mean net return was more negative than the unfiltered mean. This is a negative synthetic result, not market evidence, but it means Meta-Label is not yet validated as a solution.

**Assessment:** this is architecturally coherent as a research filter, but should not be presumed to repair the weak primary BUY signal. It needs owner-approved, authorized real-market OOS evidence before acceptance. It must remain separate from any position-sizing decision.

### Option C — Target and feature redesign (including regression)

**Proposal:** review the target semantics and information available to the model before adding more indicators or changing the estimator family.

**Evidence and trade-offs**

- Current V4-T is a three-class barrier-outcome classifier, not a simple next-candle up/down classifier. Its classes combine direction and barrier-touch behavior over 24 candles.
- The resulting class distribution is dominated by SELL; HOLD support is extremely sparse, and BUY precision is weak.
- The selected V4-T features already include returns, volatility, candle range/body, volume statistics, SMA distance, ATR, RSI, MACD and additional trend/ADX/DI features.
- The separate Volatility and Volume profiles in the extended feature module have not been tested by this run. Their existence is not evidence that adding them will improve out-of-sample economics.
- Microstructure features require appropriate L2/order-book or trade-flow data. They cannot be reconstructed honestly from the present OHLCV-only artifact.
- A regression target such as forward net return, normalized excursion or expected net edge could avoid forcing every event into BUY/SELL/HOLD classes and may support ranking. However, regression does not remove non-stationarity, heavy tails, label overlap, cost sensitivity or the need for calibrated decision thresholds. It can also predict noisy point estimates that are not tradable.

**Assessment:** the strongest issue visible here is target/class semantics and weak BUY discrimination, not proven lack of indicators. A target-design review is justified before feature expansion. Microstructure and regression are hypotheses requiring their own data/target/validation contracts, not automatic upgrades.

## 8. Recommended decision and evidence gates

### 8.1 Recommended disposition

**Keep the current V4-T candidate set at NO-GO and stop tuning. Do not start a new model run.**

For the next owner-approved design phase, the most defensible order of work is:

1. Specify what the primary model is expected to predict (barrier event occurrence, direction conditional on an event, or expected net edge) and document the complete label/support distribution by chronological fold.
2. Decide whether a Meta-Label stage is a filter worth evaluating on authorized real-market OOS data. Keep it strictly separate from risk sizing and order permission.
3. Only then consider a timeframe or feature-data change. A 1d path requires explicit support; microstructure requires an authorized source; regression requires a pre-registered target and evaluation contract.

This ordering addresses the demonstrated weaknesses without assuming that a different timeframe, a larger feature set or a second model will create an edge.

### 8.2 Evidence required before any next experiment

- Per-class counts for every train, calibration, outer-validation and Inner-CV fold, including exact failing class/fold reasons.
- Event-level label distribution and TP/SL/vertical-expiry timing, with incomplete-tail events kept distinct from no-hit outcomes.
- BUY/SELL/HOLD confusion, class-wise precision/recall, calibration and prevalence baselines on untouched forward data.
- Trade-level and portfolio-level accounting separated: unique opportunity IDs, overlap policy, candidate attribution, net costs, equity curve and true portfolio max drawdown.
- Explicit gross-versus-net results and cost sensitivity under owner-approved, realistic assumptions.
- For Meta-Labeling: OOF primary provenance, label maturity, conditional sample support, baseline comparison and end-to-end OOS calibration.
- For microstructure: authorized point-in-time raw source, exchange timestamp semantics, bid/ask/order-book/trade-flow schema, missing-data policy and leakage tests.
- For regression: exact continuous target, horizon/maturity contract, robust loss/baseline, calibration/ranking metrics and the economic decision rule.
- Independent model-risk review and an explicit Owner Directive before implementation or execution.

## 9. Artifact limitations and unresolved questions

The following cannot be recovered conclusively from Run #20's five uploaded files:

1. **Exact Inner-CV failure detail:** the report records the reason code, expected fold count and minimum class support, but not each fold's class distribution or the exact failing fold/class in ETH 1h.
2. **Underlying one-minute OHLCV:** the run log records successful full data retrieval and its row/missing-interval checks, but the uploaded artifact contains no source candle CSV. This report therefore does not independently recompute the barrier labels from raw candles.
3. **Portfolio-level result:** overlap is explicitly not modeled. The 119 shared entry timestamps and two candidate-specific replays prevent treating 423 rows as independent portfolio trades.
4. **Regime attribution:** no separate trending/ranging/regime-conditioned analysis is persisted. “Choppy market” is not proven.
5. **Microstructure:** no L2/order-book/trade-flow source is included.
6. **1d or 15m/30m economics:** not measured in this run.

These limitations are part of the diagnosis and must not be silently filled with assumptions.

## 10. Final safety and change record

Run #20's validated report states:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `sequence_model_used = false`
- `promotion_status = BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW`

V4-U made no changes to production code, labels, parameters, objective functions, CI thresholds, coverage policy, model families or workflow definitions. No model was trained and no workflow was manually dispatched for V4-U. The report is documentation-only and is committed on the V4-T research branch. The V4-T economic workflow's push path filter excludes `docs/v4u/**`, so this documentation change does not request another V4-T economic-training run; the repository's normal push-triggered CI policy remains unchanged.

**Disposition:** V4-T economic candidate remains **NO-GO**. This V4-U report is evidence for owner review only. No next-phase implementation or training is authorized by this document.
