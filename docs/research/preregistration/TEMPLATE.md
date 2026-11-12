# Research Preregistration Template — AliX

Status: REQUIRED BEFORE ANY NEW TRAINING, TUNING, SWEEP, OR ECONOMIC EVALUATION.

A preregistration is immutable once the first trial starts. Corrections require a new
preregistration revision and a linked ADR; never silently edit a hypothesis after seeing outcomes.

## 1. Trial identity and authority

- Trial ID:
- Trial family / parent registry ID:
- Preregistration version:
- Preregistered at (UTC):
- Owner directive / ADR:
- Responsible reviewer:
- Candidate code commit SHA:
- Execution workflow path + blob SHA:
- Intended outcome: edge discovery / falsification / diagnostic only:

## 2. Falsifiable hypothesis

- Hypothesis (one sentence, falsifiable):
- Null hypothesis:
- Alternative hypothesis:
- Economic mechanism:
- What observation would falsify the hypothesis?
- Explicitly forbidden post-result changes:

## 3. Target and timing

- Target / label definition:
- Target direction / class:
- Prediction horizon:
- Label window:
- Entry decision timestamp:
- Assumed fill timestamp / price:
- Exit definition:
- Maximum holding horizon:
- Target construction commit + SHA:

## 4. Market, timeframe and immutable data

- Venue:
- Market type / direction:
- Symbol:
- Timeframe:
- Exact UTC start (inclusive):
- Exact UTC end (inclusive/exclusive, state which):
- Provider / endpoint:
- Dataset artifact path:
- Dataset CSV SHA-256:
- Manifest SHA-256:
- Expected row count:
- Actual row count:
- Timestamp grid / timezone:
- Missing intervals / duplicate rows:
- Gap policy: FAIL-CLOSED; no interpolation, backfill, forward-fill, synthetic bars, masking or splice.
- Data authority: public market data only / not account, order or fill truth:
- Data retrieval workflow run ID:

## 5. Split and leakage controls

- Split method and exact row/time boundaries:
- Train fraction / rows:
- Validation fraction / rows:
- Final holdout fraction / rows:
- Fold count (minimum 3 chronological folds):
- Purge length:
- Embargo length:
- Label horizon used for purge:
- Scaler fit scope (train-only):
- Point-in-time feature rule:
- As-of alignment rule:
- Final holdout SHA-256:
- Holdout commitment date (UTC):
- Has any researcher/model previously seen this holdout? YES/NO, with evidence:
- If YES or UNKNOWN: HOLDOUT INVALID; do not claim untouched OOS evidence.

## 6. Models, weights and feature timing

For each model/arm:
- Model name and exact variant:
- Model revision / immutable revision identifier:
- Weights file or artifact path:
- Weights SHA-256:
- Runtime/library version:
- Context length:
- Warm-up rows removed:
- Earliest permissible feature timestamp:
- Feature availability / as-of timestamp:
- Pretraining corpus / provenance source:
- Known overlap of pretraining corpus with backtest window:
- Overlap evidence and uncertainty:
- Missing/unavailable model artifact behavior: FAIL-CLOSED.

## 7. Experimental controls

- Baseline definition:
- Treatment arms:
- Single intended difference between arms:
- Identical row set:
- Identical fold boundaries:
- Identical seed(s):
- Identical simulator version + SHA:
- Identical fill/stop/collision policy:
- Identical position/risk policy:
- Random seed(s):
- Number of variants planned (exact integer):
- Full list of planned variant IDs:
- Any nested tuning/random search iterations:
- Multiple-testing cohort / Trial Registry IDs:

## 8. Cost scenarios (both mandatory)

| Scenario | Round-trip cost | Per-side fee/slippage decomposition | Required |
|---|---:|---|---|
| Cost-50 | 50 bps | State maker/taker + slippage assumptions explicitly | YES |
| Cost-90 | 90 bps | State maker/taker + slippage assumptions explicitly | YES |

No silent fee substitution. Do not assume Nobitex provides a post-only guarantee.
Report both scenarios independently; never average them into one headline metric.

## 9. Acceptance gates (absolute, not relative to a losing baseline)

Every required fold and the aggregate report must show:
- Net Profit Factor >= 1.10 after costs.
- Completed trades >= 100.
- Maximum drawdown <= 15%.
- Deflated Sharpe Ratio Z > 0, using the full preregistered trial count.
- Stability in at least 3 chronological folds.
- No acceptance merely for improving a losing baseline.
- Insufficient data, a missing fold, or an undefined metric => NO-GO / INSUFFICIENT_EVIDENCE.

Record any additional owner-approved gates here without weakening the gates above:

## 10. Statistical plan

- Primary metric:
- Secondary metrics:
- DSR method and assumptions:
- PBO method and assumptions:
- Exact trial count N used:
- Source of every included trial:
- Dependence/correlation among trials:
- Anytime-valid procedure, if sequential decisions are made:
- Stopping rule:
- Multiple comparisons correction:
- Handling of failed/cancelled/partial trials:

## 11. Reproducibility and safety

- Environment lockfile SHA:
- Freqtrade version (pinned 2026.7):
- Python version (3.12):
- CI workflow run ID:
- Output artifact IDs + SHA-256:
- Champion=null:
- order_permission=false:
- trading_authorization=NO-GO:
- ALIX_DRY_RUN=true:
- ALIX_ALLOW_LIVE_TRADING=false:
- private_api_calls=0:
- real_order_attempts=0:
- No secrets in source, logs or artifacts: CONFIRMED

## 12. Review and freeze

- Preregistration reviewed by:
- Review date (UTC):
- Approval / rejection:
- Preregistration file SHA-256:
- First execution run ID:
- Deviations (must be NONE, otherwise new preregistration):
