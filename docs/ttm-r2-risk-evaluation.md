# TTM-R2 Risk-Only Evaluation Decision Record

**Status:** Registered for one offline research run  
**Base SHA:** \`9a03fa2f4fca31da0da7bc21b31b5ec7e16abe92\`  
**Working branch:** \`research/ml-core-ttm-r2-risk-eval\`  
**Authority:** Owner directive in the 11 November 2026 handover  
**Trading status:** \`NO-GO\`

## 1. Prior phases closed

- AI-6 RL is closed with \`NO STATISTICAL EDGE FOUND IN RL\`.
- DEAP is closed with \`NO STATISTICAL EDGE FOUND IN DEAP\`. Standard CI #2231 and DEAP Workflow #7 passed technically, but the winner produced four OOS trades (3.694/year) against the required frequency above 200/year. The same OOS must not be rerun or tuned.
- Chronos-2 has already been implemented and evaluated through GitHub Actions on the approved pre-hack Nobitex dataset. The existing run used 50 bps round-trip cost and three purged walk-forward folds. Baseline PF was 0.4077, Candidate PF 0.4139, maximum drawdown approximately 99.8% for both, and Candidate trades fell from 1,285 to 1,219. This is an economic failure, not a reason to rerun or tune the same period.
- Kronos-Small is therefore removed from the current roadmap. Its conditional prerequisite (Chronos-2 economic acceptance) was not met.

## 2. Narrow TTM-R2 scope

TTM-R2 is evaluated only as a zero-shot, frozen-weight, 24-hour realized-variance forecaster for offline risk research.

It MUST NOT:
- create, rank, or filter ENTRY signals;
- modify the strategy genome, labels, or model ensemble used for entry;
- write risk policy, Champion state, order permission, secrets, or Kill Switch;
- generate orders, paper orders, or exchange calls;
- increase a position multiplier above 1.0;
- receive daily/weekly entry authority or change the 15m/30m/1h/4h entry allowlist.

No model fine-tuning on Nobitex is performed. Standard-scaler parameters are fitted only on the first 70% chronological training prefix and are frozen for validation and holdout. The 15% validation split is diagnostic. The final 15% is evaluated once and is not used for model choice or calibration.

## 3. Pretraining disclosure and explicit scope exception

IBM's official Granite TTM-R2 model card lists the Monash Bitcoin dataset among the pretraining sources. Zenodo records that dataset as published on 22 July 2021 and describes 18 daily Bitcoin-related influence series, including hash rate, block size, mining difficulty, tweets, and Google search interest. The approved Nobitex series begins on 31 October 2021.

This temporal ordering lowers the specific concern of overlap with the approved Nobitex observation window, but it does not prove that IBM's entire pretraining corpus has no other overlap with the target market. The experiment therefore:
- uses only the published TTM-R2 model weights;
- downloads no Monash rows and joins no third-party market data;
- uses only the immutable pre-hack Nobitex OHLCV file for target-domain inference and evaluation;
- records the resolved immutable Hugging Face commit and weight SHA-256;
- reports the remaining pretraining-corpus uncertainty explicitly.

This is the narrow model-weights exception implied by the Owner directive to evaluate TTM-R2. It does not authorize third-party runtime features, backfills, or fine-tuning on external datasets.

## 4. Forecast target and point-in-time contract

Input: approved Nobitex hourly Spot/USDT close series.

At decision time \`t\`:
- compute \`r_t = log(close_t / close_(t-1))\`;
- model target is \`log(1 + r_t^2)\`;
- TTM context is the last 512 closed hourly observations through \`t\`;
- model output is restricted to the next 24 hourly steps;
- predicted 24-hour variance is the sum of the 24 inverse-scaled, non-negative predicted hourly variances;
- realized 24-hour variance is the sum of the next 24 squared log returns;
- persistence baseline is the sum of the preceding 24 squared log returns.

All forecast origins are non-overlapping at 24-hour intervals. Each target window must remain wholly inside its chronological split. Missing/duplicate/irregular timestamps, non-finite values, invalid prices, incomplete contexts, or incomplete horizons fail closed. No interpolation, backfill, zero-padding, or future-derived scaler is permitted.

The model uses IBM's hourly TTM-R2 512-96-r2 release (hosted on the repository's `main` branch) at immutable Hugging Face commit `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`. The `model.safetensors` file is pinned to SHA-256 `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710`; the runner verifies this digest before loading. The model is loaded with a 24-step prediction filter.

## 5. Pre-registered acceptance criteria

The single final-holdout risk forecast is accepted for further *risk-only research* only when all conditions hold:

1. At least 30 disjoint holdout forecasts are available.
2. QLIKE is at least 5% lower than the trailing-24h persistence baseline.
3. Log-variance MAE is no worse than the baseline.
4. All predictions are finite and strictly positive after the declared inverse transform.
5. The diagnostic position multiplier \`min(1, sqrt(persistence_variance / TTM_forecast_variance))\` never exceeds 1.0.

A PASS is not a trading or deployment approval. A NO-GO closes the TTM-R2 hypothesis. No threshold changes are allowed after observing the holdout. The experiment is a forecast test, not a trading backtest; therefore no trading cost is charged to QLIKE. Any later strategy-level economic evaluation must preserve the Owner's 50 bps round-trip cost and the existing trade-frequency and risk gates.

## 6. Execution and safety gate

- Code, unit tests, dependency installation, inference, and report generation run only in GitHub Actions.
- The TTM workflow is triggered only after Standard CI for the exact pushed SHA succeeds.
- The workflow has read-only repository permission and uses no GitHub secrets.
- Data access is limited to the unauthenticated Nobitex public history endpoint used by the existing immutable dataset fetcher.
- No private API calls, exchange order calls, paper orders, or real orders are made.
- \`main\` remains untouched.

Mandatory report invariants:

\`\`\`text
Champion = null
order_permission = false
trading_authorization = NO-GO
ALIX_DRY_RUN = true
ALIX_ALLOW_LIVE_TRADING = false
private_api_calls = 0
real_order_attempts = 0
paper_orders = 0
\`\`\`

## 7. Decision after the run

- \`PASS_RISK_FORECAST_RESEARCH_ONLY\`: may justify a separately reviewed risk-allocation design proposal; no runtime wiring or promotion.
- \`NO_GO\`: close TTM-R2 for this Nobitex risk target. Do not retune the final holdout.
- Technical workflow failure: no economic conclusion; fix only the implementation defect on the research branch, preserve the frozen protocol, and rerun only if the failed run did not produce a valid holdout evaluation.

Architecture First. Evidence Always. Fail Closed.
