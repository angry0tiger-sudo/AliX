# DEAP Eligibility-Gate Implementation Record

**Date:** 2026-10-25  
**Base revision:** `cfe4ef59f9effbf65c298da2628cdcaa36a26156`  
**Working branch:** `feat/deap-sample-gates-timeframe-coverage`  
**Scope:** offline GA candidate validity, mandatory timeframe coverage, and data-history verification

## Baseline findings

- The prior GA registered DEAP fitness with `weights=(1.0,)`, so positive numeric values were preferred. Its low-sample fitness was `-1.0`, which differed from the owner's new contract and could distort selection when ordinary strategy fitness was negative.
- Evolution scored training metrics only, then evaluated validation metrics for a single numeric Hall-of-Fame winner. Validation sample-count eligibility was not applied to candidates during the evolutionary search.
- Timeframes `60`, `240`, and `1440` minutes existed in the search list but had no per-generation population quota or report proving they were sampled.
- The pinned prior data artifact was about six months of BTCUSDT 1-minute candles. The data downloader already supported a maximum of 24 months through Binance's public market-data endpoint.

## Implemented acceptance contract

- Minimum completed trades are 30 on training and 10 on validation. Any candidate failing either threshold receives fitness exactly `0.0`; an explicit eligibility-aware tournament ranks sample-valid candidates ahead of invalid candidates regardless of raw numeric fitness.
- Only training expectancy contributes to the numeric objective. Validation trade count is used only as a minimum-sample gate; validation return/drawdown are evaluated after a candidate is preselected from training fitness, avoiding validation-profit optimization. The final holdout remains untouched.
- Training drawdown above 15% receives a severe non-positive penalty and disqualifies a candidate from final candidate selection. Validation drawdown above 15% or non-positive validation expectancy yields NO-GO.
- At least four unique genomes for each of 1h, 4h and 1d are injected into the population every generation. Reports include per-generation population counts and per-timeframe evaluated/qualified/rejected genome counts.
- The workflow is configured to acquire 24 months of BTCUSDT public Spot minute data, fail on coverage gaps, run at least ten generations and upload the immutable input data, manifest, JSON report and CSV diagnostics together.

## Safety

`Champion=null`, `order_permission=false`, and `trading_authorization=NO-GO` remain immutable. The workflow uses only public market-data endpoints; no private API, order/execution API, or real-order attempt is permitted.

## Validation status

This record describes the code and CI evidence contract. Actual test, data-fetch, GA, and workflow results must be appended from the completed workflow run; this document alone is not acceptance evidence.


## Additional integrity correction

- Temporal splits are now based on each resampled candle's close/availability timestamp, not its open label. A partially elapsed higher-timeframe candle can no longer be assigned to training merely because the bucket started before the cutoff. A regression test uses a cutoff inside an hourly candle.
- Required timeframe quotas count distinct genome vectors, not population slots. The injection step preserves at least four unique, newly evaluated 1h, 4h and 1d genomes per generation; duplicates cannot masquerade as coverage.


## Owner Directive update: Opportunity Hunter

The latest owner directive supersedes the earlier hard 30-training / 10-validation trade thresholds. It introduces soft sample confidence weighting rather than hard count rejection, profit-factor plus net-total-return fitness, retrospective missed-upside penalties for non-overlapping 24-hour rallies of at least 5%, regime-capture bonus terms, and evolved profit-activated trailing stops with dynamic regime exits. Mandatory timeframes now include 15m, 1h, 4h and 1d. Training-only fitness drives selection; validation does not influence tournament ranking, and final holdout remains untouched. The 15% drawdown constraint and all execution safety invariants remain in force.
