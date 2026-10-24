# AI Core Rebuild - Research Candidate

**Branch:** research/ai-core-rebuild  
**Status:** isolated implementation candidate; economic acceptance and trading authorization remain NO-GO.

## Why this is a core rebuild

The current baseline is a compact OHLCV feature pipeline with one LightGBM regressor. AliX already has independent building blocks for a strict Data Quality Gate, a versioned Gaussian HMM classifier, strategy registry/selector contracts, Triple-Barrier logic, point-in-time external data, and local NLP. This branch composes a research-only path around those contracts rather than pretending the components were absent.

## Current implementation

- **Feature schema:** the implemented schema defines 58 BTC OHLCV features and 64 when six approved, exactly aligned ETH 4H features are added. The pinned Stage A artifact currently has no positive source-volume observations, so the experiment excludes the 14 volume-derived columns and trains on 44 causal price/volatility/trend features rather than fabricating volume or quietly substituting a different venue. When positive source volume is available, the runner selects 58 BTC features; with approved aligned ETH data it selects 50 price-only + ETH cross-asset features or the full 64 when source volume is also observed. The implemented groups include multi-window SMA/EMA, trend slopes/crossovers, Bollinger width/position, RSI(6/14/28), true range/ATR, realized/downside volatility, volume changes/z-scores/turnover, VWAP distance, OHLCV-based volume-profile proxies, breakout distance and candle shape.
- **Cross-asset:** six ETH-derived return/correlation/beta/relative-strength features are implemented. They are excluded from training unless an owner-supplied, hash-pinned hourly ETH dataset covers the exact requested timestamps. GitHub Actions performs no live exchange/provider downloads. The OHLCV volume profile is a coarse histogram proxy, not trade-level volume-at-price.
- **Ensemble:** CPU-only XGBoost, Random Forest and LightGBM regressors are fit on each expanding training fold. Default aggregation is a deterministic equal-weighted mean, with model-by-model diagnostics retained. It is a testable hypothesis, not a guarantee that ensemble errors fall.
- **HMM:** a three-state diagonal-Gaussian HMM is fit within each training fold. The inference pass is causal and carries posterior state through intervening purge bars. A technical research label routes to a research-only candidate; it never changes the existing runtime model-approval contract.
- **Strategy candidates:** the isolated registry has exactly three RESEARCH_ONLY, owner_approved=false, long-only 4H entries: trend following, mean reversion, and momentum breakout. A missing/unmapped regime or invalid candidate returns NO_TRADE. The production registry is untouched.
- **NLP:** a defused XML RSS/Atom parser and point-in-time sentiment aggregator are implemented. Sentiment is **not** fetched or included in training until the RSS provider's licence/retention scope and a hash-pinned, reviewed, calibrated model are approved. Missing sentiment is not imputed to neutral.

## Research protocol

1. Load the immutable Stage A BTC development artifact and validate provenance through the existing loader.
2. Aggregate only complete, contiguous hourly bars into exact 4H candles. No resampling over gaps, interpolation, or backfill is allowed.
3. Generate features using current/past data, then create a 24-hour forward-return target solely for training/evaluation.
4. Run three expanding walk-forward folds with six 4H bars purged between train and test. The final 90-day holdout is reserved and not read by this runner.
5. Fit all three regressors per fold; report component and ensemble error metrics.
6. Fit the HMM only on the training window, forward-filter causal observations, route signals using the frozen research registry, and record non-overlapping research-only hypothetical returns after the declared 60 bps round-trip cost assumption.
7. Emit a JSON report and OOS prediction CSV tied to the exact source and code revisions.

## Known blockers and interpretation

- The pinned Stage A artifact has no positive source-volume observations. Default CI therefore trains 44 features based on observed price/volatility/trend data; all 14 volume-dependent columns are excluded and reported as blocked. The full 58-feature OHLCV schema remains implemented, but it is not falsely reported as trained.
- No approved ETH source dataset is currently part of the pinned Stage A artifact. The default CI experiment therefore reports cross-asset features as blocked. Supplying a cross-asset file is a separate governed input, not a network fetch in GitHub Actions.
- No approved Persian RSS source or calibrated sentiment model has been selected. NLP code is parser/aggregation plumbing only.
- Model/dependency licences, source contracts, rolling regime stability, calibrated probability outputs, multiple-comparison risk, economic thresholds, order-book/liquidity evidence and reviewed release approvals remain outstanding.
- HMM labels and ensemble returns are diagnostics. The strategy signals are not orders, actual fills, or evidence of economic value.
- A single symbol and a limited number of walk-forward folds do not establish generalization. Any strategy selection on the same OOS reports would create selection bias; the reserved holdout remains locked until the protocol is frozen.

## Acceptance and safety invariants

- [ ] All existing repo quality gates pass on the exact branch commit.
- [ ] The new unit tests pass and remain within the existing coverage requirements.
- [ ] Workflow report and prediction CSV are bound to the exact commit SHA and source artifact hashes.
- [ ] The optional cross-asset path fails closed for missing, duplicated, stale or non-overlapping ETH candles.
- [ ] No network/provider download occurs from this workflow.
- [ ] Champion = null, order_permission = false, production model/registry unchanged.
- [ ] Economic acceptance = **NO-GO** and trading authorization = **NO-GO**.

## Technical roadmap

**R1 - Candidate composition (this branch).** Feature schema, model ensemble, HMM forward filtering, strategy candidates, RSS parser and immutable research report.

**R2 - Dataset contracts.** Approve the cross-asset source and ingestion terms; materialize/hash a contiguous ETH history; define handling for unavailable source windows and prove point-in-time alignment.

**R3 - Statistical validation.** Lock folds/purges and acceptance criteria before viewing results; add naive/history baselines, fees/slippage sensitivity, bootstrap confidence intervals, regime-specific sample-size floors, calibration diagnostics, stability across periods and a multiple-comparison correction.

**R4 - NLP acceptance.** Approve provider rights and retention, verify Persian language handling and asset/event classification, curate a timestamped labelled dataset, evaluate out-of-time calibration, and compare ablations with/without sentiment. Until then sentiment remains excluded.

**R5 - Integration candidate.** Wire a hash-pinned HMM inference artifact through the existing HMM approval contract and pure strategy selector. Add full lineage, NO_TRADE handling, shadow-only decision ledger and independent risk-supervisor tests; no production registry population.

**R6 - Owner review and gate decision.** Review exact code/model/data hashes, dependency and data licences, OOS evidence, costs, robustness and rollback path. Only a separate owner-reviewed governance decision can change acceptance; this research branch cannot promote a Champion or grant order permission.
