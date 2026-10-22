# Phase 4 Acceptance: FreqAI Baseline

Status: **IMPLEMENTATION IN PROGRESS; ECONOMIC ACCEPTANCE NO-GO; CHAMPION BLOCKED**.

Canonical roadmap: [Phase 4 — FreqAI baseline and quantitative evaluation](AliX_Roadmap_Dynamic_Risk_FA.md). Earlier compatibility copy: [docs/roadmap path](../roadmap/AliX_Roadmap_Dynamic_Risk_FA.md).
Owner directive: [Phase 4 baseline directive](../governance/owner-directives/2026-10-22-phase-4.md); [canonical-path directive](../governance/owner-directives/2026-10-22-phase-4-knowledge-base-canonical-paths.md).
Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).

## Observed synthetic artifact (SYNTHETIC / MECHANICS ONLY)

Reviewed artifact from [CI run #39722526159](https://github.com/angry0tiger-sudo/AliX/actions/runs/39722526159), artifact ID `12401817212`, SHA-256 digest `eda6e9e0a8c6c2c2236e53913da8e4b8538e619183eeea597ac1851c93dc7bfd`. It contains `alix-phase4-synthetic.json`.

- Report status: `SYNTHETIC_MECHANICS_ONLY`; report schema `alix.freqai.walk-forward-report.v1`.
- Dataset: synthetic `SYNTHETIC/USDT`, timeframe `1h`, 1,800 generated rows from `2020-01-01T00:00:00+00:00` through `2020-03-15T23:00:00+00:00`; 1,775 valid feature/target rows; dataset SHA-256 `9c5e7276d1cebfaa5c280f0c2e8aeb8f62ad1d6bc7e57bb9f5403d8418fb72e2`.
- Contracts: feature `alix.freqai.features.v1`; target `alix.freqai.forward-return.v1`, target `&-s_forward_return`, horizon 12 candles. Config hash: `039f87f911e430a799e571ce74f20b9d1be23606a9494bfe22dab45827116333`.
- Evaluation: 3 chronological expanding folds, minimum training size 300, 12-row purge; 1,463 aggregated OOS samples. OOS RMSE was 0.0226048 versus zero-forecast RMSE 0.0213377; MAE was 0.0163096 versus 0.0152165. The zero-forecast reference was better on these forecast-error metrics for this synthetic sample.
- Costed research rule: 64 synthetic trades, turnover 128 units, cumulative fees 1,280 bps and slippage 640 bps; synthetic net return -0.0739532 and maximum drawdown -0.1223015. Only 1 of 3 folds had positive synthetic net return. These are generated-data mechanics results, not market returns.
- Model: LightGBM `LGBMRegressor`, pinned parameters recorded in the JSON; native model-string reload passed with maximum absolute prediction difference 0.0. Synthetic-run training duration totaled 0.04495 seconds; prediction latency p50 0.002737 ms/row and p95 0.002767 ms/row. These tiny synthetic-run timings are not runtime capacity claims.
- The inspected run's report `commit_sha` was `74a0391fd0e3d4302e530b2379a1678ef5651bf5` (the temporary PR merge ref), while the branch head was `54e6091ad3cf98c92b4108131233defe0cf7967a`. CI is being corrected to pass the pull request head SHA and assert that the report records it; the new exact-head run must be inspected before final review.
- Economic gates were `NOT_EVALUATED`; `champion = null`; `promotion_status = BLOCKED_NO_ECONOMIC_GATE`. This artifact does not qualify a Champion or satisfy economic acceptance.

## Acceptance checklist

### Contracts and model lifecycle
- [ ] The feature schema is versioned; all features are computed only from available current/past candles.
- [ ] Target schema/horizon are versioned; future labels never enter the feature matrix.
- [ ] The custom model loads against the pinned Freqtrade 2026.7 BaseRegressionModel/IFreqaiModel contract.
- [ ] A fixed model artifact can reload and reproduce predictions within a declared tolerance.
- [ ] The baseline model has no ability to access secrets, order lifecycle, or risk-policy writes.
- [ ] The Phase 3 Data Quality Gate runs before feature/model inference; rejected windows fail closed.

### Quantitative evaluation
- [ ] Walk-forward folds are strictly chronological and purge the full forward-label horizon before each test fold.
- [ ] OOS predictions are reported against a zero-forecast reference and a no-trade economic baseline.
- [ ] Fees and slippage are explicit, configurable, and applied to reported trade-rule returns.
- [ ] Report includes net/gross return, maximum drawdown, turnover, fees, slippage, trade count, OOS sample count and fold-level stability.
- [ ] Regime slices use thresholds fitted on training data only; confidence/uncertainty is not treated as calibrated unless separately evaluated.
- [ ] Full results reproduce from the exact commit, config, input CSV bytes, feature schema, model parameters and dependency pins.
- [ ] Training/prediction timing and sensitivity runs are recorded before final phase acceptance.

### Economics and governance
- [ ] Owner-approved economic acceptance thresholds are pre-registered before results are evaluated. Do not manufacture thresholds to make a candidate pass.
- [ ] Historical data source/license/use rights, OHLCV schema, period, pair/timeframe, data hash and quality-gate report are approved.
- [ ] OOS evidence passes every accepted threshold in multiple windows/folds/regimes and after modeled transaction costs.
- [ ] Economic and model-risk review approves the evaluation artifacts.
- [ ] Champion status remains unset unless all preceding gates pass and a separate model-promotion record is approved.

## Safety boundary

- No trade entry/exit signal is emitted by the Phase 4 strategy scaffold.
- No orders, live/dry-run sessions, exchange/API connections, market data downloads, provider calls, notifications or host networking changes are part of CI.
- The default config remains stopped, dry-run, zero open trades, zero stake, blank exchange credentials and empty pair whitelist.
- Synthetic fixtures verify mechanics, not market behavior. Synthetic returns and metrics cannot satisfy economic gates.

## Decision rules

- **GO for implementation:** exact-head lint/type/security/dependency/architecture/complexity/tests and safe runtime smoke all pass.
- **NO-GO for economic acceptance/Champion:** missing or failed approved economic criteria, unapproved or synthetic-only market data, leakage, non-reproducible artifact, unstable OOS behavior, or unmodeled costs.
- **NO-GO for every kind of trading in Phase 4**, even if implementation CI is green. Trading permission is not in Phase 4 scope.
