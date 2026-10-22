# Phase 4 Acceptance: FreqAI Baseline

Status: **IMPLEMENTATION IN PROGRESS; ECONOMIC ACCEPTANCE NO-GO; CHAMPION BLOCKED**.

Roadmap: [Phase 4 — FreqAI baseline and quantitative evaluation](../roadmap/AliX_Roadmap_Dynamic_Risk_FA.md).
Owner directive: [Phase 4 Owner Directive](../governance/owner-directives/2026-10-22-phase-4.md).
Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).

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
