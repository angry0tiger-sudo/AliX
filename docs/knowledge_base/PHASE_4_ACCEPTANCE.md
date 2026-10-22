# Phase 4 Acceptance: FreqAI Baseline

Status: **EXACT-HEAD IMPLEMENTATION CI PASS (run #326, SHA `eccd694d571da22853ea81065fa224afbfc239b6`); ECONOMIC ACCEPTANCE NO-GO; CHAMPION BLOCKED; NO TRADING**.

Canonical roadmap: [Phase 4 — FreqAI baseline and quantitative evaluation](AliX_Roadmap_Dynamic_Risk_FA.md). Earlier compatibility copy: [docs/roadmap path](../roadmap/AliX_Roadmap_Dynamic_Risk_FA.md).
Owner directive: [Phase 4 baseline directive](../governance/owner-directives/2026-10-22-phase-4.md); [canonical-path directive](../governance/owner-directives/2026-10-22-phase-4-knowledge-base-canonical-paths.md).
Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).

## Observed synthetic artifact (SYNTHETIC / MECHANICS ONLY)

Latest exact-head evidence: [GitHub Actions run #326 / ID 39724732753](https://github.com/angry0tiger-sudo/AliX/actions/runs/39724732753), branch HEAD `eccd694d571da22853ea81065fa224afbfc239b6`. Both `quality` and `freqtrade-runtime-smoke` passed. Artifact: [`phase4-synthetic-walkforward-report`, ID 12402049075](https://github.com/angry0tiger-sudo/AliX/actions/runs/39724732753/artifacts/12402049075); zip size 3,180 bytes; SHA-256 `6c4c1dff99eae3acfb0c76617a2e82a103da3c7bd6cdbae999077512e27d529b`. JSON file: `alix-phase4-synthetic.json`.

- Report provenance/status: `commit_sha=eccd694d571da22853ea81065fa224afbfc239b6`, asserted against PR head by CI; schema `alix.freqai.walk-forward-report.v1`; evaluation `SYNTHETIC_MECHANICS_ONLY`.
- Dataset: generated `SYNTHETIC/USDT`, 1h timeframe, 1,800 input rows from `2020-01-01T00:00:00+00:00` through `2020-03-15T23:00:00+00:00`, 1,775 valid feature/target rows. Dataset SHA-256 `b3d1195315f4bbc7d881ddf57790e40bcc4d230e19359d45f5e3975a572a4131`; Data Quality fingerprint `c2414f353e9e4ad6e5d33c28c274f38fd48f2a7f163d8d17980ecdfbdb189034`.
- Contracts/configuration: feature `alix.freqai.features.v1`, nine point-in-time current/past-only OHLCV features; target `alix.freqai.forward-return.v1`, column `&-s_forward_return`, horizon 12 candles and kept separate from features. Experiment config SHA-256 `039f87f911e430a799e571ce74f20b9d1be23606a9494bfe22dab45827116333`.
- Runtime versions: Python 3.12.15, Freqtrade 2026.7, LightGBM 4.7.0, NumPy 2.5.4, pandas 3.0.6 and scikit-learn 1.9.0. The model is `AliXBaselineRegressor` / LightGBM `LGBMRegressor`; recorded deterministic parameters include 80 estimators, learning rate 0.03, max depth 4, 15 leaves, random state 42, one job, `deterministic=true` and `force_col_wise=true`. Model artifact SHA-256: `67278a6514b8c5e169bbb85c42bf460098d8587b4b7bd378f199f766c617fa4a`.
- Evaluation: three chronological expanding folds, each purging 12 labels, minimum initial training size 300, no shuffle; aggregate OOS count 1,463. Fold test windows: 2020-01-14 13:00 through 2020-02-03 19:00 UTC; 2020-02-03 20:00 through 2020-02-24 03:00 UTC; 2020-02-24 04:00 through 2020-03-15 11:00 UTC. Their respective train row counts were 300, 787 and 1,275; test row counts 487, 488 and 488.
- OOS forecast RMSE 0.0226048 and MAE 0.0163096; zero-forecast RMSE 0.0213377 and MAE 0.0152165. Thus the zero-forecast reference performed better on both forecast-error metrics for this synthetic sample.
- Costed fixed-horizon research rule: 64 synthetic trades, turnover 128 units, cumulative fees 1,280 bps, cumulative slippage 640 bps, net return -0.0739532220, maximum drawdown -0.1223014807; no-trade net return 0.0 and buy-and-hold return -0.0099376443. Only 1 of 3 folds had positive net return. These synthetic values are mechanics diagnostics, not returns from an actual market or evidence of profitability.
- Native LightGBM model-string reload passed; maximum absolute prediction difference 0.0. Synthetic-run training time totaled 0.05852 seconds; prediction latency p50 0.003935 ms/row and p95 0.004199 ms/row. These measurements describe this synthetic GitHub runner only, not expected production capacity.
- Historical note: the earlier [run #310 / ID 39722526159](https://github.com/angry0tiger-sudo/AliX/actions/runs/39722526159) artifact (ID `12401817212`) reported the temporary PR merge ref `74a0391fd0e3d4302e530b2379a1678ef5651bf5` rather than its branch HEAD `54e6091ad3cf98c92b4108131233defe0cf7967a`. CI was corrected. Run #326 now asserts the report SHA equals the real PR head and passes that assertion; the latest artifact is the authoritative current mechanics report.
- Economic gates: `NOT_EVALUATED`; `champion = null`; `promotion_status = BLOCKED_NO_ECONOMIC_GATE`. This synthetic artifact does not qualify a Champion or pass economic acceptance.

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
