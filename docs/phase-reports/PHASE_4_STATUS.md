# Phase 4 Status Report — FreqAI Baseline

Status: **OFFLINE BASELINE IMPLEMENTATION IN PROGRESS / ECONOMIC GATES NOT PASSED / CHAMPION BLOCKED / NO TRADING**.

## Baseline and scope

- Repository base: Phase 3 merge a7d301874d8503f75eaed7642dc9eaa6fcbc87ac.
- Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).
- Implementation branch: feat/phase-4-freqai-baseline.
- Owner-provided source docs are being committed byte-for-byte under docs/architecture/ and docs/roadmap/.
- Pinned runtime target: freqtrade==2026.7; the custom model follows that tag's BaseRegressionModel fit contract and uses pinned LightGBM.
- Scope: versioned past-only OHLCV feature schema, isolated forward-return target, deterministic LightGBM regressor, horizon-purged expanding walk-forward evaluator, fold/regime diagnostics, reload reproducibility and versioned JSON report.
- The new research strategy validates data before freqai.start; its entry/exit signals remain zero. Existing AliXStrategy remains untouched by FreqAI signal logic.

## Evaluation design

- Default horizon: 12 candles.
- Default expanding walk-forward: 3 folds, minimum 300 training rows before purge.
- Default modeled costs: 10 bps fee per side and 5 bps slippage per side. These are research defaults only and must be replaced/approved for the eventual venue/account before economic acceptance.
- Fixed-horizon research rule is evaluated offline against the forecast threshold implied by round-trip modeled cost; no execution API is used.
- Each report records commit, pair/timeframe, input SHA-256, Data Quality Gate fingerprint, schema versions, parameters, fold timestamps, OOS metrics, costs, reload hash and promotion status.
- A seeded synthetic run is used only to verify pipeline mechanics; synthetic data is explicitly marked and cannot be treated as market performance evidence.

## Open gates

1. No approved, authoritative historical OHLCV dataset/source/license and hash has yet been supplied/registered for economic acceptance.
2. Owner-approved Phase 0 economic acceptance thresholds must be recorded before evaluating a real market candidate. No threshold is invented in this report.
3. Freqtrade full backtest with a reviewed historical dataset, broader sensitivity tests and independent quant/risk review remain outstanding.
4. Prediction latency, training duration and resource profiling still need to be captured as part of final phase evidence.
5. Any external provider/license, live route, runtime account or order reconciliation tests are outside this phase's GitHub CI and require separate authorized procedures.

## Decision

- Implementation GO is determined only by exact-head GitHub CI and focused tests.
- Economic acceptance is **NO-GO** until pre-registered thresholds and authorized historical OOS evidence pass.
- Champion must be null; promotion status remains blocked.
- No trading, order write, API connection, data provider access, notification send or host network action is authorized/performed by this phase.
- Owner review and PR merge remain pending; main is not changed by the Phase 4 branch.
