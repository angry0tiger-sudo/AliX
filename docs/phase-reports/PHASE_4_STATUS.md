# Phase 4 Status Report — FreqAI Baseline

Status: **IMPLEMENTATION CANDIDATE; PRE-FOLLOW-UP CI PASSED; NEW CANONICAL-DOCS HEAD REQUIRES EXACT-HEAD CI; ECONOMIC ACCEPTANCE NO-GO; CHAMPION BLOCKED; NO TRADING**.

## Baseline and scope

- Repository base / current `main` HEAD observed: `a7d301874d8503f75eaed7642dc9eaa6fcbc87ac`. The PR base and `main` compared identical at this SHA before the follow-up changes.
- Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).
- Pull request: [#4](https://github.com/angry0tiger-sudo/AliX/pull/4), base `main`, branch `feat/phase-4-freqai-baseline`; open, Draft, not merged.
- Baseline branch HEAD before this owner-directed documentation follow-up: `54e6091ad3cf98c92b4108131233defe0cf7967a`.
- Phase 4 source documents are retained at `docs/architecture/AliX_MRD_Dynamic_Risk_FA.md` and `docs/roadmap/AliX_Roadmap_Dynamic_Risk_FA.md`. The owner-requested stable canonical copies are now being added at `docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md` and `docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md`, with the prior copies retained for compatibility.
- Version-pinned runtime target: `freqtrade==2026.7`; custom model uses the pinned LightGBM regressor. Dependency pins are not changed by this documentation/path task.
- Scope: versioned past-only OHLCV feature schema, isolated forward-return target, deterministic LightGBM regressor, horizon-purged expanding walk-forward evaluator, fold/regime diagnostics, native model reload, and versioned JSON output.
- The research strategy validates data before `freqai.start`; its entry/exit signals remain zero. Existing `AliXStrategy` remains signal-free.

## CI evidence inspected before the knowledge-base path update

- [GitHub Actions run #310 / run ID 39722526159](https://github.com/angry0tiger-sudo/AliX/actions/runs/39722526159), on the PR branch HEAD `54e6091ad3cf98c92b4108131233defe0cf7967a`: overall `success`.
- `quality`: **success**. The exact job log shows Ruff lint/format, strict Mypy, Bandit, dependency resolution and lock validation, `pip-audit`, `lint-imports`, Xenon, unit tests, Phase 1 safety contracts, and documentation/Starter Kit validation completed successfully. Unit tests: **102 passed**, coverage **92.71%** (minimum 90%); Phase 1 safety contracts: **4 passed**.
- `freqtrade-runtime-smoke`: **success**. Log confirms Freqtrade `2026.7`, successful loading of the pinned model/strategy and closed configuration, successful synthetic walk-forward, and successful artifact upload. No exchange connection was requested.
- The above run validates the previous branch HEAD only. The current owner-directed docs/validator/workflow changes must each be checked on the final exact HEAD before they are offered for owner review.

## Synthetic artifact inspection (SYNTHETIC / MECHANICS ONLY)

- Artifact: [phase4-synthetic-walkforward-report from run 39722526159](https://github.com/angry0tiger-sudo/AliX/actions/runs/39722526159/artifacts/12401817212); artifact ID `12401817212`; zip SHA-256 digest `eda6e9e0a8c6c2c2236e53913da8e4b8538e619183eeea597ac1851c93dc7bfd`.
- JSON file: `alix-phase4-synthetic.json`; evaluation status `SYNTHETIC_MECHANICS_ONLY`; schema `alix.freqai.walk-forward-report.v1`.
- Dataset: synthetic `SYNTHETIC/USDT`, 1h candles, 1,800 generated rows from `2020-01-01T00:00:00+00:00` to `2020-03-15T23:00:00+00:00`; 1,775 valid feature/target rows; SHA-256 `9c5e7276d1cebfaa5c280f0c2e8aeb8f62ad1d6bc7e57bb9f5403d8418fb72e2`.
- Provenance recorded: config hash `039f87f911e430a799e571ce74f20b9d1be23606a9494bfe22dab45827116333`; feature version `alix.freqai.features.v1`; target version `alix.freqai.forward-return.v1`; model artifact hash `90a3b815a7b94dd17f402b68884311c3d14792bd91ba71130029978afe2d04aa`; runtime versions include Python 3.12.15, Freqtrade 2026.7, LightGBM 4.7.0, NumPy 2.5.4, pandas 3.0.6 and scikit-learn 1.9.0.
- Three expanding chronological folds, 12-candle horizon and purge, minimum 300 training rows; aggregated OOS sample count 1,463. Forecast RMSE 0.0226048 vs zero-forecast 0.0213377; MAE 0.0163096 vs 0.0152165. The simple zero-forecast baseline had lower RMSE and MAE on this synthetic dataset.
- Synthetic fixed-horizon rule: 64 trades, turnover 128, cumulative fees 1,280 bps, slippage 640 bps, net return -0.0739532, maximum drawdown -0.1223015; just 1/3 folds had positive synthetic net return. This output is strictly mechanics evidence and not a claim about real-market behavior.
- Reload result: native LightGBM model string `passed`, maximum absolute prediction difference 0.0. Measured synthetic-run training duration total 0.04495 s; prediction latency p50 0.002737 ms/row, p95 0.002767 ms/row. These measurements are specific to the synthetic runner and are not capacity guarantees.
- The report recorded `commit_sha=74a0391fd0e3d4302e530b2379a1678ef5651bf5`, the temporary PR merge ref rather than branch HEAD `54e6091ad3cf98c92b4108131233defe0cf7967a`. The Phase 4 workflow is being corrected to pass the pull-request head SHA and assert the JSON report records the same SHA. The old artifact is retained as historical evidence and is not relabelled.

## Open gates

1. No owner-authorized historical OHLCV dataset/source/license and content hash is registered for economic acceptance.
2. Owner-approved Phase 0 economic acceptance thresholds must be recorded before evaluating a real-market candidate. No threshold is invented here.
3. Full historical OOS evaluation across multiple windows/regimes, transaction-cost/slippage realism, sensitivity/stability analysis and independent quant/risk review remain outstanding.
4. Runtime latency/training/resource profiling requires an approved representative dataset and controlled environment; synthetic timings do not close this gate.
5. External provider/license checks, market metadata, account/order reconciliation and any operational/live routing tests are outside this Phase 4 offline CI and remain unauthorized here.

## Decision

- Current decision for implementation: **CONDITIONAL GO pending exact-head CI of the canonical-docs/validator update and review of the resulting artifact**.
- Economic acceptance: **NO-GO**.
- `Champion = null`; `promotion_status = BLOCKED_NO_ECONOMIC_GATE`.
- Trading authorization: **NO-GO**. Phase 4 does not authorize real, dry-run, or paper trading, order writes, exchange/API access, live market-data download, provider calls, notifications, or host/network changes.
- PR #4 remains Draft/open and must not be merged without the owner's explicit instruction.
- Phase 5 (Meta-Labeling / Triple-Barrier) has **not** started and remains blocked pending separate explicit owner approval.
