# Phase 4 Status Report — FreqAI Baseline

Status: **IMPLEMENTATION CI PASS ON EXACT HEAD `eccd694d571da22853ea81065fa224afbfc239b6` (run #326); ECONOMIC ACCEPTANCE NO-GO; CHAMPION BLOCKED; NO TRADING; OWNER REVIEW REQUIRED**.

## Repository and pull request state

- Repository: `angry0tiger-sudo/AliX`.
- `main` HEAD observed before this follow-up: `a7d301874d8503f75eaed7642dc9eaa6fcbc87ac`; comparison confirmed the PR base and main were identical at that SHA. The follow-up commits are confined to `feat/phase-4-freqai-baseline`.
- Phase 4 implementation HEAD before the owner-directed document-path update: `54e6091ad3cf98c92b4108131233defe0cf7967a`.
- PR: [#4](https://github.com/angry0tiger-sudo/AliX/pull/4), base `main`, branch `feat/phase-4-freqai-baseline`. At last read it was open, Draft, mergeable, and not merged. Never merge without explicit owner instruction.
- Tracking issue: [#3](https://github.com/angry0tiger-sudo/AliX/issues/3).
- The owner-provided MRD/Roadmap source copies remain intact at `docs/architecture/AliX_MRD_Dynamic_Risk_FA.md` and `docs/roadmap/AliX_Roadmap_Dynamic_Risk_FA.md`. Exact-content copies have been added as the canonical stable references at `docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md` and `docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md`; each copied file's Git blob SHA equals its source file's SHA. The old paths remain compatibility copies.
- `docs/PROJECT_DOCUMENTS.md` marks the knowledge-base paths canonical; `scripts/validate_project_docs.py` requires both canonical copies and the path Owner Directive; `.github/workflows/ci.yml` runs this validator. The versioned follow-up directive is `docs/governance/owner-directives/2026-10-22-phase-4-knowledge-base-canonical-paths.md`.
- Runtime baseline remains `freqtrade==2026.7`; model/runtime dependency pins were not altered by the document-path update.

## Exact-head CI evidence

- Successful check on branch HEAD `eccd694d571da22853ea81065fa224afbfc239b6`: [GitHub Actions run #326 / ID 39724732753](https://github.com/angry0tiger-sudo/AliX/actions/runs/39724732753). Run result: **success**.
- `quality`: **success**. The job log confirms compile, Ruff lint, Ruff format, strict Mypy, Bandit, dependency resolution dry run, lock verification, `pip-audit -r requirements.lock`, `lint-imports`, Xenon complexity gate, unit/coverage, Phase 1 safety contracts, and Starter Kit/docs validation completed. Unit tests: **102 passed, 1 warning**; coverage **92.71%**, above the required 90%; Phase 1 safety contracts: **4 passed**. The one pytest warning concerns deprecated NumPy timedelta unit handling in a test; it did not fail CI.
- `freqtrade-runtime-smoke`: **success**. Log confirms `freqtrade 2026.7`, model/strategy load and closed configuration smoke, no exchange connection requested, synthetic evaluator success, report-SHA assertion passed, and artifact upload succeeded.
- This report update itself creates a new branch commit and therefore triggers a new PR workflow run. Treat the latest run shown in [PR #4 checks](https://github.com/angry0tiger-sudo/AliX/pull/4) as authoritative for the resulting HEAD; do not treat run #326 as evidence for any later SHA unless it matches that SHA.

## Synthetic artifact inspection (SYNTHETIC / MECHANICS ONLY)

- Latest artifact for run #326: [`phase4-synthetic-walkforward-report`, ID 12402049075](https://github.com/angry0tiger-sudo/AliX/actions/runs/39724732753/artifacts/12402049075), ZIP size 3,180 bytes, SHA-256 `6c4c1dff99eae3acfb0c76617a2e82a103da3c7bd6cdbae999077512e27d529b`; JSON filename `alix-phase4-synthetic.json`.
- Provenance: `commit_sha=eccd694d571da22853ea81065fa224afbfc239b6`; the workflow now passes the PR head SHA and asserts exact equality. Evaluation status: `SYNTHETIC_MECHANICS_ONLY`; schema: `alix.freqai.walk-forward-report.v1`.
- Dataset: synthetic `SYNTHETIC/USDT`, 1-hour candles; 1,800 generated rows from `2020-01-01T00:00:00+00:00` through `2020-03-15T23:00:00+00:00`; 1,775 valid feature/target rows. Dataset SHA-256 `b3d1195315f4bbc7d881ddf57790e40bcc4d230e19359d45f5e3975a572a4131`; Data Quality fingerprint `c2414f353e9e4ad6e5d33c28c274f38fd48f2a7f163d8d17980ecdfbdb189034`.
- Experiment config hash: `039f87f911e430a799e571ce74f20b9d1be23606a9494bfe22dab45827116333`. Feature schema `alix.freqai.features.v1`; target schema `alix.freqai.forward-return.v1`, target column `&-s_forward_return), forecast horizon 12 candles. Feature columns are nine current/previous-candle-derived OHLCV features; future target data is kept separate.
- Runtime versions recorded: Python 3.12.15, Freqtrade 2026.7, LightGBM 4.7.0, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.0. Model `AliXBaselineRegressor` / `LGBMRegressor`; deterministic LightGBM parameters include 80 estimators, learning rate 0.03, max depth 4, 15 leaves, random state 42, one job, deterministic mode and column-wise training. Model artifact SHA-256 `67278a6514b8c5e169bbb85c42bf460098d8587b4b7bd378f199f766c617fa4a`.
- Evaluation: 3 expanding chronological folds, no shuffle, horizon 12 and 12-label purge, 300-row initial minimum training size; aggregate OOS sample count 1,463. Train rows by fold: 300, 787, 1,275; test rows: 487, 488, 488. Test windows: 2020-01-14 13:00–2020-02-03 19:00 UTC; 2020-02-03 20:00–2020-02-24 03:00 UTC; 2020-02-24 04:00–2020-03-15 11:00 UTC.
- Forecast RMSE 0.0226048 and MAE 0.0163096. Zero-forecast baseline RMSE 0.0213377 and MAE 0.0152165, so it scored better on both errors in this synthetic run.
- Costed offline rule: 64 synthetic trades; turnover 128; cumulative fees 1,280 bps; slippage 640 bps; net return -0.0739532220; max drawdown -0.1223014807; no-trade net return 0.0; buy-and-hold return -0.0099376443. Positive-return fold count: 1 of 3. Every one of these figures reflects generated data, not a real market.
- Native LightGBM model-string reload: `passed`; maximum absolute prediction difference 0.0. Training time totaled 0.05852 seconds; prediction latency p50 0.003935 ms/row and p95 0.004199 ms/row. Synthetic runner timings are not a production latency or capacity guarantee.
- The preceding [run #310 / ID 39722526159](https://github.com/angry0tiger-sudo/AliX/actions/runs/39722526159) had an artifact whose report SHA was temporary PR merge ref `74a0391fd0e3d4302e530b2379a1678ef5651bf5`, not branch HEAD `54e6091ad3cf98c92b4108131233defe0cf7967a`. That historical mismatch led to the workflow correction. Run #326 asserts the expected SHA and passed; it supersedes the earlier artifact for current mechanics evidence.
- Latest report gate fields: economic gate `NOT_EVALUATED`, `champion = null`, `promotion_status = BLOCKED_NO_ECONOMIC_GATE`.

## Open economic, data and operational gates

1. No owner-authorized historical OHLCV dataset/source/license with content hash has been registered for economic acceptance.
2. Owner-approved Phase 0 economic acceptance thresholds must be pre-registered before examining a real-market candidate. No threshold has been invented here.
3. Historical out-of-sample analysis across multiple windows and regimes, realistic fees/slippage, sensitivity and stability runs, and independent quantitative/risk review remain open.
4. Synthetic training/latency figures do not close production resource profiling.
5. External provider/license validation, current exchange market metadata, account/order reconciliation, and operational routing/recovery tests are outside Phase 4 offline CI and are not authorized by this report.

## Decision

- Implementation: **CONDITIONAL GO for owner review**, because exact-head CI run #326 passed on SHA `eccd694d571da22853ea81065fa224afbfc239b6`. The subsequent status-document commit must have its own successful exact-head run before the final PR disposition.
- Economic acceptance: **NO-GO**.
- Champion: `null`; promotion status: `BLOCKED_NO_ECONOMIC_GATE`.
- Trading authorization: **NO-GO**. This phase does not authorize live, dry-run or paper sessions, order writes, exchange/API access, live market-data download, provider requests, notifications, or host/network changes.
- PR #4 is not merged. Any transition out of Draft and any merge remain subject to the current exact-head evidence and the owner's explicit control.
- Phase 5 (Meta-Labeling / Triple-Barrier) has **not** started and remains blocked until the owner explicitly approves it.
