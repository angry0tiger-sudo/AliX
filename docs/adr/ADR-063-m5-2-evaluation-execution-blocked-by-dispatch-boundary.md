# ADR-063 — M5.2 Evaluator Ready; Economic Evaluation Blocked by GitHub Dispatch Boundary

- **Status:** **IMPLEMENTATION COMPLETE — STANDARD CI GREEN — ECONOMIC RUN NOT EXECUTED**
- **Date:** 2026-11-27
- **Branch:** \`research/nobitex-hf-strategy\`
- **Validated implementation HEAD:** \`c94924c10e1969741b19928f9e7c29be0e210370\`
- **Target:** Nobitex Spot / USDT — \`BTCUSDT\` only
- **Preregistration:** \`ALIX-M5-2-SENTIMENT-G1\`
- **Preregistration SHA-256:** \`851d6c4b39c41cacec8ccd712cf17bb40dd42dd45f2bb8813920a7f2ebe481ba\`

## 1. Decision

The M5.2 evaluator, mock tests and dispatch-only GitHub Actions workflow are implemented on the research branch. Standard CI and the Four-Arm Quality Preflight both completed successfully on the exact implementation HEAD.

The economic evaluation was **not executed**. The manual workflow \`m5-2-sentiment-evaluation.yml\` exists on \`research/nobitex-hf-strategy\`, but it does not exist on the repository's default branch, \`main\`. GitHub documents that a \`workflow_dispatch\` trigger only receives events when the workflow file exists on the default branch. The workflow has never been run once from the default branch, so selecting the research ref does not remove this prerequisite.

The Owner directive authorizes the M5.2 evaluation while separately requiring that \`main\` remain unchanged. Those two constraints cannot both be satisfied through the current GitHub Actions dispatch path. The dispatch attempt through the available browser automation also failed before a GitHub run was created. No API or workflow-run evidence exists for an M5.2 evaluation.

**Decision:** Preserve \`main\` exactly as instructed. Do not claim an economic PASS or NO-GO from nonexistent results. Record the M5.2 economic verdict as **NOT RUN — NO ECONOMIC CONCLUSION**. Trading authorization remains \`NO-GO\`.

Official GitHub documentation:
- [Events that trigger workflows — workflow_dispatch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)
- [Manually running a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

## 2. Implementation delivered

The following M5.2 files are present on the research branch:

- \`scripts/run_m5_2_sentiment_model.py\`
- \`tests/unit/test_m5_2_sentiment_model.py\`
- \`.github/workflows/m5-2-sentiment-evaluation.yml\`
- \`.github/workflows/m5-2-source-format.yml\`

The evaluator is fail-closed against immutable data hashes and the locked preregistration. It uses the existing \`scripts/align_m5_sentiment.py\` implementation and the point-in-time rule:

\`available_at <= decision_time\`

The control arm contains exactly the eight preregistered OHLCV features. The treatment arm adds only \`fear_greed_value\`. Both use the same paired decision rows, Logistic Regression pipeline, horizons, four expanding chronological folds, labels, entry/exit conventions, position sizing and costs.

The registered logical trial family remains exactly four candidates:

1. OHLCV control — 1h
2. Sentiment augmented — 1h
3. OHLCV control — 4h
4. Sentiment augmented — 4h

The model remains \`StandardScaler + LogisticRegression\`, with the frozen parameters and probability threshold from the preregistration. No DeepLOB, Transformer, parameter search, new feature, threshold sweep or alternate model was introduced.

## 3. Immutable data bindings

### 3.1 Nobitex BTCUSDT OHLCV

The evaluator is bound to the already-persisted public pre-hack dataset. No new market-data acquisition was performed for M5.2.

| Binding | Value |
|---|---|
| Venue / market | Nobitex Spot / BTCUSDT |
| Timeframe | 1h |
| Rows | 31,801 |
| Window start | 2021-10-31T20:30:00Z |
| Window end | 2025-06-17T20:30:00Z |
| Missing hourly intervals | 0 |
| CSV SHA-256 | \`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e\` |
| Source manifest SHA-256 | \`b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26\` |
| M5.2 data-lock SHA-256 | \`e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf\` |
| Scope | PRE-HACK ONLY |

The data lock is \`data/nobitex/prehack/m5_2_btcusdt_1h_data_lock.json\`. The original CSV and source manifest were not modified.

### 3.2 Alternative.me Fear & Greed

The evaluator is bound to the existing 3,218-record snapshot:

- Raw SHA-256: \`4adba36f7dd95792520e078f7578c316c23e80a7b8a0ace81df490bf06c740df\`
- Normalized SHA-256: \`e0578509cf3f8a6eeac39ddbd2edeab876e47027d759558e120c6f8b3a9f7f61\`
- Availability convention: observation timestamp plus the conservative 24-hour proxy.
- Alignment: latest observation with \`available_at <= decision_time\`; forward-fill only after availability.
- Rows with no prior available value are excluded from both paired arms.

The provider's \`available_at\` is explicitly a conservative proxy, not a provider-measured historical publication time. The sentiment series is a daily global BTC context measure; repeated hourly values are dependent observations, not independent hourly samples.

## 4. CI and quality evidence

All results below correspond to implementation HEAD \`c94924c10e1969741b19928f9e7c29be0e210370\`.

| Workflow / check | Run | Result |
|---|---:|---|
| Standard CI | [#45994508199](https://github.com/angry0tiger-sudo/AliX/actions/runs/45994508199) | **SUCCESS** |
| Four-Arm Quality Preflight | [#45994508078](https://github.com/angry0tiger-sudo/AliX/actions/runs/45994508078) | **SUCCESS** |
| M5.2 Source Formatting | [#45994508186](https://github.com/angry0tiger-sudo/AliX/actions/runs/45994508186) | SUCCESS |
| Ruff Format Assist | [#45994508113](https://github.com/angry0tiger-sudo/AliX/actions/runs/45994508113) | SUCCESS |

Standard CI quality evidence:

- Ruff lint and Ruff format check: PASS.
- Strict Mypy: PASS.
- Bandit security scan: PASS.
- Dependency resolution, transitive lock and dependency audit: PASS.
- Architecture contracts: PASS.
- Xenon complexity gate: PASS; the existing thresholds were not weakened.
- Four-arm runner module invocation contract: PASS.
- Unit tests: **1,125 passed, 5 skipped**.
- Total coverage: **90.36%**; configured minimum: 90%.
- PostgreSQL integration, Freqtrade runtime smoke and four-arm lock/bare-environment jobs: PASS.

The added M5.2 Mock tests cover the frozen feature contract, as-of alignment and no-backfill behavior, fold chronology and purge, Logistic Regression parameters, paired comparisons, economic thresholds, and portfolio mark-to-market including round-trip cost.

## 5. Economic trial outcome

No M5.2 workflow run was created. Therefore no fit, prediction, backtest, OOS trade, or economic result exists.

| Trial | Cost-50 NetPF | Cost-90 NetPF | Trades | MaxDD | Fold stability |
|---|---:|---:|---:|---:|---|
| Control 1h | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Sentiment 1h | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Control 4h | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Sentiment 4h | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

The paired comparison is also **NOT RUN**. It is therefore unknown whether adding \`fear_greed_value\` improves NetPF in either horizon, whether either arm reaches NetPF 1.10, at least 100 completed trades and MaxDD no greater than 15%, or whether the sentiment arm improves paired PnL.

No prediction CSV, M5.2 economic report, or M5.2 result artifact has been generated.

### 5.1 Adjudication

- **Economic verdict:** \`NOT RUN — NO ECONOMIC CONCLUSION\`.
- **Sentiment incremental value:** NOT MEASURED.
- **NetPF >= 1.10 in both horizons:** NOT MEASURED.
- **Paired improvement over control:** NOT MEASURED.
- **DSR:** NOT COMPUTABLE; retain \`GLOBAL_N=UNKNOWN\` and \`GLOBAL_DSR=NOT_COMPUTABLE\`.
- **First economic edge claim:** NOT PERMITTED.
- **Champion:** \`null\`.
- **order_permission:** \`false\`.
- **trading_authorization:** \`NO-GO\`.

This is an execution blocker, not evidence that sentiment succeeded or failed economically. A statement that Fear & Greed did not create an edge would be unsupported because none of the four registered candidates was evaluated.

## 6. Dispatch boundary and required prerequisite

Verified repository facts:

- Default branch: \`main\`.
- \`main\` commit: \`cd9b13d42ac0cf0edd2df066ff81580e028146d6\`.
- \`main\` tree SHA: \`5761c6c34143f8e6924f53a4e745c48932a5f791\`.
- \`m5-2-sentiment-evaluation.yml\`: present on \`research/nobitex-hf-strategy\`, absent from \`main\`.
- No M5.2 evaluation Workflow Run ID or results artifact exists.

GitHub's documented prerequisite is that a \`workflow_dispatch\` workflow file must exist on the default branch before that workflow can receive the event. Dispatching it against a non-default ref is supported only after that workflow has been registered from the default branch.

The remaining prerequisite is therefore a repository governance decision that makes this exact manual workflow available on the default branch, or another supported dispatch mechanism that meets GitHub's documented prerequisite. That would change the current \`main\` tree and conflicts with the Owner's instruction to preserve it. No such change was made here.

## 7. Safety record

- New OHLCV or sentiment acquisition for M5.2: 0.
- Private API calls: 0.
- Holdout access: prohibited; no Holdout bytes read.
- Post-Hack access: prohibited; no Post-Hack data read.
- Real order attempts: 0.
- Paper orders: 0.
- Local model fit or local backtest: not performed.
- \`Champion = null\`; \`order_permission = false\`; \`trading_authorization = NO-GO\`.

## 8. Final disposition

M5.2 engineering implementation and quality verification are complete. Economic evaluation remains pending because the dispatch-only workflow cannot be activated while it is absent from the default branch and the default branch is required to remain unchanged.

Preserve this ADR as the terminal record for the current execution attempt. Do not convert the missing run into a statistical NO-GO, do not tune the registered model, and do not access Holdout or Post-Hack data.
