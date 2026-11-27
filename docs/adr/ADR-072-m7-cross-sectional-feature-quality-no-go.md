# ADR-072 — M7 G2.0 Cross-Sectional Evaluation: FEATURE_QUALITY / NO-GO

- **Status:** CLOSED — NO-GO / BLOCKED_DATA_OR_FEATURE_QUALITY
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Supersedes:** ADR-071 for the current G2.0 candidate definitions only. ADR-071 remains the historical G1.2 record.
- **Scope:** Two G2.0 preregistered logical candidates; Nobitex public Spot PRE-HACK data only.

## 1. Executive decision

The SOLUSDT acquisition failure was correctly isolated as an incomplete historical-coverage problem, not a missing market or transient network failure. Nine other Nobitex USDT pairs already had complete, immutable, exactly aligned histories. A superseding preregistration (M7-CS-G2.0) therefore fixed a five-pair and a nine-pair candidate without duplicating symbols, altering model parameters, or changing economic gates.

The G2.0 Standard CI gate passed. The M7 evaluation workflow also completed successfully and produced its report artifact. However, **both candidate panels were empty after feature qualification**:

- `NOBITEX-M7-5PAIR-RIDGE-TOP2-02`: `BLOCKED_DATA_OR_FEATURE_QUALITY`
- `NOBITEX-M7-9PAIR-RIDGE-TOP2-01`: `BLOCKED_DATA_OR_FEATURE_QUALITY`
- Exact reason for both: `m7_panel_empty_after_feature_qualification`
- Family result: **`NO-GO`**

No Ridge fit, walk-forward fold scoring, cost simulation, or trade accounting occurred. This is not a measured negative economic result. NetPF, MaxDD and completed OOS trades are **N/A**, not zero.

## 2. SOLUSDT acquisition diagnosis and data disposition

The earlier public Nobitex acquisition found SOLUSDT in public market statistics, but the public historical endpoint returned only 25,285 rows against 31,801 required rows:

| Field | Result |
|---|---:|
| Expected rows per symbol | 31,801 |
| SOLUSDT returned rows | 25,285 |
| Missing hourly intervals | 6,516 |
| First missing timestamp | 1635712200 (2021-10-31T20:30:00Z) |
| Last missing timestamp | 1659166200 (2022-07-30T07:30:00Z) |
| Acquisition disposition | `REJECTED_DATA_QUALITY_OR_API` |

Because the pair appeared in public market statistics, the evidence does not support saying that Nobitex does not list SOLUSDT. The failure is insufficient historical coverage on the requested interval.

Nine symbols qualified on the exact BTCUSDT hourly grid. Each has 31,801 rows, zero missing intervals, no duplicate timestamps, and a SHA-256-bound CSV/sidecar record:

`ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT, TRXUSDT, LTCUSDT, LINKUSDT, DOTUSDT`

The common UTC interval is 2021-10-31T20:30:00Z through 2025-06-17T20:30:00Z inclusive. BTCUSDT remains the contextual reference series, not a target holding.

XRPUSDT and DOGEUSDT were already members of the original candidate universe. Replacing SOLUSDT with XRPUSDT as an extra item would duplicate an existing asset and would not create a new independent pair. G2.0 instead fixed the exact set of nine already-qualified assets and recorded the superseding decision before any G2.0 economic outcome was available.

The existing dataset was not reacquired or overwritten. The acquisition output directory is immutable and non-empty; all nine qualifying histories were already committed with exact hashes. The evaluation report confirms `new_data_acquisition_during_evaluation=false`.

## 3. G2.0 preregistration

- Preregistration revision: `M7-CS-G2.0`
- Preregistration config path: `config/m7/cross_sectional_preregistration.json`
- Config commit: `9eb74829336a6b5e65433eb8b6742b66e59ca655`
- Reported preregistration SHA-256: `e5a1dfdfa92912635c6949934f353f2aff89c9a26cbc5e384de89a5d0d488817`
- Immutable aggregate qualification manifest: `data/nobitex/prehack/m7/m7_data_qualification.json`
- Aggregate manifest Git blob SHA: `f47a6968ab42dc9f578aeca1cd7aae4c7e555624`
- Immutable data commit: `5cf8f867385b6481bb47c8dc560ab160b39aee5d`
- BTCUSDT source SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`

The only G2.0 candidates were:

| Trial ID | Fixed target universe | Result |
|---|---|---|
| `NOBITEX-M7-5PAIR-RIDGE-TOP2-02` | ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT | `BLOCKED_DATA_OR_FEATURE_QUALITY` |
| `NOBITEX-M7-9PAIR-RIDGE-TOP2-01` | All nine qualified symbols | `BLOCKED_DATA_OR_FEATURE_QUALITY` |

The model and policy were not tuned: StandardScaler + Ridge(alpha=1.0), the same four fixed features, positive-score top-two long-only selection, 50% initial-equity allocation per selected asset, four chronological walk-forward folds, and 50/90 bps cost scenarios. All registered thresholds and feature definitions remain unchanged.

## 4. CI and evaluation execution evidence

### 4.1 Standard CI prerequisite

- Workflow: AliX Continuous Integration
- Run: [#2671 / Run ID 46053156707](https://github.com/angry0tiger-sudo/AliX/actions/runs/46053156707)
- Exact parent SHA: `9eb74829336a6b5e65433eb8b6742b66e59ca655`
- Conclusion: **SUCCESS**
- Required jobs passed: `quality`, `postgres-integration`, `four-arm-lock-bare-env`, `freqtrade-runtime-smoke`.
- Quality gates passed, including Ruff lint/format, strict Mypy, security scan, dependency resolution/lock/audit, architecture, Xenon complexity, unit tests and coverage.
- Unrelated real-data diagnostic, ensemble research and M3 OFI jobs were skipped.

### 4.2 M7 evaluation

- Workflow: M7 Cross-Sectional Ridge Economic Evaluation
- Run: [#2 / Run ID 46053508162](https://github.com/angry0tiger-sudo/AliX/actions/runs/46053508162)
- Trigger commit: `c1695b780fe8bb26f65226d8bd616736658a0b39`
- Conclusion: **SUCCESS** (workflow execution and report validation succeeded; economic verdict is NO-GO)
- Report artifact: `alix-m7-cross-sectional-results`
- Artifact ID: `15225039615`
- Artifact ZIP SHA-256: `cd30aa83281d63e1ccb3a06413ddf1155a6afda1ac007c800eaa58b726da8467`
- Artifact URL: https://github.com/angry0tiger-sudo/AliX/actions/runs/46053508162/artifacts/15225039615

The workflow passed the exact-parent Standard CI gate, checked out the locked preregistration and immutable data, passed the two-trial and safety assertions, ran the evaluator, validated the two-trial report, and uploaded the artifact.

## 5. Feature-qualification failure

The evaluator successfully loaded the fixed G2.0 symbols from the qualification manifest and proceeded to `build_panel()`; otherwise a missing symbol would have been reported as `BLOCKED_DATA`. Instead, the panel builder raised `m7_panel_empty_after_feature_qualification` for both universes.

The existing feature contract drops a candidate row when either `beta_24h_to_btc` or `relative_volume_shock_24h` is unavailable, and also requires all model features and the 4h target to be finite. The beta calculation rejects non-finite rolling return windows or BTC variance at/below its fixed floor; the volume-shock calculation rejects a non-finite or non-positive prior-24h median volume.

The current report does **not** include per-feature rejection counts. Therefore this execution does not establish whether beta, volume shock, or both account for every rejected row. No specific root cause beyond the empty feature-qualified panel is asserted here. Isolating those counters requires a separate, read-only PRE-HACK feature diagnostic in GitHub Actions. It must not silently relax the current feature contract or change this closed G2.0 evaluation.

Because panel construction failed before `_run_candidate()`, no model was fitted and no fold or cost-scenario metrics exist.

## 6. Metric disposition and horizon

| Candidate | Cost | NetPF | MaxDD | Completed OOS trades | Status |
|---|---:|---:|---:|---:|---|
| 5PAIR G2.0 | 50 bps | N/A | N/A | N/A | Feature panel empty |
| 5PAIR G2.0 | 90 bps | N/A | N/A | N/A | Feature panel empty |
| 9PAIR G2.0 | 50 bps | N/A | N/A | N/A | Feature panel empty |
| 9PAIR G2.0 | 90 bps | N/A | N/A | N/A | Feature panel empty |

The unchanged economic gates are NetPF >= 1.10, at least 100 completed OOS trades, MaxDD <= 15%, and at least 3 valid plus 3 stable folds out of 4. None can be measured because neither model fit nor the cost simulations were reached. The report's overall verdict remains `NO-GO` under fail-closed acceptance.

**Horizon note:** M7 uses 1h input candles and one preregistered 4h forward log-return target (next-hour open to open four hours later). No separate 1h forecast-horizon trial is registered or evaluated. Accordingly, 1h-horizon NetPF/MaxDD/Trades are N/A, not inferred from the 4h target.

## 7. Safety and governance invariants

The final report explicitly records:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `holdout_bytes_read = false`
- `posthack_data_read = false`
- `new_data_acquisition_during_evaluation = false`

No local test, fitting, backtest, or data analysis was executed. Evaluation ran only in GitHub Actions. No private API or order endpoint was called. The `main` branch was not modified.

## 8. Final decision and next-step boundary

**M7 G2.0: NO-GO / BLOCKED_DATA_OR_FEATURE_QUALITY.** The SOLUSDT coverage blocker was removed from the candidate definitions through an explicit pre-outcome superseding preregistration, and nine exact-history-qualified pairs were verified. The subsequent feature panel was nevertheless empty, so there is still no eligible economic evaluation.

Do not mark the strategy as a losing model based on nonexistent PnL, and do not reopen this G2.0 result by weakening feature requirements. Any next experiment must first preregister a separate feature-availability diagnostic or model protocol, identify the beta/volume feature failure using PRE-HACK data only, retain the failed G2.0 result in the trial ledger, and pass the unchanged Standard CI and safety gates.

AliX remains an R&D project. No Champion, paper-trading permission, live-trading permission, or order authorization is created by this ADR.
