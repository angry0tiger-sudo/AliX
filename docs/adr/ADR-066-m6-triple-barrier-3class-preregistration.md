# ADR-066 — M6 Three-Class Triple-Barrier Classification Preregistration

- **Status:** **PREREGISTERED_LOCKED — IMPLEMENTATION DELIVERED — ECONOMIC EVALUATION PENDING**
- **Date:** 2026-11-27
- **Repository:** `angry0tiger-sudo/AliX`
- **Branch:** `research/nobitex-hf-strategy`
- **Target:** Nobitex Spot / USDT — `BTCUSDT` only
- **Preregistration:** `config/m6/triple_barrier_3class_preregistration.json`
- **Preregistration SHA-256:** `f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42`
- **Logical candidates:** exactly two — 1h and 4h decision cadence
- **Related prior:** ADR-056 — binary Triple-Barrier + Logistic Regression, CLOSED NO-GO

## 1. Owner decision and scope

The Owner selected Option A (target redesign) and deferred/rejected Option B (paid On-Chain data acquisition for now). This ADR locks a materially different three-class event target and one fixed RandomForestClassifier configuration.

The M6 hypothesis is that retaining vertical timeouts as a genuine third class, instead of excluding them from binary first-touch scoring, may improve the usefulness of the event model for selective long-only decisions. The registered signal is a long entry only when the raw model probability for `TP_FIRST` is **strictly greater than 0.60**.

This is an exploratory PRE-HACK-only research phase. It does not promote a model, create a Champion, grant Paper/Live permission, or open Holdout.

## 2. Explicit distinction from the closed Regime T experiment

ADR-056 already evaluated one binary experiment (`REGIME-T-TB-LOGISTIC-01`) on the same committed BTCUSDT PRE-HACK data. It used 2×ATR barriers, a 24-hour vertical window, a fixed Logistic Regression, three temporal folds, and a pooled accuracy screen. Its pooled OOF accuracy was 51.09% against the strict `>55%` screen, and the result was NO-GO.

M6 is not a rerun of that binary model:

- `TP_FIRST`, `SL_FIRST`, and `VERTICAL_TIMEOUT` are all retained as supervised classes.
- The learner is one fixed `RandomForestClassifier` rather than Logistic Regression.
- There are two registered decision cadences (1h and deterministically aggregated 4h).
- Acceptance is based on costed economic outcomes, trade count, MaxDD, and fold stability, not the old classification-accuracy screen.

ADR-056 remains a related prior trial for multiplicity and interpretation. M6 does not overwrite it, and `GLOBAL_N=UNKNOWN` remains unchanged.

## 3. Immutable data and market contract

Only the already committed public Nobitex Spot BTCUSDT PRE-HACK hourly CSV may be read:

- CSV: `data/nobitex/prehack/nobitex-btcusdt-1h.csv`
- Rows: 31,801
- Window: `2021-10-31T20:30:00Z` through `2025-06-17T20:30:00Z`
- CSV SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Manifest SHA-256: `b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26`
- M5.2 data-lock SHA-256: `e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf`

All three byte-level commitments and the PRE-HACK scope are checked before feature construction. Any mismatch fails closed. No data is fetched, purchased, appended, refreshed, converted from BTCIRT, or substituted from another venue.

The 4h decision series is made only by grouping four consecutive hash-verified 1h candles from the first committed timestamp: first open, maximum high, minimum low, final close, and summed volume. A trailing incomplete group is dropped. The underlying original 1h candles remain the event-resolution and mark-to-market source for both decision cadences.

## 4. Locked three-class target

Decision is made after a completed 1h or 4h candle. Entry is the open of the next complete selected-timeframe candle. The volatility scale is Wilder ATR(14) divided by the selected-timeframe close at the decision point.

- TP barrier: `entry_open × (1 + 2 × ATR14_fraction)`.
- SL barrier: `entry_open × (1 - 2 × ATR14_fraction)`.
- Vertical barrier: 24 elapsed hours from entry.
- `TP_FIRST = 0`, `SL_FIRST = 1`, `VERTICAL_TIMEOUT = 2`.

The event path is resolved with the original hourly OHLC bars for both candidates. The 1h candidate examines 24 hourly bars; the 4h candidate examines the same 24 underlying hours (six complete 4h periods). If neither horizontal barrier is reached, `VERTICAL_TIMEOUT` is a valid third class and exits at the close of the 24th hourly candle.

A future bar opening beyond TP or SL resolves at the bar open (favorable TP gap at open, adverse SL gap at the worse open). If one hourly candle touches both horizontal barriers without an opening-gap resolution, the event is `AMBIGUOUS`: it is not a fourth class and is excluded from classifier fit and classification metrics. If an OOS signal nevertheless corresponds to an ambiguous event, the economic ledger conservatively charges it at the SL price. No intrabar path is guessed. Incomplete 24-hour windows are unavailable and excluded.

## 5. Frozen eight-feature schema

Exactly the existing M5.2 OHLCV-only features are used, with no Fear & Greed, sentiment, OFI, order-book, tick, trade-flow, or On-Chain input:

1. `log_return_1`
2. `log_return_4`
3. `log_return_24`
4. `realized_volatility_6`
5. `realized_volatility_24`
6. `range_fraction_1`
7. `atr_fraction_14`
8. `log_volume_change_1`

All calculations use completed bars in the selected timeframe, right-aligned rolling windows, and data available by decision time. For 4h, the window units are 4h bars. Missing/warm-up rows are dropped without imputation. ADR-055 documents that the historical dataset's volume values are zero; `log_volume_change_1` is retained because the Owner requested the exact eight-feature OHLCV contract, but it must be disclosed as non-informative on this dataset. It is not silently removed or replaced.

## 6. Frozen model and signal

One `sklearn.ensemble.RandomForestClassifier` specification is registered:

- `n_estimators=300`
- `criterion=gini`
- `max_depth=5`
- `min_samples_split=40`
- `min_samples_leaf=20`
- `max_features=0.8`
- `bootstrap=true`, `max_samples=0.8`
- `class_weight=balanced_subsample`
- `random_state=42`, `n_jobs=1`

No alternative model, ensemble, scaler, calibration, class resampling, feature selection, parameter search, threshold sweep, or post-result tuning is authorized. Random Forest is not assumed to be immune to overfitting; the shallow depth and minimum leaf/split settings are fixed regularization controls, and the chronological protocol remains mandatory.

The only long-entry condition is `P(TP_FIRST) > 0.60`. Equality does not enter; `SL_FIRST` does not create a short trade; all other decisions remain in cash.

## 7. Walk-forward, purge, and embargo

Each decision cadence independently sorts feature-eligible UTC timestamps and divides them into five contiguous chronological blocks using fixed index boundaries. Four expanding folds use B1→B2, B1–B2→B3, B1–B3→B4, and B1–B4→B5.

- Only prior blocks may be used for training.
- Minimum training set: 200 rows and at least 20 examples of each of the three classes.
- Training labels must be fully resolved.
- Purge uses the maximum possible 24-hour event window even if a horizontal barrier is touched early.
- Additionally, maximum train-event end must be strictly earlier than the first test entry boundary minus the fixed 24-hour embargo.
- Test events are retained only if the full 24-hour observation window resolves inside that same test block.
- Test data is never used in fit, feature selection, calibration, or threshold selection.
- Any insufficient fold is reported; folds, horizons, minimums, or barriers are not altered to rescue sample size.

## 8. Economic gates

For each horizon and each cost scenario independently:

- Cost-50: 50 bps all-in round-trip.
- Cost-90: 90 bps all-in round-trip.
- NetPF ≥ 1.10.
- At least 100 completed costed OOS trades.
- MaxDD ≤ 15% on chronological underlying-hourly NAV marks.
- At least 3 of 4 stable folds. A stable fold requires at least 20 completed trades, positive net PnL, and fold NetPF > 1.00.

The 1h cohort notional is 1/25 of initial equity per signal; the 4h cohort notional is 1/7. This reserves one conservative boundary-overlap slot beyond the maximum 24-hour concurrency, keeping planned gross exposure at or below 1.0×. No leverage, compounding or NAV-based resizing is used. Apply the full round-trip cost once per cohort. A trade's exit is TP, SL, vertical close, or conservative SL for an ambiguous OOS event.

Both horizons must independently pass all gates under both costs for the overall economic-gates verdict to be PASS. DSR remains `NOT_COMPUTABLE` while `GLOBAL_N=UNKNOWN`; even a pass is exploratory evidence only and never grants trading authority.

## 9. Logical trial accounting

Exactly two logical candidate identities are registered. Folds, three classes, prediction rows, and Cost-50/Cost-90 replays do not create additional candidate variants.

| Trial ID | Decision cadence | Model | Status at registration |
|---|---|---|---|
| `NOBITEX-M6-1h-RF-3CLASS-01` | 1h | RandomForestClassifier | `PREREGISTERED_LOCKED` |
| `NOBITEX-M6-4h-RF-3CLASS-01` | 4h | RandomForestClassifier | `PREREGISTERED_LOCKED` |

## 10. Implementation and evaluation workflow

Implementation files:

- `scripts/run_m6_triple_barrier_3class.py`
- `tests/unit/test_m6_triple_barrier_3class.py`
- `.github/workflows/m6-3class-evaluation.yml`

The workflow is a research-branch push-triggered workflow activated only by a dedicated M6 evaluation sentinel after Standard CI has passed on the implementation commit. This avoids modifying `main` or relying on a `workflow_dispatch` file being exposed on the default branch. The workflow checks the successful Standard CI run for the exact parent implementation SHA before loading the existing PRE-HACK dataset. It is pinned to the `research/nobitex-hf-strategy` branch.

Mock tests use only synthetic fixtures; they exercise three-class outcomes, same-bar ambiguity, gap handling, 4h aggregation, feature schema, 24-hour purging/embargo, threshold strictness, Random Forest configuration, and costed portfolio bookkeeping. No real-data fit is run locally.

## 11. Safety invariants

- `BTCUSDT` only; PRE-HACK only.
- No new data acquisition, data purchase, sentiment, microstructure, or On-Chain reads.
- `holdout_bytes_read=false`; `posthack_data_read=false`.
- `private_api_calls=0`; `real_order_attempts=0`; `paper_orders=0`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.
- No changes to `main`.
- All empirical fitting/economic evaluation, if CI succeeds, occurs only in the pinned GitHub Actions workflow.

## 12. Final disposition at preregistration

M6 has two locked logical candidates. Implementation and mock tests are delivered in this change set; economic fitting and backtesting have not yet been performed at the time of this ADR. ADR-067 will record the actual CI/workflow run IDs, artifact hashes, per-fold and pooled economics, and the final outcome for both 1h and 4h. No metric is assumed or pre-judged here.

Architecture First. Evidence Always. Fail Closed.
