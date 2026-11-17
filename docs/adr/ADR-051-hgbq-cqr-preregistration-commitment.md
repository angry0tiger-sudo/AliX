# ADR-051 — HGBQ-CQR G3 Preregistration Commitment

- **Status:** Accepted; parameter lock only. Empirical execution remains **NOT AUTHORIZED**.
- **Date:** 2026-11-17
- **Branch:** `research/nobitex-hf-strategy`
- **G2 baseline SHA:** `6553e4dd2497109395fae8e730cc7722b924900e`
- **Configuration:** `config/regime_r/hgbq_cqr_preregistration.json`
- **Configuration SHA-256:** `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- **Related:** ADR-050, Regime R contract, G3 Owner Directive.
- **Scope:** immutable parameter and temporal-protocol registration only.

## 1. Decision and commitment

The HGBQ-CQR model family and its temporal evaluation protocol are frozen in the JSON configuration named above. The SHA-256 value in this ADR is the byte-level commitment to that file, including its final newline. Any modification, including whitespace or key-order changes, changes the commitment and is prohibited without explicit written Owner approval and a new superseding ADR/commitment.

The 30 stable logical trial identifiers are retained from the existing Regime R contract. Their legacy `NGBOOST` prefix is preserved for lineage and multiple-testing accounting; their preregistered implementation revision is `HGBQ-CQR-G3`. They are not 30 additional logical slots.

## 2. Frozen HGB quantile estimators

All three independent `HistGradientBoostingRegressor` estimators use the same common parameters:

| Parameter | Frozen value |
|---|---:|
| `loss` | `quantile` |
| `max_iter` | `100` |
| `learning_rate` | `0.1` |
| `max_depth` | `3` |
| `max_leaf_nodes` | `8` |
| `min_samples_leaf` | `20` |
| `l2_regularization` | `1.0` |
| `max_bins` | `255` |
| `early_stopping` | `false` |
| `random_state` | `42` |

The three frozen quantile levels are `[0.05, 0.50, 0.95]`. A shallow tree, minimum leaf size of 20, positive L2 regularization, disabled implicit early stopping, and a fixed seed are intended to constrain variance and eliminate discretionary tuning after outcomes are observed.

## 3. Frozen MAPIE and forecast horizons

- MAPIE version: `1.6.0`.
- Estimator: `ConformalizedQuantileRegressor`, prefit mode.
- `confidence_level=0.90` (`alpha=0.10` conceptually).
- Estimator order supplied to MAPIE: `[lower, upper, median]`, consistent with ADR-050.
- Calibration data must be temporally separate from training and test data. Test rows must never be used for conformalization.
- Prediction horizons: **1, 4, and 24 hours**.

## 4. Frozen purged walk-forward protocol

The protocol is a three-fold rolling, fixed-width, UTC walk-forward schedule. Every interval is start-inclusive and end-exclusive. Each fold has:

1. Six calendar months of training.
2. One calendar month of separate MAPIE calibration.
3. One full calendar month of embargo, excluded from fitting, calibration, and test scoring.
4. One calendar month of test evaluation.

The three fixed relative schedules are:

| Fold | Training offsets | Calibration offsets | Embargo offsets | Test offsets |
|---|---|---|---|---|
| WF-01 | [0, 6) months | [6, 7) | [7, 8) | [8, 9) |
| WF-02 | [3, 9) months | [9, 10) | [10, 11) | [11, 12) |
| WF-03 | [6, 12) months | [12, 13) | [13, 14) | [14, 15) |

A 24-hour target-overlap purge is applied at training/calibration and calibration/embargo boundaries because 24 hours is the maximum registered prediction horizon. Absolute UTC anchor `T0` is intentionally not selected in G3; it must be frozen by a separate Owner-approved data-window authorization before any empirical data is opened, never after outcome inspection.

## 5. Frozen cost scenarios and decision boundary

| Scenario | Round-trip cost | Return fraction |
|---|---:|---:|
| `cost_50bps` | 50 bps | 0.005 |
| `cost_90bps` | 90 bps | 0.009 |

For each scenario independently, the existing fail-closed decision is retained: `ENTER` only if the valid conformal lower bound is **strictly greater** than the applicable return fraction; equality or invalid/missing values return `ABSTAIN`. This is a research gate only and is not trading authorization.

## 6. Trial registry

The 30 stable logical IDs declared by the Regime R contract are registered in `docs/research/TRIAL_REGISTRY.md` with status `PREREGISTERED_LOCKED`, implementation revision `HGBQ-CQR-G3`, and evaluated status `No`. This is an append-only row-level registration; historical workflow/candidate records and the registry's `GLOBAL_N=UNKNOWN` accounting are not rewritten or collapsed into a guessed global count.

## 7. Explicit no-fit / anti-p-hacking boundary

> «هیچ مدلی روی داده‌های واقعی fit نشده و این پیش‌ثبت برای جلوگیری از p-hacking است. G3 فقط شامل قفل کردن پارامترهاست.»

No model has been fitted on real data. The purpose of this preregistration is to prevent p-hacking. G3 consists only of locking the parameters and protocol.

G3 does **not** authorize:
- reading, fetching, or fitting on real market data;
- Holdout access or use of any post-hack data;
- economic backtests, fold scoring, hyperparameter search, model selection, or result generation;
- network/private API access, order dispatch, paper trading, or live trading.

All 30 trial entries remain not evaluated. The G3 configuration is a frozen specification, not evidence of model quality or economic performance.

## 8. Runtime integration blocker before any future fit

The current G2 adapter is not yet wired to every G3 value: its existing constructor defaults and fit construction use earlier `max_iter` / `l2_regularization` values and do not yet expose the registered `learning_rate` and `max_depth` (nor the registered leaf/bin settings). G3 deliberately does not modify the adapter or run a fit. Before any future model fitting, a separately authorized implementation step must bind the runtime adapter to this exact JSON and add tests that fail on configuration drift. No real-data fit is permitted until that integration is verified and the Owner issues a separate explicit authorization.

## 9. Verification and preserved governance

- G2 Standard CI #2457: run ID `44044590795`, SHA `6553e4dd2497109395fae8e730cc7722b924900e`, conclusion **success**.
- G2 Quality Preflight #52: run ID `44044590785`, SHA `6553e4dd2497109395fae8e730cc7722b924900e`, conclusion **success**.
- `main` remains frozen at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`; Holdout remains sealed.

**G3 outcome:** configuration and protocol preregistered and locked only. G3 completion does not authorize G4/G5 data acquisition, real-data fitting, Holdout access, backtesting, or trading.

Architecture First. Evidence Always. Fail Closed.
