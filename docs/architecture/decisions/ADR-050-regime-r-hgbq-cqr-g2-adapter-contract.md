# ADR-050 — G2 Completion: HGBQ-CQR API Compatibility and Adapter Contract

- **Status:** Accepted for the G2 mock-only implementation; empirical execution remains NOT AUTHORIZED.
- **Date:** 2026-11-17
- **Branch baseline:** `research/nobitex-hf-strategy`
- **Baseline HEAD:** `0b65474ad860298ba197c6cf3bf8782858a2c26c`
- **Related:** ADR-045, ADR-047, ADR-048, Owner G2 Directive (2026-11-17)
- **Implementation boundary:** synthetic arrays only; no network, real-data read, Holdout access, economic backtest, private API, dispatch, nonce, paper order or live order.

## 1. Decision and relationship to prior records

The Owner's G2 directive selects **HistGradientBoosting Quantile Regression + MAPIE Conformalized Quantile Regression (HGBQ-CQR)** as the implementation target for the dependency-approved stack:

- `scikit-learn==1.9.0`
- `MAPIE==1.6.0`

This directive supersedes ADR-048's NGBoost base-learner choice for the current Regime R implementation path. It does not erase historical NGBoost research, any prior trial exposure, or the 30-slot selection-family accounting. A base-learner replacement is a candidate revision within an existing logical slot, with parent/supersedes lineage, and does not authorize adding 30 more slots.

**Repository evidence discrepancy:** the active branch at baseline `0b65474...` contained ADR-048 as its highest numbered decision record; an ADR-049 artifact was not present in `docs/architecture/decisions/`, although the Owner's instructions refer to ADR-049 as the fallback-governance record. This ADR therefore records the explicit Owner G2 direction and the relevant fallback/accounting constraints without claiming that the missing ADR-049 file was verified.

No trial registry rows are created or changed by G2. The existing 30 logical slots remain append-only and unchanged.

## 2. HGB quantile learners

The adapter constructs three independent `HistGradientBoostingRegressor` instances with:

- `loss="quantile"`
- lower quantile `(1 - confidence_level) / 2`
- median quantile `0.5`
- upper quantile `(1 + confidence_level) / 2`
- `l2_regularization` from the scikit-learn API (not an NGBoost-specific parameter)
- deterministic `random_state`
- `early_stopping=False`, so HGB does not create an implicit internal validation split.

At the default confidence level of 0.90, the quantile levels are 0.05, 0.50 and 0.95. The value of `l2_regularization` is an explicit adapter parameter and defaults to scikit-learn's documented 0.0; it is not silently set to 12.

## 3. MAPIE 1.6.0 API contract

The adapter uses the public `mapie.regression.ConformalizedQuantileRegressor` API in split-conformal prefit mode:

1. Fit the lower, median and upper HGB estimators on the supplied training arrays.
2. Construct MAPIE with `prefit=True` and the fitted estimator list in MAPIE's required order: **[lower, upper, median]**. This differs intentionally from the public forecast tuple order **[lower, median, upper]**.
3. Call `.conformalize(X_conformalize, y_conformalize)` on a separate calibration set. MAPIE's `.fit()` is not called in prefit mode.
4. Call `.predict_interval(X)`; MAPIE 1.6.0 documents the point predictions as shape `(n_samples,)` and the single confidence interval as `(n_samples, 2, 1)`.

This workflow was checked against the tagged MAPIE v1.6.0 public source and API documentation, including the required prefit estimator order and separate conformalization lifecycle:
- https://github.com/scikit-learn-contrib/MAPIE/blob/v1.6.0/mapie/regression/quantile_regression.py
- https://mapie.readthedocs.io/en/stable/api/regression/

The scikit-learn estimator is the pinned `HistGradientBoostingRegressor` API:
- https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

## 4. Forecast semantics: quantile summary is not a full distribution

The generic `DistributionForecast` contract now carries explicit `output_semantics`:

- `distributional`: a model may provide a finite mean, positive dispersion and ordered quantiles.
- `quantile_summary`: a model provides ordered conditional quantiles only. Mean and dispersion must be `None`.

HGBQ-CQR is a **quantile-summary** forecaster. It does not estimate or return a full conditional probability density/distribution. The adapter must not relabel the median as the conditional mean or the quantile spread as a standard deviation. This prevents a false NGBoost-equivalence claim while preserving the existing generic forecast envelope.

The conformal interval is represented separately as `ConformalInterval`, including bounds, nominal coverage, signed corrections and calibration sample count.

## 5. Fail-closed cost decision

For each of the two preregistered round-trip cost scenarios, the research gate is strict:

```text
ENTER(c) iff valid_conformal_lower_bound > c / 10_000
ABSTAIN otherwise
```

- Cost-50 threshold: `0.005` (50 bps).
- Cost-90 threshold: `0.009` (90 bps).
- Equality does not pass.
- The two cost ledgers are evaluated independently. Passing 50 bps does not imply passing 90 bps.
- Missing, malformed, non-finite, inverted, uncalibrated or unsupported interval/cost inputs return `ABSTAIN`.

`ENTER` is a **research-gate label only**. It is not an order, execution instruction, Champion selection or trading permission. All global safety invariants remain unchanged.

## 6. Mock-only implementation and test boundary

The only model fitting exercised by this G2 implementation is unit-test fitting on deterministic synthetic arrays generated in the test suite. Training and conformalization arrays are separate. The adapter has no market-data reader, API client, network call, file loader, backtest or execution integration.

The GitHub Actions test path must install the pinned MAPIE version and exercise:

- the three fitted HGB quantile estimators and MAPIE prefit CQR integration;
- quantile ordering and explicit quantile-only forecast semantics;
- MAPIE interval output shape and ConformalInterval validation;
- strict 50/90 bps decisions, including ENTER, ABSTAIN, equality and invalid-input cases;
- fail-closed behavior before fit and on malformed arrays/configuration.

Standard CI retains the existing coverage threshold (at least 90%) and all Ruff, strict Mypy, Bandit, Xenon, dependency and test gates. Quality Preflight is extended to lint/scan the Regime R path and install MAPIE 1.6.0; no gate is weakened.

## 7. Empirical evidence and authorization status

G2 produces **no economic result**. Synthetic unit-test outputs do not establish:

- market forecast quality;
- empirical conformal coverage on financial returns;
- temporal or conditional coverage;
- calibrated coverage under non-stationarity;
- net profitability, trade frequency, drawdown or risk-adjusted returns.

No model has been fitted on pre-hack or post-hack market data in this phase. The Holdout remains sealed. Any future real-data read, fit, calibration, inference, tuning, backtest or confirmatory evaluation requires the appropriate later Gate and separate Owner authorization.

## 8. Safety invariants

Unchanged:

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- Holdout remains SEALED
- `main=FROZEN`

**G2 scope:** API compatibility contract, synthetic-only adapter tests and fail-closed decision mechanics. It grants no authority beyond that scope.

Architecture First. Evidence Always. Fail Closed.
