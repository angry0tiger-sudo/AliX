# ADR-050 — HGBQ-CQR Adapter and API Compatibility

- **Status:** Accepted for the G2 synthetic-only implementation; empirical execution remains **NOT AUTHORIZED**.
- **Date:** 2026-11-17
- **Branch:** `research/nobitex-hf-strategy`
- **Implementation baseline:** `0b65474ad860298ba197c6cf3bf8782858a2c26c`
- **Selected stack:** `scikit-learn==1.9.0`, `MAPIE==1.6.0`
- **Related decisions:** ADR-047, ADR-048, and the Owner's G2 directive.
- **Scope:** adapter API compatibility, synthetic-only tests, and fail-closed lower-bound cost decisions.

> This is the requested `docs/adr/` publication of ADR-050. The existing
> `docs/architecture/decisions/ADR-050-regime-r-hgbq-cqr-g2-adapter-contract.md`
> is retained as the architecture-decision record; this file documents the same
> decision and is not a second, independent ADR.

## 1. Decision

Regime R G2 implements `HGBQCQRAdapter`, combining three
`HistGradientBoostingRegressor` quantile estimators with MAPIE's
`ConformalizedQuantileRegressor` (CQR) using the pinned MAPIE 1.6.0 API.

For the default 90% confidence level, the adapter trains lower, median, and
upper quantile estimators at 0.05, 0.50, and 0.95 respectively. MAPIE's prefit
CQR interface receives the fitted estimators in its required order
`[lower, upper, median]`; the public forecast keeps the semantic order
`[lower, median, upper]`. Training and conformalization arrays are supplied
separately by the caller.

The adapter's MAPIE and scikit-learn imports are isolated behind a narrow
runtime-import boundary because the pinned upstream packages do not expose
type stubs/`py.typed` metadata to the repository's strict Mypy configuration.
The quality gate remains strict; no missing-import suppression or gate
relaxation is introduced.

## 2. Forecast contract: quantile summary, not a full distribution

HGBQ-CQR is a quantile forecaster. It does not estimate a complete conditional
probability distribution, so the adapter returns a
`DistributionForecast` with:

- `quantile_levels=(0.05, 0.50, 0.95)` at the default 90% confidence level;
- `quantile_values=(lower, median, upper)` in monotonic order;
- `output_semantics="quantile_summary"`;
- `mean=None`; and
- `dispersion=None`.

The explicit `None` values are intentional. The median is not relabeled as a
conditional mean, and quantile spread is not relabeled as a standard deviation.
This prevents a quantile-only model from appearing to provide a full
distribution when it does not. The forecast contract validator enforces that a
`quantile_summary` cannot also claim non-null mean or dispersion fields.

The MAPIE conformal prediction interval is represented separately by
`ConformalInterval`, with calibrated lower/upper bounds, nominal coverage,
correction values, and calibration sample count.

## 3. Fail-closed cost gate

The research decision gate evaluates the conformal interval's **lower bound**
against the two fixed round-trip cost scenarios independently:

```text
ENTER(c)   iff valid_conformal_lower_bound > c / 10_000
ABSTAIN    otherwise
```

- 50 bps corresponds to a threshold of 0.005.
- 90 bps corresponds to a threshold of 0.009.
- The comparison is strict; equality does not pass.
- Passing the 50 bps case does not imply passing the 90 bps case.
- Missing or invalid intervals, malformed bounds, and unsupported cost inputs
  fail closed to `ABSTAIN`.

`ENTER` is only a research-gate label. It does not authorize an order,
execution, paper trading, live trading, or Champion selection.

## 4. Synthetic-only verification

The G2 unit tests use deterministic synthetic arrays generated in memory.
The adapter's test fit and separate CQR conformalization use only those
synthetic arrays. The tests cover:

- HGB quantile fitting and ordered `DistributionForecast` output;
- explicit quantile-summary semantics with `mean=None` and
  `dispersion=None`;
- MAPIE CQR prediction interval shape and interval-contract validation;
- fail-closed behavior before fitting and for malformed/non-finite inputs;
- quantile-crossing and malformed MAPIE-output rejection;
- `ENTER` for lower bounds that exceed the applicable 50/90 bps costs;
- `ABSTAIN` for negative lower bounds, equality, invalid intervals, and
  higher-cost cases that are not cleared.

The successful Standard CI run enforces the repository's existing coverage
threshold (at least 90%) for the configured test scope. The Quality Preflight
also runs Ruff, strict Mypy, Bandit, Xenon, pip-audit, and the synthetic test
suite. No quality threshold was weakened to achieve these results.

## 5. Empirical-data and execution boundary

No model in G2 has been fitted on real market data, whether pre-hack or
post-hack. No Holdout data has been read. G2 performs no economic backtest,
parameter optimization, live inference, private API call, order dispatch, or
trading execution.

CI's success is evidence of API integration, code quality, and synthetic test
behavior only. It is not evidence of forecast skill, empirical conformal
coverage, profitability, drawdown, or any other economic performance measure.
No economic metric is claimed by this ADR.

## 6. Verification record

The implementation at code SHA
`d76b92d7e33bc27fc5907bef6bf4fcf9de1bdb86` passed:

- **Standard CI #2456:** run ID `44030835159`, conclusion `success`.
- **Quality Preflight #51:** run ID `44030836722`, conclusion `success`.

Both runs checked the same implementation SHA. This ADR file is a
documentation-only addition to that already-verified implementation; the
result for the final documentation commit must be confirmed by its own CI run.

## 7. Preserved governance

The following boundaries remain unchanged:

- `Champion=null`;
- `order_permission=false`;
- `trading_authorization=NO-GO`;
- Holdout remains sealed;
- no real-data fitting or evaluation is authorized by G2; and
- `main` remains frozen.

**G2 outcome:** HGBQ-CQR API adapter and synthetic fail-closed tests are
implemented. This ADR records the technical contract; it does not grant
authorization for G3 or any empirical execution.

Architecture First. Evidence Always. Fail Closed.
