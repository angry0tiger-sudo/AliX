# ADR-048 — Regime R: انتخاب NGBoost + CQR و قرارداد R1/R2

- **Status:** Owner-selected research direction; R1/R2 contracts and mock-only scaffolding approved; empirical execution NOT AUTHORIZED.
- **Date:** 2026-11-16
- **Branch baseline:** `research/nobitex-hf-strategy`
- **Parent baseline before this change:** `32cdd57b7a31f113110eff443dd204120c2163b4`
- **Related:** ADR-012, ADR-036, ADR-045, ADR-046, ADR-047
- **Owner directive:** NGBoost conditional-distribution forecasting + Conformalized Quantile Regression (CQR), with 1h/4h/24h horizons and a maximum budget of 30 candidates.
- **Execution boundary:** no real-data read, fit, tuning, inference, backtest, holdout access, dispatch, nonce, private API call or order is authorized by this ADR.

## 1. Decision

The Owner selects **NGBoost + CQR** as the first Regime R model family.

The research question is not whether a point forecast predicts the next close. It is:

> Can a causally conditioned predictive distribution for executable forward returns produce calibrated lower prediction bounds that remain strictly above round-trip trading costs, while supporting the required trade frequency and drawdown constraints across chronological folds?

NGBoost supplies a conditional predictive distribution. CQR conformalizes lower/upper quantile estimates on a chronologically separate calibration slice. The decision policy is outside the forecaster and uses the calibrated lower bound.

Current program state:

```text
PROJECT_STATUS = PIVOTING_TO_NEW_MODELS
SELECTED_FAMILY = NGBOOST_PLUS_CQR
PHASE_STATUS = R1_R2_CONTRACTS_AND_MOCKS_ONLY
VALIDATION_STATUS =
  NOT_VALIDATABLE_UNDER_CURRENT_GOVERNANCE / INSUFFICIENT_EVIDENCE
GLOBAL_N = UNKNOWN
GLOBAL_DSR = NOT_COMPUTABLE
Champion = null
order_permission = false
trading_authorization = NO-GO
```

ADR-047 remains the broad Regime R pivot decision. This ADR narrows that approved direction to one selected family. It does not erase or reinterpret Regime P's historical record, Run #10, ADR-045, ADR-046, or historical multiple-testing exposure.

## 2. Why probabilistic forecasting, not another point model

The prior workflow already tested technical OHLCV teachers, symbolic rule search, risk allocation and zero-shot foundation forecasts. Repeating their model classes is expressly out of scope.

The new hypothesis differs in three ways:

1. **Distributional target:** the output represents a conditional distribution of future executable return, not only a class or conditional mean.
2. **Uncertainty-aware action:** CQR supplies an empirically calibrated prediction interval. The decision policy is allowed to abstain; a high mean with a weak lower bound is not a trade.
3. **Cost-aware lower-bound gate:** separate 50 bps and 90 bps ledgers accept a candidate signal only where the corresponding calibrated gross-return lower bound strictly exceeds that scenario's round-trip cost.

CQR coverage is marginal under its exchangeability assumptions; it does not magically guarantee conditional coverage for serially dependent, non-stationary financial returns. The project must report temporal coverage, interval width, coverage drift and fold-level failure. No unconditional "90% conditional coverage" claim is permitted without evidence.

## 3. Data, target and feature contract

### 3.1 Decision and execution timing

At the close of candle index `t`, features are computed using data available no later than `t`. The decision is made after that close.

- Entry reference: open of candle `t+1`.
- Exit reference for horizon `h`: close of candle `t+h`.
- Horizons: `h ∈ {1,4,24}` hourly candles.
- Gross simple return target:

```text
gross_return(t,h) = close[t+h] / open[t+1] - 1
```

The target is mature only after candle `t+h` closes. Its label-end index is therefore `t+h`. Incomplete future windows remain unlabeled; they must never be changed to zero-return examples.

This target is explicitly a gross executable-reference return. Trading costs are applied separately in the policy/simulator to avoid subtracting costs twice.

### 3.2 Causal feature schema

Initial fixed feature schema:

- `log_return_1`, `log_return_4`, `log_return_24`
- `realized_volatility_6`, `realized_volatility_24`
- `range_fraction_1`, `atr_fraction_14`
- `log_volume_change_1`, `volume_zscore_24`
- `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`

All rolling features are right-aligned and computed on closed candles. Time encodings are derived from the decision candle timestamp. No backfill, centered window, future normalization, future-derived scaler, incomplete candle, or label-derived feature is permitted. Feature values and their as-of indices must be persisted and validated against decision index `t`.

Any feature addition, removal, window change, normalization change, or timestamp semantic change is a new candidate definition and must be registered before evaluation.

### 3.3 Data independence

- The previously used 31,801-row pre-hack dataset is historically exposed. It may support code/contract development or explicitly labeled exploratory analysis only; it is not a clean confirmatory holdout.
- The post-hack holdout remains sealed, non-persisted in the prior commitment, and outside this authorization.
- Any future confirmatory dataset needs a separate data source/window decision, immutable raw commitment and independent owner authorization. This ADR does not authorize acquisition or use.

## 4. NGBoost and MAPIE/CQR integration boundary

### 4.1 NGBoost role

NGBoost must emit a predictive distribution from which the configured lower, median and upper quantiles can be obtained. The first implementation should expose distribution diagnostics such as mean, dispersion, log score/negative log-likelihood and quantile monotonicity. Distribution family and all learner/booster settings are candidate-defining configuration.

No point-prediction-only substitute is acceptable.

### 4.2 CQR role and adapter requirement

MAPIE's documented `ConformalizedQuantileRegressor` expects quantile-regression estimators. In its prefit mode it expects three fitted estimators in lower, upper and median quantile order. NGBoost instead exposes one predictive distribution. The integration therefore requires a thin, frozen scikit-learn-compatible adapter layer that exposes the selected NGBoost distribution's quantile function as lower/median/upper prediction estimators, or an explicitly reviewed equivalent implementation.

The actual integration is NOT implemented or imported in this mock-only change. Its MAPIE API contract, serialization, estimator checks and CQR conformity-score behavior must be verified in a GitHub Actions compatibility job before any model fit.

Do not pass `method="cqr"` to MAPIE's `TimeSeriesRegressor`: its documented time-series methods are ENBPI and ACI, while CQR is exposed by the dedicated `ConformalizedQuantileRegressor` API. A chronological train/calibration/test design is required. Block/rolling calibration and coverage drift diagnostics must be preregistered for temporal dependence.

### 4.3 Decision rule

For each cost scenario `c ∈ {50,90}` bps:

```text
TAKE_RESEARCH_SIGNAL(c) iff calibrated_lower_bound_gross > c / 10_000
```

The inequality is strict. Equality does not pass. The 50 bps and 90 bps scenarios use separate costed equity ledgers and separate economic metrics. A signal passing only 50 bps is not a pass at 90 bps.

The lower bound is the post-conformalization lower endpoint, not the raw NGBoost quantile. CQR calibration correction, confidence level, lower/upper quantile levels, calibration-window policy and all associated parameters must be frozen before the first fit. Their exact numeric choices remain preregistration blockers until recorded; this ADR does not invent them.

## 5. Trial family and candidate budget

- Family ID: `NOBITEX-R1-NGBOOST`
- Maximum candidate budget: `N_budget = 30`
- Horizons: 1h, 4h, 24h.
- Slots per horizon: exactly 10.
- All 30 slots form one selection family for any cross-horizon winner selection. Horizon subgroups do not erase cross-horizon multiplicity.
- A reserved ID is not automatically an evaluated economic trial. The Registry must distinguish NOT_STARTED, FIT_STARTED, CANDIDATE_EVALUATED, PARTIAL, FAILED, CANCELLED and COMPLETED, with a separate execution attempt ID.
- Every candidate's full config, model/distribution settings, feature/target hashes, fold plan, CQR/calibration settings, cost/fill contract and seed must be frozen and hashed before fitting.
- Any tuning outside these 30 slots, additional model family, post-result threshold variant or changed horizon is a new preregistration and a new budget decision.

The exact 30 canonical IDs are recorded in `docs/research/preregistration/TEMPLATE_NGBOOST_CQR.md`.

## 6. Economic and statistical gates

The Owner-mandated minimum gates apply separately to each horizon and each cost scenario:

1. `NetPF >= 1.10` at 50 bps and at 90 bps.
2. At least 100 completed, costed trades.
3. `MaxDD <= 15%`.
4. At least three chronological walk-forward folds with fold-level results retained; no pooled metric may hide a failed or unsupported fold.
5. Lower-bound decision counts, abstentions, signal attrition and interval coverage/width must be reported separately.

NetPF, trade count and MaxDD must be computed from the actual costed ledger, not from labels or forecast accuracy. No division by zero or undefined PF may be silently mapped to a passing value.

DSR is a separate evidence gate. The new family can report a family-scoped DSR only if every evaluated candidate has a comparable net-return series/Sharpe and required moments, actual candidate identity and multiplicity are reconciled, and dependence treatment is documented. The conservative `N_budget=30` sensitivity must also be shown. This does not repair or replace historical `GLOBAL_N=UNKNOWN` or `GLOBAL_DSR=NOT_COMPUTABLE`.

## 7. Preregistration blockers before R3

The template reserves trial identities and the initial contract, but is **not executable** until all items below are filled, hashed and owner-approved:

- Exact immutable configuration for all 30 candidate IDs: distribution family, NGBoost base learner, estimator count, learning rate, minibatch/column settings, regularization and seed.
- Exact quantile levels and nominal interval coverage.
- CQR estimator-adapter API, conformity score and finite-sample correction.
- Chronological calibration window length, rolling/block policy, minimum calibration sample and recalibration cadence.
- At least three exact UTC/row-index folds per horizon, with horizon-aware label purge and embargo. No shuffled split.
- Dataset SHA/window and exposure declaration; feature and target schema fingerprints.
- Cost/fill contract for 50 and 90 bps, including fee/slippage/spread interpretation and next-open/exit pricing.
- Position sizing, one-position/no-overlap policy, skipped-signal accounting and terminal liquidation.
- Baseline/persistence benchmark, uncertainty diagnostics, economic acceptance and stop rules.
- Candidate Sharpe/DSR input schema and dependence sensitivity.
- Owner approval of the complete immutable preregistration before the first fit.

The candidate configuration matrix and calibration parameters are deliberately not guessed. Until they are complete, no training or real-data evaluation may start.

## 8. Dependency and supply-chain assessment

Repository baseline at `32cdd57b7a31f113110eff443dd204120c2163b4` uses Python 3.12, `scikit-learn==1.9.0`, `numpy==2.5.4`, a pip-tools-generated `requirements.lock`, and a CI dependency audit.

- NGBoost and MAPIE are absent from `requirements.txt`, `requirements.lock` and the current CI install command.
- Current public PyPI metadata reports NGBoost 0.6.0 (2026-11-07), but the published wheel/source files have no PyPI provenance attestation. Exact compatibility with AliX's pinned sklearn/numpy stack has not been verified.
- Current public PyPI metadata reports MAPIE 1.6.0 (2026-10-19), with provenance on the release files; its documented minimums include Python 3.10+, NumPy 1.23+ and scikit-learn 1.4+.
- Neither dependency is added in R1/R2. Mock contracts intentionally use only the Python standard library. Avoid adding unverified packages to the main dependency lock.
- Before R3, use GitHub Actions only to verify candidate versions, transitive resolution, license, provenance, pip-audit, NGBoost/sklearn API compatibility, MAPIE CQR prefit-adapter compatibility, and deterministic serialization. If NGBoost 0.6.0 fails any compatibility or supply-chain gate, stop and bring an alternative pinned version/implementation back for Owner approval. Do not silently downgrade the existing sklearn stack.

## 9. Mock-only implementation scope

The only Python files added in this change are isolated contract helpers and unit tests under `research/regime_r/ngboost_cqr/` and `tests/unit/`. They validate:

- 30 canonical trial IDs and horizon allocation.
- Causal feature snapshot completeness, finite values and as-of index.
- Target horizon/index alignment and positive executable reference prices.
- Conditional distribution quantile ordering and finite dispersion.
- CQR interval correction/coverage contract.
- Strict 50/90 bps lower-bound gates.

They do not import NGBoost or MAPIE, access real candles, make network calls, fit any model, or compute economic results.

The standard CI quality gates are extended additively to include the new namespace in compile, Ruff, Mypy, Bandit, Xenon and mock test coverage. No quality gate is removed or relaxed.

## 10. Safety invariants

The following remain unchanged:

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `private_api_calls=0`
- `real_order_attempts=0`
- `ALIX_DRY_RUN=true`
- `ALIX_ALLOW_LIVE_TRADING=false`
- `authorization_status=NOT_AUTHORIZED`
- `EXECUTION_TRIGGER=PROBE`
- `ACQUISITION_TRIGGER=PROBE`
- `main=FROZEN`
- Holdout sealed / locked.

This ADR grants no access to pre-hack market data in this phase and no access to post-hack data. No training, tuning, model inference, economic backtest, private API, dispatch, nonce, paper order or live order is authorized.

**Owner decision recorded:** NGBoost + CQR is the selected family for R1/R2 design. **Next owner gate:** approve the fully completed R3 preregistration and its data/implementation authorization separately.

Architecture First. Evidence Always. Fail Closed.
