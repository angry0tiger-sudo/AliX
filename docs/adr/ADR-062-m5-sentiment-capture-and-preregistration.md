# ADR-062 — M5.1 Alternative.me Sentiment Capture Completion and M5.2 Registration

- **Status:** **M5.1 COMPLETE — M5.2 PREREGISTERED / EXECUTION NOT AUTHORIZED**
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Pre-change HEAD verified:** `3f5b1c2073e2264b6ac1d6a1a887dc171ba883d9`
- **Target:** Nobitex Spot / USDT — `BTCUSDT` only

## 1. Decision

M5.1 Alternative.me Fear & Greed capture is complete. The existing GitHub Actions workflow succeeded, and an immutable raw/normalized snapshot plus its manifest are present in the research branch. No repeat capture is required.

M5.2 is now prospectively registered as a four-candidate paired comparison: OHLCV-only Logistic Regression control and the same Logistic Regression with one additional Fear & Greed context feature, each at 1h and 4h. This ADR and the preregistration file are governance artifacts only. **No model fitting, inference, backtest, tuning, new data acquisition, Holdout/Post-Hack access, paper order or real order is authorized.**

## 2. M5.1 capture evidence

- Workflow: [M5.1 Alternative.me Sentiment Capture](https://github.com/angry0tiger-sudo/AliX/actions/workflows/m5-sentiment-capture.yml)
- Latest capture workflow on this phase: **Run #6 — completed successfully**, source commit `479eb1ad408778028d240714010ef59fc397ddc1`.
- Immutable snapshot persistence commit: `b982425141ac8527150c12bac8ce3acdc0bafe1b`.
- Snapshot directory: `data/sentiment/alternative_me/fng/`.
- Provider endpoint: `https://api.alternative.me/fng/?limit=0`.
- Snapshot fetched at: `2026-11-27T09:27:26.548110Z`.
- Observation coverage: `2018-02-01T00:00:00Z` through `2026-11-27T00:00:00Z`.
- Record count: **3,218 daily observations**.

The three persisted artifacts are:

| Artifact | Repository blob SHA |
|---|---|
| `alternative_me_fng_20261127T092726548110Z_4adba36f7dd9.raw.json` | `d635e9f6ec3abfc058791675979b0915ac8b7d0b` |
| `alternative_me_fng_20261127T092726548110Z_4adba36f7dd9.normalized.json` | `1bddb3b6416595dd14ef27ba935ee03248847d19` |
| `alternative_me_fng_20261127T092726548110Z_4adba36f7dd9.manifest.json` | `1fd542bea336c93f05c61a4c14bf14aa10885ba0` |

Manifest content hashes:

- Raw payload SHA-256: `4adba36f7dd95792520e078f7578c316c23e80a7b8a0ace81df490bf06c740df`.
- Normalized payload SHA-256: `e0578509cf3f8a6eeac39ddbd2edeab876e47027d759558e120c6f8b3a9f7f61`.
- Schema: `alix.m5.alternative-me-fng.v1`.
- Declared target context: `global_btc_sentiment_not_nobitex_pair_specific`.

The successful Workflow Run #6 is the capture execution evidence; the immutable files are separately present in the branch at the pre-change HEAD. The Standard CI on that exact pre-change HEAD is also green: [AliX Continuous Integration #2601](https://github.com/angry0tiger-sudo/AliX/actions/runs/45974829731), `completed / success`, head `3f5b1c2073e2264b6ac1d6a1a887dc171ba883d9`.

## 3. Data quality and Point-in-Time interpretation

Alternative.me does **not** provide a historical provider-measured publication timestamp. The manifest explicitly records:

- `provider_supplies_historical_available_at = false`;
- `availability_policy = observation_timestamp_plus_24h_conservative_proxy`;
- the `available_at` value is an observation timestamp plus 24 hours, **not** a verified actual publication time.

This conservative proxy and the existing alignment contract remain mandatory. At each completed BTCUSDT decision candle, only the latest sentiment record with `available_at <= decision_time` may be used. Forward-fill is allowed only after that availability time. No backward-fill, interpolation, zero-fill or use of a future observation is permitted. Rows with no prior available record are excluded from both paired model arms.

The feature is a slow-moving, global BTC context variable, not a Nobitex BTCUSDT-specific market measure. Repeating one daily value across hourly decisions does not create independent hourly sentiment observations. All future reports must retain source attribution, report source age and unique daily observations, and disclose overlap/dependence.

The existing scripts `scripts/fetch_m5_sentiment_data.py` and `scripts/align_m5_sentiment.py` were inspected and are retained unchanged. The aligner accepts only 1h/4h, checks timezone-aware and strictly increasing decision timestamps, selects the latest `available_at <= decision_time`, and reports a missing match rather than backfilling. Its documented contract also states that it does not load or merge OHLCV or run a model.

## 4. M5.2 prospective registration

The locked configuration is:

- File: `config/m5_2/sentiment_model_preregistration.json`.
- Preregistration ID: `ALIX-M5-2-SENTIMENT-G1`.
- Revision: `M5.2-SENTIMENT-G1`.
- SHA-256: `851d6c4b39c41cacec8ccd712cf17bb40dd42dd45f2bb8813920a7f2ebe481ba`.
- Trial Registry: four logical candidates, zero evaluated at registration.

The paired design is deliberately explicit:

1. OHLCV-only control: eight fixed OHLCV features.
2. Sentiment treatment: the same eight OHLCV features plus one raw `fear_greed_value` feature.
3. Both arms use one fixed `StandardScaler + LogisticRegression` pipeline, the same paired rows, the same label definition, four chronological expanding walk-forward folds and identical execution assumptions.
4. Horizons are 1h and 4h only. At `close(t)`, entry reference is `open(t+1)`; exit reference is `close(t+h)`.
5. No tuning, probability threshold search, alternate estimator, feature ablation or post-outcome change is included.

The eight OHLCV feature names are frozen in the JSON as `log_return_1`, `log_return_4`, `log_return_24`, `realized_volatility_6`, `realized_volatility_24`, `range_fraction_1`, `atr_fraction_14` and `log_volume_change_1`. The treatment adds only `fear_greed_value). The JSON records exact formulas, model parameters, data-availability semantics, split/purge rules, paired comparison and all acceptance conditions.

Before any future fit, a separate Nobitex BTCUSDT 1h OHLCV manifest and immutable digest must be pinned. This preregistration does not acquire or read that market dataset.

## 5. Frozen economic gates

The gates are applied independently for each arm, horizon and cost scenario:

- Cost-50 and Cost-90 are both mandatory; each is evaluated separately.
- Pooled OOS Net Profit Factor must be **at least 1.10**.
- At least **100 completed costed OOS trades** per registered arm/horizon.
- Absolute maximum drawdown must be **no more than 15%**.
- Four chronological folds are fixed; at least **3 of 4** folds must each contain at least 20 completed OOS trades, positive net PnL and fold NetPF greater than 1.00.
- Every fold and every failed/insufficient result must be reported. No failed fold may be hidden or replaced.
- A sentiment benefit is established only if the treatment also has strictly higher paired pooled NetPF and strictly higher paired pooled net PnL than the OHLCV-only control for both horizons under both cost scenarios.
- DSR may be reported only when its full inputs and multiplicity history are available. If not computable, retain `GLOBAL_N=UNKNOWN` and `GLOBAL_DSR=NOT_COMPUTABLE`; the result cannot be promoted to GO on an unavailable DSR.

The four logical candidates, their IDs and their locked status are appended to `docs/research/TRIAL_REGISTRY.md`. Fold fits and cost replays do not increase the candidate count.

## 6. Provenance and non-transfer of M3

M3 was closed as `CLOSED_NO_GO` on its registered BTCIRT OFI dataset. Those observations and results are not BTCUSDT data and are not reused in M5.2. The M5.2 target market is Nobitex Spot BTCUSDT only; no BTCIRT conversion, proxy or substitution is allowed.

## 7. Safety and execution boundary

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `holdout_bytes_read = false`
- `posthack_data_read = false`
- `new_sentiment_data_acquired_in_this_change = false`
- `local_execution = false`
- `model_training_authorized = false`
- `economic_backtest_authorized = false`

**Disposition:** M5.1 is complete. M5.2 is preregistered and locked. Stop after the documentation/configuration commit and successful Standard CI. No model or backtest workflow may be dispatched until a separate explicit Owner authorization is received.
