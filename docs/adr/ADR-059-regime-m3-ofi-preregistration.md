# ADR-059 — Regime M3 OFI Preregistration and Baseline Lock

- **Status:** PREREGISTERED_LOCKED — parameter/protocol lock only; implementation and empirical execution **NOT AUTHORIZED**
- **Date:** 2026-11-26
- **Branch:** `research/nobitex-hf-strategy`
- **Prere-registration ID:** `ALIX-M3-OFI-G1`
- **Configuration:** `config/regime_m3/ofi_preregistration.json`
- **Configuration SHA-256:** `c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4`
- **M1 evidence:** ADR-058 §11; M1.6 accepted five-segment public BTCIRT corpus
- **Scope:** OFI prediction study at 1h and 4h only; no 24h target

## 1. Decision and authorization boundary

M1.6 was accepted in ADR-058 §11 following independent verification of the five public capture artifacts, 105,020.393 seconds of actual Manifest capture time, zero `non_monotonic_publisher_offset` rows, successful OffsetTracker baseline recovery, explicit retention of no-replay feed gaps, and passing fail-closed invariants.

This ADR registers exactly two logical OFI trials before any M3 implementation or outcome inspection:

1. `NOBITEX-M3-OFI-1h-LOGISTIC-BASELINE-01`
2. `NOBITEX-M3-OFI-4h-LOGISTIC-BASELINE-01`

The JSON file is the canonical, byte-committed source of the hypothesis, feature definitions, labels, model parameters, temporal folds, cost treatment, simulated position sizing, reporting requirements, and acceptance gates. The digest above includes the final newline. Any byte-level change requires a new superseding preregistration and explicit Owner authorization; silent edits and post-outcome tuning are prohibited.

**This ADR authorizes preregistration only.** It does not authorize writing feature/model code, fitting/training, reading/evaluating the capture data, economic backtesting, external data acquisition, Holdout/Post-Hack access, paper trading, real orders, or trading. A separate Owner directive is required before any M3.5 implementation or empirical execution.

## 2. Registered hypothesis and horizons

The locked hypothesis is:

> عدم تعادل جریان سفارشات (OFI) در سطوح اول Order Book، قدرت پیش‌بینی‌کننده برای بازده کوتاه‌مدت (1h) و میان‌مدت (4h) دارد.

The two fixed targets are the sign of the forward mid-price log return over **1 hour** and **4 hours**. A positive forward return is class 1; zero or negative is class 0. The 24-hour horizon is explicitly excluded from G1. No alternate label, neutral band, barrier target, or post-hoc horizon is permitted.

## 3. Dataset boundary and coverage limitations

The only registered empirical source is the five independently audited M1.6 public BTCIRT capture artifacts already listed in the JSON manifest:

| Segment | Workflow Run ID | Artifact ID | ZIP SHA-256 |
|---:|---:|---:|---|
| 1 | `45553679888` | `15016992581` | `aafe15ed9da1d6f6e8badf905627665695acfd605e926b242856f43888980ce8` |
| 2 | `45620061415` | `15054886094` | `c25c202e5d556912bc40e6a9fdcf78adb5a23db32f3f760ad3a1024d79cf8961` |
| 3 | `45683066882` | `15077454919` | `c7b7ec592cda5fec117976a422d2257a908c2b260d89b4b9b89ef87ab4f20a62` |
| 4 | `45735110603` | `15095790023` | `ec4befa0ad6fc96cf1874cec4ba15b2a2dc5ca27e58adf82c33eb62f18393332` |
| 5 | `45775145557` | `15121226078` | `c52cb93bfc345e1ba66104f7f0e4f8ea3664687e87d3dd89e44ea9236279db88` |

The accepted manifests contain **29 hours, 10 minutes, 20.393 seconds** of actual capture time and **2 hours, 37 minutes, 18.216 seconds** of uncovered inter-segment time. The dataset is not continuous. Every WebSocket disconnect and every inter-segment boundary remains a hard sample boundary. No feature or target may be computed by stitching, replaying, interpolating, or carrying state across these boundaries. The 90 bps cost assumption below is not an actual fee-tier statement.

No Holdout, Post-Hack, PRE-HACK legacy, private/account data, or unaudited external source is included. No additional data may be introduced under G1.

## 4. Locked feature contract

Exactly eight numeric features are registered, computed on a one-minute UTC decision grid from a trailing five-minute information window:

- **Best-Level OFI:** event-by-event change in best bid/ask displayed quantities and prices, using the Cont–Kukanov–Stoikov event-flow sign convention. Normalized event contributions are summed over five minutes.
- **Multi-Level OFI:** the same event-flow calculation, independently for rank levels 2, 3, 4, and 5. Each level is a separate fixed input column; missing levels invalidate the sample rather than being imputed.
- **Microprice deviation:** top-of-book volume-weighted microprice minus midpoint, expressed in basis points.
- **Signed trade-volume ratio:** five-minute `buy` volume minus `sell` volume divided by total volume using the public API `type` field. This remains an API-type proxy, not an aggressor-side claim. Semantic validation is a mandatory M3.5 preflight; a failure blocks the experiment and cannot be repaired by silently dropping or remapping this registered feature.
- **Top-five Order Book Imbalance:** total bid quantity minus total ask quantity across levels 1–5, divided by total displayed quantity.

The feature equations, event aggregation, depth normalization, timestamp basis, missing-value rules, and exact column names are frozen in the JSON. Only valid WebSocket book/trade events produce flow deltas; REST snapshots can initialize state but are not counted as order-flow events. A quiet period in a live, uninterrupted socket session is not a gap; explicit disconnects and segment boundaries are.

## 5. Target, model, and fixed prediction rule

For each horizon (h \in \{1,4\}):

[
r_{t,h}=10{,}000\,\ln\left(\frac{mid_{t+h}}{mid_t}\right)
]

The binary label is 1 only when (r_{t,h}>0); otherwise it is 0. A label must fully mature inside the same uninterrupted capture segment and must not cross a disconnect, gap, or segment boundary.

One interpretable baseline is frozen:

- `StandardScaler` fitted on training rows only.
- `sklearn.linear_model.LogisticRegression`, scikit-learn 1.9.0.
- L2 penalty, `C=1.0`, `solver=lbfgs`, `class_weight=null`, `max_iter=1000`, `tol=0.0001`, `random_state=42`.
- A single fixed long-only research threshold: `P(up) >= 0.5`; otherwise abstain/hold cash.
- No Random Forest, model family alternatives, calibration, tuning, feature selection, threshold sweep, or additional candidate is registered.

## 6. Fixed temporal evaluation and economic protocol

Four fixed expanding walk-forward folds use the five capture segments in strict chronological order: WF-01 trains on segment 1 and tests on segment 2; WF-02 trains on 1–2 and tests on 3; WF-03 trains on 1–3 and tests on 4; WF-04 trains on 1–4 and tests on 5. Each test fold is exactly the next single capture segment. Training labels must be fully mature before the relevant test segment begins. The 1h/4h target intervals and trailing feature windows must remain entirely inside their source segment and uninterrupted connection interval.

Each accepted one-minute signal opens a fixed long-only research cohort held exactly for the registered horizon. Each 1h cohort uses 1/60 of initial equity; each 4h cohort uses 1/240. This keeps maximum concurrent gross exposure at 1.0x with no leverage. Every cohort is marked to mid-price and closed at its fixed horizon; incomplete, gap-crossing cohorts are excluded and counted.

A single fixed **90 bps all-in round-trip** friction assumption is subtracted once from each gross mid-to-mid trade return. It is a conservative research assumption, not the owner's actual fee tier; no second spread/slippage charge is added. This is a research simulation, not a Freqtrade execution test.

Training preflight requires at least 100 eligible training rows per horizon and at least 20 observations from each class. A test fold must have at least 20 completed OOS trades to count toward fold stability. Insufficient folds remain `INSUFFICIENT_SAMPLE`; boundaries, features, thresholds, horizons, and minimums cannot be relaxed after inspection.

## 7. Frozen acceptance gates

The gates apply **separately to each horizon**. Both 1h and 4h must pass:

| Gate | Locked threshold |
|---|---:|
| Pooled OOS Net Profit Factor | **>= 1.10** |
| Completed OOS trades | **>= 100** |
| Absolute maximum drawdown | **<= 15%** |
| Stable temporal folds | **at least 3 of the 4 fixed folds** |

A fold counts as stable only when it has at least 20 completed OOS trades, positive net PnL, and fold NetPF greater than 1.00. All four folds must be reported. Pooled NetPF is the sum of positive net trade PnL divided by the absolute sum of negative net trade PnL; an undefined denominator does not pass automatically. Maximum drawdown is measured on the chronological minute-marked portfolio NAV, including open cohort mark-to-market PnL and cost.

Both horizons must pass independently. A pass on one horizon cannot compensate for a failure or insufficient sample on the other. Failed gates produce `NO-GO`; insufficient samples produce `INSUFFICIENT_SAMPLE / NO-GO`. Always-long and no-trade baselines, training-fold majority predictions, and both predictive and economic metrics must be reported, but they do not replace or alter the registered gates.

## 8. M3.5 preflight and prohibition on implementation before approval

Before any implementation or fit, a separately authorized M3.5 task must first validate the configuration hash, exact five-artifact inventory, data/time continuity masks, all eight feature definitions, public trade-type semantics, label maturity, per-fold train counts, class balance, and attainable trade sample. Any failed preflight stops before model fitting and reports the exact blocker. No feature or fold may be changed to rescue sample size.

This registration is not permission to start M3.5. The next step remains blocked until the Owner issues a separate explicit approval.

## 9. Safety invariants

Throughout G1 and any later separately authorized M3.5 work:

- Holdout and Post-Hack contents remain sealed and unread.
- No private/account endpoint, credentials, or unauthenticated external data collection is permitted by this registration.
- No model is trained under this preregistration-only task.
- No paper or real orders are sent or attempted.
- `Champion=null`; `order_permission=false`; `trading_authorization=NO-GO`.

The repository `main` branch remains frozen. Only `research/nobitex-hf-strategy` may receive this preregistration change.

## 10. References

1. Rama Cont, Arseniy Kukanov, Sasha Stoikov, “The Price Impact of Order Book Events,” *Journal of Financial Econometrics* 12(1), 47–88 (2014), [DOI: 10.1093/jjfinec/nbt003](https://doi.org/10.1093/jjfinec/nbt003), [arXiv:1011.6402](https://arxiv.org/abs/1011.6402). Basis for the best-level OFI event sign convention.
2. Ke Xu, Martin D. Gould, Sam D. Howison, “Multi-Level Order-Flow Imbalance in a Limit Order Book,” *Market Microstructure and Liquidity* 4(3–4) (2020), [arXiv:1907.06230](https://arxiv.org/abs/1907.06230). Basis for a vector of level-specific OFI terms.
3. Nobitex official public [WebSocket documentation](https://github.com/nobitex/docs-api/blob/master/source/includes/_websocket.md); field contract and semantic caveats are also recorded in ADR-057 §3–§6.

**G1 outcome:** hypothesis, eight features, two horizons, one interpretable estimator, four fixed temporal folds, costs, portfolio construction, acceptance gates, trial IDs, and safety limits are preregistered and locked. No empirical execution or M3.5 implementation is authorized.

Architecture First. Evidence Always. Fail Closed.
