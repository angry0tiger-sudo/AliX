# ADR-065 — MRD v1.2 Status Reconciliation and Next-Phase Strategy Proposal

- **Status:** **DOCUMENTATION COMPLETE — OWNER DECISION PENDING — NO EXECUTION AUTHORIZED**
- **Date:** 2026-11-27
- **Repository:** `angry0tiger-sudo/AliX`
- **Branch:** `research/nobitex-hf-strategy`
- **Verified pre-documentation HEAD:** `95133203e70265ac96f7d4dd7c14ed213fadaf38`
- **Standard CI on verified pre-documentation HEAD:** [Run #46001067175](https://github.com/angry0tiger-sudo/AliX/actions/runs/46001067175) — `completed / success`
- **MRD v1.2 deliverable SHA-256:** `6b4f26245b4f0eaf8cc8dc01a7ba7013beb815975f1209abfa82a5ecacc8178a` (delivered as a separate DOCX artifact)
- **Supersedes:** the stale operational snapshot in MRD v1.1 dated 2026-11-16; it does not erase historical ADRs, preregistrations, or experiment outcomes.

## 1. Decision and scope

This ADR reconciles the project's current state and presents two possible directions for a future phase. It is a **documentation and strategy-proposal decision only**.

The current economic dispositions are:

- **HGBQ-CQR / Regime P:** `CLOSED_NO_GO` under ADR-055.
- **M3 OFI / Regime M:** both registered BTCIRT OFI trials are `CLOSED_NO_GO` under the Owner-approved closure in ADR-060.
- **M5.2 Fear & Greed sentiment:** all four BTCUSDT logical trials are `CLOSED_NO_GO` under ADR-064.
- **Regime T Triple-Barrier + Logistic Regression:** the existing experiment is `CLOSED_NO_GO` under ADR-056; its registered pooled OOF accuracy did not meet the owner's strict `>55%` screening threshold.

No currently evaluated model or strategy is eligible for promotion. There is no Champion and no Paper/Live permission.

**No option is selected by this ADR.** Option A is only a conditional design proposal because the exact broad idea of Triple-Barrier plus Logistic Regression has already been tried. Option B remains deferred pending source coverage, point-in-time, licensing, and total-cost qualification. Any next work requires a new Owner decision and a prospective, hash-locked preregistration.

This ADR does not authorize model implementation, feature engineering, fitting, training, prediction, backtesting, tuning, threshold sweeps, market-data acquisition, paid subscriptions, API-key creation, credit purchase, or Holdout/Post-Hack access.

## 2. Current repository and governance state

### 2.1 Repository snapshot

| Item | Verified value |
|---|---|
| Research branch | `research/nobitex-hf-strategy` |
| Pre-documentation Research HEAD | `95133203e70265ac96f7d4dd7c14ed213fadaf38` |
| Standard CI | Run `46001067175`, `SUCCESS` on the exact pre-documentation HEAD |
| Standard CI tests | 1,127 passed, 5 skipped |
| Standard CI total coverage | 90.36% (configured floor 90%) |
| Main HEAD at review | `a1c1a658906cd79be62796885e3ed782b66a267d` |
| Main tree SHA | `5761c6c34143f8e6924f53a4e745c48932a5f791` |

The `main` tree matches the previously required baseline tree exactly. The current `main` HEAD is not the original pre-exception commit because the earlier Owner-authorized temporary workflow add/delete commits remain in its history. The file contents were removed and the tree was restored. **This ADR makes no changes to `main`.**

### 2.2 Trial and experiment inventory

The following inventory reconciles the four closed paths relevant to the current MRD. It is deliberately not represented as the global DSR trial count.

| Path | Registered/evaluated logical identities | Slot / run accounting | Outcome |
|---|---:|---:|---|
| HGBQ-CQR / Regime P | 3 unique horizon candidates (1h, 4h, 24h) | 30 slots: 3 unique configurations plus 27 duplicate slots; 90 fit calls in the diagnostic lineage | `CLOSED_NO_GO` |
| M3 OFI / Regime M | 2 (1h, 4h) | Two logical trials, four folds per horizon; data was five public BTCIRT capture segments, not BTCUSDT | `CLOSED_NO_GO` |
| Regime T Triple-Barrier | 1 experiment | One fixed Logistic Regression model family across three chronological folds | `CLOSED_NO_GO` |
| M5.2 Sentiment | 4 (Control/Sentiment × 1h/4h) | Four candidates × four folds = 16 fold fits; 101,648 OOS prediction rows | `CLOSED_NO_GO` |
| **Total for these four paths** | **10 unique candidate/experiment identities** | **37 recorded slots/entries**, including 27 duplicate HGBQ slots | **10/10 rejected** |

Counting rules:

1. The three HGBQ unique candidates are the three horizon configurations. The 27 duplicate registry slots are preserved for lineage and reproducibility but are **not 27 independent candidates**. G4.5 and G4.6 belong to the same HGBQ evaluation/ablation lineage; they are not counted as an additional independent model family.
2. OFI has exactly two registered logical trials. The four folds are repeated temporal evaluations within each logical trial, not extra candidates.
3. Regime T is one registered experiment, not three candidates; the three fold fits are fold-level evidence.
4. M5.2 has four registered logical candidates; 16 fold fits do not increase its logical trial count.
5. The total above is a bounded inventory of these four named paths. It is **not** `GLOBAL_N`, not the complete historical experiment universe, and must not be used as an input to DSR.

`GLOBAL_N=UNKNOWN` and `GLOBAL_DSR=NOT_COMPUTABLE` remain unchanged.

### 2.3 Holdout and execution safety

The following invariants remain mandatory:

- `Holdout = SEALED`; no Holdout bytes have been read for this work.
- `posthack_data_read = false`.
- `Champion = null`.
- `order_permission = false`.
- `trading_authorization = NO-GO`.
- `private_api_calls = 0`; `real_order_attempts = 0`; `paper_orders = 0`.
- No new market data was downloaded or captured for this ADR.
- No model was created, fitted, trained, tuned, or evaluated for this ADR.
- No vendor subscription, trial, API credential, or API credit was activated or purchased.
- The `main` branch was not modified.

The existing PRE-HACK Nobitex BTCUSDT OHLCV file is historically exposed exploratory data, not a sealed confirmatory holdout. Its use in any future exploratory work would still require a new Owner-authorized preregistration and the existing Holdout separation rules.

## 3. Option A — materially new target redesign using existing OHLCV

### 3.1 What is already known

The broad proposal “simple Logistic Regression + Triple-Barrier labels on the existing OHLCV” is **not new**. ADR-056 already executed `REGIME-T-TB-LOGISTIC-01` on the existing 31,801-row PRE-HACK Nobitex BTCUSDT 1h dataset.

The prior locked definition was:

- Decision after the close of bar `t`; features use closed OHLC data through `t` only.
- Entry at `open(t+1)`.
- Wilder ATR(14) divided by `close(t)` as the volatility scale.
- Symmetric horizontal barriers at `±2 × ATR14_fraction` from the next-bar open.
- Vertical/time barrier at 24 future hourly bars (`t+1` through `t+24`).
- Upper-first = class 1; lower-first = class 0.
- Same-bar dual touch = `AMBIGUOUS`; no barrier touch by 24 hours = `TIMEOUT`; incomplete future window = `UNAVAILABLE`.
- `TIMEOUT`, `AMBIGUOUS`, and `UNAVAILABLE` were excluded from binary fitting and accuracy scoring.
- Eleven price/time features, no volume predictor; one fixed `StandardScaler + LogisticRegression` pipeline; one fit per each of three fixed chronological folds.

Observed result:

| Metric | Result |
|---|---:|
| Input rows | 31,801 |
| LOWER_FIRST | 12,405 |
| UPPER_FIRST | 12,335 |
| TIMEOUT | 6,970 |
| AMBIGUOUS | 54 |
| UNAVAILABLE | 37 |
| Binary events | 24,740 |
| Pooled OOF predictions | 1,652 |
| Pooled OOF accuracy | **51.09%** |
| Pooled balanced accuracy | **51.57%** |
| Owner screening gate | **Strictly greater than 55%** |
| Fold accuracies | 49.33%, 60.57%, 43.67% |
| Verdict | **NO-GO** |

The experiment did not include a costed strategy backtest. Barrier labels were statistical first-touch outcomes; they were not adjusted for fees/slippage and did not prove post-cost trading edge. The owner threshold was not met. This path is closed by ADR-056.

### 3.2 What would make a future A genuinely different

A superficial rerun of the same 2×ATR, 24-hour binary design is not recommended. If the Owner chooses A, the proposal should be rewritten around a substantively different, economically meaningful target rather than relabeling the previous experiment.

A candidate design for a **new preregistration**, not an approved implementation, is:

1. Preserve the causal timing contract: decision after the completed candle, entry reference at the next candle open, and features available no later than the decision timestamp.
2. Explicitly retain the vertical barrier as an outcome rather than silently excluding every timeout. For example, a frozen three-outcome label may distinguish `TP_FIRST`, `SL_FIRST`, and `VERTICAL_TIMEOUT`; `AMBIGUOUS` and `UNAVAILABLE` remain separately counted and censored. An alternative cost-aware target may label realized net outcomes after the vertical exit, but the exact definition must be fixed before seeing any new result.
3. Define horizontal barrier placement and vertical horizon prospectively from a documented economic hypothesis. If barriers represent executable take-profit/stop-loss outcomes, specify how the registered round-trip cost, spread/slippage assumptions, and fill convention map into the labels and the separate portfolio ledger. Do not select barrier multipliers after looking at the previous 51.09% outcome.
4. Freeze one low-capacity estimator (Logistic Regression is a suitable benchmark), its scaling, feature contract, deterministic parameters, class handling, and decision policy. Any alternative estimator or threshold is a new candidate and consumes trial budget.
5. Purge training samples whose complete label observation window overlaps the test boundary; preserve strict chronological walk-forward evaluation and an explicit embargo. Preserve every fold and every censored outcome.
6. Evaluate both prediction quality and economic value. Primary gates must include post-cost NetPF, a minimum completed-trade count, MaxDD, and fold stability under both registered cost scenarios. Accuracy, balanced accuracy, Brier/log-loss, and timeout frequency are diagnostics, not substitutes for the economic gates.
7. Record the prior Regime T attempt as a related failed experiment in the new preregistration and account for dependence/multiplicity. Do not describe a new variant as the first Triple-Barrier test.

**Assessment:** Option A is lower-cost and can reuse the already-committed pre-hack price data without acquiring another provider. However, the existing PRE-HACK data is historically exposed and cannot provide confirmatory evidence. A new target definition may improve the alignment between labels and decisions, but it cannot manufacture edge. It must be a materially new hypothesis and must pass its gates without changing them after observing results.

## 4. Option B — on-chain exchange-flow data paradigm

### 4.1 What on-chain can and cannot replace

Exchange inflow/outflow, exchange reserve changes, miner-to-exchange transfers, and related entity flows can be useful slow-moving Bitcoin-market context. They are not a direct measure of order flow on Nobitex Spot BTCUSDT.

A BTC inflow to an exchange-labeled wallet does not establish that the BTC is sold on Nobitex, sold into USDT, or executed against the Nobitex BTCUSDT order book. Exchange entity attribution, wallet-label revisions, observation timestamps, publication timestamps, and provider-side retrospective updates are material risks.

If the research objective remains a tradable Nobitex BTCUSDT strategy, the project still needs a correctly scoped BTCUSDT price/return target and an auditable post-cost execution ledger. On-chain data can at most replace or supplement the predictor/context family; it cannot replace the price series needed to define the target return, trade entry/exit, and execution economics.

### 4.2 Provider cost and entitlement snapshot (reviewed 2026-11-27)

The following information was obtained from public vendor pricing/API documentation. It is a planning snapshot, not a price quotation, account entitlement confirmation, subscription authorization, or purchase instruction.

| Provider / tier | Public price and entitlement | Important caveats |
|---|---|---|
| CryptoQuant Basic | Free; 10,000 API credits/month; 10 requests/minute; 30-day API history; market data only | No on-chain endpoints |
| CryptoQuant Advanced | $39 monthly or $29/month when billed yearly; 300,000 credits/month; 60 requests/minute; one-year API history; market data only | No on-chain endpoints. The 7-day Advanced trial requires a payment method and auto-converts to paid unless cancelled |
| CryptoQuant Professional | $109 monthly or $99/month billed yearly ($1,188 annual charge); 500,000 credits/month; 120 requests/minute; one-year API history; first tier with on-chain API | Vendor currently describes on-chain API resolution as daily. Pricing table identifies this tier as personal-use, not business-use. No evidence was found that Nobitex is a supported labeled exchange entity |
| CryptoQuant Premium | $799/month, annual billing only (about $9,588/year); unlimited credits; 800 requests/minute; full API history; hourly/per-block on-chain data | Pricing table identifies Business Use. Still does not prove Nobitex entity coverage or pair-specific BTCUSDT flow |
| Glassnode Professional | Public sales page advertises Professional from $999/month; API is an optional add-on; data credits are configured separately | Advanced does not include API. Professional lists Commercial Use and has Point-in-Time features, but API cost is additive/configured and exact Nobitex exchange coverage is unproven |
| Coin Metrics Community | Free Community API for eligible metrics | Community license is non-commercial; entitlement for the exact exchange-flow metrics and any suitable business use is not established |

CryptoQuant's current public material distinguishes market data from on-chain data: Professional includes on-chain at daily resolution, while Premium adds higher on-chain resolutions and full history. Its pricing/FAQ pages expose differences in tier and resolution descriptions; the exact endpoint, resolution, credits per call, revision flags, and history should be confirmed in writing for any selected metric. Some endpoint values may be revised and wallet/entity labels are updated over time; raw first-seen snapshots and `available_at` must be archived before historical use.

Glassnode's public pricing page lists Professional from $999/month, with API as an optional add-on and credit-based usage. Its Point-in-Time endpoints are described as append-only/immutable, but the selected metric must have an actual PIT variant; the label “PIT” must not be assumed to apply to every metric. Per-exchange Nobitex coverage and total API/data-credit cost remain unverified.

### 4.3 Purchase gate before any B work

Before the project spends money or acquires data, a separate source-qualification task would need written answers for:

- Is Nobitex individually supported as an exchange entity? Which exact entity IDs and wallet labels are available, and how are relabels/revisions handled?
- Which exact BTC exchange inflow/outflow/reserve metrics are available, at what cadence/resolution, what date range and retention, and what are the credits per request?
- Does the vendor provide a historical point-in-time archive or immutable versioned snapshots with original publication/availability timestamps?
- What license covers use in AliX research, persistent storage, cloud CI, model training, commercial deployment, and redistribution? A personal-use tier must not be treated as a business license.
- What is the all-in cost including subscription, API add-on, data credits, history/backfill, seats, tax, and renewal? Is a paid trial or auto-renewing card trial involved?
- How will the feature be aligned `as-of` each BTCUSDT decision time, and how will provider corrections after the original publication be handled?

**Assessment:** Option B has potentially richer on-chain context but materially higher cost and unresolved entity, point-in-time, license, and resolution constraints. It cannot replace the BTCUSDT price/return target for evaluating a tradable strategy. Current recommendation is `DEFER / HOLD`; no purchase or API account action is approved.

## 5. Recommendation and Owner decision gate

**Conditional recommendation:** prefer Option A for further *design work only*, because it can use the already-committed OHLCV source without purchasing new data. However, the existing Regime T design is already closed NO-GO. A may proceed only if the Owner approves a materially different target/utility hypothesis (including explicit vertical-timeout treatment and post-cost outcome evaluation) and a new prospective preregistration. Repeating ADR-056 is not recommended.

Option B remains `DEFER / HOLD` until the exact Nobitex entity coverage, metric entitlement, PIT behavior, business license, history/resolution, and total price are independently confirmed in writing. If Option A cannot be distinguished scientifically from ADR-056, neither option should proceed to implementation; a fresh proposal is needed instead of relabeling a closed experiment.

The Owner's selection is intentionally left open:

- `A — materially new target redesign`, or
- `B — on-chain source qualification / data paradigm study`.

Until that decision and a separate scope authorization are recorded, the project remains stopped at the documentation/review gate. This ADR does not authorize even the next data-source qualification call, purchase, collector, fit, or backtest.

## 6. MRD v1.2 change record

The updated Persian MRD v1.2 is delivered as a separate DOCX artifact (SHA-256 above); this repository commit carries its durable change record in ADR-065. The document:

- Replaces the stale 2026-11-16 operational snapshot with the verified Research base HEAD `95133203e70265ac96f7d4dd7c14ed213fadaf38` and explicitly notes that the current documentation commit will advance the branch afterward.
- Marks HGBQ-CQR, M3 OFI, and M5.2 Sentiment as closed `NO-GO`; it also records the previously completed Regime T Triple-Barrier `NO-GO`.
- Records the bounded trial inventory and distinguishes 10 unique identities from 37 recorded slots/entries including HGBQ duplicates.
- Replaces the stale HGBQ-CQR-active roadmap with a decision-gated A/B proposal.
- Confirms `Holdout=SEALED`, no Post-Hack access, no Champion, and no trading authorization.

The DOCX is delivered as a separate artifact and ADR-065 is stored on the Research branch. `main` is not modified.

## 7. Final safety attestation

At the time this ADR was written:

- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`
- `Holdout=SEALED`
- `holdout_bytes_read=false`
- `posthack_data_read=false`
- `private_api_calls=0`
- `real_order_attempts=0`
- `paper_orders=0`
- No model created/fitted/trained; no new market data acquired; no paid subscription/API activated.
- No `main` commit or tree mutation performed in this phase.

**Final disposition:** documentation reconciliation and the A/B proposal are complete. Implementation is stopped pending the Owner's A/B decision.  

Architecture First. Evidence Always. Fail Closed.