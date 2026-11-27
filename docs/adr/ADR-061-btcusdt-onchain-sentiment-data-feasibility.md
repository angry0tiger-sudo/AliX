# ADR-061 — BTCUSDT On-Chain & Sentiment Data Feasibility (M5)

- **Status:** **FEASIBILITY COMPLETE — CONDITIONAL GO FOR LOW-COST SOURCE QUALIFICATION ONLY**
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Target venue/market:** **Nobitex Spot / USDT — `BTCUSDT`**
- **Decision timeframes:** 1h and 4h
- **Scope:** data-source, licensing, cost, coverage, and point-in-time feasibility only
- **Code/model/data acquisition:** not authorized by this ADR

## 1. Decision and scope correction

The Owner's current directive is authoritative for the M5 target: all new market research and any future target OHLCV must use **Nobitex Spot `BTCUSDT` only**. The MRD v1.1 defines AliX as a controlled Nobitex Spot/USDT research system; this ADR records the exact BTCUSDT market identifier from the Owner's directive. No BTCIRT substitution, cross-quote normalization, or silent fallback is allowed.

A critical historical boundary is recorded explicitly: **ADR-060's closed M3 OFI evaluation used five M1.6 public BTCIRT capture artifacts**, not BTCUSDT. Its `CLOSED_NO_GO` decision remains valid for the two registered M3 BTCIRT trials, but those results are not a BTCUSDT backtest and must not be presented as evidence of BTCUSDT performance. No conversion of those historical observations is authorized.

This is a feasibility review, not a modeling decision. It authorizes none of the following:
- model/feature implementation, model fitting, training, backtesting, optimization, or candidate creation;
- downloading, acquiring, purchasing, or persistently collecting new data;
- subscription activation, credit purchase, trial activation, or external vendor account changes;
- Holdout or Post-Hack access;
- paper trading, private API calls, or order execution.

## 2. Executive decision

1. **Alternative.me Fear & Greed:** the only source with a clear, low-friction, no-cost path suitable for an initial qualification study. Public API, no API key, daily observations, and an all-available-history parameter. Attribution is required; commercial use is allowed subject to the provider's attribution rule. Public references identify the archive as beginning on 2018-02-01. It is a broad Bitcoin-market composite, not a Nobitex BTCUSDT-specific sentiment feed.
2. **CryptoQuant exchange flows:** technically the strongest immediately documented exchange-flow candidate, but its API Basic tier does not include on-chain data. Professional is the first API tier with daily on-chain access (currently listed at $109 monthly or $99/month on annual billing, $1,188 charged annually). No purchase is justified until Nobitex-specific entity coverage, the required metric set, point-in-time limitations, and the one-year API history entitlement are independently confirmed.
3. **Glassnode:** broad and potentially deep historical coverage, including Point-in-Time metrics and per-exchange metrics at Professional level, but commercial access requires the Professional plan and optional API add-on; price is configured rather than publicly fixed. The Advanced page contains inconsistent API wording and its API-light history is only 14 days daily. Not approved for purchase.
4. **Coin Metrics Community:** useful as a no-cost technical candidate for selected BTC network metrics, but Community access is restricted to non-commercial use and metric availability is a subset of the paid offering. Do not assume that a particular BTC exchange net-flow metric is available in Community until the exact metric's entitlement is checked.
5. **X:** full-archive historical search is technically available on pay-per-use, but Posts Read is currently listed at $0.005 per post. That is approximately $50 per 10,000 returned posts or $500 per 100,000, before any other costs. Legal and storage/deletion requirements apply. Too expensive for an unproven initial signal.
6. **Reddit:** not a practical default source for this project. Reddit requires permission and a contract for commercial use; its help policy identifies Reddit for Researchers as the official research avenue and says ordinary developer APIs are not authorized for research use. Bulk export is restricted and may incur fees.

**Recommendation:** proceed only with a later, explicitly approved, no-purchase qualification of Alternative.me and a check of Coin Metrics Community eligibility/metric coverage. No paid data subscription currently has sufficient demonstrated incremental value to justify spending.

## 3. Source comparison

Prices and entitlements below reflect public vendor information reviewed on 2026-11-27. Prices may change and do not constitute an offer or a purchase authorization.

| Source | Current public cost / tier | Historical coverage and resolution | Access and license | M5 assessment |
|---|---|---|---|---|
| **Alternative.me Fear & Greed** | Free public endpoint; no API key identified | Daily; `GET https://api.alternative.me/fng/?limit=0` returns all history available from the API. Public API references identify the archive from 2018-02-01. | Commercial use allowed if the source is prominently acknowledged next to displayed data. No impersonation. | **Best first candidate.** No paid cost; global BTC context only, not BTCUSDT-specific. |
| **CryptoQuant API** | Basic free: 10,000 credits/month, 10/min, 30-day API history, **no on-chain**. Advanced: 300,000 credits/month, 1-year API history, market data only. Professional: 500,000 credits/month, 1-year API history, on-chain included; $109/month or $99/month billed annually ($1,188/year). Premium: $799/month on annual billing, unlimited credits/full API history. | BTC exchange-flow endpoints support daily/hourly/block windows. Flow data is available as far back as tracked subject to plan entitlement. Professional API on-chain data is daily; hourly/block resolution and full history are Premium-level. | API key and paid plan required for on-chain endpoints. Vendor describes 200+ on-chain metrics across 490+ assets. Exchange-flow wallet labels are revised periodically; historical values can change and the endpoint is explicitly not Point-in-Time accurate. | **Strongest paid exchange-flow candidate, but HOLD.** Nobitex entity support was not established by the reviewed documentation; do not assume it exists. |
| **Glassnode** | Advanced shown at $49/month billed annually; Professional is configured/custom. Professional API is an optional add-on; no fixed public Professional/API price displayed. | Advanced Studio: 4 years of history, daily resolution; listed API Light is 14-day history at daily resolution and 50 calls/downloads per day. Professional: up to 15+ years and up to 10-minute resolution, depending on metric/tier. Point-in-Time and per-exchange metrics are Professional features. | Advanced license is Personal Use. Professional lists Commercial Use. The pricing page's feature table mentions API Light for Advanced, while its FAQ says API access is not included in Advanced; clarify directly before relying on it. | **Potentially strongest institutional-quality option, not cost-qualified.** No subscription or add-on approved. |
| **Coin Metrics Community API** | Free; no API key for Community endpoints. Professional tier requires contacting Coin Metrics. | A subset of Network Data Pro and Market Data Pro. Metric-specific history varies; daily history can be broad for some metrics. BTC exchange net-flow metrics exist in the broader product catalog, but Community entitlement for the exact required metrics was not confirmed. | Free Community data is for non-commercial use under the provider's Creative Commons terms. Community API limit: 10 requests per 6-second sliding window per IP. | **Conditional free candidate** for network context only if AliX's use qualifies and exact metrics are exposed in Community. Not yet a verified free exchange-flow source. |
| **X API (historical sentiment)** | Pay-per-use; Posts Read listed at $0.005 per returned resource. Approx. $50/10k posts; $500/100k posts. | Recent Search: last 7 days. Full-Archive Search: complete post archive back to March 2006, available to Pay-per-use and Enterprise; up to 500 posts/request. | Developer Agreement/Policy govern use. X content must be kept current; removal/deletion obligations apply. Redistribution is limited. X's current restricted-use materials prohibit using X API/content to train a foundation/frontier model (except Grok); no scraping. | **Defer.** Historical coverage is attractive, but costs scale with retrieved posts and policy obligations make this a non-trivial data dependency. |
| **Reddit Data API / Reddit for Researchers** | Reasonable basic access may be free; broader/bulk access may be paid. Commercial pricing is not public and may require a separate contract. | Official general Data API does not provide a public guarantee of a complete historical archive suitable for reproducible BTC sentiment research. Bulk export is restricted by default. | Commercial use requires Reddit's prior permission and a contract. Reddit says the official research avenue is Reddit for Researchers; ordinary developer APIs or unauthorized third-party tools are not an authorized research path. | **Not recommended at this stage.** Access and rights are unresolved; not a dependable low-cost historical source. |

### 3.1 Interpretation of source quality

- **Fear & Greed is a composite sentiment indicator**, not a direct read of BTCUSDT order flow. It is published at daily cadence and includes broad market components. Its low cost makes it suitable for data-source qualification, not proof of a predictive edge.
- **Bitcoin exchange inflow/outflow is asset-level on-chain movement.** A BTC transfer into an exchange-controlled wallet does not identify whether the asset will be sold into USDT, USD, IRT, or another pair. It also does not establish that the BTC is associated with Nobitex or the Nobitex BTCUSDT book.
- CryptoQuant's public BTC exchange-flow documentation lists aggregate `all_exchange`, `spot_exchange`, and `derivative_exchange` groupings, plus individually supported exchanges via its entity-list endpoint. The reviewed material did not confirm Nobitex as a supported labeled entity.
- Glassnode exchange metrics also depend on labeled exchange addresses and may change as wallet labels are refined.
- X and Reddit measure public conversation, not a clean market-wide truth signal. Query selection, language mix, bots, spam, coordinated posts, and sampling bias must be treated as material data-quality risks.

## 4. BTCUSDT compatibility and target data contract

### 4.1 Target market is not interchangeable with global BTC data

The dependent market series must be **Nobitex Spot BTCUSDT OHLCV** for both 1h and 4h. The quote currency is USDT. External BTC/USD, BTC/USDT from Binance/OKX, or aggregated global BTC price series may be used only as explicitly labeled contextual sources; they must never replace the Nobitex BTCUSDT target candles or silently define the target return.

On-chain BTC flows are network-wide context and are not pair-specific. They may be tested as slow-moving auxiliary variables beside Nobitex BTCUSDT OHLCV, but no source reviewed proves a direct relationship to the Nobitex BTCUSDT order book. Any future use must establish incremental value against the Nobitex BTCUSDT-only baseline after the same fees, spread, and executable-fill assumptions.

USDT supply/flow metrics could potentially add quote-asset context, but chain-specific issuance, bridge activity, and exchange wallet labels must be separated. They are not substitutes for the Nobitex BTCUSDT price/volume series.

### 4.2 Safe daily-to-1h/4h alignment

The MRD's decision convention uses completed candles: features at `close(t)`, reference entry at `open(t+1)`, and the outcome horizon ends at the registered future close. This remains the temporal anchor.

Daily features must use an **as-of join by publication/availability time**, not a join on the date label alone:

1. Store both the metric's observation period and the provider's actual publication/availability timestamp (`available_at`), all normalized to UTC.
2. At each completed Nobitex BTCUSDT 1h or 4h decision candle, use only the latest record for which `available_at <= decision_time`.
3. If a provider supplies only a daily period label and no trustworthy availability time, do not assume the day's value was known at 00:00 UTC. The conservative fallback is to expose the prior fully completed daily observation only after a full-day delay, and to document the delay.
4. Forward-fill a previously published daily value across later 1h/4h bars until the next value actually becomes available. Never backfill from a future observation and never interpolate a future daily value backward.
5. Persist raw provider responses, fetch/first-seen timestamps, source/version metadata, and hashes before any future research use. A revised value must not overwrite the historically observed version used for an earlier decision.
6. CryptoQuant exchange-flow history is explicitly mutable because wallet clustering is periodically updated and is not Point-in-Time accurate. Without archived point-in-time snapshots, its revised historical values are unsuitable for a leakage-safe historical backtest. Glassnode Point-in-Time metrics may address part of this issue but require Professional access.

Because a daily series changes at most once per day, repeating it across 24 hourly rows or six four-hour rows does **not** create 24 or 6 independent observations. Effective information frequency remains daily. The sample size and statistical uncertainty must reflect that.

### 4.3 Can these become contextual features?

**Yes, in principle**, as slow-moving context features alongside the correct Nobitex BTCUSDT OHLCV. Candidate examples include:
- one lagged Alternative.me sentiment score and its change;
- BTC exchange netflow/reserve or net inflow/outflow, if licensed and point-in-time safe;
- a separately qualified USDT supply/flow context metric;
- aggregated social sentiment, only if lawful access, sampling, and timestamp provenance are established.

This is not a feature-selection decision. No feature is approved for implementation, no predictive benefit is claimed, and no feature should be added to a model until a separate preregistration defines the metric, timing, lag, missingness, revisions, baseline, and evaluation gates.

## 5. Risks, controls, and unresolved checks

- **Pair contamination:** no BTCIRT candles or BTCIRT returns may enter the M5 target data path.
- **Venue mismatch:** global or other-exchange market prices are context only. The dependent series remains Nobitex Spot BTCUSDT.
- **Quote mismatch:** BTC on-chain inflow/outflow is not the same thing as BTCUSDT exchange trading flow. Pair-specific interpretation must not be manufactured.
- **Point-in-time revision:** if vendor metrics are mutable, preserve historical snapshots or exclude them from historical backtests.
- **Availability lag:** all data must be aligned by `available_at`, not by nominal daily date.
- **Licensing:** Alternative.me attribution must be preserved; Coin Metrics Community is non-commercial; CryptoQuant/Glassnode/X/Reddit use is subject to paid-plan terms and/or platform contracts.
- **Cost controls:** no paid tier or pay-per-use credits are approved. No account activation or trial conversion is authorized.
- **Trial governance:** `GLOBAL_N=UNKNOWN` remains unchanged. Any later M5 feature experiment must be prospectively registered, account for prior related trials, and keep Holdout/Post-Hack sealed.

## 6. Final feasibility verdict

| Source | Verdict |
|---|---|
| Alternative.me Fear & Greed | **QUALIFY FIRST — free, daily, historical; Attribution required** |
| Coin Metrics Community | **CONDITIONAL — only if exact metrics and non-commercial license fit** |
| CryptoQuant Professional | **HOLD — technically relevant but paid; Nobitex support/PIT/metric entitlement unresolved** |
| Glassnode Professional | **HOLD — rich history/PIT, but custom price/API add-on and license need confirmation** |
| X Full-Archive | **DEFER — high-volume historical reads can become expensive; policy controls required** |
| Reddit | **DEFER / NOT RECOMMENDED — research and commercial access require special authorization** |

**Investment conclusion:** one source merits time, not money: Alternative.me. Its public daily BTC sentiment archive can be qualified without a subscription. Coin Metrics Community is a secondary no-cost avenue only if its non-commercial license and exact metric entitlements are appropriate. No paid source has yet demonstrated sufficient incremental value for an investment.

**M5 status after this ADR:** feasibility completed; data acquisition not started; no data purchased; no model/feature code written; no training or backtest performed. Any next action requires a separate explicit Owner directive.

## 7. Safety invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `holdout_bytes_read = false`
- `posthack_data_read = false`
- `data_purchased = false`
- `new_data_acquired = false`
- `model_or_training_code_written = false`

## 8. References reviewed (accessed 2026-11-27)

1. AliX MRD v1.1 — managerial scope for Nobitex Spot/USDT, completed-candle timing, decision at close(t), entry reference at open(t+1).
2. ADR-060 — M3 evaluation data provenance (five M1.6 public BTCIRT segments) and Owner-approved `CLOSED_NO_GO` lifecycle state.
3. Alternative.me Fear & Greed API and use rules: https://alternative.me/crypto/fear-and-greed-index/
4. Alternative.me API: https://api.alternative.me/fng/?limit=0
5. Independent API history reference (archive since 2018-02-01): https://voiceofchain.com/academy/alternative-me-fear-greed-api
6. CryptoQuant API plans, credits, history, and on-chain entitlement: https://cryptoquant.com/apis
7. CryptoQuant pricing and billing FAQ: https://cryptoquant.com/pricing
8. CryptoQuant BTC exchange-flow endpoint, supported exchange lookup, update/revision and non-PIT note: https://userguide.cryptoquant.com/api/btc-exchange-flows
9. Glassnode plans, prices, API limits, history, licensing and Point-in-Time features: https://studio.glassnode.com/pricing
10. Glassnode exchange-flow metric definitions: https://docs.glassnode.com/basic-api/endpoints/transactions
11. Coin Metrics Community API access, scope, non-commercial terms and limits: https://github.com/coinmetrics/product-docs/blob/master/docs/access-our-data/api/README.md
12. Coin Metrics exchange net-flow metric definitions: https://docs.coinmetrics.io/network-data/network-data-overview/exchange/net-flows
13. X Developer Platform pay-per-use pricing: https://developer.x.com/
14. X Search Posts / Full-Archive Search coverage and access tier: https://github.com/xdevplatform/docs/blob/main/x-api/posts/search/introduction.mdx
15. X Developer Policy and guidelines: https://github.com/xdevplatform/docs/blob/main/developer-guidelines.mdx
16. Reddit Developer Platform and data access rules: https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data
17. Reddit Data API Terms (last revised 2026-07-20): https://redditinc.com/policies/data-api-terms
18. Nobitex official public API documentation: https://apidocs.nobitex.ir/
