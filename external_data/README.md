# External Data

Phase 6 adds versioned provider and record contracts, injected-transport adapters, separate in-memory stores and an available-at-only point-in-time join.

## AI-6 Binance Funding/OI research adapter

The AI-6 branch adds a narrowly scoped, unauthenticated Binance USD-M public market-data connector for BTCUSDT and ETHUSDT:

- Funding Rate History: `/fapi/v1/fundingRate`, paginated using provider timestamps.
- Open Interest History: `/futures/data/openInterestHist`, period 4h, latest 30 days only.
- Features: `funding_rate_zscore` and `oi_change`, joined by conservative `available_at` proxies.
- Source timestamps, content hashes, CSV hashes, coverage and missing-period counts are retained in a manifest.
- No credentials, private API, orders, notifications or account data are used.

Funding availability is conservatively modeled as `fundingTime + 5 minutes`; OI availability as `timestamp + 4 hours + 5 minutes`. These are research proxies, not provider-certified historical publication timestamps. The connector does not infer commercial-use, redistribution or long-term retention rights from public access.

Binance OI History is limited to the latest 30 days. The five-year AI-6 model/economic run therefore fails closed and skips LightGBM unless complete five-year OI coverage is evidenced. Missing periods are never interpolated, backfilled or synthesized.

All other providers remain conditional. Numeric observations continue to use the shared external-record Data Quality Gate; raw text, numeric analytics and derived NLP features remain separate. Only records actually available by the decision timestamp can enter a historical feature join.

Do not scrape or redistribute data without documented permission. TradingView remains blocked for automated/non-display use until explicit permission is recorded. Fixture tests approve test fixtures only, not any live provider or model.
