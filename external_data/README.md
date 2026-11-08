# External Data

Phase 6 provides versioned provider and record contracts, injected-transport adapters, separate in-memory stores, and available-at-only point-in-time joins.

## AI-6 Bybit Funding-only research adapter

The active AI-6 adapter is limited to unauthenticated Bybit V5 Linear public Funding Rate History for BTCUSDT and ETHUSDT:

- Endpoint: GET https://api.bybit.com/v5/market/funding/history.
- Pagination: descending endTime-only requests, max 200 records per page, moving backwards by one millisecond from the oldest record.
- Features: funding_rate_zscore and funding_rate_ma.
- Availability: fundingRateTimestamp + 5 minutes conservative research proxy.
- Joins: backward as-of only; observation age is capped at 48 hours.
- Coverage: exact 1825-day window, boundary validation, settlement-gap validation and >=90% estimated expected-event coverage.
- Provenance: source settlement timestamp, retrieval timestamp, availability timestamp, record content hash, CSV SHA-256, manifest, and safety fields.

Open Interest was deliberately removed from AI-6. The public endpoint's limited recent history cannot support the 1825-day experiment. No OI endpoint, fetch option, data class, feature, or model input is part of the active pipeline.

The connector does not infer commercial-use, redistribution, or long-term retention rights from public access. The owner directive authorizes only bounded internal research.

All other providers remain conditional. Numeric observations continue to use the shared external-record Data Quality Gate; raw text, numeric analytics, and derived NLP features remain separate. Only records actually available by the decision timestamp can enter a historical feature join.

Do not scrape or redistribute data without documented permission. TradingView remains blocked for automated/non-display use until explicit permission is recorded. Fixture tests approve test fixtures only, not any live provider or model.
