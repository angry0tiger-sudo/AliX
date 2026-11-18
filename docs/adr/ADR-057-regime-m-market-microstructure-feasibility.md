# ADR-057 — Regime M: Market Microstructure Feasibility

- **Status:** API FEASIBILITY CONFIRMED — GITHUB ACTIONS PUBLIC-FEED PROBE PENDING
- **Date:** 2026-11-18
- **Branch:** research/nobitex-hf-strategy
- **Scope:** Nobitex public Spot market data only; no account or trading operations
- **Decision class:** Data-source feasibility; no model or trading authorization

## 1. Context and objective

Regime P (HGBQ-CQR) and Regime T (Triple-Barrier) did not establish a usable OHLCV edge. This ADR assesses whether Nobitex documents public order-book and market-trade data that can support a separate microstructure research regime.

This is a data-source feasibility decision, not authorization to collect an unbounded production stream, train a candidate, or place orders. No new OHLCV model is proposed or authorized here.

## 2. Official source material reviewed

The primary source is Nobitex's public API documentation repository, `nobitex/docs-api`, whose README identifies it as the official Nobitex market API documentation and links to `https://apidocs.nobitex.ir/`.

Reviewed documentation:
- [Public market data: `_market_data.md`](https://github.com/nobitex/docs-api/blob/master/source/includes/_market_data.md)
- [WebSocket: `_websocket.md`](https://github.com/nobitex/docs-api/blob/master/source/includes/_websocket.md)
- [Market trades and account operations: `_market_trade.md`](https://github.com/nobitex/docs-api/blob/master/source/includes/_market_trade.md)
- [Official documentation repository](https://github.com/nobitex/docs-api)

The statements below distinguish documented API contracts from items that the documentation does not guarantee.

## 3. Findings: public Order Book availability

### 3.1 REST snapshots

The public market-data documentation specifies:

- `GET /v3/orderbook/SYMBOL`, for example `/v3/orderbook/BTCIRT`
- `GET /v3/orderbook/all` for all markets
- Rate limit documented as 300 requests/minute
- No API token required
- Response fields include `bids`, `asks`, `lastTradePrice`, and `lastUpdate`
- Each bid/ask level is a `[price, quantity]` pair

This is an aggregated price-level book (L2-style snapshot). The documented payload does not expose individual order identifiers, per-order queue positions, or order-level add/modify/cancel events. Therefore it must not be described as L3.

The documentation also describes `GET /v2/depth/SYMBOL` as an **experimental** depth-chart endpoint, with a documented limit of 300 requests/minute and no token requirement. It returns aggregated bid/ask price-quantity pairs and `lastUpdate`. The documentation recommends a 1–10 second interval when polling; sub-second requests may return cached copies of the previous response. The documentation recommends WebSocket for maintaining the latest order-book state.

### 3.2 Live WebSocket Order Book

The official WebSocket documentation specifies:

- Endpoint: `wss://ws.nobitex.ir/connection/websocket`
- Initial connection can use an empty connect payload (`{"connect":{}}`)
- No token is required to establish the WebSocket connection
- Public channels are documented as accessible without authentication
- Order-book channel pattern: `public:orderbook-{marketSymbol}`
- The channel publishes when the order book changes
- The publication carries bid/ask price-quantity arrays and `lastUpdate`
- Optional Centrifugo `delta: 'fossil'` mode compresses book changes; non-official clients must implement Fossil decoding or avoid that mode
- Maximum 100 simultaneous WebSocket connections per IP and 450 channel subscriptions per connection are documented

The documentation warns that quiet markets may have long gaps between book publications. A REST snapshot should therefore initialize the book before consuming the stream. A collector must also verify subscription acknowledgements, reconnect behavior, timestamp continuity, and any delta/recovery semantics before considering the resulting sequence usable.

**Conclusion:** public, real-time L2-style book data is documented and does not require a private API token. L3 access is not documented by the reviewed public contract.

## 4. Findings: Trade Flow and Tick Data

### 4.1 Public REST trades

The public market-data documentation specifies:

- `GET /v2/trades/SYMBOL`, for example `/v2/trades/BTCIRT`
- Rate limit documented as 60 requests/minute
- No API token required
- Trade records include `time`, `price`, `volume`, and `type` (`buy` or `sell`)

This establishes a public trade-tape endpoint. The reviewed documentation does **not** specify a guaranteed lookback interval, maximum record count, pagination contract, or long-term retention for this public endpoint. Its historical depth must be measured rather than assumed.

### 4.2 Public WebSocket trades

The official WebSocket documentation specifies the public channel pattern:

`public:trades-{marketSymbol}`

The channel is described as publishing all trades occurring in the selected market. The documented event fields are:
- `price`
- `time` (Unix milliseconds)
- `type` (`buy` or `sell`)
- `volume`

This is a live event stream and can support prospective tick capture and signed-volume research. The exact interpretation of `buy`/`sell` as aggressor-side classification should be validated against captured events before using it as a signed trade-flow label.

### 4.3 Historical-data limitation

The current public contract reviewed here documents current L2 snapshots and live public WebSocket streams, but does not document a queryable historical L2/L3 order-book archive or replay API.

The public `/v2/trades/SYMBOL` endpoint has no explicit retention/lookback guarantee in the reviewed documentation. A separate `GET /market/trades/list` endpoint is explicitly described as listing the authenticated user's own trades from the last three days; it requires an Authorization token and is not a public market-wide history source. It is out of scope for this investigation.

Accordingly:
- Historical L2 reconstruction is not established by the public API documentation.
- Historical L3 reconstruction is not established.
- Public trades may expose some recent records, but the documented retention horizon is unknown.
- A reliable historical research corpus would need to be collected prospectively, or sourced from an independently licensed archive whose provenance and venue coverage are verified.

## 5. Authentication and GitHub Actions compatibility

At the API-contract level, Nobitex documents the REST order-book, depth, and public trade endpoints as requiring no token. It documents the WebSocket connection itself as requiring no token and marks `public:` channels as accessible without authentication. The `/auth/ws/token` flow and account token are for private channels; these are explicitly excluded.

This makes a bounded GitHub Actions probe technically compatible with the public API without repository secrets or private API credentials. It does **not** by itself prove that Nobitex accepts egress from GitHub-hosted runner IP ranges. IP reputation, regional/network filtering, service availability, TLS/WebSocket behavior, and shared-IP rate limiting require a live public-only probe from the actual runner.

A temporary read-only Actions smoke workflow is included in this branch for that purpose. It requests one public snapshot from each documented REST endpoint and attempts to connect/subscribe to the two public WebSocket channels without an Authorization header or connection token. It makes no order, account, wallet, or private-channel calls. The smoke report will be appended after the run completes.

Execution history: the first manual dispatch attempt was blocked because the workflow file was not yet present on the selected Research ref; no job started. The first push-triggered attempt ([Run 44354401295](https://github.com/angry0tiger-sudo/AliX/actions/runs/44354401295)) then failed at checkout because the temporary workflow contained an escaped SHA expression. The REST and WebSocket probe steps were skipped, so no Nobitex requests were made in that attempt. The checkout interpolation has been corrected; the public-feed probe remains pending.

GitHub Actions is suitable for a short connectivity qualification and bounded research capture windows, but should not be treated as a low-latency, continuously available market-data collector. Production-grade event capture would require a persistent collector with explicit reconnect, gap detection, durable storage, and clock/latency monitoring.

## 6. Regime M model direction, conditional on data qualification

### 6.1 Start with Order Flow Imbalance (OFI)

If the public L2 and trade streams pass the capture-quality gates, the first registered experiment should use interpretable OFI features rather than immediately training DeepLOB.

Candidate feature families for a separately preregistered experiment:
- Best-bid/best-ask queue imbalance
- Best-level OFI from changes in bid/ask prices and displayed quantities
- Multi-level OFI over a fixed, preregistered number of price levels
- Spread, microprice, depth slope, and depth concentration
- Signed trade volume/imbalance using the public trade `type` field, subject to semantic validation
- Trade and quote event intensity, inter-arrival times, and short-horizon realized volatility

The first research stage should be an observability and data-quality study: timestamp ordering, duplicates, sequence gaps, disconnect/reconnect recovery, snapshot-to-stream synchronization, level count stability, event rates, and capture completeness. No predictive model should be fitted until those checks pass.

### 6.2 DeepLOB is conditional, not the starting assumption

DeepLOB may be considered only if the collector proves a sufficiently dense and stable multi-level L2 event sequence, with consistent level semantics and enough prospective observations. The current documentation does not promise L3 order identifiers or a historical L2 archive. Therefore DeepLOB feasibility is conditional on data quality and volume, not established merely by the existence of a WebSocket channel.

No DeepLOB or OFI implementation, model training, parameter tuning, or trading evaluation is authorized by this feasibility ADR.

## 7. Decision gates

| Gate | Required evidence | Decision |
|---|---|---|
| M0 — Public endpoint access | Unauthenticated REST responses and public WebSocket connect/subscription from GitHub Actions | Pending smoke run |
| M1 — Capture integrity | Stable timestamps, no unexplained sequence gaps, deterministic snapshot-to-stream initialization, reconnect handling | Not started |
| M2 — Historical sufficiency | Measured public trade lookback plus adequate prospective L2/trade capture | Not established |
| M3 — OFI preregistration | Frozen labels, horizons, features, event filters, temporal split, cost treatment, and baselines | Not authorized |
| M4 — DeepLOB consideration | Dense, complete multi-level sequence and enough prospective data | Deferred |

Passing M0 would establish connectivity only; it would not establish historical sufficiency, predictive edge, profitability, or trading permission.

## 8. Fallback paths if public capture is blocked or history is insufficient

1. **External historical microstructure archive:** evaluate a reputable provider or another legally accessible venue that can supply timestamped L2 snapshots/deltas and market-wide trades with documented depth, retention, licensing, and provenance. This would be a new venue/data regime and requires a separate ADR and preregistration.
2. **Close quantitative strategy research:** if no compliant, reproducible microstructure source can be acquired, stop model development rather than return to another OHLCV target or substitute unverified data. Preserve the current fail-closed posture.

## 9. Safety invariants

- PRE-HACK/POST-HACK project Holdout datasets are not accessed by this feasibility probe.
- No API token, private API call, account endpoint, wallet endpoint, or private WebSocket channel is used.
- No trading endpoint is called; no order is submitted or simulated.
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `real_order_attempts = 0`
- Public market data collection does not confer trading authorization.

## 10. Current conclusion

Nobitex documents public L2-style order-book snapshots, public live L2 order-book updates, public recent trade records, and a public live trade stream without requiring private API credentials. The reviewed documentation does not establish L3 per-order data or a historical L2 archive. Public trade-history retention is unspecified.

The API design is promising enough to justify a narrowly scoped connectivity/data-quality qualification. Whether GitHub-hosted runners can actually reach the public endpoints, and whether the public REST trade endpoint returns useful historical depth, remain empirical questions. No model work is authorized until M0–M2 are evidenced.

Architecture First. Evidence Always. Fail Closed.
