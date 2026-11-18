# ADR-057 — Regime M: Market Microstructure Feasibility

- **Status:** PUBLIC L2/TICK ACCESS CONFIRMED — HISTORICAL DEPTH UNPROVEN
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

A temporary read-only Actions smoke workflow was used for a bounded probe and was removed from both Research and main after completion.

**Successful GitHub Actions probe:** [Run 44356239046](https://github.com/angry0tiger-sudo/AliX/actions/runs/44356239046), conclusion `success`, Research commit `e7021a5055ca23452c7ad5164d414ff4bb334404`. This is direct evidence from a GitHub-hosted runner, not an inference from the API documentation.

Observed REST results for `BTCIRT`, with no Authorization header:
- `GET /v3/orderbook/BTCIRT`: reachable, API status `ok`; 24 bid levels and 24 ask levels, each represented as price-quantity pairs.
- `GET /v2/depth/BTCIRT` (documented experimental endpoint): reachable, API status `ok`; 75 bid levels and 75 ask levels, also aggregated price-quantity pairs.
- `GET /v2/trades/BTCIRT`: reachable, API status `ok`; 20 trade records with `price`, `time`, `type`, and `volume`. In this one response, the oldest and newest returned timestamps spanned approximately 26.676 seconds. This is an observation of one response, not a guarantee that the endpoint's retention is only 26.676 seconds.

Observed WebSocket results, with no connection token:
- Connected to `wss://ws.nobitex.ir/connection/websocket`.
- Both `public:orderbook-BTCIRT` and `public:trades-BTCIRT` subscriptions were accepted.
- At least one publication was received from each channel during the smoke window.
- The first order-book publication exposed `asks`, `bids`, `lastTradePrice`, and `lastUpdate`; the trade publication exposed `price`, `time`, `type`, and `volume`.

The artifact is `regime-m-public-feed-smoke`, ID `14451811167`, SHA-256 `2ddc228ecd8a827d1ff433551d5a6f07552ec14c08c57b8361181f8345e626b2`. The JSON report records `authorization_header_sent=false`, `websocket_connection_token_sent=false`, and `private_api_calls=0`.

Earlier setup attempts are retained for audit: the first manual dispatch was blocked because the workflow was absent from the selected Research ref; runs `44354401295` and `44354849726` failed during checkout before market-data requests; run `44355289468` made the three REST attempts and opened WebSocket but failed in the test parser before a report was written. Those runs are not treated as data-source failures. The successful Run `44356239046` resolves the GitHub Actions connectivity question.

The temporary smoke workflow was removed from the Research branch and main after the successful run. main's tree SHA was verified restored exactly to `5761c6c34143f8e6924f53a4e745c48932a5f791`.

Execution history: the first manual dispatch attempt was blocked because the workflow file was not yet present on the selected Research ref; no job started. The first push-triggered attempt ([Run 44354401295](https://github.com/angry0tiger-sudo/AliX/actions/runs/44354401295)) then failed at checkout because the temporary workflow contained an escaped SHA expression. The REST and WebSocket probe steps were skipped, so no Nobitex requests were made in that attempt. The checkout interpolation was still escaped in the first push-triggered smoke configuration, so [Run 44354849726](https://github.com/angry0tiger-sudo/AliX/actions/runs/44354849726) also failed at checkout; its REST and WebSocket probe steps were skipped and no Nobitex requests were made. The workflow was corrected to rely on the Actions event's default checkout ref and a static artifact name. [Run 44355289468](https://github.com/angry0tiger-sudo/AliX/actions/runs/44355289468) then passed checkout and executed the three public REST GET attempts. It subsequently established a WebSocket connection and received a server frame, but the test parser treated a newline-delimited multi-message frame as one JSON object and terminated before writing the report. REST response summaries were not persisted, and no public-channel subscription acknowledgement was recorded; those results are therefore not claimed as confirmed. The WebSocket parser now splits newline-delimited messages (commit `6c711f46abb7126bd18ad01790cb429a636d9d43`) and emits each REST probe result to the job log. A final read-only probe is pending.

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
| M0 — Public endpoint access | Unauthenticated REST responses and public WebSocket connect/subscription from GitHub Actions | **PASS** — Run 44356239046 |
| M1 — Capture integrity | Stable timestamps, no unexplained sequence gaps, deterministic snapshot-to-stream initialization, reconnect handling | Not started |
| M2 — Historical sufficiency | Measured public trade lookback plus adequate prospective L2/trade capture | Not established; public API documents no historical L2 archive or retention guarantee |
| M3 — OFI preregistration | Frozen labels, horizons, features, event filters, temporal split, cost treatment, and baselines | Not authorized |
| M4 — DeepLOB consideration | Dense, complete multi-level sequence and enough prospective data | Deferred |

M0 proves that a GitHub-hosted runner can reach the documented public REST and WebSocket feeds without private credentials. It does not establish historical sufficiency, capture continuity, predictive edge, profitability, or trading permission.

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

Nobitex public L2-style snapshots, the experimental aggregate depth endpoint, public recent trades, and both live public WebSocket channels were reachable from GitHub Actions without an Authorization header or WebSocket connection token. The runtime probe confirmed 24/24 price levels from v3 orderbook, 75/75 from experimental v2 depth, 20 recent trade records, and successful subscriptions/publications on both public channels.

The evidence confirms L2-style market data and live market-wide trade flow, but not L3 per-order events. The reviewed API documentation does not define a historical L2 archive or a guaranteed retention horizon for the public trade-list endpoint. The 20-record response observed in the smoke run is not sufficient to claim a useful long-term history.

**Decision:** proceed only to a separately authorized M1 data-quality/observability capture using public L2 and trade streams, with OFI as the preferred first statistical baseline. DeepLOB remains conditional on proving dense, complete multi-level sequence quality. No model implementation, training, parameter search, OHLCV strategy, or trading evaluation is authorized by this ADR. If compliant prospective collection or adequate history cannot be established, pursue the two fallback paths in Section 8.

Architecture First. Evidence Always. Fail Closed.
