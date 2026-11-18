# ADR-058 — Regime M Historical Microstructure Sufficiency and Live Capture

- **Status:** M2: PUBLIC HISTORICAL MICROSTRUCTURE NOT ESTABLISHED — LIVE CAPTURE REQUIRED; M1: IMPLEMENTED — PILOT PENDING
- **Date:** 2026-11-18
- **Branch:** research/nobitex-hf-strategy
- **Predecessor:** ADR-057 (M0 public-feed feasibility)
- **Market scope:** BTCIRT
- **Research stage:** M1 data acquisition and M2 historical-depth feasibility only
- **Model authorization:** NONE. OFI and DeepLOB remain prohibited pending M1/M2 acceptance.

## 1. Decision summary

M0 proved that unauthenticated public Nobitex REST and WebSocket market feeds are reachable from a GitHub-hosted runner. M2 now distinguishes historical OHLC from historical microstructure:

1. Historical OHLC candles are documented and paginated. This is price-bar history, not order-book or trade-by-trade history.
2. The documented public L2 endpoints return the current aggregate book. No public historical L2 query, archive, timestamp range, or replay interface is documented.
3. The public market-wide recent-trades endpoint returns recent trade records, but the documentation specifies no historical time filter, pagination, or guaranteed retention/lookback.
4. The public trade WebSocket is live publication, not a documented historical replay/archive.

**M2 decision:** no documented, dependable six-month public L2 or market-wide tick/trade-flow history can be obtained from the reviewed Nobitex public API. For this research, the required microstructure corpus must be collected prospectively from Live feeds starting now. This does not mean that Nobitex has no historical market data of any kind: historical OHLC candles are available.

Private account trade-history endpoints are explicitly excluded. They return a user's own activity, require authentication, and are not a source of public market-wide trades.

## 2. M2 — documented API evidence

Documentation reviewed on 2026-11-18:

| Interface | Public/authentication | Shape and behavior | Historical sufficiency |
|---|---|---|---|
| `GET /v3/orderbook/SYMBOL` | Public; no token; documented limit 300/min | Current aggregate L2 snapshot: asks/bids price-size pairs, `lastUpdate`, `lastTradePrice` | No documented `from/to`, history, archive, or replay parameters |
| `GET /v2/depth/SYMBOL` | Public; no token; experimental | Current aggregated depth snapshot; cache-aware 1–10 second polling guidance | No documented historical query or archive |
| `GET /v2/trades/SYMBOL` | Public; no token; documented limit 60/min | Recent market-wide trades with `time`, `price`, `volume`, `type` | No documented time filter, page cursor, lookback, or guaranteed retention |
| `public:orderbook-{symbol}` WebSocket | Public; no token | Complete `bids/asks` arrays with `lastUpdate` on book changes; publisher offset included | Live updates only; no public history/replay interface documented |
| `public:trades-{symbol}` WebSocket | Public; no token | Live market-wide trade publications with price, time, side, volume | Live updates only; no public history/replay interface documented |
| `GET /market/udf/history` | Public historical candle API | OHLCV candle series; documented minute history begins in year 1401; maximum 500 candles per response and `page` pagination with `from/to` | Six-month OHLC can be paginated, but it is not L2 or a trade tape |
| `/market/trades/list` (personal history) | Private/authenticated | The authenticated account's own recent trades (documented last three days) | Not public market-wide history; prohibited for this project |

The Order Book WebSocket documentation provides a full book payload and states that it publishes only when the book changes. It recommends obtaining an initial REST snapshot before subscribing. The optional Fossil delta mode is deliberately not used because decoding it would introduce an additional stateful dependency into this first raw capture.

ADR-057's bounded M0 smoke run independently observed:
- `/v3/orderbook/BTCIRT`: successful public response with 24 bid and 24 ask levels.
- `/v2/depth/BTCIRT`: successful public response with 75 bid and 75 ask levels.
- `/v2/trades/BTCIRT`: 20 recent records in that response, spanning approximately 26.676 seconds in that one observation only. This is not a retention guarantee.
- Both public WebSocket subscriptions accepted and at least one publication received per channel.
- No Authorization header, no WebSocket connection token, and zero private API calls.

The M0 observation and the documentation do not establish six months of L2 or public tick history. It would be incorrect to infer that the REST recent-trades response covers a particular fixed period.

## 3. Explicit M2 conclusion

**No dependable six-month public historical L2 or market-wide tick/trade-flow history is documented or established. Live collection is required from the start of this capture program.**

Historical OHLCV may still be used for its own clearly labelled use case, subject to separate dataset authorization and provenance. It must not be relabelled, substituted, or represented as historical microstructure.

No private credentials, private account endpoints, Holdout bytes, or Post-Hack bytes are used by this ADR or by the M1 workflow.

## 4. M1 — bounded capture design and implementation

### 4.1 Files

- `scripts/capture_nobitex_microstructure.py`
- `tests/unit/test_capture_nobitex_microstructure.py`
- `.github/workflows/capture-microstructure.yml`

The workflow is `workflow_dispatch` only. Its bounded capture duration is selectable from 1, 5, 15, 30, or 60 minutes. It has no automatic schedule, so no unbounded or unattended collection is started. It reuses the Standard CI workflow and runs capture only after that CI job succeeds.

### 4.2 Public data acquisition

The collector:
1. Connects to the documented public Centrifugo WebSocket endpoint without sending a connection token.
2. Subscribes only to `public:orderbook-BTCIRT` and `public:trades-BTCIRT`.
3. Takes an initial public REST L2 snapshot and records the best bid/ask, spread, mid-price, and first ten levels on each side.
4. Seeds the trade file with records returned by the public recent-trades endpoint, then appends observed live WebSocket trades with deterministic duplicate removal.
5. Uses ordinary non-Fossil publications. It does not request private feeds or authenticated data.

### 4.3 Output contract

The artifact contains:
- `orderbook.csv`: capture timestamp, exchange book timestamp, publisher offset, best bid/ask and sizes, spread, mid-price, and top-ten aggregated L2 levels.
- `trades.csv`: capture timestamp, exchange trade timestamp, publisher offset, side, price, volume, and source (recent REST seed or live WebSocket).
- `gaps.csv`: stream, capture interval, exchange timestamps/offsets when available, reason, and explicit `replay_attempted=false` / `replay_available=false`.
- `manifest.json`: code SHA, workflow run ID, market, source, start/end UTC, requested duration, exchange-time ranges, record counts by stream/source, gap count/status, file byte sizes, SHA-256 digests, error list, and safety invariants.
- `manifest.sha256`: SHA-256 digest of the manifest.

CSV is used for transparent first-pilot inspection and does not require introducing a new Parquet dependency. The manifest has a schema version so a later format migration can be explicit.

### 4.4 Gap and integrity policy

- A WebSocket receive/connection failure starts a gap interval for both streams. Reconnection does not erase the gap.
- After reconnection, the collector obtains a fresh REST book snapshot as a new baseline. It does not fabricate intermediate book states.
- A publisher offset jump is recorded as a gap between the last accepted and next observed offsets.
- A missing publisher offset is recorded as a gap marker; the record remains explicitly identified as lacking an offset.
- A non-monotonic/repeated publisher offset is recorded as a gap and its publication is rejected from the valid dataset.
- No linear interpolation, forward-fill, inferred trade reconstruction, or fabricated L2 update is permitted.
- A quiet book with no messages is not itself a disconnect: the documented channel publishes only when the book changes. Offset continuity and socket liveness are treated separately.

A gap-free pilot is not proof that the vendor has no unobserved upstream loss; it only means no local disconnect or observable publisher-offset discontinuity was detected during that run. Publisher offsets are treated as an integrity signal, not as a guarantee of historical recoverability.

### 4.5 Mock test coverage

The mock suite validates:
- L2 schema, best bid/ask and top-level extraction.
- Trade schema and buy/sell validation.
- Multi-message WebSocket frame decoding.
- Publisher-offset skip and non-monotonic detection.
- Rejection of a non-monotonic book publication.
- CSV record counts, SHA-256 manifests, and no-auth/no-trading invariants.

No unit test opens a network connection. All real-data interaction is isolated in the explicitly dispatched GitHub Actions capture job.

## 5. Pilot execution and acceptance

**Pilot status at ADR creation:** PENDING CI PASS AND MANUAL DISPATCH.

The intended pilot is a single 60-minute run from GitHub Actions against the Research branch. If workflow-dispatch visibility requires temporary exposure of the workflow definition on `main`, that copy is permitted only to trigger the selected Research ref, and must be deleted immediately after dispatch. The main branch Tree SHA must be rechecked against the frozen baseline:

`5761c6c34143f8e6924f53a4e745c48932a5f791`

No other main-branch paths may change.

Pilot acceptance requires:
- Standard CI succeeds without relaxing Ruff, strict Mypy, Xenon, security, dependency, or coverage gates.
- The live capture produces non-empty L2 and live-trade records.
- All artifact file SHA-256 values independently match the manifest.
- Start/end UTC and exchange-time ranges are present.
- Any disconnection or offset gap is represented in `gaps.csv`; no imputed values appear.
- The artifact is downloadable and its exact size, row counts, gap count, and digest are recorded in this ADR.
- Safety invariants remain: no Authorization header, no WebSocket token, no private API calls, no Holdout/Post-Hack reads, no real/paper orders, `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`.

This section must be updated with the actual run ID, commit SHA, artifact ID/digest, counts, byte size, gap details, and CI result after pilot completion. No pilot result is assumed in advance.

## 6. Decision gates and prohibited work

- M0: public REST and WebSocket reachability — confirmed in ADR-057.
- M2: six-month public L2/tick history — not established; prospective live capture is required.
- M1: capture integrity — implementation and Mock tests are committed; pilot evidence remains pending.
- M3: OFI pre-registration — not authorized by this ADR.
- M4: DeepLOB — not authorized by this ADR.

No OFI, DeepLOB, feature engineering, model training, model evaluation, or strategy implementation is included. No model code may be started until M1 and M2 are separately reviewed and accepted by the owner.

## 7. Safety invariants

- Public market-data feeds only.
- `authorization_header_sent=false`
- `websocket_connection_token_sent=false`
- `private_api_calls=0`
- `holdout_bytes_read=false`
- `posthack_data_read=false`
- `real_order_attempts=0`
- `paper_orders=0`
- `Champion=null`
- `order_permission=false`
- `trading_authorization=NO-GO`

Architecture First. Evidence Always. Fail Closed.
