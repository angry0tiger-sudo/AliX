# ADR-058 — Regime M Historical Microstructure Sufficiency and Live Capture

- **Status:** M2: NO DOCUMENTED PUBLIC HISTORICAL L2/TICK ARCHIVE — LIVE COLLECTION REQUIRED; M1: 5-MINUTE PILOT PASS — EXTENDED SOAK REQUIRED FOR FULL ACCEPTANCE
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

The workflow supports both a bounded `workflow_dispatch` and an hourly `schedule` trigger (`cron: 37 * * * *`, UTC). Scheduled runs capture for 55 minutes in each 60-minute cadence. Manual durations are bounded to 1, 5, 15, 30, or 55 minutes. The capture job timeout is 65 minutes; Standard CI remains a separate prerequisite job, and capture starts only after Standard CI succeeds. The cron minute is intentionally offset from the top of the hour to reduce scheduler contention.

GitHub Actions executes scheduled workflows only when the workflow file exists on the repository's default branch, and the run uses that branch. The repository's default branch is `main`; therefore, this Research-branch-only schedule cannot activate while `main` remains the default. The M1.5 activation procedure must use a temporary repository default-branch setting pointing to `research/nobitex-hf-strategy` while preserving the `main` ref and its frozen tree. Restore the default-branch setting to `main` after the bounded soak. No commit or tree mutation on `main` is authorized.

### 4.2 Public data acquisition

The collector:
1. Connects to the documented public Centrifugo WebSocket endpoint without sending a connection token.
2. Subscribes only to `public:orderbook-BTCIRT` and `public:trades-BTCIRT`.
3. Takes an initial public REST L2 snapshot and records the best bid/ask, spread, mid-price, and first ten levels on each side.
4. Seeds the trade file with records returned by the public recent-trades endpoint, then appends observed live WebSocket trades. Every observation is preserved and source-tagged; no trades are deduplicated because the public schema does not document a stable unique trade identifier, and seemingly identical trades can be distinct executions.
5. Uses ordinary non-Fossil publications. It does not request private feeds or authenticated data.

### 4.3 Output contract

The artifact contains:
- `orderbook.csv`: capture timestamp, exchange book timestamp, publisher offset, best bid/ask and sizes, spread, mid-price, and top-ten aggregated L2 levels.
- `trades.csv`: capture timestamp, exchange trade timestamp, publisher offset, side, price, volume, and source (recent REST seed or live WebSocket). Potential overlap between the REST seed and WebSocket window is preserved rather than silently deduplicated; consumers must use source/time provenance and must not assume rows equal unique exchange executions.
- `gaps.csv`: stream, capture interval, exchange timestamps/offsets when available, reason, and explicit `replay_attempted=false` / `replay_available=false`.
- `manifest.json`: code SHA, workflow run ID, market, source, start/end UTC, requested duration, exchange-time ranges, record counts by stream/source, gap count/status, file byte sizes, SHA-256 digests, error list, and safety invariants. Scheduled segments additionally declare `inter_segment_coverage_status=MISSING_COVERAGE`, the 60-minute cadence, 55-minute capture window, nominal five-minute uncovered interval, and the rule that exact adjacent-segment gap boundaries are calculated from the previous manifest's `finished_at_utc` and the next manifest's `started_at_utc` during final merge.
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

No unit test opens a network connection. All real-data interaction is isolated in the explicitly dispatched or scheduled GitHub Actions capture job.

### 4.6 M1.5 scheduled soak and missing-coverage semantics

- Schedule: once per hour at minute 37 UTC (`37 * * * *`); scheduled capture duration: exactly 55 minutes; expected nominal uncovered time per cadence: five minutes.
- Every run uploads a distinct artifact named with a UTC timestamp and GitHub run ID. Artifacts are retained for 30 days.
- Each run is an independent segment. No run attempts to stitch, replay, interpolate, forward-fill, or synthesize observations across segment boundaries.
- Each scheduled manifest records `inter_segment_coverage_status=MISSING_COVERAGE`, `schedule_interval_minutes=60`, `capture_duration_minutes_requested=55`, and `nominal_inter_segment_missing_minutes=5`. This is an explicit coverage classification, not a claim that the actual gap is exactly five minutes.
- The final audit sorts manifests by `started_at_utc` and computes each actual inter-segment uncovered interval as `[previous.finished_at_utc, next.started_at_utc)`. Delays, missed/overwritten scheduler events, failed segments, and runner queue time therefore remain visible as actual missing coverage. Segments are never labelled as continuous.
- Publisher-offset and socket-disconnect gaps remain separate observed-feed defects in `gaps.csv`; planned inter-segment missing coverage is not counted as a live WebSocket disconnect.
- This ADR does not yet grant M1 full acceptance or authorize M3. At least 24 hours of elapsed collection coverage is required; 72 hours across several market sessions remains preferred. Actual captured minutes, missing coverage, and observed-feed gaps must be reported separately.

## 5. Pilot execution and acceptance

**Pilot status: 5-MINUTE PILOT COMPLETED — PASS FOR INGESTION AND DATA-INTEGRITY SMOKE; EXTENDED SOAK REQUIRED BEFORE RESEARCH-DATASET ACCEPTANCE OR M3.**

### 5.1 Run and artifact identity

- **Workflow:** Regime M Public Microstructure Capture
- **Event / input:** `workflow_dispatch`, `duration_minutes=5`
- **Run ID:** `44369213808`
- **Run URL:** https://github.com/angry0tiger-sudo/AliX/actions/runs/44369213808
- **Selected branch:** `research/nobitex-hf-strategy`
- **Code commit SHA:** `770342e0ffa8466ec4e509b9e4188eff016f3727`
- **Started UTC:** `2026-11-18T18:29:03.963+00:00`
- **Finished UTC:** `2026-11-18T18:34:08.730+00:00`
- **Observed duration:** 304.767 seconds (5 minutes, 4.767 seconds)
- **Artifact ID:** `14458723722`
- **Artifact name:** `regime-m-microstructure-44369213808`
- **ZIP size:** 19,585 bytes
- **Artifact ZIP SHA-256:** `4aa8d41e14d3083158442eeee4aa730738920e44067b1e9aba3f7f7b2af8c472`
- **Uncompressed artifact contents:** 384,644 bytes
- **Manifest SHA-256:** `3b76365ab3301bd1d92b643df3f04bef9e90aea517e6ae2df07a278f50bab219`

GitHub reported the workflow run as **SUCCESS**. All four Standard CI jobs succeeded: `quality`, `four-arm-lock-bare-env`, `freqtrade-runtime-smoke`, and `postgres-integration`. The `real-data-diagnostic` and `ensemble-real-data-research` jobs were skipped by their existing gates; no real-data model/research job was run.

The artifact was downloaded and independently inspected. ZIP CRC validation passed. Each CSV byte count, record count, and SHA-256 matched the manifest; the manifest hash matched `manifest.sha256`. The GitHub artifact digest matched the SHA-256 of the downloaded ZIP.

### 5.2 Observed records

| File / source | Records | Bytes | SHA-256 |
|---|---:|---:|---|
| `orderbook.csv` | 488 total: 1 REST initial snapshot + 487 WebSocket publications | 375,534 | `fd8513c37ce54d3064a325459c68274163a47a1a4c466a5414ba04d0ad913fcf` |
| `trades.csv` | 79 total: 20 REST recent-trade seed + 59 WebSocket trades | 7,066 | `d524f90c140aaa034426d3028611c3cc31a29b3fab5c0588f34c16fceab8b50b` |
| `gaps.csv` | 0 gap records (header only) | 127 | `bbaaea6d0b1a672e8814745f2f0f9c20ca7411e7d41fde18b1d43311f5d604fa` |

The 487 live Order Book publications each carried a full aggregated book payload; the stored schema retained the best bid/ask and first ten levels per side. All 488 rows had valid top-ten structures, correctly ordered bid/ask levels, and no crossed books. The observed median spread was approximately 18.446 bps and the 95th percentile approximately 18.545 bps; one interval legitimately observed a one-tick 10-IRR spread.

The 59 WebSocket trade records preserved all observed buy/sell publications. The collector dropped no trades (`duplicate_trades_dropped=0`), because the documented public schema does not supply a stable unique trade ID. Consumers must not equate rows with unique executions without applying a separately specified identity policy.

### 5.3 Gap and publisher-offset evidence

- `capture_status=NO_GAPS_OBSERVED`; `gap_count=0`; `errors=[]`.
- `gaps.csv` contains its header and no data rows.
- Order Book publisher offsets were `2361136` through `2361622` across 487 publications: strictly increasing and contiguous, with no observed skip or repeat.
- Trade publisher offsets were `298805` through `298863` across 59 publications: strictly increasing and contiguous, with no observed skip or repeat.
- No WebSocket disconnect/reconnect was recorded during this run.
- Mock CI tests independently exercised publisher-offset skips and non-monotonic offsets, including recording the gap and rejecting a non-monotonic book publication.

Therefore, Gap Detection was active and evaluated the observed offsets, but no live gap event occurred in this five-minute window. This is evidence of no locally observable gap during this run, not proof of zero upstream loss or proof of reconnect recovery under fault. No missing observations were interpolated or forward-filled.

### 5.4 Timestamp and ordering audit

All timestamps parsed as valid Unix epoch milliseconds and all `captured_at_utc` values parsed as UTC. The event-to-capture delay below is the observed difference between the runner capture timestamp and the exchange event timestamp; it includes local scheduling/network effects and is not a calibrated clock-synchronization measurement.

- **Order Book exchange range:** `2026-11-18T18:29:06.365Z` to `2026-11-18T18:34:08.365Z`.
- **Order Book WebSocket capture lag:** minimum 118 ms, median 123 ms, p95 165 ms, maximum 1,178 ms. The initial REST snapshot's capture lag was 781 ms.
- **Trade WebSocket exchange range:** `2026-11-18T18:29:22.709Z` to `2026-11-18T18:33:54.180Z`.
- **Trade WebSocket capture lag:** minimum 146 ms, median 158 ms, maximum 371 ms.
- WebSocket event timestamps were monotonic within each stream during the observed run.

**REST seed ordering caveat:** the 20 `rest_recent_seed` trade rows were returned newest-first (descending exchange timestamp), spanning `2026-11-18T18:26:52.434Z` to `2026-11-18T18:27:26.811Z`. At their common capture timestamp (`18:29:08.042Z`), they were approximately 101.231–135.608 seconds old. This is consistent with a recent-history seed, not a live-event latency measure. Consequently, `trades.csv` is append/source order, not a globally ascending exchange-time series; downstream processing must sort by exchange timestamp and retain the source marker. The REST seed must not be treated as five-minute live coverage.

### 5.5 Pilot decision and remaining acceptance

The pilot **passes** the bounded M1 ingestion/integrity smoke criteria:
- non-empty public L2 and live trade-flow data were received;
- Mock tests and the complete Standard CI dependency gate succeeded;
- file counts, sizes, SHA-256 values, and manifest checksum matched independently;
- WebSocket offsets were contiguous for both streams;
- no locally observable gap, malformed book, crossed book, or invalid trade side was found;
- event timestamps and runner capture timestamps were parseable and their observed WebSocket lags were bounded as recorded above;
- all safety invariants held.

This five-minute sample does **not** establish day/night coverage, behaviour across changing liquidity/volatility regimes, long-run reconnection resilience, or adequate statistical sample size for OFI design. Accordingly, M1 is a pilot-level pass only; full data-quality acceptance remains conditional on an extended soak and review of the resulting corpus.

M1.5 is the scheduled collection phase: hourly UTC triggers at minute 37, with a 55-minute bounded capture per run and a nominal five-minute uncovered interval. Multiple timestamped, independently verifiable artifacts are expected. Their exact start/end boundaries must be retained and actual inter-run downtime must be derived from adjacent manifests as `MISSING_COVERAGE`; do not describe segmented runs as one uninterrupted 24-hour stream. Re-run the manifest/hash/offset/timestamp audit across the complete combined corpus before seeking M1 full acceptance.

M1.5 has been approved by the owner, but approval is not evidence that the schedule is active or that any soak data has been collected. Actual activation time, default-branch setting, first scheduled run, elapsed interval, artifact inventory, and final audit evidence must be appended only after they are independently verified.

The main branch was restored after temporary dispatch exposure and independently verified at the frozen Tree SHA:

`5761c6c34143f8e6924f53a4e745c48932a5f791`

No other main-branch paths were changed.

## 6. Decision gates and prohibited work

- M0: public REST and WebSocket reachability — confirmed in ADR-057.
- M2: historical six-month public L2/tick history — not documented or established; the historical-data feasibility decision is complete and prospective Live collection is required.
- M1: five-minute pilot — PASS for ingestion, manifest integrity, timestamps, and observed offset continuity; full data-quality acceptance awaits the M1.5 corpus audit.
- M1.5: OWNER-APPROVED SCHEDULED SOAK DESIGN — hourly 55-minute segments, with explicit `MISSING_COVERAGE` between segments; operational activation and accumulated evidence remain pending until recorded from GitHub.

### 6.1 M1.5 activation record (2026-11-18)

- Research-branch implementation commit: `c830dae15adc3bbabd7d8e74e09fbf662dce1311`.
- Standard CI for that exact commit: PASS; run `44379295099`.
- The repository default-branch setting has been temporarily switched from `main` to `research/nobitex-hf-strategy` so GitHub can discover and schedule the workflow on the Research branch. This is repository metadata only; no change was made to the `main` ref or its contents.
- Independently rechecked `main`: HEAD `cd9b13d42ac0cf0edd2df066ff81580e028146d6`; Tree SHA remains exactly `5761c6c34143f8e6924f53a4e745c48932a5f791`.
- The next cron tick after activation is scheduled for minute 37 UTC. At this record's verification time, no scheduled run has yet been observed. Workflow activation is not considered operationally proven until GitHub Actions reports an `event=schedule` run and its capture job succeeds.
- The default-branch setting must be restored to `main` after the M1.5 collection window and final artifact audit. No commit or tree mutation on `main` is authorized.

### 6.2 Workflow registration and first scheduled-run verification setup

- On the current default branch, GitHub's uncached Actions workflow index initially omitted `capture-microstructure.yml`, even though the file was present in the branch tree. After a comment-only refresh push on the workflow file at commit `cacf035b836102d0d70114469614b1d2f713fb13`, GitHub's Actions API listed it again as workflow ID `422952435`, name `Regime M Public Microstructure Capture`, state `active`. The underlying cause of the stale index was not established; do not infer a parser defect.
- The exact same Standard CI commit `cacf035b836102d0d70114469614b1d2f713fb13` passed in run `44383522423` (quality, four-arm lock, Freqtrade runtime, and PostgreSQL integration all successful).
- Capture job timeout is now bounded at 60 minutes; scheduled collection remains exactly 55 minutes. The schedule cron is `37 * * * *` UTC.
- At verification time `2026-11-18T19:53Z`, no `event=schedule` run had yet been observed. The first eligible trigger is `2026-11-18T20:37:00Z`, equivalent to `2026-11-19 00:37` in `Asia/Baku`. The collector itself begins only after Standard CI and capture-job bootstrap; exact data start/end are defined by each manifest, not by cron trigger time.
- Two one-time follow-up automations were created: first-run evidence check `Automation_8c414a2a09a881919874501cc6f25817` for `2026-11-19 01:53:21 Asia/Baku`; final 32-hour soak audit `Automation_81946c6e81008191823373744329ed13` for `2026-11-20 07:49:15 Asia/Baku`. They are operational reminders/checks, not evidence that any capture has already run.
- M1.5 operational activation is therefore **REGISTERED / ACTIVE, FIRST SCHEDULED CAPTURE PENDING**. M1 full acceptance and M3 authorization remain pending on actual artifact evidence.
- M3: OFI pre-registration/design implementation — NOT AUTHORIZED until the extended capture corpus passes M1 review and the owner explicitly opens M3. This ADR authorizes no OFI or DeepLOB code.
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
