# ADR-058 — Regime M Historical Microstructure Sufficiency and Live Capture

- **Status:** M2: NO DOCUMENTED PUBLIC HISTORICAL L2/TICK ARCHIVE — LIVE COLLECTION REQUIRED; M1: NOT ACCEPTED — SEVEN SCHEDULED SEGMENTS, LESS THAN 24 HOURS MAXIMUM CAPTURE BOUND; M1.5: INSUFFICIENT / FINAL ARTIFACT CONTENT AUDIT PARTIAL; M3: NOT AUTHORIZED
- **Date:** 2026-11-23
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

The workflow supports both a bounded `workflow_dispatch` and an hourly `schedule` trigger (`cron: 37 * * * *`, UTC). Scheduled runs capture for 55 minutes in each 60-minute cadence. Manual durations are bounded to 1, 5, 15, 30, or 55 minutes. The capture job timeout is 60 minutes; Standard CI remains a separate prerequisite job, and capture starts only after Standard CI succeeds. The cron minute is intentionally offset from the top of the hour to reduce scheduler contention.

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
- M1: NOT ACCEPTED — the five-minute pilot passed only bounded ingestion/integrity smoke criteria; the M1.5 scheduled soak produced no live segments and the >=24-hour actual-capture threshold was not met.
- M1.5: OWNER-APPROVED DESIGN, OPERATIONALLY FAILED — four scheduled workflow runs were recorded; all failed in Standard CI before capture and the capture job was skipped; zero scheduled live segments or M1.5 capture artifacts exist.

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
- M1.5 operational outcome is **SCHEDULE REGISTERED; FOUR SCHEDULED ATTEMPTS FAILED BEFORE CAPTURE; ZERO SOAK SEGMENTS COLLECTED**. M1 remains NOT ACCEPTED and M3 remains NOT AUTHORIZED.
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


## 8. M1.5 final soak audit (2026-11-20)

### 8.1 Repository state and complete workflow-run inventory

The audit used the GitHub Actions API for workflow ID `422952435`, path `.github/workflows/capture-microstructure.yml`, filtered to branch `research/nobitex-hf-strategy` and runs created on or after `2026-11-18T00:00:00Z`. The API returned `total_count=5` with `per_page=100`, so the single returned page contained the complete matching run set; this is not merely the five-row UI display limit.

At the start of finalization:
- Repository default branch: `research/nobitex-hf-strategy`.
- Research HEAD: `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86`; Tree SHA: `dfb8204ea988bbd6cebf763cf454adb76ede86b7`.
- Frozen main HEAD: `cd9b13d42ac0cf0edd2df066ff81580e028146d6`; Tree SHA: `5761c6c34143f8e6924f53a4e745c48932a5f791`.

| Run | Event | Created / started UTC | Updated UTC | Conclusion | Head SHA |
|---|---|---|---|---|---|
| [44369213808](https://github.com/angry0tiger-sudo/AliX/actions/runs/44369213808) | `workflow_dispatch` (manual pilot) | 2026-11-18T18:23:38Z | 2026-11-18T18:34:13Z | success | `770342e0ffa8466ec4e509b9e4188eff016f3727` |
| [44436885861](https://github.com/angry0tiger-sudo/AliX/actions/runs/44436885861) | `schedule` | 2026-11-19T01:26:13Z | 2026-11-19T01:31:09Z | failure | `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86` |
| [44496172880](https://github.com/angry0tiger-sudo/AliX/actions/runs/44496172880) | `schedule` | 2026-11-19T09:24:48Z | 2026-11-19T09:30:26Z | failure | `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86` |
| [44589431462](https://github.com/angry0tiger-sudo/AliX/actions/runs/44589431462) | `schedule` | 2026-11-19T18:35:31Z | 2026-11-19T18:43:19Z | failure | `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86` |
| [44649370365](https://github.com/angry0tiger-sudo/AliX/actions/runs/44649370365) | `schedule` | 2026-11-20T00:53:00Z | 2026-11-20T00:58:22Z | failure | `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86` |

All runs had `status=completed`; `run_started_at` equalled `created_at` for the API records. No workflow-level queued, in-progress, cancelled, or skipped run was present. All four scheduled runs failed. In each scheduled run, the `capture` job was **skipped** because its `needs: standard-ci` prerequisite did not succeed. The `real-data-diagnostic` and `ensemble-real-data-research` jobs were skipped by their existing gates. The PostgreSQL integration, four-arm lock, and Freqtrade runtime jobs succeeded in each scheduled run.

The configured schedule window from the first eligible tick `2026-11-18T20:37:00Z` through the final planned tick `2026-11-20T03:37:00Z` contains **32 nominal hourly slots** (inclusive). Only four `event=schedule` run records exist, so 28 nominal slots have no corresponding run record by count. The four API `created_at` timestamps do not themselves fall exactly on minute 37; GitHub's returned run metadata does not identify the intended cron slot for each delayed/early event. Therefore no unsupported one-to-one mapping of those four events to particular cron slots is asserted. The final tick at `2026-11-20T03:37:00Z` had no corresponding run record by the audit cutoff `2026-11-20T03:53:33Z`.

### 8.2 Scheduled-run failure and root cause

All four scheduled runs failed in `standard-ci / quality` at the same test:

`tests/unit/test_capture_nobitex_microstructure.py::test_manifest_contains_sha256_record_counts_and_safety`

The observed exception was:

`ValueError: SCHEDULED_CAPTURE_CADENCE_OR_DURATION_INVALID`

Each run reported `1 failed, 1078 passed, 5 skipped`; total measured coverage was 90.36%. The failure is an environment-isolation defect in this unit test path: the reusable Standard CI workflow inherits `GITHUB_EVENT_NAME=schedule` from the scheduled caller, while this generic manifest test does not set a schedule interval. The collector helper therefore sees the scheduled event with its default `ALIX_SCHEDULE_INTERVAL_MINUTES=0` and rejects the test's one-minute fixture. The separate scheduled-coverage test explicitly sets interval 60 and passes. This explains why the scheduled runs never reached capture; it is not evidence of a live Nobitex connection failure. No gate was weakened and no test was bypassed.

The appropriate remediation is to isolate the generic manifest test from the caller's event context (or explicitly provide a valid interval when testing scheduled metadata), retain the scheduled cadence/duration guard, and rerun Standard CI before restarting a new soak. No source or test code was changed during this final audit.

### 8.3 Complete artifact inventory and independent ZIP verification

All 13 artifacts associated with the five workflow runs were downloaded. For every ZIP, independent SHA-256 matched GitHub's artifact `digest`, the downloaded byte count matched GitHub's `size_in_bytes`, and ZIP CRC testing passed.

| Run ID | Artifact ID | Artifact name | ZIP bytes | ZIP SHA-256 |
|---|---:|---|---:|---|
| 44369213808 | 14458723722 | `regime-m-microstructure-44369213808` | 19,585 | `4aa8d41e14d3083158442eeee4aa730738920e44067b1e9aba3f7f7b2af8c472` |
| 44369213808 | 14458677699 | `phase7-risk-fault-injection-report` | 1,748 | `91b3e0b3c4a66f40000aa9d1312a4a02eb0de09b6b3ff8ed635f4deb230ccc55` |
| 44369213808 | 14458547224 | `phase5-synthetic-meta-labeling-report` | 19,146 | `ef30d1bb7e7006540390ba49d0924b87c6f435a7228e9e45af5f48499cfb4358` |
| 44369213808 | 14458402199 | `phase4-synthetic-walkforward-report` | 4,317 | `1cf2a346bf4146d8895f72cdce3e92db64ce015428db095a9fff66ce8b0e89fb` |
| 44369213808 | 14457843140 | `phase9-synthetic-e2e-report` | 2,099 | `8776ec4088a52bf471e044eab3763b4f29e4e14453d6d2e3ae22a8f3b641dce6` |
| 44436885861 | 14488892754 | `phase5-synthetic-meta-labeling-report` | 19,075 | `3e86e77d5e8bf190dca69ae8e1aa546ee4954ecfeb56295417d5e79140b038c8` |
| 44436885861 | 14488469461 | `phase4-synthetic-walkforward-report` | 4,314 | `cc58fbfbf84a593b03b8edbbf18f25ddcbcd9c8481c1666eee51b0c32810b86a` |
| 44496172880 | 14514588828 | `phase5-synthetic-meta-labeling-report` | 19,075 | `dd8e813abaeae3dd6cd16817e28ac4c7b158a25be1d7b8e766632a309c6bfe8a` |
| 44496172880 | 14513909462 | `phase4-synthetic-walkforward-report` | 4,311 | `fd1abfe9669058b62ca1c3c53e3e16213574eaaaed53a3ecec77a26c6cc4bdb9` |
| 44589431462 | 14562586659 | `phase5-synthetic-meta-labeling-report` | 19,075 | `d60d506ed42e1b3534747539f0dd54d73ad63c40565813b0dbd9f7d23f1baaa6` |
| 44589431462 | 14562596654 | `phase4-synthetic-walkforward-report` | 4,313 | `144f3ffe5b86efb428a4dcca4bd0d12350ce59f9594e4f7cb9ed1f410592479a` |
| 44649370365 | 14589883182 | `phase5-synthetic-meta-labeling-report` | 19,148 | `77bfe2102fdabe63d980cc05816a10ff5ad7780cf986cc5a41275bfba7535ce7` |
| 44649370365 | 14589913148 | `phase4-synthetic-walkforward-report` | 4,318 | `92e4446cc1690969c960547f1374aeaca606a64be566501c0e3427e1d330ba3d` |

The 12 non-capture artifacts contain synthetic Phase 4/5/7/9 JSON reports, not Nobitex microstructure segments. Those JSON members parsed successfully; they do not contain a per-artifact `manifest.json`, so manifest-level per-file count checks do not apply to them. Crucially, none of the four scheduled runs uploaded an artifact named `regime-m-microstructure-*`.

The only microstructure artifact is the manual pilot artifact `14458723722`. Its ZIP contains `orderbook.csv`, `trades.csv`, `gaps.csv`, `manifest.json`, and `manifest.sha256`. Independent verification:

| File | Records (excluding header) | Bytes | SHA-256 |
|---|---:|---:|---|
| `orderbook.csv` | 488 | 375,534 | `fd8513c37ce54d3064a325459c68274163a47a1a4c466a5414ba04d0ad913fcf` |
| `trades.csv` | 79 | 7,066 | `d524f90c140aaa034426d3028611c3cc31a29b3fab5c0588f34c16fceab8b50b` |
| `gaps.csv` | 0 | 127 | `bbaaea6d0b1a672e8814745f2f0f9c20ca7411e7d41fde18b1d43311f5d604fa` |

- `manifest.json`: 1,837 bytes; SHA-256 `3b76365ab3301bd1d92b643df3f04bef9e90aea517e6ae2df07a278f50bab219`.
- `manifest.sha256`: 80 bytes; content exactly `3b76365ab3301bd1d92b643df3f04bef9e90aea517e6ae2df07a278f50bab219  manifest.json`; its own file SHA-256 is `4f8eef9bc75f7ea1d455f37f697b5d9a09842a441fdb2ea1d7f6ef541a1e233a`.
- All three CSV byte counts, record counts, and SHA-256 values match `manifest.json`; the manifest digest matches `manifest.sha256`; the ZIP digest matches GitHub's artifact digest.

### 8.4 Actual capture, gaps, and timestamp audit

There was **zero M1.5 scheduled live capture time**. The only actual live data remains the manual five-minute pilot:

- Pilot capture window: `2026-11-18T18:29:03.963Z` to `2026-11-18T18:34:08.730Z`.
- Actual elapsed capture duration: **304.767 seconds = 5.07945 minutes**.
- Order book: **488** observations = 1 REST initial snapshot + 487 WebSocket publications.
- Trades: **79** observations = 20 REST recent-trade seed records + 59 WebSocket publications.
- Gaps: **0** rows; `gap_count=0`; `errors=[]`; `capture_status=NO_GAPS_OBSERVED`.
- Recorded reconnects: **0**.
- Order-book WebSocket offsets: 2361136–2361622, 487 values, contiguous and strictly increasing.
- Trade WebSocket offsets: 298805–298863, 59 values, contiguous and strictly increasing.
- WebSocket exchange timestamps were monotonic in both streams; no observed offset gap or non-monotonic event occurred.
- Trade deduplication: none; `duplicate_trades_dropped=0`. The public schema provides no documented stable unique trade ID, so all observations remain preserved.

The 20 REST trade seed rows span exchange timestamps `2026-11-18T18:26:52.434Z` to `2026-11-18T18:27:26.811Z`, and were 101.231–135.608 seconds old at their common capture timestamp `2026-11-18T18:29:08.042Z`. They are historical seed observations, not live five-minute trade coverage; the seed rows are descending in exchange-time order.

### 8.5 Descriptive market conditions (pilot only; no model fitting)

All observations fall in the single partial UTC hour `2026-11-18 18:00–18:59Z`; the capture itself covers only about five minutes. This is insufficient for full-session or diurnal comparison.

| UTC hour | Book observations | Median spread | P95 spread | Mid change | Mid range | Std. consecutive mid changes | Trades | Reported trade volume |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-18 18:00Z | 488 | 18.446 bps | 18.545 bps | +2.865 bps | 12.137 bps | 1.334 bps | 79 (59 WS + 20 REST seed) | 0.034186 |

The minimum observed spread was approximately 0.000000583 bps (a 10-IRR one-tick spread); the maximum was 18.545 bps. Mid-price moved from 171,341,985,200 IRR to 171,391,074,245 IRR during the sampled book sequence. The 79 trade observations comprised 46 buys and 33 sells; volume is the sum of the public `volume` field as returned, without imposing a new unit conversion. These are descriptive statistics for one short sample only, not evidence of a stable regime, predictive edge, or adequate session coverage.

### 8.6 Missing coverage and final M1/M2/M3 decision

No pair of adjacent capture-segment manifests exists: the manual pilot is the only actual capture segment and there is no subsequent `next.started_at_utc`. Consequently, no exact inter-segment interval of the form `[previous.finished_at_utc, next.started_at_utc)` can be computed. Do not substitute the nominal five-minute schedule gap for an observed interval. The verifiable no-capture interval after the pilot is open-ended at the audit cutoff: `[2026-11-18T18:34:08.730Z, 2026-11-20T03:53:33Z)`, lasting 33 hours, 19 minutes, 24.270 seconds; the upper bound is the audit cutoff, not a next segment start.

- **M1: NOT ACCEPTED.** Independently verified actual live capture is only 304.767 seconds (5.07945 minutes), versus the mandatory >=24-hour threshold. This is approximately 0.35% of the minimum. The preferred 72-hour target is also unmet. The M1.5 scheduled soak collected zero additional live observations.
- **M2: NO DOCUMENTED PUBLIC HISTORICAL L2/TICK ARCHIVE.** The earlier documented API review remains unchanged; public OHLCV history is not a substitute for L2/tick history.
- **M3: NOT AUTHORIZED.** The data does not justify owner review for OFI pre-registration. Do not begin OFI, DeepLOB, feature development, or model work. A separate remediation and new owner-approved soak are required first.
- Safety invariants remain: `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, zero private API calls, zero real/paper orders, `holdout_bytes_read=false`, and `posthack_data_read=false`. No Holdout/Post-Hack data, private endpoints, or orders were used.

After the collection window and artifact audit, the repository default-branch setting was restored in **Settings > General** from `research/nobitex-hf-strategy` to `main`. GitHub displayed the confirmation “Default branch changed to main”. A fresh repository/API check then reported `default_branch=main`; `refs/heads/main` remained exactly `cd9b13d42ac0cf0edd2df066ff81580e028146d6`, and its Tree SHA remained exactly `5761c6c34143f8e6924f53a4e745c48932a5f791`. The Research ref was unchanged at `b764a1cdd28b8b5c01f6a10a1be29a908b4d4e86` at the time of that restoration check. No main-branch commit, ref, or tree mutation occurred.

---

## 9. Owner-directed M1.5 final audit addendum — 2026-11-23

This addendum supersedes the earlier pre-soak statements in Sections 8.4–8.6 where they state that no scheduled live segments existed or that the scheduled capture failed before acquisition. The historical pilot record remains unchanged. This addendum records only the seven successful scheduled runs discovered after the 2026-11-21 audit.

### 9.1 Scope and GitHub run selection

Repository: `angry0tiger-sudo/AliX`  
Workflow: `Regime M Public Microstructure Capture`  
Workflow ID: `422952435`  
Selected event/branch: `event=schedule`, `research/nobitex-hf-strategy`  
Selected code SHA for all seven runs: `b88d30cf2b1c0ebba74cff1a60c5fe453f936711`

The GitHub Actions run collection was queried through all returned pages. Exactly seven matching scheduled runs were found after the prior audit cutoff; each completed successfully. No matching queued, in-progress, or failed run was present at the 2026-11-23 13:31 UTC audit cutoff. Earlier failed schedule runs on `b764a1c...` and the prior manual pilot are excluded from the seven-segment M1.5 totals.

| Run | Capture command start UTC (runner log) | Collector summary UTC (runner log) | Invocation span (s) | Order-book rows | Trade rows | Gap rows | Collector status | Artifact ID |
|---|---|---|---:|---:|---:|---:|---|---:|
| [44924041271](https://github.com/angry0tiger-sudo/AliX/actions/runs/44924041271) | 2026-11-21T16:26:01.449Z | 2026-11-21T17:21:07.736Z | 3306.287 | 6054 | 803 | 0 | NO_GAPS_OBSERVED | 14717007003 |
| [44953574592](https://github.com/angry0tiger-sudo/AliX/actions/runs/44953574592) | 2026-11-21T21:56:41.550Z | 2026-11-21T22:51:45.254Z | 3303.704 | 3647 | 197 | 10 | GAPPED | 14728866173 |
| [44976676483](https://github.com/angry0tiger-sudo/AliX/actions/runs/44976676483) | 2026-11-22T02:40:46.361Z | 2026-11-22T03:35:50.760Z | 3304.399 | 5572 | 167 | 0 | NO_GAPS_OBSERVED | 14737653497 |
| [45011199208](https://github.com/angry0tiger-sudo/AliX/actions/runs/45011199208) | 2026-11-22T10:16:48.256Z | 2026-11-22T11:11:58.560Z | 3310.304 | 5742 | 666 | 4 | GAPPED | 14751767157 |
| [45052497891](https://github.com/angry0tiger-sudo/AliX/actions/runs/45052497891) | 2026-11-22T18:06:06.468Z | 2026-11-22T19:01:12.294Z | 3305.826 | 4624 | 514 | 0 | NO_GAPS_OBSERVED | 14767894020 |
| [45084920861](https://github.com/angry0tiger-sudo/AliX/actions/runs/45084920861) | 2026-11-22T23:53:26.467Z | 2026-11-23T00:48:30.375Z | 3303.908 | 6161 | 108 | 0 | NO_GAPS_OBSERVED | 14781128292 |
| [45135587335](https://github.com/angry0tiger-sudo/AliX/actions/runs/45135587335) | 2026-11-23T08:12:17.553Z | 2026-11-23T09:07:21.232Z | 3303.679 | 5055 | 1270 | 0 | NO_GAPS_OBSERVED | 14802554477 |

All seven `standard-ci` prerequisite jobs and all seven `capture` jobs completed successfully. Within each capture job, the mock-only contract/gap tests, public BTCIRT capture, manifest/hash/record-count/safety validation step, and artifact upload step all completed successfully. The Standard CI sub-jobs `quality`, `four-arm-lock-bare-env`, `freqtrade-runtime-smoke`, and `postgres-integration` were successful; the real-data diagnostic and ensemble research jobs remained skipped.

### 9.2 Artifact inventory and GitHub-reported ZIP integrity metadata

Only the seven artifacts beginning `regime-m-microstructure-` are included below. Synthetic Phase 4/5/7/9 artifacts are excluded.

| Run | Artifact name | Artifact bytes | Upload Action SHA-256 / GitHub artifact API digest |
|---|---|---:|---|
| 44924041271 | `regime-m-microstructure-20261121T162601Z-44924041271` | 220704 | `e7b3bb905814de2e9373d29998c0c6862073aa09abff5858bbcbd570b7741636` |
| 44953574592 | `regime-m-microstructure-20261121T215641Z-44953574592` | 94228 | `f2792953e63431331c92d97ace8fcec8cc37da5980a32436a248a2d5be67e7e4` |
| 44976676483 | `regime-m-microstructure-20261122T024046Z-44976676483` | 164027 | `2fc752295eddb105a0e52b0cfb36b0b352ba7402335a8bfaddf2ce9ea4f88093` |
| 45011199208 | `regime-m-microstructure-20261122T101648Z-45011199208` | 180165 | `402c9ebd7e4f300ce6ef6350aca97d54f93d9b01d9ef1b36fe2a10047616d5fd` |
| 45052497891 | `regime-m-microstructure-20261122T180606Z-45052497891` | 134770 | `369d72f9e25ee4c359d70001d90a3e517448af33dfa173311b98455b2f9bb43c` |
| 45084920861 | `regime-m-microstructure-20261122T235326Z-45084920861` | 226449 | `a0cd641866eb780b105a0b9461e8b168c79de4db5ecb55504c7f1cbbddcc24f6` |
| 45135587335 | `regime-m-microstructure-20261123T081217Z-45135587335` | 166440 | `2675c142d876f4131162ecac122fa14900c3691ffa0fdf1b554c104173602b86` |

All seven ZIPs were retrieved using the GitHub artifact-download operation. The SHA-256 printed by `actions/upload-artifact@v4` in each capture log equals the digest returned by GitHub's artifact metadata API; the artifact sizes also agree. These are GitHub-side upload/API cross-checks, not an independent recomputation over the downloaded ZIP bytes.

**Artifact-content audit limitation:** no existing dedicated cloud workflow in this branch accepts these seven historical artifact IDs and reopens their ZIP contents for a second independent audit. The project handoff prohibits local test/research/data-analysis execution. Therefore, this audit did not run a local ZIP extraction or claim independent byte-for-byte recomputation of the seven manifests and CSVs. Each capture workflow's own `Validate manifest hashes, record counts, and fail-closed invariants` step did complete successfully, meaning the in-run assertions for manifest/file SHA-256, byte counts, record counts, and safety flags passed. This internal validation is preserved as evidence but is not relabelled as an independent post-download recomputation.

The downloadable ZIP SHA-256 values above must not be represented as independently recomputed by this addendum. The per-CSV digests, exact `manifest.sha256` values, REST/WS split, reconnect counts, and exact Manifest start/end timestamps remain pending independent artifact-content inspection in an authorized GitHub Actions audit environment.

### 9.3 Runner-emitted observations

The capture command emitted the following record totals:

- Order-book observations: **36,855**.
- Trade observations: **3,725**.
- Gap records: **14**.
- Duplicate trades dropped: **0** (per each collector summary).
- Segment status: five `NO_GAPS_OBSERVED`; two `GAPPED` (Runs 44953574592 and 45011199208).

These totals are sourced from the collector JSON summaries in the GitHub Actions capture logs. They are runner-emitted counts, not a second-pass recount of the downloaded CSV files. The two GAPPED segments must not be described as gap-free. The exact gap reasons and reconnect count are not exposed in the concise runner summaries; without reading `gaps.csv` and the Manifest `errors` array, reconnect count is **UNKNOWN / NOT INDEPENDENTLY VERIFIED**, not zero.

### 9.4 Capture-time bound and 24-hour acceptance

The timestamps in the table above delimit the full runner-side collector command invocation, from the GitHub log line launching `python -m scripts.capture_nobitex_microstructure` to the collector's final JSON summary. They include any command-entry/exit overhead and therefore form an upper bound on the Manifest's actual capture interval, not a substitute for `manifest.started_at_utc` and `manifest.finished_at_utc`.

- Sum of all seven runner command invocation spans: **23,138.107 seconds = 385.635 minutes = 6.4273 hours**.
- This is an upper bound for actual captured seconds. The Manifest capture intervals must lie within their respective command invocations.
- Even this upper bound is far below the mandatory 24-hour actual-capture threshold.
- No 24-hour or 48-hour continuous stream is inferred. Seven bounded segments are not concatenated.
- Exact inter-segment uncovered boundaries `[previous.finished_at_utc, next.started_at_utc)` are **NOT COMPUTED** in this addendum because those exact Manifest timestamps were not independently read. Run/command timestamps are not substituted for Manifest boundaries.

The earliest verified scheduled capture command began at `2026-11-21T16:26:01.449Z`. Its Manifest start can only be at or after that command start. Consequently, the earliest possible 48-hour point is no earlier than `2026-11-23T16:26:01.449Z`. At the audit cutoff `2026-11-23T13:31:00Z`, 48 elapsed hours from the first actual capture had **not** yet been established. This timing fact does not change the M1 verdict because the seven segments' total capture-time upper bound is already below 24 hours.

### 9.5 Safety invariants

All seven capture jobs passed their explicit workflow validation step. That step asserts:
- `authorization_header_sent == false`;
- `websocket_connection_token_sent == false`;
- `private_api_calls == 0`;
- `holdout_bytes_read == false`;
- `posthack_data_read == false`;
- `real_order_attempts == 0`;
- `paper_orders == 0`;
- `Champion == null`;
- `order_permission == false`;
- `trading_authorization == "NO-GO"`.

The passing validation steps support the safety invariants for these seven runs. No model, OFI/DeepLOB pipeline, Holdout/Post-Hack access, private endpoint, or paper/real order was authorized or invoked by this audit.

### 9.6 Updated decision

- **M1: NOT ACCEPTED.** Seven successful scheduled segments exist, but the upper bound from all collector command invocation windows is only 6.4273 hours; actual Manifest capture time is lower. This is below the required 24 hours. The runner-reported 14 gap rows and two GAPPED segments also require further data-quality investigation.
- **M1.5: INSUFFICIENT / NOT COMPLETE.** The scheduled collector runs and produced seven artifacts, but the required 24-hour actual-capture Data Gate was not met; 48 elapsed hours from the first successful segment had not yet elapsed at this audit cutoff.
- **M2: UNCHANGED.** No documented public historical L2/tick archive was established; historical OHLCV is not a substitute.
- **M3: NOT AUTHORIZED.** No owner-review request for OFI pre-registration is justified by this incomplete Data Gate. Do not build or train OFI/DeepLOB or any other model.
- **No M1 acceptance, model authorization, paper authorization, or trading authorization is created by successful Workflow runs or by this ADR update.**

### 9.7 Repository state at audit cutoff

- Research HEAD before this ADR addendum: `b88d30cf2b1c0ebba74cff1a60c5fe453f936711`.
- `main` HEAD: `cd9b13d42ac0cf0edd2df066ff81580e028146d6`.
- `main` Tree SHA: `5761c6c34143f8e6924f53a4e745c48932a5f791`.
- Repository `default_branch` remained `research/nobitex-hf-strategy` at the time this addendum was prepared.
- No main ref, commit, or tree mutation is part of this addendum.

The only intended file change for this final-audit commit is this ADR-058 document. Standard CI must pass on the resulting Research commit before the repository default-branch setting is restored to `main`. After restoration, verify `default_branch=main` and re-check the exact frozen main HEAD and Tree SHA listed above.



---

## 10. Owner-directed five-run manual Regime M soak final audit — 2026-11-25

### 10.1 Scope, evidence boundary, and decision

Repository: angry0tiger-sudo/AliX  
Workflow: Regime M Public Microstructure Capture  
Workflow ID: 422952435  
Workflow path: .github/workflows/capture-microstructure.yml  
Selected branch: research/nobitex-hf-strategy  
Selected workflow code SHA: cad729d0aa68eca65eddd67b39eaedc604b63cf6  
Market: BTCIRT  
Audit method: GitHub Actions run/job/step/log and artifact metadata inspection, followed by independent download and byte-level ZIP/CSV/manifest verification in the audit environment.

The GitHub Actions API returned exactly five repository workflow_dispatch runs created from 2026-11-23 through 2026-11-25. All five belonged to workflow 422952435, path .github/workflows/capture-microstructure.yml, branch research/nobitex-hf-strategy, and code SHA cad729d0aa68eca65eddd67b39eaedc604b63cf6. All five completed with conclusion success and run_attempt=1. The GitHub workflow-level run_number values 14–18 are repository workflow counters, not the chained input run_number values; the latter were independently read from each capture job's RUN_NUMBER environment in its logs.

The single owner/manual seed dispatch was run 45198041920, created 2026-11-23T14:47:50Z, with RUN_NUMBER=1 and CAPTURE_DURATION_MINUTES=350. Runs 2–5 were dispatched by the preceding workflow chain step. No manual dispatch, rerun, or retry was issued during this audit.

**Decision: M1 remains NOT ACCEPTED.** The sum of the five independently verified Manifest capture intervals is 105,018.751 seconds (29 hours, 10 minutes, 18.751 seconds), which exceeds the 24-hour minimum by 18,618.751 seconds. It does not meet the preferred 72-hour target. However, the mandatory data-integrity/reconnect-recovery gate fails: segment 4 records 28,978 non-monotonic publisher-offset events after reconnect, with the accepted WebSocket streams effectively ceasing to advance after publisher offsets reset. Those events are explicitly rejected and preserved as gap rows, not silently repaired. The five-run time threshold therefore does not establish an acceptable 24-hour microstructure corpus.

### 10.2 Exact five-run chain and workflow outcomes

| Chain input | GitHub run ID | API workflow run number | Created UTC | Capture RUN_NUMBER / duration | Head SHA | Overall |
|---:|---:|---:|---|---|---|---|
| 1 (manual seed) | [45198041920](https://github.com/angry0tiger-sudo/AliX/actions/runs/45198041920) | 14 | 2026-11-23T14:47:50Z | 1 / 350 minutes | cad729d0aa68eca65eddd67b39eaedc604b63cf6 | success |
| 2 (chained) | [45264275546](https://github.com/angry0tiger-sudo/AliX/actions/runs/45264275546) | 15 | 2026-11-23T20:55:06Z | 2 / 350 minutes | cad729d0aa68eca65eddd67b39eaedc604b63cf6 | success |
| 3 (chained) | [45316779276](https://github.com/angry0tiger-sudo/AliX/actions/runs/45316779276) | 16 | 2026-11-24T03:03:24Z | 3 / 350 minutes | cad729d0aa68eca65eddd67b39eaedc604b63cf6 | success |
| 4 (chained) | [45362430692](https://github.com/angry0tiger-sudo/AliX/actions/runs/45362430692) | 17 | 2026-11-24T09:09:00Z | 4 / 350 minutes | cad729d0aa68eca65eddd67b39eaedc604b63cf6 | success |
| 5 (chained final) | [45423741331](https://github.com/angry0tiger-sudo/AliX/actions/runs/45423741331) | 18 | 2026-11-24T15:17:16Z | 5 / 350 minutes | cad729d0aa68eca65eddd67b39eaedc604b63cf6 | success |

For every run:
- Standard CI prerequisite: all four jobs succeeded — quality, four-arm-lock-bare-env, freqtrade-runtime-smoke, and postgres-integration.
- Capture job: succeeded.
- Bounded-input validation: succeeded.
- Mock-only capture and auto-audit tests: succeeded.
- Public BTCIRT capture: completed.
- Long-form manifest and safety validation: succeeded.
- Auto-audit CSV counts, SHA-256, gaps, reconnect counters, and safety invariants: succeeded.
- Raw artifact upload: succeeded.
- real-data-diagnostic and ensemble-real-data-research remained skipped by their existing branch/event gates; no real-data model workflow ran.

The chain-step job succeeded in all five runs. Runs 1–4 each completed the 600-second sleep and logged the successful dispatch notice for exactly the next input segment:
- Run 1 dispatched segment 2/5 at 2026-11-23T20:55:06.412Z.
- Run 2 dispatched segment 3/5 at 2026-11-24T03:03:25.241Z.
- Run 3 dispatched segment 4/5 at 2026-11-24T09:09:00.838Z.
- Run 4 dispatched segment 5/5 at 2026-11-24T15:17:17.181Z.
- Run 5 completed “Stop after the fifth successful segment”; both the sleep and dispatch steps were skipped. No sixth dispatch was issued.

The run API and capture logs agree on the exact five run IDs, input values 1–5, duration 350, selected branch, and selected code SHA. The capture logs expose CAPTURE_DURATION_MINUTES=350 and RUN_NUMBER=1,2,3,4,5 respectively. Each Manifest independently records requested_duration_minutes=350 and the matching workflow_run_id.

### 10.3 Artifact inventory and independent ZIP verification

Only the matching regime-m-microstructure artifacts are listed. Synthetic Phase 4/5/7/9 artifacts uploaded by Standard CI are not capture evidence and are excluded.

| Run ID | Artifact ID | Artifact name | GitHub size / downloaded bytes | GitHub digest / independently computed ZIP SHA-256 |
|---|---:|---|---:|---|
| 45198041920 | 14861162379 | regime-m-microstructure-20261123T145451Z-45198041920 | 1,047,871 | 6d072cb56df8e0fc00129dfb171558630f84659197055d5fb91861a401d25674 |
| 45264275546 | 14884828194 | regime-m-microstructure-20261123T210308Z-45264275546 | 1,110,846 | b87d26a8a4b41347e36c5bd7b66a43630b9c34d40d8bdc725fb3749e6bedc0cc |
| 45316779276 | 14904127543 | regime-m-microstructure-20261124T030844Z-45316779276 | 963,631 | a45bf0d9d49bcf4ddfc5c3e86cba015a59ec87c60f9a28aec2b832d06595155c |
| 45362430692 | 14933304229 | regime-m-microstructure-20261124T091658Z-45362430692 | 802,826 | f7ec6fd5cab07d147aa0968ac17ff1abf5228910ec5a43896d6a7d694610f3c5 |
| 45423741331 | 14966045300 | regime-m-microstructure-20261124T152549Z-45423741331 | 1,226,509 | 1c297ed33aee97d9c77572f7ae9c41fa90c5dfb4ec1291053ee347dc0bca800e |

All five artifacts were downloaded by their exact GitHub artifact IDs. For every ZIP:
- Independently computed byte count equals GitHub artifact size_in_bytes.
- Independently computed SHA-256 equals the GitHub artifact API digest and the upload-action digest in the capture log.
- ZIP CRC test passed.
- The ZIP contains exactly gaps.csv, manifest.json, manifest.sha256, orderbook.csv, and trades.csv.
- Manifest workflow_run_id and code_commit_sha match the enclosing run and selected commit.
- manifest.sha256 contains the exact SHA-256 of the downloaded manifest.json followed by two spaces and manifest.json.

| Run ID | manifest.json SHA-256 |
|---|---|
| 45198041920 | 1e5999252e979b1dea3f512fcd5019766b8a3658c9aea33c44260ab41276df40 |
| 45264275546 | 2e0e554e3fee1d5148eb04f3054aa28c079a4ef184d41e379b5bd2f124209b06 |
| 45316779276 | 99ea61613b464b45b289a74dedd62bc482fbc489fc257e438602b40c2ce3a9a1 |
| 45362430692 | 6b86e214aaf1cacf4b9733ae0fe47a107bf0d7c50575da39a9d28d98da227f5c |
| 45423741331 | e8af624bcb9b81e530c1d3a9e0aef5fbdcdedbbce63e9c546ebbb319a89a2bd5 |

### 10.4 CSV contract, byte counts, row counts, and checksums

The exact CSV headers were identical in all five artifacts:

- orderbook.csv: captured_at_utc,source,exchange_timestamp_ms,offset,best_bid_price,best_bid_size,best_ask_price,best_ask_size,spread,mid_price,bids_top_json,asks_top_json
- trades.csv: captured_at_utc,source,exchange_timestamp_ms,side,price,volume,offset
- gaps.csv: stream,gap_start_utc,gap_end_utc,gap_start_ms,gap_end_ms,reason,previous_offset,next_offset,replay_attempted,replay_available

For every CSV in every artifact, independent byte count, parsed data-row count excluding the header, and SHA-256 equal the corresponding manifest.json files metadata and records values. The independent audit also confirmed record_counts and gap_count agree with the actual CSV rows.

| Run ID | File | Data rows | Bytes | SHA-256 |
|---|---|---:|---:|---|
| 45198041920 | orderbook.csv | 31,796 | 24,530,356 | e831ae32cc0bef497d7a20b135ac942257a42d987888734e8d5d9c0a15e38660 |
| 45198041920 | trades.csv | 5,186 | 458,130 | 06d56c17140d7672520ca07e731e822730f61053ca1efce242662e2360bb00cc |
| 45198041920 | gaps.csv | 4 | 651 | 86e605e57f00e858d1e938b1ac85959d2740950e36cde6f8914924638f7e2c00 |
| 45264275546 | orderbook.csv | 34,163 | 26,340,068 | 88de2566e09f0eb1819a6cf96b77305c8ed0e7610d143301d66d241a833a0629 |
| 45264275546 | trades.csv | 4,839 | 429,382 | 924302082a3f3f65fe70f242d2b2009e916c5f8c00e51e1f11cae8e551535779 |
| 45264275546 | gaps.csv | 7 | 1,032 | d00e2e6746e7d61f0d7eedbd45b5984baea8081c20b45947ea35a0f7f4cf4a73 |
| 45316779276 | orderbook.csv | 31,125 | 24,031,252 | fe541695d1dd065e8ca190f5e98b427252ca23efae20fef343e86c6a004bfaec |
| 45316779276 | trades.csv | 6,223 | 550,961 | 06be6b1375863b0e77690ea363df45b05576434264085cb141e3d0f6e80d55a5 |
| 45316779276 | gaps.csv | 4 | 651 | ae5d503e6dd7c03e2f9ec2ecc423ca82b2c9bb0812d1dbc843e7b4839884c2b3 |
| 45362430692 | orderbook.csv | 11,363 | 8,763,525 | f0e3796af0184759082c0fe497e4ba80cc6ed8e93c752bc4db81f5fa10ecd865 |
| 45362430692 | trades.csv | 1,700 | 150,025 | 3b2014f02cee068a35eb2308c45c95615308d8d670c4f3814670b52fb2dc8bce |
| 45362430692 | gaps.csv | 28,982 | 4,486,516 | cac705981f44cea1aba5e0b1002560c04e9edb76c406f0d0c0c31d20cea10efe |
| 45423741331 | orderbook.csv | 36,348 | 27,904,654 | b8ecaf25f53fb24d16da38e8f129e3485e5e1ceb4306a5daa1df1d9bd77c84b8 |
| 45423741331 | trades.csv | 3,493 | 301,168 | adf983c6cc030d2d14a076853427bd08ed0c10e97433a5187db84ee4b841fc2e |
| 45423741331 | gaps.csv | 0 | 127 | bbaaea6d0b1a672e8814745f2f0f9c20ca7411e7d41fde18b1d43311f5d604fa |

Aggregate independently recounted rows: 144,795 order-book observations, 21,441 trade observations, and 28,997 gap rows. The trade files contain 100 REST recent-trade seed rows in total and 21,341 WebSocket trade observations. The order-book files contain 5 initial REST snapshots, 6 REST reconnect snapshots, and 144,784 WebSocket order-book observations. No trade deduplication was performed; duplicate_trades_dropped is zero in every manifest.

### 10.5 Actual capture intervals and separate uncovered intervals

Capture duration below is calculated only as Manifest finished_at_utc minus Manifest started_at_utc. No wall-clock span or inter-segment downtime is added to capture time.

| Run ID | Manifest started_at_utc | Manifest finished_at_utc | Actual interval seconds | Actual interval |
|---|---|---|---:|---|
| 45198041920 | 2026-11-23T14:54:51.433+00:00 | 2026-11-23T20:44:54.611+00:00 | 21,003.178 | 350.053 minutes |
| 45264275546 | 2026-11-23T21:03:08.461+00:00 | 2026-11-24T02:53:12.689+00:00 | 21,004.228 | 350.070 minutes |
| 45316779276 | 2026-11-24T03:08:44.606+00:00 | 2026-11-24T08:58:49.190+00:00 | 21,004.584 | 350.076 minutes |
| 45362430692 | 2026-11-24T09:16:58.846+00:00 | 2026-11-24T15:07:01.654+00:00 | 21,002.808 | 350.047 minutes |
| 45423741331 | 2026-11-24T15:25:49.265+00:00 | 2026-11-24T21:15:53.218+00:00 | 21,003.953 | 350.066 minutes |

**Total actual Manifest capture interval: 105,018.751 seconds = 1,750.3125167 minutes = 29 hours, 10 minutes, 18.751 seconds.** This exceeds the 24-hour minimum by 18,618.751 seconds (5 hours, 10 minutes, 18.751 seconds). The preferred 72-hour target is short by 154,181.249 seconds (42 hours, 49 minutes, 41.249 seconds).

The exact uncovered intervals are computed separately as [previous Manifest finished_at_utc, next Manifest started_at_utc):

| Previous run → next run | Previous finish UTC | Next start UTC | Uncovered seconds | Uncovered interval |
|---|---|---|---:|---|
| 45198041920 → 45264275546 | 2026-11-23T20:44:54.611+00:00 | 2026-11-23T21:03:08.461+00:00 | 1,093.850 | 18m 13.850s |
| 45264275546 → 45316779276 | 2026-11-24T02:53:12.689+00:00 | 2026-11-24T03:08:44.606+00:00 | 931.917 | 15m 31.917s |
| 45316779276 → 45362430692 | 2026-11-24T08:58:49.190+00:00 | 2026-11-24T09:16:58.846+00:00 | 1,089.656 | 18m 09.656s |
| 45362430692 → 45423741331 | 2026-11-24T15:07:01.654+00:00 | 2026-11-24T15:25:49.265+00:00 | 1,127.611 | 18m 47.611s |

Total uncovered inter-segment time: **4,243.034 seconds = 1 hour, 10 minutes, 43.034 seconds**. Each workflow deliberately slept for 600 seconds, but the actual Manifest boundaries also include subsequent workflow dispatch, Standard CI, and capture bootstrap latency. The uncovered intervals are not capture time and were not counted toward the 24-hour threshold. The five segments are not continuous.

### 10.6 Gap-row and reconnect reconciliation

All gap rows independently parsed from gaps.csv have replay_attempted=false and replay_available=false. The exact reason totals across all five segments are:
- websocket_disconnect_no_replay: 12 rows.
- publisher_offset_gap: 7 rows.
- non_monotonic_publisher_offset: 28,978 rows.
- Total: 28,997 rows (24,491 orderbook-stream rows and 4,506 trade-stream rows).

| Run ID | Capture status | Gap rows | Gap-row detail | Reconnect attempts / successes / failures |
|---|---|---:|---|---|
| 45198041920 | GAPPED | 4 | 2 websocket_disconnect_no_replay; 2 publisher_offset_gap | 1 / 1 / 0 |
| 45264275546 | GAPPED | 7 | 4 websocket_disconnect_no_replay; 3 publisher_offset_gap | 2 / 2 / 0 |
| 45316779276 | GAPPED | 4 | 2 websocket_disconnect_no_replay; 2 publisher_offset_gap | 2 / 1 / 1 |
| 45362430692 | GAPPED | 28,982 | 4 websocket_disconnect_no_replay; 28,978 non_monotonic_publisher_offset | 5 / 2 / 3 |
| 45423741331 | NO_GAPS_OBSERVED | 0 | No gap data rows | 0 / 0 / 0 |
| **Total** | **4 GAPPED; 1 NO_GAPS_OBSERVED** | **28,997** | **12 disconnect; 7 offset-gap; 28,978 non-monotonic** | **10 / 6 / 4** |

The 12 disconnect rows are stream-specific records for disconnect windows, not 12 independent socket disconnects. The reconnect counters are separately sourced from each Manifest and satisfy attempts = successes + failures for every run. Run 3 recorded reconnect_failed:TimeoutError. Run 4 recorded three reconnect_failed:WebSocketBadStatusException entries and two websocket_disconnect events.

### 10.7 Segment 4 data-integrity failure and reconnect-baseline finding

Run 45362430692 is the decisive data-quality failure:
- It has 28,982 gap rows, of which 28,978 are non_monotonic_publisher_offset: 24,481 orderbook and 4,497 trade events.
- The first rejected order-book publication had previous_offset=3151961 and next_offset=12; subsequent rejected offsets increased only to 24500. The first rejected trade publication had previous_offset=396001 and next_offset=1; subsequent rejected offsets increased only to 4500.
- The non-monotonic gap rows span gap_start_utc 2026-11-24T11:10:26.759+00:00 through 2026-11-24T15:07:01.453+00:00, approximately 3 hours, 56 minutes, 34.694 seconds of runner UTC time. Their exchange-time span is 14,235.270 seconds.
- The accepted orderbook.csv ends at captured_at_utc=2026-11-24T11:38:34.719+00:00 and contains 11,360 WebSocket rows plus two REST reconnect snapshots and one initial REST snapshot. The accepted trades.csv ends at captured_at_utc=2026-11-24T11:09:46.226+00:00 and contains 1,680 WebSocket trades plus 20 REST seed rows. The post-reset publications were rejected, not appended as valid observations.
- Manifest counters show five reconnect attempts, two successes, and three failures. The Manifest also records websocket_disconnect and three reconnect_failed:WebSocketBadStatusException errors.

The source-level mechanism is consistent with the observed failure. CaptureState retains one OffsetTracker instance. _reconnect reuses state.offsets when processing publications from a replacement socket; it does not establish a new publisher-offset baseline for either stream. OffsetTracker.observe returns a non_monotonic_publisher_offset gap for any offset less than or equal to the last accepted offset and does not advance its stored baseline in that case. After the observed publisher reset from 3151961 to 12 for orderbook and from 396001 to 1 for trades, the incrementing post-reset offsets therefore remained below the retained previous values and were rejected. This is not a ZIP/CSV hash failure; it is a collector reconnect-state/data-continuity failure.

The in-run auto-audit passed because its implemented gates validate exact CSV headers, byte counts, hashes, record counts, manifest checksum, reconnect-counter arithmetic, positive capture interval, and safety invariants. It does not impose a maximum gap count, require zero non-monotonic events, or prove that the offset tracker recovers after a publisher reset. Consequently, a successful Auto-Audit job is not sufficient to pass the broader M1 data-quality gate.

This failure must not be hidden by the 29.17-hour sum of Manifest intervals. The capture-time interval calculation and the observed feed-integrity result are separate evidence dimensions.

### 10.8 Safety invariants and prohibited work

All five independently inspected manifests and all five successful workflow validation steps report the required fail-closed values:

| Safety field | Required value | All five runs |
|---|---|---|
| authorization_header_sent | false | PASS |
| websocket_connection_token_sent | false | PASS |
| private_api_calls | 0 | PASS |
| holdout_bytes_read | false | PASS |
| posthack_data_read | false | PASS |
| real_order_attempts | 0 | PASS |
| paper_orders | 0 | PASS |
| Champion | null | PASS |
| order_permission | false | PASS |
| trading_authorization | NO-GO | PASS |

The workflow also set ALIX_DRY_RUN=true and ALIX_ALLOW_LIVE_TRADING=false. This audit read only the five public-feed capture artifacts and repository workflow/collector/ADR source. No Holdout or Post-Hack bytes were accessed. No private endpoint or credential was used. No real or paper order was attempted. No OFI, DeepLOB, or any other model was built or trained.

### 10.9 Final M1/M2/M3 decision and required remediation

- **M1: NOT ACCEPTED.** The five Manifest intervals sum to 29 hours, 10 minutes, 18.751 seconds and pass the 24-hour interval threshold. The 72-hour preferred target was not met. More importantly, the mandatory feed-integrity/reconnect-recovery gate fails because Run 4 rejected 28,978 non-monotonic publications after the publisher offsets reset, with no replay and no effective recovery of the accepted WebSocket streams. The corpus must not be represented as an accepted 24-hour microstructure dataset.
- **M2: UNCHANGED.** No documented dependable six-month public historical L2 or market-wide tick archive has been established. Historical OHLCV is not a substitute.
- **M3: NOT AUTHORIZED.** Do not begin OFI, DeepLOB, feature engineering, model fitting, model evaluation, or strategy implementation. No model authorization follows from this five-run chain.
- Required remediation before another owner-reviewed M1 attempt: fix the reconnect offset-baseline lifecycle for both streams; add deterministic tests for publisher-offset reset after reconnect, including independent per-stream baselines and proof that valid post-reconnect publications resume; retain all rejected observations as explicit gaps; then obtain owner approval for a new bounded soak and independently audit its complete artifacts. Do not silently reinterpret or stitch this corpus into continuous coverage.

At the time of this addendum, repository default_branch was research/nobitex-hf-strategy. The frozen main ref was independently checked at HEAD cd9b13d42ac0cf0edd2df066ff81580e028146d6 and Tree SHA 5761c6c34143f8e6924f53a4e745c48932a5f791. No main-branch commit, ref, or tree mutation is authorized. The default-branch setting must be restored to main after Standard CI for this ADR-only audit commit, and the exact main HEAD and Tree SHA must be rechecked after restoration.

