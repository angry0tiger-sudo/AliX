# ADR-058 — Regime M Historical Microstructure Sufficiency and Live Capture

- **Status:** M2: NO DOCUMENTED PUBLIC HISTORICAL L2/TICK ARCHIVE — LIVE COLLECTION REQUIRED; M1: NOT ACCEPTED — ONLY A 5.08-MINUTE MANUAL PILOT; M1.5 SCHEDULED CAPTURE FAILED BEFORE DATA ACQUISITION; M3: NOT AUTHORIZED
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
