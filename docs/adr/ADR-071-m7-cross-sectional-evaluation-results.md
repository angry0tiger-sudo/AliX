# ADR-071 — M7 Cross-Sectional Evaluation: BLOCKED_DATA / NO-GO

- **Status:** CLOSED — NO-GO / INSUFFICIENT_DATA
- **Date:** 2026-11-27
- **Branch:** `research/nobitex-hf-strategy`
- **Owner decision:** Path B (Cross-Sectional Lead-Lag / Multi-Asset Rotation), as recorded in ADR-070.
- **Scope:** The two and only two preregistered M7 logical candidates. No model or strategy redesign is authorized by this ADR.

## 1. Executive decision

The M7 public-data acquisition workflow completed successfully, but it qualified only **9 of the 10 fixed candidate symbols** for the exact preregistered historical window. `SOLUSDT` failed the exact-history coverage check with 25,285 rows against 31,801 required rows (6,516 missing hourly intervals).

Because `SOLUSDT` is mandatory in both preregistered universes, both M7 candidates were reported as `BLOCKED_DATA`. The locked evaluation workflow completed successfully and returned the family verdict **`NO-GO`**. No model fitting, walk-forward economic scoring, or portfolio simulation was performed for either candidate.

This is a **data-eligibility NO-GO**, not empirical evidence that Ridge or cross-sectional momentum loses money. NetPF, MaxDD and completed-trade counts are therefore **not available** for either cost scenario. They must not be represented as zero, inferred, or estimated from a reduced universe.

## 2. Evidence and immutable execution record

### 2.1 Acquisition

- Acquisition workflow: [M7 Nobitex PRE-HACK Cross-Asset Data Acquisition, Run #3](https://github.com/angry0tiger-sudo/AliX/actions/runs/46046806684)
- Run ID: `46046806684`
- Workflow conclusion: `success`
- Data acquisition workflow source commit: `452d12d7175831b4793c71240d3e16c1cd915ffa`
- Immutable data commit: `5cf8f867385b6481bb47c8dc560ab160b39aee5d`
- Aggregate qualification manifest: `data/nobitex/prehack/m7/m7_data_qualification.json`
- Aggregate manifest Git blob SHA: `f47a6968ab42dc9f578aeca1cd7aae4c7e555624`
- Dataset Artifact: `alix-m7-nobitex-prehack-data`
- Artifact ID: `15224885056`
- Artifact SHA-256: `139b4a11613abdd4a7c215b18126d0e81d81d003fde4c1a7352825235c0ed503`

The acquisition workflow's parent Standard CI gate passed. All acquired CSVs were committed without modifying or overwriting existing data. The output directory was required to be empty before acquisition.

### 2.2 Data window and qualification

All required histories were requested from Nobitex public Spot endpoints only, on the exact one-hour BTCUSDT time grid:

- Start: `2021-10-31T20:30:00Z` (`1635712200`)
- End: `2025-06-17T20:30:00Z` (`1750192200`)
- Required observations per symbol: `31,801`
- Expected cadence: `1h`
- Allowed gaps, duplicates, off-grid timestamps or interpolation: `0`

Nine symbols qualified, each with 31,801 rows, no missing intervals and a per-symbol SHA-256 manifest:

| Symbol | Rows | CSV SHA-256 |
|---|---:|---|
| ETHUSDT | 31,801 | `c240ee0d3d78ada57c697e9520a06f6291121c554c354efc00bad2c2132c8b19` |
| BNBUSDT | 31,801 | `32fbebf9f19c2f3bc56460b6972cac9f3938926f6e6125b6d697313557ebe2da` |
| ADAUSDT | 31,801 | `d3ee43e9accbcf21c4a831e74a304abf623e5bede85a5a4680ff041a93f62bca` |
| XRPUSDT | 31,801 | `4fe43e9fbcf5892e7a60a0d2aa6e88751a94bc0f536e603358026bf8802dd966` |
| DOGEUSDT | 31,801 | `583801d4d86504a74417f60c56888fa36e1f63aadaad754178aaaf783c1c70c4` |
| TRXUSDT | 31,801 | `a3b8bb113b3ab14060f3bc0e36808bf2408af4c625fbc7a8b8d7d1a902efc9e1` |
| LTCUSDT | 31,801 | `255928a77f8ac54e82dff1f64c14e8bf27604bb237a66e73826d2cf95b218ebc` |
| LINKUSDT | 31,801 | `8d1908931712661fd1cde3f3454f30ff711ca98a70b292e62d165e6fe200fc22` |
| DOTUSDT | 31,801 | `3f1f86340510ebfe2f700466b2b121276a61996d7658dc990236b2fc7a9adec6` |

`SOLUSDT` was listed in public market stats but did not qualify:

- Actual rows returned: `25,285`
- Expected rows: `31,801`
- Missing intervals: `6,516`
- First missing timestamp: `1635712200` (`2021-10-31T20:30:00Z`)
- Last missing timestamp: `1659166200` (`2022-07-30T07:30:00Z`)
- Status: `REJECTED_DATA_QUALITY_OR_API`
- No complete 31,801-row CSV or qualified sidecar manifest was produced for SOLUSDT.

The nine successful histories are therefore useful immutable research data, but they do not satisfy either locked candidate's complete-universe requirement.

BTCUSDT context source remains bound to SHA-256:
`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`.

## 3. Locked candidate outcomes

The preregistration SHA-256 was:
`44d6b9bd01c5c163f4e4f448a68c8cc91d060bd1a182c8e841c8a939a018f6b0`.

| Logical candidate | Fixed universe | Missing symbol | Evaluation status |
|---|---|---|---|
| `NOBITEX-M7-5PAIR-RIDGE-TOP2-01` | ETH, BNB, ADA, SOL, XRP | SOLUSDT | `BLOCKED_DATA` |
| `NOBITEX-M7-10PAIR-RIDGE-TOP2-01` | All ten preregistered symbols | SOLUSDT | `BLOCKED_DATA` |

Both trials remain exactly as registered. No symbol was removed, replaced, substituted, backfilled, interpolated or given a shortened date range.

## 4. Evaluation execution and economic result

- Evaluation workflow: [M7 Cross-Sectional Ridge Economic Evaluation, Run #1](https://github.com/angry0tiger-sudo/AliX/actions/runs/46051396261)
- Run ID: `46051396261`
- Evaluation commit: `6e0baa1655e802e6dcff6baec4d791fd42214870`
- Workflow conclusion: `success`
- Parent data commit Standard CI: [Run #2668](https://github.com/angry0tiger-sudo/AliX/actions/runs/46050293146), SHA `24d1916abba25e280f4cd5804df9a9ae783b3ca2`, conclusion `success`.
- Evaluation Artifact: `alix-m7-cross-sectional-results`
- Artifact ID: `15223753731`
- Artifact SHA-256: `8a05bd2254bf2042bcac7712ffc70119fb4c2f006be9b9aaa12cbe08dd88cbe3`
- Extracted `m7_evaluation_report.json` SHA-256: `a932ba35c0bcd06ca8dc992ac5b806965de058b465c812597749752cd2ebacdb`

The evaluation workflow successfully loaded and checked the immutable data and preregistration. It did not fit a Ridge model for either candidate because both fixed universes lacked SOLUSDT. The official report records:

- `NOBITEX-M7-5PAIR-RIDGE-TOP2-01: BLOCKED_DATA`
- `NOBITEX-M7-10PAIR-RIDGE-TOP2-01: BLOCKED_DATA`
- `economic_acceptance.overall_verdict: NO-GO`

### 4.1 Required metric disposition

| Candidate | Cost | NetPF | MaxDD | Completed OOS trades | Gate disposition |
|---|---:|---:|---:|---:|---|
| 5PAIR | 50 bps | N/A | N/A | N/A | Not evaluated: BLOCKED_DATA |
| 5PAIR | 90 bps | N/A | N/A | N/A | Not evaluated: BLOCKED_DATA |
| 10PAIR | 50 bps | N/A | N/A | N/A | Not evaluated: BLOCKED_DATA |
| 10PAIR | 90 bps | N/A | N/A | N/A | Not evaluated: BLOCKED_DATA |

The numeric gates remain NetPF >= 1.10, at least 100 completed OOS asset-level round-trip trades, MaxDD <= 15%, at least 3 valid folds and at least 3 stable folds. None of these economic gates can be claimed as passed or failed by measured PnL because no eligible panel/model/fold evaluation existed. The family verdict is nevertheless NO-GO under the preregistered rule that an incomplete fixed universe is `BLOCKED_DATA` and cannot pass.

## 5. Horizon clarification

M7 uses **one-hour input candles** and a **four-hour forward target**, with the portfolio decision/rebalance cadence every four hours. The two logical candidates are the five-symbol and ten-symbol universes; they are not separate one-hour and four-hour prediction-horizon trials. No one-hour forecast-horizon model was registered or evaluated in M7.

## 6. Scientific interpretation and next-step boundary

This M7 execution does **not** establish that cross-sectional momentum or Ridge has no economic edge. It establishes that the current fixed-universe experiment cannot be economically evaluated on the available exact-window Nobitex history because one required symbol is incomplete.

Any future attempt must be handled as a separately approved, superseding preregistration before any outcome is inspected. The present ADR does not authorize:
- dropping or replacing SOLUSDT in either candidate;
- shortening the common history to fit SOLUSDT;
- using external-venue or synthetic candles;
- treating nine of ten assets as the registered ten-pair or five-pair candidate;
- tuning features, universe, alpha, rebalance rule, costs or thresholds;
- accessing Holdout or Post-Hack observations.

The nine qualified histories may be retained as immutable public PRE-HACK research data. Their existence does not create a Champion or trading permission.

## 7. Safety invariants

The acquisition manifests, workflow qualification checks and final evaluation report all preserve:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `holdout_bytes_read = false`
- `posthack_data_read = false`
- No credentials or private API were used.
- No orders or paper orders were placed.
- No local model fitting or backtest execution was performed; evaluation ran through GitHub Actions only.
- `main` remains frozen at Tree SHA `5761c6c34143f8e6924f53a4e745c48932a5f791`.

## 8. Final decision

**M7 Path B status: NO-GO / INSUFFICIENT_DATA.** No economic edge was discovered in this run because no eligible fixed-universe model evaluation could be performed. The result must not be misrepresented as a negative profitability measurement.

AliX remains an open R&D project, but these two M7 candidates are closed under their current preregistration. No Champion is created and no Paper/Live or order permission is issued.
