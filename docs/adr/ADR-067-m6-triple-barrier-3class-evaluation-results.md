# ADR-067 — M6 Three-Class Triple-Barrier Evaluation Results: BTCUSDT NO-GO

- **Status:** **EXECUTED — STANDARD CI GREEN — BOTH HORIZONS INSUFFICIENT_SAMPLE / NO-GO**
- **Date:** 2026-11-27
- **Repository:** `angry0tiger-sudo/AliX`
- **Branch:** `research/nobitex-hf-strategy`
- **Target:** Nobitex Spot / USDT — `BTCUSDT` only
- **M6 preregistration:** `config/m6/triple_barrier_3class_preregistration.json`
- **Preregistration SHA-256:** `f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42`
- **Implementation/CI parent commit:** `3192c803d15bb607d24a324e6da3356f48b46ab5`
- **Evaluation report commit:** `51dc612d27ba5af108736c7ef80a119233bc1f30`
- **Evaluation workflow:** [Run #46021926723](https://github.com/angry0tiger-sudo/AliX/actions/runs/46021926723) — SUCCESS
- **Standard CI prerequisite:** [Run #46020651916](https://github.com/angry0tiger-sudo/AliX/actions/runs/46020651916) — SUCCESS on exact implementation parent
- **M6 result artifact:** `alix-m6-btcusdt-three-class-results`, Artifact ID `15210185201`

## 1. Decision

The Owner-authorized M6 experiment completed successfully on the two locked logical candidates using the existing Nobitex Spot BTCUSDT PRE-HACK hourly dataset. Both candidates were fitted and evaluated over four expanding chronological walk-forward folds. The output contains 31,641 OOS prediction rows across the two decision cadences.

**Overall economic decision: `INSUFFICIENT_SAMPLE / NO-GO`.**

Neither the 1h nor 4h candidate met the full registered economic acceptance gates under either 50 bps or 90 bps round-trip cost. Both horizons exceeded the aggregate 100-trade count minimum, but each produced only two valid folds (the required minimum is three) and zero stable folds (the required minimum is three). NetPF was below 1.10 and MaxDD exceeded 15% in every horizon/cost combination.

The conclusion is not merely a shortage of total trades: the OOS net profitability was materially negative, MaxDD was excessive, and the model did not produce sufficiently active, profitable, temporally stable predictions. The two M6 candidates are **evaluated and closed NO-GO** for this registered design. No tuning, threshold change, alternative model, or follow-up candidate is authorized by this ADR.

This is exploratory PRE-HACK evidence only. It is not a confirmatory result, does not establish a universal impossibility result for all three-class targets, and does not create trading authority.

## 2. Execution and CI provenance

### 2.1 Standard CI

Standard CI completed successfully on the exact implementation/workflow parent SHA `3192c803d15bb607d24a324e6da3356f48b46ab5`:

- [Run #46020651916](https://github.com/angry0tiger-sudo/AliX/actions/runs/46020651916): `completed / success`.
- Tests: **1,143 passed, 5 skipped**.
- Total coverage: **90.36%**, meeting the repository's 90% floor.
- Ruff lint and formatting, strict Mypy, Bandit, dependency resolution/lock/audit, architecture contracts, Xenon, and runner contract: PASS.
- PostgreSQL integration, Freqtrade runtime smoke, and four-arm bare-environment jobs: SUCCESS.
- Real-data diagnostic, M3 OFI evaluation, and ensemble real-data research jobs: SKIPPED as outside the authorized scope.

No quality threshold was lowered to obtain this result.

### 2.2 Dispatch/trigger history

The first M6 evaluation workflow attempt, [Run #46020290406](https://github.com/angry0tiger-sudo/AliX/actions/runs/46020290406), failed at the trigger-commit scope check. The workflow checked out only the triggering commit, so the parent SHA required by `git diff` was absent in the shallow clone. The run stopped before installing the model runtime, reading the PRE-HACK data, or fitting a model. It produced no economic output and is **not** a candidate trial or experimental result.

The workflow was corrected to use `fetch-depth: 0`, then triggered again only after Standard CI passed on its exact parent implementation commit. The successful run, [#46021926723](https://github.com/angry0tiger-sudo/AliX/actions/runs/46021926723), verified:
- Standard CI success on implementation parent `3192c803d15bb607d24a324e6da3356f48b46ab5`;
- trigger commit contained only `docs/research/M6_EVALUATION_TRIGGER.txt`;
- exact preregistration and three data digests;
- M6 execution and result bindings;
- all fail-closed safety assertions;
- result artifact upload.

### 2.3 Successful run and artifacts

| Item | Verified value |
|---|---|
| Successful workflow run | `46021926723` |
| Evaluation report `commit_sha` | `51dc612d27ba5af108736c7ef80a119233bc1f30` |
| Evaluation status | `COMPLETED_PREHACK_RESEARCH` |
| Preregistration SHA-256 | `f94af9eb6f6003818beb88fcdbfbebe2ac2d7ac8a42281529db2781b44962e42` |
| Artifact ID | `15210185201` |
| Artifact archive size | 1,546,103 bytes |
| Artifact archive SHA-256 | `2b948bbefc6a26925e84702e6f430534f811d4a1548a49c73361d891a878aa43` |
| `m6_evaluation_report.json` | 28,018 bytes; SHA-256 `12a1c34343222e0358c4e1a72147ab60c8692963d5fc8a347a98d822c0ab381b` |
| `m6_oos_predictions.csv` | 6,724,456 bytes; SHA-256 `ef6edce58a5a62706bbf2b82308396534b70f2baa07f95e10e1cbde0c4a46174a` |

The archive digest was independently recomputed from the retrieved ZIP and matches the GitHub artifact digest. The two member-file digests above were computed from the extracted artifact files.

## 3. Frozen market and data provenance

Only the committed public Nobitex Spot BTCUSDT PRE-HACK 1h OHLCV artifact was used:

- CSV: `data/nobitex/prehack/nobitex-btcusdt-1h.csv`
- Rows: **31,801**
- Period: `2021-10-31T20:30:00Z` through `2025-06-17T20:30:00Z`
- CSV SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Manifest SHA-256: `b50fea25f30d282c2eafcb262751367e11d6490eda0d463de5926b660eaf1a26`
- M5.2 data-lock SHA-256: `e66616ee3c13e6ce32f94577c4f46480390eea30c774ac6987e8d519d51d4acf`

No new market data was fetched or captured. The 4h decision bars were deterministically aggregated from four consecutive committed 1h bars; the original hourly bars resolved each event and marked the hourly NAV. No BTCIRT, alternate pair, external exchange, sentiment, microstructure, or On-Chain data was accessed.

The M6 report records:
- 1h decision bars: 31,801; feature-eligible decisions: 31,776.
- 4h decision bars: 7,950; feature-eligible decisions: 7,925.
- OOS predictions: **25,325 at 1h**, **6,316 at 4h**, **31,641 total**.
- Exactly two logical candidates and four chronological folds per candidate; **8 model fits** in total.

### 3.1 Resolved event counts

| Event outcome | 1h decisions | 4h decisions |
|---|---:|---:|
| `TP_FIRST` | 12,327 | 1,425 |
| `SL_FIRST` | 12,402 | 1,413 |
| `VERTICAL_TIMEOUT` | 6,970 | 5,082 |
| `AMBIGUOUS` | 54 | 0 |
| `UNAVAILABLE_OR_INCOMPLETE` | 23 | 5 |
| **Feature-eligible decisions** | **31,776** | **7,925** |

The 8 registered features, fixed Random Forest parameters, strict `P(TP_FIRST) > 0.60` signal, barrier rules, costs, sizing, purging, and embargo were not altered after the outcomes were observed. No parameter or threshold sweep was performed.

Historical volume values in this PRE-HACK dataset are zero (as documented in ADR-055). Consequently, `log_volume_change_1` is non-informative on this source snapshot. It was nevertheless retained exactly as registered; no post-result feature removal or ablation was performed.

## 4. Classification diagnostics

Classification metrics below exclude `AMBIGUOUS` events and are pooled across the four OOS folds:

| Candidate | OOS scored rows | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| 1h | 25,275 | 35.11% | 34.53% |
| 4h | 6,316 | 32.46% | 36.80% |

These are diagnostic metrics, not the economic acceptance gates. They show weak three-class discrimination and do not alter the economic verdict.

Among predictions that actually triggered long entries at `P(TP_FIRST) > 0.60`:
- 1h: 1,177 signals; 484 subsequently resolved `TP_FIRST`, 465 `SL_FIRST`, 201 `VERTICAL_TIMEOUT`, and 27 `AMBIGUOUS` (charged as SL in the ledger).
- 4h: 501 signals; 143 subsequently resolved `TP_FIRST`, 93 `SL_FIRST`, and 265 `VERTICAL_TIMEOUT`.

A high model probability was therefore not sufficient to produce reliable realized first-touch outcomes or positive net economics.

## 5. OOS fold results

### 5.1 1h candidate — `NOBITEX-M6-1h-RF-3CLASS-01`

| Fold | OOS trades | NetPF 50 bps | NetPF 90 bps | MaxDD 50 bps | MaxDD 90 bps | Net PnL 50 bps | Net PnL 90 bps | Valid / stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WF-01 | 463 | 0.319961 | 0.023513 | 7.65% | 14.27% | -6.70% | -14.11% | Valid / No |
| WF-02 | 714 | 0.067619 | 0.007488 | 15.74% | 27.14% | -15.67% | -27.10% | Valid / No |
| WF-03 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% | Invalid / No |
| WF-04 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% | Invalid / No |
| **Pooled OOS** | **1,177** | **0.160856** | **0.013034** | **22.58%** | **41.29%** | **-22.37%** | **-41.20%** | **2/4 valid; 0/4 stable** |

WF-03 and WF-04 had no signal above the frozen threshold. Their zero-trade and zero-drawdown rows are not evidence of profitable or low-risk performance; they are inactive test folds. WF-01 and WF-02 both lost money under both cost assumptions and were not stable.

### 5.2 4h candidate — `NOBITEX-M6-4h-RF-3CLASS-01`

| Fold | OOS trades | NetPF 50 bps | NetPF 90 bps | MaxDD 50 bps | MaxDD 90 bps | Net PnL 50 bps | Net PnL 90 bps | Valid / stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WF-01 | 347 | 0.565540 | 0.312041 | 19.59% | 38.84% | -18.55% | -38.37% | Valid / No |
| WF-02 | 146 | 0.410181 | 0.164197 | 10.06% | 17.72% | -8.71% | -17.05% | Valid / No |
| WF-03 | 5 | 2.300956 | 1.314120 | 0.51% | 0.63% | +0.42% | +0.14% | Invalid / No |
| WF-04 | 3 | 0.430917 | 0.282712 | 0.42% | 0.55% | -0.34% | -0.51% | Invalid / No |
| **Pooled OOS** | **501** | **0.534557** | **0.278479** | **28.18%** | **56.06%** | **-27.17%** | **-55.80%** | **2/4 valid; 0/4 stable** |

WF-03's positive PnL and NetPF do not qualify as a stable fold because it generated only five trades, below the locked minimum of 20. WF-04 generated only three. WF-01 and WF-02 both lost money and exceeded the MaxDD limit in at least one cost scenario.

## 6. Economic gates and adjudication

The acceptance rules apply independently to each horizon and each cost scenario. The overall M6 family passes only if both horizons pass all gates at both costs.

| Gate | Registered threshold | 1h Cost-50 / 90 | 4h Cost-50 / 90 |
|---|---|---|---|
| NetPF | `>= 1.10` | **FAIL / FAIL** | **FAIL / FAIL** |
| Completed OOS trades | `>= 100` | PASS / PASS (1,177) | PASS / PASS (501) |
| MaxDD | `<= 15%` | **FAIL / FAIL** (22.58% / 41.29%) | **FAIL / FAIL** (28.18% / 56.06%) |
| Valid folds | `>= 3 of 4` with at least 20 trades | **FAIL** (2/4) | **FAIL** (2/4) |
| Stable folds | `>= 3 of 4` with at least 20 trades, positive net PnL and fold NetPF > 1 | **FAIL** (0/4) | **FAIL** (0/4) |
| **Candidate verdict** | All gates mandatory | **INSUFFICIENT_SAMPLE / NO-GO** | **INSUFFICIENT_SAMPLE / NO-GO** |

The pooled trade-count gate passed, but the valid-fold gate did not. The stable-fold gate failed completely. NetPF was below 1.10 and MaxDD above 15% for every horizon/cost scenario. Therefore neither cadence reached the registered economics gates, and the family-level verdict is **`INSUFFICIENT_SAMPLE / NO-GO`**.

The worst cost scenario was consistently 90 bps, but relaxing costs or changing the threshold after observing these results is prohibited. The result does not authorize another RF parameter set, a different classifier, an additional target, a feature ablation, or a new horizon. Such work would require a new Owner decision and a superseding preregistration.

## 7. Trial Registry update

Exactly the two preregistered logical candidate identities were evaluated:

| Trial ID | Cadence | Fits | OOS predictions | Status |
|---|---|---:|---:|---|
| `NOBITEX-M6-1h-RF-3CLASS-01` | 1h | 4 | 25,325 | **EVALUATED — INSUFFICIENT_SAMPLE / NO-GO** |
| `NOBITEX-M6-4h-RF-3CLASS-01` | 4h | 4 | 6,316 | **EVALUATED — INSUFFICIENT_SAMPLE / NO-GO** |

The two trigger/dispatch attempts do not increase the logical candidate count. The first failed before data access/fitting; the successful attempt ran the exact two registered candidates. Fold fits and cost replays are not new logical candidates.

`GLOBAL_N=UNKNOWN` and `GLOBAL_DSR=NOT_COMPUTABLE` remain unchanged. The M6 result is not a first-economic-edge claim.

## 8. Safety invariants and final disposition

The successful evaluation report recorded and the workflow asserted:

- `market = BTCUSDT`; `data_scope = PRE_HACK_ONLY`.
- `new_data_acquisition = false`; no On-Chain, sentiment, or microstructure access.
- `holdout_bytes_read = false`; `posthack_data_read = false`.
- `private_api_calls = 0`; `exchange_network_access = false`.
- `real_order_attempts = 0`; `paper_orders = 0`; `local_execution = false`.
- `Champion = null`; `order_permission = false`; `trading_authorization = NO-GO`.

The current `main` branch was not changed for M6. At the final integrity check for this ADR, `main` remains at its existing HEAD and the original main tree SHA remains `5761c6c34143f8e6924f53a4e745c48932a5f791`.

**Final decision:** M6 three-class Triple-Barrier Random Forest on Nobitex BTCUSDT is **closed NO-GO** for the exact registered 1h and 4h candidate family. Both horizons failed the registered economic gates. No Champion, Paper or Live permission, Holdout access, Post-Hack access, new data acquisition, further model, or tuning is authorized.

Architecture First. Evidence Always. Fail Closed.
