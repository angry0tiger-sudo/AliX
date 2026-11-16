# ADR-046: Model-Provenance and Data-Independence Leakage Boundary

- **Status:** Accepted — evidence boundary and governance constraints only; no research option selected
- **Date:** 2026-11-16 UTC
- **Branch:** \`research/nobitex-hf-strategy\`
- **Evidence baseline:** \`b4b076bafca665eba244f19dcc5f187300ef61db\`
- **Related:** ADR-0012, ADR-032, ADR-036, ADR-037, ADR-038, ADR-042, ADR-043, ADR-045
- **Owner decision:** none. This ADR does not authorize data access, model loading, inference, training, backtesting, dispatch, or trading.

## 1. Decision summary

AliX now distinguishes three independent barriers to statistical validation:

1. **Historical selection/multiplicity barrier:** \`GLOBAL_N=UNKNOWN\`; the historical Sharpe-per-trial distribution is incomplete; global historical \`DSR=NOT_COMPUTABLE\`.
2. **Reused-market-data barrier:** the existing 31,801-row pre-hack dataset has already been consumed by AI-6 and DEAP, and the Run #10 confirmatory prefix is drawn from that same historically used dataset. It cannot provide an untouched holdout.
3. **Foundation-model pretraining barrier:** public model provenance exists, but the published corpus descriptions do not establish a complete BTC-symbol/date-level exclusion for both pinned models. A provenance document is not proof that a later evaluation window is clean of pretraining exposure.

These barriers are logically distinct. A complete trial ledger does not establish data independence. A new market-data window does not establish independence from a foundation model's pretraining corpus. A 24-hour purge/embargo reduces local temporal label overlap at fold boundaries; it does not reverse prior researcher exposure and it does not audit a model's pretraining corpus.

**Current governance ceiling:** no current candidate is admissible for promotion/validation under the present evidence and access state. The correct status is \`NOT_VALIDATABLE_UNDER_CURRENT_GOVERNANCE / INSUFFICIENT_EVIDENCE\`. This is not a finding that the candidate is economically unprofitable, and it is not a retroactive Run #10 \`FUTILE\` or family-wide economic NO-GO.

## 2. Vector 1 — historical N and Sharpe distribution

ADR-045 remains authoritative for trial identity and DSR inputs.

- \`GLOBAL_N=UNKNOWN\`; the recorded non-HF subtotal \`N >= 7,912\` is heterogeneous and does not provide a finite, reconciled global upper bound.
- Historical Sharpe-per-trial evidence is incomplete. V4-AP retains 36 Sharpe estimates with independence unverified; V4-AI/AO has within-run Sharpe/DSR but does not include the historical cohort. For the remaining major historical families, a complete deduplicated Sharpe series is not evidenced.
- Consequently, historical global \`DSR=NOT_COMPUTABLE\`, even if someone later supplies a guessed or partial N. N and the observed Sharpe distribution are both required.

This barrier concerns multiplicity and return-distribution evidence. It is not, by itself, proof of bad economics.

## 3. Vector 2 — historical market-data reuse

### 3.1 Pre-hack dataset

The registered pre-hack source is the 31,801-row Nobitex BTC/USDT Spot 1h dataset:

- SHA-256: \`0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e\`
- Time range: \`2021-10-31T20:30:00Z\` through \`2025-06-17T20:30:00Z\`
- Repository path: \`data/nobitex/prehack/nobitex-btcusdt-1h.csv\`

The data has already been used in earlier research:

- AI-6 partitioned the same 31,801-row sequence into teacher fitting (first 45%), PPO training (45%–75%), OOS validation (75%–87.5%), and final OOS holdout (87.5%–100%). Its final 12.5% segment corresponds to rows \`27825..31800\` and the interval \`2025-01-03T05:30:00Z\` through \`2025-06-17T20:30:00Z\` (AI-6 final run ID \`43000569350\`).
- DEAP AI-2 Revisited used OOS from row \`22308\` through the final row, which includes the complete AI-6 terminal slice (run ID \`43022191014\`).
- The later four-arm preregistration restricts its confirmatory prefix to the first 27,825 rows and excludes the final 3,976 rows. That prefix is still part of the same historically consumed dataset: AI-6 used the earlier segments for fitting/training/validation, and DEAP's OOS range begins at row 22,308. The prefix must not be described as a new untouched holdout.

ADR-0012 and \`docs/research/HOLDOUT_COMMITMENT.md\` therefore remain authoritative: the historical dataset is suitable only for explicitly disclosed development/walk-forward analysis under its registered constraints. No subrange hash, alternate split, or new model family can make previously exposed observations untouched again.

A 24-hour purge and 24-hour embargo can address overlapping labels close to a particular train/evaluation boundary. They cannot erase AI-6/DEAP use of the underlying observations or choices already informed by those results.

### 3.2 Post-hack holdout

The separately committed post-hack Nobitex window is the only currently committed market-data window not consumed by prior AliX trial evaluation, based on the retained experiment records. This statement is limited to **AliX trial consumption**; it is not a statement that foundation-model pretraining has been proven not to contain the same observations.

Current commitment:

- Spec v3: \`2025-07-01T13:30:00Z\` through \`2026-11-01T00:30:00Z\`, inclusive :30 UTC hourly grid
- Expected rows: \`11,700\`
- Canonical SHA-256: \`435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee\`
- V2 status: \`FETCHED_LOCKED\`
- \`dataset_persisted=false\`; raw OHLCV bytes were not retained. The retained artifact is structural metadata/spec fingerprint, not the dataset.

No holdout access is authorized by this ADR. Any future use requires a separate Owner Release Authorization and the exact ADR-036 protocol:

1. Re-fetch the exact Spec v3 inclusive window from the same approved public Nobitex source.
2. Require exactly 11,700 rows and zero missing, off-grid, duplicate, reordered, or malformed candles; do not interpolate, backfill, mask, shift, or synthesize.
3. Reproduce the registered canonical serialization and compare SHA-256 byte-for-byte to \`435b313086b78ba8d3caa2e97834c6fabebbbd12eeddcb2865d65982279f33ee\`.
4. On any mismatch, stop. Do not inspect outcomes, select a replacement window, or continue with partial data.

The one-time property is operationally consequential: once any candidate results are observed on this holdout, it is no longer available as an untouched confirmatory set for a second selection cycle.

### 3.3 New data outside both windows

The committed public Nobitex fetcher supports bounded public retrieval for approved symbols/resolutions and bounded day windows. However, this documentation-only audit did not fetch or validate a third dataset outside both the pre-hack and post-hack windows. No immutable, gap-verified, committed third window has been evidenced at this baseline.

Therefore the actual availability, continuity, and independence of a new out-of-window dataset are **UNKNOWN / NOT VERIFIED**. Do not treat theoretical fetcher capability as evidence that a new clean dataset already exists.

## 4. Vector 3 — model pretraining exposure and actual disclosure

### 4.1 Fingerprinted contract identity

The execution-contract manifest \`config/four_arm/contract_fingerprints.json\` records:

\`license_and_corpus_disclosure_sha256 = 57548c31a6c84d9e7e05265df4f02221c64cfc52fac9004863b92e8d16c1d3c0\`

The capture implementation \`scripts/capture_four_arm_contract_hashes.py\` maps that key to:

\`docs/research/preregistration/contracts/model_provenance.md\`

and hashes the file bytes with SHA-256. The current disclosure file blob is \`1f2883c28b438ddd40b7b29460a0683d7dab70e0\`.

The actual disclosure says:

- **Kronos-Small:** MIT; trained on over 12 billion K-line records from 45 global exchanges. The available model metadata does not provide a complete per-exchange, per-symbol, per-date training ledger. Exact Nobitex BTC/USDT overlap is \`UNKNOWN / NOT EXCLUDED\`.
- **Chronos-2-Small:** Apache-2.0, 28M parameters; trained on selected real-world time-series datasets and synthetic series. The available metadata does not provide a complete symbol/date inventory for every source. Exact Nobitex BTC/USDT overlap is \`UNKNOWN / NOT EXCLUDED\`.

The public technical material adds useful but bounded facts:

- Kronos's paper describes Binance records with asset types \`Crypto, Swap\` and 997 assets, but does not disclose the per-symbol inventory. It states that pretraining data extends up to June 2024. This confirms a crypto-class presence in the broader corpus, but does **not** prove that BTCUSDT itself was included. The June 2024 cutoff is before the post-hack holdout starts in July 2025, so the declared training-period chronology does not overlap that holdout; this does not prove instrument-level independence or exclude other indirect exposures. Sources: [Kronos technical report, Table 13](https://openreview.net/pdf/1032dad0bdf374a73ec0b7aa78ff401ab8b3d650.pdf) and [forecasting setup](https://openreview.net/pdf/1032dad0bdf374a73ec0b7aa78ff401ab8b3d650.pdf).
- The pinned Chronos-2-Small model card identifies \`autogluon/chronos_datasets\` and \`Salesforce/GiftEvalPretrain\`. The Chronos-2 report describes selected Chronos/GIFT-Eval data and synthetic series; its dataset table includes M4 entries whose domain is only \`Various\`. The report documents avoiding GIFT-Eval **test** portions, while acknowledging partial overlap with some GIFT-Eval **training** portions. It does not publish a complete per-series ticker/date inventory or a complete corpus date boundary that can be compared with AliX's holdout. Therefore exact BTC/crypto presence and temporal overlap with the AliX post-hack window remain **UNKNOWN / NOT EXCLUDED**. Sources: [pinned Chronos-2-Small model card](https://huggingface.co/autogluon/chronos-2-small), [Chronos-2 training-data section and Table 6](https://arxiv.org/html/2510.15821), and [GIFT-Eval overlap discussion](https://arxiv.org/html/2510.15821).

### 4.2 Provenance is not cleanliness

The existence of a provenance document, the presence of immutable model revisions/weight hashes, and a license declaration establish identity and disclosed information. They do not establish that the holdout is absent from a model's pretraining or post-training data.

**Finding: pretraining cleanliness is not provable for the current two-model family.** For Kronos, crypto exposure is confirmed at the corpus-class level; the exact BTC symbol remains undisclosed, while the declared cutoff predates the new holdout. For Chronos-2, both exact symbol-level and temporal overlap are unresolved. This is an independent gate before any use of the post-hack holdout. ADR-045's trial accounting does not resolve it.

## 5. Three-vector leakage matrix

| Vector | What is established | What remains unresolved | Current decision |
|---|---|---|---|
| Historical N / Sharpe | ADR-045 definitions, historic lower-bound subtotal, partial per-family Sharpe records | Global finite N, complete Sharpe distribution, full candidate lineage | \`GLOBAL_N=UNKNOWN\`; \`DSR=NOT_COMPUTABLE\` |
| Reused market data | AI-6 and DEAP exposure of pre-hack; Run #10 uses a prefix of that source; post-hack holdout is not used by recorded AliX trials | Raw holdout bytes are not persisted; a new out-of-window dataset is not yet evidenced | Pre-hack cannot be an untouched holdout; post-hack remains locked and one-time |
| Foundation-model pretraining | Pinned model identity and disclosure fingerprint; Kronos Crypto class exposure and declared cutoff; Chronos-2 named corpora | Exact BTC ticker inventory for Kronos; complete symbol/date inventory and cutoff for Chronos-2 | Clean pretraining cannot be proven; mandatory gate before holdout consumption |

No one vector repairs another. A bounded family N does not establish market-data independence. A fresh data window does not supply historical Sharpe distributions. A 24-hour purge/embargo does not provide a model training-corpus audit.

## 6. Isolated-family feasibility boundary

A prospective family-scoped DSR can be mathematically meaningful without solving the historical global N, but only if the new family is truly bounded and independently governed. Its DSR must be explicitly scoped to that family; it must not be represented as a reconstruction of, or substitute for, global historical DSR.

Minimum prerequisites before the first candidate result is observed:

1. **Data boundary:** a genuinely new, immutable data commitment. For the existing post-hack window, separately authorized ADR-036 re-fetch, exact Spec v3 coverage, canonical SHA match, zero-gap validation, and sealed outcome boundary are mandatory.
2. **Pretraining boundary:** exact model revisions and weights; public/private training-data disclosure review; symbol/date-level evidence where available. If an opaque corpus prevents a clean claim, the affected model must be excluded from a clean-holdout claim or the study must explicitly remain non-independent. A provenance file alone is insufficient.
3. **Canonical trial registry:** unique \`trial_id\` per candidate decision rule; parent/duplicate links; every parameter, feature, threshold, model revision, seed, label, and execution-cost variant frozen before results.
4. **Predeclared family budget:** integer \`N_budget\`/maximum candidate count and the search protocol disclosed before evaluation. No unregistered threshold, alternative model, or post-result branch.
5. **Per-candidate Sharpe from day zero:** one comparable, net-of-cost Sharpe estimate per evaluated candidate under a common return frequency, fill/cost protocol, and sample boundary; retain the underlying return series/required moments and immutable artifact.
6. **Dependence treatment:** predeclare how correlated candidate return series are handled; include a conservative dependence/sensitivity analysis rather than assuming all candidates are independent.
7. **DSR method:** freeze the estimator and inputs. A family-scoped DSR may be advisory only, using the complete family N and a defensible conservative bound/sensitivity. Missing candidate Sharpe or an exceeded N budget is fail-closed / \`NOT_COMPUTABLE\`.
8. **One-time use:** holdout opening requires a separate Owner Release Authorization. Any result inspection consumes its untouched status. No second attempt, threshold retuning, or replacement candidate on the same holdout.

Even a clean accounting system does not, by itself, establish independence of data or independence from pretraining.

## 7. Governance status and non-equivalent verdicts

The present program is **not validatable for promotion under current governance** because the three-vector evidence is incomplete and the only committed new holdout remains \`FETCHED_LOCKED\`, non-persisted, and unauthorized for use.

This verdict MUST be kept distinct from an economic NO-GO:

- \`NOT_VALIDATABLE_UNDER_CURRENT_GOVERNANCE / INSUFFICIENT_EVIDENCE\` means the evidence contract is insufficient to make a defensible promotion decision. It says nothing conclusive about the candidate's latent economic profitability.
- \`ECONOMIC_NO_GO\` means a candidate/family was evaluated under a valid, independent, preregistered protocol and failed its specified economic gates.
- No inference from the first status to the second is permitted.

Run #10 remains \`0 PASS / 3 FAIL / 9 INSUFFICIENT_SAMPLE\`; the three FAIL references remain one repeated Donchian control as documented. ADR-042/043/044/045 and all twelve Run #10 records are unchanged. This ADR does not relabel the Run #10 family as FUTILE.

## 8. Safety, expiry, and owner boundary

At the verification snapshot \`2026-11-16T07:41:23+03:30\`, model-weight artifact \`14111134121\` remained \`expired=false\`, with exact \`expires_at=2026-12-14T11:31:29Z\` (approximately 28 days, 7 hours, 20 minutes remaining at that snapshot). Expiry removes this retained Actions artifact; it does not itself prove that the pinned public model revisions cease to exist. Any owner-selected path that depends on this exact retained bundle must decide preservation/retrievability before the deadline and re-verify every file digest. This ADR does not download, copy, extend, or reacquire it.

Current invariants remain:

- \`GLOBAL_N=UNKNOWN\`
- \`DSR=NOT_COMPUTABLE\` for the global historical family
- \`Champion=null\`
- \`order_permission=false\`
- \`trading_authorization=NO-GO\`
- \`lock=NOT_AUTHORIZED\`
- \`EXECUTION_TRIGGER=PROBE\`
- \`ACQUISITION_TRIGGER=PROBE\`
- \`ALIX_DRY_RUN=true\`
- \`ALIX_ALLOW_LIVE_TRADING=false\`
- \`private_api_calls=0\`
- \`real_order_attempts=0\`
- \`main=FROZEN\`
- holdout remains sealed

No option is selected or executed. A future owner directive must state whether to pursue a newly isolated family, perform archival-only reconciliation, or close the project with a governance-correct NO-GO record.

Architecture First. Evidence Always. Fail Closed.
