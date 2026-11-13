# Blockers for Four-Arm Kronos/Chronos Experiment Execution

**Registration:** NOBITEX-FOUR-ARM-KC-20261113  
**Trial IDs:** NOBITEX-KC26-A / B / C / D  
**Source of authority:** ADR-032 and FOUR_ARM_KRONOS_CHRONOS.md  
**Related infrastructure decision:** ADR-033  
**Status:** EXECUTION BLOCKED — preregistration and mock-only infrastructure are complete; real research execution is not authorized.

## 1. Purpose and current boundary

This document separates decisions that only the Owner can make from engineering work that an agent may implement after explicit Owner approval. It does not modify the preregistered four-arm hypothesis, trial IDs, economic gates, data scope, or holdout policy.

The four arms remain:

- A — Baseline: Donchian-20 + ATR(14) + 2-ATR trailing stop.
- B — Baseline + Kronos-Small.
- C — Baseline + Chronos-2.
- D — Baseline + Kronos-Small + Chronos-2.

All four trial IDs remain PREREGISTERED / NOT_STARTED. No candidate has been economically evaluated under this registration. Synthetic contract tests are not economic results.

### Status vocabulary

- **OPEN:** a decision or implementation has not been completed; it may proceed only within the stated authority boundary.
- **BLOCKED:** execution cannot proceed until the Owner resolves the item or supplies an immutable approved artifact.
- **RESOLVED:** evidence exists, is immutable, and has been verified against the preregistered contract.

No item in the tables below is marked RESOLVED as of this revision.

## 2. Intrinsic blockers — Owner decisions required

These items define what the experiment means. The agent must not infer or choose them from model defaults, convenience, or observed results.

### I-01 — Immutable model-weight SHA-256

**Status:** BLOCKED  
**Owner decision:** identify the exact Kronos-Small and Chronos-2 weight artifacts.

**Options**
1. Hash the exact downloaded model-weight artifacts and retain the immutable artifacts in the approved storage location.
2. Use a versioned, immutable model artifact repository whose content digest is independently verifiable.
3. Reject mutable aliases such as latest, moving branches, or unpinned tags.

**Recommendation:** require a SHA-256 for each exact weight artifact, record the artifact provenance and retrieval method, and verify the digest before loading any model. The two model hashes must be separate fields. Never substitute a repository revision for the weight-file digest.

**Resolution evidence:** model name, artifact URI or immutable storage identity, file manifest, SHA-256, and verification record.

### I-02 — Immutable model revisions

**Status:** BLOCKED  
**Owner decision:** pin the exact source revision for both model implementations.

**Options**
1. Pin a full immutable Hugging Face commit revision for each model.
2. Pin an immutable local artifact snapshot and its manifest SHA-256.
3. Record both the source revision and the locally materialized artifact identity.

**Recommendation:** record the full immutable source revision and the exact locally used artifact digest where both are available. Also pin tokenizer/configuration files and inference/runtime dependency versions. A model name or mutable branch is not a revision.

**Resolution evidence:** repository/model identifier, full revision, artifact manifest, runtime/library versions, and corresponding weight digests.

### I-03 — Forecast combination semantics

**Status:** BLOCKED  
**Owner decision:** define how model forecasts become a decision input, especially for arm D.

**Options**
1. **Equal-weight average:** normalize both model outputs to the same forecast semantics and average with fixed equal weights.
2. **Fixed weighted average:** use owner-specified weights that are frozen before evaluation.
3. **Consensus rule:** require both model forecasts to satisfy a predeclared agreement condition.
4. **Stacking:** train a separate combiner using only permitted training segments and a fully specified nested/chronological validation contract.

**Recommendation:** prefer the simplest deterministic rule that has a well-defined common forecast scale. A fixed equal-weight or fixed-weight aggregation is easier to audit than a learned stacker. Do not choose weights, thresholds, or consensus semantics after inspecting evaluation results. If stacking is selected, its training, feature, fold, and leakage contract becomes an additional preregistered model component.

**Resolution evidence:** formal input/output schema, normalization, aggregation formula or consensus predicate, missing/invalid forecast behavior, and immutable adapter SHA-256.

### I-04 — Exact chronological Fold boundaries

**Status:** BLOCKED  
**Owner decision:** approve the exact three-fold walk-forward cut points and the train/validation/OOS allocation.

**Options**
1. Three expanding-window chronological folds with exact UTC timestamps and row indices.
2. A different chronological allocation only if the Owner approves it and the preregistration is amended before any evaluation.

**Recommendation:** use exactly three or more chronological, expanding, non-shuffled folds; define every train, validation, and OOS interval by both UTC timestamp and immutable dataset row index. Apply the same eligible timestamps and boundaries to A/B/C/D. Explicitly state whether validation is separate from OOS scoring and how fold-level and pooled OOS evidence are aggregated.

**Resolution evidence:** immutable fold-plan file and SHA-256, with train/validation/OOS start and end timestamps, row-index ranges, purge/embargo exclusions, and common eligible-row identity.

### I-05 — Deterministic seed

**Status:** BLOCKED  
**Owner decision:** select the fixed seed value.

**Options**
1. One Owner-selected integer seed shared by all arms and folds.
2. A predeclared deterministic seed schedule by fold, provided every value and its mapping are frozen before execution.

**Recommendation:** use one fixed Owner-selected seed unless the pinned algorithms require a declared per-fold schedule. Record and control every relevant random source (Python, NumPy, model framework, data loader, and any combiner). Document any operation that cannot be made deterministic.

**Resolution evidence:** seed value or immutable seed schedule, implementation locations, deterministic-runtime settings, and seed-contract SHA-256. The current mock fixture seed is not automatically the economic experiment seed.

### I-06 — Triple-Barrier executable-price and collision contract

**Status:** BLOCKED  
**Preregistered anchors:** TP = 2 × ATR(14), SL = 2 × ATR(14), maximum holding = 24 hours.  
**Owner decision:** confirm the final executable-price and intrabar collision semantics.

**Options**
1. Use lower-timeframe/intrabar evidence to establish which barrier was reached first, if that evidence is available and permitted.
2. If only 1h OHLCV is available and both TP and SL are touched within one candle, use a conservative stop-first convention.
3. Mark ambiguous dual-touch events as invalid/ambiguous and apply a predeclared exclusion or loss convention consistently.

**Recommendation:** do not assume the favorable barrier was first. With 1h OHLCV only, use an Owner-approved conservative rule (for example, SL-first) or an explicit ambiguous-event rule. Also freeze the decision-time price anchor, next-open/close convention, gap-through behavior, fill price, fees, and vertical-barrier expiry.

**Resolution evidence:** executable contract covering anchor, ATR as-of timestamp, TP/SL price calculation, dual-touch ordering, gaps, fills, costs, and the 24h vertical barrier, plus a contract SHA-256.

### I-07 — Final Deflated Sharpe Ratio (DSR) method

**Status:** BLOCKED  
**Owner decision:** approve the exact DSR estimator and its inputs.

**Options**
1. Adopt one fully specified Deflated Sharpe Ratio formulation and parameterization, with source/version recorded.
2. Adopt a different explicitly documented formulation, including the assumptions and all required distributional/trial inputs.

**Recommendation:** freeze one canonical formula and a reproducible implementation before examining arm results. Record the observed Sharpe estimator, sample definition, skewness/kurtosis treatment if required by the chosen formulation, benchmark/expected-maximum-Sharpe inputs, effective trial count N, and sensitivity procedure. DSR must account for historical candidates and all four preregistered trial IDs when they reach evaluation.

**Resolution evidence:** formula reference/version, code revision and SHA-256, parameter schema, unit tests with boundary cases, and a worked synthetic-only example. Synthetic tests must not be reported as economic DSR results.

### I-08 — Known historical exposure in the final 12.5% of pre-hack data

**Status:** BLOCKED  
**Source disclosure:** ADR-032 and ADR-0012 record prior AI-6 / DEAP exposure to the final 12.5% segment of the 31,801-row pre-hack dataset. This segment is not an untouched final holdout.

**Owner decision:** determine whether the preregistered 24h purge + 24h embargo is accepted only as a temporal label-overlap control, or whether the known-exposed segment must also be excluded from confirmatory scoring.

**Options**
1. Retain the registered pre-hack dataset for development/walk-forward comparison, while explicitly treating the historically exposed 12.5% as non-independent evidence and prohibiting any untouched-holdout claim.
2. Exclude the known-exposed segment from confirmatory scoring, with exact row/timestamp boundaries and a preregistration amendment before evaluation.
3. Use the exposed segment for development only and require a separately approved, genuinely untouched dataset/period for any confirmatory claim.

**Recommendation:** do not treat 24h purge + 24h embargo as proof that historical contamination has been removed. Those controls address temporal label overlap around fold boundaries; they do not reverse prior AI-6 / DEAP exposure or prior research decisions informed by those observations. Preserve the disclosure and make no untouched-OOS claim. Any exclusion or change to the registered eligible rows requires Owner approval and a versioned preregistration amendment before evaluation.

**Resolution evidence:** explicit Owner decision, exact affected row/timestamp boundaries if exclusion is chosen, amended eligible-row contract, and a no-untouched-holdout disclosure in reports.

### I-09 — GLOBAL_N remains UNKNOWN

**Status:** BLOCKED  
**Current value:** GLOBAL_N = UNKNOWN.

**Owner decision:** choose the governance path for reconciling historical candidate evaluations; do not choose a guessed numeric N.

**Options**
1. Complete candidate-level reconciliation across historical families, deduplicating overlapping candidate identities and separating candidates from execution attempts.
2. Retain GLOBAL_N = UNKNOWN, report a defensible audited lower bound plus unresolved remainder, and run DSR sensitivity only where mathematically justified.
3. Treat DSR as INSUFFICIENT_EVIDENCE / NO-GO if the unresolved remainder prevents a defensible DSR > 0 conclusion.

**Recommendation:** keep GLOBAL_N = UNKNOWN until a candidate-level ledger proves otherwise. Do not sum heterogeneous counts (variants, genomes, workflow runs, or attempts) as if they were identical trials. Preserve all four registered trial IDs in the accounting if evaluated. If unknown N prevents a defensible DSR decision, the result is NO-GO / INSUFFICIENT_EVIDENCE.

**Resolution evidence:** candidate-level reconciliation ledger, deduplication method, count definitions, unresolved remainder, review record, and DSR sensitivity artifact.

## 3. Technical blockers — agent work requires Owner approval

These items are engineering tasks. They do not authorize training, model loading, dataset reading, or economic evaluation.

### T-01 — Connect the execution lock to every real training/evaluation entrypoint

**Status:** OPEN  
**Required work:** identify every real-data/model/evaluation entrypoint and make the independent lock validator the first control before dataset, model, or artifact I/O. Deny by default. No alternate path may bypass the validator.

**Recommendation:** implement a single shared authorization boundary and route all real execution entrypoints through it. Keep mock-only execution isolated and unable to set the authorization flag. Do not create or enable a real entrypoint until the Owner approves the implementation scope.

**Acceptance evidence:** code review, entrypoint inventory, and a fail-closed contract test proving authorization occurs before data/model access.

### T-02 — Execution-lock integration tests

**Status:** OPEN  
**Required work:** test absent/false owner authorization, unsafe dry-run/live flags, missing/corrupt lock, missing model hashes, wrong dataset hash, missing contract fingerprints, and a complete but unauthorized template.

**Recommendation:** include a sentinel path or instrumented I/O boundary proving that no dataset/model file is opened before authorization succeeds. Test the authorized contract shape without running training, fit, inference, or backtest. Verify all safety invariants remain unchanged.

**Acceptance evidence:** deterministic CI tests for every rejection branch and the authorization ordering guarantee; no gate weakening, skip, or xfail.

### T-03 — Final Cost-50 / Cost-90 fill and cost simulator

**Status:** OPEN  
**Required work:** implement the final deterministic cost/fill simulator for the preregistered 50 bps and 90 bps round-trip scenarios.

**Recommendation:** use one frozen simulator across all four arms. Define fee decomposition, slippage, fill price, gap handling, and any maker/taker assumptions explicitly. Keep 50 bps and 90 bps as separate outputs; never average the scenarios or report gross metrics as net.

**Acceptance evidence:** immutable simulator revision and SHA-256, deterministic synthetic-only tests for fees/fills/gaps/boundaries, and confirmation that all arms share identical execution assumptions.

## 4. Global execution gate

Real research execution remains blocked until all nine intrinsic decisions are resolved, all three technical items are implemented and verified, and the complete immutable execution lock is committed and reviewed.

The following are necessary but not independently sufficient:

- Owner-controlled authorization: ALIX_FOUR_ARM_EXECUTION_AUTHORIZED=true.
- ALIX_DRY_RUN=true.
- ALIX_ALLOW_LIVE_TRADING=false.
- Exact pre-hack dataset identity: 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e.
- Immutable model revisions and weight digests.
- Frozen adapter/combination, fold, seed, triple-barrier, cost/fill, and DSR contracts.
- Explicit Owner run directive after review of the lock and workflow.

The current execution-lock template is NOT_AUTHORIZED. The current four-arm workflow is workflow_dispatch-only and mock-only. Dispatching it performs synthetic contract validation, not an economic experiment. Do not represent its run as a four-arm research result.

## 5. Holdout and trading boundary

The post-hack holdout remains sealed. It is not a development, debugging, or remediation dataset. It may only be considered after at least one arm passes every preregistered gate on the permitted pre-hack evaluation and the Owner separately issues Release Authorization. Any later holdout acquisition must use the approved one-time nonce protocol; rotated or consumed nonces must never be reused.

A pre-hack PASS is at most eligibility for a separately authorized paper phase. It does not grant holdout access, private API access, or live-order permission.

## 6. Safety invariants

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- ALIX_DRY_RUN = true
- ALIX_ALLOW_LIVE_TRADING = false
- main = FROZEN

**This document is a blocker register, not an execution authorization.**
