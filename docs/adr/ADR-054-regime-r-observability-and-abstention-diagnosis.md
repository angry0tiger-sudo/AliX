# ADR-054 — Regime R G4.5 Observability and Abstention Diagnosis

- Date: 2026-11-17
- Branch: `research/nobitex-hf-strategy`
- Status: **PROVISIONAL — DIAGNOSTIC WORKFLOW DISPATCH AND ARTIFACT PENDING**
- Decision class: Research observability; no model promotion or trading authorization
- Instrumentation commit: `485948a62f87b1de7854f320ef003c0c1e9e5599`
- Standard CI: [Run #2486](https://github.com/angry0tiger-sudo/AliX/actions/runs/44123874637) — success
- Quality Preflight: [Run #76](https://github.com/angry0tiger-sudo/AliX/actions/runs/44123874369) — success
- Prior economic evaluation: [G4/G5 Run #10](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184)
- Prior economic artifact: [Artifact 14322682950](https://github.com/angry0tiger-sudo/AliX/actions/runs/44072026184/artifacts/14322682950)

## 1. Decision and current limitation

The G4/G5 evaluation established that every evaluated calibrated lower bound was at or below the applicable round-trip cost. It did **not** establish whether this was caused by predominantly negative forecasts, wide conformal intervals, or positive forecasts whose conservative lower bounds remained below cost.

The prior economic artifact did not include per-forecast lower/median/upper quantiles, calibrated interval widths, or the lower-bound-minus-cost distribution. Therefore, a causal diagnosis cannot be reconstructed from that artifact.

G4.5 adds the required observability to the existing HGBQ-CQR evaluation path. The instrumentation and diagnostics-only execution contract are committed and Standard CI is green. However, the manual `workflow_dispatch` run has **not yet been confirmed**, and no G4.5 diagnostics artifact has been retrieved in this record. Accordingly, the causes of abstention remain **UNKNOWN** and this ADR is provisional. No forecast statistics are inferred or fabricated.

## 2. Scope and frozen constraints

The diagnostics are restricted to the existing pre-hack dataset:

- Dataset: `data/nobitex/prehack/nobitex-btcusdt-1h.csv`
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- G3 preregistration SHA-256: `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- Model family: the same preregistered HGBQ-CQR adapter
- Horizons: 1h, 4h, 24h
- Registered slots: 30, representing three unique candidate configurations and 27 duplicate slots
- Walk-forward folds: three
- Expected fit calls if dispatched successfully: 90
- Cost references: 50 bps and 90 bps

No model family, G3 parameter, feature definition, cost threshold, fold definition, or acceptance gate is changed by this observability phase. The diagnostics-only main path requires `--diagnostics-only`; it does not call the hypothetical ledger, create trades, or produce a trade CSV.

## 3. Diagnostic output contract

The workflow is configured to write:

`.artifacts/regime-r-g4-g5/regime-r-g4-g5-diagnostics.json`

and upload that single file as the pre-hack diagnostics artifact. The upload step explicitly includes hidden files because the artifact path begins with `.artifacts`.

Each registered trial / fold record contains:

- Lower quantile: count, mean, population standard deviation, minimum and maximum
- Median quantile: count, mean, population standard deviation, minimum and maximum
- Upper quantile: count, mean, population standard deviation, minimum and maximum
- Raw quantile interval width (upper quantile minus lower quantile)
- Calibrated conformal interval width (calibrated upper bound minus calibrated lower bound)
- Lower and upper conformal correction summaries
- Count and percentage of strictly positive median predictions
- Signed calibrated lower-bound distance from each frozen cost scenario (50 and 90 bps), including mean/std distance, min/max, and the count/percentage above cost

Standard deviation uses population convention (`ddof=0`). Quantile and interval values are expressed as simple-return fractions; cost distances are reported both as fractions and basis points. The cost comparison uses the strict predicate `Lower Bound - Cost > 0`.

The expected JSON contains 90 trial/fold records only after a successful dispatch and completion. A configured schema or a successful CI run is not evidence that this runtime artifact has been generated.

## 4. What is currently known

The prior G4/G5 artifact recorded zero qualifying lower-bound signals at both cost thresholds across all evaluated horizons/folds. It also recorded zero trades because the strict lower-bound cost condition was never met.

The prior dataset inspection found all 31,801 volume values equal to zero. This means volume-derived features do not discriminate observations in that dataset. However, the evaluated feature set also contains price-derived return, volatility, range, ATR and time features. The evidence does **not** isolate zero volume as the cause of the abstention.

The following explanations remain unresolved until the G4.5 JSON is produced and inspected:

- Predicted medians are predominantly negative.
- Predicted medians are often positive, but calibrated lower bounds are reduced by wide conformal intervals/corrections.
- Medians are positive but expected lower-bound returns remain positive and below the round-trip cost.
- Different horizons/folds exhibit different combinations of these mechanisms.

No one of these hypotheses is adopted as a finding at this stage.

## 5. Interpretation rules for the pending artifact

The final diagnosis must be based on the per-fold and per-horizon distributions, not only one global mean:

1. **Direction / gross-value weakness:** a low positive-median rate and negative or near-zero median distributions, especially alongside negative upper quantiles, would support a weak or adverse forecast-location diagnosis.
2. **Uncertainty-driven abstention:** materially wide calibrated intervals and/or large conformal corrections, with positive medians but lower bounds below cost, would support an uncertainty-driven diagnosis.
3. **Insufficient edge after cost:** positive medians and moderate interval widths, but signed lower-bound-to-cost distances still below zero, would indicate that the conservative forecast edge does not clear execution costs.
4. **Horizon/fold instability:** materially different patterns between horizons or folds would prevent a single aggregate explanation and require a stability-aware conclusion.

These are interpretation rules, not claims about the as-yet-unretrieved diagnostics.

## 6. Recommendation pending evidence

**Do not abandon HGBQ-CQR solely because the G4/G5 run produced zero trades.** The existing evidence is insufficient to separate forecast direction/value from conformal uncertainty and cost-threshold effects.

The zero-volume limitation makes a separately preregistered volume-feature ablation a reasonable future research hypothesis, but it does not prove that volume features caused the zero qualifying signals. Any such ablation must be a new, explicitly scoped pre-hack experiment with its own feature-contract/config fingerprint and acceptance criteria. It must not silently modify the locked G3 configuration or be mixed into this observability run.

No feature removal, retraining, alternative model, parameter search, or additional economic evaluation is authorized by this provisional ADR.

## 7. Safety invariants

- `real_data_scope = PRE_HACK_ONLY`
- `holdout_bytes_read = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `hypothetical_trades_simulated = false`
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `main = FROZEN @ 44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`

## 8. Required amendment before final diagnosis

After the owner-triggered `workflow_dispatch` succeeds on the current branch, amend this ADR with:

- the exact workflow run ID and artifact ID;
- confirmation of 90/90 fit calls and 90 diagnostic records;
- measured quantile means/stds, median-positive rates, raw/conformal widths and lower-bound-to-cost distances for every horizon/fold;
- a supported finding on forecast direction versus uncertainty versus insufficient edge;
- a decision on whether to retain HGBQ-CQR for a controlled feature-ablation experiment.

Until that amendment is made, the verdict remains **NO CAUSAL DIAGNOSIS — EVIDENCE PENDING**.

Architecture First. Evidence Always. Fail Closed.
