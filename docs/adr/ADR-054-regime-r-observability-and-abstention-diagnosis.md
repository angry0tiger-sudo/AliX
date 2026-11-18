# ADR-054 — Regime R G4.5 Observability and Abstention Diagnosis

- Date: 2026-11-18
- Branch: `research/nobitex-hf-strategy`
- Status: **ACCEPTED — G4.5 DIAGNOSTIC COMPLETE; ECONOMIC VERDICT NO-GO**
- Decision class: Research observability and abstention diagnosis; no model promotion or trading authorization
- Instrumentation baseline commit: `485948a62f87b1de7854f320ef003c0c1e9e5599`
- Evaluated code snapshot: `ccc50024cc97dd1cbd09238fb5ce091d1e4bfbb9`
- Standard CI on evaluated code snapshot: [Run #2487](https://github.com/angry0tiger-sudo/AliX/actions/runs/44125075022) — success
- G4.5 diagnostic workflow: [Run #11](https://github.com/angry0tiger-sudo/AliX/actions/runs/44283184884) — success
- G4.5 artifact: [Artifact 14416704508](https://github.com/angry0tiger-sudo/AliX/actions/runs/44283184884/artifacts/14416704508)
- Artifact name: `regime-r-g4-g5-diagnostics-prehack-ccc50024cc97dd1cbd09238fb5ce091d1e4bfbb9`
- Artifact ZIP SHA-256: `37edef56e9b34b4c1b5f0a0436d067f61d97541a3ca04268023d621aac90e92c`
- JSON schema: `alix.regime-r.g4-g5-diagnostics.v1`
- Workflow Run ID embedded in JSON: `44283184884`

## 1. Decision and diagnosis

G4.5 successfully completed the previously missing observability. The artifact is tied to the authorized Research commit and the exact preregistered data/configuration hashes. It contains 90/90 fit calls and 90/90 diagnostic records. The diagnostics-only validation and all workflow safety checks passed.

**Finding:** abstention was the correct fail-closed outcome. The direct mechanical reason is unambiguous: **zero evaluated forecasts cleared the strict calibrated lower-bound gate at either 50 bps or 90 bps**. The lower bound remained substantially below execution cost across every evaluated horizon/fold.

The evidence supports a combined forecast-location and uncertainty/edge diagnosis, not a single-factor story:

- The weighted median-positive forecast rate was only **46.6788%** (3,085 of 6,609 forecast rows). This is a rate of positive model medians, **not a realized directional accuracy or win rate**; this diagnostics-only run did not evaluate whether those positive forecasts matched subsequent realized returns.
- The weighted mean forecast median was only **+10.25 bps**. The centre of the forecast distribution was therefore near zero in aggregate. The weighted mean median was slightly negative at 1h (-0.82 bps) and effectively zero at 4h (-0.03 bps), and +32.25 bps at 24h.
- The weighted mean calibrated conformal interval width was **5.3333% simple return (533.33 bps)**, versus a raw quantile interval width of **4.5249% (452.49 bps)**. Interval width rose materially with horizon: 1.9740% at 1h, 3.8515% at 4h, and 10.3229% at 24h.
- The weighted mean distance `Lower Bound - Round-trip Cost` was **-358.75 bps** at 50 bps cost and **-398.75 bps** at 90 bps cost. No horizon/fold combination had a positive qualifying count at either cost.

Accordingly, G4.5 resolves the former unknown: this is not merely a narrow miss against transaction cost. The predictive centre is weak/near-zero, and the calibrated lower tail is far below cost, with uncertainty widening sharply at 24h. The cost gate correctly rejects all forecasts. The diagnostics do not establish which individual feature caused the weak forecast distribution; in particular, the zero-volume observation below is not causal proof.

This ADR is **accepted as a completed diagnosis**, while the economic/strategy verdict remains **NO-GO**. Acceptance here does not mean the model or strategy is accepted for deployment.

## 2. Scope, provenance, and frozen constraints

The run used only the existing pre-hack dataset:

- Dataset: `data/nobitex/prehack/nobitex-btcusdt-1h.csv`
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Rows: 31,801
- Dataset scope: `PRE_HACK_NOBITEX_ONLY_EXPLORATORY`
- `confirmatory_holdout = false`
- `historically_exposed = true`
- G3 preregistration/configuration SHA-256: `92cdcf19a02f5cf9659f327704ed4d6c5cf0d85687aff2ba35b9542fec5cfdaa`
- Feature-contract SHA-256: `a898866c4a061ece282593366997dca9507dab328e3df5e72bddeff3336e6987`
- Model family: same preregistered HGBQ-CQR adapter
- Horizons: 1h, 4h, 24h
- Walk-forward folds: WF-01, WF-02, WF-03
- Cost references: 50 bps and 90 bps
- Workflow event: `workflow_dispatch`
- Run branch: `research/nobitex-hf-strategy`
- Run head SHA: `ccc50024cc97dd1cbd09238fb5ce091d1e4bfbb9`

Trial accounting from the JSON:

| Item | Observed |
|---|---:|
| Registered candidate slots | 30 |
| Unique candidate configurations | 3 |
| Duplicate slots | 27 |
| Fit calls | 90 |
| Diagnostic records | 90 |
| Unique horizon/fold evaluation slices after de-duplication | 9 |
| Forecast rows across the 9 unique slices | 6,609 |

The 27 duplicate slots reproduce the same statistics within their horizon/fold slice. They are not additional independent forecast observations and must not be counted ten times in aggregate metrics. Aggregate statistics in this ADR are therefore calculated once per unique horizon/fold slice and weighted by its prediction count.

No model family, G3 parameter, feature definition, cost threshold, fold definition, or acceptance gate was changed by G4.5. The run used `--diagnostics-only`; it did not execute a trade simulation.

## 3. Measured results

### 3.1 Aggregate metrics

All aggregate rates and widths below are weighted by prediction count across the nine unique horizon/fold slices.

| Metric | Observed |
|---|---:|
| Positive median forecasts | 3,085 / 6,609 |
| Median-positive forecast rate | 46.6788% |
| Weighted mean forecast median | +10.25 bps (+0.1025%) |
| Weighted mean raw quantile interval width | 4.5249% (452.49 bps) |
| Weighted mean conformal interval width | 5.3333% (533.33 bps) |
| Mean `Lower Bound - Cost`, 50 bps scenario | -358.75 bps |
| Mean `Lower Bound - Cost`, 90 bps scenario | -398.75 bps |
| Forecasts above 50 bps cost | 0 / 6,609 (0%) |
| Forecasts above 90 bps cost | 0 / 6,609 (0%) |

The cost gate is strict: `Lower Bound - Round-trip Cost > 0`. Equality does not pass.

### 3.2 Horizon/fold breakdown

The conformal width is shown as a percentage of simple return. Cost distances are signed basis points; negative values mean the calibrated lower bound is below the relevant round-trip cost.

| Horizon | Fold | Forecast rows | Median positive | Mean conformal width | LB − 50 bps | LB − 90 bps |
|---|---|---:|---:|---:|---:|---:|
| 1h | WF-01 | 743 | 45.6258% | 2.6131% | -190.90 bps | -230.90 bps |
| 1h | WF-02 | 743 | 42.1265% | 1.5651% | -131.57 bps | -171.57 bps |
| 1h | WF-03 | 743 | 48.0485% | 1.7439% | -141.89 bps | -181.89 bps |
| 4h | WF-01 | 740 | 52.8378% | 4.8851% | -322.13 bps | -362.13 bps |
| 4h | WF-02 | 740 | 42.5676% | 3.2426% | -210.64 bps | -250.64 bps |
| 4h | WF-03 | 740 | 38.7838% | 3.4269% | -239.99 bps | -279.99 bps |
| 24h | WF-01 | 720 | 53.3333% | 12.3219% | -813.81 bps | -853.81 bps |
| 24h | WF-02 | 720 | 50.2778% | 8.8112% | -579.48 bps | -619.48 bps |
| 24h | WF-03 | 720 | 46.8056% | 9.8356% | -626.30 bps | -666.30 bps |

### 3.3 Interpretation

1. **Direction / forecast location:** the median-positive rate is below 50% overall, and the aggregate forecast median is only +10.25 bps. This does not demonstrate directional skill. The 1h and 4h aggregate medians are approximately zero/slightly negative. The 24h median is positive on average, but is much smaller than its calibrated interval width and does not translate into a qualifying lower bound.
2. **Uncertainty:** conformal intervals expand with horizon. The weighted mean 24h width is 10.3229%, and all three 24h folds have negative lower-bound-to-cost distances well beyond the 50/90 bps cost levels. The 1h and 4h intervals are narrower but their lower bounds also fail cost in every fold.
3. **Insufficient conservative edge after cost:** the signed lower-bound-to-cost mean is negative in all nine slices under both cost scenarios. No forecast meets the required strict gate. The 40 bps difference between the cost scenarios shifts the gap by exactly 40 bps but does not change any decision.
4. **Stability:** no horizon/fold slice passes the lower-bound gate. The magnitude of the shortfall varies substantially, especially at 24h, but the NO-GO decision is stable across all nine slices.

The direct cause of abstention is therefore **systematic failure of the calibrated lower-bound economic gate**. Weak/near-zero forecast location and wide calibrated uncertainty jointly explain why the lower bound remains far below costs. The data do not support attributing the outcome solely to conformal width, model direction, cost assumptions, or volume features.

## 4. Dataset limitation: zero volume

The prior dataset inspection found all 31,801 volume values equal to zero. This makes volume-derived inputs non-informative in this dataset if their transformations remain constant. However, the G4.5 artifact does not contain a feature-level causal ablation. The evaluated feature set also includes price-derived return, volatility, range, ATR and time features.

Therefore:
- zero volume is a verified dataset limitation from the prior inspection;
- it is **not** a proven cause of the weak forecast distribution or abstention;
- no feature-removal claim is made by this ADR.

## 5. Decision and next research gate

**Economic path:** HGBQ-CQR remains **NO-GO**. Do not promote it, do not treat abstention as a trading success, and do not proceed to live or paper execution.

**Recommended next research step:** at most one separately preregistered, pre-hack-only feature ablation that explicitly tests the zero-volume/volume-derived feature family against the unchanged baseline. It must use a new experiment/configuration fingerprint and its own acceptance criteria. It must not edit the locked G3 configuration, silently change this run, or mix its results into G4.5.

The ablation should be judged on forecast-location stability and calibrated lower-bound margin against both frozen cost thresholds across all walk-forward folds, not on a higher median-positive percentage alone. If it does not produce a material, stable improvement in lower-bound-to-cost results, close the HGBQ-CQR economic path rather than continuing parameter search. A simpler model is not justified as an automatic next step until the controlled ablation shows whether the current feature contract is the dominant issue.

This ADR does not itself authorize that next experiment, retraining, feature removal, model replacement, parameter search, or any additional economic evaluation. Such work requires a separate owner-authorized preregistration and scope.

## 6. Safety invariants verified in Run #11

The JSON reports:

- `real_data_scope = PRE_HACK_ONLY`
- `holdout_bytes_read = false`
- `private_api_calls = 0`
- `real_order_attempts = 0`
- `paper_orders = 0`
- `hypothetical_trades_simulated = false`
- `trade_ledger_execution = false`
- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`

The workflow's post-run validation step succeeded, including the diagnostics-only output checks and the assertion that no trade CSV was produced. No Holdout/post-hack bytes, private API, simulated trades, paper orders, or real orders were involved.

## 7. Temporary main workflow exception and cleanup

The original frozen `main` commit was `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`. Under explicit owner authorization, the single workflow file `.github/workflows/regime-r-evaluation.yml` was temporarily exposed on `main` in commit:

- `ec75a9a6d6547f401ff9a2c8b5894843fcbbd2e6`

The workflow was dispatched only against `research/nobitex-hf-strategy`; Run #11 confirms that exact branch and SHA. No G4.5 run was dispatched on `main`. The temporary file is to be deleted from `main` after this ADR commit's Standard CI succeeds. No other `main` file or model/G3 configuration is in scope.

The cleanup is an additive deletion commit; it will leave the original commit in history and make the temporary workflow path absent from the current `main` tree. No force reset is authorized or required.

## 8. Final verdict

- **G4.5 observability:** ACCEPTED; complete and artifact-verified.
- **Abstention diagnosis:** CONFIRMED; zero lower bounds cleared either cost gate.
- **HGBQ-CQR economic strategy:** NO-GO; no Champion and no promotion.
- **Next research recommendation:** one separately preregistered volume-feature ablation at most; close the economic path if it fails the lower-bound stability/cost criteria.
- **Trading authorization:** NO-GO.

Architecture First. Evidence Always. Fail Closed.
