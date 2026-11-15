# Owner Decision Memo — Nobitex 1h Directional Family

- **Date:** 2026-11-15 UTC
- **Decision owner:** Project owner
- **Status:** Options and engineering recommendation only; no owner decision recorded
- **Evidence anchor:** Four-Arm Run #10 / Run ID `43712086361`; result artifact `14163527266`; ADR-042 and ADR-043

## Executive finding

Run #10 produced 0 PASS, 3 FAIL and 9 INSUFFICIENT_SAMPLE. The three failures
are repeated references to one threshold-invariant Donchian control, not
three independent forecast strategies. All nine B/C/D forecast cells have
fewer than 100 completed trades. Every numeric aggregate PF reported in the
cohort is below 1.0; two PF pairs are MISSING because the corresponding cells
have only one trade or zero trades. All 36 fold-stability checks failed.

This supports stopping casual threshold iteration. It does **not** establish
that all nine forecast-driven cells are economic failures, and the newly
adopted Futility predicate in ADR-043 is prospective only.

## Run #10 threshold trend

NetPF values below are aggregate NetPF@50bps.

| Arm | 240bps trades / PF | 180bps trades / PF | 120bps trades / PF |
|---|---:|---:|---:|
| B | 1 / MISSING | 11 / 0.6324 | 28 / 0.4765 |
| C | 24 / 0.5724 | 35 / 0.5240 | 70 / 0.4019 |
| D | 0 / MISSING | 7 / 0.3973 | 41 / 0.3981 |

Best observed NetPF@50bps = **0.6324**; best observed NetPF@90bps =
**0.3936**. No numeric aggregate NetPF reached 1.0. Arm B and C show a clear
inverse relationship across their three tested thresholds; Arm D does not
show a strictly inverse trend (0.3973 at 7 trades versus 0.3981 at 41).
The two missing PF values prevent claiming that every cell has a measured PF.

The signal-density artifact does not identify how many valid forecast bars
crossed each threshold or how many were suppressed by entry/position/risk
gates. The exact location of attrition remains **شاهد کافی نیست** pending the
diagnostic specified in
[`FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_SPEC.md`](../diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_SPEC.md).

## Options

### Option 1 — Lower thresholds to force trades >= 100

**Potential benefit:** More completed trades could satisfy the current sample
gate for B/C/D.

**Evidence against:**
- At 120bps, the maximum completed-trade count among B/C/D is only 70.
- B degrades from 0.6324 at 11 trades (180bps) to 0.4765 at 28 trades
  (120bps).
- C degrades from 0.5724 at 24 trades to 0.4019 at 70 trades.
- Every observed aggregate PF remains below 1.0, and the lower-threshold
  observations do not show an improving economic edge.
- D is nearly flat, not evidence of recovery; at both 7 and 41 trades its PF
  remains approximately 0.398.

**Risk:** Lowering the threshold could increase the sample while simply
confirming a negative edge. It would be a new hypothesis and would require a
new preregistration; no threshold change is authorized here.

### Option 2 — Retire the directional research line and pivot to volatility research

**Potential benefit:** Stop spending cycles on repeated directional threshold
variants and redirect research toward the TTM-R2 volatility signal.

The previously reported TTM-R2 comparison is QLIKE **0.27786 versus 0.39348**
(approximately **29.38% lower**). This is a forecast-quality result, not proof
of trading profitability. The source artifact and protocol must be attached
and reverified in any future preregistration.

**Important distinction:** ADR-043's Futility predicate is future-only and
was not preregistered for Run #10. The current data do not retroactively
trigger a formal `FUTILE` verdict: Arm D is not strictly inverse and two
forecast-cell PF values are missing. The owner may nevertheless choose to
retire the directional workstream as a **portfolio/governance decision**.
That decision must not be described as a retrospective statistical result
under ADR-043.

**Next path if selected:** design and preregister a volatility-centric TTM-R2
study with explicit forecast-to-position translation, costs, execution rules,
fold stability and independent economic gates. Do not open holdout.

### Option 3 — Close the project with a documented NO-GO

**Potential benefit:** A clean stop with no more model, threshold or strategy
experiments.

**Evidence:**
- The owner-provided project summary cites eight directional research paths,
  zero profitable cells and zero stable folds. The eight-path inventory was
  not independently reconstructed in this bounded review; its underlying
  trial IDs/artifacts should remain linked in the project ledger before
  presenting it as a fully re-audited count.
- Run #10 has zero PASS cells, three repeated-control FAIL records, nine
  under-sampled forecast cells and 0/36 stable folds.
- No numeric aggregate PF in Run #10 reaches 1.0; two forecast-cell PF pairs
  are MISSING.

**Boundary:** A project-level NO-GO can be an owner resource-allocation
decision. It is not the same as a statistically established family-wide
economic failure, because nine forecast cells did not meet the sample gate
and ADR-043 cannot be applied retrospectively.

## Engineering recommendation

**Recommend Option 2 as a governance decision, not as a formal ADR-043
FUTILE trigger:** retire the current directional threshold-search workstream,
do not lower thresholds, and move the next research design effort to a
TTM-R2-based volatility hypothesis. The reason is the absence of a measured
positive aggregate edge, the inverse PF/count trend in B and C, zero fold
stability, and the reported independent QLIKE advantage. Keep ADR-043 as a
prospective stopping rule if a new directional family is ever preregistered.

The recommendation is not an owner authorization. Option 3 remains a valid
owner choice if project closure is preferred. Option 1 is the least defensible
use of research budget given the observed PF trend.

## Non-negotiable controls

- `GLOBAL_N=UNKNOWN` remains UNKNOWN.
- DSR remains uncomputed; numeric DSR sensitivities remain MISSING.
- No Run #10 manifest or registry record is altered by this memo.
- No threshold is changed and no new trial is registered.
- No local/remote diagnostic, model inference, or economic dispatch is
  authorized by this memo.
- No holdout access, release, trading permission, or authorization change.
- Frozen `main` remains `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.

---

## Append-only addendum — TTM-R2 provenance reconstruction (2026-11-15)

The prior QLIKE comparison is now traceable to primary project records rather than an unsupported summary:

- Decision/closure documents: docs/ttm_r2_closure.md and docs/ttm-r2-risk-evaluation.md.
- Exact GitHub execution: [TTM-R2 Risk-Only Research #25, attempt 2 / Run ID 43094730462](https://github.com/angry0tiger-sudo/AliX/actions/runs/43094730462), code commit ad75f44800d3024bc50e7654b1762f49298ef0d9.
- Immutable report artifact: 13916547485, name alix-ttm-r2-risk-only, 510,786 bytes, artifact ZIP SHA-256 15718b3b763ffe8d28d2ee3dd049ba00c2aa71897c6d147fa5d2cf31a58110a8.

### Forecast protocol and exact sample

- Dataset: 31,801 native Nobitex BTCUSDT 1h rows, SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e, window 2021-10-31 20:30 UTC through 2025-06-17 20:30 UTC.
- Target: hourly log1p((log(close_t / close_(t-1)))^2); 512 closed hourly context observations; 24 hourly forecast steps. The 24 hourly inverse-scaled variance predictions are summed to form the 24h variance forecast.
- Model: ibm-granite/granite-timeseries-ttm-r2, zero-shot frozen weights, immutable revision d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f, weights SHA-256 a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710.
- Chronological split: 70% training-only scaler fit, 15% validation, final 15% holdout. Forecast origins are 24h apart. The 198 final-holdout decisions run from 2024-12-01 02:30 UTC through 2025-06-16 02:30 UTC; their target horizons end from 2024-12-02 02:30 UTC through 2025-06-17 02:30 UTC. The artifact also contains 198 separate validation forecasts.
- Actual 24h realized variance is the sum of the next 24 squared close-to-close log returns. The baseline is trailing-24h realized-variance persistence: sum of the preceding 24 squared log returns through the forecast origin.
- QLIKE is the mean of actual_variance / forecast_variance - log(actual_variance / forecast_variance) - 1, lower being better. Final holdout: 198 forecasts; TTM QLIKE 0.27785936698219454; persistence QLIKE 0.3934795684925382; improvement 29.3840%. Log-variance MAE also improved (0.63270646 vs 0.69222535).
- The QLIKE forecast comparison is cost-free by protocol; 50 bps round-trip costs were applied only to the separate portfolio replay.

### Purge / embargo clarification

The code creates 24h non-overlapping forecast origins and explicitly checks that every 24h target window lies wholly inside its assigned chronological split. The first validation target begins at the training boundary; the first holdout target begins at the validation boundary. This prevents target windows crossing split boundaries, and the 24h stride prevents overlapping target windows. However, the implementation does not reserve a separate additional 24-bar embargo gap beyond those split-contained target-window constraints. Therefore the accurate description is split-contained, non-overlapping targets; no separately encoded embargo buffer, not an unqualified claim that a conventional purge/embargo gap was applied.

### Economic interpretation and prerequisite for any future pivot

QLIKE improvement is evidence of variance-forecast quality, not proof of trading profitability. The same Run #25 artifact shows the offline 50bps portfolio replay still had 480 holdout trades, NetPF 0.3748 under TTM sizing (0.3731 fixed-size), max drawdown 18.26% (21.63% fixed-size), win rate 31.67%, and net return -18.17% (-21.56% fixed-size). Runtime sizing was not enabled.

Before any TTM-R2 volatility pivot can be preregistered, a new hypothesis must define the forecast-to-position translation (risk multiplier and exposure caps), causal entry/exit and stop behavior, fee/slippage/fill assumptions with the registered 50bps baseline and cost sensitivity, non-overlapping chronological validation with explicit boundary policy, fold-stability and sample-frequency gates, and independent economic acceptance criteria. No pivot, trial, threshold, holdout access, or execution is authorized by this provenance reconstruction.

This provenance is sufficient to cite the QLIKE observation, but it does not authorize a strategy pivot. The owner retains all three project options.

---

## Append-only addendum — harvested signal-density evidence and option reassessment (2026-11-15)

### Evidence harvested

Three separate diagnostic runs (43739166184 / artifact 14171940837; 43739941516 / 14172285941; 43740274764 / 14171877977) each produced the same extracted JSON and Markdown bytes. The ZIP wrappers have distinct SHA-256 values; the internal report file hashes are identical. The complete durable source, 27-fold table and nine histograms are now recorded in:
- docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_RESULTS.json
- docs/research/diagnostics/FOUR_ARM_SIGNAL_DENSITY_DIAGNOSTIC_HARVEST_REPORT.md

The audit used only the 27,825-row confirmatory prefix (source SHA-256 0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e) and the Run #10 forecast cache (1,138 anchors; cache SHA-256 aea6c537f3ca79a28e75faf1a118bd57d5dea4fb58b45e477eb401df03d01868). It made zero inference calls, computed no PF/PnL/equity/drawdown metric, registered zero trials, and did not access holdout.

### Bottleneck

Each B/C/D × threshold cell has 10,500 forecast-covered bars. Only 0–552 bars (0.00%–5.26%) clear the registered threshold. This is the first attrition point. Among qualified bars, 70.8%–95.8% are suppressed while a position is open. ATR-invalid and no-next-bar rejects are zero. There is no separately instrumented pre-entry risk gate, so risk rejects are MISSING, not zero.

The histogram shows a 25%-relative threshold reduction would add forecast-qualified bars: B +356/+48/+72, C +360/+144/+120, D +72/+144/+24 for the current 120/180/240 thresholds. These are bar counts only. They are not new trade counts, do not account for occupied positions, and have no PF attached.

### Reassessment of the three options

**Option 1 — Lower the thresholds.** The density evidence confirms more forecast bars would qualify at lower thresholds, but not that a 100-trade sample or profitable edge would result. Existing Run #10 economics move against lowering for B and C: NetPF@50 declines from 0.6324 at 180bps / 11 trades to 0.4765 at 120bps / 28 trades for B, and from 0.5240 / 35 trades to 0.4019 / 70 trades for C. D is nearly flat (0.3973 / 7 versus 0.3981 / 41). All numeric PF values remain below 1.0. Do not lower thresholds on this evidence. A lower cutoff would be a new hypothesis requiring separate owner approval and preregistration, with economic gates intact.

**Option 2 — Declare FUTILE and pivot to TTM-R2.** A retrospective FUTILE declaration is not supported. Run #10 is outside ADR-043's prospective scope; the prior evaluability definition yields 7/9 numeric PF cells, but D's NetPF@50 descriptive slope is positive rather than strictly negative. The 25%-lower signal counts add no economic proof. Separately, TTM-R2's 198-holdout QLIKE improvement (0.27785937 vs 0.39347957) establishes variance-forecast quality only; its 50bps holdout sizing replay still had NetPF 0.3748, MaxDD 18.26%, 480 trades, and -18.17% net return. Option 2 remains a permissible portfolio/governance choice to stop directional threshold iteration and prioritize a separately designed volatility-risk hypothesis, but it is not an ADR-043 verdict and does not authorize execution or runtime sizing.

**Option 3 — Close the project with a documented NO-GO.** This remains a valid owner-level resource-allocation choice given 0 PASS, 0/36 stable Run #10 folds, repeated-control FAIL records and nine under-sampled forecast cells. It must not be represented as a statistically proven family-wide failure: two PF pairs are null, and the nine forecast-driven cells did not meet the sample gate.

### Engineering recommendation

Recommend Option 2 as a governance priority only, with a hard boundary: retire casual directional threshold iteration, do not lower the current thresholds, retain TTM-R2 as an offline risk-forecasting result only, and require a new owner-approved preregistration translating volatility forecasts into capped positions, costs/slippage/fills, chronological fold stability and independent economic acceptance. Do not declare Run #10 FUTILE, open holdout, change runtime sizing, dispatch a trial, or arm any execution path.

Option 3 remains the owner's alternative if project closure is preferred. The owner retains the final decision. No option is executed by this memo.

### Accounting and governance boundaries

- Run #10's 12 trial registry records and ADR-042 verdict are unchanged; no new trial record was added.
- Run #10 PF evaluability remains B=2/3, C=3/3, D=2/3; B/240 and D/240 remain PF=null / not evaluable.
- GLOBAL_N=UNKNOWN; DSR numeric values remain MISSING.
- The diagnostic executable and workflow are not restored; only report data, provenance and decision records are added.
- Lock NOT_AUTHORIZED; both triggers PROBE; Champion=null; order_permission=false; trading_authorization=NO-GO.
