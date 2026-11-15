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
