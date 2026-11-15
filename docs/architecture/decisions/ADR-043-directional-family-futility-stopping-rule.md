# ADR-043: Futility Stopping Rule for the Directional Family

- **Status:** Accepted for future preregistered cohorts only
- **Date:** 2026-11-15 UTC
- **Branch:** `research/nobitex-hf-strategy`
- **Supersedes:** No existing Run #10 verdict, ADR-038 sample gate, or ADR-042 decision
- **Scope:** Future Nobitex 1h directional-family cohorts preregistered after this ADR is committed

## Context

The existing rule classifies cells with fewer than 100 completed trades as
`INSUFFICIENT_SAMPLE`, while a family-level directional NO-GO requires an
all-FAIL cohort. This can leave a weak, low-frequency strategy indefinitely
unresolved: lowering the signal frequency can keep cells below the sample
gate without producing evidence of economic viability.

Run #10 motivates a prospective stopping rule. It does not provide a
retrospective pass/fail result under this ADR because the rule was not
preregistered before Run #10 and the available cohort is mixed.

## Decision: future-only FUTILE predicate

For a future, explicitly preregistered directional-family cohort, declare the
family **FUTILE** at the next safe stopping boundary only when every condition
below is satisfied:

1. Every registered, unique, non-duplicate signal-rule cell has a numeric
   aggregate NetPF@50bps **and** NetPF@90bps, and every such value is strictly
   below 1.0. A missing, null, non-finite, or uncomputed PF means the predicate
   is **NOT EVALUABLE**, not FUTILE.
2. Duplicate trial IDs that reference the same signal rule and same effective
   signal series are deduplicated for the gradient calculation. They remain
   visible in the trial ledger but do not receive extra statistical weight.
3. Across at least three distinct, non-duplicate cells with positive completed
   trade counts, the ordinary-least-squares slope of aggregate NetPF against
   completed trade count is strictly negative for **both** cost scenarios
   (50bps and 90bps). The preregistration must freeze the cell identity,
   aggregation window, cost definitions, and slope calculation before data
   are evaluated.
4. The 100-trade sample gate does not block this futility assessment: the
   predicate may be evaluated below 100 trades whenever the aggregate PF is
   numerically defined. This does not waive any existing promotion, DSR,
   holdout, or owner-authorization gate.

If any condition is not met, report `NOT_FUTILE` or
`FUTILITY_NOT_EVALUABLE` with the failed/missing predicate. Do not lower
thresholds or add tuning variants merely to force a verdict. Once FUTILE is
established under a preregistered cohort, stop remaining directional-family
trials at the next safe boundary; do not dispatch additional cells.

## Run #10 observation — motivation, not retrospective application

The following table records the Run #10 aggregate trend. PF values shown are
NetPF@50bps. The 240bps B and D values are MISSING because their manifests
contain one and zero completed trades respectively.

| Arm | 240bps: trades / NetPF@50 | 180bps: trades / NetPF@50 | 120bps: trades / NetPF@50 |
|---|---:|---:|---:|
| B | 1 / MISSING | 11 / 0.6324 | 28 / 0.4765 |
| C | 24 / 0.5724 | 35 / 0.5240 | 70 / 0.4019 |
| D | 0 / MISSING | 7 / 0.3973 | 41 / 0.3981 |

The best observed aggregate NetPF@50bps was **0.6324** (Arm B, 180bps).
The best observed aggregate NetPF@90bps was **0.3936** (Arm B, 180bps).
No numeric aggregate PF in this cohort reached 1.0. All 36 fold records failed
the registered fold-stability gate.

The observed trend is inverse for B and C as trade count rises toward the
lower threshold. D is effectively flat and slightly higher at 41 trades than
at 7 trades; its 0-trade PF is missing. The three Arm-A threshold records are
one repeated control and are not independent gradient observations.

Accordingly, this table is an engineering motivation only. This ADR does not
label Run #10 FUTILE, does not amend its 3 FAIL / 9 INSUFFICIENT_SAMPLE result,
and does not trigger a retrospective directional-family NO-GO.

## Statistical accounting and safety

- `GLOBAL_N=UNKNOWN` remains unchanged.
- DSR was not computed; all numeric DSR sensitivities remain MISSING.
- This ADR does not create a candidate, trial, threshold, or execution
  authorization.
- No holdout release or access is authorized.
- Existing lock, trigger, dry-run, live-trading and frozen-main controls remain
  unchanged.

---

## Append-only clarification — evaluability and OLS pooling (2026-11-15)

This clarification applies only to directional-family cohorts preregistered after this addendum is committed. It does not revise the original Run #10 decision and does not retroactively declare Run #10 FUTILE.

### Evaluable cell

A registered, unique, non-duplicate signal-rule cell is EVALUABLE only when both aggregate NetPF@50bps and aggregate NetPF@90bps are present, finite numeric values. A null, missing, non-finite, or uncomputed value at either cost makes the cell NOT EVALUABLE. A zero-trade cell is not assigned PF=0.

For Run #10, Arm B / 240bps (one trade) and Arm D / 240bps (zero trades) have null PF at both costs and are not evaluable. The remaining seven forecast-driven B/C/D cells have numeric PF at both costs. All seven numeric PF pairs are below 1.0.

### Minimum evaluable cohort size

The future FUTILE predicate may be assessed only when:
- at least 6 of the 9 unique B/C/D threshold cells are evaluable in both cost scenarios; and
- each of arms B, C, and D contributes at least 2 of its 3 distinct threshold cells, with at least two distinct completed-trade-count values within that arm.

If either requirement fails, the family result is FUTILITY_NOT_EVALUABLE, not FUTILE and not an economic PASS. Missing cells remain disclosed and are never imputed.

### OLS pooling rule — per arm, not pooled across arms

The OLS gradient is calculated separately within each arm. For each arm and each cost scenario, regress that arm's aggregate NetPF on completed-trade count across its evaluable, non-duplicate threshold cells. The family gradient condition is satisfied only when the slope is strictly negative for all three arms at both costs.

Do not pool B/C/D observations into one cross-arm regression. Their forecast generators have different signal distributions and PF baselines; a pooled slope could confound between-arm level differences with within-arm threshold response (a Simpson-type aggregation error). The per-arm rule preserves the actual threshold-response question. Arm A is the repeated threshold-invariant control and is excluded from directional-family slope/evaluability counts.

The Futility predicate also requires every evaluable cell's numeric PF@50 and PF@90 to be strictly below 1.0. The 6/9 and 2-per-arm conditions define when the family-level test is sufficiently populated; they do not permit imputing missing cells.

### Run #10 descriptive OLS — not a retrospective verdict

Run #10 has 7/9 evaluable directional cells: B=2/3, C=3/3, D=2/3. The two non-evaluable cells are B/240 and D/240. OLS slopes below are descriptive slopes in PF units per completed trade, calculated separately by arm from the available numeric cells:

| Arm | Evaluable cells | OLS slope NetPF@50 vs trades | OLS slope NetPF@90 vs trades |
|---|---:|---:|---:|
| B | 2 | -0.0091689642 | -0.0054190428 |
| C | 3 | -0.0036525463 | -0.0033043201 |
| D | 2 | +0.0000242250 | -0.0014757342 |

B and C have negative slopes at both costs. D has a small positive NetPF@50 slope: PF@50 improves slightly from 0.397253 at 7 trades to 0.398077 at 41 trades; its NetPF@90 slope is negative. Thus even the descriptive Run #10 values do not satisfy the all-arm/all-cost gradient condition. More importantly, ADR-043 was not preregistered for Run #10. These slopes are motivation and method calibration only, not a formal retrospective FUTILE application.

The best observed NetPF@50 remains 0.6324 and best observed NetPF@90 remains 0.3936. No numeric aggregate PF reached 1.0. GLOBAL_N=UNKNOWN and uncomputed DSR remain unchanged.
