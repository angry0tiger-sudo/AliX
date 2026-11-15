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
