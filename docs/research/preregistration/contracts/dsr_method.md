# Four-Arm DSR Evidence Contract

**Registration:** NOBITEX-FOUR-ARM-KC-20261113  
**Date:** 2026-11-14

## Required disclosure

- The historical trial registry explicitly records `GLOBAL_N=UNKNOWN`.
- Historical families contain heterogeneous units (workflow executions, candidate variants, trained candidates, genomes and strategy/policy experiments). They cannot be blindly summed into a numeric global trial count.
- A numerical Deflated Sharpe Ratio requires a defensible trial-count input and a fixed reproducible estimator. A guessed or convenient `N` is forbidden.

## Four-arm result rule

- Report descriptive raw Sharpe only if computed from the actual, recorded OOS trade-return series; do not label it DSR.
- Set `dsr=null` and `dsr_status=INSUFFICIENT_EVIDENCE_GLOBAL_N_UNKNOWN` unless a candidate-level reconciliation establishes a defensible DSR cohort and the frozen DSR implementation can be run.
- This status cannot satisfy the registered `DSR > 0` acceptance gate.
- No model can be declared PASS, Paper-eligible, or a winner solely from PF/trades/MaxDD while DSR remains unavailable.
- The final verdict in that condition is `NO_GO_INSUFFICIENT_DSR_EVIDENCE`, not a numeric DSR estimate.

This contract deliberately fails closed. It preserves the registry's UNKNOWN instead of fabricating multiple-testing precision.
