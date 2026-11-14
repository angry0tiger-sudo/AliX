# Four-Arm DSR Evidence Contract — ADR-038

- GLOBAL_N = UNKNOWN; it must not be replaced by a guessed value.
- The audited historical lower bound carried into this experiment is at least 7,912 historical trials. Unresolved candidate identities and overlap remain disclosed.
- The twelve registered threshold × arm cells are retained in candidate accounting. The three Arm A cells are the same signal rule, computed once and disclosed as three preregistered matched-control cells.
- Report raw descriptive Sharpe only when computed from the actual recorded OOS trade-return series; never label it DSR.
- Numeric DSR is unavailable until candidate-level reconciliation establishes a defensible global trial count and the frozen estimator can be applied.
- Every trial manifest reports dsr=null, dsr_status=INSUFFICIENT_EVIDENCE_GLOBAL_N_UNKNOWN, the audited lower bound, unresolved remainder, and sensitivity multipliers 1×, 2×, 5× and 10×.
- The sensitivity table is explicitly marked NOT_COMPUTED while GLOBAL_N is UNKNOWN. No numeric DSR estimate may be invented.
- No candidate can PASS, become a winner, or become Paper-eligible while DSR is unavailable.
- A trial with fewer than 100 completed trades is INSUFFICIENT_SAMPLE, never PASS.
- Final acceptance requires NetPF >= 1.10 at both 50 and 90 bps, at least 100 trades, MaxDD <= 15%, all three fold gates, and a defensible DSR > 0.
