# Owner Directive: Self-Evolving AI Research

**Date:** 2026-10-25  
**Branch:** `research/self-evolving-ai`  
**Status:** Design baseline; research-only

## Direction

The owner wants AliX to discover its own trading hypotheses through evolutionary search, feature composition, adaptive exits/sizing, post-mortem analysis, and reinforcement learning. Do not hardwire a preferred timeframe, indicator stack, entry recipe, or take-profit rule into the strategy-search space.

## Interpretation and trade-offs

“No hardcoded rules” applies to alpha and strategy behavior. It does not authorize unlimited computation, non-causal features, fabricated data, unsafe generated code, or self-modification of governance. Use an extensible typed grammar and explicit compute budgets. Candidates may evolve strategy genes; they may not evolve the simulator's truth, scoring definition during a run, quality gates, or trading authorization.

The requested 1-minute-to-1-month timeframe range is data-dependent. The previous research artifact is hourly. Lower-than-hourly simulation requires source minute bars; do not synthesize them from hourly OHLCV.

The target win rate above 60% is a metric, not an economic guarantee. Acceptance must jointly require positive net out-of-sample expectancy, drawdown limits, adequate sample size, regime robustness, and cost sensitivity.

## Non-negotiable invariants

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- No live API calls, order submission, execution credentials, or real trades.
- All training, simulation and economic research run through GitHub Actions.
- Existing quality gates remain intact.
- No final-holdout feedback or repeated holdout tuning.

## Implementation order

Follow the phases in `docs/knowledge_base/SELF_EVOLVING_AI_ARCHITECTURE.md`. Do not combine evolutionary search and RL before each has a reproducible baseline. Keep the architecture and evidence artifacts commit-addressable. Economic failure or uncertainty leaves the system at NO-GO.


## Owner Directive: GA eligibility, drawdown and timeframe coverage (2026-10-25)

- A genome with fewer than 30 completed training trades or fewer than 10 completed validation trades has numeric fitness exactly `0.0` and is sample-ineligible.
- Eligibility is a separate ranking dimension. Numeric zero must never allow an invalid genome to beat a valid candidate with negative fitness. Selection output excludes ineligible genomes from final candidate selection.
- Training expectancy is the optimization score. Validation return and validation drawdown are not optimization objectives; validation trade count is used only as the explicit minimum-sample eligibility gate. The final 20% holdout is never read during evolution or candidate selection.
- If training max drawdown exceeds 15%, apply `min(raw_score, 0.0) - 10 * (drawdown - 0.15)`, and exclude that genome from final candidate selection. Validation drawdown above 15% or non-positive validation expectancy fails economic acceptance after the training-scored candidate is selected.
- The initial population and every generation include at least four unique genomes for each required timeframe: `1h`, `4h`, and `1d`. Report per-timeframe evaluation counts, qualification counts, best eligible fitness and rejection reasons.
- The research path fetches up to 24 calendar months of public BTCUSDT 1-minute Spot data, validates complete coverage/gaps, and publishes source CSV plus manifest/hash as a workflow artifact. Multi-asset simulation is deferred to avoid coupling this acceptance experiment to portfolio-accounting changes.
- The only permitted outcome is offline research. Keep `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, zero private API calls and zero real-order attempts.

These gates are fixed for the duration of an experiment and may not be weakened to obtain a passing report.


## Owner Directive: Opportunity Hunter (supersedes the earlier hard 30/10 rule, 2026-10-25)

This directive supersedes the earlier hard minimums of 30 training trades and 10 validation trades. No hard completed-trade-count threshold may zero out a candidate's fitness. Small samples receive continuous confidence weighting, and raw trade counts and PF are still reported.

- Fitness combines profit factor, net total return, positive regime-capture returns and a penalty for missed actionable upside windows.
- A missed-upside event is a non-overlapping 24-hour price window with a realized return of at least +5%, throughout which the candidate had no long exposure. This is a retrospective evaluation metric only, never a future-informed entry feature.
- The genome includes a profit-activated trailing stop and a dynamic exit mode: indicator threshold, indicator-or-EMA-cross, indicator-or-close-below-trend-EMA, or indicator-and-EMA-cross. Trailing-stop and activation fractions are evolved independently.
- 15m joins 1h, 4h and 1d as a mandatory timeframe. Report unique genomes, trades, PF, net return, missed upside and high-volatility/strong-trend capture.
- PF is reported as undefined when gross losses are zero. Only the bounded fitness calculation may use the configured PF cap, with soft sample weighting.
- Training-only fitness determines evolutionary selection. Validation assesses the selected candidate and is not fed back into tournament ranking. Holdout remains untouched.
- The 15% drawdown penalty and eligibility limit remain. Removing hard trade-count minima does not cancel the independent risk constraint.
- Safety invariants stay `Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, zero private API calls and zero real-order attempts.
