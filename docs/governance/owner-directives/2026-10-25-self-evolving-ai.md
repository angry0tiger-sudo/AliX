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
