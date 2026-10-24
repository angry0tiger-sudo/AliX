# Adaptive Autonomous AI: Research Architecture and Acceptance Contract

## Owner Directive and scope

This research branch implements the Owner Directive dated 2026-10-25. Within this branch, the earlier fixed-one-hour, fixed-risk-budget and fixed-take-profit assumptions are superseded for the adaptive research policy. This does not silently rewrite the prior experiment or authorize a production migration.

The policy is **research-only**. It has no exchange client, private API, order interface, production registry mutation, champion assignment or promotion path. The three safety invariants remain permanent until a separate economic review explicitly changes the governance state:

- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`

## Architecture

1. **Verified input and timestamp contract**: load the immutable Stage A artifact through the existing verified-development loader. The loader reserves the final 90-day holdout. All source candles are validated as timezone-aware, contiguous hourly OHLCV bars; zero source volume is preserved as observed, never replaced with synthetic volume.
2. **Point-in-time timeframe contexts**: construct 1h contexts and resample only complete 4h and 1d bars. Hourly candle timestamps are treated as bar-open times. Aggregated bars are labelled at their close boundary, and a context is joined to an hourly decision only when that context is already closed. Future rows must not affect prior decisions.
3. **Adaptive timeframe selection**: score each valid timeframe using trend strength and relative volatility. Rapid/high-volatility conditions increase the 1h score; strong, calmer trends favor higher timeframes. The score margin is a bounded *selection scalar*, not a calibrated probability. Missing/invalid contexts or low margin cause abstention rather than forced entry.
4. **Dynamic stop from ATR and structure**: initial long stops are placed below both the ATR envelope and the prior 20-bar support with an ATR buffer. The support calculation excludes the current bar. If the required stop is invalid or too far from entry, the policy skips the trade instead of pulling the stop closer merely to manufacture eligibility.
5. **Trailing exit with no fixed take-profit**: there is no fixed target price. After each bar closes, a strong trend with positive MACD histogram receives more room; weakening momentum tightens the trail. The stop is monotonic and never moves down. A confirmed close below the selected timeframe's EMA with negative MACD also closes the research position. Intrabar stop checks use the prior stop; if OHLC data cannot identify intrabar order, stop-first is assumed.
6. **Adaptive position size**: the per-trade risk fraction varies between a configured 0.10% floor and 0.75% ceiling based on selection confidence, trend strength and relative volatility. Exposure is capped at 95% of equity. These are guardrails around a variable risk policy, not a fixed percentage applied to every trade. The confidence scalar is not a probability guarantee.
7. **Post-trade failure analysis**: only after a position is closed, classify the observed outcome (probable tight stop, market-state shift, signal reversal, generic loss, or profitable close). Append the evidence and parameter state before/after to the learning ledger. Stop, trailing and confidence parameters may change only within configured bounds; those changes apply to subsequent entries, never to the trade that triggered them. A label is a heuristic explanation, not proof of causality.
8. **Independent baseline runs**: evaluate the adaptive selector against fixed 1h, 4h and 1d variants using the same verified development history, assumed fees/slippage and risk envelope. The result is a sequential historical simulation. Since the adaptation policy is hand-specified in this first version, these results are exploratory and must not be described as untouched OOS evidence.
9. **Evidence and safety verification**: publish a report, trade ledger, decision ledger and post-trade learning ledger. Every report must identify the exact commit and config hash, exclude the final holdout, state the assumed costs, and assert the immutable safety invariants.

## Bounded online adaptation rules

The current v1 policy deliberately uses interpretable, bounded feedback instead of an unreviewed model mutation loop:

- A losing trailing-stop exit is labelled \`stop_too_tight\` only when the initial stop distance was below 2.5 ATR and favorable excursion reached at least 1 ATR. The next-trade stop and trailing multiples rise by one bounded step.
- A known entry/exit market-state change raises the entry-confidence floor for subsequent trades.
- Signal reversal and otherwise unexplained losses make the entry gate slightly more selective.
- Three consecutive profitable closes allow a small confidence-floor relaxation and a wider future trailing distance, subject to the same bounds.
- If the evidence does not support a reliable category, the ledger records the generic category rather than inventing a market explanation.

This feedback is deliberately not an unconstrained self-modifying model. The classifier and update rules are transparent hypotheses to test, and each parameter revision is recorded.

## Non-negotiable constraints

- Research branch only; no production promotion or execution code changes.
- No private exchange endpoint, network market-data fetch, order submission or live credentials.
- \`Champion = null\`, \`order_permission = false\`, \`trading_authorization = NO-GO\`, \`economic_acceptance = NO-GO\`.
- No fixed take-profit is generated by the adaptive policy.
- No fabricated volume, spread, liquidity or order-book measurements.
- No use of the reserved final holdout for feature design, adaptation, parameter adjustment or comparisons.
- No disabled or weakened lint, typing, security, coverage, complexity or architecture gate.

## Acceptance criteria

### Technical
- Contexts for all three timeframes are aligned only to fully closed bars.
- A future-data mutation cannot change an earlier context or decision.
- Invalid OHLCV, non-finite values, unsupported stop geometry and invalid sizing fail closed.
- Strong-trend trailing gives trades more room than weak-trend trailing; the stop cannot loosen.
- Failure analysis executes only after close, emits a ledger event, and all updates remain bounded.
- Position risk responds monotonically to confidence/trend strength and is reduced by volatility.
- Config SHA-256 matches the exact module bytes.
- CI passes Ruff, strict mypy, security, dependency, architecture, complexity and coverage gates unchanged.
- Research report and all three ledgers are uploaded for the exact commit.
- All safety invariants remain asserted in test code and in the workflow report verifier.

### Economic
- Economic acceptance remains \`NO-GO\` regardless of this first run's score.
- Report trade count, win rate, net expectancy, PnL, drawdown and per-timeframe baselines with explicit assumed costs.
- Do not promote based on win rate alone or compare against only losing baselines without showing absolute negative performance.
- Require later fee/slippage sensitivity, regime/period stability, minimum sample criteria and an independently untouched validation period before economic review.
- Never use the reserved final holdout to tune the selector or feedback rules.

## Current limitations and next phases

**Current slice:** interpretable adaptive research policy, chronological simulated trades, bounded feedback and exact-commit evidence.

**Next phase:** evaluate fee/slippage sensitivity and parameter-update stability by regime. Define pre-registered economic thresholds before any new tuning. If positive source volume or venue liquidity becomes material, acquire a provenance-pinned source rather than manufacturing proxies. A trained policy or any production promotion requires a separate design review and explicit owner approval.
