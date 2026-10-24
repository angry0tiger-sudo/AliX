# Self-Evolving AI: Research Architecture and Delivery Roadmap

## Status

**Design baseline only.** This document establishes the architecture and acceptance contract for the dedicated `research/self-evolving-ai` branch. It does not claim that genetic search, neuroevolution, deep reinforcement learning, or economic validation have been implemented.

## Owner intent

The system should discover strategies rather than encode a preferred trading recipe. Timeframe choice, feature construction, indicator combinations, entry/exit behavior, and sizing policy are search variables. The system must be able to reject every candidate and remain flat when evidence is weak.

The following are **not** evolvable by candidate strategies:

- `Champion = null` until a separately reviewed economic acceptance decision.
- `order_permission = false`.
- `trading_authorization = NO-GO`.
- No private exchange API, order endpoint, live credentials, or real-order attempt.
- Data provenance, point-in-time correctness, causal evaluation, execution-cost accounting, and CI quality gates.
- No tuning on the reserved final holdout.

These are governance and scientific-integrity constraints, not trading rules. A model that can mutate them is a compromised research system, not a more intelligent one.

## Architecture principle: open search, closed execution boundary

“No hardcoded rules” means no manually selected alpha recipe or preferred timeframe is privileged. It cannot mean an infinite, unconstrained search: all computation requires a finite representation, input data, and a reproducible search budget. We will therefore use an **extensible strategy grammar**, not a giant hand-maintained list of finished strategies.

### 1. Verified data and point-in-time feature substrate

- Consume only immutable, provenance-pinned historical artifacts through the existing verified loader.
- Normalize and validate timestamps, OHLCV semantics, duplicate/missing bars, timezone, symbol boundaries, and candle-close availability.
- Build causal resampling from the finest available source bars; features from a bar may only be used after that bar is closed.
- Support candidate aggregation intervals in the intended 1-minute-to-1-month range only where the underlying source has sufficient resolution and history. Do not interpolate or synthesize lower-timeframe bars or volume.
- Store source-artifact digest, schema version, coverage range, data gaps, and eligibility in each run manifest.

**Current blocker:** the prior adaptive experiment consumes hourly candles. Hourly data can support hourly and coarser derived views, but cannot provide valid 1-minute observations. A verified minute-level source artifact is a prerequisite for claiming coverage from 1 minute upward. Month-scale candidates also need enough multi-year history and must be rejected when coverage is insufficient.

### 2. Evolvable strategy genome

Use an explicitly versioned, typed expression grammar. A genome can encode:

- A timeframe / aggregation function selected from the available causal data.
- Feature expressions composed from OHLCV, return/volatility/volume transformations, rolling statistics, and an extensible registry of validated primitive operators.
- Conditional logic and state features.
- Entry, abstain, exit, and position-sizing outputs.
- Stop/trailing behavior and model complexity budget.

The primitive registry is a vocabulary for composition, not a fixed list of approved strategies. New operators require unit tests, causal semantics, numerical bounds, and documentation. Genome validation rejects invalid lookback lengths, unsupported timeframes, non-finite expressions, look-ahead access, unbounded recursion, and resource exhaustion.

Use DEAP for evolutionary search and genetic programming (selection, crossover, mutation, diversity preservation, checkpointing). The search must retain an archive of structurally distinct candidates rather than only cloning the current top scorer. Deterministic seeds and genome serialization make every candidate reproducible.

### 3. Candidate simulator and cost model

Build one event-driven, chronological simulator used by the evolutionary engine, RL environment, baselines, and final evaluation. It must encode the observable order of candle events and prevent future-bar access. When OHLC cannot disambiguate intrabar stop/target ordering, apply the conservative rule documented by the simulator.

All candidates incur explicit, versioned assumptions for fees, slippage, latency and funding where applicable. Missing spread/order-book/liquidity evidence must be reported as missing, never fabricated. Report both gross and net outcomes, including turnover and exposure. A candidate that depends on unrealistic fills must not survive as “best.”

The simulator is a separate boundary from any exchange or order interface. Research actions produce simulated fills only.

### 4. Evolution and multi-objective selection

The objective must not reward win rate in isolation. Fitness is multi-objective and includes, at minimum:

- Net expectancy and net return after costs.
- Drawdown / tail-loss behavior.
- Stability across chronological folds and distinct market regimes.
- Turnover and sensitivity to plausible fee/slippage changes.
- Candidate complexity and behavioral diversity.
- A minimum-trade evidence requirement, set before each experiment.

Use Pareto selection or an equivalently auditable method so one scalar cannot conceal catastrophic risk. Apply early inexpensive screens, memoize feature/genome calculations, deduplicate equivalent candidates, and reserve a documented compute budget for finalists. “Thousands of strategies” means a recorded population of unique evaluated genomes, not thousands of repeated parameter variants masquerading as discovery.

Win rate above 60% is a reported target, not an acceptance criterion on its own. A high win rate with negative expectancy or severe drawdown fails.

### 5. RL / neuroevolution layer

Use a Gymnasium-compatible market environment and a pinned Stable-Baselines3 baseline (first candidate: PPO) after simulator semantics and reward tests pass. RL should learn a policy over an observation space containing only data available at decision time. Actions may include abstain/flat, position direction where supported, exposure choice, and policy parameters. The policy is not allowed to bypass the simulator or governance boundary.

The reward should be based on net portfolio growth after costs, with explicit penalties for drawdown, excessive turnover, and risk-envelope violations. Reward definitions and coefficients must be versioned and preregistered; candidates must not change the scoring definition used to evaluate themselves. Compare RL and evolutionary strategies on identical walk-forward splits and cost assumptions.

Evolution and RL are introduced in separate stages. Coupling both before each has a trustworthy baseline would make a failure almost impossible to diagnose.

### 6. Failure reflection as evidence, not storytelling

After each simulated trade or episode, record:

- The exact genome/model hash, decision-time observations, regime/context features, selected action, fills, costs, and outcome.
- A descriptive failure taxonomy with confidence and an explicit “unknown / insufficient evidence” option.
- Counterfactual diagnostics where valid: feature ablation, action alternatives, sensitivity to execution costs, and comparison with the candidate's own pre-registered exit policy.
- The resulting proposed mutation, crossover, reward/data change, or no-op, with the evidence that motivated it.

Do not claim a single “exact cause” from one losing trade. Attribution is a hypothesis and must be marked as such. Post-mortem outputs influence later training generations only; they cannot rewrite earlier predictions, labels, scoring rules, or the untouched holdout. Learning-ledger entries must be append-only and linked to run, data, genome, and code digests.

### 7. Temporal validation and overfitting defense

Use nested chronological evaluation:

1. Training window for candidate generation and RL fitting.
2. Inner walk-forward folds for search and model selection.
3. Separate validation periods for robustness and cost sensitivity.
4. A final holdout reserved before search and opened only once for a frozen candidate set.

Purging/embargo must account for overlapping labels, lookback windows, and trade holding periods. Compare with buy-and-hold and simple transparent baselines under the same cost assumptions. Report candidate count, discarded candidates, effective search budget, trial multiplicity, and the gap between in-sample and out-of-sample results. A successful CI run proves reproducibility and safety assertions, not economic merit.

### 8. Immutable safety and promotion boundary

A separate safety contract—not an optimizer-owned model/config—asserts that every run has:

- `Champion = null`
- `order_permission = false`
- `trading_authorization = NO-GO`
- `orders_or_execution_api_used = false`
- `real_order_attempts = 0`
- `final_holdout_used = false` during discovery

The final report must be generated from the exact commit and artifact digests. No trained weights are registered as a production champion. Any future promotion requires a separate owner decision after economic review; model output alone cannot trigger it.

## Phased roadmap

### Phase 0 — Contract and reproducibility (first)
- Freeze this design and explicit acceptance metrics before any strategy search.
- Add typed configuration and a run manifest (code SHA, data digest, config SHA, seeds, library versions, cost assumptions, split boundaries, genome count).
- Add contract tests that fail if any safety invariant changes.
- Add a branch-specific GitHub Actions workflow without modifying existing quality gates.

**Exit:** contract, provenance and safety tests pass; report generation works with zero candidate strategies.

### Phase 1 — Data and causal simulator
- Verify Stage A coverage; identify whether a provenance-pinned minute-bar artifact exists.
- Implement causal resampling, event ordering, fees/slippage and simulated-only fills.
- Add adversarial tests for future-data leakage, bar-boundary alignment, missing bars, impossible fills, and stop ordering.

**Exit:** same input + same seed produces the same ledger; future data mutations cannot change past decisions; simulator accounting tests pass. If minute data is unavailable, coverage claims below the source resolution remain blocked.

### Phase 2 — Genetic programming MVP
- Add DEAP as a pinned research-only dependency after license/security review.
- Implement versioned genome grammar, safe evaluator, complexity budget, checkpoints, and diverse population archive.
- Start with a small smoke budget, then a configurable multi-thousand-genome search with bounded GitHub Actions compute.
- Run transparent baselines and emit candidate-level results, discarded-candidate reasons, and evolution ledger.

**Exit:** deterministic search, no code execution from genomes, no data leakage, and reproducible artifacts across reruns.

### Phase 3 — Failure reflection
- Add append-only post-mortem and counterfactual diagnostics.
- Verify that post-mortem evidence changes only later generations and cannot rewrite earlier output.
- Test unknown attribution and low-confidence abstention.

**Exit:** every trade maps to a genome and evidence record; reflection never mutates a frozen evaluation or holdout.

### Phase 4 — RL environment and neuroevolution experiments
- Build a Gymnasium environment on the exact simulator.
- Establish a random/no-op environment contract and deterministic short training smoke test.
- Add a pinned CPU-first Stable-Baselines3 PPO experiment; later compare alternative algorithms only by a preregistered experiment matrix.
- Compare RL, evolution-only, and hybrid candidates on identical folds and costs.

**Exit:** environment compliance tests pass, the reward is fully logged, no action can call external execution, and comparisons are fair.

### Phase 5 — Robustness and economic evaluation
- Freeze candidates before validation and holdout evaluation.
- Run regime-specific, walk-forward, fee/slippage stress, turnover, complexity, bootstrap uncertainty, and multiple-testing-aware analyses.
- Report trade count, win rate, net expectancy, return, drawdown, risk-adjusted metrics, cost assumptions, folds, and stability; include all baselines.
- Evaluate the requested win-rate target only alongside expectancy and tail-risk constraints.

**Exit:** owner-defined, preregistered economic criteria pass across validation evidence. Otherwise report NO-GO, preserving every safety invariant.

## GitHub Actions and quality gates

All search, simulator tests, RL training and economic evaluation run in GitHub Actions. Existing Quality, PostgreSQL integration, runtime smoke, lint, strict typing, security, dependency, complexity, architecture and coverage gates remain unchanged. Research workflows use CPU-only dependencies initially, bounded runtimes, fixed seeds, no secrets, least-privilege permissions, and immutable artifacts.

Each run uploads a report, candidate/genome archive, trades, decisions, evolution history, learning ledger, evaluation splits, and machine-readable safety manifest. Artifacts are keyed to the exact commit and include input/config digests.

## Initial acceptance contract

A run is technically acceptable only when:
- The complete original CI suite passes without disabled, skipped, weakened, or relabeled gates.
- Every candidate is reproducible and every decision is point-in-time causal.
- All fees and costs are included; unknown costs are stated.
- Training, validation, and final holdout are disjoint and auditable.
- Candidate count refers to unique, evaluated genomes and is fully reported.
- `Champion = null`, `order_permission = false`, and `trading_authorization = NO-GO` remain true.

Economic acceptance is separate. Positive in-sample PnL or a 60% win rate is insufficient. The frozen candidate must demonstrate positive net expectancy out of sample, acceptable drawdown under pre-registered limits, adequate trade count, robustness across periods/regimes, and tolerable cost sensitivity. If any criterion fails or evidence is insufficient, the answer remains NO-GO.
