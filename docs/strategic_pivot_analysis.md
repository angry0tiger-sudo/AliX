# AliX Strategic Pivot Analysis — High-Frequency Research Only

**Date:** 2026-11-11  
**Status:** Analysis only — no implementation or experiment authorized by this document  
**Repository:** `angry0tiger-sudo/AliX`  
**Reviewed references:** AliX Master Reference Document v0.8 (2026-11-11); `AI6_RL_FINAL_VERDICT_2026-11-11.md`; `AI6-PPO-RISK-RESEARCH.md`; `external_data_pivot_aborted.md`; historical AI-2 DEAP run and eligibility-gate report.

---

## 1. Executive conclusion

The next research step must solve **economic expectancy and drawdown**, not merely increase the number of signals.

The final AI-6 result confirms that the current 1-hour Nobitex OHLCV teacher has no defensible edge under the 50 bps round-trip cost contract: baseline Net Profit Factor (PF) was 0.3743 in validation and 0.4370 in the untouched holdout, with drawdowns of 66.56% and 60.74%, respectively. PPO accepted no signal episodes and completed no trades. Its 0.00% drawdown is only the result of remaining entirely in cash, not evidence of successful risk management.

The baseline already produced approximately 500 completed trades per year when annualized over the evaluation windows. Therefore, the stated activity threshold of **more than 200 completed trades per year** is not a reason to accept weak signals or force excessive turnover. It is a minimum eligibility requirement. Every proposed path must meet it on both validation and final holdout while also demonstrating positive after-cost expectancy and controlled drawdown.

**Strategic recommendation:** prioritize a bounded **Option B — DEAP Genetic Programming re-evaluation** as the first candidate for owner approval. It directly searches for new entry and exit logic instead of trying to make the failed PPO risk filter accept a weak teacher. Keep Option A as a follow-on if multi-timeframe data coverage and signal diversity justify the additional model complexity. Do not subscribe to CoinAPI until a precise data-availability and incremental-value test is approved.

This is a recommendation for a future, owner-authorized research phase only. It does not reopen AI-6, authorize training, promote a model, or permit Paper/Live Trading.

## 2. Evidence review and binding constraints

### 2.1 Final AI-6 evidence

The canonical final verdict is the later closure report, not the interim metrics reproduced in portions of MRD v0.8.

| Segment | Model | Net PF | Win rate | Max drawdown | Statistical Z | Completed trades |
|---|---|---:|---:|---:|---:|---:|
| Validation | Baseline | 0.3743 | 32.48% | 66.56% | -4.8855 | 234 |
| Validation | PPO | N/A | N/A | 0.00%* | N/A | 0 |
| Untouched holdout | Baseline | 0.4370 | 34.22% | 60.74% | -4.0982 | 225 |
| Untouched holdout | PPO | N/A | N/A | 0.00%* | N/A | 0 |

\*PPO remained flat. PF, win rate and Z are undefined because there were no PPO trades.

Reproducibility anchors:

- Training commit: `7464e308d4e1efdb573a8225fb3a392ae7614199`
- Dataset: Nobitex BTC/USDT Spot, 1h, 31,801 pre-event candles
- Dataset SHA-256: `0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e`
- Cost: 25 bps maker fee per side, 0 bps slippage, 50 bps round trip
- [AI-6 final verdict](https://github.com/angry0tiger-sudo/AliX/blob/research/alternative-data-pivot/docs/phase-reports/AI6_RL_FINAL_VERDICT_2026-11-11.md)
- [AI-6 workflow #22](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569350)
- [Standard CI #2220](https://github.com/angry0tiger-sudo/AliX/actions/runs/43000569772)

The result is specific: the current 1h teacher's baseline signals were materially unprofitable after the stated costs, and the PPO filter abstained from every candidate. It does **not** mathematically prove that every possible OHLCV strategy or every shorter timeframe is unprofitable. It does make further tuning of the same closed AI-6 experiment unjustified.

### 2.2 Frequency was not the AI-6 bottleneck

The 234 validation trades and 225 holdout trades occurred in evaluation segments covering approximately 12.5% of the 31,801-bar hourly history each. On a rough annualized basis, that is around 500 baseline trades/year. Exact annualization must use the actual timestamps and eligible exposure window in any future report, but the conclusion is clear: the failed baseline was already above 200 trades/year.

Consequently:
- Signal count alone is not a success metric.
- The target is not to maximize trades without limit.
- A signal is counted only when it becomes a distinct, completed, flat-to-flat trade after duplicate and overlap resolution.
- A trade must not be counted multiple times merely because several timeframe models emitted the same opportunity.
- Higher activity that increases fees, slippage, whipsaw or drawdown is a regression, not an improvement.

### 2.3 AI-2 DEAP history: what it does and does not prove

MRD v0.8 records the previous AI-2 run as a technically successful 10-generation DEAP experiment with population 128 and 828 unique genomes. It used a public BTCUSDT 1-minute dataset containing 1,051,200 rows, not the immutable pre-event Nobitex dataset used by AI-6.

The selected candidate used a weekly cadence, made four training trades and one validation trade, and was marked economic NO-GO. Its training PF of 4.1477 is not persuasive evidence with four trades; validation PF was undefined because the single trade had no gross loss. That candidate is ineligible under the present timeframe directive and its metrics must not be combined with AI-6.

One correction to the historical premise is important: MRD v0.8 records the earlier DEAP simulator as charging 25 bps maker fee plus 20 bps slippage **per side**, or 90 bps round trip. The old run was therefore not cost-free. However, that historical run does not provide adequate evidence that the current, stricter causal and anti-leakage acceptance protocol was fully proven end-to-end on Nobitex data. Re-running DEAP would be a materially different experiment because the data venue, allowed timeframe universe, evaluation controls and frequency acceptance rules would change.

### 2.4 Non-negotiable timeframe and data rules

The only timeframe values eligible anywhere in the proposed entry-strategy search are:

```text
15m, 30m, 1h, 4h
```

Daily and Weekly intervals are prohibited. This current Owner Directive supersedes the older offline-diagnostic exception present in MRD v0.8. They must not appear in a genome, model registry, timeframe coverage quota, meta-model input used to select entry cadence, fallback path or candidate ranking.

Data implications:

- The immutable AI-6 artifact is hourly. It cannot be used to manufacture genuine 15m or 30m candles. Upsampling an hourly bar into smaller bars would invent intrabar information and is forbidden.
- Nobitex's public UDF history documentation lists native resolutions corresponding to 15m, 30m, 1h and 4h. It also documents a 500-bar request limit and minute-history availability beginning in Iranian year 1401. This makes native short-interval acquisition plausible, but it does not prove complete, gap-free coverage for the required pre-event window. A GitHub Actions coverage audit must prove the usable common window before any model work. [Official Nobitex market-history documentation](https://github.com/nobitex/docs-api/blob/master/source/includes/_market_data.md)
- Four-hour bars may be built from completed hourly bars with a deterministic, causal resampler, or fetched natively and reconciled against the hourly aggregation. No incomplete higher-timeframe candle may be exposed to an earlier decision.
- If native 15m/30m coverage is insufficient, the affected timeframe is data-ineligible. Do not silently substitute another venue, backfill gaps, interpolate, or extend the usable window with synthetic values.

The common evaluation window must be the same across competing models in a comparison. A longer 1h history must not be compared unfairly with a shorter 15m/30m history without an explicit paired-window analysis.

## 3. Option A — Multi-Timeframe Ensemble

### Concept

Train separate base models for 15m, 30m, 1h and 4h. Each base model produces a timestamped, calibrated signal or probability from information available at its own completed-bar boundary. A meta-learner then combines out-of-fold base predictions into one final decision stream.

The role of the timeframes should be explicit rather than four independent bots competing for the same capital. A reasonable research hypothesis is that 15m/30m identify candidate entries while 1h/4h contribute regime, trend and confidence context. The meta-learner must be trained on strictly out-of-fold predictions from the base learners.

### Conflicting-signal policy

A naive majority vote or average of raw model probabilities is not sufficient. The models may have different calibration, update cadence, sample sizes and error correlation.

A defensible design would require:

1. One master decision clock, most naturally each completed 15m close.
2. Point-in-time/as-of alignment: a 30m, 1h or 4h score is available only after that source candle has closed. No partial-bar peeking.
3. Per-timeframe calibration and out-of-fold predictions before meta-learning.
4. One portfolio-level position state. Identical or overlapping entries from multiple timeframes are merged into one trade, not counted as separate trades.
5. Explicit conflict handling: the meta-model may abstain; alternatively, a documented deterministic hierarchy can reject an entry when the higher-timeframe context is adverse. Missing or stale inputs fail closed.
6. A single exit and exposure owner. Multiple timeframe models must not independently issue incompatible exits or stack exposure.

This design can reduce dependence on one sampling interval, but it cannot guarantee that at least one timeframe will always have a valid signal. A no-trade decision remains valid when evidence is weak.

### Cost, feasibility and risks

- **Feasibility:** Medium-high, conditional on successful native 15m/30m Nobitex history validation. The existing ensemble architecture is reusable, but the current AI-6 artifact is not sufficient for all four models.
- **Computational cost:** Medium to high relative to one model. Feature generation and training are repeated across four datasets, followed by calibration, out-of-fold prediction generation and meta-model fitting. CPU-only training should be plausible for conventional tree models, but runtime and artifact volume will increase.
- **Main risk:** model disagreement and false diversification. Four models trained on the same underlying market are not four independent sources of information. Shorter bars also amplify microstructure noise, turnover and sensitivity to real slippage.
- **Time to first credible result:** approximately 2–4 weeks after data coverage is established.
- **Estimated probability of meeting both the activity and economic gates:** 20–30% (expert planning estimate, not a measured probability).

### Assessment

Option A is a good architecture for combining complementary time horizons, but it is not a solution to a frequency deficit: AI-6 already exceeded the proposed annual trade floor. Its value must be demonstrated as improved **after-cost quality, drawdown and stability**, not simply more model outputs.

## 4. Option B — DEAP Genetic Programming / AI-2 Revisited

### Concept

Use bounded Genetic Programming to discover entry and exit expressions on pre-event Nobitex Spot data. The genome may choose among the four allowed timeframes, but it must not generate unrestricted executable code. The search should evolve causal rule trees, parameters and exit logic rather than repeatedly optimizing one fixed list of named indicator recipes.

This option directly addresses the principal AI-6 failure: the primary signal generator was weak. PPO only filtered that signal and did not have authority to create a new entry edge.

### Candidate search space

A typed, bounded grammar can expose the following primitives:

- **Causal OHLCV terminals:** lagged returns, normalized candle body and range, upper/lower wick ratios, rolling volatility, range expansion, close location, volume surprise and rolling price/volume ranks.
- **Safe operators:** add/subtract/multiply, bounded division, absolute value, min/max, rolling mean/std/quantile/rank, lag, comparisons and boolean AND/OR/NOT.
- **Rule structure:** entry expression, exit expression, regime condition, maximum holding horizon, stop/trailing activation and exit policy.
- **Timeframe gene:** exactly one of 15m, 30m, 1h or 4h, subject to verified native data coverage.

Tree depth, node count, lookback bounds and operator types must be constrained. Candidate evaluation must use an allowlisted interpreter. Arbitrary Python execution, shell access, network calls, package installation and dynamic code loading are out of scope.

### Fitness and overfit controls

Fitness should reward net expectancy after costs, PF and return while imposing explicit penalties for drawdown, excess turnover, rule complexity and instability across regimes. The MRD's 15% maximum-drawdown limit remains a hard economic constraint. The search should use multi-objective or lexicographic selection so a high raw return cannot hide unacceptable risk or low trade support.

Required controls:

1. Chronological purged walk-forward folds with a horizon-aware embargo; no random shuffling.
2. Training-only evolutionary ranking. Validation is reserved for acceptance and diagnosis, not repeated tournament selection or hyperparameter tuning.
3. A sealed final holdout, evaluated only after the genome and all thresholds are frozen.
4. Multiple independent seeds and a pre-registered search budget, with every unique genome and failed candidate retained in an append-only research ledger.
5. Parsimony pressure and duplicate-genome elimination to limit bloated expressions and repeated tests of near-identical rules.
6. Robustness across market regimes and cost sensitivity at 50 bps and 90 bps round trip.
7. Statistical correction or resampling appropriate to a large strategy search (for example, block-bootstrap confidence intervals and a multiple-testing-aware performance assessment). A positive raw Z-score alone is not enough after a large search.
8. Full causal tests for feature availability, candle close time, next-open fills, stop/target collision policy and exit-state transitions.

The old DEAP run's one validation trade is a warning, not a precedent to repeat. The new search must reject a candidate that cannot demonstrate adequate out-of-sample activity and economics, regardless of its training fitness.

### Cost, feasibility and risks

- **Feasibility:** Medium-high. AliX already has DEAP research scaffolding, a bounded simulator direction and historical evidence artifacts. Adapting the pipeline to the pre-event Nobitex dataset and current timeframe contract remains necessary.
- **Computational cost:** Medium to high CPU cost because each genome requires repeated chronological simulation. It should not require a GPU. Runtime is governed mainly by population size, generations, feature computation and walk-forward fold count.
- **Main risk:** multiple-testing and strategy overfit. Genetic search can discover accidental historical patterns extremely efficiently; more generations can make this problem worse rather than better.
- **Time to first credible result:** approximately 1–3 weeks after the required Nobitex timeframe data audit; longer if native 15m/30m retrieval or simulator adaptation is blocked.
- **Estimated probability of meeting both the activity and economic gates:** 25–35% (expert planning estimate, not a measured probability).

### Assessment

Option B is the recommended first experiment, subject to owner approval. It searches for a new primary entry/exit policy, can enforce the four-timeframe allowlist directly in the genome, and reuses more existing research infrastructure than a new four-model stack. It is not low-risk: the search budget and holdout protocol must be frozen before execution.

## 5. Option C — Paid CoinAPI data plus frequent-timeframe OHLCV

### Concept

Add a narrowly selected derivative-market context feature set to Nobitex Spot OHLCV at 15m/30m/1h/4h. The first candidates worth investigating are funding rate and open interest. These features may identify crowded positioning, leverage build-up or liquidation-sensitive regimes that are not observable from spot OHLCV alone.

They are context features, not a replacement for a viable spot entry model. Funding rates and open interest are sourced from other derivatives venues; their basis, venue-specific coverage and timestamps must be considered before using them to make a Nobitex Spot decision.

### Incremental-value hypothesis

- Funding rate may help identify crowded long/short positioning and asymmetric squeeze risk.
- Open interest may distinguish a price move accompanied by new derivatives exposure from a move driven mainly by position closure.
- Neither feature guarantees direction, nor does it directly describe Nobitex Spot order-book depth, local spread or local trade flow.
- Slow-changing or periodic observations must be joined as-of their actual publication/availability time. Forward-filling a stale observation must have a documented age limit; future revisions or later-published values must never leak into earlier decisions.

CoinAPI's documentation lists historical funding-rate metrics for venues including Binance Futures and Bybit, with documented histories beginning in January 2021. Its documented open-interest venue coverage is narrower and should be checked for the exact required market and symbol before budgeting around it. [CoinAPI metrics FAQ](https://www.coinapi.io/market-data-api/faq) and [metrics API reference](https://docs.coinapi.io/market-data/metrics).

### Price and access reality

The "$25–$79/month" assumption needs correction. CoinAPI advertises $25 in starting/free credits, not a $25 recurring subscription. Its Startup plan is listed at $79/month (or $69/month equivalent when billed annually) and includes a stated daily REST-credit allowance. Actual total cost depends on endpoint usage, data volume and any metered overage. Confirm current terms before purchase. [CoinAPI pricing](https://www.coinapi.io/).

The previous direct Binance Futures and Bybit access probes from GitHub Actions failed with HTTP 451 and 403. CoinAPI is a different provider, so those failures do not prove CoinAPI will fail; however, GitHub Actions reachability, authentication, rate limits, exact symbol coverage, historical depth, retention and licence terms remain unverified. Paying for a plan does not establish any of those facts.

### Cost, feasibility and risks

- **Feasibility:** Medium-low until the exact instruments, history and GitHub Actions access path are verified.
- **Financial cost:** At least the current Startup subscription if a paid plan is required, plus possible usage charges and engineering/maintenance cost. The $25 entry credit is not a monthly plan.
- **Computational cost:** Low to medium. The difficult work is data ingestion, timestamp normalization, symbol mapping, provenance and point-in-time validation rather than model training.
- **Main risk:** the features may add little predictive information for Nobitex Spot, may be unavailable at the needed historical cadence, or may be dominated by cross-venue basis differences.
- **Time to first credible result:** approximately 2–5 weeks after a provider entitlement and GitHub Actions connectivity decision.
- **Estimated probability of meeting both the activity and economic gates:** 10–20% (expert planning estimate, not a measured probability).

### Assessment

Option C is not recommended as the first move. Before any subscription, the owner should approve a small provider-coverage validation scope. A paid feature should be retained only if a paired ablation shows a repeatable improvement over the same OHLCV baseline on the same dates, costs and validation protocol. It must not be used to disguise a weak primary signal or simply increase signal count.

## 6. Comparative decision matrix

The estimates below are planning judgments for a first credible offline research result. They are not guarantees or empirically measured probabilities.

| Dimension | Option A — MTF Ensemble | Option B — DEAP GP | Option C — CoinAPI |
|---|---|---|---|
| Feasibility | Medium-high, conditional on 15m/30m coverage | Medium-high; existing DEAP scaffolding, new Nobitex data contract required | Medium-low until vendor coverage and CI access pass |
| Compute / engineering cost | Medium-high; four model pipelines plus meta learner | Medium-high CPU; many repeated simulations, no GPU expected | Low-medium compute; medium-high integration and data governance |
| Direct recurring data cost | No new paid provider assumed | No new paid provider assumed | Startup listed at $79/month or $69/month billed annually; usage terms apply |
| Time to credible result | 2–4 weeks after data audit | 1–3 weeks after data audit | 2–5 weeks after provider approval and access audit |
| Estimated success probability | 20–30% | 25–35% | 10–20% |
| Main potential value | Better cross-horizon context and calibration | Discover genuinely new entry/exit rules | Add derivatives positioning context |
| Primary failure mode | Conflicting/correlated models; short-TF noise and turnover | Search overfit and multiple-testing false discoveries | No incremental predictive value or incomplete/unavailable external history |
| Recommendation | Second-stage option | **First option for owner consideration** | Defer pending evidence |

"Success probability" means the subjective chance of meeting **both** the required OOS activity threshold and the economic acceptance criteria below. It does not mean the chance of producing a technically successful CI run.

## 7. Shared acceptance contract for all three options

All options must be evaluated against the same non-negotiable criteria. Passing the activity gate alone is never sufficient.

### 7.1 Timeframe and trade-activity gate

- The only eligible timeframe set is `15m`, `30m`, `1h`, `4h`.
- Each option must produce **more than 200 completed, unique trades per year, annualized separately on validation and final holdout**.
- Count only realized, flat-to-flat trades after cost and overlap/deduplication rules. Do not count raw signals, partial fills as separate trades, simultaneous timeframe votes, reward bonuses or repeated entries within one position.
- Report actual trade count, exact eligible date span, annualized trade rate, median holding time and turnover.
- The 200/year threshold is a minimum activity eligibility gate, not an optimization target. More trades are not inherently better.

### 7.2 Economic and statistical gate

- Primary cost contract: 50 bps round trip, with maker fee and slippage assumptions separately parameterized and reported.
- Cost stress test: 90 bps round trip, preserving the historical 90 bps Opportunity Hunter sensitivity contract as a separate scenario.
- Net PF target: greater than 1.5, consistent with the MRD Opportunity Hunter objective; report undefined PF as `null`, never fabricate a value.
- Maximum drawdown: no more than 15% on validation and final holdout.
- Positive net expectancy and positive statistical evidence on untouched out-of-sample trades; use uncertainty estimates appropriate to serially dependent trading returns and account for multiple candidate testing.
- Compare with a simple, frozen baseline on the exact same dates, bars, fills and cost model. Report whether PF, return and drawdown improve, not merely whether they are positive.
- If the 50 bps case passes but the 90 bps stress case collapses, report the strategy as cost-fragile rather than robust.

### 7.3 Research governance

- Chronological, purged walk-forward evaluation; no random splits.
- Train-time selection must not consume validation profitability repeatedly. Final holdout remains sealed until the candidate, parameters and thresholds are frozen.
- Every input dataset must have immutable provenance, source revision, exact interval, timestamps, row count, gap report and SHA-256.
- Enforce point-in-time availability: a feature is usable only when it was available at or before the decision timestamp.
- All research, ingestion, training, tests and backtests run through GitHub/GitHub Actions. No local experiment is accepted as project evidence.
- A green CI or workflow is engineering evidence only. A candidate may only be submitted for owner review after the economic gates pass.

## 8. Recommended sequence

1. **Owner review:** consider Option B as the first bounded research authorization. No implementation is implied by this analysis.
2. **Data feasibility gate:** before model search, prove native Nobitex 15m and 30m historical coverage for the required pre-event common window. Reuse the immutable 1h artifact as a control and construct 4h only through a verified causal aggregation. Fail closed on gaps or provenance mismatch.
3. **Pre-register the experiment:** freeze the timeframe allowlist, search grammar and budget, chronological folds, fees, slippage scenarios, statistical methods and the >200 completed trades/year OOS activity floor.
4. **Run one bounded research phase:** do not simultaneously change the data source, costs, genome, feature set and evaluation protocol. Keep a same-window frozen baseline.
5. **Decision:** only a robust candidate meeting activity, PF, drawdown, expectancy and uncertainty gates in both validation and untouched holdout can be submitted for owner review. Otherwise record NO-GO and stop; do not repeatedly tune against holdout.
6. **Only after an explicit new directive:** consider Option A as a second-stage architecture or Option C as a separately budgeted data-ablation experiment.

## 9. Owner decisions and safety boundaries

This report requests a strategic choice only. It does not authorize implementation or execution.

- No code, configuration, workflow, data ingestion, training or backtest changes are made by this analysis.
- AI-6 remains CLOSED. No further PPO tuning or training is proposed.
- External Data Pivot remains ABORTED; Option C is a new, separately approved proposal, not a continuation of the aborted run.
- Nobitex Spot remains the intended execution venue, but no execution integration or order flow is authorized.
- `main` must remain untouched at `44bcaeae7eefd358cf063394b0e7f4f0b83af8eb`.
- Safety invariants remain:

```text
Champion = null
order_permission = false
trading_authorization = NO-GO
ALIX_DRY_RUN = true
ALIX_ALLOW_LIVE_TRADING = false
private_api_calls = 0
real_order_attempts = 0
nobitex_private_api_calls = 0
Paper Trading = NOT AUTHORIZED
Real Trading = NOT AUTHORIZED
```

**Final recommendation:** Option B is the best first candidate to submit for owner approval because it rethinks the primary entry/exit logic, uses the existing bounded DEAP research direction and avoids paying for unproven external data. Its approval must remain conditional on proving the required Nobitex 15m/30m coverage and on a pre-registered anti-overfitting protocol. No option is presumed profitable, and no trading authorization follows from this report.
