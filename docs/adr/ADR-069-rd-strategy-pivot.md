# ADR-069 — Strategic Pivot from OHLCV Direction Forecasting to Evidence-Led R&D

- **Status:** R&D review complete; strategy-family selection **PENDING OWNER DECISION**
- **Date:** 2026-11-27
- **Repository:** angry0tiger-sudo/AliX
- **Branch:** research/nobitex-hf-strategy
- **Reviewed base HEAD:** fde2116ab658f9478e4d9df709bcd5aa54fd467b
- **Related decisions:** ADR-056–ADR-068; MRD v1.2; M1.6 / M3 / M5.2 / M6 records
- **Decision authority:** Owner directive of 2026-11-27
- **Scope:** Documentation-only R&D assessment. No strategy, model, configuration, workflow, dataset or trial is authorized by this ADR.

## 1. Owner decision and immediate scope

The Owner has directed AliX to **stop the current single-asset OHLCV directional-prediction path, but not to close the AliX project**. The project enters an R&D selection phase intended to identify a different source of economic edge.

This ADR supersedes any interpretation of ADR-068 as a proposal to archive the entire project. ADR-068 remains a valid record of the M6 outcome and its then-pending endgame alternatives; the current Owner directive establishes that the project continues under a new research mandate.

The following work is explicitly **not authorized** by this ADR:

- No code, strategy, configuration, workflow implementation or model implementation.
- No model creation, fitting, inference, training, optimization, backtest or simulation.
- No new market-data acquisition, capture, provider trial, subscription, API credential or purchase.
- No Holdout or Post-Hack data access; no use of sealed final evaluation data.
- No private exchange API calls, account access, paper orders or real orders.
- No change to main, its ref, its tree or branch protection.
- No transition to Champion, Paper, Shadow execution with order capability, or Live.

This ADR records three candidate research families and a fourth previously discussed adjacent option. They are **draft proposals only**. There are no locked preregistration hashes, no newly registered trial identities and no approved execution steps. The Owner must select a path before a path-specific preregistration is frozen and any implementation is considered.

## 2. Audit boundary: what “12 experiments” means

The focused strategy inventory in ADR-068 reconciles **12 distinct logical candidate identities** across five workstreams:

| Workstream | Logical identities | Evidence and final status |
|---|---:|---|
| HGBQ-CQR / Regime P | 3 | 1h, 4h and 24h candidates. No calibrated lower bound passed either registered Cost-50 or Cost-90 gate; no hypothetical trade was admitted. Closed NO-GO. The 37 registry slots included 27 duplicate slots and must not be counted as 37 independent candidates. |
| M3 OFI / Regime M | 2 | BTCIRT only. 1h: 665 OOS cohorts, NetPF 0.07347, net PnL -7.9225%, stable folds 0/4. 4h: 82 cohorts, NetPF 0.23269, net PnL -0.1214%, stable folds 0/4 and only one fitted fold. Both CLOSED_NO_GO. |
| Regime T binary Triple-Barrier + Logistic Regression | 1 | 1,652 pooled OOF predictions; accuracy 51.09%, balanced accuracy 51.57%; below the strict >55% screening requirement. It was a classification screen, not a costed trading backtest. CLOSED NO-GO. |
| M5.2 Fear & Greed | 4 | Control/Sentiment × 1h/4h. All eight pooled NetPF results were below 1; sentiment was strictly worse than its paired OHLCV control in all four comparisons; stable folds 0/4. CLOSED NO-GO. |
| M6 three-class Triple-Barrier + Random Forest | 2 | 1h and 4h. Both candidates had NetPF below 1.10, MaxDD above 15%, 2/4 valid folds and 0/4 stable folds at both 50 and 90 bps. CLOSED — INSUFFICIENT_SAMPLE / NO-GO. |

**12 is not the global AliX historical trial count.** The inventory does not include all earlier AI-6/PPO, DEAP, Chronos, V4-A* and other research branches. GLOBAL_N=UNKNOWN and GLOBAL_DSR=NOT_COMPUTABLE remain in force. Candidate identities, registry slots, model fits, fold fits, workflow attempts and diagnostic scenarios are different counting units.

### 2.1 Additional historical branches found in the GitHub audit

The following historical results were independently found in old research branches and are **additional context, not additions to the 12-identity focused inventory above**:

| Historical path | Exact GitHub evidence | Interpretation |
|---|---|---|
| V4-AO 30m HMM Donchian breakout | Workflow Run 42445443790, branch commit 9a17aa082e22742e29c2866e6afc834103bc622f. Primary candidate: 94 trades, 20.21% win rate, NetPF 0.5988, MaxDD 22.68%, net PnL -1,988.88 (USD-denominated report units). Cost was a 30 bps round-trip proxy. | A Donchian/HMM strategy was already evaluated and failed its then-current gates (NetPF >1.5, MaxDD <15%, win rate >45%, at least 30 trades). It was on Binance ETHUSDT, not Nobitex BTCUSDT; it is relevant historical evidence but not a same-market replication. |
| V4-AM CUSUM/HMM regime-reactive strategy | Workflow Run 42364732134, branch commit 1b043fc791b47759bddc2b592d9bb639330fa956. Primary: 32 trades, 25.00% win rate, NetPF 0.4778, MaxDD 17.91%, net PnL -1,280.52. Meta candidate had 22 trades. | This was not actual passive market making. The “Maker” name in the historical workflow was misleading for that specific run: it used a maker-fee proxy, while passive-limit-fill simulation was false. It was a CUSUM/HMM regime-reactive signal strategy on Binance ETHUSDT, not an L2 quote-placement/queue-position strategy. |
| Stage-B “Triple-Barrier and Relative Order-Book” | Workflow Run 40109476273, branch commit 2215c00b335cf909c0694c016e55b57fde9066fb. Report: 8 model candidates executed, 8 blocked for missing book data, 7,512 OOS predictions, orderbook status NOT_PROVIDED, liquidity status BLOCKED_ZERO_SOURCE_VOLUME, economic status NOT_EVALUATED_BLOCKED_INPUT_DATA. | This was not an economic test of a real order-book strategy. The input did not contain the required order book, and the run correctly blocked economic evaluation. |

This audit materially narrows what may be called “new”:

- **Donchian itself is not new** to AliX. V4-AO failed, and Donchian also appeared as one component in V4-AM. A Donchian + TTM-R2 combination has not been established as an economic success, but it is an adjacent, previously proposed family rather than an untouched research direction.
- **Volatility targeting itself is not new**: prior V4-AM/V4-AO workflows contained capped inverse-volatility sizing. TTM-R2 provides a different forecast source, not proof that an entry edge exists.
- The old V4-AM “Maker” workflow is not evidence that passive L2 market making has been economically tested.
- The Stage-B “Orderbook” run is not evidence that real order-book data was available or evaluated.

## 3. Root-cause analysis of the failed directional path

### 3.1 Classification accuracy is not tradable expectancy

A classification score answers whether a label was predicted correctly under a specific label definition. It does not answer whether an order could have been filled at the assumed price, whether the trade had enough payoff to cover two-sided fees and slippage, how much capital was tied up, or whether losses clustered into unacceptable drawdown.

The clearest example is Regime T: pooled accuracy was 51.09% and balanced accuracy 51.57%, only a 1.09 percentage-point lift over a 50% reference. Fold accuracy ranged from 43.67% to 60.57%, so the result was temporally unstable. Its Triple-Barrier labels excluded timeout, ambiguous and unavailable observations and did not include transaction costs. It was explicitly a classification screen, not a portfolio PnL test.

The Random Forest result in M6 must not be described as “above 50% accuracy”: pooled three-class accuracy was 35.11% at 1h and 32.46% at 4h; balanced accuracy was 34.53% and 36.80%, respectively. Both horizons failed economic acceptance at both registered costs.

A strategy needs positive **net** expectancy after a realistic fill and cost model. A small accuracy edge can still lose money when false positives are frequent, the payoff ratio is unfavourable, timeouts are censored, or turnover is high.

### 3.2 Triple-Barrier labels did not create an economic edge

Triple-Barrier defines a path-dependent target (upper barrier first, lower barrier first, or vertical timeout). It can make labels more relevant than a fixed future-return sign, but it cannot create predictive information that is absent in the inputs. It also does not automatically model queue priority, partial fills, bid/ask crossing, impact, capital constraints or a self-financing position lifecycle.

The binary Regime T test excluded 7,061 censored/ambiguous/unavailable rows from 24,740 binary events. The M6 three-class target retained timeout as an outcome, but its Random Forest classifier still generated poor classification quality and failed all key economic/stability gates. These were materially different protocols and are not a clean model-complexity experiment.

### 3.3 Why Random Forest complexity did not help

The project does not have a controlled Logistic Regression-versus-Random-Forest ablation with the same labels, features, splits, cost model and decision policy. Regime T used a binary Logistic Regression screen; M6 used a three-class Random Forest protocol. Therefore, it would be incorrect to attribute the difference in results solely to estimator complexity.

The plausible failure mechanisms are:

1. The input signal-to-noise ratio was weak at the chosen 1h/4h horizons.
2. A higher-capacity tree ensemble can fit incidental regime-specific patterns when independent event samples are limited.
3. Market non-stationarity and changes in volatility/liquidity reduce the persistence of learned thresholds.
4. Class imbalance, censoring and first-touch labels can make the training objective misaligned with net economic payoff.
5. More parameters do not compensate for missing information about who is trading, at what price, and with what urgency.

The general overfitting literature warns that searching more model variants can inflate in-sample performance and selection bias. It does not prove that Random Forest is inherently inferior; it supports fixing one low-capacity estimator and one candidate definition before any future evaluation.

### 3.4 Trading costs and turnover

Trading friction can erase an apparent gross edge when expected move per trade is small. Relevant components include maker/taker fees on both sides, quoted spread, adverse selection after a fill, slippage, market impact, partial fills, cancellation/replacement costs and latency. For a passive strategy, queue position and the probability of being filled only when the market is moving against the quote are especially important.

AliX's 50 bps and 90 bps round-trip assumptions are **registered research scenarios**, not verified realized Nobitex fill costs. No complete empirical maker/taker fee schedule plus account-level fill/slippage record has been reconciled in this R&D scope.

The registered M6 results show how fragile the prior strategy was to the cost assumptions: 1h NetPF declined from 0.160856 at 50 bps to 0.013034 at 90 bps; 4h declined from 0.534557 to 0.278479. None was remotely close to 1.10. M5.2 also produced more trades after adding sentiment, but those additional trades reduced rather than improved paired net results.

The Owner-provided claim that costs consume “55%–91% of gross alpha” could not be independently verified from the supplied citation marker because no identifiable paper, author or bibliographic record was provided for that exact range. It must not be repeated as a universal empirical law. Published crypto technical-rule evidence finds that transaction-cost effects vary across cryptocurrencies and strategies; costs should be measured for the actual venue and order type rather than assumed to be a constant fraction of alpha.

### 3.5 Microstructure evidence was not adequate for a BTCUSDT strategy

M1.6 collected five scheduled BTCIRT segments totalling 105,018.751 seconds (29h 10m 18.751s). Although the elapsed capture time exceeded the 24-hour minimum, it did not achieve M1 acceptance: 28,978 non-monotonic publisher-offset events occurred in one segment after reconnect; all segments together recorded 28,997 gap rows, including disconnects and offset gaps; total uncovered inter-segment time was 4,243.034 seconds. The five segments were not continuous. M1.6 is BTCIRT, not BTCUSDT.

M3 then used the five M1.6 artifacts to evaluate best-level OFI, MLOFI levels 2–5, microprice deviation, signed API-type trade volume and top-five book imbalance. It failed both horizons and was closed. Therefore, a fresh plan cannot reuse M1.6 as though it were clean BTCUSDT data, nor may it rename the old OFI predictor and claim a new candidate.

### 3.6 What the evidence does and does not establish

- It establishes that the **registered candidate families** in the audited periods failed their economic gates.
- It does not establish that all systematic trading, all machine learning, all crypto order flow, all sentiment or all market-making strategies are impossible.
- It does establish that repeating the same single-asset OHLCV direction-label/model escalation without a different information source or economic mechanism is not justified.
- It does not establish the actual Nobitex execution-cost distribution; that remains a material unknown.

## 4. R&D literature and open-source engineering review

### 4.1 Microstructure and order flow

Cont, Kukanov and Stoikov (2014) find a short-horizon linear relation between best-quote Order Flow Imbalance (OFI) and price changes in their NYSE sample, with price impact inversely related to market depth. This is evidence that order-book events contain information about very short-term price formation; it is **not** proof that OFI is necessary for every profitable strategy or that OFI by itself survives execution costs.

Anastasopoulos et al. (2026) report predictive information in cross-border order flows across 11 major currencies for the cross-section of cryptocurrency returns. Their “world order flow” dataset is materially different from Nobitex's local BTCIRT/BTCUSDT Level-2 feed. Transferability must be tested, not assumed.

Level-2 market-by-price data aggregates displayed quantity at each price level. It does not expose each order ID or our exact queue position. NautilusTrader's backtesting documentation explicitly distinguishes L3 market-by-order data from L2 market-by-price and notes that recorded book data cannot reveal how a simulated order itself would have changed the market. A conservative queue model is therefore mandatory if a future L2-only market-making study is selected.

### 4.2 Cross-cryptocurrency return spillovers

Guo et al. (2024) report cross-cryptocurrency return predictability using lagged returns from other cryptocurrencies in Binance data; the paper reports out-of-sample long-short performance after transaction costs. AliX is Spot/long-only by current design, so a proposed Nobitex adaptation must be a long-only rotation against USDT or cash, not a direct replication of the paper's long-short portfolio.

### 4.3 On-chain flows and whale activity

Chi, Chu and Hao (2024, arXiv:2411.06327) report that USDT net inflows to cryptocurrency exchanges positively forecast BTC and ETH returns, with the strongest effect at one hour. Their abstract says BTC exchange net inflow alone did not significantly forecast BTC returns. This argues for distinguishing stablecoin flows from BTC transfers rather than treating all “exchange inflow” metrics as interchangeable.

Other research on exchange transfers and whale movements is suggestive but asset-, chain-, venue- and horizon-dependent. An on-chain transfer into an entity-labelled exchange wallet does not prove intent to sell, does not identify a Nobitex BTCUSDT order, and may be affected by revised wallet labels. Point-in-time data snapshots and license terms are essential.

ADR-061 already reviewed Alternative.me, CryptoQuant, Glassnode, Coin Metrics, X and Reddit. That was a feasibility review, not an economic evaluation. Fear & Greed was tested and failed in M5.2; exchange-flow and whale features have not been economically evaluated in the reviewed focused trial inventory. Any future paid data source must pass a written coverage, price, point-in-time, redistribution and commercial-use review before purchase.

### 4.4 Open-source systems: engineering references, not profit claims

- [Hummingbot](https://github.com/hummingbot/hummingbot) provides a modular crypto trading-bot framework. Its [Avellaneda Market Making documentation](https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/) implements inventory-aware reservation-price and spread logic. This is a useful design reference, not evidence that the strategy will be profitable on Nobitex.
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) is an event-driven trading/backtesting engine. Its [market-data documentation](https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/) is useful for understanding L1/L2/L3 fidelity and fill-simulation assumptions. Its LGPL-3.0 license requires review before integration.
- [Hummingbot Quants Lab](https://github.com/hummingbot/quants-lab) is a research/backtesting companion project; it is an engineering reference, not an audited performance record.

No open-source repository's star count, active commits or successful CI is evidence of a profitable strategy.

## 5. Three R&D candidate families

All three pathways below are **materially different from the registered single-instrument OHLCV direction models**. “Not tested” is scoped to the focused AliX evidence and selected historical branches reviewed for this ADR; because GLOBAL_N=UNKNOWN, no absolute claim is made that no related concept ever appeared anywhere in all AliX history.

### R&D-A — True L2 passive market making with inventory and adverse-selection control

**Novelty boundary:** This is not another OFI directional classifier. The hypothesis is that a passive quote-and-inventory policy might earn spread capture after adverse selection and inventory risk, using L2 events to adjust quote placement. Previous V4-AM was not passive quoting, and Stage-B did not receive an order book.

**Hypothesis.** On Nobitex Spot BTCUSDT, a fixed, inventory-aware passive quoting policy can achieve positive after-cost round-trip inventory-cycle expectancy while keeping drawdown within the project's limits, when conservative queue and adverse-selection assumptions are applied.

**Data proposal.**
- New, prospective Nobitex BTCUSDT L2 snapshots and WebSocket order-book updates, plus public trade events, with source and exchange timestamps, publisher offsets, capture timestamps and sequence/gap manifests.
- M1.6 BTCIRT artifacts are not eligible input for this test.
- Proposed qualification floor: at least seven calendar days of prospective capture and at least 72 hours of integrity-accepted active observations spanning day/night conditions. This is a data-quality floor, not proof of statistical sufficiency. Reconnects and offset discontinuities must remain visible; no forward-fill, interpolation or fabricated queue events.
- Exact Nobitex maker/taker fee schedule and the assumed execution costs must be verified before any economic evaluation. No data is collected under this ADR.

**Fixed features.**
1. Top-of-book quoted spread in basis points.
2. Normalized bid/ask depth imbalance over the first five price levels.
3. Microprice deviation from mid-price, normalized by spread.
4. L1 OFI and fixed top-five MLOFI.
5. Depth slope/concentration across the first five levels.
6. Signed aggressive trade-volume imbalance, only if the exchange trade-side field is verified.
7. Short-horizon realized volatility, quote age and current inventory deviation.

**Interpretable model / policy.** A single fixed Avellaneda–Stoikov-style reservation-price and spread policy with a predeclared inventory target and hard inventory cap. OFI/microprice may adjust quote skew or trigger a conservative adverse-selection pause; they may not become a post-hoc high-capacity directional model. All parameters and calibration procedure must be frozen before the evaluation period.

**Execution simulation.** Nobitex L2 has no order IDs or queue position. A simulated passive order must enter at the back of the visible queue; fills require observed aggressive volume to consume the conservatively estimated quantity ahead. Cancellations must not be assumed to advance our simulated queue unless supported by a conservative rule. Partial fills, stale quotes, cancel latency, market impact and adverse markouts must be represented. If this cannot be modelled credibly with the available data, the test is blocked rather than assigned optimistic fills.

**Draft evaluation.** Four chronological folds; report round-trip inventory cycles (not raw quote updates), fill rate, maker/taker mix, post-fill 1s/5s/30s markout, inventory exposure, turnover, net expectancy, NetPF and MaxDD. Apply the shared economic gates in Section 6.

**Main advantages:** a genuinely different economic mechanism; directly addresses spread, liquidity and inventory rather than trying to classify hourly direction.

**Main disadvantages:** highest data-quality and simulation burden; current M1.6 corpus is unusable for this purpose; L2 cannot reconstruct exact queue priority; actual execution costs are unverified.

### R&D-B — Cross-asset lead–lag and long-only relative-value rotation

**Novelty boundary:** This is not another BTC-only directional classifier. It asks whether information in one asset or the broader crypto cross-section predicts *relative* returns of other Nobitex-listed assets. No corresponding multi-asset Nobitex economic trial was identified in the reviewed focused registry/branches.

**Hypothesis.** Lagged BTC and leave-one-out cross-sectional market returns contain incremental information about the next four-hour relative return of other liquid Nobitex USDT spot pairs, sufficient to support a long-only rotation portfolio after costs.

**Data proposal.**
- Native Nobitex 1h OHLCV for a fixed universe of 5–10 eligible USDT spot pairs, including all available delisted/temporarily inactive symbols in the historical universe to reduce survivorship bias.
- Universe membership and liquidity ranking must be determined only from a predeclared formation window, not from future test-period turnover.
- Use only Nobitex data for the target market. The currently committed BTCUSDT series is a control, not a substitute for missing altcoin histories.
- No data is acquired under this ADR. Exact pair coverage and common historical window must be audited after Owner approval.

**Fixed features.**
1. BTC lagged 1h and 4h returns.
2. Leave-one-out cross-sectional median return of eligible pairs.
3. Target pair's lagged 1h and 4h returns.
4. Target pair's rolling beta to BTC over a fixed 24h window.
5. Target pair's relative-volume z-score over a fixed 24h window.
6. Target pair's 24h realized volatility.

**Interpretable model / policy.** One fixed StandardScaler + Ridge(alpha=1.0) regression to estimate the next four-hour return. Rebalance every four hours; hold at most the top two assets with strictly positive scores, otherwise remain in USDT. Long-only Spot, no leverage and no short positions. The control is a frozen equal-weight or BTC-only policy over exactly the same available dates and cost assumptions. The universe, rebalance, holding period and exposure caps are frozen before fitting.

**Draft evaluation.** Four expanding chronological walk-forward folds. Purge any training target whose four-hour label window crosses the test boundary and apply a four-hour embargo. No overlap or duplicate asset/time trades; report portfolio-level completed trades and asset-level attribution separately. Apply the shared economic gates in Section 6.

**Main advantages:** published cross-crypto spillover evidence; does not depend on BTC's own next-bar direction being predictable; no order-book queue simulator required.

**Main disadvantages:** needs multi-pair Nobitex historical data; the published paper uses Binance and a long-short portfolio; a long-only adaptation may be materially weaker; fees, spread and liquidity vary sharply by pair.

### R&D-C — Point-in-time on-chain stablecoin flow and whale-to-exchange activity

**Novelty boundary:** Fear & Greed has already been tested and failed. This proposal uses blockchain-flow measures, not another sentiment index. ADR-061 was a provider-feasibility study; no on-chain feature family was economically evaluated in the reviewed focused trial registry.

**Hypothesis.** Point-in-time USDT exchange net inflows and large-transfer-to-exchange activity provide incremental information about subsequent BTCUSDT returns or volatility beyond the frozen OHLCV control, after costs.

**Data proposal.**
- One licensed provider, selected only after exact metric coverage, historical resolution, point-in-time revision behavior, entity-label methodology, commercial-use rights and full price are documented.
- Candidate source fields: USDT netflow into exchange entities, BTC netflow into exchange entities, large-transfer-to-exchange share/volume, and exchange-reserve change.
- Every record must preserve observed_at, published_at, available_at, retrieved_at, provider revision and entity-label version. Only values available at or before the decision time may be used.
- Daily-only metrics must be consumed with a conservative publication delay; repeating a daily value across 24 hourly rows does not create 24 independent observations.
- Exchange wallet labels may be revised. If the provider cannot provide a point-in-time snapshot or a verifiable revision history, the test is blocked.
- Global exchange flows must not be mislabelled as Nobitex-specific flows. A BTC/USDT netflow is not proof that the flow will be traded on Nobitex.

**Fixed features.**
1. USDT exchange netflow, normalized by the relevant exchange balance/circulating-supply denominator.
2. BTC exchange netflow, normalized by exchange reserves.
3. Large-transfer-to-exchange volume share, using a threshold fixed from the training period only.

**Interpretable model / policy.** One paired StandardScaler + Ridge(alpha=1.0) model for a fixed four-hour forward return target: baseline OHLCV control versus identical OHLCV plus the three locked on-chain features. No provider metric substitution, feature expansion or retraining threshold sweep after results are observed.

**Draft evaluation.** Four chronological purged folds with a four-hour label purge/embargo, strict as-of joins and paired control/treatment comparison on identical dates and cost assumptions. Apply the shared economic gates in Section 6; incremental value must improve net performance without violating drawdown/stability gates.

**Main advantages:** different information source from exchange candle aggregates; published research specifically points to USDT exchange inflow as potentially more informative than BTC exchange netflow alone.

**Main disadvantages:** likely provider/license cost; historical data may be revised; wallet-label uncertainty; blockchain exchange inflow is not necessarily selling pressure or a Nobitex trade; point-in-time archive may be unavailable.

## 6. Shared economic and anti-overfitting gates — draft for the selected path

The Owner requested retention of the hard economic gates:

- NetPF >= 1.10 under each registered cost scenario.
- At least 100 completed out-of-sample trades. For R&D-A, a completed trade means a flat-to-flat round-trip inventory cycle, not an individual quote update or partial fill.
- MaxDD <= 15%.
- At least 3 of 4 chronological folds must be stable under a predeclared fold definition. A stable fold must have at least 20 completed trades, positive net PnL and fold NetPF > 1.0.
- Cost-50 and Cost-90 are stress scenarios unless and until actual Nobitex fees, spreads, fills and slippage have been empirically verified. The exact same economic calculation must be used for control and treatment.

Additional controls required before any path-specific preregistration is locked:

1. One primary hypothesis, one fixed feature list, one model/policy and one fixed decision rule per selected path. Any material change creates a new registered candidate.
2. Four chronological, purged walk-forward folds; no random shuffle. Purge and embargo must match the maximum outcome/holding horizon.
3. Strict point-in-time feature joins. Data availability, not just event timestamp, determines usability.
4. Immutable data manifest: source, market, symbol universe, date interval, row/record counts, gap/offset diagnostics, revisions, schema, artifact SHA-256 and workflow commit.
5. Compare against a frozen same-window baseline; report gross and net returns, fees, spread, slippage, turnover, trade counts, holding time, invalid/censored events and drawdown.
6. Freeze all thresholds and costs before the first outcome is read. No threshold change, feature sweep, model escalation or selective fold deletion.
7. Record every candidate, including failed and infrastructure-blocked candidates. Do not rewrite old registry rows.
8. Account for serial dependence and multiple testing. Report uncertainty intervals and, where the complete trial inventory becomes reconstructable, a multiple-testing-aware statistic such as Deflated Sharpe Ratio / Probability of Backtest Overfitting. Until then, GLOBAL_DSR=NOT_COMPUTABLE remains.
9. PRE-HACK data may support exploratory development only. It cannot substitute for a fresh confirmatory holdout; Holdout and Post-Hack remain sealed.
10. A successful CI run confirms engineering reproducibility only. It is not an economic pass or trading release.

## 7. Volatility / TTM-R2 + Donchian: related option, not one of the three new families

The exact TTM-R2 + Donchian pairing has not been established as an economic success. However, it is not a wholly new AliX direction: Donchian was already evaluated in V4-AO and V4-AM, volatility targeting was already used in those strategy families, and TTM-R2 sizing was replayed on 480 existing teacher trades.

TTM-R2's research-only variance forecast improved QLIKE by 29.38% versus the registered persistence baseline (198 non-overlapping forecasts). But in the 480-trade position-sizing replay, fixed sizing produced NetPF 0.3731 and MaxDD 21.63%, while TTM sizing produced NetPF 0.3748 and MaxDD 18.26%. The drawdown fell by 3.37 percentage points, but NetPF remained materially below 1 and MaxDD remained above the 15% objective. TTM improves a risk forecast; it does not create an entry edge.

If the Owner prefers to evaluate this adjacent option rather than one of the three new families, its draft contract should be:

- One fixed 1h Donchian(20) breakout entry and fixed ATR(14) exit geometry, no HMM, no model-based entry, no parameter sweep.
- A paired fixed-size baseline and TTM-only risk-size treatment on identical entries, exits and costs.
- TTM forecast must be strictly available as-of the entry; multiplier 0 < m <= 1.0; it may only reduce size and may never increase exposure, create direction or reverse an entry. Invalid/missing forecasts fail closed.
- No reuse of the existing hf_donchian.py helper unchanged: it only allows 15m/30m and uses a 0.5–1.5 multiplier, conflicting with the risk-only maximum of 1.0.
- Apply the shared economic gates and report incremental improvement over fixed sizing, not merely lower drawdown.

This route is retained as an **adjacent, previously discussed option**, not counted as one of the three new R&D families.

## 8. Comparative decision matrix

Success likelihood is an uncalibrated engineering prior, not a statistical probability. It is shown as a qualitative range because AliX has no validated base rate for these new families.

| Path | Potential advantage | Main disadvantage / blocker | Relative cost | Evidence-based success outlook |
|---|---|---|---|---|
| R&D-B — Cross-asset lead–lag rotation | Peer-reviewed cross-crypto evidence; uses spillovers rather than single-pair direction | Needs clean multi-pair Nobitex histories; Binance long-short evidence may not transfer to long-only Nobitex Spot | Medium | **Low–Medium**; best initial feasibility/evidence balance if historical pair coverage exists |
| R&D-C — On-chain USDT flows / whales | Different data domain; one study finds USDT exchange inflow more informative than BTC netflow | Paid/licensed provider, mutable entity labels, PIT and target-venue mapping risks | Medium–High | **Low–Medium**; worth pursuing only after PIT and licensing are proven |
| R&D-A — L2 passive market making | New economic mechanism: spread capture with inventory and adverse-selection control | M1.6 data failed integrity and is BTCIRT; L2 lacks queue IDs; a conservative fill simulator is difficult | High | **Low** until a new integrity-qualified BTCUSDT L2 corpus and queue model exist |
| Adjacent — Donchian + TTM-R2 | TTM forecast passed a variance benchmark and slightly reduced prior replay drawdown | Donchian V4-AO failed; prior TTM sizing remained unprofitable; exact combo still unproven | Low–Medium | **Low**; not a brand-new research family |

No path is selected by this ADR. The table supports Owner choice; it is not a recommendation to run all paths in parallel.

## 9. Proposed next governance sequence

1. Owner selects **one** R&D family (or explicitly selects the adjacent Donchian + TTM option).
2. After selection, the selected family receives a separate full preregistration, including fixed hypothesis, data contract, feature dictionary, model/policy, exact folds, all thresholds, cost model, sample-adequacy rules, statistical method and immutable artifact requirements.
3. The selected preregistration is reviewed for feasibility and locked with SHA-256. Only then may a separate Owner directive authorize a narrowly scoped data-collection or implementation step.
4. Data-quality qualification precedes model work. If required source data, time coverage, point-in-time availability or provenance fails, no fitting/backtest starts.
5. Economic evaluation occurs only through GitHub Actions on the exact approved commit and registered dataset. Any failure is recorded and closed without post-outcome retuning.
6. No selected path automatically inherits paper/live permission.

## 10. Safety invariants and branch protection

The following invariants remain unchanged:

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- paper_orders = 0
- Holdout = SEALED
- holdout_bytes_read = false
- posthack_data_read = false
- local_execution = false

The research branch is the only branch changed for this documentation update. main remains frozen at its verified base HEAD/tree; this ADR does not authorize a merge, release or default-branch change.

## 11. References

### AliX evidence
- [ADR-056 — Regime T Triple-Barrier / Logistic Regression](ADR-056-regime-t-triple-barrier.md)
- [ADR-057 — Public Microstructure Feasibility](ADR-057-regime-m-market-microstructure-feasibility.md)
- [ADR-058 — M1.6 Historical Sufficiency and Live Capture](ADR-058-regime-m-historical-sufficiency-and-live-capture.md)
- [ADR-059 — M3 OFI Preregistration](ADR-059-regime-m3-ofi-preregistration.md)
- [ADR-060 — M3 OFI Evaluation](ADR-060-regime-m3-ofi-baseline-evaluation.md)
- [ADR-061 — BTCUSDT On-Chain / Sentiment Feasibility](ADR-061-btcusdt-onchain-sentiment-data-feasibility.md)
- [ADR-064 — M5.2 Sentiment NO-GO](ADR-064-m5-2-btcusdt-sentiment-no-go.md)
- [ADR-067 — M6 Evaluation Results](ADR-067-m6-triple-barrier-3class-evaluation-results.md)
- [ADR-068 — M6 Closure and Endgame Options](ADR-068-final-endgame-decision-after-m6.md)
- V4-AO economic workflow: https://github.com/angry0tiger-sudo/AliX/actions/runs/42445443790
- V4-AM economic workflow: https://github.com/angry0tiger-sudo/AliX/actions/runs/42364732134
- Stage-B blocked order-book workflow: https://github.com/angry0tiger-sudo/AliX/actions/runs/40109476273

### External research and engineering references
- Cont, Kukanov & Stoikov (2014), “The Price Impact of Order Book Events,” Journal of Financial Econometrics. https://doi.org/10.1093/jjfinec/nbt003
- Anastasopoulos et al. (2026), “Order flow and cryptocurrency returns,” Journal of Financial Markets 79, 101047. https://doi.org/10.1016/j.finmar.2026.101047
- Guo et al. (2024), “Cross-cryptocurrency return predictability,” Journal of Economic Dynamics and Control 163, 104863. https://doi.org/10.1016/j.jedc.2024.104863
- Chi, Chu & Hao (2024), “Return-forecasting and Volatility-forecasting Power of On-chain Activities in the Cryptocurrency Market.” https://arxiv.org/abs/2411.06327
- Avellaneda & Stoikov (2008), “High-frequency trading in a limit order book.” https://doi.org/10.1080/14697680701381228
- Bailey & López de Prado (2014), “The Deflated Sharpe Ratio.” https://doi.org/10.2139/ssrn.2460551
- Bailey et al. (2015), “The Probability of Backtest Overfitting.” https://doi.org/10.2139/ssrn.2326253
- Hummingbot Avellaneda market-making documentation: https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/
- NautilusTrader backtesting data and venues: https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/

---

**Final ADR state: R&D REVIEW COMPLETE — OWNER STRATEGY SELECTION PENDING. The AliX project remains open; the previous OHLCV-only direction-prediction line is stopped. No new strategy trial or execution is authorized.**