# Open-Source Pre-trained AI Models for AliX
## Research and integration assessment

**Research date:** 10 November 2026  
**Repository / branch reviewed:** angry0tiger-sudo/AliX / research/ml-core-ai-ensemble  
**Reviewed branch base:** 511d7e5763037cd5d2d1a9a1ced9aa48db2cdb68  
**Scope:** research only. This document does not change model code, workflows, tests, trading permissions, or execution behaviour.

---

## Executive decision

**First model to evaluate: Kronos-Small.** It is the strongest match to AliX's existing OHLCV/K-line data contract: it was pre-trained specifically on financial candlesticks, includes crypto market observations in its pre-training corpus, has public MIT-licensed weights, and is small enough to test on a standard CPU VPS. Its Mini checkpoint is a useful lower-footprint alternative, but Kronos-Small has the clearer published benchmark evidence.

**Do not treat Kronos, or any other foundation model, as an already profitable trading strategy.** The reviewed research does not establish Net Profit Factor of at least 1.1 on Nobitex BTCUSDT hourly Spot data after AliX's execution costs. Kronos's own investment simulation is on Chinese A-share equities, not crypto. Its crypto forecasting results are relevant evidence, but not a substitute for an AliX walk-forward economic test.

The recommended shortlist is:

1. **Kronos-Small / Kronos-Mini** — best domain fit; first evaluation candidate.
2. **IBM Granite TinyTimeMixer (TTM-R2)** — best lightweight general-purpose challenger for CPU/VPS and GitHub Actions.
3. **Amazon Chronos-2, with Chronos-Bolt-Tiny as the low-footprint variant** — strongest general-purpose zero-shot alternative in the reviewed benchmark sources.

TimesFM 2.5 is a credible fourth candidate, but its larger checkpoint and published financial fine-tuning study both argue against assuming that generic zero-shot forecasting is already suitable for financial prices. Moirai 2.0-R-small is technically interesting but its actual Hugging Face checkpoint is marked CC BY-NC 4.0, which is a material blocker for commercial/production use. TimeGPT is not open source at the model-weight level.

**Recommendation:** start with a zero-shot Kronos-Small evaluation, keep Kronos-Mini and TTM-R2 as resource/architecture challengers, and compare all candidates using the same untouched AliX out-of-sample window and the same cost, position, and risk rules. Do not begin with fine-tuning or an ensemble of models. First establish whether one frozen pretrained checkpoint has usable incremental signal.

---

## 1. Evaluation criteria and what “open-source pretrained” means here

The shortlist uses the following distinctions:

- **Open model:** the source code and the actual checkpoint weights are downloadable under a licence suitable for the intended use. An Apache/MIT SDK that calls a closed hosted model does not qualify.
- **Pretrained:** the candidate has published checkpoint weights learned from data other than AliX's future evaluation period. “Open-source trading bot” is not evidence that its pretrained checkpoint is included.
- **Zero-shot:** the checkpoint can issue forecasts without fitting its weights on the target AliX training window. It does not mean no input preparation, model adapter, threshold calibration, or economic validation is needed.
- **Forecast benchmark:** MASE, MAE, IC, or RankIC measures predictive fit or association. None is equivalent to profitable trading after fees, slippage, funding, spread, order constraints, and position sizing.
- **Economic benchmark:** net return, profit factor, drawdown, and risk-adjusted performance from a chronological trading simulation. The benchmark is only transferable if its assets, timeframe, execution assumptions, and costs are sufficiently close to AliX.

A model can be excellent at forecasting average time-series behaviour and still fail to predict a tradable direction at the next hourly bar. Financial price series are noisy, non-stationary, heavy-tailed, and affected by venue-specific microstructure. Cross-domain forecasting leaderboards therefore help shortlist models; they do not determine AliX acceptance.

### At-a-glance comparison

| Candidate | Pretrained for finance? | Public checkpoint / licence | Approximate size | AliX fit |
|---|---|---|---:|---|
| Kronos-Small / Mini | Yes, financial K-lines and OHLCV | Yes, MIT | 24.7M / 4.1M parameters | **Best first test** |
| IBM Granite TTM-R2 | No, general multivariate time series | Yes, Apache-2.0 | Starts around 1M; small checkpoint files | **Best lightweight challenger** |
| Amazon Chronos-2 / Bolt-Tiny | No, general time series | Yes, Apache-2.0 | 120M / 8.65M parameters | **Strong general-purpose challenger** |
| Google TimesFM 2.5 | No, general time series | Yes, Apache-2.0 for 2.5 | 200M parameters; ~925 MB checkpoint | Good, but heavier and needs finance-domain scrutiny |
| Salesforce Moirai 2.0-R-small | No, general time series | Weights public, CC BY-NC 4.0 | ~45.6 MB checkpoint | **Not recommended for commercial deployment** |
| Nixtla TimeGPT | General time series | SDK is Apache-2.0; model is closed | Hosted / proprietary | Not open-weight |
| DeepAlpha | Crypto-focused trading system | Self-hosted code is MIT; pretrained models are not included | Cloud models proprietary | Not a self-hostable pretrained option |
| Qlib / FinRL | Quant research / RL frameworks | Open-source frameworks | Depends on models trained by user | Useful tooling, not pretrained models |

---

## 2. Ranked candidates: best fit for AliX

### 1) Kronos — first choice

**Sources:** [GitHub repository](https://github.com/shiyu-coder/Kronos) · [AAAI-26 paper](https://ojs.aaai.org/index.php/AAAI/article/view/39730) · [Kronos-Mini weights](https://huggingface.co/NeoQuasar/Kronos-mini) · [Kronos-Small weights](https://huggingface.co/NeoQuasar/Kronos-small)

**What it is.** Kronos is a decoder-only financial time-series foundation model built specifically for K-line/candlestick data. Its specialised tokenizer encodes multi-dimensional OHLCV observations into discrete tokens; an autoregressive Transformer then forecasts future token sequences. This is a substantially closer input contract to AliX's candle data than a general forecasting model that expects one numeric series.

**Pre-training data.** The paper reports more than **12 billion K-line observations**, across **45 global exchanges** and seven sampling frequencies. Its exchange appendix reports Binance alone at 997 assets and approximately 1.237 billion observations, with crypto and perpetual-swap coverage. The raw pre-training corpus itself is not distributed as a reproducible public dataset, so the corpus scale is author-reported rather than independently re-counted from released source data.

**Available checkpoints.**

| Checkpoint | Parameters | Context | Licence / availability |
|---|---:|---:|---|
| Kronos-Mini | 4.1M | 2,048 | Public Hugging Face checkpoint; MIT |
| Kronos-Small | 24.7M | 512 | Public Hugging Face checkpoint; MIT |
| Kronos-Base | 102.3M | 512 | Public Hugging Face checkpoint; verify exact Hub revision/licence before pinning |
| Kronos-Large | 499.2M | 512 | Not listed as an open checkpoint in the repository model zoo |

The GitHub repository is MIT-licensed. The Mini and Small Hugging Face model cards also declare MIT. For the first AliX trial, use Small as the checkpoint with the stronger published model-level evidence; include Mini as the low-memory/longer-context challenger. The published paper's main quantitative tables report Small, Base, and Large; do not assume Mini inherits the same measured performance.

**Performance evidence.**

- The AAAI-26 paper reports that, averaged over its financial price-forecasting evaluation, Kronos improves RankIC by 93% against the leading general time-series foundation-model baseline and 87% against the best non-pretrained baseline. This is a paper-reported aggregate result, not a measured AliX trading return.
- The paper's crypto subgroup is more nuanced than the headline. In its cryptocurrency price-forecasting table, Kronos-Small reports IC around **0.0247** and RankIC around **0.0138**. At least one non-pretrained comparator in that same subgroup reports a higher RankIC (about **0.0192**). Kronos therefore does not win every crypto metric or every task.
- The paper includes a visual BTC/USDT perpetual example from Binance using 15-minute candles, a 360-step lookback, and a 120-step horizon. That is useful evidence of domain coverage, but the figure is not a BTC-specific economic backtest.
- The investment simulation is a long-only top-k/drop-n portfolio on Chinese A-share equities, with a 10-day forecast horizon, 90-day lookback, five-day minimum holding period, and 0.15% transaction cost per trade. The paper reports stronger annualised excess return and information ratio than its compared baselines, but this is not a Nobitex crypto result and does not provide AliX's target Net PF acceptance evidence.

**Integration complexity.** Medium. Kronos consumes OHLCV-style K-line inputs and returns forecast paths, making it the most natural candidate for AliX's candle-centric pipeline. The public repository provides a tokenizer, model, predictor, examples, and fine-tuning scripts. It is not a drop-in FreqAI model class. AliX still needs an adapter that loads the pinned weights, builds the required candle frame, obtains a forecast distribution/path, and converts it into stable prediction columns/signals for the existing FreqAI contract. The Mini and Small sizes make CPU evaluation practical; multiple Monte Carlo trajectories increase inference cost, so the number of samples must be fixed and budgeted.

**Risks and limitations.**

- Financial-domain pretraining is a stronger prior, not proof of profitable generalisation to Nobitex Spot BTCUSDT.
- Binance perpetual/other-exchange data is not identical to Nobitex Spot; venue basis, liquidity, volume semantics, and local market microstructure may differ.
- The pre-training corpus is not fully released, limiting independent audit and exact reproduction.
- The paper's strongest investment simulation is Chinese equities, and its crypto subgroup contains mixed results.
- Autoregressive multi-step inference and Monte Carlo sampling can be slower than a single small regression model.
- Checkpoint/model-code compatibility must be pinned and checksum-verified; do not silently follow a mutable Hugging Face branch.

**AliX assessment:** best first experiment because it combines direct financial K-line design, public MIT weights, and a CPU-manageable checkpoint family. It is the best *research fit*, not a promoted model.

---

### 2) IBM Granite TinyTimeMixer (TTM-R2) — best lightweight challenger

**Sources:** [IBM Granite TSFM repository](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer) · [TTM-R2 model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) · [NeurIPS-24 paper](https://arxiv.org/abs/2401.03955)

**What it is.** TinyTimeMixer is a compact pretrained multivariate time-series forecaster based on the TSMixer family. It uses mixing blocks rather than a large attention-heavy Transformer. IBM publishes several fixed context/prediction configurations. The model card describes TTM as a “focused” family: a checkpoint is tailored to a context length, forecast length, and frequency setup rather than one universal checkpoint for every possible task.

**Pre-training data.** The TTM-R2 card reports training on diverse publicly available time-series data with augmentations. R2 is reported at approximately **700 million samples**; the R2.1 release increases the pretraining dataset to approximately **1 billion samples**. This is broad, non-financial data rather than an exchange/K-line corpus. The specific AliX-compatible asset/market coverage is not claimed.

**Size and licence.** TTM checkpoints start around **1 million parameters** and are available under Apache-2.0. The public TTM README says the models support forecasting from minutely through hourly frequencies and can run zero-shot or with few-shot fine-tuning on laptops/CPU-capable environments. It is the clearest resource-constrained option in this shortlist.

**Performance evidence.**

- IBM reports TTM outperforming several much larger pretrained forecasting models in zero-shot and few-shot evaluations, with benchmark-specific improvements generally reported in the 4–40% range.
- In IBM's GIFT-Eval work, TTM-R2 fine-tuned separately per dataset using 20% of training data for most datasets achieved normalized MASE around **0.679**. The evaluation is chronological, but it is not a crypto trading benchmark and is not a pure “no target data” zero-shot result.
- IBM's reported fine-tuning time is a few minutes on a single A100 for its GIFT-Eval setup. The model is sufficiently small for standard VPS/GitHub Actions inference; fine-tuning and batch sizes still need to be profiled for AliX's chosen checkpoint.

**Integration complexity.** Low-to-medium. Its multivariate interface is relevant to an OHLCV input and its small weight footprint is appealing for the AliX FreqAI boundary. The main engineering constraint is checkpoint selection: context length, prediction length, and frequency must match the intended inference configuration. TTM is a forecast component, not an order or strategy engine.

**Risks and limitations.**

- It is pretrained on general public time series rather than explicitly financial K-lines.
- A general multivariate model may not capture market-specific price/volume mechanics as well as Kronos.
- GIFT-Eval improvements do not imply positive IC or a profitable crypto strategy.
- Different TTM checkpoints target different context/horizon combinations; selecting an incompatible checkpoint can invalidate the comparison.
- The model card itself recommends appropriate external scaling and cautions against artificial context-length expansion.

**AliX assessment:** best second candidate when CPU memory, inference speed, and CI budget matter. It is an excellent low-cost control against Kronos, but its generic training domain makes it less compelling as the primary financial model.

---

### 3) Amazon Chronos-2 / Chronos-Bolt-Tiny — strongest general-purpose zero-shot alternative

**Sources:** [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting) · [Chronos-2 weights](https://huggingface.co/amazon/chronos-2) · [Chronos-Bolt-Tiny weights](https://huggingface.co/amazon/chronos-bolt-tiny) · [Chronos-2 technical report](https://arxiv.org/abs/2510.15821)

**What it is.** Chronos converts numeric time series into tokens and uses pretrained language-model architectures for probabilistic forecasting. Chronos-2 extends the family to zero-shot univariate, multivariate, and covariate-informed forecasting. Chronos-Bolt uses a patch-based encoder/decoder design to generate multi-step quantile forecasts more efficiently.

**Pre-training data.** Chronos' original paper reports training on a large public time-series collection combined with synthetic Gaussian-process data. Chronos-2 uses generated synthetic multivariate structures and the public Chronos/GIFT-Eval data ecosystem for broad generalisation. The Chronos team has explicitly declined to publish further specific Chronos-2 synthetic-data and training-scale details in its GitHub discussion. The model is therefore general-domain, not a financial K-line model.

**Size and licence.**

- Chronos-2: approximately 120M parameters; public checkpoint declares Apache-2.0.
- AutoGluon Chronos-2 Small: approximately 28M parameters.
- Chronos-Bolt-Tiny: approximately 8.65M parameters; public checkpoint declares Apache-2.0.
- Chronos-Bolt-Mini: approximately 21M parameters; Chronos-Bolt-Small: approximately 48M parameters.

The small Bolt checkpoints are realistic candidates for a CPU VPS. Chronos-2 itself is larger (the Hugging Face checkpoint is roughly 478 MB).

**Performance evidence.**

- The official repository reports Chronos-2 state-of-the-art zero-shot performance on fev-bench, GIFT-Eval, and Chronos Benchmark II, and a greater-than-90% head-to-head win rate against Chronos-Bolt in the authors' comparisons. These are forecasting benchmarks, not trading metrics.
- A third-party reconstruction of public GIFT-Eval results dated 9 October 2026 reports normalized MASE around **0.698** for Chronos-2, with lower being better and scores normalized to seasonal-naive baselines. That is useful cross-domain evidence, but it is not an official financial-market or crypto score.
- No reviewed source establishes Chronos-2/Bolt Net PF, win rate, or drawdown on BTCUSDT or Nobitex after AliX costs.

**Integration complexity.** Medium. Chronos-2 has a convenient DataFrame forecasting interface and quantile output. A simple first adaptation can forecast close returns or a transformed price series; a richer path can supply covariates and related OHLCV series. Chronos-Bolt-Tiny is attractive for resource-constrained inference. AliX still needs to define the target series, quantile-to-signal mapping, forecast horizon, and FreqAI adapter.

**Risks and limitations.**

- It is not pretrained specifically on financial candle mechanics.
- The public model card and repo do not provide a BTC/Nobitex economic validation.
- Exact Chronos-2 synthetic training details are not disclosed.
- Forecasting price levels directly can be misleading in non-stationary markets; testing log returns, volatility, or other stationary targets may be more meaningful.
- Quantile coverage and calibration must be checked on the AliX market regime; an accurate generic quantile forecast is not automatically a tradable edge.

**AliX assessment:** strong general-purpose zero-shot challenger and a useful comparison against Kronos. Use Bolt-Tiny if inference footprint dominates; use Chronos-2 when broader multivariate/covariate features justify the larger model.

---

## 3. Other credible foundation models

### Google TimesFM 2.5

**Sources:** [Google Research repository](https://github.com/google-research/timesfm) · [TimesFM-2.5 weights](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [Google pretraining overview](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) · [Financial Fine-tuning a Large Time Series Model](https://arxiv.org/abs/2412.09880)

TimesFM 2.5 is a 200M-parameter decoder-only forecasting model with up to 16,384 context points and quantile forecasts up to 1,000 steps with its optional quantile head. The 2.5 code and weights are Apache-2.0; the 3.0 weights are a separate non-commercial licence and should not be substituted into a commercial AliX deployment without legal review.

Google describes a large pretraining corpus of real and synthetic data, including 100 billion time points, Google Trends, and Wikipedia pageviews. The current TimesFM 2.5 model card also lists GIFT-Eval pretraining, Wikimedia Pageviews, Google Trends, and synthetic/augmented data. These are general time-series data, not a crypto OHLCV corpus.

TimesFM has strong generic benchmark performance. However, the paper **Financial Fine-tuning a Large Time Series Model** found that direct TimesFM application to financial prices was unsatisfactory because of price-series irregularity. The authors then continually pretrained a TimesFM checkpoint on 100 million financial time points and reported improved price prediction and mock-trading outcomes across financial markets. This is an important warning: a good general TSFM may need domain adaptation before it can become a useful financial signal.

**Compute:** the PyTorch checkpoint is about 925 MB. Google's repository guidance describes CPU inference as possible with at least several GB of RAM, with batch/context choices affecting memory and speed. This is feasible on a standard VPS with a suitable RAM budget, but materially heavier than Kronos-Mini or TTM.

**AliX assessment:** credible fourth candidate, especially if AliX later explores model fine-tuning or covariates. It is not the first choice for a strict pretrained-only financial experiment.

### Salesforce Moirai 2.0-R-small

**Sources:** [Salesforce Uni2TS repository](https://github.com/SalesforceAIResearch/uni2ts) · [Moirai 2.0 announcement](https://www.salesforce.com/blog/moirai-2-0/) · [Moirai 2.0-R-small weights and licence](https://huggingface.co/Salesforce/moirai-2.0-R-small)

Moirai 2.0 is a general time-series foundation model. Salesforce announced it as the best MASE model on GIFT-Eval among non-test-data-leaking models at its August 2025 release. Its pretraining mixture includes GIFT-Eval datasets, Chronos mixup data generated from non-leaking subsets, KernelSynth synthetic series, and internal Salesforce operational data. It is a capable research model, not a finance-specific market model.

**Important licensing distinction:** the Uni2TS source repository and the model weights have separate licences. The actual Moirai 2.0-R-small Hugging Face checkpoint declares **CC BY-NC 4.0**. That is non-commercial. A commercial/production AliX use case should not assume the checkpoint is cleared merely because the source code is public or differently licensed.

**AliX assessment:** retain as a non-commercial benchmark option only if the use case and counsel confirm that the NC restriction is acceptable. Do not place it in the production shortlist under the current commercial-use assumption.

---

## 4. Trading frameworks and crypto products that are often confused with pretrained models

### DeepAlpha

**Source:** [DeepAlpha GitHub repository](https://github.com/stefanoviana/deepalpha)

DeepAlpha is an open-source crypto trading system, but its own README states that the self-hosted version includes training scripts and **does not include pretrained models**. The pretrained production models are exclusive to the vendor's paid cloud offering. The repository is MIT-licensed, but that does not make the cloud weights open or redistributable.

The README advertises 70.9%, 84.6%, and 84.9% accuracy figures in different sections, as well as a profit factor of 2.91, maximum drawdown of 20.7%, and Sharpe of 0.97. These are vendor/repository-reported claims; I found no independent reproduction that establishes the claims for AliX's asset, venue, timeframe, costs, or execution assumptions. The free/self-hosted workflow requires the user to download data and train the models. Its feature descriptions also include Level-2 order-book, funding-rate, and microstructure data, which are not available in AliX's current OHLCV-only research contract.

**Verdict:** not the requested “open-source pretrained model”; the pretrained weights are not in the self-hosted repository. Do not adopt it on the premise that it is a drop-in FreqAI pretrained predictor.

### Microsoft Qlib

**Source:** [Microsoft Qlib GitHub](https://github.com/microsoft/qlib) · [Qlib benchmark results](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

Qlib is an open-source quantitative research platform with data processing, signal/model workflows, portfolio construction, and backtesting. It is not itself a downloadable pretrained price-prediction foundation model. Its public Alpha158/Alpha360 benchmark tables evaluate task-specific workflows on Chinese equity datasets such as CSI300. Those results are useful quant-research references, but they do not establish crypto or Nobitex transfer.

**Verdict:** useful research/backtesting infrastructure, not a shortcut to pretrained weights. AliX already uses Freqtrade/FreqAI; adding Qlib would risk duplicating portfolio and backtesting layers without solving the pretrained-model question.

### FinRL

**Source:** [FinRL repository](https://github.com/AI4Finance-Foundation/FinRL) · [FinRL FAQ](https://finrl.readthedocs.io/en/latest/faq.html)

FinRL is an open-source reinforcement-learning framework with trading environments and algorithms such as PPO, SAC, DDPG, TD3, and A2C. Its current offering includes example environments for stocks, crypto, and FX, but the framework requires the user to train and tune an agent for the target environment. The official FAQ (version 0.3, dated 2022) explicitly says no pretrained model is available there and cautions that its default parameters are for daily data and need tuning for intraday trading.

**Verdict:** a possible later research framework for an RL risk-management phase, not a pretrained forecasting model and not a way to avoid target-domain training.

### Nixtla TimeGPT

**Source:** [Nixtla SDK repository](https://github.com/Nixtla/nixtla)

The Nixtla SDK is Apache-2.0, but TimeGPT itself is closed source. Forecast requests execute against the proprietary model service and require an API key under the service terms. An open-source client is not equivalent to open weights that AliX can pin, inspect, self-host, and run offline.

**Verdict:** exclude from an open-source/self-hosted model shortlist unless AliX explicitly accepts a hosted commercial dependency and its data/privacy/service constraints.

### FinGPT and financial LLMs

FinGPT and similar financial language models are primarily for financial text tasks such as sentiment, news, or document analysis. They are not replacements for a pretrained numeric OHLCV forecasting model. They could become a separate feature source if AliX later approves text data and an additional point-in-time data pipeline, but they do not solve the present candle-forecasting objective.

---

## 5. FreqAI/Freqtrade integration assessment

AliX already has a FreqAI regression model boundary: the current baseline class subclasses Freqtrade's BaseRegressionModel and implements its fit contract. This gives AliX a natural place to expose an alternative predictor, but none of the shortlisted foundation models is a drop-in FreqAI strategy or risk manager.

A suitable integration should keep the responsibilities separate:

1. **Input adapter:** provide only completed, timestamp-ordered candles to the pretrained model. Kronos naturally consumes OHLCV-style bars; TTM supports multivariate series; Chronos and TimesFM can forecast selected targets with optional covariates.
2. **Inference adapter:** load a pinned, checksum-verified checkpoint and return forecasts/quantiles for a fixed horizon. The adapter must be deterministic where the model supports sampling, or explicitly fix the sampling count and seed.
3. **Signal adapter:** translate forecast paths and uncertainty into a small, well-defined prediction set for FreqAI/strategy consumption (for example forecast return distribution, directional score, and uncertainty). A foundation-model price path should not be translated directly into an order.
4. **Independent evaluation:** compare the frozen pretrained model with the current baseline on the same chronological split. Avoid changing thresholds, costs, or position rules for one model only.
5. **Independent risk/execution:** preserve the existing risk gates, dry-run policy, and execution permissions. The predictor must not receive exchange credentials or order-write capabilities.

**Integration difficulty estimate:**

| Model | FreqAI adapter effort | Runtime footprint | Main adapter issue |
|---|---|---|---|
| Kronos-Small | Medium | Low-to-medium | OHLCV tokenizer and forecast-path-to-signal mapping |
| Kronos-Mini | Medium | Very low | Less published benchmark evidence than Small |
| TTM-R2 | Low-to-medium | Very low | Choose a checkpoint matching context, horizon, and frequency |
| Chronos-2 | Medium | Medium | Define target/covariates and quantile-to-signal mapping |
| Chronos-Bolt-Tiny | Low-to-medium | Low | Smaller generic model may trade accuracy for footprint |
| TimesFM 2.5 | Medium | Medium-to-high | 200M checkpoint and financial-domain adaptation question |
| Moirai 2.0-R-small | Medium | Low-to-medium | Non-commercial weights licence is a blocker |

A custom adapter is still required because the model output contract is forecasting, while FreqAI's model interface and AliX's strategy/risk contracts are prediction-oriented. None of these models should be described as “zero integration”.

---

## 6. Recommended AliX evaluation sequence

This is a recommended future experiment, not an instruction to modify code in this research task.

### Phase A — pretrained inference only

1. Start with Kronos-Small at its published checkpoint and a pinned revision. Evaluate Kronos-Mini as a separate capacity/context variant, not as a silent replacement.
2. Use the exact same BTCUSDT Spot 1h public OHLCV data and chronological evaluation boundaries already approved for AliX. Do not add exchange account endpoints or private data.
3. Run TTM-R2 and Chronos-Bolt-Tiny as low-cost controls. Run Chronos-2 as the stronger general-purpose comparator if the runtime budget permits.
4. Freeze input preparation, prediction horizon, forecast-to-signal mapping, and decision threshold before opening the out-of-sample test.
5. Record model/checkpoint revision, weight hash, library versions, input timestamp range, inference settings, and output hash so each run is reproducible.

### Phase B — signal and forecast evaluation

Measure at least:

- Forecast MAE/MASE and quantile/interval calibration.
- Directional hit rate, IC, and RankIC, with confidence intervals where feasible.
- Performance by volatility/regime and by time segment.
- Stability under missing/late candles and model restart/reload.
- Inference latency and peak CPU RAM on the intended VPS/GitHub Actions runner.

Do not select a model on MAE alone. A model can predict price levels well while having negligible return-direction signal.

### Phase C — economic acceptance

Only after forecast evaluation should the frozen signal be tested under the existing AliX position and risk rules. Keep transaction cost, spread/slippage assumptions, trade overlap rules, and position sizing identical across candidates. Use the approved AliX economic metrics: **Net PF, win rate, max drawdown, statistical Z-score, net return, and trade count**. The acceptance target remains **Net PF ≥ 1.1**, with adequate trade count and statistically meaningful evidence. A checkpoint that fails the frozen out-of-sample test should not be promoted by repeatedly tuning thresholds on the test window.

A successful forecasting benchmark, a high accuracy claim, or a paper's portfolio simulation does not satisfy this gate.

---

## 7. Final ranking and recommendation

| Rank | Candidate | Reason |
|---:|---|---|
| **1 — Best fit** | **Kronos-Small** | Finance-native OHLCV/K-line pretraining, crypto included in corpus, MIT weights, 24.7M parameters, and published financial forecasting evidence. Best balance of domain match, openness, and practical footprint. |
| **2** | **IBM TTM-R2** | Tiny Apache-licensed pretrained multivariate model, minute/hourly support, and CPU/laptop inference. Best low-cost control, but generic rather than finance-native. |
| **3** | **Chronos-2 / Chronos-Bolt-Tiny** | Strong zero-shot general forecasting benchmark evidence, Apache checkpoints, and an 8.65M-parameter compact variant. Less financial-domain alignment and limited disclosure of Chronos-2 training detail. |

**Recommendation: evaluate Kronos-Small first, with Kronos-Mini and TTM-R2 as controls.** Kronos is the only top candidate in this shortlist whose pretraining objective, tokenizer, and data are explicitly designed around financial OHLCV candles. Its published crypto results are not uniformly best and its Chinese-equity simulation is not transferable to AliX, so this is a prioritisation decision for an experiment—not a claim that Kronos is already profitable.

Do **not** start with DeepAlpha on the assumption that a free pre-trained checkpoint or a drop-in FreqAI implementation is included. Do **not** choose TimeGPT on the basis that its SDK is open source. Do **not** use Moirai 2.0 weights for a commercial/production path unless the licence restriction is resolved. Do **not** treat Qlib or FinRL as pretrained models.

### Safety state remains unchanged

- Champion = null
- order_permission = false
- trading_authorization = NO-GO
- private_api_calls = 0
- real_order_attempts = 0
- No exchange account credentials, private endpoints, or order APIs are required for these public-weight inference experiments.

This research report does not authorise paper/live deployment or model promotion.

---

## Sources and evidence trail

1. [Kronos GitHub repository and model zoo](https://github.com/shiyu-coder/Kronos) — architecture, checkpoint sizes, tokenizer, examples, and repository licence.
2. [Kronos Hugging Face Mini checkpoint](https://huggingface.co/NeoQuasar/Kronos-mini) and [Small checkpoint](https://huggingface.co/NeoQuasar/Kronos-small) — public weights and MIT checkpoint licence.
3. [Kronos: A Foundation Model for the Language of Financial Markets, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39730) and [arXiv paper](https://arxiv.org/abs/2508.02739) — pretraining scale, exchange coverage, benchmark design, financial/crypto results, and investment-simulation design.
4. [IBM Granite TSFM / TTM repository](https://github.com/ibm-granite/granite-tsfm) and [TTM-R2 model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) — model size, Apache-2.0 licence, training-data scale, input frequency and benchmark information.
5. [Tiny Time Mixers paper](https://arxiv.org/abs/2401.03955) and [IBM research summary](https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1) — NeurIPS 2024 publication and resource-light inference claims.
6. [Amazon Chronos repository](https://github.com/amazon-science/chronos-forecasting), [Chronos-2 weights](https://huggingface.co/amazon/chronos-2), and [Chronos-Bolt-Tiny weights](https://huggingface.co/amazon/chronos-bolt-tiny) — current model family, parameter counts, usage, and Apache-2.0 licensing.
7. [Chronos-2 technical report](https://arxiv.org/abs/2510.15821) and [Chronos-2 training-data discussion](https://github.com/amazon-science/chronos-forecasting/discussions/472) — zero-shot scope and limits on disclosed training details.
8. [Google TimesFM repository](https://github.com/google-research/timesfm), [TimesFM 2.5 checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch), and [Google Research pretraining overview](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) — checkpoint size, licence, context, and pretraining data.
9. [Financial Fine-tuning a Large Time Series Model](https://arxiv.org/abs/2412.09880) — evidence that direct TimesFM zero-shot price forecasting can be unsatisfactory and that financial-domain continual pretraining changes results.
10. [Salesforce Uni2TS repository](https://github.com/SalesforceAIResearch/uni2ts), [Moirai 2.0 announcement](https://www.salesforce.com/blog/moirai-2-0/), and [Moirai 2.0-R-small checkpoint licence](https://huggingface.co/Salesforce/moirai-2.0-R-small) — architecture, benchmark claim, and non-commercial model-weight licence.
11. [Nixtla SDK repository](https://github.com/Nixtla/nixtla) — distinction between Apache-licensed SDK and closed-source TimeGPT model.
12. [DeepAlpha repository](https://github.com/stefanoviana/deepalpha) — self-hosted pretrained-weight availability, cloud-only model claims, and vendor-reported metrics.
13. [Microsoft Qlib repository](https://github.com/microsoft/qlib) and [Qlib benchmark results](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) — quantitative research platform and China-equity benchmark context.
14. [FinRL repository](https://github.com/AI4Finance-Foundation/FinRL) and [FinRL FAQ](https://finrl.readthedocs.io/en/latest/faq.html) — RL framework scope, intraday tuning caveat, and pretrained-model FAQ.
15. [GIFT-Eval result reconstruction](https://aurion.semai.id/benchmarks/gift-eval/) — third-party reconstruction of public benchmark results dated 9 October 2026; used only as cross-domain forecasting evidence, not a crypto-trading claim.

**Evidence discipline:** vendor/paper claims are labelled as such. The third-party GIFT-Eval reconstruction is not treated as a crypto backtest. No reviewed source demonstrates Net PF ≥ 1.1 on AliX's Nobitex BTCUSDT 1h Spot setup after AliX costs.
