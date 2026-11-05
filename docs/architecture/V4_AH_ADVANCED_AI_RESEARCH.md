# AliX V4-AH — Advanced AI Meta-Model Research Architecture

**Status:** Research-only implementation candidate  
**Branch:** `research/ml-core-v4ah-advanced-ai`  
**Base:** `2ac5b95d4384402bcf900169edbb9afa0a6b470d`  
**Market scope:** Binance Spot public ETHUSDT, 1h, 730 days  
**Execution authority:** None. No live or paper orders are permitted.

## 1. Evidence and problem statement

The V4-AF experiments show that the existing basic-feature LightGBM filter has not demonstrated reliable economic value. On the verified Activation 1.5 ATR baseline (Run #16), Primary-only produced 42 trades, Net PF 1.1538, win rate 30.95%, max drawdown 9.44%, and net PnL +$385.97. The Top-70 filter produced 34 trades, Net PF 1.1898, win rate 32.35%, max drawdown 7.88%, and net PnL +$410.07. This is a small uplift, not acceptance. On the later Activation 1.0 ATR sensitivity (Run #17), Primary-only produced 43 trades, Net PF 1.0185, win rate 30.23%, max drawdown 9.60%, and net PnL +$45.07; Top-70 fell to Net PF 0.9498 and net PnL -$112.80. Neither run met the owner's economic targets.

Run #17 recorded 270 matured costed OOF events, but only 41 rows in the former calibration slice. V4-AH therefore reports row counts and fails closed when the three disjoint chronological training slices do not retain minimum support. More elaborate methods are not evidence of more reliable estimates when sample support is small.

## 2. Decision architecture

The existing causal primary remains the source of directional BUY proposals: Bollinger(20,2) upper-band breakout with Close above EMA200, generated from a closed candle and executed at the next bar open in the offline simulator. The AI Meta-Model is the central accept/reject decision for those proposals. It does not receive order permission and does not bypass the primary signal.

The outer chronological data partition remains:

- First 70%: outer training and OOF label construction.
- Next 15%: forward validation, used only for final candidate comparison.
- Last 15%: final holdout, completely untouched in this phase.

Within the first 70%, matured OOF primary BUY events are split chronologically into Fit, Temperature Calibration, and Net Expectancy Threshold slices (70/15/15). Label end times are purged across both slice boundaries. The model, feature selector, temperature, and economic threshold use no forward-validation labels.

## 3. Feature and regime pipeline

The candidate feature matrix is the repository's causal Combined profile: baseline returns, RSI, MACD, ATR proxy and volume features, plus Trend, Volatility and Volume groups. It includes ADX and realized-volatility inputs. K-Means regime features are added from ATR, realized volatility, ADX and EMA spread. The scaler and K-Means centroids are fitted only on the 200-bar warm-up before the first eligible OOF event, then used only to transform later bars. Regime identity is a model feature, not a hard-coded market gate.

A first LightGBM model is fitted only on the Fit slice. TreeSHAP mean absolute attributions rank the inputs; the top 30 are retained. A second LightGBM is refitted on those selected inputs. This is a supervised feature-selection experiment, not a guarantee that the selected features generalize.

## 4. Labels, calibration, and threshold

The target is whether the isolated primary BUY replay has positive net return after fees and slippage. Each OOF event is replayed with the same execution simulator and geometry as the economic comparison: ATR(14), initial SL 2.0 ATR, TP 4.0 ATR, trailing activation 1.0 ATR, reference trailing step 1.5 ATR, and a 24-bar vertical horizon. The trailing ratchet and vertical exit are therefore part of the label outcome, not a separate fixed-horizon proxy. Future bars are accessed only for outcomes, never as model features.

Temperature scaling is fitted with L-BFGS-B on the dedicated purged calibration slice. The Net Expectancy threshold is selected from the third, separate training slice by maximizing mean costed net return among thresholds with a minimum selected sample of at least 30% of that slice and at least 12 events. If no threshold has positive observed expectancy, all Meta candidates are skipped. That fail-closed outcome is preferable to manufacturing a threshold from negative evidence.

A reliability curve, Brier score and ECE are reported on calibrated probabilities in the threshold-selection slice. This is a training diagnostic, not an independent final calibration assessment; final holdout remains untouched.

## 5. Adaptive exits and Kelly overlay

A secondary Random Forest policy chooses among the pre-registered trailing ratchet increments {1.0, 1.5, 2.0} ATR. Its event targets come from isolated, costed counterfactual replays. Every counterfactual replay must exit strictly before the temperature-calibration boundary; any event whose maximum horizon could cross that boundary is excluded. The activation threshold remains fixed at 1.0 ATR in this experiment. The selected step is a model output, not an unrestricted continuous stop.

The Kelly candidate uses the calibrated Meta probability and the Fit-slice average win/loss payoff ratio. It applies quarter-Kelly and caps the position fraction at 10% of equity. It is an offline sizing comparison only; it is not connected to the production risk engine. PnL and drawdown are calculated on the actual sized equity path. To compare signal edge fairly against the full-sized Primary control, outperformance checks use Net PF and unscaled per-trade net expectancy; raw dollar PnL is still reported but is sizing-dependent.

## 6. Candidate matrix and acceptance

The workflow reports four candidates on the same ETHUSDT 1h forward-validation window and cost contract:

1. `UNFILTERED_CONTROL` — existing Primary-only reference.
2. `META_NET_EXPECTANCY_FIXED` — SHAP LightGBM and calibrated Net Expectancy threshold with fixed V4-AF stop step.
3. `META_SHAP_ADAPTIVE_TRAIL` — same Meta decisions with the trained trailing-step policy.
4. `META_SHAP_ADAPTIVE_TRAIL_KELLY` — same Meta and trailing policy with quarter-Kelly sizing.

Each Meta candidate must meet all owner targets: Net PF > 1.5, max DD < 15%, win rate > 45%, at least 30 trades, and outperformance of Primary on Net PF and unscaled per-trade net expectancy. No candidate is promoted automatically. The last 15% remains untouched pending a separately authorized OOS decision.

## 7. Safety and engineering constraints

`Champion=null`, `order_permission=false`, `trading_authorization=NO-GO`, `private_api_calls=0`, `real_order_attempts=0`, and `sequence_model_used=false`. Only the public Binance Spot OHLCV endpoint is used. No account endpoint, private API, order endpoint, live session or paper order is used. The `risk` package and production execution remain unchanged.

All implementation, tests and the real-data experiment run in GitHub Actions. The V4-AH workflow requires Standard CI success for the exact commit SHA before fetching public data or running research. Coverage remains at least 90%; no CI gate is removed or relaxed.

## 8. Research references and limits

- Lundberg & Lee (NeurIPS 2017), *A Unified Approach to Interpreting Model Predictions*: SHAP gives additive feature attributions; the paper does not establish that using those attributions for feature pruning improves out-of-sample trading performance. <https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
- Guo et al. (ICML 2017), *On Calibration of Modern Neural Networks*: temperature scaling is a simple post-hoc calibration method in their neural-network experiments. That result is not direct proof of calibration gains for LightGBM; V4-AH measures the effect using chronological market data. <https://proceedings.mlr.press/v70/guo17a.html>
- López de Prado (2018), *Advances in Financial Machine Learning*, Chapter 3: describes Triple-Barrier labeling and Meta-Labeling. V4-AH uses the repository's costed execution simulator so the barrier exits align with the actual trailing-stop mechanics. <https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml>
- Busseti, Ryu & Boyd (2016), *Risk-Constrained Kelly Gambling*: formulates Kelly sizing with explicit drawdown-risk constraints. V4-AH's capped quarter-Kelly is a conservative heuristic experiment, not their risk-constrained optimizer. <https://web.stanford.edu/~boyd/papers/kelly.html>

None of these methods guarantees profitable trading. The only economic evidence is the paired forward-validation report produced by the exact GitHub Actions run; any holdout use requires a separate owner-authorized step.