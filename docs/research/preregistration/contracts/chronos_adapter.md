# Chronos-2-Small Forecast Adapter Contract

- Model repo: autogluon/chronos-2-small (the true Chronos-2 small family, not Chronos-T5).
- Inference: zero-shot, CPU-only, no fine-tuning.
- Context: latest 512 completed close prices through the decision-close bar.
- Forecast horizon: 24 hourly steps.
- Pipeline: Chronos2Pipeline from pinned chronos-forecasting==2.3.1 code.
- Prediction: univariate target close, context_length=512, prediction_length=24, batch_size=1, cross_learning=false.
- Forecast output: 0.5 quantile at step 24.
- Adapter output: median predicted 24h percentage return relative to decision close.
- Signal rule: positive percentage return only; zero and negative returns are rejected.
- No future covariates, fine-tuning, or post-result threshold tuning.
