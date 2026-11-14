# Kronos-Small Forecast Adapter Contract

- Model repo: NeoQuasar/Kronos-small
- Tokenizer repo: NeoQuasar/Kronos-Tokenizer-base
- Inference: zero-shot, CPU-only, no fine-tuning.
- Context: latest 512 completed BTC/USDT 1h OHLCV bars including the decision-close bar.
- Forecast horizon: 24 future hourly bars.
- Forecast input columns: open, high, low, close, volume; timestamps are UTC.
- The Nobitex source CSV has no observed `amount` field. The pinned KronosPredictor implementation deterministically derives its required amount feature as `volume * mean(open, high, low, close)` when amount is absent. This is a model-input transformation, not exchange-reported amount; it is disclosed and fixed, never tuned.
- Predictor: KronosPredictor(device="cpu", max_context=512).
- Sampling: T=1.0, top_k=1, top_p=1.0, sample_count=1 for deterministic token selection.
- Forecast output: last predicted close at horizon 24.
- Adapter output: predicted 24h percentage return relative to decision close.
- Signal rule: positive percentage return only; zero and negative returns are rejected.
- No post-result threshold or forecast normalization tuning.
