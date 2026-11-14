# Four-Arm CPU Runtime Lock

- Python: 3.12 (GitHub Actions ubuntu-latest).
- PyTorch: 2.15.0 CPU wheel, no CUDA.
- chronos-forecasting: 2.3.1 (source tag commit 7dc4435706a4454feb79df44ca9f33631f3027bf).
- Kronos source: shiyu-coder/Kronos commit 67b630e67f6a18c9e9be918d9b4337c960db1e9a.
- Hugging Face model and tokenizer revisions and weight SHA-256 values are pinned in config/four_arm/model_pins.json; tokenizer is nested at models.kronos_small.tokenizer.
- The real experiment workflow downloads four-arm-model-weights-<run_id> and verifies its manifest and every file against the pinned digest before model load.
- transformers: 5.20.2; pandas: 2.3.3; numpy: 2.5.4; safetensors: 0.8.0; huggingface-hub: 2.2.0; einops: 0.8.1; accelerate: 1.11.0; tqdm: 4.67.1.
- Device: CPU only; torch thread count 2; inference batch size 1.
- Fine-tuning/training: forbidden; zero-shot inference only.
- Deterministic seed: 42; deterministic algorithms enabled.
- Workflow timeout: 480 minutes; memory budget: <=8 GiB.
