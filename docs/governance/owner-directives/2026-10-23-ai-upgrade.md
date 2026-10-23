# Owner Directive: AI Upgrade, Causal Strategy Signals, and Shadow Trading

- directive_id: ALIX-OD-2026-10-23-AI-UPGRADE
- owner: angry0tiger-sudo
- date: 2026-10-23
- status: implementation candidate; no trading authorization
- branch: feature/ai-upgrade
- related ADR: ADR-0014-ai-training-and-shadow-execution.md

## Requirements

1. Train and technically validate a versioned three-state Gaussian HMM with the real public Nobitex BTC/USDT 1-hour dataset, exactly 8,760 candles, source manifest and dataset SHA-256.
2. Save transparent JSON model parameters and lineage in a manual-only GitHub Actions artifact. Owner approval and economic promotion are separate gates.
3. Generate strategy candidates from the validated HMM regime and an explicit upstream selector result. Trend Following and Mean Reversion are long-only. Missing or stale model/data, missing selector output, unavailable Meta-Model, and scores below 0.70 must fail closed.
4. Use ATR-based dynamic stop distance, wider for trend and tighter for range/high volatility. An open long near breakeven after 12 full timeframe candles may be exited through Freqtrade's custom-exit contract.
5. Use the existing purged OOF Meta-Label contracts with triple-barrier labels and explicit modeled fees/slippage. A 0.70 score is a filter, not Risk Gate authorization or evidence of calibrated confidence. HMM and Meta-Model technical approval must be separate source-controlled records pinning exact artifact and dataset hashes; training must never self-approve.
6. Add a SQLite-backed Mock Shadow Exchange and append-only ledger. It must include slippage and per-side fees, idempotency, replay-conflict detection and explicit synthetic provenance.
7. Do not bypass Risk Gate to create mock fills. A failed risk decision must be recorded as rejected, never simulated as a fill. No exchange credentials, private account endpoint, order endpoint, real order submission, or live adapter is permitted.
8. Preserve Champion=null, order_permission=false, promotion BLOCKED_NO_ECONOMIC_GATE, economic acceptance NO-GO and trading authorization NO-GO until all release criteria are separately passed.

## Acceptance boundaries

- CI must retain all existing quality gates, including at least 90% coverage.
- Training artifacts must be reproducible from the dataset, immutable code SHA, configuration and pinned dependencies.
- Public-data technical validation is not economic acceptance. Synthetic shadow results are mechanics evidence only.
- No artifact may set owner approval or promotion state automatically.
