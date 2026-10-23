# AI Upgrade Acceptance — Research Candidate

Status: implementation candidate; release NO-GO.

## Required evidence

- [ ] Manual GitHub Actions run has retrieved exactly 8,760 public Nobitex BTC/USDT hourly candles.
- [ ] Dataset and manifest SHA-256 agree; timestamps are unique and increasing; no gaps or interpolation; OHLCV invariants pass.
- [ ] HMM training and chronological technical holdout diagnostic complete, with model JSON hash and feature schema recorded.
- [ ] Meta-Label training has purged OOF provenance, mature triple-barrier labels, explicit modeled costs, JSON round-trip and chronological holdout metrics.
- [ ] Model artifacts are reviewed and owner approval is explicit; training cannot set it automatically.
- [ ] Trend/range candidate logic is tested with selector result absent, rejected, stale, or selected; absent/rejected cases generate no entry.
- [ ] ATR stops are regime-sensitive and protective exits remain available when the entry model is missing.
- [ ] Time-decay exit is triggered only after 12 full candles and within the defined near-breakeven band.
- [ ] Shadow fills require positive independent Risk Gate approval and score >= 0.70; a failed gate returns REJECTED with no fill.
- [ ] Shadow SQLite ledger is append-only, replay-idempotent and records conflicting decision-ID reuse as an error.
- [ ] Ruff, strict Mypy, Bandit, dependency audit, Xenon, architecture contracts, tests and >=90% coverage pass in GitHub Actions.

## Non-claims

Technical likelihood does not establish profitable regime classification. Meta-Label scores do not establish calibrated economic win probabilities. The 25 bps/side fee and 5 bps/side slippage used here are conservative research assumptions, not verified account costs. The public dataset is not account/order/fill truth. Shadow fills are synthetic and unreconciled.

Champion=null, order_permission=false, promotion_status=BLOCKED_NO_ECONOMIC_GATE, economic_acceptance=NO-GO and trading_authorization=NO-GO remain unchanged.
