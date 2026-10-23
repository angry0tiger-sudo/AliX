# ADR-0014: Audited AI Training and Risk-Gated Shadow Execution

- Status: Proposed implementation candidate; not a release or trading authorization.
- Date: 2026-10-23.
- Owner Directive: docs/governance/owner-directives/2026-10-23-ai-upgrade.md.
- Depends on: ADR-0011 versioned strategy registry and selector.
- Scope: historical Gaussian-HMM training, OOF Meta-Label training, causal strategy indicators, dynamic ATR stop, near-breakeven time-decay exit, and an isolated SQLite-backed mock exchange.

## Decision

Historical model fitting runs only from a manual GitHub Actions workflow. The workflow fetches public Nobitex OHLCV with no credential, validates the exact 8,760-row hourly grid, records the source SHA-256, performs chronological technical validation, trains final artifacts, and uploads a bounded-retention artifact. It does not commit model weights, deploy them, or authorize a strategy.

HMM model weights use a transparent versioned JSON schema with a digest over the model parameter body. The runtime loader rejects mutated weights, mismatched schema, invalid source metadata, failed technical validation, and release flags other than NO-GO. Technical approval is held in a separate source-controlled record in `config/strategy_selection/` that pins both model and dataset hashes. The training workflow never writes or approves those records; any approval change must be reviewed as a distinct repository change. This hash is tamper-evident, not a cryptographic owner signature. A trained model is not a promoted model.

Meta-Label training uses purged, chronological out-of-fold primary predictions and triple-barrier outcome labels with explicit fee and slippage assumptions. Its separate technical-approval record must pin the model and dataset hashes and must be changed through review. The 0.70 score is a research candidate filter only; it is not calibrated probability proof, a substitute for account-specific cost verification, or an override of Risk Gate.

AliXStrategy can generate a long candidate only when a valid HMM regime, the reviewed Meta-Label artifact, and an explicit upstream SELECTED result agree. A missing model, unknown regime, missing selector output, low score, invalid OHLCV, or failed approval check yields no entry. Technical exits, dynamic ATR stops, and the 12-candle near-breakeven exit do not depend on availability of the entry model.

ShadowExchange is not an exchange adapter. It has no networking code or credentials. It models round-trip reference price, slippage and per-side fees, writes an append-only SQLite record, and is idempotent by decision ID. A synthetic fill requires a positive independent Risk Gate decision, the 0.70 filter, valid inputs, and order_permission=false. A rejected gate can be recorded as REJECTED but can never become a fill.

## Consequences

- Production registry and model owner approval remain separate promotion artifacts.
- The current empty strategy registry and absent approved model bundle keep entries disabled.
- Historical training data is public market data, not account, order, fee-tier, fill, or reconciliation truth.
- HMM holdout likelihood and Meta-Label scores are technical diagnostics, not evidence of positive economic expectancy.
- Any future execution work must remain behind the official Freqtrade boundary and a separate release gate.
