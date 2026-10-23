# Owner Directive: Versioned Strategy Registry and Fail-Closed Selection

- directive_id: ALIX-OD-2026-10-23-STRATEGY-REGISTRY
- owner: angry0tiger-sudo
- timestamp: 2026-10-23.
- affected_docs: docs/architecture/decisions/ADR-0011-versioned-strategy-registry-and-selector.md; docs/knowledge_base/PHASE_STRATEGY_SELECTION_ACCEPTANCE.md; docs/knowledge_base/OPEN_TASKS.md.
- affected_components: services/strategy_selection; config/strategy_selection/strategy-registry.json; tests/unit/test_strategy_selection.py.
- base_commit: 57103cf4cf7af720ec1166794df9dd62954cd38a.
- commit_sha: 57103cf4cf7af720ec1166794df9dd62954cd38a (directive baseline; implementation SHA is recorded in the PR and phase report).
- approval_status: owner-authorized implementation and testing scope from the 2026-10-23 project handover; exact candidate still requires CI, review, and normal merge governance.
- risk_impact: selection may only return an auditable candidate; it cannot approve stake, stop distance, policy changes, Kill Switch re-arm, or trade execution. Current registry remains empty and NO_TRADE is the expected production behavior.
- security_impact: no credentials, account data, exchange writes, external-service calls, order API, or risk-policy mutation are added to the selector.
- tests_required: strict registry parser; current/future/stale timestamps; quality/data fingerprint; symbol, timeframe, features, regime, cost and liquidity restrictions; economic OOS gate; fold/regime coverage; hashes/revisions/test IDs; deterministic tie-break and fingerprint; empty registry; order_permission=false.
- go_no_go: NO-GO for selecting a production strategy or trading. This directive authorizes only the research/proposal-layer implementation.

## Direct instructions

1. Create a versioned Strategy Registry whose entries explicitly record strategy ID/version/code hash, applicable regimes, timeframes, data/feature schemas, required feature and test contracts, cost/liquidity limits, risk-profile reference, prohibited market conditions and license approvals.
2. Select only from registered, owner-approved and license-approved entries with complete current data-quality evidence and version-matched OOS performance evidence.
3. Use model outputs only to rank candidates after deterministic gates. Missing or invalid evidence, unsupported regime, excessive costs, insufficient liquidity or uncertainty must produce NO_TRADE with a recorded reason.
4. Preserve Spot/USDT/Long-only scope. Do not create Futures, Margin, Leverage, Short, direct exchange REST orders or a parallel order lifecycle.
5. Do not modify the existing intentionally signal-free AliXStrategy as a shortcut. The selector does not grant order permission; Freqtrade remains the sole order-lifecycle owner.
6. Keep the registry empty until a strategy has passed the existing pre-registered economic acceptance criteria and its evidence is approved. Do not manufacture acceptance from synthetic tests.
7. Do not weaken CI gates or label a synthetic mechanics test as real market, account, fill, or order evidence.

## Governance

This directive does not authorize merging directly to main, enabling trading, promoting a Champion, or changing release gates. Effective main-branch protection must be verified before merging exchange/data-path changes.
