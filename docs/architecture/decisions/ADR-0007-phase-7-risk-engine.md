# ADR-0007: Independent Risk Plane and Latched Halt

- Status: Accepted for isolated implementation and synthetic contract testing only. Runtime deployment, economic acceptance and trading remain NO-GO.
- Date: 2026-10-22.
- Owner directive: docs/governance/owner-directives/2026-10-22-phase-7.md.
- Base commit: 5873ba52ab827109a0be450670e971796295fa8e.

## Context

The Phase 7 roadmap requires risk allocation to respond to volatility, regime, calibrated confidence, uncertainty, transaction costs, remaining equity/budget, exposure and correlation while an absolute protective cap remains outside model authority. Missing or stale state must not permit a new entry. The previous risk package contained only a boundary README.

## Decision

1. Keep the risk plane as a standard-library-only Python package. It has no imports from NLP, external-data, strategy or exchange code.
2. Require all dynamic and absolute risk parameters in an immutable, versioned RiskPolicy supplied to the call. No daily risk amount, stake fraction, stop multiplier or cap is chosen by the allocator itself.
3. Require reconciled account/order snapshots, a point-in-time market snapshot, approved market precision rules, a proposal with explicit revision/calibration/uncertainty metadata and a fresh decision time.
4. Apply deterministic risk multipliers no greater than one for confidence, uncertainty, calibration error, ATR, regime, gross exposure and correlated exposure. Unknown regime, exhausted cap, stale state, non-positive after-cost edge or missing inputs yields no allocation.
5. Calculate stop-distance loss plus entry/exit fees and slippage conservatively. Quantize quantity downward and stop price in the conservative direction; the gate repeats the calculation independently.
6. Keep Risk Gate permission separate from order execution. The candidate can report risk approval, but order_permission is always false in Phase 7 because no execution integration or release authorization is included.
7. Store the Kill Switch as a durable, atomic, versioned latch with an append-only local event table. A newly created store starts latched, unreadable/corrupt state fails closed, and re-arm requires explicit owner approval, reconciled account/orders, healthy supervisor and a compare-and-set version.
8. Run the supervisor as a separate capability: it can latch but never automatically re-arm or mutate risk policy. Model/NLP workers must not receive policy-store or Kill Switch-controller handles.
9. Provide a pure OOS metric calculator for paired allocator/baseline returns. It reports cumulative return, drawdown, tail CVaR, costs, turnover and regime spread, but does not grant economic acceptance.
10. Use CI fault injection and downloadable JSON evidence with synthetic fixtures only. Do not invoke a provider, model download, exchange client or order lifecycle.

## Consequences and limitations

- A gate pass is a risk-plane decision only, not an order authorization.
- RiskPolicy owner_approved is a contract field, not a cryptographic signature. A trusted owner-controlled policy loader, OS process identity/ACL enforcement, heartbeat wiring and real account/order reconciliation adapter remain deployment blockers.
- SQLite persistence and a restart test verify local store mechanics, not production filesystem hardening, multi-host consensus or a deployed supervisor.
- No permitted historic Phase 7 dataset or pre-registered economic acceptance thresholds are present. OOS economic performance remains unproven and must remain NO-GO.
- Champion remains null and trading authorization remains NO-GO.
