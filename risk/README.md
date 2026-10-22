# Risk Plane

The risk package is a deterministic, model-independent safety boundary. It uses Python standard-library primitives only and does not import NLP/model inference, external-data, strategy or exchange code.

- contracts.py defines immutable, versioned policy, reconciled account, market/rules, proposal and allocation records. Monetary inputs use Decimal and timestamps require timezone awareness.
- allocator.py computes a dynamic proposal using ATR/price, regime, calibrated confidence, uncertainty, calibration error, after-cost edge, remaining daily/absolute budget, fee/slippage, free balance, exposure and correlation.
- gate.py independently recalculates estimated loss, precision, stop validity, freshness, account reconciliation and policy/exposure limits. Its risk_approved field is not order permission; order_permission is always false in Phase 7.
- kill_switch.py includes a durable SQLite latch and a test-only in-memory adapter. Missing initial state defaults to latched, persistence errors are fail-closed, and manual re-arm requires current version, owner approval, reconciliation and supervisor-health evidence.
- supervisor.py latches on absolute cap or critical health/reconciliation faults and never automatically re-arms.
- evaluation.py calculates paired walk-forward/OOS metrics; it intentionally does not decide economic acceptance.

The package is not wired into Freqtrade, strategies, a real account snapshot, order lifecycle, model process or deployed supervisor. The owner-approved flag is a contract field, not a digital signature. Trusted policy loading, OS process isolation/ACLs, runtime heartbeat, actual account/order reconciliation and authorized OOS research remain blockers. No real trading is authorized.
