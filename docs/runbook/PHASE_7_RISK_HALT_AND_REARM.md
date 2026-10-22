# Phase 7 Risk Halt, Reconciliation and Manual Re-arm Runbook

Status: operational procedure draft for future runtime integration. This document does not authorize live, paper, dry-run or exchange actions.

## 1. When a halt occurs

Triggers include absolute daily-loss or gross-exposure cap, missing/stale account or market data, unresolved order state, unapproved policy/rules, risk service error, supervisor heartbeat failure, unreadable Kill Switch state, ledger failure or ambiguous restart state.

1. Stop authorization of new entries. The Kill Switch latch is sticky; restarting a worker is not recovery.
2. Do not clear or recreate the Kill Switch database to resume service. An absent database initializes latched; a corrupt/unreadable database is a halt-unconfirmed condition and blocks entry.
3. Preserve the decision ID, policy revision, account snapshot hash, market revision, risk-gate reasons, supervisor reason, latch version and timestamps. Do not write credentials or sensitive account identifiers to logs.
4. Continue to route EXIT and REDUCE_RISK intents through the separate risk-reduction path. The Phase 7 gate does not place/cancel these orders; actual position handling remains a future, separately reviewed execution integration.
5. Mark the system HALTED or HALT_UNCONFIRMED until state and pending orders can be reconstructed.

## 2. Reconcile before recovery

A qualified operator must obtain a fresh, authoritative account snapshot and reconcile:
- equity/free balance and currency;
- realized and unrealized PnL, fees and funding if applicable;
- every open position and its remaining loss at stop;
- every open/pending/partially-filled/cancel-pending order and its reserved notional/risk;
- market tick, quantity step, min/max quantity and min/max notional;
- ledger/snapshot hashes and policy revision.

Unknown or conflicting order status is not an empty order book. Treat it as unresolved exposure and remain halted. Do not retry or cancel blindly. This phase includes no live API reconciliation.

## 3. Manual re-arm requirements

Re-arm is a separate owner-authorized operation and must use the current latch version. It requires all of:
- explicit owner approval ID and authenticated operator identity;
- reconciled account and order state;
- a healthy independent supervisor and fresh data;
- a writable/auditable ledger;
- confirmation that policy revision/caps have not changed during recovery;
- a documented incident reason and reviewable audit record.

The controller performs a compare-and-set against the version observed by the operator. A changed version, missing evidence, missing approval, unhealthy supervisor or failed reconciliation rejects re-arm. Successful re-arm creates a new version and a MANUAL_REARM audit event. No background recovery path may clear the latch.

## 4. Restart and store failure

The SQLite store persists the latch across object/process recreation and writes versioned events. A newly created store starts latched with reason initial_state_unknown. Storage read/write failure returns a fail-closed condition. The in-memory implementation is test-only and must never be used as production persistence. Production deployment must separately review directory ownership, service identity, backup/restore and filesystem access control.

## 5. Limits of this candidate

- No exchange/account connector, order submission/cancellation, notification or trading session is part of this phase.
- Risk approval does not set order_permission.
- The model/NLP process must not receive the policy loader or Kill Switch controller/store capability. OS-enforced service identities and ACLs remain an uncompleted deployment gate.
- A synthetic fault matrix does not substitute for authorized historical OOS evaluation or runtime drills.
