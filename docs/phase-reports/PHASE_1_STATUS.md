# AliX Phase 1 Status Report

Status: **IN PROGRESS / NOT ACCEPTED**

Owner approval: Phase 0 accepted and Phase 1 authorized.

## Implementation scope

- Add injected, fail-closed VPN/proxy/tunnel route drivers and route selection/failover.
- Apply selected Novaryn source contracts to a Spot/USDT Nobitex adapter boundary, without connecting to the exchange or enabling order writes.
- Add a separate Bale notification service using a pinned MIT BaleIO dependency, mock transport tests, configurable chat ID, allowlisting and disabled-by-default delivery.
- Align the Freqtrade package pin with Novaryn's 2026.7 integration baseline while retaining the existing stopped, dry-run configuration with zero allowed trades.
- Extend CI and the transitive lock to include the new integration code and dependency.

## Evidence

This is an active working report. CI is being resolved against the exact candidate commit. The phase will not be accepted until the latest candidate workflow passes, including lock consistency and security audit.

## Scope constraints

No live exchange request, order submission, real Bale send, system VPN change, proxy activation or tunnel creation has been performed. Tests inject fake transports and activation callbacks. BALE_BOT_TOKEN remains empty and reporting remains disabled in .env.example.

## Open gates

- Complete CI and committed transitive dependency lock for Freqtrade 2026.7 and BaleIO.
- Establish that the Freqtrade runtime installs and reports its version without exchange network calls.
- Preserve the fact that live read-only Nobitex and real Bale verification are deployment gates, not Phase 1 mock-test evidence.
