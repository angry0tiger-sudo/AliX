# Owner Directive: Proxy Liquidity Sensitivity and Banded Book Adapter

**Date:** 2026-10-24  
**Branch:** research/proxy-liquidity-test  
**Scope:** research-only sensitivity; no deployment or trading authorization.

## Explicit owner authorization

The owner authorized creating a separate research branch, committing and pushing the proxy sensitivity script, Triple-Barrier parameter revision, banded Binance bookDepth adapter, tests, documentation, and a dedicated GitHub Actions workflow. The owner directed that all execution, testing, data processing, and simulations take place exclusively in GitHub Actions rather than local tools.

## Non-negotiable invariants

- PR #36 is not to be edited, readied, or merged by this work; it remains Draft.
- Champion is null, order_permission is false, and trading_authorization is NO-GO.
- No private API, order endpoint, real order, production model promotion, or strategy registry update.
- The Stage A historical volume column must not be overwritten with proxy volume.
- Binance quote turnover can only be a proxy sensitivity input, with at least 90% discount and a 1,500,000 quote-unit daily upper bound for BTC. The current experiment must not apply the ETH or ZEC estimates to BTCUSDT.
- A proxy gate pass is never economic acceptance.
- Banded bookDepth data must not be presented as a 20-level L2 book. Book features are blocked unless timestamped independent reference prices, price-band consistency, full coverage, permissions, and provenance checks pass.
- No CI test, quality gate, or acceptance criterion may be weakened.

## Reporting rule

Every output must distinguish (a) actual public Binance source values, (b) assumed scaling, (c) proxy-eligible simulated trade rows, and (d) actual Nobitex economic evidence. No category may silently substitute for another.
