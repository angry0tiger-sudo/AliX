# Owner Directive: Autonomous Meta-Strategy Research

Date: 2026-10-24

The owner directs AliX research to move from single-strategy prediction toward research-only meta-selection across four independent strategy families: trend following, mean reversion, volatility breakout, and scalping. An HMM may route a candidate only when regime confidence exceeds the configured threshold and the candidate's data prerequisites are met. ATR- and confidence-derived sizing and stops/targets are simulation-only.

This directive does **not** authorize production activation. The following invariants override any conflicting strategy document or prior experiment description:

- \`Champion = null\`
- \`order_permission = false\`
- \`trading_authorization = NO-GO\`
- no real orders, no private API calls, no model promotion, and no production-registry mutation
- no synthesized positive volume, venue liquidity, spread, or order-book evidence
- the reserved final holdout remains unused
- no CI or economic acceptance gate may be weakened to make a strategy pass

A historical-bar simulation can report exploratory win rate and expectancy only under explicit fee/slippage assumptions. It does not establish live profitability, execution capacity or permission to trade.
