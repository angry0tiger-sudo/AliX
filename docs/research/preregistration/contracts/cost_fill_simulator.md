# Cost and Fill Simulator Contract

- Position direction: long-only.
- Entry fill: next candle open after the signal close.
- Exit fill: rules in the triple-barrier contract; no exchange orderbook or private API data.
- Cost scenarios are all-in round-trip cost of exactly 50 bps and 90 bps of notional.
- Net trade return = exit_price / entry_price - 1 - cost_bps / 10000.
- Do not subtract a separate entry/exit fee after applying the round-trip rate.
- No leverage, borrow, funding, maker rebate or unverified fill improvement is modeled.
- NetPF is sum(positive net trade returns) / abs(sum(negative net trade returns)); it is unavailable when there are no losses or no trades.
- MaxDD is computed on sequential compounded net trade returns from starting equity 1.
