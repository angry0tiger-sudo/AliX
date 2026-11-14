# Triple-Barrier and Baseline Execution Contract

- Decision: after candle t closes; entries are at the next candle open.
- ATR: Wilder ATR(14), computed as-of decision close only.
- Initial stop-loss: entry minus 2*ATR(decision).
- Profit target: entry plus 2*ATR(decision).
- Trailing stop: after a completed bar without exit, ratchet stop to max(prior stop, highest completed high since entry minus 2*ATR(as-of that bar)); updated stop is active from the following candle.
- Maximum holding: 24 hourly candles after entry; exit at the close of the last eligible candle.
- If candle open gaps through stop, fill at open. If it gaps above target, fill at open.
- If both horizontal barriers are touched intrabar in a 1h candle, use stop-first. If only one is touched, fill at that barrier.
- One open long position per arm; no pyramiding, short selling or leverage.
- No future data is used to set the decision, ATR, initial stop or trailing stop.
