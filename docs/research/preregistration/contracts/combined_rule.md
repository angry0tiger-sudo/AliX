# Arm D Forecast Combination Contract

- Inputs: the Kronos adapter's 24h percentage return and the Chronos-2 adapter's median 24h percentage return, both measured relative to the same decision-close price.
- Formula: `combined_return = (kronos_return + chronos_return) / 2`.
- Signal condition: `combined_return > 0`.
- If either forecast is missing, non-finite, or malformed, the Arm D signal is rejected for that refresh interval; no fallback to the other model is permitted.
- Equal weights are immutable within the current trial IDs.
