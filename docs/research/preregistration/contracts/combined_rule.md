# Arm B/C/D Forecast Threshold Contract

- Arm A is a Donchian-20 close-breakout control and is invariant to forecast threshold.
- Arm B enters independently when the pinned Kronos-Small 24h percentage return is strictly greater than the candidate threshold.
- Arm C enters independently when the pinned Chronos-2 median 24h percentage return is strictly greater than the candidate threshold.
- Arm D enters independently when the arithmetic mean of the two 24h percentage returns is strictly greater than the candidate threshold.
- The only candidate thresholds are 120, 180 and 240 bps, each a separate registered trial cell for each arm.
- B/C/D are forecast-driven entry rules; no Donchian predicate is ANDed with any forecast signal.
- Forecasts are measured relative to the same decision-close price and held for the registered 24-bar refresh interval.
- Missing, non-finite or malformed forecasts invalidate the corresponding cache/run; no fallback is permitted.
- Arm A's identical Donchian signal is computed once and referenced by its three threshold cells. All three preregistered IDs remain visible in candidate accounting.
