# Oracle Teacher + LSTM Research

This workflow is a research-only experiment that labels public hourly OHLCV data with a
Triple-Barrier oracle and trains a new CPU LSTM to predict those labels.

## Label definition

For the completed candle at index t, the reference entry is the next candle's open.
The oracle scans the next 24 candles, from t+1 through t+24:
- BUY: the +3% take-profit barrier is touched before the downside barrier.
- SELL: the -1.5% stop-loss barrier is touched before the upside barrier.
- HOLD: neither barrier is touched before the horizon expires.

With OHLC candles, the order of two barriers touched inside the same candle is unknown.
The oracle conservatively assigns SELL (stop-loss first) in that case. Labels whose
24-candle window would cross the train/validation/holdout boundary are excluded.

## Features and training protocol

Model features are causal indicators derived only from completed OHLCV candles up to the
feature timestamp: returns, candle shape, RSI, ATR, Bollinger distance, moving-average
distance, VWAP distance, and log-volume z-score. The forward-looking barrier scan is used
only for target labels and is never joined into the feature matrix. The scaler is fit on
the chronological training partition only.

The time split is 70% train, 15% validation, and 15% final holdout. Validation Macro-F1
selects the checkpoint; holdout is evaluated after training and does not influence class
weights, scaling, epoch selection, or model choices. BUY/SELL class weights are calculated
from training labels only.

Economic evaluation is a separate offline Spot long/flat simulation: a prediction at candle
close is filled at the next candle open; SELL closes an open long and never opens a short.
Fees (25 bps per side) and slippage (20 bps per side) apply on entry and exit, including
forced final liquidation. These metrics are not live-trading evidence.

## Hard safety contract

The workflow does not access private APIs or send orders. Champion remains null, order
permission remains false, and trading authorization remains NO-GO regardless of metrics.
