# Oracle LSTM + Meta-Labeling Filter

This research workflow adds a RandomForest profitability filter on top of the Triple-Barrier LSTM.

## Temporal protocol

- 0–50%: fit the base LSTM and training-only feature scaler.
- 50–55%: select the base LSTM checkpoint using Macro-F1 only.
- 55–70%: train the meta-model using BUY predictions from a base LSTM that has never trained on these rows.
- 70–85%: select one meta probability threshold using validation economics.
- 85–100%: evaluate the frozen base model, meta-model and threshold once on the final holdout.

Every forward-label window is purged at the end of its assigned partition. Holdout does not influence fitting, scaling, or threshold selection.

## Features and target

For candidate LSTM BUY signals, meta features are the predicted LSTM BUY/SELL/HOLD probabilities, causal ATR fraction, and distances from 6/24/72/168-candle moving averages. The target is 1 when the 24-candle forward trade is profitable after the 3% take-profit, 1.5% stop-loss and round-trip costs; otherwise it is 0. Timeout labels use the close at the end of the horizon and include the same costs.

The classifier is a RandomForest with balanced class weighting. The filter only blocks new BUY entries. LSTM SELL signals remain available as exits and are never filtered; short positions are not allowed.

## Economics and safety

Threshold selection uses validation only. The end-to-end simulator fills at the next candle open, enforces take-profit/stop-loss and the 24-candle time barrier, marks equity to market, and applies 25 bps fees plus 20 bps slippage per side. Same-candle touches are handled stop-first.

All work occurs in GitHub Actions from public market data. The final decision always remains research-only: Champion is null, order permission is false, and trading authorization is NO-GO.
