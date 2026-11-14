# Four-Arm Model Provenance and Corpus-Overlap Disclosure

**Date:** 2026-11-14  
**Scope:** Public model metadata for the preregistered pre-hack research only.

## Kronos-Small and tokenizer

- Model: `NeoQuasar/Kronos-small`, declared license: MIT.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`, declared license: MIT.
- The published model card describes Kronos as trained on more than 12 billion K-line records from 45 global exchanges.
- The accessible public model metadata does not provide a complete per-exchange, per-symbol, per-date training-corpus ledger sufficient to prove whether Nobitex BTC/USDT rows or derived records overlap the AliX pre-hack period.
- Corpus overlap with this experiment is therefore **UNKNOWN / NOT EXCLUDED**. Kronos model forecasts are not independent evidence with respect to every possible pretraining exposure.

## Chronos-2-Small

- Model: `autogluon/chronos-2-small`, declared license: Apache-2.0.
- It is the 28M-parameter small variant of Chronos-2.
- The public model card describes training across real-world time series and synthetic datasets, but the accessible metadata does not establish a complete symbol/date inventory for every training source.
- Exact overlap with Nobitex BTC/USDT pre-hack data is therefore **UNKNOWN / NOT EXCLUDED**.

## Interpretation

1. These disclosures identify the publicly declared licenses and available corpus description. They are not legal advice.
2. No claim is made that either model is free of pretraining overlap with the target instrument or period.
3. The four-arm results must disclose both corpus-overlap statuses as UNKNOWN. They must not be called independent of model pretraining.
4. The final 12.5% of pre-hack data has separate prior AI-6 / DEAP exposure. It remains non-independent evidence even if model-pretraining overlap were later clarified.
