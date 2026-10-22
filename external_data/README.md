# External Data

Phase 6 adds versioned provider and record contracts, injected-transport adapters, separate in-memory stores and an available-at-only point-in-time join.

No concrete HTTP client or live provider is configured. CI and tests use synthetic fixtures and fake transports only. Provider activation requires provider-specific, current approval for the actual use case. A public API or model card does not independently establish commercial processing, retention, redistribution or historical replay rights.

Numeric observations pass the shared Phase 3 external-record Data Quality Gate. Raw text, numeric analytics and derived NLP features remain separate. Unmapped assets are not guessed. Only NLP features with approved provenance and an available_at no later than decision time can enter a historical feature join.

Do not scrape or redistribute data without documented permission. TradingView remains blocked for automated/non-display use until explicit permission is recorded. Fixture tests approve test fixtures only, not any live provider or model.
