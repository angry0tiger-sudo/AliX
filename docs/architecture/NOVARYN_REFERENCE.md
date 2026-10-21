# Novaryn Reference Contract for AliX Phase 1

## Source pin

- Repository: https://github.com/angry0tiger-sudo/Novaryn
- Inspected branch: main
- Reference commit: 54e66ad40b5a7e422002619f7dad9378ecdf09f9
- The AliX implementation applies selected contracts and deterministic rules; it is not a wholesale copy of Novaryn source files.
- Novaryn is private; source access for this review was available through the connected GitHub integration.

## Reviewed source contracts

| Novaryn path | Blob SHA | Contract applied in AliX |
|---|---|---|
| docs/NOBITEX.md | 0938e663df682450c9ca62d1d49f62a215065d0e | Credentials stay outside source; public market data is read-only during validation; unknown order outcome requires reconciliation |
| docs/FREQTRADE.md | 4930b0d8dbcb9e9f337c208df084b487db1ede46 | Freqtrade interface v3; framework lifecycle stays separate from risk/business policy; baseline Freqtrade 2026.7 |
| docs/architecture/NOBITEX_FREQTRADE_INTEGRATION_DECISION.yaml | ce50fbd965e5a629022940a6582328f837830ab6 | No Freqtrade core or site-packages patching; explicit boundary; writes blocked during validation |
| src/novaryn/infrastructure/exchange/nobitex/adapter.py | b15284e366703811a9357faae16b766472b12716 | Spot/USDT constraint, base-quantity validation, client order ID bound, conservative result-state handling |
| src/novaryn/infrastructure/exchange/nobitex/market_mapping.py | 4174fe31bae45c802af553b5f1cf59a616115b44 | Reject incomplete/derivative market metadata; normalize currencies, precision and limits |
| src/novaryn/infrastructure/exchange/nobitex/status.py | 8a10c9d9add27e975a34ec8ee7793ba54bc2ca75 | Explicit known order-state map; unknown state blocks normal completion |
| src/novaryn/infrastructure/exchange/nobitex/public.py | 1642060ca8d412d18cb2113538d989c522ab239d | Public transport is a separate read-only boundary with request-policy controls |
| src/novaryn/infrastructure/exchange/nobitex/private.py | b658bb66ae9c8a0030e4cc7edc11110dbb9b0795 | Private transport requires runtime key material and validates credentials before request |
| src/novaryn/infrastructure/freqtrade/nobitex_exchange.py | 1aea026ea2096614340805706604bddce25de56d | Freqtrade-specific candle and callback behavior belongs in an explicit integration layer |
| src/novaryn/infrastructure/freqtrade/adapter.py | 56abac175d92b9eaa7343fbd2590715152491a57 | Framework callbacks delegate to registered canonical gates and fail closed when missing |

## AliX Phase 1 mapping

- integrations/nobitex/adapter.py: pure validation/mapping rules, injected read-only transport interface, payload construction, status map, and unconditional rejection of order writes.
- connectivity/routes/: route descriptors for VPN, proxy and tunnel; activation is injected, and CI does not launch external processes or change routes.
- connectivity/manager/manager.py: allowlist check, route priority and failover on unhealthy/unactivatable routes.
- services/notifications/bale.py: separate Bale adapter using MIT-licensed BaleIO pinned to immutable commit 66be3000af6e62332e1764f227aed64847a0d9ba.

## Scope and evidence boundaries

- CI never contacts Nobitex, launches VPN/tunnel processes, or sends real Bale messages.
- The Nobitex adapter has no default network transport and no write method on its transport protocol. Market stats fails closed unless a read-only transport is injected.
- Payload building is for validation only; submit_order always raises LiveExecutionDisabled.
- Bale tests use fake transports and a fake Bot factory. The owner-provided default chat ID is 832112407, but delivery remains disabled unless enabled and allowlisted.
- Novaryn's docs identify Freqtrade 2026.7 and explain that upstream CCXT lacks native Nobitex support. AliX therefore does not label the stock Freqtrade exchange as Nobitex or claim a production exchange resolver is already implemented.
- Novaryn source license/provenance is not inferred from repository access. AliX avoids copying whole source modules; confirm license before any future literal code reuse.
