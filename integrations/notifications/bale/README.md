# Bale Notification Boundary

AliX uses an independent Bale notification service in `services/notifications/bale.py`, based on the MIT-licensed BaleIO client pinned to immutable commit `66be3000af6e62332e1764f227aed64847a0d9ba`.

- Default chat ID: `832112407`, based on the owner-provided value.
- Override: `BALE_CHAT_ID`.
- Bot token: `BALE_BOT_TOKEN`, empty in `.env.example`.
- Reporting: `ALIx_REPORTING_ENABLED=false` by default.
- Recipient policy: delivery is rejected unless the target chat is explicitly present in `ALIx_BALE_CHAT_ALLOWLIST`.
- CI: only injected mock transport and fake Bot factory are used. No message is sent to Bale.

The helper checks enablement, token presence, recipient allowlist and message length before calling the transport, and closes the client after the send attempt. A successful mock assertion demonstrates the service contract only; it is not evidence of a real Bale delivery. A real send remains a later authorized runtime validation gate.
