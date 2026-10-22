# Connectivity Test Matrix

All automated checks are mock-only. CI does not contact external services, expose
credentials, launch VPN/proxy/tunnel processes, or alter host networking.

| Case | Expected result | Evidence | Status |
|---|---|---|---|
| Route not allowlisted | Deny before health probe or activation | Unit test | Implemented |
| Probe returns unhealthy | Continue to next priority route | Unit test | Implemented |
| Probe raises | Record `probe_failed`; do not expose exception text; continue | Unit test | Implemented |
| Highest-priority route unhealthy | Continue in deterministic priority order | Unit test | Implemented |
| Activation callback returns false or raises | Record `activation_failed`; continue | Unit test | Implemented |
| VPN/proxy/tunnel all fail | Clear active route; fail closed | Unit test | Implemented |
| Disabled route | Skip without probe or activation | Unit test | Implemented |
| Duplicate route IDs in one candidate set | Fail closed before probing/activation | Unit test | Implemented |
| Status snapshot | Stable per-route latest status, reason code and probe latency | Unit test | Implemented |
| Blank allowlist entry or invalid probe callback | Reject configuration | Unit test | Implemented |
| VPN/proxy/tunnel process launch | No process launch in CI | Design constraint | Enforced by injected callbacks |
| Missing proxy/VPN credential | No implicit connection attempt | Authorized local integration test | Pending |
| DNS/TLS/egress/authentication against approved endpoint | Record pass/fail without storing secrets | Authorized local integration test | Pending |
| External timeout/restart and reconnect | Record outcome; order timeouts still require reconciliation | Local failure-injection test | Pending |
| Route switch while an order outcome is unknown | Block switch/retry until reconciliation | Integration gate | Pending |

The manager reports observed callback results only. A successful mock probe is
not evidence that the route can reach Nobitex or that the exchange egress IP is
allowlisted. Never run real route activation from GitHub Actions.
