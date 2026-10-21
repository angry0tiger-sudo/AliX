# Connectivity Test Matrix

Phase 1 checks are mock-only. CI does not contact external services or change host networking.

| Case | Expected result | Test | Status |
|---|---|---|---|
| Route not allowlisted | Deny without health probe | Unit test | Implemented |
| Health probe raises | Treat as unhealthy and try next route | Unit test | Implemented |
| Highest-priority route is unhealthy | Continue to next route | Unit test | Implemented |
| Activation callback fails | Continue to next route | Unit test | Implemented |
| VPN, proxy and tunnel all fail | No active route; fail closed | Unit test | Implemented |
| Disabled route | Skip without probe or activation | Unit test | Implemented |
| Route activation exception | Treat activation as failed | Unit test | Implemented |
| VPN/proxy/tunnel process launch | No process launch in CI | Design constraint | Enforced by injected callbacks |
| Missing proxy/VPN credential | No implicit connection attempt | Deployment integration test | Pending |
| External service timeout/restart | Record failure; avoid duplicate delivery | Integration mock required | Pending |
