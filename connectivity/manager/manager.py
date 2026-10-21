"""Connectivity policy boundary. Network probes are injected; no real connection is made here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConnectivityStatus:
    route: str
    healthy: bool
    reason: str


class ConnectivityManager:
    def __init__(self, allowed_routes: set[str], probe: Callable[[str], bool]) -> None:
        self._allowed_routes = frozenset(allowed_routes)
        self._probe = probe

    def check(self, route: str) -> ConnectivityStatus:
        if route not in self._allowed_routes:
            return ConnectivityStatus(route, False, "route_not_allowlisted")
        try:
            healthy = bool(self._probe(route))
        except Exception:
            return ConnectivityStatus(route, False, "probe_failed")
        return ConnectivityStatus(
            route, healthy, "ok" if healthy else "route_unhealthy"
        )

    def may_proceed(self, route: str) -> bool:
        return self.check(route).healthy
