"""Connectivity policy boundary with deterministic mockable route failover."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from connectivity.routes.base import BaseRouteDriver


@dataclass(frozen=True)
class ConnectivityStatus:
    route: str
    healthy: bool
    reason: str


@dataclass(frozen=True)
class FailoverResult:
    healthy: bool
    active_route: str | None
    attempted_routes: tuple[str, ...]
    reason: str


class ConnectivityManager:
    """Select the first allowlisted healthy route in priority order.

    The probe is a read-only health observation. Driver activation is injected;
    this manager never launches VPN, proxy or tunnel processes itself.
    """

    def __init__(
        self, allowed_routes: set[str], probe: Callable[[str], bool]
    ) -> None:
        self._allowed_routes = frozenset(route.strip() for route in allowed_routes)
        self._probe = probe
        self.current_route: str | None = None

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

    def select_route(
        self, routes: Iterable[BaseRouteDriver]
    ) -> FailoverResult:
        """Probe and activate routes in priority order, failing closed if none work."""
        candidates = sorted(
            (route for route in routes if route.enabled),
            key=lambda route: (route.priority, route.route_id),
        )
        attempted: list[str] = []
        for route in candidates:
            attempted.append(route.route_id)
            status = self.check(route.route_id)
            if not status.healthy:
                continue
            if route.activate():
                self.current_route = route.route_id
                return FailoverResult(
                    True, self.current_route, tuple(attempted), "route_selected"
                )
        self.current_route = None
        return FailoverResult(
            False, None, tuple(attempted), "no_healthy_allowlisted_route"
        )
