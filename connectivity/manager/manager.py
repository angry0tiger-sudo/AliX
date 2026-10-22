"""Connectivity policy, health telemetry, and deterministic failover.

This boundary never launches processes or changes host networking. Route probes
and activation effects are provided by an owner-controlled composition root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import time

from connectivity.routes.base import BaseRouteDriver


@dataclass(frozen=True, slots=True)
class ConnectivityStatus:
    route: str
    healthy: bool
    reason: str
    latency_ms: float | None = None
    route_kind: str | None = None
    activation_attempted: bool = False


@dataclass(frozen=True, slots=True)
class FailoverResult:
    healthy: bool
    active_route: str | None
    attempted_routes: tuple[str, ...]
    reason: str
    route_statuses: tuple[ConnectivityStatus, ...] = ()


class ConnectivityManager:
    """Select the first allowlisted healthy route in deterministic priority order.

    Health snapshots contain route IDs, outcome codes and timing only. They never
    contain proxy URLs, endpoint credentials, tokens, or raw exception strings.
    """

    def __init__(
        self,
        allowed_routes: set[str],
        probe: Callable[[str], bool],
    ) -> None:
        normalized = {route.strip() for route in allowed_routes}
        if "" in normalized:
            raise ValueError("allowed route IDs must not be empty")
        if not callable(probe):
            raise TypeError("route health probe must be callable")
        self._allowed_routes = frozenset(normalized)
        self._probe = probe
        self.current_route: str | None = None
        self._latest_statuses: dict[str, ConnectivityStatus] = {}

    def _record(self, status: ConnectivityStatus) -> ConnectivityStatus:
        self._latest_statuses[status.route] = status
        return status

    def status_snapshot(self) -> tuple[ConnectivityStatus, ...]:
        """Return the most recent known status per route in stable route-ID order."""
        return tuple(
            self._latest_statuses[key] for key in sorted(self._latest_statuses)
        )

    def check(self, route: str) -> ConnectivityStatus:
        route_id = route.strip()
        if not route_id or route_id not in self._allowed_routes:
            return self._record(
                ConnectivityStatus(route_id, False, "route_not_allowlisted")
            )
        started = time.perf_counter()
        try:
            healthy = bool(self._probe(route_id))
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._record(
                ConnectivityStatus(route_id, False, "probe_failed", elapsed_ms)
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self._record(
            ConnectivityStatus(
                route_id,
                healthy,
                "ok" if healthy else "route_unhealthy",
                elapsed_ms,
            )
        )

    def may_proceed(self, route: str) -> bool:
        """Allow the caller to proceed only after an allowlisted route probes healthy."""
        return self.check(route).healthy

    @staticmethod
    def _enabled_routes(routes: Iterable[BaseRouteDriver]) -> list[BaseRouteDriver]:
        return sorted(
            (route for route in routes if route.enabled),
            key=lambda route: (route.priority, route.route_id),
        )

    def _reject_duplicate_ids(
        self, candidates: list[BaseRouteDriver]
    ) -> FailoverResult | None:
        route_ids = [route.route_id for route in candidates]
        duplicates = sorted(
            {route_id for route_id in route_ids if route_ids.count(route_id) > 1}
        )
        if not duplicates:
            return None
        statuses = tuple(
            self._record(ConnectivityStatus(route_id, False, "duplicate_route_id"))
            for route_id in duplicates
        )
        self.current_route = None
        return FailoverResult(
            False, None, tuple(route_ids), "duplicate_route_id", statuses
        )

    def select_route(self, routes: Iterable[BaseRouteDriver]) -> FailoverResult:
        """Probe and activate routes in priority order, failing closed if none work."""
        candidates = self._enabled_routes(routes)
        duplicate_result = self._reject_duplicate_ids(candidates)
        if duplicate_result is not None:
            return duplicate_result

        attempted: list[str] = []
        results: list[ConnectivityStatus] = []
        for route in candidates:
            attempted.append(route.route_id)
            health = self.check(route.route_id)
            if not health.healthy:
                results.append(
                    ConnectivityStatus(
                        health.route,
                        False,
                        health.reason,
                        health.latency_ms,
                        route.kind.value,
                        False,
                    )
                )
                self._record(results[-1])
                continue

            activated = route.activate()
            status = ConnectivityStatus(
                route.route_id,
                activated,
                "route_selected" if activated else "activation_failed",
                health.latency_ms,
                route.kind.value,
                True,
            )
            results.append(self._record(status))
            if activated:
                self.current_route = route.route_id
                return FailoverResult(
                    True,
                    self.current_route,
                    tuple(attempted),
                    "route_selected",
                    tuple(results),
                )

        self.current_route = None
        return FailoverResult(
            False,
            None,
            tuple(attempted),
            "no_healthy_allowlisted_route",
            tuple(results),
        )
