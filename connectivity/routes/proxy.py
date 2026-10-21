"""Proxy route descriptor; proxy credentials are not stored by this module."""

from __future__ import annotations

from connectivity.routes.base import ActivationCallback, BaseRouteDriver, RouteKind


class ProxyRouteDriver(BaseRouteDriver):
    """Proxy activation adapter with injected validation/selection behavior."""

    kind = RouteKind.PROXY

    def __init__(
        self,
        route_id: str,
        *,
        activate: ActivationCallback,
        priority: int = 20,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            route_id, activate=activate, priority=priority, enabled=enabled
        )
