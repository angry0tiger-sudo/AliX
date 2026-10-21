"""Tunnel route descriptor; this module never opens SSH or tunnel processes."""

from __future__ import annotations

from connectivity.routes.base import ActivationCallback, BaseRouteDriver, RouteKind


class TunnelRouteDriver(BaseRouteDriver):
    """Tunnel activation adapter backed by an injected, owner-controlled callback."""

    kind = RouteKind.TUNNEL

    def __init__(
        self,
        route_id: str,
        *,
        activate: ActivationCallback,
        priority: int = 30,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            route_id, activate=activate, priority=priority, enabled=enabled
        )
