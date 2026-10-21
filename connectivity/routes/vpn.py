"""VPN route descriptor with no implicit process or network management."""

from __future__ import annotations

from connectivity.routes.base import ActivationCallback, BaseRouteDriver, RouteKind


class VpnRouteDriver(BaseRouteDriver):
    """VPN activation adapter; the caller must inject and own any real VPN client."""

    kind = RouteKind.VPN

    def __init__(
        self,
        route_id: str,
        *,
        activate: ActivationCallback,
        priority: int = 10,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            route_id, activate=activate, priority=priority, enabled=enabled
        )
