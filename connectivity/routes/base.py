"""Shared network-route contract.

Route drivers deliberately delegate activation to an injected callback. They do
not launch VPN clients, alter system routes, or create tunnels on their own.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum


class RouteKind(StrEnum):
    VPN = "vpn"
    PROXY = "proxy"
    TUNNEL = "tunnel"


ActivationCallback = Callable[[], bool]


class BaseRouteDriver:
    """A configured route with a side-effect injected by its composition root."""

    kind: RouteKind

    def __init__(
        self,
        route_id: str,
        *,
        activate: ActivationCallback,
        priority: int = 100,
        enabled: bool = True,
    ) -> None:
        normalized_id = route_id.strip()
        if not normalized_id:
            raise ValueError("route_id must not be empty")
        if priority < 0:
            raise ValueError("route priority must be non-negative")
        if not callable(activate):
            raise TypeError("route activation callback must be callable")
        self.route_id = normalized_id
        self.activate_callback = activate
        self.priority = priority
        self.enabled = enabled

    def activate(self) -> bool:
        """Activate this route fail-closed; exceptions are treated as failure."""
        try:
            return bool(self.activate_callback())
        except Exception:
            return False
