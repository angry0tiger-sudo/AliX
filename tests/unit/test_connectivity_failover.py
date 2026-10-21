from __future__ import annotations

from collections.abc import Callable

from connectivity.manager.manager import ConnectivityManager
from connectivity.routes import ProxyRouteDriver, TunnelRouteDriver, VpnRouteDriver


def test_failover_selects_first_healthy_route_by_priority() -> None:
    activated: list[str] = []

    def activate(route_id: str) -> Callable[[], bool]:
        def callback() -> bool:
            activated.append(route_id)
            return True

        return callback

    manager = ConnectivityManager(
        {"vpn-primary", "proxy-secondary", "tunnel-tertiary"},
        lambda route_id: route_id != "vpn-primary",
    )
    result = manager.select_route(
        [
            TunnelRouteDriver("tunnel-tertiary", activate=activate("tunnel-tertiary")),
            VpnRouteDriver("vpn-primary", activate=activate("vpn-primary")),
            ProxyRouteDriver("proxy-secondary", activate=activate("proxy-secondary")),
        ]
    )

    assert result.healthy
    assert result.active_route == "proxy-secondary"
    assert result.attempted_routes == ("vpn-primary", "proxy-secondary")
    assert activated == ["proxy-secondary"]
    assert manager.current_route == "proxy-secondary"


def test_failover_tries_tunnel_after_probe_exception_and_activation_failure() -> None:
    activated: list[str] = []

    def activation(name: str, outcome: bool) -> Callable[[], bool]:
        def callback() -> bool:
            activated.append(name)
            return outcome

        return callback

    def probe(route_id: str) -> bool:
        if route_id == "vpn-primary":
            raise RuntimeError("mock probe failure")
        return True

    manager = ConnectivityManager(
        {"vpn-primary", "proxy-secondary", "tunnel-tertiary"}, probe
    )
    result = manager.select_route(
        [
            VpnRouteDriver("vpn-primary", activate=activation("vpn-primary", True)),
            ProxyRouteDriver(
                "proxy-secondary", activate=activation("proxy-secondary", False)
            ),
            TunnelRouteDriver(
                "tunnel-tertiary", activate=activation("tunnel-tertiary", True)
            ),
        ]
    )

    assert result.healthy
    assert result.active_route == "tunnel-tertiary"
    assert activated == ["proxy-secondary", "tunnel-tertiary"]


def test_failover_does_not_probe_unallowlisted_routes() -> None:
    probed: list[str] = []
    route = VpnRouteDriver("unlisted", activate=lambda: True)

    def probe(route_id: str) -> bool:
        probed.append(route_id)
        return True

    manager = ConnectivityManager({"approved"}, probe)

    result = manager.select_route([route])

    assert not result.healthy
    assert result.active_route is None
    assert result.attempted_routes == ("unlisted",)
    assert probed == []
    assert manager.current_route is None


def test_failover_skips_disabled_routes_and_fails_closed_if_none_work() -> None:
    probed: list[str] = []

    def probe(route_id: str) -> bool:
        probed.append(route_id)
        return False

    manager = ConnectivityManager({"vpn-disabled", "proxy-broken"}, probe)
    result = manager.select_route(
        [
            VpnRouteDriver("vpn-disabled", activate=lambda: True, enabled=False),
            ProxyRouteDriver("proxy-broken", activate=lambda: True),
        ]
    )

    assert not result.healthy
    assert result.active_route is None
    assert result.attempted_routes == ("proxy-broken",)
    assert probed == ["proxy-broken"]
    assert manager.current_route is None


def test_route_driver_activation_exception_is_fail_closed() -> None:
    route = TunnelRouteDriver(
        "tunnel", activate=lambda: (_ for _ in ()).throw(RuntimeError("mock"))
    )

    assert not route.activate()
