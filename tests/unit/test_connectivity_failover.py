from __future__ import annotations

from collections.abc import Callable

import pytest

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
    assert result.route_statuses[0].reason == "probe_failed"
    assert result.route_statuses[1].reason == "activation_failed"
    assert result.route_statuses[-1].reason == "route_selected"


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
    assert result.route_statuses[0].reason == "route_not_allowlisted"


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


def test_route_driver_rejects_empty_id_negative_priority_and_non_callable() -> None:
    with pytest.raises(ValueError, match="route_id"):
        VpnRouteDriver(" ", activate=lambda: True)
    with pytest.raises(ValueError, match="priority"):
        ProxyRouteDriver("proxy", activate=lambda: True, priority=-1)
    with pytest.raises(TypeError, match="callback"):
        TunnelRouteDriver("tunnel", activate=None)  # type: ignore[arg-type]


def test_connectivity_manager_exposes_safe_latency_and_status_snapshot() -> None:
    manager = ConnectivityManager({"proxy-main"}, lambda _route_id: True)

    status = manager.check("proxy-main")
    snapshot = manager.status_snapshot()

    assert status.healthy
    assert status.reason == "ok"
    assert status.latency_ms is not None and status.latency_ms >= 0
    assert snapshot == (status,)


def test_connectivity_manager_rejects_duplicate_route_ids_without_activation() -> None:
    activated: list[str] = []
    probe_calls: list[str] = []

    def probe(route_id: str) -> bool:
        probe_calls.append(route_id)
        return True

    manager = ConnectivityManager({"same"}, probe)
    def activate_vpn() -> bool:
        activated.append("vpn")
        return True

    def activate_proxy() -> bool:
        activated.append("proxy")
        return True

    routes = [
        VpnRouteDriver("same", activate=activate_vpn),
        ProxyRouteDriver("same", activate=activate_proxy),
    ]

    result = manager.select_route(routes)

    assert not result.healthy
    assert result.reason == "duplicate_route_id"
    assert result.route_statuses[0].reason == "duplicate_route_id"
    assert probe_calls == []
    assert activated == []
    assert manager.current_route is None


def test_connectivity_manager_normalizes_allowlist_and_redacts_probe_errors() -> None:
    def broken_probe(_route_id: str) -> bool:
        raise RuntimeError("secret-token-must-not-appear")

    manager = ConnectivityManager({" approved "}, broken_probe)

    status = manager.check("approved")
    denied = manager.check("unknown")

    assert status.reason == "probe_failed"
    assert "secret-token" not in repr(status)
    assert denied.reason == "route_not_allowlisted"
    assert manager.status_snapshot() == (
        status,
        denied,
    ) or {item.route for item in manager.status_snapshot()} == {
        "approved",
        "unknown",
    }


def test_connectivity_manager_rejects_blank_allowlist_and_invalid_probe() -> None:
    with pytest.raises(ValueError, match="allowed route IDs"):
        ConnectivityManager({" "}, lambda _: True)
    with pytest.raises(TypeError, match="probe"):
        ConnectivityManager({"route"}, None)  # type: ignore[arg-type]
