from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from control_plane.auth import (
    AuthenticationError,
    Principal,
    SingleUserTokenAuthenticator,
)
from control_plane import runtime_cli as runtime_cli_module
from control_plane.api import create_app
from control_plane import server as control_server
from control_plane.runtime_cli import (
    _api_request,
    _container_info,
    _container_name,
    _docker,
    _local_control_base_url,
    _start_worker,
    dispatch_once,
    load_env_file,
    process_command,
)
from control_plane.store import (
    ConflictError,
    ControlPlaneStore,
    ForbiddenError,
    NotFoundError,
)
from control_plane.worker_runtime import (
    WorkerAuthorizationFailure,
    WorkerRuntimeFailure,
    _validated_worker_control_url,
    run_worker,
)


class TokenMapAuthenticator:
    def __init__(self) -> None:
        self.tokens = {
            "admin-token-012345678901234567890123456789": Principal(
                "local-owner", platform_admin=True
            ),
            "owner-a-token-012345678901234567890123456789": Principal("owner-a"),
            "owner-b-token-012345678901234567890123456789": Principal("owner-b"),
            "reader-a-token-012345678901234567890123456789": Principal("reader-a"),
        }

    def authenticate(self, authorization_header: str | None) -> Principal:
        if authorization_header is None:
            raise AuthenticationError("bearer_token_required")
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("bearer_token_required")
        principal = self.tokens.get(parts[1])
        if principal is None:
            raise AuthenticationError("invalid_bearer_token")
        return principal


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        f"sqlite:///{tmp_path / 'control-plane.sqlite'}", TokenMapAuthenticator()
    )
    return TestClient(app)


def auth(token: str = "admin-token-012345678901234567890123456789") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_tenant(client: TestClient, name: str) -> str:
    response = client.post("/v1/tenants", headers=auth(), json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["tenant_id"])


def add_member(client: TestClient, tenant_id: str, subject: str, role: str) -> None:
    response = client.post(
        f"/v1/tenants/{tenant_id}/members",
        headers=auth(),
        json={"subject": subject, "role": role},
    )
    assert response.status_code == 201, response.text


def make_account(
    client: TestClient,
    tenant_id: str,
    token: str = "admin-token-012345678901234567890123456789",
    exchange_id: str = "nobitex",
    environment: str = "local",
) -> dict[str, Any]:
    response = client.post(
        f"/v1/tenants/{tenant_id}/accounts",
        headers=auth(token),
        json={
            "exchange_id": exchange_id,
            "environment": environment,
            "account_alias": f"{exchange_id}-synthetic",
            "secret_ref_id": None,
        },
    )
    assert response.status_code == 201, response.text
    account = cast(dict[str, Any], response.json())
    enabled = client.post(
        f"/v1/tenants/{tenant_id}/accounts/{account['account_id']}/enable",
        headers=auth(token),
    )
    assert enabled.status_code == 200, enabled.text
    return cast(dict[str, Any], enabled.json())


def make_worker(
    client: TestClient,
    tenant_id: str,
    account_id: str,
    token: str = "admin-token-012345678901234567890123456789",
    environment: str = "local",
) -> dict[str, Any]:
    response = client.post(
        f"/v1/tenants/{tenant_id}/workers",
        headers=auth(token),
        json={
            "account_id": account_id,
            "exchange_id": "nobitex",
            "environment": environment,
            "strategy_revision": "fixture-revision-001",
            "profile_id": "freqtrade-isolated",
        },
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def test_local_single_user_token_fails_closed() -> None:
    with pytest.raises(ValueError):
        SingleUserTokenAuthenticator("")
    with pytest.raises(ValueError):
        SingleUserTokenAuthenticator("short")
    authenticator = SingleUserTokenAuthenticator("A" * 40)
    with pytest.raises(AuthenticationError):
        authenticator.authenticate(None)
    with pytest.raises(AuthenticationError):
        authenticator.authenticate("Bearer " + "B" * 40)
    assert authenticator.authenticate("Bearer " + "A" * 40).platform_admin


def test_health_is_public_but_management_requires_auth(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/v1/tenants").status_code == 401
    assert client.get("/v1/tenants", headers=auth("wrong")).status_code == 401
    assert client.get("/v1/me", headers=auth()).json()["platform_admin"] is True


def test_tenant_creation_adds_owner_and_audit_event(client: TestClient) -> None:
    tenant_id = make_tenant(client, "Owner Tenant")
    listed = client.get("/v1/tenants", headers=auth()).json()
    assert any(item["tenant_id"] == tenant_id for item in listed)
    audit = client.get(f"/v1/tenants/{tenant_id}/audit", headers=auth()).json()
    assert any(item["action"] == "tenant.created" for item in audit)


def test_tenant_membership_is_revoked_immediately(client: TestClient) -> None:
    tenant_id = make_tenant(client, "Membership Tenant")
    add_member(client, tenant_id, "owner-a", "tenant_owner")
    add_member(client, tenant_id, "reader-a", "tenant_reader")
    assert (
        client.get(
            f"/v1/tenants/{tenant_id}",
            headers=auth("reader-a-token-012345678901234567890123456789"),
        ).status_code
        == 200
    )
    revoke = client.delete(
        f"/v1/tenants/{tenant_id}/members/reader-a",
        headers=auth("owner-a-token-012345678901234567890123456789"),
    )
    assert revoke.status_code == 204
    assert (
        client.get(
            f"/v1/tenants/{tenant_id}",
            headers=auth("reader-a-token-012345678901234567890123456789"),
        ).status_code
        == 403
    )


def test_member_cannot_list_or_read_other_tenant(client: TestClient) -> None:
    tenant_a = make_tenant(client, "Tenant A")
    tenant_b = make_tenant(client, "Tenant B")
    add_member(client, tenant_a, "owner-a", "tenant_owner")
    add_member(client, tenant_b, "owner-b", "tenant_owner")
    response = client.get(
        f"/v1/tenants/{tenant_b}",
        headers=auth("owner-a-token-012345678901234567890123456789"),
    )
    assert response.status_code == 403
    visible = client.get(
        "/v1/tenants", headers=auth("owner-a-token-012345678901234567890123456789")
    ).json()
    assert [item["tenant_id"] for item in visible] == [tenant_a]


def test_account_secret_is_reference_only_and_cross_tenant_binding_is_denied(
    client: TestClient,
) -> None:
    tenant_a = make_tenant(client, "Tenant A")
    tenant_b = make_tenant(client, "Tenant B")
    add_member(client, tenant_a, "owner-a", "tenant_owner")
    add_member(client, tenant_b, "owner-b", "tenant_owner")
    account_b = make_account(
        client, tenant_b, "owner-b-token-012345678901234567890123456789"
    )
    denied = client.post(
        f"/v1/tenants/{tenant_a}/workers",
        headers=auth("owner-a-token-012345678901234567890123456789"),
        json={
            "account_id": account_b["account_id"],
            "exchange_id": "nobitex",
            "environment": "local",
            "strategy_revision": "fixture-revision",
        },
    )
    assert denied.status_code == 404
    assert "api_key" not in account_b


def test_validation_error_does_not_echo_untrusted_secret(client: TestClient) -> None:
    tenant_id = make_tenant(client, "Validation Tenant")
    add_member(client, tenant_id, "owner-a", "tenant_owner")
    marker = "SYNTHETIC-SHOULD-NEVER-RETURN-SECRET"
    response = client.post(
        f"/v1/tenants/{tenant_id}/accounts",
        headers=auth("owner-a-token-012345678901234567890123456789"),
        json={
            "exchange_id": "nobitex",
            "environment": "local",
            "account_alias": "fixture",
            "api_key": marker,
        },
    )
    assert response.status_code == 422
    assert marker not in response.text


def test_account_binding_rejects_unapproved_environments(client: TestClient) -> None:
    tenant_id = make_tenant(client, "Environment Tenant")
    response = client.post(
        f"/v1/tenants/{tenant_id}/accounts",
        headers=auth(),
        json={
            "exchange_id": "nobitex",
            "environment": "live",
            "account_alias": "not-enabled",
        },
    )
    assert response.status_code == 422


def test_duplicate_worker_slot_is_rejected_until_previous_worker_is_stopped(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Worker Slot Tenant")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])
    conflict = client.post(
        f"/v1/tenants/{tenant_id}/workers",
        headers=auth(),
        json={
            "account_id": account["account_id"],
            "exchange_id": "nobitex",
            "environment": "local",
            "strategy_revision": "fixture-revision-002",
        },
    )
    assert conflict.status_code == 409
    stopped = client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "disable"},
    )
    assert stopped.status_code == 200
    command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-a"},
    ).json()[0]
    client.post(
        f"/v1/admin/runtime-commands/{command['command_id']}/ack",
        headers=auth(),
        json={"success": True, "container_name": None},
    )
    replacement = make_worker(client, tenant_id, account["account_id"])
    assert replacement["worker_id"] != worker["worker_id"]
    assert replacement["generation"] > worker["generation"]


def test_worker_start_claim_scoped_heartbeat_and_ack_are_idempotent(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Runtime Tenant")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])
    started = client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "start"},
    )
    assert started.status_code == 200
    assert started.json()["desired_state"] == "START_REQUESTED"
    claimed = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-a", "lease_seconds": 30},
    ).json()
    assert len(claimed) == 1
    assert (
        client.post(
            "/v1/admin/runtime-commands/claim",
            headers=auth(),
            json={"dispatcher_id": "dispatcher-b"},
        ).json()
        == []
    )
    command = claimed[0]
    token = "worker-scoped-token-value-with-enough-length-123"
    digest = hashlib.sha256(token.encode()).hexdigest()
    credential_response = client.put(
        f"/v1/admin/workers/{worker['worker_id']}/runtime-credential",
        headers=auth(),
        json={"generation": command["generation"], "credential_sha256": digest},
    )
    assert credential_response.status_code == 204
    invalid = client.post(
        f"/v1/internal/workers/{worker['worker_id']}/heartbeat",
        headers=auth("wrong"),
        json={"generation": command["generation"]},
    )
    assert invalid.status_code == 401
    heartbeat = client.post(
        f"/v1/internal/workers/{worker['worker_id']}/heartbeat",
        headers={"Authorization": f"Bearer {token}"},
        json={"generation": command["generation"]},
    )
    assert heartbeat.status_code == 204
    ack = client.post(
        f"/v1/admin/runtime-commands/{command['command_id']}/ack",
        headers=auth(),
        json={"success": True, "container_name": "alix-worker-test-g1"},
    )
    assert ack.status_code == 204
    result = client.get(f"/v1/tenants/{tenant_id}/workers", headers=auth()).json()[0]
    assert result["observed_state"] == "RUNNING"
    stale = client.post(
        f"/v1/internal/workers/{worker['worker_id']}/heartbeat",
        headers={"Authorization": f"Bearer {token}"},
        json={"generation": command["generation"] + 1},
    )
    assert stale.status_code == 401


def test_stop_revokes_worker_capability_and_audit_has_no_token(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Revoke Tenant")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])
    client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "start"},
    )
    start_command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-a"},
    ).json()[0]
    token = "worker-token-that-is-never-written-to-audit-record"
    digest = hashlib.sha256(token.encode()).hexdigest()
    client.put(
        f"/v1/admin/workers/{worker['worker_id']}/runtime-credential",
        headers=auth(),
        json={"generation": start_command["generation"], "credential_sha256": digest},
    )
    client.post(
        f"/v1/admin/runtime-commands/{start_command['command_id']}/ack",
        headers=auth(),
        json={"success": True, "container_name": "alix-worker-unit-g1"},
    )
    stopped = client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "stop"},
    )
    assert stopped.json()["desired_state"] == "STOP_REQUESTED"
    stop_command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-a"},
    ).json()[0]
    client.post(
        f"/v1/admin/runtime-commands/{stop_command['command_id']}/ack",
        headers=auth(),
        json={"success": True, "container_name": None},
    )
    assert (
        client.post(
            f"/v1/internal/workers/{worker['worker_id']}/heartbeat",
            headers={"Authorization": f"Bearer {token}"},
            json={"generation": start_command["generation"]},
        ).status_code
        == 401
    )
    audit_text = json.dumps(
        client.get(f"/v1/tenants/{tenant_id}/audit", headers=auth()).json()
    )
    assert token not in audit_text
    assert digest not in audit_text


def test_runtime_command_lease_can_be_reclaimed_after_expiry(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Lease Tenant")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])
    client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "start"},
    )
    store: ControlPlaneStore = cast(Any, client.app).state.store
    start = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    first = store.claim_runtime_commands("dispatcher-one", 1, 10, now=start)
    assert len(first) == 1
    second = store.claim_runtime_commands(
        "dispatcher-two", 1, 10, now=start + timedelta(seconds=11)
    )
    assert second[0]["command_id"] == first[0]["command_id"]
    assert second[0]["claimed_by"] == "dispatcher-two"
    with pytest.raises(ConflictError):
        store.acknowledge_runtime_command(
            first[0]["command_id"], "dispatcher-one", True, None, None
        )


def test_sqlite_worker_state_is_private_and_identity_bound(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []

    def sender(base_url: str, worker_id: str, token: str, generation: int) -> None:
        assert base_url == "http://control-plane:8080"
        calls.append((worker_id, token, generation))

    first_db = tmp_path / "worker-a" / "state.sqlite"
    second_db = tmp_path / "worker-b" / "state.sqlite"
    token_a = "A" * 40
    token_b = "B" * 40
    assert (
        run_worker(
            "worker-a",
            token_a,
            1,
            "http://control-plane:8080",
            str(first_db),
            sender,
            interval_seconds=0,
            iterations=2,
        )
        == 0
    )
    assert (
        run_worker(
            "worker-b",
            token_b,
            2,
            "http://control-plane:8080",
            str(second_db),
            sender,
            interval_seconds=0,
            iterations=1,
        )
        == 0
    )
    assert len(calls) == 3
    assert {item[0] for item in calls} == {"worker-a", "worker-b"}
    with sqlite3.connect(first_db) as connection:
        identity = connection.execute(
            "SELECT worker_id, generation FROM worker_identity"
        ).fetchone()
        heartbeats = connection.execute(
            "SELECT count(*) FROM worker_heartbeat"
        ).fetchone()[0]
    assert identity == ("worker-a", 1)
    assert heartbeats == 2
    with pytest.raises(WorkerRuntimeFailure):
        run_worker(
            "worker-a",
            token_a,
            2,
            "http://control-plane:8080",
            str(first_db),
            sender,
            iterations=1,
        )


def test_worker_fails_closed_on_revoked_identity_or_repeated_network_failure(
    tmp_path: Path,
) -> None:
    token = "worker-token-that-is-not-a-real-secret-012345"

    def rejected(
        base_url: str, worker_id: str, worker_token: str, generation: int
    ) -> None:
        raise WorkerAuthorizationFailure("revoked")

    assert (
        run_worker(
            "worker-rejected",
            token,
            1,
            "http://control-plane:8080",
            str(tmp_path / "rejected.sqlite"),
            rejected,
            interval_seconds=0,
            iterations=3,
        )
        == 2
    )

    attempts = 0

    def unavailable(
        base_url: str, worker_id: str, worker_token: str, generation: int
    ) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("synthetic_network_failure")

    assert (
        run_worker(
            "worker-offline",
            token,
            1,
            "http://control-plane:8080",
            str(tmp_path / "offline.sqlite"),
            unavailable,
            interval_seconds=0,
            iterations=10,
        )
        == 3
    )
    assert attempts == 3
    with pytest.raises(ValueError):
        run_worker(
            "worker-invalid",
            token,
            1,
            "http://untrusted.invalid:8080",
            str(tmp_path / "invalid.sqlite"),
            interval_seconds=0,
            iterations=1,
        )


def test_runtime_cli_uses_fixed_worker_service_and_scoped_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr("control_plane.runtime_cli._container_info", lambda name: None)
    monkeypatch.setattr(
        "control_plane.runtime_cli.secrets.token_urlsafe", lambda size: "S" * 48
    )

    def api_request(
        base_url: str,
        admin_token: str,
        path: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> None:
        del base_url, admin_token, timeout
        calls.append((path, payload))
        return None

    def docker(
        args: list[str],
        timeout: float = 30.0,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        calls.append(("docker-env", environment))
        calls.append(("docker", args))
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")

    monkeypatch.setattr("control_plane.runtime_cli._api_request", api_request)
    monkeypatch.setattr("control_plane.runtime_cli._docker", docker)
    name = _start_worker(
        {"worker_id": "worker-abc123", "generation": 4},
        "http://127.0.0.1:8080",
        "admin-token-not-real",
        tmp_path / "compose.yaml",
        "alix",
    )
    assert name == "alix-worker-workerabc123-g4"
    registration = calls[0][1]
    assert registration["generation"] == 4
    assert (
        registration["credential_sha256"]
        == hashlib.sha256(("S" * 48).encode()).hexdigest()
    )
    docker_environment = next(value for kind, value in calls if kind == "docker-env")
    docker_args = next(value for kind, value in calls if kind == "docker")
    assert docker_args[-1] == "worker"
    assert "control-plane" not in " ".join(docker_args)
    assert "ALIx_WORKER_TOKEN=" not in " ".join(docker_args)
    assert docker_environment["ALIx_WORKER_TOKEN"] == "S" * 48


def test_runtime_cli_container_name_is_deterministic_and_safe() -> None:
    assert _container_name("UUID-worker-123", 2) == "alix-worker-uuidworker12-g2"
    with pytest.raises(ValueError):
        _container_name("!!!", 1)


def test_runtime_cli_env_file_never_overwrites_existing_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# synthetic fixture\nALIx_ADMIN_TOKEN=local-test-token\nPOSTGRES_PASSWORD='test-db-password'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALIx_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "already-set")
    load_env_file(env_file)
    assert os.environ["ALIx_ADMIN_TOKEN"] == "local-test-token"
    assert os.environ["POSTGRES_PASSWORD"] == "already-set"


def test_worker_profile_and_compose_security_contracts() -> None:
    yaml = cast(Any, __import__("yaml"))

    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    assert compose["services"]["postgres"]["networks"] == ["control-backend"]
    api_service = compose["services"]["control-plane"]
    worker_service = compose["services"]["worker"]
    assert api_service["ports"] == ["127.0.0.1:8080:8080"]
    assert api_service["read_only"] is True
    assert worker_service["read_only"] is True
    assert worker_service["user"] == "10001:10001"
    assert worker_service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in worker_service["security_opt"]
    assert "worker-control" in worker_service["networks"]
    assert "control-backend" not in worker_service["networks"]
    assert worker_service["profiles"] == ["worker"]
    assert "volumes" not in worker_service


def test_control_plane_has_no_import_path_to_trade_execution() -> None:
    forbidden = {
        "risk",
        "integrations.nobitex",
        "user_data.strategies",
        "user_data.freqaimodels",
        "freqtrade",
        "ccxt",
    }
    for source in Path("control_plane").glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            assert not any(
                module == blocked or module.startswith(blocked + ".")
                for module in imported
                for blocked in forbidden
            ), f"execution dependency found in {source}"


def test_runtime_command_database_constraint_rejects_cross_tenant_worker(
    client: TestClient,
) -> None:
    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from control_plane.db import runtime_commands

    tenant_a = make_tenant(client, "Tenant A")
    tenant_b = make_tenant(client, "Tenant B")
    account_b = make_account(client, tenant_b)
    worker_b = make_worker(client, tenant_b, account_b["account_id"])
    store: ControlPlaneStore = cast(Any, client.app).state.store

    with pytest.raises(IntegrityError):
        with store.engine.begin() as connection:
            connection.execute(
                insert(runtime_commands).values(
                    command_id="forged-cross-tenant-command",
                    tenant_id=tenant_a,
                    worker_id=worker_b["worker_id"],
                    generation=1,
                    action="START",
                    status="PENDING",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    claimed_by=None,
                    lease_until=None,
                    completed_at=None,
                    error_code=None,
                )
            )


def test_server_fails_closed_without_token_and_rejects_oidc_until_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALIx_CONTROL_DATABASE_URL", "sqlite://")
    monkeypatch.delenv("ALIx_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ALIx_AUTH_MODE", "single_user")
    with pytest.raises(ValueError):
        control_server.build_application()

    monkeypatch.setenv("ALIx_ADMIN_TOKEN", "A" * 40)
    monkeypatch.setenv("ALIx_AUTH_MODE", "oidc")
    with pytest.raises(RuntimeError, match="unsupported_auth_mode"):
        control_server.build_application()


def test_server_main_uses_loopback_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "ALIx_CONTROL_DATABASE_URL", f"sqlite:///{tmp_path / 'server.sqlite'}"
    )
    monkeypatch.setenv("ALIx_ADMIN_TOKEN", "owner-test-token-" + "X" * 40)
    monkeypatch.setenv("ALIx_AUTH_MODE", "single_user")
    monkeypatch.delenv("ALIx_CONTROL_BIND_HOST", raising=False)
    monkeypatch.delenv("ALIx_CONTROL_PORT", raising=False)
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr("control_plane.server.uvicorn.run", fake_run)
    control_server.main()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8080


def test_local_image_uses_locked_dependencies_and_excludes_owner_secrets() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert "pip install --no-cache-dir -r requirements.lock" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert ".env" in dockerignore
    assert "secrets/" in dockerignore
    assert "!/.env.example" in dockerignore


def test_account_disable_revokes_worker_and_fences_late_start_ack(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Account Disable Tenant")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])
    started = client.post(
        f"/v1/tenants/{tenant_id}/workers/{worker['worker_id']}/lifecycle",
        headers=auth(),
        json={"action": "start"},
    )
    assert started.status_code == 200
    start_command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-start"},
    ).json()[0]
    worker_token = "synthetic-worker-token-for-account-disable-012345"
    worker_hash = hashlib.sha256(worker_token.encode()).hexdigest()
    assert (
        client.put(
            f"/v1/admin/workers/{worker['worker_id']}/runtime-credential",
            headers=auth(),
            json={
                "generation": start_command["generation"],
                "credential_sha256": worker_hash,
            },
        ).status_code
        == 204
    )

    reader = make_tenant(client, "Reader Tenant")
    add_member(client, reader, "reader-a", "tenant_reader")
    denied = client.post(
        f"/v1/tenants/{tenant_id}/accounts/{account['account_id']}/disable",
        headers=auth("reader-a-token-012345678901234567890123456789"),
    )
    assert denied.status_code == 403

    disabled = client.post(
        f"/v1/tenants/{tenant_id}/accounts/{account['account_id']}/disable",
        headers=auth(),
    )
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "DISABLED"
    current_worker = client.get(
        f"/v1/tenants/{tenant_id}/workers", headers=auth()
    ).json()[0]
    assert current_worker["desired_state"] == "DISABLED"
    assert current_worker["credential_sha256"] is None

    late_ack = client.post(
        f"/v1/admin/runtime-commands/{start_command['command_id']}/ack",
        headers=auth(),
        json={"success": True, "container_name": "late-start-container"},
    )
    assert late_ack.status_code == 204
    after_late_ack = client.get(
        f"/v1/tenants/{tenant_id}/workers", headers=auth()
    ).json()[0]
    assert after_late_ack["desired_state"] == "DISABLED"
    assert after_late_ack["container_name"] is None

    heartbeat = client.post(
        f"/v1/internal/workers/{worker['worker_id']}/heartbeat",
        headers={"Authorization": f"Bearer {worker_token}"},
        json={"generation": start_command["generation"]},
    )
    assert heartbeat.status_code == 401

    disable_command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "dispatcher-stop"},
    ).json()[0]
    assert disable_command["action"] == "DISABLE"
    assert (
        client.post(
            f"/v1/admin/runtime-commands/{disable_command['command_id']}/ack",
            headers=auth(),
            json={"success": True, "container_name": None},
        ).status_code
        == 204
    )
    final_worker = client.get(
        f"/v1/tenants/{tenant_id}/workers", headers=auth()
    ).json()[0]
    assert final_worker["observed_state"] == "DISABLED"
    replacement = client.post(
        f"/v1/tenants/{tenant_id}/workers",
        headers=auth(),
        json={
            "account_id": account["account_id"],
            "exchange_id": "nobitex",
            "environment": "local",
            "strategy_revision": "fixture-after-disable",
        },
    )
    assert replacement.status_code == 422


def test_tenant_disable_blocks_members_and_queues_worker_disable(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Tenant Disable")
    add_member(client, tenant_id, "owner-a", "tenant_owner")
    account = make_account(client, tenant_id)
    worker = make_worker(client, tenant_id, account["account_id"])

    forbidden = client.post(
        f"/v1/admin/tenants/{tenant_id}/disable",
        headers=auth("owner-a-token-012345678901234567890123456789"),
    )
    assert forbidden.status_code == 403
    assert (
        client.get(f"/v1/tenants/{tenant_id}", headers=auth()).json()["status"]
        == "ACTIVE"
    )

    disabled = client.post(f"/v1/admin/tenants/{tenant_id}/disable", headers=auth())
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "DISABLED"
    assert (
        client.get(
            f"/v1/tenants/{tenant_id}",
            headers=auth("owner-a-token-012345678901234567890123456789"),
        ).status_code
        == 403
    )

    disabled_worker = client.get(
        f"/v1/tenants/{tenant_id}/workers", headers=auth()
    ).json()[0]
    assert disabled_worker["desired_state"] == "DISABLED"

    command = client.post(
        "/v1/admin/runtime-commands/claim",
        headers=auth(),
        json={"dispatcher_id": "tenant-disable-dispatcher"},
    ).json()[0]
    assert command["action"] == "DISABLE"
    assert command["worker_id"] == worker["worker_id"]
    assert (
        client.post(
            f"/v1/admin/runtime-commands/{command['command_id']}/ack",
            headers=auth(),
            json={"success": True, "container_name": None},
        ).status_code
        == 204
    )

    final_worker = client.get(
        f"/v1/tenants/{tenant_id}/workers", headers=auth()
    ).json()[0]
    assert final_worker["observed_state"] == "DISABLED"
    assert (
        client.post(
            f"/v1/tenants/{tenant_id}/workers",
            headers=auth(),
            json={
                "account_id": account["account_id"],
                "exchange_id": "nobitex",
                "environment": "local",
                "strategy_revision": "must-not-create",
            },
        ).status_code
        == 403
    )


def test_http_transport_is_restricted_to_local_control_plane_hosts() -> None:
    assert _local_control_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert _local_control_base_url("http://localhost:8080") == "http://localhost:8080"
    for url in (
        "https://127.0.0.1:8080",
        "http://example.invalid:8080",
        "http://user:pass@127.0.0.1:8080",
        "http://127.0.0.1:bad",
    ):
        with pytest.raises(RuntimeError, match="control_plane_url_not_allowed"):
            _local_control_base_url(url)

    assert (
        _validated_worker_control_url("http://control-plane:8080")
        == "http://control-plane:8080"
    )
    assert (
        _validated_worker_control_url("http://127.0.0.1:8080")
        == "http://127.0.0.1:8080"
    )
    for url in (
        "https://control-plane:8080",
        "http://external.invalid:8080",
        "http://control-plane:8081",
        "http://user:pass@control-plane:8080",
    ):
        with pytest.raises(ValueError, match="worker_control_plane_url_not_allowed"):
            _validated_worker_control_url(url)


def test_runtime_http_request_success_and_sanitized_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self, status: int, body: bytes = b'{"ok":true}') -> None:
            self.status = status
            self.body = body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            del args

        def read(self) -> bytes:
            return self.body

    class FakeOpener:
        def __init__(self, response: FakeResponse | Exception) -> None:
            self.response = response
            self.request: Any = None
            self.timeout: float | None = None

        def open(self, request: Any, timeout: float) -> FakeResponse:
            self.request = request
            self.timeout = timeout
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    opener = FakeOpener(FakeResponse(200))
    monkeypatch.setattr(
        runtime_cli_module.urllib.request, "build_opener", lambda *args: opener
    )
    response = _api_request(
        "http://127.0.0.1:8080/",
        "synthetic-owner-token",
        "/v1/example",
        payload={"x": 1},
        timeout=2.5,
    )
    assert response == {"ok": True}
    assert opener.request.full_url == "http://127.0.0.1:8080/v1/example"
    assert opener.request.get_method() == "POST"
    assert opener.request.get_header("Authorization") == "Bearer synthetic-owner-token"
    assert json.loads(opener.request.data) == {"x": 1}
    assert opener.timeout == 2.5

    opener.response = FakeResponse(204, b"")
    assert (
        _api_request(
            "http://localhost:8080",
            "synthetic-owner-token",
            "/v1/no-content",
            method="PUT",
        )
        is None
    )

    opener.response = urllib.error.HTTPError(
        "http://127.0.0.1:8080/v1/example", 403, "synthetic", None, None
    )
    with pytest.raises(RuntimeError, match="control_plane_http_403"):
        _api_request("http://127.0.0.1:8080", "synthetic-owner-token", "/v1/example")

    opener.response = urllib.error.URLError("synthetic-unavailable")
    with pytest.raises(RuntimeError, match="control_plane_unavailable"):
        _api_request("http://127.0.0.1:8080", "synthetic-owner-token", "/v1/example")

    with pytest.raises(RuntimeError, match="control_plane_url_not_allowed"):
        _api_request("http://example.invalid:8080", "synthetic-owner-token", "/v1")


def test_docker_adapter_and_container_inspection_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("synthetic-docker-missing")

    monkeypatch.setattr(runtime_cli_module.subprocess, "run", raise_os_error)
    with pytest.raises(RuntimeError, match="docker_cli_unavailable_or_timeout"):
        _docker(["ps"])

    def raise_timeout(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired("docker", 1)

    monkeypatch.setattr(runtime_cli_module.subprocess, "run", raise_timeout)
    with pytest.raises(RuntimeError, match="docker_cli_unavailable_or_timeout"):
        _docker(["ps"])

    monkeypatch.setattr(
        runtime_cli_module,
        "_docker",
        lambda args, timeout=30.0, environment=None: subprocess.CompletedProcess(
            args, 1, "", "not-found"
        ),
    )
    assert _container_info("alix-worker-missing-g1") is None

    monkeypatch.setattr(
        runtime_cli_module,
        "_docker",
        lambda args, timeout=30.0, environment=None: subprocess.CompletedProcess(
            args, 0, "not-json", ""
        ),
    )
    with pytest.raises(RuntimeError, match="docker_inspect_invalid"):
        _container_info("alix-worker-invalid-g1")

    monkeypatch.setattr(
        runtime_cli_module,
        "_docker",
        lambda args, timeout=30.0, environment=None: subprocess.CompletedProcess(
            args, 0, "[]", ""
        ),
    )
    with pytest.raises(RuntimeError, match="docker_inspect_invalid"):
        _container_info("alix-worker-empty-g1")


def test_runtime_dispatcher_validates_existing_containers_and_stop_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    name = "alix-worker-workerabc123-g4"
    running_info = {
        "Config": {
            "Labels": {
                "alix.worker.id": "worker-abc123",
                "alix.worker.generation": "4",
            }
        },
        "State": {"Running": True},
    }
    monkeypatch.setattr(
        runtime_cli_module, "_container_info", lambda container_name: running_info
    )
    monkeypatch.setattr(
        runtime_cli_module,
        "_docker",
        lambda args, timeout=30.0, environment=None: (
            calls.append(("docker", args))
            or subprocess.CompletedProcess(args, 0, "stopped", "")
        ),
    )
    monkeypatch.setattr(
        runtime_cli_module,
        "_api_request",
        lambda *args, **kwargs: calls.append(("api", (args, kwargs))),
    )
    start_command = {"action": "START", "worker_id": "worker-abc123", "generation": 4}
    assert (
        _start_worker(
            start_command,
            "http://127.0.0.1:8080",
            "synthetic-owner-token",
            tmp_path / "compose.yaml",
            "alix",
        )
        == name
    )
    assert calls == []

    def wrong_identity(container_name: str) -> dict[str, Any]:
        del container_name
        return {
            "Config": {
                "Labels": {
                    "alix.worker.id": "other-worker",
                    "alix.worker.generation": "4",
                }
            },
            "State": {"Running": True},
        }

    monkeypatch.setattr(runtime_cli_module, "_container_info", wrong_identity)
    with pytest.raises(RuntimeError, match="container_identity_mismatch"):
        _start_worker(
            start_command,
            "http://127.0.0.1:8080",
            "synthetic-owner-token",
            tmp_path / "compose.yaml",
            "alix",
        )

    acknowledgements: list[dict[str, Any]] = []

    def capture_ack(*args: Any, **kwargs: Any) -> None:
        del args
        acknowledgements.append(kwargs["payload"])

    monkeypatch.setattr(runtime_cli_module, "_api_request", capture_ack)
    monkeypatch.setattr(
        runtime_cli_module,
        "_container_info",
        lambda container_name: {
            "Config": {
                "Labels": {
                    "alix.worker.id": "worker-abc123",
                    "alix.worker.generation": "4",
                }
            },
            "State": {"Running": True},
        },
    )
    process_command(
        {
            "action": "STOP",
            "worker_id": "worker-abc123",
            "generation": 4,
            "command_id": "synthetic-stop-command",
        },
        "http://127.0.0.1:8080",
        "synthetic-owner-token",
        tmp_path / "compose.yaml",
        "alix",
        "dispatcher-test",
    )
    assert calls[0][0] == "docker"
    assert calls[0][1][0] == "stop"
    assert acknowledgements[-1]["success"] is True
    assert acknowledgements[-1]["container_name"] is None

    monkeypatch.setattr(runtime_cli_module, "_container_info", wrong_identity)
    process_command(
        {
            "action": "DISABLE",
            "worker_id": "worker-abc123",
            "generation": 4,
            "command_id": "synthetic-disable-command",
        },
        "http://127.0.0.1:8080",
        "synthetic-owner-token",
        tmp_path / "compose.yaml",
        "alix",
        "dispatcher-test",
    )
    assert acknowledgements[-1]["success"] is False
    assert acknowledgements[-1]["error_code"] == "container_identity_mismatch"

    with pytest.raises(RuntimeError, match="unknown_runtime_command"):
        process_command(
            {"action": "EXEC", "worker_id": "worker-abc123", "generation": 4},
            "http://127.0.0.1:8080",
            "synthetic-owner-token",
            tmp_path / "compose.yaml",
            "alix",
            "dispatcher-test",
        )


def test_dispatch_once_records_runtime_failures_without_leaking_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command = {
        "action": "START",
        "worker_id": "worker-dispatch-fail",
        "generation": 1,
        "command_id": "synthetic-dispatch-command",
    }
    calls: list[tuple[str, Any]] = []

    def api_request(
        base_url: str,
        admin_token: str,
        path: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> list[dict[str, Any]] | None:
        del base_url, admin_token, method, timeout
        calls.append((path, payload))
        if path.endswith("/claim"):
            return [command]
        return None

    monkeypatch.setattr(runtime_cli_module, "_api_request", api_request)

    def fail_command(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("docker_worker_start_failed")

    monkeypatch.setattr(runtime_cli_module, "process_command", fail_command)
    count = dispatch_once(
        "http://127.0.0.1:8080",
        "synthetic-owner-token",
        tmp_path / "compose.yaml",
        "alix",
        "dispatcher-fail",
    )
    assert count == 1
    assert calls[-1][0].endswith("/synthetic-dispatch-command/ack")
    assert calls[-1][1]["success"] is False
    assert calls[-1][1]["error_code"] == "docker_worker_start_failed"

    monkeypatch.setattr(runtime_cli_module, "_api_request", lambda *args, **kwargs: [])
    assert (
        dispatch_once(
            "http://127.0.0.1:8080",
            "synthetic-owner-token",
            tmp_path / "compose.yaml",
            "alix",
            "dispatcher-empty",
        )
        == 0
    )
    monkeypatch.setattr(runtime_cli_module, "_api_request", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="runtime_command_response_invalid"):
        dispatch_once(
            "http://127.0.0.1:8080",
            "synthetic-owner-token",
            tmp_path / "compose.yaml",
            "alix",
            "dispatcher-malformed",
        )


def test_dispatcher_main_requires_owner_token_and_supports_single_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALIx_ADMIN_TOKEN=" + "X" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime_cli",
            "--env-file",
            str(env_file),
            "--compose-file",
            "compose.yaml",
            "--once",
        ],
    )
    monkeypatch.delenv("ALIx_ADMIN_TOKEN", raising=False)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        runtime_cli_module,
        "dispatch_once",
        lambda *args, **kwargs: calls.append(args) or 0,
    )
    assert runtime_cli_module.main() == 0
    assert len(calls) == 1
    assert calls[0][1] == "X" * 40

    monkeypatch.delenv("ALIx_ADMIN_TOKEN", raising=False)
    env_file.write_text("ALIx_ADMIN_TOKEN=short\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ALIx_ADMIN_TOKEN must contain"):
        runtime_cli_module.main()


def test_worker_heartbeat_http_errors_and_input_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from control_plane.worker_runtime import _send_heartbeat

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            del args

    class FakeOpener:
        def __init__(self, outcome: Any) -> None:
            self.outcome = outcome

        def open(self, request: Any, timeout: float) -> Any:
            del request, timeout
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    monkeypatch.setattr(
        "control_plane.worker_runtime.urllib.request.build_opener",
        lambda *args: FakeOpener(FakeResponse()),
    )
    with pytest.raises(WorkerRuntimeFailure, match="heartbeat_response_unexpected"):
        _send_heartbeat("http://control-plane:8080", "worker-http", "T" * 40, 1)

    monkeypatch.setattr(
        "control_plane.worker_runtime.urllib.request.build_opener",
        lambda *args: FakeOpener(
            urllib.error.HTTPError(
                "http://control-plane:8080", 403, "forbidden", None, None
            )
        ),
    )
    with pytest.raises(WorkerAuthorizationFailure, match="worker_identity_rejected"):
        _send_heartbeat("http://control-plane:8080", "worker-http", "T" * 40, 1)

    monkeypatch.setattr(
        "control_plane.worker_runtime.urllib.request.build_opener",
        lambda *args: FakeOpener(urllib.error.URLError("synthetic-network-failure")),
    )
    with pytest.raises(OSError, match="heartbeat_request_failed"):
        _send_heartbeat("http://control-plane:8080", "worker-http", "T" * 40, 1)

    with pytest.raises(ValueError, match="invalid_worker_id"):
        run_worker(
            "worker with spaces",
            "T" * 40,
            1,
            "http://control-plane:8080",
            str(tmp_path / "invalid-id.sqlite"),
            iterations=0,
        )
    with pytest.raises(ValueError, match="worker_credential_or_generation_invalid"):
        run_worker(
            "worker-valid",
            "short",
            1,
            "http://control-plane:8080",
            str(tmp_path / "invalid-token.sqlite"),
            iterations=0,
        )
    with pytest.raises(ValueError, match="worker_credential_or_generation_invalid"):
        run_worker(
            "worker-valid",
            "T" * 40,
            0,
            "http://control-plane:8080",
            str(tmp_path / "invalid-generation.sqlite"),
            iterations=0,
        )


def test_database_version_and_health_failure_are_fail_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from control_plane.db import initialize_schema, schema_meta

    store: ControlPlaneStore = client.app.state.store
    with store.engine.begin() as connection:
        connection.execute(
            schema_meta.update()
            .where(schema_meta.c.schema_key == "control_plane")
            .values(schema_value="999")
        )
    with pytest.raises(RuntimeError, match="unsupported_control_plane_schema_version"):
        initialize_schema(store.engine)

    monkeypatch.setattr(
        store, "ping", lambda: (_ for _ in ()).throw(RuntimeError("db-down"))
    )
    assert client.get("/healthz").status_code == 503


def test_disabled_tenant_members_cannot_read_tenant_metadata(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Disabled Tenant Reader")
    add_member(client, tenant_id, "owner-a", "tenant_owner")
    store: ControlPlaneStore = client.app.state.store
    store.disable_tenant(Principal("local-owner", platform_admin=True), tenant_id)
    assert (
        client.get(
            f"/v1/tenants/{tenant_id}",
            headers=auth("owner-a-token-012345678901234567890123456789"),
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/v1/tenants",
            headers=auth("owner-a-token-012345678901234567890123456789"),
        ).json()
        == []
    )


def test_store_authorization_and_missing_worker_fail_closed(
    client: TestClient,
) -> None:
    tenant_id = make_tenant(client, "Store Error Paths")
    principal = Principal("owner-a")
    store: ControlPlaneStore = client.app.state.store
    with pytest.raises(ForbiddenError):
        store.create_tenant(principal, "forbidden")
    with pytest.raises(ForbiddenError):
        store.disable_tenant(principal, tenant_id)
    with pytest.raises(ForbiddenError):
        store.get_account(principal, tenant_id, "missing-account")
    with pytest.raises(ForbiddenError):
        store.get_worker(principal, tenant_id, "missing-worker")
    with pytest.raises(NotFoundError):
        store.heartbeat_worker("missing-worker", 1, "T" * 40)
