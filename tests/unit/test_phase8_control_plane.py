from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from control_plane.auth import (
    AuthenticationError,
    Principal,
    SingleUserTokenAuthenticator,
)
from control_plane.api import create_app
from control_plane.runtime_cli import (
    _container_name,
    _start_worker,
    load_env_file,
)
from control_plane.store import ControlPlaneStore, ConflictError
from control_plane.worker_runtime import (
    WorkerAuthorizationFailure,
    WorkerRuntimeFailure,
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
    return response.json()


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
    return response.json()


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
        json={"generation": command["generation"] - 1},
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
    store: ControlPlaneStore = client.app.state.store
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
    assert __import__("os").environ["ALIx_ADMIN_TOKEN"] == "local-test-token"
    assert __import__("os").environ["POSTGRES_PASSWORD"] == "already-set"


def test_worker_profile_and_compose_security_contracts() -> None:
    import yaml

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
    store: ControlPlaneStore = client.app.state.store

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
