"""HTTP API for the local-first Control Plane."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from control_plane.auth import (
    AuthenticationError,
    Authenticator,
    AuthorizationError,
    Principal,
)
from control_plane.contracts import (
    AccountCreate,
    LifecycleRequest,
    MembershipCreate,
    RuntimeAckRequest,
    RuntimeClaimRequest,
    RuntimeCredentialRequest,
    TenantCreate,
    WorkerCreate,
    WorkerHeartbeatRequest,
)
from control_plane.db import create_engine_for_url
from control_plane.store import (
    ControlPlaneError,
    ControlPlaneStore,
    NotFoundError,
)


def create_app(database_url: str, authenticator: Authenticator) -> FastAPI:
    """Build an app with explicit database and auth dependencies for reproducible tests."""
    store = ControlPlaneStore(create_engine_for_url(database_url))
    app = FastAPI(
        title="AliX Control Plane",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.authenticator = authenticator

    def current_principal(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> Principal:
        try:
            return app.state.authenticator.authenticate(authorization)
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=401,
                detail=exc.args[0],
                headers={"WWW-Authenticate": "Bearer"},
            ) from None

    def platform_admin(principal: Principal = Depends(current_principal)) -> Principal:
        if not principal.platform_admin:
            raise HTTPException(status_code=403, detail="platform_admin_required")
        return principal

    @app.exception_handler(ControlPlaneError)
    async def control_plane_error_handler(
        request: Request, exc: ControlPlaneError
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.code},
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(
        request: Request, exc: AuthorizationError
    ) -> JSONResponse:
        del request
        status_code = (
            401
            if str(exc) in ("worker_token_required", "worker_identity_or_scope_invalid")
            else 403
        )
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
        return JSONResponse(
            status_code=status_code, content={"detail": str(exc)}, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        safe_errors = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "type": str(error.get("type", "validation_error")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.get("/healthz")
    def health() -> Response:
        try:
            store.ping()
        except Exception:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/v1/me")
    def me(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
        return {
            "subject": principal.subject,
            "platform_admin": principal.platform_admin,
            "auth_mode": "single_user" if principal.platform_admin else "injected",
        }

    @app.post("/v1/tenants", status_code=201)
    def create_tenant(
        body: TenantCreate, principal: Principal = Depends(platform_admin)
    ) -> dict[str, Any]:
        return store.create_tenant(principal, body.name)

    @app.get("/v1/tenants")
    def list_tenants(
        principal: Principal = Depends(current_principal),
    ) -> list[dict[str, Any]]:
        return store.list_tenants(principal)

    @app.get("/v1/tenants/{tenant_id}")
    def get_tenant(
        tenant_id: str, principal: Principal = Depends(current_principal)
    ) -> dict[str, Any]:
        return store.get_tenant(principal, tenant_id)

    @app.post("/v1/tenants/{tenant_id}/members", status_code=201)
    def add_member(
        tenant_id: str,
        body: MembershipCreate,
        principal: Principal = Depends(current_principal),
    ) -> dict[str, Any]:
        return store.add_membership(principal, tenant_id, body.subject, body.role)

    @app.delete("/v1/tenants/{tenant_id}/members/{subject}", status_code=204)
    def revoke_member(
        tenant_id: str,
        subject: str,
        principal: Principal = Depends(current_principal),
    ) -> Response:
        store.revoke_membership(principal, tenant_id, subject)
        return Response(status_code=204)

    @app.post("/v1/tenants/{tenant_id}/accounts", status_code=201)
    def create_account(
        tenant_id: str,
        body: AccountCreate,
        principal: Principal = Depends(current_principal),
    ) -> dict[str, Any]:
        return store.create_account(
            principal,
            tenant_id,
            body.exchange_id,
            body.environment,
            body.account_alias,
            body.secret_ref_id,
        )

    @app.get("/v1/tenants/{tenant_id}/accounts")
    def list_accounts(
        tenant_id: str, principal: Principal = Depends(current_principal)
    ) -> list[dict[str, Any]]:
        return store.list_accounts(principal, tenant_id)

    @app.post("/v1/tenants/{tenant_id}/workers", status_code=201)
    def create_worker(
        tenant_id: str,
        body: WorkerCreate,
        principal: Principal = Depends(current_principal),
    ) -> dict[str, Any]:
        return store.create_worker(
            principal,
            tenant_id,
            body.account_id,
            body.exchange_id,
            body.environment,
            body.strategy_revision,
            body.profile_id,
        )

    @app.get("/v1/tenants/{tenant_id}/workers")
    def list_workers(
        tenant_id: str, principal: Principal = Depends(current_principal)
    ) -> list[dict[str, Any]]:
        return store.list_workers(principal, tenant_id)

    @app.post("/v1/tenants/{tenant_id}/workers/{worker_id}/lifecycle")
    def worker_lifecycle(
        tenant_id: str,
        worker_id: str,
        body: LifecycleRequest,
        principal: Principal = Depends(current_principal),
    ) -> dict[str, Any]:
        return store.transition_worker(principal, tenant_id, worker_id, body.action)

    @app.get("/v1/tenants/{tenant_id}/audit")
    def audit_events(
        tenant_id: str, principal: Principal = Depends(current_principal)
    ) -> list[dict[str, Any]]:
        return store.list_audit_events(principal, tenant_id)

    @app.post("/v1/admin/runtime-commands/claim")
    def claim_runtime_commands(
        body: RuntimeClaimRequest,
        principal: Principal = Depends(platform_admin),
    ) -> list[dict[str, Any]]:
        del principal
        return store.claim_runtime_commands(
            body.dispatcher_id, body.limit, body.lease_seconds
        )

    @app.put("/v1/admin/workers/{worker_id}/runtime-credential", status_code=204)
    def register_runtime_credential(
        worker_id: str,
        body: RuntimeCredentialRequest,
        principal: Principal = Depends(platform_admin),
    ) -> Response:
        del principal
        store.register_worker_credential(
            worker_id, body.generation, body.credential_sha256
        )
        return Response(status_code=204)

    @app.post("/v1/admin/runtime-commands/{command_id}/ack", status_code=204)
    def acknowledge_command(
        command_id: str,
        body: RuntimeAckRequest,
        principal: Principal = Depends(platform_admin),
    ) -> Response:
        del principal
        command_rows = store.claimed_command_owner(command_id)
        if command_rows is None:
            raise NotFoundError("runtime_command_not_found")
        store.acknowledge_runtime_command(
            command_id,
            str(command_rows["claimed_by"]),
            body.success,
            body.container_name,
            body.error_code,
        )
        return Response(status_code=204)

    @app.post("/v1/internal/workers/{worker_id}/heartbeat", status_code=204)
    def worker_heartbeat(
        worker_id: str,
        body: WorkerHeartbeatRequest,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> Response:
        token: str | None = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        store.heartbeat_worker(worker_id, body.generation, token)
        return Response(status_code=204)

    return app
