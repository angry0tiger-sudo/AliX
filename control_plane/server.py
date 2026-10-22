"""Production entrypoint for the local Control Plane API."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from sqlalchemy.engine import URL

from control_plane.auth import SingleUserTokenAuthenticator
from control_plane.api import create_app


def _database_url() -> str:
    explicit = os.environ.get("ALIx_CONTROL_DATABASE_URL")
    if explicit:
        return explicit
    password = os.environ.get("POSTGRES_PASSWORD", "")
    username = os.environ.get("POSTGRES_USER", "alix")
    database = os.environ.get("POSTGRES_DB", "alix_control")
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD must be set in the local .env file")
    return URL.create(
        "postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    ).render_as_string(hide_password=False)


def build_application() -> FastAPI:
    mode = os.environ.get("ALIx_AUTH_MODE", "single_user")
    if mode != "single_user":
        raise RuntimeError("unsupported_auth_mode; OIDC mode is not enabled in Phase 8")
    token = os.environ.get("ALIx_ADMIN_TOKEN", "")
    authenticator = SingleUserTokenAuthenticator(token)
    return create_app(_database_url(), authenticator)


def main() -> None:
    app = build_application()
    host = os.environ.get("ALIx_CONTROL_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("ALIx_CONTROL_PORT", "8080"))
    if not 1 <= port <= 65535:
        raise RuntimeError("invalid_control_plane_port")
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="info")


if __name__ == "__main__":
    main()
