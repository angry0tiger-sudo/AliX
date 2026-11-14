"""Compatibility exports; the implementation lives in a stdlib-only module."""

from __future__ import annotations

from research.four_arm_core.lock_validator import (
    EXECUTION_AUTHORIZATION_ENV,
    LOCK_SCHEMA_VERSION,
    PREHACK_DATASET_SHA256,
    ExecutionAuthorizationError,
    load_owner_authorized_execution_lock,
)

__all__ = [
    "EXECUTION_AUTHORIZATION_ENV",
    "LOCK_SCHEMA_VERSION",
    "PREHACK_DATASET_SHA256",
    "ExecutionAuthorizationError",
    "load_owner_authorized_execution_lock",
]
