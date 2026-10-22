"""Connectivity route policies and health/failover manager."""

from connectivity.manager.manager import (
    ConnectivityManager,
    ConnectivityStatus,
    FailoverResult,
)

__all__ = ["ConnectivityManager", "ConnectivityStatus", "FailoverResult"]
