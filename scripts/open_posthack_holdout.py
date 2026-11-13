"""Require an explicit owner release reference before a consumer opens the holdout."""

from __future__ import annotations

import argparse
from pathlib import Path


class HoldoutLocked(RuntimeError):
    """Raised when the owner has not explicitly released the sealed holdout."""


def require_release_authorization(authorization: str | None = None) -> None:
    if authorization is None or not authorization.strip():
        raise HoldoutLocked(
            "HOLDOUT_LOCKED: explicit owner Release Authorization is required"
        )


def authorized_dataset_path(
    path: Path, *, owner_release_authorization: str | None = None
) -> Path:
    require_release_authorization(owner_release_authorization)
    if not path.is_file():
        raise FileNotFoundError("holdout_dataset_artifact_missing")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--owner-release-authorization")
    args = parser.parse_args()
    authorized = authorized_dataset_path(
        args.dataset,
        owner_release_authorization=args.owner_release_authorization,
    )
    print(f"Holdout release authorized; dataset path: {authorized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
