#!/usr/bin/env python3
"""Validate the preserved Starter Kit archive and key project documentation."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "AliX_GitHub_Starter_Kit.zip"
MANIFEST = ROOT / "artifacts-manifest" / "starter-kit.json"
REQUIRED_MEMBERS = (
    "docs/MRD.md",
    "docs/ROADMAP.md",
    "scripts/validate_project_docs.py",
)


def main() -> int:
    errors: list[str] = []
    if not ARCHIVE.is_file():
        errors.append(f"Missing archive: {ARCHIVE.relative_to(ROOT)}")
    if not MANIFEST.is_file():
        errors.append(f"Missing manifest: {MANIFEST.relative_to(ROOT)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    content = ARCHIVE.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    git_blob_sha1 = hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_blob = manifest.get("git_blob_sha1")
    if not expected_blob or git_blob_sha1 != expected_blob:
        errors.append(
            f"Git blob SHA-1 mismatch: expected {expected_blob!r}, "
            f"got {git_blob_sha1}"
        )

    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            bad_member = archive.testzip()
            if bad_member:
                errors.append(f"Corrupt archive member: {bad_member}")
            names = set(archive.namelist())
            missing = [
                suffix
                for suffix in REQUIRED_MEMBERS
                if not any(name.endswith(suffix) for name in names)
            ]
            if missing:
                errors.append(f"Required archive members missing: {missing}")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"Invalid ZIP archive: {exc}")

    required_project_files = (
        "docs/knowledge_base/PROJECT_RULES.md",
        "docs/knowledge_base/PHASE_0_ACCEPTANCE.md",
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
    )
    for rel in required_project_files:
        if not (ROOT / rel).is_file():
            errors.append(f"Missing required project file: {rel}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Starter Kit validation passed; git_blob_sha1={git_blob_sha1}; "
        f"sha256={sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
