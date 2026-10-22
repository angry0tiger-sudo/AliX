#!/usr/bin/env python3
"""Validate the preserved Starter Kit archive and key project documentation."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "AliX_GitHub_Starter_Kit.zip"
MANIFEST = ROOT / "artifacts-manifest" / "starter-kit.json"
WHITELIST_SAMPLE = ROOT / "config" / "environments" / "market-whitelist.sample.json"
WHITELIST_SCHEMA = ROOT / "config" / "environments" / "market-whitelist.schema.json"
REQUIRED_MEMBERS = (
    "docs/MRD.md",
    "docs/ROADMAP.md",
    "scripts/validate_project_docs.py",
)
REQUIRED_PROJECT_FILES = (
    "docs/PROJECT_DOCUMENTS.md",
    "docs/knowledge_base/AliX_MRD_Dynamic_Risk_FA.md",
    "docs/knowledge_base/AliX_Roadmap_Dynamic_Risk_FA.md",
    "docs/architecture/AliX_MRD_Dynamic_Risk_FA.md",
    "docs/roadmap/AliX_Roadmap_Dynamic_Risk_FA.md",
    "docs/governance/owner-directives/2026-10-22-phase-4-knowledge-base-canonical-paths.md",
    "docs/governance/owner-directives/2026-10-22-phase-4.md",
    "docs/knowledge_base/PHASE_4_ACCEPTANCE.md",
    "docs/phase-reports/PHASE_4_STATUS.md",
    "docs/knowledge_base/PHASE_5_ACCEPTANCE.md",
    "docs/knowledge_base/PHASE_6_ACCEPTANCE.md",
    "docs/knowledge_base/PHASE_7_ACCEPTANCE.md",
    "docs/phase-reports/PHASE_5_STATUS.md",
    "docs/phase-reports/PHASE_6_STATUS.md",
    "docs/phase-reports/PHASE_7_STATUS.md",
    "docs/architecture/decisions/ADR-0005-phase-5-plan-b.md",
    "docs/architecture/decisions/ADR-0006-phase-6-external-data-nlp.md",
    "docs/architecture/decisions/ADR-0007-phase-7-risk-engine.md",
    "docs/governance/owner-directives/2026-10-22-phase-5.md",
    "docs/governance/owner-directives/2026-10-22-phase-6.md",
    "docs/governance/owner-directives/2026-10-22-phase-7.md",
    "docs/runbook/PHASE_7_RISK_HALT_AND_REARM.md",
    "risk/__init__.py",
    "risk/contracts.py",
    "risk/allocator.py",
    "risk/gate.py",
    "risk/kill_switch.py",
    "risk/supervisor.py",
    "risk/evaluation.py",
    "tests/unit/test_risk_engine.py",
    "scripts/run_phase7_risk_fault_injection.py",
    "docs/data-provider-registry.md",
    "external_data/__init__.py",
    "external_data/contracts.py",
    "external_data/connectors.py",
    "external_data/stores.py",
    "external_data/point_in_time.py",
    "services/nlp_inference/__init__.py",
    "services/nlp_inference/service.py",
    "services/nlp_inference/huggingface_local.py",
    "tests/unit/test_phase6_external_data.py",
    "tests/unit/test_nlp_inference.py",
    "services/meta_labeling/__init__.py",
    "services/meta_labeling/triple_barrier.py",
    "services/meta_labeling/meta_model.py",
    "services/meta_labeling/walkforward.py",
    "scripts/run_meta_labeling_walkforward.py",
    "tests/unit/test_meta_labeling.py",
    "config/environments/freqai-baseline.sample.json",
    "docs/knowledge_base/PROJECT_RULES.md",
    "docs/knowledge_base/PHASE_0_ACCEPTANCE.md",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    "config/environments/market-whitelist.sample.json",
    "config/environments/market-whitelist.schema.json",
)


def _validate_checksums(content: bytes, errors: list[str]) -> tuple[str, str]:
    sha256 = hashlib.sha256(content).hexdigest()
    git_blob_sha1 = hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content,
        usedforsecurity=False,
    ).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_blob = manifest.get("git_blob_sha1")
    if not expected_blob or git_blob_sha1 != expected_blob:
        errors.append(
            f"Git blob SHA-1 mismatch: expected {expected_blob!r}, got {git_blob_sha1}"
        )
    expected_sha256 = manifest.get("sha256")
    if not expected_sha256 or sha256 != expected_sha256:
        errors.append(f"SHA-256 mismatch: expected {expected_sha256!r}, got {sha256}")
    return git_blob_sha1, sha256


def _missing_archive_members(names: set[str]) -> list[str]:
    return [
        suffix
        for suffix in REQUIRED_MEMBERS
        if not any(name.endswith(suffix) for name in names)
    ]


def _validate_zip(errors: list[str]) -> None:
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            bad_member = archive.testzip()
            if bad_member:
                errors.append(f"Corrupt archive member: {bad_member}")
            missing = _missing_archive_members(set(archive.namelist()))
            if missing:
                errors.append(f"Required archive members missing: {missing}")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"Invalid ZIP archive: {exc}")


def _validate_archive(content: bytes, errors: list[str]) -> tuple[str, str]:
    checksums = _validate_checksums(content, errors)
    _validate_zip(errors)
    return checksums


def _validate_project_files(errors: list[str]) -> None:
    for relative_path in REQUIRED_PROJECT_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"Missing required project file: {relative_path}")
    if WHITELIST_SAMPLE.is_file() and WHITELIST_SCHEMA.is_file():
        try:
            jsonschema.validate(
                instance=json.loads(WHITELIST_SAMPLE.read_text(encoding="utf-8")),
                schema=json.loads(WHITELIST_SCHEMA.read_text(encoding="utf-8")),
            )
        except (
            json.JSONDecodeError,
            jsonschema.SchemaError,
            jsonschema.ValidationError,
        ) as exc:
            errors.append(f"Market whitelist sample/schema validation failed: {exc}")


def main() -> int:
    errors: list[str] = []
    if not ARCHIVE.is_file():
        errors.append(f"Missing archive: {ARCHIVE.relative_to(ROOT)}")
    if not MANIFEST.is_file():
        errors.append(f"Missing manifest: {MANIFEST.relative_to(ROOT)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    git_blob_sha1, sha256 = _validate_archive(ARCHIVE.read_bytes(), errors)
    _validate_project_files(errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Starter Kit validation passed; git_blob_sha1={git_blob_sha1}; sha256={sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
