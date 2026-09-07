"""Generate a compact, deterministic security and repository-content receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "openai_key": re.compile(rb"sk-[A-Za-z0-9]{20,}"),
}


def validate(source: Path, junit: Path) -> dict:
    source = source.resolve()
    paths = _tracked_paths(source)
    findings = []
    oversize = []
    tracked_logs = []
    for relative in paths:
        path = source / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > 5 * 1024 * 1024:
            oversize.append({"path": relative, "size_bytes": size})
        if path.suffix.lower() == ".log":
            tracked_logs.append(relative)
        if size <= 2 * 1024 * 1024:
            content = path.read_bytes()
            for kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    findings.append({"path": relative, "kind": kind})
    names = _passing_junit_names(junit)
    injection_tests = {
        "test_hostile_source_never_scientific_claim",
        "test_report_escapes_hostile_metadata",
        "test_hashed_arbitrary_json_cannot_promote_a_gate",
    }
    license_files = [source / "LICENSE", source / "docs" / "v2" / "THIRD_PARTY_NOTICES.md"]
    result = {
        "schema_version": "security-validation-v1",
        "status": "PASS",
        "source_commit": _git(source, "rev-parse", "HEAD"),
        "tracked_files": len(paths),
        "secret_findings": len(findings),
        "secret_finding_details": findings,
        "oversize_tracked_files": len(oversize),
        "oversize_details": oversize,
        "tracked_runtime_logs": tracked_logs,
        "license_notice_files": [
            {"path": str(path.relative_to(source)), "sha256": _sha256(path)}
            for path in license_files
            if path.is_file()
        ],
        "prompt_injection_tests": sorted(injection_tests & names),
        "junit_sha256": _sha256(junit) if junit.is_file() else None,
    }
    if (
        findings
        or oversize
        or tracked_logs
        or len(result["license_notice_files"]) != 2
        or not injection_tests <= names
    ):
        result["status"] = "FAIL"
    return result


def _tracked_paths(source: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "-z"], stderr=subprocess.DEVNULL
    )
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def _passing_junit_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    if any(int(suite.get(key, "0")) for suite in suites for key in ("failures", "errors", "skipped")):
        return set()
    return {
        case.get("name")
        for case in root.iter("testcase")
        if case.get("name") and case.find("failure") is None and case.find("error") is None
    }


def _git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args.source, args.junit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
