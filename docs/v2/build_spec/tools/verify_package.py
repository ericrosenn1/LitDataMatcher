#!/usr/bin/env python3
"""Verify this build-spec package without installing anything or running the build.

Usage:
    python tools/verify_package.py
    python tools/verify_package.py /path/to/extracted/package
    python tools/verify_package.py /path/to/package.zip

Checks integrity, not the correctness or completion of LitDataMatcher itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

MANIFEST = "PACKAGE_MANIFEST.json"
SPEC_ID = "LITDATAMATCHER_V2_CODEX_BUILD_20260907"
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


class VerificationError(Exception):
    """An integrity or safety check failed."""


def safe_relative(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise VerificationError(f"Invalid relative path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(v in ("", ".", "..") for v in name.split("/")):
        raise VerificationError(f"Unsafe relative path: {name!r}")
    if any(":" in v for v in p.parts):
        raise VerificationError(f"Drive or stream syntax in path: {name!r}")
    return name


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def json_bytes(data: bytes) -> Any:
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_object,
                      parse_constant=lambda v: (_ for _ in ()).throw(
                          VerificationError(f"Non-finite JSON value: {v}")))


def verify_payload(manifest: Any, actual_names: set[str],
                   read_bytes: Callable[[str], bytes]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("spec_id") != SPEC_ID:
        raise VerificationError("Manifest is not for the expected LitDataMatcher specification.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise VerificationError("Manifest has no file inventory.")
    expected: dict[str, dict[str, Any]] = {}
    total = 0
    json_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise VerificationError("Malformed inventory row.")
        name = safe_relative(entry.get("path"))
        if name == MANIFEST or name in expected:
            raise VerificationError(f"Duplicate/self-referencing inventory entry: {name}")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or not (0 <= size <= MAX_MEMBER_BYTES):
            raise VerificationError(f"Invalid size for {name}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VerificationError(f"Invalid SHA-256 for {name}")
        total += size
        expected[name] = entry
    if total > MAX_TOTAL_BYTES:
        raise VerificationError("Package exceeds the bounded size for this instruction bundle.")
    wanted = set(expected) | {MANIFEST}
    missing, extra = wanted - actual_names, actual_names - wanted
    if missing or extra:
        raise VerificationError(f"Inventory mismatch. Missing={sorted(missing)}; extra={sorted(extra)}")
    if manifest.get("file_count") != len(entries):
        raise VerificationError("Manifest file_count is inconsistent.")
    for name, entry in expected.items():
        data = read_bytes(name)
        if len(data) != entry["size_bytes"]:
            raise VerificationError(f"Length mismatch: {name}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise VerificationError(f"SHA-256 mismatch: {name}")
        if name.endswith(".json"):
            json_bytes(data)
            json_count += 1
    return {"status": "PACKAGE_INTEGRITY_PASS", "spec_version": manifest.get("spec_version"),
            "hashed_files": len(entries), "total_payload_bytes": total,
            "json_syntax_checks": json_count, "application_readiness_tested": False}


def verify_directory(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise VerificationError("Package root must not be a symbolic link.")
    root = root.resolve(strict=True)
    actual: set[str] = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise VerificationError(f"Symbolic link in package: {p}")
        if p.is_file():
            actual.add(safe_relative(p.relative_to(root).as_posix()))
    def reader(name: str) -> bytes:
        p = root / safe_relative(name)
        if not p.is_relative_to(root) or p.stat().st_size > MAX_MEMBER_BYTES:
            raise VerificationError(f"Unsafe or oversized member: {name}")
        return p.read_bytes()
    return verify_payload(json_bytes(reader(MANIFEST)), actual, reader)


def verify_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as z:
        names: set[str] = set()
        files: set[str] = set()
        size = 0
        for info in z.infolist():
            if info.filename in names:
                raise VerificationError(f"Duplicate ZIP member: {info.filename}")
            names.add(info.filename)
            raw_name = info.filename[:-1] if info.is_dir() else info.filename
            safe_relative(raw_name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise VerificationError(f"Symbolic link in ZIP: {info.filename}")
            if info.flag_bits & 0x1:
                raise VerificationError("Encrypted ZIP members are not supported.")
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise VerificationError(f"Oversized ZIP member: {info.filename}")
            size += info.file_size
            files.add(info.filename)
        if size > MAX_TOTAL_BYTES:
            raise VerificationError("ZIP exceeds this instruction bundle's size bound.")
        manifest_paths = [v for v in files if PurePosixPath(v).name == MANIFEST]
        if len(manifest_paths) != 1:
            raise VerificationError("ZIP must contain exactly one package manifest.")
        mf = manifest_paths[0]
        prefix = mf[:-len(MANIFEST)]
        if any(not n.startswith(prefix) for n in files):
            raise VerificationError("ZIP contains files outside its package root.")
        relative_names = {safe_relative(n[len(prefix):]) for n in files}
        bad = z.testzip()
        if bad is not None:
            raise VerificationError(f"ZIP CRC check failed: {bad}")
        return verify_payload(json_bytes(z.read(mf)), relative_names,
                              lambda name: z.read(prefix + safe_relative(name)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        if args.path.is_dir():
            result = verify_directory(args.path)
        elif args.path.is_file() and zipfile.is_zipfile(args.path):
            result = verify_zip(args.path)
        else:
            raise VerificationError(f"Not an extracted package directory or ZIP: {args.path}")
        print(json.dumps(result, indent=2))
        print("PASS: build-spec package integrity verified; application not built or tested here.")
        return 0
    except (VerificationError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"PACKAGE_INTEGRITY_FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
