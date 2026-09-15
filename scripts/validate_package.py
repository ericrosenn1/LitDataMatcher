"""Validate that built LitDataMatcher distributions are public and lightweight."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_COMPONENTS = {".git", ".github", "data", "models", "test_one_paper", "__pycache__"}
REQUIRED_WHEEL_SUFFIXES = {
    "litdatamatcher/__init__.py",
    "litdatamatcher/cli.py",
    ".dist-info/METADATA",
    ".dist-info/WHEEL",
}


def _normalise_sdist_member(name: str) -> str:
    """Remove the single project-root directory that setuptools adds to an sdist."""

    parts = Path(name).parts
    return "/".join(parts[1:]) if len(parts) > 1 else ""


def _check_members(archive: Path, members: list[str], is_sdist: bool) -> list[str]:
    """Return validation errors for one distribution's logical member paths."""

    logical = [_normalise_sdist_member(member) if is_sdist else member for member in members]
    errors: list[str] = []
    for member in logical:
        if not member:
            continue
        components = set(Path(member).parts)
        forbidden = sorted(components & FORBIDDEN_COMPONENTS)
        if forbidden:
            errors.append(f"{archive.name} contains forbidden component(s) {forbidden}: {member}")
    if not is_sdist:
        for required in REQUIRED_WHEEL_SUFFIXES:
            if not any(member.endswith(required) for member in logical):
                errors.append(f"{archive.name} is missing required wheel member suffix: {required}")
    return errors


def validate(dist: Path) -> list[str]:
    """Validate one sdist and one wheel without extracting either archive."""

    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    errors: list[str] = []
    if len(sdists) != 1:
        errors.append(f"expected exactly one sdist in {dist}, found {len(sdists)}")
    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel in {dist}, found {len(wheels)}")
    for sdist in sdists:
        with tarfile.open(sdist, "r:gz") as archive:
            errors.extend(_check_members(sdist, archive.getnames(), is_sdist=True))
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            errors.extend(_check_members(wheel, archive.namelist(), is_sdist=False))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"), help="Directory containing build outputs.")
    args = parser.parse_args()
    errors = validate(args.dist)
    if errors:
        print("PACKAGE_VALIDATION_FAIL")
        print("\n".join(errors))
        return 1
    print("PACKAGE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
