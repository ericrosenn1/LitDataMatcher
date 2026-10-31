"""Strict parsing shared by checkpoints, native runs and controller receipts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def strict_json_loads(content):
    value = json.loads(content, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    _finite(value)
    return value


def validate_artifact(path: Path) -> str:
    """Validate supported serialization; zero-row JSONL is a valid empty result."""
    path = Path(path)
    suffix = path.suffix.lower()
    content = (
        path.read_text(encoding="utf-8-sig")
        if suffix in {".json", ".jsonl", ".csv", ".html"}
        else None
    )
    if suffix == ".json":
        strict_json_loads(content)
    elif suffix == ".jsonl":
        for number, line in enumerate(content.split("\n"), 1):
            if line.strip() and not isinstance(strict_json_loads(line), dict):
                raise ValueError(f"JSONL row {number} is not an object")
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.reader(handle, strict=True)
            header = next(rows, None)
            if not header or len(header) < 2:
                raise ValueError("CSV requires a header with at least two columns")
            if any(len(row) != len(header) for row in rows):
                raise ValueError("CSV row width differs from its header")
    elif suffix == ".html":
        if "<!doctype html>" not in content.casefold() or "Content-Security-Policy" not in content:
            raise ValueError("HTML report lacks document or content security policy")
    else:
        return "NOT_RUN"
    return "PASS"
