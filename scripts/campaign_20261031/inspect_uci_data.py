#!/usr/bin/env python3
"""Inspect two small processed biomedical tables from their UCI repository."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

DATASETS = {
    17: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    45: "https://archive.ics.uci.edu/dataset/45/heart+disease",
}
MAX_TOTAL_BYTES = 50 * 1024 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, path: Path, allowed_urls: set[str], budget: int) -> dict:
    if url not in allowed_urls or urlsplit(url).scheme != "https":
        raise ValueError("Unsupported source URL")
    if path.exists():
        raise ValueError("Refusing to replace a source file")
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError("Prior partial download must be preserved")
    count = 0
    try:
        with requests.get(url, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            if response.url not in allowed_urls:
                raise ValueError("Source redirected outside its declared URLs")
            with temporary.open("xb") as handle:
                for block in response.iter_content(65536):
                    count += len(block)
                    if count > min(budget, 2 * 1024 * 1024):
                        raise ValueError("Source exceeds bounded download budget")
                    handle.write(block)
            if count == 0:
                raise ValueError("Empty source file")
            final_url, content_type = response.url, response.headers.get("Content-Type")
        temporary.replace(path)
        return {
            "url": url,
            "final_url": final_url,
            "bytes": count,
            "sha256": sha256(path),
            "content_type": content_type,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        temporary.unlink(missing_ok=True)


def inspect_table(path: Path, metadata: dict) -> dict:
    variables = metadata.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("Source variable dictionary is required")
    by_name = {item["name"]: item for item in variables}
    if len(by_name) != len(variables):
        raise ValueError("Duplicate source variable names")
    missing_symbols = {"", "?", "na", "n/a", "null"}
    if metadata.get("missing_values_symbol"):
        missing_symbols.add(str(metadata["missing_values_symbol"]).casefold())
    summary = {
        name: {
            "source_role": item.get("role"),
            "source_type": item.get("type"),
            "units": item.get("units"),
            "description": item.get("description"),
            "missing_count": 0,
            "nonfinite_count": 0,
            "numeric_count": 0,
            "minimum": None,
            "maximum": None,
            "categorical_counts": Counter(),
        }
        for name, item in by_name.items()
    }
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or set(reader.fieldnames) != set(by_name)
        ):
            raise ValueError("CSV columns disagree with source dictionary")
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("CSV row width mismatch")
            rows += 1
            for name, value in record.items():
                stat, variable = summary[name], by_name[name]
                value = value.strip()
                if value.casefold() in missing_symbols:
                    stat["missing_count"] += 1
                    continue
                if variable.get("role") != "ID" and variable.get("type") == "Categorical":
                    stat["categorical_counts"][value] += 1
                try:
                    number = float(value)
                except ValueError as exc:
                    if variable.get("type") in {"Integer", "Continuous", "Real"}:
                        raise ValueError(
                            f"Invalid numeric value in declared column {name}"
                        ) from exc
                    continue
                if not math.isfinite(number):
                    stat["nonfinite_count"] += 1
                else:
                    stat["numeric_count"] += 1
                    stat["minimum"] = (
                        number if stat["minimum"] is None else min(stat["minimum"], number)
                    )
                    stat["maximum"] = (
                        number if stat["maximum"] is None else max(stat["maximum"], number)
                    )
    if rows == 0 or rows != metadata.get("num_instances"):
        raise ValueError("Observed row count disagrees with reported source count")
    features = sum(item.get("role") == "Feature" for item in variables)
    if features != metadata.get("num_features"):
        raise ValueError("Feature roles disagree with source count")
    return {
        "status": "PASS",
        "rows": rows,
        "columns": len(by_name),
        "features": features,
        "reported_instances": metadata.get("num_instances"),
        "reported_features": metadata.get("num_features"),
        "target_columns": metadata.get("target_col"),
        "id_columns": metadata.get("index_col"),
        "missing_cells": sum(item["missing_count"] for item in summary.values()),
        "nonfinite_cells": sum(item["nonfinite_count"] for item in summary.values()),
        "missing_symbols_interpreted": sorted(missing_symbols),
        "variables": summary,
        "source_sha256": sha256(path),
        "summary_kind": "computed directly from unchanged processed source CSV",
        "limitations": [
            "Rows are reported instances; subject independence is not inferred.",
            "Source feature/target definitions are retained, not reinterpreted as validated endpoints.",
            "No imputation, normalization, model fitting, or statistical-power calculation was performed.",
        ],
    }


def build(out: Path) -> dict:
    if out.exists():
        raise ValueError("Inspection output must be new")
    out.mkdir(parents=True, exist_ok=False)
    downloaded, results = [], []
    remaining = MAX_TOTAL_BYTES
    for dataset_id, repository_url in DATASETS.items():
        metadata_url = f"https://archive.ics.uci.edu/api/dataset?id={dataset_id}"
        data_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"
        allowed = {metadata_url, data_url, repository_url}
        metadata_path = out / f"uci_{dataset_id}_metadata.json"
        data_path = out / f"uci_{dataset_id}.csv"
        page_path = out / f"uci_{dataset_id}_repository.html"
        try:
            for url, path in [
                (metadata_url, metadata_path),
                (data_url, data_path),
                (repository_url, page_path),
            ]:
                entry = download(url, path, allowed, remaining)
                remaining -= entry["bytes"]
                downloaded.append(entry)
            envelope = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = envelope["data"]
            if (
                envelope.get("status") != 200
                or metadata.get("uci_id") != dataset_id
                or metadata.get("data_url") != data_url
            ):
                raise ValueError("Unexpected UCI metadata identity")
            inspection = inspect_table(data_path, metadata)
            page = page_path.read_text(encoding="utf-8")
            license_present = "creativecommons.org/licenses/by/4.0" in page
            inspection.update(
                {
                    "uci_id": dataset_id,
                    "name": metadata["name"],
                    "doi": metadata.get("dataset_doi"),
                    "repository_url": repository_url,
                    "data_url": data_url,
                    "license": "CC BY 4.0 as reported on captured repository page"
                    if license_present
                    else "UNKNOWN",
                    "license_evidence_file": page_path.name,
                    "access": "public source files retrieved",
                }
            )
            (out / f"uci_{dataset_id}_inspection.json").write_text(
                json.dumps(inspection, indent=2, ensure_ascii=True), encoding="utf-8"
            )
            results.append(
                {
                    key: inspection[key]
                    for key in [
                        "uci_id",
                        "name",
                        "doi",
                        "rows",
                        "columns",
                        "features",
                        "missing_cells",
                        "nonfinite_cells",
                        "license",
                        "status",
                    ]
                }
            )
        except (OSError, ValueError, KeyError, requests.RequestException) as exc:
            results.append({"uci_id": dataset_id, "status": "FAIL", "error": str(exc)})
            break
    manifest = {
        "schema_version": "processed-biomedical-table-inspection-v1",
        "status": "PASS"
        if len(results) == len(DATASETS) and all(item["status"] == "PASS" for item in results)
        else "PARTIAL_OR_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": results,
        "downloaded_bytes": MAX_TOTAL_BYTES - remaining,
        "max_total_bytes": MAX_TOTAL_BYTES,
        "retrievals": downloaded,
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(out.iterdir())
            if path.is_file()
        ],
    }
    (out / "INSPECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.out)
    print(
        json.dumps(
            {key: report[key] for key in ["status", "sources", "downloaded_bytes"]}, indent=2
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
