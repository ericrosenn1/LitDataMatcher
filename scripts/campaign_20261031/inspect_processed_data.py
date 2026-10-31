#!/usr/bin/env python3
"""Bounded download and measurement-only inspection of GEO series matrices."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

SOURCES = {
    "GSE2034": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/matrix/GSE2034_series_matrix.txt.gz",
    "GSE20194": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/matrix/GSE20194_series_matrix.txt.gz",
}
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MISSING = {"", "na", "n/a", "null", "."}


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_source_url(url: str, accession: str) -> None:
    parsed = urlsplit(url)
    expected = rf"/geo/series/GSE\d+nnn/{re.escape(accession)}/matrix/{re.escape(accession)}(?:-GPL\d+)?_series_matrix\.txt\.gz"
    if (parsed.scheme != "https" or parsed.netloc != "ftp.ncbi.nlm.nih.gov"
            or parsed.query or parsed.fragment or not re.fullmatch(expected, parsed.path)):
        raise ValueError("Only accession-matched primary GEO processed series matrices are supported")


def download_matrix(url: str, accession: str, destination: Path, remaining_bytes: int) -> dict:
    validate_source_url(url, accession)
    if destination.exists() or destination.with_suffix(destination.suffix + ".part").exists():
        raise ValueError("Refusing to replace an existing source download")
    if remaining_bytes < 1:
        raise ValueError("Download budget exhausted")
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest, count = hashlib.sha256(), 0
    try:
        with requests.get(url, stream=True, timeout=(10, 45), headers={"User-Agent": "LitDataMatcher/0.4 processed-data inspection"}) as response:
            response.raise_for_status()
            validate_source_url(response.url, accession)
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected HTTP content encoding; raw-file identity would be ambiguous")
            expected_size = int(response.headers.get("Content-Length", "0"))
            if expected_size > remaining_bytes:
                raise ValueError("Source file exceeds remaining compressed download budget")
            with temporary.open("xb") as handle:
                for block in response.iter_content(chunk_size=65536):
                    count += len(block)
                    if count > remaining_bytes:
                        raise ValueError("Source stream exceeded compressed download budget")
                    digest.update(block)
                    handle.write(block)
            if count == 0 or (expected_size and count != expected_size):
                raise ValueError("Empty or incomplete source download")
            response_metadata = {"http_status": response.status_code, "content_type": response.headers.get("Content-Type"),
                                 "last_modified": response.headers.get("Last-Modified"), "final_url": response.url}
        temporary.replace(destination)
        return {"accession": accession, "url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "compressed_bytes": count, "source_sha256": digest.hexdigest(), **response_metadata}
    finally:
        temporary.unlink(missing_ok=True)


def inspect_matrix(path: Path, *, max_decoded_bytes: int = 800 * 1024 * 1024) -> dict:
    if not path.name.endswith(("_series_matrix.txt", "_series_matrix.txt.gz")):
        raise ValueError("Unsupported file format; expected a GEO series matrix")
    opener = gzip.open if path.name.endswith(".gz") else open
    sample_fields, series_fields = {}, {}
    in_table, finished, header = False, False, None
    feature_ids = set()
    finite = missing = nonfinite = cells = decoded_bytes = 0
    minimum, maximum = None, None
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            decoded_bytes += len(line.encode("utf-8"))
            if decoded_bytes > max_decoded_bytes:
                raise ValueError("Decoded matrix exceeds inspection budget")
            stripped = line.strip()
            if stripped.lower() == "!series_matrix_table_begin":
                if in_table or finished:
                    raise ValueError("Unexpected second matrix table")
                in_table = True
                continue
            if stripped.lower() == "!series_matrix_table_end":
                if not in_table or header is None:
                    raise ValueError("Matrix end without a table header")
                in_table, finished = False, True
                continue
            if not stripped:
                continue
            values = next(csv.reader([line], delimiter="\t", quotechar='"'))
            if in_table:
                if header is None:
                    if len(values) < 2 or values[0] != "ID_REF" or len(set(values[1:])) != len(values) - 1:
                        raise ValueError("Unsupported or duplicate sample column headers")
                    header = values
                    continue
                if len(values) != len(header):
                    raise ValueError("Matrix row width disagrees with sample count")
                feature = values[0]
                if not feature or feature in feature_ids:
                    raise ValueError("Empty or duplicate feature identifier")
                feature_ids.add(feature)
                for value in values[1:]:
                    cells += 1
                    if value.strip().casefold() in MISSING:
                        missing += 1
                        continue
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError("Non-numeric matrix measurement") from exc
                    if not math.isfinite(number):
                        nonfinite += 1
                    else:
                        finite += 1
                        minimum = number if minimum is None else min(minimum, number)
                        maximum = number if maximum is None else max(maximum, number)
            elif values[0].startswith("!Sample_"):
                sample_fields.setdefault(values[0][1:], []).append(values[1:])
            elif values[0].startswith("!Series_"):
                series_fields.setdefault(values[0][1:], []).append(values[1:])
    if not finished or header is None or not feature_ids:
        raise ValueError("Missing or incomplete GEO expression matrix")
    accessions = sample_fields.get("Sample_geo_accession", [])
    if accessions and (len(accessions) != 1 or accessions[0] != header[1:]):
        raise ValueError("Reported sample accessions disagree with matrix columns")
    metadata_width_issues = {key: [len(row) for row in rows if len(row) != len(header) - 1]
                             for key, rows in sample_fields.items()}
    metadata_width_issues = {key: value for key, value in metadata_width_issues.items() if value}
    characteristic_labels = sorted({cell.split(":", 1)[0].strip() for key, rows in sample_fields.items()
                                    if key.startswith("Sample_characteristics") for row in rows for cell in row if ":" in cell})
    return {"status": "PARSED", "format": "GEO series matrix", "source_sha256": file_hash(path),
            "features": len(feature_ids), "sample_columns": len(header) - 1, "cells": cells,
            "finite_numeric_cells": finite, "missing_cells": missing, "nonfinite_cells": nonfinite,
            "finite_value_min": minimum, "finite_value_max": maximum,
            "feature_id_examples": sorted(feature_ids)[:5], "sample_ids": header[1:],
            "reported_characteristic_labels": characteristic_labels, "sample_metadata_fields": sample_fields,
            "series_metadata_fields": series_fields, "metadata_width_issues": metadata_width_issues,
            "derived_summary_not_source_claim": True,
            "limitations": ["Feature rows and sample columns are observed matrix dimensions, not independent genes or donors.",
                            "Values retain their source processing and units; no normalization, imputation, or power analysis was performed.",
                            "Public file access does not itself establish a redistribution license or participant-consent clearance."]}


def build_inspections(out: Path, *, max_total_bytes: int = MAX_TOTAL_BYTES) -> dict:
    if out.exists():
        raise ValueError("Inspection output must be a new directory")
    if not 0 < max_total_bytes <= MAX_TOTAL_BYTES:
        raise ValueError("Download budget must be positive and at most 50 MiB")
    out.mkdir(parents=True, exist_ok=False)
    results, used = [], 0
    for accession, url in SOURCES.items():
        path = out / f"{accession}_series_matrix.txt.gz"
        try:
            downloaded = download_matrix(url, accession, path, max_total_bytes - used)
            used += downloaded["compressed_bytes"]
            inspection = inspect_matrix(path)
            (out / f"{accession}_inspection.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=True), encoding="utf-8")
            results.append({**downloaded, "status": "PASS", "record_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                            "features": inspection["features"], "sample_columns": inspection["sample_columns"],
                            "missing_cells": inspection["missing_cells"], "nonfinite_cells": inspection["nonfinite_cells"],
                            "license_status": "NOT_ESTABLISHED_FROM_MATRIX", "access_status": "PUBLIC_FILE_RETRIEVED"})
        except (OSError, ValueError, requests.RequestException) as exc:
            results.append({"accession": accession, "url": url, "status": "FAIL", "error": str(exc)})
            if not path.exists():
                # A failed transport can consume bytes without producing a file.
                # Stop rather than underestimate the remaining network budget.
                break
    manifest = {"schema_version": "processed-matrix-inspection-v1", "status": "PASS" if all(row["status"] == "PASS" for row in results) else "PARTIAL_OR_FAIL",
                "created_utc": datetime.now(timezone.utc).isoformat(), "compressed_bytes_downloaded": used,
                "max_total_bytes": max_total_bytes, "sources": results,
                "analysis_scope": "Observed processed matrix dimensions/missingness and reported sample metadata only; no scientific feasibility or independence claim",
                "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(out.iterdir()) if path.is_file()]}
    (out / "INSPECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL_BYTES)
    args = parser.parse_args()
    result = build_inspections(args.out, max_total_bytes=args.max_total_bytes)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
