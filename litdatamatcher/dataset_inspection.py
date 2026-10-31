"""Bounded dataset inspection with field-level source evidence.

GEO SOFT sample annotations are metadata, not measured outcomes. Local processed
tables are checked against their source variable dictionary. No sample count is
silently promoted to independent subjects or statistical power.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

import requests

from .acquisition_v2 import SnapshotClient, parse_series_matrix, soft_fields
from .data_plane import atomic_json
from .modality_contract import modality_contract

VERSION = "dataset-inspection-1.0"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
RESERVED = frozenset({"GSE279879"})
LIMITATIONS = [
    "Sample annotations establish reported attributes, not measured outcomes or causal effects.",
    "GSM records are technical records; biological independence and power remain unknown.",
    "Unreported fields remain unknown; partial sampling cannot establish absence.",
    "Public metadata access does not establish data-file access or reuse permission.",
]


def _observed(value, locator, *, mapping_type="exact"):
    return {
        "value": value,
        "status": "observed",
        "source_locator": locator,
        "reason": None,
        "mapping_type": mapping_type,
    }


def _unknown(reason):
    return {"value": None, "status": "unknown", "source_locator": None, "reason": reason}


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _bounded_int(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _parse_sample(text, sample_id, accession, locator):
    fields = soft_fields(text, "!Sample_")
    if fields.get("geo_accession") != [sample_id]:
        raise ValueError("Sample source identity mismatch")
    if accession not in fields.get("series_id", []):
        raise ValueError("Sample does not declare membership in requested study")
    # Contact information is not needed in a scientific profile. The immutable
    # response still retains the source document for reproducibility.
    fields = {key: value for key, value in fields.items() if not key.startswith("contact_")}
    attributes = {}
    for field, values in fields.items():
        if re.fullmatch(r"characteristics_ch\d+", field):
            for value in values:
                if ":" in value:
                    key, val = value.split(":", 1)
                    if key.strip() and val.strip():
                        attributes.setdefault(key.strip().casefold(), []).append(val.strip())
    return {
        "sample_id": sample_id,
        "study_id": accession,
        "fields": fields,
        "characteristics": attributes,
        "source_locator": locator,
    }


def profile_geo_samples(record, series, samples, *, source_locator, complete):
    """Attach only explicitly reported facts, keeping partial coverage visible."""
    result = deepcopy(record)
    caps = result.setdefault("capabilities", {})
    for name in (
        "species",
        "assay",
        "tissue",
        "intervention",
        "comparator",
        "outcome",
        "paired",
        "timepoint",
    ):
        caps.setdefault(name, _unknown("Not established by inspected sample metadata"))
    assays = series.get("type", [])
    assay_aliases = {
        "Expression profiling by array": "microarray",
        "Expression profiling by high throughput sequencing": "RNA-seq",
    }
    if assays:
        mapped = [assay_aliases.get(item, item) for item in assays]
        caps["assay"] = _observed(mapped, source_locator + "#!Series_type", mapping_type="synonym")
        result["assay_types"] = mapped
    species = sorted(
        {
            v
            for sample in samples
            for key, values in sample["fields"].items()
            if re.fullmatch(r"organism_ch\d+", key)
            for v in values
            if v
        }
    )
    # General study organism cannot be inferred from a bounded prefix of GSMs.
    if complete and species:
        caps["species"] = _observed(species, source_locator + "#all-samples/organism")
        result["organisms"] = species
    locators = "; ".join(sample["source_locator"] for sample in samples)
    attributes = {}
    for sample in samples:
        for key, values in sample["characteristics"].items():
            attributes.setdefault(key, set()).update(values)
    for key, values in sorted(attributes.items()):
        # These names deliberately describe reported annotations; they do not
        # pretend a metadata column is a measured phenotype endpoint.
        caps["sample_attribute:" + key] = _observed(sorted(values), locators)
    if attributes:
        caps["sample_metadata_field"] = _observed(sorted(attributes), locators)
    for target, keys in {
        "tissue": ("tissue", "cell type", "cell line"),
        "intervention": ("treatment", "stimulus", "stimulation"),
        "timepoint": ("time", "time point", "timepoint"),
    }.items():
        values = sorted({v for key in keys for v in attributes.get(key, ())})
        if values and complete:
            caps[target] = _observed(values, locators)
    result.update(
        samples=samples,
        series_metadata=series,
        sample_count_reported=len(series.get("sample_id", [])),
        samples_inspected=len(samples),
        sample_coverage_complete=complete,
        independent_units=None,
        independent_unit_status="unknown",
        profile_status="sample_annotations_parsed" if samples else "series_metadata_only",
    )
    result.setdefault("title", " ".join(series.get("title", [])))
    result.setdefault("summary", " ".join(series.get("summary", [])))
    result["modality_contract"] = modality_contract(result)
    return result


def inspect_dataset(record, cache_dir, *, offline=False, max_samples=40, max_bytes=8_000_000):
    """Inspect a GEO study using bounded cached SOFT metadata requests.

    Returns an enriched copy, including ``inspection_status`` and an inspection
    receipt. A failed source is never converted into a negative compatibility
    result. ``max_bytes`` bounds total response bytes consumed, including cache
    hits; ``max_samples`` bounds sample requests. No arbitrary URL is fetched.
    """
    _bounded_int(max_samples, "max_samples", 250)
    _bounded_int(max_bytes, "max_bytes", 50_000_000)
    if not isinstance(record, dict) or not isinstance(record.get("dataset_id"), str):
        raise ValueError("Dataset record requires a string dataset_id")
    accession = record["dataset_id"]
    result = deepcopy(record)
    if accession in RESERVED:
        raise ValueError("Reserved final-evaluation study is unavailable for iterative inspection")
    if not re.fullmatch(r"GSE[1-9]\d*", accession):
        result.update(
            inspection_status="UNSUPPORTED_SOURCE",
            inspection={
                "schema_version": VERSION,
                "dataset_id": accession,
                "reason": "This inspector supports GEO GSE accessions; other adapters remain metadata-only",
            },
        )
        return result
    client = SnapshotClient(
        Path(cache_dir) / "geo", offline=offline, max_bytes=max_bytes, max_retry_wait=10
    )
    snapshots, failures, samples = [], [], []
    consumed = 0

    def fetch(identity, view):
        nonlocal consumed
        if consumed >= max_bytes:
            raise ValueError("Total inspection response budget exhausted")
        client.max_bytes = max_bytes - consumed
        before = len(client.events)
        raw, snapshot = client.get(
            GEO_URL, {"acc": identity, "targ": "self", "form": "text", "view": view}
        )
        consumed += len(raw)
        if consumed > max_bytes:
            raise ValueError("Cached response exceeds total inspection response budget")
        snapshots.append(snapshot)
        if not offline and any("attempts" in event for event in client.events[before:]):
            time.sleep(0.35)
        return raw.decode("utf-8-sig"), snapshot["url"]

    try:
        text, source = fetch(accession, "full")
        series = soft_fields(text, "!Series_")
        if series.get("geo_accession") != [accession]:
            raise ValueError("Study source identity mismatch or unavailable source record")
        relatives = set(re.findall(r"\bGSE\d+\b", " ".join(series.get("relation", []))))
        if relatives & RESERVED:
            raise ValueError("Study is explicitly linked to reserved final evaluation")
        ids = series.get("sample_id", [])
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r"GSM[1-9]\d*", x) for x in ids):
            raise ValueError("Malformed or duplicate study sample identity")
        for sample_id in ids[:max_samples]:
            try:
                sample_text, sample_url = fetch(sample_id, "brief")
                samples.append(_parse_sample(sample_text, sample_id, accession, sample_url))
            except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
                failures.append(
                    {
                        "id": sample_id,
                        "stage": "sample",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                # A failed transfer may have consumed bytes; do not underestimate
                # remaining network budget by proceeding to another sample.
                break
        complete = bool(ids) and len(samples) == len(ids)
        result = profile_geo_samples(
            result, series, samples, source_locator=source, complete=complete
        )
        status = "INSPECTED_METADATA" if complete else "PARTIAL_METADATA"
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        failures.append(
            {
                "id": accession,
                "stage": "series",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        status = "SOURCE_FAILURE"
    result.update(
        inspection_status=status,
        inspection={
            "schema_version": VERSION,
            "dataset_id": accession,
            "status": status,
            "sources": snapshots,
            "source_events": client.events,
            "failures": failures,
            "samples_inspected": len(samples),
            "max_samples": max_samples,
            "response_bytes": consumed,
            "max_bytes": max_bytes,
            "processed_measurements_inspected": False,
            "limitations": LIMITATIONS,
        },
    )
    return result


def inspect_geo_matrix(path, accession, *, source_url, max_bytes=50_000_000):
    """Validate a local processed matrix using the existing native parser."""
    if not re.fullmatch(r"GSE[1-9]\d*", accession) or accession in RESERVED:
        raise ValueError("Invalid or reserved accession")
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Matrix file exceeds inspection budget")
    data = path.read_bytes()
    decoded = data
    if data[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            decoded = handle.read(250_000_001)
    if len(decoded) > 250_000_000:
        raise ValueError("Decoded matrix exceeds inspection bound")
    begin, end, identity, header = [], [], [], None
    for index, row in enumerate(
        csv.reader(io.StringIO(decoded.decode("utf-8-sig")), delimiter="\t")
    ):
        if row == ["!series_matrix_table_begin"]:
            begin.append(index)
        elif row == ["!series_matrix_table_end"]:
            end.append(index)
        elif begin and index == begin[0] + 1:
            header = row
        elif row and row[0] == "!Series_geo_accession":
            identity.append(row[1:])
    if (
        len(begin) != 1
        or len(end) != 1
        or begin[0] >= end[0]
        or identity != [[accession]]
        or not header
        or header[0] != "ID_REF"
    ):
        raise ValueError("Incomplete matrix or source accession identity mismatch")
    samples, inspection = parse_series_matrix(data, accession)
    if inspection["status"] != "PASS":
        raise ValueError("Matrix measurements did not pass native structural validation")
    inspection.update(
        schema_version=VERSION,
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_url=source_url,
        source_path=str(path.resolve()),
        samples=samples,
    )
    return inspection


def inspect_tabular_dataset(path, metadata, *, source_url, metadata_url, max_bytes=50_000_000):
    """Profile an unchanged CSV against an explicit repository variable dictionary.

    Exact field names and source descriptions are retained. Observed table cells
    can support availability constraints, never an inferred causal design.
    """
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Processed table exceeds inspection budget")
    variables = metadata.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("Source variable dictionary required")
    by_name = {item["name"]: item for item in variables}
    if len(by_name) != len(variables):
        raise ValueError("Duplicate variable dictionary names")
    if len(by_name) > 200:
        raise ValueError("Processed table exceeds 200-column inspection bound")
    missing = {"", "?", "na", "n/a", "null"}
    if metadata.get("missing_values_symbol"):
        missing.add(str(metadata["missing_values_symbol"]).casefold())
    stats = {
        name: {
            "observed_count": 0,
            "missing_count": 0,
            "nonfinite_count": 0,
            "numeric_count": 0,
            "minimum": None,
            "maximum": None,
            "categorical_counts": Counter(),
            "source_definition": definition,
        }
        for name, definition in by_name.items()
    }
    count, observed_rows = 0, []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or set(reader.fieldnames) != set(by_name)
        ):
            raise ValueError("CSV columns disagree with source variable dictionary")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row width mismatch")
            count += 1
            if count > 100_000:
                raise ValueError("Processed table exceeds 100,000-row inspection bound")
            observed = set()
            for name, cell in row.items():
                cell = cell.strip()
                stat, definition = stats[name], by_name[name]
                if cell.casefold() in missing:
                    stat["missing_count"] += 1
                    continue
                if definition.get("type") in {"Integer", "Continuous", "Real"}:
                    try:
                        number = float(cell)
                    except ValueError as exc:
                        raise ValueError(f"Invalid numeric cell in {name}") from exc
                    if not math.isfinite(number):
                        stat["nonfinite_count"] += 1
                        continue
                    stat["numeric_count"] += 1
                    stat["minimum"] = (
                        number if stat["minimum"] is None else min(stat["minimum"], number)
                    )
                    stat["maximum"] = (
                        number if stat["maximum"] is None else max(stat["maximum"], number)
                    )
                elif definition.get("type") == "Categorical" and definition.get("role") != "ID":
                    if (
                        cell not in stat["categorical_counts"]
                        and len(stat["categorical_counts"]) >= 1000
                    ):
                        raise ValueError("Categorical vocabulary exceeds inspection bound")
                    stat["categorical_counts"][cell] += 1
                stat["observed_count"] += 1
                observed.add(name)
            observed_rows.append(observed)
    if count == 0 or (
        metadata.get("num_instances") is not None and count != metadata["num_instances"]
    ):
        raise ValueError("Observed row count disagrees with source metadata")
    source_hash = _hash(path)
    locator = source_url + "#sha256=" + source_hash
    measured = sorted(
        name
        for name, stat in stats.items()
        if stat["observed_count"] and by_name[name].get("role") != "ID"
    )
    caps = {
        "measured_variable": _observed(measured, locator + "; " + metadata_url),
        "processed_measurements": _observed(True, locator),
    }
    for name in measured:
        caps["measured_variable:" + name] = _observed(True, locator + "&column=" + quote(name))
    for name, stat in stats.items():
        if stat["categorical_counts"]:
            caps["observed_group:" + name] = _observed(
                sorted(stat["categorical_counts"]), locator + "&column=" + quote(name)
            )
    # Co-observation is availability of cells in identical source rows, not
    # evidence that rows are independent participants or paired visits.
    targets = {name for name in measured if by_name[name].get("role") == "Target"}
    pairs = {tuple(sorted((a, b))) for a in measured for b in targets if a != b}
    joint = {
        f"{a}|{b}": sum(a in row and b in row for row in observed_rows) for a, b in sorted(pairs)
    }
    dataset_id = str(metadata.get("dataset_id") or ("UCI:" + str(metadata["uci_id"])))
    return {
        "dataset_id": dataset_id,
        "source": metadata.get("source", "UCI"),
        "title": metadata.get("name", dataset_id),
        "url": metadata.get("repository_url", metadata_url),
        "capabilities": caps,
        "availability": "PUBLIC_FILE_INSPECTED",
        "independent_units": None,
        "independent_unit_status": "unknown",
        "inspection_status": "INSPECTED_MEASUREMENTS",
        "inspection": {
            "schema_version": VERSION,
            "source_url": source_url,
            "metadata_url": metadata_url,
            "source_sha256": source_hash,
            "metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest(),
            "rows": count,
            "columns": len(by_name),
            "variables": stats,
            "joint_observed_rows": joint,
            "missing_cells": sum(x["missing_count"] for x in stats.values()),
            "nonfinite_cells": sum(x["nonfinite_count"] for x in stats.values()),
            "limitations": [
                "Rows are source records; subject independence and power are not inferred.",
                "No imputation, normalization, statistical test, or outcome estimation was performed.",
                "Source variable names, definitions, units and missingness are preserved.",
            ],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    value = inspect_dataset(
        {"dataset_id": args.accession},
        args.cache,
        offline=args.offline,
        max_samples=args.max_samples,
        max_bytes=args.max_bytes,
    )
    atomic_json(args.out, value)
    print(
        json.dumps(
            {
                "dataset_id": args.accession,
                "status": value["inspection_status"],
                "output": str(args.out),
            }
        )
    )
    return 1 if value["inspection_status"] == "SOURCE_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
