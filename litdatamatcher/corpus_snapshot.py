"""Read-only qualification of a portable, previously acquired PubMed corpus.

The auditor writes a new receipt; it never repairs or changes shared source
files. Counts, schema, source identity, and hashes are independently checked.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .data_plane import atomic_json, digest

PUBMED_FIELDS = {
    "pmid",
    "doi",
    "pmcid",
    "title",
    "abstract",
    "language",
    "journal",
    "publication_date",
    "publication_types_json",
    "mesh_json",
    "source_shard",
}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file in bounded memory."""
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Corpus path escapes declared root: {relative!r}")
    return path


def _first_raw_record(path: Path) -> tuple[str | None, str | None]:
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag in {"PubmedArticle", "PubmedBookArticle"}:
                pmid, title = element.find(".//PMID"), element.find(".//ArticleTitle")
                return (
                    "".join(pmid.itertext()).strip() if pmid is not None else None,
                    "".join(title.itertext()).strip() if title is not None else None,
                )
    return None, None


def _qualify_shard(task: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    started, root = time.perf_counter(), Path(task["root"])
    result: dict[str, Any] = {"source": task["source"], "source_shard": task["name"], "issues": []}
    try:
        output = _inside(root, task["normalization"]["output_relative_path"])
        raw = _inside(root, task["raw_relative"])
        before, raw_before = output.stat(), raw.stat()
        parquet = pq.ParquetFile(output)
        if PUBMED_FIELDS - set(parquet.schema_arrow.names):
            raise ValueError("Missing required Parquet fields")
        if any(str(parquet.schema_arrow.field(field).type) != "string" for field in PUBMED_FIELDS):
            raise ValueError("Unexpected field type; expected declared string schema")
        identifiers = parquet.read(columns=["pmid", "source_shard"]).to_pydict()
        present = [value for value in identifiers["pmid"] if value]
        if len(present) != len(identifiers["pmid"]):
            result["issues"].append("missing_pmid_rows")
        wrong_source = sum(value != task["name"] for value in identifiers["source_shard"])
        if wrong_source:
            result["issues"].append(f"wrong_source_shard_rows:{wrong_source}")
        if parquet.metadata.num_rows != task["normalization"].get("records"):
            result["issues"].append("normalization_count_disagrees_with_parquet")
        if len(present) != len(set(present)):
            result["issues"].append("within_shard_duplicate_pmids")
        if task["acquisition"].get("status") != "CHECKSUM_PASS":
            result["issues"].append("acquisition_not_checksum_pass")
        if task["acquisition"].get("bytes") != raw_before.st_size:
            result["issues"].append("raw_size_disagrees_with_acquisition")
        raw_checksum = file_digest(raw, "md5") if task["verify_raw"] else None
        if task["verify_raw"] and raw_checksum != task["acquisition"].get("md5"):
            result["issues"].append("raw_md5_disagrees_with_acquisition")
        raw_pmid, raw_title = _first_raw_record(raw)
        first = parquet.read_row_group(0, columns=["pmid", "title"]).slice(0, 1).to_pylist()
        if not first or first[0]["pmid"] != raw_pmid or first[0]["title"] != raw_title:
            result["issues"].append("first_record_raw_parquet_mismatch")
        checksum = file_digest(output)
        after, raw_after = output.stat(), raw.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            result["issues"].append("parquet_changed_during_audit")
        if (raw_before.st_size, raw_before.st_mtime_ns) != (
            raw_after.st_size,
            raw_after.st_mtime_ns,
        ):
            result["issues"].append("raw_changed_during_audit")
        result.update(
            relative_path=output.relative_to(root).as_posix(),
            raw_relative_path=raw.relative_to(root).as_posix(),
            rows=parquet.metadata.num_rows,
            missing_pmid_count=len(identifiers["pmid"]) - len(present),
            bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,
            sha256=checksum,
            raw_bytes=raw_before.st_size,
            raw_mtime_ns=raw_before.st_mtime_ns,
            raw_md5=raw_checksum,
            acquisition_md5=task["acquisition"].get("md5"),
        )
    except Exception as error:
        result["issues"].append(f"{type(error).__name__}:{error}")
    result["status"] = "FAIL" if result["issues"] else "PASS"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def qualify_corpus(
    root: str | Path,
    out: str | Path,
    *,
    workers: int = 2,
    verify_raw: bool = False,
    include_updates: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Write a new integrity receipt; partial audits are explicitly PARTIAL."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to replace a corpus audit: {out}")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 to 8")
    if max_shards is not None and (type(max_shards) is not int or max_shards < 1):
        raise ValueError("max_shards must be a positive integer")
    marker = json.loads((root / "CORPUS_ROOT.json").read_text(encoding="utf-8"))
    if not marker.get("store_uuid"):
        raise ValueError("Corpus marker lacks store_uuid")
    tasks, manifests, issues = [], [], []
    for source in ["baseline", "updates"] if include_updates else ["baseline"]:
        prefix = f"pubmed_2026_{source}"
        acquired_path = root / "manifests/acquisition" / f"{prefix}_state.json"
        normalized_path = (
            root
            / "manifests/acquisition"
            / (
                "pubmed_2026_normalization_state.json"
                if source == "baseline"
                else "pubmed_2026_updates_normalization_state.json"
            )
        )
        acquired = json.loads(acquired_path.read_text(encoding="utf-8"))
        normalized = json.loads(normalized_path.read_text(encoding="utf-8")).get("shards", {})
        if not acquired.get("files"):
            issues.append(f"{source}:empty_acquisition_manifest")
        if acquired.get("expected_shards") is not None and acquired["expected_shards"] != len(
            acquired.get("files", {})
        ):
            issues.append(f"{source}:acquisition_shard_count_mismatch")
        manifests.extend(
            {"relative_path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
            for path in (acquired_path, normalized_path)
        )
        for name, item in sorted(acquired.get("files", {}).items()):
            if name not in normalized or normalized[name].get("status") != "NORMALIZED":
                issues.append(f"{source}/{name}:not_normalized")
                continue
            tasks.append(
                {
                    "root": str(root),
                    "source": source,
                    "name": name,
                    "normalization": normalized[name],
                    "acquisition": item,
                    "raw_relative": f"raw/pubmed/2026_{source}/{name}",
                    "verify_raw": verify_raw,
                }
            )
        issues.extend(
            f"{source}/{name}:no_acquisition_record"
            for name in sorted(set(normalized) - set(acquired.get("files", {})))
        )
    available = len(tasks)
    if not tasks:
        issues.append("no_qualifiable_shards")
    tasks = tasks[:max_shards] if max_shards is not None else tasks
    out.mkdir(parents=True)
    started, results = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_qualify_shard, tasks):
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(tasks):
                progress = {
                    "completed": len(results),
                    "total": len(tasks),
                    "failed": sum(row["status"] != "PASS" for row in results),
                }
                atomic_json(out / "progress.json", progress)
                print(f"Corpus audit: {progress}", flush=True)
    for manifest in manifests:
        if file_digest(_inside(root, manifest["relative_path"])) != manifest["sha256"]:
            issues.append(f"manifest_changed:{manifest['relative_path']}")
    failed = [row["source_shard"] for row in results if row["status"] != "PASS"]
    stable = [
        {key: value for key, value in row.items() if key != "elapsed_seconds"} for row in results
    ]
    receipt = {
        "schema_version": "corpus-snapshot-1.0",
        "store_uuid": marker["store_uuid"],
        "status": "FAIL" if failed or issues else "PARTIAL" if len(tasks) != available else "PASS",
        "source_root_hint": str(root),
        "shards_available": available,
        "shards_audited": len(results),
        "rows": sum(row.get("rows", 0) for row in results),
        "missing_pmid_count": sum(row.get("missing_pmid_count", 0) for row in results),
        "raw_checksums_recomputed": verify_raw,
        "input_manifests": manifests,
        "shards": results,
        "issues": issues,
        "failed_shards": failed,
        "snapshot_sha256": digest(
            {"store_uuid": marker["store_uuid"], "inputs": manifests, "shards": stable}
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "limitations": [
            "Raw/parquet first-record reconciliation is sampled per shard, not full XML semantic replay.",
            "Across-shard repeated PMIDs from updates must be version-resolved downstream.",
            "Input integrity is not biomedical truth or model validity.",
        ],
    }
    atomic_json(out / "CORPUS_SNAPSHOT.json", receipt)
    return receipt
