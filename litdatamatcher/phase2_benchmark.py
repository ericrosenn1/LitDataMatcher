"""Bounded real-metadata benchmarks with independently sourced contract labels.

This module never creates scientific source records, fits ranking weights, or
converts metadata compatibility into experimental answerability.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import psutil

from .data_plane import Catalog, atomic_json, atomic_write, digest
from .scientific_v2 import compile_evidence, rank_candidates
from .v2 import normalize_dataset

SCHEMA = "phase2_real_metadata_benchmark_v1"
INTERRUPTED_EXIT = 71
BOUNDS = {"query_p95_seconds": 5.0, "matching_seconds_per_1000": 30.0, "rss_bytes": 8 * 1024**3}
SPECIES = {
    "human": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "homo sapiens (human)": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "mus musculus (mouse)": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rattus norvegicus": "Rattus norvegicus",
    "danio rerio": "Danio rerio",
    "zebrafish": "Danio rerio",
    "drosophila melanogaster": "Drosophila melanogaster",
    "arabidopsis thaliana": "Arabidopsis thaliana",
    "escherichia coli": "Escherichia coli",
    "e. coli": "Escherichia coli",
    "gallus gallus": "Gallus gallus",
    "chicken": "Gallus gallus",
}
ASSAYS = {
    "rna-seq": "bulk_transcriptomics",
    "rna sequencing": "bulk_transcriptomics",
    "microarray": "bulk_transcriptomics",
    "transcriptomics": "bulk_transcriptomics",
    "single-cell rna-seq": "single_cell_transcriptomics",
    "scrna-seq": "single_cell_transcriptomics",
    "wgs": "sequencing_genomics",
    "whole genome sequencing": "sequencing_genomics",
    "genomics": "sequencing_genomics",
    "clinical study registry metadata": "clinical_registry",
    "clinical registry": "clinical_registry",
    "metagenomics": "microbiome_metagenomics",
    "shotgun metagenomics": "microbiome_metagenomics",
    "16s rrna sequencing": "microbiome_metagenomics",
    "proteomics": "proteomics",
    "mass spectrometry proteomics": "proteomics",
    "metabolomics": "metabolomics",
    "mass spectrometry metabolomics": "metabolomics",
}
ASSAYS.update({family: family for family in set(ASSAYS.values())})


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_acquisition_input(path: str | Path, receipt_path: str | Path) -> dict:
    """Bind a supplied input to the acquisition owner's executed receipt."""
    path, receipt_path = Path(path).resolve(), Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("status") not in {"PASS", "PASS_WITH_LIMITATIONS"}:
        raise ValueError("Acquisition receipt must report validated input")
    expected = None
    for name, metadata in receipt.get("files", {}).items():
        candidate = Path(name)
        candidate = candidate if candidate.is_absolute() else receipt_path.parent / candidate
        if candidate.resolve() == path:
            expected = metadata.get("sha256") if isinstance(metadata, dict) else metadata
    # Source-qualification receipts bind a single artifact instead of a file map.
    artifact = receipt.get("artifact", {})
    if expected is None and isinstance(artifact, dict) and artifact.get("path"):
        candidate = Path(artifact["path"])
        candidate = candidate if candidate.is_absolute() else receipt_path.parent / candidate
        if candidate.resolve() == path:
            expected = artifact.get("sha256")
    if expected is None and receipt.get("output_path"):
        candidate = Path(receipt["output_path"])
        candidate = candidate if candidate.is_absolute() else receipt_path.parent / candidate
        if candidate.resolve() == path:
            expected = receipt.get("output_sha256")
    if not isinstance(expected, str) or file_sha256(path).casefold() != expected.casefold():
        raise ValueError(f"Input lacks matching acquisition hash: {path}")
    return {
        "path": str(receipt_path),
        "sha256": file_sha256(receipt_path),
        "status": receipt["status"],
    }


def record_id(row: dict, kind: str) -> str:
    value = (
        row.get("dataset_id")
        if kind == "dataset"
        else row.get("document_id") or row.get("source_id")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing stable {kind} identity")
    return value.strip()


def unique_records(path: str | Path, kind: str) -> tuple[list[dict], dict]:
    """Read immutable input, counting identical duplicates and rejecting conflicts."""
    path = Path(path).resolve()
    records, repeated, input_count = {}, 0, 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        identity = record_id(row, kind)
        input_count += 1
        if identity in records:
            if digest(records[identity]) != digest(row):
                raise ValueError(f"Conflicting {kind} identity: {identity}")
            repeated += 1
        else:
            records[identity] = row
    if not records:
        raise ValueError(f"Empty {kind} input")
    ordered = sorted(
        records, key=lambda identity: (hashlib.sha256(identity.encode()).hexdigest(), identity)
    )
    return [records[identity] for identity in ordered], {
        "path": str(path),
        "sha256": file_sha256(path),
        "input_rows": input_count,
        "unique_ids": len(records),
        "identical_duplicate_rows": repeated,
        "identity_order_sha256": digest(ordered),
        "identity_kind": kind,
        "independent_biological_units": None,
    }


def source_locator(record: dict) -> str | None:
    provenance = record.get("source_provenance") or record.get("metadata", {}).get(
        "source_provenance"
    )
    candidates = provenance if isinstance(provenance, list) else [provenance]
    for item in candidates:
        if isinstance(item, dict):
            locator = item.get("source_locator") or item.get("source_url")
            if isinstance(locator, str) and locator.strip():
                return locator
    return None


def reference_values(record: dict, field: str) -> tuple[set[str], list[str]]:
    """Independent literal source lookup; never calls product compatibility code."""
    key, table = ("organisms", SPECIES) if field == "species" else ("assay_types", ASSAYS)
    raw = record.get(key, [])
    if not isinstance(raw, list):
        return set(), ["malformed_source_field"]
    if not source_locator(record):
        return set(), ["missing_source_locator"]
    if not raw:
        return set(), ["unreported_source_field"]
    known, reasons = set(), []
    for item in raw:
        normalized = " ".join(str(item).casefold().split())
        if normalized in table:
            known.add(table[normalized])
        else:
            reasons.append("unmapped_reference_value:" + str(item))
    return known, reasons


def reference_label(requirements: list[dict], record: dict) -> dict:
    states = []
    for requirement in requirements:
        field, expected = requirement["field"], requirement["expected"]
        known, reasons = reference_values(record, field)
        state = "MATCH" if expected in known else "UNKNOWN" if reasons or not known else "MISMATCH"
        states.append(
            {
                "field": field,
                "expected": expected,
                "state": state,
                "recognized_source_values": sorted(known),
                "reason_codes": reasons,
                "source_field": "organisms" if field == "species" else "assay_types",
            }
        )
    label = (
        "OBSERVED_MISMATCH"
        if any(x["state"] == "MISMATCH" for x in states)
        else (
            "UNKNOWN"
            if not states or any(x["state"] == "UNKNOWN" for x in states)
            else "OBSERVED_FIT"
        )
    )
    return {
        "label": label,
        "label_origin": "source_determined",
        "field_checks": states,
        "dataset_id": record["dataset_id"],
        "source_locator": source_locator(record),
        "source_record_sha256": digest(record),
        "meaning": "Narrow metadata compatibility only",
    }


def build_queries(records: list[dict], maximum: int = 24) -> list[dict]:
    """Freeze source-selected capability queries before examining ranked outputs."""
    selected = {}
    for record in sorted(records, key=lambda row: row["dataset_id"]):
        species, _ = reference_values(record, "species")
        modalities, _ = reference_values(record, "modality")
        combinations = [(("species", value),) for value in species]
        combinations += [(("modality", value),) for value in modalities]
        combinations += [
            (("species", species_value), ("modality", modality))
            for species_value in species
            for modality in modalities
        ]
        for values in combinations:
            selected.setdefault(values, record)
    result = []
    for values in sorted(selected, key=lambda item: (len(item), item))[:maximum]:
        anchor = selected[values]
        requirements = [
            {
                "field": field,
                "expected": value,
                "essential": True,
                "source_locator": source_locator(anchor),
            }
            for field, value in values
        ]
        query = " ".join(value.replace("_", " ") for _, value in values)
        result.append(
            {
                "query_id": "metadata_query_" + digest(values)[:16],
                "text": query,
                "requirements": requirements,
                "anchor_dataset_id": anchor["dataset_id"],
                "anchor_record_sha256": digest(anchor),
                "label_origin": "source_determined",
                "scope": "Source-selected metadata requirement; not an unresolved biological question",
            }
        )
    if not result:
        raise ValueError("No source-located recognized organism/modality queries")
    return result


def ranking_metrics(order: list[str], labels: dict[str, dict]) -> dict:
    """Keep unknowns in the ranking and out of negative-label denominators."""
    if len(order) != len(set(order)) or set(order) != set(labels):
        raise ValueError("Ranking must contain the complete candidate universe exactly once")
    positive = {key for key, value in labels.items() if value["label"] == "OBSERVED_FIT"}
    negative = {key for key, value in labels.items() if value["label"] == "OBSERVED_MISMATCH"}
    known = positive | negative
    top5, top10 = order[:5], order[:10]
    judged5 = [key for key in top5 if key in known]
    hits5 = sum(key in positive for key in judged5)
    hits10 = sum(key in positive for key in top10)
    known_order = [key for key in order if key in known]
    dcg = sum(
        (1 if key in positive else 0) / math.log2(index + 2)
        for index, key in enumerate(known_order[:5])
    )
    ideal = sum(1 / math.log2(index + 2) for index in range(min(5, len(positive))))
    first = next((index + 1 for index, key in enumerate(order) if key in positive), None)
    return {
        "candidate_count": len(order),
        "observed_fit_count": len(positive),
        "observed_mismatch_count": len(negative),
        "unknown_count": len(order) - len(known),
        "judged_precision_at_5": hits5 / len(judged5) if judged5 else None,
        "precision_at_5_numerator": hits5,
        "precision_at_5_denominator": len(judged5),
        "unknown_in_top5": len(top5) - len(judged5),
        "recall_at_10": hits10 / len(positive) if positive else None,
        "recall_at_10_numerator": hits10,
        "recall_at_10_denominator": len(positive),
        "reciprocal_rank": 1 / first if first else 0.0 if positive else None,
        "known_label_ndcg_at_5": dcg / ideal if ideal else None,
        "confirmed_invalid_top": bool(order and order[0] in negative),
        "unknown_top": bool(order and order[0] not in known),
        "metric_scope": "Source-determined metadata; unknown is unjudged, not negative",
    }


def search_text(record: dict) -> str:
    return (
        " ".join(
            str(record.get(key) or "") for key in ["title", "abstract", "description", "summary"]
        )
        + " "
        + " ".join(
            str(value) for key in ["organisms", "assay_types"] for value in record.get(key, [])
        )
    )


def measure(start: float, count: int) -> dict:
    elapsed = time.perf_counter() - start
    return {
        "items": count,
        "seconds": elapsed,
        "items_per_second": count / elapsed if elapsed else None,
    }


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "p50_seconds": None, "p95_seconds": None}
    return {
        "n": len(ordered),
        "p50_seconds": ordered[math.ceil(len(ordered) * 0.5) - 1],
        "p95_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "max_seconds": ordered[-1],
    }


class MemorySampler:
    """Sample parent and child RSS; an instantaneous peak can occur between samples."""

    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            rss = 0
            for item in [process, *process.children(recursive=True)]:
                try:
                    rss += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak = max(self.peak, rss)
            self.stop.wait(0.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()


@contextlib.contextmanager
def blocked_network():
    attempts = []

    def reject(*args, **_kwargs):
        attempts.append(str(args[-1])[:200] if args else "connection")
        raise OSError("Network disabled by the real-metadata benchmark")

    with (
        patch.object(socket.socket, "connect", reject),
        patch.object(socket.socket, "connect_ex", reject),
        patch.object(socket, "create_connection", reject),
    ):
        yield attempts


def catalog_digest(catalog: Catalog, kind: str) -> str:
    return digest(sorted(catalog.records(kind), key=lambda row: record_id(row, kind)))


def recovery_worker(
    input_path: str | Path, root: str | Path, receipt: str | Path, interrupt: bool
) -> None:
    """Commit partial real input, or resume only missing records in another process."""
    records, metadata = unique_records(input_path, "dataset")
    midpoint = max(1, len(records) // 2)
    catalog = Catalog(root)
    existing = {row["dataset_id"]: row for row in catalog.records("dataset")}
    expected = {row["dataset_id"]: row for row in records}
    if any(
        key not in expected or digest(row) != digest(expected[key]) for key, row in existing.items()
    ):
        raise ValueError("Recovery refuses stale or corrupted existing payloads")
    if interrupt and existing:
        raise ValueError("Interruption stage requires an empty task-owned catalog")
    written = []
    target = records[:midpoint] if interrupt else records
    for row in target:
        if row["dataset_id"] not in existing:
            catalog.upsert("dataset", row["dataset_id"], row, search_text=search_text(row))
            written.append(row["dataset_id"])
    rows = catalog.records("dataset")
    result = {
        "input_sha256": metadata["sha256"],
        "process_id": os.getpid(),
        "mode": "interrupt" if interrupt else "resume",
        "existing_count": len(existing),
        "inserted_count": len(written),
        "inserted_ids": written,
        "skipped_existing_ids": sorted(existing),
        "final_ids": sorted(row["dataset_id"] for row in rows),
        "final_payload_digest": catalog_digest(catalog, "dataset"),
        "current_count": catalog.conn.execute("SELECT COUNT(*) FROM current").fetchone()[0],
        "version_count": catalog.conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0],
    }
    atomic_json(receipt, result)
    if interrupt:
        os._exit(INTERRUPTED_EXIT)
    catalog.close()


def check_recovery(input_path: str | Path, root: Path, expected: list[dict]) -> dict:
    recovery_root = root / "recovery_catalog"
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    runs = []
    for mode in ["interrupt", "resume"]:
        receipt = root / f"recovery_{mode}.json"
        command = [
            sys.executable,
            "-B",
            "-m",
            "litdatamatcher.phase2_benchmark",
            "--worker",
            mode,
            "--datasets",
            str(input_path),
            "--root",
            str(recovery_root),
            "--receipt",
            str(receipt),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atomic_write(
            root / f"recovery_{mode}.log", (completed.stdout + completed.stderr).encode("utf-8")
        )
        expected_exit = INTERRUPTED_EXIT if mode == "interrupt" else 0
        if completed.returncode != expected_exit or not receipt.is_file():
            raise ValueError(
                f"Recovery {mode} exit {completed.returncode}, expected {expected_exit}: {completed.stderr[-1000:]}"
            )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        runs.append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "receipt_path": str(receipt),
                "receipt_sha256": file_sha256(receipt),
                **payload,
            }
        )
    first, last = runs
    expected_ids = sorted(row["dataset_id"] for row in expected)
    expected_digest = digest(sorted(expected, key=lambda row: row["dataset_id"]))
    checks = {
        "actual_separate_processes": len({first["process_id"], last["process_id"], os.getpid()})
        == 3,
        "partial_count_correct": first["current_count"] == max(1, len(expected) // 2),
        "resumed_only_missing": last["existing_count"] == first["current_count"]
        and last["inserted_count"] == len(expected) - first["current_count"],
        "preserved_existing_ids": set(first["final_ids"]) == set(last["skipped_existing_ids"]),
        "exact_ids": last["final_ids"] == expected_ids,
        "exact_payload_digest": last["final_payload_digest"] == expected_digest,
        "no_duplicate_records_or_versions": last["current_count"]
        == last["version_count"]
        == len(expected),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "partition_kind": "One source dataset identity per committed catalog transaction",
        "checks": checks,
        "child_runs": runs,
    }


def metadata_evidence(records: list[dict], source_path: str, proposition: str) -> list[dict]:
    return [
        {
            "evidence_id": "metadata_" + digest(record_id(row, "literature"))[:24],
            "source_id": record_id(row, "literature"),
            "publication_id": row.get("pmid") or row.get("doi"),
            "source_locator": source_locator(row)
            or f"{source_path}#id={record_id(row, 'literature')}",
            "related_proposition_id": proposition,
            "proposition_id": None,
            "source_record_sha256": digest(row),
            "role": "background",
            "direction": "neutral",
            "measurement_type": "metadata",
            "answers_question": False,
            "scope_match": "related",
            "title": row.get("title"),
            "label_origin": "source_determined",
            "limitation": "Performance workload context only; no biological relation extracted",
        }
        for row in records
    ]


def evaluate_rankings(
    queries: list[dict], raw: list[dict], catalog: Catalog, semantic_index=None
) -> tuple[list[dict], list[dict]]:
    normalized = [normalize_dataset(row) for row in raw]
    universe = sorted(row["dataset_id"] for row in raw)
    reports, checks = [], []
    for query in queries:
        labels = {row["dataset_id"]: reference_label(query["requirements"], row) for row in raw}
        lexical_hits = catalog.search("dataset", query["text"], 1000)
        lexical_order = lexical_hits + [key for key in universe if key not in set(lexical_hits)]
        lexical_scores = {key: -1.0 for key in universe}
        lexical_scores.update(
            {key: 1 - index / max(1, len(lexical_hits)) for index, key in enumerate(lexical_hits)}
        )
        core = rank_candidates(query["requirements"], normalized, lexical_scores)
        methods = {
            "lexical": lexical_order,
            "heuristic_without_eligibility": [
                row["dataset_id"]
                for row in sorted(core, key=lambda row: (-row["score"], row["dataset_id"]))
            ],
            "compatibility_only": [
                row["dataset_id"] for row in rank_candidates(query["requirements"], normalized)
            ],
            "compatibility_lexical": [row["dataset_id"] for row in core],
        }
        if semantic_index is not None:
            hits = semantic_index.search(query["text"], len(universe))
            scores = {row["id"]: row["score"] for row in hits}
            methods["pretrained_semantic"] = [row["id"] for row in hits]
            methods["compatibility_semantic"] = [
                row["dataset_id"]
                for row in rank_candidates(query["requirements"], normalized, scores)
            ]
        repeated = rank_candidates(
            query["requirements"], list(reversed(normalized)), lexical_scores
        )
        forbidden = [
            row["dataset_id"]
            for row in core
            if row["is_qualified"] and labels[row["dataset_id"]]["label"] == "OBSERVED_MISMATCH"
        ]
        false_exclusion = [
            row["dataset_id"]
            for row in core
            if not row["is_qualified"] and labels[row["dataset_id"]]["label"] == "OBSERVED_FIT"
        ]
        missing_promotions = [
            row["dataset_id"]
            for row in core
            if row["is_qualified"] and labels[row["dataset_id"]]["label"] == "UNKNOWN"
        ]
        check = {
            "query_id": query["query_id"],
            "forbidden_mismatch_promotions": forbidden,
            "observed_fit_exclusions": false_exclusion,
            "unknown_qualified": missing_promotions,
            "deterministic_reordered_replay": digest(core) == digest(repeated),
        }
        checks.append(check)
        reports.append(
            {
                "query": query,
                "candidate_count": len(universe),
                "candidate_ids_sha256": digest(universe),
                "labels": labels,
                "methods": {
                    name: {"order": order, "metrics": ranking_metrics(order, labels)}
                    for name, order in methods.items()
                },
                "compatibility_assessments": core,
                "checks": check,
            }
        )
    return reports, checks


def benchmark_point(
    root: Path,
    literature: list[dict],
    datasets: list[dict],
    queries: list[dict],
    literature_path: str,
) -> dict:
    catalog = Catalog(root / "catalog")
    ingested = {}
    for kind, records in [("literature", literature), ("dataset", datasets)]:
        start = time.perf_counter()
        for row in records:
            catalog.upsert(kind, record_id(row, kind), row, search_text=search_text(row))
        ingested[kind] = measure(start, len(records))
    start = time.perf_counter()
    with catalog.conn:
        catalog.conn.execute("INSERT INTO search(search) VALUES('optimize')")
    optimize = measure(start, len(literature) + len(datasets))
    latencies, replay_same = [], True
    query_texts = [query["text"] for query in queries]
    query_texts += [
        " ".join(re.findall(r"[A-Za-z]{3,}", str(row.get("title", "")))[:5])
        for row in literature[:6]
    ]
    for kind in ["literature", "dataset"]:
        for query in [text for text in query_texts if text]:
            expected = catalog.search(kind, query, 1000)
            for _ in range(20):
                start = time.perf_counter()
                observed = catalog.search(kind, query, 1000)
                latencies.append(time.perf_counter() - start)
                replay_same = replay_same and observed == expected
    start = time.perf_counter()
    normalized = [normalize_dataset(row) for row in datasets]
    normalization_seconds = time.perf_counter() - start
    for query in queries:
        rank_candidates(query["requirements"], normalized)
    matching = measure(start, len(queries) * len(datasets))
    matching["normalization_seconds_included"] = normalization_seconds
    matching["seconds_per_1000_assessments"] = matching["seconds"] * 1000 / matching["items"]
    proposition = "catalog_metadata_context_workload"
    question = {"question_id": "metadata_context_workload", "proposition_id": proposition}
    evidence = metadata_evidence(literature, literature_path, proposition)
    start = time.perf_counter()
    bundle = compile_evidence(
        question,
        evidence,
        "2026-09-13",
        [{"source": "bounded acquired metadata", "status": "partial"}],
    )
    compilation = measure(start, len(evidence))
    replay = compile_evidence(
        question,
        list(reversed(evidence)),
        "2026-09-13",
        [{"source": "bounded acquired metadata", "status": "partial"}],
    )
    cache = {}
    for kind, records in [("literature", literature), ("dataset", datasets)]:
        expected = {record_id(row, kind): digest(row) for row in records}
        start = time.perf_counter()
        observed = {record_id(row, kind): digest(row) for row in catalog.records(kind)}
        hits = sum(observed.get(key) == value for key, value in expected.items())
        cache[kind] = {
            **measure(start, len(expected)),
            "hits": hits,
            "misses": len(expected) - hits,
            "hit_rate": hits / len(expected),
            "payloads_identical": observed == expected,
            "reinserted_records": 0,
        }
    record_digests = {kind: catalog_digest(catalog, kind) for kind in ["literature", "dataset"]}
    catalog.close()
    checks = {
        "query_replay_exact": replay_same,
        "compiler_replay_exact": digest(bundle) == digest(replay),
        "compiler_no_direct_claim": bundle["independent_support_count"] is None
        and bundle["gap_status"] == "insufficient-coverage",
        "payload_cache_exact": all(item["payloads_identical"] for item in cache.values()),
    }
    return {
        "literature_count": len(literature),
        "dataset_count": len(datasets),
        "ingestion_including_fts": ingested,
        "fts_optimize": optimize,
        "fts_queries": quantiles(latencies),
        "matching": matching,
        "context_only_compilation": compilation,
        "compiler_result_sha256": digest(bundle),
        "cache_replay": cache,
        "catalog_payload_digests": record_digests,
        "checks": checks,
        "disk_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
    }


def run_benchmark(
    literature_path: str | Path,
    datasets_path: str | Path,
    acquisition_receipt: str | Path,
    output: str | Path,
    protocol: str | Path,
    model_dir: str | Path | None = None,
    dataset_supplements: list[tuple[str | Path, str | Path]] | None = None,
) -> dict:
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty versioned benchmark output directory")
    output.mkdir(parents=True, exist_ok=True)
    acquisition_receipt, protocol = Path(acquisition_receipt).resolve(), Path(protocol).resolve()
    acquisition = verify_acquisition_input(literature_path, acquisition_receipt)
    verify_acquisition_input(datasets_path, acquisition_receipt)
    start = time.perf_counter()
    literature, literature_metadata = unique_records(literature_path, "literature")
    datasets, datasets_metadata = unique_records(datasets_path, "dataset")
    input_files = [literature_metadata, datasets_metadata]
    acquisition_receipts = [acquisition]
    combined = {row["dataset_id"]: row for row in datasets}
    for supplement_path, supplement_receipt in dataset_supplements or []:
        acquisition_receipts.append(verify_acquisition_input(supplement_path, supplement_receipt))
        supplement, metadata = unique_records(supplement_path, "dataset")
        input_files.append(metadata)
        for row in supplement:
            identity = row["dataset_id"]
            if identity in combined and digest(combined[identity]) != digest(row):
                raise ValueError(f"Conflicting supplement identity: {identity}")
            combined[identity] = row
    combined_path = output / "BENCHMARK_DATASETS.jsonl"
    atomic_write(
        combined_path,
        b"".join(
            json.dumps(combined[key], sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
            for key in sorted(combined)
        ),
    )
    datasets, combined_metadata = unique_records(combined_path, "dataset")
    loading = measure(start, len(literature) + len(datasets))
    if len(literature) > 10000 or len(datasets) > 5000:
        raise ValueError("Workload exceeds the predeclared bounded metadata cap")
    queries = build_queries(datasets)
    plan = {
        "schema_version": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "protocol_path": str(protocol),
        "protocol_sha256": file_sha256(protocol),
        "input_files": input_files,
        "combined_dataset": combined_metadata,
        "acquisition_receipts": acquisition_receipts,
        "queries": queries,
        "reference_species": SPECIES,
        "reference_assays": ASSAYS,
        "bounds": BOUNDS,
        "selection": "Stable identity SHA256 ordering, nested subsets; query selection before ranking",
    }
    atomic_json(output / "BENCHMARK_PLAN.json", plan)
    atomic_json(output / "QUERY_MANIFEST.json", queries)
    points, semantics = [], {"status": "NOT_RUN", "reason": "No local model was requested"}
    with MemorySampler() as memory, blocked_network() as attempts:
        sizes = list(
            dict.fromkeys(
                [
                    (min(100, len(literature)), min(32, len(datasets))),
                    (min(500, len(literature)), min(100, len(datasets))),
                    (len(literature), len(datasets)),
                ]
            )
        )
        for literature_count, dataset_count in sizes:
            point_root = output / f"literature_{literature_count}_datasets_{dataset_count}"
            point = benchmark_point(
                point_root,
                literature[:literature_count],
                datasets[:dataset_count],
                queries,
                str(literature_path),
            )
            atomic_json(point_root / "POINT_RECEIPT.json", point)
            points.append(point)
        semantic_index = None
        if model_dir is not None:
            from .semantic_runtime import PretrainedSemanticIndex

            start = time.perf_counter()
            semantic_index = PretrainedSemanticIndex(model_dir, device="cpu")
            model_load = measure(start, 1)
            start = time.perf_counter()
            semantic_index.fit(
                [
                    {"id": row["dataset_id"], "text": search_text(row).strip() or row["dataset_id"]}
                    for row in datasets
                ]
            )
            semantics = {
                "status": "PASS",
                "model": semantic_index.manifest,
                "device": "cpu",
                "model_load": model_load,
                "encoding": measure(start, len(datasets)),
                "text_limit_tokens": 256,
                "vectors_sha256": hashlib.sha256(semantic_index.vectors.tobytes()).hexdigest(),
                "network_policy": "local_files_only and blocked socket connections",
            }
        final_catalog = Catalog(
            output / f"literature_{len(literature)}_datasets_{len(datasets)}" / "catalog"
        )
        rankings, ranking_checks = evaluate_rankings(
            queries, datasets, final_catalog, semantic_index
        )
        final_catalog.close()
        atomic_json(output / "RANKING_EVALUATION.json", rankings)
        recovery = check_recovery(combined_path, output, datasets)
        recovery["clean_catalog_payload_digest"] = points[-1]["catalog_payload_digests"]["dataset"]
        recovery["checks"]["equal_to_clean_catalog"] = (
            recovery["child_runs"][-1]["final_payload_digest"]
            == recovery["clean_catalog_payload_digest"]
        )
        recovery["status"] = "PASS" if all(recovery["checks"].values()) else "FAIL"
    coverage = Counter()
    for report in rankings:
        for label in report["labels"].values():
            coverage[label["label"]] += 1
            for field in label["field_checks"]:
                if field["state"] == "MISMATCH":
                    coverage[
                        "wrong_organism" if field["field"] == "species" else "wrong_modality"
                    ] += 1
    engineering = {
        "query_p95_within_bound": all(
            p["fts_queries"]["p95_seconds"] <= BOUNDS["query_p95_seconds"] for p in points
        ),
        "matching_within_bound": all(
            p["matching"]["seconds_per_1000_assessments"] <= BOUNDS["matching_seconds_per_1000"]
            for p in points
        ),
        "sampled_rss_within_bound": memory.peak <= BOUNDS["rss_bytes"],
        "deterministic_points": all(all(p["checks"].values()) for p in points),
        "separate_process_recovery": recovery["status"] == "PASS",
        "no_network_attempts": not attempts,
    }
    scientific = {
        "observed_positive_cases": coverage["OBSERVED_FIT"] > 0,
        "wrong_modality_negatives": coverage["wrong_modality"] > 0,
        "wrong_organism_negatives": coverage["wrong_organism"] > 0,
        "unknown_cases_retained": coverage["UNKNOWN"] > 0,
        "no_forbidden_promotions": not any(
            row["forbidden_mismatch_promotions"] for row in ranking_checks
        ),
        "no_observed_fit_exclusion": not any(
            row["observed_fit_exclusions"] for row in ranking_checks
        ),
        "no_unknown_qualified": not any(row["unknown_qualified"] for row in ranking_checks),
        "exact_rank_replay": all(row["deterministic_reordered_replay"] for row in ranking_checks),
    }
    repository = Path(__file__).resolve().parents[1]
    source_files = [
        "litdatamatcher/phase2_benchmark.py",
        "litdatamatcher/data_plane.py",
        "litdatamatcher/scientific_v2.py",
        "litdatamatcher/modality_contract.py",
        "litdatamatcher/v2.py",
        "litdatamatcher/semantic_runtime.py",
        "litdatamatcher/ontology.py",
        "litdatamatcher/schemas.py",
        "scripts/v2/run_phase2_benchmark.py",
    ]
    receipt = {
        "schema_version": SCHEMA,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": plan["command"],
        "working_directory": plan["working_directory"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "source_file_sha256": {name: file_sha256(repository / name) for name in source_files},
        "plan_path": str(output / "BENCHMARK_PLAN.json"),
        "plan_sha256": file_sha256(output / "BENCHMARK_PLAN.json"),
        "input_files": input_files,
        "combined_dataset": combined_metadata,
        "loading": loading,
        "points": points,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_bytes": psutil.virtual_memory().total,
            "backend": "CPU; SQLite FTS5; existing application matcher/compiler",
        },
        "memory": {
            "sampled_process_tree_peak_rss_bytes": memory.peak,
            "sample_interval_seconds": 0.02,
            "limitation": "Transient peaks between samples can be missed",
        },
        "semantic_baseline": semantics,
        "recovery": recovery,
        "network_attempts": attempts,
        "metadata_label_counts": dict(coverage),
        "engineering_checks": engineering,
        "metadata_evaluation_checks": scientific,
        "engineering_status": "PASS" if all(engineering.values()) else "FAIL",
        "metadata_evaluation_status": "PASS"
        if all(scientific.values())
        else "FAIL_OR_MISSING_COVERAGE",
        "status": "PASS"
        if all(engineering.values()) and all(scientific.values())
        else "FAIL_OR_MISSING_COVERAGE",
        "expert_validation": "PENDING_EXPERT_REVIEW",
        "calibration_status": "UNCALIBRATED_HEURISTIC",
        "limitations": [
            "Source-selected metadata compatibility, not held-out biological answerability or expert validation",
            "Queries derive from the evaluated catalog, so source-anchor self-rediscovery is included",
            "Catalog accession counts do not establish independent studies or subjects",
            "Compiler throughput uses real context metadata, not fresh model-extracted scientific claims",
            "Partial acquisition universe; source absence is not global novelty",
            "Original frozen alpha and sealed holdout were not executed or modified",
        ],
    }
    receipt["artifact_hashes"] = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".sqlite3"}
    }
    atomic_json(output / "BENCHMARK_RECEIPT.json", receipt)
    return receipt


def _worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["interrupt", "resume"], required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    recovery_worker(args.datasets, args.root, args.receipt, args.worker == "interrupt")


if __name__ == "__main__":
    _worker_main()
