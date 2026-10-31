"""Bounded native-CheckpointRun demonstration; no models or scientific claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psutil

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.question_discovery import _source_lifecycle, discover_questions
from litdatamatcher.workflow_checkpoint import CheckpointRun, file_identity

TOPIC = r"\b(?:breast|mammary)\s+(?:cancers?|carcinomas?|neoplasms?|tumou?rs?)\b"
EXCLUDED = r"\b(?:viruses|virus|viral|pathogens?|bacteria\w*|infect\w*|vaccin\w*|SARS\w*|HIV|hepatitis|tuberculosis)\b"
QUERY = '"breast cancer" OR "breast carcinoma" OR "mammary carcinoma" OR "breast neoplasms"'
SQL = (
    "SELECT c.pmid FROM literature_fts JOIN current_records c ON c.rowid=literature_fts.rowid "
    "JOIN versions v ON v.id=c.version_id WHERE literature_fts MATCH ? "
    "AND length(trim(v.abstract)) > 0 AND CAST(c.pmid AS INTEGER) < ? "
    "ORDER BY CAST(c.pmid AS INTEGER) DESC LIMIT ?"
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def checksum(path):
    return file_identity(path)["sha256"]


def null_count(value):
    if value is None:
        return 1
    if isinstance(value, dict):
        return sum(null_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(null_count(item) for item in value)
    return 0


def implementation():
    package = Path(sys.modules[discover_questions.__module__].__file__).parent
    names = ("question_discovery.py", "question_compiler.py", "question_compiler_rules_v1.json")
    return {name: checksum(package / name) for name in names}


def prepare(database, seed, output, count=10000):
    """Reuse a frozen first batch; append lower-PMID batches from short snapshots."""
    from litdatamatcher.literature_index import LiteratureIndex

    if count % 1000 or not 1000 <= count <= 10000:
        raise ValueError("Preparation count must be a multiple of 1000, at most 10000")
    output.mkdir(parents=True, exist_ok=False)
    seed_rows = rows(seed / "INPUT_DOCUMENTS.jsonl")
    seed_receipt = read(seed / "RECEIPT.json")
    if len(seed_rows) != 1000 or seed_receipt["implementation_sha256"] != implementation():
        raise ValueError("Seed must be the matching previously validated 1000-record run")
    if seed_receipt["result_sha256"] != checksum(seed / "QUESTIONS.json"):
        raise ValueError("Seed result changed")
    if seed_receipt["corrected_input_sha256"] != checksum(seed / "INPUT_DOCUMENTS.jsonl"):
        raise ValueError("Seed inputs changed")
    protocol = seed_receipt["selection_protocol"]
    if protocol["topic_regex"] != TOPIC or protocol["excluded_wording_regex"] != EXCLUDED:
        raise ValueError("Seed sampling filters differ")
    first = output / "batch-01.jsonl"
    shutil.copyfile(seed / "INPUT_DOCUMENTS.jsonl", first)
    batches = [
        {
            "batch": 1,
            "input": file_identity(first),
            "documents": 1000,
            "snapshot": seed_receipt["snapshot"],
            "selection_protocol": protocol,
            "seed_result": file_identity(seed / "QUESTIONS.json"),
            "seed_receipt": file_identity(seed / "RECEIPT.json"),
        }
    ]
    seen = {row["document_id"] for row in seed_rows}
    cutoff = min(int(row["pmid"]) for row in seed_rows)
    for number in range(2, count // 1000 + 1):
        selected, rejected = [], []
        start = time.perf_counter()
        with LiteratureIndex(database, read_only=True) as index:
            index.conn.execute("BEGIN")
            snapshot = index.snapshot_identity()
            identifiers = [row[0] for row in index.conn.execute(SQL, (QUERY, cutoff, 1500))]
            for pmid in identifiers:
                if len(selected) == 1000:
                    break
                doc = index.get(pmid)
                if not doc["abstract"].strip() or doc["text"] != doc["abstract"]:
                    raise ValueError("Native nonempty abstract source view required")
                text = doc["title"] + "\n" + doc["abstract"]
                reason = (
                    "outside_explicit_topic"
                    if not re.search(TOPIC, text, re.I)
                    else "excluded_pathogen_wording"
                    if re.search(EXCLUDED, text, re.I)
                    else None
                )
                if reason:
                    rejected.append({"document_id": doc["document_id"], "reason": reason})
                else:
                    if doc["document_id"] in seen:
                        raise ValueError("Native record repeated across batches")
                    seen.add(doc["document_id"])
                    selected.append(doc)
        if len(selected) != 1000:
            raise ValueError("Bounded 1500-candidate query yielded fewer than 1000 records")
        path = output / f"batch-{number:02d}.jsonl"
        path.write_text(
            "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in selected),
            encoding="utf-8",
        )
        batches.append(
            {
                "batch": number,
                "input": file_identity(path),
                "documents": len(selected),
                "snapshot": snapshot,
                "selection_protocol": {
                    "sql": SQL,
                    "parameters": [QUERY, cutoff, 1500],
                    "topic_regex": TOPIC,
                    "excluded_wording_regex": EXCLUDED,
                    "considered": len(selected) + len(rejected),
                    "rejections": rejected,
                    "selection_seconds": time.perf_counter() - start,
                },
            }
        )
        cutoff = min(int(doc["pmid"]) for doc in selected)
        atomic_json(
            output / "PREPARATION_PROGRESS.json",
            {"completed_batches": number, "documents": len(seen)},
        )
        print(
            json.dumps(
                {
                    "prepared_batch": number,
                    "documents": len(selected),
                    "snapshot_shards": snapshot["applied_shards"],
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": "question-scale-batches-1.0",
        "created_utc": now(),
        "documents": count,
        "batch_size": 1000,
        "batches": batches,
        "sampling_scope": "Frozen first batch reused; subsequent nonempty breast-oncology abstracts below its numeric PMID cutoff. Each batch has its own committed partial snapshot; this is not one corpus snapshot or a representative sample.",
        "models_used": False,
    }
    atomic_json(output / "INPUT_MANIFEST.json", manifest)
    return manifest


def validate_result(documents, result):
    by_id = {row["document_id"]: row for row in documents}
    if len(by_id) != len(documents) or len(result["document_statuses"]) != len(documents):
        raise ValueError("Document accounting mismatch")
    if {row["document_id"] for row in result["document_statuses"]} != set(by_id):
        raise ValueError("Document status identities mismatch")
    if not all(doc["text"] == doc["abstract"] and doc["abstract"].strip() for doc in documents):
        raise ValueError("Source view must be unchanged nonempty native abstract")
    for question in result["questions"]:
        doc = by_id[question["source_document_id"]]
        text, span = doc["abstract"], question["evidence_span"]
        if not (
            text[span["start"] : span["end"]]
            == span["text"]
            == question["text"]
            == question["question"]
            and hashlib.sha256(text.encode()).hexdigest() == question["source_text_sha256"]
            and doc["document_id"] == "PMID:" + str(doc["pmid"])
            and str(question["citation"]["pmid"]) == str(doc["pmid"])
            and all(question["citation"][key] == doc.get(key) for key in question["citation"])
            and question["source_provenance"] == doc["source_provenance"]
            and question["source_locator"]
            == f"{doc['source_locator']}#chars={span['start']}:{span['end']}"
            and question["novelty_status"] == "NOT_ESTABLISHED"
            and question["model_inferred"] is False
        ):
            raise ValueError("Native source fidelity failed")
    question_ids = [row["question_id"] for row in result["questions"]]
    if len(question_ids) != len(set(question_ids)) or sorted(question_ids) != sorted(
        identifier for cluster in result["clusters"] for identifier in cluster["question_ids"]
    ):
        raise ValueError("Question/cluster identities mismatch")
    statuses = Counter(row["status"] for row in result["document_statuses"])
    coverage = result["coverage"]
    if (
        coverage["failed_documents"]
        or coverage["truncated"]
        or coverage["global_novelty_established"]
    ):
        raise ValueError("Failures, truncation or unsupported novelty in scale output")
    if coverage["processed_documents"] != statuses["EMPTY"] + statuses["SUCCESS"]:
        raise ValueError("Processed-text accounting mismatch")
    for status in result["document_statuses"]:
        expected_exclusion = _source_lifecycle(by_id[status["document_id"]])["excluded"]
        if expected_exclusion != (status["status"] == "EXCLUDED_SOURCE_LIFECYCLE"):
            raise ValueError("Native lifecycle policy does not match document status")
        if status["status"] == "EXCLUDED_SOURCE_LIFECYCLE" and status.get("retained_count", 0):
            raise ValueError("Excluded lifecycle source produced a retained question")
    return {
        "status": "PASS",
        "documents": len(documents),
        "question_spans_checked": len(question_ids),
        "document_status_counts": dict(statuses),
        "literal_fidelity": "PASS",
        "native_abstract_only": "PASS",
        "native_lifecycle_policy": "PASS",
        "input_null_values_preserved": null_count(documents),
        "result_null_values_retained": null_count(result),
    }


def compute(documents):
    process, stopped = psutil.Process(), threading.Event()
    baseline = process.memory_info().rss
    samples = [baseline]

    def monitor():
        while not stopped.wait(0.02):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = discover_questions(documents, max_documents=1000, max_questions=1000)
    finally:
        duration = time.perf_counter() - start
        samples.append(process.memory_info().rss)
        stopped.set()
        thread.join()
    return {
        "result": result,
        "duration_seconds": duration,
        "created_utc": now(),
        "origin": "NEW_DETERMINISTIC_EXTRACTION",
        "memory": {
            "baseline_rss_bytes": baseline,
            "sampled_peak_rss_bytes": max(samples),
            "peak_increase_bytes": max(samples) - baseline,
            "sample_count": len(samples),
            "sample_interval_ms": 20,
        },
        "validation": validate_result(documents, result),
    }


def run_batches(manifest_path, output, *, resume=False, stop_after=None):
    manifest = read(manifest_path)
    inputs = manifest["batches"]
    identities = [file_identity(batch["input"]["path"]) for batch in inputs]
    if identities != [batch["input"] for batch in inputs]:
        raise ValueError("Frozen input changed; resume is not allowed")
    if not 1 <= len(inputs) <= 10 or sum(batch["documents"] for batch in inputs) > 10000:
        raise ValueError("At most ten batches and 10000 inputs are allowed")
    if (
        any(
            type(batch["documents"]) is not int or not 1 <= batch["documents"] <= 1000
            for batch in inputs
        )
        or sum(batch["documents"] for batch in inputs) != manifest["documents"]
        or [batch["batch"] for batch in inputs] != list(range(1, len(inputs) + 1))
    ):
        raise ValueError("Manifest counts and consecutive batch numbers must agree")
    fingerprint = {
        "manifest": file_identity(manifest_path),
        "inputs": identities,
        "implementation": implementation(),
        "dependencies": [
            file_identity(path)
            for path in sorted(
                Path(sys.modules[discover_questions.__module__].__file__).parent.glob("*.py")
            )
        ],
        "python_version": platform.python_version(),
        "psutil_version": psutil.__version__,
        "runner": file_identity(__file__),
        "scope": "bounded deterministic source-view extraction; no models",
    }
    reused, computed, imported, receipts, all_ids = [], [], [], [], set()
    with CheckpointRun(output, fingerprint, resume) as checkpoint:
        if checkpoint.completed() is not None:
            return checkpoint.completed()
        checkpoint.cached("inputs", lambda: manifest)
        previous_stop = (
            read(output / "STOP_RECEIPT.json") if (output / "STOP_RECEIPT.json").exists() else None
        )
        for batch in inputs:
            number = batch["batch"]
            name = f"extract-{number:02d}"
            documents = rows(batch["input"]["path"])
            if len(documents) != batch["documents"]:
                raise ValueError("Actual input rows differ from declared batch count")
            ids = {row["document_id"] for row in documents}
            if len(ids) != len(documents) or ids & all_ids:
                raise ValueError("Duplicate native IDs within or across chunks")
            all_ids.update(ids)
            value = checkpoint.load(name, parents=("inputs",))
            if value is not None:
                reused.append(number)
            else:
                if batch.get("seed_result"):
                    for key in ("seed_result", "seed_receipt"):
                        if file_identity(batch[key]["path"]) != batch[key]:
                            raise ValueError("Previously validated seed changed")
                    prior = read(batch["seed_receipt"]["path"])
                    if prior["implementation_sha256"] != implementation():
                        raise ValueError("Seed implementation changed")
                    result = read(batch["seed_result"]["path"])
                    value = {
                        "result": result,
                        "origin": "PRIOR_VALIDATED_RUN_REUSED",
                        "created_utc": prior["completed_utc"],
                        "duration_seconds": prior["extraction_and_clustering_seconds"],
                        "memory": prior["memory"],
                        "prior_result": batch["seed_result"],
                        "seed_identity_scope": "The prior receipt records discovery/compiler/rulebook hashes only; other historical dependency identity is not asserted. Native abstract fidelity and lifecycle policy are revalidated in this run.",
                        "validation": validate_result(documents, result),
                    }
                    imported.append(number)
                else:
                    value = compute(documents)
                    computed.append(number)
                checkpoint.save(name, value, parents=("inputs",))
            validate_result(documents, value["result"])
            receipts.append(
                {key: item for key, item in value.items() if key != "result"}
                | {
                    "batch": number,
                    "input_sha256": batch["input"]["sha256"],
                    "snapshot": batch.get("snapshot"),
                    "coverage": value["result"]["coverage"],
                    "checkpoint_artifact": checkpoint.state["stages"][name],
                }
            )
            if stop_after is not None and number >= stop_after:
                stage_evidence = {}
                for stage, info in checkpoint.state["stages"].items():
                    if stage.startswith("extract-"):
                        path = checkpoint.directory / info["file"]
                        stage_evidence[stage] = {
                            "sha256": checksum(path),
                            "mtime_ns": path.stat().st_mtime_ns,
                            "file": info["file"],
                        }
                atomic_json(
                    output / "STOP_RECEIPT.json",
                    {
                        "status": "PLANNED_BOUNDED_INTERRUPTION",
                        "stopped_utc": now(),
                        "completed_batch_count": len(receipts),
                        "newly_computed_batches": computed,
                        "imported_prior_batches": imported,
                        "checkpoint_loaded_batches": reused,
                        "stage_evidence": stage_evidence,
                        "resume_command": "Repeat the run command with --resume and without --stop-after",
                    },
                )
                raise SystemExit(75)
        total = Counter()
        if len(all_ids) != manifest["documents"]:
            raise ValueError("Actual aggregate input count differs from manifest")
        for row in receipts:
            total.update(row["validation"]["document_status_counts"])
        proof = {
            "planned_interruption_observed": previous_stop is not None,
            "loaded_without_extraction_this_invocation": reused,
            "new_extraction_calls_this_invocation": computed,
            "imported_prior_results_this_invocation": imported,
            "pre_stop_stage_hashes_and_mtimes_unchanged": None,
        }
        if previous_stop:
            for evidence in previous_stop["stage_evidence"].values():
                path = checkpoint.directory / evidence["file"]
                if (
                    checksum(path) != evidence["sha256"]
                    or path.stat().st_mtime_ns != evidence["mtime_ns"]
                ):
                    raise ValueError("Successful pre-stop stage was rewritten")
            proof["pre_stop_stage_hashes_and_mtimes_unchanged"] = True
            proof["stop_receipt"] = previous_stop
        summary = {
            "status": "PASS_WITH_LIMITATIONS",
            "completed_utc": now(),
            "documents": len(all_ids),
            "batch_count": len(receipts),
            "document_status_counts": dict(total),
            "retained_questions": sum(row["coverage"]["retained_questions"] for row in receipts),
            "new_extraction_batches_total": sum(
                row["origin"] == "NEW_DETERMINISTIC_EXTRACTION" for row in receipts
            ),
            "prior_completed_batches_reused": sum(
                row["origin"] == "PRIOR_VALIDATED_RUN_REUSED" for row in receipts
            ),
            "new_extraction_seconds_total": sum(
                row["duration_seconds"]
                for row in receipts
                if row["origin"] == "NEW_DETERMINISTIC_EXTRACTION"
            ),
            "batches": receipts,
            "resume_validation": proof,
            "sampling_scope": manifest["sampling_scope"],
            "interpretation": "Engineering accounting, exact-source fidelity and native checkpoint reuse only. EMPTY is no configured cue, not no research gap. Lifecycle exclusions retained. Grouping is within each batch; no global duplicate, novelty, independent evidence or scientific accuracy claim.",
        }
        atomic_json(output / "SUMMARY.json", summary)
        atomic_json(output / "INPUT_MANIFEST.json", manifest)
        atomic_json(output / "RESUME_VALIDATION.json", proof)
        atomic_json(
            output / "task_state.json",
            {
                "task_name": "bounded_10000_abstract_checkpoint_scale",
                "task_status": "PASS_WITH_LIMITATIONS",
                "inputs": [str(manifest_path)],
                "outputs": ["SUMMARY.json", "RESUME_VALIDATION.json", ".workflow"],
                "items_requested": manifest["documents"],
                "items_processed": total["EMPTY"] + total["SUCCESS"],
                "items_succeeded": total["EMPTY"] + total["SUCCESS"],
                "items_failed": total["EXTRACTION_FAILURE"],
                "validation_status": "PASS",
                "audit_status": "PASS",
                "repair_cycles": 0,
                "repairs_performed": [],
                "unresolved_issues": [
                    "Mixed committed partial snapshots, not a representative sample.",
                    "No reviewed extraction accuracy; within-batch clustering only.",
                ],
                "scientific_assumptions_changed": False,
                "requires_human_review": True,
                "recommended_next_action": "No scale rerun needed; review source evidence before scientific use.",
            },
        )
        native_manifest = read(output / "RUN_MANIFEST.json")
        native_manifest.update(execution_status="PASS", finished_at=now())
        native_manifest["coverage"]["unique_literature_records"] = len(all_ids)
        atomic_json(output / "RUN_MANIFEST.json", native_manifest)
        (output / "HANDOFF.md").write_text(
            f"# Bounded checkpoint scale demonstration\n\n{len(all_ids)} native nonempty abstracts in {len(receipts)} bounded batches. "
            f"{summary['prior_completed_batches_reused']} previously completed batch reused; "
            f"{summary['new_extraction_batches_total']} new deterministic batches. Counts: {dict(total)}. "
            f"Retained {summary['retained_questions']} exact source passages.\n\n"
            "Each batch retains its source snapshot, input SHA256, status for every record, measured runtime/RSS and literal-span checks. "
            + (
                "The run stopped explicitly after a subset and resumed using existing CheckpointRun; unchanged pre-stop artifact hashes and timestamps prove reuse. "
                if proof["pre_stop_stage_hashes_and_mtimes_unchanged"] is True
                else "This invocation did not demonstrate stop/resume; no interruption-reuse claim is made. "
            )
            + "No competing scheduler, model computation or scientific rule change.\n\n"
            + manifest["sampling_scope"]
            + "\n\n"
            + summary["interpretation"]
            + "\n\n"
            "Status: PASS_WITH_LIMITATIONS. No further run is needed. A completed --resume verifies and reuses final output. "
            "For a relocated copy, first update only external input paths in a new input manifest and use a new output directory; existing checkpoint fingerprints deliberately bind their original paths.\n",
            encoding="utf-8",
        )
        checkpoint.finish(summary)
    return summary


def package(output):
    manifest = read(output / "INPUT_MANIFEST.json")
    members = {
        str(path.relative_to(output)).replace("\\", "/"): path
        for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.tsv"
    }
    members["validate_question_discovery_batches.py"] = Path(__file__).resolve()
    for batch in manifest["batches"]:
        members[f"inputs/batch-{batch['batch']:02d}.jsonl"] = Path(batch["input"]["path"])
    text = "relative_path\tfile_size_bytes\tsha256\n"
    for name, path in sorted(members.items()):
        text += f"{name}\t{path.stat().st_size}\t{checksum(path)}\n"
    (output / "MANIFEST.tsv").write_text(text, encoding="utf-8")
    members["MANIFEST.tsv"] = output / "MANIFEST.tsv"
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(members):
            raise ValueError("Package integrity failed")
        for line in text.splitlines()[1:]:
            name, size, expected = line.split("\t")
            payload = archive.read(name)
            if len(payload) != int(size) or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Manifest verification failed")
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--database", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--count", type=int, default=10000)
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.database, args.seed, args.output, args.count)
        print(
            json.dumps(
                {
                    "documents": result["documents"],
                    "manifest": str(args.output / "INPUT_MANIFEST.json"),
                }
            )
        )
    else:
        result = run_batches(
            args.manifest, args.output, resume=args.resume, stop_after=args.stop_after
        )
        archive = args.output.with_suffix(".zip")
        if not archive.exists():
            package(args.output)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "documents",
                        "document_status_counts",
                        "retained_questions",
                        "new_extraction_batches_total",
                        "prior_completed_batches_reused",
                        "new_extraction_seconds_total",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
