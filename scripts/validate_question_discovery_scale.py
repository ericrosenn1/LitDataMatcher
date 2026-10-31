"""Replay preserved PubMed records with native abstracts; no retrieval or models.

The source-view correction is explicit: the historical index derived ``text``
from a title when no abstract existed. Native abstract/title/provenance fields
remain unchanged. This script measures engineering behavior, not scientific
accuracy or novelty. Original inputs and existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psutil

from litdatamatcher.question_discovery import discover_questions


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-protocol", type=Path)
    args = parser.parse_args()
    original_bytes = args.documents.read_bytes()
    documents = [json.loads(line) for line in original_bytes.decode("utf-8").splitlines()]
    if not 1 <= len(documents) <= 1000:
        raise ValueError("Provide 1..1000 preserved records")
    corrected_ids = []
    for row in documents:
        if row.get("source") != "pubmed" or not isinstance(row.get("abstract"), str):
            raise ValueError("Native PubMed abstract fields are required, including empty strings")
        if row.get("text") != row["abstract"]:
            if row.get("text") != row.get("title") or row["abstract"]:
                raise ValueError("Unexpected source view; review required before changing it")
            corrected_ids.append(row["document_id"])
        row["text"] = row["abstract"]
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    inputs = dest / "INPUT_DOCUMENTS.jsonl"
    inputs.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in documents),
        encoding="utf-8",
    )
    write_json(dest / "SNAPSHOT.json", snapshot)
    selection_protocol = (
        json.loads(args.selection_protocol.read_text(encoding="utf-8"))
        if args.selection_protocol
        else None
    )
    if selection_protocol is not None:
        write_json(dest / "SELECTION_PROTOCOL.json", selection_protocol)
    write_json(
        dest / "SOURCE_VIEW_CORRECTION.json",
        {
            "original_input_path": str(args.documents.resolve()),
            "original_input_sha256": sha(original_bytes),
            "corrected_document_ids": corrected_ids,
            "correction_count": len(corrected_ids),
            "transformation": "Derived text is native abstract, including empty abstract. Native title, abstract, identifier and provenance fields are unchanged.",
            "retrieval_repeated": False,
        },
    )
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples, stopped = [baseline], threading.Event()

    def monitor():
        while not stopped.wait(0.02):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = discover_questions(documents, max_documents=1000, max_questions=1000)
    finally:
        elapsed = time.perf_counter() - start
        samples.append(process.memory_info().rss)
        stopped.set()
        thread.join()
    write_json(dest / "QUESTIONS.json", result)
    by_id = {row["document_id"]: row for row in documents}
    for question in result["questions"]:
        doc = by_id[question["source_document_id"]]
        abstract, span = doc["abstract"], question["evidence_span"]
        assert abstract and abstract == doc["text"]
        assert abstract[span["start"] : span["end"]] == span["text"] == question["text"]
        assert question["text"] == question["question"]
        assert sha(abstract.encode("utf-8")) == question["source_text_sha256"]
        assert doc["document_id"] == "PMID:" + str(doc["pmid"])
        assert str(question["citation"]["pmid"]) == str(doc["pmid"])
        assert question["source_provenance"] == doc["source_provenance"]
        assert (
            question["source_locator"]
            == f"{doc['source_locator']}#chars={span['start']}:{span['end']}"
        )
        assert question["novelty_status"] == "NOT_ESTABLISHED"
        assert question["model_inferred"] is False
    counts = dict(Counter(row["status"] for row in result["document_statuses"]))
    question_ids = [row["question_id"] for row in result["questions"]]
    assert len(documents) == len(by_id) == len(result["document_statuses"])
    assert len(question_ids) == len(set(question_ids))
    assert sorted(question_ids) == sorted(
        identifier for cluster in result["clusters"] for identifier in cluster["question_ids"]
    )
    assert {row["document_id"] for row in result["document_statuses"]} == set(by_id)
    assert result["coverage"]["global_novelty_established"] is False
    assert all(row["text"] == row["abstract"] for row in documents)
    for status in result["document_statuses"]:
        if not by_id[status["document_id"]]["abstract"]:
            assert status["status"] in {"METADATA_ONLY_NOT_EXTRACTED", "EXCLUDED_SOURCE_LIFECYCLE"}
    validation = {
        "status": "PASS",
        "checked_question_records": len(question_ids),
        "all_source_views_equal_native_abstract_including_empty": "PASS",
        "literal_span_hash_citation_locator_native_provenance_fidelity": "PASS",
        "input_status_accounting_and_metadata_only_separation": "PASS",
        "unique_question_ids_and_complete_cluster_membership": "PASS",
    }
    write_json(dest / "VALIDATION.json", validation)
    module = Path(sys.modules[discover_questions.__module__].__file__).resolve()
    receipt = {
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "requested_documents": len(documents),
        "source_view_corrections": len(corrected_ids),
        "extraction_and_clustering_seconds": elapsed,
        "submitted_documents_per_second": len(documents) / elapsed,
        "text_bearing_processed_documents_per_second": result["coverage"]["processed_documents"]
        / elapsed,
        "memory": {
            "method": "Process RSS sampled every20ms during extraction+clustering; process-lifetime peak separate",
            "extraction_baseline_rss_bytes": baseline,
            "extraction_sampled_peak_rss_bytes": max(samples),
            "extraction_peak_rss_increase_bytes": max(samples) - baseline,
            "sample_count": len(samples),
            "process_lifetime_peak_working_set_bytes": getattr(
                process.memory_info(), "peak_wset", None
            ),
        },
        "snapshot": snapshot,
        "selection_protocol": selection_protocol,
        "coverage": result["coverage"],
        "document_status_counts": counts,
        "no_candidate_question_in_processed_source_text_documents": counts.get("EMPTY", 0),
        "question_category_counts": dict(Counter(row["category"] for row in result["questions"])),
        "clusters": len(result["clusters"]),
        "clusters_with_multiple_members": sum(
            len(row["question_ids"]) > 1 for row in result["clusters"]
        ),
        "original_input_sha256": sha(original_bytes),
        "corrected_input_sha256": sha(inputs.read_bytes()),
        "result_sha256": sha((dest / "QUESTIONS.json").read_bytes()),
        "python_version": platform.python_version(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=module.parent, text=True
        ).strip(),
        "implementation_sha256": {
            name: sha((module.parent / name).read_bytes())
            for name in (
                "question_discovery.py",
                "question_compiler.py",
                "question_compiler_rules_v1.json",
            )
        },
        "validation": validation,
        "scientific_scope": "Engineering throughput/accounting/source fidelity only. Partial historical corpus, topic-selected cohort; no precision/recall, outcome, novelty or dataset-fit claim. Metadata-only inputs were not extracted.",
    }
    write_json(dest / "RECEIPT.json", receipt)
    write_json(
        dest / "task_state.json",
        {
            "task_name": "preserved_1000_pubmed_question_discovery_demonstration",
            "task_status": "PASS_WITH_LIMITATIONS",
            "inputs": [str(args.documents.resolve())],
            "outputs": [
                "QUESTIONS.json",
                "RECEIPT.json",
                "VALIDATION.json",
                "SOURCE_VIEW_CORRECTION.json",
            ],
            "items_requested": len(documents),
            "items_processed": result["coverage"]["processed_documents"],
            "items_succeeded": result["coverage"]["processed_documents"],
            "items_failed": result["coverage"]["failed_documents"],
            "validation_status": "PASS",
            "audit_status": "PASS",
            "repair_cycles": 1 if corrected_ids else 0,
            "repairs_performed": [
                "Removed the historical index's title fallback from derived source text; original input and first run preserved as diagnostic."
            ]
            if corrected_ids
            else [],
            "unresolved_issues": [
                "Missing native abstracts, when present, limit actual extraction count.",
                "Rule-based cue accuracy is not measured; no expert labels or novelty claims.",
            ],
            "scientific_assumptions_changed": False,
            "requires_human_review": True,
            "recommended_next_action": "Review source-linked passages before scientific use; no further execution required for this demonstration.",
        },
    )
    (dest / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (dest / "HANDOFF.md").write_text(
        f"# Preserved PubMed cohort: question-discovery scale demonstration\n\n"
        f"Replayed the same {len(documents)} native records from committed partial snapshot `{snapshot['snapshot_sha256']}`. "
        f"Corrected {len(corrected_ids)} derived title fallbacks to empty native abstracts. No retrieval, model call, source-data mutation or scientific-policy change.\n\n"
        f"Statuses: {json.dumps(counts)}. Retained {len(question_ids)} literal abstract passages in {len(result['clusters'])} review groups. "
        f"Extraction+clustering took {elapsed:.3f}s. All source-span/hash/PMID/provenance checks passed.\n\n"
        "Missing abstracts are metadata-only, never negative findings. EMPTY means no configured cue matched available text. "
        "The cohort has a partial historical and topic-selected scope. This is engineering evidence, not reviewed extraction accuracy, biological validity, global novelty or dataset qualification.\n\n"
        "Status: PASS_WITH_LIMITATIONS. No rerun required. To reproduce, run the included script with --documents INPUT_DOCUMENTS.jsonl --snapshot SNAPSHOT.json --output a-new-output-directory, using the implementation hashes in RECEIPT.json. "
        "The original invalid title-based run is preserved separately as a diagnostic; cite this corrected receipt only.\n",
        encoding="utf-8",
    )
    manifest = "relative_path\tfile_size_bytes\tsha256\trole\tstatus\n"
    for path in sorted(dest.iterdir()):
        manifest += (
            f"{path.name}\t{path.stat().st_size}\t{sha(path.read_bytes())}\tfinal_artifact\tPASS\n"
        )
    (dest / "MANIFEST.tsv").write_text(manifest, encoding="utf-8")
    archive_path = dest.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {path.name for path in dest.iterdir()}
        for line in manifest.splitlines()[1:]:
            name, size, checksum, _role, _status = line.split("\t")
            data = archive.read(name)
            assert len(data) == int(size) and sha(data) == checksum
    print(json.dumps({"output": str(dest), "archive": str(archive_path), **receipt}, indent=2))


if __name__ == "__main__":
    main()
