"""Add global review grouping to completed batches; never rerun extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.question_discovery import cluster_questions
from litdatamatcher.workflow_checkpoint import file_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checksum(path):
    return file_identity(path)["sha256"]


def aggregate(source, output):
    state = read(source / ".workflow/state.json")
    summary = read(source / "SUMMARY.json")
    manifest = read(source / "INPUT_MANIFEST.json")
    if state["status"] != "COMPLETE":
        raise ValueError("Only completed native checkpoints can be aggregated")
    archive = source.with_suffix(".zip")
    frozen_archive = file_identity(archive)
    source_evidence = {
        "source_run": str(source.resolve()),
        "original_archive": frozen_archive,
        "summary": file_identity(source / "SUMMARY.json"),
        "input_manifest": file_identity(source / "INPUT_MANIFEST.json"),
        "native_checkpoint_state": file_identity(source / ".workflow/state.json"),
        "batches": [],
    }
    questions, batch_for_question, prior_groups = [], {}, 0
    for batch in manifest["batches"]:
        name = f"extract-{batch['batch']:02d}"
        stage = state["stages"][name]
        stage_path = source / ".workflow" / stage["file"]
        if stage_path.resolve().parent != (source / ".workflow").resolve():
            raise ValueError("Checkpoint stage is outside the run")
        if checksum(stage_path) != stage["sha256"]:
            raise ValueError("Extraction checkpoint checksum changed")
        input_path = Path(batch["input"]["path"])
        if file_identity(input_path) != batch["input"]:
            raise ValueError("Frozen input changed")
        documents = {
            row["document_id"]: row
            for row in (
                json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            )
        }
        result = read(stage_path)["result"]
        prior_groups += len(result["clusters"])
        source_evidence["batches"].append(
            {
                "batch": batch["batch"],
                "snapshot": batch.get("snapshot"),
                "input": batch["input"],
                "stage": file_identity(stage_path),
            }
        )
        for question in result["questions"]:
            doc = documents[question["source_document_id"]]
            span = question["evidence_span"]
            if not (
                doc["abstract"][span["start"] : span["end"]]
                == question["text"]
                == span["text"]
                == question["question"]
                and hashlib.sha256(doc["abstract"].encode()).hexdigest()
                == question["source_text_sha256"]
                and question["source_provenance"] == doc["source_provenance"]
                and all(question["citation"][key] == doc.get(key) for key in question["citation"])
            ):
                raise ValueError("Original source evidence failed validation")
            if question["question_id"] in batch_for_question:
                raise ValueError("Question identifier repeats across batches")
            batch_for_question[question["question_id"]] = batch["batch"]
            questions.append(question)
    if len(questions) != summary["retained_questions"]:
        raise ValueError("Summary and retained passage counts differ")
    if not 1 <= len(questions) <= 1000:
        raise ValueError("Existing global clustering API supports at most 1000 retained passages")
    output.mkdir(parents=True, exist_ok=False)
    before = digest(questions)
    start = time.perf_counter()
    groups = cluster_questions(questions)
    elapsed = time.perf_counter() - start
    if digest(questions) != before:
        raise ValueError("Grouping changed original question records")
    if sorted(batch_for_question) != sorted(
        identifier for group in groups for identifier in group["question_ids"]
    ):
        raise ValueError("Grouping lost or duplicated members")
    by_id = {question["question_id"]: question for question in questions}
    for group in groups:
        if group["source_document_ids"] != sorted(
            {by_id[identifier]["source_document_id"] for identifier in group["question_ids"]}
        ):
            raise ValueError("Group source identities differ from original members")
        if (
            group["scientific_equivalence"] != "NOT_ESTABLISHED"
            or group["independent_evidence_count"] is not None
        ):
            raise ValueError("Unsupported scientific grouping interpretation")
    questions_path = output / "SOURCE_QUESTIONS.jsonl"
    questions_path.write_text(
        "".join(
            json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n"
            for question in questions
        ),
        encoding="utf-8",
    )
    atomic_json(output / "GLOBAL_GROUPS.json", groups)
    atomic_json(
        output / "SOURCE_FINGERPRINT.json",
        {"fingerprint_sha256": digest(source_evidence), **source_evidence},
    )
    cross_batch = [
        group
        for group in groups
        if len({batch_for_question[key] for key in group["question_ids"]}) > 1
    ]
    multi = [group for group in groups if len(group["question_ids"]) > 1]
    receipt = {
        "status": "PASS_WITH_LIMITATIONS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_batches": len(manifest["batches"]),
        "retained_passages": len(questions),
        "source_documents": len({question["source_document_id"] for question in questions}),
        "original_per_batch_groups_total": prior_groups,
        "global_groups": len(groups),
        "multi_member_global_groups": len(multi),
        "cross_batch_groups": len(cross_batch),
        "cross_document_groups": sum(len(group["source_document_ids"]) > 1 for group in groups),
        "multi_member_relation_counts": dict(Counter(group["relation"] for group in multi)),
        "grouping_seconds": elapsed,
        "similarity_threshold": 0.9,
        "source_fingerprint_sha256": digest(source_evidence),
        "question_records_canonical_sha256": before,
        "source_questions_file_sha256": checksum(questions_path),
        "global_groups_file_sha256": checksum(output / "GLOBAL_GROUPS.json"),
        "implementation_sha256": checksum(Path(sys.modules[cluster_questions.__module__].__file__)),
        "all_source_records_and_member_ids_preserved": True,
        "all_exact_native_source_spans_revalidated": True,
        "original_archive_unchanged": file_identity(archive) == frozen_archive,
        "extraction_rerun": False,
        "source_retrieval": False,
        "model_calls": 0,
        "interpretation": "Additional global exact/near-duplicate review grouping of retained passages only. Original per-batch results remain unchanged. The existing 0.9 engineering threshold is not scientific equivalence, novelty, independent evidence, or accuracy. Original lifecycle exclusions and empty extraction records remain in the frozen source run.",
    }
    if not receipt["original_archive_unchanged"]:
        raise ValueError("Frozen extraction archive changed")
    atomic_json(output / "RECEIPT.json", receipt)
    atomic_json(
        output / "task_state.json",
        {
            "task_name": "global_grouping_of_retained_10000_abstract_passages",
            "task_status": "PASS_WITH_LIMITATIONS",
            "inputs": [str(source)],
            "outputs": [
                "GLOBAL_GROUPS.json",
                "SOURCE_QUESTIONS.jsonl",
                "SOURCE_FINGERPRINT.json",
                "RECEIPT.json",
            ],
            "items_requested": len(questions),
            "items_processed": len(questions),
            "items_succeeded": len(questions),
            "items_failed": 0,
            "validation_status": "PASS",
            "audit_status": "PASS",
            "repair_cycles": 0,
            "repairs_performed": [],
            "unresolved_issues": [
                "Near-duplicate groups require review; scientific equivalence is not established.",
                "API bound is 1000 retained passages; no corpus-global completeness claim.",
            ],
            "scientific_assumptions_changed": False,
            "requires_human_review": True,
            "recommended_next_action": "Review proposed groups before scientific use; no extraction rerun is required.",
        },
    )
    (output / "HANDOFF.md").write_text(
        f"# Added global passage grouping\n\nThe existing cluster_questions function grouped all {len(questions)} retained passages from ten completed extraction batches into {len(groups)} review groups. "
        f"There are {len(multi)} multi-member groups and {len(cross_batch)} groups spanning batches. "
        f"The original per-batch group count was {prior_groups}; that report and its verified ZIP remain frozen.\n\n"
        "All original source-linked question records, native IDs, exact passages, citations and provenance are preserved. Group membership is complete and nonduplicating. "
        "No extraction, source retrieval or model computation was repeated. Source and implementation fingerprints are retained.\n\n"
        + receipt["interpretation"]
        + "\n\nStatus: PASS_WITH_LIMITATIONS. No further execution is needed. Reproduce using the included script with --source-run pointing to the original completed run and --output a new directory.\n",
        encoding="utf-8",
    )
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest_text = "relative_path\tfile_size_bytes\tsha256\n"
    for path in sorted(output.iterdir()):
        manifest_text += f"{path.name}\t{path.stat().st_size}\t{checksum(path)}\n"
    (output / "MANIFEST.tsv").write_text(manifest_text, encoding="utf-8")
    package = output.with_suffix(".zip")
    with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.iterdir()):
            handle.write(path, path.name)
    with zipfile.ZipFile(package) as handle:
        if handle.testzip() is not None:
            raise ValueError("Global aggregation archive is corrupt")
        for line in manifest_text.splitlines()[1:]:
            name, size, expected = line.split("\t")
            data = handle.read(name)
            if len(data) != int(size) or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Global aggregation manifest differs from archive")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.source_run.resolve(), args.output.resolve()), indent=2))
