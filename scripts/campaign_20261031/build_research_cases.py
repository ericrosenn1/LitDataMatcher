#!/usr/bin/env python3
"""Freeze source-linked development cases before observing end-to-end results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            # Escape Unicode separators while retaining decoded source text exactly.
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build_cases(benchmark: Path, out: Path, *, count: int = 30,
                seed: str = "research-cases-20261031-v1") -> dict:
    if out.exists():
        raise ValueError("Case output must be a fresh directory")
    if count < 30:
        raise ValueError("At least 30 cases are required")
    manifest_path = benchmark / "BENCHMARK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("Benchmark must be qualified PASS")
    indexed = {item["path"]: item for item in manifest["outputs"]}
    for filename in ("development.jsonl", "documents.jsonl"):
        expected, path = indexed[filename], benchmark / filename
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Benchmark hash mismatch: {filename}")
    queries = load_jsonl(benchmark / "development.jsonl")
    document_rows = load_jsonl(benchmark / "documents.jsonl")
    documents = {doc["id"]: doc for doc in document_rows}
    if len(documents) != len(document_rows) or len({query["id"] for query in queries}) != len(queries):
        raise ValueError("Duplicate source document or query identities")
    strata = defaultdict(list)
    for query in queries:
        if query["split"] != "development":
            raise ValueError("Only development cases may be selected")
        if query["positive_doc_id"] not in documents:
            raise ValueError("Missing linked source document")
        strata[query["source_corpus"]].append(query)

    def rank(query):
        value = f"{seed}|{query['source_corpus']}|{query['id']}|{query['positive_doc_id']}"
        return hashlib.sha256(value.encode()).hexdigest()

    for rows in strata.values():
        rows.sort(key=rank)
    selected, seen_papers, offsets = [], set(), Counter()
    while len(selected) < count:
        added = False
        for source in sorted(strata):
            while offsets[source] < len(strata[source]):
                query = strata[source][offsets[source]]
                offsets[source] += 1
                if query["positive_doc_id"] in seen_papers:
                    continue
                seen_papers.add(query["positive_doc_id"])
                selected.append(query)
                added = True
                break
            if len(selected) == count:
                break
        if not added:
            raise ValueError("Insufficient unique development papers")
    cases, inputs = [], []
    for query in selected:
        document = documents[query["positive_doc_id"]]
        case_id = "case_" + rank(query)[:20]
        cases.append({"case_id": case_id, "question_id": query["id"], "question": query["query"],
                      "source_corpus": query["source_corpus"], "split": "development",
                      "provenance_relative_path": query["provenance_relative_path"],
                      "source_document": document, "selection_sha256": rank(query),
                      "evaluation_scope": "pipeline execution, abstention, and source grounding; no expert gold label"})
        inputs.append({"source_id": document["id"], "document_id": document["id"], "case_id": case_id,
                       "title": document["title"], "abstract": document["abstract"], "text": document["text"],
                       "pmid": document["pmid"], "pmcid": document["pmcid"], "doi": document["doi"],
                       "question": query["query"], "source_corpus": query["source_corpus"],
                       "source_shard": document["source_shard"], "source_sha256": document["source_sha256"],
                       "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{document['pmid']}/" if document.get("pmid") else ""})
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "pipeline_inputs.jsonl", inputs)
    report = {"schema_version": "source-backed-research-cases-v1", "status": "FROZEN",
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "selection_method": "source-stratified round robin of SHA-256-ranked development queries; unique linked papers",
              "selection_uses_runtime_results": False, "expert_gold_labels": False,
              "benchmark_manifest_sha256": file_hash(manifest_path),
              "benchmark_directory": str(benchmark), "count": len(cases),
              "counts_by_source": dict(Counter(case["source_corpus"] for case in cases)),
              "limitations": ["Development cases, not untouched final evaluation.",
                              "Original-paper links do not establish scientific truth or dataset feasibility.",
                              "Frozen annual baseline abstracts; updates and complete article bodies are excluded."],
              "outputs": [{"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
                          for path in sorted(out.glob("*.jsonl"))]}
    (out / "CASES_MANIFEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", default="research-cases-20261031-v1")
    args = parser.parse_args()
    report = build_cases(args.benchmark, args.out, count=args.count, seed=args.seed)
    print(json.dumps({"status": report["status"], "count": report["count"], "counts_by_source": report["counts_by_source"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
