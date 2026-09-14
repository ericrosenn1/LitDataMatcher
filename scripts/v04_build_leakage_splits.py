#!/usr/bin/env python3
"""Build deterministic, group-aware v0.4 development splits from a corpus snapshot.

The output is explicitly a development allocation.  It is not the sealed
LitDataMatcher release holdout and must not be presented as a final evaluation.
Groups are connected components over question identity, linked paper identity,
and exact canonicalized question text, preventing those relations from crossing
train/development/evaluation allocations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "v0.4-leakage-split-1.0"
SPLITS = ("train", "development", "evaluation_candidate")
TARGETS = {"train": 0.70, "development": 0.15, "evaluation_candidate": 0.15}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def canonical_question(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def choose_split(component_id: str, component_size: int, allocated: Counter[str], total: int, seed: str) -> str:
    # Deterministic priority avoids order-dependent assignments, while the
    # deficit score keeps component-weighted allocations close to targets.
    rank = int(hashlib.sha256(f"{seed}|{component_id}".encode()).hexdigest()[:16], 16)
    scored = []
    for split in SPLITS:
        desired = TARGETS[split] * total
        deficit = desired - allocated[split]
        scored.append((deficit, -rank if split == "train" else -rank + SPLITS.index(split), split))
    return max(scored)[2]


def build(args: argparse.Namespace) -> dict[str, Any]:
    questions_path = args.corpus_root / "normalized" / "questions" / "questions.parquet"
    edges_path = args.corpus_root / "normalized" / "question_paper" / "question_paper.parquet"
    questions = pq.read_table(questions_path).to_pylist()
    edges = pq.read_table(edges_path).to_pylist()
    by_question = {row["question_id"]: row for row in questions}
    if len(by_question) != len(questions):
        raise ValueError("question_id must be unique in normalized questions")

    groups = UnionFind()
    text_representative: dict[str, str] = {}
    for row in questions:
        question_node = f"q:{row['question_id']}"
        groups.find(question_node)
        key = canonical_question(row.get("question_text"))
        if key:
            if key in text_representative:
                groups.union(question_node, text_representative[key])
            else:
                text_representative[key] = question_node

    edge_questions = set()
    for edge in edges:
        question_id = edge.get("question_id")
        if question_id not in by_question:
            continue
        paper_id = (edge.get("paper_id") or "").strip()
        if paper_id:
            groups.union(f"q:{question_id}", f"p:{paper_id.casefold()}")
        edge_questions.add(question_id)

    components: dict[str, list[str]] = defaultdict(list)
    for question_id in sorted(by_question):
        components[groups.find(f"q:{question_id}")].append(question_id)

    total = len(questions)
    allocated: Counter[str] = Counter()
    component_splits: dict[str, str] = {}
    for root, members in sorted(components.items(), key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest())):
        component_splits[root] = choose_split(root, len(members), allocated, total, args.seed)
        allocated[component_splits[root]] += len(members)

    assignments: list[dict[str, Any]] = []
    for root, members in components.items():
        split = component_splits[root]
        group_id = hashlib.sha256(f"{args.seed}|{root}".encode()).hexdigest()[:20]
        for question_id in members:
            row = by_question[question_id]
            assignments.append({
                "question_id": question_id,
                "split": split,
                "group_id": group_id,
                "source_corpus": row.get("source_corpus"),
                "source_partition": row.get("source_partition"),
                "source_record_id": row.get("source_record_id"),
                "canonical_question_sha256": hashlib.sha256(canonical_question(row.get("question_text")).encode()).hexdigest(),
                "has_question_paper_edge": question_id in edge_questions,
            })
    assignments.sort(key=lambda row: row["question_id"])

    split_by_question = {row["question_id"]: row["split"] for row in assignments}
    paper_splits: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.get("question_id") in split_by_question and edge.get("paper_id"):
            paper_splits[edge["paper_id"].casefold()].add(split_by_question[edge["question_id"]])
    duplicate_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        duplicate_splits[row["canonical_question_sha256"]].add(row["split"])
    leaked_papers = sorted(paper for paper, values in paper_splits.items() if len(values) > 1)
    leaked_questions = sorted(key for key, values in duplicate_splits.items() if len(values) > 1)
    if leaked_papers or leaked_questions:
        raise AssertionError(f"leakage detected: papers={len(leaked_papers)} question_texts={len(leaked_questions)}")
    if len(assignments) != total:
        raise AssertionError("every normalized question must receive one split")

    args.out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(assignments), args.out / "split_assignments.parquet", compression="zstd")
    with (args.out / "split_assignments.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(assignments)
    source_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        source_counts[split] = dict(sorted(Counter(row["source_corpus"] for row in assignments if row["split"] == split).items()))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "purpose": "group-aware development split and leakage audit; not a final held-out evaluation",
        "seed": args.seed,
        "input_artifacts": {
            "questions": {"relative_path": str(questions_path.relative_to(args.corpus_root)), "sha256": sha256(questions_path), "rows": len(questions)},
            "question_paper": {"relative_path": str(edges_path.relative_to(args.corpus_root)), "sha256": sha256(edges_path), "rows": len(edges)}
        },
        "grouping_rules": ["same question identifier", "same canonicalized exact question text", "same normalized paper_id through question_paper edges"],
        "split_counts": dict(allocated),
        "source_counts_by_split": source_counts,
        "component_count": len(components),
        "largest_component_questions": max(len(members) for members in components.values()),
        "questions_without_question_paper_edge": len(questions) - len(edge_questions),
        "audit": {"paper_cross_split_count": len(leaked_papers), "duplicate_question_cross_split_count": len(leaked_questions), "status": "PASS"},
        "limitations": ["Does not infer study-level identity beyond normalized paper_id.", "Evaluation_candidate is a development allocation, not the sealed final holdout.", "Cross-corpus semantic question variants require future reviewed identity expansion before a release evaluation."],
    }
    (args.out / "LEAKAGE_AUDIT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="LITDATAMATCHER_V04_SPLIT_V1")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({"status": manifest["audit"]["status"], "split_counts": manifest["split_counts"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
