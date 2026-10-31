#!/usr/bin/env python3
"""Prepare original-paper retrieval data from a qualified annual PubMed baseline.

Only existing train/development assignments and their permitted official source
partitions are consumed. This is paper-link recovery, not requirements or dataset
feasibility gold. Unversioned updates and all reserved evaluation text are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VERSION = "paper-link-retrieval-baseline-v1"
SPLITS = ("train", "development")
SOURCES = {"PubMedQA", "EvidenceBench", "Evidence Inference 2.0"}
OFFICIAL_ROOT = "raw/evidence_inference/evidence-inference_official/annotations/splits"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(text: str) -> str:
    return " ".join(text.casefold().split())


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input escapes corpus root: {relative}")
    return path


def check_stat(path: Path, size: int, mtime_ns: int) -> None:
    stat = path.stat()
    if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
        raise ValueError(f"Input changed since qualified receipt: {path}")


def qualified_snapshot(path: Path, root: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("status") != "PASS" or snapshot.get("raw_checksums_recomputed") is not True:
        raise ValueError("Corpus receipt must be PASS with raw_checksums_recomputed=true")
    shards = snapshot.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Qualified receipt has no shards")
    for shard in shards:
        if shard.get("source") != "baseline":
            raise ValueError("Only qualified annual baseline shards are allowed; updates excluded")
        if shard.get("status") != "PASS" or not re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", ""))):
            raise ValueError("Every shard requires PASS and SHA-256")
        check_stat(local_path(root, shard["relative_path"]), shard["bytes"], shard["mtime_ns"])
        if shard.get("raw_relative_path"):
            check_stat(local_path(root, shard["raw_relative_path"]), shard["raw_bytes"], shard["raw_mtime_ns"])
    return snapshot


def identifier(value: object, kind: str = "") -> str | None:
    value = str(value or "").strip()
    if kind.casefold() == "pmid" or re.fullmatch(r"pmid[:\s]+\d+", value, re.I):
        number = re.sub(r"^pmid[:\s]+", "", value, flags=re.I)
        return f"pmid:{int(number)}" if number.isdigit() and int(number) > 0 else None
    if kind.casefold() == "pmcid" or re.fullmatch(r"(?:pmcid[:\s]+)?PMC\d+", value, re.I):
        number = re.sub(r"^(?:pmcid[:\s]+)?(?:PMC)?", "", value, flags=re.I)
        return f"pmcid:PMC{int(number)}" if number.isdigit() and int(number) > 0 else None
    # No implicit PMID interpretation for numeric EvidenceBench internal IDs.
    return None


def official_ids(path: Path) -> set[str]:
    values = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        value = identifier(line, "PMCID")
        if value is None:
            raise ValueError(f"Malformed official article ID: {path}")
        values.add(value)
    return values


def partition_allowed(source: str, split: str, partition: object, paper: str,
                      official_train: set[str], official_dev: set[str]) -> bool:
    if split not in SPLITS:
        return False
    part = str(partition or "").casefold()
    if source == "PubMedQA":
        return part == "pqa-l"
    if source == "EvidenceBench":
        return part == "train" if split == "train" else part in {"train", "dev"}
    if source == "Evidence Inference 2.0":
        return paper in (official_train if split == "train" else official_dev)
    return False


def load_candidates(root: Path, assignment_path: Path, counts: Counter) -> tuple[list[dict], list[Path]]:
    question_path = root / "normalized/questions/questions.parquet"
    edge_path = root / "normalized/question_paper/question_paper.parquet"
    train_path = root / OFFICIAL_ROOT / "train_article_ids.txt"
    dev_path = root / OFFICIAL_ROOT / "validation_article_ids.txt"
    official_train, official_dev = official_ids(train_path), official_ids(dev_path)
    if official_train & official_dev:
        raise ValueError("Official train and validation article IDs overlap")
    assignments = pq.read_table(assignment_path, columns=["question_id", "split"],
                                filters=[("split", "in", list(SPLITS))]).to_pylist()
    split_by_id = {row["question_id"]: row["split"] for row in assignments}
    if len(split_by_id) != len(assignments):
        raise ValueError("Duplicate allowed assignment question ID")
    if not assignments:
        raise ValueError("No train/development assignments")
    filters = [("question_id", "in", list(split_by_id))]
    # Read source metadata and links first. Disallowed query text is never materialized.
    metadata = pq.read_table(question_path, columns=["question_id", "source_corpus", "source_partition", "provenance_relative_path"], filters=filters).to_pylist()
    edges = pq.read_table(edge_path, columns=["question_id", "identifier_type", "identifier_value", "relation"], filters=filters).to_pylist()
    edge_index = defaultdict(list)
    for edge in edges:
        edge_index[edge["question_id"]].append(edge)
    eligible = []
    seen = set()
    for row in metadata:
        qid, source = row["question_id"], row["source_corpus"]
        if qid in seen:
            raise ValueError("Duplicate normalized question ID")
        seen.add(qid)
        if source not in SOURCES:
            counts["unsupported_source"] += 1
            continue
        split = split_by_id[qid]
        if source == "EvidenceBench" and not partition_allowed(source, split, row.get("source_partition"), "", official_train, official_dev):
            counts["official_partition_excluded"] += 1
            continue
        if not edge_index[qid]:
            counts["missing_paper_edge"] += 1
        for edge in edge_index[qid]:
            paper = identifier(edge.get("identifier_value"), edge.get("identifier_type", ""))
            if paper is None:
                counts["unsupported_or_missing_identifier"] += 1
                continue
            if not partition_allowed(source, split, row.get("source_partition"), paper, official_train, official_dev):
                counts["official_partition_excluded"] += 1
                continue
            if edge.get("relation") not in {"source_linked", "candidate_pool"}:
                counts["unsupported_link_relation"] += 1
                continue
            provenance = str(row.get("provenance_relative_path") or "")
            if not provenance:
                counts["missing_query_provenance"] += 1
                continue
            local_path(root, provenance)
            eligible.append({"id": qid, "paper_alias": paper, "source_corpus": source,
                             "split": split, "provenance_relative_path": provenance})
    if not eligible:
        raise ValueError(f"No eligible official-partition links: {dict(counts)}")
    allowed_ids = sorted({row["id"] for row in eligible})
    questions = pq.read_table(question_path, columns=["question_id", "question_text"],
                              filters=[("question_id", "in", allowed_ids)]).to_pylist()
    queries = {row["question_id"]: row["question_text"] for row in questions}
    candidates = []
    for row in eligible:
        query = queries.get(row["id"])
        if isinstance(query, str) and query.strip():
            candidates.append({**row, "query": query})
        else:
            counts["missing_query"] += 1
    counts["allowed_assignment_rows"] = len(assignments)
    counts["eligible_link_rows"] = len(candidates)
    return candidates, [assignment_path, question_path, edge_path, train_path, dev_path]


def shard_order(shard: dict) -> tuple[int, str]:
    name = str(shard.get("source_shard", shard["relative_path"]))
    match = re.search(r"n(\d+)", name)
    return int(match.group(1)) if match else 0, name


def resolve_documents(root: Path, shards: list[dict], aliases: set[str], counts: Counter) -> tuple[dict, dict]:
    wanted_pmids = {value[5:] for value in aliases if value.startswith("pmid:")}
    wanted_pmcids = {value[6:] for value in aliases if value.startswith("pmcid:")}
    alias_to_ids = defaultdict(set)
    documents = {}
    columns = ["pmid", "pmcid", "doi", "title", "abstract", "publication_types_json"]
    for shard_index, shard in enumerate(sorted(shards, key=shard_order), 1):
        path = local_path(root, shard["relative_path"])
        check_stat(path, shard["bytes"], shard["mtime_ns"])
        parquet, row_base = pq.ParquetFile(path), 0
        for group in range(parquet.num_row_groups):
            ids = parquet.read_row_group(group, columns=["pmid", "pmcid"])
            pmids, pmcids = ids.column("pmid"), pc.utf8_upper(ids.column("pmcid"))
            mask = pc.or_(pc.is_in(pmids, value_set=pa.array(sorted(wanted_pmids), type=pa.string())),
                          pc.is_in(pmcids, value_set=pa.array(sorted(wanted_pmcids), type=pa.string())))
            indices = pc.indices_nonzero(pc.fill_null(mask, False)).to_pylist()
            new_pmids = {str(pmids[index].as_py()) for index in indices if pmids[index].as_py()}
            if new_pmids - wanted_pmids:
                wanted_pmids.update(new_pmids)
                mask = pc.or_(mask, pc.is_in(pmids, value_set=pa.array(sorted(new_pmids), type=pa.string())))
                indices = pc.indices_nonzero(pc.fill_null(mask, False)).to_pylist()
            if indices:
                rows = parquet.read_row_group(group, columns=columns).take(pa.array(indices, type=pa.int64())).to_pylist()
                for index, row in zip(indices, rows, strict=True):
                    pmid, pmcid = identifier(row.get("pmid"), "PMID"), identifier(row.get("pmcid"), "PMCID")
                    canonical = pmid or pmcid
                    if canonical is None:
                        continue
                    for alias in (pmid, pmcid):
                        if alias and alias in aliases:
                            alias_to_ids[alias].add(canonical)
                    documents[canonical] = {**row, "id": canonical, "source_shard": shard["relative_path"], "source_sha256": shard["sha256"],
                                            "row_locator": {"row_group": group, "row_index": index, "row_number": row_base + index}}
                    counts["matching_document_versions"] += 1
            row_base += ids.num_rows
        check_stat(path, shard["bytes"], shard["mtime_ns"])
        if shard_index % 100 == 0:
            print(json.dumps({"stage": "resolve_documents", "shards_scanned": shard_index, "canonical_records": len(documents)}), flush=True)
    return documents, alias_to_ids


def select_records(candidates: list[dict], documents: dict, aliases: dict, seed: str,
                   train_limit: int, dev_limit: int, counts: Counter) -> tuple[list, list, list]:
    rows = []
    for candidate in candidates:
        ids = aliases.get(candidate["paper_alias"], set())
        if len(ids) != 1:
            counts["ambiguous_document_alias" if ids else "unresolved_document"] += 1
            continue
        doc_id = next(iter(ids))
        doc = documents[doc_id]
        try:
            types = json.loads(doc.get("publication_types_json") or "[]")
            if not isinstance(types, list):
                raise ValueError("Publication types must be a list")
        except (TypeError, ValueError):
            counts["invalid_publication_types"] += 1
            continue
        if {re.sub(r"[^a-z]", "", str(value).lower()) for value in types} & {"retractedpublication", "retractionofpublication"}:
            counts["retracted_publication"] += 1
            continue
        abstract, title = doc.get("abstract"), doc.get("title") or ""
        if not isinstance(abstract, str) or not abstract.strip():
            counts["missing_abstract"] += 1
            continue
        if not isinstance(title, str):
            counts["invalid_title"] += 1
            continue
        text = f"{title}\n\n{abstract}" if title else abstract
        rows.append({**candidate, "positive_doc_id": doc_id, "text_hash": text_hash(canonical_text(text))})
    def order(row):
        return text_hash(f"{seed}|{row['id']}|{row['positive_doc_id']}|{row['query']}")
    unique = {}
    for row in sorted(rows, key=order):
        key = row["split"], row["id"]
        if key in unique:
            counts["multiple_paper_links_deduplicated"] += 1
        else:
            unique[key] = row
    rows = list(unique.values())
    dev = [row for row in rows if row["split"] == "development"]
    dev_docs, dev_queries, dev_texts = ({row["positive_doc_id"] for row in dev},
                                     {canonical_text(row["query"]) for row in dev}, {row["text_hash"] for row in dev})
    safe = []
    for row in rows:
        if row["split"] == "train" and row["positive_doc_id"] in dev_docs:
            counts["train_development_canonical_overlap"] += 1
        elif row["split"] == "train" and canonical_text(row["query"]) in dev_queries:
            counts["train_development_query_overlap"] += 1
        elif row["split"] == "train" and row["text_hash"] in dev_texts:
            counts["train_development_document_text_overlap"] += 1
        else:
            safe.append(row)
    selected, seen_papers, source_counts = [], set(), Counter()
    for row in sorted(safe, key=order):
        key = row["split"], row["source_corpus"]
        if row["positive_doc_id"] in seen_papers:
            counts["one_question_per_paper"] += 1
        elif source_counts[key] >= (train_limit if row["split"] == "train" else dev_limit):
            counts["per_source_limit"] += 1
        else:
            seen_papers.add(row["positive_doc_id"])
            source_counts[key] += 1
            selected.append(row)
    fields = ("id", "query", "positive_doc_id", "source_corpus", "split", "provenance_relative_path")
    train = [{key: row[key] for key in fields} for row in selected if row["split"] == "train"]
    development = [{key: row[key] for key in fields} for row in selected if row["split"] == "development"]
    output_docs = []
    for doc_id in sorted(seen_papers):
        doc = documents[doc_id]
        title, abstract = doc.get("title") or "", doc["abstract"]
        output_docs.append({"id": doc_id, "text": f"{title}\n\n{abstract}" if title else abstract,
                            "title": title, "abstract": abstract, "pmid": doc.get("pmid"), "pmcid": doc.get("pmcid"), "doi": doc.get("doi"),
                            "source_shard": doc["source_shard"], "source_sha256": doc["source_sha256"], "row_locator": doc["row_locator"]})
    return train, development, output_docs


def leakage_checks(train: list, development: list, docs: list) -> dict:
    by_id = {row["id"]: row for row in docs}
    values = {}
    for label, key in (("canonical_paper_overlap", lambda row: row["positive_doc_id"]),
                       ("canonical_query_overlap", lambda row: canonical_text(row["query"])),
                       ("canonical_document_text_overlap", lambda row: canonical_text(by_id[row["positive_doc_id"]]["text"]))):
        values[label] = len({key(row) for row in train} & {key(row) for row in development})
    if any(values.values()):
        raise ValueError(f"Train/development leakage: {values}")
    return values


def build(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise ValueError("Output directory must be new; existing artifacts are never replaced")
    if args.out.resolve().is_relative_to(args.corpus_root.resolve()):
        raise ValueError("Benchmark output must be outside the protected corpus root")
    snapshot = qualified_snapshot(args.receipt, args.corpus_root)
    counts = Counter()
    candidates, inputs = load_candidates(args.corpus_root, args.assignments, counts)
    fingerprints = [{"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                    for path in [args.receipt, *inputs]]
    docs, aliases = resolve_documents(args.corpus_root, snapshot["shards"], {row["paper_alias"] for row in candidates}, counts)
    train, development, docs = select_records(candidates, docs, aliases, args.seed, args.max_train_per_source, args.max_development_per_source, counts)
    if len(development) < 30 or not train:
        raise ValueError(f"Need training rows and >=30 development cases; got {len(train)}/{len(development)}; exclusions={dict(counts)}")
    leaks = leakage_checks(train, development, docs)
    for entry in fingerprints:
        check_stat(Path(entry["path"]), entry["bytes"], entry["mtime_ns"])
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("train.jsonl", train), ("development.jsonl", development), ("documents.jsonl", docs)):
        with (args.out / filename).open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {"schema_version": VERSION, "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Original paper-link recovery; not requirements/data-feasibility gold or expert evaluation",
                "corpus_scope": "Qualified annual baseline only; updates excluded; not latest clinical truth",
                "seed": args.seed, "allowed_splits": list(SPLITS),
                "official_partition_policy": {"EvidenceBench": "train: official train; development: official train/dev; test excluded",
                                              "Evidence Inference 2.0": "official train/validation article IDs; test file never read",
                                              "PubMedQA": "PQA-L with existing grouped allocation"},
                "counts": {"train": len(train), "development": len(development), "documents": len(docs)},
                "counts_by_source": {split: dict(Counter(row["source_corpus"] for row in rows)) for split, rows in (("train", train), ("development", development))},
                "limits_per_source": {"train": args.max_train_per_source, "development": args.max_development_per_source},
                "exclusions_and_accounting": dict(counts), "inputs": fingerprints,
                "qualified_shards": [{key: shard[key] for key in ("relative_path", "sha256", "bytes", "mtime_ns")} for shard in snapshot["shards"]],
                "leakage_checks": leaks,
                "outputs": [{"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size} for path in sorted(args.out.glob("*.jsonl"))]}
    (args.out / "BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path(r"D:\LitDataMatcher-Corpora"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="litdatamatcher-retrieval-20261031")
    parser.add_argument("--max-train-per-source", type=int, default=1000)
    parser.add_argument("--max-development-per-source", type=int, default=100)
    args = parser.parse_args()
    if min(args.max_train_per_source, args.max_development_per_source) < 1:
        parser.error("Per-source limits must be positive")
    result = build(args)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
