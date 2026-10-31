"""Frozen case selection uses source data only, with no outcome-based selection."""

import json
from collections import Counter

import pytest

from scripts.campaign_20261031.build_research_cases import (
    build_cases,
    file_hash,
    load_jsonl,
    write_jsonl,
)


def fixture_benchmark(root):
    root.mkdir()
    queries, docs = [], []
    for i in range(36):
        queries.append({"id": f"q{i}", "query": f"Original query {i}", "positive_doc_id": f"pmid:{i+1}",
                        "source_corpus": "A" if i % 2 else "B", "split": "development", "provenance_relative_path": "raw/source.json"})
        docs.append({"id": f"pmid:{i+1}", "title": f"Source {i}", "abstract": "Original abstract.",
                     "text": f"Source {i}\n\nOriginal abstract.", "pmid": str(i+1), "pmcid": None, "doi": None,
                     "source_shard": "baseline.parquet", "source_sha256": "a"*64, "row_locator": {"row_index": i}})
    write_jsonl(root / "development.jsonl", queries)
    write_jsonl(root / "documents.jsonl", docs)
    (root / "BENCHMARK_MANIFEST.json").write_text(json.dumps({"status": "PASS", "outputs": [
        {"path": name, "sha256": file_hash(root/name), "bytes": (root/name).stat().st_size}
        for name in ("development.jsonl", "documents.jsonl")]}))


def test_source_stratification_is_fixed_and_reproducible(tmp_path):
    source = tmp_path / "source"
    fixture_benchmark(source)
    first, second = tmp_path / "first", tmp_path / "second"
    report = build_cases(source, first)
    build_cases(source, second)
    assert report["count"] == 30
    assert report["counts_by_source"] == {"A": 15, "B": 15}
    cases = load_jsonl(first / "cases.jsonl")
    assert len({case["source_document"]["id"] for case in cases}) == 30
    assert all("gold_label" not in case for case in cases)
    assert (first / "cases.jsonl").read_bytes() == (second / "cases.jsonl").read_bytes()
    assert Counter(case["source_corpus"] for case in cases) == {"A": 15, "B": 15}


def test_corrupt_input_and_reuse_of_output_fail(tmp_path):
    source, target = tmp_path / "source", tmp_path / "out"
    fixture_benchmark(source)
    build_cases(source, target)
    with pytest.raises(ValueError, match="fresh"):
        build_cases(source, target)
    (source / "development.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_cases(source, tmp_path / "new")
