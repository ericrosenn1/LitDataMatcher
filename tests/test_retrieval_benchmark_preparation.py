"""Engineering fixtures only: partition guards and exact paper-link recovery."""

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.campaign_20261031 import prepare_retrieval_benchmark as prep


def write_table(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([(name, pa.string()) for name in columns])), path)


def fixture_inputs(tmp_path):
    root = tmp_path / "corpus"
    questions, edges, assignments, documents = [], [], [], []
    for index in range(1, 65):
        qid = f"q{index}"
        questions.append({"question_id": qid, "question_text": f"  Original question {index}?\nUnchanged. ",
                          "source_corpus": "PubMedQA", "source_partition": "PQA-L", "provenance_relative_path": "raw/pqa.json"})
        edges.append({"question_id": qid, "identifier_type": "PMID", "identifier_value": str(index), "relation": "source_linked"})
        assignments.append({"question_id": qid, "split": "train" if index <= 32 else "development"})
        documents.append({"pmid": str(index), "pmcid": f"PMC{1000 + index}", "doi": f"10.1234/{index}",
                          "title": f"Source title {index}", "abstract": f"Source abstract {index}.", "publication_types_json": '[]'})
    # A PMCID/PMID alias crossing legacy groups must still be removed from training.
    edges[32]["identifier_type"], edges[32]["identifier_value"] = "PMCID", "PMC1001"
    documents[1]["abstract"] = None
    documents[2]["publication_types_json"] = '["Retracted Publication"]'
    documents[3]["title"], documents[3]["abstract"] = documents[39]["title"], documents[39]["abstract"]
    questions[4]["question_text"] = questions[44]["question_text"]
    for qid, source, part, kind, value, split in [
        ("reserved", "PubMedQA", "PQA-L", "PMID", "65", "evaluation_candidate"),
        ("official_eb_test", "EvidenceBench", "test", "PMID", "65", "train"),
        ("official_ei_test", "Evidence Inference 2.0", None, "PMCID", "999999", "train"),
    ]:
        questions.append({"question_id": qid, "question_text": "RESERVED TEXT MUST NOT BE MATERIALIZED", "source_corpus": source,
                          "source_partition": part, "provenance_relative_path": "raw/reserved.json"})
        edges.append({"question_id": qid, "identifier_type": kind, "identifier_value": value, "relation": "source_linked"})
        assignments.append({"question_id": qid, "split": split})
    write_table(root / "normalized/questions/questions.parquet", questions,
                ["question_id", "question_text", "source_corpus", "source_partition", "provenance_relative_path"])
    write_table(root / "normalized/question_paper/question_paper.parquet", edges,
                ["question_id", "identifier_type", "identifier_value", "relation"])
    assignment_path = tmp_path / "assignments.parquet"
    write_table(assignment_path, assignments, ["question_id", "split"])
    shard_path = root / "normalized/literature/pubmed_2026_baseline/pubmed26n0001.parquet"
    write_table(shard_path, documents, ["pmid", "pmcid", "doi", "title", "abstract", "publication_types_json"])
    official = root / prep.OFFICIAL_ROOT
    official.mkdir(parents=True)
    (official / "train_article_ids.txt").write_text("1001\n", encoding="utf-8")
    (official / "validation_article_ids.txt").write_text("1002\n", encoding="utf-8")
    stat = shard_path.stat()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "PASS", "raw_checksums_recomputed": True, "shards": [{
        "relative_path": str(shard_path.relative_to(root)), "source": "baseline", "source_shard": "pubmed26n0001.xml.gz",
        "status": "PASS", "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": prep.file_hash(shard_path),
    }]}), encoding="utf-8")
    return argparse.Namespace(corpus_root=root, assignments=assignment_path, receipt=receipt, out=tmp_path / "output",
                              seed="fixture-only", max_train_per_source=1000, max_development_per_source=100)


def test_fail_receipt_blocks_before_any_parquet_read(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"FAIL","raw_checksums_recomputed":true}')
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("Unqualified source read"))
    with pytest.raises(ValueError, match="must be PASS"):
        prep.build(argparse.Namespace(receipt=receipt, corpus_root=tmp_path / "corpus", out=tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_partition_and_identifier_rules():
    assert prep.identifier("123", "EvidenceBench_paper_id") is None
    assert prep.identifier("PMC123", "EvidenceBench_paper_id") == "pmcid:PMC123"
    assert prep.identifier("123", "PMCID") == "pmcid:PMC123"
    assert not prep.partition_allowed("EvidenceBench", "train", "test", "pmid:1", set(), set())
    assert not prep.partition_allowed("EvidenceBench", "train", "dev", "pmid:1", set(), set())
    assert prep.partition_allowed("EvidenceBench", "development", "train", "pmid:1", set(), set())
    assert not prep.partition_allowed("Evidence Inference 2.0", "train", None, "pmcid:PMC2", {"pmcid:PMC1"}, {"pmcid:PMC2"})


def test_real_link_contract_fixture_and_reserved_text_guard(tmp_path, monkeypatch):
    args = fixture_inputs(tmp_path)
    original_read = pq.read_table
    def guarded_read(path, *positional, **keywords):
        if "question_text" in keywords.get("columns", []):
            allowed = keywords["filters"][0][2]
            assert not {"reserved", "official_eb_test", "official_ei_test"} & set(allowed)
        return original_read(path, *positional, **keywords)
    monkeypatch.setattr(pq, "read_table", guarded_read)
    manifest = prep.build(args)
    assert manifest["counts"]["development"] >= 30
    assert all(value == 0 for value in manifest["leakage_checks"].values())
    counts = manifest["exclusions_and_accounting"]
    for name in ("official_partition_excluded", "missing_abstract", "retracted_publication", "train_development_canonical_overlap", "train_development_query_overlap", "train_development_document_text_overlap"):
        assert counts[name] > 0
    train = [json.loads(line) for line in (args.out / "train.jsonl").read_text().splitlines()]
    assert all(row["query"].startswith("  Original") and row["query"].endswith("Unchanged. ") for row in train)
    documents = [json.loads(line) for line in (args.out / "documents.jsonl").read_text().splitlines()]
    assert all(row["source_sha256"] and row["row_locator"]["row_number"] >= 0 for row in documents)
    for output in manifest["outputs"]:
        assert prep.file_hash(args.out / output["path"]) == output["sha256"]
    with pytest.raises(ValueError, match="must be new"):
        prep.build(args)


def test_changed_shard_and_update_receipt_are_rejected(tmp_path):
    args = fixture_inputs(tmp_path)
    receipt = json.loads(args.receipt.read_text())
    receipt["shards"][0]["source"] = "updates"
    args.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="updates excluded"):
        prep.build(args)
    receipt["shards"][0]["source"] = "baseline"
    receipt["shards"][0]["bytes"] += 1
    args.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="changed since"):
        prep.build(args)


def test_development_minimum_cannot_be_silently_weakened(tmp_path):
    args = fixture_inputs(tmp_path)
    args.max_development_per_source = 29
    with pytest.raises(ValueError, match=">=30"):
        prep.build(args)
    assert not args.out.exists()


def test_selection_is_reproducible_and_per_source_caps_are_respected(tmp_path):
    args = fixture_inputs(tmp_path)
    args.max_train_per_source = 10
    first = prep.build(args)
    original = {name: (args.out / name).read_bytes() for name in ("train.jsonl", "development.jsonl", "documents.jsonl")}
    args.out = tmp_path / "repeat"
    second = prep.build(args)
    assert first["counts"] == second["counts"]
    assert first["counts"]["train"] == 10
    assert all((args.out / name).read_bytes() == value for name, value in original.items())


def test_raw_checksum_qualification_is_required_and_sources_are_untouched(tmp_path):
    args = fixture_inputs(tmp_path)
    before = {path: prep.file_hash(path) for path in args.corpus_root.rglob("*") if path.is_file()}
    receipt = json.loads(args.receipt.read_text())
    receipt["raw_checksums_recomputed"] = False
    args.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="raw_checksums_recomputed"):
        prep.build(args)
    assert all(prep.file_hash(path) == value for path, value in before.items())
    assert not args.out.exists()
