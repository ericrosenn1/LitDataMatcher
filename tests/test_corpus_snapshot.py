"""Integrity challenges use explicit fixtures, not scientific benchmark claims."""

import gzip
import hashlib
import json

import pytest

from litdatamatcher.corpus_snapshot import PUBMED_FIELDS, qualify_corpus

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def corpus(tmp_path, count=1):
    root = tmp_path / "corpus"
    acquisition, normalized = {}, {}
    (root / "manifests/acquisition").mkdir(parents=True)
    (root / "CORPUS_ROOT.json").write_text(json.dumps({"store_uuid": "test-only"}))
    for index in range(count):
        name = f"pubmed26n{index + 1:04d}.xml.gz"
        raw = root / "raw/pubmed/2026_baseline" / name
        raw.parent.mkdir(parents=True, exist_ok=True)
        xml = f"<PubmedArticleSet><PubmedArticle><PMID>{index + 1}</PMID><ArticleTitle>Example {index + 1}</ArticleTitle></PubmedArticle></PubmedArticleSet>"
        raw.write_bytes(gzip.compress(xml.encode(), mtime=0))
        row = {field: None for field in PUBMED_FIELDS}
        row.update(pmid=str(index + 1), title=f"Example {index + 1}", source_shard=name)
        output = root / f"normalized/literature/pubmed_2026_baseline/{index + 1}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [row], schema=pa.schema([(field, pa.string()) for field in sorted(PUBMED_FIELDS)])
            ),
            output,
        )
        acquisition[name] = {
            "status": "CHECKSUM_PASS",
            "bytes": raw.stat().st_size,
            "md5": hashlib.md5(raw.read_bytes()).hexdigest(),
        }
        normalized[name] = {
            "status": "NORMALIZED",
            "records": 1,
            "output_relative_path": output.relative_to(root).as_posix(),
        }
    (root / "manifests/acquisition/pubmed_2026_baseline_state.json").write_text(
        json.dumps({"expected_shards": count, "files": acquisition})
    )
    (root / "manifests/acquisition/pubmed_2026_normalization_state.json").write_text(
        json.dumps({"shards": normalized})
    )
    return root


def test_valid_snapshot_preserves_inputs_and_binds_checksums(tmp_path):
    root = corpus(tmp_path)
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    receipt = qualify_corpus(root, tmp_path / "audit", include_updates=False, verify_raw=True)
    assert receipt["status"] == "PASS"
    assert receipt["rows"] == receipt["shards_audited"] == 1
    assert receipt["raw_checksums_recomputed"] is True
    assert receipt["shards"][0]["raw_md5"] == receipt["shards"][0]["acquisition_md5"]
    assert {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before
    with pytest.raises(FileExistsError):
        qualify_corpus(root, tmp_path / "audit", include_updates=False)


def test_sampled_audit_cannot_claim_full_qualification(tmp_path):
    receipt = qualify_corpus(
        corpus(tmp_path, 2), tmp_path / "audit", include_updates=False, max_shards=1
    )
    assert receipt["status"] == "PARTIAL"
    assert receipt["shards_available"] == 2


def test_corrupt_parquet_fails_without_replacing_source(tmp_path):
    root = corpus(tmp_path)
    path = next(root.rglob("*.parquet"))
    path.write_bytes(b"partial write")
    receipt = qualify_corpus(root, tmp_path / "audit", include_updates=False)
    assert receipt["status"] == "FAIL"
    assert path.read_bytes() == b"partial write"


def test_wrong_raw_digest_fails(tmp_path):
    root = corpus(tmp_path)
    path = root / "manifests/acquisition/pubmed_2026_baseline_state.json"
    value = json.loads(path.read_text())
    next(iter(value["files"].values()))["md5"] = "0" * 32
    path.write_text(json.dumps(value))
    receipt = qualify_corpus(root, tmp_path / "audit", include_updates=False, verify_raw=True)
    assert "raw_md5_disagrees_with_acquisition" in receipt["shards"][0]["issues"]


def test_empty_acquisition_cannot_pass(tmp_path):
    root = corpus(tmp_path, 0)
    receipt = qualify_corpus(root, tmp_path / "audit", include_updates=False)
    assert receipt["status"] == "FAIL"
    assert "no_qualifiable_shards" in receipt["issues"]


def test_missing_nonfirst_identifier_cannot_pass(tmp_path):
    root = corpus(tmp_path)
    path = next(root.rglob("*.parquet"))
    first = pq.read_table(path)
    second = {field: None for field in PUBMED_FIELDS}
    second.update(title="Missing identifier", source_shard="pubmed26n0001.xml.gz")
    pq.write_table(
        pa.concat_tables([first, pa.Table.from_pylist([second], schema=first.schema)]), path
    )
    state = root / "manifests/acquisition/pubmed_2026_normalization_state.json"
    value = json.loads(state.read_text())
    next(iter(value["shards"].values()))["records"] = 2
    state.write_text(json.dumps(value))
    result = qualify_corpus(root, tmp_path / "audit", include_updates=False)
    assert result["status"] == "FAIL"
    assert "missing_pmid_rows" in result["shards"][0]["issues"]


def test_traversal_in_normalization_manifest_fails(tmp_path):
    root = corpus(tmp_path)
    path = root / "manifests/acquisition/pubmed_2026_normalization_state.json"
    value = json.loads(path.read_text())
    next(iter(value["shards"].values()))["output_relative_path"] = "../escape.parquet"
    path.write_text(json.dumps(value))
    receipt = qualify_corpus(root, tmp_path / "audit", include_updates=False)
    assert receipt["status"] == "FAIL"
    assert "escapes declared root" in receipt["shards"][0]["issues"][0]
