"""Synthetic candidate-expansion contracts; no inference or protected question data."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.retrieval_expansion import (
    build_expansion,
    canonical_identifier,
    file_hash,
    read_rows,
    resolve_aliases,
)


def fixture(tmp_path):
    corpus, benchmark, output = (tmp_path / n for n in ("corpus", "benchmark", "output"))
    benchmark.mkdir()
    shards = []
    for part in range(3):
        path = corpus / f"baseline/part{part}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "pmid": str(n),
                "pmcid": "PMC3" if n == 3 else None,
                "doi": None,
                "title": f"Study {n}",
                "abstract": f"Distinct abstract {n}",
                "publication_types_json": json.dumps(
                    ["Retracted Publication"] if n == 5 else ["Journal Article"]
                ),
            }
            for n in range(part * 8 + 1, part * 8 + 9)
        ]
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pa.schema([(k, pa.string()) for k in rows[0]])), path
        )
        shards.append(
            {
                "relative_path": path.relative_to(corpus).as_posix(),
                "source": "baseline",
                "status": "PASS",
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    snapshot = tmp_path / "snapshot.json"
    atomic_json(snapshot, {"status": "PASS", "raw_checksums_recomputed": True, "shards": shards})
    edges = tmp_path / "edges.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"identifier_type": "PMCID", "identifier_value": "PMC3"},
                {"identifier_type": "EvidenceBench_paper_id", "identifier_value": "pubmed_7"},
            ]
        ),
        edges,
    )
    documents = [
        {
            "id": f"pmid:{n}",
            "pmid": str(n),
            "title": f"Study {n}",
            "abstract": f"Distinct abstract {n}",
            "text": f"Study {n}\n\nDistinct abstract {n}",
        }
        for n in (1, 2)
    ]
    queries = {
        "train": [
            {
                "id": "train",
                "query": "question one",
                "positive_doc_id": "pmid:1",
                "source_corpus": "fixture",
                "split": "train",
            }
        ],
        "development": [
            {
                "id": "dev",
                "query": "question two",
                "positive_doc_id": "pmid:2",
                "source_corpus": "fixture",
                "split": "development",
            }
        ],
    }
    for name, rows in {**queries, "documents": documents}.items():
        (benchmark / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(
        benchmark / "BENCHMARK_MANIFEST.json",
        {
            "status": "PASS",
            "counts": {"train": 1, "development": 1, "documents": 2},
            "leakage_checks": {
                "canonical_paper_overlap": 0,
                "canonical_query_overlap": 0,
                "canonical_document_text_overlap": 0,
            },
            "outputs": [
                {"path": p.name, "sha256": file_hash(p), "bytes": p.stat().st_size}
                for p in sorted(benchmark.glob("*.jsonl"))
            ],
        },
    )
    return corpus, snapshot, edges, benchmark, output


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("PMID", "001", "pmid:1"),
        ("PMCID", "PMC2", "pmcid:2"),
        ("EvidenceBench_paper_id", "pubmed_3", "pmid:3"),
        ("EvidenceBench_paper_id", "pmc_4", "pmcid:4"),
        ("DOI", "https://doi.org/10.1000/ABC", "doi:10.1000/abc"),
    ],
)
def test_typed_identity_rules(kind, value, expected):
    assert canonical_identifier(kind, value) == expected


@pytest.mark.parametrize(
    "kind,value",
    [
        ("EvidenceBench_paper_id", "12345"),
        ("internal", "12345"),
        ("PMID", "zero"),
        ("DOI", "unknown"),
    ],
)
def test_no_bare_identifier_reinterpretation(kind, value):
    with pytest.raises(ValueError):
        canonical_identifier(kind, value)


def test_expansion_is_deterministic_prefix_view_paired_and_source_preserving(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    originals = {p: file_hash(p) for p in args[0].rglob("*.parquet")}
    result = build_expansion(*args, sizes=(6, 10))
    assert result["status"] == "PASS"
    small = read_rows(args[-1] / "pool-6/full/documents.jsonl")
    large = read_rows(args[-1] / "pool-10/full/documents.jsonl")
    abstract = read_rows(args[-1] / "pool-10/abstract_only/documents.jsonl")
    assert [r["id"] for r in small] == [r["id"] for r in large[:6]]
    assert [r["id"] for r in large] == [r["id"] for r in abstract]
    assert all(r["text"] == r["abstract"] for r in abstract)
    assert not {"pmid:3", "pmid:5", "pmid:7"} & {r["id"] for r in large[2:]}
    for size in (6, 10):
        for view in ("full", "abstract_only"):
            for split in ("train", "development"):
                assert (args[-1] / f"pool-{size}/{view}/{split}.jsonl").read_bytes() == (
                    args[3] / f"{split}.jsonl"
                ).read_bytes()
    assert {p: file_hash(p) for p in originals} == originals

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed pool should be reused without scanning source rows")

    monkeypatch.setattr(pq, "read_table", forbidden)
    assert build_expansion(*args, sizes=(6, 10))["reused"] is True


def test_source_change_and_updates_fail_closed(tmp_path):
    args = fixture(tmp_path)
    snapshot = json.loads(args[1].read_text())
    snapshot["shards"][0]["source"] = "updates"
    atomic_json(args[1], snapshot)
    with pytest.raises(ValueError, match="annual-baseline"):
        build_expansion(*args, sizes=(6,))


def test_pool_corruption_is_not_silently_overwritten(tmp_path):
    args = fixture(tmp_path)
    build_expansion(*args, sizes=(6,))
    path = args[-1] / "pool-6/full/documents.jsonl"
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="output changed"):
        build_expansion(*args, sizes=(6,))
    assert path.read_text() == "corrupt"


def test_original_root_cannot_be_output(tmp_path):
    args = list(fixture(tmp_path))
    args[-1] = args[0] / "derived"
    with pytest.raises(ValueError, match="separate"):
        build_expansion(*args, sizes=(6,))


def test_malformed_optional_identifier_excludes_candidate_without_coercion(tmp_path):
    args = fixture(tmp_path)
    snapshot = json.loads(args[1].read_text())
    record = snapshot["shards"][0]
    path = args[0] / record["relative_path"]
    table = pq.read_table(path)
    values = table.to_pylist()
    values[5]["pmcid"] = "wh_2022_144"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), path)
    record.update(
        sha256=file_hash(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
    )
    atomic_json(args[1], snapshot)
    build_expansion(*args, sizes=(20,))
    candidates = read_rows(args[-1] / "pool-20/full/documents.jsonl")
    assert len(candidates) == 20
    assert "pmid:6" not in {row["id"] for row in candidates}


def test_alias_closure_follows_backward_transitive_links_and_normalizes(tmp_path):
    schema = pa.schema([(name, pa.string()) for name in ("pmid", "pmcid", "doi")])
    tables = [
        [
            {"pmid": "0002", "pmcid": "PMCID:PMC00200", "doi": "https://doi.org/10.1000/Test"},
            {"pmid": "2", "pmcid": "PMC100", "doi": None},
        ],
        [{"pmid": "1", "pmcid": "PMC100", "doi": None}],
    ]
    shards = []
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index, values in enumerate(tables):
        path = corpus / f"part{index}.parquet"
        pq.write_table(pa.Table.from_pylist(values, schema=schema), path)
        shards.append(
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    aliases, receipt = resolve_aliases(
        corpus, shards, {"pmid:1"}, tmp_path / "aliases.json", "test", lambda _: None
    )
    assert {"pmid:2", "pmcid:200", "doi:10.1000/test"} <= aliases
    assert receipt["complete"] and receipt["pass_number"] >= 3
