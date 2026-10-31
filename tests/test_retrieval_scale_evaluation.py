"""Synthetic plumbing tests; these are not scientific retrieval evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

import litdatamatcher.retrieval as retrieval
import litdatamatcher.retrieval_scale_evaluation as evaluation
from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.retrieval_expansion import file_hash


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _outputs(root):
    return [
        {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.jsonl"))
    ]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "pools"
    root.mkdir()
    docs = [
        {
            "id": f"pmid:{i}",
            "title": f"Study {i}",
            "abstract": f"Abstract about topic {i}",
            "text": f"Study {i}\n\nAbstract about topic {i}",
            "pmid": str(i),
        }
        for i in range(1, 10)
    ]
    queries = [
        {
            "id": f"q{i}",
            "query": f"Study {i}",
            "positive_doc_id": f"pmid:{i}",
            "source_corpus": "a" if i == 2 else "b",
            "split": "development",
        }
        for i in (2, 3)
    ]
    train = [
        {
            "id": "train1",
            "query": "Different training question",
            "positive_doc_id": "pmid:1",
            "source_corpus": "a",
            "split": "train",
        }
    ]
    seed = tmp_path / "seed"
    for name, rows in (("train", train), ("development", queries), ("documents", docs[:3])):
        _jsonl(seed / f"{name}.jsonl", rows)
    seed_manifest = {
        "status": "PASS",
        "counts": {"train": 1, "development": 2, "documents": 3},
        "outputs": _outputs(seed),
    }
    atomic_json(root / "SEED_BENCHMARK_MANIFEST.json", seed_manifest)
    inputs = {
        "version": "unlinked-pubmed-candidate-expansion-1.1",
        "sizes": [6, 9],
        "views": ["full", "abstract_only"],
        "benchmark_sha256": file_hash(root / "SEED_BENCHMARK_MANIFEST.json"),
    }
    fingerprint = digest(inputs)
    closure = {"fingerprint": fingerprint, "complete": True, "aliases": ["pmid:1"]}
    atomic_json(root / "EXCLUSIONS.json", {**closure, "content_sha256": digest(closure)})
    receipts = []
    for size in (6, 9):
        for view in ("full", "abstract_only"):
            folder = root / f"pool-{size}" / view
            for name, rows in (("train", train), ("development", queries)):
                _jsonl(folder / f"{name}.jsonl", rows)
            records = [
                dict(row, text=row["abstract"] if view == "abstract_only" else row["text"])
                for row in docs[:size]
            ]
            _jsonl(folder / "documents.jsonl", records)
            manifest = {
                "status": "PASS",
                "counts": {"train": 1, "development": 2, "documents": size},
                "candidate_view": view,
                "expansion_fingerprint": fingerprint,
                "expansion": {"seed_documents": 3},
                "outputs": _outputs(folder),
                "ordered_candidate_ids_sha256": digest([row["id"] for row in records]),
            }
            atomic_json(folder / "BENCHMARK_MANIFEST.json", manifest)
            receipts.append(
                {
                    "path": str((folder / "BENCHMARK_MANIFEST.json").relative_to(root)),
                    "sha256": file_hash(folder / "BENCHMARK_MANIFEST.json"),
                }
            )
    atomic_json(
        root / "EXPANSION.json",
        {
            "status": "PASS",
            "inputs": inputs,
            "fingerprint": fingerprint,
            "sizes": [6, 9],
            "views": ["full", "abstract_only"],
            "manifest_receipts": receipts,
        },
    )
    models = []
    for name in ("pretrained", "trained"):
        path = tmp_path / name
        path.mkdir()
        weights = path / "weights.safetensors"
        weights.write_bytes(name.encode())  # Not deserialized: the encoder is an explicit fake.
        atomic_json(
            path / "MODEL_MANIFEST.json",
            {
                "model_id": name,
                "revision": ("1" if name == "pretrained" else "2") * 40,
                "license": "fixture",
                "files": [{"path": weights.name, "sha256": file_hash(weights)}],
            },
        )
        models.append(path)
    FakeEncoder.calls = []
    monkeypatch.setattr(retrieval, "PretrainedSemanticIndex", FakeEncoder)
    return root, *models, tmp_path / "results"


class FakeEncoder:
    calls = []

    def __init__(self, path, device="cpu"):
        self.device = device
        self.manifest = json.loads((Path(path) / "MODEL_MANIFEST.json").read_text())

    def encode(self, texts):
        self.calls.append((self.manifest["model_id"], tuple(texts)))
        vectors = []
        for text in texts:
            values = np.array(
                [1 + int(digest(text)[i : i + 2], 16) for i in range(0, 16, 2)], dtype=np.float32
            )
            vectors.append(values / np.linalg.norm(values))
        return np.asarray(vectors, dtype=np.float32)


def test_full_run_shared_encodings_resume_and_metrics(fixture, monkeypatch):
    result = evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    assert result["status"] == "PASS"
    assert len(result["conditions"]) == 4
    assert len(FakeEncoder.calls) == 6
    assert sorted(len(texts) for _, texts in FakeEncoder.calls) == [2, 2, 9, 9, 9, 9]
    condition = result["conditions"]["9/full"]
    assert set(condition["metrics"]) == set(evaluation.METHODS)
    # Tokenizer omits single-digit numbers, so the lexical fixture has exact ties.
    assert condition["metrics"]["bm25"]["mrr_at_k"] == pytest.approx(5 / 12)
    assert set(condition["paired"]["trained_vs_bm25"]["by_source"]) == {"a", "b"}
    assert "query_results" not in condition["metrics"]["trained"]
    out = fixture[-1]
    hashes = {path: file_hash(path) for path in out.rglob("rankings.jsonl.gz")}
    assert len(hashes) == 20

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must neither initialize encoders nor rebuild BM25")

    monkeypatch.setattr(retrieval, "PretrainedSemanticIndex", forbidden)
    monkeypatch.setattr(evaluation, "BM25Index", forbidden)
    resumed = evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert resumed["conditions"] == result["conditions"]
    assert {path: file_hash(path) for path in hashes} == hashes


def test_interruption_preserves_completed_methods_and_vectors(fixture, monkeypatch):
    original = evaluation._method

    def interrupted(out, folder, run_fp, method, *args):
        if method == "trained":
            raise KeyboardInterrupt("fixture interruption")
        return original(out, folder, run_fp, method, *args)

    monkeypatch.setattr(evaluation, "_method", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    assert json.loads((fixture[-1] / "RUN.json").read_text())["status"] == "INTERRUPTED"
    path = fixture[-1] / "conditions/pool-6/full/pretrained/rankings.jsonl.gz"
    before = file_hash(path)
    calls = len(FakeEncoder.calls)
    monkeypatch.setattr(evaluation, "_method", original)
    result = evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert result["status"] == "PASS" and file_hash(path) == before
    assert calls == 4 and len(FakeEncoder.calls) == 6


def test_corrupt_vectors_fail_without_reencoding(fixture):
    evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    path = next((fixture[-1] / "vectors").rglob("vectors.npy"))
    path.write_bytes(b"damaged")
    count = len(FakeEncoder.calls)
    with pytest.raises(ValueError, match="integrity failure"):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert len(FakeEncoder.calls) == count


def test_changed_model_fingerprint_preserves_completed_run(fixture):
    evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    saved = (fixture[-1] / "RUN.json").read_bytes()
    manifest = fixture[2] / "MODEL_MANIFEST.json"
    value = json.loads(manifest.read_text())
    value["revision"] = "3" * 40
    atomic_json(manifest, value)
    with pytest.raises(ValueError, match="identical"):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert (fixture[-1] / "RUN.json").read_bytes() == saved


def test_no_resume_cannot_overwrite_completed_results(fixture):
    evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    saved = (fixture[-1] / "RUN.json").read_bytes()
    with pytest.raises(ValueError, match="--resume"):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    assert (fixture[-1] / "RUN.json").read_bytes() == saved


def test_input_tampering_is_rejected(fixture):
    root = fixture[0]
    path = root / "pool-6/full/development.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="integrity failure"):
        evaluation.validate_pools(root, sizes=(6, 9))
    assert not fixture[-1].exists()


def test_dense_prefix_exact_scores_ties_and_nonfinite():
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    queries = np.array([[1, 0]], dtype=np.float32)
    records = [{"id": name} for name in ("b", "c", "a")]
    assert [row["id"] for row in evaluation._rank_dense(vectors, queries, records)[0]] == [
        "a",
        "b",
        "c",
    ]
    assert [row["id"] for row in evaluation._rank_dense(vectors[:2], queries, records[:2])[0]] == [
        "b",
        "c",
    ]
    queries[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        evaluation._rank_dense(vectors, queries, records)


def test_output_cannot_overlap_protected_inputs(fixture):
    with pytest.raises(ValueError, match="separate"):
        evaluation.evaluate_scale(*fixture[:3], fixture[0] / "results", sizes=(6, 9))


def _mutate_condition(root, size, view, mutate, filename="documents.jsonl"):
    folder = root / f"pool-{size}" / view
    path = folder / filename
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    _jsonl(path, rows)
    manifest_path = folder / "BENCHMARK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = _outputs(folder)
    if filename == "documents.jsonl":
        manifest["ordered_candidate_ids_sha256"] = digest([row["id"] for row in rows])
    atomic_json(manifest_path, manifest)
    expansion_path = root / "EXPANSION.json"
    expansion = json.loads(expansion_path.read_text())
    for row in expansion["manifest_receipts"]:
        if Path(row["path"]) == manifest_path.relative_to(root):
            row["sha256"] = file_hash(manifest_path)
    atomic_json(expansion_path, expansion)


def test_missing_title_is_preserved_and_not_fabricated(fixture):
    def remove_title(rows):
        rows[-1].update(title="", text=rows[-1]["abstract"])

    for view in ("full", "abstract_only"):
        _mutate_condition(fixture[0], 9, view, remove_title)
    assert len(evaluation.validate_pools(fixture[0], sizes=(6, 9))["queries"]) == 2


def test_rehashed_changed_query_is_not_a_new_frozen_baseline(fixture):
    def change_query(rows):
        rows[0]["query"] = "Altered query text"

    _mutate_condition(fixture[0], 9, "full", change_query, "development.jsonl")
    with pytest.raises(ValueError, match="original frozen"):
        evaluation.validate_pools(fixture[0], sizes=(6, 9))


def test_same_ids_do_not_allow_changed_prefix_text(fixture):
    def change_text(rows):
        rows[0]["abstract"] += " altered"
        rows[0]["text"] = rows[0]["title"] + "\n\n" + rows[0]["abstract"]

    _mutate_condition(fixture[0], 9, "full", change_text)
    with pytest.raises(ValueError, match="content differs"):
        evaluation.validate_pools(fixture[0], sizes=(6, 9))


def test_corrupt_completed_ranking_fails_without_new_inference(fixture):
    evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    path = next(fixture[-1].rglob("rankings.jsonl.gz"))
    path.write_bytes(b"bad gzip bytes")
    calls = len(FakeEncoder.calls)
    with pytest.raises(ValueError, match="integrity failure"):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert len(FakeEncoder.calls) == calls


def test_interrupted_query_encoding_retains_complete_document_cache(fixture, monkeypatch):
    encode = FakeEncoder.encode

    def interrupt_query(self, texts):
        if len(texts) == 2:
            raise KeyboardInterrupt("query encoding interrupted")
        return encode(self, texts)

    monkeypatch.setattr(FakeEncoder, "encode", interrupt_query)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9))
    assert len(FakeEncoder.calls) == 1 and len(FakeEncoder.calls[0][1]) == 9
    monkeypatch.setattr(FakeEncoder, "encode", encode)
    resumed = evaluation.evaluate_scale(*fixture, sizes=(6, 9), resume=True)
    assert resumed["status"] == "PASS" and len(FakeEncoder.calls) == 6
    assert (
        resumed["encodings"]["full/pretrained"]["timings"]["document_first_origin"]
        == "verified_cache_replay"
    )
    fresh = fixture[-1].with_name("fresh-results")
    evaluation.evaluate_scale(*fixture[:3], fresh, sizes=(6, 9))
    for path in fixture[-1].rglob("rankings.jsonl.gz"):
        assert file_hash(path) == file_hash(fresh / path.relative_to(fixture[-1]))


@pytest.mark.parametrize("target", ["query", "model"])
def test_input_drift_during_run_cannot_publish_pass(fixture, target):
    changed = []

    def progress(event):
        if changed:
            return
        if event["stage"] == "ranking_or_verified_replay":
            path = (
                fixture[0] / "pool-6/full/development.jsonl"
                if target == "query"
                else fixture[1] / "weights.safetensors"
            )
            path.write_bytes(path.read_bytes() + b" ")
            changed.append(True)

    with pytest.raises(ValueError, match="integrity failure|Model integrity"):
        evaluation.evaluate_scale(*fixture, sizes=(6, 9), progress=progress)
    assert json.loads((fixture[-1] / "RUN.json").read_text())["status"] == "FAIL"
