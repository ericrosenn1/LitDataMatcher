"""Regression contracts for candidate retrieval; fake vectors test plumbing only."""

import hashlib
import json

import numpy as np
import pytest

import litdatamatcher.retrieval as retrieval


class FakeEncoder:
    calls = 0
    revision = "a" * 40

    def __init__(self, model_dir, device="cpu"):
        self.device = device
        self.manifest = {"revision": self.revision, "model_id": "test-only"}
        self.vectors = None
        self.records = []

    def encode(self, texts):
        type(self).calls += 1
        vectors = np.array(
            [list(hashlib.sha256(text.encode()).digest()[:4]) for text in texts], dtype=np.float32
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query, k=10):
        scores = self.vectors @ self.encode([query])[0]
        order = sorted(
            range(len(scores)), key=lambda i: (-float(scores[i]), self.records[i]["id"])
        )[:k]
        return [{"id": self.records[i]["id"], "score": float(scores[i])} for i in order]


@pytest.fixture
def encoder(monkeypatch):
    FakeEncoder.calls, FakeEncoder.revision = 0, "a" * 40
    monkeypatch.setattr(retrieval, "PretrainedSemanticIndex", FakeEncoder)
    return FakeEncoder


RECORDS = [
    {"id": "a", "text": "Sleep cognition and human ageing."},
    {"id": "b", "text": "Human glucose and insulin metabolism."},
]


def test_second_fit_uses_verified_vectors_without_encoding(encoder, tmp_path):
    first = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert first.receipt["origin"] == "fresh_model_encoding"
    assert encoder.calls == 1
    second = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert second.receipt["origin"] == "verified_cache_replay"
    assert encoder.calls == 1
    assert np.array_equal(first.index.vectors, second.index.vectors)


def test_dependency_drift_uses_new_vector_cache_without_overwriting_old(
    encoder, monkeypatch, tmp_path
):
    versions = {"tokenizers": "fixture-1", "safetensors": "fixture-1"}
    monkeypatch.setattr(retrieval, "runtime_dependency_versions", lambda: dict(versions))
    first = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    old = tmp_path / first.fingerprint / "vectors.npy"
    original = old.read_bytes()
    versions["tokenizers"] = "fixture-2"
    second = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert second.fingerprint != first.fingerprint
    assert encoder.calls == 2
    assert old.read_bytes() == original


def test_text_order_and_model_changes_invalidate_cache(encoder, tmp_path):
    first = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    reordered = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS[::-1])
    encoder.revision = "b" * 40
    revised = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert len({first.fingerprint, reordered.fingerprint, revised.fingerprint}) == 3
    assert encoder.calls == 3


def test_corruption_fails_closed_until_explicit_rebuild(encoder, tmp_path):
    first = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    (tmp_path / first.fingerprint / "vectors.npy").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert encoder.calls == 1
    repaired = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS, force_rebuild=True)
    assert repaired.receipt["origin"] == "fresh_model_encoding"
    assert encoder.calls == 2


def test_batch_search_matches_individual_ranking(encoder, tmp_path):
    index = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    individual = [index.search(query) for query in ["sleep", "insulin"]]
    batched = index.search_many(["sleep", "insulin"])
    assert [[r["id"] for r in rows] for rows in individual] == [
        [r["id"] for r in rows] for rows in batched
    ]


def test_invalid_vectors_cannot_be_published(encoder, monkeypatch, tmp_path):
    monkeypatch.setattr(
        encoder, "encode", lambda self, texts: np.full((len(texts), 4), np.nan, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="Non-finite"):
        retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    assert not list(tmp_path.rglob("manifest.json"))


def test_bm25_relevance_and_empty_query():
    index = retrieval.BM25Index().fit(RECORDS)
    assert index.search("sleep")[0]["id"] == "a"
    assert index.search("insulin")[0]["id"] == "b"
    assert index.search("unrelated") == []
    assert index.search("") == []


def test_duplicate_records_are_rejected(encoder, tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval.BM25Index().fit(RECORDS + RECORDS)
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS + RECORDS)


def test_fusion_has_stable_ties_and_preserves_components():
    result = retrieval.reciprocal_rank_fusion(
        [[{"id": "b", "score": 1}], [{"id": "a", "score": 100}]]
    )
    assert [r["id"] for r in result] == ["a", "b"]
    assert result[0]["score_kind"] == "rrf_not_probability"
    assert result[0]["components"][0]["original_score"] == 100
    json.dumps(result, allow_nan=False)


def test_fusion_rejects_duplicate_component_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval.reciprocal_rank_fusion([[{"id": "a"}, {"id": "a"}]])


def test_cache_rejects_changed_shape_metadata(encoder, tmp_path):
    index = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    path = tmp_path / index.fingerprint / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["record_count"] = 999
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata integrity"):
        retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)


def test_nonfinite_queries_are_rejected(encoder, monkeypatch, tmp_path):
    index = retrieval.CachedSemanticIndex("fake", tmp_path).fit(RECORDS)
    monkeypatch.setattr(
        index.index, "encode", lambda texts: np.full((len(texts), 4), np.nan, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="Non-finite"):
        index.search_many(["query"])
    with pytest.raises(ValueError, match="Non-finite"):
        index.search("query")


def test_fusion_rejects_nonfinite_original_scores():
    with pytest.raises(ValueError, match="Non-finite"):
        retrieval.reciprocal_rank_fusion([[{"id": "a", "score": float("nan")}]])
