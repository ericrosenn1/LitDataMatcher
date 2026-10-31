"""Measured candidate retrieval with persistent, fingerprinted model vectors.

Retrieval scores rank source candidates; they are never scientific compatibility
or truth probabilities. The default scientific assessor remains authoritative.
"""

from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import inspect
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .controller import integration_lease
from .data_plane import atomic_json, atomic_write, digest
from .semantic_runtime import PretrainedSemanticIndex
from .text import tokenize

CACHE_VERSION = "semantic-vector-cache-1.0"


def top_k_indices(scores: np.ndarray, identifiers: list[str], k: int) -> list[int]:
    """Exact top-k with stable ID ties, avoiding a Python sort of every score.

    Partitioning selects the score boundary in linear time. Every boundary tie
    is considered before choosing the lexicographically smallest identifiers.
    This is exact retrieval, not an approximate nearest-neighbor index.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if scores.ndim != 1 or len(scores) != len(identifiers) or not np.isfinite(scores).all():
        raise ValueError("Finite scores and equally sized identifiers are required")
    count = min(k, len(scores))
    if not count:
        return []
    boundary = float(np.partition(scores, len(scores) - count)[len(scores) - count])
    above = np.flatnonzero(scores > boundary).tolist()
    tied = heapq.nsmallest(count - len(above), np.flatnonzero(scores == boundary).tolist(),
                          key=lambda i: identifiers[i])
    return sorted(above + tied, key=lambda i: (-float(scores[i]), identifiers[i]))


def _validate_records(records: list[dict]) -> list[dict]:
    if not records:
        raise ValueError("At least one retrieval record is required")
    if any(
        not isinstance(row.get("id"), str)
        or not row["id"]
        or not isinstance(row.get("text"), str)
        or not row["text"].strip()
        for row in records
    ):
        raise ValueError("Retrieval records require nonempty string id/text fields")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate retrieval identity")
    return [dict(row) for row in records]


class BM25Index:
    """Dependency-light inverted index with explicit BM25 parameters."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.records: list[dict] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = np.empty(0)
        self.average_length = 0.0

    def fit(self, records: list[dict]) -> BM25Index:
        self.records = _validate_records(records)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for index, record in enumerate(self.records):
            frequencies = Counter(tokenize(record["text"]))
            lengths.append(sum(frequencies.values()))
            for term, count in frequencies.items():
                postings[term].append((index, count))
        self.postings = dict(postings)
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.average_length = float(np.mean(self.lengths))
        return self

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not self.records:
            raise ValueError("Index is not fitted")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        scores: dict[int, float] = defaultdict(float)
        count = len(self.records)
        for term in set(tokenize(query)):
            hits = self.postings.get(term, [])
            idf = math.log(1 + (count - len(hits) + 0.5) / (len(hits) + 0.5))
            for index, frequency in hits:
                normalizer = self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + normalizer)
        order = heapq.nsmallest(k, scores, key=lambda index: (-scores[index], self.records[index]["id"]))
        return [
            {
                "id": self.records[index]["id"],
                "score": float(scores[index]),
                "score_kind": "bm25_not_probability",
                "algorithm": "bm25-v1",
                "input_sha256": digest(self.records[index]),
            }
            for index in order
        ]


class CachedSemanticIndex:
    """Cache model-derived vectors, preserving the exact record/order contract.

    Cache writes use the project's existing OS lease. Corruption fails closed;
    rebuilding requires an explicit ``force_rebuild=True``. No model download or
    deserialization of pickled objects occurs here.
    """

    def __init__(self, model_dir: str | Path, cache_root: str | Path, *, device: str = "cpu"):
        self.index = PretrainedSemanticIndex(model_dir, device=device)
        self.cache_root = Path(cache_root).resolve()
        self.receipt: dict[str, Any] = {}
        self.fingerprint: str | None = None

    @staticmethod
    def _vectors_valid(vectors: np.ndarray, count: int) -> None:
        if (
            vectors.dtype != np.float32
            or vectors.ndim != 2
            or vectors.shape[0] != count
            or vectors.shape[1] < 1
        ):
            raise ValueError("Unexpected vector dtype or dimensions")
        if not np.isfinite(vectors).all():
            raise ValueError("Non-finite retrieval vectors")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Retrieval vectors are not unit normalized")

    def fit(self, records: list[dict], *, force_rebuild: bool = False) -> CachedSemanticIndex:
        records = _validate_records(records)
        versions = {}
        for name in ("numpy", "torch", "transformers"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not-installed"
        fingerprint = digest(
            {
                "version": CACHE_VERSION,
                "model": self.index.manifest,
                "records": records,
                "versions": versions,
                "encoder": inspect.getsource(PretrainedSemanticIndex),
                "device": str(self.index.device),
                "max_tokens": 256,
            }
        )
        destination = self.cache_root / fingerprint
        started = time.perf_counter()
        with integration_lease(destination, "semantic-vector-cache"):
            manifest_path, vectors_path = destination / "manifest.json", destination / "vectors.npy"
            if manifest_path.exists() and not force_rebuild:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                content = vectors_path.read_bytes()
                if manifest.get("fingerprint") != fingerprint or hashlib.sha256(
                    content
                ).hexdigest() != manifest.get("vectors_sha256"):
                    raise ValueError("Semantic vector cache integrity failure")
                vectors = np.load(io.BytesIO(content), allow_pickle=False)
                self._vectors_valid(vectors, len(records))
                if (
                    manifest.get("version") != CACHE_VERSION
                    or manifest.get("record_count") != len(records)
                    or manifest.get("dimensions") != vectors.shape[1]
                    or manifest.get("records_sha256") != digest(records)
                    or manifest.get("model") != self.index.manifest
                ):
                    raise ValueError("Semantic vector cache metadata integrity failure")
                origin = "verified_cache_replay"
            else:
                vectors = np.asarray(
                    self.index.encode([row["text"] for row in records]), dtype=np.float32
                )
                self._vectors_valid(vectors, len(records))
                buffer = io.BytesIO()
                np.save(buffer, vectors, allow_pickle=False)
                content = buffer.getvalue()
                atomic_write(vectors_path, content)
                manifest = {
                    "version": CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "records_sha256": digest(records),
                    "record_count": len(records),
                    "dimensions": vectors.shape[1],
                    "model": self.index.manifest,
                    "vectors_sha256": hashlib.sha256(content).hexdigest(),
                    "runtime_versions": versions,
                }
                atomic_json(manifest_path, manifest)
                origin = "fresh_model_encoding"
        self.index.records, self.index.vectors = records, vectors
        self.fingerprint = fingerprint
        self.receipt = {
            **manifest,
            "origin": origin,
            "elapsed_seconds": time.perf_counter() - started,
        }
        return self

    def search(self, query: str, k: int = 10) -> list[dict]:
        return self.search_many([query], k=k)[0]

    def search_many(self, queries: list[str], k: int = 10) -> list[list[dict]]:
        """Encode queries in batches while preserving deterministic tie handling."""
        if self.index.vectors is None:
            raise ValueError("Index is not fitted")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        query_vectors = self.index.encode(queries)
        self._vectors_valid(query_vectors, len(queries))
        if query_vectors.shape[1] != self.index.vectors.shape[1]:
            raise ValueError("Query/document vector dimensions differ")
        results = []
        identifiers = [row["id"] for row in self.index.records]
        for query in query_vectors:
            scores = self.index.vectors @ query
            order = top_k_indices(scores, identifiers, k)
            results.append(
                [
                    {
                        "id": self.index.records[index]["id"],
                        "score": float(scores[index]),
                        "score_kind": "cosine_similarity_not_probability",
                        "model_revision": self.index.manifest["revision"],
                        "input_sha256": digest(self.index.records[index]),
                        "text_limit_tokens": 256,
                    }
                    for index in order
                ]
            )
        return results


def reciprocal_rank_fusion(
    rankings: list[list[dict]], *, k: int = 10, rank_constant: int = 60
) -> list[dict]:
    """Fuse independent candidate rankings without treating scores as calibrated."""
    if type(k) is not int or k < 1 or type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("Positive integer k and rank_constant are required")
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[dict]] = defaultdict(list)
    for method_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        for rank, row in enumerate(ranking, 1):
            identity = row["id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("Candidate identity must be a nonempty string")
            if row.get("score") is not None and not math.isfinite(row["score"]):
                raise ValueError("Non-finite component ranking score")
            if identity in seen:
                raise ValueError("Duplicate identity in a component ranking")
            seen.add(identity)
            scores[identity] += 1 / (rank_constant + rank)
            evidence[identity].append(
                {"method": method_index, "rank": rank, "original_score": row.get("score")}
            )
    order = sorted(scores, key=lambda identity: (-scores[identity], identity))[:k]
    return [
        {
            "id": identity,
            "score": scores[identity],
            "score_kind": "rrf_not_probability",
            "rank_constant": rank_constant,
            "components": evidence[identity],
        }
        for identity in order
    ]
