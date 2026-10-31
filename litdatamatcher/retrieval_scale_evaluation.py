"""Fixed-query retrieval stress test; no training or promotion decisions."""

from __future__ import annotations

import gzip
import importlib.metadata
import io
import json
import math
import re
import time
from functools import partial
from pathlib import Path

import numpy as np

from .controller import integration_lease
from .data_plane import atomic_json, atomic_write, digest
from .retrieval import (
    CACHE_VERSION,
    BM25Index,
    CachedSemanticIndex,
    reciprocal_rank_fusion,
    top_k_indices,
)
from .retrieval_evaluation import evaluate_retrieval, paired_comparison
from .retrieval_expansion import file_hash, normalize_text, read_json, read_rows
from .semantic_runtime import verify_model

VERSION = "fixed-query-retrieval-scale-1.0"
METHODS = ("bm25", "pretrained", "trained", "bm25_pretrained_rrf", "bm25_trained_rrf")
CONFIG = {
    "cutoff": 10,
    "ranking_depth": 100,
    "rrf_constant": 60,
    "bootstrap_seed": 1729,
    "max_tokens": 256,
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
}


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its root")
    return path


def _receipt(root, path):
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }


def _verify_outputs(root, outputs):
    if not outputs or len({row["path"] for row in outputs}) != len(outputs):
        raise ValueError("Missing or duplicate artifact receipts")
    for row in outputs:
        path = _inside(root, row["path"])
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"Artifact integrity failure: {path}")


def _stage(root, path, fingerprint):
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("fingerprint") != fingerprint:
        raise ValueError(f"Stage fingerprint or status mismatch: {path}")
    _verify_outputs(root, value["outputs"])
    return value


def validate_pools(root, *, sizes=(50000, 100000)):
    """Verify every frozen input, exact query bytes, both views, and ID prefixes."""
    root = Path(root).resolve()
    expansion = read_json(root / "EXPANSION.json")
    if expansion.get("status") != "PASS" or digest(expansion["inputs"]) != expansion["fingerprint"]:
        raise ValueError("Expansion receipt is not a valid completed build")
    if sorted(expansion["sizes"]) != list(sizes) or set(expansion["views"]) != {
        "full",
        "abstract_only",
    }:
        raise ValueError("Expected the two predeclared pool sizes and views")
    seed_path = root / "SEED_BENCHMARK_MANIFEST.json"
    if file_hash(seed_path) != expansion["inputs"]["benchmark_sha256"]:
        raise ValueError("Original benchmark manifest changed")
    seed_manifest = read_json(seed_path)
    seed_outputs = {row["path"]: row for row in seed_manifest["outputs"]}
    qualification = None
    if expansion["inputs"].get("version") == "unlinked-pubmed-candidate-expansion-1.0":
        qualification = read_json(root / "CLOSURE_QUALIFICATION.json")
        closure_path = root / "EXCLUSIONS_FIXED_POINT.json"
        closure = read_json(closure_path)
        checksum = closure.pop("content_sha256", None)
        if (
            qualification.get("status") != "PASS"
            or digest(qualification.get("inputs")) != qualification.get("fingerprint")
            or qualification.get("leaking_distractors")
            or not closure.get("complete")
            or digest(closure) != checksum
            or qualification.get("fingerprint") != closure.get("fingerprint")
            or qualification.get("inputs", {}).get("snapshot_sha256")
            != expansion["inputs"]["snapshot_sha256"]
            or qualification.get("inputs", {}).get("edges_sha256")
            != expansion["inputs"]["edges_sha256"]
        ):
            raise ValueError("Legacy expansion requires valid fixed-point exclusion qualification")
    elif expansion["inputs"].get("version") != "unlinked-pubmed-candidate-expansion-1.1":
        raise ValueError("Unknown expansion protocol version")
    else:
        closure = read_json(root / "EXCLUSIONS.json")
        checksum = closure.pop("content_sha256", None)
        if (
            not closure.get("complete")
            or digest(closure) != checksum
            or closure.get("fingerprint") != expansion["fingerprint"]
        ):
            raise ValueError("Completed fixed-point exclusion receipt is required")
    manifests, expected = {}, {(size, view) for size in sizes for view in ("full", "abstract_only")}
    query_bytes, full_ids, query_rows, seed_records = None, {}, None, None
    for item in expansion["manifest_receipts"]:
        path = _inside(root, item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("Candidate manifest changed")
        manifest = read_json(path)
        key = (manifest["counts"]["documents"], manifest["candidate_view"])
        if key not in expected or key in manifests or manifest.get("status") != "PASS":
            raise ValueError("Missing, duplicated or unexpected candidate condition")
        if manifest.get("expansion_fingerprint") != expansion["fingerprint"]:
            raise ValueError("Candidate and expansion identities differ")
        if {row["path"] for row in manifest["outputs"]} != {
            "train.jsonl",
            "development.jsonl",
            "documents.jsonl",
        }:
            raise ValueError("Unexpected candidate outputs")
        _verify_outputs(path.parent, manifest["outputs"])
        for split in ("train", "development"):
            if (
                file_hash(path.parent / f"{split}.jsonl")
                != seed_outputs[f"{split}.jsonl"]["sha256"]
            ):
                raise ValueError("Query file differs from original frozen benchmark")
        if manifest["counts"]["development"] != seed_manifest["counts"]["development"]:
            raise ValueError("Frozen query denominator changed")
        current_bytes = tuple(
            (path.parent / f"{split}.jsonl").read_bytes() for split in ("train", "development")
        )
        if query_bytes is None:
            query_bytes = current_bytes
            query_rows = read_rows(path.parent / "development.jsonl")
            if len(query_rows) != seed_manifest["counts"]["development"]:
                raise ValueError("Development query count does not match its receipt")
            if tuple(sizes) == (50000, 100000) and len(query_rows) != 175:
                raise ValueError("Predeclared scale experiment requires the same 175 queries")
            train = read_rows(path.parent / "train.jsonl")
            for rows, split in ((train, "train"), (query_rows, "development")):
                if (
                    not rows
                    or len({q["id"] for q in rows}) != len(rows)
                    or len({q["positive_doc_id"] for q in rows}) != len(rows)
                    or any(
                        q.get("split") != split
                        or not isinstance(q.get("query"), str)
                        or not q["query"].strip()
                        for q in rows
                    )
                ):
                    raise ValueError("Invalid frozen query identity or split")
            if {q["positive_doc_id"] for q in train} & {
                q["positive_doc_id"] for q in query_rows
            } or {normalize_text(q["query"]) for q in train} & {
                normalize_text(q["query"]) for q in query_rows
            }:
                raise ValueError("Training/development overlap")
        elif current_bytes != query_bytes:
            raise ValueError("Frozen query bytes differ between candidate conditions")
        records = read_rows(path.parent / "documents.jsonl")
        ids = [row["id"] for row in records]
        if (
            len(ids) != key[0]
            or len(set(ids)) != len(ids)
            or digest(ids) != manifest["ordered_candidate_ids_sha256"]
        ):
            raise ValueError("Candidate count, uniqueness or ordering failed")
        if any(
            not isinstance(row.get("text"), str)
            or not row["text"].strip()
            or not isinstance(row.get("abstract"), str)
            or not row["abstract"].strip()
            for row in records
        ):
            raise ValueError("Empty candidate text or abstract")
        if key[1] == "abstract_only" and any(row["text"] != row["abstract"] for row in records):
            raise ValueError("Abstract-only view contains other text")
        if key[1] == "full" and any(
            row["text"]
            != (row["title"] + "\n\n" + row["abstract"] if row.get("title") else row["abstract"])
            for row in records
        ):
            raise ValueError("Full candidate text is not the declared title/abstract view")
        if not {q["positive_doc_id"] for q in query_rows} <= set(ids):
            raise ValueError("Missing development target")
        if key[1] == "full":
            full_ids[key[0]] = ids
            if seed_records is None:
                seed_records = records[: manifest["expansion"]["seed_documents"]]
        manifests[key] = {
            "manifest": manifest,
            "path": path,
            "id_digest": digest(ids),
            "record_digest": digest(records),
        }
    if set(manifests) != expected:
        raise ValueError("Incomplete candidate conditions")
    for size in sizes:
        if (
            manifests[(size, "full")]["id_digest"]
            != manifests[(size, "abstract_only")]["id_digest"]
        ):
            raise ValueError("Candidate views have different IDs")
    if qualification:
        expected_hashes = {
            str(Path(row["path"])): row["sha256"]
            for row in qualification["validated_output_hashes"]
        }
        actual_hashes = {
            str((item["path"].parent / row["path"]).relative_to(root)): row["sha256"]
            for item in manifests.values()
            for row in item["manifest"]["outputs"]
        }
        if expected_hashes != actual_hashes:
            raise ValueError("Candidate outputs differ from fixed-point qualification")
    if full_ids[sizes[0]] != full_ids[sizes[-1]][: sizes[0]]:
        raise ValueError("Small pool is not an exact ID prefix")
    # Compare complete records for paired views and size prefixes, not only IDs.
    for view in ("full", "abstract_only"):
        large = read_rows(manifests[(sizes[-1], view)]["path"].parent / "documents.jsonl")
        small = read_rows(manifests[(sizes[0], view)]["path"].parent / "documents.jsonl")
        if large[: sizes[0]] != small:
            raise ValueError("Small-pool content differs from large-pool prefix")
    full = read_rows(manifests[(sizes[-1], "full")]["path"].parent / "documents.jsonl")
    abstract = read_rows(manifests[(sizes[-1], "abstract_only")]["path"].parent / "documents.jsonl")
    if any(
        {k: v for k, v in a.items() if k != "text"} != {k: v for k, v in b.items() if k != "text"}
        for a, b in zip(full, abstract, strict=True)
    ):
        raise ValueError("Candidate views differ beyond the declared text field")
    original = {row["id"]: row for row in seed_records}
    title_overlap = sum(
        re.sub(r"\W+", "", q["query"]).casefold()
        == re.sub(r"\W+", "", original[q["positive_doc_id"]].get("title", "")).casefold()
        for q in query_rows
    )
    return {
        "root": root,
        "expansion": expansion,
        "manifests": manifests,
        "queries": query_rows,
        "receipt": {
            "expansion_sha256": file_hash(root / "EXPANSION.json"),
            "seed_manifest_sha256": file_hash(seed_path),
            "exclusion_sha256": file_hash(
                root / ("EXCLUSIONS_FIXED_POINT.json" if qualification else "EXCLUSIONS.json")
            ),
            "qualification_sha256": file_hash(root / "CLOSURE_QUALIFICATION.json")
            if qualification
            else None,
            "query_sha256": digest(query_rows),
            "query_file_sha256": file_hash(
                next(iter(manifests.values()))["path"].parent / "development.jsonl"
            ),
            "query_count": len(query_rows),
            "query_equals_target_title_ignoring_punctuation": title_overlap,
            "manifests": [
                {"condition": f"{size}/{view}", "sha256": file_hash(item["path"])}
                for (size, view), item in sorted(manifests.items())
            ],
        },
    }


def _implementation():
    root = Path(__file__).parent
    return {
        name: file_hash(root / name)
        for name in (
            "retrieval_scale_evaluation.py",
            "retrieval.py",
            "retrieval_evaluation.py",
            "retrieval_expansion.py",
            "semantic_runtime.py",
            "text.py",
            "strict_artifacts.py",
        )
    }


def _recheck_inputs(validated, models, implementation):
    """Long-running inference must not publish PASS after source/code drift."""
    root, receipt = validated["root"], validated["receipt"]
    checks = {
        "EXPANSION.json": receipt["expansion_sha256"],
        "SEED_BENCHMARK_MANIFEST.json": receipt["seed_manifest_sha256"],
        "EXCLUSIONS_FIXED_POINT.json"
        if receipt["qualification_sha256"]
        else "EXCLUSIONS.json": receipt["exclusion_sha256"],
    }
    if receipt["qualification_sha256"]:
        checks["CLOSURE_QUALIFICATION.json"] = receipt["qualification_sha256"]
    if any(file_hash(root / name) != checksum for name, checksum in checks.items()):
        raise ValueError("Frozen input receipts changed during evaluation")
    expected = {row["condition"]: row["sha256"] for row in receipt["manifests"]}
    for (size, view), item in validated["manifests"].items():
        if file_hash(item["path"]) != expected[f"{size}/{view}"]:
            raise ValueError("Frozen candidate manifest changed during evaluation")
        _verify_outputs(item["path"].parent, item["manifest"]["outputs"])
    if _implementation() != implementation:
        raise ValueError("Evaluation implementation changed during execution")
    for model in models.values():
        if _model_identity(Path(model["path"])) != model:
            raise ValueError("Frozen model changed during evaluation")


def _model_identity(path):
    value = verify_model(path)
    return {
        "path": str(path),
        "manifest_sha256": file_hash(path / "MODEL_MANIFEST.json"),
        "model_id": value["model_id"],
        "revision": value["revision"],
        "revision_type": value.get("revision_type", "upstream_git_commit"),
        "license": value["license"],
    }


def _vectors(out, fingerprint, view, name, model_path, records, queries, device):
    folder = out / "encodings" / view / name
    path = folder / "ENCODING.json"
    stage_fp = digest({"run": fingerprint, "view": view, "model": name})
    documents = [{"id": row["id"], "text": row["text"]} for row in records]
    query_records = [{"id": row["id"], "text": row["query"]} for row in queries]
    saved = _stage(out, path, stage_fp)
    if saved:
        started = time.perf_counter()
        model = read_json(model_path / "MODEL_MANIFEST.json")
        arrays = []
        covered = {item["path"] for item in saved["outputs"]}
        for key, values in (("document_vectors", documents), ("query_vectors", query_records)):
            vector_path = _inside(out, saved[key])
            manifest_path = vector_path.parent / "manifest.json"
            if saved[key] not in covered or str(manifest_path.relative_to(out)) not in covered:
                raise ValueError("Vector references are not covered by checkpoint hashes")
            manifest = read_json(manifest_path)
            vectors = np.load(vector_path, allow_pickle=False)
            CachedSemanticIndex._vectors_valid(vectors, len(values))
            if (
                manifest.get("records_sha256") != digest(values)
                or manifest.get("model") != model
                or manifest.get("fingerprint") != vector_path.parent.name
                or manifest.get("version") != CACHE_VERSION
                or manifest.get("vectors_sha256") != file_hash(vector_path)
                or manifest.get("dimensions") != vectors.shape[1]
            ):
                raise ValueError("Cached vector metadata differs from the input/model contract")
            vectors.flags.writeable = False
            arrays.append(vectors)
        doc, query = arrays
        return doc, query, saved, time.perf_counter() - started
    started = time.perf_counter()
    index = CachedSemanticIndex(model_path, out / "vectors", device=device)
    initialization = time.perf_counter() - started
    index.fit(documents)
    doc, first_doc = index.index.vectors, dict(index.receipt)
    doc_fp = index.fingerprint
    index.fit(documents)
    if index.receipt["origin"] != "verified_cache_replay":
        raise ValueError("Warm document cache check unexpectedly re-encoded")
    warm_doc = index.receipt["elapsed_seconds"]
    index.fit(query_records)
    query, first_query = index.index.vectors, dict(index.receipt)
    query_fp = index.fingerprint
    index.fit(query_records)
    if index.receipt["origin"] != "verified_cache_replay":
        raise ValueError("Warm query cache check unexpectedly re-encoded")
    warm_query = index.receipt["elapsed_seconds"]
    doc_path, query_path = (out / "vectors" / fp / "vectors.npy" for fp in (doc_fp, query_fp))
    artifacts = [
        p
        for fp in (doc_fp, query_fp)
        for p in (out / "vectors" / fp / "vectors.npy", out / "vectors" / fp / "manifest.json")
    ]
    saved = {
        "status": "PASS",
        "fingerprint": stage_fp,
        "document_vectors": str(doc_path.relative_to(out)),
        "query_vectors": str(query_path.relative_to(out)),
        "outputs": [_receipt(out, p) for p in artifacts],
        "timings": {
            "model_initialization_seconds": initialization,
            "document_first_origin": first_doc["origin"],
            "document_first_fit_seconds": first_doc["elapsed_seconds"],
            "document_warm_verified_fit_seconds": warm_doc,
            "query_first_origin": first_query["origin"],
            "query_first_fit_seconds": first_query["elapsed_seconds"],
            "query_warm_verified_fit_seconds": warm_query,
        },
        "documents_encoded_for_both_sizes": len(records),
        "queries_encoded": len(queries),
    }
    atomic_json(path, saved)
    doc.flags.writeable = False
    query.flags.writeable = False
    return doc, query, saved, None


def _rank_dense(vectors, queries, records):
    CachedSemanticIndex._vectors_valid(vectors, len(records))
    if queries.ndim != 2 or queries.shape[1] != vectors.shape[1]:
        raise ValueError("Query/document vector dimensions differ")
    CachedSemanticIndex._vectors_valid(queries, len(queries))
    ids = [row["id"] for row in records]
    results = []
    for query in queries:
        scores = vectors @ query
        results.append(
            [
                {"id": ids[i], "score": float(scores[i])}
                for i in top_k_indices(scores, ids, CONFIG["ranking_depth"])
            ]
        )
    return results


def _rankings_file(path, queries, rankings):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as compressed:
        for query, hits in zip(queries, rankings, strict=True):
            compressed.write(
                (
                    json.dumps(
                        {"query_id": query["id"], "hits": hits},
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )
    atomic_write(path, buffer.getvalue())


def _read_rankings(path, queries):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if [row["query_id"] for row in rows] != [q["id"] for q in queries]:
        raise ValueError("Ranking query order or denominator changed")
    return [row["hits"] for row in rows]


def _rank_bm25(records, queries):
    started = time.perf_counter()
    index = BM25Index(k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"]).fit(records)
    fitted = time.perf_counter()
    rankings = [index.search(q["query"], k=CONFIG["ranking_depth"]) for q in queries]
    return rankings, {
        "index_build_seconds": fitted - started,
        "query_ranking_seconds": time.perf_counter() - fitted,
    }


def _rank_fused(first, second):
    return [
        reciprocal_rank_fusion(
            [a, b], k=CONFIG["ranking_depth"], rank_constant=CONFIG["rrf_constant"]
        )
        for a, b in zip(first, second, strict=True)
    ]


def _method(out, folder, run_fp, method, queries, records, calculate):
    fingerprint = digest(
        {"run": run_fp, "condition": str(folder.relative_to(out)), "method": method}
    )
    destination = folder / method
    stage_path = destination / "STAGE.json"
    saved = _stage(out, stage_path, fingerprint)
    if saved:
        return read_json(destination / "METRICS.json"), _read_rankings(
            destination / "rankings.jsonl.gz", queries
        )
    started = time.perf_counter()
    rankings = calculate()
    elapsed = time.perf_counter() - started
    timings = {"query_ranking_seconds": elapsed}
    if isinstance(rankings, tuple):
        rankings, timings = rankings
    ids = {row["id"] for row in records}
    if len(rankings) != len(queries) or any(
        len(hits) > CONFIG["ranking_depth"]
        or len({h["id"] for h in hits}) != len(hits)
        or any(h["id"] not in ids or not math.isfinite(h["score"]) for h in hits)
        for hits in rankings
    ):
        raise ValueError("Invalid calculated rankings")
    metrics = evaluate_retrieval(
        queries, rankings, k=CONFIG["cutoff"], seed=CONFIG["bootstrap_seed"]
    )
    metrics["by_source"] = {
        source: {
            k: v
            for k, v in evaluate_retrieval(
                [queries[i] for i in positions],
                [rankings[i] for i in positions],
                k=CONFIG["cutoff"],
            ).items()
            if k not in {"by_source", "query_results"}
        }
        for source, positions in _source_indices(queries).items()
    }
    metrics["timings"] = {
        **timings,
        "mean_query_ranking_seconds": timings["query_ranking_seconds"] / len(queries),
    }
    _rankings_file(destination / "rankings.jsonl.gz", queries, rankings)
    atomic_json(destination / "METRICS.json", metrics)
    atomic_json(
        stage_path,
        {
            "status": "PASS",
            "fingerprint": fingerprint,
            "outputs": [
                _receipt(out, destination / name) for name in ("rankings.jsonl.gz", "METRICS.json")
            ],
        },
    )
    return metrics, rankings


def _source_indices(queries):
    sources = sorted({row["source_corpus"] for row in queries})
    return {
        source: [i for i, row in enumerate(queries) if row["source_corpus"] == source]
        for source in sources
    }


def _compare(left, right, queries):
    result = paired_comparison(left, right)
    result["recall_at_1_delta"] = right["known_link_recall_at_1"] - left["known_link_recall_at_1"]
    result["recall_at_10_delta"] = right["known_link_recall_at_k"] - left["known_link_recall_at_k"]
    result["ndcg_at_10_delta"] = right["ndcg_at_k"] - left["ndcg_at_k"]
    result["by_source"] = {}
    for source, positions in _source_indices(queries).items():
        a, b = (
            {**metrics, "query_results": [metrics["query_results"][i] for i in positions]}
            for metrics in (left, right)
        )
        result["by_source"][source] = paired_comparison(a, b)
    return result


def _run(
    root,
    pretrained,
    trained,
    out,
    *,
    device="cpu",
    resume=False,
    sizes=(50000, 100000),
    progress=None,
    _on_start=None,
):
    root, pretrained, trained, out = (Path(p).resolve() for p in (root, pretrained, trained, out))
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (root, pretrained, trained)):
        raise ValueError("Output must be separate from immutable inputs and models")
    notify = progress or (lambda event: None)
    validated = validate_pools(root, sizes=sizes)
    models = {"pretrained": _model_identity(pretrained), "trained": _model_identity(trained)}
    if models["pretrained"]["manifest_sha256"] == models["trained"]["manifest_sha256"]:
        raise ValueError("The pretrained and trained models must be distinct")
    versions = {}
    for name in ("numpy", "torch", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    identity = {
        "version": VERSION,
        "inputs": validated["receipt"],
        "models": models,
        "config": CONFIG,
        "device": device,
        "implementation": _implementation(),
        "runtime_versions": versions,
    }
    fingerprint = digest(identity)
    state_path = out / "RUN.json"
    if state_path.exists():
        saved = read_json(state_path)
        if not resume or saved.get("fingerprint") != fingerprint:
            raise ValueError(
                "Existing run requires --resume with identical inputs/configuration/code"
            )
    state = {
        "status": "RUNNING",
        "fingerprint": fingerprint,
        "identity": identity,
        "interpretation": "Fixed development diagnostic; no training, promotion or sealed-holdout evaluation.",
    }
    atomic_json(state_path, state)
    if _on_start:
        _on_start(fingerprint)
    queries, results, encoding_receipts, all_metrics = validated["queries"], {}, {}, {}
    for view in ("full", "abstract_only"):
        records = read_rows(
            validated["manifests"][(sizes[-1], view)]["path"].parent / "documents.jsonl"
        )
        dense = {}
        for name, path in (("pretrained", pretrained), ("trained", trained)):
            notify({"stage": "encoding_or_verified_replay", "view": view, "model": name})
            vectors, query_vectors, receipt, replay_seconds = _vectors(
                out, fingerprint, view, name, path, records, queries, device
            )
            dense[name] = (vectors, query_vectors)
            encoding_receipts[f"{view}/{name}"] = {
                "path": str((out / "encodings" / view / name / "ENCODING.json").relative_to(out)),
                "sha256": file_hash(out / "encodings" / view / name / "ENCODING.json"),
                "timings": receipt["timings"],
                "current_resume_load_seconds": replay_seconds,
            }
        for size in sizes:
            selected = records[:size]
            folder = out / "conditions" / f"pool-{size}" / view
            rankings, metrics = {}, {}
            for method in METHODS:
                notify(
                    {
                        "stage": "ranking_or_verified_replay",
                        "view": view,
                        "size": size,
                        "method": method,
                    }
                )
                if method == "bm25":
                    calculate = partial(_rank_bm25, selected, queries)
                elif method in dense:
                    calculate = partial(
                        _rank_dense, dense[method][0][:size], dense[method][1], selected
                    )
                else:
                    component = "trained" if method == "bm25_trained_rrf" else "pretrained"
                    calculate = partial(_rank_fused, rankings["bm25"], rankings[component])
                metrics[method], rankings[method] = _method(
                    out, folder, fingerprint, method, queries, selected, calculate
                )
            condition = {
                "candidate_count": size,
                "view": view,
                "candidate_manifest_sha256": file_hash(
                    validated["manifests"][(size, view)]["path"]
                ),
                "ordered_candidate_ids_sha256": validated["manifests"][(size, view)]["id_digest"],
                "query_sha256": validated["receipt"]["query_sha256"],
                "metrics": {
                    name: {k: v for k, v in value.items() if k != "query_results"}
                    for name, value in metrics.items()
                },
                "paired": {
                    f"{name}_vs_{baseline}": _compare(metrics[baseline], metrics[name], queries)
                    for baseline in ("bm25", "pretrained")
                    for name in METHODS
                    if name != baseline
                },
            }
            atomic_json(folder / "COMPARISON.json", condition)
            results[f"{size}/{view}"] = condition
            all_metrics[(size, view)] = metrics
    contrasts = {
        "abstract_only_minus_full": {
            str(size): {
                name: _compare(
                    all_metrics[(size, "full")][name],
                    all_metrics[(size, "abstract_only")][name],
                    queries,
                )
                for name in METHODS
            }
            for size in sizes
        },
        "larger_minus_smaller_pool": {
            view: {
                name: _compare(
                    all_metrics[(sizes[0], view)][name],
                    all_metrics[(sizes[-1], view)][name],
                    queries,
                )
                for name in METHODS
            }
            for view in ("full", "abstract_only")
        },
        "interpretation": "Predeclared diagnostics on the same development queries; no promotion or best-view selection.",
    }
    atomic_json(out / "DIAGNOSTIC_CONTRASTS.json", contrasts)
    _recheck_inputs(validated, models, identity["implementation"])
    state.update(
        status="PASS",
        conditions=results,
        encodings=encoding_receipts,
        outputs=[
            _receipt(out, out / "conditions" / f"pool-{size}" / view / "COMPARISON.json")
            for size in sizes
            for view in ("full", "abstract_only")
        ]
        + [_receipt(out, out / "DIAGNOSTIC_CONTRASTS.json")],
    )
    atomic_json(state_path, state)
    return state


def evaluate_scale(root, pretrained, trained, out, **kwargs):
    """Run or resume, preserving finished model-vector and ranking checkpoints."""
    paths = [Path(p).resolve() for p in (root, pretrained, trained)]
    destination = Path(out).resolve()
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in paths):
        raise ValueError("Output must be separate from immutable inputs and models")
    with integration_lease(destination / ".owner", "retrieval-scale-evaluation"):
        started = []
        try:
            return _run(root, pretrained, trained, out, _on_start=started.append, **kwargs)
        except BaseException as exc:
            path = Path(out).resolve() / "RUN.json"
            if started and path.exists():
                value = read_json(path)
                if value.get("status") == "RUNNING" and value.get("fingerprint") == started[0]:
                    value.update(
                        status="INTERRUPTED"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "FAIL",
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                    atomic_json(path, value)
            raise
