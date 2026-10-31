"""Bounded local source-link retriever training and development evaluation.

This experiment learns recorded paper links. In-batch alternatives are unjudged,
and none of its scores or development metrics establish scientific suitability.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .controller import integration_lease
from .data_plane import atomic_json, digest
from .retrieval import BM25Index, CachedSemanticIndex, reciprocal_rank_fusion
from .retrieval_evaluation import evaluate_retrieval, paired_comparison
from .semantic_runtime import runtime_dependency_versions, verify_model
from .strict_artifacts import strict_json_loads


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    temperature: float = 0.05
    max_tokens: int = 256
    seed: int = 1729
    gradient_clip: float = 1.0


FIXED_CONFIG = TrainingConfig()


def file_sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _read(path):
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    rows = [
        _read_line(line)
        for line in Path(path).read_text(encoding="utf-8-sig").split("\n")
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Benchmark JSONL requires object records")
    return rows


def _read_line(line):
    return strict_json_loads(line)


def _normalized(text):
    return " ".join(text.casefold().split())


def _aliases(document):
    values = {document["id"].casefold()}
    if document.get("pmid"):
        values.add("pmid:" + str(document["pmid"]).casefold().removeprefix("pmid:"))
    if document.get("pmcid"):
        values.add("pmcid:" + str(document["pmcid"]).casefold().removeprefix("pmcid:"))
    if document.get("doi"):
        values.add("doi:" + str(document["doi"]).casefold().removeprefix("https://doi.org/"))
    return values


def validate_benchmark(root: Path) -> dict:
    """Verify the preparer's exact output receipts and independently reject leakage."""
    root = Path(root).resolve()
    manifest_path = root / "BENCHMARK_MANIFEST.json"
    manifest = _read(manifest_path)
    if manifest.get("status") != "PASS":
        raise ValueError("Benchmark manifest must report PASS")
    receipts = manifest.get("outputs")
    if not isinstance(receipts, list):
        raise ValueError("Benchmark outputs must be a list of file receipts")
    expected = {"train.jsonl", "development.jsonl", "documents.jsonl"}
    if len(receipts) != 3 or {item.get("path") for item in receipts} != expected:
        raise ValueError("Benchmark receipts must name exactly train/development/documents JSONL")
    for item in receipts:
        path = root / item["path"]
        if (
            path.resolve().parent != root
            or file_sha(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
        ):
            raise ValueError(f"Benchmark file integrity failure: {item['path']}")
    train, development, documents = (
        _rows(root / name) for name in ("train.jsonl", "development.jsonl", "documents.jsonl")
    )
    if len(train) < 2 or not development or not documents:
        raise ValueError(
            "Need at least two training links, development links and candidate documents"
        )
    if manifest.get("counts") != {
        "train": len(train),
        "development": len(development),
        "documents": len(documents),
    }:
        raise ValueError("Benchmark counts disagree with its records")
    by_id = {}
    for document in documents:
        if (
            not isinstance(document.get("id"), str)
            or not document["id"]
            or not isinstance(document.get("text"), str)
            or not document["text"].strip()
        ):
            raise ValueError("Documents need nonempty string id/text")
        if document["id"] in by_id:
            raise ValueError("Duplicate candidate document identity")
        by_id[document["id"]] = document
    global_ids = set()
    split_keys = []
    for split, queries in (("train", train), ("development", development)):
        ids, queries_seen, paper_aliases, texts_seen = set(), set(), set(), set()
        for row in queries:
            if any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in (
                    "id",
                    "query",
                    "positive_doc_id",
                    "source_corpus",
                    "split",
                    "provenance_relative_path",
                )
            ):
                raise ValueError(
                    "Query records require string identity, query, source, split and positive link"
                )
            if (
                row["split"] != split
                or row["id"] in global_ids
                or row["positive_doc_id"] not in by_id
            ):
                raise ValueError("Wrong split, duplicate query ID or missing positive document")
            normalized = _normalized(row["query"])
            aliases = _aliases(by_id[row["positive_doc_id"]])
            if normalized in queries_seen or aliases & paper_aliases:
                raise ValueError("Duplicate query or more than one query per source paper")
            global_ids.add(row["id"])
            ids.add(row["positive_doc_id"])
            queries_seen.add(normalized)
            paper_aliases.update(aliases)
            texts_seen.add(_normalized(by_id[row["positive_doc_id"]]["text"]))
        split_keys.append((ids, queries_seen, paper_aliases, texts_seen))
    if any(left & right for left, right in zip(split_keys[0], split_keys[1], strict=True)):
        raise ValueError("Train/development paper, alias, query or source-text leakage")
    if any(
        manifest.get("leakage_checks", {}).get(key) != 0
        for key in (
            "canonical_paper_overlap",
            "canonical_query_overlap",
            "canonical_document_text_overlap",
        )
    ):
        raise ValueError("Preparer leakage checks are missing or nonzero")
    return {
        "train": train,
        "development": development,
        "documents": documents,
        "manifest": manifest,
        "manifest_sha256": file_sha(manifest_path),
    }


def training_fingerprint(benchmark, parent_manifest, config, device):
    implementations = [
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in (
                "retrieval.py",
                "retrieval_evaluation.py",
                "semantic_runtime.py",
                "strict_artifacts.py",
            )
        ],
    ]
    versions = runtime_dependency_versions()
    protocol = Path(__file__).with_name("protocols") / "retrieval-training-v1.md"
    if not protocol.is_file():
        raise FileNotFoundError(
            f"Required packaged retrieval-training protocol missing: {protocol}"
        )
    return {
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "parent_model": parent_manifest,
        "config": asdict(config),
        "device": device,
        "runtime_versions": versions,
        "precision": "float32",
        "protocol_sha256": file_sha(protocol),
        "implementation": {path.name: file_sha(path) for path in implementations},
    }


def seed_runtime(seed, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def pooled_embeddings(model, inputs):
    import torch.nn.functional as functional

    tokens = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1)
    return functional.normalize(pooled, p=2, dim=1)


def train_epoch(model, tokenizer, optimizer, pairs, documents, config, epoch, device):
    import torch
    import torch.nn.functional as functional

    seed_runtime(config.seed + epoch, device)
    model.train()
    order = list(range(len(pairs)))
    random.Random(config.seed + epoch).shuffle(order)
    losses, skipped = [], 0
    started = time.perf_counter()
    for offset in range(0, len(order), config.batch_size):
        batch = [pairs[index] for index in order[offset : offset + config.batch_size]]
        if len(batch) < 2:
            skipped += len(batch)
            continue

        def tokenize(texts):
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_tokens,
                return_tensors="pt",
            ).to(device)

        optimizer.zero_grad(set_to_none=True)
        queries = pooled_embeddings(model, tokenize([row["query"] for row in batch]))
        positives = pooled_embeddings(
            model, tokenize([documents[row["positive_doc_id"]]["text"] for row in batch])
        )
        scores = queries @ positives.T / config.temperature
        labels = torch.arange(len(batch), device=device)
        loss = (
            functional.cross_entropy(scores, labels) + functional.cross_entropy(scores.T, labels)
        ) / 2
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite contrastive training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("No usable contrastive batches")
    return {
        "epoch": epoch,
        "batches": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "batch_losses": losses,
        "skipped_singleton_examples": skipped,
        "order_sha256": digest(order),
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_epoch(model, tokenizer, optimizer, out, epoch, run_fingerprint, parent, stats, provenance):
    import torch

    destination = out / f"epoch-{epoch:02d}"
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing trained epoch")
    staging = out / (".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    files = [
        {
            "path": path.relative_to(staging).as_posix(),
            "sha256": file_sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    ]
    content_fingerprint = digest(
        {"files": files, "training_fingerprint": run_fingerprint, "epoch": epoch}
    )
    model_manifest = {
        "model_id": f"local/litdatamatcher-source-link-epoch-{epoch}",
        "revision": content_fingerprint[:40],
        "revision_type": "local_training_content_fingerprint",
        "training_content_sha256": content_fingerprint,
        "license": parent["license"],
        "parent_model": {
            "model_id": parent["model_id"],
            "revision": parent["revision"],
            "manifest_sha256": digest(parent),
        },
        "training_fingerprint": run_fingerprint,
        "training_provenance": provenance,
        "files": files,
    }
    atomic_json(staging / "MODEL_MANIFEST.json", model_manifest)
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    torch.load(staging / "optimizer.pt", map_location="cpu", weights_only=True)
    receipt = {
        "epoch": epoch,
        "training_fingerprint": run_fingerprint,
        "statistics": stats,
        "model_manifest_sha256": file_sha(staging / "MODEL_MANIFEST.json"),
        "optimizer_sha256": file_sha(staging / "optimizer.pt"),
    }
    atomic_json(staging / "EPOCH_RECEIPT.json", receipt)
    staging.replace(destination)
    return receipt


def validate_epoch(path, epoch, fingerprint):
    receipt = _read(path / "EPOCH_RECEIPT.json")
    if receipt.get("epoch") != epoch or receipt.get("training_fingerprint") != fingerprint:
        raise ValueError("Epoch fingerprint mismatch")
    manifest = verify_model(path)
    if (
        manifest.get("training_fingerprint") != fingerprint
        or receipt["model_manifest_sha256"] != file_sha(path / "MODEL_MANIFEST.json")
        or receipt["optimizer_sha256"] != file_sha(path / "optimizer.pt")
    ):
        raise ValueError("Epoch model/optimizer integrity failure")
    return receipt


def _metric_report(queries, rankings):
    result = evaluate_retrieval(queries, rankings, k=10)
    result["source_strata"] = {}
    for source in sorted({row["source_corpus"] for row in queries}):
        positions = [i for i, row in enumerate(queries) if row["source_corpus"] == source]
        metrics = evaluate_retrieval(
            [queries[i] for i in positions], [rankings[i] for i in positions], k=10
        )
        metrics.pop("query_results")
        result["source_strata"][source] = metrics
    return result


def semantic_evaluation(model_dir, documents, queries, cache_root, device):
    started = time.perf_counter()
    index = CachedSemanticIndex(model_dir, cache_root, device=device)
    load_seconds = time.perf_counter() - started
    index.fit(documents)
    initial = dict(index.receipt)
    started = time.perf_counter()
    rankings = index.search_many([row["query"] for row in queries], k=min(100, len(documents)))
    query_seconds = time.perf_counter() - started
    index.fit(documents)
    warm = dict(index.receipt)
    started = time.perf_counter()
    replay = index.search_many([row["query"] for row in queries], k=min(100, len(documents)))
    warm_query_seconds = time.perf_counter() - started
    if [[row["id"] for row in ranking] for ranking in rankings] != [
        [row["id"] for row in ranking] for ranking in replay
    ]:
        raise ValueError("Warm retrieval differs from initial ranking")
    return {
        "metrics": _metric_report(queries, rankings),
        "rankings": rankings,
        "timing": {
            "model_load_seconds": load_seconds,
            "initial_fit": initial,
            "warm_fit": warm,
            "query_seconds": query_seconds,
            "warm_query_seconds": warm_query_seconds,
        },
    }


def _save_evaluation(path, value, fingerprint):
    atomic_json(path, value)
    atomic_json(
        path.with_suffix(".receipt.json"),
        {"sha256": file_sha(path), "training_fingerprint": fingerprint},
    )


def _cached_evaluation(path, fingerprint, function):
    if path.exists():
        receipt = _read(path.with_suffix(".receipt.json"))
        if receipt.get("training_fingerprint") != fingerprint or receipt["sha256"] != file_sha(
            path
        ):
            raise ValueError("Evaluation integrity failure; existing results were preserved")
        return _read(path)
    value = function()
    _save_evaluation(path, value, fingerprint)
    return value


def promotion_decision(baseline, candidates):
    rows = []
    for name, candidate in candidates.items():
        paired = paired_comparison(baseline["metrics"], candidate["metrics"])
        recall_delta = (
            candidate["metrics"]["known_link_recall_at_k"]
            - baseline["metrics"]["known_link_recall_at_k"]
        )
        eligible = paired["mean_reciprocal_rank_delta"] >= 0.01 and recall_delta >= -0.005
        rows.append(
            {
                "candidate": name,
                "eligible": eligible,
                "paired": paired,
                "recall_at_10_delta": recall_delta,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["paired"]["mean_reciprocal_rank_delta"],
                row["recall_at_10_delta"],
                -int(row["candidate"].split("-")[-1]),
            ),
        )
        if eligible
        else None
    )
    return {
        "status": "EXPERIMENTAL_DEVELOPMENT_SELECTION" if selected else "INCUMBENT_RETAINED",
        "selected": selected["candidate"] if selected else "pretrained",
        "candidates": rows,
        "thresholds": {"minimum_mrr_at_10_improvement": 0.01, "minimum_recall_at_10_delta": -0.005},
        "limitations": "Development selection only; source-link recovery is not scientific validity or independent test validation.",
    }


def run_training(
    benchmark_root,
    model_dir,
    out,
    *,
    device="cuda",
    resume=False,
    config=FIXED_CONFIG,
    progress=None,
):
    """Execute only the predeclared experiment; all artifacts use a new run root."""
    if config != FIXED_CONFIG:
        raise ValueError("This experiment uses the predeclared fixed training configuration")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda explicitly")
    benchmark_root, model_dir, out = (
        Path(path).resolve() for path in (benchmark_root, model_dir, out)
    )
    for protected in (benchmark_root, model_dir):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError(
                "Training output must be separate from benchmark and parent model directories"
            )
    benchmark = validate_benchmark(benchmark_root)
    parent = verify_model(model_dir)
    fingerprint_inputs = training_fingerprint(benchmark, parent, config, device)
    fingerprint = digest(fingerprint_inputs)
    if out.exists() and not resume:
        raise ValueError("Training run already exists; use --resume with identical inputs")
    if resume and not (out / "TRAINING_STATE.json").is_file():
        raise ValueError("No resumable training state exists")
    with integration_lease(out / ".owner", "retriever-training"):
        state_path = out / "TRAINING_STATE.json"
        if not resume and state_path.exists():
            raise ValueError("Training run was created by another writer; use --resume")
        if resume:
            state = _read(state_path)
            if state.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Training fingerprint mismatch: data, model, configuration or implementation changed"
                )
        else:
            state = {
                "fingerprint": fingerprint,
                "fingerprint_inputs": fingerprint_inputs,
                "status": "RUNNING",
                "completed_epochs": [],
                "started_at": time.time(),
            }
            atomic_json(state_path, state)
        notify = progress or (lambda event: None)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            seed_runtime(config.seed, device)
            documents, development, training = (
                benchmark["documents"],
                benchmark["development"],
                benchmark["train"],
            )
            by_id = {row["id"]: row for row in documents}

            def baseline_results():
                start = time.perf_counter()
                bm25 = BM25Index().fit(documents)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                lexical_rankings = [
                    bm25.search(row["query"], k=min(100, len(documents))) for row in development
                ]
                lexical = {
                    "metrics": _metric_report(development, lexical_rankings),
                    "rankings": lexical_rankings,
                    "timing": {
                        "fit_seconds": fit_seconds,
                        "query_seconds": time.perf_counter() - start,
                    },
                }
                dense = semantic_evaluation(
                    model_dir, documents, development, out / "vectors/pretrained", device
                )
                fused = [
                    reciprocal_rank_fusion([left, right], k=10, rank_constant=60)
                    for left, right in zip(lexical_rankings, dense["rankings"], strict=True)
                ]
                return {
                    "bm25": lexical,
                    "pretrained": dense,
                    "hybrid": {
                        "metrics": _metric_report(development, fused),
                        "rankings": fused,
                        "component_depth": min(100, len(documents)),
                        "rank_constant": 60,
                    },
                }

            notify({"stage": "baseline_evaluation", "fingerprint": fingerprint})
            baseline = _cached_evaluation(
                out / "BASELINE_EVALUATION.json", fingerprint, baseline_results
            )
            candidates = {}
            previous = model_dir
            model = tokenizer = optimizer = None
            for epoch in range(1, config.epochs + 1):
                destination = out / f"epoch-{epoch:02d}"
                if destination.exists():
                    receipt = validate_epoch(destination, epoch, fingerprint)
                    notify({"stage": "epoch_reused", "epoch": epoch})
                else:
                    if epoch in state["completed_epochs"]:
                        raise ValueError(
                            "A previously completed epoch is missing; refusing silent retraining"
                        )
                    if model is None:
                        tokenizer = AutoTokenizer.from_pretrained(
                            previous, local_files_only=True, trust_remote_code=False
                        )
                        model = (
                            AutoModel.from_pretrained(
                                previous,
                                local_files_only=True,
                                trust_remote_code=False,
                                use_safetensors=True,
                            )
                            .float()
                            .to(device)
                        )
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                        )
                        if epoch > 1:
                            optimizer.load_state_dict(
                                torch.load(
                                    previous / "optimizer.pt",
                                    map_location=device,
                                    weights_only=True,
                                )
                            )
                    notify(
                        {
                            "stage": "epoch_training",
                            "epoch": epoch,
                            "training_examples": len(training),
                        }
                    )
                    stats = train_epoch(
                        model, tokenizer, optimizer, training, by_id, config, epoch, device
                    )
                    if (
                        validate_benchmark(benchmark_root)["manifest_sha256"]
                        != benchmark["manifest_sha256"]
                    ):
                        raise ValueError("Benchmark changed during training")
                    receipt = save_epoch(
                        model,
                        tokenizer,
                        optimizer,
                        out,
                        epoch,
                        fingerprint,
                        verify_model(previous),
                        stats,
                        fingerprint_inputs,
                    )
                    notify(
                        {
                            "stage": "epoch_saved",
                            "epoch": epoch,
                            "statistics": receipt["statistics"],
                        }
                    )
                state["completed_epochs"] = sorted(set(state["completed_epochs"] + [epoch]))
                state["status"] = "RUNNING"
                atomic_json(state_path, state)
                candidates[f"epoch-{epoch:02d}"] = _cached_evaluation(
                    out / f"EVALUATION_EPOCH_{epoch:02d}.json",
                    fingerprint,
                    lambda destination=destination, epoch=epoch: semantic_evaluation(
                        destination,
                        documents,
                        development,
                        out / f"vectors/epoch-{epoch:02d}",
                        device,
                    ),
                )
                previous = destination
            result = {
                "status": "PASS",
                "experiment": "source-link-retrieval-development-only",
                "fingerprint": fingerprint,
                "configuration": asdict(config),
                "precision": "float32",
                "benchmark_purpose": benchmark["manifest"].get("purpose"),
                "failure_receipts": sorted(path.name for path in out.glob("FAILURE_*.json")),
                "benchmark_manifest_sha256": benchmark["manifest_sha256"],
                "counts": benchmark["manifest"]["counts"],
                "exclusions": benchmark["manifest"].get("exclusions_and_accounting", {}),
                "baseline": baseline,
                "candidates": candidates,
                "selection": promotion_decision(baseline["pretrained"], candidates),
                "truncation_policy": "256 tokenizer tokens for both queries and source-paper title/abstract; dynamic padding within batches",
                "negative_policy": "Other source-paper links in each batch are unjudged negatives, not verified irrelevant articles.",
                "validation_scope": "Paper-disjoint development links from the same source corpora; not a source-corpus holdout, sealed holdout, dataset feasibility, or expert-validated scientific accuracy.",
            }
            if (
                training_fingerprint(
                    validate_benchmark(benchmark_root), verify_model(model_dir), config, device
                )
                != fingerprint_inputs
            ):
                raise ValueError("Training fingerprint changed during execution")
            _save_evaluation(out / "TRAINING_RESULT.json", result, fingerprint)
            state.update(
                status="PASS",
                finished_at=time.time(),
                result_sha256=file_sha(out / "TRAINING_RESULT.json"),
            )
            state.pop("error", None)
            state.pop("failed_at", None)
            atomic_json(state_path, state)
            notify({"stage": "complete", "selection": result["selection"]})
            return result
        except BaseException as exc:
            state.update(
                status="INTERRUPTED"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "FAILED",
                error={"type": type(exc).__name__, "message": str(exc)},
                failed_at=time.time(),
            )
            atomic_json(
                out / f"FAILURE_{time.time_ns()}.json",
                {
                    "fingerprint": fingerprint,
                    "status": state["status"],
                    "error": state["error"],
                    "completed_epochs": state["completed_epochs"],
                    "failed_at": state["failed_at"],
                },
            )
            atomic_json(state_path, state)
            raise
