"""Bounded application workflow using the native scientific engine and checkpoints.

Network discovery, extraction, inspection and qualification are separate persisted
stages. A completed computation never implies that a question is novel or that a
dataset has adequate statistical power.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .adapters import DATASET_ADAPTERS, LITERATURE_ADAPTERS, _merge_literature_duplicate
from .data_plane import atomic_json, atomic_write, digest
from .discovery import discover
from .provenance import repository_identity
from .question_compiler import compile_question
from .retrieval import BM25Index
from .scientific_v2 import rank_candidates
from .strict_artifacts import strict_json_loads
from .tabular_contracts import scope_tabular_comparison
from .v2 import normalize_dataset, read_rows
from .workflow_checkpoint import CheckpointRun, file_identity

VERSION = "platform-workflow-1.0"


def validate_request(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Request must be an object")
    allowed = {
        "question",
        "query",
        "requirements",
        "documents",
        "document_ids",
        "literature_sources",
        "dataset_sources",
        "limit",
        "inspection_limit",
        "extraction_mode",
        "ranking",
    }
    if set(raw) - allowed:
        raise ValueError("Unknown request fields: " + ", ".join(sorted(set(raw) - allowed)))
    value = dict(raw)
    for field in ("question", "query"):
        value.setdefault(field, "")
        if not isinstance(value[field], str) or len(value[field]) > 4000:
            raise ValueError(f"{field} must contain at most 4000 characters")
        value[field] = value[field].strip()
    if not (value["question"] or value["query"]):
        raise ValueError("Provide a scientific question or literature query")
    value.setdefault("requirements", [])
    if not isinstance(value["requirements"], list) or len(value["requirements"]) > 30:
        raise ValueError("At most 30 explicit requirements are supported")
    # Validate using the owning compiler; never accept a client-supplied assessment.
    compile_question(value["question"] or value["query"], user_constraints=value["requirements"])
    for key, registry, default in (
        ("literature_sources", LITERATURE_ADAPTERS, ["europepmc"]),
        ("dataset_sources", DATASET_ADAPTERS, ["geo"]),
    ):
        value.setdefault(key, default)
        if (
            not isinstance(value[key], list)
            or len(value[key]) > 4
            or any(not isinstance(s, str) or s not in registry for s in value[key])
            or len(set(value[key])) != len(value[key])
        ):
            raise ValueError(f"{key} must contain up to four unique supported sources")
    for key, default, maximum in (("limit", 5, 20), ("inspection_limit", 3, 10)):
        value.setdefault(key, default)
        if (
            type(value[key]) is not int
            or not 0 <= value[key] <= maximum
            or (key == "limit" and not value[key])
        ):
            raise ValueError(f"{key} is out of range")
    value.setdefault("extraction_mode", "rules")
    value.setdefault("ranking", "bm25")
    if value["extraction_mode"] not in {"rules", "model"} or value["ranking"] not in {
        "bm25",
        "dense",
        "hybrid",
    }:
        raise ValueError("Unsupported extraction or ranking mode")
    value.setdefault("documents", [])
    value.setdefault("document_ids", [])
    if (
        not isinstance(value["document_ids"], list)
        or len(value["document_ids"]) > 20
        or any(
            not isinstance(key, str) or not 1 <= len(key) <= 160 for key in value["document_ids"]
        )
    ):
        raise ValueError("At most twenty bounded source document IDs are supported")
    value["document_ids"] = list(dict.fromkeys(value["document_ids"]))
    if not isinstance(value["documents"], list) or len(value["documents"]) > 10:
        raise ValueError("At most ten submitted documents are supported")
    cleaned = []
    for document in value["documents"]:
        if not isinstance(document, dict) or set(document) - {"title", "text", "url"}:
            raise ValueError("Documents accept title, text and optional source URL")
        if (
            any(not isinstance(v, str) for v in document.values())
            or not document.get("text", "").strip()
            or len(document["text"]) > 100_000
        ):
            raise ValueError("Each document requires 1..100000 characters of source text")
        if document.get("url") and not document["url"].startswith(("http://", "https://")):
            raise ValueError("Document URL must be HTTP(S)")
        identifier = "submitted:" + digest(document)[:24]
        cleaned.append(
            dict(
                document,
                document_id=identifier,
                source_locator=document.get("url") or identifier,
                source_type="user_submitted_unverified",
            )
        )
    value["documents"] = cleaned
    return value


def _local_rows(root, filename, limit=10000):
    path = Path(root) / "catalog" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Configured catalog is missing: {filename}")
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(strict_json_loads(line))
            if len(rows) >= limit:
                break
    return rows


def _retrieval_text(row):
    parts = [str(row.get(f, "")) for f in ("title", "text", "abstract", "summary", "description")]
    for field, observation in row.get("capabilities", {}).items():
        if (
            isinstance(observation, dict)
            and observation.get("status") == "observed"
            and observation.get("source_locator")
        ):
            parts.extend([field, str(observation.get("value", ""))])
    return " ".join(parts)


def _select(rows, query, limit, id_field, *, ranking="bm25", config=None, work=None):
    unique = {}
    for row in rows:
        identifier = row.get(id_field) or row.get("source_id") or row.get("id")
        if identifier:
            unique.setdefault(str(identifier), dict(row, **{id_field: str(identifier)}))
    records = [
        {
            "id": key,
            "text": _retrieval_text(row),
        }
        for key, row in unique.items()
    ]
    records = [r for r in records if r["text"].strip()]
    if not records:
        return []
    hits = BM25Index().fit(records).search(query, k=limit)
    if ranking in {"dense", "hybrid"}:
        from .retrieval import CachedSemanticIndex, reciprocal_rank_fusion

        if not config or not config.get("embedding_dir"):
            raise ValueError("Semantic candidate generation requires a local embedding model")
        index = CachedSemanticIndex(
            config["embedding_dir"],
            Path(config["cache_root"]) / "catalog-vectors"
            if config.get("cache_root")
            else work / "catalog-vectors",
            device=config.get("device", "cpu"),
        )
        index.fit(records)
        dense = index.search(query, k=limit)
        selected = dense if ranking == "dense" else reciprocal_rank_fusion([hits, dense], k=limit)
        return [unique[r["id"]] for r in selected]
    return [unique[r["id"]] for r in hits if r["score"] > 0]


def _merge_documents(documents):
    unique = {}
    for raw in documents:
        identifier = raw.get("document_id") or raw.get("source_id") or raw.get("id")
        if not identifier:
            continue
        document = dict(raw)
        source_id = str(document.get("source_id") or identifier)
        pmid = str(document.get("pmid") or "")
        if not pmid:
            match = re.fullmatch(r"(?:pmid|pubmed):(\d+)", str(identifier), re.I)
            pmid = match.group(1) if match else ""
        if re.fullmatch(r"[0-9]+", pmid) and int(pmid) > 0:
            pmid = str(int(pmid))
            identifier = "PMID:" + pmid
            document["pmid"] = pmid
        document.update(document_id=identifier, source_id=source_id)
        if identifier in unique:
            if document != unique[identifier]:
                _merge_literature_duplicate(unique[identifier], document)
        else:
            unique[identifier] = document
    return list(unique.values())


def _acquire(request, config, work):
    query = request["query"] or request["question"]
    documents = list(request["documents"])
    datasets, sources = [], []
    unresolved = set(request["document_ids"])
    if config.get("catalog_root"):
        root = config["catalog_root"]
        catalog_documents = _local_rows(root, "literature.jsonl")
        for document in catalog_documents:
            pmid = str(document.get("pmid") or "")
            keys = {document.get("document_id"), document.get("id"), document.get("source_id")}
            if pmid.isdigit():
                keys.update({"PMID:" + pmid, "pmid:" + pmid, "pubmed:" + pmid})
            selected = unresolved & keys
            if selected:
                documents.append(document)
                unresolved -= selected
        documents.extend(_select(catalog_documents, query, request["limit"], "document_id"))
        datasets.extend(
            _select(
                _local_rows(root, "studies.jsonl"),
                query + " " + " ".join(str(r["expected"]) for r in request["requirements"]),
                request["limit"] * 4,
                "dataset_id",
                ranking=request["ranking"],
                config=config,
                work=work,
            )
        )
        sources.append(
            {"source": "configured_local_catalog", "status": "SEARCHED", "scan_limit": 10000}
        )
    if config.get("literature_index"):
        from .literature_index import LiteratureIndex

        with LiteratureIndex(config["literature_index"], read_only=True) as index:
            index.conn.execute("BEGIN")
            for identifier in list(unresolved):
                match = re.fullmatch(r"(?:pmid|pubmed):(\d+)", identifier, re.I)
                document = index.get(match.group(1)) if match else None
                if document:
                    documents.append(document)
                    unresolved.remove(identifier)
            documents.extend(index.search(query, limit=request["limit"]))
            sources.append(
                {
                    "source": "materialized_pubmed",
                    "status": "SEARCHED",
                    "snapshot": index.snapshot_identity(),
                }
            )
    if unresolved:
        raise ValueError(
            "Selected documents are unavailable in the configured source snapshot: "
            + ", ".join(sorted(unresolved))
        )
    if request["literature_sources"] or request["dataset_sources"]:
        plan = {
            "schema_version": "discovery-plan-1.0",
            "queries": [
                {
                    "id": "research",
                    "query": query,
                    "limit": request["limit"],
                    "literature_sources": request["literature_sources"],
                    "dataset_sources": request["dataset_sources"],
                }
            ],
        }
        location = work / "discovery"
        receipt = discover(
            plan,
            location,
            workers=1,
            offline=config.get("offline", False),
            resume=(location / "DISCOVERY_STATE.json").exists(),
            retry_failed=True,
            cache_root=Path(config["cache_root"]) / "http" if config.get("cache_root") else None,
        )
        documents.extend(read_rows(location / "catalog/literature.jsonl"))
        datasets.extend(read_rows(location / "catalog/studies.jsonl"))
        sources.append({"source": "public_repository_discovery", "receipt": receipt})
    dataset_ids = {}
    for dataset in datasets:
        dataset_ids.setdefault(dataset["dataset_id"], dataset)
    return {
        "documents": _merge_documents(documents)[: request["limit"] * 4],
        "datasets": list(dataset_ids.values())[: request["limit"] * 4],
        "sources": sources,
    }


def _extract(request, config, acquired, work):
    from .question_discovery import discover_questions

    runtime = None
    if request["extraction_mode"] == "model":
        from .semantic_runtime import LocalSemanticRuntime, RuntimeConfig

        if not config.get("model_dir"):
            raise ValueError("Model extraction needs a configured verified local model")
        runtime = LocalSemanticRuntime(
            config["model_dir"],
            RuntimeConfig(
                device=config.get("device", "cpu"),
                decoding_mode="source_literal",
                dtype="bfloat16" if config.get("device") == "cuda" else "float32",
                max_new_tokens=512,
            ),
        )
    result = discover_questions(
        acquired["documents"],
        runtime=runtime,
        cache_root=Path(config["cache_root"]) / "extraction"
        if config.get("cache_root")
        else work / "extraction-cache",
        max_documents=80,
        max_questions=40,
    )
    if request["question"]:
        result["selected_question"] = {
            "question_id": "question:" + digest(request["question"])[:24],
            "question": request["question"],
            "source_locator": "user-submitted-question",
            "provenance_category": "USER_SPECIFIED",
            "novelty_status": "NOT_ASSESSED",
        }
    elif result["questions"]:
        result["selected_question"] = result["questions"][0]
    else:
        result["selected_question"] = None
    return result


def _inspect(request, config, acquired, work, workflow=None):
    datasets = []
    for i, raw in enumerate(acquired["datasets"]):
        if raw.get("inspection_status") == "INSPECTED_MEASUREMENTS":
            # Server-configured, provenance-bearing processed inspection input.
            # Its file hash is pinned with the catalog; preserve successful work.
            datasets.append(normalize_dataset(raw))
            continue
        if i < request["inspection_limit"]:
            from .dataset_inspection import inspect_dataset

            try:
                key = "dataset-inspection:" + digest(raw)
                cached = workflow.load(key) if workflow else None
                if cached is not None:
                    raw = cached
                else:
                    raw = inspect_dataset(
                        raw,
                        Path(config["cache_root"]) / "inspection"
                        if config.get("cache_root")
                        else work / "inspection-cache",
                        offline=config.get("offline", False),
                        max_samples=40,
                        max_bytes=8_000_000,
                    )
                    if workflow and not _inspection_failed(raw):
                        workflow.save(key, raw)
            except (ValueError, OSError, RuntimeError) as exc:
                raw = dict(raw, inspection_status="FAILED", inspection_error=str(exc))
        else:
            raw = dict(raw, inspection_status="NOT_REQUESTED")
        datasets.append(normalize_dataset(raw))
    return {"datasets": datasets}


def _qualify(request, config, extracted, inspected, acquired, work):
    from .question_discovery import check_addressed_evidence, compile_research_question

    question = extracted["selected_question"]
    if not question:
        return {
            "status": "NO_SOURCE_QUESTION",
            "question": None,
            "compilation": None,
            "candidates": [],
            "reason": "No explicit unanswered question was extracted. Submit a question to continue.",
        }
    text = question.get("question") or question.get("question_text") or question.get("quote")
    document = next(
        (
            doc
            for doc in acquired["documents"]
            if doc.get("document_id") == question.get("source_document_id")
        ),
        {},
    )
    compiled = compile_research_question(
        text,
        source_text=document.get("text") or document.get("abstract"),
        source_locator=question.get("source_locator") or "selected-question",
        user_constraints=request["requirements"],
    )
    requirements = compiled["matching_requirements"]
    datasets = inspected["datasets"]
    records = [
        {
            "id": d["dataset_id"],
            "text": _retrieval_text(d).strip() or d["dataset_id"],
        }
        for d in datasets
    ]
    lexical = BM25Index().fit(records).search(text, k=len(records)) if records else []
    dense = []
    if records and request["ranking"] in {"dense", "hybrid"}:
        from .retrieval import CachedSemanticIndex

        if not config.get("embedding_dir"):
            raise ValueError("Semantic ranking needs a configured verified local embedding model")
        index = CachedSemanticIndex(
            config["embedding_dir"],
            Path(config["cache_root"]) / "qualified-vectors"
            if config.get("cache_root")
            else work / "vectors",
            device=config.get("device", "cpu"),
        )
        index.fit(records)
        dense = index.search(text, k=len(records))
    cosine = {r["id"]: r["score"] for r in dense}
    lexical_rank = {r["id"]: i + 1 for i, r in enumerate(lexical)}
    dense_rank = {r["id"]: i + 1 for i, r in enumerate(dense)}
    # Project only this compiled contract; keep persisted catalog profiles unchanged.
    scoped = [scope_tabular_comparison(dataset, requirements) for dataset in datasets]
    ranked = rank_candidates(requirements, scoped, cosine) if requirements else []
    lookup = {d["dataset_id"]: d for d in scoped}
    for row in ranked:
        key = row["dataset_id"]
        row["dataset"] = lookup[key]
        row["retrieval"] = {
            "method": request["ranking"],
            "lexical_rank": lexical_rank.get(key),
            "dense_rank": dense_rank.get(key),
            "cosine": cosine.get(key),
            "rrf": sum(
                1 / (60 + ranks[key]) for ranks in (lexical_rank, dense_rank) if key in ranks
            ),
        }
        row["explanation"] = [
            f"{r['field']}: {r['status']} (required {r['expected']})"
            for r in row["assessment"]["requirements"]
        ]
    order = {"DIRECT_FIT": 0, "PARTIAL_FIT": 1, "REQUIRES_INSPECTION": 2, "NOT_QUALIFIED": 3}
    ranked.sort(
        key=lambda r: (
            order[r["assessment"]["eligibility"]],
            -(
                r["retrieval"]["rrf"]
                if request["ranking"] == "hybrid"
                else cosine.get(r["dataset_id"], 0)
                if request["ranking"] == "dense"
                else 1 / lexical_rank.get(r["dataset_id"], 10000)
            ),
            -r["assessment"]["coverage"],
            r["dataset_id"],
        )
    )
    return {
        "status": "QUESTION_UNDERDETERMINED"
        if not requirements
        else "ASSESSED"
        if ranked
        else "NO_CANDIDATES",
        "question": question,
        "compilation": compiled,
        "candidates": ranked,
        "addressed_evidence": check_addressed_evidence(question, acquired["documents"]),
        "retrieval_baselines": {"bm25": lexical, "dense": dense},
        "ranking_policy": "Scientific eligibility first, selected retrieval relevance second, requirement coverage third; scores are not probabilities.",
        "scientific_status": "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW",
        "score_type": "UNCALIBRATED_HEURISTIC",
    }


def _report(result):
    question = result.get("question") or {}
    lines = [
        "# LitDataMatcher research opportunity dossier",
        "",
        f"Result: {result['result_id']}",
        f"Scientific status: {result['scientific_status']}",
        f"Processing status: {result['processing_status']}",
        "Retryable stages: " + (", ".join(result["retryable_stages"]) or "none"),
        "",
        question.get("question", "No source question selected"),
        "Question origin: "
        + str(question.get("origin") or question.get("provenance_category") or "not reported"),
        "Source locator: " + str(question.get("source_locator") or "not reported"),
        "Citation: " + json.dumps(question.get("citation", {}), ensure_ascii=False),
        "Quoted source passage: "
        + str(
            question.get("evidence_span", {}).get("text")
            or "No source passage; user-specified question."
        ),
        "",
        "No global novelty, statistical adequacy, or expert validation is asserted.",
        "",
    ]
    for row in result["candidates"]:
        lines.extend(
            [
                f"## {row['dataset_id']} — {row['dataset'].get('title', '')}",
                f"Decision: {row['assessment']['eligibility']}",
                f"Compatibility: {row['assessment']['compatibility_status']}",
                "",
            ]
        )
        for requirement in row["assessment"]["requirements"]:
            observation = requirement["observation"]
            lines.append(
                f"- {requirement['field']}: {requirement['status']}; required {requirement['expected']}; observed {observation['value']}; source {observation['source_locator'] or 'unreported'}"
            )
        scope = row["dataset"].get("comparison_scope")
        if scope:
            evidence = next(
                item
                for item in row["dataset"]["inspection"]["comparison_evidence"]
                if item["evidence_id"] == scope["evidence_id"]
            )
            lines.append(
                "Exact comparison: "
                + scope["outcome_variable"]
                + " by "
                + scope["grouping_variable"]
                + " groups "
                + ", ".join(scope["group_values"])
            )
            lines.append(
                "Joint observed rows by group: " + json.dumps(evidence["joint_rows_by_group"])
            )
            for definition in evidence["definition_evidence"]:
                lines.append(
                    f"Source definition of {definition['variable']}: {definition['text']} ({definition['source_locator']})"
                )
            lines.extend("- " + text for text in evidence["limitations"])
        lines.append("")
    lines.extend(
        ["## Coverage and limitations", "", *["- " + value for value in result["limitations"]]]
    )
    return "\n".join(lines) + "\n"


def _inspection_failed(row):
    return row.get("inspection_status") in {
        "FAILED",
        "SOURCE_FAILURE",
        "PARTIAL_SOURCE_FAILURE",
    } or bool(row.get("inspection", {}).get("failures"))


def _retryable(stage, value):
    if stage == "discovery":
        return any(s.get("receipt", {}).get("failed_source_jobs", 0) for s in value["sources"])
    if stage == "questions":
        return bool(value.get("coverage", {}).get("failed_documents"))
    if stage == "inspection":
        return any(_inspection_failed(d) for d in value["datasets"])
    return False


def run_workflow(request, config, out: Path, *, progress_path=None):
    # Request is already canonicalized at submission; a frozen on-disk request is
    # the only input the controller executes. No shell or paths come from the UI.
    out = Path(out)
    package = Path(__file__).parent
    from .semantic_runtime import runtime_dependency_versions

    used_models = []
    if request["extraction_mode"] == "model" and config.get("model_dir"):
        used_models.append(config["model_dir"])
    if request["ranking"] != "bm25" and config.get("embedding_dir"):
        used_models.append(config["embedding_dir"])
    identity = {
        "version": VERSION,
        "request": request,
        "configuration": config,
        "implementation": [file_identity(p) for p in sorted(package.glob("*.py"))],
        "rules": file_identity(package / "question_compiler_rules_v1.json"),
        "catalog": [
            file_identity(Path(config["catalog_root"]) / "catalog" / f)
            for f in ("literature.jsonl", "studies.jsonl")
        ]
        if config.get("catalog_root")
        else [],
        "models": [file_identity(path) for path in used_models],
        "runtime_dependencies": runtime_dependency_versions(),
    }
    resume = (out / ".workflow/state.json").exists()
    start = time.perf_counter()
    with CheckpointRun(out, identity, resume=resume) as workflow:
        completed = workflow.completed()
        if completed is not None:
            # Controller requires fresh publication when repairing a missing receipt.
            for name in ("result.json", "dossier.md", "candidates.csv"):
                path = out / name
                atomic_write(path, path.read_bytes())
            return completed

        def stage(name, function, parents=()):
            if progress_path:
                atomic_json(
                    progress_path, {"stage": name, "status": "RUNNING", "updated_at": time.time()}
                )
            prior = workflow.load(name, parents)
            if prior is None or _retryable(name, prior):
                prior = workflow.save(name, function(), parents)
            return prior

        acquired = stage("discovery", lambda: _acquire(request, config, out))
        extracted = stage(
            "questions", lambda: _extract(request, config, acquired, out), ("discovery",)
        )
        inspected = stage(
            "inspection", lambda: _inspect(request, config, acquired, out, workflow), ("discovery",)
        )
        qualified = stage(
            "qualification",
            lambda: _qualify(request, config, extracted, inspected, acquired, out),
            ("questions", "inspection"),
        )
        retryable_stages = [
            name
            for name, value in (
                ("discovery", acquired),
                ("questions", extracted),
                ("inspection", inspected),
            )
            if _retryable(name, value)
        ]
        workflow.state["retryable_stage_failures"] = retryable_stages
        result = dict(
            qualified,
            processing_status="PARTIAL_RETRYABLE" if retryable_stages else "COMPLETE",
            retryable_stages=retryable_stages,
            schema_version=VERSION,
            result_id="result:" + digest([identity, acquired])[:24],
            request=request,
            questions=extracted,
            sources=acquired["sources"],
            documents=acquired["documents"],
            datasets=inspected["datasets"],
            scientific_status="SOURCE_ASSISTED_PENDING_EXPERT_REVIEW",
            created_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - start,
            software={
                "source": repository_identity(package.parent),
                "python": platform.python_version(),
                "workflow": VERSION,
            },
            input_fingerprint=digest([identity, acquired]),
            limitations=[
                "Search coverage is bounded; source failure is not evidence of absence.",
                "Source-assisted compatibility is not expert validation or a power analysis.",
                "Unreported requirements remain unknown; no global novelty is asserted.",
                "Submitted text is user-supplied evidence and is not independently authenticated.",
            ],
        )
        atomic_json(out / "result.json", result)
        atomic_write(out / "dossier.md", _report(result).encode("utf-8"))
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset_id",
                "eligibility",
                "compatibility",
                "coverage",
                "unknown_fields",
                "score_type",
            ]
        )
        for row in result["candidates"]:
            a = row["assessment"]
            writer.writerow(
                [
                    row["dataset_id"],
                    a["eligibility"],
                    a["compatibility_status"],
                    a["coverage"],
                    a["unknown_fields"],
                    row["score_type"],
                ]
            )
        atomic_write(out / "candidates.csv", stream.getvalue().encode("utf-8"))
        # The platform manifest has its own explicit schema; don't leave a native
        # RUN_MANIFEST implying that unexecuted native stages succeeded.
        (out / "RUN_MANIFEST.json").unlink(missing_ok=True)
        atomic_json(
            out / "platform_manifest.json",
            {
                "schema_version": VERSION,
                "result_id": result["result_id"],
                "inputs": identity,
                "outputs": [
                    file_identity(out / name)
                    for name in ("result.json", "dossier.md", "candidates.csv")
                ],
            },
        )
        workflow.finish(result)
        if progress_path:
            atomic_json(
                progress_path,
                {
                    "stage": "complete",
                    "status": result["processing_status"],
                    "updated_at": time.time(),
                },
            )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress")
    args = parser.parse_args(argv)
    saved = strict_json_loads(Path(args.request).read_text(encoding="utf-8"))
    result = run_workflow(
        saved["request"], saved["configuration"], Path(args.out), progress_path=args.progress
    )
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "status": result["status"],
                "candidates": len(result["candidates"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
