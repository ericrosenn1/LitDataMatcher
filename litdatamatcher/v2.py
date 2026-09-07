"""Local v2 commands: independently sync, infer, match and review real sources."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from .data_plane import Catalog, atomic_json, atomic_write, digest
from .schemas import stable_id
from .scientific_v2 import compile_evidence, discover_cross_document_gaps, rank_candidates


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_rows(path, rows):
    atomic_write(
        path,
        b"".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False).encode() + b"\n"
            for r in rows
        ),
    )


def normalize_dataset(record):
    """Explicit acquisition-v2.1 -> experimental contract migration; raw retained."""
    result = dict(record)
    caps = {}
    for key, raw in record.get("capabilities", {}).items():
        value = dict(raw)
        if value.get("status") == "known":
            value["status"] = "observed"
        locator = value.get("source_locator")
        if isinstance(locator, list):
            value["source_locator"] = "; ".join(map(str, locator)) or None
        value = {
            k: v
            for k, v in value.items()
            if k in {"value", "status", "source_locator", "reason", "mapping_type"}
        }
        caps[key] = value
    result["capabilities"] = caps
    result["availability"] = record.get("access_status", "UNKNOWN")
    result["migration"] = {
        "from": record.get("schema_version"),
        "to": "experimental-contract-2.0",
        "raw_record_digest": digest(record),
        "changed_semantics": False,
    }
    return result


def source_chunks(document, max_chars=1800, max_chunks=2):
    """Explicit section windows; retain parent offsets and visible coverage limits."""
    text = document.get("text", "")
    sections = document.get("sections") or [
        {"start": 0, "end": len(text), "text": text, "section": "unstructured"}
    ]
    def priority(section):
        return (
            0
            if re.search("result|discussion|conclusion", section.get("section", ""), re.I)
            else 1,
            section.get("start", 0),
        )
    result = []
    for section in sorted(sections, key=priority):
        start = section.get("start", 0)
        end = section.get("end", len(text))
        if text[start:end] != section.get("text", text[start:end]):
            raise ValueError("Section offset corruption")
        if not text[start:end].strip():
            continue
        # Keep only complete sentences within bounded context, never silently truncate.
        while start < end and len(result) < max_chunks:
            stop = min(end, start + max_chars)
            if stop < end:
                candidates = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", text[start:stop])]
                if not candidates:
                    break
                stop = start + candidates[-1]
            view = text[start:stop]
            result.append(
                {
                    "document_id": document["document_id"],
                    "title": document.get("title", ""),
                    "text": view,
                    "parent_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "parent_start": start,
                    "parent_end": stop,
                    "section": section.get("section", "unknown"),
                    "source_provenance": {
                        "source_locator": document.get("source_locator"),
                        "source_snapshot": document.get(
                            "fulltext_snapshot", document.get("source_snapshot")
                        ),
                    },
                }
            )
            start = stop
        if len(result) >= max_chunks:
            break
    return result


_EXPLICIT_GAP = re.compile(
    r"\b(?:further (?:studies|research)|remains? (?:unclear|unknown|unresolved)|"
    r"not (?:yet )?known)\b",
    re.I,
)


def explicit_unresolved_questions(document: dict, view: dict) -> list[dict]:
    """Create source-linked gap records only from explicit source language."""
    questions = []
    text = view["text"]
    for sentence in re.finditer(r"(?:^|(?<=[.!?])\s+)([^.!?]*[.!?])", text, re.S):
        statement = sentence.group(1).strip()
        if not statement or not _EXPLICIT_GAP.search(statement):
            continue
        local_start = text.find(statement, sentence.start(1), sentence.end(1))
        start = view["parent_start"] + local_start
        end = start + len(statement)
        if document["text"][start:end] != statement:
            raise ValueError("Explicit-gap parent span mismatch")
        locator = document.get("source_locator", document["document_id"])
        question_id = stable_id("question", document["document_id"], start, end)
        questions.append(
            {
                "question_id": question_id,
                "question": statement,
                "text": statement,
                "origin": "explicit_unresolved_source",
                "proposition_id": stable_id("proposition", statement),
                "requirements": [],
                "conditions": {},
                "gap_status": "insufficient-coverage",
                "source_document_id": document["document_id"],
                "source_locator": f"{locator}#chars={start}:{end}",
                "evidence_span": {"start": start, "end": end, "text": statement},
                "novelty_claim": "None; source future-work language alone does not establish novelty",
            }
        )
    return questions


def rebase_runtime_item(item: dict, document: dict, view: dict, view_id: str) -> dict:
    """Map a view-local runtime record to stable parent-document coordinates."""
    result = dict(item)
    span = dict(result["evidence_span"])
    span["start"] += view["parent_start"]
    span["end"] += view["parent_start"]
    if document["text"][span["start"] : span["end"]] != span["text"]:
        raise ValueError("Parent span mismatch")
    prefix = "clm" if "claim_id" in result else "que"
    id_field = "claim_id" if prefix == "clm" else "question_id"
    result[id_field] = (
        prefix
        + "_"
        + digest([document["document_id"], span["start"], span["end"], span["text"]])[:24]
    )
    result["evidence_span"] = span
    result["source_document_id"] = document["document_id"]
    result["source_view_id"] = view_id
    locator = document.get("source_locator", document["document_id"])
    result["source_locator"] = f"{locator}#chars={span['start']}:{span['end']}"
    return result


def qualified_runtime(qualification_path: Path, runtime) -> tuple[bool, str | None]:
    """Accept a backend qualification only when its inspected model identity matches."""
    if not qualification_path.is_file():
        return False, None
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        passed = (
            qualification.get("status") == "PASS"
            and qualification["fresh"]["inference_manifest"]["origin"] == "fresh_local_inference"
            and qualification["fresh"]["inference_manifest"]["model_revision"]
            == runtime.model_manifest["revision"]
            and qualification.get("replay_origin") == "cache_replay"
            and qualification.get("network_control", {}).get("blocked_probe")
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, None
    return bool(passed), hashlib.sha256(qualification_path.read_bytes()).hexdigest()


def validate_run_artifact(path: Path) -> str:
    """Parse every persisted artifact type before recording validation PASS."""
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        return "FAIL"
        elif path.suffix == ".csv":
            import csv

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            if not rows or len(rows[0]) < 2:
                return "FAIL"
        elif path.suffix == ".html":
            content = path.read_text(encoding="utf-8")
            if (
                "<!doctype html>" not in content.casefold()
                or "Content-Security-Policy" not in content
            ):
                return "FAIL"
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            return "NOT_RUN"
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "FAIL"
    return "PASS"


def evidence_from_claim(claim, document):
    # A direction of increase is not automatically support for an arbitrary question.
    # Claim propositions include their verbatim relation and context.
    prop = stable_id(
        "proposition", claim["subject"], claim["predicate"], claim["object"], claim.get("context")
    )
    return {
        "evidence_id": claim["claim_id"],
        "proposition_id": prop,
        "role": "direct_test" if claim["status"] == "direct_experiment" else claim["status"],
        "direction": "inconclusive",
        "reported_direction": claim["direction"],
        "source_id": document["document_id"],
        "source_document_id": document["document_id"],
        "publication_id": document.get("pmid") or document.get("doi"),
        "publication_date": document.get("publication_date"),
        "study_id": None,
        "cohort_id": None,
        "source_of_source": None,
        "conditions": {"source_context": claim.get("context")},
        "measurement_type": "observation"
        if claim["status"] == "direct_experiment"
        else "interpretation",
        "source_locator": claim["source_locator"],
        "scope_match": "unresolved",
        "answers_question": False,
        "statement": claim["statement"],
        "claim": claim,
    }


def render_report(run: Path) -> Path:
    manifest = json.loads((run / "RUN_MANIFEST.json").read_text())
    matches = read_rows(run / "matches.jsonl")
    questions = {q["question_id"]: q for q in read_rows(run / "questions.jsonl")}
    bundles = {b["question_id"]: b for b in read_rows(run / "evidence_bundles.jsonl")}
    def esc(value):
        return html.escape(str(value), quote=True)
    cards = []
    for match in matches:
        q = questions[match["question_id"]]
        a = match["assessment"]
        bundle = bundles[q["question_id"]]
        rows = "".join(
            f"<tr><td>{esc(r['field'])}</td><td>{esc(r['expected'])}</td><td>{esc(r['status'])}</td><td>{esc(r['observation']['source_locator'])}</td></tr>"
            for r in a["requirements"]
        )
        evidence = "".join(
            f"<li>{esc(e.get('statement', e.get('description', '')))} <small>{esc(e['source_locator'])}</small></li>"
            for e in bundle["evidence_items"]
        )
        cards.append(
            f"<article><h2>{esc(q.get('question', q.get('text')))}</h2><h3>{esc(match['dataset_id'])}: {esc(a['eligibility'])}</h3><p>{esc(bundle['gap_status'])}, as of {esc(bundle['as_of'])}. Review priority {match['score']:.3f} (uncalibrated heuristic).</p><table><thead><tr><th>Requirement</th><th>Expected</th><th>Fit</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table><p>Independent units: {esc(a['independent_units'])}. Statistical adequacy: {esc(a['statistical_adequacy'])}.</p><p>Next analysis: verify missing variables, usable contrasts and design-specific power; then analyze the submitted measurements under a documented estimand. No downstream experiment has been performed by this ranking.</p><details><summary>Evidence and dependence</summary><ul>{evidence}</ul><pre>{esc(json.dumps(bundle['dependence_groups'], indent=2))}</pre></details></article>"
        )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>LitDataMatcher v2 review</title><style>body{{max-width:1100px;margin:auto;padding:2rem;font:16px system-ui;color:#20313a;background:#f4f7f7}}article{{background:white;border:1px solid #ccdada;border-radius:10px;margin:1rem 0;padding:1.5rem}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{border-bottom:1px solid #ddd;padding:.5rem;text-align:left;overflow-wrap:anywhere}}small,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}h1{{color:#17515c}}</style><h1>LitDataMatcher v2 scientific review</h1><p>Run {esc(manifest["run_id"])} · {esc(manifest["execution_status"])}. Experimental fit and evidence remain inspectable; scores are not probabilities.</p><p>Acquisition coverage: {esc(json.dumps(manifest["coverage"]))}</p><p>Inference coverage is explicitly bounded to selected document sections. Metadata availability does not establish statistical answerability. Expert calibration pending.</p>{"".join(cards) or "<p>No assessable questions or matches. Inspect failures and source coverage.</p>"}</html>"""
    target = run / "report.html"
    atomic_write(target, content.encode())
    return target


def analyze(
    root: Path,
    out: Path,
    model_dir: Path,
    embedding_dir: Path,
    *,
    question=None,
    requirements=None,
    limit=3,
    chunks=2,
    fresh=False,
    device="cpu",
    document_path=None,
):
    from .semantic_runtime import LocalSemanticRuntime, PretrainedSemanticIndex, RuntimeConfig

    if out.exists():
        raise ValueError("Run output already exists; use a new run ID or the resume command")
    out.mkdir(parents=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    documents = read_rows(root / "catalog/literature.jsonl")
    all_datasets = [normalize_dataset(r) for r in read_rows(root / "catalog/studies.jsonl")]
    if document_path:
        from .ingestion import ingest_literature_sources

        localout = out / "local_ingestion"
        localout.mkdir()
        ingest_literature_sources([str(document_path)], localout / "literature.jsonl")
        documents = read_rows(localout / "literature.jsonl")
    # Reserved transfer/final families are never silently consumed in development.
    selected = sorted(
        [
            d
            for d in documents
            if d.get("split_context", "development") == "development" and d.get("text")
        ],
        key=lambda d: d["document_id"],
    )[:limit]
    datasets = [d for d in all_datasets if d.get("split_context", "development") == "development"]
    runtime = LocalSemanticRuntime(
        model_dir, RuntimeConfig(device=device, dtype="bfloat16" if device == "cuda" else "float32")
    )
    catalog = Catalog(root / "normalized")
    claims = []
    questions = []
    inferences = []
    failures = []
    views = []
    evidence = []
    try:
        for dataset in datasets:
            catalog.upsert(
                "dataset",
                dataset["dataset_id"],
                dataset,
                search_text=dataset["title"] + " " + dataset.get("summary", ""),
            )
        for document in selected:
            catalog.upsert(
                "document", document["document_id"], document, search_text=document.get("title", "")
            )
            for view in source_chunks(document, max_chunks=chunks):
                viewid = digest(view)
                views.append(
                    dict(
                        view_id=viewid,
                        document_id=document["document_id"],
                        start=view["parent_start"],
                        end=view["parent_end"],
                        section=view["section"],
                    )
                )
                runtime_view = dict(view, document_id=viewid)
                try:
                    result = runtime.extract(
                        runtime_view, root / "inference_cache", force_fresh=fresh
                    )
                except (ValueError, RuntimeError, OSError) as exc:
                    failures.append(
                        {
                            "document_id": document["document_id"],
                            "view_id": viewid,
                            "stage": "inference",
                            "description": str(exc),
                        }
                    )
                    continue
                inferences.append(result["inference_manifest"])
                result["claims"] = [
                    rebase_runtime_item(item, document, view, viewid) for item in result["claims"]
                ]
                result["questions"] = [
                    rebase_runtime_item(item, document, view, viewid)
                    for item in result["questions"]
                ]
                for item in result["claims"] + result["questions"]:
                    item["inference_fingerprint"] = result["inference_manifest"]["fingerprint"]
                for claim in result["claims"]:
                    catalog.upsert(
                        "claim", claim["claim_id"], claim, [("document", document["document_id"])]
                    )
                    claims.append(claim)
                    evidence.append(evidence_from_claim(claim, document))
                for q in result["questions"]:
                    q.update(
                        question=q["text"],
                        proposition_id=stable_id("proposition", q["text"]),
                        requirements=[],
                        conditions={},
                    )
                    questions.append(q)
                questions.extend(explicit_unresolved_questions(document, view))
                if result["rejected"]:
                    failures.append(
                        {
                            "document_id": document["document_id"],
                            "view_id": viewid,
                            "stage": "source_guard",
                            "description": "Model records rejected by source guard",
                            "rejections": result["rejected"],
                        }
                    )
        questions.extend(discover_cross_document_gaps(evidence, started[:10]))
        if question:
            questions.insert(
                0,
                {
                    "question_id": stable_id("question", question),
                    "question": question,
                    "origin": "user",
                    "proposition_id": stable_id("proposition", question),
                    "requirements": requirements or [],
                    "conditions": {},
                    "gap_status": "unassessed",
                },
            )
        questions = list({q["question_id"]: q for q in questions}.values())
        index = PretrainedSemanticIndex(embedding_dir, device=device)
        if datasets:
            index.fit(
                [
                    {"id": d["dataset_id"], "text": d["title"] + " " + d.get("summary", "")}
                    for d in datasets
                ]
            )
        matches = []
        bundles = []
        for q in questions:
            # Related retrieved claims remain context; exact proposition support requires a separate justified mapping.
            qtext = q["question"]
            query_tokens = set(re.findall(r"\w+", qtext.lower()))
            contextual = []
            for item in evidence:
                if len(query_tokens & set(re.findall(r"\w+", item["statement"].lower()))) >= 3:
                    contextual.append(dict(item, related_proposition_id=q["proposition_id"]))
            resource_path = root / "external_evidence/uniprot"
            if resource_path.exists():
                from .external_evidence import import_resource, query_resource

                records = import_resource(root, offline=True)
                entities = [
                    e for e in ["TNF", "IL6", "IL1B"] if re.search(r"\b" + e + r"\b", qtext, re.I)
                ]
                contextual.extend(
                    query_resource(records, entities, proposition_id=q["proposition_id"])
                )
            bundle = compile_evidence(q, contextual, started[:10], [])
            bundles.append(bundle)
            semantic = (
                {r["id"]: r["score"] for r in index.search(qtext, k=min(50, len(datasets)))}
                if datasets
                else {}
            )
            lexical = set(catalog.search("dataset", qtext, 50))
            candidates = [d for d in datasets if d["dataset_id"] in lexical | set(semantic)]
            ranked = rank_candidates(q.get("requirements", []), candidates, semantic)
            for m in ranked[:10]:
                m.update(
                    question_id=q["question_id"],
                    match_id=stable_id("match", q["question_id"], m["dataset_id"]),
                    evidence_bundle_id=bundle["bundle_id"],
                )
                matches.append(m)
        for name, rows in [
            ("claims", claims),
            ("questions", questions),
            ("datasets", datasets),
            ("evidence_bundles", bundles),
            ("matches", matches),
            ("inferences", inferences),
            ("source_views", views),
        ]:
            write_rows(out / f"{name}.jsonl", rows)
        inspected = read_rows(root / "catalog/processed_inspections.jsonl")
        coverage = {
            "unique_literature_records": len({d["document_id"] for d in documents}),
            "parsed_full_texts": sum(d.get("fulltext_status") == "parsed" for d in documents),
            "unique_accession_studies": len(
                {d.get("dependence_group", d["dataset_id"]) for d in all_datasets}
            ),
            "sample_profiled_studies": sum(
                d.get("profile_status") == "sample_annotations_parsed" for d in all_datasets
            ),
            "inspected_processed_studies": len(
                {
                    d["dataset_id"]
                    for d in inspected
                    if d.get("processed_measurements_present") and d.get("sample_alignment")
                }
            ),
            "external_structured_resources": int((root / "external_evidence/uniprot").exists()),
            "distinct_pilot_contexts": len({d.get("topic") for d in documents if d.get("topic")}),
            "case_dossiers": len({(m["question_id"], m["dataset_id"]) for m in matches}),
        }
        source_root = Path(__file__).parents[1]
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "installed-distribution"
        qualification_path = root / "runtime-qualification/qualified_7b_pass1.json"
        backend_qualified, qualification_digest = qualified_runtime(qualification_path, runtime)
        lock_path = source_root / "requirements-v2.lock"
        source_snapshots = []
        for document in selected:
            snapshot = document.get("fulltext_snapshot") or document.get("source_snapshot")
            if not isinstance(snapshot, dict) or not snapshot.get("sha256"):
                continue
            source_snapshots.append(
                {
                    "source": snapshot.get(
                        "url", document.get("source_locator", document["document_id"])
                    ),
                    "snapshot_id": snapshot["sha256"],
                    "retrieved_at": snapshot.get("retrieved_at", started),
                    "manifest_digest": digest(snapshot),
                }
            )
        manifest = {
            "schema_version": "2.0",
            "run_id": out.name,
            "execution_status": "PARTIAL" if failures else "PASS",
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "repository": "ericrosenn1/LitDataMatcher",
                "commit": commit,
                "working_tree_digest": digest(
                    {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")
                    }
                ),
                "spec_digest": hashlib.sha256(
                    (source_root / "docs/v2/build_spec/PACKAGE_MANIFEST.json").read_bytes()
                ).hexdigest(),
                "config_digest": digest(
                    {
                        "model": str(model_dir),
                        "question": question,
                        "requirements": requirements,
                        "limit": limit,
                        "chunks": chunks,
                    }
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependency_lock_digest": hashlib.sha256(lock_path.read_bytes()).hexdigest()
                if lock_path.is_file()
                else None,
                "hardware_record": json.dumps(
                    {"machine": platform.machine(), "processor": platform.processor()},
                    sort_keys=True,
                ),
            },
            "models": [
                {
                    "id": runtime.model_manifest["model_id"],
                    "revision": runtime.model_manifest["revision"],
                    "runtime": "transformers",
                    "license_status": str(runtime.model_manifest["license"]),
                    "prompt_version": inferences[0]["fingerprint"]["prompt_sha256"]
                    if inferences
                    else "not-executed",
                }
            ],
            "source_snapshots": source_snapshots,
            "commands": [
                {
                    "command": " ".join(sys.argv),
                    "cwd": str(Path.cwd()),
                    "started_at": started,
                    "exit_code": 0,
                    "log_reference": "inferences.jsonl",
                }
            ],
            "evaluation": {
                "protocol_version": "EP-20260907-1",
                "split_id": "development",
                "split_role": "DEVELOPMENT",
                "label_origins": ["unreviewed"],
                "holdout_exposed_to_tuning": False,
            },
            "coverage": coverage,
            "artifacts": [],
            "network": {
                "mode": "OFFLINE",
                "offline_block_test": False,
                "external_requests_observed": None,
            },
            "inference": {
                "fresh_calls": sum(i["origin"] == "fresh_local_inference" for i in inferences),
                "cache_replays": sum(i["origin"] == "cache_replay" for i in inferences),
                "backend_qualified": backend_qualified,
            },
            "failures": failures,
        }
        if qualification_digest:
            manifest["models"][0]["qualification_sha256"] = qualification_digest
        atomic_json(out / "RUN_MANIFEST.json", manifest)
        report = render_report(out)
        import csv
        import io

        sheet = io.StringIO(newline="")
        writer = csv.writer(sheet)
        writer.writerow(
            [
                "match_id",
                "question_id",
                "dataset_id",
                "eligibility",
                "score",
                "label_origin",
                "review_label",
            ]
        )
        for m in matches:
            writer.writerow(
                [
                    m["match_id"],
                    m["question_id"],
                    m["dataset_id"],
                    m["assessment"]["eligibility"],
                    m["score"],
                    "unreviewed",
                    "",
                ]
            )
        atomic_write(out / "review_sheet.csv", sheet.getvalue().encode())
        manifest["artifacts"] = [
            {
                "path": p.name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "size_bytes": p.stat().st_size,
                "kind": "run_output",
                "validation": validate_run_artifact(p),
            }
            for p in out.iterdir()
            if p.is_file() and p.name != "RUN_MANIFEST.json"
        ]
        if any(item["validation"] == "FAIL" for item in manifest["artifacts"]):
            manifest["execution_status"] = "FAIL"
            manifest["failures"].append(
                {
                    "stage": "artifact_validation",
                    "description": "One or more persisted outputs failed parse validation",
                }
            )
        atomic_json(out / "RUN_MANIFEST.json", manifest)
        return {
            "run": str(out),
            "report": str(report),
            "coverage": coverage,
            "claims": len(claims),
            "questions": len(questions),
            "failures": len(failures),
            "status": manifest["execution_status"],
        }
    finally:
        catalog.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--root", required=True)

    sync = sub.add_parser("sync")
    sync.add_argument("--root", required=True)
    sync.add_argument("--stage", choices=["literature", "datasets"], required=True)
    sync.add_argument("--offline", action="store_true")
    sync.add_argument("--expanded", action="store_true")

    run = sub.add_parser("analyze")
    run.add_argument("--root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--embeddings", required=True)
    run.add_argument("--question")
    run.add_argument("--requirements")
    run.add_argument("--document")
    run.add_argument("--limit", type=int, default=3)
    run.add_argument("--chunks", type=int, default=2)
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    report = sub.add_parser("report")
    report.add_argument("--run", required=True)

    acceptance = sub.add_parser(
        "acceptance",
        help="Derive a strict machine acceptance report from hashed run evidence",
    )
    acceptance.add_argument(
        "--evidence", required=True, help="Versioned acceptance evidence ledger JSON"
    )
    acceptance.add_argument(
        "--out", required=True, help="ACCEPTANCE_REPORT.json output path"
    )

    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .resources import ResourceGovernor

        result = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "data_root": str(Path(args.root).resolve()),
            "resources": ResourceGovernor().admission(),
            "paid_api_enabled": False,
        }
    elif args.command == "sync":
        from .acquisition_v2 import sync_datasets, sync_literature

        if args.stage == "literature":
            result = sync_literature(
                args.root,
                limit=200 if args.expanded else 50,
                fulltexts=50 if args.expanded else 20,
                offline=args.offline,
            )
        else:
            result = sync_datasets(
                args.root,
                limit=100 if args.expanded else 50,
                profiles=30 if args.expanded else 20,
                offline=args.offline,
            )
    elif args.command == "report":
        result = {"report": str(render_report(Path(args.run)))}
    elif args.command == "acceptance":
        from .acceptance import validate_acceptance

        result = validate_acceptance(args.evidence, output_path=args.out)
    else:
        if args.limit < 1 or args.chunks < 1:
            raise ValueError("Positive limits required")
        requirements = (
            json.loads(Path(args.requirements).read_text()) if args.requirements else None
        )
        result = analyze(
            Path(args.root),
            Path(args.out),
            Path(args.model),
            Path(args.embeddings),
            question=args.question,
            requirements=requirements,
            limit=args.limit,
            chunks=args.chunks,
            fresh=args.fresh,
            device=args.device,
            document_path=Path(args.document) if args.document else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("status") == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
