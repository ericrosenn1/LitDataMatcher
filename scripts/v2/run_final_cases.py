"""Prepare derivative real case inputs and execute local inference with denied networking.

This is a bounded source-assisted software qualification, never the sealed holdout.
Input snapshots and prior attempts are immutable; output directories must be new.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher.data_plane import atomic_json, digest
from litdatamatcher.scientific_dossier import render_dossier, validate_dossier
from litdatamatcher.v2 import analyze, document_lifecycle_status, read_rows, write_rows


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


class AbstractText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.headings = []
        self.in_heading = False

    def handle_starttag(self, tag, attrs):
        if tag == "h4":
            self.in_heading = True
        if tag in {"h4", "p", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "h4":
            self.in_heading = False
        if tag in {"h4", "p", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)
        if self.in_heading:
            self.headings.append(data.strip())


def abstract_document(abstract):
    parser = AbstractText()
    parser.feed(abstract)
    # Unicode horizontal whitespace is a recorded presentation transformation;
    # words, numbers, punctuation and paragraph boundaries are preserved.
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in "".join(parser.parts).splitlines() if line.strip())
    headers = [match for match in re.finditer(r"(?m)^.+$", text) if match.group() in parser.headings]
    sections = []
    for index, match in enumerate(headers):
        start = match.end() + 1
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        if start < end:
            sections.append({"start": start, "end": end, "text": text[start:end], "section": match.group()})
    return text, sections


def prepare(args):
    root = args.root.resolve()
    if root.exists():
        raise ValueError("Preserve existing case inputs; choose a new derivative root")
    root.mkdir(parents=True)
    expanded = args.expanded.resolve()
    lineage = {}
    for item in read_rows(expanded / "RECORD_SOURCE_LINEAGE.jsonl"):
        lineage.setdefault(item["source_record_id"], []).append(item)
    documents, cases = [], []
    seen = set()
    for domain, count in (("neurologic", 2), ("cancer", 2), ("environmental", 1), ("metabolic", 1)):
        selected = 0
        for raw in read_rows(expanded / "domains" / domain / "literature.jsonl"):
            row = copy.deepcopy(raw)
            entries = lineage.get(row["document_id"], [])
            if not entries:
                raise ValueError("Missing raw literature lineage")
            entry = next((item for item in entries if item["partition"].startswith(domain + "_")), entries[0])
            snapshot = Path(entry["response_object_path"])
            if file_ref(snapshot)["sha256"] != entry["response_sha256"]:
                raise ValueError("Source snapshot changed")
            source = row["source_provenance"]
            source["metadata"]["cache_snapshot"] = {"cache_path": str(snapshot), "cache_content_sha256": entry["response_sha256"], "retrieval_time_utc": entry["retrieved_at_utc"], "cache_status": "immutable_source_snapshot"}
            text, sections = abstract_document(row.get("abstract", ""))
            row.update(text=text, topic=domain, split_context="development", source_locator=source["source_locator"], fulltext_status="NOT_RETRIEVED_ABSTRACT_ONLY", source_snapshot={"sha256": entry["response_sha256"], "url": entry["request_url"], "retrieved_at": entry["retrieved_at_utc"], "object_path": str(snapshot), "json_pointer": entry["json_pointer"]}, derivation={"method": "HTMLParser entity decoding; explicit block boundaries; strip outer line whitespace", "input_abstract_sha256": hashlib.sha256(raw.get("abstract", "").encode()).hexdigest(), "raw_record_lineage": entry, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            row["sections"] = sections
            row["derivation"]["method"] += "; normalize horizontal Unicode whitespace; retain source h4 headings and exact derived-text section offsets"
            if row["document_id"] not in seen:
                documents.append(row)
                seen.add(row["document_id"])
            # Predeclared acquisition-order rule: no selection by model outcome.
            if selected < count and len(text) >= 500 and document_lifecycle_status(row) == "ACTIVE_METADATA_ONLY":
                selected += 1
                case_id = f"{domain}_{selected}"
                document = dict(row, topic=case_id)
                document_path = root / "documents" / f"{case_id}.json"
                atomic_json(document_path, document)
                cases.append({"case_id": case_id, "domain": domain, "document": file_ref(document_path), "document_id": row["document_id"], "question": f"Which public datasets can test the observations described in {row['title'].rstrip('.?')}?", "requirements": [{"field": "species", "expected": "Homo sapiens", "essential": True}] if domain != "environmental" else [{"field": "modality", "expected": "microbiome_metagenomics", "essential": True}], "question_origin": "predeclared development case; not expert review", "selection": "first acquired active record with at least500 normalized abstract characters"})
        if selected != count:
            raise ValueError(f"Insufficient real case inputs in {domain}")
    # Each individual run sees its selected real document while sharing all real studies.
    selected_docs = [json.loads(Path(case["document"]["path"]).read_text(encoding="utf-8")) for case in cases]
    write_rows(root / "catalog/literature.jsonl", selected_docs)
    write_rows(root / "source_catalog/literature.jsonl", documents)
    studies = read_rows(expanded / "corpus/datasets.jsonl") + read_rows(args.omics / "studies.jsonl")
    by_id = {}
    for raw in studies:
        row = copy.deepcopy(raw)
        entries = lineage.get(row["dataset_id"], [])
        if entries:
            entry = entries[0]
            metadata = row.setdefault("metadata", {})
            source = row.get("source_provenance") or metadata.get("source_provenance")
            if source:
                source.setdefault("metadata", {})["cache_snapshot"] = {"cache_path": entry["response_object_path"], "cache_content_sha256": entry["response_sha256"], "retrieval_time_utc": entry["retrieved_at_utc"], "cache_status": "immutable_source_snapshot"}
            metadata["record_source_lineage"] = entries
        previous = by_id.get(row["dataset_id"])
        if previous and digest(previous) != digest(row):
            raise ValueError("Conflicting study identity across source qualifications")
        by_id[row["dataset_id"]] = row
    write_rows(root / "catalog/studies.jsonl", [by_id[key] for key in sorted(by_id)])
    write_rows(root / "catalog/processed_inspections.jsonl", [])
    qualification = root / "runtime-qualification/qualified_7b_pass1.json"
    qualification.parent.mkdir(parents=True)
    shutil.copyfile(args.qualification, qualification)
    protocol = {"schema_version": "final_real_case_protocol_v1", "created_at": datetime.now(timezone.utc).isoformat(), "data_origin": "real", "cases": cases, "limit": 1, "chunks": 2, "scope": "Six new abstract-assisted development cases in four domains. No full-text, independent-cohort, biological-answerability, expert-label or calibrated-probability claim.", "frozen_holdout_executed": False, "inputs": [file_ref(expanded / "corpus/literature.jsonl"), file_ref(expanded / "corpus/datasets.jsonl"), file_ref(expanded / "RECORD_SOURCE_LINEAGE.jsonl"), file_ref(args.omics / "QUALIFICATION.json"), file_ref(args.omics / "studies.jsonl")], "derived_study_ids": len(by_id), "source_catalog_literature_count": len(documents)}
    atomic_json(root / "PROTOCOL.json", protocol)
    print(json.dumps({"status": "PREPARED", "cases": len(cases), "study_ids": len(by_id), "root": str(root)}), flush=True)


def execute(args):
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Execution output exists; preserve it and use a new output")
    out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    blocked = []

    def deny(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise PermissionError("Final real cases deny Python sockets and DNS")

    sys.addaudithook(deny)
    try:
        with socket.socket() as probe:
            probe.connect(("127.0.0.1", 9))
    except PermissionError:
        pass
    else:
        raise RuntimeError("Network denial probe failed")
    probe_count = len(blocked)
    summaries, dossiers, locators = [], [], []
    for case in protocol["cases"]:
        print(json.dumps({"stage": "starting", "case_id": case["case_id"], "mode": "cache_replay" if args.replay else "fresh"}), flush=True)
        run = out / case["case_id"]
        result = analyze(root, run, args.model, args.embeddings, question=case["question"], requirements=case["requirements"], limit=1, chunks=protocol["chunks"], fresh=not args.replay, device="cuda", topic=case["case_id"], question_source_id=case["document_id"])
        manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        accepted_statuses = {"PASS", "PARTIAL"}
        analysis_valid = result.get("status") in accepted_statuses and manifest.get("execution_status") in accepted_statuses
        run_dossiers = read_rows(run / "scientific_dossiers.jsonl")
        if not analysis_valid:
            result["case_validation"] = "FAIL_ANALYSIS_EXECUTION"
        elif not run_dossiers:
            result["case_validation"] = "FAIL_NO_SOURCE_LINKED_DOSSIER"
        else:
            dossier = run_dossiers[0]
            if not validate_dossier(dossier):
                raise ValueError("Invalid generated scientific dossier")
            target = out / "dossiers" / f"{case['case_id']}.json"
            atomic_json(target, dossier)
            target.with_suffix(".html").write_text(render_dossier(dossier), encoding="utf-8")
            dossiers.append({"case_id": case["case_id"], "domain": case["domain"], "dossier": file_ref(target), "run_manifest": file_ref(run / "RUN_MANIFEST.json")})
            result["case_validation"] = "PASS_SOURCE_ASSISTED"
        document = json.loads(Path(case["document"]["path"]).read_text(encoding="utf-8"))
        locators.append({"locator": document["source_locator"], "artifact": case["document"]})
        summaries.append(dict(case_id=case["case_id"], **result))
        atomic_json(out / "PROGRESS.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)
        gc.collect()
    control = {"kind": "Python audit hook; no native-library firewall claim", "blocked_probe_count": probe_count, "unexpected_requests": len(blocked) - probe_count}
    receipt = {"schema_version": "final_real_case_execution_v1", "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "protocol": file_ref(root / "PROTOCOL.json"), "dataset_catalog": file_ref(root / "catalog/studies.jsonl"), "mode": "cache_replay" if args.replay else "fresh_local_inference", "network_control": control, "cases": dossiers, "source_locators": locators, "observations": summaries, "status": "PASS" if len(dossiers) == 6 and all(row["case_validation"] == "PASS_SOURCE_ASSISTED" for row in summaries) and control["unexpected_requests"] == 0 else "FAIL"}
    atomic_json(out / "CASE_EXECUTION.json", receipt)
    print(json.dumps({"status": receipt["status"], "dossiers": len(dossiers), "out": str(out)}), flush=True)
    return int(receipt["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--expanded", type=Path, required=True)
    p.add_argument("--omics", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
