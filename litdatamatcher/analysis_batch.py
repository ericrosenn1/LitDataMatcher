"""Durable serial batches for native v2 analysis; no scheduler or model workers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .analysis_session import AnalysisSession
from .controller import integration_lease
from .data_plane import atomic_json, atomic_write, canonical_json, digest
from .strict_artifacts import strict_json_loads, validate_artifact
from .workflow_checkpoint import file_identity


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _safe_case_id(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value)
        or value.endswith(".")
    ):
        raise ValueError("Case IDs must be safe ASCII basenames of 1–96 characters")
    if value.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("Case ID is a reserved filename")
    return value


def load_batch_inputs(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Batch input exceeds 50 MiB")
    cases, seen = [], set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = strict_json_loads(line)
            if not isinstance(row, dict):
                raise ValueError("Batch requires JSONL object records")
            case_id = _safe_case_id(
                row["case_id"] if "case_id" in row else "case_" + digest(row)[:20]
            )
            if case_id.casefold() in seen:
                raise ValueError("Duplicate case ID")
            seen.add(case_id.casefold())
            source = row.get("source_document", row)
            if not isinstance(source, dict):
                raise ValueError("Case source_document must be an object")
            document = dict(source)
            document_id = (
                document.get("document_id") or document.get("id") or document.get("source_id")
            )
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("Case document requires an explicit source identity")
            text = document.get("text") or document.get("abstract")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Case document text is empty")
            document["document_id"] = document_id
            document.setdefault("source_id", document_id)
            document["text"] = text
            document.setdefault("split_context", "development")
            if (
                document["split_context"] != "development"
                or row.get("split", "development") != "development"
            ):
                raise ValueError("Batch accepts development cases only")
            question = row.get("question") or row.get("query")
            if question is not None and (not isinstance(question, str) or not question.strip()):
                raise ValueError("Case question must be nonempty text when present")
            cases.append(
                {
                    "case_id": case_id,
                    "document": document,
                    "question": question,
                    "input_sha256": digest(row),
                }
            )
            if len(cases) > 1000:
                raise ValueError("Batch exceeds 1000 cases")
    if not cases:
        raise ValueError("Batch input contains no cases")
    return cases


def _run_path(out, case_id, relative):
    path = (out / relative).resolve()
    if not path.is_relative_to(out / "cases" / case_id):
        raise ValueError("Per-case output path escapes its case directory")
    return path


def _verified_terminal(out, case_id, entry):
    if not entry.get("terminal") or not entry.get("outputs"):
        return False
    run = _run_path(out, case_id, entry["run_dir"])
    expected = {str(run / "RUN_MANIFEST.json"), str(run / ".workflow/state.json")}
    paths = set()
    for item in entry["outputs"]:
        path = (out / item["path"]).resolve()
        if path.parent not in {run, run / ".workflow"}:
            raise ValueError("Batch receipt points outside the case output")
        paths.add(str(path))
        if not path.is_file() or _file_sha(path) != item["sha256"]:
            return False
        validate_artifact(path)
    return expected <= paths


def _case_receipt(out, run):
    checkpoint = run / ".workflow/state.json"
    state = strict_json_loads(checkpoint.read_text(encoding="utf-8"))
    paths = [path for path in run.iterdir() if path.is_file()] + [checkpoint]
    return bool(state.get("completed")), [
        {"path": path.relative_to(out).as_posix(), "sha256": _file_sha(path)}
        for path in sorted(paths)
    ]


def run_batch(
    root,
    out,
    model_dir,
    embedding_dir,
    inputs,
    *,
    resume=False,
    stop_on_error=False,
    device="cpu",
    vector_cache=None,
    limit=1,
    chunks=2,
    fresh=False,
    requirements=None,
    runtime_config=None,
    runtime_qualification=None,
):
    """Share a serial model session; independently journal every case and failure."""
    from .v2 import analyze

    root, out, model_dir, embedding_dir, inputs = (
        Path(path).resolve() for path in (root, out, model_dir, embedding_dir, inputs)
    )
    if out.exists() != bool(resume):
        raise ValueError("Use a new batch output, or --resume an existing batch")
    if any(out == path or out.is_relative_to(path) for path in (model_dir, embedding_dir)):
        raise ValueError("Batch output cannot be inside model directories")
    if limit < 1 or chunks < 1:
        raise ValueError("Positive analysis limits required")
    input_identity = file_identity(inputs)
    cases = load_batch_inputs(inputs)
    if file_identity(inputs) != input_identity:
        raise ValueError("Batch input changed while it was being read")
    session = AnalysisSession(
        root,
        model_dir,
        embedding_dir,
        device=device,
        vector_cache=vector_cache,
        runtime_config=runtime_config,
        runtime_qualification=runtime_qualification,
    )
    options = {
        "device": device,
        "vector_cache": session.vector_cache,
        "limit": limit,
        "chunks": chunks,
        "fresh": fresh,
        "requirements": requirements,
        "runtime_config": session.runtime_config,
        "runtime_qualification": session.runtime_qualification,
    }
    identity = {
        "schema_version": "native-batch-v1",
        "input": input_identity,
        "options": options,
        "shared_analysis": session.fingerprint({}),
        "case_inputs": {case["case_id"]: case["input_sha256"] for case in cases},
    }
    state_path = out / "BATCH_STATE.json"
    with integration_lease(out / ".batch_owner", "native-analysis-batch"):
        # Preflight is advisory; only this locked read may authorize a write.
        if resume:
            state = strict_json_loads(state_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != digest(identity):
                raise ValueError(
                    "Batch resume fingerprint mismatch: inputs, configuration, models or implementation changed"
                )
            if not isinstance(state.get("cases"), dict) or set(state["cases"]) - set(
                identity["case_inputs"]
            ):
                raise ValueError("Batch state contains invalid case identities")
        else:
            if any(path.name != ".batch_owner" for path in out.iterdir()):
                raise ValueError(
                    "Batch output was created by another owner; preserve it and use --resume"
                )
            state = {
                "schema_version": "native-batch-v1",
                "fingerprint": digest(identity),
                "inputs": identity,
                "created_utc": utc_now(),
                "cases": {},
            }
        state["status"] = "RUNNING"
        atomic_json(state_path, state)
        try:
            for case in cases:
                case_id = case["case_id"]
                entry = state["cases"].setdefault(
                    case_id, {"input_sha256": case["input_sha256"], "attempts": []}
                )
                if entry.get("input_sha256") != case["input_sha256"]:
                    raise ValueError("Per-case input identity changed")
                if _verified_terminal(out, case_id, entry):
                    entry["last_action"] = "VERIFIED_SKIP"
                    atomic_json(state_path, state)
                    continue
                document_path = out / "inputs" / f"{case_id}.jsonl"
                content = canonical_json(case["document"]) + b"\n"
                if document_path.exists() and document_path.read_bytes() != content:
                    raise ValueError("Materialized case input changed")
                if not document_path.exists():
                    atomic_write(document_path, content)
                existing = (
                    _run_path(out, case_id, entry["run_dir"]) if entry.get("run_dir") else None
                )
                resumable = existing is not None and (existing / ".workflow/state.json").is_file()
                run = (
                    existing
                    if resumable
                    else out / "cases" / case_id / f"attempt-{len(entry['attempts']) + 1:04d}"
                )
                attempt = {
                    "run_dir": run.relative_to(out).as_posix(),
                    "started_utc": utc_now(),
                    "resume": resumable,
                }
                entry.update(status="RUNNING", run_dir=attempt["run_dir"], last_action="EXECUTED")
                entry["attempts"].append(attempt)
                atomic_json(state_path, state)
                try:
                    result = analyze(
                        root,
                        run,
                        model_dir,
                        embedding_dir,
                        question=case["question"],
                        question_source_id=case["document"]["document_id"]
                        if case["question"]
                        else None,
                        document_path=document_path,
                        resume=resumable,
                        _session=session,
                        **options,
                    )
                    terminal, outputs = _case_receipt(out, run)
                    entry.update(
                        status=result["status"], result=result, terminal=terminal, outputs=outputs
                    )
                    attempt.update(
                        status=result["status"], result=result, terminal=terminal, outputs=outputs
                    )
                    if stop_on_error and (result["status"] == "FAIL" or not terminal):
                        raise RuntimeError(
                            "Case returned a failed or retryable native result; batch stopped by policy"
                        )
                except BaseException as error:
                    entry.update(
                        status="INTERRUPTED"
                        if isinstance(error, (KeyboardInterrupt, SystemExit))
                        else "FAIL",
                        terminal=False,
                    )
                    attempt.update(
                        status=entry["status"],
                        error={"type": type(error).__name__, "message": str(error)},
                    )
                    atomic_json(state_path, state)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)) or stop_on_error:
                        raise
                finally:
                    attempt["finished_utc"] = utc_now()
                    if "error" in attempt or attempt.get("status") != "PASS":
                        atomic_json(
                            out / "failures" / f"{case_id}-{len(entry['attempts']):04d}.json",
                            attempt,
                        )
                    atomic_json(state_path, state)
            session.validate_unchanged(full=True)
            if file_identity(inputs) != identity["input"]:
                raise ValueError("Batch input changed during execution")
            statuses = [entry["status"] for entry in state["cases"].values()]
            counts = {
                "passed": statuses.count("PASS"),
                "partial": statuses.count("PARTIAL"),
                "failed": sum(value in {"FAIL", "INTERRUPTED"} for value in statuses),
                "verified_skips": sum(
                    entry.get("last_action") == "VERIFIED_SKIP" for entry in state["cases"].values()
                ),
                "retryable": sum(
                    not entry.get("terminal", False) for entry in state["cases"].values()
                ),
            }
            state.update(
                status="PASS"
                if counts["passed"] == len(cases)
                else "PARTIAL"
                if counts["passed"] or counts["partial"]
                else "FAIL",
                counts=counts,
                finished_utc=utc_now(),
            )
            result = {
                "status": state["status"],
                "cases": len(cases),
                **counts,
                "manifest": str(state_path),
            }
            atomic_json(out / "BATCH_RESULT.json", result)
            atomic_json(state_path, state)
            return result
        except BaseException as error:
            state.update(
                status="INTERRUPTED"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "FAIL",
                error={"type": type(error).__name__, "message": str(error)},
            )
            atomic_json(state_path, state)
            raise
        finally:
            session.close(validate=False)
